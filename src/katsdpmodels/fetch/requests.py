################################################################################
# Copyright (c) 2020, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Fetch models over HTTP."""

import errno
import functools
import io
import re
import logging
import os
import sys
import urllib.parse
from typing import (
    List, Generator, Dict, MutableMapping, Optional, Type, Callable,
    TypeVar, Any, TYPE_CHECKING
)

import requests

from .. import models, fetch

if TYPE_CHECKING:
    import katsdptelstate


_logger = logging.getLogger(__name__)
_T = TypeVar('_T')
_M = TypeVar('_M', bound=models.Model)
_F = TypeVar('_F', bound='Fetcher')
_TF = TypeVar('_TF', bound='TelescopeStateFetcher')
_Req = TypeVar('_Req')
_Resp = TypeVar('_Resp')


def _run_generator(gen: Generator[_Req, _Resp, _T],
                   handle_request: Callable[[_Req], _Resp]) -> _T:
    try:
        request = next(gen)      # Start it going
        while True:
            try:
                response = handle_request(request)
            except Exception:
                request = gen.throw(*sys.exc_info())
            else:
                request = gen.send(response)
    except StopIteration as exc:
        return exc.value
    finally:
        gen.close()


class HttpFile(io.RawIOBase):
    """File-like object that fetches byte ranges via HTTP.

    This requires the server to advertise support for byte-range requests and
    to provide a Content-Length. It is currently *not* robust against the
    content changing.

    Raises
    ------
    FileNotFoundError
        HTTP 404 error
    PermissionError
        HTTP 403 error
    OSError
        Other HTTP errors, or the server doesn't implement the required features.
    """

    # urllib3 by default sets this to gzip, deflate, but that won't play nice
    # with byte ranges because we need byte ranges from the original file, not
    # from the compressed representation.
    _HEADERS = {
        'Accept-Encoding': 'identity'
    }

    def __init__(self, session: requests.Session, url: str) -> None:
        self._session = session
        self._offset = 0
        with session.head(url, headers=self._HEADERS, allow_redirects=True) as resp:
            if resp.status_code == 404:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), url)
            elif resp.status_code in {401, 403}:
                raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), url)
            resp.raise_for_status()
            if resp.headers.get('Accept-Ranges', 'none') != 'bytes':
                raise OSError(None, 'Server does not accept byte ranges', url)
            if 'Content-Encoding' in resp.headers:
                raise OSError(None, 'Server provided Content-Encoding header', url)
            try:
                self._length = int(resp.headers['Content-Length'])
            except (KeyError, ValueError):
                raise OSError(None, 'Server did not provide Content-Length header', url) from None
            # TODO: consider storing ETag/Last-Modified to check for data
            # changing under us.
            self._url = resp.url
            self.content_type = resp.headers.get('Content-Type')

    @property
    def url(self) -> str:
        """Return the final URL.

        This may differ from the constructor argument if HTTP redirects occurred.
        """
        return self._url

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_offset = offset
        elif whence == io.SEEK_CUR:
            new_offset = self._offset + offset
        elif whence == io.SEEK_END:
            new_offset = self._length + offset
        else:
            raise ValueError(f'invalid whence ({whence}, should be 0, 1 or 2)')
        if new_offset < 0:
            raise OSError(errno.EINVAL, 'Invalid argument')
        self._offset = new_offset
        return new_offset

    def tell(self) -> int:
        return self._offset

    def readinto(self, b) -> int:
        start = min(self._length, self._offset)
        end = min(self._length, start + len(b)) - 1    # End is inclusive
        with self._session.get(
                self._url,
                headers={'Range': f'bytes={start}-{end}', **self._HEADERS},
                stream=True) as resp:
            resp.raise_for_status()
            if 'Content-Encoding' in resp.headers:
                raise OSError('Server provided Content-Encoding header')
            content_range = resp.headers.get('Content-Range', '')
            # RFC 7233 specifies the format
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(?:\*|\d+)', content_range)
            if (resp.status_code != requests.codes.PARTIAL_CONTENT or not match
                    or int(match.group(1)) != start or int(match.group(2)) != end):
                # Tornado does not send partial content if the entire content
                # was requested, so allow that case.
                if start != 0 or end != self._length - 1:
                    raise OSError('Did not receive expected byte range')
            bytes_read = resp.raw.readinto(b)
            self._offset += bytes_read
            return bytes_read

    def close(self) -> None:
        super().close()


class Fetcher(fetch.FetcherBase):
    """Fetches and caches models, using the :mod:`requests` library.

    It caches every URL it fetches (ignoring any cache control headers), so it
    should not be reused over a long time.  It is best suited to fetching a
    batch of models, some of which may turn out to be aliases of each other.

    It should be closed with :meth:`close` when no longer in use. It also
    implements the context manager protocol for this purpose. This will close
    the retrieved models, so must only be done once the models are no longer in
    use.

    This class is *not* thread-safe.

    Parameters
    ----------
    session
        Interface for doing the actual HTTP requests. If not specified, an
        internal session will be created (which supports ``file://`` URLs), and
        closed when the fetcher is closed. If a custom session is provided it
        will *not* be closed (so it can be shared between multiple
        :class:`Fetcher` instances).
    model_cache
        A dictionary for caching models by URL. This is not typically needed,
        as the fetcher will use an internal cache if one is not provided, but
        allows fetchers to share a cache (but not in a thread-safe way!).
        If a custom cache is provided, then :meth:`close` will not close the
        models in it, and the caller is responsible for doing so.

    Raises
    ------
    :exc:`.ModelError`
        For any issues with the model itself
    :exc:`requests.RequestException`
        For any issues at the HTTP level
    """

    def __init__(self, *,
                 session: Optional[requests.Session] = None,
                 model_cache: Optional[MutableMapping[str, models.Model]] = None) -> None:
        super().__init__(model_cache=model_cache)
        if session is None:
            self._session = requests.Session()
            self._close_session = True
        else:
            self._session = session
            self._close_session = False

    @property
    def session(self) -> requests.Session:
        return self._session

    def close(self) -> None:
        self._close()
        if self._close_session:
            self._session.close()

    def __enter__(self: _F) -> _F:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _handle_request(self, request: fetch.Request, *,
                        lazy: bool = False) -> fetch.Response:
        assert request.response_type in {fetch.ResponseType.TEXT, fetch.ResponseType.FILE}
        parts = urllib.parse.urlsplit(request.url)
        if parts.scheme == 'file':
            return self._handle_file_scheme(request, lazy)
        elif request.response_type == fetch.ResponseType.TEXT:
            with self._session.get(request.url) as resp:
                resp.raise_for_status()
                return fetch.TextResponse(resp.url, resp.headers, resp.text)
        elif not lazy:
            with self._session.get(request.url) as resp:
                resp.raise_for_status()
                content = resp.content
                file = io.BytesIO(content)
                return fetch.FileResponse(
                    resp.url, resp.headers, file=file, content=content)
        else:
            fh = HttpFile(self._session, request.url)
            # TODO: make HttpFile return the full headers
            headers = requests.structures.CaseInsensitiveDict(
                {'Content-type': fh.content_type or 'application/octet-stream'}
            )
            return fetch.FileResponse(fh.url, headers, file=fh, content=None)

    def resolve(self, url: str) -> List[str]:
        """Follow a chain of aliases.

        Return a list of URLs found along the chain. The first element is the
        given URL and the final element is the resolved model.

        Raises
        ------
        .TooManyAliasesError
            If there were more than :data:`.MAX_ALIASES` aliases or a cycle was found.
        """
        return _run_generator(self._resolve(url), self._handle_request)

    def get(self, url: str, model_class: Type[_M], *,
            lazy: bool = False) -> _M:
        """Retrieve a single model.

        The caller must *not* close the retrieved model, as it is cached and
        a future request for the model would return the closed model. Instead,
        the caller must close the fetcher once it no longer needs any of the
        models.

        Parameters
        ----------
        lazy
            If true, create a view of the HDF5 file that only retrieves data
            as it is needed. Whether this actually allows data to be loaded
            lazily depends on the `model_class`: some model classes will read
            all the data into memory on creation, in which case lazy loading
            may perform significantly worse.

            Lazy loading imposes some additional requirements:

            1. The session must not be closed while the model is in use, even
               if the fetcher is no longer needed.
            2. The checksum stored in the filename is not validated.
            3. If the model is already in the cache, the laziness setting is
               ignored and the cached model is returned.

        Raises
        ------
        .ModelError
            If there are high-level errors with the model.
        requests.exception.RequestException
            Any exceptions raised by the underlying session.
        """
        return _run_generator(
            self._get(url, model_class),
            functools.partial(self._handle_request, lazy=lazy)
        )


def fetch_model(url: str, model_class: Type[_M], *,
                session: Optional[requests.Session] = None) -> _M:
    """Retrieve a single model.

    This is a convenience function that should only be used when loading just a
    single model. If multiple models will be used instead, construct an
    instance of :class:`Fetcher` and use it to fetch models, as this will allow
    models that turn out to be the same to be shared.
    """
    # Custom cache so that fetcher won't close the model
    model_cache: Dict[str, models.Model] = {}
    with Fetcher(session=session, model_cache=model_cache) as fetcher:
        return fetcher.get(url, model_class)


class TelescopeStateFetcher(fetch.TelescopeStateFetcherBase['katsdptelstate.TelescopeState']):
    """Fetches models that are referenced by telescope state.

    The telescope state must have a ``sdp_model_base_url`` key with a base
    URL, and a key per model with an URL relative to this one. If it is
    missing then a :exc:`KeyError` will be raised from :meth:`get`, rather
    than the constructor.

    If no fetcher is provided, an internal one will be created, and closed
    when this object is closed.
    """

    def __init__(self,
                 telstate: 'katsdptelstate.TelescopeState',
                 fetcher: Optional[Fetcher] = None) -> None:
        super().__init__(telstate)
        if fetcher is not None:
            self.fetcher = fetcher
            self._close_fetcher = False
        else:
            self.fetcher = Fetcher()
            self._close_fetcher = True

    @staticmethod
    def _handle_request(
            request: fetch.TelescopeStateRequest['katsdptelstate.TelescopeState']) -> str:
        return request.telstate[request.key]

    def get(self, key: str, model_class: Type[_M], *,
            telstate: Optional['katsdptelstate.TelescopeState'] = None,
            lazy: bool = False) -> _M:
        """Retrieve a single model.

        The semantics are the same as for :meth:`Fetcher.get`. Any problems
        with getting the keys from the telescope state will raise
        :exc:`.TelescopeStateError`.

        If `telstate` is provided, it is used instead of the constructor
        argument for fetching the model key; but the constructor telstate is
        still used to fetch ``sdp_model_base_url``.
        """
        url = _run_generator(self._get_url(key, telstate=telstate), self._handle_request)
        return self.fetcher.get(url, model_class, lazy=lazy)

    def close(self) -> None:
        """Clean up resources."""
        if self._close_fetcher:
            self.fetcher.close()

    def __enter__(self: _TF) -> _TF:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
