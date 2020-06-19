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
import io
import re
import logging
import os
from typing import List, Generator, Optional, Type, TypeVar, cast
from typing_extensions import Protocol

import requests
import requests_file

from . import models, fetch_base


MAX_ALIASES = 30
_logger = logging.getLogger(__name__)
_T = TypeVar('_T')
_M = TypeVar('_M', bound=models.Model)


class Session(Protocol):
    """Generalization of :class:`requests.Session`."""

    def get(self, url: str, **kwargs) -> requests.Response: ...   # pragma: nocover
    def head(self, url: str, **kwargs) -> requests.Response: ...  # pragma: nocover
    def close(self) -> None: ...                                  # pragma: nocover


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

    def __init__(self, session: Session, url: str) -> None:
        self._session = session
        self._offset = 0
        # TODO do we need to set Accept-Encoding: none? Not sure how transfer
        # encoding interact with byte ranges.
        with session.head(url, headers=self._HEADERS) as resp:
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
        """The actual URL with the content.

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


class Fetcher(fetch_base.FetcherBase):
    """Fetches and caches models.

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

        It need not be an instance of :class:`requests.Session`; one can
        create any class as long as it implements the :class:`Session`
        protocol.

    Attributes
    ----------
    close_models
        If true (the default), the cached models will be closed when the
        fetcher is closed. If set to false, the user must close the models.
        This is only intended for use by :func:`fetch_model`.
    close_session
        If true, the session used for HTTP requests will be closed when the
        fetcher is closed. It defaults to true if an internally-created session
        is used, and false if the user provided a session.

    Raises
    ------
    :exc:`.ModelError`
        For any issues with the model itself
    :exc:`requests.exceptions.RequestException`
        For any issues at the HTTP level
    """

    def __init__(self, *, session: Optional[Session] = None) -> None:
        super().__init__()
        self._session: Session
        if session is None:
            self._session = requests.Session()
            # TODO: requests_file is convenient, but it would be more efficient to
            # open the file directly with h5py rather than sucking it into a
            # BytesIO.
            self._session.mount('file://', requests_file.FileAdapter())
            self.close_session = True
        else:
            self._session = session
            self.close_session = False

    @property
    def session(self) -> Session:
        return self._session

    def close(self) -> None:
        """Release the resources associated with the fetcher.

        See also
        --------
        :attr:`close_session`, :attr:`close_models`
        """
        super().close()
        if self.close_session:
            self._session.close()

    def _handle_request(self, request: fetch_base.Request, *,
                        lazy: bool = False) -> fetch_base.Response:
        if request.response_type == fetch_base.ResponseType.TEXT:
            with self._session.get(request.url) as resp:
                resp.raise_for_status()
                return fetch_base.TextResponse(resp.url, resp.headers, resp.text)
        elif not lazy:
            with self._session.get(request.url) as resp:
                resp.raise_for_status()
                content = resp.content
                file = io.BytesIO(content)
                return fetch_base.FileResponse(
                    resp.url, resp.headers, file=cast(io.IOBase, file), content=content)
        else:
            fh = HttpFile(self._session, request.url)
            # TODO: make HttpFile return the full headers
            headers = requests.structures.CaseInsensitiveDict(
                {'Content-type': fh.content_type or 'application/octet-stream'}
            )
            return fetch_base.FileResponse(fh.url, headers, file=fh, content=None)

    def _run(self, gen: Generator[fetch_base.Request, fetch_base.Response, _T], *,
             lazy: bool = False) -> _T:
        try:
            request = next(gen)      # Start it going
            while True:
                response = self._handle_request(request)
                request = gen.send(response)
        except StopIteration as exc:
            return exc.value
        finally:
            gen.close()

    def resolve(self, url: str) -> List[str]:
        """Follow a chain of aliases.

        Return a list of URLs found along the chain. The first element is the
        given URL and the final element is the resolved model.

        Raises
        ------
        .models.TooManyAliasesError
            If there were more than :const:`MAX_ALIASES` aliases or a cycle was found.
        """
        return self._run(self._resolve(url))

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

            1. If a custom session is passed to the constructor, it must
               support the RFC 7233 ``Range`` header and return a response
               with the matching ``Content-Range``. It must also return a
               ``Content-Length`` header for :meth:`Session.head` requests.
            2. The session must not be closed while the model is in use, even
               if the fetcher is no longer needed.
            3. The checksum stored in the filename is not validated.
            4. If the model is already in the cache, the laziness setting is
               ignored and the cached model is returned.
            5. It does not work with ``file://`` URLs.

        Raises
        ------
        .ModelError
            If there are high-level errors with the model.
        requests.exception.RequestException
            Any exceptions raised by the underlying session.
        """
        return self._run(self._get(url, model_class), lazy=lazy)


def fetch_model(url: str, model_class: Type[_M], *,
                session: Optional[Session] = None) -> _M:
    """Convenience function for retrieving a single model.

    This should only be used when loading just a single model. If multiple
    models will be used instead, construct an instance of :class:`Fetcher`
    and use it to fetch models, as this will allow models that turn out to be
    the same to be shared.
    """
    with Fetcher(session=session) as fetcher:
        fetcher.close_models = False
        return fetcher.get(url, model_class)
