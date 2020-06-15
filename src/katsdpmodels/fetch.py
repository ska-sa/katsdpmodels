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

import contextlib
import errno
import io
import hashlib
import re
import logging
import os
import urllib.parse
from typing import Dict, Tuple, Optional, Type, TypeVar, Any, cast
from typing_extensions import Protocol

import requests
import requests_file

from . import models


MAX_ALIASES = 30
_logger = logging.getLogger(__name__)
_F = TypeVar('_F', bound='Fetcher')
_M = TypeVar('_M', bound='models.Model')


class Session(Protocol):
    """Generalization of :class:`requests.Session`."""

    def get(self, url: str, **kwargs) -> requests.Response: ...   # pragma: nocover
    def head(self, url: str, **kwargs) -> requests.Response: ...  # pragma: nocover
    def close(self) -> None: ...                                  # pragma: nocover


class HttpFile(io.RawIOBase):
    """File-like object that fetches byte ranges via HTTP.

    This requires the server to advertise for support byte-range requests and
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

    def __init__(self, session: Session, url: str) -> None:
        self._session = session
        self._offset = 0
        # TODO do we need to set Accept-Encoding: none? Not sure how transfer
        # encoding interact with byte ranges.
        with session.head(url) as resp:
            if resp.status_code == 404:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), url)
            elif resp.status_code in {401, 403}:
                raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), url)
            resp.raise_for_status()
            if resp.headers.get('Accept-Ranges', 'none') != 'bytes':
                raise OSError(None, 'Server does not accept byte ranges', url)
            try:
                self._length = int(resp.headers['Content-Length'])
            except (KeyError, ValueError):
                raise OSError(None, 'Server did not provide Content-Length header', url) from None
            # TODO: consider storing ETag/Last-Modified to check for data
            # changing under us.
            self._url = resp.url

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
                headers={'Range': f'bytes={start}-{end}'},
                stream=True) as resp:
            resp.raise_for_status()
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


class Fetcher:
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
        self._alias_cache: Dict[str, str] = {}
        self._model_cache: Dict[str, models.Model] = {}
        self.close_models = True

    @property
    def session(self) -> Session:
        return self._session

    def close(self) -> None:
        """Release the resources associated with the fetcher.

        See also
        --------
        :attr:`close_session`, :attr:`close_models`
        """
        if self.close_session:
            self._session.close()
        self._alias_cache.clear()
        if self.close_models:
            for model in self._model_cache.values():
                model.close()
        self._model_cache.clear()

    def resolve(self, url: str) -> str:
        """Find the canonical URL, following aliases."""
        original_url = url
        aliases = 0
        parts = urllib.parse.urlparse(url)
        while parts.path.endswith('.alias'):
            aliases += 1
            if aliases > MAX_ALIASES:
                raise models.TooManyAliasesError.with_urls(
                    f'Reached limit of {MAX_ALIASES} levels of aliases',
                    url=url, original_url=original_url)
            if url in self._alias_cache:
                new_url = self._alias_cache[url]
            else:
                with self._session.get(url) as resp:
                    resp.raise_for_status()
                    rel_path = resp.text.rstrip()
                    new_url = urllib.parse.urljoin(resp.url, rel_path)
                self._alias_cache[url] = new_url
            _logger.debug('Redirecting from %s to %s', url, new_url)
            url = new_url
            parts = urllib.parse.urlparse(url)
        return url

    def _get_eager(self, url: str) -> Tuple[io.IOBase, str, str]:
        with self._session.get(url) as resp:
            resp.raise_for_status()
            data = resp.content
            url = resp.url     # Handle HTTP redirects
        checksum = hashlib.sha256(data).hexdigest()
        parts = urllib.parse.urlparse(url)
        match = re.search(r'/sha256_([a-z0-9]+)\.[^/]+$', parts.path)
        if match:
            if checksum != match.group(1):
                raise models.ChecksumError('Content did not match checksum in URL')
        # typeshed doesn't reflect that BytesIO inherits from BufferedIOBase
        # (fixed in master, but not in mypy 0.780).
        return cast(io.IOBase, io.BytesIO(data)), url, checksum

    def _get_lazy(self, url: str) -> Tuple[io.IOBase, str]:
        fh = HttpFile(self._session, url)
        return fh, fh.url

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
        original_url = url
        url = self.resolve(url)
        if url in self._model_cache:
            model = self._model_cache[url]
            if model_class.model_type != model.model_type:
                raise models.ModelTypeError.with_urls(
                    f'Expected a model of type {model_class.model_type!r}, '
                    f'not {model.model_type!r}',
                    url=url, original_url=original_url)
            if not isinstance(model, model_class):
                raise TypeError('model_class should be the base class for the model type')
            return model

        try:
            if lazy:
                fh, url = self._get_lazy(url)
                checksum: Optional[str] = None
            else:
                fh, url, checksum = self._get_eager(url)
        except models.ModelError as exc:
            exc.original_url = original_url
            exc.url = url
            raise

        try:
            with contextlib.ExitStack() as exit_stack:
                exit_stack.callback(fh.close)
                new_model = model_class.from_file(fh, url)
                exit_stack.pop_all()   # new_model now owns fh, or has closed it
        except models.ModelError as exc:
            exc.original_url = original_url
            exc.url = url
            raise
        except Exception as exc:
            raise models.DataError.with_urls(
                f'Failed to load model from {url}: {exc}',
                url=url, original_url=original_url) from exc
        new_model.checksum = checksum
        self._model_cache[url] = new_model
        return new_model

    def __enter__(self: _F) -> _F:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


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
