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

import contextlib
import io
import hashlib
import re
import logging
import urllib.parse
from typing import BinaryIO, Dict, Tuple, Optional, Mapping, Type, TypeVar, Any
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

    def get(self, url: str, *, headers: Optional[Mapping[str, str]] = None) -> requests.Response: ...   # pragma: nocover  # noqa: E501
    def head(self, url: str, *, headers: Optional[Mapping[str, str]] = None) -> requests.Response: ...  # pragma: nocover  # noqa: E501
    def close(self) -> None: ...                          # pragma: nocover


class Fetcher:
    """Fetches and caches models.

    It caches every URL it fetches, so it should not be held for a long time.
    It is best suited to fetching a batch of models, some of which may be turn
    out to be aliases of each other.

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
        fetch is closed. It defaults to true if an internally-created session
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

    def _get_eager(self, url: str) -> Tuple[BinaryIO, str, str]:
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
        return io.BytesIO(data), url, checksum

    def _get_lazy(self, url: str) -> Tuple[BinaryIO, str]:
        # TODO: implement
        fh, url, _ = self._get_eager(url)
        return fh, url

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
    and use it to fetch models.
    """
    with Fetcher(session=session) as fetcher:
        fetcher.close_models = False
        return fetcher.get(url, model_class)
