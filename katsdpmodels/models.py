"""Base functionality common to all model types."""

from abc import ABC, abstractmethod
import io
import hashlib
import logging
import re
import urllib.parse
from typing import List, Dict, Iterable, Optional, Any, ClassVar, Type, TypeVar
from typing_extensions import Protocol

import h5py
import requests
import requests_file


MAX_ALIASES = 30
_E = TypeVar('_E', bound='ModelError')
_M = TypeVar('_M', bound='Model')
_F = TypeVar('_F', bound='Fetcher')
_logger = logging.getLogger(__name__)


class ModelError(ValueError):
    """A model was found, but the content was incorrect."""

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.url: Optional[str] = None
        self.original_url: Optional[str] = None

    @classmethod
    def with_urls(cls: Type[_E], *args,
                  url: Optional[str] = None, original_url: Optional[str] = None) -> _E:
        exc = cls(args)
        exc.url = url
        exc.original_url = original_url
        return exc


class ModelTypeError(ModelError):
    """The ``model_type`` attribute was missing or did not match the expected value."""


class ModelFormatError(ModelError):
    """The ``model_format`` attribute was missing or did match a known value."""


class DataError(ModelError):
    """The model was missing some data or it had the wrong format."""


class ChecksumError(DataError):
    """The model did not match the checksum embedded in the filename."""


class TooManyAliasesError(ModelError):
    """The limit on the number of alias redirections was reached."""


class Model(ABC):
    """Base class for models.

    Models can either be loaded from instances of :class:`h5py` or constructed
    directly. Models loaded by the fetcher store the checksum of the raw data,
    and are considered equal if the checksums match. Otherwise, equality is by
    object identity.

    Models loaded by the fetcher should not be modified, as they may be shared
    by other users. Instead, make a copy and modify that.
    """

    model_type: ClassVar[str]
    model_format: ClassVar[str]
    checksum: Optional[str] = None

    @classmethod
    @abstractmethod
    def from_hdf5(cls: Type[_M], hdf5: h5py.File) -> _M:
        """Load a model from raw data."""

    def __eq__(self, other: object) -> Any:
        if not isinstance(other, Model):
            return NotImplemented
        elif self.checksum is not None and other.checksum is not None:
            return self.checksum == other.checksum and type(self) is type(other)
        else:
            return self is other

    def __hash__(self) -> int:
        if self.checksum is not None:
            return hash(self.checksum)
        else:
            return super().__hash__()


class Session(Protocol):
    """Generalization of :class:`requests.Session`."""

    def get(self, url: str) -> requests.Response: ...
    def close(self) -> None: ...


class Fetcher:
    """Fetches and caches models.

    It caches every URL it fetches, so it should not be held for a long time.
    It is best suited to fetching a batch of models, some of which may be turn
    out to be aliases of each other.

    It should be closed with :meth:`close` when no longer in use. It also
    implements the context manager protocol for this purpose.

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
    """

    def __init__(self, *, session: Optional[Session] = None) -> None:
        self._session: Session
        if session is None:
            self._session = requests.Session()
            # TODO: requests_file is convenient, but it would be more efficient to
            # open the file directly with h5py rather than sucking it into a
            # BytesIO.
            self._session.mount('file://', requests_file.FileAdapter())
            self._close_session = True
        else:
            self._session = session
            self._close_session = False
        self._alias_cache: Dict[str, str] = {}
        self._model_cache: Dict[str, Model] = {}

    def close(self) -> None:
        if self._close_session:
            self._session.close()
        self._alias_cache.clear()
        self._model_cache.clear()

    def resolve(self, url: str) -> str:
        """Find the canonical URL, following aliases."""
        original_url = url
        aliases = 0
        parts = urllib.parse.urlparse(url)
        while parts.path.endswith('.alias'):
            aliases += 1
            if aliases > MAX_ALIASES:
                raise TooManyAliasesError.with_urls(
                    f'Reached limit of {MAX_ALIASES} levels of aliases',
                    url=url, original_url=original_url)
            if url in self._alias_cache:
                new_url = self._alias_cache[url]
            else:
                with self._session.get(url) as resp:
                    rel_path = resp.text.rstrip()
                new_url = urllib.parse.urljoin(url, rel_path)
                self._alias_cache[url] = new_url
            _logger.debug('Redirecting from %s to %s', url, new_url)
            url = new_url
            parts = urllib.parse.urlparse(url)
        return url

    def get(self, url: str, model_class: Type[_M]) -> _M:
        original_url = url
        url = self.resolve(url)
        if url in self._model_cache:
            model = self._model_cache[url]
            if model_class.model_type != model.model_type:
                raise ModelTypeError.with_urls(
                    f'Expected a model of type {model_class.model_type!r}, '
                    f'not {model.model_type!r}',
                    url=url, original_url=original_url)
            if not isinstance(model, model_class):
                raise TypeError('model_class should be the base class for the model type')

        with self._session.get(url) as resp:
            data = resp.content
        checksum = hashlib.sha256(data).hexdigest()
        parts = urllib.parse.urlparse(url)
        match = re.search(r'/sha256_([a-z0-9]+)\.[^/]+$', parts.path)
        if match:
            if checksum != match.group(1):
                raise ChecksumError.with_urls(
                    'Content did not match checksum in URL',
                    url=url, original_url=original_url)

        with h5py.File(io.BytesIO(data), 'r') as hdf5:
            model_type = hdf5.attrs.get('model_type')
            if model_type != model_class.model_type:
                raise ModelTypeError.with_urls(
                    f'Expected a model of type {model_class.model_type!r}, not {model_type!r}',
                    url=url, original_url=original_url)
            try:
                new_model = model_class.from_hdf5(hdf5)
            except ModelError as exc:
                exc.original_url = original_url
                exc.url = url
                raise
            new_model.checksum = checksum
            return new_model

    def __enter__(self: _F) -> _F:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def fetch_model(url: str, model_class: Type[_M], *,
                session: Optional[Session] = None) -> _M:
    with Fetcher(session=session) as fetcher:
        return fetcher.get(url, model_class)


def fetch_models(urls: Iterable[str], model_class: Type[_M], *,
                 session: Optional[Session] = None) -> List[_M]:
    with Fetcher(session=session) as fetcher:
        return [fetcher.get(url, model_class) for url in urls]
