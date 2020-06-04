"""Base functionality common to all model types."""

from abc import ABC, abstractmethod
import io
import hashlib
import logging
import re
import urllib.parse
from typing import Mapping, Optional, Any, ClassVar, Type, TypeVar

import attr
import h5py
import requests
import requests_file


MAX_ALIASES = 30
_E = TypeVar('_E', bound='ModelError')
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


@attr.s
class RawModel:
    """Raw data for a model.

    This is a thin container for a :class:`h5py.File` which also records the
    URL and sha256 checksum.
    """

    hdf5: h5py.File = attr.ib()
    url: str = attr.ib()
    original_url: str = attr.ib()
    checksum: str = attr.ib()

    def __enter__(self) -> 'RawModel':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.hdf5.close()


class Model(ABC):
    """Base class for models.

    Models can either be loaded from instances of :class:`RawModel` or
    constructed directly. Models loaded from :class:`RawModel` store the
    checksum of the raw data, and are considered equal if the checksums
    match. Otherwise, equality is by object identity.

    If a model loaded from :class:`RawModel` is modified, clear the
    :attr:`checksum` attribute since it will no longer be valid.
    """

    model_type: ClassVar[str]
    model_format: ClassVar[str]
    checksum: Optional[str]

    def __init__(self, *, raw: Optional[RawModel] = None) -> None:
        self.checksum = raw.checksum if raw is not None else None

    @classmethod
    @abstractmethod
    def from_raw(cls, raw: RawModel) -> 'Model':
        """Load a model from raw data."""

    @classmethod
    def from_url(cls, url: str) -> 'Model':
        raw = fetch_raw(url, cls.model_type)
        try:
            with raw:
                return cls.from_raw(raw)
        except ModelError as exc:
            exc.original_url = raw.original_url
            exc.url = raw.url
            raise

    def __eq__(self, other: object) -> Any:
        if not isinstance(other, Model):
            return NotImplemented
        elif self.checksum is not None and other.checksum is not None:
            return self.checksum == other.checksum
        else:
            return self is other

    def __hash__(self) -> int:
        if self.checksum is not None:
            return hash(self.checksum)
        else:
            return super().__hash__()


def fetch_raw(
        url: str,
        model_type: str,
        get_options: Mapping[str, Any] = {}) -> RawModel:
    original_url = url
    with requests.session() as session:
        session.mount('file://', requests_file.FileAdapter())
        aliases = 0
        while True:
            parts = urllib.parse.urlparse(url)
            with session.get(url, **get_options) as resp:
                if parts.path.endswith('.alias'):
                    aliases += 1
                    if aliases > MAX_ALIASES:
                        raise TooManyAliasesError.with_urls(
                            f'Reached limit of {MAX_ALIASES} levels of aliases',
                            url=url, original_url=original_url)
                    rel_path = resp.text.rstrip()
                    new_url = urllib.parse.urljoin(url, rel_path)
                    _logger.debug('Redirecting from %s to %s', url, new_url)
                    url = new_url
                    continue
                data = resp.content
            break

    checksum = hashlib.sha256(data).hexdigest()
    match = re.search(r'/sha256_([a-z0-9]+)\.[^/]+$', parts.path)
    if match:
        if checksum != match.group(1):
            raise ChecksumError.with_urls(
                'Content did not match checksum in URL',
                url=url, original_url=original_url)

    # TODO: validate checksum if embedded in URL
    hdf5 = h5py.File(io.BytesIO(data), 'r')
    actual_model_type = hdf5.attrs.get('model_type')
    if actual_model_type != model_type:
        raise ModelTypeError.with_urls(
            f'Expected a model of type {model_type!r}, not {actual_model_type!r}',
            url=url, original_url=original_url)
    return RawModel(hdf5=hdf5, url=url, original_url=original_url, checksum=checksum)
