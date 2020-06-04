"""Base functionality common to all model types."""

from abc import ABC, abstractmethod
import io
import hashlib
import logging
import re
import urllib.parse
from typing import Mapping, Optional, Any, ClassVar, Type, TypeVar

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


class Model(ABC):
    """Base class for models."""

    model_type: ClassVar[str]
    model_format: ClassVar[str]

    @classmethod
    @abstractmethod
    def from_hdf5(cls, hdf5: h5py.File) -> 'Model':
        """Load a model from an open HDF5 file."""

    @classmethod
    def from_url(cls, url: str) -> 'Model':
        hdf5, new_url = _fetch_hdf5(url, cls.model_type)
        try:
            with hdf5:
                return cls.from_hdf5(hdf5)
        except ModelError as exc:
            exc.original_url = url
            exc.url = new_url
            raise


def _fetch_hdf5(
        url: str,
        model_type: str,
        get_options: Mapping[str, Any] = {}) -> h5py.File:
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
    h5 = h5py.File(io.BytesIO(data), 'r')
    actual_model_type = h5.attrs.get('model_type')
    if actual_model_type != model_type:
        raise ModelTypeError.with_urls(
            f'Expected a model of type {model_type!r}, not {actual_model_type!r}',
            url=url, original_url=original_url)
    return h5, url
