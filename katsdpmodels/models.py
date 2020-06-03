"""Base functionality common to all model types."""

from abc import ABC, abstractmethod
import io
import logging
import urllib.parse
from typing import Mapping, Optional, Any, ClassVar

import h5py
import requests
import requests_file


_logger = logging.getLogger(__name__)


class ModelError(ValueError):
    """A model was found, but the content was incorrect."""

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.original_url: Optional[str] = None
        self.url: Optional[str] = None


class ModelTypeError(ModelError):
    """The ``model_type`` attribute was missing or did not match the expected value."""


class ModelFormatError(ModelError):
    """The ``model_format`` attribute was missing or did match a known value."""


class DataError(ModelError):
    """The model was missing some data or it had the wrong format."""


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
    with requests.session() as session:
        session.mount('file://', requests_file.FileAdapter())
        while True:
            parts = urllib.parse.urlparse(url)
            with session.get(url, **get_options) as resp:
                if parts.path.endswith('.alias'):
                    rel_path = resp.text.rstrip()
                    new_url = urllib.parse.urljoin(url, rel_path)
                    _logger.debug('Redirecting from %s to %s', url, new_url)
                    url = new_url
                    continue
                data = resp.content
            break

    # TODO: validate checksum if embedded in URL
    h5 = h5py.File(io.BytesIO(data), 'r')
    actual_model_type = h5.attrs.get('model_type')
    if actual_model_type != model_type:
        raise ValueError(f'Expected a model of type {model_type!r}, not {actual_model_type!r}')
    return h5, url
