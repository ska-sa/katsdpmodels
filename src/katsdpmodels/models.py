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

"""Base functionality common to all model types."""

from abc import ABC, abstractmethod
import io
import urllib
import h5py
from typing import BinaryIO, Optional, Union, Any, ClassVar, Type, TypeVar

import numpy as np


_E = TypeVar('_E', bound='ModelError')
_M = TypeVar('_M', bound='Model')
_H = TypeVar('_H', bound='SimpleHDF5Model')


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


class FileTypeError(ModelError):
    """The file type (as determined by extension) was not recognised."""


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

    Models can either be loaded from file-like objects or constructed directly.
    Models loaded by the fetcher store the checksum of the raw data, and are
    considered equal if the checksums match. Otherwise, equality is by object
    identity.

    Models loaded by the fetcher should not be modified, as they may be shared
    by other users. Instead, make a copy and modify that.
    """

    model_type: ClassVar[str]
    model_format: ClassVar[str]
    checksum: Optional[str] = None

    @classmethod
    @abstractmethod
    def from_file(cls: Type[_M], file: BinaryIO, url: str) -> _M:
        """Load a model from raw data.

        On success, the callee takes responsibility for closing `file`, either
        within the function itself or the :meth:`close` method of the returned
        model.

        The `url` may be used to determine the file type.
        """

    def close(self) -> None:
        """Close external resources associated with the model.

        Attempting to use the model after that results in undefined behavior.
        However, it is legal to call :meth:`close` multiple times.
        """

    def __enter__(self: _M) -> _M:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

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


class SimpleHDF5Model(Model):
    """Helper base class for models that load data from HDF5.

    It does not handle lazy loading: the :meth:`from_hdf5` class method must
    load all the data out of the HDF5 file as it will be closed by the caller.
    """

    @classmethod
    def from_file(cls: Type[_H], file: BinaryIO, url: str) -> _H:
        """Load a model from raw data.

        On success, the callee takes responsibility for closing `file`, either
        within the function itself or the :meth:`close` method of the returned
        model.

        The `url` may be used to determine the file type.
        """
        with file:
            parts = urllib.parse.urlparse(url)
            if not parts.path.endswith(('.h5', '.hdf5')):
                raise FileTypeError(f'Filename extension not recognised in {url}')
            with h5py.File(file, 'r') as hdf5:
                model_type = ensure_str(hdf5.attrs.get('model_type', ''))
                if model_type != cls.model_type:
                    raise ModelTypeError(
                        f'Expected a model of type {cls.model_type!r}, not {model_type!r}')
                return cls.from_hdf5(hdf5)

    @classmethod
    @abstractmethod
    def from_hdf5(cls: Type[_H], hdf5: h5py.File) -> _H:
        """Load a model from an HDF5 file."""


def ensure_str(s: Union[bytes, str]) -> str:
    """Decode bytes to string if necessary.

    This is provided to work around for https://github.com/h5py/h5py/issues/379.

    Raises
    ------
    TypeError
        if `s` is neither :class:`bytes` nor :class:`str`.
    UnicodeDecodeError
        if `s` is :class:`bytes` and is not valid UTF-8.
    """
    if isinstance(s, str):
        return s
    elif isinstance(s, bytes):
        return s.decode('utf-8')
    else:
        raise TypeError('Expected bytes or str, received {}'.format(type(s)))


def require_columns(array: Any, dtype: np.dtype) -> Any:
    """Validate the columns in a table.

    The `dtype` is the expected dtype, which must be a structured dtype. The
    array is checked for compatibility: it must have all the required fields
    (but may have more), and they must be castable to the the expected dtype.

    The return value is the input array restricted to the required columns
    and cast to the required dtype. It may be either a view or a copy,
    depending on whether any casting was required.

    This function is not designed to support nested structuring, and will not
    recursively filter out unwanted sub-structures.

    Raises
    ------
    DataError
        if the types are not compatible
    """
    if array.dtype == dtype:
        return np.asanyarray(array)
    if array.dtype.names is None:
        raise DataError('Array does not have named columns')
    for name in dtype.names:
        if name not in array.dtype.names:
            raise DataError(f'Column {name} is missing')
        if not np.can_cast(array.dtype[name], dtype[name], 'same_kind'):
            raise DataError(f'Column {name} has type {array.dtype[name]}, expected {dtype[name]}')
    out = np.empty(array.shape, dtype)
    np.lib.recfunctions.assign_fields_by_name(out, array)
    return out
