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
from typing import Optional, Union, Any, ClassVar, Type, TypeVar

import numpy as np
import h5py


_E = TypeVar('_E', bound='ModelError')
_M = TypeVar('_M', bound='Model')


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

    Models can either be loaded from instances of :class:`h5py.File` or
    constructed directly. Models loaded by the fetcher store the checksum of
    the raw data, and are considered equal if the checksums match. Otherwise,
    equality is by object identity.

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
