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
from datetime import datetime, timezone
import io
import numbers
import urllib.parse
import h5py
from typing import Mapping, Optional, Any, ClassVar, Type, TypeVar, overload
from typing_extensions import Literal

import numpy as np
import numpy.lib.recfunctions
import strict_rfc3339


_E = TypeVar('_E', bound='ModelError')
_M = TypeVar('_M', bound='Model')
_H = TypeVar('_H', bound='SimpleHDF5Model')
_T = TypeVar('_T')


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
    """The file type (as determined by Content-Type or extension) was not recognised."""


class ModelVersionError(ModelError):
    """The ``model_version`` attribute was missing."""


class ModelTypeError(ModelError):
    """The ``model_type`` attribute was missing or did not match the expected value."""


class ModelFormatError(ModelError):
    """The ``model_format`` attribute was missing or did not match a known value."""


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

    Subclassing should generally be done in two layers:

    1. A class that defines `model_type` and defines the interface for that
       model type. This will be passed to :class:`.Fetcher` to indicate what
       model type is expected. Due to limitations in mypy, this should not
       use ``@abstractmethod`` for the interface methods.
    2. A concrete implementation that defines `model_format`.
    """

    model_type: ClassVar[str]
    model_format: ClassVar[str]
    checksum: Optional[str] = None
    target: Optional[str] = None
    comment: Optional[str] = None
    author: Optional[str] = None
    # It's required in loaded files, but optional here to allow models to be
    # built programmatically.
    version: Optional[int] = None
    created: Optional[datetime] = None

    @classmethod
    @abstractmethod
    def from_file(cls: Type[_M], file: io.IOBase, url: str, *,
                  content_type: Optional[str] = None) -> _M:
        """Load a model from raw data.

        On success, the callee takes responsibility for closing `file`, either
        within the function itself or the :meth:`close` method of the returned
        model.

        If `content_type` is given, it should be used to determine the file
        type; otherwise `url` may be used instead.
        """

    def close(self) -> None:
        """Close external resources associated with the model.

        Attempting to use the model after that results in undefined behavior.
        However, it is legal to call :meth:`close` multiple times.

        Models also implement the context manager protocol.
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
    The implementation of :meth:`from_hdf5` does not need to pull out the
    generic metadata (comment, target, author, created, version).
    """

    @classmethod
    def from_file(cls: Type[_H], file: io.IOBase, url: str, *,
                  content_type: Optional[str] = None) -> _H:
        with file:
            if content_type is not None:
                if content_type != 'application/x-hdf5':
                    raise FileTypeError(f'Expected application/x-hdf5, not {content_type}')
            else:
                parts = urllib.parse.urlparse(url)
                if not parts.path.endswith(('.h5', '.hdf5')):
                    raise FileTypeError(f'Filename extension not recognised in {url}')
            with h5py.File(file, 'r') as hdf5:
                model_type = get_hdf5_attr(hdf5.attrs, 'model_type', str)
                if model_type != cls.model_type:
                    raise ModelTypeError(
                        f'Expected a model of type {cls.model_type!r}, not {model_type!r}')
                model = cls.from_hdf5(hdf5)
                try:
                    model.comment = get_hdf5_attr(hdf5.attrs, 'model_comment', str)
                    model.target = get_hdf5_attr(hdf5.attrs, 'model_target', str)
                    model.author = get_hdf5_attr(hdf5.attrs, 'model_author', str)
                    try:
                        model.version = get_hdf5_attr(hdf5.attrs, 'model_version', int,
                                                      required=True)
                    except (KeyError, TypeError) as exc:
                        raise ModelVersionError(str(exc)) from exc
                    created = get_hdf5_attr(hdf5.attrs, 'model_created', str)
                    if created is not None:
                        try:
                            model.created = rfc3339_to_datetime(created)
                        except ValueError:
                            raise DataError(f'Invalid creation timestamp {created!r}') from None
                except Exception:
                    model.close()
                    raise
                else:
                    return model

    @classmethod
    @abstractmethod
    def from_hdf5(cls: Type[_H], hdf5: h5py.File) -> _H:
        """Load a model from an HDF5 file.

        Subclasses will implement this method, but it is not intended to be
        used directly.
        """


@overload
def get_hdf5_attr(attrs: Mapping[str, object], name: str, required_type: Type[_T], *,
                  required: Literal[True]) -> _T: ...


@overload
def get_hdf5_attr(attrs: Mapping[str, object], name: str, required_type: Type[_T], *,
                  required: bool = False) -> Optional[_T]: ...


def get_hdf5_attr(attrs, name, required_type, *, required=False):
    """Retrieve an attribute from an HDF5 object and verify its type.

    Pass the ``attrs`` attribute of the HDF5 file, group or dataset as the
    `attrs` parameter. If the `name` is not present, returns ``None``, unless
    ``required=True`` is passed.

    The implementation includes a workaround for
    https://github.com/h5py/h5py/issues/379, which will decode byte attributes
    to Unicode. It also turns numpy integer types into plain integers.

    Raises
    ------
    KeyError
        if `name` is not present and `required` is true.
    TypeError
        if `name` is present but is not of type `type`.
    UnicodeDecodeError
        if the attribute is :class:`bytes` that are not valid UTF-8 (only if
        `type` is :class:`str`).
    """
    try:
        value = attrs[name]
    except KeyError:
        if required:
            # The original message from h5py is less readable
            raise KeyError(f'attribute {name!r} is missing') from None
        else:
            return None
    actual_type = type(value)
    if actual_type == required_type:
        return value
    elif required_type == str and isinstance(value, bytes):
        return value.decode('utf-8')
    elif (required_type == int and isinstance(value, numbers.Integral)
            and not isinstance(value, bool)):
        return int(value)
    else:
        raise TypeError(f'Expected {required_type} for {name!r}, received {actual_type}')


def rfc3339_to_datetime(timestamp: str) -> datetime:
    """Convert a string in RFC 3339 format to a timezone-aware datetime object.

    The original timezone in the string is lost, and the returned datetime is
    based on UTC.
    """
    unix_time = strict_rfc3339.rfc3339_to_timestamp(timestamp)
    return datetime.fromtimestamp(unix_time, timezone.utc)


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
