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

"""Tests for :mod:`katsdpmodels.models`."""

from datetime import datetime, timezone
import hashlib
import io
import pathlib
from typing import Dict, Type, Union

import numpy as np
import pytest

from katsdpmodels import models
from test_utils import DummyModel


@pytest.fixture
def dummy_model() -> DummyModel:
    ranges = np.array(
       [(1, 4.5), (2, -5.5)],
       dtype=[('a', 'i4'), ('b', 'f8')]
    )
    model = DummyModel(ranges)
    model.author = 'Test author'
    model.comment = 'Test comment'
    model.target = 'Test target'
    model.version = 1
    model.created = datetime(2020, 6, 15, 14, 11, tzinfo=timezone.utc)
    return model


def test_eq_hash() -> None:
    # model1 and model2 have the same checksum, model3 a different checksum
    model1 = DummyModel(None)
    model2 = DummyModel(None)
    model3 = DummyModel(None)
    model4 = DummyModel(None)
    model1.checksum = hashlib.sha256(b'foo').hexdigest()
    model2.checksum = model1.checksum
    model3.checksum = hashlib.sha256(b'bar').hexdigest()
    assert model1 == model1
    assert model1 == model2
    assert model1 != model3
    assert model3 == model3
    assert model3 != model4
    assert model4 == model4
    assert hash(model1) == hash(model2)
    assert hash(model1) != hash(model3)
    assert hash(model3) != hash(model4)
    assert model1 != 1
    assert model4 != 1


def test_get_hdf5_attr_missing() -> None:
    attrs: Dict[str, object] = {}
    assert models.get_hdf5_attr(attrs, 'missing', str) is None
    with pytest.raises(KeyError):
        models.get_hdf5_attr({}, 'missing', str, required=True)


def test_get_hdf5_attr_type_mismatch() -> None:
    with pytest.raises(TypeError, match="Expected <class 'int'> for 'foo', received <class 'str'>"):
        models.get_hdf5_attr({'foo': 'bar'}, 'foo', int)


def test_get_hdf5_attr_decode_bytes() -> None:
    assert models.get_hdf5_attr({'foo': 'café'.encode()}, 'foo', str) == 'café'


def test_get_hdf5_attr_numpy_int() -> None:
    assert models.get_hdf5_attr({'foo': np.int64(1)}, 'foo', int) == 1


def test_get_hdf5_attr_bad_utf8() -> None:
    with pytest.raises(UnicodeDecodeError):
        models.get_hdf5_attr({'foo': b'\xff'}, 'foo', str)
    # Should raise the type error first if we aren't expecting strings
    with pytest.raises(TypeError):
        models.get_hdf5_attr({'foo': b'\xff'}, 'foo', int)


def test_get_hdf5_attr_bool_not_int() -> None:
    with pytest.raises(TypeError):
        models.get_hdf5_attr({'foo': True}, 'foo', int)
    with pytest.raises(TypeError):
        models.get_hdf5_attr({'foo': 2}, 'foo', bool)


def test_get_hdf5_attr_success() -> None:
    attrs = {'string': 'hello', 'int': 42}
    assert models.get_hdf5_attr(attrs, 'string', str) == 'hello'
    assert models.get_hdf5_attr(attrs, 'int', int) == 42


def test_require_columns_missing_column() -> None:
    dtype1 = np.dtype([('a', 'f8'), ('b', 'i4')])
    dtype2 = np.dtype([('a', 'f8'), ('c', 'i4')])
    array = np.zeros((5,), dtype1)
    with pytest.raises(models.DataError, match='Column c is missing'):
        models.require_columns(array, dtype2)


def test_require_columns_unstructured() -> None:
    dtype = np.dtype([('a', 'f8'), ('b', 'i4')])
    array = np.zeros((5,), np.float32)
    with pytest.raises(models.DataError, match='Array does not have named columns'):
        models.require_columns(array, dtype)


def test_require_columns_dtype_mismatch() -> None:
    dtype1 = np.dtype([('a', 'f8')])
    dtype2 = np.dtype([('a', 'i4')])
    array = np.zeros((5,), dtype1)
    with pytest.raises(models.DataError, match='Column a has type float64, expected int32'):
        models.require_columns(array, dtype2)


def test_require_columns_same_dtype() -> None:
    dtype = np.dtype([('a', 'f8'), ('b', 'i4')])
    array = np.array([(1.5, 1), (3.5, 3)], dtype=dtype)
    out = models.require_columns(array, dtype)
    np.testing.assert_array_equal(array, out)
    assert np.shares_memory(array, out)


def test_require_columns_change_byteorder() -> None:
    dtype1 = np.dtype([('a', '>f8')])
    dtype2 = np.dtype([('a', '<f8')])
    array = np.array([1.0, 1.5, 2.0], dtype=dtype1)
    out = models.require_columns(array, dtype2)
    np.testing.assert_array_equal(array, out)
    # Some versions of numpy considered dtypes to be equal even if the byte
    # order was different, so explicitly compare byte order.
    assert out.dtype['a'].byteorder == dtype2['a'].byteorder


def test_require_columns_extra_column() -> None:
    dtype1 = np.dtype([('a', 'f8'), ('b', 'f8')])
    dtype2 = np.dtype([('b', 'f8')])
    array = np.array([(1.0, 10.0), (1.5, 15.0), (2.0, 20.0)], dtype=dtype1)
    expected = np.array([10.0, 15.0, 20.0], dtype=dtype2)
    out = models.require_columns(array, dtype2)
    np.testing.assert_equal(out, expected)


def assert_models_equal(model1: DummyModel, model2: DummyModel):
    assert type(model1) == type(model2)
    assert model1.model_type == model2.model_type
    assert model1.model_format == model2.model_format
    assert model1.comment == model2.comment
    assert model1.author == model2.author
    assert model1.target == model2.target
    assert model1.created == model2.created
    assert model1.version == model2.version
    np.testing.assert_array_equal(model1.ranges, model2.ranges)


@pytest.mark.parametrize('clear_metadata', [False, True])
def test_hdf5_to_file(clear_metadata: bool, dummy_model: DummyModel) -> None:
    if clear_metadata:
        dummy_model.comment = None
        dummy_model.author = None
        dummy_model.target = None
        dummy_model.created = None
    fh = io.BytesIO()
    dummy_model.to_file(fh, content_type='application/x-hdf5')
    fh.seek(0)
    new_model = DummyModel.from_file(fh, 'http://test.invalid/dummy.h5',
                                     content_type='application/x-hdf5')
    assert_models_equal(dummy_model, new_model)


def test_hdf5_to_file_no_version(dummy_model: DummyModel) -> None:
    dummy_model.version = None
    with pytest.raises(ValueError):
        dummy_model.to_file(io.BytesIO(), content_type='application/x-hdf5')


def test_hdf5_to_file_no_content_type_or_filename(dummy_model: DummyModel) -> None:
    with pytest.raises(AttributeError):
        dummy_model.to_file(io.BytesIO())


def test_hdf5_to_file_bad_content_type(dummy_model: DummyModel) -> None:
    with pytest.raises(models.FileTypeError, match='Expected application/x-hdf5, not image/png'):
        dummy_model.to_file(io.BytesIO(), content_type='image/png')


@pytest.mark.parametrize('path_type', [pathlib.Path, str])
def test_hdf5_to_file_bad_extension(path_type: Union[Type[pathlib.Path], Type[str]],
                                    dummy_model: DummyModel,
                                    tmp_path: pathlib.Path) -> None:
    with pytest.raises(models.FileTypeError,
                       match=r'Expected extension of \.h5 or \.hdf5, not \.foo'):
        dummy_model.to_file(path_type(tmp_path / 'test.foo'))


@pytest.mark.parametrize('path_type', [pathlib.Path, str])
def test_hdf5_to_file_good_extension(path_type: Union[Type[pathlib.Path], Type[str]],
                                     dummy_model: DummyModel,
                                     tmp_path: pathlib.Path) -> None:
    path = path_type(tmp_path / 'test.h5')
    dummy_model.to_file(path)
    new_model = dummy_model.from_file(open(path, 'rb'), 'test.h5',
                                      content_type='application/x-hdf5')
    assert_models_equal(dummy_model, new_model)
