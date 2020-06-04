"""Tests for :mod:`katsdpmodels.models`."""

import os
import pathlib
import hashlib
from typing import Any, ClassVar
from typing_extensions import Literal

import pytest
import h5py
import responses

from katsdpmodels import models


DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


def get_data(filename: str) -> bytes:
    with open(os.path.join(DATA_DIR, filename), 'rb') as f:
        return f.read()


def get_data_url(filename: str) -> str:
    path = os.path.join(DATA_DIR, 'rfi_mask_ranges.hdf5')
    return pathlib.PurePath(path).as_uri()


class DummyModel(models.Model):
    model_type: ClassVar[Literal['rfi_mask']] = 'rfi_mask'

    def __init__(self, ranges: Any) -> None:
        self.ranges = ranges

    @classmethod
    def from_hdf5(cls, hdf5: h5py.File) -> 'DummyModel':
        return cls(hdf5['/ranges'][:])


class DummyModel2(models.Model):
    model_type: ClassVar[Literal['something_else']] = 'something_else'

    # Implement this just so that it's not an abstract class
    @classmethod
    def from_hdf5(cls, hdf5: h5py.File) -> 'DummyModel2':
        return cls()


@responses.activate
def test_fetch_model_simple() -> None:
    url = 'http://test.invalid/test/rfi_mask_ranges.hdf5'
    responses.add(responses.GET, url, body=get_data('rfi_mask_ranges.hdf5'))
    model = models.fetch_model(url, DummyModel)
    assert len(model.ranges) == 2


def test_fetch_model_file() -> None:
    url = get_data_url('rfi_mask_ranges.hdf5')
    model = models.fetch_model(url, DummyModel)
    assert len(model.ranges) == 2


@responses.activate
def test_fetch_model_alias() -> None:
    alias_url = 'http://test.invalid/test/blah/test.alias'
    real_url = 'http://test.invalid/test/rfi_mask_ranges.hdf5'
    responses.add(responses.GET, alias_url, body='../rfi_mask_ranges.hdf5')
    responses.add(responses.GET, real_url, body=get_data('rfi_mask_ranges.hdf5'))
    model = models.fetch_model(alias_url, DummyModel)
    assert len(model.ranges) == 2


@responses.activate
def test_fetch_model_alias_loop() -> None:
    url = 'http://test.invalid/test/blah/test.alias'
    responses.add(responses.GET, url, body='../blah/test.alias')
    with pytest.raises(models.TooManyAliasesError) as exc_info:
        models.fetch_model(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url


@responses.activate
def test_fetch_model_model_type_error() -> None:
    alias_url = 'http://test.invalid/test/blah/test.alias'
    real_url = 'http://test.invalid/test/rfi_mask_ranges.hdf5'
    responses.add(responses.GET, alias_url, body='../rfi_mask_ranges.hdf5')
    responses.add(responses.GET, real_url, body=get_data('rfi_mask_ranges.hdf5'))
    with pytest.raises(models.ModelTypeError) as exc_info:
        models.fetch_model(alias_url, DummyModel2)
    assert exc_info.value.url == real_url
    assert exc_info.value.original_url == alias_url
    assert 'rfi_mask' in str(exc_info.value)


@responses.activate
def test_fetch_model_checksum_ok() -> None:
    data = get_data('rfi_mask_ranges.hdf5')
    digest = hashlib.sha256(data).hexdigest()
    url = f'http://test.invalid/test/sha256_{digest}.hdf5'
    responses.add(responses.GET, url, body=data)
    model = models.fetch_model(url, DummyModel)
    assert model.checksum == digest


@responses.activate
def test_fetch_model_checksum_bad() -> None:
    data = get_data('rfi_mask_ranges.hdf5')
    digest = hashlib.sha256(data).hexdigest()
    url = f'http://test.invalid/test/sha256_{digest}.hdf5'
    # Now invalidate the digest
    data += b'blahblahblah'
    responses.add(responses.GET, url, body=data)
    with pytest.raises(models.ChecksumError) as exc_info:
        models.fetch_model(url, DummyModel)
    assert exc_info.value.url == url


@responses.activate
def test_eq_hash() -> None:
    data = get_data('rfi_mask_ranges.hdf5')
    url1 = 'http://test.invalid/test/rfi_mask_ranges.hdf5'
    url2 = 'http://test.invalid/another_test.hdf5'
    responses.add(responses.GET, url1, body=data)
    responses.add(responses.GET, url2, body=data)
    with models.Fetcher() as fetcher:
        model1 = fetcher.get(url1, DummyModel)
        model2 = fetcher.get(url2, DummyModel)
    model3 = DummyModel(None)
    model4 = DummyModel(None)
    assert model1 == model1
    assert model1 == model2
    assert model1 != model3
    assert model3 == model3
    assert model3 != model4
    assert hash(model1) == hash(model2)
    assert hash(model1) != hash(model3)
    assert hash(model3) != hash(model4)
