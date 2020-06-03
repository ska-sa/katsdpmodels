"""Tests for :mod:`katsdpmodels.models`."""

import os
import pathlib

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


@responses.activate
def test_fetch_hdf5_simple():
    url = 'http://example.com/test/rfi_mask_ranges.hdf5'
    responses.add(responses.GET, url, body=get_data('rfi_mask_ranges.hdf5'))
    hdf5, new_url = models._fetch_hdf5(url, 'rfi_mask')
    assert new_url == url
    assert isinstance(hdf5, h5py.File)
    assert hdf5.attrs['model_format'] == 'ranges'


def test_fetch_hdf5_file():
    url = get_data_url('rfi_mask_ranges.hdf5')
    hdf5, new_url = models._fetch_hdf5(url, 'rfi_mask')
    assert new_url == url
    assert isinstance(hdf5, h5py.File)
    assert hdf5.attrs['model_format'] == 'ranges'


@responses.activate
def test_fetch_hdf5_alias():
    alias_url = 'http://example.com/test/blah/test.alias'
    real_url = 'http://example.com/test/rfi_mask_ranges.hdf5'
    responses.add(responses.GET, alias_url, body='../rfi_mask_ranges.hdf5')
    responses.add(responses.GET, real_url, body=get_data('rfi_mask_ranges.hdf5'))
    hdf5, new_url = models._fetch_hdf5(alias_url, 'rfi_mask')
    assert new_url == real_url
    assert isinstance(hdf5, h5py.File)
    assert hdf5.attrs['model_format'] == 'ranges'


@responses.activate
def test_fetch_hdf5_alias_loop():
    url = 'http://example.com/test/blah/test.alias'
    responses.add(responses.GET, url, body='../blah/test.alias')
    with pytest.raises(models.TooManyAliasesError) as exc_info:
        models._fetch_hdf5(url, 'rfi_mask')
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url


@responses.activate
def test_fetch_hdf5_model_type_error():
    alias_url = 'http://example.com/test/blah/test.alias'
    real_url = 'http://example.com/test/rfi_mask_ranges.hdf5'
    responses.add(responses.GET, alias_url, body='../rfi_mask_ranges.hdf5')
    responses.add(responses.GET, real_url, body=get_data('rfi_mask_ranges.hdf5'))
    with pytest.raises(models.ModelTypeError) as exc_info:
        models._fetch_hdf5(alias_url, 'bad_type')
    assert exc_info.value.url == real_url
    assert exc_info.value.original_url == alias_url
    assert 'rfi_mask' in str(exc_info.value)
