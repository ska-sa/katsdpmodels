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

"""Tests for :mod:`katsdpmodels.fetch`."""

import hashlib

import pytest
import requests
import responses

from katsdpmodels import models, fetch
from test_utils import get_data, get_data_url, DummyModel, DummyModel2


@responses.activate
def test_fetch_model_simple() -> None:
    url = 'http://test.invalid/test/rfi_mask_ranges.hdf5'
    responses.add(responses.GET, url, body=get_data('rfi_mask_ranges.hdf5'))
    model = fetch.fetch_model(url, DummyModel)
    assert len(model.ranges) == 2


def test_fetch_model_file() -> None:
    url = get_data_url('rfi_mask_ranges.hdf5')
    model = fetch.fetch_model(url, DummyModel)
    assert len(model.ranges) == 2


@responses.activate
def test_fetch_model_alias() -> None:
    alias_url = 'http://test.invalid/test/blah/test.alias'
    real_url = 'http://test.invalid/test/rfi_mask_ranges.hdf5'
    responses.add(responses.GET, alias_url, body='../rfi_mask_ranges.hdf5')
    responses.add(responses.GET, real_url, body=get_data('rfi_mask_ranges.hdf5'))
    model = fetch.fetch_model(alias_url, DummyModel)
    assert len(model.ranges) == 2


@responses.activate
def test_fetch_model_alias_loop() -> None:
    url = 'http://test.invalid/test/blah/test.alias'
    responses.add(responses.GET, url, body='../blah/test.alias')
    with pytest.raises(models.TooManyAliasesError) as exc_info:
        fetch.fetch_model(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url


@responses.activate
def test_fetch_model_model_type_error() -> None:
    alias_url = 'http://test.invalid/test/blah/test.alias'
    real_url = 'http://test.invalid/test/rfi_mask_ranges.hdf5'
    responses.add(responses.GET, alias_url, body='../rfi_mask_ranges.hdf5')
    responses.add(responses.GET, real_url, body=get_data('rfi_mask_ranges.hdf5'))
    with pytest.raises(models.ModelTypeError) as exc_info:
        fetch.fetch_model(alias_url, DummyModel2)
    assert exc_info.value.url == real_url
    assert exc_info.value.original_url == alias_url
    assert 'rfi_mask' in str(exc_info.value)


@responses.activate
def test_fetch_model_checksum_ok() -> None:
    data = get_data('rfi_mask_ranges.hdf5')
    digest = hashlib.sha256(data).hexdigest()
    url = f'http://test.invalid/test/sha256_{digest}.hdf5'
    responses.add(responses.GET, url, body=data)
    model = fetch.fetch_model(url, DummyModel)
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
        fetch.fetch_model(url, DummyModel)
    assert exc_info.value.url == url


@responses.activate
def test_fetch_model_bad_http_status() -> None:
    url = 'http://test.invalid/test/rfi_mask_ranges.hdf5'
    responses.add(responses.GET, url, body=get_data('rfi_mask_ranges.hdf5'), status=404)
    with pytest.raises(requests.HTTPError) as exc_info:
        fetch.fetch_model(url, DummyModel)
    assert exc_info.value.response.status_code == 404


@responses.activate
def test_fetch_model_alias_bad_http_status() -> None:
    url = 'http://test.invalid/test/blah/test.alias'
    responses.add(responses.GET, url, body='test.alias', status=404)
    with pytest.raises(requests.HTTPError) as exc_info:
        fetch.fetch_model(url, DummyModel)
    assert exc_info.value.response.status_code == 404


@responses.activate
def test_fetch_model_connection_error() -> None:
    # responses raises ConnectionError for any unregistered URL
    with pytest.raises(requests.ConnectionError):
        fetch.fetch_model('http://test.invalid/test/rfi_mask_ranges.hdf5', DummyModel)
