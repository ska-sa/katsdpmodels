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
from test_utils import get_data, get_data_url, get_file_url, DummyModel, DummyModel2


@pytest.mark.parametrize('url_gen', [get_data_url, get_file_url])
@pytest.mark.parametrize('filename', ['rfi_mask_ranges.hdf5', 'direct.alias', 'indirect.alias'])
def test_fetch_model_simple(url_gen, filename, mock_responses) -> None:
    url = url_gen(filename)
    model = fetch.fetch_model(url, DummyModel)
    assert len(model.ranges) == 2


def test_fetch_model_alias_loop(mock_responses) -> None:
    url = get_data_url('loop.alias')
    with pytest.raises(models.TooManyAliasesError) as exc_info:
        fetch.fetch_model(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url


@pytest.mark.parametrize('filename', ['rfi_mask_ranges.hdf5', 'direct.alias'])
def test_fetch_model_model_type_error(filename, mock_responses) -> None:
    url = get_data_url(filename)
    with pytest.raises(models.ModelTypeError) as exc_info:
        fetch.fetch_model(url, DummyModel2)
    assert exc_info.value.url == get_data_url('rfi_mask_ranges.hdf5')
    assert exc_info.value.original_url == url
    assert 'rfi_mask' in str(exc_info.value)


def test_fetch_model_checksum_ok(mock_responses) -> None:
    data = get_data('rfi_mask_ranges.hdf5')
    digest = hashlib.sha256(data).hexdigest()
    url = get_data_url(f'sha256_{digest}.hdf5')
    mock_responses.add(responses.GET, url, body=data)
    model = fetch.fetch_model(url, DummyModel)
    assert model.checksum == digest


def test_fetch_model_checksum_bad(mock_responses) -> None:
    data = get_data('rfi_mask_ranges.hdf5')
    digest = hashlib.sha256(data).hexdigest()
    url = get_data_url(f'sha256_{digest}.hdf5')
    # Now invalidate the digest
    data += b'blahblahblah'
    mock_responses.add(responses.GET, url, body=data)
    with pytest.raises(models.ChecksumError) as exc_info:
        fetch.fetch_model(url, DummyModel)
    assert exc_info.value.url == url


@pytest.mark.parametrize('filename', ['rfi_mask_ranges.hdf5', 'direct.alias'])
def test_fetch_model_bad_http_status(filename, mock_responses) -> None:
    url = get_data_url(filename)
    mock_responses.replace(
        responses.GET,
        get_data_url('rfi_mask_ranges.hdf5'),
        body=get_data('rfi_mask_ranges.hdf5'),
        status=404
    )
    with pytest.raises(requests.HTTPError) as exc_info:
        fetch.fetch_model(url, DummyModel)
    assert exc_info.value.response.status_code == 404


def test_fetch_model_connection_error(mock_responses) -> None:
    # responses raises ConnectionError for any unregistered URL
    with pytest.raises(requests.ConnectionError):
        fetch.fetch_model(get_data_url('does_not_exist.hdf5'), DummyModel)
