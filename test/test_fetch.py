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

import contextlib
import hashlib

import h5py
import pytest
import requests
import responses

from katsdpmodels import models, fetch
from test_utils import get_data, get_data_url, get_file_url, DummyModel


@pytest.mark.parametrize('url_gen', [get_data_url, get_file_url])
@pytest.mark.parametrize('filename', ['rfi_mask_ranges.hdf5', 'direct.alias', 'indirect.alias'])
def test_fetch_model_simple(url_gen, filename, mock_responses) -> None:
    url = url_gen(filename)
    with fetch.fetch_model(url, DummyModel) as model:
        assert len(model.ranges) == 2
        assert not model.is_closed
    assert model.is_closed


def test_fetch_model_alias_loop(mock_responses) -> None:
    url = get_data_url('loop.alias')
    with pytest.raises(models.TooManyAliasesError) as exc_info:
        fetch.fetch_model(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url


@pytest.mark.parametrize('filename', ['bad_model_type.hdf5', 'no_model_type.hdf5'])
def test_fetch_model_model_type_error(filename, mock_responses) -> None:
    url = get_data_url(filename)
    with pytest.raises(models.ModelTypeError) as exc_info:
        fetch.fetch_model(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url
    assert 'rfi_mask' in str(exc_info.value)


def test_fetch_model_cached_model_type_error(mock_responses, monkeypatch) -> None:
    class OtherModel(models.Model):
        model_type = 'other'

        @classmethod
        def from_hdf5(cls, hdf5: h5py.File) -> 'OtherModel':
            return cls()

    url = get_data_url('rfi_mask_ranges.hdf5')
    with fetch.Fetcher() as fetcher:
        fetcher.get(url, DummyModel)
        with pytest.raises(models.ModelTypeError) as exc_info:
            fetcher.get(url, OtherModel)
        assert exc_info.value.url == url
        assert exc_info.value.original_url == url


def test_fetch_model_not_hdf5(mock_responses) -> None:
    url = get_data_url('not_hdf5.hdf5')
    with pytest.raises(models.DataError) as exc_info:
        fetch.fetch_model(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url


def test_fetch_model_checksum_ok(mock_responses) -> None:
    data = get_data('rfi_mask_ranges.hdf5')
    digest = hashlib.sha256(data).hexdigest()
    url = get_data_url(f'sha256_{digest}.hdf5')
    mock_responses.add(responses.GET, url, body=data)
    with fetch.fetch_model(url, DummyModel) as model:
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


def test_fetch_model_http_redirect(mock_responses) -> None:
    url = get_data_url('subdir/redirect.alias')
    mock_responses.add(responses.GET, url, headers={'Location': '../direct.alias'}, status=307)
    with fetch.fetch_model(url, DummyModel) as model:
        assert len(model.ranges) == 2


def test_fetch_model_connection_error(mock_responses) -> None:
    # responses raises ConnectionError for any unregistered URL
    with pytest.raises(requests.ConnectionError):
        fetch.fetch_model(get_data_url('does_not_exist.hdf5'), DummyModel)


def test_fetcher_caching(mock_responses) -> None:
    with fetch.Fetcher() as fetcher:
        model1 = fetcher.get(get_data_url('rfi_mask_ranges.hdf5'), DummyModel)
        model2 = fetcher.get(get_data_url('indirect.alias'), DummyModel)
        model3 = fetcher.get(get_data_url('direct.alias'), DummyModel)
        assert model1 is model2
        assert model1 is model3
        assert not model1.is_closed
    assert len(mock_responses.calls) == 3
    assert model1.is_closed


class DummySession:
    def __init__(self) -> None:
        self._session = requests.Session()
        self.closed = False
        self.calls = 0

    def get(self, url: str) -> requests.Response:
        self.calls += 1
        return self._session.get(url)

    def close(self) -> None:
        self._session.close()
        self.closed = True


def test_custom_session(mock_responses):
    with contextlib.closing(DummySession()) as session:
        fetch.fetch_model(get_data_url('direct.alias'), DummyModel, session=session)
        assert session.calls == 2
        assert not session.closed
