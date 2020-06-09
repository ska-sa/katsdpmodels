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


@pytest.mark.parametrize('use_file', [True, False])
@pytest.mark.parametrize('filename', ['rfi_mask_ranges.hdf5', 'direct.alias', 'indirect.alias'])
def test_fetch_model_simple(use_file, filename, web_server) -> None:
    url = get_file_url(filename) if use_file else web_server(filename)
    with fetch.fetch_model(url, DummyModel) as model:
        assert len(model.ranges) == 2
        assert not model.is_closed
    assert model.is_closed


def test_fetch_model_alias_loop(web_server) -> None:
    url = web_server('loop.alias')
    with pytest.raises(models.TooManyAliasesError) as exc_info:
        fetch.fetch_model(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url


@pytest.mark.parametrize('filename', ['bad_model_type.hdf5', 'no_model_type.hdf5'])
def test_fetch_model_model_type_error(filename, web_server) -> None:
    url = web_server(filename)
    with pytest.raises(models.ModelTypeError) as exc_info:
        fetch.fetch_model(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url
    assert 'rfi_mask' in str(exc_info.value)


def test_fetch_model_cached_model_type_error(web_server) -> None:
    class OtherModel(models.SimpleHDF5Model):
        model_type = 'other'

        @classmethod
        def from_hdf5(cls, hdf5: h5py.File) -> 'OtherModel':
            return cls()

    url = web_server('rfi_mask_ranges.hdf5')
    with fetch.Fetcher() as fetcher:
        fetcher.get(url, DummyModel)
        with pytest.raises(models.ModelTypeError) as exc_info:
            fetcher.get(url, OtherModel)
        assert exc_info.value.url == url
        assert exc_info.value.original_url == url


def test_fetch_model_file_type_error(web_server) -> None:
    url = web_server('wrong_extension.blah')
    with pytest.raises(models.FileTypeError) as exc_info:
        fetch.fetch_model(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url


def test_fetch_model_not_hdf5(web_server) -> None:
    url = web_server('not_hdf5.hdf5')
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


@pytest.mark.parametrize('filename', ['does_not_exist.hdf5', 'does_not_exist.alias'])
def test_fetch_model_bad_http_status(filename, web_server) -> None:
    url = web_server(filename)
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

    def get(self, url: str, **kwargs) -> requests.Response:
        self.calls += 1
        return self._session.get(url, **kwargs)

    def head(self, url: str, **kwargs) -> requests.Response:
        self.calls += 1
        return self._session.head(url, **kwargs)

    def close(self) -> None:
        self._session.close()
        self.closed = True


def test_custom_session(web_server) -> None:
    with contextlib.closing(DummySession()) as session:
        fetch.fetch_model(web_server('direct.alias'), DummyModel, session=session)
        assert session.calls == 2
        assert not session.closed
