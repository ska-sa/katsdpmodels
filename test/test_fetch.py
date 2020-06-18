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
import io

import h5py
import pytest
import requests
import responses

from katsdpmodels import models, fetch
from test_utils import get_data, get_data_url, get_file_url, DummyModel


@pytest.fixture
def http_file(web_server):
    with requests.Session() as session:
        with fetch.HttpFile(session, web_server('all_bytes.bin')) as file:
            yield file


def test_http_file_seek_tell(http_file):
    assert http_file.tell() == 0
    http_file.seek(10, io.SEEK_SET)
    assert http_file.tell() == 10
    assert http_file.read(2) == b'\x0A\x0B'
    assert http_file.tell() == 12
    http_file.seek(5)
    assert http_file.tell() == 5
    http_file.seek(5, io.SEEK_CUR)
    assert http_file.tell() == 10
    http_file.seek(-5, io.SEEK_CUR)
    assert http_file.tell() == 5
    http_file.seek(-10, io.SEEK_END)
    assert http_file.tell() == 246
    with pytest.raises(ValueError):
        http_file.seek(0, 17)
    with pytest.raises(OSError):
        http_file.seek(-10000, io.SEEK_END)


def test_http_file_close(http_file):
    assert not http_file.closed
    http_file.close()
    assert http_file.closed
    http_file.close()
    assert http_file.closed


def test_http_file_read(http_file):
    assert http_file.read(2) == b'\x00\x01'
    assert http_file.read(3) == b'\x02\x03\x04'
    http_file.seek(-2, io.SEEK_END)
    # Short read at end of file
    assert http_file.read(4) == b'\xFE\xFF'
    assert http_file.tell() == 256


def test_http_file_not_found(web_server):
    with requests.Session() as session:
        with pytest.raises(FileNotFoundError) as exc_info:
            fetch.HttpFile(session, web_server('does_not_exist'))
    assert exc_info.value.filename == web_server('does_not_exist')


def test_http_file_forbidden(mock_responses):
    url = get_data_url('does_not_exist')
    mock_responses.add(responses.HEAD, url, status=403)
    with requests.Session() as session:
        with pytest.raises(PermissionError) as exc_info:
            fetch.HttpFile(session, url)
    assert exc_info.value.filename == url


def test_http_file_ranges_not_accepted(mock_responses):
    url = get_data_url('rfi_mask_ranges.h5')
    with requests.Session() as session:
        with pytest.raises(OSError, match='Server does not accept byte ranges') as exc_info:
            fetch.HttpFile(session, url)
    assert exc_info.value.filename == url


def test_http_file_no_content_length(mock_responses):
    url = get_data_url('rfi_mask_ranges.h5')
    mock_responses.replace(responses.HEAD, url, content_type='application/x-hdf5',
                           headers={'Accept-Ranges': 'bytes'})
    with requests.Session() as session:
        with pytest.raises(OSError,
                           match='Server did not provide Content-Length header') as exc_info:
            fetch.HttpFile(session, url)
    assert exc_info.value.filename == url


def test_http_file_content_encoding(mock_responses):
    url = get_data_url('rfi_mask_ranges.h5')
    mock_responses.replace(
        responses.HEAD, url,
        headers={
            'Content-Length': str(len(get_data('rfi_mask_ranges.h5'))),
            'Accept-Ranges': 'bytes',
            'Content-Encoding': 'gzip'
        }
    )
    with requests.Session() as session:
        with pytest.raises(OSError,
                           match='Server provided Content-Encoding header') as exc_info:
            fetch.HttpFile(session, url)
    assert exc_info.value.filename == url


def test_http_file_content_encoding_get(mock_responses):
    url = get_data_url('rfi_mask_ranges.h5')
    mock_responses.replace(
        responses.HEAD, url,
        headers={
            'Content-Length': str(len(get_data('rfi_mask_ranges.h5'))),
            'Accept-Ranges': 'bytes'
        }
    )
    mock_responses.replace(
        responses.GET, url,
        headers={
            'Content-Length': str(len(get_data('rfi_mask_ranges.h5'))),
            'Accept-Ranges': 'bytes',
            'Content-Encoding': 'gzip'
        }
    )
    with requests.Session() as session:
        with fetch.HttpFile(session, url) as file:
            with pytest.raises(OSError, match='Server provided Content-Encoding header'):
                file.read()


def test_http_file_range_ignored(mock_responses):
    url = get_data_url('rfi_mask_ranges.h5')
    data = get_data('rfi_mask_ranges.h5')
    mock_responses.replace(
        responses.HEAD, url,
        headers={
            'Accept-Ranges': 'bytes',
            'Content-Length': str(len(data))
        }
    )
    mock_responses.replace(responses.GET, url, body=data, stream=True)
    with requests.Session() as session:
        with fetch.HttpFile(session, url) as file:
            with pytest.raises(OSError, match='Did not receive expected byte range'):
                file.read(10)
            # Reading the whole file should work even if the server doesn't send
            # back partial content.
            file.seek(0)
            test_data = file.read(len(data))
            assert test_data == data


@pytest.mark.parametrize('use_file', [True, False])
@pytest.mark.parametrize('filename', ['rfi_mask_ranges.h5', 'direct.alias', 'indirect.alias'])
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


def test_fetch_model_too_many_aliases(web_server, monkeypatch) -> None:
    monkeypatch.setattr(fetch, 'MAX_ALIASES', 1)
    with pytest.raises(models.TooManyAliasesError):
        fetch.fetch_model(web_server('indirect.alias'), DummyModel)
    # Check that 1 level of alias is still permitted
    with fetch.fetch_model(web_server('direct.alias'), DummyModel):
        pass


@pytest.mark.parametrize('filename', ['bad_model_type.h5', 'no_model_type.h5'])
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

    url = web_server('rfi_mask_ranges.h5')
    with fetch.Fetcher() as fetcher:
        fetcher.get(url, DummyModel)
        with pytest.raises(models.ModelTypeError) as exc_info:
            fetcher.get(url, OtherModel)
        assert exc_info.value.url == url
        assert exc_info.value.original_url == url


@pytest.mark.parametrize('lazy', [True, False])
def test_fetch_model_bad_content_type(mock_responses, lazy) -> None:
    data = get_data('rfi_mask_ranges.h5')
    url = get_data_url('bad_content_type.h5')
    mock_responses.add(
        responses.HEAD, url, content_type='image/png',
        headers={
            'Accept-Ranges': 'bytes',
            'Content-Length': str(len(data))
        }
    )
    mock_responses.add(responses.GET, url, content_type='image/png', body=data)
    with pytest.raises(models.FileTypeError,
                       match='Expected application/x-hdf5, not image/png') as exc_info:
        with fetch.Fetcher() as fetcher:
            fetcher.get(url, DummyModel, lazy=lazy)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url


@pytest.mark.parametrize('lazy', [True, False])
def test_fetch_model_bad_extension(web_server, lazy) -> None:
    url = web_server('wrong_extension.blah')
    with pytest.raises(models.FileTypeError) as exc_info:
        with fetch.Fetcher() as fetcher:
            fetcher.get(url, DummyModel, lazy=lazy)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url


def test_fetch_model_not_hdf5(web_server) -> None:
    url = web_server('not_hdf5.h5')
    with pytest.raises(models.DataError) as exc_info:
        fetch.fetch_model(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url


def test_fetch_model_checksum_ok(mock_responses) -> None:
    data = get_data('rfi_mask_ranges.h5')
    digest = hashlib.sha256(data).hexdigest()
    url = get_data_url(f'sha256_{digest}.h5')
    mock_responses.add(responses.GET, url, content_type='application/x-hdf5', body=data)
    with fetch.fetch_model(url, DummyModel) as model:
        assert model.checksum == digest


def test_fetch_model_checksum_bad(mock_responses) -> None:
    data = get_data('rfi_mask_ranges.h5')
    digest = hashlib.sha256(data).hexdigest()
    url = get_data_url(f'sha256_{digest}.h5')
    # Now invalidate the digest
    data += b'blahblahblah'
    mock_responses.add(responses.GET, url, content_type='application/x-hdf5', body=data)
    with pytest.raises(models.ChecksumError) as exc_info:
        fetch.fetch_model(url, DummyModel)
    assert exc_info.value.url == url


@pytest.mark.parametrize('filename', ['does_not_exist.h5', 'does_not_exist.alias'])
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
        fetch.fetch_model(get_data_url('does_not_exist.h5'), DummyModel)


def test_fetcher_caching(mock_responses) -> None:
    with fetch.Fetcher() as fetcher:
        model1 = fetcher.get(get_data_url('rfi_mask_ranges.h5'), DummyModel)
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


def test_fetcher_custom_session(web_server) -> None:
    with contextlib.closing(DummySession()) as session:
        with fetch.Fetcher(session=session) as fetcher:
            assert fetcher.session is session
            fetcher.get(web_server('direct.alias'), DummyModel)
        assert session.calls == 2
        assert not session.closed


def test_custom_session(web_server) -> None:
    with contextlib.closing(DummySession()) as session:
        fetch.fetch_model(web_server('direct.alias'), DummyModel, session=session)
        assert session.calls == 2
        assert not session.closed


def test_lazy(web_server) -> None:
    with fetch.Fetcher() as fetcher:
        model = fetcher.get(web_server('rfi_mask_ranges.h5'), DummyModel, lazy=True)
        assert len(model.ranges) == 2
        assert not model.is_closed
    assert model.is_closed
