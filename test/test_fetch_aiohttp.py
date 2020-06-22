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

"""Tests for :mod:`katsdpmodels.fetch.aiohttp`."""

import hashlib
import types
from typing import List

import h5py
import pytest
import aiohttp
import yarl

from katsdpmodels import models, fetch
import katsdpmodels.fetch.aiohttp as fetch_aiohttp
from test_utils import get_data, get_data_url, get_file_url, DummyModel


@pytest.mark.parametrize('use_file', [True, False])
@pytest.mark.parametrize('filename', ['rfi_mask_ranges.h5', 'direct.alias', 'indirect.alias'])
@pytest.mark.asyncio
async def test_fetch_model_simple(use_file, filename, web_server) -> None:
    url = get_file_url(filename) if use_file else web_server(filename)
    with await fetch_aiohttp.fetch_model(url, DummyModel) as model:
        assert len(model.ranges) == 2
        assert not model.is_closed
    assert model.is_closed


@pytest.mark.asyncio
async def test_fetch_model_alias_loop(web_server) -> None:
    url = web_server('loop.alias')
    with pytest.raises(models.TooManyAliasesError) as exc_info:
        await fetch_aiohttp.fetch_model(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url


@pytest.mark.asyncio
async def test_fetch_model_too_many_aliases(web_server, monkeypatch) -> None:
    monkeypatch.setattr(fetch, 'MAX_ALIASES', 1)
    with pytest.raises(models.TooManyAliasesError):
        await fetch_aiohttp.fetch_model(web_server('indirect.alias'), DummyModel)
    # Check that 1 level of alias is still permitted
    with await fetch_aiohttp.fetch_model(web_server('direct.alias'), DummyModel):
        pass


@pytest.mark.asyncio
async def test_fetch_model_to_file_alias(web_server) -> None:
    url = web_server('to_file.alias')
    with pytest.raises(models.LocalRedirectError) as exc_info:
        await fetch_aiohttp.fetch_model(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url


@pytest.mark.parametrize('filename', ['bad_model_type.h5', 'no_model_type.h5'])
@pytest.mark.asyncio
async def test_fetch_model_model_type_error(filename, web_server) -> None:
    url = web_server(filename)
    with pytest.raises(models.ModelTypeError) as exc_info:
        await fetch_aiohttp.fetch_model(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url
    assert 'rfi_mask' in str(exc_info.value)


@pytest.mark.asyncio
async def test_fetch_model_bad_created(web_server) -> None:
    url = web_server('bad_created.h5')
    with pytest.raises(models.DataError, match='Invalid creation timestamp') as exc_info:
        await fetch_aiohttp.fetch_model(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url


@pytest.mark.parametrize('filename', ['bad_model_version.h5', 'no_model_version.h5'])
@pytest.mark.asyncio
async def test_fetch_model_model_version_error(filename, web_server) -> None:
    url = web_server(filename)
    with pytest.raises(models.ModelVersionError) as exc_info:
        await fetch_aiohttp.fetch_model(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url
    assert 'model_version' in str(exc_info.value)


@pytest.mark.asyncio
async def test_fetch_model_cached_model_type_error(web_server) -> None:
    class OtherModel(models.SimpleHDF5Model):
        model_type = 'other'

        @classmethod
        def from_hdf5(cls, hdf5: h5py.File) -> 'OtherModel':
            return cls()

    url = web_server('rfi_mask_ranges.h5')
    async with fetch_aiohttp.Fetcher() as fetcher:
        await fetcher.get(url, DummyModel)
        with pytest.raises(models.ModelTypeError) as exc_info:
            await fetcher.get(url, OtherModel)
        assert exc_info.value.url == url
        assert exc_info.value.original_url == url


@pytest.mark.asyncio
async def test_fetch_model_bad_content_type(mock_aioresponses) -> None:
    data = get_data('rfi_mask_ranges.h5')
    url = get_data_url('bad_content_type.h5')
    mock_aioresponses.get(url, content_type='image/png', body=data)
    with pytest.raises(models.FileTypeError,
                       match='Expected application/x-hdf5, not image/png') as exc_info:
        async with fetch_aiohttp.Fetcher() as fetcher:
            await fetcher.get(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url


@pytest.mark.asyncio
async def test_fetch_model_bad_extension(web_server) -> None:
    url = web_server('wrong_extension.blah')
    with pytest.raises(models.FileTypeError) as exc_info:
        async with fetch_aiohttp.Fetcher() as fetcher:
            await fetcher.get(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url


@pytest.mark.asyncio
async def test_fetch_model_not_hdf5(web_server) -> None:
    url = web_server('not_hdf5.h5')
    with pytest.raises(models.DataError) as exc_info:
        await fetch_aiohttp.fetch_model(url, DummyModel)
    assert exc_info.value.url == url
    assert exc_info.value.original_url == url


@pytest.mark.asyncio
async def test_fetch_model_checksum_ok(mock_aioresponses) -> None:
    data = get_data('rfi_mask_ranges.h5')
    digest = hashlib.sha256(data).hexdigest()
    url = get_data_url(f'sha256_{digest}.h5')
    mock_aioresponses.get(url, content_type='application/x-hdf5', body=data)
    with await fetch_aiohttp.fetch_model(url, DummyModel) as model:
        assert model.checksum == digest


@pytest.mark.asyncio
async def test_fetch_model_checksum_bad(mock_aioresponses) -> None:
    data = get_data('rfi_mask_ranges.h5')
    digest = hashlib.sha256(data).hexdigest()
    url = get_data_url(f'sha256_{digest}.h5')
    # Now invalidate the digest
    data += b'blahblahblah'
    mock_aioresponses.get(url, content_type='application/x-hdf5', body=data)
    with pytest.raises(models.ChecksumError) as exc_info:
        await fetch_aiohttp.fetch_model(url, DummyModel)
    assert exc_info.value.url == url


@pytest.mark.parametrize('filename', ['does_not_exist.h5', 'does_not_exist.alias'])
@pytest.mark.asyncio
async def test_fetch_model_bad_http_status(filename, web_server) -> None:
    url = web_server(filename)
    with pytest.raises(aiohttp.ClientError) as exc_info:
        await fetch_aiohttp.fetch_model(url, DummyModel)
    assert exc_info.value.status == 404


@pytest.mark.asyncio
async def test_fetch_model_http_redirect(mock_aioresponses) -> None:
    url = get_data_url('subdir/redirect.alias')
    # aioresponses doesn't properly handle relative URLs in redirects, so we
    # need to use an absolute URL.
    new_url = get_data_url('direct.alias')
    mock_aioresponses.get(url, headers={'Location': new_url}, status=307)
    for match in mock_aioresponses._matches:
        print(match.method, match.url_or_pattern)
    with await fetch_aiohttp.fetch_model(url, DummyModel) as model:
        assert len(model.ranges) == 2


@pytest.mark.asyncio
async def test_fetch_model_connection_error(mock_aioresponses) -> None:
    # aioresponses raises ConnectionError for any unregistered URL
    with pytest.raises(aiohttp.ClientConnectionError):
        await fetch_aiohttp.fetch_model(get_data_url('does_not_exist.h5'), DummyModel)


@pytest.mark.asyncio
async def test_fetcher_caching(mock_aioresponses) -> None:
    async with fetch_aiohttp.Fetcher() as fetcher:
        model1 = await fetcher.get(get_data_url('rfi_mask_ranges.h5'), DummyModel)
        model2 = await fetcher.get(get_data_url('indirect.alias'), DummyModel)
        model3 = await fetcher.get(get_data_url('direct.alias'), DummyModel)
        assert model1 is model2
        assert model1 is model3
        assert not model1.is_closed
    # Not supported by aioresponses
    # assert len(mock_aioresponses.calls) == 3
    assert model1.is_closed


def _tracing_session(urls: List[yarl.URL]) -> aiohttp.ClientSession:
    async def record_urls(session: aiohttp.ClientSession,
                          trace_config_ctx: types.SimpleNamespace,
                          params: aiohttp.TraceRequestStartParams) -> None:
        urls.append(params.url)

    trace_config = aiohttp.TraceConfig()
    trace_config.on_request_start.append(record_urls)     # type: ignore
    return aiohttp.ClientSession(trace_configs=[trace_config])


@pytest.mark.asyncio
async def test_fetcher_custom_session(web_server) -> None:
    urls: List[yarl.URL] = []
    async with _tracing_session(urls) as session:
        async with fetch_aiohttp.Fetcher(session=session) as fetcher:
            assert fetcher.session is session
            await fetcher.get(web_server('direct.alias'), DummyModel)
        assert len(urls) == 2
        assert not session.closed
    assert session.closed


@pytest.mark.asyncio
async def test_custom_session(web_server) -> None:
    urls: List[yarl.URL] = []
    async with _tracing_session(urls) as session:
        await fetch_aiohttp.fetch_model(web_server('direct.alias'), DummyModel, session=session)
        assert len(urls) == 2
        assert not session.closed
    assert session.closed


@pytest.mark.asyncio
async def test_fetcher_resolve(web_server) -> None:
    url = web_server('indirect.alias')
    async with fetch_aiohttp.Fetcher() as fetcher:
        urls = await fetcher.resolve(url)
    assert urls == [
        web_server('indirect.alias'),
        web_server('direct.alias'),
        web_server('rfi_mask_ranges.h5')
    ]
