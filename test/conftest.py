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

import asyncio
import concurrent.futures
import os
import pathlib
import threading
import urllib.parse
from typing import Tuple, Callable, Generator

import pytest
import responses
import tornado.httpserver
import tornado.netutil
import tornado.web

import test_utils


_Info = Tuple[asyncio.AbstractEventLoop, asyncio.Event, str]


@pytest.fixture
def mock_responses() -> Generator[responses.RequestsMock, None, None]:
    """Fixture to make test data available via mocked HTTP."""
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        for (dirpath, dirnames, filenames) in os.walk(data_dir):
            for name in filenames:
                path = pathlib.Path(dirpath) / name
                rel_url = path.relative_to(data_dir).as_posix()
                url = urllib.parse.urljoin(test_utils.BASE_URL, rel_url)
                with open(path, 'rb') as f:
                    data = f.read()
                rsps.add(responses.GET, url, body=data)
                rsps.add(responses.HEAD, url)
        yield rsps


@pytest.fixture
def web_server() -> Generator[Callable[[str], str], None, None]:
    """Fixture that runs an aiohttp web server in a separate thread.

    It makes the test data available. The return value is a function that
    converts a relative URL to an absolute one for the server.
    """

    async def server_coro(info_future: 'concurrent.futures.Future[_Info]') -> None:
        try:
            data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
            app = tornado.web.Application([], static_path=data_dir)
            sockets = tornado.netutil.bind_sockets(0, '127.0.0.1')
            port = sockets[0].getsockname()[1]
            server = tornado.httpserver.HTTPServer(app)
            server.add_sockets(sockets)
            server.start()
            finished_event = asyncio.Event()
            data_url = f'http://127.0.0.1:{port}/static/'
            info_future.set_result((asyncio.get_event_loop(), finished_event, data_url))
            await finished_event.wait()
            server.stop()
        except Exception as exc:
            info_future.set_exception(exc)

    def server_thread(info_future: 'concurrent.futures.Future[_Info]') -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server_coro(info_future))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

    def generate_url(relative: str) -> str:
        return urllib.parse.urljoin(base_url, relative)

    info_future: concurrent.futures.Future[_Info] = concurrent.futures.Future()
    thread = threading.Thread(target=server_thread, args=(info_future,))
    thread.start()
    # Wait for the server to be ready to serve
    loop, event, base_url = info_future.result()

    yield generate_url

    loop.call_soon_threadsafe(event.set)
    thread.join()
