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

"""Fetch models asynchronously over HTTP."""

import io
from typing import List, Dict, Generator, Optional, MutableMapping, Type, TypeVar, Any, cast

import aiohttp

from .. import models, fetch


_T = TypeVar('_T')
_M = TypeVar('_M', bound=models.Model)
_F = TypeVar('_F', bound='Fetcher')


class Fetcher(fetch.FetcherBase):
    """Fetches and caches models, using the :mod:`aiohttp` library.

    It caches every URL it fetches (ignoring any cache control headers), so it
    should not be reused over a long time.  It is best suited to fetching a
    batch of models, some of which may turn out to be aliases of each other.

    It should be closed with :meth:`close` when no longer in use. It also
    implements the context manager protocol for this purpose. This will close
    the retrieved models, so must only be done once the models are no longer in
    use.

    This class is *not* thread-safe.

    Parameters
    ----------
    session
        Interface for doing the actual HTTP requests. If not specified, an
        internal session will be created, and closed when the fetcher is
        closed. If a custom session is provided it will *not* be closed (so it
        can be shared between multiple :class:`Fetcher` instances).
    model_cache
        A dictionary for caching models by URL. This is not typically needed,
        as the fetcher will use an internal cache if one is not provided, but
        allows fetchers to share a cache (but not in a thread-safe way!).
        If a custom cache is provided, then :meth:`close` will not close the
        models in it, and the caller is responsible for doing so.

    Raises
    ------
    :exc:`.ModelError`
        For any issues with the model itself
    :exc:`aiohttp.ClientError`
        For any issues at the HTTP level
    """

    def __init__(self, *,
                 session: Optional[aiohttp.ClientSession] = None,
                 model_cache: Optional[MutableMapping[str, models.Model]] = None) -> None:
        super().__init__(model_cache=model_cache)
        if session is None:
            self._session = aiohttp.ClientSession()
            # TODO: support for file:// URLs
            self._close_session = True
        else:
            self._session = session
            self._close_session = False

    @property
    def session(self) -> aiohttp.ClientSession:
        return self._session

    async def close(self) -> None:
        self._close()
        if self._close_session:
            await self._session.close()

    async def __aenter__(self: _F) -> _F:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def _handle_request(self, request: fetch.Request) -> fetch.Response:
        assert request.response_type in {fetch.ResponseType.TEXT, fetch.ResponseType.FILE}
        if request.response_type == fetch.ResponseType.TEXT:
            async with self._session.get(request.url, raise_for_status=True) as resp:
                text = await resp.text()
                return fetch.TextResponse(str(resp.url), resp.headers, text)
        else:
            async with self._session.get(request.url, raise_for_status=True) as resp:
                content = await resp.read()
                file = io.BytesIO(content)
                return fetch.FileResponse(
                    str(resp.url), resp.headers, file=cast(io.IOBase, file), content=content)

    async def _run(self, gen: Generator[fetch.Request, fetch.Response, _T]) -> _T:
        try:
            request = next(gen)      # Start it going
            while True:
                response = await self._handle_request(request)
                request = gen.send(response)
        except StopIteration as exc:
            return exc.value
        finally:
            gen.close()

    async def resolve(self, url: str) -> List[str]:
        """Follow a chain of aliases.

        Return a list of URLs found along the chain. The first element is the
        given URL and the final element is the resolved model.

        Raises
        ------
        .TooManyAliasesError
            If there were more than :const:`.MAX_ALIASES` aliases or a cycle was found.
        """
        return await self._run(self._resolve(url))

    async def get(self, url: str, model_class: Type[_M]) -> _M:
        """Retrieve a single model.

        The caller must *not* close the retrieved model, as it is cached and
        a future request for the model would return the closed model. Instead,
        the caller must close the fetcher once it no longer needs any of the
        models.

        Raises
        ------
        .ModelError
            If there are high-level errors with the model.
        aiohttp.ClientError
            Any exceptions raised by the underlying session.
        """
        return await self._run(self._get(url, model_class))


async def fetch_model(url: str, model_class: Type[_M], *,
                      session: Optional[aiohttp.ClientSession] = None) -> _M:
    """Convenience function for retrieving a single model.

    This should only be used when loading just a single model. If multiple
    models will be used instead, construct an instance of :class:`Fetcher`
    and use it to fetch models, as this will allow models that turn out to be
    the same to be shared.
    """
    # Custom cache so that fetcher won't close the model
    model_cache: Dict[str, models.Model] = {}
    async with Fetcher(session=session, model_cache=model_cache) as fetcher:
        return await fetcher.get(url, model_class)
