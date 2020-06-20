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

import contextlib
import enum
import hashlib
import logging
import io
import re
import urllib.parse
from typing import List, Dict, Generator, Mapping, MutableMapping, Optional, Type, TypeVar, Any

from .. import models


MAX_ALIASES = 30
_logger = logging.getLogger(__name__)
_F = TypeVar('_F', bound='FetcherBase')
_M = TypeVar('_M', bound=models.Model)


class ResponseType(enum.Enum):
    TEXT = 0
    FILE = 1


class Request:
    """Bare essentials of an HTTP request."""

    # TODO: use attrs?
    def __init__(self, url: str, response_type: ResponseType) -> None:
        self.url = url
        self.response_type = response_type


class Response:
    """Bare essentials of a response to a :class:`Request`.

    Parameters
    ----------
    url
        The final URL of the request, after handling any redirects.
    headers
        A case-insensitive mapping of HTTP headers
    """

    def __init__(self, url: str, headers: Mapping[str, str]) -> None:
        self.url = url
        self.headers = headers

    @property
    def content_type(self) -> Optional[str]:
        """The MIME content type of the request.

        Any parameters (like encoding) are stripped off. If the Content-Type
        header is missing or it is application/octet-stream, returns
        ``None``.
        """
        raw = self.headers.get('Content-Type', 'application/octet-stream')
        # Split parameters like encoding
        content_type = raw.split(';')[0].strip()
        if content_type == 'application/octet-stream':
            # This is a generic/fallback content type that doesn't convey any
            # useful information.
            return None
        else:
            return content_type


class TextResponse(Response):
    """Response to a :const:`ResponseType.TEXT` request."""

    def __init__(self, url: str, headers: Mapping[str, str], text: str) -> None:
        super().__init__(url, headers)
        self.text = text


class FileResponse(Response):
    """Response to a :const:`ResponseType.FILE` request.

    Parameters
    ----------
    url
        The final URL of the request, after handling any redirects.
    headers
        A case-insensitive mapping of HTTP headers
    file
        A file-like object from which the binary content can be read. The
        receiver of the response is responsible for closing it.
    content
        If available, the content of the response. It is used only for
        verifying checksums.
    """

    def __init__(self, url: str, headers: Mapping[str, str],
                 file: io.IOBase, content: Optional[bytes] = None) -> None:
        super().__init__(url, headers)
        self.file = file
        self.content = content


class FetcherBase:
    """Base class for HTTP fetcher implementations.

    It does not perform any I/O itself. Instead, it provides generators
    that yield :class:`Request`s and expects to receive :class:`Response`s in
    reply. The subclass is responsible for producing the responses to
    requests. This design allows the core logic to be shared between
    synchronous and asynchronous implementations.

    Parameters
    ----------
    model_cache
        A dictionary for caching models by URL. This is not typically needed,
        as the fetcher will use an internal cache if one is not provided, but
        allows fetchers to share a cache (but not in a thread-safe way!).
        If a custom cache is provided, then :meth:`close` will not close the
        models in it, and the caller is responsible for doing so.
    """

    def __init__(self, *, model_cache: Optional[MutableMapping[str, models.Model]] = None) -> None:
        self._alias_cache: Dict[str, str] = {}
        if model_cache is not None:
            self._model_cache = model_cache
            self._close_models = False
        else:
            self._model_cache = {}
            self._close_models = True

    def _resolve(self, url: str) -> Generator[Request, Response, List[str]]:
        chain = [url]
        parts = urllib.parse.urlparse(url)
        while parts.path.endswith('.alias'):
            if len(chain) > MAX_ALIASES:
                raise models.TooManyAliasesError.with_urls(
                    f'Reached limit of {MAX_ALIASES} levels of aliases',
                    url=url, original_url=chain[0])
            if url in self._alias_cache:
                new_url = self._alias_cache[url]
            else:
                request = Request(url, ResponseType.TEXT)
                response = yield request
                assert isinstance(response, TextResponse)
                rel_path = response.text.rstrip()
                new_url = urllib.parse.urljoin(response.url, rel_path)
                self._alias_cache[url] = new_url
            if new_url in chain:
                raise models.TooManyAliasesError.with_urls(
                    f'Cycle detected starting from {new_url}',
                    url=new_url, original_url=chain[0])
            chain.append(new_url)
            _logger.debug('Redirecting from %s to %s', url, new_url)
            url = new_url
            parts = urllib.parse.urlparse(url)
        return chain

    def _get(self, url: str, model_class: Type[_M]) -> Generator[Request, Response, _M]:
        original_url = url
        url = (yield from self._resolve(url))[-1]
        if url in self._model_cache:
            model = self._model_cache[url]
            if model_class.model_type != model.model_type:
                raise models.ModelTypeError.with_urls(
                    f'Expected a model of type {model_class.model_type!r}, '
                    f'not {model.model_type!r}',
                    url=url, original_url=original_url)
            if not isinstance(model, model_class):
                raise TypeError('model_class should be the base class for the model type')
            return model

        with contextlib.ExitStack() as exit_stack:
            checksum: Optional[str] = None
            try:
                request = Request(url, ResponseType.FILE)
                response = yield request
                assert isinstance(response, FileResponse)
                exit_stack.callback(response.file.close)
                url = response.url
                if response.content is not None:
                    checksum = hashlib.sha256(response.content).hexdigest()
                    parts = urllib.parse.urlparse(url)
                    match = re.search(r'/sha256_([a-z0-9]+)\.[^/]+$', parts.path)
                    if match and checksum != match.group(1):
                        raise models.ChecksumError('Content did not match checksum in URL')
                try:
                    new_model = model_class.from_file(response.file, url,
                                                      content_type=response.content_type)
                except models.ModelError:
                    raise
                except Exception as exc:
                    raise models.DataError(f'Failed to load model from {url}: {exc}') from exc
            except models.ModelError as exc:
                exc.original_url = original_url
                exc.url = url
                raise
            else:
                exit_stack.pop_all()   # new_model now owns fh, or has closed it

        new_model.checksum = checksum
        self._model_cache[url] = new_model
        return new_model

    def close(self) -> None:
        """Release the resources associated with the fetcher.

        After this the fetcher should not be used further, except that it is
        legal to call this method multiple times.
        """
        self._alias_cache.clear()
        if self._close_models:
            for model in self._model_cache.values():
                model.close()
            self._model_cache.clear()
        else:
            self._model_cache = {}     # Allow garbage collection of the old cache

    def __enter__(self: _F) -> _F:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
