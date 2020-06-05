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

import os
import pathlib
import urllib.parse

import pytest
import responses

import test_utils


@pytest.fixture
def mock_responses():
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
        yield rsps
