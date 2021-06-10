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

"""Tests for :mod:`katsdpmodels.local_sky`"""

from typing import Generator
import pytest
from katsdpmodels import KatpointSkyModel
import katpoint

@pytest.fixture
def dummy_local_sky() -> LocalSkyModel:
    cat = katpoint.Catalogue()
    lsm = KatpointSkyModel(cat)
    return lsm

def test_