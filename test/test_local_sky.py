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

import io
from typing import Generator

import astropy.units as u
import astropy.table
import numpy as np
import pytest

from katsdpmodels import models, local_sky
import katsdpmodels.fetch.requests as fetch_requests
from test_utils import get_data_url

def test_bad_model_format(mock_responses) -> None:
    url = get_data_url('local_sky_bad_format.h5')
    with pytest.raises(models.ModelFormatError) as exc_info:
        fetch_requests.fetch_model(url, local_sky.KatpointSkyModel)
    assert str(exc_info.value) == "Unknown model_format 'notkatpointskymodel' for local_sky"