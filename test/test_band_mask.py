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

"""Tests for :mod:`katsdpmodels.band_mask`"""

import io
from typing import Generator

import astropy.units as u
import numpy as np
import pytest

from katsdpmodels import models, band_mask
import katsdpmodels.fetch.requests as fetch_requests
from test_utils import get_data_url


@pytest.fixture
def spectral_window() -> band_mask.SpectralWindow:
    return band_mask.SpectralWindow(400 * u.MHz, 1000 * u.MHz)


@pytest.fixture
def ranges_model(web_server) -> Generator[band_mask.BandMask, None, None]:
    with fetch_requests.fetch_model(web_server('band_mask_ranges.h5'), band_mask.BandMask) as model:
        yield model


def test_spectral_window_attributes(spectral_window) -> None:
    assert spectral_window.bandwidth == 400 * u.MHz
    assert spectral_window.centre_frequency == 1000 * u.MHz
    assert spectral_window.min_frequency == 800 * u.MHz
    assert spectral_window.max_frequency == 1200 * u.MHz


@pytest.mark.parametrize(
    'frequency,result',
    [
        (801 * u.MHz, True),
        (810 * u.MHz, True),
        (819 * u.MHz, True),
        (821 * u.MHz, False),
        (1000 * u.MHz, False),
        (1179 * u.MHz, False),
        (1181 * u.MHz, True),
        (1199 * u.MHz, True),
        (1201 * u.MHz, False),
        (800e6 * u.Hz, True),
        (1000e6 * u.Hz, False),
        (1200e6 * u.Hz, True)
    ])
def test_is_masked_scalar(frequency: u.Quantity, result: bool,
                          ranges_model: band_mask.BandMask,
                          spectral_window: band_mask.SpectralWindow) -> None:
    # Window is 800-1200 MHz, mask is 800-820 and 1180-1200
    assert ranges_model.is_masked(spectral_window, frequency) == result


def test_is_masked_vector(ranges_model: band_mask.BandMask,
                          spectral_window: band_mask.SpectralWindow) -> None:
    frequency = [801, 810, 819, 821, 1000, 1179, 1181, 1199, 1201] * u.MHz
    np.testing.assert_array_equal(
        ranges_model.is_masked(spectral_window, frequency),
        [True, True, True, False, False, False, True, True, False]
    )


def test_bad_shape(mock_responses) -> None:
    url = get_data_url('band_mask_ranges_2d.h5')
    with pytest.raises(models.DataError, match='ranges dataset should have 1 dimension, found 2'):
        fetch_requests.fetch_model(url, band_mask.BandMask)


def test_bad_model_format(mock_responses) -> None:
    url = get_data_url('band_mask_bad_format.h5')
    with pytest.raises(models.ModelFormatError) as exc_info:
        fetch_requests.fetch_model(url, band_mask.BandMask)
    assert str(exc_info.value) == "Unknown model_format 'not_ranges' for band_mask"


def test_to_file(ranges_model: band_mask.BandMaskRanges) -> None:
    fh = io.BytesIO()
    ranges_model.to_file(fh, content_type='application/x-hdf5')
    fh.seek(0)
    new_model = band_mask.BandMask.from_file(fh, 'http://test.invalid/test.h5',
                                             content_type='application/x-hdf5')
    assert isinstance(new_model, band_mask.BandMaskRanges)
    np.testing.assert_array_equal(new_model.ranges, ranges_model.ranges)
