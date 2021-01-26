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

"""Tests for :mod:`katsdpmodels.rfi_mask`"""

import io
from typing import Generator

import astropy.units as u
import astropy.table
import numpy as np
import pytest

from katsdpmodels import models, rfi_mask
import katsdpmodels.fetch.requests as fetch_requests
from test_utils import get_data_url


@pytest.fixture
def ranges_model(web_server) -> Generator[rfi_mask.RFIMask, None, None]:
    with fetch_requests.fetch_model(web_server('rfi_mask_ranges.h5'), rfi_mask.RFIMask) as model:
        yield model


@pytest.mark.parametrize(
    'frequency,baseline_length,result',
    [
        (90e6 * u.Hz, 2 * u.m, False),
        (110e6 * u.Hz, 2 * u.m, True),
        (110 * u.MHz, 2 * u.m, True),
        (110e6 * u.Hz, 2000 * u.m, False),
        (100 * u.MHz, 2 * u.km, False),
        (600e6 * u.Hz, 2000 * u.m, True),
        (600 * u.MHz, 2 * u.km, True),
        (600 * u.MHz, 0 * u.m, False)
    ])
def test_is_masked_scalar(frequency: u.Quantity, baseline_length: u.Quantity, result: bool,
                          ranges_model: rfi_mask.RFIMask) -> None:
    assert ranges_model.is_masked(frequency, baseline_length) == result


def test_is_masked_vector(ranges_model: rfi_mask.RFIMaskRanges):
    frequency = u.Quantity([90, 110, 90, 110, 300, 600, 600, 600], u.MHz)
    baseline_length = u.Quantity([2, 2, 2000, 2000, 2, 2, 2000, 0], u.m)
    result = ranges_model.is_masked(frequency, baseline_length)
    expected = np.array([False, True, False, False, False, True, True, False])
    np.testing.assert_array_equal(result, expected)
    # Test broadcasting of inputs against each other
    result = ranges_model.is_masked(frequency, 2 * u.m)
    expected = np.array([False, True, False, True, False, True, True, True])
    np.testing.assert_array_equal(result, expected)


def test_is_masked_channel_width(ranges_model: rfi_mask.RFIMaskRanges):
    frequency = u.Quantity([70, 90, 110, 190, 210, 230], u.MHz)
    result = ranges_model.is_masked(frequency, 1 * u.m, 20 * u.MHz)
    expected = np.array([False, True, True, True, True, False])
    np.testing.assert_array_equal(result, expected)


def test_is_masked_auto_correlations(ranges_model: rfi_mask.RFIMaskRanges) -> None:
    ranges_model = rfi_mask.RFIMaskRanges(ranges_model.ranges, True)
    assert ranges_model.is_masked(600 * u.MHz, 0 * u.m)


@pytest.mark.parametrize(
    'frequency,result',
    [
        (90e6 * u.Hz, -1 * u.m),
        (90 * u.MHz, -1 * u.m),
        (110e6 * u.Hz, 1000 * u.m),
        (110 * u.MHz, 1000 * u.m),
        (300 * u.MHz, -1 * u.m),
        (600 * u.MHz, np.inf * u.m)
    ])
def test_max_baseline_length_scalar(frequency: u.Quantity, result: u.Quantity,
                                    ranges_model: rfi_mask.RFIMask) -> None:
    assert ranges_model.max_baseline_length(frequency) == result


def test_max_baseline_length_vector(ranges_model: rfi_mask.RFIMask) -> None:
    frequency = u.Quantity([90, 110, 300, 600, 900], u.MHz)
    result = ranges_model.max_baseline_length(frequency)
    np.testing.assert_array_equal(
        result.to_value(u.m),
        [-1, 1000, -1, np.inf, -1]
    )


def test_max_baseline_length_channel_width(ranges_model: rfi_mask.RFIMask) -> None:
    frequency = u.Quantity([70, 90, 110, 190, 210, 230, 470, 490], u.MHz)
    result = ranges_model.max_baseline_length(frequency, 20 * u.MHz)
    np.testing.assert_array_equal(
        result.to_value(u.m),
        [-1, 1000, 1000, 1000, 1000, -1, -1, np.inf]
    )


def test_max_baseline_length_empty(ranges_model: rfi_mask.RFIMaskRanges) -> None:
    ranges_model.ranges.remove_rows(np.s_[:])
    assert ranges_model.max_baseline_length(1 * u.Hz) == -1 * u.m
    result = ranges_model.max_baseline_length([1, 2] * u.Hz)
    np.testing.assert_array_equal(result.to_value(u.m), [-1.0, -1.0])


@pytest.mark.parametrize(
    'filename',
    ['rfi_mask_missing_dataset.h5', 'rfi_mask_ranges_is_group.h5'])
def test_missing_dataset(filename: str, mock_responses) -> None:
    url = get_data_url(filename)
    with pytest.raises(models.DataError, match='Model is missing ranges dataset'):
        fetch_requests.fetch_model(url, rfi_mask.RFIMask)


def test_bad_shape(mock_responses) -> None:
    url = get_data_url('rfi_mask_ranges_2d.h5')
    with pytest.raises(models.DataError,
                       match='ranges dataset should be 1-dimensional, but is 2-dimensional'):
        fetch_requests.fetch_model(url, rfi_mask.RFIMask)


def test_bad_model_format(mock_responses) -> None:
    url = get_data_url('rfi_mask_bad_format.h5')
    with pytest.raises(models.ModelFormatError) as exc_info:
        fetch_requests.fetch_model(url, rfi_mask.RFIMask)
    assert str(exc_info.value) == "Unknown model_format 'not_ranges' for rfi_mask"


def test_metadata(web_server) -> None:
    with fetch_requests.fetch_model(
            web_server('rfi_mask_ranges_metadata.h5'), rfi_mask.RFIMask) as model:
        assert model.comment == 'Test model'
        assert model.author == 'Test author'
        assert model.target == 'Test target'
        assert model.created is not None
        assert model.created.isoformat() == '2020-06-11T11:11:00+00:00'


def test_construct_bad_units() -> None:
    ranges = astropy.table.Table(
        [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
        names=('min_frequency', 'max_frequency', 'max_baseline'),
        dtype=(np.float64, np.float64, np.float64)
    )
    # No units at all
    with pytest.raises(u.UnitConversionError):
        rfi_mask.RFIMaskRanges(ranges, True)
    # Incompatible units for baseline
    ranges['min_frequency'].unit = u.Hz
    ranges['max_frequency'].unit = u.Hz
    ranges['max_baseline'].unit = u.Hz
    with pytest.raises(u.UnitConversionError):
        rfi_mask.RFIMaskRanges(ranges, True)


def test_construct_incompatible_types() -> None:
    ranges = astropy.table.Table(
        [(1, 2, 3), (4, 5, 6), ('a', 'b', 'c')],
        names=('min_frequency', 'max_frequency', 'max_baseline'),
        dtype=(np.float64, np.float64, 'S1')
    )
    with pytest.raises(ValueError):
        rfi_mask.RFIMaskRanges(ranges, True)


def test_construct_missing_column() -> None:
    ranges = astropy.table.Table(
        [(1, 2, 3), (4, 5, 6)],
        names=('min_frequency', 'max_frequency'),
        dtype=(np.float64, np.float64)
    )
    with pytest.raises(KeyError):
        rfi_mask.RFIMaskRanges(ranges, True)


def test_to_file(ranges_model):
    fh = io.BytesIO()
    ranges_model.to_file(fh, content_type='application/x-hdf5')
    fh.seek(0)
    new_model = rfi_mask.RFIMask.from_file(fh, 'http://test.invalid/test.h5',
                                           content_type='application/x-hdf5')
    np.testing.assert_array_equal(new_model.ranges, ranges_model.ranges)
    assert new_model.mask_auto_correlations == ranges_model.mask_auto_correlations
