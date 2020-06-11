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

import astropy.units as u
import numpy as np
import pytest

from katsdpmodels import models, fetch, rfi_mask
from test_utils import get_data_url


@pytest.fixture
def ranges_model(web_server):
    return fetch.fetch_model(web_server('rfi_mask_ranges.hdf5'), rfi_mask.RFIMask)


@pytest.mark.parametrize(
    'frequency,baseline_length,result',
    [
        (90e6 * u.Hz, 2 * u.m, False),
        (110e6 * u.Hz, 2 * u.m, True),
        (110 * u.MHz, 2 * u.m, True),
        (110e6 * u.Hz, 2000 * u.m, False),
        (100 * u.MHz, 2 * u.km, False),
        (600e6 * u.Hz, 2000 * u.m, True),
        (600 * u.MHz, 2 * u.km, True)
    ])
def test_is_masked_scalar(frequency, baseline_length, result, ranges_model):
    assert ranges_model.is_masked(frequency, baseline_length) == result


def test_is_masked_vector(ranges_model):
    frequency = u.Quantity([90, 110, 90, 110, 300, 600, 600], u.MHz)
    baseline_length = u.Quantity([2, 2, 2000, 2000, 2, 2, 2000], u.m)
    result = ranges_model.is_masked(frequency, baseline_length)
    expected = np.array([False, True, False, False, False, True, True])
    np.testing.assert_array_equal(result, expected)
    # Test broadcasting of inputs against each other
    result = ranges_model.is_masked(frequency, 2 * u.m)
    expected = np.array([False, True, False, True, False, True, True])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    'frequency,result',
    [
        (90e6 * u.Hz, 0 * u.m),
        (90 * u.MHz, 0 * u.m),
        (110e6 * u.Hz, 1000 * u.m),
        (110 * u.MHz, 1000 * u.m),
        (300 * u.MHz, 0 * u.m),
        (600 * u.MHz, np.inf * u.m)
    ])
def test_max_baseline_length_scalar(frequency, result, ranges_model):
    assert ranges_model.max_baseline_length(frequency) == result


def test_max_baseline_length_vector(ranges_model):
    frequency = u.Quantity([90, 110, 300, 600, 900], u.MHz)
    result = ranges_model.max_baseline_length(frequency)
    np.testing.assert_array_equal(
        result.to_value(u.m),
        [0, 1000, 0, np.inf, 0]
    )


def test_max_baseline_length_empty(ranges_model):
    ranges_model.ranges.remove_rows(np.s_[:])
    assert ranges_model.max_baseline_length(1 * u.Hz) == 0 * u.m
    result = ranges_model.max_baseline_length([1, 2] * u.Hz)
    np.testing.assert_array_equal(result.to_value(u.m), [0.0, 0.0])


@pytest.mark.parametrize(
    'filename',
    ['rfi_mask_missing_dataset.hdf5', 'rfi_mask_ranges_is_group.hdf5'])
def test_missing_dataset(filename, mock_responses):
    url = get_data_url(filename)
    with pytest.raises(models.DataError, match='Model is missing ranges dataset'):
        fetch.fetch_model(url, rfi_mask.RFIMask)


def test_bad_shape(mock_responses):
    url = get_data_url('rfi_mask_ranges_2d.hdf5')
    with pytest.raises(models.DataError, match='ranges dataset should have 1 dimension, found 2'):
        fetch.fetch_model(url, rfi_mask.RFIMask)


def test_bad_model_format(mock_responses):
    url = get_data_url('rfi_mask_bad_format.hdf5')
    with pytest.raises(models.ModelFormatError) as exc_info:
        fetch.fetch_model(url, rfi_mask.RFIMask)
    assert str(exc_info.value) == "Unknown model_format 'not_ranges' for rfi_mask"