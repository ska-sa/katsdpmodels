################################################################################
# Copyright (c) 2021, National Research Foundation (SARAO)
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

"""Tests for :mod:`katsdpmodels.primary_beam`"""

from typing import Generator

import astropy.units as u
import numpy as np
import h5py
import pytest

from katsdpmodels import primary_beam
import katsdpmodels.fetch.requests as fetch_requests


def make_aperture_plane_model_file(h5file: h5py.File) -> None:
    """Create an aperture-plane model for testing.

    It writes the model directly rather than using the PrimaryBeam API
    to ensure the model adheres to the file format even if bugs are
    introduced into the API.
    """
    h5file.attrs['model_type'] = 'primary_beam'
    h5file.attrs['model_format'] = 'aperture_plane'
    h5file.attrs['model_comment'] = 'Dummy model for testing. Do not use in production.'
    h5file.attrs['model_created'] = '2021-04-20T12:11:00Z'
    h5file.attrs['model_version'] = 1
    h5file.attrs['model_author'] = 'MeerKAT SDP developers <sdpdev+katsdpmodels@ska.ac.za>'
    h5file.attrs['antenna'] = 'm999'
    h5file.attrs['receiver'] = 'r123'
    h5file.attrs['band'] = 'z'
    # x and y are deliberately set up differently to catch bugs that mix them up.
    h5file.attrs['x_start'] = -10.0
    h5file.attrs['x_step'] = 0.5
    h5file.attrs['y_start'] = 8.0
    h5file.attrs['y_step'] = -0.25
    frequency = np.arange(64) * 1e7 + 1e9
    h5file.create_dataset('frequency', data=frequency)

    rs = np.random.RandomState()
    shape = (2, 2, len(frequency), 80, 40)
    data = rs.random_sample(shape) + 1j * rs.random_sample(shape)
    # Adjust data so that beam is the identity at the centre.
    data -= np.mean(data, axis=(3, 4), keepdims=True)
    data[0, 0] += 1
    data[1, 1] += 1
    h5file.create_dataset('aperture_plane', data=data.astype(np.complex64))


@pytest.fixture
def aperture_plane_model(tmp_path) \
        -> Generator[primary_beam.PrimaryBeam, None, None]:
    path = tmp_path / 'aperture_plane_test.h5'
    with h5py.File(path, 'w') as h5file:
        make_aperture_plane_model_file(h5file)
    with fetch_requests.fetch_model(path.as_uri(),
                                    primary_beam.PrimaryBeam) as model:
        yield model


@pytest.fixture
def aperture_plane_model_no_optional(tmp_path) \
        -> Generator[primary_beam.PrimaryBeam, None, None]:
    path = tmp_path / 'aperture_plane_test.h5'
    with h5py.File(path, 'w') as h5file:
        make_aperture_plane_model_file(h5file)
        del h5file.attrs['receiver']
        del h5file.attrs['antenna']
        del h5file.attrs['band']
    with fetch_requests.fetch_model(path.as_uri(),
                                    primary_beam.PrimaryBeam) as model:
        yield model


def test_properties(aperture_plane_model):
    model = aperture_plane_model
    # approx because speed of light is not exactly 3e8 m/s
    assert model.spatial_resolution(1 * u.GHz) == pytest.approx(0.015, rel=1e-3)
    assert model.frequency_range() == (1000 * u.MHz, 1630 * u.MHz)
    assert model.frequency_resolution() == 10 * u.MHz
    # approx because speed of light is not exactly 3e8 m/s
    assert model.inradius(1.5 * u.GHz) == pytest.approx(0.2, rel=1e-3)
    # Radius in x is 0.3, in y is 0.6 because it has twice the aperture-plane
    # resolution.
    assert model.circumradius(1 * u.GHz) == pytest.approx(np.hypot(0.3, 0.6), rel=1e-3)
    assert not model.is_circular
    assert not model.is_unpolarized
    assert model.antenna == 'm999'
    assert model.receiver == 'r123'
    assert model.band == 'z'


def test_no_optional_properties(aperture_plane_model_no_optional):
    model = aperture_plane_model_no_optional
    assert model.antenna is None
    assert model.receiver is None
    assert model.band is None
