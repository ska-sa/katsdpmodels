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

import contextlib
import pathlib
from typing import Generator

import astropy.units as u
import numpy as np
import h5py
import pytest

from katsdpmodels import models, primary_beam
import katsdpmodels.fetch.requests as fetch_requests


@pytest.fixture
def aperture_plane_model_file(tmp_path) -> h5py.File:
    """Create an aperture-plane model for testing.

    It writes the model directly rather than using the PrimaryBeam API
    to ensure the model adheres to the file format even if bugs are
    introduced into the API.
    """
    path = tmp_path / 'aperture_plane_test.h5'
    h5file = h5py.File(path, 'w')
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
    return h5file


@contextlib.contextmanager
def serve_model(model_file: h5py.File) -> Generator[primary_beam.PrimaryBeam, None, None]:
    path = pathlib.Path(model_file.filename)
    model_file.close()  # Ensures data is written to the file
    with fetch_requests.fetch_model(path.as_uri(), primary_beam.PrimaryBeam) as model:
        yield model


@pytest.fixture
def aperture_plane_model(aperture_plane_model_file) \
        -> Generator[primary_beam.PrimaryBeam, None, None]:
    with serve_model(aperture_plane_model_file) as model:
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


def test_no_optional_properties(aperture_plane_model_file):
    h5file = aperture_plane_model_file
    del h5file.attrs['receiver']
    del h5file.attrs['antenna']
    del h5file.attrs['band']
    with serve_model(h5file) as model:
        assert model.antenna is None
        assert model.receiver is None
        assert model.band is None


def test_single_frequency(aperture_plane_model_file):
    h5file = aperture_plane_model_file
    old_frequency = h5file['frequency']
    old_samples = h5file['aperture_plane']
    del h5file['frequency']
    del h5file['aperture_plane']
    h5file.create_dataset('frequency', data=old_frequency[:1])
    h5file.create_dataset('aperture_plane', data=old_samples[:, :, :1])
    with serve_model(h5file) as model:
        assert model.frequency_range() == (1000 * u.MHz, 1000 * u.MHz)
        assert model.frequency_resolution() == 0


def test_wrong_leading_dimensions(aperture_plane_model_file):
    h5file = aperture_plane_model_file
    del h5file['aperture_plane']
    h5file.create_dataset('aperture_plane', shape=(3, 3, 64, 80, 40), dtype=np.complex64)
    with pytest.raises(models.DataError, match='aperture_plane must by 2x2 on leading dimensions'):
        with serve_model(h5file):
            pass


def test_frequency_bad_size(aperture_plane_model_file):
    h5file = aperture_plane_model_file
    old = h5file['frequency']
    del h5file['frequency']
    h5file.create_dataset('frequency', data=old[:-1])
    with pytest.raises(models.DataError,
                       match='aperture_plane and frequency have inconsistent sizes'):
        with serve_model(h5file):
            pass


def test_frequency_bad_ordering(aperture_plane_model_file):
    h5file = aperture_plane_model_file
    h5file['frequency'][3] = 2e9
    with pytest.raises(models.DataError, match='frequencies must be strictly increasing'):
        with serve_model(h5file):
            pass


def test_missing_attr(aperture_plane_model_file):
    h5file = aperture_plane_model_file
    del h5file.attrs['x_start']
    with pytest.raises(models.DataError, match="attribute 'x_start' is missing"):
        with serve_model(h5file):
            pass


def test_bad_model_format(aperture_plane_model_file):
    h5file = aperture_plane_model_file
    h5file.attrs['model_format'] = 'not_this'
    with pytest.raises(models.ModelFormatError,
                       match="Unknown model_format 'not_this' for primary_beam"):
        with serve_model(h5file):
            pass
