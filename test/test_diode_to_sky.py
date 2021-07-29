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

"""Tests for :mod:`katsdpmodels.diode_to_sky`"""

# import astropy.units as u
import contextlib
import h5py
import io
import numpy as np
import pathlib
import pytest

from katsdpmodels import models, diode_to_sky
from typing import Any, Generator, Optional, cast
try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any  # type: ignore

import katsdpmodels.fetch.requests as fetch_requests


COEFS = np.array([15.439, 15.385, 14.456, 13.058, 11.515, 11.301, 11.554, 12.309, 13.132, 13.716,
                  13.993, 14.380, 15.393, 17.0477, 18.917, 24.137, 25.159, 26.408, 0.0, 0.0, 0.0,
                  0.0, 0.0])
KNOTS = np.array([544.0, 544.0, 544.0, 544.0, 544.0, 544.5, 587.0, 629.5, 672.0, 714.5, 757.0,
                  799.5, 842.0, 884.5, 927.0, 969.5, 1012.0, 1054.5, 1087.5, 1087.5, 1087.5,
                  1087.5, 1087.5])


@contextlib.contextmanager
def serve_model(model_file: h5py.File) \
        -> Generator[diode_to_sky.BSplineModel, None, None]:
    path = pathlib.Path(model_file.filename)
    model_file.close()  # Ensures data is written to the file
    with fetch_requests.fetch_model(path.as_uri(), diode_to_sky.BSplineModel) as model:
        yield cast(diode_to_sky.BSplineModel, model)


@pytest.fixture
def bspline_model_file(tmp_path) -> h5py.File:
    """Explicitly create a model for testing.

    Writes the model directly rather than using the API to decouple testing.
    """
    path = tmp_path / 'diode_test.h5'
    h5file = h5py.File(path, 'w')
    h5file.attrs['model_type'] = 'diode_to_sky'
    h5file.attrs['model_format'] = 'bspline'
    h5file.attrs['model_created'] = '2021-07-29T12:19:00Z'
    h5file.attrs['model_version'] = 1
    h5file.attrs['model_author'] = 'MeerKAT SDP developers <sdpdev+katsdpmodels@ska.ac.za>'
    h5file.attrs['antenna'] = 'm001'
    h5file.attrs['receiver'] = 'r001'
    h5file.attrs['band'] = 'UHF'
    h5file.attrs['degree'] = 3
    h5file.create_dataset('coefs', data=COEFS.astype(np.float64))
    h5file.create_dataset('knots', data=KNOTS.astype(np.float64))
    return h5file


@pytest.fixture
def bspline_model(bspline_model_file: h5py.File) \
        -> Generator[diode_to_sky.BSplineModel, None, None]:
    with serve_model(bspline_model_file) as model:
        yield model


def test_properties(bspline_model: diode_to_sky.BSplineModel) -> None:
    model = bspline_model
    assert model.antenna == 'm001'
    assert model.receiver == 'r001'
    assert model.band == 'UHF'


def test_no_optional_properties(bspline_model_file: h5py.File) -> None:
    h5file = bspline_model_file
    del h5file.attrs['receiver']
    del h5file.attrs['antenna']
    with serve_model(h5file) as model:
        assert model.antenna is None
        assert model.receiver is None


def test_no_band(bspline_model_file: h5py.File) -> None:
    h5file = bspline_model_file
    del h5file.attrs['band']
    with pytest.raises(models.DataError, match="attribute 'band' is missing"):
        with serve_model(h5file):
            pass


@pytest.mark.parametrize(
    'antenna, receiver',
    [
        ('m001', 'r001'),
        (None, None)
    ]
)
def test_to_file(bspline_model: diode_to_sky.BSplineModel,
                 antenna: Optional[str], receiver: Optional[str]) -> None:
    model = bspline_model
    model._antenna = antenna
    model._receiver = receiver
    fh = io.BytesIO()
    model.to_file(fh, content_type='application/x-hdf5')
    fh.seek(0)
    new_model = diode_to_sky.BSplineModel.from_file(
        fh, 'http://test.invalid/test.h5', content_type='application/x-hdf5')
    assert isinstance(new_model, diode_to_sky.BSplineModel)
    assert new_model.antenna == antenna
    assert new_model.receiver == receiver
    assert new_model.band == model.band
    np.testing.assert_equal(new_model.knots, model.knots)


def test_bad_model_format(bspline_model_file: h5py.File) -> None:
    h5file = bspline_model_file
    h5file.attrs['model_format'] = 'BAD_FORMAT'
    with pytest.raises(models.ModelFormatError,
                       match="Unknown model_format 'BAD_FORMAT' for diode_to_sky"):
        with serve_model(h5file):
            pass
