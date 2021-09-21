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

"""Tests for :mod:`katsdpmodels.sefd`"""

# import astropy.units as u
import contextlib
import h5py
import io
import numpy as np
import pathlib
import pytest

from katsdpmodels import models, sefd
from typing import Any, Generator, List, Optional, cast
try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any  # type: ignore

import katsdpmodels.fetch.requests as fetch_requests


COEFS = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
ANTS = ['m001', 'm123']
RECS = ['r987', 'r123']


@contextlib.contextmanager
def serve_model(model_file: h5py.File) -> Generator[sefd.SEFDPoly, None, None]:
    path = pathlib.Path(model_file.filename)
    model_file.close()  # Ensures data is written to the file
    with fetch_requests.fetch_model(path.as_uri(), sefd.SEFDModel) as model:
        yield cast(sefd.SEFDPoly, model)


@pytest.fixture
def poly_model_file(tmp_path) -> h5py.File:
    """Explicitly create a model for testing.

    Writes the model directly rather than using the API to decouple testing.
    """
    path = tmp_path / 'sefd_test.h5'
    h5file = h5py.File(path, 'w')
    h5file.attrs['model_type'] = 'sefd'
    h5file.attrs['model_format'] = 'poly'
    h5file.attrs['model_created'] = '2021-08-11T13:46:00Z'
    h5file.attrs['model_version'] = 1
    h5file.attrs['model_author'] = 'MeerKAT SDP developers <sdpdev+katsdpmodels@ska.ac.za>'
    h5file.create_dataset('frequency', data=np.arange(64) * 1e7 + 1e9)
    h5file.create_dataset('coefs', data=COEFS)
    h5file.attrs['correlator_efficiency'] = 0.85
    h5file.attrs['band'] = 'UHF'
    h5file.create_dataset('antennas', data=ANTS, track_times=False)
    h5file.create_dataset('receivers', data=RECS, track_times=False)
    return h5file


@pytest.fixture
def poly_model(poly_model_file) -> Generator[sefd.SEFDPoly, None, None]:
    with serve_model(poly_model_file) as model:
        yield model


def test_properties(poly_model) -> None:
    model = poly_model
    # assert model.antennas == 'm001'
    # assert model.receivers == 'r001'
    # assert model.antennas == ANTS
    assert model.band == 'UHF'
    np.testing.assert_equal(model.coefs, COEFS)


def test_no_optional_properties(poly_model_file) -> None:
    h5file = poly_model_file
    del h5file['receivers']
    del h5file['antennas']
    with serve_model(h5file) as model:
        assert model.antennas is None
        assert model.receivers is None


def test_no_band(poly_model_file) -> None:
    h5file = poly_model_file
    del h5file.attrs['band']
    with pytest.raises(models.DataError, match="attribute 'band' is missing"):
        with serve_model(h5file):
            pass


@pytest.mark.parametrize(
    'antennas, receivers',
    [
        (['m001'], ['r001']),
        (['m123'], ['r987']),
        (None, None)
    ]
)
def test_to_file(poly_model: sefd.SEFDPoly, antennas: Optional[List[str]],
                 receivers: Optional[List[str]]) -> None:
    model = poly_model
    model._antennas = antennas
    model._receivers = receivers
    fh = io.BytesIO()
    model.to_file(fh, content_type='application/x-hdf5')
    fh.seek(0)
    new_model = sefd.SEFDPoly.from_file(fh, 'http://test.invalid/test.h5',
                                        content_type='application/x-hdf5')
    assert isinstance(new_model, sefd.SEFDPoly)
    assert new_model.band == model.band
    np.testing.assert_equal(new_model.coefs, model.coefs)


def test_bad_model_format(poly_model_file: h5py.File) -> None:
    h5file = poly_model_file
    h5file.attrs['model_format'] = 'BAD_FORMAT'
    with pytest.raises(models.ModelFormatError,
                       match=f"Unknown model_format '{h5file.attrs['model_format']}' "
                             f"for sefd"):
        with serve_model(h5file):
            pass


def test_missing_required_attr(poly_model_file: h5py.File) -> None:
    h5file = poly_model_file
    del h5file.attrs['correlator_efficiency']
    with pytest.raises(models.DataError, match="attribute 'correlator_efficiency' is missing"):
        with serve_model(h5file):
            pass


def test_call(poly_model_file: h5py.File) -> None:
    pass
    '''h5file = poly_model_file
    with serve_model(h5file) as model:
        model()'''
