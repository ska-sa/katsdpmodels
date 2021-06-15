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

# from typing import Generator
import os

import h5py
import pytest
from katsdpmodels.local_sky import KatpointSkyModel
import katpoint
import astropy.units as units
import numpy as np


_TRG_A = 'A, radec, 20:00:00.00, -60:00:00.0, (200.0 12000.0 1.0 0.5)'
_TRG_B = 'B, radec, 8:00:00.00, 60:00:00.0, (200.0 12000.0 2.0)'
_TRG_C = 'C, radec, 21:00:00.00, -60:00:00.0, (800.0 43200.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.8 -0.7 0.6)'
_PC = 'pc, radec, 20:00:00.00, -60:00:00.0'


@pytest.fixture
def dummy_local_sky() -> KatpointSkyModel:
    t1 = katpoint.Target(_TRG_A)
    t2 = katpoint.Target(_TRG_B)
    t3 = katpoint.Target(_TRG_C)
    cat = katpoint.Catalogue([t1, t2, t3])
    pc = katpoint.Target(_PC)
    return KatpointSkyModel(cat, pc)


def test_model_type(dummy_local_sky: KatpointSkyModel):
    model = dummy_local_sky
    assert model.model_type == "lsm"


def test_flux_density(dummy_local_sky: KatpointSkyModel):
    model = dummy_local_sky
    flux = model.flux_density(1e10 * units.Hz)
    np.testing.assert_allclose(flux, [
        [1000, 0, 0, 0],
        [100, 0, 0, 0],
        [10, 8, -7, 6]])


def test_phase_centre(dummy_local_sky: KatpointSkyModel):
    # phase_centre = np.array([300, -60]) * units.deg  # RA 20.0
    model = dummy_local_sky
    assert model._PhaseCentre is not None


def test_to_hdf5(dummy_local_sky: KatpointSkyModel):
    model = dummy_local_sky
    f = h5py.File('test_cat.h5')
    model.to_hdf5(f)
    f.close()
    # newmodel = KatpointSkyModel.from_hdf5('test_cat.h5')
    # assert newmodel.type is KatpointSkyModel
    os.remove('test_cat.h5')
