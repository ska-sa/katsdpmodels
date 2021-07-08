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
import pytest
import katsdpmodels.local_sky
import katpoint
import astropy.units as units
import numpy as np

from katsdpmodels.local_sky import KatpointSkyModel


_TRG_A = 'A, radec, 20:00:00.00, -60:00:00.0, (200.0 12000.0 1.0 0.5)'
_TRG_B = 'B, radec, 8:00:00.00, 60:00:00.0, (200.0 12000.0 2.0)'
_TRG_C = 'C, radec, 21:00:00.00, -60:00:00.0, (800.0 43200.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.8 ' \
         '-0.7 0.6)'
_TRG_WSC_A = ''
_PC = 'pc, radec, 20:00:00.00, -60:00:00.0'


@pytest.fixture
def dummy_catalogue() -> katpoint.Catalogue:
    t1 = katpoint.Target(_TRG_A)
    t2 = katpoint.Target(_TRG_B)
    t3 = katpoint.Target(_TRG_C)
    return katpoint.Catalogue([t1, t2, t3])


@pytest.fixture
def dummy_local_sky(dummy_catalogue: katpoint.Catalogue) -> KatpointSkyModel:
    cat = dummy_catalogue
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
    model = dummy_local_sky
    assert model.phase_centre is not None


def test_catalogue_from_katpoint_csv(dummy_catalogue: katpoint.Catalogue):
    cat = dummy_catalogue
    cat.save('example_catalogue_from_katpoint.csv')
    cat2 = katsdpmodels.local_sky.catalogue_from_katpoint('example_catalogue_from_katpoint.csv')
    assert cat == cat2  # TODO define equality relationship
