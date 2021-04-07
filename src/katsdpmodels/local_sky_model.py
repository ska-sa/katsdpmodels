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

"""Local Sky Model"""

import enum
from typing import Tuple, ClassVar, Union, Optional, Any
from typing_extensions import Literal

import numpy as np
try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any  # type: ignore
import astropy.units as u
from astropy.coordinates import SkyCoord
import h5py

from . import models


class AltAzFrame:
    """Coordinate system aligned with the antenna.

    The l coordinate is horizontal and increases with increasing azimuth (north
    to east), while the m coordinate is vertical and increases with increasing
    altitude. Both are defined by an orthographic (SIN) projection, with the
    nominal pointing centre at zero.
    """


class RADecFrame:
    """Coordinate system aligned with a celestial sphere.

    The l coordinate is aligned with right ascension and the m coordinate with
    declination, and increase in the corresponding directions. Both are defined by
    an orthographic (SIN) projection, with the nominal pointing centre at zero.

    Any RA/Dec system can be used (e.g. ICRS/GCRS/CIRS) as long as the
    parallactic angle is computed for the same system.

    Parameters
    ----------
    parallactic_angle
        Rotation angle to align the celestial sphere with the antenna. See
        :meth:`katpoint.Target.parallactic_angle` for a definition.
    """

    def __init__(self, parallactic_angle: u.Quantity) -> None:
        self.parallactic_angle = parallactic_angle.to(u.rad)  # Just to verify unit type

    @staticmethod
    def from_sky_coord(self, target: SkyCoord) -> 'RADecFrame':
        """Generate a frame from a target (assuming an AltAz mount).

        The `target` must have ``obstime`` and ``location`` properties, and
        must be scalar. It will be converted to ICRS if necessary.
        """
        # TODO: implement
        raise NotImplementedError


class LocalSkyModel(models.SimpleHDF5Model):
    model_type: ClassVar[Literal['lsm']] = 'lsm'
    # Methods are not marked @abstractmethod as it causes issues with mypy:
    # https://github.com/python/mypy/issues/4717

    @classmethod
    def from_hdf5(cls, hdf5: h5py.File) -> 'LocalSkyModel':
        raise NotImplementedError()      # pragma: nocover

    def to_hdf5(self, hdf5: h5py.File) -> None:
        raise NotImplementedError()      # pragma: nocover

