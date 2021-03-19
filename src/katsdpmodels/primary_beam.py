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

"""Primary beam models."""

import enum
from typing import Sequence, Tuple, Dict, ClassVar, Union, Any
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
    """Coordinate system aligned with the celestial sphere.

    The l coordinate is aligned with right ascension and the m coordinate with
    declination, and increase in the corresponding directions. Both are defined by
    an orthographic (SIN) projection, with the nominal pointing centre at zero.

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

        The `target` must have ``obstime`` and ``obslocation`` properties, and
        must be scalar. It will be converted to ICRS if necessary.
        """
        # TODO: implement
        raise NotImplementedError


class OutputType(enum.Enum):
    JONES_HV = 1
    JONES_XY = 2
    MUELLER = 3
    UNPOLARIZED_POWER = 4


class Parameter:
    """Description of a parameter accepted by a model.

    Parameters
    ----------
    name
        Name of the parameter used when passing it to sampling functions
    description
        Human-readable description of the parameter
    unit
        Units that must be used for the parameter
    required
        Whether the parameter is required to use the model
    """

    def __init__(self, name: str, description: str, unit: str, required: bool) -> None:
        self.name = name
        self.description = description
        self.unit = unit
        self.required = required


class PrimaryBeam(models.SimpleHDF5Model):
    r"""Base class for primary beams.

    The phase of the electric field at a point increases with time i.e., the
    phasor is

    .. math:: e^{(\omega t - kz)i}

    The "ideal" complex voltage is multiplied by the beam model to obtain the
    measured voltage.

    If you stick your right arm out and left hand up, then pretend to be an
    antenna (lean back and look at the sky) then they correspond to the
    directions of positive horizontal and vertical polarization respectively
    (the absolute signs don't matter but the signs relative to each other do
    for cross-hand terms).

    An instance of this class describes either a single antenna or an average
    of multiple antennas; to do antenna-specific modelling you will need a
    separate instance of this class per antenna.
    """

    model_type: ClassVar[Literal['primary_beam']] = 'primary_beam'

    def resolution(self, frequency: u.Quantity) -> float:
        """Approximate spatial resolution of the model, in units of projected coordinates.

        Sampling a grid at significantly higher resolution than this will have
        diminishing returns compared to sampling at this resolution and
        interpolating.
        """
        raise NotImplementedError()

    @property
    def frequency_range(self) -> Tuple[u.Quantity, u.Quantity]:
        """Minimum and maximum frequency covered by the model."""
        raise NotImplementedError()

    @property
    def frequency_resolution(self) -> u.Quantity:
        """Approximate frequency resolution of the model.

        Sampling at significantly higher spectral resolution than this will
        have diminishing returns compared to sampling at this resolution and
        interpolating.
        """
        raise NotImplementedError()

    def min_radius(self, frequency: u.Quantity) -> float:
        """Maximum distance from the pointing centre at which model has full coverage."""
        raise NotImplementedError()

    def max_radius(self, frequency: u.Quantity) -> float:
        """Maximum distance from the pointing centre at which model has any coverage."""
        raise NotImplementedError()

    @property
    def is_circular(self) -> bool:
        """Whether this model is circularly-symmetric about the pointing centre."""
        raise NotImplementedError()

    @property
    def is_unpolarized(self) -> bool:
        """Whether this model ignores polarization.

        If true, it is guaranteed that the Jones matrices describing the beam are
        scaled identity matrices.
        """
        raise NotImplementedError()

    @property
    def parameters(self) -> Sequence[Parameter]:
        """Parameters of the model."""
        raise NotImplementedError()

    def sample(self, l: ArrayLike, m: ArrayLike, frequency: u.Quantity,   # noqa: E741
               frame: Union[AltAzFrame, RADecFrame],
               output_type: OutputType,
               parameters: Dict[str, u.Quantity] = {}) -> np.ndarray:
        """Sample the primary.

        A sample is returned for each combination of a position (given by `l`,
        `m`) with a frequency. The dimensions of the output will be first those
        of `frequency`, then those of `l` and `m` (which are broadcast with
        each other), and finally the row and column for matrices if
        `output_type` is one of the matrix types.

        Parameters
        ----------
        l
            Horizontal coordinates (interpreted according to `frame`).
        m
            Vertical coordinates (interpreted according to `frame`).
        frequency
            Frequencies to sample
        frame
            Specifies how to interpret the coordinates.
        output_type
            The value to compute. It must be one of

            OutputType.JONES_HV
                Jones matrix with linear basis corresponding to horizontal and
                vertical directions (see the class documentation for sign
                conventions).
            OutputType.JONES_XY
                Jones matrix with linear basis corresponding to the IAU X
                (north) and Y (east) directions on the celestial sphere.
            OutputType.MUELLER
                A 4x4 Mueller matrix describing the effect on each Stokes
                parameter (IQUV), assuming that both antennas share the same
                beam.
            OutputType.UNPOLARISED_POWER
                Power attenuation of unpolarized sources, assuming that both
                antennas share the same beam. This is the same as the first
                element of Mueller matrix above.
        parameters
            Additional parameters to the model. Parameters that are not used
            are silently ignored.

        Raises
        ------
        KeyError
            if a required parameter is missing (TODO: more specific error)?
        ValueError
            if `output_type` is :data:`OutputType.JONES_XY` and `frame` is not
            an instance of :class:`RADecFrame`.
        astropy.units.UnitConversionError
            if any parameter has the wrong units
        astropy.units.UnitConversionError
            if `frequency` is not specified with a spectral unit
        """
        raise NotImplementedError()

    def sample_grid(self, l: ArrayLike, m: ArrayLike, frequency: u.Quantity,   # noqa: E741
                    frame: Union[AltAzFrame, RADecFrame],
                    output_type: OutputType,
                    parameters: Dict[str, u.Quantity] = {}) -> np.ndarray:
        """Sample the primary beam on a regular grid.

        This is equivalent to
        :code:`sample(l[np.newaxis, :], m[:, np.newaxis], ...)`, but may be
        significantly faster (depending on the implementation), and is not
        guaranteed to give bit-identical results. This advantage may be lost
        when using :class:`AltAzFrame` with a non-zero parallactic angle.

        The grid need not be regularly spaced, but an output is generated for
        each combination of `l`, `m` and `frequency`.

        Refer to :meth:`sample` for further details of the parameters and
        return value.
        """
        raise NotImplementedError()

    def to_hdf5(self, hdf5: h5py.File) -> None:
        raise NotImplementedError()      # pragma: nocover
