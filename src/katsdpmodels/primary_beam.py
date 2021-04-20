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
from typing import Tuple, ClassVar, Union, Optional, Type, TypeVar, Any
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


_P = TypeVar('_P', bound='PrimaryBeamAperturePlane')


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


class OutputType(enum.Enum):
    JONES_HV = 1
    """Jones matrix with linear basis corresponding to horizontal and vertical
    directions. See :class:`PrimaryBeam` for sign conventions.
    """

    JONES_XY = 2
    """Jones matrix with linear basis corresponding to the IAU X (north) and Y
    (east) directions on the celestial sphere.
    """

    MUELLER = 3
    """A 4x4 Mueller matrix describing the effect on each Stokes parameter
    (IQUV), assuming that both antennas share the same beam.
    """

    UNPOLARIZED_POWER = 4
    """Scalar power attenuation of unpolarized sources, assuming that both
    antennas share the same beam. This is the same as the first element of
    :data:`MUELLER`.
    """


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

    This provides a normalized primary beam, describing the effect on off-axis
    pointing relative to the pointing centre. Thus, at the pointing centre it
    is the identity Jones matrix. If the primary beam Jones matrix in a given
    direction (returned from this model) is :math:`E` and the
    direction-independent effects such as receiver gain and leakage are
    :math:`G`, then the combined effect is :math:`GE`.
    """

    model_type: ClassVar[Literal['primary_beam']] = 'primary_beam'

    def spatial_resolution(self, frequency: u.Quantity) -> float:
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

    def inradius(self, frequency: u.Quantity) -> float:
        """Maximum distance from the pointing centre at which model has full coverage."""
        raise NotImplementedError()

    def circumradius(self, frequency: u.Quantity) -> float:
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
    def antenna(self) -> Optional[str]:
        """The antenna to which this model applies.

        If this model is not antenna-specific or does not carry this
        information, it will be ``None``.
        """
        raise NotImplementedError()

    @property
    def receiver(self) -> Optional[str]:
        """The receiver ID to which this model applies.

        If this model is not specific to a single receiver or the model does
        not carry this information, it will be ``None``.
        """
        raise NotImplementedError()

    @property
    def band(self) -> Optional[str]:
        """String identifier of the receiver band to which this model applies.

        If not known, it will be ``None``.
        """

    def sample(self, l: ArrayLike, m: ArrayLike, frequency: u.Quantity,   # noqa: E741
               frame: Union[AltAzFrame, RADecFrame],
               output_type: OutputType, *,
               out: Optional[np.ndarray] = None) -> np.ndarray:
        """Sample the primary.

        A sample is returned for each combination of a position (given by `l`,
        `m`) with a frequency. The dimensions of the output will be first those
        of `frequency`, then those of `m` and `l` (which are broadcast with
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
            The value to compute. See :class:`OutputType` for details.
        out
            If specified, provides the memory into which the result will be
            written. It must have the correct shape the dtype must be
            ``complex64``.

        Raises
        ------
        ValueError
            if `output_type` is :data:`OutputType.JONES_XY` and `frame` is not
            an instance of :class:`RADecFrame`.
        ValueError
            if `out` is specified and has the wrong shape.
        TypeError
            if `out` is specified and has the wrong dtype.
        astropy.units.UnitConversionError
            if `frequency` is not specified with a spectral unit
        """
        raise NotImplementedError()

    def sample_grid(self, l: ArrayLike, m: ArrayLike, frequency: u.Quantity,   # noqa: E741
                    frame: Union[AltAzFrame, RADecFrame],
                    output_type: OutputType, *,
                    out: Optional[np.ndarray] = None) -> np.ndarray:
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

    @classmethod
    def from_hdf5(cls, hdf5: h5py.File) -> 'PrimaryBeam':
        model_format = models.get_hdf5_attr(hdf5.attrs, 'model_format', str) or ''
        if model_format == 'aperture_plane':
            return PrimaryBeamAperturePlane.from_hdf5(hdf5)
        else:
            raise models.ModelFormatError(
                f'Unknown model_format {model_format!r} for {cls.model_type}')

    def to_hdf5(self, hdf5: h5py.File) -> None:
        raise NotImplementedError()      # pragma: nocover


class PrimaryBeamAperturePlane(PrimaryBeam):
    """Primary beam model represented in the aperture plane.

    This is a Fourier transform of the primary beam response. See
    :doc:`user/formats` for details.
    """

    model_format: ClassVar[Literal['aperture_plane']] = 'aperture_plane'

    def __init__(
            self,
            x_start: u.Quantity, y_start: u.Quantity,
            x_step: u.Quantity, y_step: u.Quantity,
            frequencies: u.Quantity, samples: u.Quantity,
            *,
            antenna: Optional[str] = None,
            receiver: Optional[str] = None,
            band: Optional[str] = None):
        super().__init__()
        # Canonicalise the units to simplify to_hdf5 (and also remove the
        # cost of conversions when methods are called with canonical units,
        # with a side benefit of failing hard if the wrong units are
        # provided).
        self.x_start = x_start.to(u.m)
        self.y_start = y_start.to(u.m)
        self.x_step = x_step.to(u.m)
        self.y_step = y_step.to(u.m)
        self.frequencies = frequencies.astype(np.float32, copy=False, casting='same_kind')
        self.samples = samples.astype(np.complex64, copy=False, casting='same_kind')
        if len(frequencies) > 1:
            self._frequency_resolution = np.min(np.diff(frequencies))
            if self._frequency_resolution <= 0 * u.Hz:
                raise ValueError('Frequencies must be strictly increasing')
        else:
            self._frequency_resolution = 0 * u.Hz
        self._antenna = antenna
        self._receiver = receiver
        self._band = band

    @property
    def x(self) -> np.ndarray:
        """x coordinates associated with the samples."""
        return np.arange(self.samples.shape[-1]) * self.x_step + self.x_start

    @property
    def y(self) -> np.ndarray:
        """y coordinates associated with the samples."""
        return np.arange(self.samples.shape[-2]) * self.y_step * self.y_start

    def spatial_resolution(self, frequency: u.Quantity) -> u.Quantity:
        # Compute the Nyquist frequency, taking the minimum between x and y
        x = self.x
        y = self.y
        scale = min(max(abs(x[0]), abs(x[-1])), max(abs(y[0]), y[-1]))
        wavelength = frequency.to(u.m, equivalencies=u.spectral(), copy=False)
        return 0.5 * wavelength / scale

    def frequency_range(self) -> Tuple[u.Quantity, u.Quantity]:
        return self.frequencies[0], self.frequencies[-1]

    def frequency_resolution(self) -> u.Quantity:
        return self._frequency_resolution

    def inradius(self, frequency: u.Quantity) -> float:
        wavelength = frequency.to(u.m, equivalencies=u.spectral(), copy=False)
        # Assume that the model extends to the Nyquist limit
        return 0.5 * float(wavelength / max(self.x_step, self.y_step))

    def circumradius(self, frequency: u.Quantity) -> float:
        wavelength = frequency.to(u.m, equivalencies=u.spectral(), copy=False)
        return 0.5 * float(np.hypot(wavelength / self.x_step, wavelength / self.y_step))

    @property
    def is_circular(self) -> bool:
        return False

    @property
    def is_unpolarized(self) -> bool:
        return False

    @property
    def antenna(self) -> Optional[str]:
        return self._antenna

    @property
    def receiver(self) -> Optional[str]:
        return self._receiver

    @property
    def band(self) -> Optional[str]:
        return self._band

    @classmethod
    def from_hdf5(cls: Type[_P], hdf5: h5py.File) -> _P:
        samples = models.get_hdf5_dataset(hdf5, 'samples')
        samples = models.require_columns('samples', samples, np.complex64, 5)
        frequencies = models.get_hdf5_dataset(hdf5, 'frequencies')
        frequencies = models.require_columns('frequencies', frequencies, np.float32, 1)
        frequencies <<= u.Hz
        if samples.shape[:2] != (2, 2):
            raise models.DataError('samples must by 2x2 on leading dimensions')
        if frequencies.shape[0] != samples.shape[2]:
            raise models.DataError('samples and frequencies have inconsistent sizes')
        x_start = models.get_hdf5_attr(hdf5, 'x_start', float, required=True) * u.m
        y_start = models.get_hdf5_attr(hdf5, 'y_start', float, required=True) * u.m
        x_step = models.get_hdf5_attr(hdf5, 'x_step', float, required=True) * u.m
        y_step = models.get_hdf5_attr(hdf5, 'y_step', float, required=True) * u.m
        return cls(x_start, y_start, x_step, y_step, frequencies, samples,
                   antenna=models.get_hdf5_attr(hdf5, 'antenna', str),
                   receiver=models.get_hdf5_attr(hdf5, 'receiver', str),
                   band=models.get_hdf5_attr(hdf5, 'band', str))
