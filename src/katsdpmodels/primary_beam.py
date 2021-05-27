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
    from numpy.typing import ArrayLike, DTypeLike
except ImportError:
    ArrayLike = Any  # type: ignore
    DTypeLike = Any  # type: ignore
import numba
import scipy.interpolate
import astropy.units as u
from astropy.coordinates import SkyCoord
import h5py

from . import models


_P = TypeVar('_P', bound='PrimaryBeamAperturePlane')

# Matrix to convert Stokes parameters to XY-basis brightness vector.
# Labeled S in eq (25) of Smirnov 2011, Revisiting the radio interferometer
# measurement equation. I. A full-sky Jones formalism.
_IQUV_TO_XY = np.array(
    [
        [1, 1, 0, 0],
        [0, 0, 1, 1j],
        [0, 0, 1, -1j],
        [1, -1, 0, 0]
    ],
    dtype=np.complex64
)
# Inverse of _IQUV_TO_XY
_XY_TO_IQUV = np.linalg.inv(_IQUV_TO_XY)


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

    This currently only models parallactic angle rotation. Relativistic
    aberration can cause a small scaling of offsets between barycentric and
    topocentric frames, but it's ignored as it is typically much smaller than
    features of the primary beam.

    This class should not be constructed directly, as the constructor
    signature is subject to change in future. Use the factory methods
    such as :meth:`from_parallactic_angle` and :meth:`from_sky_coord`.

    Parameters
    ----------
    parallactic_angle
        Rotation angle to align the celestial sphere with the antenna. See
        :meth:`katpoint.Target.parallactic_angle` for a definition.
    """

    def __init__(self, parallactic_angle: u.Quantity) -> None:
        self.parallactic_angle = parallactic_angle.to(u.rad)  # Just to verify unit type

    def lm_to_hv(self) -> np.ndarray:
        """Matrix that converts (ra, dec) offsets to (h, v) offsets."""
        c = np.cos(self.parallactic_angle)
        s = np.sin(self.parallactic_angle)
        # Note: this is reflected rotation matrix, because RADec and AltAz
        # have opposite handedness.
        return np.array([[-c, s], [s, c]])

    def jones_hv_to_xy(self) -> np.ndarray:
        """Jones matrix that converts voltages from HV to XY.

        See :class:`OutputType` for further clarification.
        """
        c = np.cos(self.parallactic_angle)
        s = np.sin(self.parallactic_angle)
        # No handedness change, but H aligns with X and V with Y at a
        # PA of 90°.
        return np.array([[s, c], [-c, s]])

    @classmethod
    def from_parallactic_angle(cls, parallactic_angle: u.Quantity) -> 'RADecFrame':
        """Generate a frame from a parallactic angle."""
        return cls(parallactic_angle)

    @classmethod
    def from_sky_coord(cls, target: SkyCoord) -> 'RADecFrame':
        """Generate a frame from a target (assuming an AltAz mount).

        The `target` must have ``obstime`` and ``location`` properties, and
        must be scalar. It will be converted to ICRS if necessary.
        """
        # Construct a point that is displaced from the pointing by a small
        # quantity towards the nearest pole (north or south). It's necessary to
        # use a small finite difference rather than the pole itself, because
        # the transformation to AzEl is not rigid (does not preserve great
        # circles).
        target_icrs = target.icrs
        if target_icrs.dec > 0:
            sign = -1
        else:
            sign = 1
        pole = target_icrs.directional_offset_by(0 * u.rad, sign * 1e-5 * u.rad)
        # directional_offset_by doesn't preserve these extra attributes
        pole.obstime = target_icrs.obstime
        pole.location = target_icrs.location
        pa = target.altaz.position_angle(pole.altaz)
        if sign == -1:
            pa += np.pi * u.rad
        return cls.from_parallactic_angle(pa)


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
    (IQUV), assuming that both antennas share the same beam and parallactic
    angle.
    """

    UNPOLARIZED_POWER = 4
    """Scalar power attenuation of unpolarized sources, assuming that both
    antennas share the same beam and parallactic angle. This is the same as the
    first element of :data:`MUELLER`.
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

    def spatial_resolution(self, frequency: u.Quantity) -> np.ndarray:
        """Approximate spatial resolution of the model, in units of projected coordinates.

        Sampling a grid at significantly higher resolution than this will have
        diminishing returns compared to sampling at this resolution and
        interpolating.
        """
        raise NotImplementedError()      # pragma: nocover

    @property
    def frequency_range(self) -> Tuple[u.Quantity, u.Quantity]:
        """Minimum and maximum frequency covered by the model."""
        raise NotImplementedError()      # pragma: nocover

    @property
    def frequency_resolution(self) -> u.Quantity:
        """Approximate frequency resolution of the model.

        Sampling at significantly higher spectral resolution than this will
        have diminishing returns compared to sampling at this resolution and
        interpolating.
        """
        raise NotImplementedError()      # pragma: nocover

    def inradius(self, frequency: u.Quantity) -> float:
        """Maximum distance from the pointing centre at which model has full coverage."""
        raise NotImplementedError()      # pragma: nocover

    def circumradius(self, frequency: u.Quantity) -> float:
        """Maximum distance from the pointing centre at which model has any coverage."""
        raise NotImplementedError()      # pragma: nocover

    @property
    def is_circular(self) -> bool:
        """Whether this model is circularly-symmetric about the pointing centre."""
        raise NotImplementedError()      # pragma: nocover

    @property
    def is_unpolarized(self) -> bool:
        """Whether this model ignores polarization.

        If true, it is guaranteed that the Jones matrices describing the beam are
        scaled identity matrices.
        """
        raise NotImplementedError()      # pragma: nocover

    @property
    def antenna(self) -> Optional[str]:
        """The antenna to which this model applies.

        If this model is not antenna-specific or does not carry this
        information, it will be ``None``.
        """
        raise NotImplementedError()      # pragma: nocover

    @property
    def receiver(self) -> Optional[str]:
        """The receiver ID to which this model applies.

        If this model is not specific to a single receiver or the model does
        not carry this information, it will be ``None``.
        """
        raise NotImplementedError()      # pragma: nocover

    @property
    def band(self) -> str:
        """String identifier of the receiver band to which this model applies."""

    def sample(self, l: ArrayLike, m: ArrayLike, frequency: u.Quantity,   # noqa: E741
               frame: Union[AltAzFrame, RADecFrame],
               output_type: OutputType, *,
               out: Optional[np.ndarray] = None) -> np.ndarray:
        """Sample the primary beam.

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
            written. It must have the correct shape, the dtype must be
            ``complex64`` (or ``float32`` for
            :data:`OutputType.UNPOLARIZED_POWER`) and it must be C
            contiguous.

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
        raise NotImplementedError()      # pragma: nocover

    def sample_grid(self, l: ArrayLike, m: ArrayLike, frequency: u.Quantity,   # noqa: E741
                    frame: Union[AltAzFrame, RADecFrame],
                    output_type: OutputType, *,
                    out: Optional[np.ndarray] = None) -> np.ndarray:
        """Sample the primary beam on a grid aligned to the axes.

        This is equivalent to
        :code:`sample(l[np.newaxis, :], m[:, np.newaxis], ...)`, but may be
        significantly faster (depending on the implementation), and is not
        guaranteed to give bit-identical results. This advantage may be lost
        when using :class:`RADecFrame` with a non-zero parallactic angle.

        The grid need not be regularly spaced, but an output is generated for
        each combination of `l`, `m` and `frequency`.

        Refer to :meth:`sample` for further details of the parameters and
        return value.
        """
        raise NotImplementedError()      # pragma: nocover

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


@numba.vectorize([numba.complex64(numba.float32)])
def _expjm2pi(x):
    """Equivalent to ``exp(-2j * np.pi * x)`` where `x` is real.

    x is reduced to a small value before multiplication, which
    improves precision at a small cost in performance.
    """
    y = np.float32(-2 * np.pi) * (x - np.rint(x))
    return complex(np.cos(y), np.sin(y))


def _asarray(x: ArrayLike, dtype: Optional[DTypeLike] = None) -> np.ndarray:
    """Convert an array-like to an array.

    Unlike np.ndarray, this will reject astropy Quantities with dimensions
    and convert dimensionless quantities correctly even if they have scale.

    When a dtype is specified, uses ``same_kind`` casting.
    """
    if isinstance(x, u.Quantity):
        array = x.to_value(u.dimensionless_unscaled)
    else:
        array = np.asarray(x)
    if dtype is not None:
        array = array.astype(dtype, copy=False, casting='same_kind')
    return array


def _check_out(out: Optional[np.ndarray], output_type: OutputType) -> None:
    if out is not None:
        expected_dtype: np.dtype
        if output_type != OutputType.UNPOLARIZED_POWER:
            expected_dtype = np.dtype(np.complex64)
        else:
            expected_dtype = np.dtype(np.float32)
        if out.dtype != expected_dtype:
            raise TypeError(f'out must have dtype {expected_dtype}, not {out.dtype}')
        if not out.flags.c_contiguous:
            raise ValueError('out must be C contiguous')


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
            frequency: u.Quantity, samples: np.ndarray,
            *,
            band: str,
            antenna: Optional[str] = None,
            receiver: Optional[str] = None) -> None:
        super().__init__()
        # Canonicalise the units to simplify to_hdf5 (and also remove the
        # cost of conversions when methods are called with canonical units,
        # with a side benefit of failing hard if the wrong units are
        # provided).
        self.x_start = x_start.to(u.m)
        self.y_start = y_start.to(u.m)
        self.x_step = x_step.to(u.m)
        self.y_step = y_step.to(u.m)
        self.frequency = frequency.astype(np.float32, copy=False, casting='same_kind')
        if len(frequency) > 1:
            self._frequency_resolution = np.min(np.diff(frequency))
            if self._frequency_resolution <= 0 * u.Hz:
                raise ValueError('frequencies must be strictly increasing')
        else:
            # We can set _frequency_resolution easily enough, but
            # scipy.interpolate also refuses to work with just a single (or zero)
            # elements on the interpolation axis.
            raise NotImplementedError('at least 2 frequencies are currently required')
        self.samples = samples.astype(np.complex64, copy=False, casting='same_kind')
        scale = samples.shape[-1] * samples.shape[-2]   # Normalisation factor
        self._interp_samples = scipy.interpolate.interp1d(
            self.frequency.to_value(u.Hz), self.samples / scale,
            axis=0, copy=False, bounds_error=False, fill_value=np.nan,
            assume_sorted=True)
        self._antenna = antenna
        self._receiver = receiver
        self._band = band

    @property
    def x(self) -> u.Quantity:
        """x coordinates associated with the samples."""
        return np.arange(self.samples.shape[-1]) * self.x_step + self.x_start

    @property
    def y(self) -> u.Quantity:
        """y coordinates associated with the samples."""
        return np.arange(self.samples.shape[-2]) * self.y_step + self.y_start

    def spatial_resolution(self, frequency: u.Quantity) -> np.ndarray:
        # Compute the Nyquist frequency, taking the maximum between x and y
        x = self.x
        y = self.y
        scale = max(max(abs(x[0]), abs(x[-1])), max(abs(y[0]), y[-1]))
        wavelength = frequency.to(u.m, equivalencies=u.spectral(), copy=False)
        return (0.5 * wavelength / scale).to_value(u.dimensionless_unscaled)

    def frequency_range(self) -> Tuple[u.Quantity, u.Quantity]:
        return self.frequency[0], self.frequency[-1]

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
    def band(self) -> str:
        return self._band

    def _prepare_samples(self, frequency: u.Quantity,
                         frame: Union[AltAzFrame, RADecFrame],
                         output_type: OutputType) -> np.ndarray:
        """Interpolate aperture plane to selected frequencies, and optionally rotate."""
        frequency_Hz = frequency.to_value(u.Hz).astype(np.float32, copy=False, casting='same_kind')
        samples = self._interp_samples(frequency_Hz)
        if output_type in {OutputType.JONES_XY, OutputType.MUELLER}:
            if not isinstance(frame, RADecFrame):
                raise ValueError('JONES_XY required a RADecFrame')
            jones = frame.jones_hv_to_xy().astype(np.complex64, copy=False)
            # Matrix multiply, but tensordot/matmul would require shuffling
            # the axes around. The Jones dimensions are axes -4 and -3 on both
            # the input and output (i, j and k refer to the axes involved in
            # the matrix multiply).
            # TODO: see if it's faster to move into Numba inner loop.
            samples = np.einsum('ij,...jkxy->...ikxy', jones, samples)
        return samples

    @staticmethod
    def _finalize(values: np.ndarray, output_type: OutputType,
                  out: Optional[np.ndarray]) -> np.ndarray:
        """Handle :data:`OutputType.MUELLER` and `OutputType.UNPOLARIZED_POWER`.

        Parameters
        ----------
        values
            Stacked 2x2 Jones matrices, suitable for `output_type`
        output_type
            Target output type, either :data:`OutputType.MUELLER` or
            :data:`OutputType.UNPOLARIZED_POWER`.
        """
        if output_type == OutputType.MUELLER:
            conj = np.conj(values)
            # Take Kronecker product. Unfortunately np.kron doesn't allow
            # operating over subsets of dimensions.
            M = np.block([[values[..., 0:1, 0:1] * conj, values[..., 0:1, 1:2] * conj],
                          [values[..., 1:2, 0:1] * conj, values[..., 1:2, 1:2] * conj]])
            return np.matmul(_XY_TO_IQUV @ M, _IQUV_TO_XY, out=out)
        elif output_type == OutputType.UNPOLARIZED_POWER:
            # Compute sum of squared magnitudes across the 4 Jones terms.
            # Viewing at float32 simplifies summing squared magnitudes.
            assert values.dtype == np.dtype(np.complex64)
            ret = np.sum(np.square(values.view(np.float32)), axis=(-2, -1), out=out)
            ret *= 0.5
            return ret
        else:
            raise ValueError(f'Unrecognised output_type {output_type}')

    @staticmethod
    @numba.njit
    def _sample_impl(aperture: np.ndarray, xf: np.ndarray, yf: np.ndarray,
                     l: np.ndarray, m: np.ndarray, out: np.ndarray) -> None:
        """Numba implementation details of :meth:`_sample_altaz_jones`.

        Parameters
        ----------
        aperture
            Aperture-plane samples, already interpolated onto the desired frequencies
        xf, yf
            x and y in wavelength, with the frequency axes first.
        l, m
            1D l and m coordinates (already broadcast with each other)
        out
            Output array
        """
        for freq_idx in np.ndindex(xf.shape[:-1]):
            # Frequency-specific 1D arrays
            x = xf[freq_idx]
            y = yf[freq_idx]
            ap = aperture[freq_idx]
            coeff1 = _expjm2pi(np.outer(l, x))
            coeff2 = _expjm2pi(np.outer(m, y))
            for i in range(2):
                for j in range(2):
                    tmp = coeff2 @ ap[i, j]
                    out_chunk = out[freq_idx + (...,) + (i, j)]
                    for lm_idx in np.ndindex(l.shape):
                        out_chunk[lm_idx] = np.dot(tmp[lm_idx], coeff1[lm_idx])

    def _sample_altaz_jones(
            self, l: np.ndarray, m: np.ndarray, frequency: u.Quantity,
            frame: Union[AltAzFrame, RADecFrame], output_type: OutputType, *,
            out: Optional[np.ndarray] = None) -> np.ndarray:
        """Partial implementation of :meth:`sample`.

        It takes `l` and `m` in AltAz frame, and produces Jones matrices in
        either :data:`OutputType.JONES_HV` or :data:`OutputType.JONES_XY`.
        The provided `frame` and `output_type` are used only for polarization
        rotation.

        l and m must already be broadcast to the same shape, and must be float32.
        """
        assert l.shape == m.shape
        assert l.dtype == np.float32
        assert m.dtype == np.float32

        out_shape = frequency.shape + l.shape + (2, 2)
        if out is None:
            out = np.empty(out_shape, np.complex64)
        elif out.shape != out_shape:
            raise ValueError(f'out must have shape {out_shape}, not {out.shape}')

        # Compute x and y in wavelengths
        wavenumber = frequency.to('m^-1', equivalencies=u.spectral())
        xf = _asarray(np.multiply.outer(wavenumber, self.x), np.float32)
        yf = _asarray(np.multiply.outer(wavenumber, self.y), np.float32)
        # Numba can't handle the broadcasting involved in multi-dimensional
        # l/m, so flatten. Assign to shape instead of reshape to ensure no
        # copying.
        out_view = out.view()
        out_view.shape = frequency.shape + (l.size, 2, 2)
        samples = self._prepare_samples(frequency, frame, output_type)
        self._sample_impl(samples, xf, yf, l.ravel(), m.ravel(), out_view)

        # Check if there are any points that may lie outside the valid l/m
        # region. If not (common case) we can avoid computing masks.
        max_l = np.max(np.abs(l))
        max_m = np.max(np.abs(m))
        max_wavenumber = np.max(wavenumber)
        limit_l = 0.5 / abs(self.x_step)
        limit_m = 0.5 / abs(self.y_step)
        if (max_l * max_wavenumber > limit_l
                or max_m * max_wavenumber > limit_m):
            invalid = (
                (np.multiply.outer(wavenumber, np.abs(l)) > limit_l)
                | (np.multiply.outer(wavenumber, np.abs(m)) > limit_m)
            )
            # Add the axes for the Jones matrix dimensions
            invalid = invalid[..., np.newaxis, np.newaxis]
            np.copyto(out, np.nan, where=invalid)

        return out

    def sample(self, l: ArrayLike, m: ArrayLike, frequency: u.Quantity,   # noqa: E741
               frame: Union[AltAzFrame, RADecFrame],
               output_type: OutputType, *,
               out: Optional[np.ndarray] = None) -> np.ndarray:
        _check_out(out, output_type)
        l_ = _asarray(l)
        m_ = _asarray(m)
        l_, m_ = np.broadcast_arrays(l_, m_)
        # numba seems to trigger a FutureWarning when it checks the writeable
        # flag on these broadcast arrays. Suppress it by making them
        # explicitly readonly.
        l_.flags.writeable = False
        m_.flags.writeable = False
        if isinstance(frame, RADecFrame):
            # Form a matrix with just two rows
            lm = np.stack([l_.ravel(), m_.ravel()], axis=0)
            # Convert to AltAz frame
            lm = frame.lm_to_hv() @ lm
            # Unpack again
            l_ = lm[0].reshape(l_.shape)
            m_ = lm[1].reshape(m_.shape)
        elif not isinstance(frame, AltAzFrame):
            raise TypeError(f'frame must be RADecFrame or AltAzFrame, not {type(frame)}')
        l_ = _asarray(l_, np.float32)
        m_ = _asarray(m_, np.float32)

        if output_type in {OutputType.JONES_XY, OutputType.JONES_HV}:
            return self._sample_altaz_jones(l_, m_, frequency, frame, output_type, out=out)
        else:
            jones = self._sample_altaz_jones(l_, m_, frequency, frame, output_type)
            return self._finalize(jones, output_type, out=out)

    @staticmethod
    @numba.njit
    def _sample_grid_impl(x_m: np.ndarray, y_m: np.ndarray,
                          l: np.ndarray, m: np.ndarray,
                          wavenumber: np.ndarray,
                          samples: np.ndarray,
                          out: np.ndarray) -> None:
        for freq_idx in np.ndindex(wavenumber.shape):
            x = x_m * wavenumber[freq_idx]
            y = y_m * wavenumber[freq_idx]
            coeff1 = _expjm2pi(np.outer(l, x))
            coeff2 = _expjm2pi(np.outer(m, y))
            # Shove the polarizations into extra columns in a matrix. Matrix
            # multiplication treats the columns in the RHS independently,
            # which is what we want for polarizations, and the result then
            # has the right memory layout.
            tmp = np.zeros((len(y), len(l) * 4), np.complex64)
            s = samples[freq_idx]
            for i in range(2):
                for j in range(2):
                    tmp[:, (i * 2 + j)::4] = s[i, j] @ coeff1.T
            out[freq_idx] = (coeff2 @ tmp).reshape(out[freq_idx].shape)

    def _sample_grid_altaz_jones(
            self, l: np.ndarray, m: np.ndarray, frequency: u.Quantity, *,
            out: Optional[np.ndarray] = None) -> np.ndarray:
        """Partial implementation of :meth:`sample_grid`.

        It takes `l` and `m` in AltAz frame and produces Jones HV matrices.
        """
        assert l.ndim == 1
        assert l.shape == m.shape
        assert l.dtype == np.dtype(np.float32)
        assert m.dtype == np.dtype(np.float32)
        x_m = _asarray(self.x.to_value(u.m), np.float32)
        y_m = _asarray(self.y.to_value(u.m), np.float32)
        wavenumber = frequency.to('m^-1', equivalencies=u.spectral())
        wavenumber_m = _asarray(wavenumber.value, np.float32)
        samples = self._prepare_samples(frequency, AltAzFrame(), OutputType.JONES_HV)
        out_shape = frequency.shape + m.shape + l.shape + (2, 2)
        if out is None:
            out = np.empty(out_shape, np.complex64)
        elif out.shape != out_shape:
            raise ValueError(f'out must have shape {out_shape}, not {out.shape}')
        self._sample_grid_impl(x_m, y_m, l, m, wavenumber_m, samples, out)

        # Check if there are any points that may lie outside the valid l/m
        # region. If not (common case) we can avoid computing masks.
        max_l = np.max(np.abs(l))
        max_m = np.max(np.abs(m))
        max_wavenumber = np.max(wavenumber)
        limit_l = 0.5 / abs(self.x_step)
        limit_m = 0.5 / abs(self.y_step)
        if max_l * max_wavenumber > limit_l:
            invalid = np.multiply.outer(wavenumber, np.abs(l)) > limit_l
            # Insert axes for m and Jones terms
            invalid = invalid[..., np.newaxis, :, np.newaxis, np.newaxis]
            np.copyto(out, np.nan, where=invalid)
        if max_m * max_wavenumber > limit_m:
            invalid = np.multiply.outer(wavenumber, np.abs(m)) > limit_m
            # Insert axes for l and Jones terms
            invalid = invalid[..., np.newaxis, np.newaxis, np.newaxis]
            np.copyto(out, np.nan, where=invalid)

        return out

    def sample_grid(self, l: ArrayLike, m: ArrayLike, frequency: u.Quantity,   # noqa: E741
                    frame: Union[AltAzFrame, RADecFrame],
                    output_type: OutputType, *,
                    out: Optional[np.ndarray] = None) -> np.ndarray:
        # This is a completely unoptimised placeholder implementation.
        l_ = _asarray(l, np.float32)
        m_ = _asarray(m, np.float32)
        if l_.ndim != 1 or m_.ndim != 1:
            raise ValueError('l and m must be 1D')
        if (not isinstance(frame, AltAzFrame)
                or output_type not in {OutputType.JONES_HV, OutputType.UNPOLARIZED_POWER}):
            # Can't take advantage of separability when applying an arbitrary
            # parallactic angle rotation. However, there may be some value in
            # supporting a fast path when PA is a multiple of pi/2, for users
            # that will interpolate from that grid to other parallactic
            # angles but don't want to deal with the handedness change.
            return self.sample(
                l_[np.newaxis, :], m_[:, np.newaxis], frequency,
                frame, output_type, out=out)
        else:
            _check_out(out, output_type)
            if output_type == OutputType.JONES_HV:
                return self._sample_grid_altaz_jones(l_, m_, frequency, out=out)
            else:
                mid = self._sample_grid_altaz_jones(l_, m_, frequency)
                return self._finalize(mid, output_type, out=out)

    @classmethod
    def from_hdf5(cls: Type[_P], hdf5: h5py.File) -> _P:
        samples = models.get_hdf5_dataset(hdf5, 'aperture_plane')
        samples = models.require_columns('aperture_plane', samples, np.complex64, 5)
        frequency = models.get_hdf5_dataset(hdf5, 'frequency')
        frequency = models.require_columns('frequency', frequency, np.float32, 1)
        # Quantity doesn't play nice with h5py numpy-like's, so force loading
        frequency = frequency[:]
        frequency <<= u.Hz
        if samples.shape[1:3] != (2, 2):
            raise ValueError('aperture_plane must by 2x2 on Jones dimensions')
        if frequency.shape[0] != samples.shape[0]:
            raise ValueError('aperture_plane and frequency have inconsistent sizes')
        attrs = hdf5.attrs
        x_start = models.get_hdf5_attr(attrs, 'x_start', float, required=True) * u.m
        y_start = models.get_hdf5_attr(attrs, 'y_start', float, required=True) * u.m
        x_step = models.get_hdf5_attr(attrs, 'x_step', float, required=True) * u.m
        y_step = models.get_hdf5_attr(attrs, 'y_step', float, required=True) * u.m
        return cls(x_start, y_start, x_step, y_step, frequency, samples,
                   antenna=models.get_hdf5_attr(attrs, 'antenna', str),
                   receiver=models.get_hdf5_attr(attrs, 'receiver', str),
                   band=models.get_hdf5_attr(attrs, 'band', str, required=True))

    def to_hdf5(self, hdf5: h5py.File) -> None:
        hdf5.attrs['band'] = self.band
        if self.antenna is not None:
            hdf5.attrs['antenna'] = self.antenna
        if self.receiver is not None:
            hdf5.attrs['receiver'] = self.receiver
        hdf5.attrs['x_start'] = self.x_start.to_value(u.m)
        hdf5.attrs['x_step'] = self.x_step.to_value(u.m)
        hdf5.attrs['y_start'] = self.y_start.to_value(u.m)
        hdf5.attrs['y_step'] = self.y_step.to_value(u.m)
        hdf5.create_dataset(
            'frequency', data=self.frequency.to_value(u.Hz), track_times=False)
        # Use chunked storage so that individual frequencies can be loaded
        # independently in future.
        hdf5.create_dataset(
            'aperture_plane',
            data=self.samples,
            chunks=(1,) + self.samples.shape[1:],
            track_times=False
        )

    @classmethod
    def from_katholog(cls: Type[_P], model, *,
                      antenna: Optional[str] = None,
                      band: Optional[str] = None) -> _P:
        """Load a model represented in the :mod:`katholog` package.

        Parameters
        ----------
        model : :class:`katholog.Aperture`
            The katholog model from which to load.
        antenna
            The antenna name for which to load data. If no value is specified,
            the array average is loaded.
        band
            The name of the band to set in the returned model. If no value is
            provided, will try to determine it from the katholog model.

        Raises
        ------
        ValueError
            If `model` does not indicate a band and `band` was not provided.
        """
        if antenna is None:
            antenna_idx = -1
        else:
            antenna_idx = model.scanantennanames.index(antenna)

        frequency = model.freqMHz * u.MHz
        x_start = -0.5 * model.mapsize * u.m
        y_start = x_start
        x_step = model.mapsize / model.gridsize * u.m
        y_step = x_step
        # Select only the desired antenna
        samples = model.apert[:, antenna_idx]
        # katholog stores polarizations as HH, HV, VH, VV (first letter is
        # feed, second is radiation). Reshape into Jones matrix.
        samples = samples.reshape((2, 2) + samples.shape[1:])
        # Move the Jones axes to the trailing dimensions for normalisation
        samples = np.moveaxis(samples, (0, 1), (3, 4))
        # Normalise samples so that the central value (in the image plane, which is
        # the mean of the aperture-plane values) is the identity.
        c = np.mean(samples, axis=(1, 2), keepdims=True)
        samples = np.linalg.inv(c) @ samples
        # Move the Jones axes to their proper place
        samples = np.moveaxis(samples, (3, 4), (1, 2))

        if band is None and hasattr(model.env, 'band'):
            # Undo katdal band renaming
            BAND_RENAME = {'L': 'l', 'UHF': 'u', 'S': 's'}
            band = str(model.env.band)   # It's originally a numpy scalar string
            band = BAND_RENAME.get(band, band)
        if band is None:
            raise ValueError('Model does not indicate band - it must be passed explicitly')

        if antenna is not None:
            receiver = model.env.receivers[antenna]
        else:
            receiver = None
        return cls(x_start, y_start, x_step, y_step, frequency, samples,
                   antenna=antenna, band=band, receiver=receiver)
