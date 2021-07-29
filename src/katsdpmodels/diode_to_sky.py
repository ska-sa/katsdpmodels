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

"""Diode-to-Sky Models"""
import numpy as np

import astropy.units as u
import h5py
import logging
import io
import scipy.interpolate

from pathlib import Path
from typing import Any, BinaryIO, ClassVar, Optional, Tuple, Type, TypeVar, Union
from typing_extensions import Literal

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any  # type: ignore

from . import models

# use a type alias for file_like objects
_FileLike = Union[io.IOBase, io.BytesIO, BinaryIO]

_B = TypeVar('_B', bound='BSplineModel')

logger = logging.getLogger(__name__)


class NoDiodeToSkyModelError(Exception):
    """Attempted to load a bandpass phase model but it does not exist"""
    pass


class DiodeToSkyModel(models.SimpleHDF5Model):
    """
    Base class for diode-to-sky bandpass phase models.

    A 'Diode-to-Sky' model is a bandpass phase model representing the phase modification applied to
    a signal due to all components that are not captured during noise-diode calibration,
    (such as the effect of the ionosphere and other atmospheric sources, persistent low-level
    RFI, the OMT, etc). During cal, this phase screen model is removed from the signal.

    model_type: diode_to_sky

    Diode-to-Sky has the following attributes:
    """
    model_type: ClassVar[Literal['diode_to_sky']] = 'diode_to_sky'

    @property
    def band(self) -> str:
        """String identifier of the receiver band to which this model applies."""

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
    def frequency_range(self) -> Tuple[u.Quantity, u.Quantity]:
        """Minimum and maximum frequency covered by the model."""
        raise NotImplementedError()  # pragma: nocover

    @property
    def frequency_resolution(self) -> u.Quantity:
        """Approximate frequency resolution of the model."""
        raise NotImplementedError()      # pragma: nocover

    @classmethod
    def from_hdf5(cls, hdf5: h5py.File) -> 'DiodeToSkyModel':
        raise NotImplementedError()  # pragma: nocover

    def to_hdf5(self, hdf5: h5py.File) -> None:
        raise NotImplementedError()  # pragma: nocover


class BSplineModel(DiodeToSkyModel):
    """
    captures set of knot locations and spline parameters as a scipy.interpolate.bspline object
    model_format: 'scipy.interpolate.BSpline'

    """

    model_format: ClassVar[Literal['scipy_b_spline']] = 'scipy_b_spline'

    def __init__(self,
                 knots: ArrayLike,
                 coefficients: ArrayLike,
                 degree: int,
                 *,
                 band: str,
                 antenna: Optional[str] = None,
                 receiver: Optional[str] = None) -> None:
        super().__init__()
        self._knots = knots
        self._coefficients = coefficients
        self._degree = degree
        self._band = band
        self._antenna = antenna
        self._receiver = receiver
        self._model = scipy.interpolate.BSpline(knots, coefficients, degree)

    @property
    def knots(self) -> ArrayLike:
        """A set of knot locations over the band."""
        return self._knots

    @property
    def coefficients(self) -> ArrayLike:
        """A set of coefficients characterising the BSpline model."""
        return self._coefficients

    @property
    def antenna(self) -> Optional[str]:
        return self._antenna

    @property
    def receiver(self) -> Optional[str]:
        return self._receiver

    @property
    def band(self) -> str:
        return self._band

    @classmethod
    def from_hdf5(cls: Type[_B], hdf5: h5py.File) -> _B:
        """"""
        attrs = hdf5.attrs
        band = models.get_hdf5_attr(attrs, 'band', str, required=True)
        antenna = models.get_hdf5_attr(attrs, 'band', str, required=False)
        receiver = models.get_hdf5_attr(attrs, 'receiver', str, required=False)
        degree = models.get_hdf5_attr(attrs, 'degree', int, required=True)

        knots = models.get_hdf5_dataset(hdf5, 'knots')
        knots = models.require_columns('knots', knots, np.float64, 1)
        coefs = models.get_hdf5_dataset(hdf5, 'coefs')
        coefs = models.require_columns('coefs', coefs, np.float64, 1)

        return cls(knots, coefs, degree, band=band, antenna=antenna, receiver=receiver)

    def to_hdf5(self, hdf5: h5py.File) -> None:
        """"""
        hdf5.attrs['band'] = self._band
        hdf5.attrs['degree'] = self._degree
        if self.antenna is not None:
            hdf5.attrs['antenna'] = self._antenna
        if self.receiver is not None:
            hdf5.attrs['receiver'] = self._receiver
        hdf5.create_dataset('knots', data=self._knots, track_times=False)
        hdf5.create_dataset('coefs', data=self._knots, track_times=False)

