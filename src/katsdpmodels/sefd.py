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

"""System-Equivalent Flux Density Models
Models MUST provide separate H and V SEFD.
    Needed to simulate visibilities.
Models MUST allow dish dependence.
    Required for heterogeneous MeerKAT+ array.
TODO
Models MAY provide combined (Stokes-I) SEFD.
    It’s a handy convenience for estimating image noise, but should probably be computed on the
    fly rather than stored.
Models SHOULD provide sensible values even at RFI-affected frequencies.
    Needed for simulation, and looks better in an imaging report.
"""
import numpy as np

import astropy.units as u
import enum
import h5py
import logging
import io

from typing import Any, BinaryIO, ClassVar, List, Optional, Tuple, Type, TypeVar, Union
from typing_extensions import Literal

from .models import DataError

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any  # type: ignore

from . import models

# use a type alias for file_like objects
_FileLike = Union[io.IOBase, io.BytesIO, BinaryIO]

_P = TypeVar('_P', bound='SEFDPoly')

logger = logging.getLogger(__name__)


class Pol(enum.Enum):
    H = 1
    V = 2
    Stokes_I = 3


class SEFDModel(models.SimpleHDF5Model):
    """
    Base class for SEFD models. The System-Equivalent Flux Density (SEFD) is defined as the
    flux density of a radio source that doubles the system temperature ($T_sys$) of a radiometer.
    Lower values of the SEFD indicate more sensitive performance.

    The SEFD for a particular radio telescope is a measured quantity, and thus relates to a set of
    antennas and receiver chains. If it is available, the information about these components is
    able to be stored in the model.

    TBC: This quantity is not expected to be nor modeled as significantly variable over timescales
    shorter than the measurement interval or between antennas of the same physical design.

    model_type: sefd

    SEFD has the following attributes:
    """
    model_type: ClassVar[Literal['sefd']] = 'sefd'

    @property
    def band(self) -> str:
        """String identifier of the frequency band to which this model applies."""
        raise NotImplementedError()

    @property
    def antennas(self) -> Optional[List[str]]:
        """The set of antenna IDs to which this model applies, implemented as a list of
        MK-format antenna identifiers.

        If this model is not antenna-set-specific or does not carry this
        information, it will be ``None``.
        """
        raise NotImplementedError()      # pragma: nocover

    @property
    def receivers(self) -> Optional[List[str]]:
        """The set of receiver IDs to which this model applies, implemented as a list of
        MK-format receiver identifiers.

        If this model is not specific to a single set of receivers
        or the model does not carry this information, it will be ``None``.
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
    def from_hdf5(cls, hdf5: h5py.File) -> 'SEFDModel':
        model_format = models.get_hdf5_attr(hdf5.attrs, 'model_format', str)
        if model_format == 'poly':
            return SEFDPoly.from_hdf5(hdf5)
        else:
            raise models.ModelFormatError(
                f'Unknown model_format {model_format!r} for {cls.model_type}')

    def to_hdf5(self, hdf5: h5py.File) -> None:
        raise NotImplementedError()  # pragma: nocover


class SEFDPoly(SEFDModel):
    """
    captures a polynomial SEFD model
    model_format: 'poly'
    """

    model_format: ClassVar[Literal['poly']] = 'poly'

    def __init__(self,
                 frequency: u.Quantity,
                 coefs: Tuple[ArrayLike, ArrayLike],
                 correlator_efficiency: Optional[float],
                 *,
                 band: str,
                 antennas: Optional[List[str]] = None,
                 receivers: Optional[List[str]] = None) -> None:
        super().__init__()
        self.frequency = frequency.astype(np.float32, copy=False, casting='same_kind')
        if len(frequency) > 1:
            self._frequency_resolution = np.min(np.diff(frequency))
            if self._frequency_resolution <= 0 * u.Hz:
                raise ValueError('frequencies must be strictly increasing')
        else:
            raise NotImplementedError('at least 2 frequencies are currently required')
        self.coefs = coefs
        if correlator_efficiency is not None:
            self._correlator_efficiency = correlator_efficiency
        else:
            self._correlator_efficiency = 1.0
        self._band = band
        self._antennas = antennas if antennas is not None else None
        self._receivers = receivers if receivers is not None else None

    def __call__(self, pol: Optional[Pol] = None):
        pol_sefd = np.polynomial.polynomial.polyval(
            self.frequency, self.coefs, tensor=True)
        if pol in {Pol.H, Pol.V}:
            sefd = pol_sefd  # pol_sefd[pol.value-1]
        else:  # elif pol in {Pol.Stokes_I, None}:
            sefd = np.sqrt(np.mean(np.square(pol_sefd), axis=0))
        return sefd / self.correlator_efficiency

    @property
    def frequency_range(self) -> Tuple[u.Quantity, u.Quantity]:
        return self.frequency[0], self.frequency[-1]

    @property
    def frequency_resolution(self) -> u.Quantity:
        return self._frequency_resolution

    @property
    def correlator_efficiency(self) -> Optional[float]:
        return self._correlator_efficiency

    @property
    def antennas(self) -> Optional[List[str]]:
        return self._antennas

    @antennas.setter
    def antennas(self, new_set: List[str]):
        self._antennas = new_set

    @property
    def receivers(self) -> Optional[List[str]]:
        return self._receivers

    @receivers.setter
    def receivers(self, new_set: List[str]):
        self._receivers = new_set

    @property
    def band(self) -> str:
        return self._band

    @classmethod
    def from_hdf5(cls: Type[_P], hdf5: h5py.File) -> _P:
        """"""
        attrs = hdf5.attrs
        frequency = models.get_hdf5_dataset(hdf5, 'frequency')
        frequency = models.require_columns('frequency', frequency, np.float32, 1)
        # u.Quantity has issues with h5py numpy-like's, so force loading
        frequency = frequency[:]
        frequency <<= u.Hz
        coefs = models.get_hdf5_dataset(hdf5, 'coefs')
        coefs = models.require_columns('coefs', coefs, np.float64, 2)
        band = models.get_hdf5_attr(attrs, 'band', str, required=True)
        correlator_efficiency = models.get_hdf5_attr(attrs, 'correlator_efficiency', float,
                                                     required=True)
        try:
            antennas = models.get_hdf5_dataset(hdf5, 'antennas')
            antennas = models.require_columns('antennas', antennas, str, 1)
        except DataError:
            antennas = None
        try:
            receivers = models.get_hdf5_dataset(hdf5, 'receivers')
            receivers = models.require_columns('receivers', receivers, str, 1)
        except DataError:
            receivers = None
        return cls(frequency, coefs, correlator_efficiency,
                   band=band, antennas=antennas, receivers=receivers)

    def to_hdf5(self, hdf5: h5py.File) -> None:
        """"""
        hdf5.create_dataset('frequency', data=self.frequency, track_times=False)
        hdf5.create_dataset('coefs', data=self.coefs, track_times=False)
        hdf5.attrs['correlator_efficiency'] = self.correlator_efficiency
        hdf5.attrs['band'] = self.band
        if self.antennas is not None:
            hdf5.create_dataset('antennas', data=self._antennas, track_times=False)
        if self.receivers is not None:
            hdf5.create_dataset('receivers', data=self._receivers, track_times=False)
