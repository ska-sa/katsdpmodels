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
Models MAY provide combined (Stokes-I) SEFD.
    It’s a handy convenience for estimating image noise, but should probably be computed on the
    fly rather than stored.
Models SHOULD provide sensible values even at RFI-affected frequencies.
    Needed for simulation, and looks better in an imaging report.
Models MUST allow dish dependence.
    Required for heterogeneous MeerKAT+ array.
"""
import numpy as np

import astropy.units as u
import h5py
import logging
import io

# from pathlib import Path
from typing import Any, BinaryIO, ClassVar, Optional, Tuple, Type, TypeVar, Union
from typing_extensions import Literal

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any  # type: ignore

from . import models

# use a type alias for file_like objects
_FileLike = Union[io.IOBase, io.BytesIO, BinaryIO]

_B = TypeVar('_B', bound='BSplineSEFDModel')
_P = TypeVar('_P', bound='PolySEFDModel')

logger = logging.getLogger(__name__)


class NoSEFDModelError(Exception):
    """Attempted to load a SEFD model but it does not exist"""
    pass


class SEFDModel(models.SimpleHDF5Model):
    """
    Base class for SEFD models.

    A System-Equivalent Flux Density (SEFD) model is a ...

    model_type: sefd

    SEFD has the following attributes:
    """
    model_type: ClassVar[Literal['sefd']] = 'sefd'

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
    def from_hdf5(cls, hdf5: h5py.File) -> 'SEFDModel':
        model_format = models.get_hdf5_attr(hdf5.attrs, 'model_format', str)
        logger.error(model_format)
        if model_format == 'poly':
            return PolySEFDModel.from_hdf5(hdf5)
        elif model_format == 'bspline':
            return BSplineSEFDModel.from_hdf5(hdf5)
        else:
            raise models.ModelFormatError(
                f'Unknown model_format {model_format!r} for {cls.model_type}')

    def to_hdf5(self, hdf5: h5py.File) -> None:
        raise NotImplementedError()  # pragma: nocover


class PolySEFDModel(SEFDModel):
    """
    captures a polynomial SEFD model
    model_format: 'poly'
    """

    model_format: ClassVar[Literal['poly']] = 'poly'

    def __init__(self,
                 min_frequency: u.Quantity,
                 max_frequency: u.Quantity,
                 frequency_unit: u.Unit,
                 coefs: u.Quantity,
                 correlator_efficiency: Optional[float],
                 *,
                 band: str,
                 antenna: Optional[str] = None,
                 receiver: Optional[str] = None) -> None:
        super().__init__()
        self._min_frequency = min_frequency
        self._max_frequency = max_frequency
        self._frequency_unit = frequency_unit
        self._coefs = coefs
        self._correlator_efficiency = correlator_efficiency
        self._band = band
        self._antenna = antenna
        self._receiver = receiver

    @property
    def coefs(self) -> ArrayLike:
        """ TODO: check that u.Quantity is ArrayLike """
        return self._coefs

    @property
    def min_frequency(self) -> Optional[u.Quantity]:
        return self._min_frequency

    @property
    def max_frequency(self) -> Optional[u.Quantity]:
        return self._max_frequency

    @property
    def frequency_unit(self) -> Optional[u.Unit]:
        return self._frequency_unit

    @property
    def correlator_efficiency(self) -> Optional[float]:
        return self._correlator_efficiency

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
    def from_hdf5(cls: Type[_P], hdf5: h5py.File) -> _P:
        """"""
        attrs = hdf5.attrs
        min_frequency = models.get_hdf5_attr(attrs, 'min_frequency', float, required=False)
        max_frequency = models.get_hdf5_attr(attrs, 'max_frequency', float, required=False)
        frequency_unit = models.get_hdf5_attr(attrs, 'frequency_unit', int, required=False)
        correlator_efficiency = models.get_hdf5_attr(attrs, 'correlator_efficiency', float,
                                                     required=False)
        band = models.get_hdf5_attr(attrs, 'band', str, required=True)
        antenna = models.get_hdf5_attr(attrs, 'antenna', str, required=False)
        receiver = models.get_hdf5_attr(attrs, 'receiver', str, required=False)

        coefs = models.get_hdf5_dataset(hdf5, 'coefs')
        coefs = models.require_columns('coefs', coefs, np.float64, 1)

        return cls(min_frequency, max_frequency, frequency_unit,
                   coefs, correlator_efficiency,
                   band=band, antenna=antenna, receiver=receiver)

    def to_hdf5(self, hdf5: h5py.File) -> None:
        """"""
        hdf5.attrs['band'] = self._band
        if self.antenna is not None:
            hdf5.attrs['antenna'] = self._antenna
        if self.receiver is not None:
            hdf5.attrs['receiver'] = self._receiver
        if self.min_frequency is not None:
            hdf5.attrs['min_frequency'] = self._min_frequency
        if self.max_frequency is not None:
            hdf5.attrs['max_frequency'] = self._max_frequency
        if self.frequency_unit is not None:
            hdf5.attrs['frequency_unit'] = self._frequency_unit
        if self.correlator_efficiency is not None:
            hdf5.attrs['correlator_efficiency'] = self._correlator_efficiency
        hdf5.create_dataset('coefs', data=self._coefs, track_times=False)


class BSplineSEFDModel(SEFDModel):
    """
    captures a BSpline SEFD model
    model_format: 'bspline'
    """

    model_format: ClassVar[Literal['bspline']] = 'bspline'

    def __init__(self,
                 params,  # TODO
                 *,
                 band: str,
                 antenna: Optional[str] = None,
                 receiver: Optional[str] = None) -> None:
        super().__init__()
        self._params = params
        self._band = band
        self._antenna = antenna
        self._receiver = receiver

    @property
    def params(self) -> ArrayLike:
        """TODO"""
        return self._params

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
        antenna = models.get_hdf5_attr(attrs, 'antenna', str, required=False)
        receiver = models.get_hdf5_attr(attrs, 'receiver', str, required=False)

        # TODO
        params = models.get_hdf5_dataset(hdf5, 'params')
        params = models.require_columns('params', params, np.float64, 1)

        return cls(params, band=band, antenna=antenna, receiver=receiver)

    def to_hdf5(self, hdf5: h5py.File) -> None:
        """"""
        hdf5.attrs['band'] = self._band
        if self.antenna is not None:
            hdf5.attrs['antenna'] = self._antenna
        if self.receiver is not None:
            hdf5.attrs['receiver'] = self._receiver
        hdf5.create_dataset('params', data=self._params, track_times=False)
