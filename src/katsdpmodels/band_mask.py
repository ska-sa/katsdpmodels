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

"""Masks for roll-off at the edges of a band."""

from typing import Type, TypeVar, ClassVar, Any
from typing_extensions import Literal

import numpy as np
import astropy.units as u
import astropy.table
import h5py

from . import models

_B = TypeVar('_B', bound='BandMaskRanges')


class Band:
    """Defines a band for use with :class:`BandMask`."""

    def __init__(self, bandwidth: u.Quantity, centre_frequency: u.Quantity):
        self.bandwidth = bandwidth.to(u.Hz)
        self.centre_frequency = centre_frequency.to(u.Hz)

    @property
    def min_frequency(self) -> u.Quantity:
        return self.centre_frequency - self.bandwidth / 2

    @property
    def max_frequency(self) -> u.Quantity:
        return self.centre_frequency + self.bandwidth / 2


class BandMask(models.SimpleHDF5Model):
    model_type: ClassVar[Literal['band_mask']] = 'band_mask'

    def is_masked(self, band: Band, frequency: u.Quantity) -> Any:
        raise NotImplementedError()      # pragma: nocover

    @classmethod
    def from_hdf5(cls, hdf5: h5py.File) -> 'BandMask':
        model_format = models.get_hdf5_attr(hdf5.attrs, 'model_format', str) or ''
        if model_format == 'ranges':
            return BandMaskRanges.from_hdf5(hdf5)
        else:
            raise models.ModelFormatError(
                f'Unknown model_format {model_format!r} for {cls.model_type}')


class BandMaskRanges(BandMask):
    model_format: ClassVar[Literal['ranges']] = 'ranges'

    def __init__(self, ranges: astropy.table.Table) -> None:
        super().__init__()
        cols = ('min_fraction', 'max_fraction')
        self.ranges = astropy.table.Table(
            [ranges[col] for col in cols],
            names=cols,
            dtype=(np.float64, np.float64)
        )

    def is_masked(self, band: Band, frequency: u.Quantity) -> Any:
        fraction = (frequency - band.min_frequency) / band.bandwidth
        # Add an axis that will broadcast with the ranges
        fraction = fraction[..., np.newaxis]
        mask = (fraction >= self.ranges['min_fraction']) & (fraction <= self.ranges['max_fraction'])
        return np.any(mask, axis=-1)

    @classmethod
    def from_hdf5(cls: Type[_B], hdf5: h5py.File) -> _B:
        data = models.get_hdf5_dataset(hdf5, 'ranges')
        if data.ndim != 1:
            raise models.DataError(f'ranges dataset should have 1 dimension, found {data.ndim}')
        expected_dtype = np.dtype([
            ('min_fraction', 'f8'),
            ('max_fraction', 'f8'),
        ])
        data = models.require_columns(data, expected_dtype)
        ranges = astropy.table.Table(data[...], copy=False)
        ranges['min_fraction'].unit = u.dimensionless_unscaled
        ranges['max_fraction'].unit = u.dimensionless_unscaled
        return cls(ranges)

    def to_hdf5(self, hdf5: h5py.File) -> None:
        hdf5.create_dataset('ranges', data=self.ranges.as_array(), track_times=False)
