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

"""Masks for radio-frequency interference."""

from typing import Type, TypeVar, ClassVar, Any
from typing_extensions import Literal

import numpy as np
import astropy.units as u
import astropy.table
import h5py

from . import models


_R = TypeVar('_R', bound='RFIMaskRanges')


class RFIMask(models.SimpleHDF5Model):
    model_type: ClassVar[Literal['rfi_mask']] = 'rfi_mask'

    # Methods are not marked @abstractmethod as it causes issues with mypy:
    # https://github.com/python/mypy/issues/4717

    def is_masked(self, frequency: u.Quantity, baseline_length: u.Quantity,
                  channel_width: u.Quantity = 0 * u.Hz) -> Any:
        """Determine whether given frequency is masked for given baseline length.

        The return value is either a boolean (if `frequency`, `baseline_length`
        and `channel_width` are scalar) or an array of boolean if they're
        arrays, with the usual broadcasting rules applying.

        A channel is masked if any part of the channel overlaps with RFI. The
        channel has width `channel_width` and is centred on `frequency`. This
        also honours the :attr:`mask_auto_correlations` property for baseline
        lengths of zero.
        """
        raise NotImplementedError()      # pragma: nocover

    def max_baseline_length(self, frequency: u.Quantity,
                            channel_width: u.Quantity = 0 * u.Hz) -> Any:
        """Maximum baseline length for which data at `frequency` should be masked.

        If the frequency is not masked at all, returns a negative length, and
        if it is masked at all baseline lengths, returns +inf. One may also
        supply an array of frequencies and receive an array of responses. This
        does *not* include the effect of the :attr:`mask_auto_correlations`
        property when returning a value of zero.
        """
        raise NotImplementedError()      # pragma: nocover

    @property
    def mask_auto_correlations(self) -> bool:
        """Return whether auto-correlations should be masked too.

        Auto-correlations are defined as baselines with zero length, which
        includes cross-hand polarisation products. This property indicates
        whether auto-correlations should be masked, in which case it is done
        for frequencies covered by any of the ranges. That is, if this is
        `False`, no auto-correlations will be masked for RFI. If this is
        `True`, auto-correlations are treated like any other baselines.
        """
        raise NotImplementedError()      # pragma: nocover

    @classmethod
    def from_hdf5(cls, hdf5: h5py.File) -> 'RFIMask':
        model_format = models.get_hdf5_attr(hdf5.attrs, 'model_format', str) or ''
        if model_format == 'ranges':
            return RFIMaskRanges.from_hdf5(hdf5)
        else:
            raise models.ModelFormatError(
                f'Unknown model_format {model_format!r} for {cls.model_type}')

    def to_hdf5(self, hdf5: h5py.File) -> None:
        raise NotImplementedError()      # pragma: nocover


class RFIMaskRanges(RFIMask):
    model_format: ClassVar[Literal['ranges']] = 'ranges'

    def __init__(self, ranges: astropy.table.Table, mask_auto_correlations: bool) -> None:
        super().__init__()
        cols = ('min_frequency', 'max_frequency', 'max_baseline')
        units = (u.Hz, u.Hz, u.m)
        self.ranges = astropy.table.QTable(
            [ranges[col] for col in cols],
            names=cols,
            dtype=(np.float64, np.float64, np.float64)
        )
        # Canonicalise the units to simplify to_hdf5 (and also remove the
        # cost of conversions when methods are called with canonical units).
        for col, unit in zip(cols, units):
            # Ensure we haven't been given unit-less data, as <<= will inject
            # the unit (see https://github.com/astropy/astropy/issues/10514).
            if self.ranges[col].unit is None:
                raise u.UnitConversionError(f'Column {col} has no units')
            self.ranges[col] <<= unit
        self._mask_auto_correlations = mask_auto_correlations

    @property
    def mask_auto_correlations(self) -> bool:
        return self._mask_auto_correlations

    def is_masked(self, frequency: u.Quantity, baseline_length: u.Quantity,
                  channel_width: u.Quantity = 0 * u.Hz) -> Any:
        # Add extra axis which will broadcast with the masks
        low = (frequency - 0.5 * channel_width)[..., np.newaxis]
        high = (frequency + 0.5 * channel_width)[..., np.newaxis]
        b = baseline_length[..., np.newaxis]
        in_range = (
            (self.ranges['min_frequency'] <= high)
            & (low <= self.ranges['max_frequency'])
            & (b <= self.ranges['max_baseline'])
        )
        if not self.mask_auto_correlations:
            in_range &= b > 0
        return np.any(in_range, axis=-1)

    def max_baseline_length(self, frequency: u.Quantity,
                            channel_width: u.Quantity = 0 * u.Hz) -> Any:
        # Add extra axis which will broadcast with the masks, and compute channel edges
        low = (frequency - 0.5 * channel_width)[..., np.newaxis]
        high = (frequency + 0.5 * channel_width)[..., np.newaxis]
        in_range = (self.ranges['min_frequency'] <= high) & (low <= self.ranges['max_frequency'])
        b = np.broadcast_to(self.ranges['max_baseline'], in_range.shape, subok=True)
        return np.max(b,
                      axis=-1, where=in_range,
                      initial=-1.0 * self.ranges['max_baseline'].unit)

    @classmethod
    def from_hdf5(cls: Type[_R], hdf5: h5py.File) -> _R:
        dataset_name = 'ranges'
        data = models.get_hdf5_dataset(hdf5, dataset_name)
        expected_dtype = np.dtype([
            ('min_frequency', 'f8'),
            ('max_frequency', 'f8'),
            ('max_baseline', 'f8')
        ])
        data = models.require_columns(f'{dataset_name} dataset', data, expected_dtype, 1)
        ranges = astropy.table.Table(data[...], copy=False)
        ranges['min_frequency'].unit = u.Hz
        ranges['max_frequency'].unit = u.Hz
        ranges['max_baseline'].unit = u.m
        mask_auto_correlations = models.get_hdf5_attr(
            hdf5.attrs, 'mask_auto_correlations', bool, required=True)
        return cls(ranges, mask_auto_correlations)

    def to_hdf5(self, hdf5: h5py.File) -> None:
        hdf5.attrs['mask_auto_correlations'] = self.mask_auto_correlations
        # The constructor ensures we're using unscaled units
        hdf5.create_dataset('ranges', data=self.ranges.as_array(), track_times=False)
