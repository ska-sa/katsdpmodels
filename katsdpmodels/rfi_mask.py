"""Masks for radio-frequency interference."""

from abc import abstractmethod
from typing import Optional, Type, TypeVar, ClassVar
from typing_extensions import Literal

import numpy as np
import astropy.units as u
import astropy.table
import h5py

from . import models


_R = TypeVar('_R', bound='RFIMaskRanges')


class RFIMask(models.Model):
    model_type: ClassVar[Literal['rfi_mask']] = 'rfi_mask'

    @abstractmethod
    def is_masked(self, frequency: u.Quantity, baseline_length: u.Quantity) -> bool:
        """Determine whether given frequency is masked for the given baseline length."""

    @classmethod
    def from_raw(cls, raw: models.RawModel) -> 'RFIMask':
        model_format = raw.hdf5.attrs.get('model_format')
        if model_format == 'ranges':
            return RFIMaskRanges.from_raw(raw)
        else:
            raise models.ModelFormatError(
                f'Unknown model_format {model_format!r} for {cls.model_type}')


class RFIMaskRanges(RFIMask):
    model_format: ClassVar[Literal['rfi_format']] = 'rfi_format'

    def __init__(self, ranges: astropy.table.QTable, *,
                 raw: Optional[models.RawModel] = None) -> None:
        super().__init__(raw=raw)
        # TODO: validate the shape and units
        # TODO: document what the requirements are
        self.ranges = ranges

    def is_masked(self, frequency: u.Quantity, baseline_length: u.Quantity) -> bool:
        # TODO: should this be vectorised so that one can get a mask across
        # multiple frequencies or baselines (or both)?
        in_range = (
            (self.ranges['min_frequency'] <= frequency)
            & (frequency <= self.ranges['max_frequency'])
            & (baseline_length <= self.ranges['max_baseline'])
        )
        return np.any(in_range)

    @classmethod
    def from_raw(cls: Type[_R], raw: models.RawModel) -> _R:
        try:
            data = raw.hdf5['/ranges']
            if isinstance(data, h5py.Group):
                raise KeyError        # It should be a dataset, not a group
        except KeyError:
            raise models.DataError('Model is missing ranges dataset') from None
        ranges = astropy.table.QTable(data[:])
        ranges['min_frequency'] <<= u.Hz
        ranges['max_frequency'] <<= u.Hz
        ranges['max_baseline'] <<= u.m
        return cls(ranges, raw=raw)
