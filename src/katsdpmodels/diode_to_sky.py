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

import astropy.units as units
import numpy as np
import logging


from pathlib import Path
from typing import Any, ClassVar, Optional, Type, TypeVar, Union
from typing_extensions import Literal

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any  # type: ignore

import katdal.sensordata
import katpoint
import katsdptelstate

from . import models


logger = logging.getLogger(__name__)

_D = TypeVar('_K', bound='DiodeToSkyModel')

# use a type alias for file_like objects
_FileLike = Union[io.IOBase, io.BytesIO, BinaryIO]


class NoDiodeToSkyModelError(Exception):
    """Attempted to load a bandpass phase model but it does not exist"""
    pass


class DiodeToSkyModel(models.Model):
    """ Base class for bandpass phase models
    Diode-to-sky model (bandpass phase model for calibration)
    model_type: diode_to_sky
    model_format: ?? --
    target: ??

    Diode-to-Sky has the following attributes:
    TODO
    """
    model_type: ClassVar[Literal['dts']] = 'dts'

    @classmethod
    def from_file(cls, file: Union[str, Path, _FileLike], url: str, *,
                  content_type: Optional[str] = None) -> 'LocalSkyModel':
        raise NotImplementedError()  # pragma: nocover

    def to_file(self, file: Union[str, Path, _FileLike], *,
                content_type: Optional[str] = None) -> None:
        raise NotImplementedError()  # pragma: nocover


class SplineDiodeModel(DiodeToSkyModel):
    """ captures set of 'knot' locations and spline paramters """
    pass