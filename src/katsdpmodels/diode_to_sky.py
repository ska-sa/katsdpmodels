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

import logging
import io
import scipy.interpolate

from pathlib import Path 
from typing import Any, BinaryIO, ClassVar, Optional, TypeVar, Union
from typing_extensions import Literal

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any  # type: ignore

from . import models


logger = logging.getLogger(__name__)

_D = TypeVar('_D', bound='DiodeToSkyModel')

# use a type alias for file_like objects
_FileLike = Union[io.IOBase, io.BytesIO, BinaryIO]


class NoDiodeToSkyModelError(Exception):
    """Attempted to load a bandpass phase model but it does not exist"""
    pass


class DiodeToSkyModel(models.Model):
    """ Base class for bandpass phase models
    Diode-to-sky model (bandpass phase model for calibration)
    model_type: diode_to_sky
    model_format: ??
    target: ??
    
    Diode-to-Sky has the following attributes:
    TODO
    """
    model_type: ClassVar[Literal['dsm']] = 'dsm'
    

    @classmethod
    def from_file(cls, file: Union[str, Path, _FileLike], url: str, *,
                  content_type: Optional[str] = None) -> 'DiodeToSkyModel':
        raise NotImplementedError()  # pragma: nocover

    def to_file(self, file: Union[str, Path, _FileLike], *,
                content_type: Optional[str] = None) -> None:
        raise NotImplementedError()  # pragma: nocover


class BSplineModel(DiodeToSkyModel):
    """ captures set of knot locations and spline parameters as a scipy bspline object"""
    model_format: ClassVar[Literal['ScipyBSpline']] = 'ScipyBSpline'

    def __init__(self, *, 
                 knots: ArrayLike, 
                 coefficients: ArrayLike, 
                 degree: int, 
                 target: str) -> None:
        self.knots = knots
        self.coefficients = coefficients
        self.degree = degree
        self.target = target
        self.bspline = scipy.interpolate.BSpline(knots, coefficients, degree)

    
class CSplineModel(DiodeToSkyModel):
    pass
