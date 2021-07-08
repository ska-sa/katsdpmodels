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

"""Local Sky Models"""
import enum
from pathlib import Path
from typing import Any, ClassVar, Optional, Type, TypeVar, Union, BinaryIO

import numpy as np
from typing_extensions import Literal

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any  # type: ignore

import logging
import urllib.parse
import io
import katdal.sensordata
import katpoint
import katsdptelstate
from katsdpmodels.primary_beam import PrimaryBeam

from . import models

import astropy.units as units

logger = logging.getLogger(__name__)

_K = TypeVar('_K', bound='KatpointSkyModel')
_L = TypeVar('_L', bound='LocalSkyModel')

# use a type alias for file_like objects
_FileLike = Union[io.IOBase, io.BytesIO, BinaryIO]


class NoSkyModelError(Exception):
    """Attempted to load a sky model but it does not exist"""
    pass


class NoPrimaryBeamError(Exception):
    """Attempted to get the associated primary beam model but it hasn't been set"""
    pass


class NoPhaseCentreError(Exception):
    """Attempted to get the phase centre target, but it hasn't been set"""
    pass


class FluxDensity(enum.Enum):
    PERCEIVED = '1'
    TRUE = '2'


class LocalSkyModel(models.Model):
    """ Base class for sky models
    Local sky
    model_type: local_sky
    model_format: katpoint_catalogue
    target: flux_density/name, where flux_density is either perceived or true, and name is the
    J-name for the calibrator e.g. J1441+3030

    Local Sky has the following attributes:
    flux_density: Union[Literal[perceived], Literal[ true]] -  to indicate whether the flux
                    densities in the model have been modulated by a primary beam or not (and which
                    matches the target). In the MeerKAT+ era, perceived will not be used as there
                    will no longer be a common perceived sky.

    Primary_beam: Optional[models.Primary_Beam] - this exists here so that calibration can use the
                    same beam used to derive this sky model to predict visibilities.

    Individual components are stored in a katpoint.Catalogue called components.
                    TODO katpoint.Catalogue has been extended to support wsclean sources.
    """
    model_type: ClassVar[Literal['lsm']] = 'lsm'

    # Methods are not marked @abstractmethod as it causes issues with mypy:
    # https://github.com/python/mypy/issues/4717

    @property
    def pb_model(self) -> Optional[PrimaryBeam]:
        """Minimum and maximum frequency covered by the model."""
        raise NotImplementedError()  # pragma: nocover

    @property
    def flux_density(self) -> Union[Literal[FluxDensity.TRUE], Literal[FluxDensity.PERCEIVED]]:
        """ enum to indicates whether the flux densities in the model have been modulated by a
        primary beam or not """
        raise NotImplementedError()  # pragma: nocover

    @classmethod
    def from_file(cls, file: Union[str, Path, _FileLike], url: str, *,
                  content_type: Optional[str] = None) -> 'LocalSkyModel':
        raise NotImplementedError()  # pragma: nocover

    def to_file(self, file: Union[str, Path, _FileLike], *,
                content_type: Optional[str] = None) -> None:
        raise NotImplementedError()  # pragma: nocover



class KatpointSkyModel(LocalSkyModel):
    model_format: ClassVar[Literal['katpoint_catalogue']] = 'katpoint_catalogue'

    def __init__(self, cat: Optional[katpoint.Catalogue] = None,
                 pc: Optional[katpoint.Target] = None,
                 pb: Optional[PrimaryBeam] = None):
        if cat:
            self._components = cat  # TODO:
        self._PhaseCentre = pc
        self._PBModel = pb
        super().__init__()

    @property
    def pb_model(self) -> PrimaryBeam:
        if self._PBModel is None:
            raise NoPrimaryBeamError
        return self._PBModel

    @pb_model.setter
    def pb_model(self, pb: PrimaryBeam) -> None:
        # TODO check pb exists and is accessible
        self._PBModel = pb

    @property
    def phase_centre(self) -> katpoint.Target:
        if self._PhaseCentre is None:
            raise NoPhaseCentreError
        return self._PhaseCentre

    @phase_centre.setter
    def phase_centre(self, pc: katpoint.Target) -> None:
        self._PhaseCentre = pc

    @units.quantity_input(wavelength=units.m, equivalencies=units.spectral())
    def flux_density(self, wavelength):
        freq_MHz = wavelength.to(units.MHz, equivalencies=units.spectral()).value
        out = np.stack([source.flux_density_stokes(freq_MHz) for source in self._components])
        return np.nan_to_num(out, copy=False)

    @classmethod
    def from_katpoint_catalogue(cls: Type[_K], cat: katpoint.Catalogue) -> _K:
        return cls(cat)

    @classmethod
    def from_file(cls: Type[_K], file: Union[str, Path, _FileLike], url: str, *,
                  content_type: Optional[str] = None) -> 'KatpointSkyModel':
        if url:
            cat = catalogue_from_katpoint(url)
            return KatpointSkyModel(cat)
        else:
            return KatpointSkyModel()

    def to_file(self, file: Union[str, Path, _FileLike], *,
                content_type: Optional[str] = None) -> None:
        pass


def catalogue_from_katpoint(url: str) -> katpoint.Catalogue:
    """Load a katpoint sky model from file. Katpoint stores catalogues as `.csv' files.
    Parameters
    ----------
    url : str
         A ``file://`` URL for a katpoint catalogue file (in CSV format -- although this need
         not be the extension of the url supplied)

    Raises
    ------
        ValueError
            if `format` was not recognised, the URL doesn't contain the
            expected query parameters, or the URL scheme is not supported
        IOError, OSError
            if there was a low-level error reading a file
        Exception
            any exception raised by katdal in opening the file

    Returns
    -------
        KatpointSkyModel
    """
    if urllib.parse.urlparse(url, scheme='file').scheme != 'file':
        raise ValueError('Only file:// URLs are supported for katpoint sky model format')
    with open(urllib.parse.urlparse(url, scheme='file').path) as f:
        return katpoint.Catalogue(f)


def catalogue_from_telstate(telstate: Union[katsdptelstate.TelescopeState,
                                            katdal.sensordata.TelstateToStr],
                            capture_block_id: str,
                            continuum: Union[str, None],
                            target: katpoint.Target) -> katpoint.Catalogue:
    """Extract a katpoint catalogue written to katsdptelstate.
    Parameters
    ----------
    telstate : :class:`katsdptelstate.TelescopeState` or :class:`katdal.sensordata.TelstateToStr`
        Telescope state
    capture_block_id : str
        Capture block ID
    continuum : str or ``None``
        Name of the continuum imaging stream (used to form the telstate view).
        If ``None``, there must be exactly one continuum imaging stream in the
        data set, which is used.
    target : :class:`katpoint.Target`
        Target field
    Raises
    ------
    NoSkyModelError
        If no sky model could be found for the given parameters
    Returns
    -------
    katpoint.Catalogue
    """
    from katdal.sensordata import TelstateToStr

    telstate = TelstateToStr(telstate)
    try:
        # Find the continuum image stream
        if continuum is None:
            archived_streams = telstate['sdp_archived_streams']
            for stream_name in archived_streams:
                view = telstate.view(stream_name, exclusive=True)
                view = view.view(telstate.join(capture_block_id, stream_name))
                stream_type = view.get('stream_type', 'unknown')
                # The correct value is 'sdp.continuum_image', but due to a bug
                # there are observations in the wild with just 'continuum_image'.
                if stream_type not in {'sdp.continuum_image', 'continuum_image'}:
                    continue
                if continuum is not None:
                    raise NoSkyModelError(
                        'Multiple continuum image streams found - need to select one')
                continuum = stream_name
            if continuum is None:
                raise NoSkyModelError('No continuum image streams found')

        view = telstate.view(continuum, exclusive=True)
        view = view.view(telstate.join(capture_block_id, continuum))
        target_namespace = view['targets'][target.description]
        prefix = telstate.join(capture_block_id, continuum, target_namespace, 'target0')
        data = view.view(prefix)['clean_components']
        # Should always match, but for safety
        if katpoint.Target(data['description']) == target:
            return katpoint.Catalogue(data['components'])
    except KeyError:
        logger.debug('KeyError', exc_info=True)
    raise NoSkyModelError('Sky model for target {} not found'.format(target.name))
