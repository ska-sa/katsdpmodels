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

from typing import ClassVar, Any, Union, TypeVar, Type, Optional
from typing_extensions import Literal

from katsdpmodels.primary_beam import PrimaryBeam

try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any  # type: ignore

# import astropy.units as u
import h5py
import logging
import katdal
import katpoint
import katsdptelstate

from . import models

logger = logging.getLogger(__name__)

_C = TypeVar('_C', bound='ComponentSkyModel')
_K = TypeVar('_K', bound='KatpointSkyModel')


class NoSkyModelError(Exception):
    """Attempted to load a sky model for continuum subtraction but there isn't one"""
    pass


class LocalSkyModel(models.SimpleHDF5Model):
    """ Base class for sky models """
    model_type: ClassVar[Literal['lsm']] = 'lsm'

    # Methods are not marked @abstractmethod as it causes issues with mypy:
    # https://github.com/python/mypy/issues/4717

    @classmethod
    def from_hdf5(cls, hdf5: h5py.File) -> 'LocalSkyModel':
        raise NotImplementedError()  # pragma: nocover

    def to_hdf5(self, hdf5: h5py.File) -> None:
        raise NotImplementedError()  # pragma: nocover


def catalogue_from_telstate(telstate: Union[katsdptelstate.TelescopeState,
                                            katdal.sensordata.TelstateToStr],
                            capture_block_id: str,
                            continuum: Union[str, None],
                            target: katpoint.Target) -> katpoint.Catalogue:
    """Extract a katpoint catalogue written by katsdpcontim.
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


class KatpointSkyModel(LocalSkyModel):
    model_format: ClassVar[Literal['katpoint_catalogue']] = 'katpoint_catalogue'

    def __init__(self, cat: katpoint.Catalogue, pb: Optional[PrimaryBeam]):
        self
        self._cat = cat
        if pb:
            self._PBModel = pb
        super().__init__()

    @property
    def primaryBeamModel(self, pb: PrimaryBeam) -> PrimaryBeam:
        # check pb exists and is accessible
        self._PBModel = pb
        return self._PBModel

    @classmethod
    def from_hdf5(cls: Type[_C], hdf5: h5py.File) -> _C:
        cat = models.get_hdf5_dataset(hdf5, 'catalogue')
        return cls(cat)

    def to_hdf5(self, hdf5: h5py.File) -> None:
        hdf5.attrs['cat'] = self._cat
        hdf5.create_dataset('cat', data=self._cat, track_times=False)


class ComponentSkyModel(LocalSkyModel):
    model_format: ClassVar[Literal['skymodel']] = 'skymodel'

    def __init__(self, cat):
        self.cat = cat
        super().__init__()

    @classmethod
    def from_hdf5(cls: Type[_C], hdf5: h5py.File) -> _C:
        cat = models.get_hdf5_dataset(hdf5, 'catalogue')
        return cls(cat)

    def to_hdf5(self, hdf5: h5py.File) -> None:
        hdf5.attrs['cat'] = self.cat
        hdf5.create_dataset('cat', data=self.cat, track_times=False)
