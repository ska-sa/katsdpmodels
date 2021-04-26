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

"""Tests for :mod:`katsdpmodels.primary_beam`"""

import contextlib
import pathlib
from typing import Generator, Any, Union, cast

import astropy.units as u
from astropy import constants
from astropy.coordinates import (
    EarthLocation, AltAz, SkyCoord,
    CartesianRepresentation, UnitSphericalRepresentation)
from astropy.time import Time
from astropy.coordinates.matrix_utilities import rotation_matrix
import numpy as np
try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any  # type: ignore
import h5py
import pytest

from katsdpmodels import models, primary_beam
import katsdpmodels.fetch.requests as fetch_requests


FRAME_OUTPUT_TYPE_COMBOS = [
    (frame, output_type)
    for frame in [primary_beam.AltAzFrame(),
                  primary_beam.RADecFrame(parallactic_angle=40 * u.deg)]
    for output_type in primary_beam.OutputType
    if (isinstance(frame, primary_beam.RADecFrame)
        or output_type in {primary_beam.OutputType.JONES_HV,
                           primary_beam.OutputType.UNPOLARIZED_POWER})
]


@pytest.fixture
def aperture_plane_model_file(tmp_path) -> h5py.File:
    """Create an aperture-plane model for testing.

    It writes the model directly rather than using the PrimaryBeam API
    to ensure the model adheres to the file format even if bugs are
    introduced into the API.
    """
    path = tmp_path / 'aperture_plane_test.h5'
    h5file = h5py.File(path, 'w')
    h5file.attrs['model_type'] = 'primary_beam'
    h5file.attrs['model_format'] = 'aperture_plane'
    h5file.attrs['model_comment'] = 'Dummy model for testing. Do not use in production.'
    h5file.attrs['model_created'] = '2021-04-20T12:11:00Z'
    h5file.attrs['model_version'] = 1
    h5file.attrs['model_author'] = 'MeerKAT SDP developers <sdpdev+katsdpmodels@ska.ac.za>'
    h5file.attrs['antenna'] = 'm999'
    h5file.attrs['receiver'] = 'r123'
    h5file.attrs['band'] = 'z'
    # x and y are deliberately set up differently to catch bugs that mix them up.
    h5file.attrs['x_start'] = -10.0
    h5file.attrs['x_step'] = 0.5
    h5file.attrs['y_start'] = 8.0
    h5file.attrs['y_step'] = -0.25
    frequency = np.arange(64) * 1e7 + 1e9
    h5file.create_dataset('frequency', data=frequency)

    rs = np.random.RandomState(1)
    shape = (len(frequency), 2, 2, 80, 40)
    data = rs.random_sample(shape) + 1j * rs.random_sample(shape)
    # Adjust data so that beam is the identity at the centre.
    data -= np.mean(data, axis=(3, 4), keepdims=True)
    data[:, 0, 0] += 1
    data[:, 1, 1] += 1
    h5file.create_dataset('aperture_plane', data=data.astype(np.complex64))
    return h5file


@contextlib.contextmanager
def serve_model(model_file: h5py.File) \
        -> Generator[primary_beam.PrimaryBeamAperturePlane, None, None]:
    path = pathlib.Path(model_file.filename)
    model_file.close()  # Ensures data is written to the file
    with fetch_requests.fetch_model(path.as_uri(), primary_beam.PrimaryBeam) as model:
        yield cast(primary_beam.PrimaryBeamAperturePlane, model)


@pytest.fixture
def aperture_plane_model(aperture_plane_model_file: h5py.File) \
        -> Generator[primary_beam.PrimaryBeamAperturePlane, None, None]:
    with serve_model(aperture_plane_model_file) as model:
        yield model


def test_properties(aperture_plane_model: primary_beam.PrimaryBeamAperturePlane) -> None:
    model = aperture_plane_model
    # approx because speed of light is not exactly 3e8 m/s
    assert model.spatial_resolution(1 * u.GHz) == pytest.approx(0.015, rel=1e-3)
    assert model.frequency_range() == (1000 * u.MHz, 1630 * u.MHz)
    assert model.frequency_resolution() == 10 * u.MHz
    # approx because speed of light is not exactly 3e8 m/s
    assert model.inradius(1.5 * u.GHz) == pytest.approx(0.2, rel=1e-3)
    # Radius in x is 0.3, in y is 0.6 because it has twice the aperture-plane
    # resolution.
    assert model.circumradius(1 * u.GHz) == pytest.approx(np.hypot(0.3, 0.6), rel=1e-3)
    assert not model.is_circular
    assert not model.is_unpolarized
    assert model.antenna == 'm999'
    assert model.receiver == 'r123'
    assert model.band == 'z'
    # These are not expected to be used by users, but sampling tests rely on
    # them so we need to be sure they're accurate.
    np.testing.assert_equal(model.x, (np.arange(40) * 0.5 - 10) * u.m)
    np.testing.assert_equal(model.y, (np.arange(80) * -0.25 + 8) * u.m)


def test_no_optional_properties(aperture_plane_model_file: h5py.File) -> None:
    h5file = aperture_plane_model_file
    del h5file.attrs['receiver']
    del h5file.attrs['antenna']
    del h5file.attrs['band']
    with serve_model(h5file) as model:
        assert model.antenna is None
        assert model.receiver is None
        assert model.band is None


def test_single_frequency(aperture_plane_model_file: h5py.File) -> None:
    h5file = aperture_plane_model_file
    old_frequency = h5file['frequency']
    old_samples = h5file['aperture_plane']
    del h5file['frequency']
    del h5file['aperture_plane']
    h5file.create_dataset('frequency', data=old_frequency[:1])
    h5file.create_dataset('aperture_plane', data=old_samples[:1])
    with pytest.raises(models.DataError, match='at least 2 frequencies'):
        with serve_model(h5file):
            pass


def test_wrong_leading_dimensions(aperture_plane_model_file: h5py.File) -> None:
    h5file = aperture_plane_model_file
    del h5file['aperture_plane']
    h5file.create_dataset('aperture_plane', shape=(64, 3, 3, 80, 40), dtype=np.complex64)
    with pytest.raises(models.DataError, match='aperture_plane must by 2x2 on Jones dimensions'):
        with serve_model(h5file):
            pass


def test_frequency_bad_size(aperture_plane_model_file: h5py.File) -> None:
    h5file = aperture_plane_model_file
    old = h5file['frequency']
    del h5file['frequency']
    h5file.create_dataset('frequency', data=old[:-1])
    with pytest.raises(models.DataError,
                       match='aperture_plane and frequency have inconsistent sizes'):
        with serve_model(h5file):
            pass


def test_frequency_bad_ordering(aperture_plane_model_file: h5py.File) -> None:
    h5file = aperture_plane_model_file
    h5file['frequency'][3] = 2e9
    with pytest.raises(models.DataError, match='frequencies must be strictly increasing'):
        with serve_model(h5file):
            pass


def test_missing_attr(aperture_plane_model_file: h5py.File) -> None:
    h5file = aperture_plane_model_file
    del h5file.attrs['x_start']
    with pytest.raises(models.DataError, match="attribute 'x_start' is missing"):
        with serve_model(h5file):
            pass


def test_bad_model_format(aperture_plane_model_file: h5py.File) -> None:
    h5file = aperture_plane_model_file
    h5file.attrs['model_format'] = 'not_this'
    with pytest.raises(models.ModelFormatError,
                       match="Unknown model_format 'not_this' for primary_beam"):
        with serve_model(h5file):
            pass


def _compute_expected(model: primary_beam.PrimaryBeamAperturePlane,
                      samples: ArrayLike,
                      l: ArrayLike, m: ArrayLike,
                      frequency: u.Quantity) -> np.ndarray:
    l = np.asarray(l)
    m = np.asarray(m)
    samples = np.asarray(samples)
    # Use complex128 to get better precision with the accumulations and make
    # sure that the real implementation's use of complex64 isn't an issue.
    expected = np.zeros((len(l), 2, 2), np.complex128)
    for i in range(len(l)):
        for j in range(len(model.x)):
            for k in range(len(model.y)):
                s = np.exp(-2j * np.pi * (model.x[j] * l[i] + model.y[k] * m[i])
                           * frequency / constants.c)
                for pol in np.ndindex((2, 2)):
                    expected[(i,) + pol] += s * samples[pol + (k, j)]
    expected /= len(model.x) * len(model.y)
    return expected.astype(np.complex64)


def test_sample_exact_scalar_freq(
        aperture_plane_model: primary_beam.PrimaryBeamAperturePlane) -> None:
    model = aperture_plane_model
    l = [0.0, 0.05, -0.002]
    m = [0.0, 0.03, 0.004]
    frequency_idx = 50
    frequency = 1500 * u.MHz
    assert model.frequency[frequency_idx] == frequency

    actual = aperture_plane_model.sample(
        l, m, frequency,
        primary_beam.AltAzFrame(),
        primary_beam.OutputType.JONES_HV)

    expected = _compute_expected(model, model.samples[frequency_idx], l, m, frequency)

    # atol is more appropriate than rtol since there is cancellation of small terms
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    # Check that we get identity matrix at the origin
    np.testing.assert_allclose(actual[0], np.eye(2), rtol=0, atol=1e-6)


def test_sample_interp_scalar_freq(
        aperture_plane_model: primary_beam.PrimaryBeamAperturePlane) -> None:
    model = aperture_plane_model
    l = [0.0, 0.05, -0.02]
    m = [0.0, 0.03, 0.04]
    frequency_idx = 50
    frequency = np.mean(model.frequency[frequency_idx : frequency_idx + 2])

    actual = aperture_plane_model.sample(
        l, m, frequency,
        primary_beam.AltAzFrame(),
        primary_beam.OutputType.JONES_HV)

    samples = np.mean(model.samples[frequency_idx : frequency_idx + 2], axis=0)
    expected = _compute_expected(model, samples, l, m, frequency)

    # atol is more appropriate than rtol since there is cancellation of small terms
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    # Check that we get identity matrix at the origin
    np.testing.assert_allclose(actual[0], np.eye(2), rtol=0, atol=1e-6)


def test_sample_multi_dim_lm(aperture_plane_model: primary_beam.PrimaryBeamAperturePlane) -> None:
    rs = np.random.RandomState(1)
    shape = (2, 3, 4)
    l = rs.random(shape) * 0.05
    m = rs.random(shape) * 0.05
    frequency = 1500 * u.MHz

    multi = aperture_plane_model.sample(
        l, m, frequency,
        primary_beam.AltAzFrame(),
        primary_beam.OutputType.JONES_HV)
    flat = aperture_plane_model.sample(
        l.ravel(), m.ravel(), frequency,
        primary_beam.AltAzFrame(),
        primary_beam.OutputType.JONES_HV)
    np.testing.assert_array_equal(multi, flat.reshape(multi.shape))
    # Make sure that we're comparing meaningful values, not just NaNs
    assert not np.any(np.isnan(multi))


def test_sample_scalar_lm(aperture_plane_model: primary_beam.PrimaryBeamAperturePlane) -> None:
    l = 0.01
    m = 0.02
    frequency = 1500 * u.MHz

    scalar = aperture_plane_model.sample(
        l, m, frequency,
        primary_beam.AltAzFrame(),
        primary_beam.OutputType.JONES_HV)
    vector = aperture_plane_model.sample(
        np.array([l]), np.array([m]), frequency,
        primary_beam.AltAzFrame(),
        primary_beam.OutputType.JONES_HV)
    assert scalar.shape == (2, 2)
    np.testing.assert_array_equal(scalar, vector[0])


def test_sample_frequency_array(
        aperture_plane_model: primary_beam.PrimaryBeamAperturePlane) -> None:
    l = np.array([0.01])
    m = np.array([0.02])
    frequency = np.array([[1000, 1200], [1100, 1500]]) * u.MHz

    # Evaluate one frequency at a time
    expected = np.zeros(frequency.shape + (1, 2, 2), np.complex64)
    # Astropy units don't work with np.ndenumerate, hence npindex instead
    for idx in np.ndindex(frequency.shape):
        expected[idx] = aperture_plane_model.sample(
            l, m, frequency[idx],
            primary_beam.AltAzFrame(),
            primary_beam.OutputType.JONES_HV)
    # Evaluate them all together
    actual = aperture_plane_model.sample(
        l, m, frequency,
        primary_beam.AltAzFrame(),
        primary_beam.OutputType.JONES_HV)

    np.testing.assert_array_equal(actual, expected)


def test_sample_lm_broadcast(aperture_plane_model: primary_beam.PrimaryBeamAperturePlane) -> None:
    l = np.array([0.01, 0.02])
    m = np.array([0.03, -0.04, 0.01])
    frequency = 1500 * u.MHz

    actual = aperture_plane_model.sample(
        l[np.newaxis, :], m[:, np.newaxis], frequency,
        primary_beam.AltAzFrame(),
        primary_beam.OutputType.JONES_HV)

    expected = np.zeros((len(m), len(l), 2, 2), np.complex64)
    for i in range(len(m)):
        for j in range(len(l)):
            expected[i, j] = aperture_plane_model.sample(
                l[j], m[i], frequency,
                primary_beam.AltAzFrame(),
                primary_beam.OutputType.JONES_HV)

    # It isn't exact, presumably because the matrix multiplications sum
    # in a different order depending on size.
    np.testing.assert_allclose(actual, expected, atol=1e-7)


@pytest.mark.parametrize(
    'l, m, frequency',
    [
        (-0.3, 0.002, 1500 * u.MHz),      # l too small
        (0.3, 0.002, 1500 * u.MHz),       # l too large
        (0.001, -0.7, 1500 * u.MHz),      # m too small
        (0.001, 1.0, 1500 * u.MHz),       # m too large
        (0.001, 0.002, 900 * u.MHz),      # frequency too low
        (0.001, 0.002, 2000 * u.MHz)      # frequency too high
    ]
)
def test_sample_out_of_range(
        aperture_plane_model: primary_beam.PrimaryBeamAperturePlane,
        l: float, m: float, frequency: u.Quantity) -> None:
    actual = aperture_plane_model.sample(
        l, m, frequency,
        primary_beam.AltAzFrame(),
        primary_beam.OutputType.JONES_HV)
    expected = np.full((2, 2), np.nan)
    np.testing.assert_array_equal(actual, expected)


def test_sample_partially_out_of_range(
        aperture_plane_model: primary_beam.PrimaryBeamAperturePlane) -> None:
    model = aperture_plane_model
    l = [0.1, -0.19, 0.21]     # 0.2 is (roughly) the limit at 1.5 GHz
    m = [0.0]
    frequency = [1000, 1500, 1630, 2000] * u.MHz
    actual = model.sample(
        l, m, frequency,
        primary_beam.AltAzFrame(),
        primary_beam.OutputType.JONES_HV)

    expected_nan = np.array([
        [False, False, False],
        [False, False, True],
        [False, True, True],
        [True, True, True]     # frequency is out of range
    ])

    for i in range(2):
        for j in range(2):
            np.testing.assert_array_equal(np.isnan(actual[..., i, j]), expected_nan)


@pytest.mark.parametrize(
    'out, expectation',
    [
        pytest.param(
            np.empty((2, 2, 2, 2), np.complex64),
            pytest.raises(ValueError,
                          match=r'out must have shape \(2, 2, 1, 2, 2\), not \(2, 2, 2, 2\)'),
            id='shape'
        ),
        pytest.param(
            np.empty((2, 2, 1, 2, 2), np.float64),
            pytest.raises(TypeError,
                          match=r'out must have dtype complex64, not float64'),
            id='dtype'
        ),
        pytest.param(
            np.empty((2, 2, 1, 2, 2), np.complex64, order='F'),
            pytest.raises(ValueError,
                          match=r'out must be C contiguous'),
            id='contiguous'
        )
    ]
)
def test_sample_bad_out(
        aperture_plane_model: primary_beam.PrimaryBeamAperturePlane,
        out: np.ndarray,
        expectation) -> None:
    model = aperture_plane_model
    l = np.array([0.01])
    m = np.array([0.02])
    frequency = np.array([[1000, 1200], [1100, 1500]]) * u.MHz
    with expectation:
        model.sample(
            l, m, frequency,
            primary_beam.AltAzFrame(),
            primary_beam.OutputType.JONES_HV,
            out=out)


@pytest.mark.parametrize('frame, output_type', FRAME_OUTPUT_TYPE_COMBOS)
def test_sample_out(
        aperture_plane_model: primary_beam.PrimaryBeamAperturePlane,
        frame: Union[primary_beam.AltAzFrame, primary_beam.RADecFrame],
        output_type: primary_beam.OutputType) -> None:
    model = aperture_plane_model
    l = np.array([0.01, 0.02, 1])[np.newaxis, :]
    m = np.array([-0.02, -0.03])[:, np.newaxis]
    frequency = np.array([[1000, 1200], [1100, 1500]]) * u.MHz
    expected = model.sample(l, m, frequency, frame, output_type)
    out = np.zeros(expected.shape, expected.dtype)
    actual = model.sample(l, m, frequency, frame, output_type, out=out)
    assert actual is out
    np.testing.assert_array_equal(out, expected)


def _skyoffset_matrix(origin: SkyCoord):
    """Reproduce the matrix used by :class:`astropy.coordinates.SkyOffsetFrame`.

    This is the matrix for transforming from a reference frame to an offset
    frame. It does not support the rotation attribute of
    :class:`astropy.coordinates.SkyOffsetFrame`.
    """
    # Based on reference_to_skyoffset in the astropy code.
    origin_sph = origin.spherical
    maty = rotation_matrix(-origin_sph.lat, 'y')
    matz = rotation_matrix(origin_sph.lon, 'z')
    return maty @ matz


def _coords_to_lm(coords: SkyCoord, origin: SkyCoord) -> np.ndarray:
    """Convert sky coordinates to l/m direction cosines.

    The `coords` and `origin` must be in the same frame.

    This would be simpler with SkyOffsetFrame, but unfortunately
    https://github.com/astropy/astropy/issues/11277 makes it break randomly.
    """
    assert coords.frame.is_equivalent_frame(origin.frame)
    mat = _skyoffset_matrix(origin)
    offsets = coords.represent_as(CartesianRepresentation).transform(mat)
    return offsets.xyz[1:]    # astropy spherical (0, 0) at +x, so y, z are l, m


def _lm_to_coords(l: ArrayLike, m: ArrayLike, origin: SkyCoord) -> SkyCoord:
    """Convert l/m cooordinates relative to an origin into coordinates.

    This would be simpler with SkyOffsetFrame, but unfortunately
    https://github.com/astropy/astropy/issues/11277 makes it break randomly.
    """
    l = np.asarray(l)
    m = np.asarray(m)
    n = np.sqrt(1 - (l * l + m * m))
    nlm = CartesianRepresentation(np.stack([n, l, m]))
    mat = _skyoffset_matrix(origin).T   # Transposing a rotation matrix inverts it
    xyz = nlm.transform(mat)
    return SkyCoord(xyz.represent_as(UnitSphericalRepresentation), frame=origin)


@pytest.mark.parametrize('lat', [-30 * u.deg, 35 * u.deg])
def test_sample_radec(
        aperture_plane_model: primary_beam.PrimaryBeamAperturePlane,
        lat: u.Quantity) -> None:
    model = aperture_plane_model
    location = EarthLocation.from_geodetic(18 * u.deg, lat=lat, height=100 * u.m)
    obstime = Time('2021-04-22T13:00:00Z')
    frequency = 1 * u.GHz
    altaz_frame = AltAz(obstime=obstime, location=location)
    target_altaz = SkyCoord(alt=70 * u.deg, az=150 * u.deg, frame=altaz_frame)

    l_altaz = [-0.002, 0.001, 0.0, 0.0, 0.0]
    m_altaz = [0.0, 0.02, 0.0, -0.03, 0.01]
    coords_altaz = _lm_to_coords(l_altaz, m_altaz, target_altaz)

    target_icrs = target_altaz.icrs
    coords_icrs = coords_altaz.icrs
    l_icrs, m_icrs = _coords_to_lm(coords_icrs, target_icrs)

    out_radec = model.sample(
        l_icrs, m_icrs, frequency,
        primary_beam.RADecFrame.from_sky_coord(target_icrs),
        primary_beam.OutputType.JONES_HV)
    out_altaz = model.sample(
        l_altaz, m_altaz, frequency,
        primary_beam.AltAzFrame(),
        primary_beam.OutputType.JONES_HV)
    # Tolerance is high because RADecFrame doesn't account for aberration
    np.testing.assert_allclose(out_radec, out_altaz, atol=1e-4)


def test_samples_jones_xy(aperture_plane_model: primary_beam.PrimaryBeamAperturePlane) -> None:
    model = aperture_plane_model
    l = [-0.002, 0.001, 0.0, 0.0, 0.0]
    m = [0.0, 0.02, 0.0, -0.03, 0.01]
    # Use at least one dimension in frequency to check that tensor products use
    # the right axes.
    frequency = [1.25, 1.5] * u.GHz
    frame = primary_beam.RADecFrame(30 * u.deg)

    out_hv = model.sample(
        l, m, frequency, frame, primary_beam.OutputType.JONES_HV)
    out_xy = model.sample(
        l, m, frequency, frame, primary_beam.OutputType.JONES_XY)
    expected_xy = frame.jones_hv_to_xy() @ out_hv
    np.testing.assert_allclose(out_xy, expected_xy, rtol=0, atol=1e-6)


def test_sample_mueller(aperture_plane_model: primary_beam.PrimaryBeamAperturePlane) -> None:
    # A brightness matrix is an expectation of an outer product of the
    # voltage vector with itself. Pick a few sample voltages to determine a
    # brightness matrix, and cross-check by processing those individual
    # voltages through JONES_XY.
    model = aperture_plane_model
    rs = np.random.RandomState(1)
    voltages_shape = (2, 6)
    voltages = rs.normal(size=voltages_shape) + 1j * rs.normal(size=voltages_shape)
    l = [-0.002, 0.001, 0.0, 0.0, 0.0]
    m = [0.0, 0.02, 0.0, -0.03, 0.01]
    # Use at least one dimension in frequency to check that tensor products use
    # the right axes.
    frequency = [1.25, 1.5] * u.GHz
    frame = primary_beam.RADecFrame(30 * u.deg)

    mueller = model.sample(l, m, frequency, frame, primary_beam.OutputType.MUELLER)
    B = voltages @ voltages.T.conj()            # Matrix of [[XX, XY], [YX, YY]]
    B = primary_beam._XY_TO_IQUV @ B.ravel()    # Brightness in IQUV
    actual = mueller @ B                        # Apparent IQUV

    xy = model.sample(l, m, frequency, frame, primary_beam.OutputType.JONES_XY)
    obs_voltages = xy @ voltages
    vis = obs_voltages @ obs_voltages.swapaxes(-1, -2).conj()  # Matrix of [[XX, XY], [YX, YY]]
    vis = vis.reshape(vis.shape[:-2] + (4,))    # Flatten to [XX, XY, YX, YY]
    print(vis.shape)
    # Treat vis as a stack of vectors to multiply by the matrix
    vis = np.tensordot(vis, primary_beam._XY_TO_IQUV, axes=((-1,), (-1,)))

    np.testing.assert_allclose(actual, vis, rtol=0, atol=1e-5)


def test_sample_unpolarized_power(
        aperture_plane_model: primary_beam.PrimaryBeamAperturePlane) -> None:
    model = aperture_plane_model
    l = [-0.002, 0.001, 0.0, 0.0, 0.0]
    m = [0.0, 0.02, 0.0, -0.03, 0.01]
    frequency = [1.25, 1.5] * u.GHz
    frame = primary_beam.RADecFrame(parallactic_angle=30 * u.deg)

    mueller = model.sample(
        l, m, frequency, frame, primary_beam.OutputType.MUELLER)
    unpol = model.sample(
        l, m, frequency, frame, primary_beam.OutputType.UNPOLARIZED_POWER)

    np.testing.assert_allclose(unpol, mueller[..., 0, 0], atol=1e-5)


@pytest.mark.parametrize(
    'frequency',
    [
        1500 * u.MHz,
        [1250, 1300] * u.MHz,
        [[1250, 2000], [900, 1500]] * u.MHz
    ]
)
@pytest.mark.parametrize('frame, output_type', FRAME_OUTPUT_TYPE_COMBOS)
def test_sample_grid(
        aperture_plane_model: primary_beam.PrimaryBeamAperturePlane,
        frequency: u.Quantity,
        frame: Union[primary_beam.AltAzFrame, primary_beam.RADecFrame],
        output_type: primary_beam.OutputType) -> None:
    model = aperture_plane_model
    l = [-0.002, 0.001, 0.0, 0.0, 0.0]
    m = [0.0, 0.02, 0.0, -0.03, 0.01]

    actual = model.sample_grid(l, m, frequency, frame, output_type)
    expected = model.sample(
        np.array(l)[np.newaxis, :], np.array(m)[:, np.newaxis], frequency,
        frame, output_type)
    np.testing.assert_allclose(actual, expected, atol=1e-5)

    # Test output parameters
    out = np.zeros(expected.shape, expected.dtype)
    ret = model.sample(
        np.array(l)[np.newaxis, :], np.array(m)[:, np.newaxis], frequency,
        frame, output_type,
        out=out)
    assert ret is out
    np.testing.assert_array_equal(out, actual)
