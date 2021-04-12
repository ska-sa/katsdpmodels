Model file formats
==================

Models are stored in HDF5 files. Where boolean or complex values are stored,
they use the conventions set by `h5py`_. The extension should be :file:`.h5`
(:file:`.hdf5` is also accepted by this package). When served over HTTP, the
``Content-Type`` should be ``application/x-hdf5``.

.. _h5py: https://docs.h5py.org/en/stable/

The base filename should be a checksum of the file content: this ensures that
models can be cached indefinitely as a new version of a model is guaranteed to
have a new checksum. The filename should thus be
:file:`{algorithm}_{checksum}.h5` where the only currently supported value for
:samp:`{algorithm}` is ``sha256``. The package verifies the checksum if it is
recognised, but using a filename without a checksum is not an error.

Alternatively, a model with an extension of :file:`.alias` or ``Content-Type``
of ``text/plain`` is an :dfn:`alias`. It must contain a relative path to the
actual model. Aliases can also point at other aliases, but there is an
implementation-defined maximum redirection depth. Aliases may not change
origin (scheme, host or port); this makes it practical to mirror a set of
models in a filesystem or on another server without needing to rewrite the
aliases.

Model files will contain the following attributes on the root (the ``model_*``
namespace is reserved for the general framework, and shouldn't be used for
type-specific data):

model_type
    The model type (see :doc:`concepts`). See later sections for model types
    defined by this specification. New model types may be defined in future.
    To avoid collisions, users may define their own types with the prefix
    ``x_``.

model_format
    Indicates how the model is encoded in the file, and is specific to each
    model type. If backwards-incompatible changes are made to the format, a
    new ``model_format`` must be used. On the other hand, an existing format
    may be extended with new datasets, attributes, etc. while maintaining
    backwards compatibility.

model_version
    Simple integer version of the model for a particular configuration,
    incremented only when semantic content changes to the model occur. Thus,
    ``model_format`` may be updated without changing this version, provided
    that the new parameters describe the same model.

model_target (optional)
    Human-readable string describing the applicable targets and / or
    configurations of the model. Note that because models may be shared
    (e.g., multiple receptors may have the same SEFD model), this might be
    more general than the target the user requested.

model_comment (optional)
    Human-readable string which may describe the provenance of the model or
    any other comments about it.

model_created (optional)
    Timestamp at which the model was created, in RFC 3339 format.

model_author (optional)
    Name of the person who created the model. It may optionally include an
    email address in angle brackets (name-addr format in RFC 2822).

Physical units are unscaled. In other words, frequencies are in units of Hz,
flux density in units Jy and so on. Angles are in radians, or expressed as
direction cosines when appropriate.

RFI Mask
--------
This specifies frequencies which are affected by Radio Frequency
Interference (RFI). It is assumed that the RFI environment is consistent
across a whole telescope.

Attributes
^^^^^^^^^^
model_type
    ``rfi_mask``

model_format
    ``ranges``

mask_auto_correlations
    A boolean to indicate whether auto-correlations (both same-hand and
    cross-hand) should be masked, in which case it is done for frequencies
    covered by any of the ranges. That is, if this is ``False``, no
    auto-correlations will be masked for RFI. If this is ``True``,
    auto-correlations are considered to be very short (zero-length) baselines
    and treated like any other baselines.

Datasets
^^^^^^^^
ranges
    A 1D dataset with the following columns

    min_frequency, max_frequency (float)
        Frequencies of an RFI-affected region of the spectrum. Channels are
        masked if any part of the channel lies inside any of the ranges.

    max_baseline (float)
        Maximum (inclusive) baseline length for which these frequencies should
        be masked. This should be non-negative. To mask all baselines, use ∞.

Band mask
---------
A band mask is similar to an RFI mask, but indicates frequencies that are
unusable due to the receiver response rather than RFI (it can also include
digital effects, such as a digital band-pass filter). It specifies ranges
relative to the digitiser band, making it suitable for describing effects that
occur after down-conversion in a heterodyne system.

Attributes
^^^^^^^^^^
model_type
    ``band_mask``

model_format
    ``ranges``

Datasets
^^^^^^^^
fractional_ranges
    A 1D dataset with the following columns

    min_fraction, max_fraction (float)
        Range of the band to mask. The values are between 0.0 and 1.0, with
        0.0 indicating the lowest nominal frequency and 1.0 the highest
        nominal frequency (both in the digitised bandwidth). Channel i should
        be masked if :math:`[\frac{i-0.5}{nchans}, \frac{i+0.5}{nchans}]`
        overlaps any of the ranges. Note that this means that channel 0
        is centred at 0.0 but extends below it.

Primary beam
------------
The file contains the information in a compact Fourier-transform ("aperture
plane") form. It comprises a 3-dimensional array J of Jones matrices. Another
three arrays :math:`\nu`, :math:`y` and :math:`x` indicate the frequency (in
Hz) and spatial position (in metres) of each sample along the axes.

To determine the response at a sampled frequency :math:`\nu_f` and some
direction, turn the direction into direction cosines :math:`l` and :math:`m`
relative to the pointing centre (as defined in
:class:`katsdpmodels.primary_beam.AltAzFrame`). Then the Jones
matrix for the response is

.. math:: \frac{1}{|x|\cdot|y|}
          \sum_{j,k} e^{-2\pi i (x_j l + y_k m)\nu_f/c} J_{f,k,j},

where :math:`|x|`, :math:`|y|` are the number of elements in the respective
arrays and :math:`c` is the speed of light.
The Jones matrices correspond to :attr:`.OutputType.JONES_HV`. Note the
reversed axis order in accessing :math:`J`.

To sample at an intermediate frequency, use linear interpolation along the
frequency axis in the aperture plane.

See :class:`katsdpmodels.primary_beam.PrimaryBeam` for other definitions and
sign conventions.

Attributes
^^^^^^^^^^
model_type
    ``primary_beam``

model_format
    ``aperture_plane``

antenna
    The name of the antenna to which this model applies. Absent if this model
    is not specific to a single antenna (the more generic ``model_target`` may
    provide human-readable information about the range of applicable antennas).

receiver
    Serial number of the receiver to which this model applies. Absent if this
    model is not specific to a single receiver.

x_start, y_start
    Coordinates associated with the first sample along the respective axes.

y_step, y_step
    Spacing between samples along the respective axes.

The `antenna` and `receiver` may be compared to the actual antenna
and receiver identifiers in use to detect incorrect models (for example, if a
receiver was swapped out but the model was not updated).

Datasets
^^^^^^^^
frequency
    1D array of sampled frequencies.

aperture_plane
    5D array, with axes corresponding to (in order)

    - row of Jones matrix (length 2)
    - column of Jones matrix (length 2)
    - frequency
    - y
    - x
