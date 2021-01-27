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
    and treated like any other baseline.

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
