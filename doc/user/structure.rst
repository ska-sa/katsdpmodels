Directory structure
===================
This package is agnostic to the arrangement of model data; it simply fetches
and processes URLs. Here we document the directory structure adopted for the
MeerKAT telescope, both as a suggestion for other telescopes and for users who
wish to explore the available models.

Models are arranged in three levels, connected by alias files:

1. The models themselves are stored as
   :samp:`{model_type}/fixed/sha256_{hash}.h5`, where :samp:`{model_type}` is
   the value stored in the HDF5 ``model_type`` attribute.

2. For each configuration of each target (see :doc:`concepts`), the current
   model is pointed to by :samp:`{model_type}/config/{target}/{config}.alias`.

3. For each target, the alias for the current configuration is pointed to by
   :samp:`{model_type}/current/{target}.alias`.

The :samp:`{target}` may contain multiple components separated by slashes. It
may also be empty for telescope-level models, in which case the level 2 alias
becomes :samp:`{model_type}/config/{config}.alias` and the level 3 alias
becomes :samp:`{model_type}/current.alias`. The :samp:`{config}` is intended
to be a flat version string for the configuration. See :doc:`telstate` for an
example of how these aliases point to each other.

Below, we list the naming conventions currently in use for MeerKAT for
:samp:`{target}` and :samp:`{config}`. Other telescopes will likely use
a different configuration management system and need to describe a different
range of targets, so are not expected to use the same naming conventions.

RFI mask
--------
target
    Empty string (i.e. a telescope-level model)

config
    ``meerkat``

Band mask
---------
target
    :samp:`{band}/nb_ratio={ratio}`, where :samp:`{band}` is the single-letter
    abbreviation for the receiver band (``l``, ``s``, ``u`` or ``x`` in
    MeerKAT) and :samp:`{ratio}` is the integer ratio between the digitised
    bandwidth and the output correlator bandwidth. It is 1 for a wideband
    instrument and larger for narrowband instruments.

config
    The document number of the release note for the correlator.
