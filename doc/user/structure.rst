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
    None

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

Primary beam
------------
target
    :samp:`{group}/{antenna}/{band}`, where :samp:`{group}` is either
    ``individual`` or ``cohort``). The choice is made by the user depending on
    whether antenna-specific models are desirable or not. Readers that want the
    most accurate possible models should use ``individual``, while readers that
    will benefit from having many antennas share the same model should use
    ``cohort``. For MeerKAT, ``cohort`` will ensure that all antennas share the
    same model, although this will not hold for the MeerKAT Extension as it has
    heterogeneous dishes. The :samp:`{band}` is the single-letter abbreviation
    for the receiver band (``l``, ``s``, ``u`` or ``x``).

    Additionally, for each cohort there is a target
    :samp:`cohort/{cohort}/{band}`. For MeerKAT the cohort name is simply
    ``meerkat``. For the new dishes in the MeerKAT Extension it will be
    ``meerkat_extension``.

    Note that even if ``individual`` is requested, many or all of the antennas
    may still share the same model if per-antenna models have not been
    produced.

    The ``current/`` directory must contain all the targets defined
    above, but the ``config/`` directory need not.

config
    TBD. For cohorts it is assumed that the average beam properties will not
    change significantly over time, so an unversioned name (such as ``meerkat``)
    may suffice. For antenna-specific models the receiver serial number should
    be included, as well as some form of version number that can be updated for
    changes not related to the receiver identity.
