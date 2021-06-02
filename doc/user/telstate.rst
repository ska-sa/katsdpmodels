Telescope state integration
===========================
The MeerKAT telescope stores information about each observation using the
`katsdptelstate`_ package. Information about the models used in an observation
(and where to retrieve them) is stored in the telescope state so that they may
be retrieved later.

.. _katsdptelstate: https://katsdptelstate.readthedocs.io

The telescope state contains an ``sdp_model_base_url`` attribute which holds
a base URL for the models. All other URLs in the telescope state are relative
to this. For each model used there are two entries in the telescope state. One
has the suffix ``fixed`` and references the exact model that was current
at the time of the observation.  This allows any analysis that was done at the
time to be reproduced. The second has the suffix ``config`` and references an
alias for the configuration.  If an improved model is computed for this
configuration and uploaded to the model store, using this alias will
automatically reference this updated model.  This allows analysis to use the
best current model when it is not necessary to match any online use of the
model done when the observation was made.

Because the MeerKAT Science Data Processor (SDP) allows multiple streams of
each type to be grouped into a single telescope state, some keys for
specifying models are scoped by the name of the
``antenna-channelised-voltage`` input stream to which they apply. This is
represented as :samp:`{acv}` in the names below.

Example
-------
While katsdpmodels doesn't yet support primary beam models, we'll use it for
an example because it's a complex case that demonstrates multiple aspects of
the design. Consider a primary beam model for the UHF receiver on antenna
m012. For simplicity, we'll assume the configuration depends only on the dish
and the receiver, with no other versioning. Then the initial setup as aliases
may be as seen in the left side of the figure below. After an observation has
been run, the telescope state (as serialized to the :file:`.rdb` file) appears on
the right.

.. tikz::
   :libs: chains, positioning, fit
   :include: example_aliases1.tex

Now suppose further holography allowed the model for this configuration to be
improved, and a new model was uploaded. This would lead to the following
situation:

.. tikz::
   :libs: chains, positioning, fit
   :include: example_aliases2.tex

The telescope state has not been altered, but by using the
``model_primary_beam_config`` key, one can access the improved model, without
losing track of the model that was used online.

Next, suppose the receiver on m012 was swapped out for a different one. Then
the situation might change to look like this:

.. tikz::
   :libs: chains, positioning, fit
   :include: example_aliases3.tex

The existing :file:`.rdb` file continues to reference the old configuration,
but if a new observation were started now, it would follow the alias chain
down the left side to reference the new configuration and model.

Model keys by type
------------------
Each of these keys contains a relative URL, as described above. The
:samp:`{type}` is either ``fixed`` or ``config``.

RFI mask
    :samp:`model_rfi_mask_{type}`

Band mask
    :samp:`{acv}_model_band_mask_{type}`

Primary beam
    :samp:`{acv}_model_primary_beam_{group}_{type}`, where
    :samp:`{group}` is one of ``individual`` or ``cohort``. This is an indexed
    telescope state key, with the antenna name as the index.

API
---
In future `katdal`_ may be updated to hide these details and allow models to
be fetched directly from a katdal data set. Until then, one can obtain the
underlying telescope state object from a dataset as
``dataset.source.telstate``. Pass it to the constructor of
:class:`katsdpmodels.fetch.aiohttp.TelescopeStateFetcher` (asynchronous) or
:class:`katsdpmodels.fetch.requests.TelescopeStateFetcher` (synchronous), along
with an (optional) underlying fetcher. Then use
:meth:`~katsdpmodels.fetch.requests.TelescopeStateFetcher.get` to retrieve
models. Instead of passing an URL (as for the underlying fetcher classes),
pass the name of the telescope state key holding the relative URL.

.. _katdal: https://katdal.readthedocs.io/

In some cases one may wish to look up the key within a telescope state view.
This can be done by passing the view as a ``telstate`` keyword argument. Here
is an example of fetching a band mask model from a view called
``telstate_cbf`` which refers to the ``antenna-channelised-voltage`` stream:

.. code:: python

    with katsdpmodels.fetch.aiohttp.TelescopeStateFetcher(telstate) as fetcher:
        band_mask_model_key = telstate_cbf.join('model', 'band_mask', 'fixed')
        try:
            band_mask_model = await fetcher.get(band_mask_model_key,
                                                katsdpmodels.band_mask.BandMask,
                                                telstate=telstate_cbf)
            return band_mask_model
        except (aiohttp.ClientError, katsdpmodels.models.ModelError) as exc:
            logger.warning('Failed to load band_mask model: %s', exc)
            return None
