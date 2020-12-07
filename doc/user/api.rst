API
===

Class hierarchy
---------------
Models are organised in a class hierarchy. At the base of the hierarchy is
:class:`katsdpmodels.model.Model`. In contains general metadata about the
model, as well as a :meth:`~katsdpmodels.model.Model.to_file` method to
serialise the model to a file.

For each model type there is a subclass. For example, RFI masks are described
by :class:`katsdpmodels.rfi_mask.RFIMask`. These classes contain further
properties and methods to interrogate the model. They are further subclassed
into concrete classes that are specific to the format (such as
:class:`~katsdpmodels.rfi_mask.RFIMaskRanges`). To use a model it is generally
not necessary to know about these concrete classes, and it is recommended that
one works just with the abstract classes; this allows for new model formats to
be defined and deployed without requiring changes to your code. When creating
new models it is necessary to use the format-specific classes.

Fetching
--------
The :mod:`katsdmodels.fetch` sub-package provides modules for retrieving
models over HTTP. For code using :mod:`asyncio`, use
:mod:`katsdpmodels.fetch.aiohttp`; for synchronous code, use
:mod:`katsdpmodels.fetch.requests`. These depend on the
:mod:`aiohttp` and :mod:`requests` packages respectively. When
installing katsdpmodels, specify the dependency as ``katsdpmodels[aiohttp]``
or ``katsdpmodels[requests]`` to pull in the appropriate dependencies.

Other than the use of coroutines, the two APIs are very similar, so only the
synchronous API is described here.

The fetcher operates on URLs. It does not know how to construct an URL for a
specific target or configuration. For retrieving models stored in a MeerKAT
data set, refer to :doc:`telstate` for a convenience wrapper that extracts
URLs encoded into the data set.

To fetch models one first creates a
:class:`~katsdpmodels.fetch.requests.Fetcher`. This wraps a
:class:`requests.Session`. Once it is no longer needed, it can be closed to
release the OS resources. It can also be used as a context manager for this
purpose. Then use :meth:`~katsdpmodels.fetch.requests.Fetcher.get` to retrieve
a model.

.. warning::

   Closing a fetcher invalidates all the models it returned. The fetcher
   should only be closed once the models are no longer being used.

A fetcher caches all the models it fetches. It is thus reasonably cheap
to re-fetch a model. More importantly, some models with different URLs may be
in fact be the same model (due to :ref:`aliasing <aliases>`), and the fetcher
will re-use the same model in such cases rather than fetching a duplicate
copy.

When fetching a model, the type should be known in advance. This is indicated
by passing the type-specific class (such as :class:`.RFIMask`, *not*
:class:`.RFIMaskRanges`). The fetcher validates the retrieved model against
this type and :ref:`raises an exception <exception-handling>` if it does not
match.

It is possible to provide your own :class:`requests.Session` when constructing
the fetcher. This could be used to set additional headers (such as for
authentication) or otherwise customize behaviour. If you supply your own
session, you are responsible for closing it.

.. _exception-handling:

Exception handling
------------------
The base class :exc:`.ModelError` is used for all errors relating to the
content of model files. It has properties
:attr:`~katsdpmodels.models.ModelError.original_url` and
:attr:`~katsdpmodels.models.ModelError.url` which respectively contain the
requested URL and the final URL (after redirections and aliases) of the
problematic model. If the model was not retrieved via HTTP these may be
``None``. See the reference documentation for subclasses that indicate more
specific errors.

This exception is *not* used for transport-level errors when fetching a model,
such as :exc:`OSError` for file errors or exceptions from the HTTP library
when fetching over HTTP.
