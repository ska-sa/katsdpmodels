User manual
===========

This package contains code for defining and accessing models of various parts
of a radio telescope. It is designed for use with MeerKAT, but should be
generic enough to apply to other radio telescopes. It does not contain any
specific models; rather, it provides APIs to retrieve models that are
associated with observations.

The package defines :doc:`file formats <formats>` for a number of types of models.
The format documentation is useful if you want to make your own models,
understand the limitations of the models, or access them without using this
package. However, an :doc:`API <api>` is provided that provides ways to retrieve
and sample the models.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   concepts
   formats
   api
   structure
   telstate
