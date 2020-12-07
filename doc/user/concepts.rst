Concepts
========

The package defines models of various :dfn:`types`, with an associated Python
class for each type. For example, :class:`~.RFIMask` is the class
corresponding to RFI masks.

Models describe :dfn:`targets`. The target is an abstract description,
independent of time, such as "the UHF receiver on antenna m031". Over time, a
target may change. For example, a receiver may become faulty and be swapped
out with a spare, which will have different characteristics. Each such variant
of a target is called a :dfn:`configuration`.

In an ideal world, each configuration would have only a single model that
describes it perfectly. In practice, models are seldom perfect and so new
versions might be produced that better describe the same configuration. The
difference between a new model and a new configuration is that one should
always use a model appropriate to the configuration that was used at the time
of the observation, but can safely use a newer model for that configuration
(assuming the newer model is actually better).

Each model also has a :dfn:`format`, which determines what set of parameters
is required and how they are evaluated to define a model. For example, a
bandpass response might be defined by a piecewise-linear function, a spline,
or a polynomial, and these would be different formats for the same type.

.. _aliases:

Models are stored in files, which could be in a local filesystem or accessed
remotely via HTTP. In some cases, several targets or configurations may share
the same model. To support this, :dfn:`alias` files point at a model stored
elsewhere in the same model database.
