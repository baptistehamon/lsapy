=======================
LandSuitabilityAnalysis
=======================

The `LandSuitabilityAnalysis` is the top-level class in LSAPy and defines the LSA framework.

.. currentmodule:: lsapy

.. autosummary::
   :toctree: generated/

   LandSuitabilityAnalysis

Methods
-------
.. autosummary::
   :toctree: generated/

   LandSuitabilityAnalysis.run

Attributes
----------
.. autosummary::
   :toctree: generated/

   LandSuitabilityAnalysis.land_use
   LandSuitabilityAnalysis.criteria
   LandSuitabilityAnalysis.data
   LandSuitabilityAnalysis.category
   LandSuitabilityAnalysis.criteria_by_category
   LandSuitabilityAnalysis.weights_by_category
   LandSuitabilityAnalysis.attrs


Criteria Aggregation
--------------------

The underlying criteria aggregation method used in ``LandSuitabilityAnalysis.run`` is implemented in the `aggregate` module.

.. autosummary::
   :toctree: generated/

   aggregate.aggregate
