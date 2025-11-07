=========================
Standardization Functions
=========================

The ``lsapy.standardize`` module provides various standardization functions to transform raw input data into standardized suitability scores.
These functions can be broadly categorized into discrete, sigmoid-like, and Gaussian-like standardization.

.. currentmodule:: lsapy.standardize

Discrete Standardization
------------------------
.. autosummary::
   :toctree: generated/

   boolean
   discrete

Sigmoid-like Standardization
----------------------------

.. autosummary::
   :toctree: generated/

   logistic
   sigmoid
   vetharaniam2022_eq3
   vetharaniam2022_eq5

Gaussian-like Standardization
-----------------------------

.. autosummary::
   :toctree: generated/

   vetharaniam2024_eq8
   vetharaniam2024_eq10

Helper Functions
----------------

.. autosummary::
   :toctree: generated/

   fit
