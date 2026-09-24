.. _terminology:

Terminology
===========

*This guide aims to provide a clear understanding of the terminology used in LSAPy,
as some terms may have a slightly different meaning than its common usage in other contexts.*

.. note::

    LSAPy is a relatively new package and the terminology used may evolve over time.
    While we strive to maintain consistency, some terms may be subject to change in future developments.

.. glossary::

    Indicator
        A `xarray.DataArray`__ representing a measure or variable of a characteristic of
        the system/component of interest (e.g., slope or precipitation for land).

        __ https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html

    Standardization Function
        A function that converts input values into a common scale,
        typically between 0 and 1, for subsequent analysis.
        A wide range of standardization functions are provided in the
        ``lsapy.standardize`` module.

    SuitabilityCriteria
        A data structure defining a set of elements and rules, including an
        :term:`Indicator` and a :term:`Standardization Function`, used to
        determine its suitability. A ``SuitabilityCriteria`` differs from an
        :term:`Indicator` in that the latter is used as input to the former,
        and different indicators can be used for the same ``SuitabilityCriteria``.
        For example, a ``SuitabilityCriteria`` used to assess the water need
        of a crop may use a monthly or annual total precipitation indicator.

        .. note::

            While *criteria* in ``SuitabilityCriteria`` might suggest a plural form,
            it refers to a single criterion in the context of LSAPy.

    LandSuitabilityAnalysis
        A data structure evaluating land suitability for a specific use or purpose,
        by aggregating a set of :term:`SuitabilityCriteria`.

    Aggregating
        Aggregating refers to the process of combining several `xarray.DataArray`__
        objects into a single one, by applying a specific operation (e.g., min, mean,
        geometric mean...) across the input arrays. In LSAPy, this is used to combine
        multiple :term:`SuitabilityCriteria`.

        __ https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html
