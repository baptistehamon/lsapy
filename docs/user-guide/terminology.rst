.. _terminology:

Terminology
===========

*This guide defines the main terms used in LSAPy. Some terms may have a slightly
different meaning here than in other contexts.*

.. note::

    LSAPy is a relatively new package, so its terminology may evolve over time.
    We aim to keep these definitions consistent as the package develops.

.. glossary::

    Indicator
        An :py:class:`xarray.DataArray` containing a measured or derived variable
        used to evaluate a characteristic of the area of interest. Examples for
        land include slope, soil drainage, and precipitation. An indicator may
        have spatial, temporal, or other dimensions.

    Standardization Function
        A function that converts input values into a common scale,
        typically between 0 and 1, for subsequent analysis. The function
        can represent a threshold, discrete classes, or a gradual
        response around an optimum. LSAPy provides a range of standardization
        functions in the ``lsapy.standardize`` module. In some contexts,
        this type of function is also called a `membership function`_.

    Suitability Score
        A standardized value expressing how suitable a location is for a
        particular criteria. In the usual LSAPy workflow, a standardization
        function produces the score from an :term:`Indicator`; scores are then
        combined to calculate overall suitability.

    SuitabilityCriteria
        A :py:class:`SuitabilityCriteria` object defining a criteria.
        It combines an :term:`Indicator` with a :term:`Standardization Function`
        and can also include function parameters, a weight, a category, and
        metadata. Computing the criteria applies the standardization function
        to the indicator and produces a :term:`Suitability Score`.

        The same criteria can be have different indicators. For example,
        a heat stress criteria for a crop may use the mean summer maximum
        temperature or the number of days above a threshold temperature, depending
        on the analysis.

        A data structure defining a set of elements and rules, including an
        :term:`Indicator` and a :term:`Standardization Function`, used to
        determine its suitability. A ``SuitabilityCriteria`` differs from an
        :term:`Indicator` in that the latter is used as input to the former,
        and different indicators can be used for the same ``SuitabilityCriteria``.
        For example, a ``SuitabilityCriteria`` used to assess the water need
        of a crop may use a monthly or annual total precipitation indicator.

        .. note::

            While *criteria* in ``SuitabilityCriteria`` might suggest a plural form,
            it refers to a single criteria in the context of LSAPy.

    LandSuitabilityAnalysis
        A data structure that evaluates land suitability for a specific land use
        by computing and aggregating a set of :term:`SuitabilityCriteria`.
        The aggregation method determines how the individual criterion scores
        are combined into the analysis result.

    Aggregating
        Aggregating refers to the process of combining several :py:class:`xarray.DataArray`
        objects into a single one, by applying a specific operation (e.g., min, mean,
        geometric mean...) across the input arrays. Weights and categories can be used to
        control how criteria contribute to the result. In LSAPy, this is used to combine
        multiple :term:`SuitabilityCriteria`.

.. _membership function: https://en.wikipedia.org/wiki/Membership_function_(mathematics)
