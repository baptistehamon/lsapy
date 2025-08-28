______________________________________________________________________

title: 'LSAPy: Land Suitability Analysis in Python'
tags:

- land suitability analysis
- land evaluation
- geospatial analysis
- GIS
- python
  authors:
- name: Baptiste Hamon
  orcid: 0009-0007-4530-9772
  affiliation: 1
  affiliations:
- name: Department of Civil and Environmental Engineering, University of Canterbury, Christchurch, New Zealand
  index: 1
  date: 28 August 2025
  bibliography: paper.bib

______________________________________________________________________

# Summary

LSAPy is a highly customizable Python library designed to streamline and enhance Land Suitability Analysis (LSA) workflows. The package implements a fuzzy-logic approach and provides three core objects—`SuitabilityFunction`, `SuitabilityCriteria`, and `LandSuitabilityAnalysis`—that work together to deliver a flexible and user-defined LSA framework. By relying on `xarray` objects for computation [@hoyer2017], LSAPy seamlessly integrates with the broader Python ecosystem, such as `dask` for efficient parallel processing and `matplotlib` for data visualisation. Its modular design addresses some limitations of existing LSA tools by offering greater flexibility, reproducibility, and scalability for research and practical applications.

# Statement of need

In the past decades, several software programs have been developed to perform land evaluation or suitability analysis, including ALES [@johnson1991], Micro-LEIS [@delarosa1992; @delarosa2004], LEIGIS [@kalogirou2002], ALSE [@elsheikh2013] and general-purpose GIS platforms such as ArcGIS and QGIS. While each of these tools offers distinct advantages, a limitation that often arises is the software-imposed constraints and the lack of freedom left to users [@elsheikh2013; @chen2022; @asaad2022]. These limitations manifest in several ways. Desktop GIS solutions, for example, often present challenges related to operating system dependency, the cost of proprietary software (e.g., ArcGIS), and difficulties in integration with broader analytical frameworks [@chen2022]. Furthermore, many specialised programs (e.g., Micro-LEIS, LEIGIS, ALSE) do not permit modification of the land characteristics used in the analysis [@elsheikh2013; @asaad2022]. This rigidity is problematic, as the predefined land characteristics may not apply to all crops or may require augmentation with additional parameters. Additionally, some tools (e.g., LEIGIS) support only a limited selection of crops for evaluation [@elsheikh2013].

Recently, two libraries – ALUES [@asaad2022] and PyLUSAT [@chen2022] – have been developed to address some of these limitations. ALUES, an R package, supports the evaluation of 56 crops and allows the addition of new ones. While it addresses some of the limitations discussed above, it falls short in others. For example, it relies on fixed criteria grouped into three unmodifiable categories [@asaad2022]. PyLUSAT, a Python package, enables land suitability analysis using vector data, which offers specific advantages [@chen2022]. Nevertheless, raster-based approaches are generally more efficient for data combinations and complex calculations [carr2007], particularly when integrating climate data, which is often distributed in netCDF format and best processed using raster routines. Converting such data to a vector format for analysis is technically feasible but suboptimal, leading to increased computational overhead and memory usage. Consequently, raster-based methods are essential for large-scale analyses, such as land suitability assessments using climate projections.

Beyond these technical constraints, the lack of user flexibility in existing software introduces further limitations. First, when programs offer a restricted set of functionalities, they hinder the reproducibility of scientific results. For instance, ALSE aggregates criteria suitability using the maximum limitation method [@elsheikh2013], whereas ALUES provides only minimum, maximum, and average aggregation options [@asaad2022]. This discrepancy renders the reproduction of cross-software results impossible. Second, tools designed for specific use cases (e.g., agriculture) limit broader adoption and, in the case of free and open-source software (FOSS), may impede community engagement and growth. Finally, rigid frameworks restrict the ability to explore edge cases or out-of-the-box analyses, which are increasingly important in a rapidly changing environment.

The limitations of existing software programs motivated the development of LSAPy (Land Suitability Analysis in Python), a new, highly customizable Python package that supports raster-like data. Thanks to its custom-based approach, frameworks used in most software programs mentioned here can be implemented in LSAPy.

# Key features

LSAPy provides three core objects that operate together to perform the land suitability analysis (LSA) according to user-defined frameworks \\autoref{fig:lsapy}.

![Overview of LSAPy’s object structures and their associated properties and methods. * is used as an abbreviation of .abel{fig:lsapy}](lsapy.png)

## Suitability Function

`SuitabilityFunction` is built around a function that transforms input data into suitability values \\autoref{fig:lsapy}. LSAPy includes built-in functions for both discrete and continuous data. For the latter, the package follows a fuzzy-logic approach, implementing Gaussian-like and sigmoid-like membership functions previously used in LSA studies. If the provided functions do not meet user requirements, custom functions can be defined.

## Suitability Criteria

The `SuitabilityCriteria` defines an individual criteria used in LSA. Its `indicator` property refers to the input data, while `func` specifies the associated `SuitabilityFunction` \\autoref{fig:lsapy}. The `weight` and `category` properties allow users to customise how each criteria is aggregated with others in the analysis. The `compute()` method applies the `SuitabilityFunction` to the given `indicator` to calculate the suitability score.

## Land Suitability Analysis

`LandSuitabilityAnalysis` is the top-level class in LSAPy, defining the LSA framework. All criteria for the analysis are stored in the `criteria` property. The `run()` method executes the LSA, with parameters specifying the level of suitability to compute (i.e., criteria, category, or overall land suitability) and the aggregation method to use. Currently, supported aggregation methods include median, mean, weighted mean, geometric mean, weighted geometric mean, and limiting factor.

## Additional Features

- The `stats` module offers functions for computing spatial (e.g., national, regional...) summary statistics of land use suitability.
- LSAPy’s `open_data` function provides access to sample datasets, including soil/land data from NZGLID [@hamon2025d] and climate data from NEX-GDDP-CMIP6 [@thrasher2022] datasets for tutorial and training purposes.

# Research Applications

LSAPy has been used to assess the impact of climate change on apples, cherries, maize and wheat in New Zealand [@hamon2025a; @hamon2025c]. Additionally, ongoing research investigating the influence of land suitability analysis criteria on land use suitability also utilises LSAPy[@hamon2025b].

# Acknowledgements

The development of LSAPy began as part of a PhD funded by the [Food Transition 2050](<%5Bhttps://www.foodtransitions2050.ac.nz/%5D(https://www.foodtransitions2050.ac.nz/)>) Joint Postgraduate School.

# References
