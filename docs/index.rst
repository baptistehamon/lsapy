.. lsapy documentation master file, created by
   sphinx-quickstart on Tue Apr 22 19:32:21 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:html_theme.sidebar_secondary.remove:

===================
LSAPy Documentation
===================

Description
-----------

`LSAPy` (Land Suitability Analysis in Python) is a highly customizable, open-source Python library designed
to streamline and enhance Land Suitability Analysis (LSA) workflows. Its objective is to make conducting LSA
in Python easier and more accessible to users. The package implements a fuzzy-logic approach and provides a
set of objects that work together to deliver a flexible and user-defined LSA framework.

By relying on `xarray`_ objects for computation, `LSAPy` is specifically designed to facilitate LSA under climate
change projections. Moreover, the use of `xarray`_ allows for seamless integration with the broader Python ecosystem,
such as `dask`_ for efficient parallel processing and `matplotlib`_ for data visualisation. The modular design of
`LSAPy` addresses some limitations of existing LSA tools by offering greater flexibility, reproducibility, and
scalability for research and practical applications.

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents:

   About <readme>
   Getting Started <getting_started/index>
   User Guide <notebooks/index>
   API Reference <api>
   Community <community/index>

.. toctree::
   :titlesonly:

   changelog
   references
