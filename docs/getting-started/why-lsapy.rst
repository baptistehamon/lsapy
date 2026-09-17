Why LSAPy?
==========

Purpose
-------

There is a growing interest of conducting Land Suitability Analysis (LSA) under climate change, as it is crucial to understand how changing climate conditions may affect the suitability of land for various uses.
However, climate data are often provided in grid-based formats, such as netCDF files, which can be complex and large, making it challenging to analyze using existing tools.
Moreover, existing tools (see `Alternative tools`_) for LSA often have limitations in terms of flexibility and scalability, which can hinder the ability to explore specific cases.
These limitations motivated the development of **LSAPy**, a Python library that can handle grid-based data and integrate seamlessly with the broader Python ecosystem.
Although LSAPy was initially developed for agricultural land suitability analysis, it is designed to be versatile and can be applied to various land use types,
including forestry, urban planning, and conservation, or for risks assessment.

Alternative tools
-----------------

**PyLUSAT**
    | *"Python Land-Use Suitability Analysis Toolkit"* (`Source code <https://github.com/chjch/pylusat>`_, `Documentation <https://pylusat.readthedocs.io/en/latest/>`_, `Article <https://doi.org/10.1016/j.envsoft.2022.105362>`_).
    | PyLUSAT is a Python library for vector-based LSA.

**ALUES**
    | *"Agricultural Land Use Evaluation System"* (`Source code <https://github.com/alstat/ALUES/tree/master>`_, `Documentation <https://alstat.github.io/ALUES/>`_, `Article <https://doi.org/10.21105/joss.04228>`_).
    | ALUES is a R package to evaluate the land suitability of different crops based on the Food and Agriculture Organization (FAO) and International Rice Research Institute (IRRI) methodology.

**Other software (including early tools)**
    * ALES (`Article <https://doi.org/10.1111/j.1475-2743.1991.tb00881.x>`_).
    * Micro-LEIS: Computer-based land evaluation information system (`Article <https://doi.org/10.1111/j.1475-2743.1992.tb00900.x>`_).
    * LEIGIS (`Article <https://doi.org/10.1016/S0198-9715(01)00031-X>`_).
    * ALSE: Agricultural Land Suitability Evaluator (`Article <https://doi.org/10.1016/j.compag.2013.02.003>`_).
    * General-purpose platforms (e.g., ArcGIS, QGIS).
