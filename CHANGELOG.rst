=========
Changelog
=========

v0.3.1 (2025-11-14)
-------------------
Contributor to this version: Baptiste Hamon (@baptistehamon).

Bug fixes
^^^^^^^^^
* The `np.vectorize` decorator has been removed of standardization functions to fix fitting functions issues (issue `#101 <https://github.com/baptistehamon/lsapy/issues/101>`_, PR `#102 <https://github.com/baptistehamon/lsapy/pull/102>`_).

v0.3.0 (2025-11-10)
-------------------
Contributor to this version: Baptiste Hamon (@baptistehamon).

Announcements
^^^^^^^^^^^^^
* `LSAPy` is now available on `conda-forge <https://anaconda.org/conda-forge/lsapy>`_ (issue `#68 <https://github.com/baptistehamon/lsapy/issues/68>`_, PR `#74 <https://github.com/baptistehamon/lsapy/pull/74>`_).

New features
^^^^^^^^^^^^
* A new ``boolean`` suitability function has been added (issue `#85 <https://github.com/baptistehamon/lsapy/issues/85>`_, PR `#86 <https://github.com/baptistehamon/lsapy/pull/86>`_).
* The ``name`` and ``indicator`` arguments of ``SuitabilityCriteria`` can now be optional (issue `#84 <https://github.com/baptistehamon/lsapy/issues/84>`_, PR `#87 <https://github.com/baptistehamon/lsapy/issues/87>`_).
* A ``setter`` has been added to ``SuitabilityCriteria`` attributes (issue `#84 <https://github.com/baptistehamon/lsapy/issues/84>`_, PR `#87 <https://github.com/baptistehamon/lsapy/issues/87>`_).
* The ``lsapy.standardize`` module has been added with categorical and membership standardization functions (issue `#89 <https://github.com/baptistehamon/lsapy/issues/89>`_, PR `#91 <https://github.com/baptistehamon/lsapy/issues/91>`_).
* The ``SuitabilityFunction`` is not longer required to define the standardization function in ``SuitabilityCriteria`` (issue `#89 <https://github.com/baptistehamon/lsapy/issues/89>`_, PR `#91 <https://github.com/baptistehamon/lsapy/issues/91>`_).
* ``SuitabilityCriteria.name`` is now a property of the class (PR `#97 <https://github.com/baptistehamon/lsapy/issues/97>`_).
* The ``LandSuitabilityAnalysis`` properties have been properly defined (PR `#98 <https://github.com/baptistehamon/lsapy/issues/98>`_).
* The API Reference documentation has been updated and improved (issue `#96 <https://github.com/baptistehamon/lsapy/issues/96>`_, PR `#99 <https://github.com/baptistehamon/lsapy/issues/99>`_).

Breaking changes
^^^^^^^^^^^^^^^^
* The ``lsapy.statistics`` module has been renamed to ``lsapy.stats`` (PR `#71 <https://github.com/baptistehamon/lsapy/pull/71>`_).
* The ``lsapy.core.aggregation`` has been moved and renamed to ``lsapy.aggregate`` (PR `#71 <https://github.com/baptistehamon/lsapy/pull/71>`_).
* The ``statistical_summary`` function has been renamed to ``stats_summary`` (PR `#82 <https://github.com/baptistehamon/lsapy/pull/82>`_).
* The ``spatial_statistical_summary`` function has been renamed to ``spatial_stats_summary`` (PR `#82 <https://github.com/baptistehamon/lsapy/pull/82>`_).
* The ``lsapy.functions`` module as well as ``SuitabilityFunction`` have been marked as deprecated and will be removed in a future release (issue `#89 <https://github.com/baptistehamon/lsapy/issues/89>`_, PR `#91 <https://github.com/baptistehamon/lsapy/issues/91>`_).

Internal changes
^^^^^^^^^^^^^^^^
* The name and year in the license file have been updated (PR `#76 <https://github.com/baptistehamon/lsapy/pull/76>`_).
* The logo has been updated (PR `#80 <https://github.com/baptistehamon/lsapy/pull/80>`_).
* Tests have been added for the ``lsapy.stats`` module functions (PR `#82 <https://github.com/baptistehamon/lsapy/pull/82>`_).
* The README file has been updated integrating new standardization function workflow (issue `#92 <https://github.com/baptistehamon/lsapy/issues/92>`_, PR `#93 <https://github.com/baptistehamon/lsapy/issues/93>`_).

Bug fixes
^^^^^^^^^
* Fix small issue of ``SuitabilityCriteria`` string representation (PR `#73 <https://github.com/baptistehamon/lsapy/pull/73>`_).

v0.2.0 (2025-08-20)
-------------------
Contributor to this version: Baptiste Hamon (@baptistehamon).

New features
^^^^^^^^^^^^
* Functions' alternative names can now be used in ``SuitabilityFunction`` (PR `#43 <https://github.com/baptistehamon/lsapy/pull/43>`_).
* The documentation of membership functions has been improved (PR `#43 <https://github.com/baptistehamon/lsapy/pull/43>`_).
* `LSAPy` officially supports Python>=3.10 and is OS independent (PR `#46 <https://github.com/baptistehamon/lsapy/pull/46>`_).
* ``repr`` methods of ``SuitabilityFunction``, ``SuitabilityCriteria`` and ``LandSuitabilityAnalysis`` have been modified to provide more user-friendly information (issue `#55 <https://github.com/baptistehamon/lsapy/issues/55>`_, PR `#59 <https://github.com/baptistehamon/lsapy/pull/59>`_).
* A setter has been added to the ``attrs`` method of ``SuitabilityCriteria`` and ``LandSuitabilityAnalysis`` (issue `#55 <https://github.com/baptistehamon/lsapy/issues/55>`_, PR `#59 <https://github.com/baptistehamon/lsapy/pull/59>`_).
* The ``open_data`` function has been added to the ``lsapy.utils`` module to load data from the LSAPy data repository (issue `#60 <https://github.com/baptistehamon/lsapy/issues/60>`_, PR `#62 <https://github.com/baptistehamon/lsapy/pull/62>`_).
* A ``median`` aggregation method has been added (PR `#63 <https://github.com/baptistehamon/lsapy/pull/63>`_).

Breaking changes
^^^^^^^^^^^^^^^^
* The deprecated ``SuitabilityFunction.map`` method has been removed (PR `#44 <https://github.com/baptistehamon/lsapy/pull/44>`_).
* ``short_name``, ``long_name``, ``description`` and ``comment`` attributes of ``SuitabilityCriteria`` and ``LandSuitabilityAnalysis`` have been removed and are now stored in the ``attrs`` attribute (issue `#55 <https://github.com/baptistehamon/lsapy/issues/55>`_, PR `#59 <https://github.com/baptistehamon/lsapy/pull/59>`_).
* ``load_climate_data`` and ``load_soil_data`` functions have been removed (issue `#60 <https://github.com/baptistehamon/lsapy/issues/60>`_, PR `#62 <https://github.com/baptistehamon/lsapy/pull/62>`_).
* Some names of aggregation methods have been changed (PR `#63 <https://github.com/baptistehamon/lsapy/pull/63>`_):
    * ``weighted_mean`` is now ``wmean``
    * ``geomean`` is now ``gmean``
    * ``weighted_geomean`` is now ``wgmean``
    * ``limiting_factor`` is now ``limfactor``
* The ``vars_weighted_mean``, ``vars_mean``, ``vars_geomean``, ``vars_weighted_geomean`` and ``limiting_factor`` aggregation methods have been removed (PR `#63 <https://github.com/baptistehamon/lsapy/pull/63>`_).

Internal changes
^^^^^^^^^^^^^^^^
* Tests have been added for currently implemented `LSAPy` functionalities (issue `#7 <https://github.com/baptistehamon/lsapy/issues/7>`_, PR `#46 <https://github.com/baptistehamon/lsapy/pull/46>`_).
    * ``pytest`` is used as the testing framework to run all unit tests, doctests and test notebooks.
    * ``nox`` has been set up and is used to run tests in CI workflows.
* A CI GitHub Actions workflow has been added (issue `#8 <https://github.com/baptistehamon/lsapy/issues/8>`_, PR `#46 <https://github.com/baptistehamon/lsapy/pull/46>`_)
    * The CI runs tests on Python 3.10 to 3.13, on Ubuntu, macOS and Windows.
    * The coverage, doctests and notebook tests are run on Ubuntu under Python 3.12.
* New pre-commit hooks have been added and the package pyproject has been updated (PR `#58 <https://github.com/baptistehamon/lsapy/pull/58>`_)
    * New hooks: ``yamllint``, ``vulture``, ``nbstripout``, ``pygrep-hooks``, ``mdformat``, ``blackdoc``, ``formatbibtex``, ``gitleaks`` and ``meta``.
    * Update dependencies: remove unused ``Shapely`` and add dependencies for new hooks.
    * Add deptry config to track dependencies.
    * Update package metadata: keywords, classifiers (python versions) and project URLs.
    * Update package sdist files.
* The ``lsapy.core.formatting`` module has been added and contains ``repr`` formatting functions (issue `#55 <https://github.com/baptistehamon/lsapy/issues/55>`_, PR `#59 <https://github.com/baptistehamon/lsapy/pull/59>`_).
* `LSAPy` sample data management has been improved (issue `#60 <https://github.com/baptistehamon/lsapy/issues/60>`_, PR `#62 <https://github.com/baptistehamon/lsapy/pull/62>`_):
    * `LSAPy` now uses `pooch` to fetch sample data.
    * Old data files have been removed from the data folder, and the new climate data file has been added.
    * A registry file has been added to store sample data file names, hashes and URLs.
* Aggregation functions have been moved to the `lsapy.core.aggregation` module (PR `#63 <https://github.com/baptistehamon/lsapy/pull/63>`_).
* A relaxed configuration of `mypy` has been added to the project (PR `#66 <https://github.com/baptistehamon/lsapy/pull/66>`_).

Bug fixes
^^^^^^^^^
* Fix issues with representations of ``SuitabilityFunction`` when no parameters are provided (issue `#61 <https://github.com/baptistehamon/lsapy/issues/61>`_, PR `#65 <https://github.com/baptistehamon/lsapy/pull/65>`_).
* Add `**kwargs` to `SuitabilityCriteria.compute` and `LandSuitabilityAnalysis.run` to allow handling Dask arrays (issue `#64 <https://github.com/baptistehamon/lsapy/issues/64>`_, PR `#65 <https://github.com/baptistehamon/lsapy/pull/65>`_).
* The codebase has been modified to improve typing and fix mypy errors (issue `#35 <https://github.com/baptistehamon/lsapy/issues/35>`_, PR `#66 <https://github.com/baptistehamon/lsapy/pull/66>`_).

v0.1.1 (2025-07-26)
-------------------
Contributor to this version: Baptiste Hamon (@baptistehamon).

Internal changes
^^^^^^^^^^^^^^^^
* The documentation has been updated to reflect the changes in ``LandSuitabilityAnalysis`` workflow (issue `#41 <https://github.com/baptistehamon/lsapy/issues/41>`_, PR `#42 <https://github.com/baptistehamon/lsapy/pull/42>`_).

v0.1.0 (2025-07-25)
-------------------
Contributor to this version: Baptiste Hamon (@baptistehamon).

New features
^^^^^^^^^^^^
* New function ``membership.fit_membership`` implemented as replacement of the deprecated ``MembershipSuitFunction.fit`` method (issue `#29 <https://github.com/baptistehamon/lsapy/issues/29>`_, PR `#30 <https://github.com/baptistehamon/lsapy/pull/30>`_).

Breaking changes
^^^^^^^^^^^^^^^^
* ``MembershipSuitFunction`` and ``DiscreteSuitFunction`` have been removed (issue `#29 <https://github.com/baptistehamon/lsapy/issues/29>`_, PR `#30 <https://github.com/baptistehamon/lsapy/pull/30>`_).
* Changes in ``SuitabilityFunction`` (issue `#15 <https://github.com/baptistehamon/lsapy/issues/15>`_, PR `#23 <https://github.com/baptistehamon/lsapy/pull/23>`_ & PR `#30 <https://github.com/baptistehamon/lsapy/pull/30>`_):
    * ``func_method`` and ``func_params`` have been renamed to ``name`` and ``params`` respectively.
    * ``map`` has been deprecated because of its redundancy with the ``__call__`` method. Changes will be permanent in LSAPy v0.1.0. Call the function directly instead.
* ``LandSuitability`` has been renamed to ``LandSuitabilityAnalysis``. (issue `#15 <https://github.com/baptistehamon/lsapy/issues/15>`_, PR `#26 <https://github.com/baptistehamon/lsapy/pull/26>`_)
    * ``name`` has been renamed to ``land_use``.
    * ``compute_criteria_suitability``, ``compute_category_suitability``, and ``compute_suitability`` methods have been removed and the method ``run`` has been implemented as replacement (issue `#15 <https://github.com/baptistehamon/lsapy/issues/15>`_, PR `#38 <https://github.com/baptistehamon/lsapy/pull/38>`_)
    * ``mask``, ``statistics`` and ``spatial_statistics`` methods have been removed.

Internal changes
^^^^^^^^^^^^^^^^
* Templates for requesting new features, asking question and submitting PR have been added (issue `#11 <https://github.com/baptistehamon/lsapy/issues/11>`_, PR `#12 <https://github.com/baptistehamon/lsapy/pull/12>`_).
* The README has been updated to make links permanent and to add a docs badge (PR `#13 <https://github.com/baptistehamon/lsapy/pull/13>`_).
* A configuration file for Zenodo integration has been added to the repository (PR `#14 <https://github.com/baptistehamon/lsapy/pull/14>`_).
* `Pre-commit` has been setup and `ruff`, `codespell` and `numpydoc` hooks have been added (issue `#8 <https://github.com/baptistehamon/lsapy/issues/8>`_, PR `#18 <https://github.com/baptistehamon/lsapy/pull/18>`_/PR `#19 <https://github.com/baptistehamon/lsapy/pull/19>`_).
* The autoupdate schedule of `pre-commit` has been set to weekly (PR `#21 <https://github.com/baptistehamon/lsapy/pull/21>`_)
* The unused ``introduction.ipynb`` notebook has been removed (issue `#15 <https://github.com/baptistehamon/lsapy/issues/15>`_, PR `#20 <https://github.com/baptistehamon/lsapy/pull/20>`_).
* The structure around ``SuitabilityFunction`` (PR `#30 <https://github.com/baptistehamon/lsapy/pull/30>`_):
    * The ``SuitabilityFunction`` has been moved to LSAPy ``function._suitability`` module.
    * The membership functions have been moved to the ``function.membership`` module.
    * The discrete function has been moved to the ``function._discrete`` module.
    * The ``equation`` decorator has been rename to ``declare_equation`` and moved to the ``core.function`` module.
    * The ``get_function_from_name`` function has been moved to the ``core.function`` module.
* Changes on ``SuitabilityCriteria`` (issue `#15 <https://github.com/baptistehamon/lsapy/issues/15>`_, PR `#31 <https://github.com/baptistehamon/lsapy/pull/31>`_):
    * It now has a ``comment`` and ``is_computed`` attributes.
    * ``func`` parameter is now optional, useful when the criteria is already computed.
* LSAPy logo has been added: README and documentation have been updated to use it (PR `#27 <https://github.com/baptistehamon/lsapy/pull/27>`_)

v0.1.0-dev2 (2025-05-25)
------------------------
Contributor to this version: Baptiste Hamon (@baptistehamon).

Internal changes
^^^^^^^^^^^^^^^^
* Major changes for documentation (issue `#2 <https://github.com/baptistehamon/lsapy/issues/2>`_, PR `#9 <https://github.com/baptistehamon/lsapy/pull/9>`_):
    * All public objects are now documented using the `NumPy-style <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
    * *introduction.ipynb* has been slip into three different ones: *criteria.ipynb*, *function.ipynb*, and *lsa.ipynb*.
    * The top-level documentation has been updated/created:
        * The format of README and CHANGELOG files is now reStructuredText (RST).
        * A proper README has been created.
        * A CODE_OF_CONDUCT file adopting the `Contributor Covenant <https://www.contributor-covenant.org/>`_ code of conduct has been added.
        * A CONTRIBUTING.md providing guidelines on how to contribute to the project has been added.
    * FT20250 and UC logos used in the documentation have been added to the repository.
    * The documentation building using `Sphinx <https://www.sphinx-doc.org/en/master/>`_ has been setup:
        * The documentation uses the `PyData theme <https://pydata-sphinx-theme.readthedocs.io/en/stable/>`_.
        * A User-facing documentation is now available and has been published on `Read the Docs <https://readthedocs.org/>`_.
    * The project dependencies have been updated and made consistent across *pyproject.toml* and *environments.yml* files.

v0.1.0-dev1 (2025-05-16)
------------------------
Contributor to this version: Baptiste Hamon (@baptistehamon).

New features
^^^^^^^^^^^^
* Add ruff configuration to the project.

Bug fixes
^^^^^^^^^
* Fix the fit of ``MembershipSuitFunction`` returning the wrong best fit (issue `#1 <https://github.com/baptistehamon/lsapy/issues/1>`_, PR `#5 <https://github.com/baptistehamon/lsapy/pull/5>`_)

v0.1.0-dev0 (2025-03-12)
------------------------
Contributor to this version: Baptiste Hamon (@baptistehamon).

* First release on PyPI.

New features
^^^^^^^^^^^^
* ``SuitabilityFunction`` to define the function used for suitability computation.
* ``SuitabilityCriteria`` to define criteria to consider in the LSA
* ``LandSuitability`` to conduct LSA.
