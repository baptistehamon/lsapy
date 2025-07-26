=========
Changelog
=========

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
* The README has been updated to make links permanent and to add a docs badge (PR `#13 <https://github.com/baptistehamon/lsapy/pull/13>`_, PR `#15 <https://github.com/baptistehamon/lsapy/pull/15>`).
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
