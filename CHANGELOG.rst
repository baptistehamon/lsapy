=========
Changelog
=========

v0.1.0 (unreleased)
-------------------
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
    * The project dependencies have been updated and made consistent accross *pyproject.toml* and *environments.yml* files.

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