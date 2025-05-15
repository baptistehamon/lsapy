=========
Changelog
=========

v0.1.0-dev1 (2025-05-16)
------------------------
Contributor to this version: Baptiste Hamon (@baptistehamon).

New features
^^^^^^^^^^^^
* Add ruff configuration to the project.

Bug fixes
^^^^^^^^^
* Fix the fit of `MembershipSuitFunction` returning the wrong best fit (issue `#1 <https://github.com/baptistehamon/lsapy/issues/1>`_, PR `#5 <https://github.com/baptistehamon/lsapy/pull/5>`_)

v0.1.0-dev0 (2025-03-12)
------------------------
Contributor to this version: Baptiste Hamon (@baptistehamon).
* First release on PyPI.

New features
^^^^^^^^^^^^
* ``SuitabilityFunction`` to define the function used for suitability computation.
* ``SuitabilityCriteria`` to define criteria to consider in the LSA
* ``LandSuitability`` to conduct LSA.