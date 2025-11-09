.. image:: https://raw.githubusercontent.com/baptistehamon/lsapy/main/docs/logos/lsapy/LSAPy_Logo_FullColour.png
    :class: dark-light
    :align: center
    :target: https://github.com/baptistehamon/lsapy
    :width: 400px
    :alt: LSAPy Logo

LSAPy: Land Suitability Analysis in Python
==========================================
|pypi| |conda| |python-versions| |status| |testing| |coverage| |ruff| |pre-commit| |docs| |question| |zenodo|

`LSAPy` stands for Land Suitability Analysis (LSA) in Python. Its objective is to make conducting
LSA in Python easier and more accessible to users. It provides a set of objects built around
`xarray`_ and operating together, making LSA's workflow straight forward and easy to understand.

.. _`xarray`: https://xarray.pydata.org/en/stable/

Quick Start
-----------
To install `LSAPy`, you can use `pip`:

.. code-block:: shell

    pip install lsapy

or `conda`:

.. code-block:: shell

    conda install -c conda-forge lsapy

You can now perform your LSA:

.. code-block:: python

    # import modules
    from lsapy import LandSuitabilityAnalysis, SuitabilityCriteria

    # define your criteria
    criteria = {
        "crit1": SuitabilityCriteria(
            name="criteria1",
            indicator=indicator1,  # xarray object
            func="function_name",
            fparams={"param1": value1, "param2": value2},
        ),
        "crit2": SuitabilityCriteria(
            name="criteria2",
            indicator=indicator2,  # xarray object
            func="another_function_name",
            fparams={"param1": value1, "param2": value2},
        ),
        # add all necessary criteria
    }

    # define your land suitability
    lsa = LandSuitabilityAnalysis(
        land_use="land_use_name",
        criteria=criteria,
    )

    # run your analysis
    lsa.run(params)

More detailed tutorials and examples can be found in the `User Guide`_.

.. _`User Guide`: https://lsapy.readthedocs.io/en/latest/notebooks/index.html


Contributing
------------

`LSAPy` is an open-source project and we welcome contributions from the community. If you are interested in contributing, please
refer to the `Contribution`_ section for guidelines on how to get started helping us improve the library.

.. _`Contribution`: https://lsapy.readthedocs.io/en/latest/community/contributing.html

Credits
-------

The development of `LSAPy` started as part of a PhD, funded by the `Food Transition 2050`_  Joint Postgraduate School and hosted
by the `University of Canterbury`_ in New Zealand.

|FT2050| |UC-white| |UC-black|

The Python package has been created following the `pyOpenSci Guidebook`_.

.. _`Food Transition 2050`: https://www.foodtransitions2050.ac.nz/
.. _`University of Canterbury`: https://www.canterbury.ac.nz/
.. _`pyOpenSci Guidebook`: https://www.pyopensci.org/python-package-guide/

.. |logo| image:: https://raw.githubusercontent.com/baptistehamon/lsapy/main/docs/logos/lsapy/LSAPy_Logo_FullColour.png
    :class: dark-light
    :target: https://github.com/baptistehamon/lsapy
    :width: 400px
    :alt: LSAPy Logo

.. |FT2050| image:: https://raw.githubusercontent.com/baptistehamon/lsapy/main/docs/logos/FT2050-full_colour.png
    :class: dark-light
    :target: https://www.foodtransitions2050.ac.nz/
    :width: 200px
    :alt: Food Transition 2050 Logo

.. |UC-white| image:: https://raw.githubusercontent.com/baptistehamon/lsapy/main/docs/logos/UCWhite.png
    :class: only-dark
    :target: https://www.canterbury.ac.nz/
    :width: 100px
    :alt: University of Canterbury Logo

.. |UC-black| image:: https://raw.githubusercontent.com/baptistehamon/lsapy/main/docs/logos/UCBlack.png
    :class: only-light
    :target: https://www.canterbury.ac.nz/
    :width: 100px
    :alt: University of Canterbury Logo

.. |pypi| image:: https://img.shields.io/pypi/v/lsapy.svg
    :target: https://pypi.python.org/pypi/lsapy
    :alt: Python Package Index Build

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/lsapy
    :target: https://anaconda.org/conda-forge/lsapy
    :alt: Conda Version

.. |status| image:: https://www.repostatus.org/badges/latest/active.svg
    :target: https://www.repostatus.org/#active
    :alt: Project Status: Active - The project has reached a stable, usable state and is being actively developed.

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.15015111.svg
    :target: https://doi.org/10.5281/zenodo.15015111
    :alt: Zenodo DOI

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. |docs| image:: https://readthedocs.org/projects/lsapy/badge/
    :target: https://lsapy.readthedocs.io/en/latest/
    :alt: Documentation Status

.. |question| image:: https://img.shields.io/badge/Question_%3F-blue?style=social&logo=github
    :target: https://github.com/baptistehamon/lsapy/labels/question
    :alt: Link to question issues

.. |pre-commit| image:: https://results.pre-commit.ci/badge/github/baptistehamon/lsapy/main.svg
   :target: https://results.pre-commit.ci/latest/github/baptistehamon/lsapy/main
   :alt: pre-commit.ci status

.. |testing| image:: https://github.com/baptistehamon/lsapy/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/baptistehamon/lsapy/actions/workflows/testing.yml
    :alt: Testing Status

.. |coverage| image:: https://coveralls.io/repos/github/baptistehamon/lsapy/badge.svg
    :target: https://coveralls.io/github/baptistehamon/lsapy
    :alt: Coverage Status (Coveralls)

.. |python-versions| image:: https://img.shields.io/pypi/pyversions/lsapy
        :target: https://pypi.python.org/pypi/lsapy
        :alt: Supported Python Versions
