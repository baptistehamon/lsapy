===========================================
LSAPy: Land Suitability Analysis in Python
===========================================

|pypi| |zenodo| |ruff|

`LSAPy` stand for Land Suitability Analysis (LSA) in Python. Its objective is to make conducting
LSA in Python easier and more accessible to users. It provides a set of objects built around
`xarray`_ and operating together, making LSA's workflow straight forward and easy to understand.

.. _`xarray`: https://xarray.pydata.org/en/stable/

Quick Start
-------------
To install `LSAPy`, you can use `pip`:

.. code-block:: shell

    pip install lsapy


You can now perform your LSA:

.. code-block:: python

    # import modules
    from lsapy import SuitabilityFunction, SuitabilityCriteria, LandSuitability

    # define your criteria
    criteria = {
        'crit1': SuitabilityCriteria(
            name='criteria1',
            indicator='indicator1',
            suitability_function= SuitabilityFunction("relevant-function")
        ),
        'crit2': SuitabilityCriteria(
            name='criteria2',
            indicator='indicator2',
            suitability_function= SuitabilityFunction("relevant-function")
        )
        # add all necessary criteria
    }

    # define your land suitability
    ls = LandSuitability(
        name= 'name_of_your_lsa',
        criteria= criteria,
    )

    # run your analysis
    ls.compute_suitability(params)

More detailed tutorials and examples can be found in the :doc:`notebooks/index`.

Contributing
------------

`LSAPy` is an open-source project and we welcome contributions from the community. If you are interested in contributing, please
refer to the :doc:`Contribution </community/contributing>` section for guidelines on how to get started helping us improve the library.

Credits
-------

The development of `LSAPy` started as part of a PhD, funded by the the `Food Transition 2050`_  Joint Postgraduate School and hosted
by the `University of Canterbury`_ in New Zealand.

|FT2050| |UC-black|

The Python package has been created following the `pyOpenSci Guidebook`_.

.. _`Food Transition 2050`: https://www.foodtransitions2050.ac.nz/
.. _`University of Canterbury`: https://www.canterbury.ac.nz/
.. _`pyOpenSci Guidebook`: https://www.pyopensci.org/python-package-guide/

.. |FT2050| image:: /logos/FT2050-full_colour.png
    :target: https://www.foodtransitions2050.ac.nz/
    :width: 200px
    :alt: Food Transition 2050 Logo 

.. |UC-black| image:: /logos/UCBlack.png
    :target: https://www.canterbury.ac.nz/
    :width: 100px
    :alt: University of Canterbury Logo
    

.. |pypi| image:: https://img.shields.io/pypi/v/lsapy.svg
    :target: https://pypi.python.org/pypi/lsapy
    :alt: Python Package Index Build

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.15015111.svg
    :target: https://doi.org/10.5281/zenodo.15015111
    :alt: Zenodo DOI

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff