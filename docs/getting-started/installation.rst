.. _installation:

Installation
============

Stable release
^^^^^^^^^^^^^^

The stable version of LSAPy can be installed from `PyPI`_ using `pip`:

.. code-block:: shell

   python -m pip install lsapy

or from `conda-forge`_ using `conda`:

.. code-block:: shell

   conda install -c conda-forge lsapy

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

LSAPy has a few optional dependencies that are not required for the core functionality of the package, but may be needed for certain features or workflows.
To open and use the sample datasets provided by LSAPy through the ``lsapy.tutorial.open_dataset`` function, `pooch`_ is required to download the datasets, and
a netCDF library is required to read the them. We recommend installing one of the following netCDF backend libraries:

* `netCDF4`_
* `h5netcdf`_

For parallel computation on chunked arrays, `dask`_ is required.

With `pip`, these optional dependencies can be installed using the ``tutorial`` and ``parallel`` extras as follows:

.. code-block:: shell

   python -m pip install "lsapy[tutorial]" # install optional dependencies for tutorial/sample data
   python -m pip install "lsapy[parallel]" # install dask for parallel computation
   python -m pip install "lsapy[complete]" # install all the above

Development version
^^^^^^^^^^^^^^^^^^^

The latest development version of LSAPy can be installed directly from the GitHub repository using `pip`:

.. code-block:: shell

   python -m pip install git+https://github.com/baptistehamon/lsapy

Or if you want to contribute to the development of LSAPy, you can clone the repository and install it in editable mode:

.. code-block:: shell

   git clone git@github.com:baptistehamon/lsapy.git
   cd lsapy
   python -m pip install -e .[dev]

You can find more information about contributing to LSAPy in the `Contribution`_ section of the documentation.

.. _dask: https://docs.dask.org/en/stable/
.. _netCDF4: https://github.com/Unidata/netcdf4-python
.. _h5netcdf: https://h5netcdf.org/
.. _pooch : https://www.fatiando.org/pooch/latest/index.html
.. _Contribution: https://lsapy.readthedocs.io/en/latest/community/contributing.html
.. _PyPI: https://pypi.org/project/laspy/
.. _conda-forge: https://conda-forge.org/
