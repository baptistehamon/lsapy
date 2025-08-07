=====================
Contributing to LSAPy
=====================

No matter the scope of your contribution, help is always welcome!

The `issue tracker`_ is the best place to start if you want to contribute to `LSAPy`. You can get involved in different ways:

* **Submitting feedbacks**: We welcome any feedback on the library, whether it's about the API, the documentation, or anything else.
* **Suggesting new features**: If you have an idea for a new feature or improvement. Explain your idea in details and provide any thoughts you have on how it could be implemented.
* **Reporting bugs**: If you find a bug, please report it making sure to include as much information as possible to help us reproduce the bug (e.g., operating system, local setup details, steps to reproduce it...).

The `issue tracker`_ is also the place to ask questions about the library. The maintainers and the community will be happy to help you.

Contribution Workflow
---------------------
Below is the steps to follow for contributing to the project.

#. Fork the `LSAPy` repository on GitHub to your own account.

#. Clone your forked repository to your local machine.

    .. code-block:: shell

        git clone git@github.com:<your_username>/lsapy.git
        cd lsapy/

#. Create a virtual environment (``conda`` is recommended) and install the dependencies.

    .. code-block:: shell

        conda env create -f environment.yml
        conda activate lsapy
        python -m pip install -e .

#. Create a new branch for your changes.

    .. code-block:: shell

        git checkout -b name-of-your-bugfix-or-feature

    You can now make changes.

#. Before committing your changes, we ask that you install ``pre-commit`` in your development environment and run git hooks to ensure that your code adheres to the project's coding standards:

    .. code-block:: shell

        # To install the necessary pre-commit hooks:
        pre-commit install
        # To run pre-commit hooks manually:
        pre-commit run --all-files

    This will automatically format your code and fix any linting issues.
    Instead of ``pre-commit``, you can check your changes using `nox`:

        .. code-block:: shell

            nox -s lint  # to run all linters and formatters

    Or check individual hooks manually:

        .. code-block:: shell

            ruff check --fix --show-fixes src/lsapy/
            ruff format src/lsapy/
            codespell src/lsapy/ tests/ docs/
            numpydoc src/lsapy/**/*.py

#. The next step is to ensure your changes do not introduce any breaking issues using ``pytest``:

    .. code-block:: shell

        pytest --nbval docs/notebooks/ # for only notebooks
        pytest --doctest-modules src/lsapy/ # for only doctests
        pytest # for all unit tests, excluding doctests and notebooks.

    Alternatively, you can run all tests using `nox`:

        .. code-block:: shell

            nox -s tests  # for all unit tests, excluding doctests and notebooks
            nox -s notebooks doctests  # for notebooks and doctests
            nox # to run all unit tests, doctests, and notebooks

#. Finally, you need to make sure that the documentation will build correctly on ReadTheDocs. You can do this as follows:

    .. code-block:: shell

        make -C docs html
        # or
        nox -s docs

#. Commit your changes and push your branch.

    .. code-block:: shell

        git add *
        git commit -m "Short description of your changes"
        git push origin name-of-your-bugfix-or-feature

#. Create a pull request on GitHub.

    Before creating a pull request, we first ask you to open an issue in the `GitHub repository`_. Describe the bug you would
    like to fix or the feature you would like to add. Link the issue to your pull request.

.. note::

    **Longer Term Commitment ?**

    While the project is still in its early stages, a bigger maintainers team may be required in the future if the project
    grows. If you like the project and are interested in joining the maintainers team, please reach out to us. We will be happy to
    discuss it with you.

Reminder for maintainers
------------------------

This section provides some useful information for maintainers.

Versioning
^^^^^^^^^^

The project follows `Semantic Versioning`_ scheme:

.. code-block:: shell

    major.minor.patch-releaseX
      |     |     |      |   |
      |     |     |      |   +--- Build number (e.g., 1, 2, 3...)
      |     |     |      +------- Degree of production (e.g., dev, alpha, beta)
      |     |     +-------------- Patch release (e.g., bug fixes)
      |     +-------------------- Minor release (e.g., new features)
      +-------------------------- Major release (e.g., breaking changes)


Packaging and Deployment
^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    This section comes largely from the `xclim Contributing guidelines`_. Small edits have been made to match `LSAPy` project.

This section serves as a reminder for the maintainers on how to prepare the library for a tagged version and how to deploy packages to TestPyPI and PyPI.

When a new version has been minted (features have been successfully integrated test coverage and stability is adequate), maintainers should update the pip-installable package (wheel and source release) on PyPI.

From a new branch (e.g. ``prepare-v123``), open a Pull Request and make sure all your changes to support a new version are committed (**update the entry for newest version in CHANGELOG.rst**), then run:

.. code-block:: shell

    bump-my-version bump <option>  # possible options: major / minor / patch / release / build

These commands will increment the version and create a commit with an autogenerated message.

For PyPI releases/stable versions, ensure that the last version bumping command run is ``$ bump-my-version bump release`` to remove the ``-dev``. These changes can now be merged to the ``prepare-v123`` branch:

.. code-block:: shell

    git push origin prepare-v123

With this performed, we can tag a version that will act as the GitHub-provided stable source archive. **Be sure to only tag from the `main` branch when all changes from PRs have been merged!** The commands needed are:

.. code-block:: shell

    git tag v1.2.3
    git push --tags

.. note::

    All tags pushed to GitHub will trigger a build and publish a package to TestPyPI by default.

The Automated Approach
~~~~~~~~~~~~~~~~~~~~~~

The simplest way to package `LSAPy` is to "publish" a version on GitHub. GitHub CI Actions are presently configured to build the library and publish the packages on PyPI automatically.

.. warning::

    A published version on PyPI can never be overwritten. Be sure to verify that the package published at https://test.pypi.org/project/lsapy/ matches expectations before publishing a version on GitHub.

The Manual Approach
~~~~~~~~~~~~~~~~~~~

The manual approach to library packaging for general support (pip wheels) requires that the `flit`_ library is installed.

From the command line on your Linux distribution, simply run the following from the clone's main dev branch:

.. code-block:: shell

    # To build the packages (sources and wheel)
    flit build

    # To upload to PyPI
    flit publish

The new version based off of the version checked out will now be available via ``pip`` (``$ pip install lsapy``).

Credits
-------

This document is inspired by the `xclim Contributing guidelines`_.

.. _`GitHub Repository`: https://github.com/baptistehamon/lsapy
.. _`issue tracker`: https://github.com/baptistehamon/lsapy/issues
.. _`xclim Contributing guidelines`: https://github.com/Ouranosinc/xclim/blob/main/CONTRIBUTING.rst
.. _`Semantic Versioning`: https://semver.org/
.. _`flit`: https://flit.pypa.io/en/stable/index.html
