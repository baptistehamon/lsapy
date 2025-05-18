# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../src/"))

import lsapy

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "lsapy"
copyright = "2025, Baptiste Hamon"
author = "Baptiste Hamon"

# The short X.Y version.
version = lsapy.__version__.split("-")[0]
# The full version, including alpha/beta/rc tags.
release = lsapy.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "sphinx_mdinclude",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = 'lsapy'
html_short_title = 'lsapy'

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# Pygments light and dark theme styles
# pygments_style = "codex" #TODO: check why this creates an error when building the docs
# pygments_dark_style = "lightbulb"