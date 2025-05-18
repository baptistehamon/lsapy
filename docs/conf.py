# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

from pybtex.plugin import register_plugin  # noqa
from pybtex.style.formatting.alpha import Style as AlphaStyle  # noqa
from pybtex.style.labels import BaseLabelStyle  # noqa

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
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "sphinx_mdinclude",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# Bibliography stuff (stolen from xclim package)
# a simple label style which uses the bibtex keys for labels
class XCLabelStyle(BaseLabelStyle):
    def format_labels(self, sorted_entries):
        for entry in sorted_entries:
            yield entry.key


class XCStyle(AlphaStyle):
    default_label_style = XCLabelStyle


register_plugin("pybtex.style.formatting", "xcstyle", XCStyle)
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "xcstyle"
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