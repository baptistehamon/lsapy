"""
Suitability Functions Module.

Deprecated: This package is deprecated and will be removed in a future release.
Use `standardize` module instead.
"""

from __future__ import annotations

import warnings

from lsapy.functions._discrete import *
from lsapy.functions._suitability import SuitabilityFunction
from lsapy.functions.membership import *

warnings.warn(
    "The 'lsapy.functions' module is deprecated and will be removed in a future release. "
    "Use 'lsapy.standardize' module instead.",
    FutureWarning,
    stacklevel=2,
)
