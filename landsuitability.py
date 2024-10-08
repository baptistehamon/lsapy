"""
Miscellaneous Testing Utilities
===============================

...
"""

from pathlib import Path

import xarray as xr

from landsuitability.testing.utils import load_data


class LandSuitability:
    
    def __init__(self) -> None:
        pass


class LandCriteria:
    
    def __init__(self) -> None:
        pass


class SuitabilityCriteria:
    
    def __init__(self) -> None:
        pass



from landsuitability.testing.utils import load_data
from landsuitability.functions import SuitabilityFunction, MembershipSuitFunction, DiscreteSuitFunction
import numpy as np
import matplotlib.pyplot as plt

SuitabilityFunction(func_name='logistic', func_params={'a': 1, 'b': 2})

MembershipSuitFunction.fit(x=[1500,1350,1250,1150,1000], plot=True)

f_suit = SuitabilityFunction(func_name='vetharaniam2022_eq5', func_params={'a': 0.8759, 'b': 1248.5})

x = np.linspace(500, 2000, 100)
y = f_suit(x)
plt.plot(x, y)
plt.show()