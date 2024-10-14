"""
Miscellaneous Testing Utilities
===============================

...
"""

from pathlib import Path
import xarray as xr
from typing import Optional, Union, Any

from landsuitability.testing.utils import load_data
from landsuitability.core import LandSuitability
from landsuitability.criteria import SuitabilityCriteria
from landsuitability.criteria.indicators import CriteriaIndicator
from landsuitability.functions import SuitabilityFunction, MembershipSuitFunction, DiscreteSuitFunction

# Testing Utilities
import numpy as np
import matplotlib.pyplot as plt
from xclim.indicators.atmos import growing_degree_days, precip_accumulation


# import data
soil_data = load_data('soil')
clim_data = load_data('climate').interp_like(soil_data, 'nearest')
data = xr.merge([soil_data, clim_data])


# test suitabiliy function
SuitabilityFunction(func_method='logistic', func_params={'a': 1, 'b': 2})
MembershipSuitFunction.fit(x=[1500,1350,1250,1150,1000], plot=True)
f_suit = SuitabilityFunction(func_method='vetharaniam2022_eq5', func_params={'a': 0.8759, 'b': 1248.5})

x = np.linspace(500, 2000, 100)
y = f_suit(x)
plt.plot(x, y)
plt.show()


# test criteria indicator
ci = CriteriaIndicator(
    var_name = 'GDD',
    standard_name = 'Growing Degree Days',
    long_name = 'Growing Degree Days',
    units = 'degC',
    func = growing_degree_days,
    func_params = {'thresh': '10 degC'},
    data_arg_name = 'ds'
)
gdd = ci.compute(clim_data)


# test suitability criteria
sc = SuitabilityCriteria(
    name = 'Temp_requirements',
    long_name = 'Temperature Requirements',
    description = 'Temperature requirements for crop growth',
    indicator=ci,
    func = SuitabilityFunction(func_method='vetharaniam2022_eq5', func_params={'a': -1.41, 'b': 801})
)
gdd_suit = sc.compute(clim_data)


fig, ax = plt.subplots(1, 2, figsize=(12, 4))
gdd.isel(time=0).plot(ax=ax[0])
gdd_suit.isel(time=0).plot(ax=ax[1])
plt.show()


# test land suitability criteria
ls = LandSuitability(
    name = 'land_suitability',
    data = data,
    criteria = {
        'water_req' : SuitabilityCriteria(
            name = 'Annual Rainfall Requirement',
            weight=0.5,
            category='Climate',
            indicator=CriteriaIndicator(
                var_name='pr',
                standard_name='Precipitation',
                long_name='Annual Precipitation',
                units='mm',
                func=precip_accumulation,
                func_params={'freq': 'YS'},
                data_arg_name='ds'
            ),
            func = SuitabilityFunction(func_method='vetharaniam2022_eq5', func_params={'a': 0.876, 'b': 1248})
        ),
        'temp_req' : SuitabilityCriteria(
            name = 'Temperature Requirements',
            weight=1,
            category='Climate',
            indicator = CriteriaIndicator(
                var_name='GDD',
                standard_name='Growing Degree Days',
                long_name='Growing Degree Days',
                units='degC',
                func=growing_degree_days,
                func_params={'thresh': '10 degC'},
                data_arg_name='ds'
            ),
            func = SuitabilityFunction(func_method='vetharaniam2022_eq5', func_params={'a': -1.41, 'b': 801})
        ),
        'prd' : SuitabilityCriteria(
            name = 'Soil Depth',
            weight=0.5,
            category='TerrainSoil',
            indicator=CriteriaIndicator(
                var_name='soil_depth',
                standard_name='Soil Depth',
                long_name='Soil Depth',
                units='cm',
                func=lambda x: x['PRD'],
                func_params={},
                data_arg_name=None
            ),
            func = SuitabilityFunction(func_method='vetharaniam2022_eq5', func_params={'a': -11.98, 'b': 0.459})
        )
    }
)


# test criteria suitability
ls_out = ls.compute_criteria_suitability(inplace=True)

fig, ax = plt.subplots(1, 3, figsize=(18, 4))
ls_out['water_req'].isel(time=0).plot(ax=ax[0])
ls_out['temp_req'].isel(time=0).plot(ax=ax[1])
ls_out['prd'].plot(ax=ax[2])
plt.show()


# test category suitability
ls_out = ls.compute_category_suitability(method='weighted_geomean', inplace=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ls_out['Climate'].isel(time=0).plot(ax=ax[0])
ls_out['Terrain/Soil'].plot(ax=ax[1])
plt.show()


# test land suitability
ls_out = ls.compute_suitability(method='weighted_mean', keep_category=True, inplace=False)

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ls_out['Climate'].isel(time=0).plot(ax=ax[0,0])
ls_out['Terrain/Soil'].plot(ax=ax[0,1])
ls_out['suitability'].isel(time=0).plot(ax=ax[1,0])
ax[1,1].axis('off')
plt.show()

import pandas as pd

