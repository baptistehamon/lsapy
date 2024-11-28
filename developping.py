"""
Development Script
===============================

...
"""

from pathlib import Path
import xarray as xr
from typing import Optional, Union, Any

from lsapy.testing.utils import load_data
from lsapy.core import LandSuitability
from lsapy.criteria import SuitabilityCriteria
from lsapy.criteria.indicators import CriteriaIndicator
from lsapy.functions import SuitabilityFunction, MembershipSuitFunction, discrete

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


sc = SuitabilityCriteria(
    name = 'Drainage Class',
    weight= 3,
    category= 'SoilTerrain',
    indicator = CriteriaIndicator(
        var_name= 'DRC',
        standard_name= 'Drainage Class',
        long_name= 'Drainage Class',
        units= '',
        func= lambda x: x['DRC'],
    ),
    func=SuitabilityFunction(func_method='discrete', func_params={'rules': {'1': 0, '2': 0.1, '3': 0.5, '4': 0.9, '5': 1}}),
)
drc = sc.compute(soil_data)

drc.plot()
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

from xclim.indicators.atmos._temperature import TempWithIndexing
from xclim import indices
from xclim.indicators import atmos


hot_spell_frequency_indexing = TempWithIndexing(
    title="Hot spell frequency_indx",
    identifier="hot_spell_frequency_indx",
    long_name="Number of hot periods of {window} day(s) or more, during which the temperature on a "
    "window of {window} day(s) is above {thresh}.",
    description="The {freq} number of hot periods of {window} day(s) or more, during which the temperature on a "
    "window of {window} day(s) is above {thresh}.",
    abstract="The frequency of hot periods of `N` days or more, during which the temperature "
    "over a given time window of days is above a given threshold.",
    units="",
    cell_methods="",
    compute=indices.hot_spell_frequency,
)


# test selection by doy
def _get_doys(_start, _end, _inclusive):
    if _start <= _end:
        _doys = np.arange(_start, _end + 1)
    else:
        _doys = np.concatenate((np.arange(_start, 367), np.arange(0, _end + 1)))
    if not _inclusive[0]:
        _doys = _doys[1:]
    if not _inclusive[1]:
        _doys = _doys[:-1]
    return _doys

_get_doys(355, 10, [True, True])

gss = atmos.first_day_tg_below(clim_data['tas'], thresh='5 degC', after_date='01-01') # use first day below 5 degC as proxy for growing season start
gse = atmos.first_day_tg_above(clim_data['tas'], thresh='5 degC', after_date='07-01') # use first day above 5 degC as proxy for growing season end

doy = clim_data['tas'].time.dt.dayofyear

cond = (gss) & (doy <= gse)

gsl = gse - gss

fig, ax = plt.subplots(1, 3, figsize=(18, 4))
gss.isel(time=6).plot(ax=ax[0])
gse.isel(time=6).plot(ax=ax[1])
gsl.isel(time=6).plot(ax=ax[2])
plt.show()


# test for aggregate_between_dates
from xclim.indices.generic import aggregate_between_dates, season
from xclim.indices import (
    growing_season_start, growing_season_end, effective_growing_degree_days,
    first_day_temperature_above, first_day_temperature_below
)
from xclim.core.units import convert_units_to

tas = clim_data['tas'].sel(time=slice('2000-01-01', '2002-12-31'))
tas = convert_units_to(tas, 'degC')
tasmin = clim_data['tasmin'].sel(time=slice('2000-01-01', '2002-12-31'))
tasmin = convert_units_to(tasmin, 'degC')
tasmax = clim_data['tasmax'].sel(time=slice('2000-01-01', '2002-12-31'))
tasmax = convert_units_to(tasmax, 'degC')

# effective growing degree days
egdd = effective_growing_degree_days(tasmax, tasmin, thresh='5 degC', freq='YS', after_date='07-01')
egdd.isel(time=2).plot()
plt.show()

# first day temperature above
fda = first_day_temperature_above(tas, thresh='12 degC', freq='YS-JUL', window=1, after_date='07-01')
fda_ = first_day_temperature_above(tas, thresh='12 degC', freq='YS', window=1)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
fda.isel(time=2).plot(ax=ax[0])
fda_.isel(time=2).plot(ax=ax[1])
plt.show()


# first day temperature below
fdb = first_day_temperature_below(tas, thresh='5 degC', freq='YS-JUL', window=1, after_date='01-01')
fdb_ = first_day_temperature_below(tas, thresh='5 degC', freq='YS', window=1)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
fdb.isel(time=2).plot(ax=ax[0])
fdb_.isel(time=2).plot(ax=ax[1])
plt.show()



# south hemisphere
gss = first_day_temperature_above(tas, thresh='12 degC', freq='YS-JUL', window=1, after_date='07-01')
gse = first_day_temperature_below(tas, thresh='5 degC', freq='YS-JUL', window=1, after_date='01-01')
tas_agg = aggregate_between_dates(tas, gss, gse, 'sum', freq='YS-JUL')

fig, ax = plt.subplots(1, 3, figsize=(18, 4))
gss.isel(time=2).plot(ax=ax[0])
gse.isel(time=2).plot(ax=ax[1])
tas_agg.isel(time=2).plot(ax=ax[2])
plt.show()

# north hemisphere
gss = growing_season_end(tas, mid_date='01-02', freq='YS')
gse = growing_season_start(tas, mid_date='07-01')

gss.isel(time=0).plot()
plt.show()