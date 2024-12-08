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
# from lsapy.criteria.indicators import CriteriaIndicator
from lsapy.functions import SuitabilityFunction, MembershipSuitFunction, discrete

# Testing Utilities
import numpy as np
import matplotlib.pyplot as plt
from xclim.indicators.atmos import growing_degree_days, precip_accumulation


# import data
soil_data = load_data('soil')
clim_data = load_data('climate').interp_like(soil_data, 'nearest')
data = xr.merge([soil_data, clim_data])

# compute criteria indicator
gdd = growing_degree_days(clim_data['tas'], thresh='10 degC', freq='YS-JUL')
anr = precip_accumulation(clim_data['pr'], freq='YS-JUL')

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
gdd.isel(time=1).plot(ax=ax[0])
anr.isel(time=1).plot(ax=ax[1])
plt.show()


# test suitabiliy function
SuitabilityFunction(func_method='logistic', func_params={'a': 1, 'b': 2})
MembershipSuitFunction.fit(x=[1500,1350,1250,1150,1000], plot=True)
f_suit = SuitabilityFunction(func_method='vetharaniam2022_eq5', func_params={'a': 0.8759, 'b': 1248.5})
x = np.linspace(500, 2000, 100)
y = f_suit(x)
plt.plot(x, y)
plt.show()



# test suitability criteria
sc = SuitabilityCriteria(
    name = 'temo_req',
    long_name = 'Temperature Requirements',
    description = 'Temperature requirements for crop growth',
    indicator=gdd,
    func = SuitabilityFunction(func_method='vetharaniam2022_eq5', func_params={'a': -1.41, 'b': 801})
)
gdd_suit = sc.compute()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
gdd.isel(time=2).plot(ax=ax[0])
gdd_suit.isel(time=2).plot(ax=ax[1], vmin=0, vmax=1)
plt.show()


sc = SuitabilityCriteria(
    name = 'Drainage Class',
    weight= 3,
    category= 'SoilTerrain',
    indicator = soil_data['DRC'],
    func=SuitabilityFunction(func_method='discrete', func_params={'rules': {'1': 0, '2': 0.1, '3': 0.5, '4': 0.9, '5': 1}}),
)
drc = sc.compute()
drc.plot()
plt.show()


# test land suitability criteria
ls = LandSuitability(
    name = 'land_suitability',
    criteria = {
        'water_req' : SuitabilityCriteria(
            name = 'Annual Rainfall Requirement',
            weight=0.5,
            category='Climate',
            indicator= anr,
            func = SuitabilityFunction(func_method='vetharaniam2022_eq5', func_params={'a': 0.876, 'b': 1248})
        ),
        'temp_req' : SuitabilityCriteria(
            name = 'Temperature Requirements',
            weight=1,
            category='Climate',
            indicator = gdd,
            func = SuitabilityFunction(func_method='vetharaniam2022_eq5', func_params={'a': -1.41, 'b': 801})
        ),
        'prd' : SuitabilityCriteria(
            name = 'Soil Depth',
            weight=0.5,
            category='TerrainSoil',
            indicator= soil_data['PRD'],
            func = SuitabilityFunction(func_method='vetharaniam2022_eq5', func_params={'a': -11.98, 'b': 0.459})
        )
    }
)


# test criteria suitability
ls_out = ls.compute_criteria_suitability(inplace=False)

fig, ax = plt.subplots(1, 3, figsize=(18, 4))
ls_out['water_req'].isel(time=1).plot(ax=ax[0])
ls_out['temp_req'].isel(time=1).plot(ax=ax[1])
ls_out['prd'].plot(ax=ax[2])
plt.show()


# test category suitability
ls_out = ls.compute_category_suitability(method='weighted_geomean', keep_criteria=True, inplace=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ls_out['Climate'].isel(time=1).plot(ax=ax[0])
ls_out['TerrainSoil'].plot(ax=ax[1])
plt.show()


# test land suitability
ls.compute_criteria_suitability(inplace=True)
ls.compute_category_suitability(method='weighted_geomean', keep_criteria=True, inplace=True)
ls_out = ls.compute_suitability(method='weighted_mean', keep_category=True, inplace=False)

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ls_out['Climate'].isel(time=1).plot(ax=ax[0,0])
ls_out['TerrainSoil'].plot(ax=ax[0,1])
ls_out['Suitability'].isel(time=1).plot(ax=ax[1,0])
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
from xclim.core.calendar import get_calendar
from xclim.core.calendar import doy_to_days_since
import xclim.indices.run_length as rl

tas = clim_data['tas'].sel(time=slice('2000-01-01', '2002-12-31'))
tas = convert_units_to(tas, 'degC')
tasmin = clim_data['tasmin'].sel(time=slice('2000-01-01', '2002-12-31'))
tasmin = convert_units_to(tasmin, 'degC')
tasmax = clim_data['tasmax'].sel(time=slice('2000-01-01', '2002-12-31'))
tasmax = convert_units_to(tasmax, 'degC')
deg_days = (tas - 10).clip(min=0)

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
tas_agg = aggregate_between_dates(deg_days, gss, gse, 'sum', freq='YS-JUL')

fig, ax = plt.subplots(1, 3, figsize=(18, 4))
gss.isel(time=1).plot(ax=ax[0], vmin=0, vmax=366)
gse.isel(time=1).plot(ax=ax[1], vmin=0, vmax=366)
tas_agg.isel(time=1).plot(ax=ax[2])
plt.show()

# north hemisphere
gss = growing_season_end(tas, mid_date='01-02', freq='YS')
gse = growing_season_start(tas, mid_date='07-01', freq='YS')

gss.isel(time=0).plot()
plt.show()

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

### aggregate between dates workfow
start, end = gss, gse # use growing season start and end as proxy for growing season
op, freq = 'sum', None



doy_bounds = [gse, gss]
doy_bounds = ["08-01", gse]
da = tas
include_bounds = [True, True]

### select_time
# def select_time(da: xr.DataArray, doy_bounds: tuple, include_bounds: tuple[bool, bool] = [True, True]) -> xr.DataArray:

#     if isinstance(doy_bounds[0], int) and isinstance(doy_bounds[1], int):
#         mask = da.time.dt.dayofyear.isin(_get_doys(*doy_bounds, [True, True]))

#     else:
#         start, end = doy_bounds
#         if isinstance(start, int):
#             start = xr.where(end.isnull(), np.nan, start)
#         if isinstance(end, int):
#             end = xr.where(start.isnull(), np.nan, end)
        
#         frequencies = []
#         for bound in [start, end]:
#             try:
#                 frequencies.append(xr.infer_freq(bound.time))
#             except AttributeError:
#                 frequencies.append(None)
        
#         good_freq = set(frequencies) - {None}

#         if len(good_freq) != 1:
#             raise ValueError(
#                 f"Non-inferrable resampling frequency or inconsistent frequencies. Got start, end = {frequencies}."
#                 " Please consider providing `freq` manually."
#             )
#         freq = good_freq.pop()

#         cal = get_calendar(da, dim="time")

#         start = start.convert_calendar(cal)
#         start.attrs["calendar"] = cal
#         start = doy_to_days_since(start)

#         end = end.convert_calendar(cal)
#         end.attrs["calendar"] = cal
#         end = doy_to_days_since(end)

#         out = []
#         for base_time, indexes in da.resample(time=freq).groups.items():
#             # get group slice
#             group = da.isel(time=indexes)

#             start_d = start.sel(time=base_time)
#             end_d = end.sel(time=base_time)

#             if not include_bounds[0]:
#                 start_d += 1
#             if not include_bounds[1]:
#                 end_d -= 1

#             # select days between start and end for group
#             days = (group.time - base_time).dt.days
#             days[days < 0] = np.nan

#             mask = (days >= start_d) & (days <= end_d)
#             out.append(mask)
#         mask = xr.concat(out, dim="time")

#     return da.where(mask, drop=False)



# test select_time
start, end = gss, gse

deg_days

gdd = aggregate_between_dates(deg_days, start, end, 'sum', freq='YS-JUL')
gdd_ = select_time(deg_days, [start, end]).resample(time='YS-JUL').sum(dim='time', skipna=True)
gdd_ = xr.where(((start.isnull()) | (end.isnull())), np.nan, gdd_)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
gdd.isel(time=2).plot(ax=ax[0], vmin=0, vmax=1500)
gdd_.isel(time=2).plot(ax=ax[1], vmin=0, vmax=1500)
plt.show()

gdd.isel(time=2).values[2]
gdd_.isel(time=2).values[2]

from xclim.core.calendar import select_time

tas_ = select_time(tas, doy_bounds=[gss, gse])

lat_i, lon_i = 25, 25

gss.isel(lat=lat_i, lon=lon_i).values
gse.isel(lat=lat_i, lon=lon_i).values


tas_.isel(time=slice(0,366), lat=25, lon=25)