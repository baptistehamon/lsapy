from typing import Optional, Union, Callable, Any

import xarray as xr

_all__ = ["CriteriaIndicator"]

class CriteriaIndicator:
    def __init__(
            self,
            var_name: str,
            standard_name: str,
            long_name: str,
            units: str = '',
            func: Callable[...,xr.DataArray] = lambda x: x,
            func_params: Optional[dict[str,]] = None,
            data_arg_name: Optional[str] = None,
            description: Optional[str] = None,
            comment: Optional[str] = None
    ) -> None:
        self.var_name = var_name
        self.standard_name = standard_name
        self.long_name = long_name
        self.units = units
        self.func = func
        self.func_params = func_params
        self.data_arg_name = data_arg_name
        self.description = description
        self.comment = comment
    
    def compute(self, data: xr.DataArray | xr.Dataset) -> xr.DataArray:
        if self.data_arg_name is not None:
            self.func_params[self.data_arg_name] = data
            return self.func(**self.func_params)
        else:
            if self.func_params is None:
                return self.func(data)
            return self.func(data, **self.func_params)
    


# from xclim.indicators.atmos import huglin_index
# import matplotlib.pyplot as plt

# from landsuitability.testing.utils import load_data

# soil_data = load_data('soil')
# climate_data = load_data('climate')

# ci = CriteriaIndicator(
#     var_name='Huglin Index',
#     standard_name='Huglin Index',
#     long_name='Huglin heliothermal index',
#     units='degC',
#     func= lambda x: huglin_index(ds=x, freq='YS', thresh='0 degC')
# )

# huglin = ci.compute(climate_data)
# huglin.isel(time=0).plot()
# plt.show()

# def test_function(ds: xr.Dataset) -> xr.DataArray:
#     tasmax = ds['tasmax']
#     tasmin = ds['tasmin']
#     return (tasmax + tasmin) / 2

# ci = CriteriaIndicator(
#     var_name='Temperature',
#     standard_name='Temperature',
#     long_name='Temperature',
#     units='degC',
#     func=test_function
# )

# tas = ci.compute(climate_data)
# tas.isel(time=0).plot()
# plt.show()

# ci = CriteriaIndicator(
#     var_name='Test',
#     standard_name='Test',
#     long_name='Test',
# )

# test = ci.compute(climate_data['tas'])
# test