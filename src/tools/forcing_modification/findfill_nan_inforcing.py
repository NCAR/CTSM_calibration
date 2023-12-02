import os, glob
import xarray as xr
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from concurrent.futures import ProcessPoolExecutor

def fill_nan_with_nearest(ds):
    """
    Fill NaN values in all variables of an xarray dataset 
    with dimensions (lat, lon, time) using nearest neighbor interpolation.

    Parameters:
    ds (xarray.Dataset): The input dataset.

    Returns:
    xarray.Dataset: The dataset with NaN values filled in the specified variables.
    """
    ds_filled = ds.copy()

    # Create a mesh grid for latitude and longitude
    lon, lat = np.meshgrid(ds_filled.lon, ds_filled.lat)
    points = np.array([lat.ravel(), lon.ravel()]).T

    # Iterate over each variable in the dataset
    for var in ds_filled.data_vars:
        # Check if the variable has the required dimensions
        if set(ds_filled[var].dims) == {'lat', 'lon', 'time'}:
            for t in range(len(ds_filled.time)):
                # Extract the 2D slice for the current time step
                data_2d = ds_filled[var].isel(time=t).values

                # Find indices of NaN and non-NaN values
                valid = ~np.isnan(data_2d)
                invalid = np.isnan(data_2d)

                if np.any(invalid):
                    # Perform nearest neighbor interpolation
                    interpolator = NearestNDInterpolator(points[valid.ravel()], data_2d[valid])
                    data_2d[invalid] = interpolator(points[invalid.ravel()])

                    # Update the dataset
                    ds_filled[var].isel(time=t).values[:] = data_2d

    return ds_filled


def check_and_fill(file, allvar, y):
    ds = xr.open_dataset(file)
    
    flag = False
    for var in allvar:
        v = ds[var].values
        d = np.sum(np.sum(np.isnan(v), axis=1),axis=1)
        n = np.sum(d>0)
        if n>0:
            print(f'{var}-{y}: Time steps with NaN: {n}. Max nan grid: {np.max(d)}')
            flag = True
    
    ds_filled = ds
    if flag==True:
        ds_filled = fill_nan_with_nearest(ds)
        for var in allvar:
            v = ds_filled[var].values
            d = np.sum(np.sum(np.isnan(v), axis=1),axis=1)
            n = np.sum(d>0)
            print(f'After nearest filling {var}-{y}: Time steps with NaN: {n}. Max nan grid: {np.max(d)}')
            if np.nanmax(d)>0:
                sys.exit('Error. Failed filling!')
        
    return ds_filled, flag


def save_filled_file(file, flag, ds):
    if flag:
        print(f'Save ds to the new {file}')
        os.system(f'mv {file} {file}-withNaN')
        ds.to_netcdf(file, format='NETCDF3_CLASSIC')

        
def process_path(path):
    """
    Process a single path.
    """
    print('#'*50)
    print('Processing:', path)

    year = [f'{y}-{y+4}' for y in range(1951, 2020, 5)]
    year[-1] = '2016-2019'

    for y in year:
        # Process each file type
        for file_type in ['Precip', 'Solar', 'TPQWL']:
            if file_type == 'Precip':
                file = f'{path}/Precip/subset_clmforc.E5LEME.c2023.010x010.Precip.{y}.nc'
                ds, flag = check_and_fill(file, ['PRECTmms'], y)
            elif file_type == 'Solar':
                file = f'{path}/Solar/subset_clmforc.E5LEME.c2023.010x010.Solar.{y}.nc'
                ds, flag = check_and_fill(file, ['FSDS'], y)
            else:  # TPQWL
                file = f'{path}/TPQWL/subset_clmforc.E5LEME.c2023.010x010.TPQWL.{y}.nc'
                ds, flag = check_and_fill(file, ['FLDS', 'PSRF', 'QBOT', 'TBOT', 'WIND'], y)
            
            save_filled_file(file, flag, ds)
        

allpaths1 = [f'/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/level1_{i}_SubsetForcing' for i in range(627)]
allpaths2 = [f'/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/level2_{i}_SubsetForcing' for i in range(40)]
allpaths3 = [f'/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/level3_{i}_SubsetForcing' for i in range(4)]
allpaths = allpaths1 + allpaths2 + allpaths3


# Using ProcessPoolExecutor to parallelize
with ProcessPoolExecutor(max_workers=36) as executor:
    # Submit tasks to the executor
    futures = [executor.submit(process_path, path) for path in allpaths]

    # Wait for all tasks to complete (optional, for synchronization)
    for future in futures:
        future.result()