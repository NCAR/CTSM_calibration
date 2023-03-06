# disaggregate the surface dataset file into >500 files. Each file will contain just one basin.

import netCDF4 as nc4
import xarray as xr
import numpy as np
import os, sys, pathlib


def save_to_netcdf(ds, outfile, format):
    print('Saving to', outfile)
    # encoding = {}
    for v in ds.data_vars:
        ec = ds[v].encoding
        if not '_FillValue' in ec:
            ec['_FillValue'] = None
        if not 'coordinates' in ec:
            ec['coordinates'] = None
        ds[v].encoding = ec
    ds.to_netcdf(outfile, format=format)

########################################################################################################################
# Basin domain file

infile_basins = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/surfdata_CAMELS_split_nested_hist_78pfts_CMIP6_simyr2000_c230105.nc'
outpath = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_basin_divide_split_nest'
os.makedirs(outpath, exist_ok=True)

# get netcdf file format
with nc4.Dataset(infile_basins) as ncd:
    netcdf_format = ncd.file_format
    print(f'Netcdf format of {infile_basins}: {netcdf_format}')

# read and disaggregate
dsall = xr.load_dataset(infile_basins)

name = pathlib.Path(infile_basins).name
for i in range(len(dsall.gridcell)):
    namei = name.replace('.nc', f'_basin{i}.nc')
    outfilei = f'{outpath}/{namei}'
    dsi = dsall.isel(gridcell= [i] ) # even just one element, the "gridcell" dim has to be preserved
    save_to_netcdf(dsi, outfilei, netcdf_format)