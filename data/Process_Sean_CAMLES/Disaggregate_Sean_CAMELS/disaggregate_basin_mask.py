# based on the raw MESH file from Sean, setting the mask variable to only choose one basin in model run

import netCDF4 as nc4
import xarray as xr
import numpy as np
import os, sys


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

infile_basins = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/ESMFmesh_ctsm_HCDN_nhru_final_671_v0_8e-3.nc'
outpath = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_basin_mask'
os.makedirs(outpath, exist_ok=True)


# get netcdf file format
with nc4.Dataset(infile_basins) as ncd:
    netcdf_format = ncd.file_format
    print(f'Netcdf format of {infile_basins}: {netcdf_format}')

# read and disaggregate
dsall = xr.load_dataset(infile_basins)
numbasin = dsall.elementCount.size

for i in range(numbasin):
    outfilei = f'{outpath}/ESMFmesh_ctsm_HCDN_nhru_final_671_v0_8e-3_basin{i}.nc'
    elementMask = dsall.elementMask.values
    elementMask[:] = 0
    elementMask[i] = 1
    dsall.elementMask.values = elementMask
    save_to_netcdf(dsall, outfilei, netcdf_format)

