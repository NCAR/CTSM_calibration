# disaggregate the domain file into >500 files. Each file will contain just one basin.

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
outpath = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_basin_domains'
os.makedirs(outpath, exist_ok=True)

# get netcdf file format
with nc4.Dataset(infile_basins) as ncd:
    netcdf_format = ncd.file_format
    print(f'Netcdf format of {infile_basins}: {netcdf_format}')

# read and disaggregate
dsall = xr.load_dataset(infile_basins)

numbasin = dsall.elementCount.size
nodeCoords = dsall.nodeCoords.values
elementConn = dsall.elementConn.values
numElementConn = dsall.numElementConn.values

outfilei = f'{outpath}/ESMFmesh_ctsm_HCDN_nhru_final_671_v0_8e-3_basin1-2.nc'

# dsi = dsall.isel(elementCount=[0, 1], connectionCount=slice(0, 101))
dsi = dsall.isel(elementCount=[0, 1])
# dsi = dsall.isel(elementCount=slice(0, 587))
# dsi = dsall
save_to_netcdf(dsi, outfilei, netcdf_format)

########################################################################################################################
# Surface data file
infile_surfdata = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/surfdata_CAMELS_hist_78pfts_CMIP6_simyr2000_c221004.nc'
# infile_surfdata = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/surfdata_CAMELS_hist_78pfts_CMIP6_simyr1850_c221004.nc'

# get netcdf file format
with nc4.Dataset(infile_surfdata) as ncd:
    netcdf_format = ncd.file_format
    print(f'Netcdf format of {infile_surfdata}: {netcdf_format}')

# read and disaggregate
dsall = xr.load_dataset(infile_surfdata)
outfilei = f'{outpath}/surfdata_CAMELS_hist_78pfts_CMIP6_simyr2000_c221004_basin1-2.nc'
dsi = dsall.isel(gridcell=[0, 1])
# dsi = dsall.isel(gridcell=slice(0, 587))
# dsi = dsall
save_to_netcdf(dsi, outfilei, netcdf_format)
