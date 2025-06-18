# disaggregate the domain file into >500 files. Each file will contain just one basin.

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

infile_basins = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/ESMFmesh_ctsm_HCDN_nhru_final_671.buff_fix_holes_polygons_simplified_5e-4_split_nested.nc'
outpath = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_basin_divide_split_nest'
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

name = pathlib.Path(infile_basins).name

for i in range(numbasin):
    namei = name.replace('.nc', f'_basin{i}.nc')
    outfilei = f'{outpath}/{namei}'
    # print(f'Processing basin {i} -- {numbasin}')
    # index used to select basin
    index_elementCount = [i]
    index_connectionCount = np.arange(np.sum(numElementConn[:i]), np.sum(numElementConn[:i+1])).astype(int)
    index_nodeCount = elementConn[index_connectionCount].astype(int) - 1 # index starts from 0
    # print(index_elementCount)
    # print(index_connectionCount)
    # print(index_nodeCount)
    dsi = dsall.isel(elementCount=index_elementCount, connectionCount=index_connectionCount, nodeCount=index_nodeCount)
    # re-assign elementConn
    # dsi['elementConn'].values = dsi['elementConn'].values - dsi['elementConn'].values.min() + 1
    dsi['elementConn'].values = np.arange(len(index_nodeCount)) + 1.0 # needs to make sure this is counterclockwise
    save_to_netcdf(dsi, outfilei, netcdf_format)