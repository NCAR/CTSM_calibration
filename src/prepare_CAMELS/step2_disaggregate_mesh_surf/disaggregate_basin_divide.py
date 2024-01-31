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

outpath = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/data_mesh_surf/HillslopeHydrology/disaggregation'
os.makedirs(outpath, exist_ok=True)

        
for level in [1, 2, 3]:

    infile_basins = f'/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/data_mesh_surf/HillslopeHydrology/sfcdata/surfdata_CAMELSandTDX_areabias_fix.simp0.001.level{level}_hist_78pfts_CMIP6_simyr2000_HAND_4_col_hillslope_geo_params_nlcd_bedrock.nc'

    os.makedirs(outpath, exist_ok=True)

    # get netcdf file format
    with nc4.Dataset(infile_basins) as ncd:
        netcdf_format = ncd.file_format
        print(f'Netcdf format of {infile_basins}: {netcdf_format}')

    # read and disaggregate
    dsall = xr.load_dataset(infile_basins)

    name = pathlib.Path(infile_basins).name
    for i in range(len(dsall.gridcell)):
        namei = name.replace('.nc', f'_{i}.nc')
        outfilei = f'{outpath}/{namei}'
        
        if os.path.isfile(outfilei):
            continue
        
        dsi = dsall.isel(gridcell= [i] ) # even just one element, the "gridcell" dim has to be preserved
        save_to_netcdf(dsi, outfilei, netcdf_format)