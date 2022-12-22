# Create a csv file containing basin information

import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd

infile_Sean_MESH = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/ESMFmesh_ctsm_HCDN_nhru_final_671_v0_8e-3.nc'
inpath_CAMELS_q = '/glade/p/ral/hap/common_data/camels/obs_flow_met/basin_dataset_public_v1p2/usgs_streamflow/all'
infile_CAMELS_shp = '/glade/p/ral/hap/common_data/camels/shapefile/HCDN_nhru_final_671.shp'
outfile_info = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/Sean_MESH_CAMELS_basin_info.csv'

ds = xr.load_dataset(infile_Sean_MESH)
shp = gpd.read_file(infile_CAMELS_shp)
shp = shp.drop(columns=['geometry'])
shp_lat_cen = shp['lat_cen'].values
shp_lon_cen = shp['lon_cen'].values

basin_info = pd.DataFrame()
index_all = []
for i in range(ds.elementCount.size):
    lat_cen_mesh = ds['centerCoords'].values[i, 1]
    lon_cen_mesh = ds['centerCoords'].values[i, 0] - 360
    diff = np.abs(shp_lat_cen-lat_cen_mesh)+np.abs(shp_lon_cen-lon_cen_mesh)
    indexi = np.argmin(diff)
    shpi = shp[indexi:indexi+1]
    shpi = shpi.reindex()
    shpi['distdiff'] = np.nanmin(diff)
    id = shpi['hru_id'].values[0]
    shpi['file_obsQ'] = f'{inpath_CAMELS_q}/{id:08}_streamflow_qc.txt'
    basin_info = pd.concat([basin_info, shpi])
    index_all.append(indexi)


index_all = np.array(index_all)
if len(np.unique(index_all)) < len(index_all):
    print('Error!!! Basins overlapped!')

basin_info = basin_info.reindex()
basin_info.to_csv(outfile_info, index=False)

