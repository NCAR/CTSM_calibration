# Create a csv file containing basin information
# Basins in mesh file and the raw shapefile are the same. No basin dependence is considered.

import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os


for level in [1, 2, 3]:

    infile_ESMFmesh = f'/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/data_mesh_surf/HillslopeHydrology/esmf_mesh_files/corrected_HCDN_nhru_final_671_buff_fix_holes.CAMELSandTDX_areabias_fix.simp0.001.level{level}_polygons_neighbor_group_esmf_mesh.nc'
    inpath_CAMELS_q = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/CAMLES_Qobs'
    infile_CAMELS_shp = f'/glade/work/guoqiang/CAMELS_TDXHydro/HCDN_nhru_final_671.buff_fix_holes.CAMELSandTDX_areabias_fix.simp0.001.level{level}.gpkg'
    outfile_info = f'/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/data_mesh_surf/HillslopeHydrology/CAMELS_level{level}_basin_info.csv'

    ds_mesh = xr.load_dataset(infile_ESMFmesh)
    shp_camels = gpd.read_file(infile_CAMELS_shp)
    shp_lon_cen, shp_lat_cen = shp_camels['lon_cen'].values, shp_camels['lat_cen'].values

    basin_info = pd.DataFrame()
    index_all = []
    for i in range(ds_mesh.elementCount.size):
        # use the nearest center point to match mesh and shapefile basins
        lat_cen_mesh = ds_mesh['centerCoords'].values[i, 1]
        lon_cen_mesh = ds_mesh['centerCoords'].values[i, 0] - 360

        # Create a Point object using lat_cen_mesh and lon_cen_mesh
        point = Point(lon_cen_mesh, lat_cen_mesh)

        # Perform a spatial query to find the row(s) in shp_camels that contain the point
        matching_rows = shp_camels[shp_camels.geometry.contains(point)]

        if len(matching_rows) > 1:
            print('Point contained in >1 rows')
        elif len(matching_rows) == 0:
            print('No row contains the point, probably because mesh center is outside the basin shape')
            # find the basin with the closest center lat/lon
            diff = np.abs(shp_lat_cen - lat_cen_mesh) + np.abs(shp_lon_cen - lon_cen_mesh)
            indexi = np.argmin(diff)
            matching_row = shp_camels.iloc[indexi:indexi + 1].copy()  # Make a copy to avoid the SettingWithCopyWarning
        else:
            indexi = matching_rows.index[0]
            matching_row = matching_rows.copy()  # Make a copy to avoid the SettingWithCopyWarning

        hru_id = matching_row['hru_id'].values[0]
        fileq = f'{inpath_CAMELS_q}/{hru_id:08}_streamflow_qc.txt'
        matching_row.loc[indexi, 'file_obsQ'] = fileq

        if not os.path.isfile(fileq):
            print('file_obsQ does not exist', fileq)

        basin_info = pd.concat([basin_info, matching_row])

        index_all.append(indexi)


    index_all = np.array(index_all)
    if len(np.unique(index_all)) < len(index_all):
        print('Error!!! Basins overlapped!')

    basin_info = basin_info.reindex()
    basin_info.to_csv(outfile_info, index=False)