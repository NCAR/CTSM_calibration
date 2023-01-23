# We need the basin boundary of a shapefile to remap gridded forcing to the basin

import pandas as pd
import geopandas as gpd
import os

# shapefile of all basins, which is generated during basin preparation
infile_mesh_shp = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/shp/ESMFmesh_ctsm_HCDN_nhru_final_671.buff_fix_holes_polygons_simplified_5e-4_split_nested.shp'

# basin information to link basin ID and mesh shp
infile_info = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/info_ESMFmesh_ctsm_HCDN_nhru_final_671.buff_fix_holes_polygons_simplified_5e-4_split_nested.csv'

# outpath
outpath = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/shp/split_nest'
os.makedirs(outpath, exist_ok=True)


info = pd.read_csv(infile_info)
shp = gpd.read_file(infile_mesh_shp)
num = len(shp)

for i in range(num):
    shpi = shp.loc[[i]]
    idi = info.loc[info['mesh_id']==i]['hru_id'].iloc[0]
    outfile = f'{outpath}/{idi:08}_split_nest.shp'
    shpi.to_file(outfile)
