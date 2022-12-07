# convert Sean's mesh domain file to shapefile to faciliate comparison with Andy's raw CAMELS polygons


import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

# source files: /glade/p/ral/hap/common_data/camels/shapefile/*
infile = '/glade/work/guoqiang/CTSM_cases/CAMELS_Sean/shared_data_Sean/ESMFmesh_ctsm_HCDN_nhru_final_671_v0_8e-3.nc'
outfile = '/glade/work/guoqiang/CTSM_cases/CAMELS_Sean/shared_data_Sean/ESMFmesh_ctsm_HCDN_nhru_final_671_v0_8e-3.shp'

dsall = xr.load_dataset(infile)
numbasin = dsall.elementCount.size
nodeCoords = dsall.nodeCoords.values
elementConn = dsall.elementConn.values
numElementConn = dsall.numElementConn.values
centerCoords = dsall.centerCoords.values

df = gpd.GeoDataFrame()

polygons = []
for i in range(numbasin):
    index_elementCount = [i]
    index_connectionCount = np.arange(np.sum(numElementConn[:i]), np.sum(numElementConn[:i+1])).astype(int)
    index_nodeCount = elementConn[index_connectionCount].astype(int) - 1 # index starts from 0
    nodeCoordsi = nodeCoords[index_nodeCount]

    lat_point_list = nodeCoordsi[:, 1]
    lon_point_list = nodeCoordsi[:, 0] - 360

    polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
    polygons.append(polygon_geom)

gdf = gpd.GeoDataFrame({'SeanID': np.arange(numbasin), 'lon_cen': 180-centerCoords[:,0], 'lat_cen': centerCoords[:,1]}, geometry=polygons, crs="EPSG:4326")
gdf.to_file(outfile)


# ###
# # check a basin with weird shape
# sid = 556
# dsall.isel(elementCount=sid)
# np.sum(numElementConn[0:sid])
# latlon = nodeCoords[elementConn[15715:15715+96-1].astype(int)]
# import matplotlib.pyplot as plt
# plt.scatter(latlon[:,0],latlon[:,1])