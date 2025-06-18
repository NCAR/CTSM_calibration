# convert Sean's mesh domain file to shapefile to faciliate comparison with Andy's raw CAMELS polygons


import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

for level in [1, 2, 3]:

    infile = f'/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/data_mesh_surf/HillslopeHydrology/esmf_mesh_files/corrected_HCDN_nhru_final_671_buff_fix_holes.CAMELSandTDX_areabias_fix.simp0.001.level{level}_polygons_neighbor_group_esmf_mesh.nc'
    outfile = f'/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/data_mesh_surf/HillslopeHydrology/esmf_mesh_files/corrected_HCDN_nhru_final_671_buff_fix_holes.CAMELSandTDX_areabias_fix.simp0.001.level{level}_polygons_neighbor_group_esmf_mesh.gpkg'

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

    gdf = gpd.GeoDataFrame({'SeanID': np.arange(numbasin), 'lon_cen': centerCoords[:,0] - 360, 'lat_cen': centerCoords[:,1]}, geometry=polygons, crs="EPSG:4326")
    gdf.to_file(outfile)
