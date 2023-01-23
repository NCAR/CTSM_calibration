import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import shapely
import warnings
import numpy as np
import sys


##############################################################
#### GIS section
##############################################################

def intersection_shp(
                     shp_1,
                     shp_2):
    import geopandas as gpd
    from shapely.geometry import Polygon
    import shapefile  # pyshed library
    import shapely
    """
    @ author:                  Shervan Gharari
    @ Github:                  https://github.com/ShervanGharari/EASYMORE
    @ author's email id:       sh.gharari@gmail.com
    @license:                  GNU-GPLv3
    This fucntion intersect two shapefile. It keeps the fiels from the first and second shapefiles (identified by S_1_ and
    S_2_). It also creats other field including AS1 (area of the shape element from shapefile 1), IDS1 (an arbitary index
    for the shapefile 1), AS2 (area of the shape element from shapefile 1), IDS2 (an arbitary index for the shapefile 1),
    AINT (the area of teh intersected shapes), AP1 (the area of the intersected shape to the shapes from shapefile 1),
    AP2 (the area of teh intersected shape to the shapefes from shapefile 2), AP1N (the area normalized in the case AP1
    summation is not 1 for a given shape from shapefile 1, this will help to preseve mass if part of the shapefile are not
    intersected), AP2N (the area normalized in the case AP2 summation is not 1 for a given shape from shapefile 2, this
    will help to preseve mass if part of the shapefile are not intersected)
    Arguments
    ---------
    shp_1: geo data frame, shapefile 1
    shp_2: geo data frame, shapefile 2
    Returns
    -------
    result: a geo data frame that includes the intersected shapefile and area, percent and normalized percent of each shape
    elements in another one
    """
    # get the column name of shp_1
    column_names = shp_1.columns
    column_names = list(column_names)
    # removing the geometry from the column names
    column_names.remove('geometry')
    # renaming the column with S_1
    for i in range(len(column_names)):
        shp_1 = shp_1.rename(
            columns={column_names[i]: 'S_1_' + column_names[i]})
    # Caclulating the area for shp1
    shp_1['AS1'] = shp_1.area
    shp_1['IDS1'] = np.arange(shp_1.shape[0]) + 1
    # get the column name of shp_2
    column_names = shp_2.columns
    column_names = list(column_names)
    # removing the geometry from the colomn names
    column_names.remove('geometry')
    # renaming the column with S_2
    for i in range(len(column_names)):
        shp_2 = shp_2.rename(
            columns={column_names[i]: 'S_2_' + column_names[i]})
    # Caclulating the area for shp2
    shp_2['AS2'] = shp_2.area
    shp_2['IDS2'] = np.arange(shp_2.shape[0]) + 1
    # making intesection
    result = spatial_overlays(shp_1, shp_2, how='intersection')
    # Caclulating the area for shp2
    result['AINT'] = result['geometry'].area
    result['AP1'] = result['AINT'] / result['AS1']
    result['AP2'] = result['AINT'] / result['AS2']
    # taking the part of data frame as the numpy to incread the spead
    # finding the IDs from shapefile one
    ID_S1 = np.array(result['IDS1'])
    AP1 = np.array(result['AP1'])
    AP1N = AP1  # creating the nnormalized percent area
    ID_S1_unique = np.unique(ID_S1)  # unique idea
    for i in ID_S1_unique:
        INDX = np.where(ID_S1 == i)  # getting the indeces
        AP1N[INDX] = AP1[INDX] / AP1[INDX].sum()  # normalizing for that sum
    # taking the part of data frame as the numpy to incread the spead
    # finding the IDs from shapefile one
    ID_S2 = np.array(result['IDS2'])
    AP2 = np.array(result['AP2'])
    AP2N = AP2  # creating the nnormalized percent area
    ID_S2_unique = np.unique(ID_S2)  # unique idea
    for i in ID_S2_unique:
        INDX = np.where(ID_S2 == i)  # getting the indeces
        AP2N[INDX] = AP2[INDX] / AP2[INDX].sum()  # normalizing for that sum
    result['AP1N'] = AP1N
    result['AP2N'] = AP2N
    return result


def spatial_overlays(
                     df1,
                     df2,
                     how='intersection',
                     reproject=True):
    import geopandas as gpd
    from shapely.geometry import Polygon
    import shapefile  # pyshed library
    import shapely
    """
    Perform spatial overlay between two polygons.
    Currently only supports data GeoDataFrames with polygons.
    Implements several methods that are all effectively subsets of
    the union.
    author: Omer Ozak
    https://github.com/ozak
    https://github.com/geopandas/geopandas/pull/338
    license: GNU-GPLv3
    Parameters
    ----------
    df1: GeoDataFrame with MultiPolygon or Polygon geometry column
    df2: GeoDataFrame with MultiPolygon or Polygon geometry column
    how: string
        Method of spatial overlay: 'intersection', 'union',
        'identity', 'symmetric_difference' or 'difference'.
    use_sindex : boolean, default True
        Use the spatial index to speed up operation if available.
    Returns
    -------
    df: GeoDataFrame
        GeoDataFrame with new set of polygons and attributes
        resulting from the overlay
    """
    df1 = df1.copy()
    df2 = df2.copy()
    df1['geometry'] = df1.geometry.buffer(0)
    df2['geometry'] = df2.geometry.buffer(0)
    if df1.crs != df2.crs and reproject:
        print('Data has different projections.')
        print('Converted data to projection of first GeoPandas DatFrame')
        df2.to_crs(crs=df1.crs, inplace=True)
    if how == 'intersection':
        # Spatial Index to create intersections
        spatial_index = df2.sindex
        df1['bbox'] = df1.geometry.apply(lambda x: x.bounds)
        df1['sidx'] = df1.bbox.apply(lambda x: list(spatial_index.intersection(x)))
        pairs = df1['sidx'].to_dict()
        nei = []
        for i, j in pairs.items():
            for k in j:
                nei.append([i, k])
        # pairs = gpd.GeoDataFrame(nei, columns=['idx1','idx2'], crs=df1.crs)
        pairs = gpd.GeoDataFrame(nei, columns=['idx1', 'idx2'])
        pairs = pairs.merge(df1, left_on='idx1', right_index=True)
        pairs = pairs.merge(df2, left_on='idx2', right_index=True, suffixes=['_1', '_2'])
        pairs['Intersection'] = pairs.apply(lambda x: (x['geometry_1'].intersection(x['geometry_2'])).buffer(0), axis=1)
        # pairs = gpd.GeoDataFrame(pairs, columns=pairs.columns, crs=df1.crs)
        pairs = gpd.GeoDataFrame(pairs, columns=pairs.columns)
        cols = pairs.columns.tolist()
        cols.remove('geometry_1')
        cols.remove('geometry_2')
        cols.remove('sidx')
        cols.remove('bbox')
        cols.remove('Intersection')
        dfinter = pairs[cols + ['Intersection']].copy()
        dfinter.rename(columns={'Intersection': 'geometry'}, inplace=True)
        # dfinter = gpd.GeoDataFrame(dfinter, columns=dfinter.columns, crs=pairs.crs)
        dfinter = gpd.GeoDataFrame(dfinter, columns=dfinter.columns, crs=df1.crs)
        dfinter = dfinter.loc[dfinter.geometry.is_empty == False]
        dfinter.drop(['idx1', 'idx2'], inplace=True, axis=1)
        return dfinter
    elif how == 'difference':
        spatial_index = df2.sindex
        df1['bbox'] = df1.geometry.apply(lambda x: x.bounds)
        df1['sidx'] = df1.bbox.apply(lambda x: list(spatial_index.intersection(x)))
        df1['new_g'] = df1.apply(lambda x: reduce(lambda x, y: x.difference(y).buffer(0),
                                                  [x.geometry] + list(df2.iloc[x.sidx].geometry)), axis=1)
        df1.geometry = df1.new_g
        df1 = df1.loc[df1.geometry.is_empty == False].copy()
        df1.drop(['bbox', 'sidx', 'new_g'], axis=1, inplace=True)
        return df1
    elif how == 'symmetric_difference':
        df1['idx1'] = df1.index.tolist()
        df2['idx2'] = df2.index.tolist()
        df1['idx2'] = np.nan
        df2['idx1'] = np.nan
        dfsym = df1.merge(df2, on=['idx1', 'idx2'], how='outer', suffixes=['_1', '_2'])
        dfsym['geometry'] = dfsym.geometry_1
        dfsym.loc[dfsym.geometry_2.isnull() == False, 'geometry'] = dfsym.loc[
            dfsym.geometry_2.isnull() == False, 'geometry_2']
        dfsym.drop(['geometry_1', 'geometry_2'], axis=1, inplace=True)
        dfsym = gpd.GeoDataFrame(dfsym, columns=dfsym.columns, crs=df1.crs)
        spatial_index = dfsym.sindex
        dfsym['bbox'] = dfsym.geometry.apply(lambda x: x.bounds)
        dfsym['sidx'] = dfsym.bbox.apply(lambda x: list(spatial_index.intersection(x)))
        dfsym['idx'] = dfsym.index.values
        dfsym.apply(lambda x: x.sidx.remove(x.idx), axis=1)
        dfsym['new_g'] = dfsym.apply(lambda x: reduce(lambda x, y: x.difference(y).buffer(0),
                                                      [x.geometry] + list(dfsym.iloc[x.sidx].geometry)), axis=1)
        dfsym.geometry = dfsym.new_g
        dfsym = dfsym.loc[dfsym.geometry.is_empty == False].copy()
        dfsym.drop(['bbox', 'sidx', 'idx', 'idx1', 'idx2', 'new_g'], axis=1, inplace=True)
        return dfsym
    elif how == 'union':
        dfinter = spatial_overlays(df1, df2, how='intersection')
        dfsym = spatial_overlays(df1, df2, how='symmetric_difference')
        dfunion = dfinter.append(dfsym)
        dfunion.reset_index(inplace=True, drop=True)
        return dfunion
    elif how == 'identity':
        dfunion = spatial_overlays(df1, df2, how='union')
        cols1 = df1.columns.tolist()
        cols2 = df2.columns.tolist()
        cols1.remove('geometry')
        cols2.remove('geometry')
        cols2 = set(cols2).intersection(set(cols1))
        cols1 = list(set(cols1).difference(set(cols2)))
        cols2 = [col + '_1' for col in cols2]
        dfunion = dfunion[(dfunion[cols1 + cols2].isnull() == False).values]
        return dfunion






shp = gpd.read_file('/Users/guoqiang/Downloads/test_source_shapefile.shp')


self_tolerance =  10**-5 # tolerance


if not shp.crs:
    print('inside shp_lon_correction, no crs is provided for the shapefile; EASYMORE will allocate WGS84 to correct for lon above 180')
    shp = shp.set_crs("epsg:4326")
#
col_names = shp.columns.to_list()
col_names.remove('geometry')
df_attribute = pd.DataFrame()
if col_names:
    df_attribute = shp.drop(columns = 'geometry')
shp = shp.drop(columns = col_names)
shp['ID'] = np.arange(len(shp))+1
# get the maximum and minimum bound of the total bound
min_lon, min_lat, max_lon, max_lat = shp.total_bounds
if (360 < max_lon) and (min_lon<0):
    sys.exit('The minimum longitude is higher than 360 while the minimum longitude is lower that 0')
if (max_lon < 180) and (-180 < min_lon):
    print('EASYMORE detects that shapefile longitude is between -180 and 180, no correction is performed')
    shp_final = shp
else:
    shp_int1 = pd.DataFrame()
    shp_int2 = pd.DataFrame()
    # intersection if shp has a larger lon of 180 so it is 0 to 360,
    if (180 < max_lon) and (-180 < min_lon):
        print('EASYMORE detects that shapefile longitude is between 0 and 360, correction is performed to transfer to -180 to 180')
        # shapefile with 180 to 360 lon
        gdf1 = {'geometry': [Polygon([(  180.0+self_tolerance, -90.0+self_tolerance), ( 180.0+self_tolerance,  90.0-self_tolerance),\
                                      (  360.0-self_tolerance,  90.0-self_tolerance), ( 360.0-self_tolerance, -90.0+self_tolerance)])]}
        gdf1 = gpd.GeoDataFrame(gdf1)
        gdf1 = gdf1.set_crs ("epsg:4326")
        warnings.simplefilter('ignore')
        shp_int1 = intersection_shp(shp, gdf1)
        warnings.simplefilter('default')
        col_names = shp_int1.columns
        col_names = list(filter(lambda x: x.startswith('S_1_'), col_names))
        col_names.append('geometry')
        shp_int1 = shp_int1[shp_int1.columns.intersection(col_names)]
        col_names.remove('geometry')
        # rename columns without S_1_
        for col_name in col_names:
            col_name = str(col_name)
            col_name_n = col_name.replace("S_1_","");
            shp_int1 = shp_int1.rename(columns={col_name: col_name_n})
        #
        for index, _ in shp_int1.iterrows():
            polys = shp_int1.geometry.iloc[index] # get the shape
            polys = shapely.affinity.translate(polys, xoff=-360.0, yoff=0.0, zoff=0.0)
            shp_int1.geometry.iloc[index] = polys
        # shapefile with -180 to 180 lon
        gdf2 = {'geometry': [Polygon([( -180.0+self_tolerance, -90.0+self_tolerance), (-180.0+self_tolerance,  90.0-self_tolerance),\
                                      (  180.0-self_tolerance,  90.0-self_tolerance), ( 180.0-self_tolerance, -90.0+self_tolerance)])]}
        gdf2 = gpd.GeoDataFrame(gdf2)
        gdf2 = gdf2.set_crs ("epsg:4326")
        warnings.simplefilter('ignore')
        shp_int2 = intersection_shp(shp, gdf2)
        warnings.simplefilter('default')
        col_names = shp_int2.columns
        col_names = list(filter(lambda x: x.startswith('S_1_'), col_names))
        col_names.append('geometry')
        shp_int2 = shp_int2[shp_int2.columns.intersection(col_names)]
        col_names.remove('geometry')
        # rename columns without S_1_
        for col_name in col_names:
            col_name = str(col_name)
            col_name_n = col_name.replace("S_1_","");
            shp_int2 = shp_int2.rename(columns={col_name: col_name_n})

a=1
b=2