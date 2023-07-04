# Create a csv file containing basin information
# for nested basins, split them (i.e., bigger basin substracts smaller basin)
# Basins in mesh file and the raw shapefile are not the same for split basins. Upstream basins are also found

import os, shutil
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

def check_if_a_basin_contain_another(shpi, shpj):
    if shpj['AREA'] < shpi['AREA']:
        shpj_cen = Point(shpj['lon_cen'], shpj['lat_cen'])
        if shpi.geometry.contains(shpj_cen):
            flag = 1 # shpi contain shpj
        else:
            flag = 0 # no contain
    else:
        shpi_cen = Point(shpi['lon_cen'], shpi['lat_cen'])
        if shpj.geometry.contains(shpi_cen):
            flag = -1 # shpj contain shpi
        else:
            flag = 0
    return flag


def find_independent_nest_basins(index, shp_camels):
    # if there are multiple layers of nested basins, only the highest level is needed
    if len(index) == 1:
        indep_next = index
    else:
        indep_next = []
        for i in range(len(index)):
            addflag = True
            for j in range(len(index)):
                if i != j:
                    flag = check_if_a_basin_contain_another(shp_camels.iloc[index[i]], shp_camels.iloc[index[j]])
                    if flag == -1: # i is contained in j
                        addflag = False
                        break
            if addflag == True: # i is not contained in any other basins
                indep_next = indep_next + [index[i]]
    return indep_next



infile_Sean_MESH = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/ESMFmesh_ctsm_HCDN_nhru_final_671.buff_fix_holes_polygons_simplified_5e-4_split_nested.nc'
inpath_CAMELS_q = '/glade/p/ral/hap/common_data/camels/obs_flow_met/basin_dataset_public_v1p2/usgs_streamflow/all'
infile_CAMELS_shp = '/glade/p/ral/hap/common_data/camels/shapefile/HCDN_nhru_final_671.shp'
outfile_info = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/info_ESMFmesh_ctsm_HCDN_nhru_final_671.buff_fix_holes_polygons_simplified_5e-4_split_nested.csv'
outpath_CAMELS_q = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/CAMLES_q_split_nest'

ds_mesh = xr.load_dataset(infile_Sean_MESH)
shp_camels = gpd.read_file(infile_CAMELS_shp)

########################################################################################################################
# find nested basins
# Example: 1 contains 2 and 3, and 2 contains 3
# nest_index_all [ [1, 2, 3], ... ] or [ [1, 3, 2], ... ]
# nest_index_ind [ [1, 2], ...]


nest_index_all = {}
for i in range(len(shp_camels)):
    shpi = shp_camels.iloc[i]
    indi = []
    for j in range(len(shp_camels)):
        if i != j:
            shpj = shp_camels.iloc[j]
            flag = check_if_a_basin_contain_another(shpi, shpj)
            if flag == 1:
                indi.append(j)
    if len(indi) > 0:
        nest_index_all[i] = indi


nest_index_ind = {}
for ind1, ind2 in nest_index_all.items():
    if len(ind2) == 1:
        nest_index_ind[ind1] = ind2
    else:
        ind2 = find_independent_nest_basins(ind2, shp_camels)
        nest_index_ind[ind1] = ind2

########################################################################################################################
# plot nested basins
# just for visualization and check

# import matplotlib.pyplot as plt
# import matplotlib.backends.backend_pdf
# pdf = matplotlib.backends.backend_pdf.PdfPages("nested_basins.pdf")
# colors = 'krbyg'
# for ind1, ind2 in nest_index_all.items():
#     fig, ax = plt.subplots(1, 2, figsize=[6, 4])
#     indi = [ind1] + ind2
#     for j in range(len(indi)):
#         if j == 0:
#             shp_camels.iloc[[indi[j]]].plot(ax=ax[0], color=colors[j])
#         else:
#             shp_camels.iloc[[indi[j]]].boundary.plot(ax=ax[0], color=colors[j])
#         ax[0].text(shp_camels.iloc[[indi[j]]]['lon_cen'], shp_camels.iloc[[indi[j]]]['lat_cen'], indi[j], color='w')
#
#     ind3 = nest_index_ind[ind1]
#     indi = [ind1] + ind3
#     for j in range(len(indi)):
#         if j == 0:
#             shp_camels.iloc[[indi[j]]].plot(ax=ax[1], color=colors[j])
#         else:
#             shp_camels.iloc[[indi[j]]].boundary.plot(ax=ax[1], color=colors[j])
#         ax[1].text(shp_camels.iloc[[indi[j]]]['lon_cen'], shp_camels.iloc[[indi[j]]]['lat_cen'], indi[j], color='w')
#     pdf.savefig(fig)
# pdf.close()


########################################################################################################################
# split nest streamflow
# Example: 1 contains 2 and 3, and 2 contains 3
# Q_1_new = Q_1_raw - Q_2_raw
# Q_2_new = Q_2_raw - Q_3_raw
# Q_3_new = Q_3_raw


os.makedirs(outpath_CAMELS_q, exist_ok=True)
splitbasins = [nest_index_ind[n][0] for n in nest_index_ind]

for i in range(len(shp_camels)):
    infilei = f"{inpath_CAMELS_q}/{shp_camels.iloc[i]['hru_id']:08}_streamflow_qc.txt"
    outfilei = f"{outpath_CAMELS_q}/{shp_camels.iloc[i]['hru_id']:08}_streamflow_qc.txt"
    if not i in splitbasins:
        _ = shutil.copy(infilei, outfilei)
    else:
        dfi = pd.read_csv(infilei, delim_whitespace=True, header=None)
        indi = splitbasins.index(i)
        for j in range(len(nest_index_ind[indi])-1):
            infilej = f"{inpath_CAMELS_q}/{shp_camels.iloc[nest_index_ind[indi][j+1]]['hru_id']:08}_streamflow_qc.txt"
            dfj = pd.read_csv(infilej, delim_whitespace=True, header=None)
            for q in range(len(dfj)):
                if dfj.loc[q][4] >= 0:
                    ind = np.where((dfi[1].values == dfj.iloc[q][1]) & (dfi[2].values == dfj.iloc[q][2]) & (dfi[3].values == dfj.iloc[q][3]))[0]
                    if len(ind) == 1 and dfi.loc[ind][4].values>=0:
                        dfi.loc[ind, 4] = dfi.loc[ind][4] - dfj.loc[q][4]
        dfi.to_csv(outfilei, index=False, header=False, sep=' ')


########################################################################################################################
# post-process these streamflow files

for i in range(len(shp_camels)):
    infilei = f"{outpath_CAMELS_q}/{shp_camels.iloc[i]['hru_id']:08}_streamflow_qc.txt"
    outfilei = f"{outpath_CAMELS_q}/{shp_camels.iloc[i]['hru_id']:08}_Q_postprocess.csv"

    df_q_in = pd.read_csv(infilei, delim_whitespace=True, header=None)
    years = df_q_in[1].values
    months = df_q_in[2].values
    days = df_q_in[3].values
    dates = [f'{years[i]}-{months[i]:02}-{days[i]:02}' for i in range(len(years))]
    dates = pd.to_datetime(dates)
    q_obs = df_q_in[4].values * 0.028316847 # cfs to cms
    q_obs[q_obs<0] = -9999.0
    df = pd.DataFrame({'Date': dates, 'Runoff_cms': q_obs})

    # fill possible missing values
    df.set_index('Date', inplace=True)
    date_range = pd.date_range(start='1980-01-01', end='2014-12-31', freq='D')
    df = df.reindex(date_range)
    df.fillna(-9999, inplace=True)
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'Date'})

    df.to_csv(outfilei, index=False)

    idi = df_q_in[0].iloc[0]
    df = df.rename(columns={'Runoff_cms':idi})
    if i == 0:
        dfall = df
    else:
        if len(dfall) != len(df):
            print('Warning! Different lengths')
        dfall = pd.concat([dfall, df[ idi ]], axis=1)

outfileall = f"{outpath_CAMELS_q}/All_CAMELS_Q_postprocess.csv"
dfall.to_csv(outfileall, index=False)


########################################################################################################################
# link mesh file basin to CAMELS shp_camels because mesh shapefile order and shp order are different

loop_mesh = np.arange(len(ds_mesh.elementCount))
loop_shp = shp_camels.copy()
index_camels = np.zeros([len(ds_mesh.elementCount), 2], dtype=int) # [mesh index, camels index]

flag = 0
while len(loop_mesh) > 0:
    for i in range(len(loop_mesh)):
        point = Point(ds_mesh['centerCoords'].values[loop_mesh[i], 0] - 360, ds_mesh['centerCoords'].values[loop_mesh[i], 1])
        flagcontain = loop_shp.geometry.contains(point)
        indexi = flagcontain.index.values[np.where(flagcontain)[0]]
        if len(indexi) == 1:
            index_camels[flag, 0] = loop_mesh[i]
            index_camels[flag, 1] = indexi
            flag = flag+1
            loop_mesh = np.delete(loop_mesh, np.where(loop_mesh==loop_mesh[i]))
            loop_shp = loop_shp.drop(index=indexi)
            break
        if len(indexi) == 0: # sometimes the center point is outside the shapefile
            # find the nearest point
            diff = (point.x - loop_shp.geometry.centroid.x)**2 + (point.y - loop_shp.geometry.centroid.y)**2
            indmin = np.argmin(diff)
            indexi = loop_shp.index.values[indmin]
            index_camels[flag, 0] = loop_mesh[i]
            index_camels[flag, 1] = indexi
            flag = flag+1
            loop_mesh = np.delete(loop_mesh, np.where(loop_mesh==loop_mesh[i]))
            loop_shp = loop_shp.drop(index=indexi)

index_camels = index_camels[np.argsort(index_camels[:,0]), :]

# # check whether match is correct
# import matplotlib.pyplot as plt
# shp_mesh = gpd.read_file('/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/shp/ESMFmesh_ctsm_HCDN_nhru_final_671.buff_fix_holes_polygons_simplified_5e-4_split_nested.shp')
# for index in index_camels:
#     fig, ax = plt.subplots(figsize=[6, 4])
#     shp_mesh.loc[[index[0]]].plot(ax=ax, color='k')
#     shp_camels.loc[[index[1]]].boundary.plot(ax=ax, color='r')

########################################################################################################################
# create basin info output file

basin_info = pd.DataFrame()
for index in index_camels:
    shpi = shp_camels.loc[[index[1]]].copy()
    shpi = shpi.drop(columns=['geometry'])

    shpi = shpi.reindex()
    shpi['distdiff'] = np.nanmin(diff)
    id = shpi['hru_id'].values[0]

    shpi['mesh_id'] = index[0]
    shpi['shp_id'] = index[1]
    if index[1] in nest_index_all:
        shpi['shp_id_allup'] = ','.join([str(n) for n in nest_index_all[index[1]]])
        shpi['shp_id_indup'] = ','.join([str(n) for n in nest_index_ind[index[1]]])

        f = []
        for j in nest_index_all[index[1]]:
            idj = shp_camels.loc[j]['hru_id']
            f.append(f'{inpath_CAMELS_q}/{idj:08}_streamflow_qc.txt')
        shpi['file_obsQ_allup'] = ','.join(f)

        f = []
        for j in nest_index_ind[index[1]]:
            idj = shp_camels.loc[j]['hru_id']
            f.append(f'{inpath_CAMELS_q}/{idj:08}_streamflow_qc.txt')
        shpi['file_obsQ_indup'] = ','.join(f)

    else:
        shpi['shp_id_allup'] = ''
        shpi['shp_id_indup'] = ''
        shpi['file_obsQ_allup'] = ''
        shpi['file_obsQ_indup'] = ''

    shpi['file_obsQ'] = f'{inpath_CAMELS_q}/{id:08}_streamflow_qc.txt'

    basin_info = pd.concat([basin_info, shpi])

basin_info = basin_info.reindex()
basin_info.to_csv(outfile_info, index=False)