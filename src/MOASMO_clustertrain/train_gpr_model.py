# concatenate basin attributes to parameter inputs

import glob
import os
import sys

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append("../MOASMO_support")
from MOASMO_parameters import *

sys.path.append("/glade/u/home/guoqiang/CTSM_repos/ctsm_optz/MO-ASMO/src")
import NSGA2

# load basin info
infile_basin_info = f"/glade/work/guoqiang/CTSM_CAMELS/data_mesh_surf/HillslopeHydrology/CAMELS_level1_basin_info.csv"
df_info = pd.read_csv(infile_basin_info)

# load cluster info
infile = "../camels_cluster/Manuela_Brunner_2020/flood_cluster_memberships_CAMELS.txt"
df_cluster = pd.read_csv(infile)
df_cluster = df_cluster.rename(
    columns={"Camels_IDs": "hru_id", "flood_cluster": "clusters"}
)
df_cluster2 = pd.DataFrame()

for id in df_info["hru_id"].values:
    dfi = df_cluster.loc[df_cluster["hru_id"] == id]
    df_cluster2 = pd.concat([df_cluster2, dfi])

df_cluster2.sel_index = np.arange(len(df_cluster2))
df_cluster = df_cluster2
del df_cluster2
df_cluster["clusters"] = df_cluster["clusters"] - 1  # starting from 0

if np.any(df_info["hru_id"].values - df_cluster['hru_id'].values != 0):
    print("Mistmatch between basins and clusters")
else:
    print("basins and clusters match")


# load basin attributes for this cluster
attfiles = [
    "/glade/campaign/ral/hap/common/camels/camels_geol.txt",
    "/glade/campaign/ral/hap/common/camels/camels_hydro.txt",
    "/glade/campaign/ral/hap/common/camels/camels_clim.txt",
    "/glade/campaign/ral/hap/common/camels/camels_loc_topo.txt",
    "/glade/campaign/ral/hap/common/camels/camels_soil.txt",
    "/glade/campaign/ral/hap/common/camels/camels_vege.txt",
]

for i in range(len(attfiles)):
    dfi = pd.read_csv(attfiles[i], delimiter=";")
    if i == 0:
        df_att = dfi
    else:
        df_att = pd.merge(df_att, dfi, on="gauge_id")

df_att = df_att.loc[df_att["gauge_id"].isin(df_info["hru_id"].values)]
df_att.sel_index = np.arange(len(df_att))
if np.any(df_att["gauge_id"].values != df_info["hru_id"].values):
    sys.exit("Mismatch between att and info ids")
else:
    df_att["hru_id"] = df_info["hru_id"].values

print("All columns")
print(df_att.columns)


# select a cluster
sel_cluster = 2
sel_index = df_cluster["clusters"].values == sel_cluster
print('Number', np.sum(sel_index))
print(np.where(sel_index)[0])

# plt.figure(figsize=[5, 3])
# plt.scatter(df_info["lon_cen"], df_info["lat_cen"])
# plt.scatter(df_info["lon_cen"].values[sel_index], df_info["lat_cen"].values[sel_index])
# plt.title(f"Number of basins: {np.sum(sel_index)}")
# plt.show()

# select basin information for this cluster
df_att = df_att[sel_index]
df_cluster = df_cluster[sel_index]
df_info = df_info[sel_index]

inpath_moasmo = "/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange"
basin_index = np.where(sel_index)[0]


# check whether parameter names are the same
first_parameters = None
all_same = True

for i in basin_index:
    file = (
        f"{inpath_moasmo}/level1_{i}_MOASMOcalib/param_sets/all_default_parameters.pkl"
    )
    df_defaparam = pd.read_pickle(file)
    if first_parameters is None:
        first_parameters = df_defaparam["Parameter"].values
    else:
        if not (df_defaparam["Parameter"].values == first_parameters).all():
            all_same = False
            break

if all_same:
    print("All 'Parameter' values are the same across all files.")
else:
    sys.exit("There are differences in 'Parameter' values between the files.")

print("Parameter names:", first_parameters)


# load parameter values from all basins

df_param = pd.DataFrame()
df_metric = pd.DataFrame()
param_names = df_defaparam["Parameter"].values  # exclude binded parameters

flag = 0
for i in basin_index:
    file_param = (
        f"{inpath_moasmo}/level1_{i}_MOASMOcalib/ctsm_outputs/iter0_all_meanparam.csv"
    )
    file_metric = (
        f"{inpath_moasmo}/level1_{i}_MOASMOcalib/ctsm_outputs/iter0_all_metric.csv"
    )

    df1 = pd.read_csv(file_param)
    df1 = df1[param_names]

    df2 = pd.read_csv(file_metric)

    df2["basin_num"] = flag
    df2["basin_id"] = i
    df2["hru_id"] = df_info["hru_id"].values[flag]

    if len(df_param) == 0:
        df_param = df1
        df_metric = df2
    else:
        df_param = pd.concat([df_param, df1])
        df_metric = pd.concat([df_metric, df2])

    flag = flag + 1

df_basinid = df_metric[["basin_num", "basin_id", "hru_id"]]
df_metric = df_metric[["metric1", "metric2"]]

print("Number of parameter sets:", len(df_param))

# parameter upper/lower bound
param_lb_mean = np.array([np.nanmean(v) for v in df_defaparam["Lower"]])
param_ub_mean = np.array([np.nanmean(v) for v in df_defaparam["Upper"]])

# att names adjusted to raw CAMELS names

att_Feng2020 = {
    "mean_elev": {"description": "Catchment mean elevation", "unit": "m"},
    "mean_slope": {"description": "Catchment mean slope", "unit": "m/km"},
    "area_gauges2": {"description": "Catchment area (GAGESII estimate)", "unit": "km2"},
    "frac_forest": {"description": "Forest fraction", "unit": "—"},
    "lai_max": {
        "description": "Maximum monthly mean of the leaf area index",
        "unit": "—",
    },
    "lai_diff": {
        "description": "Difference between the maximum and minimum monthly mean of the leaf area index",
        "unit": "—",
    },
    "dom_land_cover_frac": {
        "description": "Fraction of the catchment area associated with the dominant land cover",
        "unit": "—",
    },
    "dom_land_cover": {"description": "Dominant land cover type", "unit": "—"},
    "root_depth_50": {
        "description": "Root depth at 50th percentile, extracted from a root depth distribution based on the International Geosphere-Biosphere Programme (IGBP) land cover",
        "unit": "m",
    },
    "soil_depth_statsgo": {"description": "Soil depth", "unit": "m"},
    "soil_porosity": {"description": "Volumetric soil porosity", "unit": "—"},
    "soil_conductivity": {
        "description": "Saturated hydraulic conductivity",
        "unit": "cm/hr",
    },
    "max_water_content": {"description": "Maximum water content", "unit": "m"},
    "geol_1st_class": {
        "description": "Most common geologic class in the catchment basin",
        "unit": "—",
    },
    "geol_2nd_class": {
        "description": "Second most common geologic class in the catchment basin",
        "unit": "—",
    },
    "geol_porostiy": {"description": "Subsurface porosity", "unit": "—"},
    "geol_permeability": {"description": "Subsurface permeability", "unit": "m2"},
}

att_Xie2021 = {
    "p_mean": {"description": "Mean daily precipitation", "unit": "mm"},
    "pet_mean": {
        "description": "Mean daily potential evapotranspiration",
        "unit": "mm",
    },
    "aridity": {"description": "Ratio of mean PET to mean precipitation", "unit": "—"},
    "p_seasonality": {
        "description": "Seasonality and timing of precipitation",
        "unit": "mm",
    },
    "frac_snow": {
        "description": "Fraction of precipitation falling on days with temperatures below 0 °C",
        "unit": "—",
    },
    "high_prec_freq": {
        "description": "Frequency of high-precipitation days (≥ 5 times mean daily precipitation)",
        "unit": "—",
    },
    "high_prec_dur": {
        "description": "Average duration of high-precipitation events",
        "unit": "day",
    },
    "low_prec_freq": {"description": "Frequency of dry days (≤ 1 mm/day)", "unit": "—"},
    "low_prec_dur": {"description": "Average duration of dry periods", "unit": "day"},
    "mean_elev": {"description": "Catchment mean elevation", "unit": "m"},
    "mean_slope": {"description": "Catchment mean slope", "unit": "m/km"},
    "area_gauges2": {"description": "Catchment area", "unit": "km2"},
    "frac_forest": {"description": "Forest fraction", "unit": "—"},
    "lai_max": {"description": "Maximum monthly mean of leaf area index", "unit": "—"},
    "lai_diff": {
        "description": "Difference between the max. and min. mean of the leaf area index",
        "unit": "—",
    },
    "dom_land_cover_frac": {
        "description": "Fraction of the catchment area associated with the dominant land cover",
        "unit": "—",
    },
    "dom_land_cover": {"description": "Dominant land cover type", "unit": "—"},
    "soil_depth_pelletier": {
        "description": "Depth to bedrock (maximum 50 m)",
        "unit": "m",
    },
    "soil_depth_statsgo": {"description": "Soil depth (maximum 1.5 m)", "unit": "m"},
    "soil_porosity": {"description": "Volumetric porosity", "unit": "—"},
    "soil_conductivity": {
        "description": "Saturated hydraulic conductivity",
        "unit": "cm/hr",
    },
    "max_water_content": {
        "description": "Maximum water content of the soil",
        "unit": "m",
    },
    "sand_frac": {"description": "Fraction of sand in the soil", "unit": "—"},
    "silt_frac": {"description": "Fraction of silt in the soil", "unit": "—"},
    "clay_frac": {"description": "Fraction of clay in the soil", "unit": "—"},
    "carbonate_rocks_frac": {
        "description": "Fraction of Carbonate sedimentary rocks",
        "unit": "—",
    },
    "geol_permeability": {"description": "Surface permeability (log10)", "unit": "m2"},
}

# Prepare model input and output

useattrs0 = list(att_Xie2021.keys())
useattrs = []
for i in useattrs0:
    if not isinstance(df_att.iloc[0][i], str):
        useattrs.append(i)


print("The number of attributes used:", len(useattrs))


df_input = df_param.copy()
df_input["hru_id"] = df_basinid["hru_id"]
df_input = df_input.merge(df_att[useattrs + ["hru_id"]], on="hru_id", how="left")
df_input = df_input.drop(["hru_id"], axis=1)

inputnames = list(df_param.columns) + useattrs
print("Training input names:", inputnames)

df_output = df_metric.copy()

x = df_input[inputnames].values
y = df_output[["metric1", "metric2"]].values

print("Input shape:", x.shape)
print("Output shape:", y.shape)



file_sm_gpr_all = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/emulators/emulator_gpr_AllSample_allbasin_cluster{sel_cluster}'
if os.path.isfile(file_sm_gpr_all):
    with open(file_sm_gpr_all, 'rb') as file:
        sm_gpr_all = pickle.load(file)
else:

    # train over all basins: GRP
    xlb_mean = np.min(x, axis=0)
    xub_mean = np.max(x, axis=0)
    
    xlb_mean[: len(param_lb_mean)] = param_lb_mean
    xub_mean[: len(param_ub_mean)] = param_ub_mean
    
    ind = np.where((xlb_mean == 0) & (xub_mean == 0))
    xlb_mean[ind] = -0.01
    xub_mean[ind] = 0.01
    
    # use sparse samples
    np.random.seed(1234567890)
    
    # define hyperparameter
    alpha = 1e-3
    leng_lb = 1e-3
    leng_ub = 1e3
    nu = 2.5
    sm_gpr_all = gp.GPR_Matern(
        x,
        y,
        x.shape[1],
        y.shape[1],
        x.shape[0],
        xlb_mean,
        xub_mean,
        alpha=alpha,
        leng_sb=[leng_lb, leng_ub],
        nu=nu,
    )

    pickle.dump(sm_gpr_all, open(file_sm_gpr_all, 'wb'))



file_sm_gpr_optmz = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/emulators/emulator_gpr_AllSample_allbasin_cluster{sel_cluster}_optmz_outputs.npz'
if os.path.isfile(file_sm_gpr_optmz):
    d = np.load(file_sm_gpr_optmz)
    bestx_sm_all_gpr = d['bestx_sm_all_gpr']
    besty_sm_all_gpr = d['besty_sm_all_gpr']

else:

    bestx_sm_all_gpr = []
    besty_sm_all_gpr = []
    
    for tarbasin in range(len(df_info)):
        print(tarbasin)
    
        index = np.where(df_basinid["basin_num"].values == tarbasin)[0]
        hruid = df_basinid["hru_id"].values[index[0]]
        attrvalues = df_att.loc[df_att["gauge_id"] == hruid][useattrs].values[0]
    
        x_tar = x[index, :]
        y_tar = y[index, :]
        xlb_mean = np.hstack([param_lb_mean, attrvalues])
        xub_mean = np.hstack([param_ub_mean, attrvalues])
    
        nInput = len(xlb_mean)
        nOutput = 2
    
        pop = 100
        gen = 100
        crossover_rate = 0.9
        mu = 20
        mum = 20
    
        bestx_sm, besty_sm, x_sm, y_sm = NSGA2.optimization(
            sm_gpr_all,
            nInput,
            nOutput,
            xlb_mean,
            xub_mean,
            pop,
            gen,
            crossover_rate,
            mu,
            mum,
        )
        # D = NSGA2.crowding_distance(besty_sm)
        # print('model sample number:', D.shape[0])
    
        bestx_sm_all_gpr.append(bestx_sm)
        besty_sm_all_gpr.append(besty_sm)

    np.savez_compressed(file_sm_gpr_optmz, bestx_sm_all_gpr=bestx_sm_all_gpr, besty_sm_all_gpr=besty_sm_all_gpr)