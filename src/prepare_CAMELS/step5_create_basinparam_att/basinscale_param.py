# For the csv file containing all parameters and ranges, create single basin param file based on that

import xarray as xr
import numpy as np
import pandas as pd
import glob, os, sys

sys.path.append('../../MOASMO_support')
from MOASMO_parameters import read_save_load_all_default_parameters


infile_basin_info = f'/glade/work/guoqiang/CTSM_CAMELS/data_mesh_surf/HillslopeHydrology/CAMELS_level1_basin_info.csv'
df_info = pd.read_csv(infile_basin_info)

infile_param = f'../../parameter/CTSM_CAMELS_calibparam_2410.csv'
df_param = pd.read_csv(infile_param)
df_param

for basin_num in range(len(df_info)):

    idi = df_info.iloc[basin_num]['hru_id']

    paramfile_new = f'/glade/work/guoqiang/CTSM_CAMELS/data_paramcailb/ParamCalib_same4basins_{idi}.csv'
    
    if os.path.isfile(paramfile_new):
        continue
    
    df0 = df_param.copy()
    df0 = df0.rename(columns={'Default':'Default_glob', 'Lower':'Lower_glob', 'Upper':'Upper_glob'})

    # read default param
    path_CTSM_case = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/level1_{basin_num}'
    dfnew = read_save_load_all_default_parameters(infile_param, '', path_CTSM_case=path_CTSM_case, savefile=False)

    defa = np.array([np.mean(i) for i in dfnew['Value']])
    df0['Default'] = defa
    if np.any(defa > df0['Upper_glob'].values) or np.any(defa < df0['Lower_glob'].values):
        print('Warning! Bound problem')
    
    # df0['Lower'] = (defa/df0['Default_glob'].values) * df0['Lower_glob'].values
    # df0['Upper'] = (defa/df0['Default_glob'].values) * df0['Upper_glob'].values
    df0['Lower'] = df0['Default_glob']
    df0['Upper'] = df0['Default_glob']

    print('saving to', paramfile_new)
    df0.to_csv(paramfile_new)
