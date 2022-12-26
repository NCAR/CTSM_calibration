# CTSM uses a global parameter as a default setting, e.g., /glade/p/cesmdata/cseg/inputdata/lnd/clm2/paramdata/ctsm51_params.c211112.nc
# This script calculate new_parameter = default_parameter * multipliers

# # using xarray to update parameters is also feasible. Must using format='NETCDF3_CLASSIC' to save netcdf. For paramfile, using NETCDF4 is wrong. Other files are not necessarily wrong.
# infile = '/glade/u/home/guoqiang/CTSM_cases/CAMELS_Calib/CAMELS_LumpCalib_Ostrich/tempdir/ctsm51_params.c211112_Ostrich.nc-backup'
# outfile = '/glade/u/home/guoqiang/CTSM_cases/CAMELS_Calib/CAMELS_LumpCalib_Ostrich/tempdir/ctsm51_params.c211112_Ostrich.nc'
# ds = xr.load_dataset(infile)
# encoding = {}
# for v in ds.data_vars:
#     ec = ds[v].encoding
#     if not '_FillValue' in ec:
#         ec['_FillValue'] = None
#     if not 'coordinates' in ec:
#         ec['coordinates'] = None
#     ds[v].encoding = ec
# ds.to_netcdf(outfile, format='NETCDF3_CLASSIC')

import xarray as xr
import pandas as pd
import numpy as np
import sys


def change_param_value(vold, factor, method):
    vold_mean = np.nanmean(vold)
    if method == 'Multiplicative':
        vnew = vold * factor
    elif method == 'Additive':
        vnew = vold + factor
    else:
        sys.exit('Unknown method!')
    vnew_mean = np.nanmean(vnew)
    return vold_mean, vnew, vnew_mean




print('#'* 50)
print('Updating parameters ...')
print('#'* 50)

########################################################################################################################
# Input arguments
if len(sys.argv) != 4:
    print("The number of input argument is wrong!")
    sys.exit(0)


infile_param_info = sys.argv[1]  # multipliers of parameters
inpath_CTSMcase = sys.argv[2] # CTSM case
infile_factor_value = sys.argv[3] # factor value

print('infile_param_info:', infile_param_info)
print('inpath_CTSMcase:', inpath_CTSMcase)
print('infile_factor_value:', infile_factor_value)

########################################################################################################################
# read parameter information (names, methods (e.g., multiplicative _mtp or additive _add, and factor values)

df_calibparam = pd.read_csv(infile_param_info)
df_calibparam['factor'] = 0.0

# read parameter factors
with open(infile_factor_value, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('!') and not line.startswith("'"):
            splits = line.split('|')
            df_calibparam.loc[df_calibparam['Parameter'] == splits[0].strip(), 'factor'] = float(splits[1].strip())

########################################################################################################################
# check if there is any binded variable
# if there is, add binded variables to df_calibparam

df_bind = pd.DataFrame()
for i in range(len(df_calibparam)):
    bindvari = df_calibparam.iloc[i]['Binding']
    if bindvari != 'None':
        bindvari = bindvari.split(',')
        for bv in bindvari:
            dftmp = df_calibparam.iloc[[i]].copy()
            dftmp['Parameter'] = bv
            df_bind = pd.concat([df_bind, dftmp])

df_calibparam = pd.concat([df_calibparam, df_bind])
df_calibparam.index = np.arange(df_calibparam)

########################################################################################################################
# Read parameter and file information


# CTSM files
file_user_nl_clm = f'{inpath_CTSMcase}/user_nl_clm'
with open(file_user_nl_clm, 'r') as f:
    lines_nlclm = f.readlines()


# flag: whether to change some files
change_param = False
change_surfdata = False
change_nlclm = False

########################################################################################################################
# update parameters

## 1. parameter file

df_param1 = df_calibparam[df_calibparam['Source']=='Param']
if len(df_param1) > 0:
    file_param_base = df_calibparam[df_calibparam['Source']=='Param']['Source_file'].values[0]
    outfile_newparam = df_calibparam[df_calibparam['Source'] == 'Param']['OstrichTrial_file'].values[0]
    ds_param = xr.load_dataset(file_param_base)
    for i in range(len(df_param1)):
        pn = df_param1.iloc[i]['Parameter']
        pm = df_param1.iloc[i]['factor']
        if not pn in ds_param.data_vars:
            print(f'Error!!! Variable {pn} is not find in parameter file {file_param_base}!!!')
            sys.exit()
        else:
            method = df_calibparam.loc[df_calibparam['Parameter'] == pn]['Method'].values
            bindvar = df_calibparam.loc[df_calibparam['Parameter'] == pn]['Binding'].values
            vold = ds_param[pn].values
            vold_mean, vnew, vnew_mean = change_param_value(vold, pm, method)
            ds_param[pn].values = vnew
            print(f'  -- Updating parameter {pn}: old mean value {vold_mean} * multiplier {pm} = new mean value {vnew_mean}')
            change_param = True

if change_param == True:
    ds_param.to_netcdf(outfile_newparam, format='NETCDF3_CLASSIC')

## 2. surface data file
df_param2 = df_calibparam[df_calibparam['Source']=='Surfdata']
if len(df_param2) > 0:
    file_surfdata_base = df_calibparam[df_calibparam['Source']=='Surfdata']['Source_file'].values[0]
    outfile_newsurf = df_calibparam[df_calibparam['Source']=='Surfdata']['OstrichTrial_file'].values[0]
    ds_surf = xr.load_dataset(file_surfdata_base)
    for i in range(len(df_param2)):
        pn = df_param2.iloc[i]['Parameter']
        pm = df_param2.iloc[i]['factor']
        if not pn in ds_surf.data_vars:
            print(f'Error!!! Variable {pn} is not find in parameter file {file_surfdata_base}!!!')
            sys.exit()
        else:
            method = df_calibparam.loc[df_calibparam['Parameter'] == pn]['Method'].values
            bindvar = df_calibparam.loc[df_calibparam['Parameter'] == pn]['Binding'].values
            vold = ds_surf[pn].values
            vold_mean, vnew, vnew_mean = change_param_value(vold, pm, method)
            ds_surf[pn].values = vnew
            print(f'  -- Updating parameter {pn}: old mean value {vold_mean} * multiplier {pm} = new mean value {vnew_mean}')
            change_surfdata = True

if change_surfdata == True:
    ds_surf.to_netcdf(outfile_newsurf, format='NETCDF3_CLASSIC')

# check if new parameter name is already in the user_nl_clm file
for i in range(len(lines_nlclm)):
    if outfile_newparam in lines_nlclm[i]:
        # already in ...
        break
    if i == len(lines_nlclm) - 1:
        # not in
        lines_nlclm.append(f"\nparamfile='{outfile_newparam}'\n")
        change_nlclm = True

## 3. namelist
df_param3 = df_calibparam[df_calibparam['Source']=='Namelist']
if len(df_param3) > 0:
    file_lndin_base = df_calibparam[df_calibparam['Source'] == 'Namelist']['Source_file'].values[0]
    with open(file_lndin_base, 'r') as f:
        lines_lndin = f.readlines()
        varname_lndin = [l.split('=')[0].strip() for l in lines_lndin]
    for i in range(len(df_param3)):
        pn = df_param3.iloc[i]['Parameter']
        pm = df_param3.iloc[i]['factor']
        if not pn in varname_lndin:
            print(f'Error!!! Variable {pn} is not find in parameter file {file_lndin_base}!!!')
            sys.exit()
        else:
            vold = np.nan
            for line in lines_lndin:
                line = line.strip()
                if line.startswith(pn):
                    vold = np.array(float(line.split('=')[-1].strip().replace('\'', '').split('d')[0]))
                    break
            method = df_calibparam.loc[df_calibparam['Parameter'] == pn]['Method'].values
            bindvar = df_calibparam.loc[df_calibparam['Parameter'] == pn]['Binding'].values
            vold_mean, vnew, vnew_mean = change_param_value(vold, pm, method)
            # write to user_nl_clm
            flag = False
            for i in range(len(lines_nlclm)):
                line = lines_nlclm[i].strip()
                if line.startswith(pn):
                    lines_nlclm[i] = f'{pn} = {vnew}\n'
                    flag = True
            if flag == False:
                lines_nlclm.append(f'{pn} = {vnew}\n')
            print(f'  -- Updating parameter {pn}: old mean value {vold_mean} * multiplier {pm} = new mean value {vnew_mean}')
            change_nlclm = True

########################################################################################################################
# write to user_nl_clm with changes
if change_nlclm == True:
    with open(file_user_nl_clm, 'w') as f:
        for line in lines_nlclm:
            f.write(line)


print('#'* 50)
print('Successfully update parameters!')
print('#'* 50)


