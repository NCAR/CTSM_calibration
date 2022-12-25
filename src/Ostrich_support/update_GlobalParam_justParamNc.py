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

import os
import netCDF4 as nc4
import numpy as np
import sys, subprocess, shutil, pathlib

print('#'* 50)
print('Updating parameters ...')
print('#'* 50)

########################################################################################################################
# Input arguments
if len(sys.argv) != 5:
    print("The number of input argument is wrong!")
    sys.exit(0)

infile_multiplier = sys.argv[1]  # multipliers of parameters
infile_oldparam = sys.argv[2]  # old (or reference or base ...) parameter file
outfile_newparam = sys.argv[3]  # output updated param file = infile_oldparam * infile_multiplier
file_user_nl_clm = sys.argv[4] # let CLM use updated parameter file

print('infile_multiplier:', infile_multiplier)
print('infile_oldparam:', infile_oldparam)
print('outfile_newparam:', outfile_newparam)

########################################################################################################################
# Read parameter names and multipliers

# read multipliers and parameter names
param_multipliers = []
param_var_names = [] # variable name in param file
with open(infile_multiplier, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('!') and not line.startswith("'"):
            splits = line.split('|')
            param_var_names.append(splits[0].strip())
            param_multipliers.append(float(splits[1].strip()))

########################################################################################################################
# update parameters

# # don't use xarray which will cause error when running CTSM
# ds_param = xr.load_dataset(infile_oldparam)
# for pn, pm in zip(param_names, param_multipliers):
#     if not pn in ds_param.data_vars:
#         print(f'Error!!! Variable {pn} is not find in parameter file {infile_oldparam}!!!')
#         sys.exit()
#     else:
#         vold = ds_param[pn].values.mean()
#         ds_param[pn].values = vold * pm
#         vnew = ds_param[pn].values.mean()
#         print(f'  -- Updating parameter {pn}: old mean value {vold} * multiplier {pm} = new mean value {vnew}')
# ds_param.to_netcdf(outfile_newparam)


# apply multipliers to existing HRU values
outpath_newparam = str(pathlib.Path(outfile_newparam).parent)
os.makedirs(outpath_newparam, exist_ok=True)
_ = shutil.copyfile(infile_oldparam, outfile_newparam)

dataset_pattern = nc4.Dataset(infile_oldparam ,'r')
dataset = nc4.Dataset(outfile_newparam ,'r+')

for i in range(len(param_var_names)):
    var_name = param_var_names[i]
    marr = dataset.variables[var_name][:]
    vold = np.nanmean(marr)
    pm = param_multipliers[i]
    arr_value = marr.data * pm
    dataset.variables[var_name][:] = np.ma.array(arr_value, mask=np.ma.getmask(marr), fill_value=marr.get_fill_value())
    vnew = np.nanmean(dataset.variables[var_name][:])
    print(f'  -- Updating parameter {var_name}: old mean value {vold} * multiplier {pm} = new mean value {vnew}')

dataset.close()
dataset_pattern.close()

########################################################################################################################

# write filename
with open(file_user_nl_clm, 'r') as f:
    lines = f.readlines()

flag = True
for line in lines:
    if outfile_newparam in line:
        flag = False
        break

if flag == True:
    with open(file_user_nl_clm, 'a') as f:
        f.write(f"paramfile='{outfile_newparam}'\n")


print('#'* 50)
print('Successfully update parameters!')
print('#'* 50)
