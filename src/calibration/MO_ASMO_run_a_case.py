# based on a CTSM case
# 1. clone a case
# 2. update parameters
# 3. run model
# 4. move outputs to another directory
# 5. delete cloned case

import os, sys, subprocess, pathlib
import numpy as np
import pandas as pd
import xarray as xr


def xmlquery_output(pathCTSM, keyword):
    os.chdir(pathCTSM)
    out = subprocess.run(f'./xmlquery {keyword}', shell=True, capture_output=True)
    out = out.stdout.decode().strip().split(' ')[-1]
    return out


def clone_CTSM_model_case(script_clone, source_modelcase, target_modelcase, target_cimeoutput=''):
    if len(target_cimeoutput) > 0:
        settings = f'--cime-output-root {target_cimeoutput}'
    else:
        settings = ''
    _ = subprocess.run(f"{script_clone} --case {target_modelcase} --clone {source_modelcase} {settings}", shell=True)
    os.chdir(target_modelcase)
    _ = subprocess.run('./xmlchange BUILD_COMPLETE=TRUE', shell=True)


def update_CTSM_parameter_ParamFile(param_names, param_values, path_CTSM_case, outfile_newparam):
    # param_names and param_values: list
    file_param_base = ''
    file_user_nl_clm = f'{path_CTSM_clone}/user_nl_clm'
    with open(file_user_nl_clm) as f:
        for line in f:
            line = line.strip()
            if line.startswith('paramfile'):
                file_param_base = line.split('=')[-1].strip().replace('\'', '')
    if len(file_param_base) == 0:
        infile_lndin = f'{path_CTSM_case}/Buildconf/clmconf/lnd_in'
        with open(infile_lndin, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('paramfile'):
                    file_param_base = line.split('=')[-1].strip().replace('\'', '')

    ds_param = xr.load_dataset(file_param_base)
    for i in range(len(param_names)):
        pn = param_names[i]
        vnew = param_values[i]
        if not pn in ds_param.data_vars:
            print(f'Error!!! Variable {pn} is not find in parameter file {file_param_base}!!!')
            sys.exit()
        else:
            vold = ds_param[pn].values
            print(
                f'  -- Updating parameter {pn}: old mean value is {np.nanmean(vold)}. New mean value {np.nanmean(vnew)}')
            ds_param[pn].values = vnew
    # write to new parameter file
    ds_param.to_netcdf(outfile_newparam, format='NETCDF3_CLASSIC')

def update_CTSM_parameter_NamelistFile(param_names, param_values, file_namelist):
    if len(param_names) > 0:
        with open(file_namelist, 'a') as f:
            print('# Writing new parameter values')
        for i in range(len(param_names)):
                f.write(f'{param_names[i]} = {param_values[i]}\n')
                print(f'  -- Writing parameter {param_names[i]} to namelist file. New value is {param_values[i]}')


def update_CTSM_parameter_SurfdataFile(param_names, param_values, path_CTSM_case, outfile_newsurfdata):

    file_surfdata = ''
    file_user_nl_clm = f'{path_CTSM_clone}/user_nl_clm'
    with open(file_user_nl_clm) as f:
        for line in f:
            line = line.strip()
            if line.startswith('fsurdat'):
                file_surfdata = line.split('=')[-1].strip().replace('\'', '')
    if len(file_surfdata) == 0:
        infile_lndin = f'{path_CTSM_case}/Buildconf/clmconf/lnd_in'
        with open(infile_lndin, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('fsurdat'):
                    file_surfdata = line.split('=')[-1].strip().replace('\'', '')

    ds_surf = xr.load_dataset(file_surfdata)
    for i in range(len(param_names)):
        pn = param_names[i]
        vnew = param_values[i]
        if not pn in ds_surf.data_vars:
            print(f'Error!!! Variable {pn} is not find in parameter file {file_surfdata}!!!')
            sys.exit()
        else:
            vold = ds_surf[pn].values
            print(
                f'  -- Updating parameter {pn}: old mean value is {np.nanmean(vold)}. New mean value {np.nanmean(vnew)}')
            ds_surf[pn].values = vnew

    ds_surf.to_netcdf(outfile_newsurfdata, format='NETCDF3_CLASSIC')


def update_ctsm_parameters(path_CTSM_case, file_parameter_set):

    file_user_nl_clm = f'{path_CTSM_case}/user_nl_clm'
    df_calibparam = pd.read_csv(file_parameter_set)

    dfi = df_calibparam.loc[df_calibparam['Source'] == 'Namelist']
    if len(dfi) > 0:
        param_names = dfi['Parameter'].values
        param_values = dfi['Value'].values
        update_CTSM_parameter_NamelistFile(param_names, param_values, file_user_nl_clm)

    dfi = df_calibparam.loc[df_calibparam['Source'] == 'Param']
    if len(dfi) > 0:
        outfile_newparam = f'{path_CTSM_case}/updated_parameter.nc'
        param_names = dfi['Parameter'].values
        param_values = dfi['Value'].values
        update_CTSM_parameter_ParamFile(param_names, param_values, path_CTSM_case, outfile_newparam)
        # add to the name list file
        with open(file_user_nl_clm, 'a') as f:
            f.write(f"paramfile='{outfile_newparam}'\n")

    dfi = df_calibparam.loc[df_calibparam['Source'] == 'Surfdata']
    if len(dfi) > 0:
        outfile_newsurfdata = f'{path_CTSM_case}/updated_surfdata.nc'
        param_names = dfi['Parameter'].values
        param_values = dfi['Value'].values
        update_CTSM_parameter_SurfdataFile(param_names, param_values, path_CTSM_case, outfile_newsurfdata)
        # add to the name list file
        with open(file_user_nl_clm, 'a') as f:
            f.write(f"fsurdat='{outfile_newsurfdata}'\n")

    print('Successfully update parameters!')

# input arguments
script_clone = sys.argv[0]
path_CTSM_base = sys.argv[1]
file_parameter_set = sys.argv[2] # three columns: ['Parameter', 'Value', 'Source']
path_archive = sys.argv[3]
caseflag = sys.argv[4]

delete_clone = True

# clone case
path_CTSM_base = str(pathlib.Path(path_CTSM_base))
path_CTSM_clone = f'{path_CTSM_base}_{caseflag}'
clone_case_name = pathlib.Path(path_CTSM_clone).name
clone_CTSM_model_case(script_clone, path_CTSM_base, path_CTSM_clone)

# change parameters, which won't affect files of path_CTSM_base
update_ctsm_parameters(path_CTSM_clone, file_parameter_set)

# run model in "no-batch" mode
os.chdir(path_CTSM_clone)
subprocess.run('./case.submit --no-batch', shell=True)

# move output to the archive folder
output_dir = xmlquery_output(path_CTSM_clone, 'DOUT_S_ROOT')
target_dir = f'{path_archive}/{caseflag}'
if os.path.isdir(target_dir):
    print(f'Warning!!! {target_dir} exists before moving archived files')
    _ = os.system(f'rm -r {target_dir}')
_ = subprocess.run(f'mv {output_dir} {target_dir}', shell=True)
_ = subprocess.run(f'cp {file_parameter_set} {target_dir}', shell=True)

# delete cloned cases to save space
if delete_clone:
    cimeoutroot = xmlquery_output(path_CTSM_clone, 'CIME_OUTPUT_ROOT')
    _ = subprocess.run(f'rm -r f{cimeoutroot}/{clone_case_name}')
    _ = subprocess.run(f'rm -r {path_CTSM_clone}')
