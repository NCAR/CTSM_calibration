# run a CTSM case for one iteration and one trial
# 1. clone a case
# 2. update parameters
# 3. run model
# 4. move outputs to another directory
# 5. delete cloned case

# The Derecho version can run many cases within one node to be efficient, but manual submission is needed


import os, sys, glob, subprocess, pathlib, random, time
import numpy as np
import pandas as pd
import xarray as xr
from mo_evaluation import mo_evaluate
import mo_evaluation_nonstandard
from MOASMO_parameters import get_parameter_from_Namelist_or_lndin
from xml_addcpubind import insert_cpu_bind

def read_parameter_csv(file_parameter_list):
    df_calibparam = pd.read_csv(file_parameter_list)

    for c in ['Upper', 'Lower', 'Factor', 'Value']:
        if c in df_calibparam.columns:
            if isinstance(df_calibparam.iloc[0][c], str):
                arr = []
                for i in range(len(df_calibparam)):
                    vi = df_calibparam.iloc[i][c]
                    if ',' in vi:
                        arr.append(np.array(vi.split(',')).astype(np.float64))
                    elif '[' in vi:
                        arr.append(np.array(vi.strip('[]').replace('\n', '').split(), dtype=np.float64))
                    else:
                        try:
                            arr.append(np.array([np.float64(vi)]))
                        except:
                            arr.append(np.array([-99999]))
                df_calibparam[c] = arr

    return df_calibparam

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
    _ = subprocess.run(f"{script_clone} --case {target_modelcase} --clone {source_modelcase} {settings} --keepexe", shell=True)
    os.chdir(target_modelcase)
    _ = subprocess.run('./xmlchange BUILD_COMPLETE=TRUE', shell=True)


def update_CTSM_parameter_ParamFile(param_names, param_values, path_CTSM_case, outfile_newparam):
    # param_names and param_values: list
    file_param_base = ''
    # file_user_nl_clm = f'{path_CTSM_clone}/user_nl_clm'
    # with open(file_user_nl_clm) as f:
    #     for line in f:
    #         line = line.strip()
    #         if line.startswith('paramfile'):
    #             file_param_base = line.split('=')[-1].strip().replace('\'', '')
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
            print(f'  -- Updating parameter {pn}: old mean value is {np.nanmean(vold)}. New mean value {np.nanmean(vnew)}')
            if vnew.size==1 and ds_param[pn].values.size>1:
                print(f'Warning! New parameter size=1, but raw parameter size={ds_param[pn].values.size}. Force the mean equal to the new parameter value.')
                m = np.nanmean(ds_param[pn].values)
                ds_param[pn].values = ds_param[pn].values - m + np.squeeze(vnew)
            else:
                ds_param[pn].values = np.squeeze(vnew)
    # write to new parameter file
    ds_param.to_netcdf(outfile_newparam, format='NETCDF3_CLASSIC')

def update_CTSM_parameter_NamelistFile(param_names, param_values, file_namelist):
    if len(param_names) > 0:
        with open(file_namelist, 'a') as f:
            print('# Writing new parameter values')
            for i in range(len(param_names)):
                    _ = f.write(f'{param_names[i]} = {np.squeeze(param_values[i])}\n')
                    print(f'  -- Writing parameter {param_names[i]} to namelist file. New value is {param_values[i]}')


def update_CTSM_parameter_SurfdataFile(param_names, param_values, path_CTSM_case, outfile_newsurfdata):

    file_surfdata = ''
    # file_user_nl_clm = f'{path_CTSM_clone}/user_nl_clm'
    # with open(file_user_nl_clm) as f:
    #     for line in f:
    #         line = line.strip()
    #         if line.startswith('fsurdat'):
    #             file_surfdata = line.split('=')[-1].strip().replace('\'', '')
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
            print(f'  -- Updating surfdata {pn}: old mean value is {np.nanmean(vold)}. New mean value {np.nanmean(vnew)}')
            if vnew.size==1 and ds_surf[pn].values.size>1:
                print(f'Warning! New surfdata size=1, but raw surfdata size={ds_surf[pn].values.size}. Force the mean equal to the new surfdata value.')
                m = np.nanmean(ds_surf[pn].values)
                ds_surf[pn].values[:] = ds_surf[pn].values - m + np.squeeze(vnew)
            else:
                ds_surf[pn].values[:] = np.squeeze(vnew)

    ds_surf.to_netcdf(outfile_newsurfdata, format='NETCDF3_CLASSIC')


def update_ctsm_parameters(path_CTSM_case, file_parameter_set):

    file_user_nl_clm = f'{path_CTSM_case}/user_nl_clm'
    df_calibparam = pd.read_pickle(file_parameter_set)
    #df_calibparam = read_parameter_csv(file_parameter_set)

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

if __name__ == '__main__':

    # input arguments
    script_clone = sys.argv[1]
    path_CTSM_base = sys.argv[2]
    file_parameter_set = sys.argv[3] # three columns: ['Parameter', 'Value', 'Source']
    path_archive = sys.argv[4]
    caseflag = sys.argv[5]
    date_start = sys.argv[6]
    date_end = sys.argv[7]
    ref_streamflow = sys.argv[8]
    add_flow_file = sys.argv[9]


    use_cpu_bind = False
    if use_cpu_bind == True:
        cpubind_path = sys.argv[10] 
        cpuuse = sys.argv[11] 
    else:
        cpubind_path = 'NA'
        cpuuse = 'NA'

    # script_clone = '/glade/u/home/guoqiang/CTSM_repos/CTSM/cime/scripts/create_clone'
    # path_CTSM_base = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest_LMWG/CAMELS_100'
    # file_parameter_set = '/glade/scratch/guoqiang/moasmo_test/param_sets/paramset_iter0_trial0.csv'
    # path_archive = '/glade/scratch/guoqiang/moasmo_test/ctsm_outputs'
    # caseflag = 'iter0_trial0'
    #
    # date_start = '1994-10-01'
    # date_end = '1998-09-30'
    # ref_streamflow = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest_LMWG/CAMELS_100_OstCalib/refdata/streamflow_data.csv'
    # add_flow_file = 'nofile'

    delete_clone = True
    overwrite_previous = False # delete previous simulation results in the archive folder
    
    ########################################################################################################################
    # check whether previous simulation exists
    
    simufiles = glob.glob(f'{path_archive}/{caseflag}/lnd/hist/*.nc')
    if len(simufiles) > 0 and overwrite_previous == False:
        print(f'There are .nc files in {path_archive}/{caseflag}/lnd/hist. No need to run model again')
        runmodel = False
    else:
        runmodel = True
    
    
    path_CTSM_base = str(pathlib.Path(path_CTSM_base))
    exclusive_flag = False
    
    if runmodel == True:
        ########################################################################################################################
        # clone case
        path_CTSM_clone = f'{path_CTSM_base}_{caseflag}'
        clone_case_name = pathlib.Path(path_CTSM_clone).name
        clone_CTSM_model_case(script_clone, path_CTSM_base, path_CTSM_clone)

        ########################################################################################################################
        # define CPU bind information (optional)
        if cpubind_path != 'NA':

            if cpuuse == 'Automatic':
                print('Do no set cpuuse')

            else:
            
                if cpuuse != 'NA':
                    # use the predefined cpu 
                    cpubind = f'{cpuuse}:{cpuuse}'
                    cpufile_use = f'{cpubind_path}/idlecpu_{cpuuse}'
                    cpufile_use2 = cpufile_use.replace('idlecpu','busycpu')
                    os.rename(cpufile_use, cpufile_use2)

                    cpunum = cpufile_use.split('_')[-1]
                    cpubind = f'{cpunum}:{cpunum}'
                    os.chdir(path_CTSM_clone)
                    insert_cpu_bind(f'{path_CTSM_clone}/env_mach_specific.xml',cpubind)

                    exclusive_flag = True
                
                else:
                    # find available cpu
                    trytimes = 10
                    
                    while (exclusive_flag==False) and (trytimes>0):
                        cpufiles = glob.glob(f'{cpubind_path}/idlecpu_*')
                        cpufiles.sort()
                        
                        # cpufile_use = cpufiles[0]
                        cpufile_use = random.choice(cpufiles)
                        cpufile_use2 = cpufile_use.replace('idlecpu','busycpu')
                        
                        if os.path.isfile(cpufile_use): # good. one cpu is found
                            os.rename(cpufile_use, cpufile_use2)
                         
                            cpunum = cpufile_use.split('_')[-1]
                            cpubind = f'{cpunum}:{cpunum}'
                            os.chdir(path_CTSM_clone)
                            insert_cpu_bind(f'{path_CTSM_clone}/env_mach_specific.xml',cpubind)

                            exclusive_flag = True

                        trytimes = trytimes - 1
                        time.sleep(1)

                    if exclusive_flag==False: # this means no available cpu is found
                        sys.exit('failed to find available CPU')

        ########################################################################################################################
        # change parameters, which won't affect files of path_CTSM_base
        update_ctsm_parameters(path_CTSM_clone, file_parameter_set)

        ########################################################################################################################
        # run model in "no-batch" mode
        os.chdir(path_CTSM_clone)
        subprocess.run('./case.submit --no-batch', shell=True)

        ########################################################################################################################
        # move output to the archive folder
        output_dir = xmlquery_output(path_CTSM_clone, 'DOUT_S_ROOT')
        target_dir = f'{path_archive}/{caseflag}'
        os.makedirs(str(pathlib.Path(target_dir).parent), exist_ok=True)
        if os.path.isdir(target_dir):
            print(f'Warning!!! {target_dir} exists before moving archived files')
            _ = os.system(f'rm -r {target_dir}')

        _ = subprocess.run(f'mv {output_dir} {target_dir}', shell=True)
        _ = subprocess.run(f'cp {file_parameter_set} {target_dir}', shell=True)

        # delete large restart files
        restfiles = glob.glob(f'{target_dir}/rest/*/*.nc')
        for f in restfiles:
            if not '.clm2.r.' in f:
                _ = os.system(f'rm {f}')

        # delete log files
        logfolder = f'{target_dir}/log'
        _ = os.system(f'rm {logfolder}/*')

        # delete cloned cases to save space
        if delete_clone:
            os.chdir(path_CTSM_base)
            RUNDIR = xmlquery_output(path_CTSM_clone, 'RUNDIR')
            _ = subprocess.run(f'rm -r {str(pathlib.Path(RUNDIR).parent)}', shell=True)
            _ = subprocess.run(f'rm -r {path_CTSM_clone}', shell=True)

    ########################################################################################################################
    # evaluate model results
    outfile_metric = f'{path_archive}/{caseflag}/evaluation_metric.csv'
    if os.path.isfile(outfile_metric) and overwrite_previous == False:
        print('The evaluation metric file exists. no need to run evaluation')
    else:
        infilelist = glob.glob(f'{path_archive}/{caseflag}/lnd/hist/*.clm2.h1.*.nc')
        infilelist.sort()
        fsurdat = get_parameter_from_Namelist_or_lndin('fsurdat', f'{path_CTSM_base}/user_nl_clm', f'{path_CTSM_base}/Buildconf/clmconf/lnd_in', type='str')
    
        print('Use standard lump evaluation')
        mo_evaluate(outfile_metric, infilelist, fsurdat, date_start, date_end, ref_streamflow, add_flow_file)

    # move cpu file from busy to idle
    if exclusive_flag==True:
        os.rename(cpufile_use2, cpufile_use) # release the cpu
