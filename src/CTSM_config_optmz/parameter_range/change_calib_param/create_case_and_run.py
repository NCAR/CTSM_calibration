# run one selected parameter and archive

import os, sys, glob, subprocess, pathlib, random, time, toml
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append('/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support')
from run_one_paramset_Derecho import *
from MOASMO_parameters import get_parameter_from_Namelist_or_lndin

tarbasin = int(sys.argv[1])
# tarbasin = 30

if len(sys.argv) == 3:
    selr = int(sys.argv[2]) # select the rank of objective functions used for simulation (1 is the best; 2 is the 2nd best ...)
else:
    selr = 1 # default is the best

if selr < 1:
    sys.exit('Error. Selr has to be > 1')

print('processing basin', tarbasin)
print('processing selr', selr)

# input arguments
script_clone = '/glade/u/home/guoqiang/CTSM_repos/CTSM/cime/scripts/create_clone'
path_CTSM_base = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/level1_{tarbasin}'

caseflag = f'normKGEr{selr}'

config_file = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/configuration/_level1-{tarbasin}_config_MOASMO.toml'
config = toml.load(config_file)
ref_streamflow = config['file_Qobs']
# evaluation period
RUN_STARTDATE = config['RUN_STARTDATE']
ignore_month = config['ignore_month']
STOP_OPTION = config['STOP_OPTION']
STOP_N = config['STOP_N']
date_start = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=ignore_month)).strftime('%Y-%m-%d') # ignor the first year when evaluating model
if STOP_OPTION == 'nyears':
    date_end = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(years=STOP_N)).strftime('%Y-%m-%d')
elif STOP_OPTION == 'nmonths':
    date_end = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=STOP_N)).strftime('%Y-%m-%d')
else:
    sys.exit(f'STOP_OPTION must be nyears or nmonths. {STOP_OPTION} is not accepted.')


if 'add_flow_file' in config:
    add_flow_file = config['add_flow_file']
else:
    add_flow_file = 'NA'


run_startdate = RUN_STARTDATE
stop_n = 72 


########################################################################################################################
# find a parameter set to run the model
path0 = '/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange'
dfall = pd.DataFrame()
for iter in range(0, 4):
    filei = f'{path0}/level1_{tarbasin}_MOASMOcalib/ctsm_outputs_emutest/iter{iter}_many_metric.csv'
    dfi = pd.read_csv(filei)
    dfi['iter'] = iter
    dfi['trial'] = np.arange(len(dfi))
    dfall = pd.concat([dfall, dfi])

# indi = np.nanargmax(dfall['kge'].values)
tarmet = -dfall['kge'].values.copy()
tarmet[np.isnan(tarmet)] = np.inf
indexsort = np.argsort(tarmet)
indi = indexsort[selr - 1]

iter_tar = dfall['iter'].values[indi]
trial_tar = dfall['trial'].values[indi]
file_parameter_set = f'{path0}/level1_{tarbasin}_MOASMOcalib/param_sets_emutest/paramset_iter{iter_tar}_trial{trial_tar}.pkl'
print('Using parameter file:', file_parameter_set)
print('Optmz kge:', dfall['kge'].values[indi])

for zb in [0.5, 1, 3, 5]:

    ########################################################################################################################
    # check whether previous simulation exists
    
    path_archive = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_{tarbasin}_MOASMOcalib/ctsm_outputs_LowZbedrock/iter{iter_tar}_trial{trial_tar}_zb{zb}'
    
    if len(glob.glob(f'{path_archive}/*.nc'))>0:
        print('Find nc files in ', path_archive)
        sys.exit(0)
        
    path_CTSM_base = str(pathlib.Path(path_CTSM_base))
    
    ########################################################################################################################
    # clone case
    path_CTSM_clone = f'{path_CTSM_base}_{caseflag}'
    clone_case_name = pathlib.Path(path_CTSM_clone).name
    clone_CTSM_model_case(script_clone, path_CTSM_base, path_CTSM_clone)
    
    ########################################################################################################################
    # change parameters, which won't affect files of path_CTSM_base
    update_ctsm_parameters(path_CTSM_clone, file_parameter_set)
    
    ########################################################################################################################
    # change zbedrock surf
    outfile_newsurfdata = f'{path_CTSM_clone}/updated_surfdata_lowZbedrock.nc'
    update_CTSM_parameter_SurfdataFile(['zbedrock'], [zb], path_CTSM_clone, outfile_newsurfdata)
    
    file_user_nl_clm = f'{path_CTSM_clone}/user_nl_clm'
    with open(file_user_nl_clm, 'a') as f:
        f.write(f"fsurdat='{outfile_newsurfdata}'\n")
    
    ########################################################################################################################
    # run model in "no-batch" mode
    os.chdir(path_CTSM_clone)
    _ = subprocess.run(f'./xmlchange RUN_STARTDATE={run_startdate}', shell=True)
    _ = subprocess.run(f'./xmlchange STOP_N={stop_n}', shell=True)
    subprocess.run('./case.submit --no-batch', shell=True)
    
    ########################################################################################################################
    # archive results
    archive_keyword = "clm2*.nc" # what archive files to be saved? only for PreserveBestModel
    out = subprocess.run('./xmlquery DOUT_S_ROOT', shell=True, capture_output=True)
    out = out.stdout.decode().strip()
    path_dsout = out.split(':')[1].strip()
    filelist_simulations = glob.glob(f'{path_dsout}/lnd/hist/*{archive_keyword}*')
    filelist_simulations.sort()
    
    out = subprocess.run('./xmlquery RUNDIR', shell=True, capture_output=True)
    out = out.stdout.decode().strip()
    path_rundir = out.split(':')[1].strip()
    filelist_rest = glob.glob(f'{path_rundir}/*clm2.r.*.nc')
    filelist_rest.sort()
    
    filelist = filelist_simulations + filelist_rest
    
    if len(filelist) == 0:
        print(f'Do not find model output files! Check {path_archive}/lnd/hist/*{archive_keyword}*')
    else:
        for f in filelist:
            # print('Archive best model output:', f)
            os.makedirs(path_archive, exist_ok=True)
            _ = subprocess.run(f'mv {f} {path_archive}', shell=True)
    
    _ = subprocess.run(f'cp {file_parameter_set} {path_archive}', shell=True)
    _ = subprocess.run(f'cp user_nl_clm {path_archive}', shell=True)
    _ = subprocess.run(f'cp user_nl_datm_streams {path_archive}', shell=True)
    
    _ = subprocess.run(f'cp updated_parameter.nc {path_archive}', shell=True)
    _ = subprocess.run(f'cp updated_surfdata.nc {path_archive}', shell=True)
    
    
    # remove simulation folder
    os.chdir('..')
    _ = os.system(f'rm -r {path_CTSM_clone}')
    _ = os.system(f'rm -r {path_rundir}')
    _ = os.system(f'rm -r {path_dsout}')
    
    
    ########################################################################################################################
    # evaluation results
    
    fsurdat = get_parameter_from_Namelist_or_lndin('fsurdat', f'{path_CTSM_base}/user_nl_clm', f'{path_CTSM_base}/Buildconf/clmconf/lnd_in', type='str')
    
    outfile_metric = f'{path_archive}/evaluation_many_metrics.csv'
    infilelist = glob.glob(f'{path_archive}/*.clm2.h1.*.nc')
    mo_evaluate_return_many_metrics(outfile_metric, infilelist, fsurdat, date_start, date_end, ref_streamflow, add_flow_file)
