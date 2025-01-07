# run one selected parameter and archive

import os, sys, glob, subprocess, pathlib, random, time
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append('/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support')
from run_one_paramset_Derecho import *

tarbasin = int(sys.argv[1])

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
path_CTSM_base = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_emulator/level1_{tarbasin}'

caseflag = f'normKGEr{selr}'
path_archive = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_emulator_LongTermSimu/LSEallbasin/level1_{tarbasin}/{caseflag}'

run_startdate = '1951-10-01'
# run_enddate = '2019-12-31'
stop_n = 819
# stop_n = 5 

########################################################################################################################
# check whether previous simulation exists

if len(glob.glob(f'{path_archive}/*.nc'))>0:
    print('Find nc files in ', path_archive)
    sys.exit(0)
    
path_CTSM_base = str(pathlib.Path(path_CTSM_base))

########################################################################################################################
# find a parameter set to run the model
path0 = '/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_emulator'
dfall = pd.DataFrame()
for iter in range(0, 8):
    filei = f'{path0}/level1_{tarbasin}_calib/ctsm_outputs_LSEnormKGE/iter{iter}_many_metrics_mizuroute_s-1.csv'
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
file_parameter_set = f'{path0}/level1_{tarbasin}_calib/param_sets_LSEnormKGE/paramset_iter{iter_tar}_trial{trial_tar}.pkl'
print('Using parameter file:', file_parameter_set)
print('Optmz kge:', dfall['kge'].values[indi])

########################################################################################################################
# clone case
path_CTSM_clone = f'{path_CTSM_base}_{caseflag}'
clone_case_name = pathlib.Path(path_CTSM_clone).name
clone_CTSM_model_case(script_clone, path_CTSM_base, path_CTSM_clone)

########################################################################################################################
# change parameters, which won't affect files of path_CTSM_base
update_ctsm_parameters(path_CTSM_clone, file_parameter_set)

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
