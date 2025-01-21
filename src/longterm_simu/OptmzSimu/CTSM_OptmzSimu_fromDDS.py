# run one selected parameter and archive

import os, sys, glob, subprocess, pathlib, random, time
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append('/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support')
from run_one_paramset_Derecho import *

tarbasin = int(sys.argv[1])
# tarbasin = 0

selr = 1 # default is the best

if selr < 1:
    sys.exit('Error. Selr has to be > 1')

print('processing basin', tarbasin)
print('processing selr', selr)

# input arguments
script_clone = '/glade/u/home/guoqiang/CTSM_repos/CTSM/cime/scripts/create_clone'
path_CTSM_base = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_Ostrich_kge/level1_{tarbasin}'

caseflag = f'normKGEr{selr}'
path_archive = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/LongTermSimu/DDS/level1_{tarbasin}/{caseflag}'

path_bestparam = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_Ostrich_kge/level1_{tarbasin}_OSTRICHcalib/archive/PreserveBestModel'

run_startdate = '1951-10-01'
stop_n = 819

# run_startdate = '2008-10-01'
# stop_n = 72

########################################################################################################################
# check whether previous simulation exists

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
os.chdir(path_CTSM_clone)

os.system(f'cp {path_bestparam}/user_nl_clm .')

file_param = f'{path_bestparam}/ostrich_trial_parameters.nc'
file_surf = f'{path_bestparam}/ostrich_trial_surfdata.nc'
os.system(f"echo \"paramfile='{file_param}'\" >> user_nl_clm")
os.system(f"echo \"fsurdat='{file_surf}'\" >> user_nl_clm")

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

_ = subprocess.run(f'cp {file_surf} {path_archive}', shell=True)
_ = subprocess.run(f'cp {file_param} {path_archive}', shell=True)


# remove simulation folder
os.chdir('..')
_ = os.system(f'rm -r {path_CTSM_clone}')
_ = os.system(f'rm -r {path_rundir}')
_ = os.system(f'rm -r {path_dsout}')
