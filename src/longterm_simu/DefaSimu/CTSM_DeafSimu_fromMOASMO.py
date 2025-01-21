# use the MO-ASMO iter-0 trial-0 parameter (default parameters in my current workflow), make sure this is true before running the script

import glob, os, sys, subprocess

# inputs
basinnum = int(sys.argv[1])
# basinnum = 0
basinuse = f'level1_{basinnum}'
print('Processing basin:', basinuse)

# basic settings
path_CTSM_repo = '/glade/u/home/guoqiang/CTSM_repos/CTSM'
path_CTSM_source = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/{basinuse}'
path_archive = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/LongTermSimu/Defa/{basinuse}'


if len(glob.glob(f'{path_archive}/*.nc'))>0:
    print('Find nc files in ', path_archive)
    sys.exit(0)


if not os.path.isdir(path_CTSM_source):
    sys.exit(f'Error! {path_CTSM_source} does not exist!')
    
suffix = 'DefaultSimu'
run_startdate = '1951-10-01'
# run_enddate = '2019-12-31'
stop_n = 819
# stop_n = 2

path_CTSM_target = f'{path_CTSM_source}_{suffix}'
clonescript = f'{path_CTSM_repo}/cime/scripts/create_clone'

# clone a case for simulation
if os.path.isdir(path_CTSM_target):
    _ = os.system(f'rm -r {path_CTSM_target}')
    
_ = subprocess.run(f'{clonescript} --case {path_CTSM_target} --clone {path_CTSM_source} --keepexe', shell=True)
cwd = os.getcwd()
os.chdir(path_CTSM_target)

_ = subprocess.run(f'./xmlchange RUN_STARTDATE={run_startdate}', shell=True)
_ = subprocess.run(f'./xmlchange STOP_N={stop_n}', shell=True)


# Run model
_ = os.system('./case.setup')
_ = os.system('./case.submit --no-batch')

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

_ = subprocess.run(f'cp user_nl_clm {path_archive}', shell=True)
_ = subprocess.run(f'cp user_nl_datm_streams {path_archive}', shell=True)

# remove simulation folder
os.chdir('..')
_ = os.system(f'rm -r {path_CTSM_target}')
_ = os.system(f'rm -r {path_rundir}')
_ = os.system(f'rm -r {path_dsout}')