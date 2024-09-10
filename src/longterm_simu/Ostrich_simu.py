# after calibration, using the optimal parameter to force the model and perform the simulation
import glob, os, sys, subprocess


# inputs
basinnum = int(sys.argv[1])
basins = [f'level1_{i}' for i in range(627)] + [f'level2_{i}' for i in range(40)] + [f'level3_{i}' for i in range(4)]

if basinnum > len(basins)-1:
    sys.exit(f'{basinnum} is larger than {len(basins)}')

basinuse = basins[basinnum]
print('Processing basin:', basinuse)

# basic settings
path_CTSM_repo = '/glade/u/home/guoqiang/CTSM_repos/CTSM_hillslope'
path_CTSM_source = f'/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_Ostrich/{basinuse}'
path_calib_param = f'/glade/campaign/cgd/tss/people/guoqiang/CTSMcases/CAMELS_Calib/Calib_all_HH_Ostrich/{basinuse}_Ostrich/archive/PreserveBestModel'
path_archive0 = f'/glade/campaign/cgd/tss/people/guoqiang/CTSMcases/CAMELS_Calib/Calib_all_HH_Ostrich/{basinuse}_Ostrich/archive/'
#file_init = '/glade/p/cesmdata/cseg/inputdata/lnd/clm2/initdata_map/clmi.I2000Clm50BgcCrop.2011-01-01.1.9x2.5_gx1v7_gl4_simyr2000_c190312.nc'

if not os.path.isdir(path_CTSM_source):
    sys.exit(f'Error! {path_CTSM_source} does not exist!')
    
suffix = 'Bestsimu1'
run_startdate = '1951-10-01'
run_enddate = '2019-12-31'
stop_n = 819

path_CTSM_target = f'{path_CTSM_source}_{suffix}'
path_archive = f'{path_archive0}/{suffix}'
clonescript = f'{path_CTSM_repo}/cime/scripts/create_clone'

# clone a case for simulation
if os.path.isdir(path_CTSM_target):
    _ = os.system(f'rm -r {path_CTSM_target}')
    
_ = subprocess.run(f'{clonescript} --case {path_CTSM_target} --clone {path_CTSM_source} --keepexe', shell=True)
cwd = os.getcwd()
os.chdir(path_CTSM_target)

_ = subprocess.run(f'./xmlchange RUN_STARTDATE={run_startdate}', shell=True)
_ = subprocess.run(f'./xmlchange STOP_N={stop_n}', shell=True)

# change user_nl_clm
new_nlclm = f'{path_calib_param}/user_nl_clm' # note: this include namelist parameters
new_param = f'{path_calib_param}/ostrich_trial_parameters.nc'
_ = os.system('mv user_nl_clm user_nl_clm-fromclone')
_ = os.system(f'cp {new_nlclm} user_nl_clm')
with open('user_nl_clm', 'a') as f:
    f.write(f"paramfile='{new_param}'\n")
        
# change surface dataset files
# pending        

# Run model
_ = os.system('./case.submit --no-batch')

# archive results
archive_keyword = "clm2*.nc" # what archive files to be saved? only for PreserveBestModel
out = subprocess.run('./xmlquery DOUT_S_ROOT', shell=True, capture_output=True)
out = out.stdout.decode().strip()
patharchive = out.split(':')[1].strip()
filelist_simulations = glob.glob(f'{patharchive}/lnd/hist/*{archive_keyword}*')
filelist_simulations.sort()

out = subprocess.run('./xmlquery RUNDIR', shell=True, capture_output=True)
out = out.stdout.decode().strip()
patharchive = out.split(':')[1].strip()
filelist_rest = glob.glob(f'{patharchive}/*clm2.r.*.nc')
filelist_rest.sort()

filelist = filelist_simulations + filelist_rest

if len(filelist) == 0:
    print(f'Do not find model output files! Check {patharchive}/lnd/hist/*{archive_keyword}*')
else:
    for f in filelist:
        # print('Archive best model output:', f)
        os.makedirs(path_archive, exist_ok=True)
        _ = subprocess.run(f'mv {f} {path_archive}', shell=True)
        