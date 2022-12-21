# An independent basin

# Create a CTSM model case
# Settings based on single basin files
# Note: manual check of settings are needed before running this script

import sys, subprocess, time, os

##############
# settings

bnum  =  sys.argv[1]
STOP_N  =  sys.argv[2]
RUN_STARTDATE  =  sys.argv[3]
buildoption = sys.argv[4]

# bnum = 311 # basin number
# STOP_N = 3
# RUN_STARTDATE = '2000-01-01'
# buildoption = 'qcmd' # qcmd or direct

STOP_OPTION = 'nmonths'
input_type = 'mask' # mask: set mask based on Sean's setting; divide: independent basin based on Sean's setting
NTASKS = 1

##############
# less active settings
# pathCTSMrepo: CTSM model source used to create model cases. git clone from Github repos
pathCTSMrepo = "/glade/u/home/guoqiang/CTSM_repos/CTSM"

# pathParent: everything of the model will be saved here
pathParent = "/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib"

# CIME_OUTPUT_ROOT, which also defines run folder and archive folder
pathCIMEout = "/glade/scratch/guoqiang/CTSM_outputs/CAMELS_Calib/Lump_calib"

# CTSM model case folder
casename = f"CAMELS_{bnum}"
pathCTSMcase = f'{pathParent}/{casename}'

# surface file and domain file
if input_type == 'mask':
    fileDomain = f"/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_basin_mask/ESMFmesh_ctsm_HCDN_nhru_final_671_v0_8e-3_basin{bnum}.nc"
    fileSurf = f"/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/surfdata_CAMELS_hist_78pfts_CMIP6_simyr2000_c221004.nc"
elif input_type == 'divide':
    fileDomain = f"/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_basin_divide/ESMFmesh_ctsm_HCDN_nhru_final_671_v0_8e-3_basin{bnum}.nc"
    fileSurf = "/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_basin_mask/surfdata_CAMELS_hist_78pfts_CMIP6_simyr2000_c221004_basin{bnum}.nc"
else:
    sys.exit('Unknown input type!')

# projectCode: a project is needed to run on Cheyenne
projectCode = "P08010000"


#####################
# Model settings to be changed: list
user_nl_clm_settings = [f"fsurdat = '{fileSurf}'",
                        "hist_nhtfrq = 0,-24",
                        "hist_mfilt = 1,365",
                        "hist_fincl2 = 'QRUNOFF','H2OSNO','ZWT','SOILWATER_10CM','EFLX_LH_TOT','QDRAI','QOVER','RAIN'",
                        ]

xmlchange_settings = [f"ATM_DOMAIN_MESH={fileDomain}",
                      f"LND_DOMAIN_MESH={fileDomain}",
                      f"MASK_MESH={fileDomain}",
                      # build/run parent path
                      f"CIME_OUTPUT_ROOT={pathCIMEout}",
                      # turn off MOSART_MODE to save time
                      "MOSART_MODE=NULL",
                      # change forcing data
                      "DATM_MODE=CLMNLDAS2",
                      # change the run time of mode case
                      f"STOP_N={STOP_N}",
                      f"RUN_STARTDATE={RUN_STARTDATE}",
                      f"STOP_OPTION={STOP_OPTION}",
                      # change computation resource requirement if needed
                      # NTASKS: the total number of MPI tasks, a negative value indicates nodes rather than tasks
                      f"NTASKS={NTASKS}",
                      # one cpu one job
                      "COST_PES=1",
                      "TOTALPES=1",
                      "MAX_TASKS_PER_NODE=1",
                      "MAX_MPITASKS_PER_NODE=1",
                      "COST_PES=1",
                      ]

xmlquery_settings = 'ATM_DOMAIN_MESH,LND_DOMAIN_MESH,MASK_MESH,RUNDIR,DOUT_S_ROOT,MOSART_MODE,DATM_MODE,RUN_STARTDATE,STOP_N,STOP_OPTION,NTASKS,NTASKS_PER_INST'

########################################################################################################################
# 2. create model cases

print(f"processing CAMELS basin {bnum}")

pwd = os.getcwd()

################################
# (1) create new case
newcase_settings = f"--compset I2000Clm51Sp --driver nuopc --compiler intel --res f09_g16 --run-unsupported --project {projectCode}"
_ = subprocess.run(f'{pathCTSMrepo}/cime/scripts/create_newcase --case {pathCTSMcase} {newcase_settings}', shell=True)

################################
# (2) change dir
os.chdir(pathCTSMcase)

################################
# (3) change settings (optional)

# change user_nl_clm
with open('user_nl_clm', 'a') as f:
    for s in user_nl_clm_settings:
        _ = f.write(s+'\n')

# change land domain and MESH files
for s in xmlchange_settings:
    _ = subprocess.run(f'./xmlchange {s}', shell=True)

# xmlquery
_ = subprocess.run(f'./xmlquery {xmlquery_settings}', shell=True)

################################
# (4) compile the model
# _ = subprocess.run('./case.setup --reset', shell=True)
_ = subprocess.run('./case.setup', shell=True)
_ = subprocess.run('./case.build --clean-all', shell=True)
if buildoption == 'qcmd':
    _ = subprocess.run(f'qcmd -l select=1:ncpus=1:mpiprocs=1 -l walltime=0:20:00 -A {projectCode} -q share -- ./case.build', shell=True)
elif buildoption == 'direct':
    _ = subprocess.run(f'./case.build', shell=True)
else:
    sys.exit('Unknown buildoption')

################################
# (5) submit jobs (optional)
# Can be used to check whether the case can be successfully run before calibration
# _ = subprocess.run('./case.submit', shell=True)
