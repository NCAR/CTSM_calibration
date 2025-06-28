# An independent basin

# Create a CTSM model case
# Settings based on single basin files
# Note: manual check of settings are needed before running this script

import sys, subprocess, time, os


# more straitforward
path_CTSM_source = '/glade/work/swensosc/ctsm_code/hillslope_hydrology'
path_CTSM_case = f'/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Distr_Calib_split_nest/CAMELS_test_HS_MOSART'
path_CTSM_CIMEout = f'/glade/scratch/guoqiang/CTSM_outputs/CAMELS_Calib/Distr_Calib_split_nest/CAMELS_test_HS_MOSART'
file_CTSM_mesh = "/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/ESMFmesh_ctsm_HCDN_nhru_final_671.buff_fix_holes_polygons_simplified_5e-4_split_nested.nc"
file_CTSM_surfdata = "/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/surfdata_CAMELS_split_nested_hist_78pfts_CMIP6_simyr2000_c230105.nc"

createcase = "--machine cheyenne --compset I2000Clm51Sp --driver nuopc --compiler intel --res f09_g16 --handle-preexisting-dirs r --run-unsupported"
RUN_STARTDATE = '2000-01-01'
STOP_N = 12
STOP_OPTION = 'nmonths'
casebuild = 'direct'
projectCode = 'P08010000'
NTASKS = 144

#####################
# Model settings to be changed: list
user_nl_clm_settings = [f"fsurdat = '{file_CTSM_surfdata}'",
                        "hist_nhtfrq = 0,-24",
                        "hist_mfilt = 1,365",
                        "hist_fincl2 = 'QRUNOFF','H2OSNO','ZWT','SOILWATER_10CM','EFLX_LH_TOT','QDRAI','QOVER','RAIN'",
                        ]

xmlchange_settings = [f"ATM_DOMAIN_MESH={file_CTSM_mesh}",
                      f"LND_DOMAIN_MESH={file_CTSM_mesh}",
                      f"MASK_MESH={file_CTSM_mesh}",
                      # build/run parent path
                      f"CIME_OUTPUT_ROOT={path_CTSM_CIMEout}",
                      # turn off MOSART_MODE to save time
                      "MOSART_MODE=ACTIVE",
                      # change forcing data
                      "DATM_MODE=CLMNLDAS2",
                      # change the run time of mode case
                      f"STOP_N={STOP_N}",
                      f"RUN_STARTDATE={RUN_STARTDATE}",
                      f"STOP_OPTION={STOP_OPTION}",
                      f"NTASKS={NTASKS}",
                      ]


xmlquery_settings = 'ATM_DOMAIN_MESH,LND_DOMAIN_MESH,MASK_MESH,RUNDIR,DOUT_S_ROOT,MOSART_MODE,DATM_MODE,RUN_STARTDATE,STOP_N,STOP_OPTION,NTASKS,NTASKS_PER_INST'

########################################################################################################################
# create model cases

pwd = os.getcwd()

################################
# (1) create new case
newcase_settings = f"{createcase} --project {projectCode}"
_ = subprocess.run(f'{path_CTSM_source}/cime/scripts/create_newcase --case {path_CTSM_case} {newcase_settings}', shell=True)

################################
# (2) change dir
os.chdir(path_CTSM_case)

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
if casebuild == 'qcmd':
    _ = subprocess.run(f'qcmd -l select=1:ncpus=1:mpiprocs=1 -l walltime=0:20:00 -A {projectCode} -q share -- ./case.build', shell=True)
elif casebuild == 'direct':
    _ = subprocess.run(f'./case.build', shell=True)
else:
    sys.exit('Unknown casebuild')

################################
# (5) submit jobs (optional)
# Can be used to check whether the case can be successfully run before calibration
# _ = subprocess.run('./case.submit', shell=True)
