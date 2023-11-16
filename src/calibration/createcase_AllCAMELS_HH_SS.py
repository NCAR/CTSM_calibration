# An independent basin

# Create a CTSM model case
# Settings based on single basin files
# Note: manual check of settings are needed before running this script

import sys, subprocess, time, os


# more straitforward
path_CTSM_source = '/glade/u/home/guoqiang/CTSM_repos/CTSM_hillslope'
path_CTSM_case = f'/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Distr_Calib_no_nest/hillslope_SS587'
path_CTSM_CIMEout = f'/glade/scratch/guoqiang/CTSM_outputs/CAMELS_Calib/Distr_Calib_no_nest'
file_CTSM_mesh = f'/glade/work/swensosc/polygons/esmf_mesh_files/ESMFmesh_ctsm_HCDN_nhru_final_671_v0_8e-3.nc'
file_CTSM_surfdata = '/glade/work/swensosc/polygons/sfcdata/surfdata_CAMELS_8e-3_hist_78pfts_CMIP6_simyr2000_HAND_hillslope_geo_params_section_quad_20Chunk.nc'

createcase = "--machine cheyenne --compset I2000Clm51Sp --driver nuopc --compiler intel --res f09_g16 --handle-preexisting-dirs r --run-unsupported"
RUN_STARTDATE = '1901-01-01'
STOP_N = 113
STOP_OPTION = 'nyears'
#RESUBMIT = 0
casebuild = 'direct'
projectCode = 'P93300041'

#####################
# Model settings to be changed: list
user_nl_clm_settings = ["! Ref: /glade/u/home/swensosc/cases/hillslope_CAMELS/user_nl_clm", 
                        f"fsurdat = '{file_CTSM_surfdata}'",
                        "\n",
                        "use_hillslope = .true.", 
                        "use_hillslope_routing = .true.", 
                        "n_dom_pfts = 2",
                        "\n",
                        "hillslope_soil_profile_method =  'SetLowlandUpland'", 
                        "\n",
                        "hist_nhtfrq = 0,-24,-24", 
                        "hist_mfilt = 1,365,365", 
                        "hist_dov2xy = .true., .true., .false.", 
                        "\n",
                        "! for hillslope, add streamflow", 
                        "hist_fincl2 = 'QRUNOFF','H2OSNO','H2OSFC','ZWT','ZWT_PERCH','SOILWATER_10CM','EFLX_LH_TOT','QDRAI','QDRAI_PERCH','QOVER','QH2OSFC','QFLX_SNOW_DRAIN','RAIN','SOILLIQ','SOILICE','VOLUMETRIC_STREAMFLOW','STREAM_WATER_DEPTH','STREAM_WATER_VOLUME'",
                        "hist_fincl3 = 'QRUNOFF','H2OSNO','H2OSFC','ZWT','ZWT_PERCH','QDRAI','QDRAI_PERCH','QOVER','QH2OSFC','QFLX_SNOW_DRAIN','SNOW','SOILLIQ','SOILICE'",
                        ]

xmlchange_settings = [f"ATM_DOMAIN_MESH={file_CTSM_mesh}",
                      f"LND_DOMAIN_MESH={file_CTSM_mesh}",
                      f"MASK_MESH={file_CTSM_mesh}",
                      # build/run parent path
                      f"CIME_OUTPUT_ROOT={path_CTSM_CIMEout}",
                      # turn off MOSART_MODE to save time
                      "MOSART_MODE=NULL",
                      # change forcing data
                      "DATM_MODE=CLMGSWP3v1",
                      # change the run time of mode case
                      f"STOP_N={STOP_N}",
                      f"RUN_STARTDATE={RUN_STARTDATE}",
                      f"STOP_OPTION={STOP_OPTION}",
                      f"NTASKS=-10",
                      f"NTASKS_ATM=-1",
                      f"NTASKS_ESP=1",
                      #f"RESUBMIT={RESUBMIT}",
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
    # _ = subprocess.run(f'qcmd -l select=1:ncpus=1:mpiprocs=1 -l walltime=0:20:00 -A {projectCode} -q share -- ./case.build', shell=True)
    _ = subprocess.run(f'qcmd -- ./case.build', shell=True)
elif casebuild == 'direct':
    _ = subprocess.run(f'./case.build', shell=True)
else:
    sys.exit('Unknown casebuild')

################################
# (5) submit jobs (optional)
# Can be used to check whether the case can be successfully run before calibration
# _ = subprocess.run('./case.submit', shell=True)
