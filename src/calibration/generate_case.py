# An independent basin
import shutil
# Create a CTSM model case
# Settings based on single basin files
# Note: manual check of settings are needed before running this script

import sys, subprocess, time, os, toml



config_file_CTSMcase = sys.argv[1]

print('Create CTSM case ...')
print('Reading configuration from:', config_file_CTSMcase)

########################################################################################################################
# settings used for create CTSE case

##############
# parse settings

config_CTSMcase = toml.load(config_file_CTSMcase)

# for key,val in config_CTSMcase.items():
#         exec(key + '=val')

# more straitforward
path_CTSM_source = config_CTSMcase['path_CTSM_source']
path_CTSM_case = config_CTSMcase['path_CTSM_case']
path_CTSM_CIMEout = config_CTSMcase['path_CTSM_CIMEout']
file_CTSM_mesh = config_CTSMcase['file_CTSM_mesh']
file_CTSM_surfdata = config_CTSMcase['file_CTSM_surfdata']

createcase = config_CTSMcase['createcase']
RUN_STARTDATE = config_CTSMcase['RUN_STARTDATE']
STOP_N = config_CTSMcase['STOP_N']
STOP_OPTION = config_CTSMcase['STOP_OPTION']
NTASKS = config_CTSMcase['NTASKS']
casebuild = config_CTSMcase['casebuild']
projectCode = config_CTSMcase['projectCode']
CLONEROOT = config_CTSMcase['CLONEROOT']
CLONEsettings = config_CTSMcase['CLONEsettings']


#####################
# Model settings to be changed: list
user_nl_clm_settings = [f"fsurdat = '{file_CTSM_surfdata}'",
                        "hist_nhtfrq = 0,-24",
                        "hist_mfilt = 1,365",
                        "hist_fincl2 = 'QRUNOFF','H2OSNO','ZWT','SOILWATER_10CM','EFLX_LH_TOT','QDRAI','QOVER','RAIN'",
                        ]

# no need to re compile
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
                      # change computation resource requirement if needed
                      # NTASKS: the total number of MPI tasks, a negative value indicates nodes rather than tasks
                      ]

if NTASKS > -999:
    xmlchange_settings.append(f"NTASKS={NTASKS}")

if NTASKS == 1:
    # need to recompile
    xmlchange_settings2 = [# one cpu one job
                           "COST_PES=1",
                           "TOTALPES=1",
                           "MAX_TASKS_PER_NODE=1",
                           "MAX_MPITASKS_PER_NODE=1",
                           "COST_PES=1",
                           "--file env_mach_pes.xml --id ROOTPE --val 0",
                           "--file env_mach_pes.xml --id TOTALPES --val 1",
                          ]
    xmlchange_settings = xmlchange_settings + xmlchange_settings2

# if clone is used, only change these settings to avoid recompiling
select_settings_if_clone = ['ATM_DOMAIN_MESH', 'LND_DOMAIN_MESH', 'MASK_MESH', 'DATM_MODE', 'STOP_N', 'RUN_STARTDATE', 'STOP_OPTION']

xmlquery_settings = 'ATM_DOMAIN_MESH,LND_DOMAIN_MESH,MASK_MESH,RUNDIR,DOUT_S_ROOT,MOSART_MODE,DATM_MODE,RUN_STARTDATE,STOP_N,STOP_OPTION,NTASKS,NTASKS_PER_INST'


########################################################################################################################
# create model cases

pwd = os.getcwd()

################################
# (0) delete raw folders
_ = subprocess.run(f'rm -r {path_CTSM_case}', shell=True)

################################
# (1) create new case

if not os.path.isdir(CLONEROOT):
    print(f'Use create_newcase because CLONEROOT {CLONEROOT} does not exist')
    newcase_settings = f"{createcase} --project {projectCode}"
    _ = subprocess.run(f'{path_CTSM_source}/cime/scripts/create_newcase --case {path_CTSM_case} {newcase_settings}', shell=True)
    flag_clone = False
else:
    print(f'Use create_clone because CLONEROOT {CLONEROOT} exists')
    clone_settings = f"--cime-output-root {path_CTSM_CIMEout} {CLONEsettings} --project {projectCode}"
    _ = subprocess.run(f'{path_CTSM_source}/cime/scripts/create_clone --case {path_CTSM_case} --clone {CLONEROOT} {clone_settings}', shell=True)
    flag_clone = True

    # abandon some original settings
    file = f'{path_CTSM_case}/user_nl_clm'
    with open(file, 'w') as f:
        pass

    file = f'{path_CTSM_case}/user_nl_datm_streams'
    with open(file, 'w') as f:
        pass

    os.system(f'rm {path_CTSM_case}/user_nl_datm_streams_*')

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
if flag_clone == True:
    tmp = []
    for s1 in xmlchange_settings:
        for s2 in select_settings_if_clone:
            if s2 in s1:
                tmp.append(s1)
                break
    xmlchange_settings = tmp

for s in xmlchange_settings:
    _ = subprocess.run(f'./xmlchange {s}', shell=True)

# xmlquery
_ = subprocess.run(f'./xmlquery {xmlquery_settings}', shell=True)

################################
# (4) compile the model

if flag_clone == False:
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
# (5) replace files if they are provided
if 'replacefiles' in config_CTSMcase:
    replacefiles = config_CTSMcase['replacefiles']
    if isinstance(replacefiles, dict):
        
        for name, file in replacefiles.items():
            
            if os.path.isfile(name):
                print(f'use {file} to replace {name}')
                _ = subprocess.run(f'cp {name} {name}-backup', shell=True)
                _ = subprocess.run(f'cp {file} {name}', shell=True)
        
        _ = subprocess.run('./check_case', shell=True) # to make changed files shown in buildconf

        
################################
# (6) submit jobs (optional)
# Can be used to check whether the case can be successfully run before calibration
# _ = subprocess.run('./case.submit', shell=True)
