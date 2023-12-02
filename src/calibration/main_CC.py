# Build cases for CAMELS basins. one basin one case
# better to run within a computation node due to the time cost of case.build, spin up, forcing subset

import subprocess, sys, os, toml, pathlib

import parse_configuration as parsconfig

########################################################################################################################
# input config file
# config_file = '../config_templates/example.MO_ASMO.config.toml'
config_file = sys.argv[1]

# tasks that will be executed. by default, all tasks will be run
# runtasks = "Build,Ostrich,SubForc,NameList,SpinUp" # run complete Ostrich calibration
runtasks = "MOASMO" # run complete MO-ASMO calibration
if len(sys.argv) == 3:
    runtasks = sys.argv[2]

print('Run tasks:', runtasks)

# remove intermediate configuration files (.e.g., _cstm.config_CTMScase.toml)
rm_interconfig = False

config_file = os.path.abspath(config_file)

########################################################################################################################
print(f"Settings are read from {config_file}")
config = toml.load(config_file)
config['path_config_file'] = str(pathlib.Path(config_file).parent)
config['name_config_file'] = pathlib.Path(config_file).name

########################################################################################################################
# step-1: Create model case

if 'Build' in runtasks:
    script_CTSMcase = config['calib']['files']['path_script_calib'] + '/' + 'generate_case.py'
    file_config_CTSMcase = parsconfig.parse_CTSMcase_config(config)
    _ = subprocess.run(f'python {script_CTSMcase} {file_config_CTSMcase}', shell=True)
else:
    print('No need to build case')


########################################################################################################################
# step-2, option-1: Create Ostrich settings
if 'Ostrich' in runtasks:
    print('Ostrich calibration is selected.')
    script_GenOstrich = config['calib']['files']['path_script_calib'] + '/' + 'generate_ostrich_settings.py'
    file_config_Ostrich = parsconfig.parse_Ostrich_config(config)
    _ = subprocess.run(f'python {script_GenOstrich} {file_config_Ostrich}', shell=True)
else:
    print('No need to create Ostrich settings')


# step-2, option-2: Create MO-ASMO settings
moasmo_mode = 'Once' # Once or Resubmit
if 'MOASMO' in runtasks:
    print('MO-ASMO calibration is selected.')
    script_GenMOASMO = config['calib']['files']['path_script_calib'] + '/' + 'generate_MOASMO_settings.py'
    file_config_Ostrich = parsconfig.parse_MOASMO_config(config)
    _ = subprocess.run(f'python {script_GenMOASMO} {file_config_Ostrich} {moasmo_mode}', shell=True)
else:
    print('No need to create MO-ASMO settings')

########################################################################################################################
# step-3: Create forcing subsets and change model settings
if 'SubForc' in runtasks:
    script_SubForc = config['calib']['files']['path_script_calib'] + '/' + 'generate_forcing_subset.py'
    file_config_SubForc = parsconfig.parse_SubForc_config(config)
    _ = subprocess.run(f'python {script_SubForc} {file_config_SubForc}', shell=True)
else:
    print('No need to subset forcing')

########################################################################################################################
# step-4: Modify namelist
# will be add to other sections
if 'NameList' in runtasks:
    script_NL = config['calib']['files']['path_script_calib'] + '/' + 'generate_namelist.py'
    file_config_NL = parsconfig.parse_namelist_config(config)
    _ = subprocess.run(f'python {script_NL} {file_config_NL}', shell=True)
else:
    print('No need to modify namelist')

########################################################################################################################
# step-5: Model spin up
if 'SpinUp' in runtasks:
    script_spinup = config['calib']['files']['path_script_calib'] + '/' + 'generate_spinup.py'
    file_config_spinup = parsconfig.parse_spinup_config(config)
    _ = subprocess.run(f'python {script_spinup} {file_config_spinup}', shell=True)
else:
    print('No need to spin up model')

########################################################################################################################
# finally ...
if rm_interconfig == True:
    _ = subprocess.run('rm _*.toml', shell=True)

print('Model creation successful!')
