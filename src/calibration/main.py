# Build cases for CAMELS basins. one basin one case
# better to run within a computation node due to the time cost of case.build, spin up, forcing subset

import subprocess, sys
import toml
import pathlib

# functions for parsing configurations
def parse_CTSMcase_config(config):
    config_CTSMcase = {'path_CTSM_source': config['CTSM']['files']['path_CTSM_source'],
                       'path_CTSM_case': config['CTSM']['files']['path_CTSM_case'],
                       'path_CTSM_CIMEout': config['CTSM']['files']['path_CTSM_CIMEout'],
                       'file_CTSM_mesh': config['CTSM']['files']['file_CTSM_mesh'],
                       'file_CTSM_surfdata': config['CTSM']['files']['file_CTSM_surfdata'],

                       'CLONEROOT': config['CTSM']['settings']['CLONEROOT'],
                       'CLONEsettings': config['CTSM']['settings']['CLONEsettings'],
                       'createcase': config['CTSM']['settings']['createcase'],
                       'RUN_STARTDATE': config['CTSM']['settings']['RUN_STARTDATE'],
                       'STOP_N': config['CTSM']['settings']['STOP_N'],
                       'STOP_OPTION': config['CTSM']['settings']['STOP_OPTION'],
                       'NTASKS': config['CTSM']['settings']['NTASKS'],
                       'casebuild': config['CTSM']['settings']['casebuild'],

                       'projectCode': config['HPC']['projectCode'],
                       }
    # # simpler but less obvious
    # config_CTSMcase = config['CTSM']['files'] | config['CTSM']['settings'] | config['HPC']
    file_config_CTSMcase = config['path_config_file'] + '/_' + config['name_config_file'].replace('.toml', '') + '_CTMScase.toml'
    with open(file_config_CTSMcase, 'w') as f:
        toml.dump(config_CTSMcase, f)
    return file_config_CTSMcase


def parse_Ostrich_config(config):
    config_Ostrich = {'path_script_calib': config['calib']['files']['path_script_calib'],
                      'path_script_Ostrich': config['calib']['files']['path_script_Ostrich'],
                      'path_CTSM_case': config['CTSM']['files']['path_CTSM_case'],
                      'file_calib_param': config['calib']['files']['file_calib_param'],
                      'file_Qobs': config['calib']['files']['file_Qobs'],
                      'ignore_month': config['calib']['eval']['ignore_month'],
                      'RUN_STARTDATE': config['CTSM']['settings']['RUN_STARTDATE'],
                      'STOP_N': config['CTSM']['settings']['STOP_N'],
                      'STOP_OPTION': config['CTSM']['settings']['STOP_OPTION'],
                      'projectCode': config['HPC']['projectCode'],
                      'jobsetting': config['calib']['job']['jobsetting']
                      }
    file_config_Ostrich = config['path_config_file'] + '/_' + config['name_config_file'].replace('.toml', '') + '_Ostrich.toml'
    with open(file_config_Ostrich, 'w') as f:
        toml.dump(config_Ostrich, f)
    return file_config_Ostrich

def parse_SubForc_config(config):
    config_SubForc = {'path_CTSM_case': config['CTSM']['files']['path_CTSM_case'],
                      'subset_length': config['CTSM']['settings']['subset_length'],
                      'forcing_YearStep': config['CTSM']['settings']['forcing_YearStep'],
                      }
    file_config_SubForc = config['path_config_file'] + '/_' + config['name_config_file'].replace('.toml', '') + '_SubForc.toml'
    with open(file_config_SubForc, 'w') as f:
        toml.dump(config_SubForc, f)
    return file_config_SubForc

def parse_namelist_config(config):
    config_NL = {'path_CTSM_case': config['CTSM']['files']['path_CTSM_case'],
                      'AddToNamelist': config['CTSM']['AddToNamelist'],
                      }
    file_config_NL = config['path_config_file'] + '/_' + config['name_config_file'].replace('.toml', '') + '_namelist.toml'
    with open(file_config_NL, 'w') as f:
        toml.dump(config_NL, f)
    return file_config_NL

def parse_spinup_config(config):
    config_spinup = {'path_CTSM_case': config['CTSM']['files']['path_CTSM_case'],
                     'spinup_month': config['spinup']['spinup_month'],
                     'spinup_mode': config['spinup']['spinup_mode'],
                     'force_Jan_start': config['spinup']['force_Jan_start'],
                     'update_restart': config['spinup']['update_restart'],
                     }
    file_config_spinup = config['path_config_file'] + '/_' + config['name_config_file'].replace('.toml', '') + '_spinup.toml'
    with open(file_config_spinup, 'w') as f:
        toml.dump(config_spinup, f)
    return file_config_spinup

########################################################################################################################
# input config file
# config_file = './example.cstm.config.toml'
config_file = sys.argv[1]

# all: build case and do following task
# only: only build cases, don't do other tasks
# except: do all tasks except building cases
buildcase_option = 'all' # default is doing all jobs
if len(sys.argv[1]) == 3:
    buildcase_option = 'only'
    if buildcase_option in ['all', 'only', 'except']:
        print('buildcase_option:', buildcase_option)
    else:
        sys.exit('Unknown buildcase_option!')

# remove intermediate configuration files (.e.g., _cstm.config_CTMScase.toml)
rm_interconfig = False

########################################################################################################################
print(f"Settings are read from {config_file}")
config = toml.load(config_file)
config['path_config_file'] = str(pathlib.Path(config_file).parent)
config['name_config_file'] = pathlib.Path(config_file).name

########################################################################################################################
# step-1: Create model case

if buildcase_option in ['all', 'only']:
    script_CTSMcase = config['calib']['files']['path_script_calib'] + '/' + 'generate_case.py'
    file_config_CTSMcase = parse_CTSMcase_config(config)
    _ = subprocess.run(f'python {script_CTSMcase} {file_config_CTSMcase}', shell=True)

if buildcase_option in ['all', 'except']:
    ########################################################################################################################
    # step-2: Create Ostrich settings

    script_GenOstrich = config['calib']['files']['path_script_calib'] + '/' + 'generate_ostrich_settings.py'
    file_config_Ostrich = parse_Ostrich_config(config)
    _ = subprocess.run(f'python {script_GenOstrich} {file_config_Ostrich}', shell=True)

    ########################################################################################################################
    # step-3: Create forcing subsets and change model settings

    script_SubForc = config['calib']['files']['path_script_calib'] + '/' + 'generate_forcing_subset.py'
    file_config_SubForc = parse_SubForc_config(config)
    _ = subprocess.run(f'python {script_SubForc} {file_config_SubForc}', shell=True)

    ########################################################################################################################
    # step-4: Modify namelist

    script_NL = config['calib']['files']['path_script_calib'] + '/' + 'generate_namelist.py'
    file_config_NL = parse_namelist_config(config)
    _ = subprocess.run(f'python {script_NL} {file_config_NL}', shell=True)

    ########################################################################################################################
    # step-5: Model spin up

    script_spinup = config['calib']['files']['path_script_calib'] + '/' + 'generate_spinup.py'
    file_config_spinup = parse_spinup_config(config)
    _ = subprocess.run(f'python {script_spinup} {file_config_spinup}', shell=True)

########################################################################################################################
# finally ...

if rm_interconfig == True:
    _ = subprocess.run('rm _*.toml', shell=True)


print('Model creation successful!')
