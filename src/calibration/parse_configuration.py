# functions for parsing and processing configurations

import toml


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

def parse_MOASMO_config(config):
    config_MOASMO =  {'path_CTSM_source': config['CTSM']['files']['path_CTSM_source'],
                      'path_script_calib': config['calib']['files']['path_script_calib'],
                      'path_script_MOASMO': config['calib']['files']['path_script_MOASMO'],
                      'path_CTSM_case': config['CTSM']['files']['path_CTSM_case'],
                      'file_calib_param': config['calib']['files']['file_calib_param'],
                      'file_Qobs': config['calib']['files']['file_Qobs'],
                      'ignore_month': config['calib']['eval']['ignore_month'],
                      'RUN_STARTDATE': config['CTSM']['settings']['RUN_STARTDATE'],
                      'STOP_N': config['CTSM']['settings']['STOP_N'],
                      'STOP_OPTION': config['CTSM']['settings']['STOP_OPTION'],
                      'projectCode': config['HPC']['projectCode'],
                      'job_CTSMiteration': config['calib']['job']['job_CTSMiteration'],
                      'job_controlMOASMO': config['calib']['job']['job_controlMOASMO'],
                      'sampling_method': config['calib']['settings']['sampling_method'],
                      'num_init': config['calib']['settings']['num_init'],
                      'num_per_iter': config['calib']['settings']['num_per_iter'],
                      'num_iter': config['calib']['settings']['num_iter'],
                      }
    file_config_MOASMO = config['path_config_file'] + '/_' + config['name_config_file'].replace('.toml', '') + '_MOASMO.toml'
    with open(file_config_MOASMO, 'w') as f:
        toml.dump(config_MOASMO, f)
    return file_config_MOASMO

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