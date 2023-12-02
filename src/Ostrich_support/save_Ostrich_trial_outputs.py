# Two modes:
# 1. Save the best model
# 2. Save all model outputs for each trial

import os, sys, shutil, glob, subprocess, datetime

import pandas as pd


########################################################################################################################
# define some functions
def get_target_archive_files(pathCTSM, keyword):
    # get the list of archived files of the latest model run
    # # settings
    # pathCTSM = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/CAMELS_LumpCalib'
    # keyword = ".clm2.h1."
    # find files
    st_archive_file = glob.glob(f'{pathCTSM}/st_archive.*')
    st_archive_file.sort()
    st_archive_file = st_archive_file[-1] # only the latest one
    filelist = []
    print('Getting simulaiton outputs    from CTSM model case path ...')
    print('pathCTSM:', pathCTSM)
    print('keyword:', keyword)
    with open(st_archive_file, 'r')  as f:
        for line in f:
            if line.startswith('moving') or line.startswith('copying'):
                if keyword in line:
                    file = line.split(' to ')[-1].strip()
                    if os.path.isfile(file):
                        print('Append to file list:', file)
                        filelist.append(file)
    return filelist, st_archive_file


def get_value_from_settingfile(file, varname):
    # setting file format: varname = 'var value'
    varvalue = f'Do not find {varname} in {file}'
    with open(file, 'r') as f:
        for line in f:
            line=line.strip()
            if line.startswith(varname):
                varvalue = line.split('=')[-1].strip()
                varvalue = varvalue.replace('\'','')
    return varvalue


########################################################################################################################
# arguments
mode = sys.argv[1]
pathCTSM = sys.argv[2]
pathOstrichRun = sys.argv[3]
pathOstrichSave = sys.argv[4]

if not  mode in ['PreserveBestModel', 'PreserveModelOutput']:
    sys.exit(f'Saving mode {mode} is not accepted!!!')
else:
    print(f'Save calibration intermediate outputs using mode {mode}')

######## hard coded parameters (may be changed later)
#archive_keyword = ".clm2.h1." # what archive files to be saved? only for PreserveBestModel
archive_keyword = "clm2*.nc" # what archive files to be saved? only for PreserveBestModel

########################################################################################################################
# define folder name

file_OstModel = f'{pathOstrichRun}/OstModel0.txt'
df_OstModel = pd.read_csv(file_OstModel, delim_whitespace=True)
current_run = int(df_OstModel.iloc[-1]['Run'])

nowtime_UTC = datetime.datetime.utcnow()
nowtime_UTC = nowtime_UTC.strftime("%Y-%m-%m-%H-%M-%S")

if mode == 'PreserveBestModel':
    pathOstrichSave = f'{pathOstrichSave}/{mode}'
elif mode == 'PreserveModelOutput':
    pathOstrichSave = f'{pathOstrichSave}/{mode}/Run_{current_run}'
else:
    sys.exit('Unknown mode for saving!')
    
os.makedirs(pathOstrichSave, exist_ok=True)

########################################################################################################################
# get necessary file/path information from CTSM settings

# parameter file and append a UTC time suffix to it
file_lnd_in = f'{pathCTSM}/Buildconf/clmconf/lnd_in'
file_parameter = get_value_from_settingfile(file_lnd_in, 'paramfile')
file_surfdata = get_value_from_settingfile(file_lnd_in, 'fsurdat')
file_user_nl_clm = f'{pathCTSM}/user_nl_clm'

########################################################################################################################
# save files

# # get file list: method-1. do not work for ./case.submit --no-batch
# filelist_simulations, st_archive_file = get_target_archive_files(pathCTSM, archive_keyword)

# method-2: just search the output archive folder
cwd = os.getcwd()
os.chdir(pathCTSM)
out = subprocess.run('./xmlquery DOUT_S_ROOT', shell=True, capture_output=True)
out = out.stdout.decode().strip()
patharchive = out.split(':')[1].strip()
filelist_simulations = glob.glob(f'{patharchive}/lnd/hist/*{archive_keyword}*')
filelist_simulations.sort()
os.chdir(cwd)

if len(filelist_simulations) == 0:
    print(f'Do not find model output files! Check {patharchive}/lnd/hist/*{archive_keyword}*')
else:
    for f in filelist_simulations:
        # print('Archive best model output:', f)
        _ = subprocess.run(f'cp {f} {pathOstrichSave}', shell=True)

# save parameters
_ = subprocess.run(f'cp {file_parameter} {pathOstrichSave}', shell=True)
_ = subprocess.run(f'cp {file_surfdata} {pathOstrichSave}', shell=True)
_ = subprocess.run(f'cp {file_lnd_in} {pathOstrichSave}', shell=True)
_ = subprocess.run(f'cp {file_user_nl_clm} {pathOstrichSave}', shell=True)

# save OstModel
_ = subprocess.run(f'cp {pathOstrichRun}/OstModel0.txt {pathOstrichSave}', shell=True)

#if mode == 'PreserveBestModel':
if True:
    # save Ostrich outputs
    files_in_pathOstrichRun = ['*.txt', '*.tpl', 'timetrack.log']
    for f in files_in_pathOstrichRun:
        _ = subprocess.run(f'cp {pathOstrichRun}/{f} {pathOstrichSave}', shell=True)



