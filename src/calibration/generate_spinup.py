# spin up using a continuous period simulation

import sys, subprocess, time, os, shutil, glob, pathlib, toml
import pandas as pd
import func_run_CTSM_model as func_runCTSM



config_file_spinup = sys.argv[1]

print('Create spin up ...')
print('Reading configuration from:', config_file_spinup)

########################################################################################################################
# settings used for create CTSE case

##############
# parse settings

config_spinup = toml.load(config_file_spinup)


path_CTSM_case = config_spinup['path_CTSM_case']
spinup_month = config_spinup['spinup_month']
spinup_mode = config_spinup['spinup_mode']
force_Jan_start = config_spinup['force_Jan_start']
update_restart = config_spinup['update_restart']

##############
# default settings

save_spinup_simulations = True # save the lnd model output files from spin up run

spinup_info = 'spinup_info.csv'

if path_CTSM_case[-1]=='/':
    path_CTSM_case = path_CTSM_case[:-1]

pathSpinup = path_CTSM_case + '_SpinupFiles'
os.makedirs(pathSpinup, exist_ok=True)

t1 = time.time()

########################################################################################################################
# check if restart file has been generated
files = glob.glob(f'{pathSpinup}/*clm2.r.*.nc')
if len(files)>0:
    print('Restart files exist:', files)
    sys.exit('Restart files exist. No need to spin up')


########################################################################################################################
# change model settings
cwd = os.getcwd()

# get model settings
os.chdir(path_CTSM_case)

settingnames = ['RUN_STARTDATE', 'STOP_N', 'STOP_OPTION']

df_setting = pd.DataFrame(index=settingnames)
df_setting['before_spinup'] = [func_runCTSM.get_xmlquery_output(s) for s in settingnames]

date0 = pd.Timestamp(df_setting['before_spinup']['RUN_STARTDATE'])
datee = (date0 - pd.offsets.DateOffset(hours=1)).strftime('%Y-%m-%d')
dates = (date0 - pd.offsets.DateOffset(months=spinup_month)).strftime('%Y-%m-%d')

if (force_Jan_start == True) and (dates[-5:] != '01-01'):
    print('Force the spin up start date to be Jan 1st.')
    spinup_month = spinup_month + int(dates[5:7]) - 1
    dates = dates[:4] + '-01-01'

df_setting['run_spinup'] = [dates, spinup_month, 'nmonths']
df_setting.to_csv(spinup_info, index=False)

# change model settings
for s in settingnames:
    v = df_setting['run_spinup'][s]
    _ = subprocess.run(f'./xmlchange {s}={v}', shell=True)


########################################################################################################################
# run the model to generate restart files
try:
    RUNDIR = func_runCTSM.submit_and_run_CTSM_model(direct_run=True, rm_old=True)
    sucess_flag = True

except:
    print('Failed to create restart files!')
    sucess_flag = False



########################################################################################################################
# copy restart files

if sucess_flag == True:
    
    RUNDIR = func_runCTSM.get_xmlquery_output('RUNDIR')
    DOUT_S_ROOT = func_runCTSM.get_xmlquery_output('DOUT_S_ROOT')
    path_archive = f'{pathSpinup}/archive'
    os.makedirs(path_archive, exist_ok=True)
    
    # copy the restart file to a target folder and change model settings
    file_restart = glob.glob(f'{RUNDIR}/*.clm2.r.*.nc')
    file_restart = file_restart[-1]
    file_restart_archive = pathSpinup + '/' + pathlib.Path(file_restart).name
    file_info = file_restart_archive.replace('.nc', '_info.csv')

    _ = shutil.copy(file_restart, pathSpinup)
    _ = shutil.copy(spinup_info, file_info)
    
    if update_restart == True:
        with open('user_nl_clm', 'a') as f:
            f.write(f"finidat = '{file_restart_archive}'\n")
            
    # copy other restart related files
    file_restart = glob.glob(f'{RUNDIR}/*.clm2.rh*.*.nc')
    for f in file_restart:
        f_archive = pathSpinup + '/' + pathlib.Path(f).name
        _ = shutil.copy(f, f_archive)
    
    # copy timing information (the most recent two files
    path_time = f'{path_archive}/timing'
    os.makedirs(path_time, exist_ok=True)
    files = glob.glob(os.path.join("./timing/*"))
    files.sort(key=os.path.getmtime, reverse=True)

    for f in files[:2]:
        _ = subprocess.run(f'cp {f} {path_time}', shell=True)
    
    # copy save_spinup_simulations
    if save_spinup_simulations == True:
        _ = subprocess.run(f'cp -r {DOUT_S_ROOT}/lnd {path_archive}', shell=True)



########################################################################################################################
# change model settings back to the original
for s in settingnames:
    v = df_setting['before_spinup'][s]
    _ = subprocess.run(f'./xmlchange {s}={v}', shell=True)

# change dir
os.chdir(cwd)

if sucess_flag == True:
    print('Finish spin up!')
else:
    sys.exit('Failed spin up!')

t2 = time.time()
print('Time cost (sec):', t2-t1)