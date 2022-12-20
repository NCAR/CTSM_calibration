# spin up using a continuous period simulation

import sys, subprocess, time, os, shutil, glob, pathlib
import pandas as pd
import func_run_CTSM_model as func_runCTSM


########################################################################################################################
# settings

# pathCTSMcase = '/glade/u/home/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib/CAMELS_0'
# spinup_months = 9
pathCTSMcase = sys.argv[1]
spinup_months = int(sys.argv[2])

force_Jan_start = True # CTSM default initial conditions start at Jan 1st. So, using Jan as the start date could be better
spinup_info = 'spinup_info.csv'
update_restart = True # after spin-up is done, add the restart file to user_nl_clm

if pathCTSMcase[-1]=='/':
    pathCTSMcase = pathCTSMcase[:-1]

pathSpinup = pathCTSMcase + '_SpinupFiles'
os.makedirs(pathSpinup, exist_ok=True)

t1 = time.time()

########################################################################################################################
# change model settings
cwd = os.getcwd()

# get model settings
os.chdir(pathCTSMcase)

settingnames = ['RUN_STARTDATE', 'STOP_N', 'STOP_OPTION']

df_setting = pd.DataFrame(index=settingnames)
df_setting['before_spinup'] = [func_runCTSM.get_xmlquery_output(s) for s in settingnames]

date0 = pd.Timestamp(df_setting['before_spinup']['RUN_STARTDATE'])
datee = (date0 - pd.offsets.DateOffset(hours=1)).strftime('%Y-%m-%d')
dates = (date0 - pd.offsets.DateOffset(months=spinup_months)).strftime('%Y-%m-%d')

if (force_Jan_start == True) and (dates[-5:] != '01-01'):
    print('Force the spin up start date to be Jan 1st.')
    spinup_months = spinup_months + int(dates[5:7]) - 1
    dates = dates[:4] + '-01-01'

df_setting['run_spinup'] = [dates, spinup_months, 'nmonths']
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

    # copy restart files to a target folder and change model settings
    file_restart = glob.glob(f'{RUNDIR}/*.clm2.r.*.nc')
    file_restart.sort()
    file_restart = file_restart[-1]
    file_restart_archive = pathSpinup + '/' + pathlib.Path(file_restart).name
    file_info = file_restart_archive.replace('.nc', '_info.csv')

    _ = shutil.copy(file_restart, pathSpinup)
    _ = shutil.copy(spinup_info, file_info)

    # copy restart files to a target folder and change model settings to before spin up
    if update_restart == True:
        with open('user_nl_clm', 'a') as f:
            f.write(f"finidat = '{file_restart_archive}'\n")


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