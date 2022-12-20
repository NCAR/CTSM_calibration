# spin up using a continuous period simulation

import sys, subprocess, time, os, shutil, glob, pathlib
import pandas as pd

def get_xmlquery_output(keyword):
    out = subprocess.run(f'./xmlquery {keyword}', shell=True, capture_output=True)
    out = out.stdout.decode().strip().split(':')[1].strip()
    if keyword in ['STOP_N']:
        out = int(out)
    return out

def detect_laststatus_of_CaseStatus(file_CaseStatus, keyword):
    with open(file_CaseStatus, 'r') as f:
        lines = f.readlines()
    line = lines[-2] # last line is --- after submission
    if keyword in line:
        return True
    else:
        return False


def check_job_status_use_qstat(jobid, file_CaseStatus, wait_gap=30):
    print(f'Start checking the status of job {jobid} using qstat')
    t1 = time.time()
    # check if the target job is queuing or running
    # stop until job is complete

    # qstat sometimes needs some time to show the job status
    # time.sleep(30) # not safe enough. use detect_laststatus_of_CaseStatus instead
    while detect_laststatus_of_CaseStatus(file_CaseStatus, keyword='case.submit success'):
        time.sleep(10)

    # check status
    flag = True
    iternum = 0
    while flag:
        tasklist = subprocess.run(f'qstat {jobid}', shell=True, capture_output=True)
        tasklist = tasklist.stdout.decode()
        if len(tasklist) == 0:
            if iternum < 2:
                flag = False
            else:
                flag = True
        else:
            # print('Waiting ...')
            time.sleep(wait_gap)
        iternum = iternum + 1
    t2 = time.time()
    print(f'Job {jobid} cannot be found using qstat after {t2 - t1} seconds.')
    print('Stop checking the status.')

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

########################################################################################################################
# change model settings
cwd = os.getcwd()

# get model settings
os.chdir(pathCTSMcase)

settingnames = ['RUN_STARTDATE', 'STOP_N', 'STOP_OPTION']

df_setting = pd.DataFrame(index=settingnames)
df_setting['before_spinup'] = [get_xmlquery_output(s) for s in settingnames]

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

try:
    ########################################################################################################################
    # run the model to generate restart files

    # clean run folder
    RUNDIR = get_xmlquery_output('RUNDIR')
    _ = subprocess.run(f'rm {RUNDIR}/*.nc', shell=True)

    # submit
    out = subprocess.run('./case.submit', capture_output=True)
    out = out.stdout.decode().split('\n')
    for line in out:
        if 'Submitted job id is ' in line:
            id_run = line.replace('Submitted job id is ', '').split('.')[0]
        if 'Submitted job case.st_archive with id ' in line:
            id_archive = line.replace('Submitted job case.st_archive with id ', '').split('.')[0]

    # hold until the job is finished
    file_CaseStatus = f'{pathCTSMcase}/CaseStatus'
    check_job_status_use_qstat(id_archive, file_CaseStatus, wait_gap=60)

    if detect_laststatus_of_CaseStatus(file_CaseStatus, 'st_archive success'):
        print('Spin up run is successfully finished!')
    else:
        sys.exit('Spin up run failed!')

    ########################################################################################################################
    # copy restart files to a target folder and change model settings

    file_restart = glob.glob(f'{RUNDIR}/*.clm2.r.*.nc')
    file_restart.sort()
    file_restart = file_restart[-1]
    file_restart_archive = pathSpinup + '/' + pathlib.Path(file_restart).name
    file_info = file_restart_archive.replace('.nc', '_info.csv')

    _ = shutil.copy(file_restart, pathSpinup)
    _ = shutil.copy(spinup_info, file_info)

    ########################################################################################################################
    # copy restart files to a target folder and change model settings to before spin up

    if update_restart == True:
        with open('user_nl_clm', 'a') as f:
            f.write(f"finidat = '{file_restart_archive}'\n")

    sucess_flag = True

except:
    print('Failed to create restart files!')
    sucess_flag = False


# change model settings back to the original
for s in settingnames:
    v = df_setting['before_spinup'][s]
    _ = subprocess.run(f'./xmlchange {s}={v}', shell=True)


os.chdir(cwd)



if sucess_flag == True:
    print('Finish spin up!')
else:
    sys.exit('Failed spin up!')
