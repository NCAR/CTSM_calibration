
import sys, subprocess, time, os

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
                flag = True
            else:
                flag = False
        else:
            # print('Waiting ...')
            time.sleep(wait_gap)
        iternum = iternum + 1
    t2 = time.time()
    print(f'Job {jobid} cannot be found using qstat after {t2 - t1} seconds.')
    print('Stop checking the status.')


def submit_and_run_CTSM_model(direct_run=False, rm_old=True):
    # this function assume that we are already in the folder of model case
    # rm_old: remove old files in RUNDIR
    # direct_run: use "./case.submit --no-batch" to skip job allocation

    # clean run folder
    if rm_old == True:
        RUNDIR = get_xmlquery_output('RUNDIR')
        print('Remove old *.nc output files in RUNDIR:', RUNDIR)
        _ = subprocess.run(f'rm {RUNDIR}/*.nc', shell=True)

    # submit
    if direct_run == True:
        _ = os.system('./case.submit --no-batch')

    else:
        # submit and get job id
        out = subprocess.run('./case.submit', shell=True, capture_output=True)
        out = out.stdout.decode().split('\n')
        for line in out:
            if 'Submitted job id is ' in line:
                id_run = line.replace('Submitted job id is ', '').split('.')[0]
            if 'Submitted job case.st_archive with id ' in line:
                id_archive = line.replace('Submitted job case.st_archive with id ', '').split('.')[0]

        # hold until the job is finished
        check_job_status_use_qstat(id_archive, 'CaseStatus', wait_gap=60)

    if detect_laststatus_of_CaseStatus('CaseStatus', 'success'):
        # "st_archive success" for ./case.submit or "case.submit success" for "./case.submit --no-batch"
        print('Model run is successfully finished!')
    else:
        sys.exit('Model run failed!')

    return RUNDIR