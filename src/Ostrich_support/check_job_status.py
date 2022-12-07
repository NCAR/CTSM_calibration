# check whether a submitted job is finished
# here for CLM, check st_archive status

import os, subprocess, sys, time

def get_JobID_from_CaseStatus(file_CaseStatus, keyword):
    # CaseStatus is a file of CLM
    if not os.path.isfile(file_CaseStatus):
        sys.exit(f'Error!!! file_CaseStatus does not exist:{file_CaseStatus}')
    with open(file_CaseStatus, 'r') as f:
        lines = f.readlines()
    jobid = -999
    for i in range(-1, -len(lines), -1):
        linei = lines[i]
        if keyword in linei:
            jobid = linei.strip().split(keyword)[-1].split('.')[0].strip()
            break
    if jobid == -999:
        sys.exit('Fail to find the st_archive job ID!!!')
    else:
        print(f'st_archive job ID is {jobid} found in CaseStatus: {linei}')
    return jobid


def detect_laststatus_of_CaseStatus(file_CaseStatus, keyword):
    with open(file_CaseStatus, 'r') as f:
        lines = f.readlines()
    line = lines[-2] # last line is --- after submission
    if keyword in line:
        return True
    else:
        return False


def check_job_status_use_qstat(jobid, wait_gap=30):
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
    while flag:
        tasklist = subprocess.run(f'qstat {jobid}', shell=True, capture_output=True)
        tasklist = tasklist.stdout.decode()
        if len(tasklist) == 0:
            flag = False
        else:
            # print('Waiting ...')
            time.sleep(wait_gap)
    t2 = time.time()
    print(f'Job {jobid} cannot be found using qstat after {t2 - t1} seconds.')
    print('Stop checking the status.')


########################################################################################################################
# arguments
# pathCTSM = '/glade/u/home/guoqiang/CTSM_cases/CAMELS_Calib/CAMELS_LumpCalib'
pathCTSM = sys.argv[1]


# default parameters
keyword_CaseStatus = 'case.submit success'
wait_gap = 60 # seconds

########################################################################################################################
# check the job status of the latest st_archive in the CaseStatus file

file_CaseStatus = f'{pathCTSM}/CaseStatus'
jobid = get_JobID_from_CaseStatus(file_CaseStatus, keyword_CaseStatus)

check_job_status_use_qstat(jobid, wait_gap)