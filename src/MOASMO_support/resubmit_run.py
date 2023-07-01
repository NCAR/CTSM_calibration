# For Cheyenne, it has time limit. Thus, resubmitting jobs is needed to finish a long job
import sys

import numpy as np
import toml, subprocess, os, time

time.sleep(5)

submit_script = sys.argv[1]
run_script = sys.argv[2]
configfile = sys.argv[3]
logfile = sys.argv[4]
iter_num_per_sub = int(sys.argv[5])


# decide iteration number for each submission. note that iteration-0 is an independent job
config = toml.load(configfile)
num_iter = config['num_iter']


# submit jobs
if not os.path.isfile(logfile):
    iter_start = 0
    iter_end = 0 + 1
else:
    print(f'Logfile {logfile} exists. Load iter_start and iter_end information from it.')
    d = np.loadtxt(logfile)
    iter_start = int(d[-1])
    iter_end = int(d[-1]) + iter_num_per_sub + 1
    if iter_end >= num_iter:
        iter_end = num_iter

out = subprocess.run(f'python {run_script} {configfile} {iter_start} {iter_end}', shell=True)

if out.returncode == 0:
    print('Job is successful. Update log file')
    with open(logfile, 'w') as f:
        f.write(f'{iter_start}\n')
        f.write(f'{iter_end}\n')


# resubmit a job and exit this job
if iter_end < num_iter:
    print('Submit a new job')
    subprocess.run(f'qsub {submit_script}', shell=True)
    print('Finish submitting the job')
else:
    print('All runs are finished')
    os.remove(logfile)
    with open(f'{logfile}0', 'w') as f:
        f.write('Successful!\n')

exit()
