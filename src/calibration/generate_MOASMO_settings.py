# generate MO_ASMO_settings so we can submit MO-ASMO calibration jobs
# there are two modes for the control task (i.e., coordinate everything including real model run):
# 1. direct submit, which could be limited by the time
# 2. resubmit. this has not been fulled tested

import sys, toml, os

def create_control_once_job(file_control, job_controlMOASMO, configfile, script_main):
    lines = ['module load ncarenv/23.09', 'module load conda/latest cdo', 'conda activate npl-2023b',
             '\n',
             f'python {script_main} {configfile}'
             ]
    lines = job_controlMOASMO + lines

    with open(file_control, 'w') as f:
        for li in lines:
            _ = f.write(li + '\n')

    _ = os.system(f'chmod +x {file_control}')



def create_control_resubmit_job(file_control, job_controlMOASMO, configfile, script_main, submit_script, script_resubmit, iter_num_per_sub):
    lines = ['module load ncarenv/23.09', 'module load conda/latest cdo', 'conda activate npl-2023b',
             '\n',
             f'submit_script={submit_script}',
             f'script_main={script_main}',
             f'configfile={configfile}',
             'logfile=resubmit.log',
             f'iter_num_per_sub={iter_num_per_sub}',
             '\n',
             'python resubmit_run.py ${submit_script} ${script_main} ${configfile} ${logfile} ${iter_num_per_sub}',
             ]
    lines = job_controlMOASMO + lines

    with open(file_control, 'w') as f:
        for li in lines:
            _ = f.write(li + '\n')

    _ = os.system(f'chmod +x {file_control}')


configfile = sys.argv[1]
mode = sys.argv[2]

config = toml.load(configfile)

job_controlMOASMO = config['job_controlMOASMO']
path_CTSM_base = config['path_CTSM_case']

if config['path_calib'] == 'NA':
    path_submit = f'{path_CTSM_base}_MOASMOcalib/submit'
else:
    path_calib = config['path_calib']
    path_submit = f'{path_calib}/submit'

os.makedirs(path_submit, exist_ok=True)

path_script_MOASMO = config['path_script_MOASMO']
script_main = f'{path_script_MOASMO}/main_MOASMO.py'

if mode == 'Once':
    file_control = f'{path_submit}/MOASMO_one_submit.sh'
    create_control_once_job(file_control, job_controlMOASMO, configfile, script_main)

elif mode == 'Resubmit':
    print('Note resubmit functions are not tested yet')
    file_control = f'{path_submit}/MOASMO_iter_submit.sh'
    script_main = f'{path_script_MOASMO}/main_MOASMO.py'
    script_resubmit = f'{path_script_MOASMO}/resubmit_run.py'
    iter_num_per_sub = 5
    submit_script = 'MOASMO_iter_submit.sh'
    create_control_resubmit_job(file_control, job_controlMOASMO, configfile, script_main, submit_script, script_resubmit, iter_num_per_sub)

