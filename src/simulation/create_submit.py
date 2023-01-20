# create submission scripts
import toml, os

# input config file
config_file = './example.simu.config.toml'
# config_file = sys.argv[1]


# load configurations
config = toml.load(config_file)
path_CTSM_case = config['path_CTSM_case']
path_output = config['path_output']
jobsetting = config['jobsetting']
projectCode = config['projectCode']
script_simu = config['script_simu']

path_submit = f'{path_output}/submit'
os.makedirs(path_submit, exist_ok=True)
file_submit = f'{path_submit}/submit.simu.sh'


# create submit sh
lines1 = ['#!/bin/bash -l', f'#PBS -A {projectCode}'] + jobsetting

lines2 = []
template_file = f'{path_CTSM_case}/.case.run'
with open(template_file, 'r') as f:
    for li in f:
        if li.startswith('#PBS') and (not li.startswith('#PBS -N')):
            lines2.append(li.strip())

lines3 = ['\n', 'module load conda/latest', 'conda activate npl-2022b', '\n', f'cd {path_CTSM_case}', f'python {script_simu} {config_file}', '\n']

with open(file_submit, 'w') as f:
    for li in lines1+lines2+lines3:
        _ = f.write(li+'\n')
