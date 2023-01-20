# run CTSM and save outputs
import os
import toml, sys, pathlib, datetime, shutil, subprocess

def isnumber(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def get_parameters_from_namelist_file(file):
    param = {}
    with open(file, 'r') as f:
        for line in f:
            if not line.startswith('!'):
                linesplit = line.strip().split('=')
                name = linesplit[0].strip()
                value = linesplit[1].strip()
                if isnumber(value):
                    param[name] = value # later values will overwrite previous values
    return param

def get_xmlquery_output(keyword):
    out = subprocess.run(f'./xmlquery {keyword}', shell=True, capture_output=True)
    out = out.stdout.decode().strip().split(':')[1].strip()
    if keyword in ['STOP_N']:
        out = int(out)
    return out

########################################################################################################################
# input config file
# config_file = './example.simu.config.toml'
config_file = sys.argv[1]

########################################################################################################################
print(f"Settings are read from {config_file}")
config = toml.load(config_file)
current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

########################################################################################################################
print("Back up original model settings")
path_CTSM_case = config['path_CTSM_case']
path_backup = f'{path_CTSM_case}/backup_{current_time}/'
os.makedirs(path_backup, exist_ok=True)

backupfiles = ['user_nl_clm', 'env_run.xml', 'Buildconf/clmconf/lnd_in']
for f in backupfiles:
    _ = shutil.copy(f'{path_CTSM_case}/{f}', path_backup)

########################################################################################################################
print("Start model simulation")
cwd = os.getcwd()
os.chdir(path_CTSM_case)

for period in config['periods']:

    # change period settings
    _ = subprocess.run(f'./xmlchange RUN_STARTDATE={period[0]}', shell=True)
    _ = subprocess.run(f'./xmlchange STOP_N={period[1]}', shell=True)
    _ = subprocess.run(f'./xmlchange STOP_OPTION={period[2]}', shell=True)

    # change parameter files
    paramfile_Param = config['paramfile_Param']
    paramfile_Surf = config['paramfile_Surf']
    pramfile_NL = config['pramfile_NL']

    if any(os.path.isfile(f) for f in [paramfile_Param, paramfile_Surf, pramfile_NL]):
        addlines = []
        addlines.append('! \n')
        addlines.append(f'! {current_time}. Add parameter settings\n')
        if os.path.isfile(paramfile_Param):
            addlines.append(f"paramfile='{paramfile_Param}'\n")
        if os.path.isfile(paramfile_Surf):
            addlines.append(f"fsurdat='{paramfile_Surf}'\n")
        if os.path.isfile(pramfile_NL):
            params = get_parameters_from_namelist_file(pramfile_NL)
            for name, value in params.items():
                addlines.append(f"{name} = {value}\n")
        addlines.append('! \n')
        # add parameter settings to namelist file
        casefile_NL = f'{path_CTSM_case}/user_nl_datm'
        with open(casefile_NL, 'a') as f:
            for line in addlines:
                _ = f.write(line)

    # clear run and archive folder
    rundir = get_xmlquery_output('RUNDIR')
    archivedir = get_xmlquery_output('DOUT_S_ROOT')
    _ = subprocess.run(f'rm {rundir}/*.nc', shell=True)
    _ = subprocess.run(f'rm {archivedir}/lnd/hist/*.nc', shell=True)

    # model simulation
    _ = subprocess.run('./case.submit --no-batch', shell=True)

    # archive files
    savefile = config['savefile']
    path_output = config['path_output']
    path_output = f'{path_output}/{period[0]}-{period[1]}-{period[2]}'
    os.makedirs(path_output, exist_ok=True)
    for f in savefile:
        f = f.split(':')
        if f[0] == 'archive':
            _ = subprocess.run(f'cp {archivedir}/{f[1]} {path_output}', shell=True)
        elif f[0] == 'run':
            _ = subprocess.run(f'cp {rundir}/{f[1]} {path_output}', shell=True)