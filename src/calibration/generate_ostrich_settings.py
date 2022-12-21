# Create a PBS script to submit Ostrich calibration
# Here, we incorporate case.submit ntask settings in this step. therefore, case.submit needs to be run first (no need to finish running)

import os, pathlib, sys, shutil, subprocess
import toml
import numpy as np
import pandas as pd
import xarray as xr

def update_txt_file(file, newsettings, start, sep, comment):
    # start, sep, and comment are '', '\'', and ! for summa fileManager.txt
    # start, sep, and comment are '<', ' ', and ! for mizuroute control file
    # start, sep, and comment are '', ' ', and # for ostIn.txt
    # start, sep, and comment are '', '=', and # for run_trial.sh
    if (len(newsettings) > 0) and os.path.isfile(file):
        # read raw data
        with open(file) as f:
            contents = f.readlines()
        # save a new file
        file_new = file + '-temp'
        with open(file_new, 'w') as f:
            for line in contents:
                for name, value in newsettings.items():
                    if line.startswith(start + name):
                        line2 = line.split(comment)[0].strip()
                        if line2.count(sep) == 2: # format: xxx_sep_value_sep (only summa fileManager.txt)
                            oldvalue = line2.split(sep)[1].strip()
                        else:
                            oldvalue = line2.split(sep)[-1].strip()
                        if not isinstance(value, str):
                            value = str(value)
                        line = line.replace(oldvalue, value)
                f.write(line)
        # replace old file
        os.remove(file)
        shutil.move(file_new, file)




config_file_Ostrich = sys.argv[1]

print('Create Ostrich settings ...')
print('Reading configuration from:', config_file_Ostrich)


########################################################################################################################
# settings

##############
# parse settings

config_Ostrich = toml.load(config_file_Ostrich)

path_script_calib = config_Ostrich['path_script_calib']
path_script_Ostrich = config_Ostrich['path_script_Ostrich']
path_CTSM_case = config_Ostrich['path_CTSM_case']
file_calib_param = config_Ostrich['file_calib_param']
file_Qobs = config_Ostrich['file_Qobs']
ignore_month = config_Ostrich['ignore_month']
RUN_STARTDATE = config_Ostrich['RUN_STARTDATE']
STOP_N = config_Ostrich['STOP_N']
STOP_OPTION = config_Ostrich['STOP_OPTION']
projectCode = config_Ostrich['projectCode']

#############
# default settings

outpathOstCalib = path_CTSM_case + "_OstCalib"
ostrichRunDir = f"{outpathOstCalib}/run"  # calib directory
ostrichArchive = f'{outpathOstCalib}/archive'
ostrichRefDir = f"{outpathOstCalib}/refdata"  # calib directory
ostrichParam = f"{outpathOstCalib}/parameters"

# template files
infile_ostin_template = f'{path_script_Ostrich}/ostIn_KGE_DDS.tpl'
infile_runtrial_template = f'{path_script_Ostrich}/CTSM_run_trial.sh'
infile_savebest_template = f'{path_script_Ostrich}/call_PreserveBestModel.sh'
infile_savemodel_template = f'{path_script_Ostrich}/call_PreserveModelOutput.sh'

# scripts
script_gen_ostIn = f'{path_script_Ostrich}/auxiliary_create_ostin.py'
script_save_file = f'{path_script_Ostrich}/save_Ostrich_trial_outputs.py'

infile_lndin = f'{path_CTSM_case}/Buildconf/clmconf/lnd_in'
infile_param_link = f'{ostrichParam}/base_parameters.nc'

outfile_param_ost = f'{ostrichParam}/ostrich_trial_parameters.nc'

exeOstrich = "/glade/u/home/guoqiang/model_sources/Ostrich_v17.12.19/Source/OstrichGCC"

os.makedirs(ostrichRunDir, exist_ok=True)
os.makedirs(ostrichArchive, exist_ok=True)
os.makedirs(ostrichRefDir, exist_ok=True)
os.makedirs(ostrichParam, exist_ok=True)

########################################################################################################################
# what parameters to calibrate

# get priori parameter file location
# to ensure robustness, create a hyperlink

if not os.path.isfile(infile_param_link):

    with open(infile_lndin, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('paramfile'):
                infile_param = line.split('=')[-1].strip().replace('\'', '')
                break

    _ = subprocess.run(f'ln -s {infile_param} {infile_param_link}', shell=True)


ds_param = xr.load_dataset(infile_param_link)
df_calibparam = pd.read_csv(file_calib_param)

flags = np.zeros(len(df_calibparam))
for i in range(len(df_calibparam)):
    parami_name = df_calibparam.iloc[i]['Parameter']
    if parami_name in ds_param.data_vars:
        flags[i] = 1

df_calibparam = df_calibparam[flags == 1]

# copy raw parameters to calibrate parameters
_ = shutil.copy(infile_param_link, outfile_param_ost)

########################################################################################################################
# Create Ostrich setting files

#############
# Ostrich exe
_ = subprocess.run(f'ln -sf {exeOstrich} {ostrichRunDir}/', shell=True)

#############
# create nc_multiplier.tpl
outfile_mtp_tpl = f'{ostrichRunDir}/nc_multiplier.tpl'
with open(outfile_mtp_tpl, 'w') as f:
    for p in df_calibparam['Parameter'].values:
        linep = f'{p:30} |  {p}_mtp\n'
        _ = f.write(linep)

#############
# create ostIn.txt
outfile_ostin_txt = f'{ostrichRunDir}/ostIn.txt'
_ = subprocess.run(f'python {script_gen_ostIn} {infile_lndin} {infile_ostin_template} {file_calib_param} {outfile_ostin_txt}', shell=True)

#############
# create CTSM_run_trial.sh
outfile_runtrial = f'{ostrichRunDir}/CTSM_run_trial.sh'
_ = shutil.copy(infile_runtrial_template, outfile_runtrial)

runtrial_setting = {}
runtrial_setting['pathCTSMcase'] = path_CTSM_case
runtrial_setting['pathOstrich'] = outpathOstCalib
runtrial_setting['paramfile_priori'] = infile_param_link
runtrial_setting['paramfile_ostrich'] = outfile_param_ost
runtrial_setting['ostrichRunDir'] = ostrichRunDir

# evaluation period
DateEvalStart = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=ignore_month)).strftime('%Y-%m-%d') # ignor the first year when evaluating model
if STOP_OPTION == 'nyears':
    DateEvalEnd = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(years=STOP_N)).strftime('%Y-%m-%d')
elif STOP_OPTION == 'nmonths':
    DateEvalEnd = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=STOP_N)).strftime('%Y-%m-%d')
else:
    sys.exit(f'STOP_OPTION must be nyears or nmonths. {STOP_OPTION} is not accepted.')
runtrial_setting['DateEvalStart'] = DateEvalStart
runtrial_setting['DateEvalEnd'] = DateEvalEnd

# update setting file
update_txt_file(outfile_runtrial, runtrial_setting, start='', sep='=', comment='#')
_ = subprocess.run(f'chmod +x {outfile_runtrial}', shell=True)

#############
# save files

outfile_savebest = f'{ostrichRunDir}/call_PreserveBestModel.sh'
outfile_savemodel = f'{ostrichRunDir}/call_PreserveModelOutput.sh'

_ = shutil.copy(infile_savebest_template, outfile_savebest)
_ = shutil.copy(infile_savemodel_template, outfile_savemodel)

save_setting = {}
save_setting['script_save_file'] = script_save_file
save_setting['pathCTSM'] = path_CTSM_case
save_setting['pathOstrichRun'] = ostrichRunDir
save_setting['pathOstrichSave'] = ostrichArchive
update_txt_file(outfile_savebest, save_setting, start='', sep='=', comment='#')
update_txt_file(outfile_savemodel, save_setting, start='', sep='=', comment='#')
_ = subprocess.run(f'chmod +x {outfile_savebest}', shell=True)
_ = subprocess.run(f'chmod +x {outfile_savemodel}', shell=True)

########################################################################################################################
# create submission file
file_submit = f'{ostrichRunDir}/submit.Ostrich.sh'

lines1 = ['#!/bin/bash -l', '#PBS -N OstrichCalib', '#PBS -q share', '#PBS -l walltime=12:00:00', f'#PBS -A {projectcode}']

lines2 = []
template_file = f'{path_CTSM_case}/.case.run'
with open(template_file, 'r') as f:
    for li in f:
        if li.startswith('#PBS') and (not li.startswith('#PBS -N')):
            lines2.append(li.strip())

# clean archive folder of CTSM cases
cwd = os.getcwd()
os.chdir(path_CTSM_case)
out = subprocess.run('./xmlquery DOUT_S_ROOT', shell=True, capture_output=True)
DOUT_S_ROOT = out.stdout.decode().strip().split(' ')[-1]
os.chdir(cwd)
lines3 = ['\n', f'rm -r {DOUT_S_ROOT}\*']

lines4 = ['\n', 'module load conda/latest', 'conda activate npl-2022b', '\n', './OstrichGCC']

with open(file_submit, 'w') as f:
    for li in lines1+lines2+lines3+lines4:
        _ = f.write(li+'\n')

########################################################################################################################
# streamflow file data
outfile_Qobs = f'{ostrichRefDir}/streamflow_data.csv'
df_q_in = pd.read_csv(file_Qobs, delim_whitespace=True, header=None)
years = df_q_in[1].values
months = df_q_in[2].values
days = df_q_in[3].values
dates = [f'{years[i]}-{months[i]:02}-{days[i]:02}' for i in range(len(years))]
q_obs = df_q_in[4].values * 0.028316847 # cfs to cms
q_obs[q_obs<0] = -9999.0
df_q_out = pd.DataFrame({'Date': dates, 'Runoff_cms': q_obs})
df_q_out.to_csv(outfile_Qobs, index=False)

########################################################################################################################
# end
_ = subprocess.run(f'chmod +x {outfile_savemodel}', shell=True)
print('Finish building Ostrich.')