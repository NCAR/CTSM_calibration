# Create a PBS script to submit Ostrich calibration
# Here, we incorporate case.submit ntask settings in this step. therefore, case.submit needs to be run first (no need to finish running)
# deal with both multiplier and addition (mtp and add)

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
jobsetting = config_Ostrich['jobsetting']

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

outfile_lndin_base = f'{ostrichParam}/base_lnd_in'
outfile_param_base = f'{ostrichParam}/base_parameters.nc'
outfile_surfdata_base = f'{ostrichParam}/base_surfdata.nc'
outfile_nlclm_base = f'{ostrichParam}/base_user_nl_clm'
outfile_param_ost = f'{ostrichParam}/ostrich_trial_parameters.nc'
outfile_surfdata_ost = f'{ostrichParam}/ostrich_trial_surfdata.nc'
outfile_param_info = f'{ostrichParam}/calib_parameter_info.csv'

exeOstrich = "/glade/u/home/guoqiang/model_sources/Ostrich_v17.12.19/Source/OstrichGCC"

_ = subprocess.run(f'rm -r {outpathOstCalib}', shell=True)

os.makedirs(ostrichRunDir, exist_ok=True)
os.makedirs(ostrichArchive, exist_ok=True)
os.makedirs(ostrichRefDir, exist_ok=True)
os.makedirs(ostrichParam, exist_ok=True)

########################################################################################################################
# back-up original parameter files (parameter nc file, lnd_in text file, user_nl_clm file) which will be used as base parameter source

# parameter nc file
if not os.path.isfile(outfile_param_base):
    with open(infile_lndin, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('paramfile'):
                infile_param = line.split('=')[-1].strip().replace('\'', '')
    _ = subprocess.run(f'ln -s {infile_param} {outfile_param_base}', shell=True)
    _ = shutil.copy(infile_param, outfile_param_ost)


# surfdata file
infile_user_nl_clm = path_CTSM_case + '/user_nl_clm'
if not os.path.isfile(outfile_surfdata_base):
    with open(infile_lndin, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('fsurdat'):
                infile_surfdata = line.split('=')[-1].strip().replace('\'', '')
    with open(infile_user_nl_clm) as f:
        for line in f:
            line = line.strip()
            if line.startswith('fsurdat'):
                infile_surfdata = line.split('=')[-1].strip().replace('\'', '')
    _ = subprocess.run(f'ln -s {infile_surfdata} {outfile_surfdata_base}', shell=True)
    _ = shutil.copy(infile_surfdata, outfile_surfdata_ost)


# lnd_in file
if not os.path.isfile(outfile_lndin_base):
    _ = subprocess.run(f'cp {infile_lndin} {outfile_lndin_base}', shell=True)

# user_nl_clm file
if not os.path.isfile(outfile_nlclm_base):
    _ = subprocess.run(f'cp {infile_user_nl_clm} {outfile_nlclm_base}', shell=True)


########################################################################################################################
# read and check parameter information file
df_calibparam = pd.read_csv(file_calib_param)

cols_required = ['Parameter', 'Lower', 'Upper']
for col in cols_required:
    if not (col in df_calibparam.columns):
        sys.exit(f'Error! Parameter info file {file_calib_param} does not contain column {col}!')

cols_auxiliary = ['Source', 'Method', 'Binding']
cols_aux_fill = ['Param', 'Multiplicative', 'None']
for col, fillv in zip(cols_auxiliary, cols_aux_fill):
    if not (col in df_calibparam.columns):
        print(f'Warning! Parameter info file {file_calib_param} does not contain column {col}.')
        print(f'Fill the column {col} using default value: {fillv}. Check if this is right!')
        df_calibparam[col] = fillv

# cols_others = ['Default', 'Type', 'etc'] # not necessary but can provide some useful infromation

########################################################################################################################
# select parameters that can be found in parameter source files

flags = np.zeros(len(df_calibparam))

with open(outfile_lndin_base, 'r') as f:
    lines = f.readlines()
    varname_lndin = [l.split('=')[0].strip() for l in lines]

ds_param = xr.open_dataset(outfile_param_base)
ds_surfdata = xr.open_dataset(outfile_surfdata_base)
for i in range(len(df_calibparam)):
    parami_name = df_calibparam.iloc[i]['Parameter']
    if (parami_name in ds_param.data_vars) or (parami_name in varname_lndin) or (parami_name in ds_surfdata.data_vars):
        flags[i] = 1
    else:
        print('Cannot find parami_name in parameter nc file or lnd_in file or surfdata nc file.')

df_calibparam = df_calibparam[flags == 1]
df_calibparam.index = np.arange(len(df_calibparam))

########################################################################################################################
# Create Ostrich setting files

#############
# Ostrich exe
_ = subprocess.run(f'ln -sf {exeOstrich} {ostrichRunDir}/', shell=True)

#############
# create param_factor.tpl
df_calibparam['Parameter_Ost'] = ''
df_calibparam['Source_file'] = ''
df_calibparam['OstrichTrial_file'] = ''

outfile_param_tpl = f'{ostrichRunDir}/param_factor.tpl'
with open(outfile_param_tpl, 'w') as f:
    for i in range(len(df_calibparam)):

        if df_calibparam.iloc[i]['Method'] == 'Multiplicative':
            suffix1 = 'mtp'
        elif df_calibparam.iloc[i]['Method'] == 'Additive':
            suffix1 = 'add'
        else:
            sys.exit('Error! Method must be Multiplicative or Additive.')

        if df_calibparam.iloc[i]['Source'] == 'Param':
            suffix2 = 'P'
            df_calibparam.at[i, 'Source_file'] = outfile_param_base
            df_calibparam.at[i, 'OstrichTrial_file'] = outfile_param_ost
        elif df_calibparam.iloc[i]['Source'] == 'Namelist':
            suffix2 = 'N'
            df_calibparam.at[i, 'Source_file'] = outfile_lndin_base
            df_calibparam.at[i, 'OstrichTrial_file'] = infile_user_nl_clm
        elif df_calibparam.iloc[i]['Source'] == 'Surfdata':
            suffix2 = 'S'
            df_calibparam.at[i, 'Source_file'] = outfile_surfdata_base
            df_calibparam.at[i, 'OstrichTrial_file'] = outfile_surfdata_ost
        else:
            sys.exit('Error! Source must be Param, Namelist, or Surfdata.')

        suffix = f'{suffix2}_{suffix1}'
        # suffix = 'fct'

        p_raw = df_calibparam.iloc[i]['Parameter']
        p_ost = f'{p_raw}_{suffix}'
        df_calibparam.at[i, 'Parameter_Ost'] = p_ost

        linep = f'{p_raw:50} |     {p_ost}\n'
        _ = f.write(linep)

# save for later use
df_calibparam.to_csv(outfile_param_info, index=False)

#############
# create ostIn.txt
outfile_ostin_txt = f'{ostrichRunDir}/ostIn.txt'
_ = subprocess.run(f'python {script_gen_ostIn} {infile_lndin} {infile_ostin_template} {outfile_param_info} {outfile_ostin_txt}', shell=True)

#############
# create CTSM_run_trial.sh
outfile_runtrial = f'{ostrichRunDir}/CTSM_run_trial.sh'
_ = shutil.copy(infile_runtrial_template, outfile_runtrial)

runtrial_setting = {}
runtrial_setting['pathCTSMcase'] = path_CTSM_case
runtrial_setting['pathOstrich'] = outpathOstCalib
runtrial_setting['paramfile_priori'] = outfile_param_base
runtrial_setting['paramfile_ostrich'] = outfile_param_ost
runtrial_setting['lndinfile_priori'] = infile_lndin
runtrial_setting['lndinfile_ostrich'] = outfile_lndin_base
runtrial_setting['ostrichRunDir'] = ostrichRunDir
runtrial_setting['file_param_info'] = outfile_param_info
runtrial_setting['ostrichScriptDir'] = path_script_Ostrich


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

# lines1 = ['#!/bin/bash -l', '#PBS -N OstrichCalib', '#PBS -q share', '#PBS -l walltime=12:00:00', f'#PBS -A {projectCode}']
lines1 = ['#!/bin/bash -l', f'#PBS -A {projectCode}'] + jobsetting

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
out = subprocess.run('./xmlquery RUNDIR', shell=True, capture_output=True)
RUNDIR = out.stdout.decode().strip().split(' ')[-1]
os.chdir(cwd)
lines3 = ['\n', f'rm -r {DOUT_S_ROOT}/*\n', f'rm -r {RUNDIR}/*.nc']

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