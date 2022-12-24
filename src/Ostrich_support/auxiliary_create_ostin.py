# create ostin.txt from a template file
# how to deal with parameters with different dims such as pft?


import os, sys, time, pathlib, subprocess
import xarray as xr
import pandas as pd
import numpy as np


infile_lndin = sys.argv[1]
infile_ostin_template = sys.argv[2]
infile_calibparam = sys.argv[3]
outfile_ostin_txt = sys.argv[4]

# infile_lndin = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib/CAMELS_2/Buildconf/clmconf/lnd_in'
# infile_ostin_template = '/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/Ostrich_support/ostIn_KGE_DDS.tpl'
# infile_calibparam = '/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/parameter/param_ASG_20221206.csv'
# outfile_ostin_txt = '/glade/u/home/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib/CAMELS_2_OstCalib/run/ostIn.txt'

trial_num = 400
OstrichWarmStart = 'no'

########################################################################################################################
# base files
df_calibparam = pd.read_csv(infile_calibparam)

# base parameter file
dftmp = df_calibparam[df_calibparam['Source'] == 'Param']
if len(dftmp) > 0:
    infile_param = dftmp.iloc[0]['Source_file']
    ds_param = xr.load_dataset(infile_param)
else:
    infile_param = ''
    ds_param = []

# base parameter file
dftmp = df_calibparam[df_calibparam['Source'] == 'Surfdata']
if len(dftmp) > 0:
    infile_surfdata = dftmp.iloc[0]['Source_file']
    ds_surf = xr.load_dataset(infile_surfdata)
else:
    infile_surfdata = ''
    ds_surf = []

# land mask file
cwd = os.getcwd()
os.chdir(pathlib.Path(infile_lndin).parents[2])
out = subprocess.run('./xmlquery LND_DOMAIN_MESH', shell=True, capture_output=True)
infileMESH = out.stdout.decode().strip().split(':')[1].strip()
os.chdir(cwd)
ds_mesh = xr.load_dataset(infileMESH)

########################################################################################################################
# get target parameter list and calculate multiplier range

# get priori parameter file location
with open(infile_lndin, 'r') as f:
    lines_lndin = f.readlines()

df_calibparam['Lower_factor'] = np.nan
df_calibparam['Upper_factor'] = np.nan
df_calibparam['Initi_factor'] = np.nan

for i in range(len(df_calibparam)):
    parami_name = df_calibparam.iloc[i]['Parameter']
    Sourcei = df_calibparam.iloc[i]['Source']
    Methodi = df_calibparam.iloc[i]['Method']

    # get default param value
    if Sourcei == 'Param': # parameter file
        if not parami_name in ds_param.data_vars:
            print(f'Cannot find parameter {parami_name} in {infile_param}!!!')
            parami_values = np.array(np.nan)
        else:
            parami_values = ds_param[parami_name].values
            parami_values = parami_values[parami_values != 0]
    elif Sourcei == 'Surfdata':  # surface data file
        if not parami_name in ds_surf.data_vars:
            print(f'Cannot find parameter {parami_name} in {infile_surfdata}!!!')
            parami_values = np.array(np.nan)
        else:
            elementMask = ds_mesh['elementMask'].values
            parami_values = ds_surf[parami_name].values
            parami_values = parami_values[elementMask == 1]
    elif Sourcei == 'Namelist': # name list file
        flag = False
        for line in lines_lndin:
            line = line.strip()
            if line.startswith(parami_name):
                parami_values = np.array(float(line.split('=')[-1].strip().replace('\'', '').split('d')[0]))
                flag = True
        if flag == False:
            print(f'Cannot find parameter {parami_name} in {infile_lndin}!!!')
            parami_values = np.array(np.nan)
    else:
        sys.exit(f'Unknown Source {Sourcei} for {parami_name}')

    # calculate Ostrich parameter range
    parami_lower = df_calibparam.iloc[i]['Lower']
    parami_upper = df_calibparam.iloc[i]['Upper']
    # parami_priori_min = parami_values.min()
    # parami_priori_max = parami_values.max()
    parami_priori_mean = np.nanmean(parami_values)

    if Methodi == 'Multiplicative':
        if parami_upper <= 0 and parami_lower <= 0:
            # df_calibparam.at[i, 'Lower_factor'] = parami_upper / parami_priori_max
            # df_calibparam.at[i, 'Upper_factor'] = parami_lower / parami_priori_min
            df_calibparam.at[i, 'Lower_factor'] = parami_upper / parami_priori_mean
            df_calibparam.at[i, 'Upper_factor'] = parami_lower / parami_priori_mean
        elif parami_upper >= 0 and parami_lower >= 0:
            # df_calibparam.at[i, 'Upper_factor'] = parami_upper / parami_priori_max
            # df_calibparam.at[i, 'Lower_factor'] = parami_lower / parami_priori_min
            df_calibparam.at[i, 'Upper_factor'] = parami_upper / parami_priori_mean
            df_calibparam.at[i, 'Lower_factor'] = parami_lower / parami_priori_mean
        else:
            sys.exit(f'The upper and lower values of {parami_name} have different signs!')

        if df_calibparam.iloc[i]['Upper_factor'] > 1 and df_calibparam.iloc[i]['Lower_factor'] < 1:
            # start from the default parameter value
            df_calibparam.at[i, 'Initi_factor'] = 1
        else:
            df_calibparam.at[i, 'Initi_factor'] = (df_calibparam.iloc[i]['Upper_factor'] + df_calibparam.iloc[i]['Lower_factor']) / 2

    # elif Methodi == 'Additive': # new param = parami_lower + (parami_upper - parami_lower) * factor
    #     df_calibparam.at[i, 'Upper_factor'] = 1
    #     df_calibparam.at[i, 'Lower_factor'] = 0
    #     df_calibparam.at[i, 'Initi_factor'] = (parami_priori_mean - parami_lower) / (parami_upper - parami_priori_mean)
    elif Methodi == 'Additive': # new param = factor
        df_calibparam.at[i, 'Upper_factor'] = parami_upper - parami_priori_mean
        df_calibparam.at[i, 'Lower_factor'] = parami_lower - parami_priori_mean

        if df_calibparam.iloc[i]['Upper_factor'] > 0 and df_calibparam.iloc[i]['Lower_factor'] < 0:
            # start from the default parameter value
            df_calibparam.at[i, 'Initi_factor'] = 0
        else:
            df_calibparam.at[i, 'Initi_factor'] = (df_calibparam.iloc[i]['Upper_factor'] + df_calibparam.iloc[i]['Lower_factor']) / 2

    else:
        sys.exit(f'Unknown Method {Methodi} for {parami_name}')


########################################################################################################################
# generate ostin txt file

if os.path.exists(outfile_ostin_txt):
    os.remove(outfile_ostin_txt)

with open(infile_ostin_template) as f:
    content = f.readlines()

# find lines specifying parameters (BeginParams to EndParams), and insert parameter factors
for i in range(len(content)):
    line = content[i]
    if line.startswith('BeginParams'):
        lns = i
    if line.startswith('EndParams'):
        lne = i

content1 = content[:lns+1]
content2 = content[lne:]

content1 = content1 + ['#parameter  init  lwr  upr  txInN  txOst  txOut  fmt  \n']
for i in range(len(df_calibparam)):

    parami_name = df_calibparam.iloc[i]['Parameter']
    parami_Ost_name = df_calibparam.iloc[i]['Parameter_Ost']

    factori_lower = df_calibparam.iloc[i]['Lower_factor']
    factori_initi = df_calibparam.iloc[i]['Initi_factor']
    factori_upper = df_calibparam.iloc[i]['Upper_factor']
    if ~np.isnan(factori_lower):
        linei = f'{parami_Ost_name:50} {factori_initi:15.5f} {factori_lower:15.5f} {factori_upper:15.5f}    none    none    none    free\n'
        content1 = content1 + [linei]

content_new = content1 + content2

# write with some new settings
with open(outfile_ostin_txt, 'w') as f:
    for line in content_new:
        if line.startswith('MaxIterations'):
            line = f'MaxIterations {trial_num}\n'
        if line.startswith('OstrichWarmStart'):
            str_old = line.split('#')[0].strip().split(' ')[-1]
            line = line.replace(str_old, OstrichWarmStart)
        _ = f.write(line)

