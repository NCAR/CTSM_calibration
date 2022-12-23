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

# infile_lndin = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib/CAMELS_100/Buildconf/clmconf/lnd_in'
# infile_ostin_template = '/glade/u/home/guoqiang/CTSM_repos/CTSM_Guoqiang/Calibration/Ostrich_calib_support/ostIn_KGE_DDS.tpl'
# infile_calibparam = '/glade/u/home/guoqiang/CTSM_repos/CTSM_Guoqiang/Calibration/Calib_params/param_yifan.csv'
# outfile_ostin_txt = '/glade/u/home/guoqiang/test_ostin.txt'

trial_num = 400
OstrichWarmStart = 'no'

########################################################################################################################
# get priori parameter file location
with open(infile_lndin, 'r') as f:
    lines_lndin = f.readlines()

for line in lines_lndin:
    line = line.strip()
    if line.startswith('paramfile'):
        infile_param = line.split('=')[-1].strip().replace('\'', '')
    if line.startswith('fsurdat'):
        infile_surfdata = line.split('=')[-1].strip().replace('\'', '')

ds_param = xr.load_dataset(infile_param)

# get user_nl_clm location
infile_user_nl_clm = str(pathlib.Path(infile_lndin).parents[2]) + '/user_nl_clm'

# get surface data location (replace the one in lndin if user_nl_clm has one)
with open(infile_user_nl_clm) as f:
    for line in f:
        line = line.strip()
        if line.startswith('fsurdat'):
            infile_surfdata = line.split('=')[-1].strip().replace('\'', '')

ds_surf = xr.load_dataset(infile_surfdata)

# get land mask file
cwd = os.getcwd()
os.chdir(pathlib.Path(infile_lndin).parents[2])
out = subprocess.run('./xmlquery LND_DOMAIN_MESH', shell=True, capture_output=True)
infileMESH = out.stdout.decode().strip().split(':')[1].strip()
os.chdir(cwd)

ds_mesh = xr.load_dataset(infileMESH)

########################################################################################################################
# get target parameter list and calculate multiplier range
df_calibparam = pd.read_csv(infile_calibparam)
df_calibparam['Lower_mtp'] = np.nan
df_calibparam['Upper_mtp'] = np.nan
df_calibparam['Initi_mtp'] = np.nan

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
        if not parami_name in ds_param.data_vars:
            print(f'Cannot find parameter {parami_name} in {infile_surfdata}!!!')
            parami_values = np.array(np.nan)
        else:
            elementMask = ds_mesh['elementMask'].values
            parami_values = ds_mesh[parami_name].values
            parami_values = parami_values[elementMask == 1]
    elif Sourcei == 'Namelist': # name list file
        flag = False
        for line in lines_lndin:
            if line.startswith(parami_name):
                parami_values = np.array(float(line.split('=')[-1].strip().replace('\'', '').split('d')[0]))
                flag = True
        if flag == False:
            print(f'Cannot find parameter {parami_name} in {infile_lndin}!!!')
            parami_values = np.array(np.nan)
    else:
        sys.exit(f'Unknown Source {Sourcei} for {parami_name}')

    # calculate Ostrich parameter range
    if Methodi == 'Multiplicative':
        parami_priori_min = parami_values.min()
        parami_priori_max = parami_values.max()
        parami_priori_mean = parami_values.mean()
        parami_lower = df_calibparam.iloc[i]['Lower']
        parami_upper = df_calibparam.iloc[i]['Upper']
        if parami_upper < 0 or parami_lower < 0:
            # df_calibparam.at[i, 'Lower_mtp'] = parami_upper / parami_priori_max
            # df_calibparam.at[i, 'Upper_mtp'] = parami_lower / parami_priori_min
            df_calibparam.at[i, 'Lower_mtp'] = parami_upper / parami_priori_mean
            df_calibparam.at[i, 'Upper_mtp'] = parami_lower / parami_priori_mean
        else:
            # df_calibparam.at[i, 'Upper_mtp'] = parami_upper / parami_priori_max
            # df_calibparam.at[i, 'Lower_mtp'] = parami_lower / parami_priori_min
            df_calibparam.at[i, 'Upper_mtp'] = parami_upper / parami_priori_mean
            df_calibparam.at[i, 'Lower_mtp'] = parami_lower / parami_priori_mean
        if df_calibparam.iloc[i]['Upper_mtp'] > 1 and df_calibparam.iloc[i]['Lower_mtp'] < 1:
            df_calibparam.at[i, 'Initi_mtp'] = 1
        else:
            df_calibparam.at[i, 'Initi_mtp'] = (df_calibparam.iloc[i]['Upper_mtp'] + df_calibparam.iloc[i]['Lower_mtp']) / 2
    # elif Methodi == 'Additive': # new param = parami_lower + (parami_upper - parami_lower) * mtp
    #     df_calibparam.at[i, 'Upper_mtp'] = 1
    #     df_calibparam.at[i, 'Lower_mtp'] = 0
    #     df_calibparam.at[i, 'Initi_mtp'] = (parami_priori_mean - parami_lower) / (parami_upper - parami_priori_mean)
    elif Methodi == 'Additive': # new param = mtp
        df_calibparam.at[i, 'Upper_mtp'] = parami_upper
        df_calibparam.at[i, 'Lower_mtp'] = parami_lower
        df_calibparam.at[i, 'Initi_mtp'] = parami_priori_mean
    else:
        sys.exit(f'Unknown Method {Methodi} for {parami_name}')


    ########################################################################################################################
# generate ostin txt file

if os.path.exists(outfile_ostin_txt):
    os.remove(outfile_ostin_txt)

f_out = open(outfile_ostin_txt, 'w')
with open(infile_ostin_template) as f:
    content = f.readlines()

# find lines specifying parameters, and remove previous paramters
for i in range(len(content)):
    line = content[i]
    if line.startswith('BeginParams'):
        lns = i
    if line.startswith('EndParams'):
        lne = i

content1 = content[:lns+1]
content2 = content[lne:]

# add new params
content1 = content1 + ['#parameter  init  lwr  upr  txInN  txOst  txOut  fmt  \n']
for i in range(len(df_calibparam)):
    parami_name = df_calibparam.iloc[i]['Parameter']
    mtpi_name = parami_name + '_mtp'
    mtpi_lower = df_calibparam.iloc[i]['Lower_mtp']
    mtpi_initi = df_calibparam.iloc[i]['Initi_mtp']
    mtpi_upper = df_calibparam.iloc[i]['Upper_mtp']
    if ~np.isnan(mtpi_lower):
        linei = f'{mtpi_name}  {mtpi_initi:.4f} {mtpi_lower:.4f} {mtpi_upper:.4f}  none  none  none  free\n'
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


