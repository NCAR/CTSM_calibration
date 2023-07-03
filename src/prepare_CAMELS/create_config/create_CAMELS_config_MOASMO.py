# Create config.toml files for CAMELS basins

import pandas as pd
import numpy as np
import os, toml, sys
import decide_CalibValid_periods as decidePeriod


########################################################################################################################
# settings
basin_num = int(sys.argv[1])
# basin_num = 0

infile_basin_info = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/info_ESMFmesh_ctsm_HCDN_nhru_final_671.buff_fix_holes_polygons_simplified_5e-4_split_nested.csv'
inpath_camels_data = '/glade/work/guoqiang/basin_dataset/camels_data'

outpath_case = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest_MOASMO'
outpath_out = '/glade/scratch/guoqiang/CTSM_outputs/CAMELS_Calib/Lump_calib_split_nest_MOASMO'
outpath_config = f'{outpath_case}/configuration'
outfile_config = f'{outpath_config}/CAMELS-{basin_num}_config.toml'

os.makedirs(outpath_case, exist_ok=True)
os.makedirs(outpath_out, exist_ok=True)
os.makedirs(outpath_config, exist_ok=True)

########################################################################################################################
# basic configurations
config_intro = {'author': 'Guoqiang Tang',
                'version': '0.0.1',
                'name': 'CTSM calibration',
                'date': '2023-07',
                'affiliation': 'NCAR CGD'}

create_case_settings = "--machine cheyenne --compset I2000Clm51Sp --driver nuopc --compiler intel --res f09_g16 --handle-preexisting-dirs r --run-unsupported"

config_HPC = {'projectCode': 'P08010000'}

########################################################################################################################
# decide calibration period

df_info = pd.read_csv(infile_basin_info)
id = df_info.iloc[basin_num]['hru_id']
file_Qobs = df_info.iloc[basin_num]['file_obsQ']
data, date = decidePeriod.get_tmean_series_masked_by_q(inpath_camels_data, id)

# # method-1
# settings = {}
# settings['method'] = 1
# settings['calibyears'] = 5 # how many years are used to calibrate the model
# settings['validratio'] = 0.8 # ratio of valid Q records during the period
# settings['trial_start_date'] = '1985-10-01' # only use data after this period. month can be used to define the start of a water year
# RUN_STARTDATE, STOP_N, STOP_OPTION, STOP_DATE = decidePeriod.calibration_period_CTSMformat(data, date, settings)

# method-2
settings = {}
settings['method'] = 2
settings['startmonth'] = 10 #  the start of a year (e.g., 1 or 10)
settings['periodlength'] = 5 # calib years
settings['window'] = 5 # years of rolling mean
RUN_STARTDATE, STOP_N, STOP_OPTION, STOP_DATE = decidePeriod.calibration_period_CTSMformat(data, date, settings)

# # Method-3: default start period to utilize the existing restart file
# STOP_OPTION = 'nmonths'
# STOP_N = 36 # 3 years
# RUN_STARTDATE = '2000-01-01'
# STOP_DATE = '2002-12-31'

########################################################################################################################
# CTSM configurations
config_CTSM = {}
config_CTSM['files'] = {}
config_CTSM['files']['path_CTSM_source'] = '/glade/u/home/guoqiang/CTSM_repos/CTSM'
config_CTSM['files']['path_CTSM_case'] = f'{outpath_case}/CAMELS_{basin_num}'
config_CTSM['files']['path_CTSM_CIMEout'] = f'{outpath_out}/CAMELS_{basin_num}'
config_CTSM['files']['file_CTSM_mesh'] = f'/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_basin_mask_split_nest/ESMFmesh_ctsm_HCDN_nhru_final_671.buff_fix_holes_polygons_simplified_5e-4_split_nested_basin{basin_num}.nc'
config_CTSM['files']['file_CTSM_surfdata'] = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/surfdata_CAMELS_split_nested_hist_78pfts_CMIP6_simyr2000_c230105.nc'

config_CTSM['settings'] = {}

if basin_num == 0:
    print('Will create new CTSM case from scratch.')
    config_CTSM['settings']['CLONEROOT'] = ''
else:
    print('Will clone CTSM case from basin 0 using --keepexe')
    config_CTSM['settings']['CLONEROOT'] = f'{outpath_case}/CAMELS_0'

config_CTSM['settings']['CLONEsettings'] = "--keepexe"
config_CTSM['settings']['createcase'] = create_case_settings
config_CTSM['settings']['RUN_STARTDATE'] = RUN_STARTDATE
config_CTSM['settings']['STOP_N'] = STOP_N
config_CTSM['settings']['STOP_OPTION'] = STOP_OPTION
config_CTSM['settings']['NTASKS'] = 1
config_CTSM['settings']['casebuild'] = 'direct'
config_CTSM['settings']['subset_length'] = 'all'
config_CTSM['settings']['forcing_YearStep'] = 5

config_CTSM['AddToNamelist'] = {}
config_CTSM['AddToNamelist']['user_nl_datm_streams'] = ['topo.observed:meshfile=/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/topo_data/ESMFmesh_ctsm_elev_Conus_0.125d_210810.cdf5.nc',
                                                        'topo.observed:datafiles=/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/topo_data/ctsm_elev_Conus_0.125d.cdf5.nc']
config_CTSM['AddToNamelist']['user_nl_datm'] = ['']
config_CTSM['AddToNamelist']['user_nl_clm'] = ['']

########################################################################################################################
# calibration configurations

config_calib = {}
config_calib['files'] = {}
config_calib['files']['path_script_calib'] = '/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration'
config_calib['files']['path_script_MOASMO'] = "/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support"
config_calib['files']['file_calib_param'] = '/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/parameter/param_ASG_20221206_moasmo.csv'
config_calib['files']['file_Qobs'] = file_Qobs
config_calib['eval'] = {}
config_calib['eval']['ignore_month'] = 12
config_calib['job'] = {}
config_calib['job']['jobsetting'] = ['#PBS -N OstrichCalib', '#PBS -q share', '#PBS -l walltime=6:00:00']
# config_calib['job']['jobsetting'] = ['#PBS -N OstrichCalib', '#PBS -q casper', '#PBS -l walltime=24:00:00']


config_calib['settings'] = {}
config_calib['settings']['sampling_method'] = 'glp'
config_calib['settings']['num_init'] = 72 # initial number of samples
config_calib['settings']['num_per_iter'] = 20 # number of selected pareto parameter sets for each iteration
config_calib['settings']['num_iter'] = 15 # including the initial iteration


config_calib['job'] = {}
config_calib['job']['job_mode'] = 'lumpsubmit'

# job to run each iteration (including iteration 0). multiple CTMS cases will be run
config_calib['job']['job_CTSMiteration'] = ['#PBS -N CTSMiter', '#PBS -q regular', '#PBS -l select=1:ncpus=36', '#PBS -l walltime=12:00:00', '#PBS -A P08010000']

# one job that controls all jobs, and perform basic functions such as parameter generation and emulator construction
config_calib['job']['job_controlMOASMO'] = ['#PBS -N MOAcontrol', '#PBS -q share', '#PBS -l select=1:ncpus=1', '#PBS -l walltime=6:00:00', '#PBS -A P08010000']

########################################################################################################################
# spinup configurations
config_spinup = {}
config_spinup['spinup_mode'] = "continuous" # continuous: period before RUN_STARTDATE. No other option for now
config_spinup['spinup_month'] = 60 # 5-year spin up
config_spinup['force_Jan_start'] = True # CTSM default initial conditions start at Jan 1st. So, using Jan as the start date could be better
config_spinup['update_restart'] = True # after spin-up is done, add the restart file to user_nl_clm

########################################################################################################################
# save configurations to toml
config_all = {}
config_all['intro'] = config_intro
config_all['HPC'] = config_HPC
config_all['CTSM'] = config_CTSM
config_all['calib'] = config_calib
config_all['spinup'] = config_spinup

with open(outfile_config, 'w') as f:
    toml.dump(config_all, f)