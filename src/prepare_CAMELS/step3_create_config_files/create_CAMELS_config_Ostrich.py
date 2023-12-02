# Create config.toml files for CAMELS basins

import pandas as pd
import numpy as np
import os, toml, sys
from datetime import datetime
import decide_CalibValid_periods as decidePeriod


########################################################################################################################
# settings
basin_num = int(sys.argv[1])
level = sys.argv[2]

# level = 'level1'
clonecase = 'level1_0'

projectcode = 'NCGD0013'

infile_basin_info = f'/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/data_mesh_surf/HillslopeHydrology/CAMELS_{level}_basin_info.csv'
inpath_camels_data = '/glade/p/ral/hap/common_data/camels/obs_flow_met'
inpath_Qobs = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/CAMLES_Qobs'

outpath_case = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_Ostrich'
outpath_out = '/glade/scratch/guoqiang/CTSM_outputs/CAMELS_Calib/Calib_all_HH_Ostrich'
outpath_config = f'{outpath_case}/configuration'
outfile_config = f'{outpath_config}/{level}-{basin_num}_config.toml'

os.makedirs(outpath_case, exist_ok=True)
os.makedirs(outpath_out, exist_ok=True)
os.makedirs(outpath_config, exist_ok=True)

calibyears = 6
wy_month = 10 # October: water year start

########################################################################################################################
# basic configurations
config_intro = {'author': 'Guoqiang Tang',
                'version': '0.0.1',
                'name': 'CAMELS-CTSM',
                'date': '2023-11',
                'affiliation': 'CGD/TSS, NCAR'}

create_case_settings = "--machine cheyenne --compset I2000Clm51Sp --driver nuopc --compiler intel --res f09_g16 --handle-preexisting-dirs r --run-unsupported"

config_HPC = {'projectCode': projectcode}

########################################################################################################################
# decide calibration period

df_info = pd.read_csv(infile_basin_info)

if len(df_info)<basin_num:
    sys.exit(f'basin_num {basin_num} exceeds the length of basin info csv')

hru_id = df_info.iloc[basin_num]['hru_id']
name = df_info.iloc[basin_num]['file_obsQ'].split('/')[-1]
file_Qobs = f'{inpath_Qobs}/{name}'

df_q = decidePeriod.read_raw_CAMELS_Q_to_df(file_Qobs)
date = df_q['date'].values
data = df_q['Qobs'].values

# the most recent period is used for calibration
trial_start_date, trial_end_date = decidePeriod.calib_period_Ending(data, date, calibyears, validratio=0.8, wy_month=10)
RUN_STARTDATE = trial_start_date
STOP_N = calibyears * 12
STOP_OPTION = 'nmonths'

# decide the spin up period uisng the data starts from 1950-01
input_date = datetime.strptime(trial_start_date, '%Y-%m-%d')
reference_date = datetime(1951, 1, 1)
months_difference = (input_date.year - reference_date.year) * 12 + (input_date.month - reference_date.month)



# # method-1
# settings = {}
# settings['method'] = 1
# settings['calibyears'] = 5 # how many years are used to calibrate the model
# settings['validratio'] = 0.8 # ratio of valid Q records during the period
# settings['trial_start_date'] = '1985-10-01' # only use data after this period. month can be used to define the start of a water year
# RUN_STARTDATE, STOP_N, STOP_OPTION, STOP_DATE = decidePeriod.calibration_period_CTSMformat(data, date, settings)

# # method-2
# settings = {}
# settings['method'] = 2
# settings['startmonth'] = 10 #  the start of a year (e.g., 1 or 10)
# settings['periodlength'] = 5 # calib years
# settings['window'] = 5 # years of rolling mean
# RUN_STARTDATE, STOP_N, STOP_OPTION, STOP_DATE = decidePeriod.calibration_period_CTSMformat(data, date, settings)

# # Method-3: default start period to utilize the existing restart file
# STOP_OPTION = 'nmonths'
# STOP_N = 36 # 3 years
# RUN_STARTDATE = '2000-01-01'
# STOP_DATE = '2002-12-31'

########################################################################################################################
# CTSM configurations
config_CTSM = {}
config_CTSM['files'] = {}
config_CTSM['files']['path_CTSM_source'] = '/glade/u/home/guoqiang/CTSM_repos/CTSM_hillslope'
config_CTSM['files']['path_CTSM_case'] = f'{outpath_case}/{level}_{basin_num}'
config_CTSM['files']['path_CTSM_CIMEout'] = f'{outpath_out}/{level}_{basin_num}'
config_CTSM['files']['file_CTSM_mesh'] = f'/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/data_mesh_surf/HillslopeHydrology/disaggregation/corrected_HCDN_nhru_final_671_buff_fix_holes.CAMELSandTDX_areabias_fix.simp0.001.{level}_polygons_neighbor_group_esmf_mesh_{basin_num}.nc'
config_CTSM['files']['file_CTSM_surfdata'] = f'/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/data_mesh_surf/HillslopeHydrology/disaggregation/surfdata_CAMELS_{level}_hist_78pfts_CMIP6_simyr2000_c231117_{basin_num}.nc'

config_CTSM['settings'] = {}

if basin_num == 0 and level == 'level1':
    print('Will create new CTSM case from scratch.')
    config_CTSM['settings']['CLONEROOT'] = ''
else:
    print('Will clone CTSM case from basin 0 using --keepexe')
    config_CTSM['settings']['CLONEROOT'] = f'{outpath_case}/{clonecase}'

config_CTSM['settings']['CLONEsettings'] = "--keepexe"
config_CTSM['settings']['createcase'] = create_case_settings
config_CTSM['settings']['RUN_STARTDATE'] = RUN_STARTDATE
config_CTSM['settings']['STOP_N'] = STOP_N
config_CTSM['settings']['STOP_OPTION'] = STOP_OPTION
config_CTSM['settings']['NTASKS'] = 1
config_CTSM['settings']['casebuild'] = 'direct'
config_CTSM['settings']['subset_length'] = 'existing'
config_CTSM['settings']['forcing_YearStep'] = 5 # merge monthly/yearly forcings to the target time step

config_CTSM['AddToNamelist'] = {}
config_CTSM['AddToNamelist']['user_nl_datm_streams'] = ['topo.observed:meshfile=/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/data_topo/ESMFmesh_ctsm_elev_Conus_0.125d_210810.cdf5.nc',
                                                        'topo.observed:datafiles=/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/data_topo/ctsm_elev_Conus_0.125d.cdf5.nc']
config_CTSM['AddToNamelist']['user_nl_datm'] = ['']
config_CTSM['AddToNamelist']['user_nl_clm'] = ['']

config_CTSM['replacefiles'] = {}
config_CTSM['replacefiles']['user_nl_datm_streams'] = '/glade/campaign/cgd/tss/common/lm_forcing/hybrid/era5land_eme/user_nl_datm_streams'

########################################################################################################################
# calibration configurations
config_calib = {}
config_calib['files'] = {}
config_calib['files']['path_script_calib'] = '/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration'
config_calib['files']['path_script_Ostrich'] = '/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/Ostrich_support'
config_calib['files']['file_calib_param'] = '/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/parameter/param_ASG_20221206.csv'
config_calib['files']['file_Qobs'] = file_Qobs
config_calib['files']['path_calib'] = f"/glade/campaign/cgd/tss/people/guoqiang/CTSMcases/CAMELS_Calib/Calib_all_HH_Ostrich/{level}_{basin_num}_Ostrich" # if not provided, just use default settings (i.e., a folder within the same folder with the CTSM case)

config_calib['eval'] = {}
config_calib['eval']['ignore_month'] = 12 # the first few months are ignored during evaluation due to spin up
config_calib['job'] = {}
config_calib['job']['jobsetting'] = ['#PBS -N OstrichCalib', '#PBS -q share', '#PBS -l walltime=6:00:00']
# config_calib['job']['jobsetting'] = ['#PBS -N OstrichCalib', '#PBS -q casper', '#PBS -l walltime=24:00:00']

########################################################################################################################
# spinup configurations
config_spinup = {}
config_spinup['spinup_mode'] = "continuous" # continuous: period before RUN_STARTDATE. No other option for now
config_spinup['spinup_month'] = months_difference # 5-year spin up
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