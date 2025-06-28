# Create config.toml files for CAMELS basins
# This uses MO-ASMO ability to do sensitivity analysis (one iteration)

import pandas as pd
import numpy as np
import os, toml, sys, glob
from datetime import datetime
import decide_CalibValid_periods as decidePeriod

########################################################################################################################
# settings
basin_num = int(sys.argv[1])
level = sys.argv[2]

# level = 'level1'
clonecase = 'level1_0'

projectcode = 'NCGD0013'

infile_basin_info = f'/glade/work/guoqiang/CTSM_CAMELS/data_mesh_surf/HillslopeHydrology/CAMELS_{level}_basin_info.csv'
inpath_camels_data = '/glade/campaign/ral/hap/common/camels/obs_flow_met'
inpath_Qobs = '/glade/work/guoqiang/CTSM_CAMELS/CAMLES_Qobs'

outpath_case = '/glade/work/guoqiang/CTSM_CAMELS/SA_HH_MOASMO'
outpath_out = '/glade/derecho/scratch/guoqiang/CTSM_outputs/CAMELS_Calib/SA_HH_MOASMO'
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
                'name': 'CTSM calibration',
                'date': '2024-01',
                'affiliation': 'CGD/TSS NCAR'}

create_case_settings = "--machine derecho --compset I2000Clm51Sp --driver nuopc --compiler intel --res f09_g16 --handle-preexisting-dirs r --run-unsupported"

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

########################################################################################################################
# CTSM configurations
config_CTSM = {}
config_CTSM['files'] = {}
config_CTSM['files']['path_CTSM_source'] = '/glade/u/home/guoqiang/CTSM_repos/CTSM_hillslope'
config_CTSM['files']['path_CTSM_case'] = f'{outpath_case}/{level}_{basin_num}'
config_CTSM['files']['path_CTSM_CIMEout'] = f'{outpath_out}/{level}_{basin_num}'
config_CTSM['files']['file_CTSM_mesh'] = f'/glade/work/guoqiang/CTSM_CAMELS/data_mesh_surf/HillslopeHydrology/disaggregation/corrected_HCDN_nhru_final_671_buff_fix_holes.CAMELSandTDX_areabias_fix.simp0.001.{level}_polygons_neighbor_group_esmf_mesh_{basin_num}.nc'


config_CTSM['files']['file_CTSM_surfdata'] = f'/glade/work/guoqiang/CTSM_CAMELS/data_mesh_surf/HillslopeHydrology/disaggregation/surfdata_CAMELSandTDX_areabias_fix.simp0.001.{level}_hist_78pfts_CMIP6_simyr2000_HAND_4_col_hillslope_geo_params_nlcd_bedrock_{basin_num}.nc'

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
config_CTSM['AddToNamelist']['user_nl_datm_streams'] = ['']
config_CTSM['AddToNamelist']['user_nl_datm'] = ['']


config_CTSM['AddToNamelist']['user_nl_clm'] = ["use_hillslope = .true.", "use_hillslope_routing = .true.", "n_dom_pfts = 2", 
                                              "hist_nhtfrq = 0,-24", "hist_mfilt = 1,365", 
                                              "hist_fincl2 = 'QRUNOFF','H2OSNO','H2OSFC','ZWT','ZWT_PERCH','SOILWATER_10CM','EFLX_LH_TOT','QDRAI','QDRAI_PERCH','QOVER','QH2OSFC','QFLX_SNOW_DRAIN','RAIN','SOILLIQ','SOILICE','VOLUMETRIC_STREAMFLOW','STREAM_WATER_DEPTH','STREAM_WATER_VOLUME'"]

config_CTSM['replacefiles'] = {}
config_CTSM['replacefiles']['user_nl_datm_streams'] = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/{level}_{basin_num}/user_nl_datm_streams'
config_CTSM['replacefiles']['user_nl_clm'] = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/{level}_{basin_num}/user_nl_clm'

########################################################################################################################
# calibration configurations

config_calib = {}
config_calib['files'] = {}
config_calib['files']['path_script_calib'] = '/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration'
config_calib['files']['path_script_MOASMO'] = "/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support"
config_calib['files']['file_calib_param'] = '/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/parameter/param_PPE_DailyQ.csv'
config_calib['files']['file_Qobs'] = file_Qobs
config_calib['files']['path_calib'] = f"/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/SA_HH_MOASMO/{level}_{basin_num}_SA" # if not provided, just use default settings (i.e., a folder within the same folder with the CTSM case)

config_calib['eval'] = {}
config_calib['eval']['ignore_month'] = 12

config_calib['job'] = {}
config_calib['job']['jobsetting'] = ['#PBS -N OstrichCalib', '#PBS -q share', '#PBS -l walltime=6:00:00']
# config_calib['job']['jobsetting'] = ['#PBS -N OstrichCalib', '#PBS -q casper', '#PBS -l walltime=24:00:00']

config_calib['settings'] = {}
config_calib['settings']['sampling_method'] = 'lh'
config_calib['settings']['num_init'] = 256 # initial number of samples
config_calib['settings']['num_per_iter'] = 36 # number of selected pareto parameter sets for each iteration
config_calib['settings']['num_iter'] = 1 # including the initial iteration

config_calib['job'] = {}
# config_calib['job']['job_mode'] = 'lumpsubmit' # lumpsubmit can do, for example, run 36 one-basin calibrations within one node to fully utilize one node. every iteration will be resubmitted, so there will much queue time. It can supprot resubmit function, which means if there are many iterations, the control job will resubmit itself to extend the run time.

config_calib['job']['job_mode'] = 'lumpsubmit_allinone' # lumpsubmit_allinone. This will run everything within the control job, which means only the job_controlMOASMO will be submitted, and other iterations will be within this job. No need to submit other jobs. make sure 

# one job that controls all jobs, and perform basic functions such as parameter generation and emulator construction
config_calib['job']['job_controlMOASMO'] = ['#PBS -N MOAcontrol', '#PBS -q main', '#PBS -l select=1:ncpus=128', '#PBS -l walltime=12:00:00', f'#PBS -A {projectcode}']

if config_calib['job']['job_mode'] == 'lumpsubmit_allinone':
    config_calib['job']['job_CTSMiteration'] = config_calib['job']['job_controlMOASMO'] # only the cpu information will be used
else:
    # job to run each iteration (including iteration 0). multiple CTMS cases will be run
    config_calib['job']['job_CTSMiteration'] = ['#PBS -N CTSMiter', '#PBS -q main', '#PBS -l select=1:ncpus=128', '#PBS -l walltime=12:00:00', f'#PBS -A {projectcode}']


########################################################################################################################
# spinup configurations
config_spinup = {}
config_spinup['spinup_mode'] = "continuous" # continuous: period before RUN_STARTDATE. No other option for now
config_spinup['spinup_month'] = months_difference
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
