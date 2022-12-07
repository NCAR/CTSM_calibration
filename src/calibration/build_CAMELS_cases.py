# Build cases for CAMELS basins. one basin one case
# call step1, step2

import pandas as pd
import numpy as np
import os, sys, subprocess
import decide_CalibValid_periods as decidePeriod

########################################################################################################################
# settings
# basin_num = int(sys.argv[1])
basin_num = 11
infile_basin_info = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/Sean_MESH_CAMELS_basin_info.csv'

# less active settings
script_step1 = '/glade/u/home/guoqiang/CTSM_repos/CTSM_Guoqiang/Calibration/OneBasin/step1_create_case_SingleBasin.py'
script_step4 = '/glade/u/home/guoqiang/CTSM_repos/CTSM_Guoqiang/Calibration/OneBasin/step4_create_ostrich_settings.py'
script_step5 = '/glade/u/home/guoqiang/CTSM_repos/CTSM_Guoqiang/Calibration/OneBasin/step5_subset_datm_forcing.py'
inpathOstSource = "/glade/u/home/guoqiang/CTSM_repos/CTSM_Guoqiang/Calibration/Ostrich_calib_support"
infileCalibParam = "/glade/u/home/guoqiang/CTSM_repos/CTSM_Guoqiang/Calibration/Calib_params/param_yifan.csv"

inpathCTSMcase = f"/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib/CAMELS_{basin_num}"

infileMESH_divide = f'/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_basin_divide/ESMFmesh_ctsm_HCDN_nhru_final_671_v0_8e-3_basin{basin_num}.nc'

ignore_year = 1 # the first part of simulation is ignored when evaluating the model for better spin up
buildoption = 'direct' # direct or qcmd. no need to use qcmd if this script is submitted through qbs

########################################################################################################################
# basin info
df_info = pd.read_csv(infile_basin_info)
infile_Qobs = df_info.iloc[basin_num]['file_obsQ']

########################################################################################################################
# Create model case

# ############### Different methods defining calibration period:
# ############### Method-1
# # use streamflow to decide calibration period
# calibyears = 5 # how many years are used to calibrate the model
# validratio = 0.8 # ratio of valid Q records during the period
# trial_start_date = '1980-01-01' # only use data after this period
# year_start, year_end = decidePeriod.calib_period_QBeginning(infile_Qobs, calibyears, validratio, trial_start_date)
# RUN_STARTDATE = f'{year_start}-10-01'
# STOP_DATE = f'{year_end}-09-30'
# STOP_N = calibyears * 12 # for STOP_OPTION: nmonths
#
# ############### Method-2: climate anomaly years
# # use streamflow to decide calibration period
# startmonth = 10 #  the start of a year (e.g., 1 or 10)
# periodlength = 5 # calib years
# window = 5 # years of rolling mean
# period_extreme = decidePeriod.cal_period_extreme(infile_Qobs, startmonth=10, periodlength=5, window=5)
# RUN_STARTDATE = period_extreme['date_start'].loc['min']
# STOP_DATE = period_extreme['date_end'].loc['min']
# # diff = pd.Timestamp(STOP_DATE).to_period('M') - pd.Timestamp(RUN_STARTDATE).to_period('M')
# # STOP_N = diff.n + 1
# STOP_N = periodlength * 12 # for STOP_OPTION: nmonths

############### Method-3: default start period to utilize the existing restart file
STOP_N = 36 # 3 years
RUN_STARTDATE = '2000-01-01'
STOP_DATE = '2002-12-31'

_ = subprocess.run(f'python {script_step1} {basin_num} {STOP_N} {RUN_STARTDATE} {buildoption}', shell=True)

########################################################################################################################
# Create Ostrich settings
DateEvalStart = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(years=1)).strftime('%Y-%m-%d') # ignor the first year when evaluating model
DateEvalEnd = STOP_DATE

_ = subprocess.run(f'python {script_step4} {inpathCTSMcase} {inpathOstSource} {infileCalibParam} {infile_Qobs} {DateEvalStart} {DateEvalEnd}', shell=True)

########################################################################################################################
# Model spin up
# empty for now. does not affect the previous and next steps

########################################################################################################################
# Create forcing subsets and change model settings
_ = subprocess.run(f'python {script_step5} {inpathCTSMcase} {infileMESH_divide}', shell=True)
