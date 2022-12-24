#!/bin/bash

# The script run one trial of the calibration: run model, evaluate model (or objective function), update parameter
# Python environment needs to be set before running this script
# module load conda/latest
# conda activate npl-2022b

########################################################################################################################

# pathCTSMcase: CTSM model case folder (i.e., create_newcase output dir)
pathCTSMcase="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/CAMELS_LumpCalib"

# pathOstrich: everything related to Ostrich is saved
pathOstrich="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/CAMELS_LumpCalib_Ostrich"

# parameter file: parameter nc
paramfile_priori="/glade/p/cesmdata/cseg/inputdata/lnd/clm2/paramdata/ctsm51_params.c211112.nc"
paramfile_ostrich="${pathOstrich}/tempdir/ctsm51_params.c211112_Ostrich.nc"

# parameter file: lnd_in namelist
lndinfile_priori="${pathCTSMcase}/Buildconf/clmconf/lnd_in"
lndinfile_ostrich="${pathOstrich}/tempdir/lnd_in"

# calib parameter information
file_param_info="${pathOstrich}/parameters/"

# Evaluation period (could be different from calibration period)
DateEvalStart="2000-01-01"      # start date for calculating statistics
DateEvalEnd="2001-12-31"       # end date for calculating statistics

# scripts
ostrichScriptDir="/glade/u/home/guoqiang/CTSM_repos/CTSM_Guoqiang/Calibration/Ostrich_calib_support"
script_updateparam="${ostrichScriptDir}/update_GlobalParam.py"
script_calculate_stats="${ostrichScriptDir}/cal_metrics_NoRouting.py"
#script_check_job_status="${ostrichScriptDir}/check_job_status.py"

########################### inactive settings (depending on information from pre-defined folder structures)
# reference data for calibration
ref_flow_file="${pathOstrich}/refdata/streamflow_data.csv"

# ostrich run folders and files
ostrichRunDir="${pathOstrich}/run"  # calib directory

file_trial_stats="$ostrichRunDir/trial_stats.txt" # statistics of simulated runoff

# param_txtfile_tpl="$ostrichRunDir/param_factor.tpl"      # template files containing variable names
param_txtfile_ost="$ostrichRunDir/param_factor.txt"      # multiplier values

# user_nl_clm file: change settings
file_user_nl_clm="${pathCTSMcase}/user_nl_clm"

# parameter info (e.g., method, source)
file_ostin_source="${pathOstrich}/parameters/calib_parameter_info.csv"

########################################################################################################################
# ========= start main function ===============================

echo "===== executing trial ====="
echo " "
date | awk '{printf("%s: ---- executing new trial ----\n",$0)}' >> $ostrichRunDir/timetrack.log

# ------------------------------------------------------------------------------
# --- 1.  update params (takes basin id as arg)                              ---
# ------------------------------------------------------------------------------
# the script will also change paramfile path in the first trial

date | awk '{printf("%s: updating params\n",$0)}' >> $ostrichRunDir/timetrack.log
python ${script_updateparam} ${file_ostin_source} ${pathCTSMcase} ${param_txtfile_ost}
echo " "


# ------------------------------------------------------------------------------
# --- 2.  run CTSM                                                           ---
# ------------------------------------------------------------------------------
# this is problematic because the script may not wait for case.submit to finish calculation

date | awk '{printf("%s: submitting CTSM\n",$0)}' >> $ostrichRunDir/timetrack.log

cwd=$PWD
cd ${pathCTSMcase} || exit

## Method-1: run case.submit which costs queue time
#./case.submit
## check job status of case.submit to ensure the below codes are excuted after case.submit is complete
#python ${script_check_job_status} ${pathCTSMcase}

# Method-2: direct run
#Python .case.run
./case.submit --no-batch
#Python case.st_archive

cd $cwd

# ------------------------------------------------------------------------------
# --- 3.  calculate statistics for Ostrich                                   ---
# ------------------------------------------------------------------------------

echo "calculating statistics"
date | awk '{printf("%s: calculating statistics\n",$0)}' >> timetrack.log

\rm $file_trial_stats
python ${script_calculate_stats} $file_trial_stats ${pathCTSMcase} $DateEvalStart $DateEvalEnd $ref_flow_file


date >> trial_stats_allrecords.txt
cat ${file_trial_stats} >> trial_stats_allrecords.txt
echo " " >> trial_stats_allrecords.txt

# ------------------------------------------------------------------------------
# --- Done                                                                   ---
# ------------------------------------------------------------------------------
date | awk '{printf("%s: done with trial\n",$0)}' >> $ostrichRunDir/timetrack.log
wait
exit 0

