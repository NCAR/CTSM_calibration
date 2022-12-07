#!/bin/bash

# save the best calibrated model so far

mode="PreserveBestModel"

script_save_file="/glade/u/home/guoqiang/CTSM_repos/CTSM_Guoqiang/Calibration/Ostrich_calib_support/save_Ostrich_trial_outputs.py"
pathCTSM="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/CAMELS_LumpCalib"
pathOstrichRun="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/CAMELS_LumpCalib_Ostrich/run"
pathOstrichSave="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/CAMELS_LumpCalib_Ostrich/archive"

python ${script_save_file} ${mode} ${pathCTSM} ${pathOstrichRun} ${pathOstrichSave}
