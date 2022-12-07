#!/bin/bash

# Model spinup run and formal model run have different settings such as periods
# Currently, I use the same model case (i.e., pathCTSMcase) folder for the two runs. Therefore, this script is needed.
# Probably, a better organization is to build two different cases for spinup and formal run. But this step could still be useful
# because users may want to make some changes to model settings before calibration, which are not necessary done in step1


# environment
module load conda
conda activate npl-2022b

########################################################################################################################
# 1. paths and files

# pathParent: everything of the model will be saved here
pathParent="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib"

# casename
casename="CAMELS_LumpCalib"

# pathCTSMcase: CTSM model case folder
pathCTSMcase="${pathParent}/${casename}"

########################################################################################################################
# 2. run the model

cd ${pathCTSMcase}

# change the run time of mode case (critical)
./xmlchange STOP_N=2
./xmlchange STOP_OPTION=nyears
./xmlchange RUN_STARTDATE=2001-01-01
./xmlquery RUN_STARTDATE,STOP_N,STOP_OPTION


