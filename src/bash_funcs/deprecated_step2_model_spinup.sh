#!/bin/bash

# model spin up is needed to get reasonable initial state for a model
# there are several ways to do spin up. Here, I run the model during a period before the formal simulation period to get restart files.



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

# pathSpinup: where spinup outputs are saved
pathSpinup="${pathParent}/${casename}_Spinup1"

# compress the model case and save it in the spin up folder
fileCTSMcaseBackup="${pathSpinup}/${casename}.tar.gz"

# date of restart file
dateRestart=2000-01-01

########################################################################################################################
# 2. run the model

cd ${pathCTSMcase}

# change the run time of mode case (critical)
./xmlchange STOP_N=2
./xmlchange STOP_OPTION=nyears
./xmlchange RUN_STARTDATE=1998-01-01
./xmlquery RUN_STARTDATE,STOP_N,STOP_OPTION

# submit
./case.submit

# check whether spinup run is successful
tail CaseStatus

########################################################################################################################
# 3. copy restart files and relevant settings to a folder

mkdir -p ${pathSpinup}

# compress model files
tar -czf ${fileCTSMcaseBackup} ${pathCTSMcase}

# copy restart files
pathSimu="/glade/scratch/guoqiang/${casename}/run"
filenameRestart="${casename}.clm2.r.${dateRestart}-00000.nc"
cp ${pathSimu}/${filenameRestart} ${pathSpinup}

########################################################################################################################
# 4. change restart file in the model case

cd ${pathCTSMcase}
echo "finidat='${pathSpinup}/${filenameRestart}'" >> user_nl_clm


