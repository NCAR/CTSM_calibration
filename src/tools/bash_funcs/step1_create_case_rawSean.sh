#!/bin/bash

# Create a CTSM model case
# Settings follow Sean's CAMELS example: /glade/u/home/swensosc/cases/hillslope_CAMELS
# Note: manual check of settings are needed before running this script

# environment
module load conda
conda activate npl-2022b

########################################################################################################################
# 1. inpaths and files

# pathCTSMrepo: CTSM model source used to create model cases. git clone from Github repos
pathCTSMrepo="/glade/u/home/guoqiang/CTSM_repos/CTSM"

# pathParent: everything of the model will be saved here
pathParent="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib2"

# pathBasinData: basin domain information, surface data information ...
# pathBasinData_source: where do the basin data come from? This case is based on Sean's data. Not necessary if pathBasinData already contains files needed
pathBasinData_source="/glade/work/guoqiang/CTSM_cases/CAMELS_Sean/shared_data_Sean"
pathBasinData="${pathParent}/shared_data_Sean"

# pathCTSMcase: CTSM model case folder
pathCTSMcase="${pathParent}/CAMELS_LumpCalib22"

# projectCode: a project is needed to run on Cheyenne
projectCode="P08010000"

########################################################################################################################
# 2. create folders and populate files

mkdir -p ${pathBasinData}

cp -r ${pathBasinData_source}/* ${pathBasinData}

########################################################################################################################
# 3. create model cases

################################

# (1) create new case
newcase_settings="--compset I2000Clm51Sp --driver nuopc --compiler intel --res f09_g16 --run-unsupported --project ${projectCode} --machine cheyenne"
${pathCTSMrepo}/cime/scripts/create_newcase --case ${pathCTSMcase} ${newcase_settings}

cd ${pathCTSMcase} || exit

# (2) set up
./case.setup

################################
# (3) change settings (optional)

# change user_nl_clm
echo "fsurdat = '${pathBasinData}/surfdata_CAMELS_hist_78pfts_CMIP6_simyr2000_c221004.nc'" >> user_nl_clm
echo "hist_nhtfrq = 0,-24"  >> user_nl_clm
echo "hist_mfilt = 1,365"  >> user_nl_clm
echo "hist_fincl2 = 'QRUNOFF','H2OSNO','ZWT','SOILWATER_10CM','EFLX_LH_TOT','QDRAI','QOVER','RAIN'"  >> user_nl_clm

# change land domain and MESH files
./xmlchange ATM_DOMAIN_MESH=${pathBasinData}/ESMFmesh_ctsm_HCDN_nhru_final_671_v0_8e-3.nc
./xmlchange LND_DOMAIN_MESH=${pathBasinData}/ESMFmesh_ctsm_HCDN_nhru_final_671_v0_8e-3.nc
./xmlchange MASK_MESH=${pathBasinData}/ESMFmesh_ctsm_HCDN_nhru_final_671_v0_8e-3.nc
./xmlquery ATM_DOMAIN_MESH,LND_DOMAIN_MESH,MASK_MESH

# change forcing data
./xmlchange DATM_MODE=CLMNLDAS2
./xmlquery DATM_MODE

# change the run time of mode case
./xmlchange STOP_N=2
./xmlchange STOP_OPTION=nyears
./xmlchange RUN_STARTDATE=2000-01-01
./xmlquery RUN_STARTDATE,STOP_N,STOP_OPTION

## others
#./xmlchange DOUT_S_SAVE_INTERIM_RESTART_FILES=TRUE

# change computation resource requirement if needed
# NTASKS: the total number of MPI tasks, a negative value indicates nodes rather than tasks
./xmlchange NTASKS=1
./xmlchange NTASKS_ATM=1
./xmlchange NTASKS_ESP=1
./xmlchange NTASKS_PER_INST_ATM=36
./xmlchange NTASKS_PER_INST_ESP=1

################################
# (4) compile the model
./case.setup --reset
./case.build --clean-all
qcmd -l select=1:ncpus=1 -l walltime=1:00:00 -A P08010000 -q casper -- ./case.build

################################
# (5) submit jobs (optional)
# Can be used to check whether the case can be successfully run before calibration
#./case.submit











