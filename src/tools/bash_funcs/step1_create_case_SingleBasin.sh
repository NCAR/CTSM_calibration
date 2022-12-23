#!/bin/bash
# An independent basin

# Create a CTSM model case
# Settings based on single basin files
# Note: manual check of settings are needed before running this script

# environment
module load conda
conda activate npl-2022b

########################################################################################################################
# 1. inpaths and files

##############
# input arguments
#bnum=99 # basin number
#STOP_N=3
#RUN_STARTDATE=2000-01-01
bnum=$1
STOP_N=$2
RUN_STARTDATE=$3

##############
# less active settings
# pathCTSMrepo: CTSM model source used to create model cases. git clone from Github repos
pathCTSMrepo="/glade/u/home/guoqiang/CTSM_repos/CTSM"

# pathParent: everything of the model will be saved here
pathParent="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib"

# CIME_OUTPUT_ROOT, which also defines run folder and archive folder
pathCIMEout="/glade/scratch/guoqiang/CTSM_outputs/CAMELS_Calib/Lump_calib"

# CTSM model case folder
casename="CAMELS_${bnum}"
pathCTSMcase=${pathParent}/${casename}

# surface file and domain file
fileDomain="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_basin_mask/ESMFmesh_ctsm_HCDN_nhru_final_671_v0_8e-3_basin${bnum}.nc"

## for mesh mask type:
#fileSurf="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_basin_mask/surfdata_CAMELS_hist_78pfts_CMIP6_simyr2000_c221004_basin${bnum}.nc"
# for independent basins:
fileSurf="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/surfdata_CAMELS_hist_78pfts_CMIP6_simyr2000_c221004.nc"


# projectCode: a project is needed to run on Cheyenne
projectCode="P08010000"

#####################
# Model settings to be changed
nl_clm_settings="fsurdat = '${fileSurf}'"

xmlchange_settings1="ATM_DOMAIN_MESH=${fileDomain}"
xmlchange_settings2="LND_DOMAIN_MESH=${fileDomain}"
#xmlchange_settings3=""
xmlchange_settings3="MASK_MESH=${fileDomain}" # this is needed if try to masking out some basins
xmlchange_settings4="CIME_OUTPUT_ROOT=${pathCIMEout}"
xmlchange_settings5="MOSART_MODE=NULL"
xmlchange_settings="${xmlchange_settings1} ${xmlchange_settings2} ${xmlchange_settings3} ${xmlchange_settings4} ${xmlchange_settings5}"

########################################################################################################################
# 2. create model cases

echo "processing CAMELS basin ${bnum}"

################################

# (1) create new case
newcase_settings="--compset I2000Clm51Sp --driver nuopc --compiler intel --res f09_g16 --run-unsupported --project ${projectCode}"
  ${pathCTSMrepo}/cime/scripts/create_newcase --case ${pathCTSMcase} ${newcase_settings}

cd ${pathCTSMcase} || exit

# (2) set up
./case.setup

################################
# (3) change settings (optional)

# change user_nl_clm
echo "${nl_clm_settings}"  >> user_nl_clm
echo "hist_nhtfrq = 0,-24"  >> user_nl_clm
echo "hist_mfilt = 1,365"  >> user_nl_clm
echo "hist_fincl2 = 'QRUNOFF','H2OSNO','ZWT','SOILWATER_10CM','EFLX_LH_TOT','QDRAI','QOVER','RAIN'"  >> user_nl_clm

# change land domain and MESH files
for s in ${xmlchange_settings}
do
  ./xmlchange ${s}
done
./xmlquery ATM_DOMAIN_MESH,LND_DOMAIN_MESH,MASK_MESH,RUNDIR,DOUT_S_ROOT

# change forcing data
./xmlchange DATM_MODE=CLMNLDAS2
./xmlquery DATM_MODE

# change the run time of mode case
./xmlchange STOP_N=${STOP_N}
./xmlchange RUN_STARTDATE=${RUN_STARTDATE}
./xmlchange STOP_OPTION=nmonths
./xmlquery RUN_STARTDATE,STOP_N,STOP_OPTION

## others
#./xmlchange DOUT_S_SAVE_INTERIM_RESTART_FILES=TRUE

# change computation resource requirement if needed
# NTASKS: the total number of MPI tasks, a negative value indicates nodes rather than tasks
./xmlchange NTASKS=1
./xmlchange NTASKS_PER_INST=1
./xmlquery NTASKS,NTASKS_PER_INST

################################
# (4) compile the model
./case.setup --reset
./case.build --clean-all
#qcmd -- ./case.build
qcmd -l select=1:ncpus=1:mpiprocs=1 -l walltime=1:00:00 -A ${projectCode} -q share -- ./case.build

################################
# (5) submit jobs (optional)
# Can be used to check whether the case can be successfully run before calibration
#./case.submit

