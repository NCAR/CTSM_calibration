# create multiple CTSM cases to test parallel run

# preparation
module load conda
conda activate npl-2022b

projectCode="P08010000"

path0="/glade/scratch/$USER/CTSM_testpara"
mkdir -p $path0
cd $path0

###########  step-0:
# git clone
git clone https://github.com/ESCOMP/CTSM.git
cd CTSM
./manage_externals/checkout_externals

########### step-1:
# create a CTSM casee
# note that case.build will cost ~10 min
newcase_settings="--compset I2000Clm51Sp --driver nuopc --compiler intel --res f09_g16 --run-unsupported --project ${projectCode}"
${path0}/CTSM/cime/scripts/create_newcase --case $path0/case_0 ${newcase_settings}

cd ${path0}/case_0 || exit
./case.setup

nl_clm_settings="fsurdat = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/surfdata_CAMELS_split_nested_hist_78pfts_CMIP6_simyr2000_c230105.nc'"
echo "${nl_clm_settings}"  >> user_nl_clm
echo "hist_nhtfrq = 0,-24"  >> user_nl_clm
echo "hist_mfilt = 1,365"  >> user_nl_clm
echo "hist_fincl2 = 'QRUNOFF','H2OSNO','ZWT','SOILWATER_10CM','EFLX_LH_TOT','QDRAI','QOVER','RAIN'"  >> user_nl_clm

xmlchange_settings1="ATM_DOMAIN_MESH=/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_basin_mask_split_nest/ESMFmesh_ctsm_HCDN_nhru_final_671.buff_fix_holes_polygons_simplified_5e-4_split_nested_basin0.nc"
xmlchange_settings2="LND_DOMAIN_MESH=/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_basin_mask_split_nest/ESMFmesh_ctsm_HCDN_nhru_final_671.buff_fix_holes_polygons_simplified_5e-4_split_nested_basin0.nc"
xmlchange_settings3="MASK_MESH=/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_basin_mask_split_nest/ESMFmesh_ctsm_HCDN_nhru_final_671.buff_fix_holes_polygons_simplified_5e-4_split_nested_basin0.nc"
xmlchange_settings4="CIME_OUTPUT_ROOT=${path0}/output/case_0"
xmlchange_settings5="MOSART_MODE=NULL"
xmlchange_settings="${xmlchange_settings1} ${xmlchange_settings2} ${xmlchange_settings3} ${xmlchange_settings4} ${xmlchange_settings5}"
for s in ${xmlchange_settings}
do
  ./xmlchange ${s}
done
./xmlchange DATM_MODE=CLMNLDAS2
./xmlchange STOP_N=12
./xmlchange RUN_STARTDATE=2000-01-01
./xmlchange STOP_OPTION=nmonths
./xmlchange NTASKS=1
./xmlchange COST_PES=1
./xmlchange TOTALPES=1
./xmlchange MAX_TASKS_PER_NODE=1
./xmlchange MAX_MPITASKS_PER_NODE=1
./xmlchange COST_PES=1

./case.setup --reset
./case.build --clean-all
qcmd -l select=1:ncpus=1 -l walltime=1:00:00 -A ${projectCode} -q share -- ./case.build

#./case.submit # just to check if the model can really work in your workspace

########### step-2L
# clone multiple cases. they are the same with case_0 but compiling is skipped to save tme
for i in {1..17}
do
  ${path0}/CTSM/cime/scripts/create_clone --case $path0/case_${i} --clone $path0/case_0 --cime-output-root ${path0}/output/case_${i} --keepexe --project ${projectCode}
done
