#!/bin/bash

CASEDIR="/glade/work/guoqiang/CTSM_CAMELS/SA_HH_allbasins/level1_newHH"

/glade/u/home/guoqiang/CTSM_repos/CTSM/cime/scripts/create_newcase --case "${CASEDIR}" --machine derecho --compset I2000Clm51Sp --driver nuopc --compiler intel --res f09_g16 --handle-preexisting-dirs r --run-unsupported --project P08010000

cd "${CASEDIR}"

./xmlchange ATM_DOMAIN_MESH=/glade/work/guoqiang/CTSM_CAMELS/data_mesh_surf/HillslopeHydrology/esmf_mesh_files/corrected_HCDN_nhru_final_671_buff_fix_holes.CAMELSandTDX_areabias_fix.simp0.001.level1_polygons_neighbor_group_esmf_mesh.nc

./xmlchange LND_DOMAIN_MESH=/glade/work/guoqiang/CTSM_CAMELS/data_mesh_surf/HillslopeHydrology/esmf_mesh_files/corrected_HCDN_nhru_final_671_buff_fix_holes.CAMELSandTDX_areabias_fix.simp0.001.level1_polygons_neighbor_group_esmf_mesh.nc

./xmlchange MASK_MESH=/glade/work/guoqiang/CTSM_CAMELS/data_mesh_surf/HillslopeHydrology/esmf_mesh_files/corrected_HCDN_nhru_final_671_buff_fix_holes.CAMELSandTDX_areabias_fix.simp0.001.level1_polygons_neighbor_group_esmf_mesh.nc

./xmlchange CIME_OUTPUT_ROOT=/glade/derecho/scratch/guoqiang/CTSM_outputs/CAMELS_SA/level1_newHH

./xmlchange MOSART_MODE=NULL

./xmlchange DATM_MODE=CLMGSWP3v1

# ./xmlchange STOP_N=12

# ./xmlchange RUN_STARTDATE=1998-10-01

./xmlchange RUN_STARTDATE=2000-01-01
./xmlchange STOP_N=60
./xmlchange STOP_OPTION=nmonths


./xmlchange NTASKS=-2
./xmlchange NTASKS_ATM=-1
./xmlchange NTASKS_PER_INST_ATM=-1
# ./xmlchange ROOTPE=0
./xmlchange NTASKS_ESP=1
./xmlchange NTASKS_PER_INST_ESP=1


./case.setup
./case.build
