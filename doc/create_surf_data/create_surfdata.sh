#!/bin/bash
#PBS -A P08010000
#PBS -N gensurf
#PBS -q regular
#PBS -l walltime=06:00:00
##PBS -j oe
#PBS -e ./log/
#PBS -o ./log/
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=109G

path_ctsm="/glade/u/home/$USER/CTSM_repos/CTSM" # CTSM codes
path_tar="/glade/scratch/$USER/test_create_surfdata" # where files will be saved
gridfile="${path_tar}/SCRIPgrid_test_nomask_c230227.nc" # target region

cd ${path_tar}

# create mapping files
script=${path_ctsm}/tools/mkmapdata/mkmapdata.sh
usr_gname="testgrid"
$script -f ${gridfile} -r $usr_gname -t regional


# create surface dataset
script="${path_ctsm}/tools/mksurfdata_map/mksurfdata_regional.pl"
usr_gname="testgrid" # same with that in Step-2
usr_gdate="230227"
year="2000"
scenario="hist"
${script} -r usrspec -usr_gname ${usr_gname} -usr_gdate ${usr_gdate} -usr_mapdir ${path_tar} -y ${year} -ssp_rcp ${scenario}