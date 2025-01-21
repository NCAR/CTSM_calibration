#PBS -N buildbasin1x
#PBS -q regular
#PBS -l walltime=12:00:00
#PBS -A NCGD0013
#PBS -l select=1:ncpus=36
#PBS -e /glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/logs/create_cases/
#PBS -o /glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/logs/create_cases/

# Not working

module load conda
conda activate npl-2023b
module load cdo
module load parallel


export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

# necessary for the regular queue to run CTSM model in parallel. The share queue already has this setting.
# unfortunately, this only works for node=1. for multiple nodes, CTSM create_case or clone_case do not run at all
# and even for one node, when I try to run CTSM at the same time, many cores jut failed and only some of tasks are running
export MPI_DSM_DISTRIBUTE=0

parallel -j 20 < /glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/submission/create_cases_part1.txt
#parallel --jobs 360 "python main.py {}" ::: /glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/configuration/level1-*_config.toml

# use MPMD 
#export MPI_SHEPHERD=true
#cmdfile="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/submission/create_cases_part2.txt"
#mpiexec_mpt launch_cf.sh $cmdfile