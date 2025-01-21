#PBS -N simudefa
#PBS -q regular
#PBS -l walltime=6:00:00
#PBS -A NCGD0013
#PBS -l select=1:ncpus=36
#PBS -e /glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_Ostrich/logs/create_cases/
#PBS -o /glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_Ostrich/logs/create_cases/
#PBS -J 2-19:1

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

# Calculate the range of $num for this array job
START=$(( (PBS_ARRAY_INDEX - 1) * 36 ))
END=$(( PBS_ARRAY_INDEX * 36 - 1 ))
echo "Processing basins $START to $END"

# Path to your script
SCRIPT="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/simulation_aftercalib/Ostrich_simuDefault.py"

# Run the Python script using GNU parallel
parallel -j 36 python $SCRIPT ::: $(seq $START $END)
