#PBS -N geninitparam
#PBS -q main
#PBS -l job_priority=economy
#PBS -l walltime=1:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=128

module load conda cdo
conda activate npl-2022b


export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_MOASMO_Derecho_part1_initparams.py"
config_path="/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/configuration"

iter_start=0
iter_end=1

# Generate a list of configuration files
config_files=$(find $config_path -name '_level*_config_MOASMO.toml')

# Use parallel with :::
# parallel -j 120 python $script ::: $config_files ::: $iter_start ::: $iter_end
find $config_path -name '_level*_config_MOASMO.toml' | \
parallel -j 100 python $script {} $iter_start $iter_end