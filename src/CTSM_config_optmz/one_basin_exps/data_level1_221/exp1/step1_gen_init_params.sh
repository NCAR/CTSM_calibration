#PBS -N geninitparam
#PBS -q develop
#PBS -l walltime=1:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=1

# generate hundreds of parameter sets (iteration 0)

module load conda cdo
conda activate npl-2024a-tgq


export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_MOASMO_Derecho_part1_initparams.py"

iter_start=0
iter_end=1 # due to the method range(iter_start, iter_end), it will only run iteration 0

config_file="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/CTSM_config_optmz/one_basin_exps/data_level1_221/exp1/_level1-221_config_MOASMO.toml"

python $script ${config_file} $iter_start $iter_end