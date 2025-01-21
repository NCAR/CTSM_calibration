# PBS -N gennewparam
# PBS -q develop
# PBS -l walltime=1:00:00
# PBS -A P08010000
# PBS -l select=1:ncpus=1

# based on results from iteration-0, generate new parameter sets for iteration-1

module load conda cdo
conda activate npl-2024a-tgq


export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_MOASMO_Derecho_part3_newparam.py"

iter_end=3

config_file="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/CTSM_config_optmz/one_basin_exps/data_level1_221/exp1/_level1-221_config_MOASMO.toml"

python $script ${config_file}  $iter_end
