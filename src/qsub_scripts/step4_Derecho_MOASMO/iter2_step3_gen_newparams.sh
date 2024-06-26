#PBS -N gennewparam
#PBS -q main
#PBS -l walltime=3:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=128

# based on results from iteration-0, generate new parameter sets for iteration-1

module load conda cdo
conda activate npl-2022b


export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_MOASMO_Derecho_part3_newparam.py"
config_path="/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/configuration"

iter_end=3

# Generate a list of configuration files
config_files=$(find $config_path -name '_level*_config_MOASMO.toml')

# example of one basin
# python $script /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/configuration/_level1-0_config_MOASMO.toml 1

find $config_path -name '_level*_config_MOASMO.toml' | \
parallel -j 128 python $script {}  $iter_end
