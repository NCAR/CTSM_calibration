#PBS -A P08010000
#PBS -q develop
#PBS -l walltime=1:00:00
#PBS  -l select=1:ncpus=1:mem=4GB

module load conda
module load cdo
conda activate npl-2023b

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

configfile=/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_emulator/configuration/level1-0_config.toml
echo "Configuration file is $configfile"
python /glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/main_CC.py $configfile Build,MOASMO,NameList
