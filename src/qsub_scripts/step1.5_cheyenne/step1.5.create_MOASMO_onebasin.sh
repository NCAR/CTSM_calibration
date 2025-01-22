#PBS -N buildbasin1
#PBS -q share
#PBS -l walltime=6:00:00
#PBS -A NCGD0013
#PBS -l select=1:ncpus=1
#PBS -e /glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/logs/create_cases/
#PBS -o /glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/logs/create_cases/

# Not working

module load conda
conda activate npl-2023b
module load cdo
module load parallel


export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

python /glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/main_CC.py /glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/configuration/level1-0_config.toml