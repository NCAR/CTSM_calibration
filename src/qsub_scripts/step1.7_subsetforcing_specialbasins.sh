# PBS -N buildbasin0
# PBS -q share
# PBS -l walltime=6:00:00
# PBS -A NCGD0013
# PBS  -l select=1:ncpus=18
# PBS -e /glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/logs/create_cases/
# PBS -o /glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/logs/create_cases/

module load conda
module load cdo
conda activate npl-2023b

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

#for b in 239 249 316 325 537 541
for b in 239
do
configfile=/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/configuration/_level1-${b}_config_SubForc.toml
echo "Configuration file is $configfile"
python /glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/generate_forcing_subset_largebuffer.py $configfile &
done