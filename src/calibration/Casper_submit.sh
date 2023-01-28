#PBS -N buildcalib
#PBS -q casper
#PBS -l walltime=3:00:00
#PBS -A P08010000
#PBS  -l select=1:ncpus=1
#PBS -e ./log/
#PBS -o ./log/
#PBS -J 0-2:1

module load conda
module load cdo
conda activate npl-2022b-tgq

configfile=/glade/u/home/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest/configuration/CAMELS-${PBS_ARRAY_INDEX}_config.toml

echo "Configuration file is $configfile"

python main.py $configfile
