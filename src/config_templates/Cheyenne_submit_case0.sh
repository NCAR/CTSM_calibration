#PBS -N buildcalib
#PBS -q share
#PBS -l walltime=3:00:00
#PBS -A P08010000
#PBS  -l select=1:ncpus=1
#PBS -e ./log/
#PBS -o ./log/

# This will submit basin 0, which will be used for other cases to clone (exe)

module load conda
module load cdo
conda activate npl-2022b-tgq

id0=0
id1=0

for ((id=$id0;id<=$id1;id++))
do

configfile=/glade/u/home/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest/configuration/CAMELS-${id}_config.toml
echo "Configuration file is $configfile"
python main.py $configfile

done