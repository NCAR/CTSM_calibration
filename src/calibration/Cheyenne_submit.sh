#PBS -N buildcalib
#PBS -q share
#PBS -l walltime=6:00:00
#PBS -A P08010000
#PBS  -l select=1:ncpus=1
#PBS -e ./log/
#PBS -o ./log/
#PBS -J 80-180:10

module load conda
module load cdo
conda activate npl-2022b-tgq

buildcase_option='only'

id0=${PBS_ARRAY_INDEX}
id1=$((PBS_ARRAY_INDEX+9))

for ((id=$id0;id<=$id1;id++))
do
configfile=/glade/u/home/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest/configuration/CAMELS-${id}_config.toml
echo "Configuration file is $configfile"
python main_onlycase.py $configfile $buildcase_option
done

