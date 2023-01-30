#PBS -N buildcalib
#PBS -q regular
#PBS -l walltime=12:00:00
#PBS -A P08010000
#PBS  -l select=4:ncpus=36
#PBS -e ./log/
#PBS -o ./log/



module load conda
module load cdo
module load parallel
conda activate npl-2022b-tgq

#individual jobs
#configfile=/glade/u/home/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest/configuration/CAMELS-${PBS_ARRAY_INDEX}_config.toml
#echo "Configuration file is $configfile"
#python main.py $configfile


parallel "python main.py {}" ::: /glade/u/home/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest/configuration/CAMELS-10*_config.toml


