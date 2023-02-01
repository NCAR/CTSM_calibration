#PBS -N buildcalib
#PBS -q regular
#PBS -l walltime=12:00:00
#PBS -A P08010000
#PBS  -l select=1:ncpus=36
#PBS -e ./log/
#PBS -o ./log/

# case.build does not work well in this way. compiling is very very slow or even not possible...


module load conda
module load cdo
module load parallel
conda activate npl-2022b-tgq

buildcase_option='except'
parallel --jobs 36 "python main.py {} ${buildcase_option}" ::: /glade/u/home/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest/configuration/CAMELS-10*_config.toml


