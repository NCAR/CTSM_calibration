#PBS -A P08010000
#PBS -q casper
#PBS -l walltime=12:00:00
#PBS  -l select=1:ncpus=1

module load conda
module load cdo
conda activate npl-2023b

python compare_emulator_methods_useIter0.py