#PBS -N buildcalib
#PBS -q share
#PBS -l walltime=6:00:00
#PBS -A P08010000
#PBS  -l select=1:ncpus=1
#PBS -J 20-40:10

module load conda
module load cdo
conda activate npl-2022b-tgq

python main_MOASMO_forCESM.py ${PBS_ARRAY_INDEX}