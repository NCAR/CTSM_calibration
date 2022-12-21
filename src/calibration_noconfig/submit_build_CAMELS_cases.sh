#PBS -N Buildcase
#PBS -q share
#PBS -l walltime=1:00:00
#PBS -A P08010000
#PBS  -l select=1:ncpus=1

# command: qsub -v num=0 submit_build_CAMELS_cases.sh

module load conda
conda activate npl-2022b-tgq

#bnum=0
#bnum=$1
echo processing basin $bnum
python build_CAMELS_cases.py $bnum



