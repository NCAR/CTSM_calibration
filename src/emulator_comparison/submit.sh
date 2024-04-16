# PBS -A P08010000
# PBS -q main
# PBS -l job_priority=economy
# PBS -l walltime=2:00:00
# PBS  -l select=1:ncpus=128

module load conda
module load cdo
conda activate npl-2023b

python compare_emulator_methods.py
