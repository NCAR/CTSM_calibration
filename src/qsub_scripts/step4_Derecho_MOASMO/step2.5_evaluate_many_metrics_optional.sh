#PBS -N MOAcalib
#PBS -q main
#PBS -l select=1:ncpus=128
#PBS -l walltime=6:00:00
#PBS -l job_priority=economy
#PBS -A P08010000


module load conda nco cdo
conda activate npl-2024a

script='/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_MOASMO_Derecho_part2.5_optional_evaluate.py'

python $script 128 627 400
