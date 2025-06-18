#PBS -N MOAcalib
#PBS -q develop
#PBS -l select=1:ncpus=36
#PBS -l walltime=1:00:00
#PBS -A P08010000


module load conda nco cdo
conda activate npl-2024a

script='/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_MOASMO_Derecho_part2.5_optional_evaluate_combinecsv.py'

python $script
