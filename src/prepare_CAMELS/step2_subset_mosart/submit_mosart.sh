#PBS -N gennewparam
#PBS -q main
#PBS -l walltime=3:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=128


parallel -j 128 python extract_MOSART_subset.py ::: {0..626}