#PBS -N pyvisc
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=128
#PBS -l job_priority=economy

module load conda
conda activate PyVISCOUS

python step5_SA_pyviscous_calculation.py