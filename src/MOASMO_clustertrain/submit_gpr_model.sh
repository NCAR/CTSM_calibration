#PBS -N traingpr
#PBS -q casper
#PBS -l walltime=12:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=1:mem=40G


python train_gpr_model.py
