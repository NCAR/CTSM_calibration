#PBS -N buildcalib
#PBS -q regular
#PBS -l walltime=6:00:00
#PBS -A P08010000
#PBS  -l select=1:ncpus=36:mem=100gb
#PBS -e ./log/
#PBS -o ./log/
#PBS -J 1-6:1



module load conda
module load cdo
module load parallel
conda activate npl-2022b-tgq

# necessary for the regular queue to run CTSM model in parallel. The share queue already has this setting.
export MPI_DSM_DISTRIBUTE=0

parallel --jobs 36 "python main.py {}" ::: /glade/u/home/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest/configuration/CAMELS-${PBS_ARRAY_INDEX}*_config.toml


