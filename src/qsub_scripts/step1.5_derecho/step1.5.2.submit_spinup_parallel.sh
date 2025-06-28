#PBS -N buildbasin1x
#PBS -q main
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00
#PBS -A P08010000
#PBS -l select=5:ncpus=128
#PBS -e /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/logs/create_cases/
#PBS -o /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/logs/create_cases/
#PBS -J 2-5

module load conda cdo
conda activate npl-2023b

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

cmdfile=/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/submission/spinup_part${PBS_ARRAY_INDEX}.txt

echo "Processing ${cmdfile}"

parallel < ${cmdfile}
