#PBS -N buildbasin1x
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -A NCGD0013
#PBS -l select=1:ncpus=128
#PBS -l job_priority=economy
#PBS -e /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/logs/create_cases/
#PBS -o /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/logs/create_cases/
# ###PBS -J 1-6:1

PBS_ARRAY_INDEX=1

module load conda nco cdo
conda activate npl-2023b

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

cmdfile=/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/submission/spinup_part${PBS_ARRAY_INDEX}.txt

echo "Processing ${cmdfile}"

cat $cmdfile | xargs -I {} -P 128 sh -c '{}'
