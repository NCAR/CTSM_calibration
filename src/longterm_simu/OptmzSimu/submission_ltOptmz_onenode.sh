#PBS -N LTOptmz
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=128
#PBS -l job_priority=economy
#PBS -o /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/submission/LongTermSimu/
#PBS -e /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/submission/LongTermSimu/

module load conda cdo
conda activate npl-2024a-tgq

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

# Define command file and calculate which lines this job will execute
cmdfile="/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/submission/LongTermSimu/Optmz3.txt"

parallel -j 128 < $cmdfile
