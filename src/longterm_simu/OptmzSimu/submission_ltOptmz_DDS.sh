#PBS -N LTOptmz
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=128
#PBS -o /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_Ostrich_kge/submission/LongTermSimu/
#PBS -e /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_Ostrich_kge/submission/LongTermSimu/
#####PBS -l job_priority=economy
#####PBS -J 1-5

module load conda cdo
conda activate npl-2024a-tgq

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

# Define command file and calculate which lines this job will execute
cmdfile="/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_Ostrich_kge/submission/LongTermSimu/Optmz1.txt"

parallel -j 128 < $cmdfile
