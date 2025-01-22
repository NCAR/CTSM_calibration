#PBS -N calibost
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=128
#PBS -e /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_Ostrich/logs/runmodel/
#PBS -o /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_Ostrich/logs/runmodel/
####PBS -J 1-5:1


module load conda
conda activate npl-2023b
module load cdo
module load parallel

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

PBS_ARRAY_INDEX=1
cmdfile=/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_Ostrich_kge/submission/Ostcalib_part${PBS_ARRAY_INDEX}.txt

echo "Processing ${cmdfile}"

parallel < ${cmdfile}
