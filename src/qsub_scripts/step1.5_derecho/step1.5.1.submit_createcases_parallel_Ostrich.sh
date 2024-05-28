#PBS -N buildbasin1x
#PBS -q main
#PBS -l walltime=1:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=128
#PBS -e /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_Ostrich_kge/logs/create_cases/
#PBS -o /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_Ostrich_kge/logs/create_cases/


module load conda cdo
conda activate npl-2023b

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

cmdfile=/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_Ostrich_m2err_smallrange/submission/create_cases_1-671.txt

echo "Processing ${cmdfile}"

parallel -j 128 < ${cmdfile}
