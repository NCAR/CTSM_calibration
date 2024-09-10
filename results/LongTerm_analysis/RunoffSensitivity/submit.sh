#PBS -N Qsenst
#PBS -q casper
#PBS -l walltime=6:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=36:mem=220G
###### PBS -J 2-5:1

module load conda cdo
conda activate npl-2024a-tgq

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

python cal_runoff_sensitivity.py 5
