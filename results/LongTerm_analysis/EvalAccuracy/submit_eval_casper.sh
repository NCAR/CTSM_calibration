#PBS -N Eval
#PBS -q casper
#PBS -l walltime=6:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=36:mem=200G

module load conda cdo
conda activate npl-2024a-tgq

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

python evaluate_runoff.py
