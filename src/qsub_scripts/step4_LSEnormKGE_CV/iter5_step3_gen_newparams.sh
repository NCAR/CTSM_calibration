#PBS -N gennewparam
#PBS -q develop
#PBS -l walltime=3:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=40:mem=200GB

# based on results from iteration-0, generate new parameter sets for iteration-1

module load conda cdo
conda activate npl-2024a-tgq


export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_Derecho_part3_newparam_LSE_CV.py"

iter_end=6

python $script ${iter_end} 40
