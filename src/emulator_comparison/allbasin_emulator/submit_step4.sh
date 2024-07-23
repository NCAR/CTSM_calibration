#PBS -N gennewparam
#PBS -q casper
#PBS -l walltime=6:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=1
#PBS -r y
#PBS -J 101-627:20

# based on results from iteration-0, generate new parameter sets for iteration-1

module load conda cdo
conda activate npl-2024a-tgq


export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

s=${PBS_ARRAY_INDEX}
e=$((PBS_ARRAY_INDEX+19))

python step4_emulator_generate_optmz_params_RF_GA.py $s $e
