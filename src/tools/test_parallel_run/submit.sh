#PBS -N paratest
#PBS -q share
#PBS -l walltime=1:00:00
#PBS -A P08010000
#PBS  -l select=1:ncpus=18:mpiprocs=18:ompthreads=1:mem=109GB

module load conda/latest
conda activate npl-2022b

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

export MPI_SHEPHERD=true
mpiexec_mpt launch_cf.sh cmdfile
