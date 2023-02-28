# Source: https://gist.github.com/uturuncoglu/9c638f003e0bf9dc089c8298d5e24c0a

#export TMPDIR=/glade/scratch/$USER/temp
#mkdir -p $TMPDIR
export MPI_USE_ARRAY=false

# load modules
module purge
module load ncarenv/1.3
module load intel/19.0.2
module load mpt/2.19
module load netcdf-mpi/4.7.1
module load pnetcdf/1.11.0
module load ncarcompilers/0.5.0
module use /glade/work/turuncu/PROGS/modulefiles/esmfpkgs/intel/19.0.2
module load esmf-8.0.0-ncdfio-mpt-O 
module load nco

#file_i='test_basins_SCRIPunstructured.nc'
#file_o='test_basins_ESMFmesh.nc'

file_i='CAMELS_80.nc'
file_o='CAMELS_80_ESMFmesh.nc'

echo "$file_i --> $file_o"
mpiexec_mpt `hostname` -np 2 ESMF_Scrip2Unstruct $file_i $file_o 0 ESMF
nccopy -k 5 $file_o ${file_o/_ESMFmesh_/_ESMFmesh_cdf5_} 
