#PBS -N buildbasin1x
#PBS -q regular
#PBS -l walltime=12:00:00
#PBS -A NCGD0013
#PBS -l select=1:ncpus=36

module load conda
conda activate npl-2023b
module load nco cdo parallel

export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

basins=""
# Add 'level1_' elements
for i in {0..626}; do
  basins="${basins} level1_$i"
done

# Add 'level2_' elements
for i in {0..39}; do
  basins="${basins} level2_$i"
done

# Add 'level3_' elements
for i in {0..3}; do
  basins="${basins} level3_$i"
done

parallel -j 36 ./extract_nc_variables.sh ::: $basins
