#PBS -N Merge
#PBS -q main
#PBS -l walltime=6:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=128


module load conda cdo
conda activate npl-2024a

# Define the function
merge_files() {
    local i=$1      # The first argument is the index
    local normKGE=$2  # The second argument is the normKGE value (e.g., normKGEr2, normKGEr3, etc.)
    local path0="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/LongTermSimu/LSEallbasin"
    local pathi="${path0}/level1_${i}/${normKGE}"

    echo "processing level1_${i} with ${normKGE}"
    # Navigate to the directory and run the ncrcat command
    cd $pathi
    ncrcat -O level1_${i}_*.clm2.h0.*.nc merged_h0_1951-2019.nc
}

export -f merge_files  # Export the function so that GNU parallel can access it

# Run the function in parallel using GNU parallel, looping through normKGEr2 to normKGEr4 (or more)
for normKGE in normKGEr20; do
    seq 0 626 | parallel -j 128 merge_files {} $normKGE
done
