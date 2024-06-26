
iter=$1

threshold=40

path="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange"


echo "print folder information if its $iter iteration parameter.pkl number is smaller than $threshold"

for i in {0..626}
do
    # Count the number of files
    count=$(ls ${path}/level1_${i}_MOASMOcalib/param_sets/paramset_iter${iter}_trial*.pkl 2>/dev/null | wc -l)

    # Check if the count is less than the threshold
    if [ "$count" -lt "$threshold" ]; then
        echo "basin $i: $count files"
    fi
done




