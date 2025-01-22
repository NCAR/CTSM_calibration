cd /glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_emulator/run_allbasin_iter0/
mkdir iter0_trial0
cd iter0_trial0

# Define the file path pattern
file_pattern="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_emulator/run_allbasin_iter0/iter0/batch*/batch_*.txt"

# Output file path
output_file="iter0_trial0_extracted.txt"

# Create or clear the output file
> "$output_file"

# Loop through all files matching the pattern
for file in $file_pattern; do
    # Extract lines containing 'iter0_trial0' and append them to the output file
    grep 'iter0_trial0' "$file" >> "$output_file"
done



cp ../iter0/batch0/submission.sh .