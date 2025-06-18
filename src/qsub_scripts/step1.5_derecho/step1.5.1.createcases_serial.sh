#!/bin/bash

# Path to your file
file="/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/submission/create_cases_1-671.txt"

# Check if the file exists
if [ -f "$file" ]; then
    # Read each line from the file
    while IFS= read -r line; do
        # Execute the line as a command
        eval "$line"
    done < "$file"
else
    echo "The file does not exist."
fi

