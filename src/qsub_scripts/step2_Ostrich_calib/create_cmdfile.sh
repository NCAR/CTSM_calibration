#!/bin/bash
# basin 1-671

calibpath="/glade/campaign/cgd/tss/people/guoqiang/CTSMcases/CAMELS_Calib/Calib_all_HH_Ostrich"
cmdpath="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_Ostrich/submission"
mkdir -p $cmdpath


# Variables for looping through basins and files
total_basins=671  # Updated total basins
basins_per_file=36
file_count=1
basin_count=1

# Loop through all basins
for i in $(seq 0 $(($total_basins - 1)))
do
  # Determine the current level and basin number
  if [ $i -le 626 ]; then  # Including basin 626 in Level 1
    level="level1"
    basin_number=$i
  elif [ $i -le 666 ]; then  # Adjusting range for Level 2
    level="level2"
    basin_number=$(($i - 627))  # Subtract 627 since Level 2 starts after basin 626
  else
    level="level3"
    basin_number=$(($i - 667))  # Adjust for Level 3 start
  fi


  # Command to be added
  cmd="cd ${calibpath}/${level}_${basin_number}_Ostrich/run && ./OstrichGCC"

  # File to write to
  cmdfile="${cmdpath}/Ostcalib_part${file_count}.txt"

  # Create or append to the command file
  if [ $basin_count -eq 1 ]; then
    echo $cmd > $cmdfile
  else
    echo $cmd >> $cmdfile
  fi

  # Increment basin_count, reset if it reaches basins_per_file and increment file_count
  basin_count=$(($basin_count + 1))
  if [ $basin_count -gt $basins_per_file ]; then
    basin_count=1
    file_count=$(($file_count + 1))
  fi
done
