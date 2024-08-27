#!/bin/bash

~/CTSM_repos/CTSM/cime/scripts/create_clone --case level1_1_newsu2 --clone level1_1 --keepexe

casedir="/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/level1_1_newsu2"
cd $casedir
./case.setup

# model settings
./xmlchange RUN_STARTDATE=1951-10-01
./xmlchange STOP_OPTION=nmonths
./xmlchange STOP_N=12

# Define directories

rundir=$(./xmlquery RUNDIR | grep "RUNDIR:" | awk '{print $2}')
restarch=${casedir}_RecurSpinup
mkdir -p $restarch

echo "Rundir $rundir"
echo "restarch $restarch"

# Loop through iterations
for i in {1..5}; do
  # Submit the job
  ./case.submit --no-batch
  
  # Locate the CLM restart file
  restart_file=$(ls ${rundir}/*.clm2.r.* 2>/dev/null)
  
  if [[ -z "$restart_file" ]]; then
    echo "No restart file found in ${rundir} for iteration ${i}. Exiting."
    # exit 1
  fi

  # Construct output file name based on the original file name
  base_filename=$(basename "$restart_file")
  outfile="${restarch}/run${i}_${base_filename}"

  # Move the restart file to the new location
  mv "$restart_file" "$outfile"

  # Clean up the rundir
  rm ${rundir}/*.nc
  rm ${rundir}/*log*

  # Update user_nl_clm with the new finidat
  echo "finidat = '${outfile}'" >> user_nl_clm

  echo "Iteration ${i} completed. Restart file moved to ${outfile}."
done
