l#!/bin/bash

# receive input argument
tarbasin=$1
echo "Processing basin $tarbasin"

# clone a case and change settings
newcase=level1_${tarbasin}_SUrun
casedir="/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/$newcase"
~/CTSM_repos/CTSM/cime/scripts/create_clone --case $casedir --clone /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/level1_${tarbasin} --keepexe


cd $casedir
./case.setup


./xmlchange RUN_STARTDATE=1951-10-01
./xmlchange STOP_OPTION=nmonths
./xmlchange STOP_N=12

# get directories
rundir=$(./xmlquery RUNDIR | grep "RUNDIR:" | awk '{print $2}')
restarch=${casedir}_RecurSpinup
mkdir -p $restarch

echo "Rundir $rundir"
echo "restarch $restarch"

# Loop through iterations
for i in {1..50}; do


  # Check if the final restart file already exists
  restart_file="${restarch}/run${i}_level1_${tarbasin}_newsu2.clm2.r.1952-10-01-00000.nc"
  if [ -f "$restart_file" ]; then
    echo "Restart file ${restart_file} already exists. Skipping iteration ${i}."
    echo "finidat = '${restart_file}'" >> user_nl_clm
    continue
  fi


  # Submit the job
  ./case.submit --no-batch
  
  # copy files
  # Define the restart files to copy
  level1_1_SUrun.clm2.r.1952-10-01-00000.nc
  allfiles=("level1_${tarbasin}_SUrun.clm2.r.1952-10-01-00000.nc" 
            "level1_${tarbasin}_SUrun.clm2.rh0.1952-10-01-00000.nc" 
            "level1_${tarbasin}_SUrun.clm2.rh1.1952-10-01-00000.nc")
  for restart_filename in "${allfiles[@]}"; do
    infile="${rundir}/${restart_filename}"
    outfile="${restarch}/run${i}_${restart_filename}"
    
    # Check if the input file exists before moving
    if [ -f "$infile" ]; then
      mv "$infile" "$outfile"
      echo "Moved $infile to ${outfile}"
    else
      echo "$infile does not exist in $rundir. Skipping."
    fi
  done


  # Clean up the rundir
  rm ${rundir}/*.nc
  rm ${rundir}/*log*

  # Update user_nl_clm with the new finidat
  restart_file="${restarch}/run${i}_level1_${tarbasin}_newsu2.clm2.r.1952-10-01-00000.nc"
  
  echo "finidat = '${restart_file}'" >> user_nl_clm
  echo "Iteration ${i} completed."
done

# rm simulation results
doutdir=$(./xmlquery DOUT_S_ROOT | grep "DOUT_S_ROOT:" | awk '{print $2}')
echo $doutdir

echo "remove directories"
cd ..
rm -r $doutdir
rm -r $rundir
rm -r $casedir

