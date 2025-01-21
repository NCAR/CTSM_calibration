# generate commands to generate CTSM cases
# Parallel commands cannot run within multiple nodes (as Nov 27, 2023) on NCAR HPCs. Therefore, this script generate some cmdfiles with fewer lines so they can be run on different nodes (36 CPUs on Cheyenne and 128 CPUs on Derecho)


script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/main_CC.py"
cmdpath="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/submission"
mkdir -p $cmdpath

# ###### part-1: 1-164 have been finished (subset forcing)
cmdfile="${cmdpath}/create_cases_pernode_part1.txt"
if [ -f $cmdfile ]; then
    rm $cmdfile
fi

level="level1" 
for i in {1..164} # zero has been created
do
  configfile="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/configuration/${level}-${i}_config.toml"
  cmd="python $script $configfile"
  echo $cmd  >> $cmdfile
done


# ###### part-2:
start=(165 237 309 381 453 525)
end=(236 308 380 452 524 596)
flag=2

for i in "${!start[@]}"; do
    echo "Flag: $flag, Start: ${start[$i]}, End: ${end[$i]}"
    
    cmdfile="${cmdpath}/create_cases_pernode_part${flag}.txt"
    if [ -f "$cmdfile" ]; then
        rm "$cmdfile"
    fi

    level="level1" 
    for j in $(seq ${start[$i]} ${end[$i]}); do
      configfile="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/configuration/${level}-${j}_config.toml"
      cmd="python $script $configfile"
      echo "$cmd"  >> "$cmdfile"
    done

    ((flag++))
done


# ###### part-2: all remaining
cmdfile="${cmdpath}/create_cases_pernode_part8.txt"
if [ -f $cmdfile ]; then
    rm $cmdfile
fi

level="level1" 
for i in {597..626} 
do
  configfile="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/configuration/${level}-${i}_config.toml"
  cmd="python $script $configfile"
  echo $cmd  >> $cmdfile
done

level="level2"
for i in {0..39}
do
  configfile="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/configuration/${level}-${i}_config.toml"
  cmd="python $script $configfile"
  echo $cmd  >> $cmdfile
done

level="level3"
for i in {0..3}
do
  configfile="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/configuration/${level}-${i}_config.toml"
  cmd="python $script $configfile"
  echo $cmd  >> $cmdfile
done


