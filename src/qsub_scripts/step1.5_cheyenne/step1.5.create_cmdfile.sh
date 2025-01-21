# generate commands to generate CTSM cases

####### part-1: 36 basins for test

script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/main_CC.py"
cmdpath="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/submission"
cmdfile="${cmdpath}/create_cases_part1.txt"
mkdir -p $cmdpath

if [ -f $cmdfile ]; then
    rm $cmdfile
fi

level="level1" 
for i in {1..36} # zero has been created
do
  configfile="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/configuration/${level}-${i}_config.toml"
  cmd="python $script $configfile"
  echo $cmd  >> $cmdfile
done

####### part-2: other basins

script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/main_CC.py"
cmdpath="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/submission"
cmdfile="${cmdpath}/create_cases_part2.txt"
mkdir -p $cmdpath

if [ -f $cmdfile ]; then
    rm $cmdfile
fi

level="level1" 
for i in {37..626} # zero has been created
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


