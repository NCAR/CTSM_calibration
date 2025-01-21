#!/bin/bash
# basin 1-671

# Script and command path settings
script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/main_CC.py"
cmdpath="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_Ostrich/submission"
mkdir -p $cmdpath

cmdfile="${cmdpath}/create_cases.txt"
rm $cmdfile


for i in {1..626}
do
configfile=/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_Ostrich/configuration/level1-${i}_config.toml
cmd="python /glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/main_CC.py $configfile"
echo $cmd >> $cmdfile
done

for i in {0..39}
do
configfile=/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_Ostrich/configuration/level2-${i}_config.toml
cmd="python /glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/main_CC.py $configfile"
echo $cmd >> $cmdfile
done

for i in {0..3}
do
configfile=/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_Ostrich/configuration/level3-${i}_config.toml
cmd="python /glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/main_CC.py $configfile"
echo $cmd >> $cmdfile
done
