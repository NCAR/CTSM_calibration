#!/bin/bash
# basin 1-671
# just create cases which is fast

script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/main_CC.py"
cmdpath="/glade/work/guoqiang/CTSM_CAMELS/SA_HH_MOASMO/submission"
mkdir -p $cmdpath

cmdfile="${cmdpath}/create_cases_1-671.txt"
rm $cmdfile


for i in {1..626}
do
configfile=/glade/work/guoqiang/CTSM_CAMELS/SA_HH_MOASMO/configuration/level1-${i}_config.toml
#python /glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/main_CC.py $configfile Build,MOASMO,NameList
cmd="python /glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/main_CC.py $configfile Build,MOASMO,NameList"
echo $cmd >> $cmdfile
done

for i in {0..39}
do
configfile=/glade/work/guoqiang/CTSM_CAMELS/SA_HH_MOASMO/configuration/level2-${i}_config.toml
#python /glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/main_CC.py $configfile Build,MOASMO,NameList
cmd="python /glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/main_CC.py $configfile Build,MOASMO,NameList"
echo $cmd >> $cmdfile
done

for i in {0..3}
do
configfile=/glade/work/guoqiang/CTSM_CAMELS/SA_HH_MOASMO/configuration/level3-${i}_config.toml
#python /glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/main_CC.py $configfile Build,MOASMO,NameList
cmd="python /glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/main_CC.py $configfile Build,MOASMO,NameList"
echo $cmd >> $cmdfile
done
