#!/bin/bash

cmdfile="/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/submission/RecurSpinup/cmdlist.txt"

for i in {0..626}
do
  command="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/RecursiveSpinup/RecurSpinup_onebasin.sh $i"
  echo $command >> $cmdfile
done