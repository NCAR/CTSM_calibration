
pathcase=/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib/
pathout=/glade/scratch/guoqiang/CTSM_outputs/CAMELS_Calib/Lump_calib/


for bnum in {1..3}
do
  rm -r ${pathcase}/CAMELS_${bnum}
  rm -r ${pathcase}/CAMELS_${bnum}_OstCalib
  #rm -r ${pathcase}/CAMELS_${bnum}_SubsetForcing
  #rm -r ${pathcase}/CAMELS_${bnum}_SpinupFiles
  rm ${pathcase}/configuration/*CAMELS-${bnum}*

  rm -r ${pathout}/CAMELS_${bnum}
  rm -r  ${pathout}/archive/CAMELS_${bnum}
done