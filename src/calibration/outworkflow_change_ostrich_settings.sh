
path="/glade/u/home/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest_LMWG"
for i in {0..670..10}
do
  filei="${path}/CAMELS_${i}_OstCalib/run/ostIn.txt"
  sed -i s/"OstrichWarmStart no"/"OstrichWarmStart yes"/g $filei
done