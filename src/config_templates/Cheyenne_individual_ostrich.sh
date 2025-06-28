
pathparent="/glade/u/home/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest_LMWG"

for i in {380..670..10}
do
 cd ${pathparent}/CAMELS_${i}_OstCalib/run
 pwd
 qsub submit.Ostrich.sh
done

