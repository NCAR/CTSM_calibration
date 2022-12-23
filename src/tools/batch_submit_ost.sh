# submit basin Ostrich calib

for n in {1..3}
do
  cd /glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib/CAMELS_${n}_OstCalib/run
  qsub submit.Ostrich.sh
done