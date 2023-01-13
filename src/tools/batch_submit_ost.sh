# submit basin Ostrich calib

for n in {0..2}
do
  cd /glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib/CAMELS_${n}_OstCalib/run
  qsub submit.Ostrich.sh
done