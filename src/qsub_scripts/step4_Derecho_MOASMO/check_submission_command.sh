
iter=$1

path="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO/run_model_mpiserial"

for f in ${path}/iter${iter}/batch*/*.txt
do

#echo checking $f
cat $f | wc -l

done

