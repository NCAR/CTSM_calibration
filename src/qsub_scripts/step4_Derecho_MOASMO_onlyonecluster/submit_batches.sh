
iter=iter8

path=/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/run_model_mpiserial/$iter

for b in {0..14}
do

pathb=${path}/batch${b}

# ls $pathb
cd $pathb
qsub submission.sh

done
