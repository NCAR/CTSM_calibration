#PBS -N gennewparam
#PBS -q develop
#PBS -l walltime=6:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=1

sleep 21400
qsub iter7_step3_gen_newparams.sh

# sleep 5000
# ./iter8_step2_gen_submission.sh

# sleep 120
# cd /glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_emulator/run_allbasin_LSEnormKGECV0test/iter8/batch0/
# qsub submission.sh

# for i in {0..33}
# do
# cd /glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_emulator/run_allbasin_LSEnormKGECV0/iter8/batch${i}
# qsub submission.sh
# done


