#PBS -N gennewparam
#PBS -q develop
#PBS -l walltime=3:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=1

sleep 7200
/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/qsub_scripts/step4_SSEnormKGE/iter5_step2_gen_submission.sh

sleep 120

for i in {0..41}
do
cd /glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_emulator/run_allbasin_SSEnormKGE/iter5/batch${i}/
qsub submission.sh
done

