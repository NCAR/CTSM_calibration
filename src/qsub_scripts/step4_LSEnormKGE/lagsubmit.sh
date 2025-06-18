#PBS -N gennewparam
#PBS -q casper
#PBS -l walltime=8:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=1

sleep 21600
qsub iter7_step3_gen_newparams.sh


