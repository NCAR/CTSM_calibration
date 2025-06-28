#PBS -N MOAiterrun
#PBS -q share
#PBS -l walltime=6:00:00
#PBS -A P08010000
#PBS  -l select=1:ncpus=1

module load conda
conda activate npl-2022b-tgq

submit_script="MOASMO_iter_submit.sh"
run_script="resubmit_run.py"
configfile="file"
logfile="resubmit.log"
iter_num_per_sub=5

python resubmit_run.py ${submit_script} ${run_script} ${configfile} ${logfile} ${iter_num_per_sub}

echo "finish"

