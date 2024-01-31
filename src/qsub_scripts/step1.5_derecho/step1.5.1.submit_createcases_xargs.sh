#PBS -N buildbasin1x
#PBS -q main
#PBS -l walltime=1:00:00
#PBS -A NCGD0013
#PBS -l select=1:ncpus=128
#PBS -e /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/logs/create_cases/
#PBS -o /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/logs/create_cases/


# this is the method that I finally used

module load conda cdo
conda activate npl-2023b


export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

# necessary for the regular queue to run CTSM model in parallel. The share queue already has this setting.
# unfortunately, this only works for node=1. for multiple nodes, CTSM create_case or clone_case do not run at all
# and even for one node, when I try to run CTSM at the same time, many cores jut failed and only some of tasks are running

cmdfile=/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/submission/create_cases_1-671.txt

echo "Processing ${cmdfile}"


cat $cmdfile | xargs -I {} -P 100 sh -c '{}'

#while IFS= read -r line; do
#    $line &
#done < $cmdfile
#wait