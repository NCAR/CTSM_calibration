#PBS -N geninitparam
#PBS -q develop
#PBS -l walltime=6:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=50:mem=150GB

# generate hundreds of parameter sets (iteration 0)

module load conda cdo
conda activate npl-2024a-tgq

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

##### Generate init-1 params
script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_MOASMO_Derecho_part1_initparams.py"

iter_start=0
iter_end=1 # Due to the method range(iter_start, iter_end), it will only run iteration 0

config_file="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/CTSM_config_optmz/one_basin_exps/data_level1_4/exp3/_level1-4_config_MOASMO.toml"

python $script ${config_file} $iter_start $iter_end

##### Main loop
for i in {0..6}
do

    # Run the model
    file=/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/MOASMO_exps/level1_4_exp3/run_model_normKGE/iter${i}/commands_run_iter${i}.txt
    if [ $i -eq 0 ]; then  # Corrected syntax for the conditional statement
        file=/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/MOASMO_exps/level1_4_exp3/run_model/iter${i}/commands_run_iter0.txt
    fi
    
    parallel --jobs 50 < $file  # Ensure that the commands in the file are correctly formatted


    if [ $i -eq 0 ]; then
        mv /glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/MOASMO_exps/level1_4_exp3/ctsm_outputs /glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/MOASMO_exps/level1_4_exp3/ctsm_outputs_normKGE
    fi

    # Calculate new parameters
    script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_MOASMO_Derecho_part3_newparam.py"
    iter_end=$((i+1))
    python $script ${config_file} $iter_end

done
