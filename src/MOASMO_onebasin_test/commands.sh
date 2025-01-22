# b=97
b=256
# b=514
# b=329
# b=288
# b=202
# b=160

# 1. generate init parameter sets

script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_MOASMO_Derecho_part1_initparams.py"
config_file="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_onebasin_test/_level1-${b}_config_MOASMO_shortparam.toml"

iter_start=0
iter_end=1 # due to the method range(iter_start, iter_end), it will only run iteration 0

python $script ${config_file} $iter_start $iter_end

# 2. generate new parameters for iter-1
script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_MOASMO_Derecho_part3_newparam.py"
config_file="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_onebasin_test/_level1-${b}_config_MOASMO_shortparam.toml"

iter_end=1
python $script $config_file  $iter_end

# 2. generate new parameters for iter-2
script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_MOASMO_Derecho_part3_newparam.py"
config_file="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_onebasin_test/_level1-${b}_config_MOASMO_shortparam.toml"

iter_end=2
python $script $config_file  $iter_end

# 3. generate new parameters for iter-3
script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_MOASMO_Derecho_part3_newparam.py"
config_file="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_onebasin_test/_level1-${b}_config_MOASMO.toml"

iter_end=3
python $script $config_file  $iter_end

# 4. generate new parameters for iter-4
script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_MOASMO_Derecho_part3_newparam.py"
config_file="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_onebasin_test/_level1-${b}_config_MOASMO.toml"

iter_end=4
python $script $config_file  $iter_end


# 5. generate new parameters for iter-5
script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_MOASMO_Derecho_part3_newparam.py"
config_file="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_onebasin_test/_level1-${b}_config_MOASMO.toml"

iter_end=5
python $script $config_file  $iter_end



