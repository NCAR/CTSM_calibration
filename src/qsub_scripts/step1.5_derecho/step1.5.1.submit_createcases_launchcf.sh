# This works on Derecho and Casper, not on Cheyenne
# -J 0-127 means use 128 arrays. 
# it can only require one node, 128 CPUs on Derecho.
# all 235gb memory is assigned
# put config_env.sh in the same folder with the cmdfile
# (Jan 3, 2024). Need to submit from the folder containing config_env.sh so this can be loaded
# a work around is to generate multiple cmd files

# not work as expected

launch_cf -A NCGD0013 -q main -l select=1:ncpus=128 -l walltime=1:00:00 -J 0-127 /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/submission/create_cases_1-671.txt
