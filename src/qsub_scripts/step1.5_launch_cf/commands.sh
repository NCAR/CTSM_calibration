# This works on Derecho and Casper, not on Cheyenne
# -J 0-127 means use 128 arrays. 
# it can only require one node, 128 CPUs on Derecho.
# all 235gb memory is assigned
# put config_env.sh in the same folder with the cmdfile
# a work around is to generate multiple cmd files

launch_cf -A NCGD0013 -q main -l select=1:ncpus=128 -l walltime=12:00:00 -J 0-127 /glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO/submission/create_cases_part2.txt