# directly change settings in /glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_MOASMO_Derecho_part2_submission.py

# generate scripts for submission. this script does not submit

iter=0 # iteration number
basin_on_one_node=8 # number of basins in one node
script=/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_emulator_part2_submission.py

python $script $iter $basin_on_one_node
