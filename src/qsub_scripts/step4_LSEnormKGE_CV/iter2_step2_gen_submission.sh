# directly change settings in /glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_MOASMO_Derecho_part2_submission.py

# generate scripts for submission. this script does not submit

iter=2 # iteration number
basin_on_one_node_train=15 # number of basins in one node
basin_on_one_node_test=627 # number of basins in one node
script=/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/main_emulator_part2_submission_LSECV.py

python $script $iter $basin_on_one_node_train $basin_on_one_node_test
