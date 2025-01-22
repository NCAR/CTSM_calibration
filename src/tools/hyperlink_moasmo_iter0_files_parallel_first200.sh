#!/bin/bash


process_iteration() {
  local i=$1

  src_dir="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_${i}_MOASMOcalib/ctsm_outputs"
  dst_dir="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_${i}_MOASMOcalib/ctsm_outputs_normKGE_200iter0"
  mkdir -p $dst_dir
  for t in {0..199}
  do
    ln -s "$src_dir/iter0_trial${t}" "$dst_dir"
  done 
  # ln -s "$src_dir/iter0_all_meanparam.csv" "$dst_dir"
  # ln -s "$src_dir/iter0_all_metric.csv" "$dst_dir"
  # ln -s "$src_dir/iter0_many_metric.csv" "$dst_dir"

  src_dir="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_${i}_MOASMOcalib/param_sets"
  dst_dir="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_${i}_MOASMOcalib/param_sets_normKGE_200iter0"
  mkdir -p $dst_dir
  for t in {0..199}
  do
    ln -s "$src_dir/paramset_iter0_trial${t}.pkl" "$dst_dir"
  done 
  ln -s "$src_dir/all_default_parameters.pkl" "$dst_dir"

}

export -f process_iteration

# Parallel execution
parallel -j 10 process_iteration ::: {0..626}
# parallel -j 1 process_iteration ::: {0..0}
# process_iteration 0