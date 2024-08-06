#!/bin/bash

create_symlinks() {
  local src_dir=$1
  local dst_dir=$2
  local keyword=$3

  # Get the absolute paths
  local abs_src_dir=$(realpath "$src_dir")
  local abs_dst_dir=$(realpath "$dst_dir")

  # Create the destination directory if it doesn't exist
  mkdir -p "$abs_dst_dir"

  # Loop through the files and directories in the source directory
  for item in "$abs_src_dir"/*; do
    if [[ "$item" == *"$keyword"* ]]; then
      local base_item=$(basename "$item")
      local src_path="$abs_src_dir/$base_item"
      local dst_path="$abs_dst_dir/$base_item"
      
      # Create the symbolic link in the destination directory
      if [ ! -e "$dst_path" ]; then
        ln -s "$src_path" "$dst_path"
        echo "Created symlink for $base_item"
      else
        echo "Symlink for $base_item already exists"
      fi
    fi
  done
}

process_iteration() {
  local i=$1

  # src_dir="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_${i}_MOASMOcalib/ctsm_outputs"
  # dst_dir="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_${i}_MOASMOcalib/ctsm_outputs_LSEspaceCV"
  # create_symlinks "$src_dir" "$dst_dir" "iter0"

  # src_dir="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_${i}_MOASMOcalib/param_sets"
  # dst_dir="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_${i}_MOASMOcalib/param_sets_LSEspaceCVr"
  # create_symlinks "$src_dir" "$dst_dir" "iter0"
  # create_symlinks "$src_dir" "$dst_dir" "all_default_parameters.pkl"

  # # Remove files that should not be linked
  # dst_dir="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_${i}_MOASMOcalib/param_sets_LSEspaceCV"
  # rm -f ${dst_dir}/RF_for_iter0_CV_kge.csv
  # rm -f ${dst_dir}/surrogate_model_for_iter0
  # rm -f ${dst_dir}/GPR_for_iter0_CV_kge.csv
  # rm -f ${dst_dir}/GPR_for_iter0_try*_CV_kge.csv

  src_dir="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_${i}_MOASMOcalib/param_sets"
  dst_dir="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_${i}_MOASMOcalib/param_sets_LSEspaceCV"
  mkdir -p $dst_dir
  ln -s "$src_dir/paramset_iter0_trial0.pkl" "$dst_dir/"
}

export -f create_symlinks
export -f process_iteration

# Parallel execution
parallel -j 1 process_iteration ::: {0..626}
