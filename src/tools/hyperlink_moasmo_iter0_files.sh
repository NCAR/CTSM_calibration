#!/bin/bash

# Copy iter-0 results to another folder so we can use the workflow to create new simulations

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

for i in {0..626..1}; do
  src_dir="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_${i}_MOASMOcalib/ctsm_outputs"
  dst_dir="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_${i}_MOASMOcalib/ctsm_outputs_emutest"
  create_symlinks "$src_dir" "$dst_dir" "iter0"

  src_dir="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_${i}_MOASMOcalib/param_sets"
  dst_dir="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_${i}_MOASMOcalib/param_sets_emutest"
  create_symlinks "$src_dir" "$dst_dir" "iter0"
  create_symlinks "$src_dir" "$dst_dir" "all_default_parameters.pkl"
done
