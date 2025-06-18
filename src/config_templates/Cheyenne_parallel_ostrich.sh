#PBS -N paraOst
#PBS -q share
#PBS -l walltime=1:00:00
#PBS -A P08010000
#PBS  -l select=1:ncpus=18

# submit Ostrich calibration job after model is built
# parallel ostrich calibration

module load conda/latest
module load parallel
module load cdo
conda activate npl-2022b

# necessary for the regular queue to run CTSM model in parallel. The share queue already has this setting.
export MPI_DSM_DISTRIBUTE=0

pathparent="/glade/u/home/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest"
#tarfolders=$(ls -d ${pathparent}/CAMELS_10*_OstCalib)
tarfolders=""
for i in {101..118..1}
do
folderi=${pathparent}/CAMELS_${i}_OstCalib
tarfolders="${tarfolders} ${folderi}"
done

echo ${tarfolders}

parallel --jobs 18 'cd {}/run && ./OstrichGCC' ::: ${tarfolders}
