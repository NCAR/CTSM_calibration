#PBS -N paraOst
#PBS -q share
#PBS -l walltime=1:00:00
#PBS -A P08010000
#PBS  -l select=1:ncpus=11

#submit Ostrich calibration job after model is built
#parallel ostrich calibration

module load conda/latest
conda activate npl-2022b

pathparent="/glade/u/home/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest"
tarfolders=$(ls -d ${pathparent}/CAMELS_10*_OstCalib)

parallel --jobs 11 'cd {}/ && ./OstrichGCC' ::: ${tarfolders}