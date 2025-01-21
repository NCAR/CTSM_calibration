#PBS -N LTDefa
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -A P08010000
#PBS -l select=1:ncpus=128
#PBS -o /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/submission/LongTermSimu/
#PBS -e /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/submission/LongTermSimu/
#PBS -J 1-5

module load conda cdo
conda activate npl-2024a

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

# Define command file and calculate which lines this job will execute
cmdfile="/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/submission/LongTermSimu/Defa.txt"

# Calculate the start and end line numbers for this job in the job array
start=$(( ($PBS_ARRAY_INDEX - 1) * 128 + 1 ))
end=$(( $start + 127 ))

# Ensure end doesn't go beyond the total number of commands
total_lines=$(wc -l < $cmdfile)
if [ $end -gt $total_lines ]; then
  end=$total_lines
fi

# Extract the commands for this specific job in the array
sed -n "${start},${end}p" $cmdfile | parallel -j 128
