
script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/longterm_simu/DefaSimu/CTSM_DeafSimu_fromMOASMO.py"
path="/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/submission/LongTermSimu"
mkdir -p $path
cmdfile="${path}/Defa.txt"

rm $cmdfile

for i in {0..626}
do

    echo "python $script $i" >> $cmdfile
    
done