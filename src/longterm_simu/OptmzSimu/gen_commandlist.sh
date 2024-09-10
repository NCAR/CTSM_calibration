
script="/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/longterm_simu/OptmzSimu/CTSM_OptmzSimu_fromMOASMO.py"
path="/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/submission/LongTermSimu"
mkdir -p $path


for r in {2..35}
do

    cmdfile="${path}/Optmz${r}.txt"
    
    rm $cmdfile
    
    for i in {0..626}
    do
        echo "python $script $i $r" >> $cmdfile
    done

done