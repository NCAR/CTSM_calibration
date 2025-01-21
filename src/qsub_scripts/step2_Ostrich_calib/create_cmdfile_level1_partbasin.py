

calibpath="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_Ostrich_m2err"
cmdpath="/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_Ostrich_m2err/submission"
mkdir -p $cmdpath

total_basins=627  # Updated total basins
basins_per_file=126


for b in range(5):
    bi = b
    cmdfile=f"{cmdpath}/Ostcalib_part{b+1}.txt"
    with open(cmdfile, 'w') as f:
        while bi <= total_basins-1:
            cmd=f"cd {calibpath}/level1_{bi}_OSTRICHcalib/run && ./OstrichGCC"
            bi = bi + 5
            _ = f.write(f'{cmd}\n')
      