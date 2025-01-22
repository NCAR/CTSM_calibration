#!/bin/bash


START_TIME=$SECONDS

module load cdo nco

basin=$1
echo "processing basin $basin"

path0="/glade/campaign/cgd/tss/people/guoqiang/CTSMcases/CAMELS_Calib/Calib_all_HH_Ostrich/"
inpath_data=${path0}/${basin}_Ostrich/archive/DefaultSimu/
outpath=${path0}/${basin}_Ostrich/archive_stats/
type="h0"
vars="RAIN,SNOW,RAIN_FROM_ATM,SNOW_FROM_ATM,QRUNOFF,TBOT"
outfile=${outpath}/DefaultSimu_QPT.nc

mkdir -p ${outpath}

if [ -f "$outfile" ]; then
    echo "$outfile exists. No need to run"
  
else

    ###### method-1: cdo
    # Error: cdo(1) cat (Abort): Input streams have different number of variables per timestep!
    # Because time-step 1 has some variables (attributes) not contained in following time steps
    # cdo selname,FSNO -cat ${inpath_data}/*${type}* ${outpath}/lndhist_${type}_FSNO.nc

    ###### method-2: nco

    # Create a temporary directory for intermediate files
    tmpdir=$(mktemp -d -p ${outpath} tmp.XXXXXX)

    # Extract the variable from each file and save to the temporary directory
    echo "Subsetting input files"
    for file in ${inpath_data}/*${type}*; do
        outfiletmp="${tmpdir}/$(basename ${file})"
        # echo "Saving subset file to ${outfiletmp}"
        ncks -v ${vars} ${file} ${outfiletmp}
    done

    # Concatenate the extracted files into one
    echo "Merging all subset files to $outfile"
    ncrcat -O ${tmpdir}/* $outfile

    # Clean up the temporary directory
    echo "Remove temp dir"
    rm -r ${tmpdir}

    ELAPSED_TIME=$(($SECONDS - $START_TIME))
    echo "Total runtime: $ELAPSED_TIME seconds"

fi