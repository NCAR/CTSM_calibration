# use forcing/restart files from MOASMO to force Ostrich

pathMOASMO="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_MOASMO"
pathOstrich="/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Calib_all_HH_Ostrich"


for i in {0..626}
do
file=${pathMOASMO}/level1_${i}/user_nl_datm_streams
cp ${file} ${pathOstrich}/level1_${i}

file=$(realpath ${pathMOASMO}/level1_${i}_SpinupFiles/*.r.*.nc)
line="finidat = '${file}'"
echo $line >> ${pathOstrich}/level1_${i}/user_nl_clm
done


for i in {0..39}
do
file=${pathMOASMO}/level2_${i}/user_nl_datm_streams
cp ${file} ${pathOstrich}/level2_${i}

file=$(realpath ${pathMOASMO}/level2_${i}_SpinupFiles/*.r.*.nc)
line="finidat = '${file}'"
echo $line >> ${pathOstrich}/level2_${i}/user_nl_clm
done

for i in {0..3}
do
file=${pathMOASMO}/level3_${i}/user_nl_datm_streams
cp ${file} ${pathOstrich}/level3_${i}

file=$(realpath ${pathMOASMO}/level3_${i}_SpinupFiles/*.r.*.nc)
line="finidat = '${file}'"
echo $line >> ${pathOstrich}/level3_${i}/user_nl_clm
done