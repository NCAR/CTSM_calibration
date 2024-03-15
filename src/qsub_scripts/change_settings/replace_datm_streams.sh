
sroucefolder=Calib_HH_MOASMO_backup
tarfolder=Calib_HH_MOASMO
cd $sroucefolder

for folder in $(ls -d level*/)
do
 echo cp $folder/user_nl_datm_streams ../${tarfolder}/$folder/user_nl_datm_streams
done

cd ..