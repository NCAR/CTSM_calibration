# Nested basins, e.g., Basin 1 contains 2 and 3 (2 and 3 are indepdent)
# We may use the split-basin strategy, that is simulating Basin 2, Basin 3, and Basin 1 - 2 -3
# For Basin 1 - 2 - 3, Basin 2 and 3 outlet streamflow should be added to the simulated streamflow at the outlet to be compared to Basin 1 outlet streamflow


# This function changes the setting: add_flow_file in CTSM_run_trial.sh for CAMELS-based calibration
import pandas as pd
import os, shutil

def update_txt_file(file, newsettings, start, sep, comment):
    # start, sep, and comment are '', '\'', and ! for summa fileManager.txt
    # start, sep, and comment are '<', ' ', and ! for mizuroute control file
    # start, sep, and comment are '', ' ', and # for ostIn.txt
    # start, sep, and comment are '', '=', and # for run_trial.sh
    if (len(newsettings) > 0) and os.path.isfile(file):
        # read raw data
        with open(file) as f:
            contents = f.readlines()
        # save a new file
        file_new = file + '-temp'
        with open(file_new, 'w') as f:
            for line in contents:
                for name, value in newsettings.items():
                    if line.startswith(start + name):
                        line2 = line.split(comment)[0].strip()
                        if line2.count(sep) == 2: # format: xxx_sep_value_sep (only summa fileManager.txt)
                            oldvalue = line2.split(sep)[1].strip()
                        else:
                            oldvalue = line2.split(sep)[-1].strip()
                        if not isinstance(value, str):
                            value = str(value)
                        line = line.replace(oldvalue, value)
                f.write(line)
        # replace old file
        os.remove(file)
        shutil.move(file_new, file)



infile_basin_info = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/info_ESMFmesh_ctsm_HCDN_nhru_final_671.buff_fix_holes_polygons_simplified_5e-4_split_nested.csv'
outpath_parent = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest'

df_info = pd.read_csv(infile_basin_info)
for i in range(len(df_info)):
    filei = f'{outpath_parent}/CAMELS_{i}_OstCalib/run/CTSM_run_trial.sh'
    if os.path.isfile(filei) and isinstance(df_info.iloc[i]['file_obsQ_indup'], str):
        file_obsQ_indup = [f for f in df_info.iloc[i]['file_obsQ_indup'].split(',') if os.path.isfile(f)]
        if len(file_obsQ_indup) > 0:
            file_obsQ_indup = ','.join(file_obsQ_indup)
            runtrial_setting = {'add_flow_file': file_obsQ_indup}
            update_txt_file(filei, runtrial_setting, start='', sep='=', comment='#')
            _ = os.system(f'chmod +x {filei}')