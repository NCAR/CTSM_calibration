# get values from xmlquery

import os, glob
import sys

def get_target_archive_files(pathCTSM, keyword):
    # get the list of archived files of the latest model run
    # # settings
    # pathCTSM = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/CAMELS_LumpCalib'
    # keyword = ".clm2.h1."
    # find files
    st_archive_files = glob.glob(f'{pathCTSM}/st_archive.*')
    st_archive_files.sort()
    st_archive_files = st_archive_files[-1]
    filelist = []
    with open(st_archive_files, 'r')  as f:
        for line in f:
            if line.startswith('moving') or line.startswith('copying'):
                if keyword in line:
                    file = line.split(' to ')[-1].strip()
                    if os.path.isfile(file):
                        print('Append to file list:', file)
                        filelist.append(file)
                    else:
                        sys.exit('File does not exist:', file)
    return filelist
