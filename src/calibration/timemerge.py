import xarray as xr
import numpy as np
import glob, re, os, sys, subprocess, pathlib


def highlevel_cdo_mergetime(infiles, tarYearStep=1, rmRawFiles=True, outpath=''):
    # infiles: a list of input forcing files
    # tarYearStep: years of data saved in one file

    # not good ...
    # this is only used to define output file names. can be deleted
    dateformat = '\d+-\d+'  # use re to find the date in the file name (e.g., 2012-05). \d+ - one or more digits.

    tarPath = str(pathlib.Path(infiles[0]).parent)
    outfile_inout_mapping = f'{tarPath}/inout_list_timemerge.txt'
    if os.path.isfile(outfile_inout_mapping):
        sys.exit(f'timemerge has been performed! Check {outfile_inout_mapping}.')

    ########################################################################################################################
    # get the year tag of each file
    num = len(infiles)
    year_tag = np.nan * np.zeros(num)
    for i in range(num):
        infilei = infiles[i]
        with xr.open_dataset(infilei) as ds:
            year_tag[i] = ds.time.values[0].year

    year_unique = np.unique(year_tag)
    batch = np.ceil(len(year_unique) / tarYearStep).astype(int)

    ########################################################################################################################
    # create files
    maplist_infile = []
    maplist_outfile = []
    all_outfiles = ''

    for i in range(batch):
        # find files correspoding to the batch
        year_s = year_unique[i * tarYearStep].astype(int)
        year_e = year_unique[np.min([(i + 1) * tarYearStep, len(year_unique)]) - 1].astype(int)
        file_in = infiles[(year_tag >= year_s) & (year_tag <= year_e)]
        olddate = re.findall(dateformat, file_in[0])
        if len(olddate) == 1:
            olddate = olddate[0]
        else:
            sys.exit(f"Error! Find multiple {dateformat} in the name of {file_in[0]}")

        # define output file name
        file_out = file_in[0].replace(olddate, f'{year_s}-{year_e}')
        if len(outpath) == 0:
            # save in the raw directory
            pass
        else:
            # save in a new directory
            os.makedirs(outpath, exist_ok=True)
            tmp = pathlib.Path(file_out).name
            file_out = f'{outpath}/{tmp}'

        # merge using cdo
        print(' ')
        filetmp = ''
        for f in file_in:
            print('Merging input file:', f)
            maplist_infile.append(f)
            maplist_outfile.append(file_out)
            filetmp = filetmp + f + ' '
        all_outfiles = all_outfiles + file_out + ','
        _ = subprocess.run(f'cdo mergetime {filetmp} {file_out}', shell=True)
        print('Merging output file:', file_out)

    # generate mapping file list
    with open(outfile_inout_mapping, 'w') as f:
        for i in range(len(maplist_infile)):
            _ = f.write(f'{maplist_infile[i]} -> {maplist_outfile[i]}\n')

    # delete raw files
    if rmRawFiles == True:
        for f in infiles:
            _ = os.remove(f)

    return all_outfiles[:-1], maplist_infile, maplist_outfile



