# run mizuroute model as a post-processing step to CTSM

import pandas as pd
import numpy as np
import os, sys, glob, shutil
import matplotlib.pyplot as plt
import xarray as xr


def change_text_value(file, newsettings, separator, comment):
    # file: target file
    # start: start string (note for mizuroute.control file, this should be "<name" because mizuroute settings are within "<>")
    # newsettings: dict contain target variables and values
    # separator: character separating variables and values (e.g., "=" for variable=value)
    # comment: character used as character

    # separator, and comment are '\'', and ! for summa fileManager.txt
    # separator, and comment are ' ', and ! for mizuroute control file
    # separator, and comment are ' ', and # for ostIn.txt
    # separator, and comment are '=', and # for run_trial.sh

    # example:
    # summa_setting = {}
    # summa_setting['settingsPath'] = '/the/path/'
    # summa_setting['simStartTime'] = '2009-07-01 00:00'
    # change_text_value('summa_fileManager.txt', summa_setting, separator='\'', comment='!')

    if (len(newsettings) > 0) and os.path.isfile(file):
        # read raw data
        with open(file) as f:
            contents = f.readlines()
        # save a new file
        file_new = file + '-temp'
        with open(file_new, 'w') as f:
            for line in contents:
                for name, value in newsettings.items():
                    if line.startswith(name):
                        line2 = line.split(comment)[0].strip()
                        if line2.count(separator) == 2: # format: xxx_sep_value_sep (only summa fileManager.txt)
                            oldvalue = line2.split(separator)[1].strip()
                        else:
                            oldvalue = line2.split(separator)[-1].strip()
                        if not isinstance(value, str):
                            value = str(value)
                        line = line.replace(oldvalue, value)
                f.write(line)
        # replace old file
        os.remove(file)
        shutil.move(file_new, file)

def main_run_mizuroute(inpath_ctsm, inpath_mizusetting, mizuEXE, outpath, basinID=None, caseflag='trial',):

    ########
    # input arguments
    
    # inpath_ctsm = '/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_emulator/level1_0_calib/ctsm_outputs/iter0_trial0/lnd/hist'
    # inpath_mizusetting = '/glade/work/guoqiang/CTSM_CAMELS/mizuroute_settings/level1_0/'
    # outpath = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_emulator/level1_0_calib/ctsm_outputs/iter0_trial0/mizuroute/'
    # caseflag = 'iter0_trial0'
    # basinID = 1013500
    # mizuEXE = '/glade/u/home/mizukami/model/mizuRoute/route/bin/route_runoff.intel.cesm-coupling.n02_v2.1.4-standalone'
    
    # get CTSM file list
    infilelist_CTSM = glob.glob(f'{inpath_ctsm}/level1_*.clm2.h1.*.nc')

    if len(infilelist_CTSM)==0:
        print('No files found for routing in', inpath_ctsm)
        return
    
    infilelist_CTSM.sort()

    if not outpath.endswith('/'):
        outpath = outpath + '/'
    os.makedirs(outpath, exist_ok=True)
    
    
    ########
    # prepare mizuroute runoff input

    # get basin ID (temporary method)
    if basinID == None:
        infile_topo = glob.glob(f'{inpath_mizusetting}/ntopo_MERIT_Hydro_v1_*.nc')[0]
        basinID = int(infile_topo.split('_')[-1][:-3])
    
    # Extract the dates from the file paths
    dates = [path.split('.')[-2] for path in infilelist_CTSM]
    start_date = dates[0][:-6]
    end_date = dates[-1][:-6]
    
    outfile_clmrunoff = f'{outpath}/CTSM_runoff_{start_date}-to-{end_date}.nc'
    
    if os.path.isfile(outfile_clmrunoff):
        print('clm runoff file exists:', outfile_clmrunoff)
    else:
        print('extracting clm outputs to:', outfile_clmrunoff)
        ds_clm = xr.open_mfdataset(infilelist_CTSM)
        ds_clm_out = ds_clm[['QRUNOFF']].load()
        ds_clm_out = ds_clm_out.rename({'lndgrid':'gru'})
        ds_clm_out.coords['gru'] = [basinID]
        ds_clm_out['gruId'] = xr.DataArray([basinID], dims=('gru'))
        ds_clm_out.to_netcdf(outfile_clmrunoff)
    
    ########
    # copy mizuroute param
    os.system(f'cp {inpath_mizusetting}/param.nml.default {outpath}')
    
    # create a control file for this routing
    
    file_control = f'{outpath}/{caseflag}_control.txt'
    os.system(f'cp {inpath_mizusetting}/control.txt {file_control}')
    
    newsettings = { '<input_dir>': outpath, 
                    '<output_dir>': outpath,
                    '<sim_start>': start_date,
                    '<sim_end>': end_date,
                    '<fname_qsim>': f'CTSM_runoff_{start_date}-to-{end_date}.nc'
                    }
    change_text_value(file_control, newsettings, ' ', '!')
    
    
    ########
    # run mizuroute
    os.system(f'{mizuEXE} {file_control}')