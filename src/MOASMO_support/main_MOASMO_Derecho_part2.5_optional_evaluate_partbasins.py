    ####################################################
# parallel version

import numpy as np
import os, glob, sys, toml
from run_one_paramset_Derecho import *
from mo_evaluation import mo_evaluate_return_many_metrics
from multiprocessing import Pool

def run_trial(params):
    basin, caseflag, configfile = params
    config = toml.load(configfile)
    
    # inputs
    path_CTSM_base = config['path_CTSM_case']
    ref_streamflow = config['file_Qobs']
    
    if 'add_flow_file' in config:
        add_flow_file = config['add_flow_file']
    else:
        add_flow_file = 'NA'
    
    # evaluation period
    RUN_STARTDATE = config['RUN_STARTDATE']
    ignore_month = config['ignore_month']
    STOP_OPTION = config['STOP_OPTION']
    STOP_N = config['STOP_N']
    
    date_start = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=ignore_month)).strftime('%Y-%m-%d')
    if STOP_OPTION == 'nyears':
        date_end = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(years=STOP_N)).strftime('%Y-%m-%d')
    elif STOP_OPTION == 'nmonths':
        date_end = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=STOP_N)).strftime('%Y-%m-%d')
    else:
        print(f'STOP_OPTION must be nyears or nmonths. {STOP_OPTION} is not accepted.')
        return
    
    if config['path_calib'] == 'NA':
        path_MOASMOcalib = f'{path_CTSM_base}_MOASMOcalib'
    else:
        path_MOASMOcalib = config['path_calib']
    path_archive = f'{path_MOASMOcalib}/ctsm_outputs_normKGE'
    
    # evaluate model results
    infilelist = glob.glob(f'{path_archive}/{caseflag}/lnd/hist/*.clm2.h1.*.nc')
    infilelist.sort()
    fsurdat = get_parameter_from_Namelist_or_lndin('fsurdat', f'{path_CTSM_base}/user_nl_clm', f'{path_CTSM_base}/Buildconf/clmconf/lnd_in', type='str')
    
    outfile_metric = f'{path_archive}/{caseflag}/evaluation_many_metrics.csv'
    if not os.path.isfile(outfile_metric):
        print('saving', outfile_metric)
        if len(infilelist) > 0:
            mo_evaluate_return_many_metrics(outfile_metric, infilelist, fsurdat, date_start, date_end, ref_streamflow, add_flow_file)
        else:
            print(f"No input files found for {caseflag} in basin {basin}.")

    # the other two periods
    date_start = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=ignore_month)).strftime('%Y-%m-%d')
    if STOP_OPTION == 'nyears':
        date_end = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(years=4)).strftime('%Y-%m-%d')
    elif STOP_OPTION == 'nmonths':
        date_end = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=48)).strftime('%Y-%m-%d')
    else:
        print(f'STOP_OPTION must be nyears or nmonths. {STOP_OPTION} is not accepted.')
        return

    outfile_metric = f'{path_archive}/{caseflag}/evaluation_many_metrics_period1.csv'
    if not os.path.isfile(outfile_metric):
        print('saving', outfile_metric)
        if len(infilelist) > 0:
            mo_evaluate_return_many_metrics(outfile_metric, infilelist, fsurdat, date_start, date_end, ref_streamflow, add_flow_file)
        else:
            print(f"No input files found for {caseflag} in basin {basin}.")


    date_start = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=36)).strftime('%Y-%m-%d')
    if STOP_OPTION == 'nyears':
        date_end = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(years=STOP_N)).strftime('%Y-%m-%d')
    elif STOP_OPTION == 'nmonths':
        date_end = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=STOP_N)).strftime('%Y-%m-%d')
    else:
        print(f'STOP_OPTION must be nyears or nmonths. {STOP_OPTION} is not accepted.')
        return

    outfile_metric = f'{path_archive}/{caseflag}/evaluation_many_metrics_period2.csv'
    if not os.path.isfile(outfile_metric):
        print('saving', outfile_metric)
        if len(infilelist) > 0:
            mo_evaluate_return_many_metrics(outfile_metric, infilelist, fsurdat, date_start, date_end, ref_streamflow, add_flow_file)
        else:
            print(f"No input files found for {caseflag} in basin {basin}.")


if __name__ == '__main__':
    # Example usage

    num_processes = 128
    basin_num = 627

    pool = Pool(processes=num_processes)
    
    tasks = []
    for basin in range(basin_num):
        print('basin', basin)
        configfile = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/configuration/_level1-{basin}_config_MOASMO.toml'

        pattern = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_{basin}_MOASMOcalib/ctsm_outputs_normKGE/iter1_trial*'
        matching_folders = [f.split('/')[-1] for f in glob.glob(pattern)] # e.g., iter0_trial58
        print('number of folders:', len(matching_folders))
        
        for caseflag in matching_folders:
            tasks.append((basin, caseflag, configfile))
    
    pool.map(run_trial, tasks)
    pool.close()  # Close the pool to prevent any more tasks from being submitted
    pool.join()   # Wait for the worker processes to terminate

