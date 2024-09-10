# calculate the accuracy metrics of long-term simulations

import numpy as np
import os, glob, sys, toml
from multiprocessing import Pool

sys.path.append('/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support')
from run_one_paramset_Derecho import *
from mo_evaluation import mo_evaluate_return_many_metrics

def run_trial(params):
    folder, configfile = params
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
    
    # evaluate model results
    infilelist = glob.glob(f'{folder}/*.clm2.h1.*.nc')
    infilelist.sort()
    fsurdat = get_parameter_from_Namelist_or_lndin('fsurdat', f'{path_CTSM_base}/user_nl_clm', f'{path_CTSM_base}/Buildconf/clmconf/lnd_in', type='str')


    # evaluation during test period1
    date_end = (pd.Timestamp(RUN_STARTDATE)).strftime('%Y-%m-%d')
    if STOP_OPTION == 'nyears':
        date_start = (pd.Timestamp(RUN_STARTDATE) - pd.offsets.DateOffset(years=STOP_N-ignore_month)).strftime('%Y-%m-%d')
    elif STOP_OPTION == 'nmonths':
        date_start = (pd.Timestamp(RUN_STARTDATE) - pd.offsets.DateOffset(months=STOP_N-ignore_month)).strftime('%Y-%m-%d')
    else:
        print(f'STOP_OPTION must be nyears or nmonths. {STOP_OPTION} is not accepted.')
        return
    print('eval period:', date_start, date_end)
    
    outfile_metric = f'{folder}/evaluation_metrics_5y_test.csv'
    if not os.path.isfile(outfile_metric):
        print('saving', outfile_metric)
        if len(infilelist) > 0:
            mo_evaluate_return_many_metrics(outfile_metric, infilelist, fsurdat, date_start, date_end, ref_streamflow, add_flow_file)
        else:
            print(f"No input files found")
    else:
        print('outfile exists', outfile_metric)

    # evaluation during test period2
    date_end = (pd.Timestamp(RUN_STARTDATE)).strftime('%Y-%m-%d')
    date_start = '1980-10-01'
    print('eval period:', date_start, date_end)
    
    outfile_metric = f'{folder}/evaluation_metrics_Ally_test.csv'
    if not os.path.isfile(outfile_metric):
        print('saving', outfile_metric)
        if len(infilelist) > 0:
            mo_evaluate_return_many_metrics(outfile_metric, infilelist, fsurdat, date_start, date_end, ref_streamflow, add_flow_file)
        else:
            print(f"No input files found")
    else:
        print('outfile exists', outfile_metric)
        

    # evaluation during training period
    date_start = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=ignore_month)).strftime('%Y-%m-%d')
    if STOP_OPTION == 'nyears':
        date_end = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(years=STOP_N)).strftime('%Y-%m-%d')
    elif STOP_OPTION == 'nmonths':
        date_end = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=STOP_N)).strftime('%Y-%m-%d')
    else:
        print(f'STOP_OPTION must be nyears or nmonths. {STOP_OPTION} is not accepted.')
        return
    print('eval period:', date_start, date_end)
    
    outfile_metric = f'{folder}/evaluation_metrics_train.csv'
    if not os.path.isfile(outfile_metric):
        print('saving', outfile_metric)
        if len(infilelist) > 0:
            mo_evaluate_return_many_metrics(outfile_metric, infilelist, fsurdat, date_start, date_end, ref_streamflow, add_flow_file)
        else:
            print(f"No input files found")
    else:
        print('outfile exists', outfile_metric)

    # evaluation during all years
    date_start = '1980-10-01'
    date_end = '2014-09-30'
    print('eval period:', date_start, date_end)
    
    outfile_metric = f'{folder}/evaluation_metrics_ally.csv'
    if not os.path.isfile(outfile_metric):
        print('saving', outfile_metric)
        if len(infilelist) > 0:
            mo_evaluate_return_many_metrics(outfile_metric, infilelist, fsurdat, date_start, date_end, ref_streamflow, add_flow_file)
        else:
            print(f"No input files found")
    else:
        print('outfile exists', outfile_metric)

if __name__ == '__main__':
    # parallel
    num_processes = 128
    basin_num = 627

    pool = Pool(processes=num_processes)
    
    tasks = []
    for basin in range(basin_num):
        print('basin', basin)
        configfile = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/configuration/_level1-{basin}_config_MOASMO.toml'
        # folder = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/LongTermSimu/Defa/level1_{basin}'
        # tasks.append((folder, configfile))
        # for i in range(2, 6):
        for i in [10, 20]:
            folder = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/LongTermSimu/LSEallbasin/level1_{basin}/normKGEr{i}'
            tasks.append((folder, configfile))
    
    pool.map(run_trial, tasks)
    pool.close()  # Close the pool to prevent any more tasks from being submitted
    pool.join()   # Wait for the worker processes to terminate

    # # serial
    # basin_num = 627
    # for basin in range(basin_num):
    #     print('basin', basin)
    #     configfile = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/configuration/_level1-{basin}_config_MOASMO.toml'
    #     folder = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/LongTermSimu/LSEallbasin/level1_{basin}/normKGEr1'        
    #     run_trial(folder, configfile)
    