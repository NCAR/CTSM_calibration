
# run a CTSM case for one iteration and one trial
# 1. clone a case
# 2. update parameters
# 3. run model
# 4. move outputs to another directory
# 5. delete cloned case

# The Derecho version can run many cases within one node to be efficient, but manual submission is needed


import os, sys, glob, subprocess, pathlib, random, time
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append('/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support/')

from mo_evaluation import mo_evaluate, mo_evaluate_return_many_metrics, mo_evaluate_return_many_metrics_mizuroute
from MOASMO_parameters import get_parameter_from_Namelist_or_lndin

sys.path.append('/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/calibration/')
from run_mizuroute import main_run_mizuroute

def xmlquery_output(pathCTSM, keyword):
    os.chdir(pathCTSM)
    out = subprocess.run(f'./xmlquery {keyword}', shell=True, capture_output=True)
    out = out.stdout.decode().strip().split(' ')[-1]
    return out
    
if __name__ == '__main__':

    # input arguments
    path_CTSM_base = sys.argv[1]
    date_start  = sys.argv[2]
    date_end  = sys.argv[3]
    ref_streamflow = sys.argv[4]

    add_flow_file = 'na'
    ########################################################################################################################
    # run model in "no-batch" mode
    os.chdir(path_CTSM_base)
    subprocess.run('./case.submit --no-batch', shell=True)

    ########################################################################################################################
    # run mizuroute

    ########################################################################################################################
    # run mizuroute
    # hard coded paths ..

    path_archive = xmlquery_output(path_CTSM_base, 'DOUT_S_ROOT')
    outpath_mizu = f'{path_archive}/mizuroute'
    if True:
        print('run routing')
        
        inpath_ctsmout = f'{path_archive}/lnd/hist'
        mizuEXE = '/glade/u/home/mizukami/model/mizuRoute/route/bin/route_runoff.intel.cesm-coupling.n02_v2.1.4-standalone'

        basinnum = int(path_CTSM_base.split('_')[-1])
        inpath_mizusetting = f'/glade/work/guoqiang/CTSM_CAMELS/mizuroute_settings/level1_{basinnum}'
        
        main_run_mizuroute(inpath_ctsmout, inpath_mizusetting, mizuEXE, outpath_mizu)

    
    ########################################################################################################################
    # evaluate model results

    # # clm output files
    # infilelist = glob.glob(f'{path_archive}/lnd/hist/*.clm2.h1.*.nc')
    # infilelist.sort()

    # # surf data
    # fsurdat = get_parameter_from_Namelist_or_lndin('fsurdat', f'{path_CTSM_base}/user_nl_clm', f'{path_CTSM_base}/Buildconf/clmconf/lnd_in', type='str')

    # ################ CLM evaluation
    # # 24 error metrics
    # outfile_metric = f'{path_archive}/evaluation_many_metrics.csv'
    # if not os.path.isfile(outfile_metric):
    #     print('saving', outfile_metric)
    #     if len(infilelist) > 0:
    #         mo_evaluate_return_many_metrics(outfile_metric, infilelist, fsurdat, date_start, date_end, ref_streamflow, 0, add_flow_file)
    #     else:
    #         print(f"No input files found for basin {basin}.")


    # ########## mizuroute evaluation
    # infilelist_mizuroute = glob.glob(f'{path_archive}/mizuroute/sflow*.nc')
    # infilelist_mizuroute.sort()

    # outfile_metric_mizuroute = f'{path_archive}/evaluation_many_metrics_mizuroute_s-1.csv'
    # if not os.path.isfile(outfile_metric_mizuroute):
    #     mo_evaluate_return_many_metrics_mizuroute(outfile_metric_mizuroute, infilelist_mizuroute, date_start, date_end, ref_streamflow, shift=-1)
