# group different basins in one node
# after part2 simulations are done, if we want to check other evaluation metrics, this script can be run to do the evaluation

import numpy as np
import os, glob, sys, toml
from run_one_paramset_Derecho import *
from mo_evaluation import mo_evaluate_return_many_metrics

if __name__ == '__main__':

    # configfile = sys.argv[1]
    # iterflag = int(sys.argv[2])
    # trialflag = int(sys.argv[3])

    for basin in range(10):
        print('basin', basin)
                   
        configfile = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/configuration/_level1-{basin}_config_MOASMO.toml'
        iterflag = 0
    
        for trialflag in range(400):
            if np.mod(trialflag, 100)==0:
                print(trialflag)
        
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
        
            date_start = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=ignore_month)).strftime('%Y-%m-%d') # ignor the first year when evaluating model
            if STOP_OPTION == 'nyears':
                date_end = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(years=STOP_N)).strftime('%Y-%m-%d')
            elif STOP_OPTION == 'nmonths':
                date_end = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=STOP_N)).strftime('%Y-%m-%d')
            else:
                sys.exit(f'STOP_OPTION must be nyears or nmonths. {STOP_OPTION} is not accepted.')
        
        
            if config['path_calib'] == 'NA':
                path_MOASMOcalib = f'{path_CTSM_base}_MOASMOcalib'
            else:
                path_MOASMOcalib = config['path_calib']
            path_archive = f'{path_MOASMOcalib}/ctsm_outputs'
        
            caseflag = f'iter{iterflag}_trial{trialflag}'
        
            # evaluate model results
            infilelist = glob.glob(f'{path_archive}/{caseflag}/lnd/hist/*.clm2.h1.*.nc')
            infilelist.sort()
            fsurdat = get_parameter_from_Namelist_or_lndin('fsurdat', f'{path_CTSM_base}/user_nl_clm', f'{path_CTSM_base}/Buildconf/clmconf/lnd_in', type='str')
            outfile_metric = f'{path_archive}/{caseflag}/evaluation_many_metrics.csv'

            if not os.path.isfile(outfile_metric):
                if len(infilelist)>0:
                    mo_evaluate_return_many_metrics(outfile_metric, infilelist, fsurdat, date_start, date_end, ref_streamflow, add_flow_file)
