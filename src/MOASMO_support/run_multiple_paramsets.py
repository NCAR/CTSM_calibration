# for multiple parameter sets by calling run_one_paramset.py
# this script creates submission scripts and run those parameters in parallel

import os, glob, time
import pandas as pd
import numpy as np
from MOASMO_parameters import read_parameter_csv

def generate_and_submit_multi_CTSM_runs(iterflag, path_allruns, path_paramset, path_CTSM_base, path_archive,
                                        script_singlerun, script_clone, date_start, date_end, ref_streamflow, add_flow_file):

    # iterflag = 0
    # path_allruns = '/glade/scratch/guoqiang/moasmo_test/run_model'
    # path_paramset = '/glade/scratch/guoqiang/moasmo_test/param_sets'
    # script_singlerun = '/glade/u/home/guoqiang/CTSM_repos/moasmo_test/run_one_paramset.py'
    # script_clone = '/glade/u/home/guoqiang/CTSM_repos/CTSM/cime/scripts/create_clone'
    # path_CTSM_base = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest_LMWG/CAMELS_100'
    # path_archive = '/glade/scratch/guoqiang/moasmo_test/ctsm_outputs'
    # date_start = '1994-10-01'
    # date_end = '1998-09-30'
    # ref_streamflow = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest_LMWG/CAMELS_100_OstCalib/refdata/streamflow_data.csv'
    # add_flow_file = 'nofile'

    # create command file
    path_runmodel = f'{path_allruns}/iter{iterflag}'
    os.makedirs(path_runmodel, exist_ok=True)

    param_filelist = glob.glob(f'{path_paramset}/*iter{iterflag}_trial*.csv')
    param_filelist.sort()

    commands_run_model = f'{path_runmodel}/commands_run_iter{iterflag}.txt'
    with open(commands_run_model, 'w') as f:
        for i in range(len(param_filelist)):
            caseflag = f'iter{iterflag}_trial{i}'
            file_parameter_set = f'{path_paramset}/paramset_iter{iterflag}_trial{i}.csv'
            commandi = f"python {script_singlerun} {script_clone} {path_CTSM_base} {file_parameter_set} {path_archive} {caseflag} {date_start} {date_end} {ref_streamflow} {add_flow_file}"
            _ = f.write(f'{commandi}\n')

    # create submission file
    script_submission = f'{path_runmodel}/submit_iter{iterflag}.sh'
    cpus = 36 # note that Cheyenne charges for the entire node even this value is smaller than 36

    lines = ['#PBS -N moasmo', '#PBS -q regular',
             '#PBS -l walltime=12:00:00', '#PBS -A P08010000',
             f'#PBS -l select=1:ncpus={cpus}',
             '\n',
             'module load conda/latest parallel cdo', 'conda activate npl-2022b',
             '\n',
             'export MPI_DSM_DISTRIBUTE=0',
             '\n'
             f"echo 'Running {commands_run_model}'",
             f'parallel --jobs {cpus} --joblog joblog.txt < {commands_run_model}']

    with open(script_submission, 'w') as f:
        for li in lines:
            _ = f.write(li + '\n')

    _ = os.system(f'chmod +x {script_submission}')

    # submit job
    os.chdir(path_runmodel)
    os.system(f'qsub submit_iter{iterflag}.sh')



def check_if_all_runs_are_finsihed(path_archive, iterflag, tarnum, sleep=10):
    # check the archive folder. if the archive folder contains the right number of folders, all runs are regarded as finished

    flag = True
    while flag:
        folders = glob.glob(f'{path_archive}/*{iterflag}*/')
        if len(folders) == tarnum:
            print(f'All runs dones: the number of {path_archive}/*{iterflag}*/ folders equal to {tarnum}')
            merge_parameter_metric_csv(path_archive, iterflag, tarnum)
            flag = False
        else:
            flag = True
            time.sleep(sleep)

    return

def merge_parameter_metric_csv(path_archive, iterflag, tarnum):
    # merge parameters and metric values from all runs to one csv file
    for i in range(tarnum):
        file_param = f'{path_archive}/iter{iterflag}_trial{i}/paramset_iter{iterflag}_trial{i}.csv'
        file_metric = f'{path_archive}/iter{iterflag}_trial{i}/evaluation_metric.csv'
        df_met = pd.read_csv(file_metric)
        df_param = read_parameter_csv(file_param)
        if i == 0:
            param_names = list(df_param['Parameter'].values)
            param_meanvalues = np.zeros([tarnum, len(param_names)])

            metric_names = list(df_met.columns)
            metric_values = np.zeros([tarnum, len(metric_names)])

        param_meanvalues[i, :] = np.array([np.nanmean(v) for v in df_param['Value'].values])
        metric_values[i, :] = df_met.iloc[0].values

    outfile_metric = f'{path_archive}/iter{iterflag}_all_metric.csv'
    print(f'Write all metrics for {tarnum} trials in iteration {iterflag} to {outfile_metric}')
    dfout = pd.DataFrame(metric_values, columns=metric_names)
    dfout.to_csv(outfile_metric, index=False)

    outfile_param = f'{path_archive}/iter{iterflag}_all_meanparam.csv'
    print(f'Write all parameters (mean value) for {tarnum} trials in iteration {iterflag} to {outfile_param}')
    dfout = pd.DataFrame(param_meanvalues, columns=param_names)
    dfout.to_csv(outfile_param, index=False)

    return
