# for multiple parameter sets by calling run_one_paramset.py
# this script creates submission scripts and run those parameters in parallel

import os, glob, time, re
import sys

import pandas as pd
import numpy as np
from MOASMO_parameters import read_parameter_csv

def generate_and_submit_multi_CTSM_runs(iterflag, path_submit, path_paramset, path_CTSM_base, path_archive,
                                        script_singlerun, script_clone, date_start, date_end, ref_streamflow, add_flow_file, job_CTSMiteration, job_mode):
    # job_mode:
    # lumpsubmit: a job only addressed one grid and thus one submission job can sequentially deal with many jobs. but each iteration will be submitted again
    # lumpsubmit_allinone: a control job with many CPUs will be submitted. all simulation will be within this job. no need to submit 
    # casesubmit: use the casesubmit function provided by CTSM to run each model case independently

    # iterflag = 0
    # path_submit = '/glade/scratch/guoqiang/moasmo_test/run_model'
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
    path_runmodel = f'{path_submit}/iter{iterflag}'
    os.makedirs(path_runmodel, exist_ok=True)
    cpubind_path = path_runmodel

    
    param_filelist = glob.glob(f'{path_paramset}/*iter{iterflag}_trial*.pkl')
    param_filelist.sort()

    commands_run_model = f'{path_runmodel}/commands_run_iter{iterflag}.txt'
    commands_all = []
    with open(commands_run_model, 'w') as f:
        for i in range(len(param_filelist)):
            caseflag = f'iter{iterflag}_trial{i}'
            file_parameter_set = f'{path_paramset}/paramset_iter{iterflag}_trial{i}.pkl'
            commandi = f"python {script_singlerun} {script_clone} {path_CTSM_base} {file_parameter_set} {path_archive} {caseflag} {date_start} {date_end} {ref_streamflow} {add_flow_file} {cpubind_path}"
            _ = f.write(f'{commandi}\n')
            commands_all.append(commandi)

    if job_mode == 'lumpsubmit':
        # create submission file
        script_submission = f'{path_runmodel}/submit_iter{iterflag}.sh'
        # '#PBS -l select=1:ncpus=1'
        for s in job_CTSMiteration:
            if 'ncpus' in s:
                pattern = r'select=(\d+):ncpus=(\d+)'
                match = re.search(pattern, s)
                numnode = int(match.group(1))
                numcpu = int(match.group(2))
                cpus = numnode * numcpu
                break

        lines = [#'module load conda/latest parallel cdo', 
                 'conda activate npl-2024a-tgq',
                 '\n',
                 # 'export MPI_DSM_DISTRIBUTE=0',
                 '\n'
                 f"echo 'Running {commands_run_model}'",
                 f'parallel --jobs {cpus} --joblog joblog.txt < {commands_run_model}']

        lines = job_CTSMiteration + lines

        with open(script_submission, 'w') as f:
            for li in lines:
                _ = f.write(li + '\n')

        _ = os.system(f'chmod +x {script_submission}')

        # # submit job
        # os.chdir(path_runmodel)
        # os.system(f'qsub submit_iter{iterflag}.sh')
        
    elif job_mode == 'lumpsubmit_allinone':
        # create submission file
        script_submission = f'{path_runmodel}/run_iter{iterflag}.sh'
        for s in job_CTSMiteration:
            if 'ncpus' in s:
                pattern = r'select=(\d+):ncpus=(\d+)'
                match = re.search(pattern, s)
                numnode = int(match.group(1))
                numcpu = int(match.group(2))
                cpus = numnode * numcpu
                break

        lines = ['module load conda/latest cdo', 'conda activate npl-2023b',
                 '\n',
                 'export MPI_DSM_DISTRIBUTE=0',
                 '\n'
                 f"echo 'Running {commands_run_model}'",
                 f'parallel --jobs {cpus} --joblog joblog.txt < {commands_run_model}']

        with open(script_submission, 'w') as f:
            for li in lines:
                _ = f.write(li + '\n')

        _ = os.system(f'chmod +x {script_submission}')

        # # submit job
        # os.chdir(path_runmodel)
        # os.system(f'./run_iter{iterflag}.sh')

    elif job_mode == 'casesubmit':

        # generate submission scripts
        lines1 = []
        keywords = ['-q', 'walltime', '-A']
        for l in job_CTSMiteration:
            for k in keywords:
                if k in l:
                    lines1.append(l)
                    break

        lines2 = []
        template_file = f'{path_CTSM_base}/.case.run'
        with open(template_file, 'r') as f:
            for li in f:
                if li.startswith('#PBS'):
                    if np.all([not k in li for k in keywords]):
                        lines2.append(li.strip())

        lines3 = ['\n', 'module load conda/latest parallel cdo', 'conda activate npl-2022b', '\n',]

        subscriptall = []
        for i in range(len(commands_all)):
            script_submission = f'{path_runmodel}/submit_iter{iterflag}_trial{i}.sh'
            with open(script_submission, 'w') as f:
                lines4 = [ commands_all[i] ]
                for li in lines1 + lines2 + lines3 + lines4:
                    _ = f.write(li + '\n')
            os.system(f'chmod +x {script_submission}')
            subscriptall.append(script_submission)

        # don't submit
        # # submit scripts
        # os.chdir(path_runmodel)
        # for i in range(len(commands_all)):
        #     os.system(f'qsub {subscriptall[i]}')

    else:
        sys.exit('Unknown job_mode')



def check_if_all_runs_are_finsihed(path_archive, iterflag, tarnum, sleep=10):
    # check the archive folder. if the archive folder contains the right number of folders, all runs are regarded as finished

    flag = True
    while flag:
        folders = glob.glob(f'{path_archive}/*iter{iterflag}_*/')
        # if len(folders) == tarnum:
        if True:
            # print(f'All runs dones: the number of {path_archive}/*iter{iterflag}_*/ folders equal to {tarnum}')
            outfile_metric, outfile_param = merge_parameter_metric_csv(path_archive, iterflag, tarnum)

            merge_many_metric_csv(path_archive, iterflag, tarnum) # merge many metric csv output
            
            flag = False

    return outfile_metric, outfile_param

def merge_parameter_metric_csv(path_archive, iterflag, tarnum):
    # merge parameters and metric values from all runs to one csv file

    outfile_metric = f'{path_archive}/iter{iterflag}_all_metric.csv'
    outfile_param = f'{path_archive}/iter{iterflag}_all_meanparam.csv'
    if (not os.path.isfile(outfile_metric)) or (not os.path.isfile(outfile_param)):

        flag = True
        metric_values = []
        for i in range(tarnum):
            file_metric = f'{path_archive}/iter{iterflag}_trial{i}/evaluation_metric.csv'

            if not os.path.isfile(file_metric):
                print(f'Warning! File does not exist: {file_metric}')
                continue
            
            df_met = pd.read_csv(file_metric)
            if flag:
                metric_names = list(df_met.columns)
                metric_values = np.nan * np.zeros([tarnum, len(metric_names)])
                flag = False

            metric_values[i, :] = df_met.iloc[0].values

        flag = True
        param_meanvalues = []
        for i in range(tarnum):
            file_param = f'{path_archive}/iter{iterflag}_trial{i}/paramset_iter{iterflag}_trial{i}.pkl'

            if not os.path.isfile(file_param):
                print(f'Warning! File does not exist: {file_param}')
                continue
            
            df_param = pd.read_pickle(file_param)
            if flag:
                param_names = list(df_param['Parameter'].values)
                param_meanvalues = np.nan * np.zeros([tarnum, len(param_names)])
                flag = False
    
            param_meanvalues[i, :] = np.array([np.nanmean(v) for v in df_param['Value'].values])

        
        if len(metric_values) == 0:
            file_metric = f'{path_archive}/iter0_trial0/evaluation_metric.csv'
            df_met = pd.read_csv(file_metric)
            metric_names = list(df_met.columns)
            metric_values = np.nan * np.zeros([tarnum, len(metric_names)])

        if len(param_meanvalues) == 0:
            file_param = f'{path_archive}/iter{iterflag}_trial{i}/paramset_iter0_trial0.pkl'
            df_param = pd.read_csv(file_param)
            param_names = list(df_param.columns)
            param_meanvalues = np.nan * np.zeros([tarnum, len(param_names)])
            
        print(f'Write all metrics for {tarnum} trials in iteration {iterflag} to {outfile_metric}')
        dfout = pd.DataFrame(metric_values, columns=metric_names)
        dfout.to_csv(outfile_metric, index=False)
    
        print(f'Write all parameters (mean value) for {tarnum} trials in iteration {iterflag} to {outfile_param}')
        dfout = pd.DataFrame(param_meanvalues, columns=param_names)
        dfout.to_csv(outfile_param, index=False)

    return outfile_metric, outfile_param


def merge_many_metric_csv(path_archive, iterflag, tarnum):
    # merge parameters and metric values from all runs to one csv file

    outfile_metric = f'{path_archive}/iter{iterflag}_many_metrics_mizuroute_s-1.csv'
    if os.path.isfile(outfile_metric):
        print('Outfile exists:', outfile_metric)
    else:
        flag = True
        metric_values = []
        for i in range(tarnum):
            file_metric = f'{path_archive}/iter{iterflag}_trial{i}/evaluation_many_metrics_mizuroute_s-1.csv'
            if not os.path.isfile(file_metric):
                print(f'Warning! File does not exist: {file_metric}')
                continue
            df_met = pd.read_csv(file_metric)
            if flag:
                metric_names = list(df_met.columns)
                metric_values = np.nan * np.zeros([tarnum, len(metric_names)])
                flag = False
            metric_values[i, :] = df_met.iloc[0].values

        print(f'Write all metrics for {tarnum} trials in iteration {iterflag} to {outfile_metric}')
        if len(metric_values) == 0:
            file_metric = f'{path_archive}/iter0_trial0/evaluation_many_metrics_mizuroute_s-1.csv'
            df_met = pd.read_csv(file_metric)
            metric_names = list(df_met.columns)
            metric_values = np.nan * np.zeros([tarnum, len(metric_names)])
            
        dfout = pd.DataFrame(metric_values, columns=metric_names)
        dfout.to_csv(outfile_metric, index=False)

        
    # outfile_metric = f'{path_archive}/iter{iterflag}_many_metric_period1.csv'
    # if (not os.path.isfile(outfile_metric)):
    #     flag = True
    #     metric_values = []
    #     for i in range(tarnum):
    #         file_metric = f'{path_archive}/iter{iterflag}_trial{i}/evaluation_many_metrics_period1.csv'
    #         if not os.path.isfile(file_metric):
    #             print(f'Warning! File does not exist: {file_metric}')
    #             continue
    #         df_met = pd.read_csv(file_metric)
    #         if flag:
    #             metric_names = list(df_met.columns)
    #             metric_values = np.nan * np.zeros([tarnum, len(metric_names)])
    #             flag = False
    #         metric_values[i, :] = df_met.iloc[0].values

    #     print(f'Write all metrics for {tarnum} trials in iteration {iterflag} to {outfile_metric}')
    #     dfout = pd.DataFrame(metric_values, columns=metric_names)
    #     dfout.to_csv(outfile_metric, index=False)


    # outfile_metric = f'{path_archive}/iter{iterflag}_many_metric_period2.csv'
    # if (not os.path.isfile(outfile_metric)):
    #     flag = True
    #     metric_values = []
    #     for i in range(tarnum):
    #         file_metric = f'{path_archive}/iter{iterflag}_trial{i}/evaluation_many_metrics_period2.csv'
    #         if not os.path.isfile(file_metric):
    #             print(f'Warning! File does not exist: {file_metric}')
    #             continue
    #         df_met = pd.read_csv(file_metric)
    #         if flag:
    #             metric_names = list(df_met.columns)
    #             metric_values = np.nan * np.zeros([tarnum, len(metric_names)])
    #             flag = False
    #         metric_values[i, :] = df_met.iloc[0].values

    #     print(f'Write all metrics for {tarnum} trials in iteration {iterflag} to {outfile_metric}')
    #     dfout = pd.DataFrame(metric_values, columns=metric_names)
    #     dfout.to_csv(outfile_metric, index=False)
