# functions needed for MO-ASMO calibration for one basin
# Gong et al., (2015) Multiobjective adaptive surrogate modeling-based optimization for parameter estimation of large, complex geophysical models, WRR
# https://github.com/gongw03/MO-ASMO

import os, sys, subprocess, time
import toml
from MOASMO_parameters import generate_initial_parameter_sets, surrogate_model_train_and_pareto_points
import run_multiple_paramsets

# Four step design following Gong et all., (2015)

########################################################################################################################
# load configurations
# config_file = 'config.toml'
# config = toml.load(config_file)

# inputs
file_parameter_list = '/glade/u/home/guoqiang/CTSM_repos/moasmo_test/param_ASG_20221206_moasmo.csv'
path_CTSM_base = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest_LMWG/CAMELS_100'
script_singlerun = '/glade/u/home/guoqiang/CTSM_repos/moasmo_test/run_one_paramset.py'
script_clone = '/glade/u/home/guoqiang/CTSM_repos/CTSM/cime/scripts/create_clone'
ref_streamflow = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest_LMWG/CAMELS_100_OstCalib/refdata/streamflow_data.csv'
add_flow_file = 'nofile'

# outputs
path_paramset = '/glade/scratch/guoqiang/moasmo_camels100/param_sets'
path_submit = '/glade/scratch/guoqiang/moasmo_camels100/run_model'
path_archive = '/glade/scratch/guoqiang/moasmo_camels100/ctsm_outputs'

# evaluation period
date_start = '1994-10-01'
date_end = '1998-09-30'

# MO-ASMO parameters
sampling_method = 'glp'
num_init = 36 # initial number of samples
num_per_iter = 20 # number of selected pareto parameter sets for each iteration
num_iter = 4 # including the initial iteration
cpus_per_iter = 36 # how many cpus are used to run each iteration (i.e., the number of cpus of an entire node)

########################################################################################################################
# MO-ASMO main

file_metric_all = []
file_param_all = []

for it in range(num_iter):
    print('#'*50)
    print(f'Start iterattion {it}. Total iteration number: {num_iter}')
    t1 = time.time()

    iterflag = it

    if it == 0:
        # Initial sampling which generates many parameter sets
        print('Generating initial parameters')
        init_param_filelist = generate_initial_parameter_sets(file_parameter_list, sampling_method, path_paramset, path_CTSM_base, num_init)
        sample_num = num_init
    else:
        sample_num = num_per_iter

    # Run models based on all parameter samples for this iteration. Individual jobs will be submitted
    run_multiple_paramsets.generate_and_submit_multi_CTSM_runs(iterflag, path_submit, path_paramset, path_CTSM_base,
                                                               path_archive, script_singlerun, script_clone,
                                                               date_start, date_end, ref_streamflow, add_flow_file, cpus_per_iter)

    # Don't continue until all runs are finished
    file_metric_iter, file_param_iter = run_multiple_paramsets.check_if_all_runs_are_finsihed(path_archive, iterflag, sample_num)
    file_metric_all.append(file_metric_iter)
    file_param_all.append(file_param_iter)

    # train a surrogate model and select pareto parameter sets
    surrogate_model_train_and_pareto_points(file_parameter_list, file_param_all, file_metric_all, path_paramset, iterflag, num_per_iter, path_CTSM_base)

    t2 = time.time()
    print(f'Iteration {it} is complete. Time cost (s) is {t2 - t1}')

########################################################################################################################
# Main loop: running following trials