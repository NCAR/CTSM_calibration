# functions needed for MO-ASMO calibration for one basin
# Gong et al., (2015) Multiobjective adaptive surrogate modeling-based optimization for parameter estimation of large, complex geophysical models, WRR
# https://github.com/gongw03/MO-ASMO

import os, sys, subprocess
import numpy as np
import pandas as pd
import xarray as xr
import toml

from MOASMO_parameters import generate_initial_parameter_sets
import run_multiple_paramsets





# import MO-ASMO functions
path_MOASMO = '/glade/u/home/guoqiang/model_sources/MO-ASMO/src'
sys.path.append(path_MOASMO)
import sampling

# Four step design following Gong et all., (2015)

########################################################################################################################
# load configurations
config_file = 'config.toml'
config = toml.load(config_file)

########################################################################################################################
# Initial sampling which generates many parameter sets
file_parameter_list = '/glade/u/home/guoqiang/CTSM_repos/moasmo_test/param_ASG_20221206.csv'
sampling_method = 'glp'
path_paramset = '/glade/scratch/guoqiang/moasmo_test/param_sets'
path_CTSM_case = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest_LMWG/CAMELS_100'
num_init = 200
print('Generating initial parameters')
init_param_filelist = generate_initial_parameter_sets(file_parameter_list, sampling_method, path_paramset, path_CTSM_case, num_init)


########################################################################################################################
# Run models based on all initial parameters. Individual jobs will be submitted
iterflag = 0
path_allruns = '/glade/scratch/guoqiang/moasmo_test/run_model'
# path_paramset = '/glade/scratch/guoqiang/moasmo_test/param_sets'
script_singlerun = '/glade/u/home/guoqiang/CTSM_repos/moasmo_test/run_one_paramset.py'
script_clone = '/glade/u/home/guoqiang/CTSM_repos/CTSM/cime/scripts/create_clone'
path_CTSM_base = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest_LMWG/CAMELS_100'
path_archive = '/glade/scratch/guoqiang/moasmo_test/ctsm_outputs'
date_start = '1994-10-01'
date_end = '1998-09-30'
ref_streamflow = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest_LMWG/CAMELS_100_OstCalib/refdata/streamflow_data.csv'
add_flow_file = 'nofile'

run_multiple_paramsets.generate_and_submit_multi_CTSM_runs(iterflag, path_allruns, path_paramset, path_CTSM_base, path_archive, script_singlerun, script_clone, date_start, date_end, ref_streamflow, add_flow_file)
run_multiple_paramsets.check_if_all_runs_are_finsihed(path_archive, iterflag, num_init) # this function will not stop until all runs are finished

obtain_optimal_parameters_from_surrogate_model()

########################################################################################################################
# Main loop: running following trials