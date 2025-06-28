# functions needed for MO-ASMO calibration for one basin
# Gong et al., (2015) Multiobjective adaptive surrogate modeling-based optimization for parameter estimation of large, complex geophysical models, WRR
# https://github.com/gongw03/MO-ASMO

# The Derecho version can run many cases within one node to be efficient, but manual submission is needed

# generate new parameter sets by using MO-ASMO algorithms


import os, sys, subprocess, time, toml
import pandas as pd
from MOASMO_parameter_allbasin_emulator import allbasin_emulator_train_and_optimize
import run_multiple_paramsets_Derecho
from multiprocessing import Pool

iter_end = int(sys.argv[1]) # e.g., iter_end=2 means outputs from iter0 and iter1 will be used to generate new paprameters for iter 2
ncpus = int(sys.argv[2]) 
# ncpus = 20
# iterend = 1
numruns = 100

only_checkruns = False

infile_basin_info = f"/glade/work/guoqiang/CTSM_CAMELS/data_mesh_surf/HillslopeHydrology/CAMELS_level1_basin_info.csv"
infile_param_info = '/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/parameter/CTSM_CAMELS_SA_param_240202.csv'
infile_attr_foruse = '/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/data/camels_attributes_table_TrainModel.csv'
inpath_moasmo = "/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange"
path_CTSM_case_all = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange'


########################################################################################################################
# check whether runs are finished and merge output csv/pkl files
def check_runs_and_merge(tarbasin, iter_end, numruns, path_CTSM_case_all):
    config_file = f'{path_CTSM_case_all}/configuration/_level1-{tarbasin}_config_MOASMO.toml'
    config = toml.load(config_file)

    path_CTSM_base = config['path_CTSM_case']
    if config['path_calib'] == 'NA':
        path_MOASMOcalib = f'{path_CTSM_base}_MOASMOcalib'
    else:
        path_MOASMOcalib = config['path_calib']
    path_archive = f'{path_MOASMOcalib}/ctsm_outputs_emutest'
        
    os.makedirs(path_MOASMOcalib, exist_ok=True) 

    # check whether runs are finished and merge output csv/pkl files
    num_init = config['num_init'] # initial number of samples
    # num_per_iter = config['num_per_iter'] # number of selected pareto parameter sets for each iteration
    num_per_iter = numruns
    for it in range(0, iter_end):
        if it == 0:
            sample_num = num_init
        else:
            sample_num = num_per_iter
        file_metric_iter, file_param_iter = run_multiple_paramsets_Derecho.check_if_all_runs_are_finsihed(path_archive, it, sample_num)
    return (tarbasin, file_metric_iter, file_param_iter)

def parallel_check_and_merge(iter_end, ncpus, numruns, infile_param_info, infile_attr_foruse, inpath_moasmo, path_CTSM_case_all):
    # Create a pool of workers
    with Pool(processes=ncpus) as pool:
        # Prepare the arguments for each process
        args = [(tarbasin, iter_end, numruns, path_CTSM_case_all) for tarbasin in range(627)]
        
        # Run the processes in parallel
        results = pool.starmap(check_runs_and_merge, args)
        
        # Process the results if needed
        for result in results:
            tarbasin, file_metric_iter, file_param_iter = result
            print(f"Processed basin {tarbasin}: {file_metric_iter}, {file_param_iter}")

parallel_check_and_merge(iter_end, ncpus, numruns, infile_param_info, infile_attr_foruse, inpath_moasmo, path_CTSM_case_all)

if only_checkruns == True:
    sys.exit(0)

########################################################################################################################
# build emulator and generate outputs

allbasin_emulator_train_and_optimize(infile_basin_info, infile_param_info, infile_attr_foruse, inpath_moasmo, path_CTSM_case_all, iter_end, ncpus, numruns)


########################################################################################################################
# generate submission settings


for tarbasin in range(627):
    config_file = f'{path_CTSM_case_all}/configuration/_level1-{tarbasin}_config_MOASMO.toml'
    config = toml.load(config_file)
    
    # inputs
    file_parameter_list = config['file_calib_param']
    path_CTSM_base = config['path_CTSM_case']
    path_script_MOASMO = config['path_script_MOASMO']
    path_CTSM_source = config['path_CTSM_source']
    ref_streamflow = config['file_Qobs']
    
    if 'add_flow_file' in config:
        add_flow_file = config['add_flow_file']
    else:
        add_flow_file = 'NA'
    
    script_singlerun = f'{path_script_MOASMO}/run_one_paramset_Derecho.py'
    script_clone = f'{path_CTSM_source}/cime/scripts/create_clone'

    if config['path_calib'] == 'NA':
        path_MOASMOcalib = f'{path_CTSM_base}_MOASMOcalib'
    else:
        path_MOASMOcalib = config['path_calib']
        
    # outputs
    path_paramset = f'{path_MOASMOcalib}/param_sets_emutest'
    path_submit = f'{path_MOASMOcalib}/run_model_emutest'
    path_archive = f'{path_MOASMOcalib}/ctsm_outputs_emutest'
        
    os.makedirs(path_MOASMOcalib, exist_ok=True) 
    
    # evaluation period
    RUN_STARTDATE = config['RUN_STARTDATE']
    ignore_month = config['ignore_month']
    STOP_OPTION = config['STOP_OPTION']
    STOP_N = config['STOP_N']
    
    if 'nonstandard_evaluation' in config:
        nonstandard_evaluation = config['nonstandard_evaluation']
    else:
        nonstandard_evaluation = 'NA'
    
    # HPC job settings
    job_mode = config['job_mode']
    job_CTSMiteration = config['job_CTSMiteration']
    # job_controlMOASMO = config['job_controlMOASMO'] # not needed here
    
    date_start = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=ignore_month)).strftime('%Y-%m-%d') # ignor the first year when evaluating model
    if STOP_OPTION == 'nyears':
        date_end = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(years=STOP_N)).strftime('%Y-%m-%d')
    elif STOP_OPTION == 'nmonths':
        date_end = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=STOP_N)).strftime('%Y-%m-%d')
    else:
        sys.exit(f'STOP_OPTION must be nyears or nmonths. {STOP_OPTION} is not accepted.')

    # generate submission commands (note, this won't submit a real job on Derecho)
    run_multiple_paramsets_Derecho.generate_and_submit_multi_CTSM_runs(iter_end, path_submit, path_paramset, path_CTSM_base, 
                                                                       path_archive, script_singlerun, script_clone, 
                                                                       date_start, date_end, ref_streamflow, add_flow_file,
                                                                       job_CTSMiteration, job_mode)
