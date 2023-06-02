# functions needed for MO-ASMO calibration for one basin
# Gong et al., (2015) Multiobjective adaptive surrogate modeling-based optimization for parameter estimation of large, complex geophysical models, WRR
# https://github.com/gongw03/MO-ASMO

import os, sys, subprocess
import numpy as np
import xarray as xr
import toml

from MOASMO_parameters import generate_initial_parameter_sets






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
outpath = '/glade/scratch/guoqiang/moasmo_test/param_sets'
path_CTSM_case = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest_LMWG/CAMELS_100'
num_init = 200
print('Generating initial parameters')
init_param_filelist = generate_initial_parameter_sets(file_parameter_list, sampling_method, outpath, path_CTSM_case, num_init)


########################################################################################################################
# Run models based on all initial parameters. Individual jobs will be submitted



def create_script_to_run_CTSM_in_batch(ctsm_case_paths, script_file):
    # a list of all CTSM model cases to be run in parallel
    with open(script_file, 'w') as f:
        for p in ctsm_case_paths:
            f.write(f"cd {p} && ./case.submit --no-batch\n")
    _ = subprocess.run(f'chmod +x {script_file}', shell=True)

def create_script_for_submission(lines):
    lines4 = ['\n', 'module load conda/latest parallel', 'conda activate npl-2022b', '\n']
    lines = ['parallel --jobs 4 < script.sh']





path_CTSM_source = ''
path_CTSM_case_base = ''
path_calib = ''

moasmo_num_init = 200 # initial number of model runs
moasmo_num_pareto_select = 20 # select a portion of the solutions to represent the Pareto Frontier
moasmo_stop_maxiter = 15 # max number of iterations to stop calibration
moasmo_stop_objdiff = 0.01 # min difference between objective functions to stop calibration

# here model parallel run is used instead of ninst because of the 1-cpu requirement
# num_batch is set to 36, the number of cores per node, meaning 36 cases will be run in parallel. of course it can be larger.
# for one-basin calibration, the required cpus will be num_batch. this job will run all moasmo_num_init or moasmo_num_pareto_select runs in parallel

# for example, for cpus_per_submission=36, moasmo_num_init=200, modelruns_per_submission=72, the script will submit 2 jobs:
# job-1 with 36 cpus and 12 hours, runs 0-72 in parallel
# job-2 with 36 cpus and 12 hours, runs 72-144 in parallel
# job-3 with 36 cpus and 12 hours, runs 144-200 in parallel (not ideal, so carefully design these numbers can improve efficency)

cpus_per_submission = 36 # note that Cheyenne charges for the entire node even this value is smaller than 36

# cheyenne has a limit of 12 hours. If running a model needs 2 hours and 36 cpus are required,  modelruns_per_submission should be < 36 * 12 / 2
# modelruns_per_submission is recommended to be >= num_batch
modelruns_per_submission = 100



for iter_now in range(moasmo_stop_maxiter):
    print('Current iteration:', iter_now)

    if iter_now == 0:
        print(f'This is the initial iteration. {moasmo_num_init} cases will be created and run in parallel')

        num_modelruns = moasmo_num_init
        num_submissions = np.floor(moasmo_num_init / modelruns_per_submission)

        # clone model cases
        xmlquery_output(path_CTSM_case_base, 'CIME_OUTPUT_ROOT')

        clone_CTSM_model_case(script_clone, source_modelcase, target_modelcase, target_cimeoutput)





        # change parameters

        # submit jobs

        # check whether all jobs are finished

        # train surrogate model

        # get and save Pareto parameter sets for the next iteration

    else:
        # same with the previous step
        num_modelruns = moasmo_num_pareto_select
        pass







########################################################################################################################
# 3. Main loop of MO-ASMO
# a. Surrogate model training
# b. Run NSGA-II multiobjective optimization on the surrogate model and obtain the Pareto optimal points
# c. From the matrix given by NSGA-II, select a portion of the solutions (typically 20%) that have the largest
# crowding distances (or the largest weighted crowding distance, if the reference point is provided).
# Run the dynamic model with the selected optimal points
# d. Append X' and Y' to the input-output pool
# e. Return to step a) if termination conditions are not satisfied

def train_surrogate_model():
    pass

def obtain_Pareto_optimal_parameters():
    pass

########################################################################################################################
# 4. End of MO-ASMO
# when meeting the defined criteria, end MO-ASMO calib.


########################################################################################################################
# main functions

