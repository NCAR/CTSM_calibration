# functions needed for MO-ASMO calibration
# Gong et al., (2015) Multiobjective adaptive surrogate modeling-based optimization for parameter estimation of large, complex geophysical models, WRR
# https://github.com/gongw03/MO-ASMO

import os, sys, subprocess
import numpy as np
import xarray as xr


# import MO-ASMO functions
path_MOASMO = '/glade/u/home/guoqiang/model_sources/MO-ASMO/src'
sys.path.append(path_MOASMO)
import sampling

# Four step design following Gong et all., (2015)

########################################################################################################################
# 1. Problem definition
# y=f(x) where f(.) is the dynamic model, which has N adjustable parameters and M objective functions,
# x is the N dimensional parameter vector and y is the M dimensional objective vector.
def objective_function():
    return


########################################################################################################################
# 2. Initial sampling:
# Generate a TXN matrix X using the Good Lattice Points method with RGS de-correlation,
# where T is the number of sample points. Run the dynamic model for T times and obtain the multiobjective results Y
# Y=f(X)
# where Y is a TXM matrix containing the objective functions

def generate_initial_parameter_sets(param_upper_bound, param_lower_bound, sampling_method, num_init=-1):
    # example parameters
    # sampling_method = 'lh'  # lh: LatinHypercubeDesign, slh: SymmetricLatinHypercubeDesign, glp: GoodLatticePointsDesign
    # param_upper_bound = {'param1': np.array(15), 'param2': np.array([1, 2, 3])}
    # param_lower_bound = {'param1': np.array(3), 'param2': np.array([0.2, 1.5, 2.2])}

    # dimension sizes
    num_param = len(param_lower_bound) # number of parameters to be calibrated
    if not num_init > 0:
        num_init = num_param * 20 # number of initial samples (i.e., initial model runs). A proper initial sample size should be 15–20 times the number of parameters (Gong et al., 2015)

    # get initial factors between 0 and 1 which will be used to scale real parameters
    # init_factors: [num_init, num_param]
    if sampling_method == 'lh':
        init_factors = sampling.lh(num_init, num_param)
    elif sampling_method == 'slh':
        init_factors = sampling.slh(num_init, num_param)
    elif sampling_method == 'glp': # glp is used by Gong et al., 2015
        init_factors = sampling.glp(num_init, num_param)
    else:
        sys.exit('Unknown sampling method!')

    # obtain parameter sets using the scaling factors
    param_set_all = []
    for i in range(num_init):
        parami = {}
        for j, key in enumerate(param_upper_bound):
            parami[key] = init_factors[i, j] * (param_upper_bound[key] - param_lower_bound[key]) + param_lower_bound[key]
        param_set_all.append(parami)

    return param_set_all


########################################################################################################################
# check if there is any binded variable
# if there is, add binded variables to df_calibparam

df_bind = pd.DataFrame()
for i in range(len(df_calibparam)):
    bindvari = df_calibparam.iloc[i]['Binding']
    if bindvari != 'None':
        bindvari = bindvari.split(',')
        for bv in bindvari:
            dftmp = df_calibparam.iloc[[i]].copy()
            dftmp['Parameter'] = bv
            # mask other cols
            for col in ['Default', 'Lower', 'Upper', 'Binding', 'Parameter_Ost']:
                if col in dftmp.columns:
                    dftmp[col] = 'None'
            df_bind = pd.concat([df_bind, dftmp])

df_calibparam = pd.concat([df_calibparam, df_bind])
df_calibparam.index = np.arange(len(df_calibparam))




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




