# Based on data from the notebook, use pyviscous to calculate sensitivity analysis results
import os, time, sys, glob
import numpy as np
import pandas as pd
import concurrent.futures
import pyviscous
import random

# Assuming the rest of your code for loading data remains the same

# def calculate_sensitivity_for_grid_point(i, params, kge, nparam):
#     x = params
#     y = kge[:,i]
#     ind = ~np.isnan(y)
#     x, y = x[ind,:], y[ind]
#     y = y[:, np.newaxis]
#     sens_indx_first = np.zeros(nparam)
#     sens_indx_total = np.zeros(nparam)
    
#     for xIndex in range(nparam): 
#         sens_indx_first[xIndex], gmcm_first = pyviscous.viscous(x, y, xIndex, 'first', MSC='AIC', verbose=False) 
#         sens_indx_total[xIndex], gmcm_total = pyviscous.viscous(x, y, xIndex, 'total', MSC='AIC', verbose=False)
    
#     return sens_indx_first, sens_indx_total


def calculate_sensitivity_for_grid_point(i, params, kge, nparam):
    print('Processing', i)

    # Temporary suppress print statements
    original_stdout = sys.stdout  # Save a reference to the original standard output
    sys.stdout = open(os.devnull, 'w')  # Redirect standard output to null device
    
    x = params
    y = kge[:,i]
    ind = ~np.isnan(y)
    x, y = x[ind,:], y[ind]
    y = y[:, np.newaxis]
    sens_indx_first = np.zeros(nparam)
    sens_indx_total = np.zeros(nparam)

    for xIndex in range(nparam): 
        sens_indx_first[xIndex], gmcm_first = pyviscous.viscous(x, y, xIndex, 'first', MSC='AIC', verbose=False) 
        sens_indx_total[xIndex], gmcm_total = pyviscous.viscous(x, y, xIndex, 'total', MSC='AIC', verbose=False)


    # Restore print functionality
    sys.stdout.close()
    sys.stdout = original_stdout  # Reset the standard output to its original value

    return sens_indx_first, sens_indx_total

# load data
dtmp = np.load('SA_data.npz', allow_pickle=True)
df_param = pd.DataFrame(dtmp['df_param'])
df_info = pd.DataFrame(dtmp['df_info'])
params = dtmp['params']
paramnames = dtmp['paramnames'] 
kge = dtmp['kge']

# calculate sensitivity analysis
nparam = params.shape[1]
ngrid = kge.shape[1]

sens_indx_first = np.nan * np.zeros([nparam, ngrid])
sens_indx_total = np.nan * np.zeros([nparam, ngrid])

# Set the number of CPUs you wish to use
num_cpus = 120  # For example, to use 4 CPUs

random.seed(123456789)

with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
    futures = [executor.submit(calculate_sensitivity_for_grid_point, i, params, kge, nparam) for i in range(ngrid)]
    for i, future in enumerate(concurrent.futures.as_completed(futures)):
        sens_indx_first[:, i], sens_indx_total[:, i] = future.result()

np.savez_compressed('SA_pyviscous_output.npz', sens_indx_first=sens_indx_first, sens_indx_total=sens_indx_total)