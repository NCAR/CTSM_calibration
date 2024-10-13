import os
import numpy as np
import pandas as pd
import sys
import pyviscous
from concurrent.futures import ProcessPoolExecutor, as_completed

def calculate_sensitivity_for_grid_point(params, kge):
    # Temporary suppress print statements
    original_stdout = sys.stdout  # Save a reference to the original standard output
    sys.stdout = open(os.devnull, 'w')  # Redirect standard output to null device
    
    x = params
    y = kge
    ind = ~np.isnan(y)
    x, y = x[ind, :], y[ind]
    y = y[:, np.newaxis]
    nparam = x.shape[1]  # Number of parameters
    sens_indx_first = np.zeros(nparam)

    for xIndex in range(nparam): 
        sens_indx_first[xIndex], _ = pyviscous.viscous(x, y, xIndex, 'first', MSC='AIC', verbose=False) 

    # Restore print functionality
    sys.stdout.close()
    sys.stdout = original_stdout  # Reset the standard output to its original value

    return sens_indx_first

def process_basin(basin_index):
    # Construct file paths
    param_file = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_{basin_index}_MOASMOcalib/ctsm_outputs/iter0_all_meanparam.csv'
    metric_file = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/level1_{basin_index}_MOASMOcalib/ctsm_outputs/iter0_all_metric.csv'
    
    # Read the parameter and metric files
    df_param = pd.read_csv(param_file)
    df_metric = pd.read_csv(metric_file)

    # Extract params and kge
    params = df_param.values
    if 'precip_repartition_nonglc_all_rain_t' in df_metric.columns:
        mask = []
        for c in df_metric.columns:
            if ('precip' in c) and (c != 'precip_repartition_nonglc_all_rain_t'):
                mask.append(False)
            else:
                mask.append(True)
        mask = np.array(mask)
        params = params[:,mask]
    
    kge = df_metric['kge'].values

    # Calculate sensitivity for the current basin
    sens = calculate_sensitivity_for_grid_point(params, kge)
    
    return basin_index, sens, df_param.columns.tolist()

# Initialize a list to store futures and results
all_sensitivities = []
all_param_names = set()

# Use ProcessPoolExecutor to run calculations in parallel
with ProcessPoolExecutor() as executor:
    futures = {executor.submit(process_basin, i): i for i in range(627)}

    for future in as_completed(futures):
        basin_index, sens, param_names = future.result()
        all_sensitivities.append(sens)
        all_param_names.update(param_names)

# Create a DataFrame to hold sensitivities for all basins
sensitivity_df = pd.DataFrame(all_sensitivities)

# Rank parameters for each basin
ranked_sensitivities = sensitivity_df.rank(axis=1, ascending=False)

# Create a complete parameter array with NaN for missing parameters
complete_param_array = pd.DataFrame(index=np.arange(627), columns=all_param_names)

# Populate the complete parameter array
for basin_index in range(627):
    complete_param_array.iloc[basin_index, :len(all_sensitivities[basin_index])] = sensitivity_df.iloc[basin_index]

# Calculate average sensitivity across all basins for each parameter
average_sensitivity = complete_param_array.mean(axis=0)

# Print or save results as needed
print("Average Sensitivity for All Parameters:")
print(average_sensitivity)

# Optionally, save the ranked sensitivities and average sensitivities to CSV
ranked_sensitivities.to_csv('ranked_sensitivities.csv', index=False)
average_sensitivity.to_csv('average_sensitivity.csv')
