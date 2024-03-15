# Check emulator ability
# Compare the performance of different emulators

# use observed streamflow data to evaluate model outputs
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os, sys

####### load data
niter = 200
nparam = 27
params = np.nan * np.zeros([niter, nparam])

for i in range(0, niter):
    if i>=0:
        file = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/SA_HH_allbasins/level1/param_sets/paramset_iter0_trial{i}.pkl'
    else:
        file = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/SA_HH_allbasins/level1/param_sets/all_default_parameters.pkl' # -1: default paramters
        
    df_param = pd.read_pickle(file)
    va = df_param['Value'].values
    for j in range(nparam):
        params[i, j]=np.mean(va[j])


ngrid = 627
outpath = '/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/SA_HH_allbasins/level1/ctsm_outputs_evaluation'
kge = np.nan * np.zeros([niter, ngrid])
for t in range(0, niter): # -1 is default
    # outfile metric
    outfile_metric = f'{outpath}/metric_iter0_trial{t}.csv'
    df_metric = pd.read_csv(outfile_metric)
    kge[t, :] = df_metric['KGEmod'].values

index = ~np.isnan(kge[:,0])
kge = kge[index,:]
params = params[index,:]
# kge.shape, params.shape

####### emulator compare
# runbasin = 3

##### RF
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
import numpy as np

# Assuming `kge` and `params` are your datasets with shapes (179, 627) and (179, 27), respectively

# Function to train a model and perform cross-validation for a single basin
def train_and_evaluate(basin_idx, kge_basin, params):
    print(f'Processing basin {basin_idx}')
    
    # Initialize the Random Forest regressor
    model = RandomForestRegressor()
    
    # Perform 5-fold cross-validation and return the mean score
    scores = cross_val_score(model, params, np.ravel(kge_basin), cv=5, scoring='neg_mean_squared_error')
    mean_score = np.mean(scores)
    
    return mean_score

# Split the kge dataset into 627 separate arrays, one for each basin
kge_per_basin = np.split(kge, 627, axis=1)

# Use joblib to parallelize the training and evaluation across basins
cv_scores_rf = Parallel(n_jobs=-1, verbose=10)(delayed(train_and_evaluate)(basin_idx, kge_basin, params) for basin_idx, kge_basin in enumerate(kge_per_basin))

print('Mean CV scores RF with parallel processing:', np.mean(cv_scores_rf))


# cv_scores_rf now contains the cross-validation score for each basin's model
cv_scores_rf = np.array(cv_scores_rf)
print('mean cv_scores_rf', np.nanmean(cv_scores_rf))

np.savez_compressed('cv_scores_rf.npz', cv_scores_rf=cv_scores_rf)


######## GPR
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
import numpy as np

# Assuming `kge` and `params` are already loaded

# Define a function for training and evaluating a single basin
def train_and_evaluate_gpr(basin_idx, kge_basin, params):
    print(f'Processing basin {basin_idx}')
    
    # Define a kernel for the Gaussian Process using the Matérn kernel
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
    
    # Create a pipeline with normalization and GPR
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    pipeline = Pipeline([('normalize', MinMaxScaler()), ('gpr', gp)])
    
    # Perform 5-fold cross-validation and return the mean score
    scores = cross_val_score(pipeline, params, np.ravel(kge_basin), cv=5, scoring='neg_mean_squared_error')
    mean_score = np.mean(scores)
    
    return mean_score

# Split the kge dataset into 627 separate arrays, one for each basin
kge_per_basin = np.split(kge, 627, axis=1)

# Use joblib to parallelize the training and evaluation across basins
cv_scores_gpr = Parallel(n_jobs=-1, verbose=10)(
    delayed(train_and_evaluate_gpr)(basin_idx, kge_basin, params) for basin_idx, kge_basin in enumerate(kge_per_basin)
)

print('Mean CV scores GPR with parallel processing:', np.mean(cv_scores_gpr))

np.savez_compressed('cv_scores_gpr.npz', cv_scores_gpr=cv_scores_gpr)


##### SVM
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
import numpy as np

# Assuming `kge` and `params` are your datasets

# Function to perform training and evaluation for a single basin
def train_and_evaluate_svm(basin_idx, kge_basin, params):
    print(f'Processing basin {basin_idx}')
    svr_pipeline = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
    scores = cross_val_score(svr_pipeline, params, np.ravel(kge_basin), cv=5, scoring='neg_mean_squared_error')
    return np.mean(scores)

# Split the kge dataset into 627 separate arrays, one for each basin
kge_per_basin = np.split(kge, 627, axis=1)

# Parallelize the training and evaluation process across basins
cv_scores_svm = Parallel(n_jobs=-1, verbose=10)(
    delayed(train_and_evaluate_svm)(basin_idx, kge_basin, params)
    for basin_idx, kge_basin in enumerate(kge_per_basin)
)

print('Mean CV scores SVM with parallel processing:', np.nanmean(cv_scores_svm))

# Save the scores
np.savez_compressed('cv_scores_svm.npz', cv_scores_svm=cv_scores_svm)


######## ANN
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
import numpy as np

# Assuming `kge` and `params` are already loaded with shapes (179, 627) and (179, 27), respectively

# Function to perform cross-validation for a single basin
def evaluate_basin(basin_idx, kge_basin, params):
    print(f'Processing basin {basin_idx}')
    
    # Create an MLPRegressor with adjusted parameters
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', 
                       alpha=0.001, batch_size='auto', learning_rate='adaptive', 
                       learning_rate_init=0.001, max_iter=1000, shuffle=True, 
                       random_state=1, tol=0.0001, verbose=False, early_stopping=True, 
                       validation_fraction=0.1, n_iter_no_change=10)
    
    # Use a pipeline to automate scaling
    pipeline = make_pipeline(StandardScaler(), mlp)
    
    # Perform cross-validation and return the mean score
    scores = cross_val_score(pipeline, params, kge_basin.flatten(), cv=5, scoring='neg_mean_squared_error')
    mean_score = np.mean(scores)
    
    return mean_score

# Split the kge dataset into 627 separate arrays, one for each basin
kge_per_basin = np.split(kge, 627, axis=1)

# Parallelize the evaluation across basins
cv_scores_mlp = Parallel(n_jobs=-1, verbose=10)(
    delayed(evaluate_basin)(basin_idx, kge_basin, params) for basin_idx, kge_basin in enumerate(kge_per_basin))

# Convert the scores list to a NumPy array and print the mean CV score
cv_scores_mlp = np.array(cv_scores_mlp)
print('Mean CV Scores for MLP:', np.nanmean(cv_scores_mlp))

# Save the scores to a compressed numpy file
np.savez_compressed('cv_scores_mlp.npz', cv_scores_mlp=cv_scores_mlp)

# ##### MOASMO 
# import sys
# sys.path.append('../../../ctsm_optz/MO-ASMO/src')
# from gp import *

# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import KFold
# import numpy as np

# # Sample data setup -- replace with your actual data
# # params = np.random.randn(179, 27)  # Example feature matrix
# # kge = np.random.randn(179, 627)    # Example target matrix

# n_splits = 5  # Number of folds for cross-validation
# kf = KFold(n_splits=n_splits)

# # Assuming the bounds for the GPR_Matern class are known
# xlb = np.min(params, axis=0)
# xub = np.max(params, axis=0)

# cv_scores_mgpr = []

# # Process each basin separately
# for basin_index in range(kge.shape[1]):

#     if np.mod(basin_index, 50) == 0:
#         print('processing basin', basin_index)
        
#     basin_scores = []
#     y = kge[:, basin_index]

#     for train_index, test_index in kf.split(params):
#         X_train, X_test = params[train_index], params[test_index]
#         y_train, y_test = y[train_index], y[test_index]
        
#         # Initialize and fit the GPR_Matern model
#         model = GPR_Matern(X_train, y_train, nInput=27, nOutput=1, N=len(train_index), xlb=xlb, xub=xub)
        
#         # Predict using the fitted model
#         y_pred = model.predict(X_test)
        
#         # Calculate the negative mean squared error
#         score = -mean_squared_error(y_test, y_pred)
        
#         basin_scores.append(score)
    
#     # Output the average score for the basin
#     avg_score = np.mean(basin_scores)

#     # Append the mean score to the list
#     cv_scores_mgpr.append(avg_score)
    
#     # Stop early for the sake of demonstration
#     if basin_index > runbasin:
#         break

# # Convert the scores list to a NumPy array for convenience
# cv_scores_mgpr = np.array(cv_scores_mgpr)
# print('Mean CV Scores for MOASMO GPR:', np.nanmean(cv_scores_mgpr))

# # Save the scores to a compressed numpy file
# np.savez_compressed('cv_scores_mgpr.npz', cv_scores_mgpr=cv_scores_mgpr)
