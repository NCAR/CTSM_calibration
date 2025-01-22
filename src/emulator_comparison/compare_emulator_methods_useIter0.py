# Check emulator ability
# Compare the performance of different emulators

# use observed streamflow data to evaluate model outputs
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os, sys, toml

def get_modified_KGE(obs, sim):
    ind = (~np.isnan(obs)) & (~np.isnan(sim))
    obs = obs[ind]
    sim = sim[ind]

    try:
        sd_sim = np.std(sim, ddof=1)
        sd_obs = np.std(obs, ddof=1)
        m_sim = np.mean(sim)
        m_obs = np.mean(obs)
        r = (np.corrcoef(sim, obs))[0, 1]
        relvar = (float(sd_sim)/float(m_sim))/(float(sd_obs)/float(m_obs))
        bias = float(m_sim)/float(m_obs)
        kge = 1.0 - np.sqrt((r-1)**2 + (relvar-1)**2 + (bias-1)**2)
    except:
        kge = np.nan

    return kge

metric_out = np.nan * np.zeros([627, 4])

basin = int(sys.argv[1])
print('basin', basin)

if os.path.isfile(f'CAMELS_4model_emulator_metric_{basin}.npz'):
    sys.exit('file exists')


iterflag = 0

####### load data

nmet = 22  # Or nmet = 19 based on your comment
metrics = np.nan * np.zeros([400, nmet])  # Removed basinnum dimension, assuming single basin focus
params = np.nan * np.zeros([400, 30])  # Removed basinnum dimension, assuming single basin focus


configfile = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/configuration/_level1-{basin}_config_MOASMO.toml'
config = toml.load(configfile)

for trialflag in range(400):

	if config['path_calib'] == 'NA':
		path_MOASMOcalib = f'{path_CTSM_base}_MOASMOcalib'  # Ensure path_CTSM_base is defined somewhere
	else:
		path_MOASMOcalib = config['path_calib']
	path_archive = f'{path_MOASMOcalib}/ctsm_outputs'
	caseflag = f'iter{iterflag}_trial{trialflag}'  # Ensure iterflag is defined or handled as needed
	outfile_metric = f'{path_archive}/{caseflag}/evaluation_many_metrics.csv'

	# load metric
	if os.path.isfile(outfile_metric):
		try:
			df = pd.read_csv(outfile_metric)
			metrics[trialflag, :] = df.values[0]  # Adjusted for the removed basin dimension
		except:
			print('failed reading')

	# load parameter
	dfparam = pd.read_pickle(f'{path_archive}/{caseflag}/paramset_{caseflag}.pkl')
	param = dfparam['Value'].values
	param = np.array([np.mean(i) for i in param])
	
	lower = np.array([i != 'None' for i in dfparam['Lower'].values])
	param = param[lower]
	
	params[trialflag, 0:len(param)] = param

params = params[:, ~np.isnan(params[0,:])]

metnames = df.columns.values
# Adjusting metrics based on new definitions
for m in ['kge', 'cc', 'nse', 'kge_log_q', 'kge_summer', 'kge_winter', 'kge_spring', 'kge_autumn']:
	ind1 = np.where(metnames == m)[0][0]
	metrics[:, ind1] = 1 - metrics[:, ind1]  # Minimize values, adjusted for the removed basin dimension
	metnames[ind1] = '1-' + m


met1 = 'max_mon_abs_err'
met2 = 'mae'
ind1 = np.where(metnames == met1)[0][0]
ind2 = np.where(metnames == met2)[0][0]
metrics_use = metrics[:, [ind1, ind2]]

ind=~np.isnan(metrics_use[:,0]+metrics_use[:,1])
params = params[ind, :]
metrics_use = metrics_use[ind, :]


####### emulator compare

##### RF
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def train_and_evaluate_with_predictions(params, metrics_use):
	print('Processing...')

	cv = KFold(n_splits=5, shuffle=True, random_state=42)
	kge_scores = []  # To store the KGE score for each fold

	# Initialize the Random Forest regressor
	model = RandomForestRegressor(random_state=42)

	for train_idx, test_idx in cv.split(params):
		X_train, X_test = params[train_idx], params[test_idx]
		y_train, y_test = metrics_use[train_idx], metrics_use[test_idx]

		model.fit(X_train, y_train)  # Fit the model on the training data
		y_pred = model.predict(X_test)  # Predict on the test set

		# Calculate KGE for each column (target) and take the mean
		fold_kge_scores = [get_modified_KGE(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
		fold_mean_kge = np.nanmean(fold_kge_scores)  # Compute the mean KGE score for this fold, ignoring any NaN values
		kge_scores.append(fold_mean_kge)

	mean_kge_score = np.mean(kge_scores)  # Calculate the mean KGE score across all folds

	return mean_kge_score

mean_kge_score = train_and_evaluate_with_predictions(params, metrics_use)
print('Mean KGE score:', mean_kge_score)

# np.savez_compressed(f'cv_scores_rf_{basin}.npz', mean_kge_score=mean_kge_score)
metric_out[basin, 0] = mean_kge_score

######## GPR
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
from sklearn.model_selection import KFold
import numpy as np

def train_and_evaluate_with_predictions_gpr(params, metrics_use):
	print('Processing...')

	cv = KFold(n_splits=5, shuffle=True, random_state=42)
	kge_scores = []  # To store the KGE score for each fold

	# Define the kernel with Matern function
	kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
	
	# Initialize the GPR model
	model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True, random_state=42)

	for train_idx, test_idx in cv.split(params):
		X_train, X_test = params[train_idx], params[test_idx]
		y_train, y_test = metrics_use[train_idx], metrics_use[test_idx]

		# Normalize features
		scaler = MinMaxScaler().fit(X_train)
		X_train_scaled = scaler.transform(X_train)
		X_test_scaled = scaler.transform(X_test)

		model.fit(X_train_scaled, y_train)  # Fit the model on the training data
		y_pred = model.predict(X_test_scaled)  # Predict on the test set

		# Calculate KGE for each column (target) and take the mean
		fold_kge_scores = [get_modified_KGE(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
		fold_mean_kge = np.nanmean(fold_kge_scores)  # Compute the mean KGE score for this fold, ignoring any NaN values
		kge_scores.append(fold_mean_kge)

	mean_kge_score = np.mean(kge_scores)  # Calculate the mean KGE score across all folds

	return mean_kge_score

# Execute the function with your data
mean_kge_score_gpr = train_and_evaluate_with_predictions_gpr(params, metrics_use)
print('Mean KGE score for GPR:', mean_kge_score_gpr)

# Optionally, save the results
# np.savez_compressed(f'cv_scores_gpr_{basin}.npz', mean_kge_score=mean_kge_score_gpr)
metric_out[basin, 1] = mean_kge_score_gpr

##### SVM
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

def train_and_evaluate_svm(params, metrics_use):
	scores = []
	predictions = []

	# Split the data
	X_train, X_test, y_train, y_test = train_test_split(params, metrics_use, test_size=0.3, random_state=42)

	# Train one model for each target variable
	for i in range(y_train.shape[1]):
		model = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
		model.fit(X_train, y_train[:, i])
		y_pred = model.predict(X_test)
		predictions.append(y_pred)
		
		# Here you would calculate your score for each target, for example using KGE
		# Assuming get_modified_KGE function is defined
		score = get_modified_KGE(y_test[:, i], y_pred)
		scores.append(score)
	
	mean_score = np.mean(scores)  # You can calculate mean score across all outputs
	predictions = np.array(predictions).T  # Transpose to match the shape of y_test
	
	return mean_score, predictions

# Call the function
mean_score_svm, predictions_svm = train_and_evaluate_svm(params, metrics_use)
print('Mean score for SVM:', mean_score_svm)

metric_out[basin, 2] = mean_score_svm

######## ANN
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
import numpy as np

def train_and_evaluate_ann_with_kge(params, metrics_use):
	print('Processing...')

	cv = KFold(n_splits=5, shuffle=True, random_state=42)
	kge_scores = []

	ann_pipeline = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', 
																solver='adam', alpha=0.001, batch_size='auto', 
																learning_rate='adaptive', learning_rate_init=0.001, 
																max_iter=1000, shuffle=True, random_state=1, 
																tol=0.0001, verbose=False, early_stopping=True, 
																validation_fraction=0.1, n_iter_no_change=10))

	for train_idx, test_idx in cv.split(params):
		X_train, X_test = params[train_idx], params[test_idx]
		y_train, y_test = metrics_use[train_idx], metrics_use[test_idx]

		ann_pipeline.fit(X_train, y_train)  # Fit the model on the training data
		y_pred = ann_pipeline.predict(X_test)  # Predict on the test set
		
		# Calculate KGE for each column (target) and take the mean
		kge_score = np.mean([get_modified_KGE(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])])
		kge_scores.append(kge_score)

	mean_kge_score = np.mean(kge_scores)  # Calculate the mean KGE score across all folds
	return mean_kge_score

mean_kge_score_ann = train_and_evaluate_ann_with_kge(params, metrics_use)
print('Mean KGE score for ANN:', mean_kge_score_ann)
metric_out[basin, 3] = mean_kge_score_ann

np.savez_compressed(f'CAMELS_4model_emulator_metric_{basin}.npz', metric_out=metric_out)