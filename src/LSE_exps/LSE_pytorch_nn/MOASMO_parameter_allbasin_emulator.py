# all functions needed implement all basin emulator
import glob, os, sys, toml, pickle, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from os import path

from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Pool, cpu_count
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

path_MOASMO = '/glade/u/home/guoqiang/CTSM_repos/ctsm_optz/MO-ASMO/src/'
sys.path.append(path_MOASMO)
import gp
import NSGA2

sys.path.append("../../moasmo_test")
from MOASMO_parameters import *

###################################################################################################
# Pytorch neural network train-validation
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network model outside the function
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    # Add a predict method similar to sklearn's
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        self.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            predictions = self.forward(X)
        return predictions.numpy()  # Convert the predictions back to numpy


def train_nn_model_pytorch(x_train_scaled, y_train, x_val_scaled, y_val, n_epochs=1000, patience=10, lr=0.001, model_file="model.pth"):

    # Check if model file exists, and load the model if it does
    if os.path.isfile(model_file):
        print(f"Loading model from {model_file}")
        model = SimpleNN(input_size=x_train_scaled.shape[1])
        model.load_state_dict(torch.load(model_file))
        return model

    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    # y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    x_val_tensor = torch.tensor(x_val_scaled, dtype=torch.float32)
    # y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)


    # Initialize the model, loss function, and optimizer
    model = SimpleNN(input_size=x_train_tensor.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Early stopping parameters
    best_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        predictions = model(x_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_predictions = model(x_val_tensor)
            val_loss = criterion(val_predictions, y_val_tensor)

        print(f'Epoch {epoch+1}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

        # Early stopping logic
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), model_file)  # Save the best model
            print(f"Model saved to {model_file}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('Early stopping triggered.')
            break

    # Load the best model state before returning
    model.load_state_dict(torch.load(model_file))

    # Return the trained model, now with predict method
    return model

###################################################################################################


def read_paramfile(filename):
    var_names = []
    var_value = []

    with open (filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('!') and not line.startswith("'"):
                splits = line.split('|')
                if isinstance(splits[0].strip(), str):
                    var_names.append(splits[0].strip())
                    var_value.append(str_to_float(splits[1].strip()))

    return var_names, var_value
    
def str_to_float(data_str):
    if 'd' in data_str:
        x = data_str.split('d')[0]+'e'+data_str.split('d')[1]
        return float(x)
    else:
        return float(data_str) 

def list_folders(path):
    folder_list = []
    # Iterate over all items in the directory
    for item in os.listdir(path):
        # Check if the item is a directory
        if os.path.isdir(os.path.join(path, item)):
            folder_list.append(item)
    return folder_list

def read_parameters(basin_name, base_settings_dir, valid_params):
    settings_dir = os.path.join(base_settings_dir, basin_name)
    local_param = os.path.join(settings_dir, 'localParamInfo.txt')
    basin_param = os.path.join(settings_dir, 'basinParamInfo.txt')
    
    # Read variable ranges from Local and Basin param files
    local_var_names, local_var_values = read_paramfile(local_param)
    basin_var_names, basin_var_values = read_paramfile(basin_param)
    
    # Combine local and basin parameters
    all_var_names = local_var_names + basin_var_names
    all_var_values = local_var_values + basin_var_values
    
    # Filter parameters based on valid_params
    filtered_vars = {name: value for name, value in zip(all_var_names, all_var_values) if name in valid_params}
    
    return filtered_vars

Basin_list = sorted(list_folders('/glade/campaign/cgd/tss/people/mozhgana/projects/routing/camels'))

###################################################################################################
# data preparation functions

def read_camels_attributes(infile_basin_info, outfile, train_index):
    # outfile = 'camels_627basin_allinfo.pkl'

    if os.path.isfile(outfile):
        print('File exists:', outfile)
        df_att = pd.read_csv(outfile)

    else:

        # Load basin info
        # infile_basin_info= f"/glade/work/guoqiang/CTSM_CAMELS/data_mesh_surf/HillslopeHydrology/CAMELS_level1_basin_info.csv"
        df_basin_info = pd.read_csv(infile_basin_info)
    
        # Load basin attributes for this cluster
        attfiles = [
            "/glade/campaign/ral/hap/common/camels/camels_geol.txt",
            "/glade/campaign/ral/hap/common/camels/camels_hydro.txt",
            "/glade/campaign/ral/hap/common/camels/camels_clim.txt",
            "/glade/campaign/ral/hap/common/camels/camels_loc_topo.txt",
            "/glade/campaign/ral/hap/common/camels/camels_soil.txt",
            "/glade/campaign/ral/hap/common/camels/camels_vege.txt",
        ]
    
        for i in range(len(attfiles)):
            dfi = pd.read_csv(attfiles[i], delimiter=";")
            if i == 0:
                df_att = dfi
            else:
                df_att = pd.merge(df_att, dfi, on="gauge_id")
    
        df_att = df_att.loc[df_att["gauge_id"].isin(df_basin_info["hru_id"].values)]
        df_att.sel_index = np.arange(len(df_att))
        if np.any(df_att["gauge_id"].values != df_basin_info["hru_id"].values):
            sys.exit("Mismatch between att and info ids")
        else:
            print("att and info ids match")
            df_att["hru_id"] = df_basin_info["hru_id"].values

        # select train basins
        df_att = df_att.iloc[train_index]
        
        print('save attributes to file', outfile)
        df_att.to_csv(outfile, index=False)
        
    return df_att


def read_allbasin_defa_params(infile_param_info, outfile_defa_param, Basin_list, train_index):

    # infile_param_info = '/glade/u/home/mozhgana/mywork/model_calibration/src/moasmo_test/param_file_tpl.csv'
    # outfile_defa_param = '/glade/u/home/mozhgana/mywork/model_calibration/src/parameter/camels_627basin_summa_defa_param.csv'

    # Basin_list = list_folders('/glade/campaign/cgd/tss/people/mozhgana/projects/routing/camels')

    df_param_info = pd.read_csv(infile_param_info)
    # load default parameters for each basin
    param_names = df_param_info['Parameter'].values
    base_settings_dir = "/glade/campaign/cgd/tss/people/mozhgana/projects/SUMMA/settings/"

    
    if os.path.isfile(outfile_defa_param):
        df_param_defa = pd.read_csv(outfile_defa_param)
    else:
        # Initialize an empty list to hold data for DataFrame construction
        data = []
        
        for ii in train_index:
            basin_name = Basin_list[ii]  # Get the basin name using the index

            try:
                filtered_vars = read_parameters(basin_name, base_settings_dir, param_names)
                
                # Create a row for the current basin
                row = {'basin_id': basin_name}
                for param in param_names:
                    row[param] = filtered_vars.get(param, None)  # Use None if the parameter is not found
                
                # Append the row to the data list
                data.append(row)
                
            except Exception as e:
                print(f"Error processing basin {basin_name}: {e}")
        
        df_param_defa = pd.DataFrame(data)  

        # Sort the DataFrame by 'basin_id'
        df_param_defa = df_param_defa.sort_values(by='basin_id')        
        df_param_defa.to_csv(outfile_defa_param, index=False)

    return df_param_defa


def load_basin_param_bounds(inpath_moasmo, df_param_defa, file_param_lb, file_param_ub):
    # file_param_lb = 'camels_627basin_summa_all_param_lb.csv.gz'
    # file_param_ub = 'camels_627basin_summa_all_param_ub.csv.gz'


    if os.path.isfile(file_param_lb) and os.path.isfile(file_param_ub):
        df_param_lb = pd.read_csv(file_param_lb, compression='gzip')
        df_param_ub = pd.read_csv(file_param_ub, compression='gzip')
    else:
        param_lb_values = df_param_defa.values[:, 1:].copy()  # Skip 'basin_id'
        param_ub_values = df_param_defa.values[:,1:].copy()

        for id in df_param_defa['basin_id']:
            
            id_str = str(id)

            # Check if the hru_id needs a leading zero (it should be 8 characters long)
            if len(id_str) == 7:
                id_str = '0' + id_str
                
            file = f"{inpath_moasmo}/param_sets/{id_str}/all_default_parameters.csv"
            dfi = pd.read_csv(file)
        
            for j in range(len(dfi['Parameter'].values)):
    
                indj = np.where(df_param_defa.columns.values == dfi['Parameter'].values[j])[0][0] - 1  # Adjust index since 'basin_id' is removed
                param_lb_values[df_param_defa['basin_id'] == id, indj] = dfi['Lower'].values[j]
                param_ub_values[df_param_defa['basin_id'] == id, indj] = dfi['Upper'].values[j]
    

        # Recreate DataFrames without 'basin_id' column
        df_param_lb = pd.DataFrame(param_lb_values, columns=df_param_defa.columns.values[1:])
        df_param_ub = pd.DataFrame(param_ub_values, columns=df_param_defa.columns.values[1:])
        
        df_param_lb.to_csv(file_param_lb, index=False, compression='gzip')
        df_param_ub.to_csv(file_param_ub, index=False, compression='gzip')

    return df_param_lb, df_param_ub

def load_all_basin_params_metrics(inpath_moasmo,infile_param_info, df_param_defa, df_basin_info, tariter, file_all_param, file_all_metric, file_all_basinid, train_index, suffix):
    # file_all_param = 'camels_627basin_summa_all_param.csv.gz'
    # file_all_metric = 'camels_627basin_summa_all_metric.csv.gz'
    # file_all_basinid = 'camels_627basin_summa_all_basinid.csv.gz'

    df_param_info = pd.read_csv(infile_param_info)
    # load default parameters for each basin
    param_names = df_param_info['Parameter'].values

    if os.path.isfile(file_all_param) and os.path.isfile(file_all_metric) and os.path.isfile(file_all_basinid):
        df_param = pd.read_csv(file_all_param, compression='gzip')
        df_metric = pd.read_csv(file_all_metric, compression='gzip')
        df_basinid = pd.read_csv(file_all_basinid, compression='gzip')
    else:
        df_metric = pd.DataFrame()
        df_param = pd.DataFrame()
    
        flag = 0

        df_basin_info=df_basin_info.iloc[train_index]
        
        for i in range(len(train_index)):
            id= df_basin_info['hru_id'].values[i]
            id_str = str(id)
            
            # Check if the hru_id needs a leading zero (it should be 8 characters long)
            if len(id_str) == 7:
                id_str = '0' + id_str
             
            # load param
            file_param = f"{inpath_moasmo}/moasmo_evaluation_{suffix}/{id_str}/iter{tariter}_all_meanparam.csv"

            # Set the number of rows based on iteration (400 for iter0, 100 for iter1)
            num_rows = 400 if tariter == 0 else 100

            if os.path.isfile(file_param):
                df1 = pd.read_csv(file_param)
                df1 = df1.iloc[:, :14]  # Skip heightCanopyBottom
            else:
                print(f"Warning: {file_param} not found, filling with NaN")
                df1 = pd.DataFrame(np.nan, index=range(num_rows), columns=param_names[:14])  # Create NaN-filled DataFrame

            parami = np.tile(df_param_defa.iloc[i].values[1:], (len(df1), 1))

            # parami = np.tile(df_param_defa.loc[df_param_defa.basin_id == id].values[:, 1:], (len(df1), 1))  # Skip basin_id
            for j in range(len(df1.columns)):
                if df1.columns[j] in param_names:
                    indj = np.where(param_names == df1.columns[j])[0][0]
                    
                    parami[:, indj] = df1.values[:, j]

        
            df1 = pd.DataFrame(parami, columns=param_names)

            if len(df_param) == 0:
                df_param = df1
            else:
                df_param = pd.concat([df_param, df1])

            # load metric
            file_metric = f"{inpath_moasmo}/moasmo_evaluation_{suffix}/{id_str}/iter{tariter}_all_metric.csv"

            if os.path.isfile(file_metric):
                df2 = pd.read_csv(file_metric)
            else:
                print(f"Warning: {file_metric} not found, filling with NaN")
                df2 = pd.DataFrame(np.nan, index=range(num_rows), columns=['kge', 'nse', 'rmse', 'max_mon_abs_err', 'abs_err']) 

            # Add columns for basin information
            df2["basin_num"] = flag
            df2["basin_id"] = train_index[i]
        
            # Concatenate the dataframes
            if len(df_metric) == 0:
                df_metric = df2
            else:
                df_metric = pd.concat([df_metric, df2])

            flag += 1
    
        df_param.index = np.arange(len(df_param))
        df_metric.index = np.arange(len(df_metric))

        # Create a dataframe for basin IDs
        df_basinid = df_metric[["basin_num", "basin_id", "basin_name"]]
        
        # Select relevant metrics
        selected_met = ['kge', 'nse', 'rmse', 'max_mon_abs_err', 'abs_err']
        df_metric = df_metric[selected_met]

        # Save the metric and basin ID dataframes to CSV files
        df_param.to_csv(file_all_param, index=False, compression='gzip')
        df_metric.to_csv(file_all_metric, index=False, compression='gzip')
        df_basinid.to_csv(file_all_basinid, index=False, compression='gzip')

    return df_param, df_metric, df_basinid

###################################################################################################
# optimiztion
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination
from multiprocessing import Pool

class MyProblem(Problem):
    def __init__(self, xlb_mean, xub_mean, em_model):
        super().__init__(n_var=len(xlb_mean), n_obj=1, n_constr=0, xl=xlb_mean, xu=xub_mean, elementwise_evaluation=False)
        self.em_model = em_model

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = -self.em_model.predict(x)  # Return negative to maximize


def run_ga_optimization(em_model, xlb_mean, xub_mean, num_runs=100, pop_size=100, num_generations=100, times=3):
    ga_all_solutions = []
    ga_all_outputs = []
    print(f'Run GA {times * num_runs} times and extract the best {num_runs} solutions')

    # Run the model `2 * num_runs` times
    for _ in range(times * num_runs):
        problem = MyProblem(xlb_mean, xub_mean, em_model)
        
        algorithm = GA(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SimulatedBinaryCrossover(prob=0.9, eta=15),
            mutation=PolynomialMutation(eta=20),
            eliminate_duplicates=True
        )
        
        res = minimize(problem,
                       algorithm,
                       termination=get_termination("n_gen", num_generations),
                       verbose=False)
        
        optimized_features = res.X
        max_output = -res.F
        
        ga_all_solutions.append(optimized_features)
        ga_all_outputs.append(np.squeeze(max_output))

    # Convert to numpy arrays for easier sorting
    ga_all_solutions = np.array(ga_all_solutions)
    ga_all_outputs = np.array(ga_all_outputs)

    # Sort the results based on the max_output (in descending order)
    sorted_indices = np.argsort(ga_all_outputs)[::-1]  # Sort in descending order
    
    # Select the top `num_runs` solutions
    top_indices = sorted_indices[:num_runs]
    
    top_solutions = ga_all_solutions[top_indices]
    top_outputs = ga_all_outputs[top_indices]
    
    return top_solutions, top_outputs
    


def run_nsga2_optimization(em_model, xlb_mean, xub_mean, num_runs=100, pop_size=100, num_generations=100):

    # perform optimization using the surrogate model
    pop = 100
    if pop <= num_runs:
        pop = num_runs * 2
    
    gen = 100
    crossover_rate = 0.9
    mu = 20
    mum = 20
    bestx_sm, besty_sm, x_sm, y_sm = NSGA2.optimization(em_model, len(xlb_mean), 2, xlb_mean, xub_mean, pop, gen, crossover_rate, mu, mum)
    D = NSGA2.crowding_distance(besty_sm)
        
    idxr = D.argsort()[::-1][:num_runs]
    nsga2_all_solutions = bestx_sm[idxr, :]
    nsga2_all_outputs = besty_sm[idxr, :]

    return np.array(nsga2_all_solutions), np.array(nsga2_all_outputs)


def generate_param_files(basin_id, em_model, xlb_mean, xub_mean, param_names, inpath_moasmo, numruns, iterend, suffix, num_objfunc=1, times=3):
    
    basin_id = str(basin_id)

    # Check if the hru_id needs a leading zero (it should be 8 characters long)
    if len(basin_id) == 7:
        basin_id = '0' + basin_id
    
    outpath = f'{inpath_moasmo}/param_sets_{suffix}/{basin_id}/'
    os.makedirs(outpath, exist_ok=True)

    if num_objfunc == 1:
        outfile_ga = f'{outpath}/ga_output_iter{iterend}.npz'
        if os.path.isfile(outfile_ga):
            dtmp = np.load(outfile_ga)
            ga_all_solutions = dtmp['ga_all_solutions']
            ga_all_outputs = dtmp['ga_all_outputs']
        else:
            ga_all_solutions, ga_all_outputs = run_ga_optimization(em_model, xlb_mean, xub_mean, num_runs=numruns, pop_size=100, num_generations=100, times=times)
            np.savez_compressed(outfile_ga, ga_all_solutions=ga_all_solutions, ga_all_outputs=ga_all_outputs)
    
        final_solutions_array = np.array(ga_all_solutions)

    elif num_objfunc == 2:
        outfile_nsga2 = f'{outpath}/nsga2_output_iter{iterend}.npz'
        if os.path.isfile(outfile_nsga2):
            dtmp = np.load(outfile_nsga2)
            nsga2_all_solutions = dtmp['nsga2_all_solutions']
            nsga2_all_outputs = dtmp['nsga2_all_outputs']
        else:
            nsga2_all_solutions, nsga2_all_outputs = run_nsga2_optimization(em_model, xlb_mean, xub_mean, num_runs=numruns, pop_size=100, num_generations=100)
            np.savez_compressed(outfile_nsga2, nsga2_all_solutions=nsga2_all_solutions, nsga2_all_outputs=nsga2_all_outputs)
    
        final_solutions_array = np.array(nsga2_all_solutions)
    
    df_info = pd.read_csv(f'{inpath_moasmo}/param_sets/{basin_id}/paramset_iter0_trial0.csv').loc[:13,:]
    df_info = df_info.loc[df_info['Value'] != 'None']
    
    df_info['Factor'] = np.nan
    df_info['Value'] = np.nan

    indexp = [np.where(param_names == p)[0][0] for p in df_info['Parameter'].values if p in param_names]


    # Read heightCanopyBottom from the file
    paramfile = f'/glade/campaign/cgd/tss/people/mozhgana/projects/SUMMA/settings/{basin_id}/trialParams.camels.nc'
    dataset = Dataset(paramfile)
    height_canopy_bottom_value = dataset.variables['heightCanopyBottom'][:][0]

    for i in range(final_solutions_array.shape[0]):
        outfile = f'{outpath}/paramset_iter{iterend}_trial{i}.csv'
        if os.path.isfile(outfile):
            continue

        dfi = df_info.copy()
        dfi['Value'] = final_solutions_array[i, indexp]

        # Ensure heightCanopyTop is greater than heightCanopyBottom
        if 'heightCanopyTop' in dfi['Parameter'].values:
            height_canopy_top_value = dfi[dfi['Parameter'] == 'heightCanopyTop']['Value'].values[0]

            if height_canopy_top_value <= height_canopy_bottom_value:
                # Adjust heightCanopyTop to be greater than heightCanopyBottom
                height_canopy_top_value = height_canopy_bottom_value + 0.5  # Add a buffer (e.g., 1.0)
                dfi.loc[dfi['Parameter'] == 'heightCanopyTop', 'Value'] = height_canopy_top_value
                print(f"Adjusted heightCanopyTop to {height_canopy_top_value} for basin {basin_id}")
        
        dfi.to_csv(outfile, index=False)

    print('finish basin', basin_id)


def process_basin(args):
    basin_id = None  # Initialize basin_id before the try block
    try:
        tarbasin, tarbasin_id, df_basinid, df_param_lb, df_param_ub, x_all, df_input, y_all, param_names, inputnames, em_model, inpath_moasmo, numruns, iterend, suffix, times = args
        index = np.where(df_basinid["basin_num"].values == tarbasin_id)[0]
        # print(df_basinid.shape, x_all.shape, tarbasin, tarbasin_id)

        attnames = [i for i in inputnames if i not in param_names]
        
        param_lb_mean = df_param_lb.values[tarbasin_id, :]
        param_ub_mean = df_param_ub.values[tarbasin_id, :]
        
        attrvalues = x_all[index[0], len(param_lb_mean):]  # same with df_input
                          
        if np.any(attrvalues != df_input.iloc[index[0]][attnames].values):
            print('Warning! att problem')

        xlb_mean = np.hstack([param_lb_mean, attrvalues])
        xub_mean = np.hstack([param_ub_mean, attrvalues])

        if y_all.ndim == 1 or y_all.shape[1] == 1:
            num_objfunc = 1
        elif y_all.shape[1] == 2:
            num_objfunc = 2
        else:
            sys.exit('Error: y_all.shape[1] is larger than 2.')
        print('num_objfunc:',num_objfunc)

        basin_id= str(int(df_basinid.loc[df_basinid['basin_id'] == tarbasin, 'basin_name'].values[0]))
        if len(basin_id) == 7:
            basin_id = '0' + basin_id

        print(f"Processing basin: {basin_id}, Index: {tarbasin}")
        
        generate_param_files(basin_id, em_model, xlb_mean, xub_mean, param_names, inpath_moasmo, numruns, iterend, suffix, num_objfunc, times)
    except Exception as e:
        print(f"Error processing basin {basin_id}-{tarbasin}: {e}")


def parallel_process_basins(df_basinid, df_param_lb, df_param_ub, x_all, df_input, y_all, param_names, inputnames, em_model, inpath_moasmo, ncpus, numruns, iterend, basin_index, suffix):
    times = 3
    args = [(basin_index[tarbasin_id], tarbasin_id, df_basinid, df_param_lb, df_param_ub, x_all, df_input, y_all, param_names, inputnames, em_model, inpath_moasmo, numruns, iterend, suffix, times) for tarbasin_id in range(len(basin_index))]
    
    with Pool(processes=ncpus) as pool:
        pool.map(process_basin, args)


def process_basin_predictunseen(args):
    try:
        # Unpack arguments
        tarbasin, tarbasin_id, df_basinid, em_model, xlb_mean, xub_mean, param_names, inpath_moasmo, numruns, iterend, suffix, num_objfunc, times = args

        basin_id= str(int(df_basinid.loc[df_basinid['basin_id'] == tarbasin, 'basin_name'].values[0]))
        if len(basin_id) == 7:
            basin_id = '0' + basin_id

        print(f"Processing basin: {basin_id}, Index: {tarbasin}")
       
        # Call the function to generate parameter files        
        generate_param_files(basin_id, em_model, xlb_mean, xub_mean, param_names, inpath_moasmo, numruns, iterend, suffix, num_objfunc, times)

    except Exception as e:
        print(f"Error processing basin {basin_id}-{tarbasin}: {e}")

def parallel_process_basins_predictunseen(df_basinid, xlb_mean_all, xub_mean_all, param_names, em_model, inpath_moasmo, ncpus, numruns, iterend, basin_index, suffix, num_objfunc=1):
    # Prepare the list of arguments to pass to the function
    times = 100
    args = [(basin_index[tarbasin_id], tarbasin_id, df_basinid, em_model, xlb_mean_all[tarbasin_id, :], xub_mean_all[tarbasin_id, :], 
             param_names, inpath_moasmo, numruns, iterend, suffix, num_objfunc,times) 
            for tarbasin_id in range(len(basin_index))]

    # Run in parallel using Pool
    with Pool(processes=ncpus) as pool:
        pool.map(process_basin_predictunseen, args)


###################################################################################################
# functions with normalization

def generate_param_files_norm(basin_id, em_model, xlb_mean, xub_mean, param_names, inpath_moasmo, numruns, iterend, suffix, num_objfunc=1, normdict={}, times=3):
    
    basin_id = str(basin_id)

    # Check if the hru_id needs a leading zero (it should be 8 characters long)
    if len(basin_id) == 7:
        basin_id = '0' + basin_id
    
    outpath = f'{inpath_moasmo}/param_sets_{suffix}/{basin_id}/'
    os.makedirs(outpath, exist_ok=True)

    if num_objfunc == 1:
        outfile_ga = f'{outpath}/ga_output_iter{iterend}.npz'
        if os.path.isfile(outfile_ga):
            dtmp = np.load(outfile_ga)
            ga_all_solutions = dtmp['ga_all_solutions']
            ga_all_outputs = dtmp['ga_all_outputs']
        else:
            ga_all_solutions, ga_all_outputs = run_ga_optimization(em_model, xlb_mean, xub_mean, num_runs=numruns, pop_size=100, num_generations=100, times=times)
            np.savez_compressed(outfile_ga, ga_all_solutions=ga_all_solutions, ga_all_outputs=ga_all_outputs)
    
        final_solutions_array = np.array(ga_all_solutions)

    elif num_objfunc == 2:
        outfile_nsga2 = f'{outpath}/nsga2_output_iter{iterend}.npz'
        if os.path.isfile(outfile_nsga2):
            dtmp = np.load(outfile_nsga2)
            nsga2_all_solutions = dtmp['nsga2_all_solutions']
            nsga2_all_outputs = dtmp['nsga2_all_outputs']
        else:
            nsga2_all_solutions, nsga2_all_outputs = run_nsga2_optimization(em_model, xlb_mean, xub_mean, num_runs=numruns, pop_size=100, num_generations=100)
            np.savez_compressed(outfile_nsga2, nsga2_all_solutions=nsga2_all_solutions, nsga2_all_outputs=nsga2_all_outputs)
    
        final_solutions_array = np.array(nsga2_all_solutions)
    
    df_info = pd.read_csv(f'{inpath_moasmo}/param_sets/{basin_id}/paramset_iter0_trial0.csv').loc[:13,:]
    df_info = df_info.loc[df_info['Value'] != 'None']
    
    df_info['Factor'] = np.nan
    df_info['Value'] = np.nan

    indexp = [np.where(param_names == p)[0][0] for p in df_info['Parameter'].values if p in param_names]

    # reverse normalization
    if 'method' in normdict:
        if normdict['method'] == 'z-score':
            print('reverse normalization')
            final_solutions_array = final_solutions_array*normdict['std'] + normdict['mean']


    # Read heightCanopyBottom from the file
    paramfile = f'/glade/campaign/cgd/tss/people/mozhgana/projects/SUMMA/settings/{basin_id}/trialParams.camels.nc'
    dataset = Dataset(paramfile)
    height_canopy_bottom_value = dataset.variables['heightCanopyBottom'][:][0]

    for i in range(final_solutions_array.shape[0]):
        outfile = f'{outpath}/paramset_iter{iterend}_trial{i}.csv'
        if os.path.isfile(outfile):
            continue

        dfi = df_info.copy()
        dfi['Value'] = final_solutions_array[i, indexp]

        # Ensure heightCanopyTop is greater than heightCanopyBottom
        if 'heightCanopyTop' in dfi['Parameter'].values:
            height_canopy_top_value = dfi[dfi['Parameter'] == 'heightCanopyTop']['Value'].values[0]

            if height_canopy_top_value <= height_canopy_bottom_value:
                # Adjust heightCanopyTop to be greater than heightCanopyBottom
                height_canopy_top_value = height_canopy_bottom_value + 0.5  # Add a buffer (e.g., 1.0)
                dfi.loc[dfi['Parameter'] == 'heightCanopyTop', 'Value'] = height_canopy_top_value
                print(f"Adjusted heightCanopyTop to {height_canopy_top_value} for basin {basin_id}")
        
        dfi.to_csv(outfile, index=False)

    print('finish basin', basin_id)


def process_basin_norm(args):
    basin_id = None  # Initialize basin_id before the try block
    try:
        tarbasin, tarbasin_id, df_basinid, df_param_lb, df_param_ub, x_all, df_input, y_all, param_names, inputnames, em_model, inpath_moasmo, numruns, iterend, suffix, normdict, times = args
        index = np.where(df_basinid["basin_num"].values == tarbasin_id)[0]
        # print(df_basinid.shape, x_all.shape, tarbasin, tarbasin_id)

        attnames = [i for i in inputnames if i not in param_names]
        
        param_lb_mean = df_param_lb.values[tarbasin_id, :]
        param_ub_mean = df_param_ub.values[tarbasin_id, :]
        
        attrvalues = x_all[index[0], len(param_lb_mean):]  # same with df_input
                          
        if np.any(attrvalues != df_input.iloc[index[0]][attnames].values):
            print('Warning! att problem')

        xlb_mean = np.hstack([param_lb_mean, attrvalues])
        xub_mean = np.hstack([param_ub_mean, attrvalues])

        if y_all.ndim == 1 or y_all.shape[1] == 1:
            num_objfunc = 1
        elif y_all.shape[1] == 2:
            num_objfunc = 2
        else:
            sys.exit('Error: y_all.shape[1] is larger than 2.')
        print('num_objfunc:',num_objfunc)

        if normdict['method'] == 'z-score':
            xlb_mean_scaled = (xlb_mean - normdict['mean']) / normdict['std']
            xub_mean_scaled = (xub_mean - normdict['mean']) / normdict['std']
        else:
            sys.exit('empty normdict')

        

        basin_id= str(int(df_basinid.loc[df_basinid['basin_id'] == tarbasin, 'basin_name'].values[0]))
        if len(basin_id) == 7:
            basin_id = '0' + basin_id

        print(f"Processing basin: {basin_id}, Index: {tarbasin}")
        
        generate_param_files_norm(basin_id, em_model, xlb_mean_scaled, xub_mean_scaled, param_names, inpath_moasmo, numruns, iterend, suffix, num_objfunc,normdict, times)
    except Exception as e:
        print(f"Error processing basin {basin_id}-{tarbasin}: {e}")


def parallel_process_basins_norm(df_basinid, df_param_lb, df_param_ub, x_all, df_input, y_all, param_names, inputnames, em_model, inpath_moasmo, ncpus, numruns, iterend, basin_index, suffix, normdict):
    times = 3
    args = [(basin_index[tarbasin_id], tarbasin_id, df_basinid, df_param_lb, df_param_ub, x_all, df_input, y_all, param_names, inputnames, em_model, inpath_moasmo, numruns, iterend, suffix,normdict, times) for tarbasin_id in range(len(basin_index))]
    
    with Pool(processes=ncpus) as pool:
        pool.map(process_basin_norm, args)


def process_basin_predictunseen_norm(args):
    try:
        # Unpack arguments
        tarbasin, tarbasin_id, df_basinid, em_model, xlb_mean, xub_mean, param_names, inpath_moasmo, numruns, iterend, suffix, num_objfunc, normdict, times = args

        basin_id= str(int(df_basinid.loc[df_basinid['basin_id'] == tarbasin, 'basin_name'].values[0]))
        if len(basin_id) == 7:
            basin_id = '0' + basin_id

        print(f"Processing basin: {basin_id}, Index: {tarbasin}")
       
        # Call the function to generate parameter files        
        generate_param_files_norm(basin_id, em_model, xlb_mean, xub_mean, param_names, inpath_moasmo, numruns, iterend, suffix, num_objfunc, normdict, times)

    except Exception as e:
        print(f"Error processing basin {basin_id}-{tarbasin}: {e}")

def parallel_process_basins_predictunseen_norm(df_basinid, xlb_mean_all, xub_mean_all, param_names, em_model, inpath_moasmo, ncpus, numruns, iterend, basin_index, suffix, num_objfunc=1, normdict={}):
    # Prepare the list of arguments to pass to the function
    times = 100
    args = [(basin_index[tarbasin_id], tarbasin_id, df_basinid, em_model, xlb_mean_all[tarbasin_id, :], xub_mean_all[tarbasin_id, :], 
             param_names, inpath_moasmo, numruns, iterend, suffix, num_objfunc, normdict, times) 
            for tarbasin_id in range(len(basin_index))]

    # Run in parallel using Pool
    with Pool(processes=ncpus) as pool:
        pool.map(process_basin_predictunseen_norm, args)


###################################################################################################

###################################################################################################

def allbasin_emulator_train_and_optimize(infile_basin_info, infile_param_info, infile_attr_foruse, inpath_moasmo, outpathname, iterend, ncpus, train_index, suffix, numruns=100, objfunc='normKGE'):


    outpath = f"{inpath_moasmo}/{outpathname}"
    os.makedirs(outpath, exist_ok=True)
    
    # Load data: same for all iterations
    df_basin_info = pd.read_csv(infile_basin_info)
    df_basin_info = df_basin_info.iloc[train_index]
    df_basin_info.index = np.arange(len(df_basin_info))
    
    file_defa_param = f'{outpath}/camels_{len(train_index)}basin_summa_defa_param.csv'
    df_param_defa = read_allbasin_defa_params(infile_param_info, file_defa_param, Basin_list, train_index)
    df_param_defa['basin_id'] = df_param_defa['basin_id'].astype(str).apply(lambda x: x.zfill(8))


    file_param_lb = f'{outpath}/camels_{len(train_index)}basin_summa_all_param_lb.gz'
    file_param_ub = f'{outpath}/camels_{len(train_index)}basin_summa_all_param_ub.gz'
    df_param_lb, df_param_ub = load_basin_param_bounds(inpath_moasmo, df_param_defa, file_param_lb, file_param_ub)

    file_camels_attribute = f'{outpath}/camels_{len(train_index)}basin_attribute.pkl'
    df_att = read_camels_attributes(infile_basin_info, file_camels_attribute, train_index)
    
    df_att_foruse = pd.read_csv(infile_attr_foruse)
    useattrs = list(df_att_foruse[df_att_foruse['att_Xie2021'].values]['Attribute_text'].values)
    print("The number of attributes used:", len(useattrs))
    print(useattrs)

    # Load data: outputs from each iteration
    for iter in range(0, iterend):
        file_all_param = f'{outpath}/camels_{len(train_index)}basin_summa_all_param_iter{iter}.gz'
        file_all_metric = f'{outpath}/camels_{len(train_index)}basin_summa_all_metric_iter{iter}.gz'
        file_all_basinid = f'{outpath}/camels_{len(train_index)}basin_summa_all_basinid_iter{iter}.gz'
        
        df_param_i, df_metric_i, df_basinid_i = load_all_basin_params_metrics(inpath_moasmo,infile_param_info, df_param_defa, df_basin_info, iter, file_all_param, file_all_metric, file_all_basinid, suffix)
        
        df_basinid_i['iter'] = iter
        
        if iter == 0:
            df_param = df_param_i
            df_metric = df_metric_i
            df_basinid = df_basinid_i
        else:
            df_param = pd.concat([df_param, df_param_i])
            df_metric = pd.concat([df_metric, df_metric_i])
            df_basinid = pd.concat([df_basinid, df_basinid_i])

    df_param = df_param.apply(pd.to_numeric, errors='coerce')

    
    df_param.index = np.arange(len(df_param))
    df_metric.index = np.arange(len(df_metric))
    df_basinid.index = np.arange(len(df_basinid))

    index = np.isnan(np.sum(df_metric.values, axis=1) + np.sum(df_param.values, axis=1))
    df_param = df_param[~index]
    df_metric = df_metric[~index]
    df_basinid = df_basinid[~index]
    
    df_param.index = np.arange(len(df_param))
    df_metric.index = np.arange(len(df_metric))
    df_basinid.index = np.arange(len(df_basinid))
    
    print('Number of nan samples:', np.sum(index))
    print("Number of original parameter sets:", len(index))
    print("Number of final parameter sets:", len(df_param))

    # Prepare model input and output
    df_input = df_param.copy()
    df_input["hru_id"] = df_basinid["basin_name"]
    df_input = df_input.merge(df_att[useattrs + ["hru_id"]], on="hru_id", how="left")
    df_input = df_input.drop(["hru_id"], axis=1)
    
    inputnames = list(df_param.columns) + useattrs

    # One-hot encoding for categorical attributes
    for att in useattrs:
        if df_input[att].dtype == "object":
            print('Convert', att, 'to one-hot encoding')
            enc = OneHotEncoder(sparse=False)
            enc.fit(df_input[[att]])
            encnames = [att + "_" + str(i) for i in range(len(enc.categories_[0]))]
            print('New columns:', encnames)
            df_enc = pd.DataFrame(enc.transform(df_input[[att]]), columns=encnames)
            df_input = pd.concat([df_input, df_enc], axis=1)
            df_input = df_input.drop([att], axis=1)
            inputnames = [i for i in inputnames if i != att] + encnames

    x_all = df_input[inputnames].values.copy()
    print("Input shape:", x_all.shape)


    if objfunc == 'normKGE':
        print('Use normalized KGE as output')
        df_output = df_metric.copy()
        y_all = df_output[["kge"]].values.copy()
        y_all = y_all / (2 - y_all)  # Normalize KGE
    
        # Train a random forest emulator
        outfile = f'{outpath}/RF_emulator_for_iter{iterend}.pkl'
        if os.path.isfile(outfile):
            with open(outfile, 'rb') as file:
                em_model = pickle.load(file)
        else:
            modelconfig = {'n_estimators': 100, 'random_state': 42, 'max_depth': 40}
            em_model = RandomForestRegressor(**modelconfig, n_jobs=ncpus)
            em_model.fit(x_all, y_all)
            with open(outfile, 'wb') as file:
                pickle.dump(em_model, file)

    elif objfunc == 'norm2err':
        print('Use normalized mae and mmae as output')
        # normalization is performed for each basin
        df_output = df_metric.copy()
        metvalues = df_output[['mae', 'max_mon_abs_err']].values.copy()
        y_all = np.nan * metvalues
        for i in range(len(df_basin_info)):
            indi = df_basinid['basin_id'].values==i
            di = metvalues[indi, :]
            di = (di - np.nanmin(di,axis=0) ) / (np.nanmax(di,axis=0) - np.nanmin(di,axis=0) )
            y_all[indi, :] = di
        
        # Train a random forest emulator
        outfile = f'{outpath}/RF_emulator_2errOBJfunc_for_iter{iterend}.pkl'
        if os.path.isfile(outfile):
            with open(outfile, 'rb') as file:
                em_model = pickle.load(file)
        else:
            modelconfig = {'n_estimators': 200, 'random_state': 42, 'max_depth': 40}
            em_model = RandomForestRegressor(**modelconfig, n_jobs=ncpus)
            em_model.fit(x_all, y_all)
            with open(outfile, 'wb') as file:
                pickle.dump(em_model, file)
            
    
    df_param_info = pd.read_csv(infile_param_info)
    # load default parameters for each basin
    param_names = df_param_info['Parameter'].values
    
    parallel_process_basins(df_basinid, df_param_lb, df_param_ub, x_all, df_input, y_all, param_names, inputnames, em_model, inpath_moasmo, ncpus, numruns, iterend, train_index, suffix)


def allbasin_emulator_CV_traintest_and_optimize(infile_basin_info, infile_param_info, infile_attr_foruse, inpath_moasmo, outpathname, iterend, ncpus, suffix, numruns=100, objfunc='normKGE'):

    cv_num = 5
    
    outpath = f"{inpath_moasmo}/{outpathname}"
    os.makedirs(outpath, exist_ok=True)
    
    # Load data: same for all iterations
    df_basin_info = pd.read_csv(infile_basin_info)
    df_basin_info.index = np.arange(len(df_basin_info))
    all_index = np.arange(len(df_basin_info))
    
    # divide into train/test index
    outfile = f'{outpath}/train_test_CV_indices.npz'
    if os.path.isfile(outfile):
        dtmp = np.load(outfile, allow_pickle=True)
        train_indices, test_indices = dtmp['train_indices'], dtmp['test_indices']
    else:
        train_indices = []
        test_indices = []
        for i in range(cv_num):
            ind1 = all_index[i::cv_num]
            ind2 = np.setdiff1d(all_index, ind1)
            test_indices.append(ind1)
            train_indices.append(ind2)
            print('test index', ind1)
            print('train index', ind2)
        np.savez_compressed(outfile, test_indices=np.array(test_indices, dtype=object), train_indices=np.array(train_indices, dtype=object))
    
    # information for all basins
    df_param_info = pd.read_csv(infile_param_info)
    
    file_defa_param = f'{outpath}/camels_{len(all_index)}basin_summa_defa_param.csv'
    df_param_defa = read_allbasin_defa_params(infile_param_info, file_defa_param, Basin_list, all_index)    
    df_param_defa['basin_id'] = df_param_defa['basin_id'].astype(str).apply(lambda x: x.zfill(8))
    
    file_param_lb = f'{outpath}/camels_{len(all_index)}basin_summa_all_param_lb.gz'
    file_param_ub = f'{outpath}/camels_{len(all_index)}basin_summa_all_param_ub.gz'
    df_param_lb, df_param_ub = load_basin_param_bounds(inpath_moasmo, df_param_defa, file_param_lb, file_param_ub)
    
    
    file_camels_attribute = f'{outpath}/camels_{len(all_index)}basin_attribute.pkl'
    df_att = read_camels_attributes(infile_basin_info, file_camels_attribute, all_index)
    
    
    df_att_foruse = pd.read_csv(infile_attr_foruse)
    useattrs = list(df_att_foruse[df_att_foruse['att_Xie2021'].values]['Attribute_text'].values)
    print("The number of attributes used:", len(useattrs))
    print(useattrs)
    
    # Load data: outputs from each iteration
    for iter in range(0, iterend):
    
        file_all_param = f'{outpath}/camels_{len(all_index)}basin_summa_all_param_iter{iter}.gz'
        file_all_metric = f'{outpath}/camels_{len(all_index)}basin_summa_all_metric_iter{iter}.gz'
        file_all_basinid = f'{outpath}/camels_{len(all_index)}basin_summa_all_basinid_iter{iter}.gz'
    
        df_param_i, df_metric_i, df_basinid_i = load_all_basin_params_metrics(inpath_moasmo, infile_param_info, df_param_defa, df_basin_info, iter, file_all_param, file_all_metric, file_all_basinid, suffix)
    
        
        df_basinid_i['iter'] = iter
        
        if iter == 0:
            df_param = df_param_i
            df_metric = df_metric_i
            df_basinid = df_basinid_i
        else:
            df_param = pd.concat([df_param, df_param_i])
            df_metric = pd.concat([df_metric, df_metric_i])
            df_basinid = pd.concat([df_basinid, df_basinid_i])
    
    df_param = df_param.apply(pd.to_numeric, errors='coerce')
    
    df_param.index = np.arange(len(df_param))
    df_metric.index = np.arange(len(df_metric))
    df_basinid.index = np.arange(len(df_basinid))
    
    index = np.isnan(np.sum(df_metric.values, axis=1) + np.sum(df_param.values, axis=1))
    df_param = df_param[~index]
    df_metric = df_metric[~index]
    df_basinid = df_basinid[~index]
    
    df_param.index = np.arange(len(df_param))
    df_metric.index = np.arange(len(df_metric))
    df_basinid.index = np.arange(len(df_basinid))
    
    print('Number of nan samples:', np.sum(index))
    print("Number of original parameter sets:", len(index))
    print("Number of final parameter sets:", len(df_param))
    
    # Prepare model input and output
    df_input = df_param.copy()
    df_input["hru_id"] = df_basinid["basin_name"]
    df_input = df_input.merge(df_att[useattrs + ["hru_id"]], on="hru_id", how="left")
    df_input = df_input.drop(["hru_id"], axis=1)
    
    inputnames = list(df_param.columns) + useattrs
    
    # One-hot encoding for categorical attributes
    for att in useattrs:
            if df_input[att].dtype == "object":
                print('Convert', att, 'to one-hot encoding')
                enc = OneHotEncoder(sparse=False)
                enc.fit(df_input[[att]])
                encnames = [att + "_" + str(i) for i in range(len(enc.categories_[0]))]
                print('New columns:', encnames)
                df_enc = pd.DataFrame(enc.transform(df_input[[att]]), columns=encnames)
                df_input = pd.concat([df_input, df_enc], axis=1)
                df_input = df_input.drop([att], axis=1)
                inputnames = [i for i in inputnames if i != att] + encnames
    
    
    x_all = df_input[inputnames].values.copy()
    print("Input shape:", x_all.shape)
    
    
    for cvind in range(len(train_indices)):
        
        train_index = train_indices[cvind]
        test_index = test_indices[cvind]
    
        train_index_allsample = df_basinid['basin_num'].isin(train_index).values
        test_index_allsample = df_basinid['basin_num'].isin(test_index).values
    
        # Debugging prints to verify the number of selected samples
        print('Train/test model')
        print('Train index:', train_index)
        print('Test index:', test_index)
        print(f"Number of training samples: {np.sum(train_index_allsample)}")
        print(f"Number of testing samples: {np.sum(test_index_allsample)}")
    
        if objfunc == 'normKGE':
            print('Use normalized KGE as output')
            df_output = df_metric.copy()
            y_all = df_output[["kge"]].values.copy()
            y_all = y_all / (2 - y_all)  # Normalize KGE
        
            # Train a random forest emulator
            outfile = f'{outpath}/RF_emulator_for_iter{iterend}_CVFold{cvind}.pkl'
            outfile_eval = f'{outpath}/RF_emulator_for_iter{iterend}_CVFold{cvind}_eval.npz'
            if os.path.isfile(outfile):
                with open(outfile, 'rb') as file:
                    em_model = pickle.load(file)
            else:
                modelconfig = {'n_estimators': 100, 'random_state': 42, 'max_depth': 40}
                em_model = RandomForestRegressor(**modelconfig, n_jobs=ncpus)
                em_model.fit(x_all[train_index_allsample], y_all[train_index_allsample])
                with open(outfile, 'wb') as file:
                    pickle.dump(em_model, file)
    
                # Evaluate the model on testing samples
                y_test_pred = em_model.predict(x_all[test_index_allsample])
                np.savez_compressed(outfile_eval, 
                                    y_test_pred=y_test_pred, 
                                    y_test=y_all[test_index_allsample], 
                                    basin_id=df_basinid['basin_id'].values[test_index_allsample]) 
    
        elif objfunc == 'norm2err':
            print('Use normalized MAE and MMAE as output')
            df_output = df_metric.copy()
            metvalues = df_output[['mae', 'max_mon_abs_err']].values.copy()
            y_all = np.nan * metvalues
            for i in range(len(df_basin_info)):
                indi = df_basinid['basin_num'].values == i
                di = metvalues[indi, :]
                di = (di - np.nanmin(di, axis=0)) / (np.nanmax(di, axis=0) - np.nanmin(di, axis=0))
                y_all[indi, :] = di
            
            # Train a random forest emulator
            outfile = f'{outpath}/RF_emulator_2errOBJfunc_for_iter{iterend}_CVFold{cvind}.pkl'
            outfile_eval = f'{outpath}/RF_emulator_2errOBJfunc_for_iter{iterend}_CVFold{cvind}_eval.npz'
            if os.path.isfile(outfile):
                with open(outfile, 'rb') as file:
                    em_model = pickle.load(file)
            else:
                modelconfig = {'n_estimators': 200, 'random_state': 42, 'max_depth': 40}
                em_model = RandomForestRegressor(**modelconfig, n_jobs=ncpus)
                em_model.fit(x_all[train_index_allsample], y_all[train_index_allsample])
                with open(outfile, 'wb') as file:
                    pickle.dump(em_model, file)
    
                # Evaluate the model on testing samples
                y_test_pred = em_model.predict(x_all[test_index_allsample])
                np.savez_compressed(outfile_eval, 
                                    y_test_pred=y_test_pred, 
                                    y_test=y_all[test_index_allsample], 
                                    basin_id=df_basinid['basin_id'].values[test_index_allsample])
    
        # Pass the necessary parameters for parallel processing of basins
        param_names = df_param_info['Parameter'].values
        parallel_process_basins(
            df_basinid[test_index_allsample], 
            df_param_lb.iloc[test_index], 
            df_param_ub.iloc[test_index], 
            x_all[test_index_allsample], 
            df_input[test_index_allsample], 
            y_all[test_index_allsample], 
            param_names, 
            inputnames, 
            em_model, 
            inpath_moasmo, 
            ncpus, 
            numruns, 
            iterend, 
            test_index, 
            suffix
        )



    
def allbasin_emulator_CV_traintest_and_optimize_2(infile_basin_info, infile_param_info, infile_attr_foruse,
                                                  inpath_moasmo, outpathname, iterend, ncpus, suffix,
                                                  train_index, numruns=100, objfunc='normKGE'):
    # implementation - 2
    # (1) iter-0 simulations
    # (2) divide basins into 5 folds.
    # (3) For each fold, using 80% for training and 20% for testing
    # Iterative training based on the 80% basins. In this process, the 20% basins are never used. This could lead to iter-1, iter-2, …, iter-x, until saturated calibration. The implementation is the same with a typical joint emulator
    # For trained emulator in each iteration (iter-1, iter-2, iter-3, …, iter-x), it can be used to predict parameters in 20% testing basins. The number of predicted parameters can range from 1 to inf. This number can be smaller than 100 since we don’t need 100 optimized parameter sets in unseen basins. In practice, simply using iter-x is also fine
    # (3) is repeated five times to get validation results in each testing basins. (edited)

    # to implement that in a CV way, divide all basins into five folds. For each fold, name the suffix as "CV1", "CV2", "CV3", ... "CV5".
    # then, using train_index (80% stations) and test_index (20% stations)  as inputs
    # note other inputs such as infile_basin_info still contain all basins and should correspond to the index from train_index and test_index
    # train_index and test_index needs to be generated outside this function, maybe following the below method
    # train_indices = []
    # test_indices = []
    # for i in range(cv_num):
    #     ind1 = all_index[i::cv_num]
    #     ind2 = np.setdiff1d(all_index, ind1)
    #     test_indices.append(ind1)
    #     train_indices.append(ind2)

    suffix_defa_source = 'LSEnormKGE'

    outpath = f"{inpath_moasmo}/{outpathname}"
    os.makedirs(outpath, exist_ok=True)

    # Load data: same for all iterations
    df_basin_info = pd.read_csv(infile_basin_info)
    df_basin_info.index = np.arange(len(df_basin_info))
    all_index = np.arange(len(df_basin_info))

    test_index = np.setdiff1d(all_index, train_index)

    # information for all basins
    df_param_info = pd.read_csv(infile_param_info)

    file_defa_param = f'{outpath}/camels_summa_defa_param_train_{suffix}.csv'
    df_param_defa_train = read_allbasin_defa_params(infile_param_info, file_defa_param, Basin_list, train_index)

    file_defa_param = f'{outpath}/camels_summa_defa_param_test_{suffix}.csv'
    df_param_defa_test = read_allbasin_defa_params(infile_param_info, file_defa_param, Basin_list, test_index)

    file_param_lb = f'{outpath}/camels_summa_all_param_lb_train_{suffix}.gz'
    file_param_ub = f'{outpath}/camels_summa_all_param_ub_train_{suffix}.gz'
    
    df_param_lb_train, df_param_ub_train = load_basin_param_bounds(inpath_moasmo, df_param_defa_train, file_param_lb, file_param_ub)

    file_param_lb = f'{outpath}/camels_summa_all_param_lb_test_{suffix}.gz'
    file_param_ub = f'{outpath}/camels_summa_all_param_ub_test_{suffix}.gz'
    df_param_lb_test, df_param_ub_test = load_basin_param_bounds(inpath_moasmo, df_param_defa_test, file_param_lb, file_param_ub)

    
    file_camels_attribute = f'{outpath}/camels_basin_attribute_train_{suffix}.pkl'
    df_att_train = read_camels_attributes(infile_basin_info, file_camels_attribute, train_index)
    file_camels_attribute = f'{outpath}/camels_basin_attribute_test_{suffix}.pkl'
    df_att_test = read_camels_attributes(infile_basin_info, file_camels_attribute, test_index)

    df_att_foruse = pd.read_csv(infile_attr_foruse)
    useattrs = list(df_att_foruse[df_att_foruse['att_Xie2021'].values]['Attribute_text'].values)
    print("The number of attributes used:", len(useattrs))
    print(useattrs)
    
    suffixtest = suffix+'test'

    # Load data: outputs from each iteration from training basins
    for iter in range(0, iterend):
        file_all_param = f'{outpath}/camels_summa_all_param_train_{suffix}_iter{iter}.gz'
        file_all_metric = f'{outpath}/camels_summa_all_metric_train_{suffix}_iter{iter}.gz'
        file_all_basinid = f'{outpath}/camels_summa_all_basinid_train_{suffix}_iter{iter}.gz'
    
        file_all_param_test = f'{outpath}/camels_summa_all_param_test_{suffix}_iter{iter}.gz'
        file_all_metric_test = f'{outpath}/camels_summa_all_metric_test_{suffix}_iter{iter}.gz'
        file_all_basinid_test = f'{outpath}/camels_summa_all_basinid_test_{suffix}_iter{iter}.gz'
    
        if iter == 0:
    
            df_param_i, df_metric_i, df_basinid_i = load_all_basin_params_metrics(inpath_moasmo, infile_param_info, df_param_defa_train,
                                                                                  df_basin_info, iter, file_all_param,
                                                                                  file_all_metric, file_all_basinid,
                                                                                  train_index, suffix_defa_source)
        
            df_param_i_test, df_metric_i_test, df_basinid_i_test = load_all_basin_params_metrics(inpath_moasmo, infile_param_info, df_param_defa_test,
                                                                                  df_basin_info, iter, file_all_param_test,
                                                                                  file_all_metric_test, file_all_basinid_test,
                                                                                  test_index, suffix_defa_source)
        else:
            df_param_i, df_metric_i, df_basinid_i = load_all_basin_params_metrics(inpath_moasmo, infile_param_info, df_param_defa_train,
                                                                                  df_basin_info, iter, file_all_param,
                                                                                  file_all_metric, file_all_basinid,
                                                                                  train_index, suffix)
        
            df_param_i_test, df_metric_i_test, df_basinid_i_test = load_all_basin_params_metrics(inpath_moasmo, infile_param_info, df_param_defa_test,
                                                                                  df_basin_info, iter, file_all_param_test,
                                                                                  file_all_metric_test, file_all_basinid_test,
                                                                                  test_index, suffixtest)
    
        df_basinid_i['iter'] = iter
        df_basinid_i_test['iter'] = iter
    
        if iter == 0:
            df_param = df_param_i
            df_metric = df_metric_i
            df_basinid = df_basinid_i
            
            df_param_test = df_param_i_test
            df_metric_test = df_metric_i_test
            df_basinid_test = df_basinid_i_test
        else:
            df_param = pd.concat([df_param, df_param_i])
            df_metric = pd.concat([df_metric, df_metric_i])
            df_basinid = pd.concat([df_basinid, df_basinid_i])
            
            df_param_test = pd.concat([df_param_test, df_param_i_test])
            df_metric_test = pd.concat([df_metric_test, df_metric_i_test])        
            df_basinid_test = pd.concat([df_basinid_test, df_basinid_i_test])

    df_param = df_param.apply(pd.to_numeric, errors='coerce')
    df_param_test = df_param_test.apply(pd.to_numeric, errors='coerce')


    df_param.index = np.arange(len(df_param))
    df_metric.index = np.arange(len(df_metric))
    df_basinid.index = np.arange(len(df_basinid))
    
    df_param_test.index = np.arange(len(df_param_test))
    df_metric_test.index = np.arange(len(df_metric_test))
    df_basinid_test.index = np.arange(len(df_basinid_test))
    
    
    index = np.isnan(np.sum(df_metric.values, axis=1) + np.sum(df_param.values, axis=1))
    df_param = df_param[~index]
    df_metric = df_metric[~index]
    df_basinid = df_basinid[~index]
    
    index_test = np.isnan(np.sum(df_metric_test.values, axis=1) + np.sum(df_param_test.values, axis=1))
    df_param_test = df_param_test[~index_test]
    df_metric_test = df_metric_test[~index_test]
    df_basinid_test = df_basinid_test[~index_test]
    
    
    df_param.index = np.arange(len(df_param))
    df_metric.index = np.arange(len(df_metric))
    df_basinid.index = np.arange(len(df_basinid))
    
    df_param_test.index = np.arange(len(df_param_test))
    df_metric_test.index = np.arange(len(df_metric_test))
    df_basinid_test.index = np.arange(len(df_basinid_test))
    
    
    print('Number of nan samples:', np.sum(index))
    print("Number of original parameter sets:", len(index))
    print("Number of final parameter sets:", len(df_param))
    
    
    # One-hot encoding for categorical attributes
    df_att = pd.concat([df_att_train, df_att_test])
    df_att.index = np.arange(len(df_att))
    df_att_use = df_att[useattrs + ["hru_id"]]

    for att in useattrs:
        if df_att_use[att].dtype == "object":
            print('Convert', att, 'to one-hot encoding')
            enc = OneHotEncoder(sparse=False)
            enc.fit(df_att_use[[att]])
            encnames = [att + "_" + str(i) for i in range(len(enc.categories_[0]))]
            print('New columns:', encnames)
            df_enc = pd.DataFrame(enc.transform(df_att_use[[att]]), columns=encnames)
            df_att_use = pd.concat([df_att_use, df_enc], axis=1)
            df_att_use = df_att_use.drop([att], axis=1)

    df_att_use_train = df_att_use[:len(df_att_train)]
    df_att_use_test = df_att_use[len(df_att_train):]
    df_att_use_train.index = np.arange(len(df_att_use_train))
    df_att_use_test.index = np.arange(len(df_att_use_test))
    
    useattrs = list(df_att_use_train.columns)
    useattrs.remove('hru_id')

    # Prepare model input and output
    df_input = df_param.copy()
    df_input["hru_id"] = df_basinid["basin_name"]
    df_input = df_input.merge(df_att_use_train[useattrs + ["hru_id"]], on="hru_id", how="left")
    df_input = df_input.drop(["hru_id"], axis=1)

    inputnames = list(df_param.columns) + useattrs
    x_all = df_input[inputnames].values.copy()
    print("Input shape:", x_all.shape)

    print('Train/test model')
    print('Train index:', train_index)


    if objfunc == 'normKGE':
        print('Use normalized KGE as output')
        df_output = df_metric.copy()
        y_all = df_output[["kge"]].values.copy()
        y_all = y_all / (2 - y_all)  # Normalize KGE

        # Train a random forest emulator
        outfile = f'{outpath}/RF_emulator_for_iter{iterend}_{suffix}.pkl'
        if os.path.isfile(outfile):
            with open(outfile, 'rb') as file:
                em_model = pickle.load(file)
        else:
            modelconfig = {'n_estimators': 100, 'random_state': 42, 'max_depth': 40}
            em_model = RandomForestRegressor(**modelconfig, n_jobs=ncpus)
            em_model.fit(x_all, y_all)
            with open(outfile, 'wb') as file:
                pickle.dump(em_model, file)


    elif objfunc == 'norm2err':
        print('Use normalized mae and mmae as output')
        # normalization is performed for each basin
        df_output = df_metric.copy()
        metvalues = df_output[['mae', 'max_mon_abs_err']].values.copy()
        y_all = np.nan * metvalues
        for i in range(len(train_index)):
            indi = df_basinid['basin_num'].values == train_index[i]
            di = metvalues[indi, :]
            di = (di - np.nanmin(di, axis=0)) / (np.nanmax(di, axis=0) - np.nanmin(di, axis=0))
            y_all[indi, :] = di

        # Train a random forest emulator
        outfile = f'{outpath}/RF_emulator_2errOBJfunc_for_iter{iterend}_{suffix}.pkl'
        outfile_eval = f'{outpath}/RF_emulator_2errOBJfunc_for_iter{iterend}_{suffix}_eval.npz'
        if os.path.isfile(outfile):
            with open(outfile, 'rb') as file:
                em_model = pickle.load(file)
        else:
            modelconfig = {'n_estimators': 200, 'random_state': 42, 'max_depth': 40}
            em_model = RandomForestRegressor(**modelconfig, n_jobs=ncpus)
            em_model.fit(x_all, y_all)
            with open(outfile, 'wb') as file:
                pickle.dump(em_model, file)


    param_names = df_param_info['Parameter'].values
    parallel_process_basins(df_basinid, df_param_lb_train, df_param_ub_train,
                            x_all, df_input, y_all,
                            param_names, inputnames, em_model, inpath_moasmo, ncpus, numruns, iterend, train_index, suffix)

    #### predict parameter in unseen basins
    suffixtest = suffix+'test'
    numruns_test = 20 # can be smaller
    if objfunc == 'normKGE':
        num_objfunc=1
    else:
        sys.exit('Not tested objfunc')

    df_att_use_test2 = df_att_use_test.drop(['hru_id'], axis=1)

    xlb_mean_test = np.nan * np.zeros([len(df_param_lb_test), len(inputnames)])
    xub_mean_test = np.nan * np.zeros([len(df_param_ub_test), len(inputnames)])   
    for i in range(len(df_param_lb_test)):
        param_lb_mean = df_param_lb_test.values[i, :]
        param_ub_mean = df_param_ub_test.values[i, :]
        attrvalues = df_att_use_test2.values[i,:]
        xlb_mean_test[i,:] = np.hstack([param_lb_mean, attrvalues])
        xub_mean_test[i,:] = np.hstack([param_ub_mean, attrvalues])
    

    parallel_process_basins_predictunseen(df_basinid_test, xlb_mean_test, xub_mean_test, param_names, em_model, inpath_moasmo, ncpus, numruns_test, iterend, test_index, suffixtest, num_objfunc)


def allbasin_emulator_CV_traintest_and_optimize_2_ann(infile_basin_info, infile_param_info, infile_attr_foruse,
                                                  inpath_moasmo, outpathname, iterend, ncpus, suffix,
                                                  train_index, numruns=100, objfunc='normKGE'):
    # implementation - 2
    # (1) iter-0 simulations
    # (2) divide basins into 5 folds.
    # (3) For each fold, using 80% for training and 20% for testing
    # Iterative training based on the 80% basins. In this process, the 20% basins are never used. This could lead to iter-1, iter-2, …, iter-x, until saturated calibration. The implementation is the same with a typical joint emulator
    # For trained emulator in each iteration (iter-1, iter-2, iter-3, …, iter-x), it can be used to predict parameters in 20% testing basins. The number of predicted parameters can range from 1 to inf. This number can be smaller than 100 since we don’t need 100 optimized parameter sets in unseen basins. In practice, simply using iter-x is also fine
    # (3) is repeated five times to get validation results in each testing basins. (edited)

    # to implement that in a CV way, divide all basins into five folds. For each fold, name the suffix as "CV1", "CV2", "CV3", ... "CV5".
    # then, using train_index (80% stations) and test_index (20% stations)  as inputs
    # note other inputs such as infile_basin_info still contain all basins and should correspond to the index from train_index and test_index
    # train_index and test_index needs to be generated outside this function, maybe following the below method
    # train_indices = []
    # test_indices = []
    # for i in range(cv_num):
    #     ind1 = all_index[i::cv_num]
    #     ind2 = np.setdiff1d(all_index, ind1)
    #     test_indices.append(ind1)
    #     train_indices.append(ind2)

    # infile_basin_info = f"/glade/work/guoqiang/CTSM_CAMELS/data_mesh_surf/HillslopeHydrology/CAMELS_level1_basin_info.csv"
    # infile_param_info = '/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/parameter/CTSM_CAMELS_calibparam_2410.csv'
    # infile_attr_foruse = '/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/data/camels_attributes_table_TrainModel.csv'
    # inpath_moasmo = "/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_emulator"
    # outpathname = "emulator_CV_test"
    # path_CTSM_case = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_emulator'
    # train_index = np.setdiff1d(np.arange(627), np.arange(0, 627, 5))
    # numruns=100
    # objfunc='normKGE'
    # ncpus = 1
    # iterend = 1
    # suffix = 'CV1'
    # suffix_defa_source = 'LSEnormKGE' # temporary suffix if CV1 has not been generated
    
    suffix_defa_source = 'LSEnormKGE'

    outpath = f"{inpath_moasmo}/{outpathname}"
    os.makedirs(outpath, exist_ok=True)

    # Load data: same for all iterations
    df_basin_info = pd.read_csv(infile_basin_info)
    df_basin_info.index = np.arange(len(df_basin_info))
    all_index = np.arange(len(df_basin_info))

    test_index = np.setdiff1d(all_index, train_index)

    # information for all basins
    df_param_info = pd.read_csv(infile_param_info)

    file_defa_param = f'{outpath}/camels_summa_defa_param_train_{suffix}.csv'
    df_param_defa_train = read_allbasin_defa_params(infile_param_info, file_defa_param, Basin_list, train_index)

    file_defa_param = f'{outpath}/camels_summa_defa_param_test_{suffix}.csv'
    df_param_defa_test = read_allbasin_defa_params(infile_param_info, file_defa_param, Basin_list, test_index)

    file_param_lb = f'{outpath}/camels_summa_all_param_lb_train_{suffix}.gz'
    file_param_ub = f'{outpath}/camels_summa_all_param_ub_train_{suffix}.gz'
    
    df_param_lb_train, df_param_ub_train = load_basin_param_bounds(inpath_moasmo, df_param_defa_train, file_param_lb, file_param_ub)

    file_param_lb = f'{outpath}/camels_summa_all_param_lb_test_{suffix}.gz'
    file_param_ub = f'{outpath}/camels_summa_all_param_ub_test_{suffix}.gz'
    df_param_lb_test, df_param_ub_test = load_basin_param_bounds(inpath_moasmo, df_param_defa_test, file_param_lb, file_param_ub)

    
    file_camels_attribute = f'{outpath}/camels_basin_attribute_train_{suffix}.pkl'
    df_att_train = read_camels_attributes(infile_basin_info, file_camels_attribute, train_index)
    file_camels_attribute = f'{outpath}/camels_basin_attribute_test_{suffix}.pkl'
    df_att_test = read_camels_attributes(infile_basin_info, file_camels_attribute, test_index)

    df_att_foruse = pd.read_csv(infile_attr_foruse)
    useattrs = list(df_att_foruse[df_att_foruse['att_Xie2021'].values]['Attribute_text'].values)
    print("The number of attributes used:", len(useattrs))
    print(useattrs)

    suffixtest = suffix+'test'

    # Load data: outputs from each iteration from training basins
    for iter in range(0, iterend):
        file_all_param = f'{outpath}/camels_summa_all_param_train_{suffix}_iter{iter}.gz'
        file_all_metric = f'{outpath}/camels_summa_all_metric_train_{suffix}_iter{iter}.gz'
        file_all_basinid = f'{outpath}/camels_summa_all_basinid_train_{suffix}_iter{iter}.gz'
    
        file_all_param_test = f'{outpath}/camels_summa_all_param_test_{suffix}_iter{iter}.gz'
        file_all_metric_test = f'{outpath}/camels_summa_all_metric_test_{suffix}_iter{iter}.gz'
        file_all_basinid_test = f'{outpath}/camels_summa_all_basinid_test_{suffix}_iter{iter}.gz'
    
        if iter == 0:
    
            df_param_i, df_metric_i, df_basinid_i = load_all_basin_params_metrics(inpath_moasmo, infile_param_info, df_param_defa_train,
                                                                                  df_basin_info, iter, file_all_param,
                                                                                  file_all_metric, file_all_basinid,
                                                                                  train_index, suffix_defa_source)
        
            df_param_i_test, df_metric_i_test, df_basinid_i_test = load_all_basin_params_metrics(inpath_moasmo, infile_param_info, df_param_defa_test,
                                                                                  df_basin_info, iter, file_all_param_test,
                                                                                  file_all_metric_test, file_all_basinid_test,
                                                                                  test_index, suffix_defa_source)
        else:
            df_param_i, df_metric_i, df_basinid_i = load_all_basin_params_metrics(inpath_moasmo, infile_param_info, df_param_defa_train,
                                                                                  df_basin_info, iter, file_all_param,
                                                                                  file_all_metric, file_all_basinid,
                                                                                  train_index, suffix)
        
            df_param_i_test, df_metric_i_test, df_basinid_i_test = load_all_basin_params_metrics(inpath_moasmo, infile_param_info, df_param_defa_test,
                                                                                  df_basin_info, iter, file_all_param_test,
                                                                                  file_all_metric_test, file_all_basinid_test,
                                                                                  test_index, suffixtest)
    
        df_basinid_i['iter'] = iter
        df_basinid_i_test['iter'] = iter
    
        if iter == 0:
            df_param = df_param_i
            df_metric = df_metric_i
            df_basinid = df_basinid_i
            
            df_param_test = df_param_i_test
            df_metric_test = df_metric_i_test
            df_basinid_test = df_basinid_i_test
        else:
            df_param = pd.concat([df_param, df_param_i])
            df_metric = pd.concat([df_metric, df_metric_i])
            df_basinid = pd.concat([df_basinid, df_basinid_i])
            
            df_param_test = pd.concat([df_param_test, df_param_i_test])
            df_metric_test = pd.concat([df_metric_test, df_metric_i_test])        
            df_basinid_test = pd.concat([df_basinid_test, df_basinid_i_test])

    df_param = df_param.apply(pd.to_numeric, errors='coerce')
    df_param_test = df_param_test.apply(pd.to_numeric, errors='coerce')


    df_param.index = np.arange(len(df_param))
    df_metric.index = np.arange(len(df_metric))
    df_basinid.index = np.arange(len(df_basinid))
    
    df_param_test.index = np.arange(len(df_param_test))
    df_metric_test.index = np.arange(len(df_metric_test))
    df_basinid_test.index = np.arange(len(df_basinid_test))
    
    
    index = np.isnan(np.sum(df_metric.values, axis=1) + np.sum(df_param.values, axis=1))
    df_param = df_param[~index]
    df_metric = df_metric[~index]
    df_basinid = df_basinid[~index]
    
    index_test = np.isnan(np.sum(df_metric_test.values, axis=1) + np.sum(df_param_test.values, axis=1))
    df_param_test = df_param_test[~index_test]
    df_metric_test = df_metric_test[~index_test]
    df_basinid_test = df_basinid_test[~index_test]
    
    
    df_param.index = np.arange(len(df_param))
    df_metric.index = np.arange(len(df_metric))
    df_basinid.index = np.arange(len(df_basinid))
    
    df_param_test.index = np.arange(len(df_param_test))
    df_metric_test.index = np.arange(len(df_metric_test))
    df_basinid_test.index = np.arange(len(df_basinid_test))
    
    
    print('Number of nan samples:', np.sum(index))
    print("Number of original parameter sets:", len(index))
    print("Number of final parameter sets:", len(df_param))
    
    
    # One-hot encoding for categorical attributes
    df_att = pd.concat([df_att_train, df_att_test])
    df_att.index = np.arange(len(df_att))
    df_att_use = df_att[useattrs + ["hru_id"]]

    for att in useattrs:
        if df_att_use[att].dtype == "object":
            print('Convert', att, 'to one-hot encoding')
            enc = OneHotEncoder(sparse=False)
            enc.fit(df_att_use[[att]])
            encnames = [att + "_" + str(i) for i in range(len(enc.categories_[0]))]
            print('New columns:', encnames)
            df_enc = pd.DataFrame(enc.transform(df_att_use[[att]]), columns=encnames)
            df_att_use = pd.concat([df_att_use, df_enc], axis=1)
            df_att_use = df_att_use.drop([att], axis=1)

    df_att_use_train = df_att_use[:len(df_att_train)]
    df_att_use_test = df_att_use[len(df_att_train):]
    df_att_use_train.index = np.arange(len(df_att_use_train))
    df_att_use_test.index = np.arange(len(df_att_use_test))
    
    useattrs = list(df_att_use_train.columns)
    useattrs.remove('hru_id')

    # Prepare model input and output
    df_input = df_param.copy()
    df_input["hru_id"] = df_basinid["basin_name"]
    df_input = df_input.merge(df_att_use_train[useattrs + ["hru_id"]], on="hru_id", how="left")
    df_input = df_input.drop(["hru_id"], axis=1)

    inputnames = list(df_param.columns) + useattrs
    x_all = df_input[inputnames].values.copy()
    print("Input shape:", x_all.shape)

    print('Train/test model')
    print('Train index:', train_index)

    
    # divide samples into training and validation sets (70% vs 30%)
    hru_idu = np.unique(df_basinid["basin_name"].values)
    index_val_tmp = np.linspace(0, len(hru_idu)-1, int(len(hru_idu) * 0.3 )).astype(int)
    index_train_tmp = np.setdiff1d(np.arange(len(hru_idu)), index_val_tmp)
    hru_idu_train = hru_idu[index_train_tmp]
    hru_idu_val = hru_idu[index_val_tmp]

    index_train = df_basinid["basin_name"].isin(hru_idu_train)
    index_val = df_basinid["basin_name"].isin(hru_idu_val)

    x_train, x_val = x_all[index_train, :], x_all[index_val, :]

    # Normalize the features
    # scaler = StandardScaler()
    # x_train_scaled = scaler.fit_transform(x_train)
    # x_val_scaled = scaler.transform(x_val)
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train_scaled = (x_train - x_train_mean) / x_train_std
    x_val_scaled = (x_val - x_train_mean) / x_train_std


    if objfunc == 'normKGE':
        print('Use normalized KGE as output')
        df_output = df_metric.copy()
        y_all = df_output[["kge"]].values.copy()
        y_all = y_all / (2 - y_all)  # Normalize KGE
        y_train, y_val = y_all[index_train], y_all[index_val]
        
        # Train a random forest emulator
        outfile = f'{outpath}/ANN_emulator_for_iter{iterend}_{suffix}'
        em_model = train_nn_model_pytorch(x_train_scaled, y_train, x_val_scaled, y_val, model_file=outfile)


    normdict = {'method': 'z-score',
                'mean': x_train_mean,
                'std': x_train_std,
               }
    param_names = df_param_info['Parameter'].values
    parallel_process_basins_norm(df_basinid, df_param_lb_train, df_param_ub_train,
                            x_all, df_input, y_all,
                            param_names, inputnames, em_model, inpath_moasmo, ncpus, numruns, iterend, train_index, suffix, normdict)

    #### predict parameter in unseen basins
    suffixtest = suffix+'test'
    numruns_test = 1 # can be smaller
    if objfunc == 'normKGE':
        num_objfunc=1
    else:
        sys.exit('Not tested objfunc')

    df_att_use_test2 = df_att_use_test.drop(['hru_id'], axis=1)

    xlb_mean_test = np.nan * np.zeros([len(df_param_lb_test), len(inputnames)])
    xub_mean_test = np.nan * np.zeros([len(df_param_ub_test), len(inputnames)])   
    for i in range(len(df_param_lb_test)):
        param_lb_mean = df_param_lb_test.values[i, :]
        param_ub_mean = df_param_ub_test.values[i, :]
        attrvalues = df_att_use_test2.values[i,:]
        xlb_mean_test[i,:] = np.hstack([param_lb_mean, attrvalues])
        xub_mean_test[i,:] = np.hstack([param_ub_mean, attrvalues])

    xlb_mean_test_scaled = (xlb_mean_test - x_train_mean) / x_train_std
    xub_mean_test_scaled = (xub_mean_test - x_train_mean) / x_train_std
    normdict = {'method': 'z-score',
                'mean': x_train_mean,
                'std': x_train_std,
               }
    
    parallel_process_basins_predictunseen_norm(df_basinid_test, xlb_mean_test_scaled, xub_mean_test_scaled, param_names, em_model, inpath_moasmo, ncpus, numruns_test, iterend, test_index, suffixtest, num_objfunc, normdict)


    








