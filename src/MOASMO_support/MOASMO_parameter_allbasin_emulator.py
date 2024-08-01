# all functions needed implement all basin emulator
import glob, os, sys, toml, pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Pool, cpu_count
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from MOASMO_parameters import *

def read_ctsm_default_parameters(param_names, param_sources, path_CTSM_case):
    # use functions from MOASMO_parameters
    param0 = []
    for j in range(len(param_names)):
        param0.append(get_parameter_value_from_CTSM_case(param_names[j], param_sources[j], path_CTSM_case))
    
    return param0
    
###################################################################################################
# data preparation functions

def read_camels_attributes(infile_basin_info, outfile):

    if os.path.isfile(outfile):
        print('File exists:', outfile)
        df_att = pd.read_csv(outfile)

    else:

        # Load basin info
         # = f"/glade/work/guoqiang/CTSM_CAMELS/data_mesh_surf/HillslopeHydrology/CAMELS_level1_basin_info.csv"
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

        print('save attributes to file', outfile)
        df_att.to_csv(outfile, index=False)
        
    return df_att


def read_allbasin_defa_params(pathctsm, infile_param_info, outfile_defa_param, basinnum):

    # infile_param_info = '/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/parameter/CTSM_CAMELS_SA_param_240202.csv'\
    # pathctsm = '/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange'
    # outfile_defa_param = 'camels_627basin_ctsm_defa_param.csv'
    
    df_param_info = pd.read_csv(infile_param_info)

    # load default parameters for each basin
    param_names = df_param_info['Parameter'].values
    param_sources = df_param_info['Source'].values
    
    
    
    if os.path.isfile(outfile_defa_param):
        df_param_defa = pd.read_csv(outfile_defa_param)
    else:
        
        param_defa = np.nan * np.zeros([basinnum, len(param_names)])
        for i in range(basinnum):
            path_CTSM_case = f'{pathctsm}/level1_{i}'
            parami_all = read_ctsm_default_parameters(param_names, param_sources, path_CTSM_case)
            parami_mean = [np.mean(p) for p in parami_all]
            param_defa[i, :] = parami_mean
        
        df_param_defa = pd.DataFrame(param_defa, columns=param_names)
        df_param_defa.to_csv(outfile_defa_param, index=False)

    return df_param_defa


def load_basin_param_bounds(inpath_moasmo, df_param_defa, file_param_lb, file_param_ub):
    # file_param_lb = 'camels_627basin_ctsm_all_param_lb.csv.gz'
    # file_param_ub = 'camels_627basin_ctsm_all_param_ub.csv.gz'

    if os.path.isfile(file_param_lb) and os.path.isfile(file_param_ub):
        df_param_lb = pd.read_csv(file_param_lb, compression='gzip')
        df_param_ub = pd.read_csv(file_param_ub, compression='gzip')
    else:
        param_lb_values = df_param_defa.values.copy()
        param_ub_values = df_param_defa.values.copy()

        for i in range(len(df_param_defa)):
            file = f"{inpath_moasmo}/level1_{i}_MOASMOcalib/param_sets_emutest/all_default_parameters.pkl"
            dfi = pd.read_pickle(file)

            for j in range(len(dfi['Parameter'].values)):
                indj = np.where(df_param_defa.columns.values == dfi['Parameter'].values[j])[0][0]
                param_lb_values[i, indj] = dfi['Lower'].values[j]
                param_ub_values[i, indj] = dfi['Upper'].values[j]

        df_param_lb = pd.DataFrame(param_lb_values, columns=df_param_defa.columns.values)
        df_param_ub = pd.DataFrame(param_ub_values, columns=df_param_defa.columns.values)

        df_param_lb.to_csv(file_param_lb, index=False, compression='gzip')
        df_param_ub.to_csv(file_param_ub, index=False, compression='gzip')

    return df_param_lb, df_param_ub


def load_all_basin_params_metrics(inpath_moasmo, df_param_defa, df_basin_info, tariter, file_all_param, file_all_metric, file_all_basinid):

    param_names = np.array(df_param_defa.columns)
    
    if os.path.isfile(file_all_param) and os.path.isfile(file_all_metric) and os.path.isfile(file_all_basinid):
        df_param = pd.read_csv(file_all_param, compression='gzip')
        df_metric = pd.read_csv(file_all_metric, compression='gzip')
        df_basinid = pd.read_csv(file_all_basinid, compression='gzip')
    else:
        df_metric = pd.DataFrame()
        df_param = pd.DataFrame()
        flag = 0

        for i in range(len(df_basin_info)):

            # load param
            file_param = f"{inpath_moasmo}/level1_{i}_MOASMOcalib/ctsm_outputs_emutest/iter{tariter}_all_meanparam.csv"
            df1 = pd.read_csv(file_param)

            parami = np.tile(df_param_defa.iloc[i].values, (len(df1), 1))
            for j in range(len(df1.columns)):
                if df1.columns[j] in param_names:  # Skip binded parameters
                    indj = np.where(param_names == df1.columns[j])[0][0]
                    parami[:, indj] = df1.values[:, j]

            df1 = pd.DataFrame(parami, columns=param_names)
            if df_param.empty:
                df_param = df1
            else:
                df_param = pd.concat([df_param, df1])

            # load metric
            file_metric = f"{inpath_moasmo}/level1_{i}_MOASMOcalib/ctsm_outputs_emutest/iter{tariter}_many_metric.csv"
            df2 = pd.read_csv(file_metric)
            metnames = df2.columns

            df2["basin_num"] = flag
            df2["basin_id"] = i
            df2["hru_id"] = df_basin_info["hru_id"].values[flag]

            if df_metric.empty:
                df_metric = df2
            else:
                df_metric = pd.concat([df_metric, df2])

            flag += 1

        df_param.index = np.arange(len(df_param))
        
        df_metric.index = np.arange(len(df_metric))
        df_basinid = df_metric[["basin_num", "basin_id", "hru_id"]]

        selected_met = ['kge', 'mae', 'n_mae', 'nse', 'cc', 'rmse', 'max_mon_abs_err', 'n_max_mon_abs_err', 'kge_log_q']
        df_metric = df_metric[selected_met]

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

from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

class MyProblem(Problem):
    def __init__(self, xlb_mean, xub_mean, em_model):
        super().__init__(n_var=len(xlb_mean), n_obj=1, n_constr=0, xl=xlb_mean, xu=xub_mean, elementwise_evaluation=False)
        self.em_model = em_model

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = -self.em_model.predict(x)  # Return negative to maximize


def run_ga_optimization(em_model, xlb_mean, xub_mean, num_runs=100, pop_size=100, num_generations=100):
    ga_all_solutions = []
    ga_all_outputs = []

    for _ in range(num_runs):  # run the model `num_runs` times
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
    
    return np.array(ga_all_solutions), np.array(ga_all_outputs)


def generate_param_files(tarbasin, em_model, xlb_mean, xub_mean, param_names, inpath_moasmo, numruns, iterend):
    path_CTSM_case = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/level1_{tarbasin}'
    outpath = f'{inpath_moasmo}/level1_{tarbasin}_MOASMOcalib/param_sets_emutest'
    os.makedirs(outpath, exist_ok=True)

    outfile_ga = f'{outpath}/ga_output_iter{iterend}.npz'
    if os.path.isfile(outfile_ga):
        dtmp = np.load(outfile_ga)
        ga_all_solutions = dtmp['ga_all_solutions']
        ga_all_outputs = dtmp['ga_all_outputs']
    else:
        ga_all_solutions, ga_all_outputs = run_ga_optimization(em_model, xlb_mean, xub_mean, num_runs=numruns, pop_size=100, num_generations=100)
        np.savez_compressed(outfile_ga, ga_all_solutions=ga_all_solutions, ga_all_outputs=ga_all_outputs)

    df_info = pd.read_pickle(f'{inpath_moasmo}/level1_{tarbasin}_MOASMOcalib/param_sets_emutest/paramset_iter0_trial0.pkl')
    df_info = df_info.loc[df_info['Default'] != 'None']
    df_info['Factor'] = np.nan
    df_info['Value'] = np.nan

    ga_all_solutions_array = np.array(ga_all_solutions)
    indexp = [np.where(param_names == p)[0][0] for p in df_info['Parameter'].values if p in param_names]

    for i in range(ga_all_solutions_array.shape[0]):
        outfile = f'{outpath}/paramset_iter{iterend}_trial{i}.pkl'
        if os.path.isfile(outfile):
            continue

        dfi = df_info.copy()
        dfi['Value'] = ga_all_solutions_array[i, indexp]
        dfi = check_and_generate_binded_parameters(dfi, path_CTSM_case)
        dfi.to_pickle(outfile)

    print('finish basin', tarbasin)

def process_basin(args):
    try:
        tarbasin, df_basinid, df_param_lb, df_param_ub, x_all, df_input, y_all, param_names, inputnames, em_model, inpath_moasmo, numruns, iterend = args
        print(tarbasin)
        index = np.where(df_basinid["basin_num"].values == tarbasin)[0]
        attnames = [i for i in inputnames if i not in param_names]
        param_lb_mean = df_param_lb.values[tarbasin, :]
        param_ub_mean = df_param_ub.values[tarbasin, :]
        attrvalues = x_all[index[0], len(param_lb_mean):]  # same with df_input
        if np.any(attrvalues != df_input.iloc[index[0]][attnames].values):
            print('Warning! att problem')

        x_tar = x_all[index, :]
        xlb_mean = np.hstack([param_lb_mean, attrvalues])
        xub_mean = np.hstack([param_ub_mean, attrvalues])
        generate_param_files(tarbasin, em_model, xlb_mean, xub_mean, param_names, inpath_moasmo, numruns, iterend)
    except Exception as e:
        print(f"Error processing basin {tarbasin}: {e}")


def parallel_process_basins(df_basin_info, df_basinid, df_param_lb, df_param_ub, x_all, df_input, y_all, param_names, inputnames, em_model, inpath_moasmo, ncpus, numruns, iterend):
    args = [(tarbasin, df_basinid, df_param_lb, df_param_ub, x_all, df_input, y_all, param_names, inputnames, em_model, inpath_moasmo, numruns, iterend) for tarbasin in range(len(df_basin_info))]
    
    with Pool(processes=ncpus) as pool:
        pool.map(process_basin, args)

###################################################################################################

def allbasin_emulator_train_and_optimize(infile_basin_info, infile_param_info, infile_attr_foruse, inpath_moasmo, path_CTSM_case, iterend, ncpus, numruns=100):
    # infile_basin_info = f"/glade/work/guoqiang/CTSM_CAMELS/data_mesh_surf/HillslopeHydrology/CAMELS_level1_basin_info.csv"
    # infile_param_info = '/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/parameter/CTSM_CAMELS_SA_param_240202.csv'
    # infile_attr_foruse = '/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/data/camels_attributes_table_TrainModel.csv'
    # inpath_moasmo = "/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange"
    # path_CTSM_case = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange'
    # ncpus = 20
    # iterend = 1

    outpath = f"{inpath_moasmo}/allbasin_emulator"
    os.makedirs(outpath, exist_ok=True)
    
    # Load data: same for all iterations
    df_basin_info = pd.read_csv(infile_basin_info)
    df_param_info = pd.read_csv(infile_param_info)
    
    file_defa_param = f'{outpath}/camels_627basin_ctsm_defa_param.csv'
    df_param_defa = read_allbasin_defa_params(path_CTSM_case, infile_param_info, file_defa_param, len(df_basin_info))

    file_param_lb = f'{outpath}/camels_627basin_ctsm_all_param_lb.gz'
    file_param_ub = f'{outpath}/camels_627basin_ctsm_all_param_ub.gz'
    df_param_lb, df_param_ub = load_basin_param_bounds(inpath_moasmo, df_param_defa, file_param_lb, file_param_ub)

    file_camels_attribute = f'{outpath}/camels_627basin_attribute.pkl'
    df_att = read_camels_attributes(infile_basin_info, file_camels_attribute)
    
    df_att_foruse = pd.read_csv(infile_attr_foruse)
    useattrs = list(df_att_foruse[df_att_foruse['att_Xie2021'].values]['Attribute_text'].values)
    print("The number of attributes used:", len(useattrs))
    print(useattrs)

    # Load data: outputs from each iteration
    for iter in range(0, iterend):
        file_all_param = f'{outpath}/camels_627basin_ctsm_all_param_iter{iter}.gz'
        file_all_metric = f'{outpath}/camels_627basin_ctsm_all_metric_iter{iter}.gz'
        file_all_basinid = f'{outpath}/camels_627basin_ctsm_all_basinid_iter{iter}.gz'
        
        df_param_i, df_metric_i, df_basinid_i = load_all_basin_params_metrics(inpath_moasmo, df_param_defa, df_basin_info, iter, file_all_param, file_all_metric, file_all_basinid)
        
        df_basinid_i['iter'] = iter
        
        if iter == 0:
            df_param = df_param_i
            df_metric = df_metric_i
            df_basinid = df_basinid_i
        else:
            df_param = pd.concat([df_param, df_param_i])
            df_metric = pd.concat([df_metric, df_metric_i])
            df_basinid = pd.concat([df_basinid, df_basinid_i])
    
    df_param.index = np.arange(len(df_param))
    df_metric.index = np.arange(len(df_metric))
    df_basinid.index = np.arange(len(df_basinid))

    index = np.isnan(np.sum(df_metric.values, axis=1))
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
    df_input["hru_id"] = df_basinid["hru_id"]
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
    
    # Use normalized KGE as output
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

    param_names = df_param_info['Parameter'].values
    parallel_process_basins(df_basin_info, df_basinid, df_param_lb, df_param_ub, x_all, df_input, y_all, param_names, inputnames, em_model, inpath_moasmo, ncpus, numruns, iterend)

