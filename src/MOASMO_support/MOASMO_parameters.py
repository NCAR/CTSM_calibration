# functions for generating and saving MO-ASMO parameter sets, including initial and non-dominant parameters
import os, sys, subprocess, pickle, random
import numpy as np
import pandas as pd
import xarray as xr
import pickle
from mo_evaluation import get_modified_KGE
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# import MO-ASMO functions
# path_MOASMO = '/glade/u/home/guoqiang/model_sources/MO-ASMO/src'
path_MOASMO = '/glade/u/home/guoqiang/CTSM_repos/ctsm_optz/MO-ASMO/src/'
sys.path.append(path_MOASMO)
import sampling
import gp
import NSGA2

def get_parameter_from_Namelist_or_lndin(name, file_user_nl_clm, file_lndin, type='number'):
    # check Namelist file first, and then check
    flag = False
    with open(file_user_nl_clm, 'r') as f:
        for l in f:
            l = l.strip()
            if l.startswith(name):
                if type == 'number':
                    if 'd' in l:
                        l = l.replace('d', 'e')
                    value = np.array(float(l.split('=')[-1].strip().replace('\'', '')))
                elif type == 'str':
                    value = l.split('=')[-1].strip().replace('\'', '')
                flag = True
                break
    if not flag:
        with open(file_lndin, 'r') as f:
            for l in f:
                l = l.strip()
                if l.startswith(name):
                    if type == 'number':
                        if 'd' in l:
                            l = l.replace('d', 'e')
                        value = np.array(float(l.split('=')[-1].strip().replace('\'', '')))
                    elif type == 'str':
                        value = l.split('=')[-1].strip().replace('\'', '')
                    break
    return value


def get_parameter_value_from_CTSM_case(param_name, param_source, path_CTSM_case):
    # get the parameter values of an existing CTSM case

    file_user_nl_clm = f'{path_CTSM_case}/user_nl_clm'
    file_lndin = f'{path_CTSM_case}/Buildconf/clmconf/lnd_in'

    if param_source == 'Namelist':
        param_value = get_parameter_from_Namelist_or_lndin(param_name, file_user_nl_clm, file_lndin, 'number')
    elif param_source == 'Param':
        paramfile = get_parameter_from_Namelist_or_lndin('paramfile', file_user_nl_clm, file_lndin, 'str')
        with xr.open_dataset(paramfile) as ds:
            param_value = ds[param_name].values
    elif param_source == 'Surfdata':
        fsurdat = get_parameter_from_Namelist_or_lndin('fsurdat', file_user_nl_clm, file_lndin, 'str')
        with xr.open_dataset(fsurdat) as ds:
            param_value = ds[param_name].values
    else:
        sys.exit(f'Unknown param_source: {param_source}')

    return param_value


def check_and_generate_binded_parameters(df_param, path_CTSM_case):
    # check if there is any binded variable
    # if there is, add binded variables to df_calibparam
    # it is assumed that binded variables have the same parameter range so we can scale them equally
    if 'Binding' in df_param.columns:
        df_bind = pd.DataFrame()
        for i in range(len(df_param)):
            rawvari_value = get_parameter_value_from_CTSM_case(df_param.iloc[i]['Parameter'], df_param.iloc[i]['Source'], path_CTSM_case)
            bindvari = df_param.iloc[i]['Binding']
            if isinstance(bindvari, str):
                if bindvari != 'None':
                    bindvari = bindvari.split(',')
                    for bv in bindvari:
                        dftmp = df_param.iloc[[i]].copy()
                        dftmp['Parameter'] = bv
                        # mask other cols
                        for col in ['Default', 'Lower', 'Upper', 'Binding', 'Parameter_Ost']:
                            if col in dftmp.columns:
                                dftmp[col] = 'None'
                        # generate parameter values
                        bind_var_value0 = get_parameter_value_from_CTSM_case(bv, dftmp['Source'].values[0], path_CTSM_case)
                        dftmp['Value'] = bind_var_value0 + (df_param.iloc[i]['Value'] - rawvari_value)
                        df_bind = pd.concat([df_bind, dftmp])

        df_param = pd.concat([df_param, df_bind])

    return df_param


########################################################################################################################
# Initial sampling:
# Generate a TXN matrix X using the Good Lattice Points method with RGS de-correlation,
# where T is the number of sample points. Run the dynamic model for T times and obtain the multiobjective results Y
# Y=f(X)
# where Y is a TXM matrix containing the objective functions


def read_parameter_csv(file_parameter_list):

    df_calibparam = pd.read_csv(file_parameter_list)
    for c in ['Upper', 'Lower', 'Factor', 'Value']:
        if c in df_calibparam.columns:
            if isinstance(df_calibparam.iloc[0][c], str):
                arr = []
                for i in range(len(df_calibparam)):
                    vi = df_calibparam.iloc[i][c]
                    if ',' in vi:
                        arr.append(np.array(vi.split(',')).astype(np.float64))
                    elif '[' in vi:
                        arr.append(np.array(vi.strip('[]').replace('\n', '').split(), dtype=np.float64))
                    else:
                        try:
                            arr.append(np.array([np.float64(vi)]))
                        except:
                            arr.append(np.array([-99999]))
                df_calibparam[c] = arr
    return df_calibparam


def read_save_load_all_default_parameters(file_parameter_list, outpath, path_CTSM_case='', savefile=True):
    
    outfile = f'{outpath}/all_default_parameters.pkl'
    
    if not os.path.isfile(outfile):
        
        df_calibparam = read_parameter_csv(file_parameter_list)
        param_names = df_calibparam['Parameter'].values
        param_sources = df_calibparam['Source'].values
        
        dfi = df_calibparam.copy()
        param0 = []
        for j in range(len(param_names)):
            param0.append(get_parameter_value_from_CTSM_case(param_names[j], param_sources[j], path_CTSM_case))

        dfi['Value'] = param0

        if savefile == True:
            dfi.to_pickle(outfile) # preserve arrays
    
    
    print('Load default parameter values from:', outfile)
    if os.path.isfile(outfile):
        dfi = pd.read_pickle(outfile)

    return dfi


def generate_initial_parameter_sets(file_parameter_list, sampling_method, outpath, path_CTSM_case='', num_init=-1, adddefault=True):
    # example parameters
    # sampling_method = 'lh'  # lh: LatinHypercubeDesign, slh: SymmetricLatinHypercubeDesign, glp: GoodLatticePointsDesign
    # param_upper_bound = {'param1': np.array(15), 'param2': np.array([1, 2, 3])}
    # param_lower_bound = {'param1': np.array(3), 'param2': np.array([0.2, 1.5, 2.2])}
    # path_CTSM_case must be provided if there are any binded parameters for calibration
    # if adddefault=True, the first parameter set will be replaced by the default parameter of CTSM

    os.makedirs(outpath, exist_ok=True)

    df_calibparam = read_parameter_csv(file_parameter_list)
    param_upper_bound = df_calibparam['Upper'].values
    param_lower_bound = df_calibparam['Lower'].values
    
    param_upper_bound_mean = np.array([np.nanmean(p) for p in param_upper_bound])
    param_lower_bound_mean = np.array([np.nanmean(p) for p in param_lower_bound])

    # dimension sizes
    num_param = len(param_lower_bound_mean) # number of parameters to be calibrated
    if not num_init > 0:
        num_init = num_param * 20 # number of initial samples (i.e., initial model runs). A proper initial sample size should be 15–20 times the number of parameters (Gong et al., 2015)
        
    # check whether parameter files have been generated
    flag = False
    outfiles_all = []
    for i in range(num_init):
        outfile = f'{outpath}/paramset_iter0_trial{i}.pkl'
        outfiles_all.append(outfiles_all)
        if not os.path.isfile(outfile):
            flag = True
            break
    
    if flag == False:
        print('All ini parameter csv/pkl files have been generated. Skip this step')
    else:
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

        # save factors
        df_factor = pd.DataFrame(init_factors, columns=df_calibparam['Parameter'].values)
        df_factor.to_csv(f'{outpath}/paramset_iter0_scalefactors.csv', index=False)
        
        # load default parameter dataframe (file will be saved after first generation)
        df_defaultparam = read_save_load_all_default_parameters(file_parameter_list, outpath, path_CTSM_case)
        param0 = df_defaultparam['Value'].values

        # generate a dataframe for every set of parameters and deal with binding parameters
        outfiles_all = []
        for i in range(num_init):
            #outfile = f'{outpath}/paramset_iter0_trial{i}.csv'
            outfile = f'{outpath}/paramset_iter0_trial{i}.pkl'
            print('Generating parameter file:', outfile)
            dfi = df_calibparam.copy()

            if i == 0 and adddefault == True:
                print('For iteration 0, the default parameters will be used. scaling factors are not adopted')

                param_names = df_calibparam['Parameter'].values
                param_sources = df_calibparam['Source'].values
                
                factor0 = []
                for j in range(len(param_names)):
                    factor0.append( ( np.nanmean(param0[j]) - param_lower_bound_mean[j] )/(param_upper_bound_mean[j] - param_lower_bound_mean[j]) )

                dfi['Value'] = param0
                dfi['Factor'] = factor0

            else:

                dfi['Factor'] = init_factors[i, :]
                meanparam = init_factors[i, :] * (param_upper_bound_mean - param_lower_bound_mean) + param_lower_bound_mean
                newparam =  [meanparam[j] / np.nanmean(param0[j]) * param0[j] for j in range(len(param0))]
                dfi['Value'] = newparam

            # process binded parameters
            dfi = check_and_generate_binded_parameters(dfi, path_CTSM_case)

            #dfi.to_csv(outfile, index=False)
            dfi.to_pickle(outfile)
            
            outfiles_all.append(outfiles_all)
    
    
    return outfiles_all


########################################################################################################################
# Pareto optimal points:

def gpr_emulator_cv(x, y, alpha, leng_lb, leng_ub, nu, xlb_mean, xub_mean, outpath, iterflag):

    random.seed(1234567890)
    np.random.seed(1234567890)

    n_splits = 5
    
    kf = KFold(n_splits=n_splits, shuffle=True) 
    kge_scores = np.nan * np.zeros([n_splits, y.shape[1]])
    
    for fold_idx, (train_index, test_index) in enumerate(kf.split(x), 1):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Initialize and train your GPR model here; adjust parameters as needed
        sm = gp.GPR_Matern(x_train, y_train, x_train.shape[1], y_train.shape[1], x_train.shape[0], xlb_mean, xub_mean, alpha=alpha, leng_sb=[leng_lb, leng_ub], nu=nu)
        
        # Predict using the trained model
        y_pred = sm.predict(x_test)  # Adjust this method call based on your model's API
        
        # Evaluate the model using KGE
        for i in range(y.shape[1]):
            kge_scores[fold_idx-1, i] = get_modified_KGE(y_test[:,i], y_pred[:,i])
    
    # Calculate the mean KGE score across all folds
    mean_kge_score = np.nanmean(kge_scores, axis=0)[np.newaxis, :]
    kge_scores = np.concatenate([kge_scores, mean_kge_score])

    # Convert the list of KGE scores into a pandas DataFrame
    kge_scores_df = pd.DataFrame()
    kge_scores_df['Fold'] = list(np.arange(n_splits)+1) + ['mean']
    kge_scores_df['kge1'] = kge_scores[:, 0]
    kge_scores_df['kge2'] = kge_scores[:, 1]
    kge_scores_df['kge_mean'] = (kge_scores[:, 0] + kge_scores[:, 1])/2
    
    print("GPR CV KGE Score for metric1/metric2:")
    print(kge_scores_df)
    
    csv_file_path =  f'{outpath}/GPR_for_iter{iterflag}_CV_kge.csv'
    kge_scores_df.to_csv(csv_file_path, index=False)

    return kge_scores_df


def rf_emulator_cv(x, y, outpath, iterflag):

    random.seed(1234567890)
    np.random.seed(1234567890)
    
    n_splits = 5
    
    kf = KFold(n_splits=n_splits, shuffle=True) 
    kge_scores = np.nan * np.zeros([n_splits, y.shape[1]])
    
    for fold_idx, (train_index, test_index) in enumerate(kf.split(x), 1):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Initialize and train your GPR model here; adjust parameters as needed
        sm = RandomForestRegressor()
        sm.fit(x_train, y_train)
        
        # Predict using the trained model
        y_pred = sm.predict(x_test)  # Adjust this method call based on your model's API
        
        # Evaluate the model using KGE
        for i in range(y.shape[1]):
            kge_scores[fold_idx-1, i] = get_modified_KGE(y_test[:,i], y_pred[:,i])
    
    # Calculate the mean KGE score across all folds
    mean_kge_score = np.nanmean(kge_scores, axis=0)[np.newaxis, :]
    kge_scores = np.concatenate([kge_scores, mean_kge_score])

    # Convert the list of KGE scores into a pandas DataFrame
    kge_scores_df = pd.DataFrame()
    kge_scores_df['Fold'] = list(np.arange(n_splits)+1) + ['mean']
    kge_scores_df['kge1'] = kge_scores[:, 0]
    kge_scores_df['kge2'] = kge_scores[:, 1]
    kge_scores_df['kge_mean'] = (kge_scores[:, 0] + kge_scores[:, 1])/2
    
    print("RF CV KGE Score for metric1/metric2:")
    print(kge_scores_df)
    
    csv_file_path =  f'{outpath}/RF_for_iter{iterflag}_CV_kge.csv'
    kge_scores_df.to_csv(csv_file_path, index=False)

    return kge_scores_df


def surrogate_model_train_and_pareto_points(param_infofile, param_filelist, metric_filelist, outpath, iterflag, num_per_iter, path_CTSM_case=''):
    # path_CTSM_case must be provided if there are any binded parameters for calibration

    random.seed(1234567890)
    
    # check whether files have been generated
    flag = False
    for i in range(num_per_iter):
        outfile = f'{outpath}/paramset_iter{iterflag+1}_trial{i}.csv'
        if not os.path.isfile(outfile):
            flag = True
            break
            
    if flag == False:
        print('All parameter csv files have been generated. Skip this step')
    else:
        # define hyper parameters
        pop = 100
        gen = 100
        crossover_rate = 0.9
        mu = 20
        mum = 20

        # define hyperparameter
        alpha = 1e-3
        leng_lb = 1e-3
        leng_ub = 1e3
        nu = 2.5

        n_sample = num_per_iter # number of selected optimal points

        # input data x (parameter sets) and output data y (objective function values)
        df_param = pd.concat(map(pd.read_csv, param_filelist))
        df_metric = pd.concat(map(pd.read_csv, metric_filelist))
        df_info = read_parameter_csv(param_infofile)

        param_names = df_info['Parameter'].values # exclude binded parameters
        df_param = df_param[param_names]

        xlb_mean = np.array([np.nanmean(v) for v in df_info['Lower']])
        xub_mean = np.array([np.nanmean(v) for v in df_info['Upper']])

        x = df_param.to_numpy()
        y = df_metric.to_numpy()

        ind = ~np.isnan( np.sum(x,axis=1) + np.sum(y,axis=1))
        x, y = x[ind, :], y[ind, :]

        nInput = x.shape[1]
        nOutput = y.shape[1]


        # decide the most suitable emulator based on cross validation
        os.makedirs(outpath, exist_ok=True)
        gpr_kge_cv = gpr_emulator_cv(x, y, alpha, leng_lb, leng_ub, nu, xlb_mean, xub_mean, outpath, iterflag)
        rf_kge_cv = rf_emulator_cv(x, y, outpath, iterflag)

        # train the surrogate model 
        if gpr_kge_cv['kge_mean'].values[-1] > rf_kge_cv['kge_mean'].values[-1]:
        # if True: # always use GPR
            print('Use GPR model')
            sm = gp.GPR_Matern(x, y, nInput, nOutput, x.shape[0], xlb_mean, xub_mean, alpha=alpha, leng_sb=[leng_lb, leng_ub], nu=nu)
            flag = 1
        else:
            print('Use RF model')
            sm = RandomForestRegressor()
            sm.fit(x, y)
            flag = 2

        
        # perform optimization using the surrogate model
        bestx_sm, besty_sm, x_sm, y_sm = NSGA2.optimization(sm, nInput, nOutput, xlb_mean, xub_mean, pop, gen, crossover_rate, mu, mum)
        D = NSGA2.crowding_distance(besty_sm)
        print('model sample number:', D.shape[0])
        if D.shape[0] < n_sample:
            print('Too few samples. Use the other method')
            # use the other model
            if flag == 1:
                sm = RandomForestRegressor()
                sm.fit(x, y)
            elif flag == 2:
                sm = gp.GPR_Matern(x, y, nInput, nOutput, x.shape[0], xlb_mean, xub_mean, alpha=alpha, leng_sb=[leng_lb, leng_ub], nu=nu)
                bestx_sm, besty_sm, x_sm, y_sm = NSGA2.optimization(sm, nInput, nOutput, xlb_mean, xub_mean, pop, gen, crossover_rate, mu, mum)
                D = NSGA2.crowding_distance(besty_sm)
                print('model sample number:', D.shape[0])

        sm_filename = f'{outpath}/surrogate_model_for_iter{iterflag}'
        pickle.dump(sm, open(sm_filename, 'wb'))
        
        idxr = D.argsort()[::-1][:n_sample]
        x_resample = bestx_sm[idxr, :]
        y_resample = besty_sm[idxr, :]
        # y_resample = sm.predict(x_resample)

        # # plot
        # import matplotlib.pyplot as plt
        # plt.scatter(y[:, 0], y[:, 1])
        # plt.scatter(besty_sm[:, 0], besty_sm[:, 1])
        # plt.scatter(besty_sm[idxr, 0], besty_sm[idxr, 1])

        param_upper_bound = df_info['Upper'].values
        param_lower_bound = df_info['Lower'].values
        param_upper_bound_mean = np.array([np.nanmean(p) for p in param_upper_bound])
        param_lower_bound_mean = np.array([np.nanmean(p) for p in param_lower_bound])
        
        # load default parameter dataframe (file will be saved after first generation)
        df_defaultparam = read_save_load_all_default_parameters(param_filelist, outpath, path_CTSM_case)
        param0 = df_defaultparam['Value'].values


        # generate a parameter dataframe for next trial
        for i in range(x_resample.shape[0]):
            # outfile = f'{outpath}/paramset_iter{iterflag+1}_trial{i}.csv'
            outfile = f'{outpath}/paramset_iter{iterflag+1}_trial{i}.pkl'
            print('Generating parameter file:', outfile)

            dfi = df_info.copy()
            factors = (x_resample[i, :] - xlb_mean) / (xub_mean - xlb_mean)
            factors[factors<0] = 0.01
            factors[factors>1] = 0.99
            dfi['Factor'] = factors
            
            meanparam = factors * (param_upper_bound_mean - param_lower_bound_mean) + param_lower_bound_mean
            newparam =  [meanparam[j] / np.nanmean(param0[j]) * param0[j] for j in range(len(param0))]
            dfi['Value'] = newparam

            # process binded parameters
            dfi = check_and_generate_binded_parameters(dfi, path_CTSM_case)

            # write
            #dfi.to_csv(outfile, index=False)
            dfi.to_pickle(outfile)



########################
# for experiments
import numpy as np
import copy
def fast_non_dominated_sort(Y):
    ''' a fast non-dominated sorting method
        Y: output objective matrix
    '''
    N, d = Y.shape
    Q = [] # temp array of Pareto front index
    Sp = [] # temp array of points dominated by p
    S = [] # temp array of Sp
    rank = np.zeros(N) # Pareto rank
    n = np.zeros(N)  # domination counter of p
    dom = np.zeros((N, N))  # the dominate matrix, 1: i doms j, 2: j doms i

    # compute the dominate relationship online, much faster
    for i in range(N):
        for j in range(N):
            if i != j:
                if dominates(Y[i,:], Y[j,:]):
                    dom[i,j] = 1
                    Sp.append(j)
                elif dominates(Y[j,:], Y[i,:]):
                    dom[i,j] = 2
                    n[i] += 1
        if n[i] == 0:
            rank[i] = 0
            Q.append(i)
        S.append(copy.deepcopy(Sp))
        Sp = []

    F = []
    F.append(copy.deepcopy(Q))
    k = 0
    while len(F[k]) > 0:
        Q = []
        for i in range(len(F[k])):
            p = F[k][i]
            for j in range(len(S[p])):
                q = S[p][j]
                n[q] -= 1
                if n[q] == 0:
                    rank[q]  = k + 1
                    Q.append(q)
        k += 1
        F.append(copy.deepcopy(Q))

    return rank, dom

def dominates(p,q):
    ''' comparison for multi-objective optimization
        d = True, if p dominates q
        d = False, if p not dominates q
        p and q are 1*nOutput array
    '''
    if sum(p > q) == 0:
        d = True
    else:
        d = False
    return d



def surrogate_model_train_and_pareto_points_experiment(param_infofile, param_filelist, metric_filelist, outpath, iterflag, num_per_iter, path_CTSM_case='',innum=200):
    # path_CTSM_case must be provided if there are any binded parameters for calibration

    random.seed(1234567890)
    
    # check whether files have been generated
    flag = False
    for i in range(num_per_iter):
        outfile = f'{outpath}/paramset_iter{iterflag+1}_trial{i}.csv'
        if not os.path.isfile(outfile):
            flag = True
            break
            
    if flag == False:
        print('All parameter csv files have been generated. Skip this step')
    else:
        # define hyper parameters
        pop = 200
        gen = 100
        crossover_rate = 0.9
        mu = 20
        mum = 20

        # define hyperparameter
        alpha = 1e-3
        leng_lb = 1e-3
        leng_ub = 1e3
        nu = 2.5

        n_sample = num_per_iter # number of selected optimal points

        # input data x (parameter sets) and output data y (objective function values)
        df_param = pd.concat(map(pd.read_csv, param_filelist))
        df_metric = pd.concat(map(pd.read_csv, metric_filelist))
        df_info = read_parameter_csv(param_infofile)

        param_names = df_info['Parameter'].values # exclude binded parameters
        df_param = df_param[param_names]

        xlb_mean = np.array([np.nanmean(v) for v in df_info['Lower']])
        xub_mean = np.array([np.nanmean(v) for v in df_info['Upper']])

        x = df_param.to_numpy()
        y = df_metric.to_numpy()

        ind = ~np.isnan( np.sum(x,axis=1) + np.sum(y,axis=1))
        x, y = x[ind, :], y[ind, :]

        
        # # select some non-dominant parameters
        # # print: using the best 200 parameter sets
        # print('Using the best y inputs')
        # print('raw x/y size', x.shape, y.shape)
        # print('raw ymean', np.nanmean(y, axis=0))
        
        # rank,dom=fast_non_dominated_sort(y)
        # index = np.argsort(rank)[:innum]
        # x = x[index,:]
        # y = y[index,:]
        
        # print('new x/y size', x.shape, y.shape)
        # print('new ymean', np.nanmean(y, axis=0))
        
        nInput = x.shape[1]
        nOutput = y.shape[1]


        # decide the most suitable emulator based on cross validation
        os.makedirs(outpath, exist_ok=True)
        gpr_kge_cv = gpr_emulator_cv(x, y, alpha, leng_lb, leng_ub, nu, xlb_mean, xub_mean, outpath, iterflag)
        rf_kge_cv = rf_emulator_cv(x, y, outpath, iterflag)

        # train the surrogate model 
        if gpr_kge_cv['kge_mean'].values[-1] > rf_kge_cv['kge_mean'].values[-1]:
        # if True: # always use GPR
            print('Use GPR model')
            sm = gp.GPR_Matern(x, y, nInput, nOutput, x.shape[0], xlb_mean, xub_mean, alpha=alpha, leng_sb=[leng_lb, leng_ub], nu=nu)
            flag = 1
        else:
            print('Use RF model')
            sm = RandomForestRegressor()
            sm.fit(x, y)
            flag = 2

        
        # perform optimization using the surrogate model
        bestx_sm, besty_sm, x_sm, y_sm = NSGA2.optimization(sm, nInput, nOutput, xlb_mean, xub_mean, pop, gen, crossover_rate, mu, mum)
        D = NSGA2.crowding_distance(besty_sm)
        print('model sample number:', D.shape[0])
        if D.shape[0] < n_sample:
            print('Too few samples. Use the other method')
            # use the other model
            if flag == 1:
                sm = RandomForestRegressor()
                sm.fit(x, y)
            elif flag == 2:
                sm = gp.GPR_Matern(x, y, nInput, nOutput, x.shape[0], xlb_mean, xub_mean, alpha=alpha, leng_sb=[leng_lb, leng_ub], nu=nu)
                bestx_sm, besty_sm, x_sm, y_sm = NSGA2.optimization(sm, nInput, nOutput, xlb_mean, xub_mean, pop, gen, crossover_rate, mu, mum)
                D = NSGA2.crowding_distance(besty_sm)
                print('model sample number:', D.shape[0])

        sm_filename = f'{outpath}/surrogate_model_for_iter{iterflag}'
        pickle.dump(sm, open(sm_filename, 'wb'))
        
        idxr = D.argsort()[::-1][:n_sample]
        x_resample = bestx_sm[idxr, :]
        y_resample = besty_sm[idxr, :]
        # y_resample = sm.predict(x_resample)

        # # plot
        # import matplotlib.pyplot as plt
        # plt.scatter(y[:, 0], y[:, 1])
        # plt.scatter(besty_sm[:, 0], besty_sm[:, 1])
        # plt.scatter(besty_sm[idxr, 0], besty_sm[idxr, 1])

        param_upper_bound = df_info['Upper'].values
        param_lower_bound = df_info['Lower'].values
        param_upper_bound_mean = np.array([np.nanmean(p) for p in param_upper_bound])
        param_lower_bound_mean = np.array([np.nanmean(p) for p in param_lower_bound])
        
        # load default parameter dataframe (file will be saved after first generation)
        df_defaultparam = read_save_load_all_default_parameters(param_filelist, outpath, path_CTSM_case)
        param0 = df_defaultparam['Value'].values


        # generate a parameter dataframe for next trial
        for i in range(x_resample.shape[0]):
            # outfile = f'{outpath}/paramset_iter{iterflag+1}_trial{i}.csv'
            outfile = f'{outpath}/paramset_iter{iterflag+1}_trial{i}.pkl'
            print('Generating parameter file:', outfile)

            dfi = df_info.copy()
            factors = (x_resample[i, :] - xlb_mean) / (xub_mean - xlb_mean)
            factors[factors<0] = 0.01
            factors[factors>1] = 0.99
            dfi['Factor'] = factors
            
            meanparam = factors * (param_upper_bound_mean - param_lower_bound_mean) + param_lower_bound_mean
            newparam =  [meanparam[j] / np.nanmean(param0[j]) * param0[j] for j in range(len(param0))]
            dfi['Value'] = newparam

            # process binded parameters
            dfi = check_and_generate_binded_parameters(dfi, path_CTSM_case)

            # write
            #dfi.to_csv(outfile, index=False)
            dfi.to_pickle(outfile)



