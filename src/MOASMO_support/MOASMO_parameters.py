# functions for generating and saving MO-ASMO parameter sets, including initial and non-dominant parameters
import os, sys, subprocess, pickle
import numpy as np
import pandas as pd
import xarray as xr
import pickle

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

    # save factors
    df_factor = pd.DataFrame(init_factors, columns=df_calibparam['Parameter'].values)
    df_factor.to_csv(f'{outpath}/paramset_iter0_scalefactors.csv', index=False)

    # generate a dataframe for every set of parameters and deal with binding parameters
    outfiles_all = []
    for i in range(num_init):
        outfile = f'{outpath}/paramset_iter0_trial{i}.csv'
        print('Generating parameter file:', outfile)
        dfi = df_calibparam.copy()

        if i == 0 and adddefault == True:
            print('For iteration 0, the default parameters will be used. scaling factors are not adopted')

            param_names = df_calibparam['Parameter'].values
            param_sources = df_calibparam['Source'].values
            param0 = []
            factor0 = []
            for j in range(len(param_names)):
                param0.append(get_parameter_value_from_CTSM_case(param_names[j], param_sources[j], path_CTSM_case))
                factor0.append( (np.nanmean(param0[j]) - param_lower_bound[j])/(param_upper_bound[j] - param_lower_bound[j]) )

            dfi['Value'] = param0
            dfi['Factor'] = factor0

        else:

            dfi['Factor'] = init_factors[i, :]
            dfi['Value'] = init_factors[i, :] * (param_upper_bound - param_lower_bound) + param_lower_bound


        # process binded parameters
        dfi = check_and_generate_binded_parameters(dfi, path_CTSM_case)

        dfi.to_csv(outfile, index=False)
        outfiles_all.append(outfiles_all)



########################################################################################################################
# Pareto optimal points:

def surrogate_model_train_and_pareto_points(param_infofile, param_filelist, metric_filelist, outpath, iterflag, num_per_iter, path_CTSM_case=''):
    # path_CTSM_case must be provided if there are any binded parameters for calibration

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

    nInput = x.shape[1]
    nOutput = y.shape[1]

    # train the surrogate model
    # https://github.com/NCAR/ctsm_optz/blob/89e3689e73180574c62d1f5aa555a57e886a7cec/workflow/scripts/MOASMO_onestep.pe_basin.py#LL311C1-L315C41
    # sm = gp.GPR_Matern(x, y, nInput, nOutput, x.shape[0], xlb_mean, xub_mean, alpha=alpha, leng_sb=[leng_lb, leng_ub], nu=nu)
    # os.makedirs(outpath, exist_ok=True)
    # sm_filename = f'{outpath}/surrogate_model_for_iter{iterflag}'
    # pickle.dump(sm, open(sm_filename, 'wb'))

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    sm = RandomForestRegressor()
    sm.fit(x, y)
    os.makedirs(outpath, exist_ok=True)
    sm_filename = f'{outpath}/surrogate_model_for_iter{iterflag}'
    pickle.dump(sm, open(sm_filename, 'wb'))


    # perform optimization using the surrogate model
    bestx_sm, besty_sm, x_sm, y_sm = NSGA2.optimization(sm, nInput, nOutput, xlb_mean, xub_mean, pop, gen, crossover_rate, mu, mum)
    D = NSGA2.crowding_distance(besty_sm)
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

    # generate a parameter dataframe for next trial
    for i in range(x_resample.shape[0]):
        outfile = f'{outpath}/paramset_iter{iterflag+1}_trial{i}.csv'
        print('Generating parameter file:', outfile)

        dfi = df_info.copy()
        factors = (x_resample[i, :] - xlb_mean) / (xub_mean - xlb_mean)
        factors[factors<0] = 0.01
        factors[factors>1] = 0.99
        dfi['Factor'] = factors
        dfi['Value'] = factors * (param_upper_bound - param_lower_bound) + param_lower_bound

        # process binded parameters
        dfi = check_and_generate_binded_parameters(dfi, path_CTSM_case)

        # write
        dfi.to_csv(outfile, index=False)



