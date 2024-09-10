# calculate runoff sensitivity to P/T

# T: K
# P: mm/day
# Q: mm/day


import numpy as np
import os, glob, sys, toml
from multiprocessing import Pool
import statsmodels.api as sm
sys.path.append('/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/MOASMO_support')
from run_one_paramset_Derecho import *
from mo_evaluation import *
import cftime

def assign_nan_to_calibperiod(ds_sim_optm, calibstart, calibend, leapcheck=False):
    if True:

        
        # Convert calibstart and calibend to cftime.DatetimeNoLeap if they are not already
        if leapcheck:

            # Convert calibstart and calibend to standard Python datetime objects
            calibstart = pd.to_datetime(calibstart).to_pydatetime()
            calibend = pd.to_datetime(calibend).to_pydatetime()
            
            if not isinstance(calibstart, cftime._cftime.DatetimeNoLeap):
                calibstart = cftime.DatetimeNoLeap(calibstart.year, calibstart.month, calibstart.day)
            if not isinstance(calibend, cftime._cftime.DatetimeNoLeap):
                calibend = cftime.DatetimeNoLeap(calibend.year, calibend.month, calibend.day)

        else:
            # Convert calibstart and calibend to standard Python datetime objects
            calibstart = pd.to_datetime(calibstart)
            calibend = pd.to_datetime(calibend)

        # Create a boolean mask for the time period between calibstart and calibend
        time_mask = (ds_sim_optm['time'].values >= calibstart) & (ds_sim_optm['time'].values <= calibend)

        # Loop through each data variable
        for var_name, variable in ds_sim_optm.data_vars.items():
            if 'time' in variable.dims:
                # Apply the mask and set values to nan for the specified time period
                ds_sim_optm[var_name].loc[dict(time=time_mask)] = np.nan
                
    return ds_sim_optm

def load_and_fill_missing_dates_CAMELSq(file_q):
    df_q = pd.read_csv(file_q, delim_whitespace=True, header=None)
    df_q.columns = ['name', 'year', 'month', 'day', 'Qobs', 'code']
    df_q['date'] = pd.to_datetime(df_q[['year', 'month', 'day']])
    q = df_q['Qobs'].values
    q[q<0] = np.nan
    df_q['Qobs'] = q

    df_q.set_index('date', inplace=True)
    date_range = pd.date_range(start='1980-01-01', end='2014-12-31', freq='D')
    df_reindexed = df_q.reindex(date_range)
    df_reset = df_reindexed.reset_index().rename(columns={'index': 'date'})
    
    return df_reset


def cal_delta(Q, P, T):
    indi = ~np.isnan(Q+P+T)
    if np.sum(indi)/len(indi)<0.6:
        print('Warning! Invalid ratio >0.8')
    Q, P, T = Q[indi], P[indi], T[indi]
    
    dQ = (Q - np.mean(Q)) / np.mean(Q) * 100
    dP = (P - np.mean(P)) / np.mean(P) * 100
    dT = T - np.mean(T)

    return dQ, dP, dT

def Q_sensitivity_to_P_T(dQ, dP, dT, conflevel=0.10):

    # Sample data
    data = {'dQ':dQ, 'dP': dP, 'dT':dT}

    df = pd.DataFrame(data)
    df['dPdT'] = df['dP'] * df['dT']

    # Define independent variables (X) and dependent variable (y)
    X = df[['dP', 'dT', 'dPdT']]
    X = sm.add_constant(X)  # Adds a constant column for the intercept
    y = df['dQ']

    # Perform regression
    model = sm.OLS(y, X).fit()

    # get coefficients
    a = model.params['dP']
    b = model.params['dT']
    c = model.params['dPdT']

    # print(f"a (coefficient for dP) = {a}")
    # print(f"b (coefficient for dT) = {b}")
    # print(f"c (coefficient for dPdT) = {c}")

    # Get the confidence intervals
    conf_intervals = model.conf_int(alpha = conflevel)  # 10% significance level for 5-95% CIs

    # Extract the intervals for dP (a) and dT (b)
    a_interval = conf_intervals.loc['dP'].values
    b_interval = conf_intervals.loc['dT'].values
    c_interval = conf_intervals.loc['dPdT'].values

    # print("Confidence interval for a (dP):", a_interval)
    # print("Confidence interval for b (dT):", b_interval)
    # print("Confidence interval for c (dPdT):", c_interval)

    return [a, b, c, a_interval[0], a_interval[1], b_interval[0], b_interval[1], c_interval[0], c_interval[1]]



def run_trial(params):
    
    folder, configfile, areab = params
    
    outfile_metric = f'{folder}/Q_sensitivity_nocalib.csv'
    if os.path.isfile(outfile_metric):
        print('outfile exists', outfile_metric)
        return

    config = toml.load(configfile)
    
    # inputs
    path_CTSM_base = config['path_CTSM_case']
    ref_streamflow = config['file_Qobs']

    # read observation Q
    date_range = pd.date_range(start='1980-01-01', end='2014-12-31', freq='D') # 1980-2014 CAMELS periods
    df_obs = load_and_fill_missing_dates_CAMELSq(ref_streamflow)
    ds_obs = xr.Dataset()
    ds_obs.coords['time'] = date_range
    Qobs = df_obs['Qobs'].values * 0.0283168 * 60 * 60 * 24 / areab * 1000
    ds_obs['Qobs'] = xr.DataArray(Qobs, dims=('time'))

    ds_obs = ds_obs.where((ds_obs['time.month'] != 2) | (ds_obs['time.day'] != 29), drop=True) # drop Feb-29

    # read simulated CTSM data
    infile = f'{folder}/merged_h0_1951-2019.nc'

    if not os.path.isfile(infile):
        print('infile does not exist:', infile)

    else:
    
        fsurdat = get_parameter_from_Namelist_or_lndin('fsurdat', f'{path_CTSM_base}/user_nl_clm', f'{path_CTSM_base}/Buildconf/clmconf/lnd_in', type='str')
    
        ds_sim = xr.open_dataset(infile)
        ds_sim = ds_sim[['RAIN', 'SNOW', 'TBOT', 'QRUNOFF']]
        ds_sim = ds_sim.load()
        ds_sim['PRECIP'] = ds_sim['RAIN'] + ds_sim['SNOW']
        ds_sim = ds_sim.squeeze()
    
        for v in ['RAIN', 'SNOW', 'PRECIP', 'QRUNOFF']:
            if v in ds_sim.data_vars:
                ds_sim[v] = ds_sim[v] * 60 * 60 * 24
    
        # drop calibration period (nan)
        RUN_STARTDATE = config['RUN_STARTDATE']
        ignore_month = config['ignore_month']
        STOP_OPTION = config['STOP_OPTION']
        STOP_N = config['STOP_N']    
    
        date_start = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=ignore_month)).strftime('%Y-%m-%d')
        if STOP_OPTION == 'nyears':
            date_end = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(years=STOP_N)).strftime('%Y-%m-%d')
        elif STOP_OPTION == 'nmonths':
            date_end = (pd.Timestamp(RUN_STARTDATE) + pd.offsets.DateOffset(months=STOP_N)).strftime('%Y-%m-%d')
        else:
            print(f'STOP_OPTION must be nyears or nmonths. {STOP_OPTION} is not accepted.')
            return
        
        ds_sim = assign_nan_to_calibperiod(ds_sim, date_start, date_end, True)
        ds_obs = assign_nan_to_calibperiod(ds_obs, date_start, date_end, False)
    
        # calculate water year mean
        month_start = 10
        month_end = 9
        years = [1980, 2014]
        
        # CAMELS obs to water year mean
        tarvar = ['Qobs',]
        
        flag = 0
        n1 = years[1]-years[0]
        n2 = len(tarvar)
        
        values = np.nan * np.zeros([n1, n2])
        
        for i in range(n1):
            yi = years[0] + i
            for j in range(n2):
                dij = ds_obs[tarvar[j]].sel(time=slice(f'{yi}-{month_start:02}', f'{yi+1}-{month_end:02}'))
                values[i, j] = dij.mean(dim='time').values
                
        ds_obs_wy = xr.Dataset()
        ds_obs_wy.coords['time'] = pd.date_range(f'{years[0]}-{month_start}', f'{years[1]}-{month_start}', freq='Y')
        
        for i in range(n2):
            ds_obs_wy[tarvar[i]] = xr.DataArray(values[:, i], dims=('time',))
            ds_obs_wy[tarvar[i]].attrs = ds_obs[tarvar[i]].attrs
            
        ds_obs_wy = ds_obs_wy.rename({'Qobs': 'Q', })
        ds_obs = ds_obs.rename({'Qobs': 'Q', })
    
    
        # CAMELS obs to water year mean
        tarvar = ['TBOT', 'QRUNOFF', 'PRECIP']
        n1 = years[1]-years[0]
        n2 = len(tarvar)
        
        values = np.nan * np.zeros([n1, n2])
        
        for i in range(n1):
            yi = years[0] + i
            for j in range(n2):
                dij = ds_sim[tarvar[j]].sel(time=slice(f'{yi}-{month_start:02}', f'{yi+1}-{month_end:02}'))
                values[i, j] = np.squeeze(dij.mean(dim='time').values)
                
        ds_sim_wy = xr.Dataset()
        ds_sim_wy.coords['time'] = pd.date_range(f'{years[0]}-{month_start}', f'{years[1]}-{month_start}', freq='Y')
        
        for i in range(n2):
            ds_sim_wy[tarvar[i]] = xr.DataArray(values[:, i], dims=('time',))
            ds_sim_wy[tarvar[i]].attrs = ds_sim[tarvar[i]].attrs
            
        ds_sim_wy = ds_sim_wy.rename({'QRUNOFF': 'Q', 'TBOT': 'T', 'PRECIP': 'P',})
        ds_sim = ds_sim.rename({'QRUNOFF': 'Q', 'TBOT': 'T', 'PRECIP': 'P',})
    
        # calculate sensitivity
        period = [1980, 2013]
    
        Q = np.squeeze(ds_obs_wy['Q'].sel(time=slice(str(period[0]), str(period[1]))).values)
        P = np.squeeze(ds_sim_wy['P'].sel(time=slice(str(period[0]), str(period[1]))).values)
        T = np.squeeze(ds_sim_wy['T'].sel(time=slice(str(period[0]), str(period[1]))).values)
        dQ, dP, dT = cal_delta(Q, P, T)
        senst_obs = Q_sensitivity_to_P_T(dQ, dP, dT, conflevel=0.10)
    
        Q = np.squeeze(ds_sim_wy['Q'].sel(time=slice(str(period[0]), str(period[1]))).values)
        P = np.squeeze(ds_sim_wy['P'].sel(time=slice(str(period[0]), str(period[1]))).values)
        T = np.squeeze(ds_sim_wy['T'].sel(time=slice(str(period[0]), str(period[1]))).values)
        dQ, dP, dT = cal_delta(Q, P, T)
        senst_sim = Q_sensitivity_to_P_T(dQ, dP, dT, conflevel=0.10)
    
        # save output file
        df_out = pd.DataFrame({'Obs': senst_obs, 'Simu': senst_sim})
        df_out['senst'] = ['P', 'T', 'PT', 'Plow', 'Pup', 'Tlow', 'Tup', 'PTlow', 'PTup']
        
        outfile_metric = f'{folder}/Q_sensitivity_nocalib.csv'
        if not os.path.isfile(outfile_metric):
            print('saving', outfile_metric)
            df_out.to_csv(outfile_metric, index=False)
        else:
            print('outfile exists', outfile_metric)

if __name__ == '__main__':

    infile_info = '/glade/work/guoqiang/CTSM_CAMELS/data_mesh_surf/HillslopeHydrology/CAMELS_level1_basin_info.csv'
    df_info = pd.read_csv(infile_info)
    area = df_info['areaUSGS'].values * 1e6 # m2

    rin = int(sys.argv[1])
    print('processing', rin)
    
    # parallel
    num_processes = 36
    basin_num = 627

    pool = Pool(processes=num_processes)
    
    tasks = []
    for basin in range(basin_num):
        print('basin', basin)
        areab = area[basin]
        configfile = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/configuration/_level1-{basin}_config_MOASMO.toml'
        # folder = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/LongTermSimu/LSEallbasin/level1_{basin}/normKGEr1'
        # folder = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/LongTermSimu/Defa/level1_{basin}'
        # tasks.append((folder, configfile, areab))
        for i in range(rin, rin+1):
            folder = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/LongTermSimu/LSEallbasin/level1_{basin}/normKGEr{i}'
            tasks.append((folder, configfile, areab))
    
    pool.map(run_trial, tasks)
    pool.close()  # Close the pool to prevent any more tasks from being submitted
    pool.join()   # Wait for the worker processes to terminate

    # # serial
    # basin_num = 627
    # for basin in range(basin_num):
    #     print('basin', basin)
    #     areab = area[basin]
    #     configfile = f'/glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO_bigrange/configuration/_level1-{basin}_config_MOASMO.toml'
    #     # folder = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/LongTermSimu/LSEallbasin/level1_{basin}/normKGEr1'
    #     folder = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/LongTermSimu/Defa/level1_{basin}'     
    #     run_trial((folder, configfile, areab))

