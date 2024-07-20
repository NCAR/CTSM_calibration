# multi-objective evaluation of CTSM model outputs
# Guoqiang Tang
# Note: Here the average of outputs is used to compare to observed streamflow


import numpy as np
import xarray as xr
import pandas as pd
import sys, glob, os, subprocess

# turn off all warnings (not always necessary)
import warnings
warnings.filterwarnings("ignore")

########################################################################################################################
# define functions for calculating metrics

def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    ind = np.array([bind.get(itm, np.nan) for itm in a])
    ind1 = np.where(~np.isnan(ind))[0]
    ind2 = ind[ind1]
    return ind1.astype(int), ind2.astype(int) # None can be replaced by any other "not in b" value

def get_modified_KGE(obs,sim):
    ind = (~np.isnan(obs)) & (~np.isnan(sim))
    obs = obs[ind]
    sim = sim[ind]

    try:
        sd_sim=np.std(sim, ddof=1)
        sd_obs=np.std(obs, ddof=1)
        m_sim=np.mean(sim)
        m_obs=np.mean(obs)
        r=(np.corrcoef(sim,obs))[0,1]
        relvar=(float(sd_sim)/float(m_sim))/(float(sd_obs)/float(m_obs))
        bias=float(m_sim)/float(m_obs)
        kge=1.0-np.sqrt((r-1)**2 +(relvar-1)**2 + (bias-1)**2)
    except:
        kge = np.nan

    return kge

def get_CC(obs,sim):
    ind = (~np.isnan(obs)) & (~np.isnan(sim))
    obs = obs[ind]
    sim = sim[ind]
    cc = np.corrcoef(obs, sim)[0, 1]
    return cc

def get_RMSE(obs,sim):
    rmse = np.sqrt(np.nanmean(np.power((sim - obs),2)))
    return rmse

def get_mean_error(obs, sim):
    bias_err = np.nanmean(sim - obs)
    abs_err = np.nanmean(np.absolute(sim - obs))
    return bias_err, abs_err

def get_mean_abs_error(obs, sim):
    mean_abs_err = np.nanmean(np.absolute(sim - obs))
    return mean_abs_err

def get_max_abs_error(d1, d2):
    return np.nanmax(np.abs(d1-d2))

def get_nse(obs, sim):     
    return 1-(np.nansum((sim-obs)**2)/np.nansum((obs-np.nanmean(obs))**2))


########################################################################################################################
# define functions for reading CTSM outputs

def main_read_CTSM_streamflow(fsurdat, CTSMfilelist, date_start, date_end, clm_q_name):
    ########################################################################################################################
    # read files
    ds_simu = xr.open_mfdataset(CTSMfilelist)
    ds_simu = ds_simu[[clm_q_name]]

    if date_start == 'default' or date_end == 'default':
        print(
            'Either date_start or date_end is default. Evaluation period will be the overlapped period of referene data and simulations')
    else:
        ds_simu = ds_simu.sel(time=slice(date_start, date_end))

    ds_simu = ds_simu.load()

    # change time format
    ds_simu['time'] = ds_simu.indexes['time'].to_datetimeindex()

    ########################################################################################################################
    # get the area of a basin to convert the unit of QRUNOFF from mm/s to m3/s

    with xr.open_dataset(fsurdat) as ds_surdat:
        area = ds_surdat.AREA.values

    # calculate streamflow: although mean is used, for Sean's setting, only one basin should be allowed effective in the calibration
    # streamflow? Use mean for this test
    ds_simu[clm_q_name].values = (ds_simu[clm_q_name].values / 1000) * (area * 1e6)  # raw q: mm/s; raw area km2; target: m3/s

    return ds_simu


########################################################################################################################
# define functions for reading CAMELS data

def read_CAMELS_Q(file_Qobs):
    df_q_in = pd.read_csv(file_Qobs, delim_whitespace=True, header=None)
    years = df_q_in[1].values
    months = df_q_in[2].values
    days = df_q_in[3].values
    dates = [f'{years[i]}-{months[i]:02}-{days[i]:02}' for i in range(len(years))]
    dates = pd.to_datetime(dates)
    q_obs = df_q_in[4].values * 0.028316847  # cfs to cms
    q_obs[q_obs < 0] = -9999.0
    df_q_out = pd.DataFrame({'Date': dates, 'Runoff_cms': q_obs})
    
    # fill possible missing values
    df_q_out.set_index('Date', inplace=True)
    date_range = pd.date_range(start=dates[0], end=dates[-1], freq='D')
    df_q_out = df_q_out.reindex(date_range)
    df_q_out.fillna(-9999, inplace=True)
    df_q_out.reset_index(inplace=True)
    df_q_out = df_q_out.rename(columns={'index': 'Date'})

    return df_q_out

def read_CAMELS_Q_and_to_xarray(ref_streamflow, ref_q_date, ref_q_name):
    ########################################################################################################################
    # load observation streamflow
    print('Use streamflow reference file:', ref_streamflow)
    #df_q_obs = pd.read_csv(ref_streamflow)
    df_q_obs = read_CAMELS_Q(ref_streamflow)
    ds_q_obs = xr.Dataset()
    ds_q_obs.coords['time'] = pd.to_datetime(df_q_obs[ref_q_date].values)
    ds_q_obs[ref_q_name] = xr.DataArray(df_q_obs[ref_q_name].values, dims=['time']) # flexible time
    for i in range(10000):
        coli = ref_q_name + str(i)
        if coli in df_q_obs.columns:
            ds_q_obs[coli] = xr.DataArray(df_q_obs[coli].values, dims=['time'])  # flexible time
        else:
            break
    return ds_q_obs


def add_upstream_flow(add_flow_file, ds_simu, ref_q_date, ref_q_name, clm_q_name):
    ########################################################################################################################
    # add upstream flows to simulated streamflow

    add_flow_file = [f for f in add_flow_file.split(',') if len(f)>0]
    if len(add_flow_file) > 0:
        add_flow_file2 = []
        for f in add_flow_file:
            if not os.path.isfile(f):
                print('File does not exist:', f)
                print('Remove it from add flow file list')
            else:
                add_flow_file2.append(f)
        add_flow_file = add_flow_file2

    if len(add_flow_file) > 0:
        print('Flow files will be added to the incremental downstream basin:', add_flow_file)
        q_dd = np.zeros(len(ds_simu.time))
        num = np.zeros(len(ds_simu.time))
        time0 = ds_simu.time.values
        for i in range(len(add_flow_file)):
            df_addi = read_CAMELS_Q(add_flow_file[i])
            # df_addi = pd.read_csv(add_flow_file[i])
            timei = pd.to_datetime(df_addi[ref_q_date].values)
            ind1, ind2 = ismember(np.array(timei), time0)
            q_dd[ind2] = q_dd[ind2] + df_addi[ref_q_name].values[ind1]
            num[ind2] = num[ind2] + 1
        q_dd[num==0] = np.nan

        ds_simu[clm_q_name].values = ds_simu[clm_q_name].values + q_dd
        ratio = np.sum(~np.isnan(ds_simu[clm_q_name].values)) / len(ds_simu[clm_q_name].values)
        if ratio < 0.5:
            print('Warning!!!')
        print(f'The valid ratio of simulated streamflow is {ratio} after add upstream flow')

    return ds_simu


def mo_evaluate_return_many_metrics(outfile_metric, CTSMfilelist, fsurdat, date_start, date_end, ref_streamflow, add_flow_file=''):

    ######## default variable names
    clm_q_name = 'QRUNOFF' # default runoff variable name
    clm_q_sdim = 'lndgrid' # spatial dim name
    ref_q_name = 'Runoff_cms'
    ref_q_date = 'Date'

    ########################################################################################################################
    # load CTSM streamflow (m3/s)
    ds_simu = main_read_CTSM_streamflow(fsurdat, CTSMfilelist, date_start, date_end, clm_q_name)
    ds_simu = ds_simu.mean(dim=clm_q_sdim, skipna=True)

    ########################################################################################################################
    # load CAMELS observation streamflow (m3/s)
    ds_q_obs = read_CAMELS_Q_and_to_xarray(ref_streamflow, ref_q_date, ref_q_name)

    ########################################################################################################################
    # add upstream flows to simulated streamflow
    ds_simu = add_upstream_flow(add_flow_file, ds_simu, ref_q_date, ref_q_name, clm_q_name)

    ########################################################################################################################
    # evaluation

    # match sim-obs time and data
    ds_q_obs = ds_q_obs.sel(time=ds_q_obs.time.isin(ds_simu.time))
    ds_simu = ds_simu.sel(time=ds_simu.time.isin(ds_q_obs.time))

    d1 = ds_q_obs[ref_q_name].values
    d2 = ds_simu[clm_q_name].values
    d1[d1<0] = np.nan
    d2[d2<0] = np.nan
    ds_q_obs[ref_q_name].values = d1
    ds_simu[clm_q_name].values = d2

    # calculate kge'
    kge_q = get_modified_KGE(obs=d1, sim=d2)

    # calculate mean daily MAE
    abs_err = get_mean_abs_error(obs=d1, sim=d2)
    n_abs_err = abs_err / np.nanmean(d1)

    # calculate nse
    nse = get_nse(obs=d1, sim=d2)

    # calculate cc
    cc = get_CC(obs=d1, sim=d2)

    # get rmse
    rmse = get_RMSE(obs=d1, sim=d2)

    # peak flow / low flow: 90% or 10% threshold mae
    indtmp = d1>np.nanpercentile(d1, 90)
    q90_mae = get_mean_abs_error(obs=d1[indtmp], sim=d2[indtmp])

    indtmp = d1>np.nanpercentile(d1, 10)
    q10_mae = get_mean_abs_error(obs=d1[indtmp], sim=d2[indtmp])

    # peak flow / low flow: 90% or 10% threshold mae of duration days
    threshold = np.nanpercentile(d1, 90)
    q90_days_err = np.abs( np.sum(d1>threshold) - np.sum(d2>threshold))

    threshold = np.nanpercentile(d1, 10)
    q10_days_err = np.abs( np.sum(d1<threshold) - np.sum(d2<threshold))

    # mae from thresholds
    indtmp = d1>=np.nanpercentile(d1, 50)
    ge_q50_mae = get_mean_abs_error(obs=d1[indtmp], sim=d2[indtmp])

    indtmp = d1>=np.nanpercentile(d1, 25)
    ge_q25_mae = get_mean_abs_error(obs=d1[indtmp], sim=d2[indtmp])

    indtmp = d1>=np.nanpercentile(d1, 75)
    ge_q75_mae = get_mean_abs_error(obs=d1[indtmp], sim=d2[indtmp])
    
    # calculate log kge'
    indtmp = (d1>0)&(d2>0)
    kge_log_q = get_modified_KGE(obs=np.log(d1[indtmp]), sim=np.log(d2[indtmp]))

    # calculate summer kge' / mae
    summer = [6, 7, 8]
    winter = [12, 1, 2]
    months = ds_simu.time.dt.month.values
    indsummer = (months==6)|(months==7)|(months==8)
    indwinter = (months==12)|(months==1)|(months==2)
    indspring = (months==3)|(months==4)|(months==5)
    indautumn = (months==9)|(months==10)|(months==11)
    kge_summer = get_modified_KGE(obs=d1[indsummer], sim=d2[indsummer])
    kge_winter = get_modified_KGE(obs=d1[indwinter], sim=d2[indwinter])
    kge_spring = get_modified_KGE(obs=d1[indspring], sim=d2[indspring])
    kge_autumn = get_modified_KGE(obs=d1[indautumn], sim=d2[indautumn])

    mae_summer = get_mean_abs_error(obs=d1[indsummer], sim=d2[indsummer])
    mae_winter = get_mean_abs_error(obs=d1[indwinter], sim=d2[indwinter])
    mae_spring = get_mean_abs_error(obs=d1[indspring], sim=d2[indspring])
    mae_autumn = get_mean_abs_error(obs=d1[indautumn], sim=d2[indautumn])

    # calculate max monthly error
    d1_monthly = ds_q_obs[ref_q_name].groupby('time.month').mean().values
    d2_monthly = ds_simu[clm_q_name].groupby('time.month').mean().values
    maxabserror_q = get_max_abs_error(d1_monthly, d2_monthly)
    n_maxabserror_q = maxabserror_q / np.nanmean(d1_monthly)

########################################################################################################################
    # write objective functions to file.
    # metrics will be minimized during optimization
    # dfout = pd.DataFrame([[1 - kge_q, maxabserror_q]], columns=['metric1', 'metric2'])
    dfout = pd.DataFrame([[kge_q, abs_err, n_abs_err, nse, cc, rmse, maxabserror_q, n_maxabserror_q, q90_mae, q10_mae, 
                           q90_days_err, q10_days_err, kge_log_q, 
                           kge_summer, kge_winter, kge_spring, kge_autumn,
                           mae_summer, mae_winter, mae_spring, mae_autumn, 
                           ge_q25_mae, ge_q50_mae, ge_q75_mae]], 
                         columns=['kge', 'mae', 'n_mae', 'nse', 'cc', 'rmse', 'max_mon_abs_err', 'n_max_mon_abs_err', 'q90_mae', 'q10_mae', 
                                 'q90_days_err', 'q10_days_err', 'kge_log_q', 
                                 'kge_summer', 'kge_winter', 'kge_spring', 'kge_autumn', 
                                 'mae_summer', 'mae_winter', 'mae_spring', 'mae_autumn',
                                 'ge_q25_mae', 'ge_q50_mae', 'ge_q75_mae'])
    dfout.to_csv(outfile_metric, index=False)



def mo_evaluate(outfile_metric, CTSMfilelist, fsurdat, date_start, date_end, ref_streamflow, add_flow_file=''):

    ######## default variable names
    clm_q_name = 'QRUNOFF' # default runoff variable name
    clm_q_sdim = 'lndgrid' # spatial dim name
    ref_q_name = 'Runoff_cms'
    ref_q_date = 'Date'

    ########################################################################################################################
    # load CTSM streamflow (m3/s)
    ds_simu = main_read_CTSM_streamflow(fsurdat, CTSMfilelist, date_start, date_end, clm_q_name)
    ds_simu = ds_simu.mean(dim=clm_q_sdim, skipna=True)

    ########################################################################################################################
    # load CAMELS observation streamflow (m3/s)
    ds_q_obs = read_CAMELS_Q_and_to_xarray(ref_streamflow, ref_q_date, ref_q_name)

    ########################################################################################################################
    # add upstream flows to simulated streamflow
    ds_simu = add_upstream_flow(add_flow_file, ds_simu, ref_q_date, ref_q_name, clm_q_name)

    ########################################################################################################################
    # evaluation

    # match sim-obs time and data
    ds_q_obs = ds_q_obs.sel(time=ds_q_obs.time.isin(ds_simu.time))
    ds_simu = ds_simu.sel(time=ds_simu.time.isin(ds_q_obs.time))

    d1 = ds_q_obs[ref_q_name].values
    d2 = ds_simu[clm_q_name].values
    d1[d1<0] = np.nan
    d2[d2<0] = np.nan
    ds_q_obs[ref_q_name].values = d1
    ds_simu[clm_q_name].values = d2

    # calculate kge'
    kge_q = get_modified_KGE(obs=d1, sim=d2)

    # calculate log kge'
    # d1[d1<0.001] = 0.001
    # d2[d2<0.001] = 0.001
    # kge_logq = get_modified_KGE(obs=np.log(d1), sim=np.log(d2))

    # calculate mean daily MAE
    abs_err = get_mean_abs_error(obs=d1, sim=d2)

    # calculate max monthly error
    d1_monthly = ds_q_obs[ref_q_name].groupby('time.month').mean().values
    d2_monthly = ds_simu[clm_q_name].groupby('time.month').mean().values
    maxabserror_q = get_max_abs_error(d1_monthly, d2_monthly)

    # calculate RMSE
    # rmse_q = get_RMSE(obs=ds_q_obs[ref_q_name].values, sim=ds_simu[clm_q_name].values)

    print(f'Evaluation result: kge_q={kge_q}, maxabserror_q={maxabserror_q}, MAE={abs_err}')

    ########################################################################################################################
    # write objective functions to file.
    # metrics will be minimized during optimization
    # dfout = pd.DataFrame([[1 - kge_q, maxabserror_q]], columns=['metric1', 'metric2'])
    dfout = pd.DataFrame([[abs_err, maxabserror_q]], columns=['metric1', 'metric2'])
    dfout.to_csv(outfile_metric, index=False)

