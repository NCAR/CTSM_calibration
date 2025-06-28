# Guoqiang Tang
# Note: Here the average of outputs is used to compare to observed streamflow


import numpy as np
import datetime
import xarray as xr
import pandas as pd
import sys, glob, os, re, subprocess

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
    sim[sim<0] = np.nan
    obs[obs<0] = np.nan
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

def xmlquery_output(pathCTSM, keyword):
    os.chdir(pathCTSM)
    out = subprocess.run(f'./xmlquery {keyword}', shell=True, capture_output=True)
    out = out.stdout.decode().strip().split(' ')[-1]
    return out
    
########################################################################################################################
# define functions for reading CAMELS data

def read_CAMELS_Q(file_Qobs):
    df_q_in = pd.read_csv(file_Qobs, delim_whitespace=True, header=None)
    years = df_q_in[1].values
    months = df_q_in[2].values
    days = df_q_in[3].values
    dates = [f'{years[i]}-{months[i]:02}-{days[i]:02}' for i in range(len(years))]
    q_obs = df_q_in[4].values * 0.028316847  # cfs to cms
    q_obs[q_obs < 0] = -9999.0
    df_q_out = pd.DataFrame({'Date': dates, 'Runoff_cms': q_obs})
    return df_q_out

def read_CAMELS_Q_and_to_xarray(ref_streamflow, ref_q_date, ref_q_name):
    ########################################################################################################################
    # load observation streamflow
    print('Use streamflow reference file:', ref_streamflow)
    df_q_obs = pd.read_csv(ref_streamflow)
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


# main
if __name__ == '__main__':

    ########################################################################################################################
    # input arguments
    # future improvements can use more advanced argparse. currently, only sys is used

    ######## required input arguments
    outfile_statistics = sys.argv[1]
    pathCTSM = sys.argv[2]  # CTSM model case
    date_start = sys.argv[3]  # '%Y-%m-%d' or 'default'. if default, the date from control_file_summa will be used
    date_end = sys.argv[4]

    objfunc = sys.argv[5]

    # reference files (streamflow, snow cover). if a file cannot be found, it won't be inclulded in the calibration
    ref_streamflow = sys.argv[6]

    # add_flow_file. sometimes upstream flow needs to be added to the incremental downstream area runoff
    add_flow_file = sys.argv[7]

    ######## default variable names
    clm_q_name = 'DWroutedRunoff'
    ref_q_name = 'Runoff_cms'
    ref_q_date = 'Date'
    keyword = ".clm2.h1."
    outlet_segId = None
    shift = -1 # for CTSM-CAMELS setting, shift=-1 is the best for mizuroute model-obs match

    ########################################################################################################################
    # load mizuroute streamflow (m3/s)
    path_archive = xmlquery_output(pathCTSM, 'DOUT_S_ROOT')
    infilelist_mizuroute = glob.glob(f'{path_archive}/mizuroute/sflow*.nc')
    infilelist_mizuroute.sort()

    try:
        ds_simu = xr.open_mfdataset(infilelist_mizuroute)

        ds_simu = ds_simu.sel(time=slice(date_start, date_end))
        ds_simu['time'] = ds_simu.indexes['time'].to_datetimeindex()

        if outlet_segId == None:
            # select one seg with the largest mean flow
            ind = np.argmax(ds_simu[clm_q_name].mean(dim='time').values)
            ds_simu = ds_simu.isel(seg=ind)
        else:
            ds_simu = ds_simu.isel(seg=outlet_segId)

        if shift != 0:
            ds_simu = ds_simu.shift(time=shift)

        ds_simu = ds_simu[[clm_q_name]].load()

    except:
        print('Fail to read CTSMfilelist:', infilelist_mizuroute)
        sys.exit(0)

    ########################################################################################################################
    # load CAMELS observation streamflow (m3/s)
    ds_q_obs = read_CAMELS_Q_and_to_xarray(ref_streamflow, ref_q_date, ref_q_name)

    ########################################################################################################################
    # evaluation

    # match sim-obs time and data
    ds_q_obs = ds_q_obs.sel(time=ds_q_obs.time.isin(ds_simu.time))
    ds_simu = ds_simu.sel(time=ds_simu.time.isin(ds_q_obs.time))

    d1 = ds_q_obs[ref_q_name].values
    d2 = ds_simu[clm_q_name].values
    d1[d1 < 0] = np.nan
    d2[d2 < 0] = np.nan
    ds_q_obs[ref_q_name].values = d1
    ds_simu[clm_q_name].values = d2

    # calculate kge'
    kge_q = get_modified_KGE(obs=d1, sim=d2)

    # calculate rmse
    rmse_q = get_RMSE(obs=ds_q_obs[ref_q_name].values, sim=ds_simu[clm_q_name].values)

    # calculate mean daily MAE
    abs_err = get_mean_abs_error(obs=d1, sim=d2)

    # calculate max monthly error
    d1_monthly = ds_q_obs[ref_q_name].groupby('time.month').mean().values
    d2_monthly = ds_simu[clm_q_name].groupby('time.month').mean().values
    maxabserror_q = get_max_abs_error(d1_monthly, d2_monthly)

    mean_2error = (abs_err + maxabserror_q) / 2
    print(f'Evaluation result: kge_q={kge_q}, maxabserror_q={maxabserror_q}, MAE={abs_err}, mean_2error={mean_2error}')

    ########################################################################################################################
    # write metric to file
    stat_lines = np.array([f'{kge_q:.6f}\t#streamflow_KGE\n',
                           f'{rmse_q:.6f}\t#streamflow_RMSE\n',
                           f'{abs_err:.6f}\t#streamflow_abserr\n',
                           f'{maxabserror_q:.6f}\t#streamflow_maxmontherr\n',
                           f'{mean_2error:.6f}\t#streamflow_mean2err\n'])

    if objfunc == 'kge':
        stat_lines = stat_lines
    elif objfunc == 'mean_mae_mme':
        stat_lines = stat_lines[[4, 0, 1, 2, 3]]
    elif objfunc == 'rmse':
        stat_lines = stat_lines[[1, 0, 2, 3, 4]]
    elif objfunc == 'maxmontherr':
        stat_lines = stat_lines[[3, 0, 1, 2, 4]]
    else:
        sys.exit(f'Unknown objfunc: {objfunc}')

    with open(outfile_statistics, 'w+') as f:
        for l in stat_lines:
            f.write(l)

