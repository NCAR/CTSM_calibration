# this is a non-standard evaluation script and cannot be directly used in the general workflow
# functions are written for special use
# this script contains hard-coded settings including file names


from mo_evaluation import *

# turn off all warnings (not always necessary)
import warnings
warnings.filterwarnings("ignore")

def evaluate_allCAMELS(outfile_metric, CTSMfilelist, fsurdat, date_start, date_end):

    ref_streamflow = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/CAMLES_q_split_nest/All_CAMELS_Q_postprocess.csv'

    ######## default variable names
    clm_q_name = 'QRUNOFF'  # default runoff variable name
    clm_q_sdim = 'lndgrid'  # spatial dim name
    ref_q_name = 'Runoff_cms'
    ref_q_date = 'Date'

    ########################################################################################################################
    # load CTSM streamflow (m3/s)
    ds_simu = main_read_CTSM_streamflow(fsurdat, CTSMfilelist, date_start, date_end, clm_q_name, clm_q_sdim)
    ds_simu = ds_simu.transpose('time', clm_q_sdim)

    ########################################################################################################################
    # load CAMELS observation streamflow (m3/s)
    df_q_obs = pd.read_csv(ref_streamflow)
    ds_q_obs = xr.Dataset()
    ds_q_obs.coords['time'] = pd.to_datetime(df_q_obs[ref_q_date].values)
    ds_q_obs[ref_q_name] = xr.DataArray(df_q_obs.values[:, 1:], dims=['time', clm_q_sdim]) # flexible time

    ########################################################################################################################
    # evaluation

    ds_q_obs = ds_q_obs.sel(time=ds_q_obs.time.isin(ds_simu.time))
    ds_simu = ds_simu.sel(time=ds_simu.time.isin(ds_q_obs.time))

    d1 = ds_q_obs[ref_q_name].values
    d2 = ds_simu[clm_q_name].values
    d1[d1 < 0] = np.nan
    d2[d2 < 0] = np.nan
    ds_q_obs[ref_q_name].values = d1
    ds_simu[clm_q_name].values = d2

    kge_q_all = []
    for i in range(d1.shape[1]):
        kge_q = get_modified_KGE(obs=d1[:,i], sim=d2[:,i])
        kge_q_all.append(kge_q)

    d1 = ds_q_obs[ref_q_name].groupby('time.month').mean().values
    d2 = ds_simu[clm_q_name].groupby('time.month').mean().values
    maxabserror_q_all = []
    for i in range(d1.shape[1]):
        maxabserror_q = get_max_abs_error(d1[:, i], d2[:, i])
        maxabserror_q_all.append(maxabserror_q)

    kge_q_median = np.nanmedian(kge_q_all)
    maxabserror_q_median = np.nanmedian(maxabserror_q_all)
    print(f'Evaluation result: kge_q_median={kge_q_median}, maxabserror_q_median={maxabserror_q_median}')

    ########################################################################################################################
    # write objective functions to file.
    # metrics will be minimized during optimization
    dfout = pd.DataFrame([[1 - kge_q_median, maxabserror_q_median]], columns=['metric1', 'metric2'])
    dfout.to_csv(outfile_metric, index=False)



