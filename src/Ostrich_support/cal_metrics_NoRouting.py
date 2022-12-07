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


def get_RMSE(obs,sim):
    sim[sim<0] = np.nan
    obs[obs<0] = np.nan
    rmse = np.sqrt(np.nanmean(np.power((sim - obs),2)))
    return rmse

def get_mean_error(obs,sim):
    bias_err = np.nanmean(sim - obs)
    abs_err = np.nanmean(np.absolute(sim - obs))
    return bias_err, abs_err

def get_month_mean_flow(obs,sim,sim_time):
    month = [dt.month for dt in sim_time]

    data = {'sim':sim, 'obs':obs, 'month':month} 
    df = pd.DataFrame(data, index = sim_time)
    
    gdf = df.groupby(['month'])
    sim_month_mean = gdf.aggregate({'sim':np.nanmean})
    obs_month_mean = gdf.aggregate({'obs':np.nanmean})
    return obs_month_mean, sim_month_mean

def get_target_archive_files_from_starchive(pathCTSM, keyword):
    # get the list of archived files of the latest model run
    # # settings
    # pathCTSM = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/CAMELS_LumpCalib'
    # keyword = ".clm2.h1."
    # find files
    st_archive_files = glob.glob(f'{pathCTSM}/st_archive.*')
    st_archive_files.sort()
    st_archive_files = st_archive_files[-1]
    filelist = []
    print('Getting simulaiton outputs from CTSM model case path ...')
    print('pathCTSM:', pathCTSM)
    print('keyword:', keyword)
    with open(st_archive_files, 'r')  as f:
        for line in f:
            if line.startswith('moving') or line.startswith('copying'):
                if keyword in line:
                    file = line.split(' to ')[-1].strip()
                    if os.path.isfile(file):
                        print('Append to file list:', file)
                        filelist.append(file)
                    else:
                        sys.exit(f'File does not exist: {file}')
    return filelist


def get_target_archive_files_from_archivefolder(pathCTSM, keyword):
    # get the list of archived files of the latest model run
    cwd = os.getcwd()
    os.chdir(pathCTSM)
    out = subprocess.run('./xmlquery DOUT_S_ROOT', shell=True, capture_output=True)
    DOUT_S_ROOT = out.stdout.decode().strip().split(' ')[-1]
    os.chdir(cwd)
    filelist = glob.glob(f'{DOUT_S_ROOT}/lnd/hist/*{keyword}*')
    filelist.sort()
    return filelist


# main
if __name__ == '__main__':

    ########################################################################################################################
    # input arguments
    # future improvements can use more advanced argparse. currently, only sys is used

    ######## required input arguments
    outfile_statistics = sys.argv[1]
    pathCTSM = sys.argv[2] # CTSM model case
    date_start = sys.argv[3] # '%Y-%m-%d' or 'default'. if default, the date from control_file_summa will be used
    date_end = sys.argv[4]

    # reference files (streamflow, snow cover). if a file cannot be found, it won't be inclulded in the calibration
    ref_streamflow = sys.argv[5]

    ######## default variable names
    clm_q_name = 'QRUNOFF' # default runoff variable name
    clm_q_sdim = 'lndgrid' # spatial dim name
    ref_q_name = 'Runoff_cms'
    ref_q_date = 'Date'
    keyword = ".clm2.h1."

    ########################################################################################################################
    # read files
    CTSMfilelist = get_target_archive_files_from_archivefolder(pathCTSM, keyword)
    ds_simu = xr.open_mfdataset(CTSMfilelist)
    ds_simu = ds_simu[[clm_q_name]]

    if date_start == 'default' or date_end == 'default':
        print('Either date_start or date_end is default. Evaluation period will be the overlapped period of referene data and simulations')
    else:
        ds_simu = ds_simu.sel(time=slice(date_start, date_end))

    ds_simu = ds_simu.load()

    # change time format
    ds_simu['time'] = ds_simu.indexes['time'].to_datetimeindex()

    ########################################################################################################################
    # get the area of a basin to convert the unit of QRUNOFF from mm/s to m3/s

    # elementArea is in radians^2, not real area
    # cwd = os.getcwd()
    # os.chdir(pathCTSM)
    # out = subprocess.run('./xmlquery LND_DOMAIN_MESH', shell=True, capture_output=True)
    # LND_DOMAIN_MESH = out.stdout.decode().strip().split(' ')[-1]
    # os.chdir(cwd)

    # get surface data file from user_nl_clm or lnd_in. user_nl_clm may not be reliable because it may not contain this file
    # file = f'{pathCTSM}/user_nl_clm'
    file = f'{pathCTSM}/Buildconf/clmconf/lnd_in'
    fsurdat = ''
    with open(file, 'r') as f:
        for line in f:
            line=line.strip()
            if line.startswith('fsurdat'):
                fsurdat = line.split('=')[-1].strip()
                fsurdat = fsurdat.replace('\'','')

    if not os.path.isfile(fsurdat):
        sys.exit(f'File not found! fsurdat: {fsurdat}')

    with xr.open_dataset(fsurdat) as ds_surdat:
        area = ds_surdat.AREA.values


    # calculate streamflow: although mean is used, for Sean's setting, only one basin should be allowed effective in the calibration
    # streamflow? Use mean for this test
    ds_simu[clm_q_name].values = (ds_simu[clm_q_name].values / 1000) * (area * 1e6) # raw q: mm/s; raw area km2; target: m3/s
    ds_simu = ds_simu.mean(dim=clm_q_sdim, skipna=True)

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

    ########################################################################################################################
    # evaluation

    ds_q_obs = ds_q_obs.sel(time=ds_q_obs.time.isin(ds_simu.time))
    ds_simu = ds_simu.sel(time=ds_simu.time.isin(ds_q_obs.time))

    kge_q = get_modified_KGE(obs=ds_q_obs[ref_q_name].values, sim=ds_simu[clm_q_name].values)
    rmse_q = get_RMSE(obs=ds_q_obs[ref_q_name].values, sim=ds_simu[clm_q_name].values)

    ########################################################################################################################
    # write metric to file
    with open(outfile_statistics, 'w+') as f:
        f.write(f'{kge_q:.6f}\t#streamflow_KGE\n')
        f.write(f'{rmse_q:.6f}\t#streamflow_RMSE\n')


