# Decide the periods of calibration and validation using different methods

import pandas as pd
import numpy as np
import os, sys, subprocess


def read_raw_CAMELS_Q_to_df(infile_q):
    # read streamflow fom raw CAMELS text files, and change column names
    df_q = pd.read_csv(infile_q, delim_whitespace=True, header=None)
    df_q.columns = ['name', 'year', 'month', 'day', 'Qobs', 'code']
    df_q['date'] = pd.to_datetime(df_q[['year', 'month', 'day']])
    q = df_q['Qobs'].values
    q[q<0] = np.nan
    df_q['Qobs'] = q
    return df_q


def get_tmean_series_masked_by_q(inpath_camels_data, id):
    # load prcp/tmean

    infilei = f'{inpath_camels_data}/basin_mean_forcing/nldas/all/{id:08}_lump_nldas_forcing_leap.txt'
    df_met = pd.read_csv(infilei, skiprows=3, delim_whitespace=True)
    df_met = df_met.rename(columns={'Mnth': 'Month'})
    df_met['date'] = pd.to_datetime(df_met[['Year', 'Month', 'Day']])
    df_met['Tmean(C)'] = (df_met['Tmin(C)'].values + df_met['Tmax(C)'].values) / 2

    # load q
    infilei_q = f'{inpath_camels_data}/usgs_streamflow/{id:08}_streamflow_qc.txt'
    df_q = read_raw_CAMELS_Q_to_df(infilei_q)
    df_all = df_q.merge(df_met, how='outer', on='date')
    df_all = df_all.sort_values(by='date')

    # choose a variable to decide periods
    data = df_all['Tmean(C)'].values
    q = df_all['Qobs'].values
    data[np.isnan(q)] = np.nan # ensure period is consistent with streamflow
    date = df_all['date'].values

    return data, date

def cal_shift_annual_mean(data, date, startmonth=1):
    # startmonth: the start of a year (e.g., 1 or 10)
    years = np.unique(pd.DatetimeIndex(date).year.values)
    values = np.nan * np.zeros(len(years))
    for i in range(len(years)):
        yeari = years[i]
        date_start = pd.Timestamp(f'{yeari}-{startmonth}-1')
        date_end = date_start + pd.offsets.DateOffset(years=1) - pd.offsets.DateOffset(hours=1)
        datai = data[(date>=date_start) & (date<=date_end)]
        if np.sum(~np.isnan(datai))/len(datai) >= 0.9:
            values[i] = np.nanmean(datai)
    return years, values


def time_series_anomaly_analysis(data, date, startmonth=1, periodlength=5, window=5):
    # startmonth: the start of a year (e.g., 1 or 10)
    # window: years of rolling mean
    # calculate period segmental mean of a time series
    years, values = cal_shift_annual_mean(data, date, startmonth)
    # calculate rolling mean
    values_rolling = np.squeeze(pd.DataFrame(values).rolling(window).mean().values)
    # calculate period mean
    values_anomaly = values_rolling - np.nanmean(values)
    period_stat = []
    for i in range(0, len(values_anomaly)-periodlength):
        datei1 = f'{years[i]}-{startmonth}-01'
        datei2 = (pd.Timestamp(f'{years[i+periodlength]}-{startmonth}-01') - pd.offsets.DateOffset(hours=1)).strftime('%Y-%m-%d')
        datai = np.mean(values_anomaly[i:i+periodlength])
        period_stat.append([datei1, datei2, datai])
    period_stat = pd.DataFrame(period_stat, columns=['date_start', 'date_end', 'mean'])
    # sort stat from low to high
    period_stat = period_stat[~np.isnan(period_stat['mean'].values)]
    period_stat = period_stat.iloc[period_stat['mean'].values.argsort()]
    period_stat.index = np.arange(len(period_stat))
    return period_stat

def get_most_extreme_periods(period_stat):
    period_extreme = pd.DataFrame()
    indmin = period_stat['mean'].values.argmin()
    period_extreme = pd.concat([period_extreme, period_stat[indmin:indmin + 1]])
    indmax = period_stat['mean'].values.argmax()
    period_extreme = pd.concat([period_extreme, period_stat[indmax:indmax + 1]])
    period_extreme.index = ['min', 'max']
    return period_extreme


def calib_period_Beginning(data, date, calibyears, validratio, trial_start_date):
    # find the period from the beginning of the series

    # df_q: dataframe of Q. It must contain two columns: date and Qobs
    # calibyears: int, number of calibration years
    # validratio: the ratio of valid sample numbers during the selected period
    # trial_start_date: e.g., 2000-10-01. The function will find a start period >=2000 which also meets validratio.
    # The year trial_start_date to trial_start_date+1 must have valid samples >= validratio

    print('Decide calibration period')

    # df_q = read_raw_CAMELS_Q_to_df(infile_q)

    # decide the trial start date
    trial_start_date = pd.Timestamp(trial_start_date)
    if pd.Timestamp(trial_start_date) > date[-1]:
        sys.exit('Error!!! trial_start_date is too large!')

    flag = True
    while flag:
        if pd.Timestamp(trial_start_date) < date[0]:
            trial_start_date = trial_start_date + pd.offsets.DateOffset(years=1)
        else:
            trial_start_date2 = trial_start_date + pd.offsets.DateOffset(years=1)
            index = (date >= trial_start_date) & (date <= trial_start_date2)
            if np.sum(index) > 180:
                qtrial = data[index]
                if np.sum(qtrial>=0)/len(qtrial) >= validratio:
                    flag = False
                else:
                    trial_start_date = trial_start_date2
            else:
                flag = False
                print('Stop searching because reaching the end the series ...')

    # find a calibration period that meets validratio
    flag = True
    while flag:
        trial_start_date2 = trial_start_date + pd.offsets.DateOffset(years=calibyears)
        index = (date >= trial_start_date) & (date <= trial_start_date2)
        if np.sum(index) > calibyears*365*validratio:
            qtrial = data[index]
            if np.sum(qtrial >= 0) / len(qtrial) >= validratio:
                flag = False
            else:
                trial_start_date = trial_start_date2
        else:
            flag = False
            print('Stop searching because reaching the end the series ...')

    # end date of the calibration period
    trial_end_date = trial_start_date + pd.offsets.DateOffset(years=calibyears) - pd.offsets.DateOffset(hours=1)

    print(f'Calibration period is {trial_start_date} to {trial_end_date}')
    return trial_start_date.strftime('%Y-%m-%d'), trial_end_date.strftime('%Y-%m-%d')


def calibration_period_CTSMformat(data, date, settings):
    if settings['method'] == 1:
        print('Decide calibration period using streamflow data')
        #### Method-1
        # use streamflow to decide calibration period
        # calibyears = 5 # how many years are used to calibrate the model
        # validratio = 0.8 # ratio of valid Q records during the period
        # trial_start_date = '1980-01-01' # only use data after this period
        calibyears = settings['calibyears']
        validratio = settings['validratio']
        trial_start_date = settings['trial_start_date']
        RUN_STARTDATE, STOP_DATE = calib_period_Beginning(data, date, calibyears, validratio, trial_start_date)
        STOP_OPTION = 'nmonths'
        STOP_N = calibyears * 12 # for STOP_OPTION: nmonths
    elif settings['method'] == 2:
        print('Decide calibration period using anomaly years')
        #### Method-2: climate anomaly years
        # startmonth = 10 #  the start of a year (e.g., 1 or 10)
        # periodlength = 5 # calib years
        # window = 5 # years of rolling mean
        startmonth = settings['startmonth']
        periodlength = settings['periodlength']
        # trial_start_date = settings['trial_start_date']
        window = settings['window']
        period_stat = time_series_anomaly_analysis(data, date, startmonth, periodlength, window)
        period_extreme = get_most_extreme_periods(period_stat)
        RUN_STARTDATE = period_extreme['date_start'].loc['min']
        STOP_DATE = period_extreme['date_end'].loc['min']
        # diff = pd.Timestamp(STOP_DATE).to_period('M') - pd.Timestamp(RUN_STARTDATE).to_period('M')
        # STOP_N = diff.n + 1
        STOP_OPTION = 'nmonths'
        STOP_N = periodlength * 12 # for STOP_OPTION: nmonths
    else:
        sys.exit('Method must be 1 or 2!')
    return RUN_STARTDATE, STOP_N, STOP_OPTION, STOP_DATE

if __name__ == '__main__':
    infile_q='/glade/p/ral/hap/common_data/camels/obs_flow_met/basin_dataset_public_v1p2/usgs_streamflow/all/01022500_streamflow_qc.txt'
    df_q = read_raw_CAMELS_Q_to_df(infile_q)
    date = df_q['date']
    data = df_q['Qobs'].values
    # method-1:
    print('Method-1')
    trial_start_date, trial_end_date = calib_period_Beginning(data, date, calibyears=5, validratio=0.8, trial_start_date='1980-01-01')
    # method-2:
    print('#'*50)
    print('Method-2')
    df_q = read_raw_CAMELS_Q_to_df(infile_q)
    date = df_q['date']
    data = df_q['Qobs'].values
    period_stat = time_series_anomaly_analysis(data, date, startmonth=10, periodlength=5, window=5)
    period_extreme = get_most_extreme_periods(period_stat)
    print(period_extreme)
