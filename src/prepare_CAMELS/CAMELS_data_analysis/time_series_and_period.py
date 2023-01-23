# use the meteorological data from camels dataset to decide calibration periods

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

def read_raw_CAMELS_Q_to_df(infile_q):
    # read streamflow fom raw CAMELS text files, and change column names
    df_q = pd.read_csv(infile_q, delim_whitespace=True, header=None)
    df_q.columns = ['name', 'year', 'month', 'day', 'Qobs', 'code']
    df_q['date'] = pd.to_datetime(df_q[['year', 'month', 'day']])
    q = df_q['Qobs'].values
    q[q<0] = np.nan
    df_q['Qobs'] = q
    return df_q


def cal_shift_annual_mean(data, date, startmonth=1):
    # startmonth: the start of a year (e.g., 1 or 10)
    years = np.unique(pd.DatetimeIndex(date).year.values)
    values = np.nan * np.zeros(len(years))
    for i in range(len(years)):
        yeari = years[i]
        date_start = pd.Timestamp(f'{yeari}-{startmonth}-1')
        date_end = date_start + pd.offsets.DateOffset(years=1) - pd.offsets.DateOffset(hours=1)
        datai = data[(date>=date_start) & (date<=date_end)]
        if np.sum(~np.isnan(datai))/len(datai) >= 0.9: # valid ratio during this period
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
    annual_anomaly = values - np.nanmean(values)
    rolling_anomaly = values_rolling - np.nanmean(values)
    tmean_stat = []
    for i in range(0, len(rolling_anomaly)-periodlength):
        datei1 = f'{years[i]}-{startmonth}-01'
        datei2 = (pd.Timestamp(f'{years[i+periodlength]}-{startmonth}-01') - pd.offsets.DateOffset(hours=1)).strftime('%Y-%m-%d')
        datai = np.mean(rolling_anomaly[i:i+periodlength])
        tmean_stat.append([datei1, datei2, datai])
    tmean_stat = pd.DataFrame(tmean_stat, columns=['date_start', 'date_end', 'mean'])
    # sort stat from low to high
    tmean_stat = tmean_stat[~np.isnan(tmean_stat['mean'].values)]
    tmean_stat = tmean_stat.iloc[tmean_stat['mean'].values.argsort()]
    tmean_stat.index = np.arange(len(tmean_stat))
    return tmean_stat, rolling_anomaly, annual_anomaly, years

def get_most_extreme_periods(tmean_stat):
    data_minmax = pd.DataFrame()
    indmin = tmean_stat['mean'].values.argmin()
    data_minmax = pd.concat([data_minmax, tmean_stat[indmin:indmin + 1]])
    indmax = tmean_stat['mean'].values.argmax()
    data_minmax = pd.concat([data_minmax, tmean_stat[indmax:indmax + 1]])
    data_minmax.index = ['min', 'max']
    return data_minmax


def plot_series(data, date, ax, indexmin, indexmax):
    # use tmean to decide min/max periods
    data_stat, rolling_anomaly, annual_anomaly, years = time_series_anomaly_analysis(data, date, startmonth=10, periodlength=5, window=5)
    data_minmax = get_most_extreme_periods(data_stat)

    if len(indexmin) == 0 or len(indexmax) == 0:
        year1 = int(data_minmax['date_start'].loc['min'][:4])
        year2 = int(data_minmax['date_end'].loc['min'][:4])
        indexmin = (years >= year1) & (years <= year2)

        year1 = int(data_minmax['date_start'].loc['max'][:4])
        year2 = int(data_minmax['date_end'].loc['max'][:4])
        indexmax = (years>=year1) & (years<=year2)

    # rolling anomaly
    l1 = ax.plot(years, rolling_anomaly, '-o', color='tab:orange', label='5-yr rolling mean', zorder=4)
    l2 = ax.plot(years[indexmin], rolling_anomaly[indexmin], '-b', linewidth=8, alpha=0.5, label='min 5-yr rolling', zorder=5)
    l3 = ax.plot(years[indexmax], rolling_anomaly[indexmax], '-r', linewidth=8, alpha=0.5, label='max 5-yr rolling', zorder=6)

    # annual anomaly
    l4 = ax.plot(years, annual_anomaly, '-*', color='grey', label='annual mean', zorder=0)
    l5 = ax.plot(years[indexmin], annual_anomaly[indexmin], '-', color='grey', linewidth=8, alpha=0.5, label='corresponding min/max', zorder=1)
    l6 = ax.plot(years[indexmax], annual_anomaly[indexmax], '-', color='grey', linewidth=8, alpha=0.5, zorder=2)

    # fig/ax attributes
    ax.plot(years, years*0, '-k')
    ax.set_ylabel('Anomaly')
    limi = np.nanmax(np.abs(rolling_anomaly)) * 2.5
    ax.set_ylim(-limi, limi)
    ax.set_xlim(years[0], years[-1])
    ax.grid()

    # legend
    leg = l1 + l2 + l3 + l4 + l5
    labs = [l.get_label() for l in leg]
    ax.legend(leg, labs, loc='lower center', ncol=2, frameon=False)

    return ax, indexmin, indexmax



inpath_camels = '/glade/scratch/guoqiang/CAMELS_data/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2'
infile_info = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/shared_data_Sean/info_ESMFmesh_ctsm_HCDN_nhru_final_671.buff_fix_holes_polygons_simplified_5e-4_split_nested.csv'

df_info = pd.read_csv(infile_info)


pdf = matplotlib.backends.backend_pdf.PdfPages("CAMELS_MetFlow.pdf")

# for i in range(len(df_info)):
for i in range(3):
    id = df_info.iloc[i]['hru_id']

    # load prcp/tmean
    infilei = f'{inpath_camels}/basin_mean_forcing/nldas/all/{id:08}_lump_nldas_forcing_leap.txt'
    dfi = pd.read_csv(infilei, skiprows=3, delim_whitespace=True)
    dfi = dfi.rename(columns={'Mnth':'Month'})
    date = pd.to_datetime(dfi[['Year', 'Month', 'Day']])
    prcpi = dfi['PRCP(mm/day)'].values
    tmini = dfi['Tmin(C)'].values
    tmaxi = dfi['Tmax(C)'].values
    tmeani = (tmini + tmaxi) / 2

    # load q
    infilei_q = f'{inpath_camels}/usgs_streamflow/{id:08}_streamflow_qc.txt'
    df_q = read_raw_CAMELS_Q_to_df(infilei_q)

    fig, ax = plt.subplots(3, 1, figsize=[8, 8])

    ax[0], indexmin, indexmax  =plot_series(tmeani, date, ax[0], [], [])
    ax[0].set_title(f'(a) {id:08}: Tmean (period base)')

    ax[1], indexmin, indexmax = plot_series(prcpi, date, ax[1], indexmin, indexmax)
    ax[1].set_title(f'(b) {id:08}: Precipitation')
    ax[1].get_legend().remove()

    ax[2], indexmin, indexmax = plot_series(df_q['Qobs'].values, df_q['date'].values, ax[2], indexmin, indexmax)
    ax[2].set_title(f'(c) {id:08}: Streamflow')
    ax[2].get_legend().remove()

    plt.tight_layout()
    # plt.show()
    pdf.savefig(fig)

pdf.close()