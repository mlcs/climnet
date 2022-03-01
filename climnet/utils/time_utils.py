import climnet.utils.general_utils as gut
from tqdm import tqdm
import pandas as pd
import itertools
from importlib import reload
import climnet.utils.statistic_utils as sut
import xarray as xr
import numpy as np
import copy

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
          'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def get_index_of_month(month):
    idx = -1
    idx = months.index(month)
    if idx == -1:
        raise ValueError(f"This month does not exist:{month}!")

    return idx


def is_in_month_range(month, start_month, end_month):
    start_month_idx = get_index_of_month(start_month)+1
    end_month_idx = get_index_of_month(end_month)+1

    if start_month_idx <= end_month_idx:
        mask = (month >= start_month_idx) & (month <= end_month_idx)
    else:
        mask = (month >= start_month_idx) | (month <= end_month_idx)
    return mask


def get_month_range_data(dataset, start_month='Jan', end_month='Dec'):
    """
    This function generates data within a given month range.
    It can be from smaller month to higher (eg. Jul-Sep) but as well from higher month
    to smaller month (eg. Dec-Feb)

    Parameters
    ----------
    start_month : string, optional
        Start month. The default is 'Jan'.
    end_month : string, optional
        End Month. The default is 'Dec'.

    Returns
    -------
    seasonal_data : xr.dataarray
        array that contains only data within month-range.

    """
    seasonal_data = dataset.sel(time=is_in_month_range(
        dataset['time.month'], start_month, end_month))

    return seasonal_data


def get_sy_ey_time(times, sy=None, ey=None, sm=None, em=None):
    """Returns the start and end year of a xr Dataarray
    datetime object

    Args:
        times (xr.Datetime): xr.Datetime object that contains time
        sy (int, optional): other startyear if specified. Defaults to None.
        ey (int, optional): end endyear. Defaults to None.

    Returns:
        int, int: start and end year
    """
    if sy is None:
        start_year = int(times[0].time.dt.year)
    else:
        start_year = sy
    if ey is None:
        end_year = int(times[-1].time.dt.year) + 1
    else:
        end_year = ey

    if sm is not None and em is not None:
        smi = get_index_of_month(sm)+1
        emi = get_index_of_month(em)+1
        if emi < smi:
            end_year -= 1

    return start_year, end_year


def get_start_end_date(sy, ey, sm='Jan', em='Dec'):
    smi = get_index_of_month(sm)+1
    emi = get_index_of_month(em)+1
    start_date = np.datetime64(
        f"{int(sy)}-{int(smi):02}-{int(1):02}",
        'D')
    if em == 'Feb':
        end_day = 28
    elif em in ['Jan', 'Mar', 'May', 'Jul', 'Aug', 'Oct', 'Dec']:
        end_day = 31
    else:
        end_day = 30

    ey = copy.deepcopy(ey)
    if emi < smi:
        ey = ey + 1
    end_date = np.datetime64(
        f"{int(ey)}-{int(emi):02}-{int(end_day):02}",
        'D')

    return start_date, end_date


def get_start_end_date_shift(sm, em, sy, ey, shift=0):
    """Same as normal get_start_date_year but the start and end shifted by shift days

    Args:
        sm (str): Start Month
        em (str): End Month
        sy (int): start Year
        ey (int): end Year
        shift (int, optional): shift by days. Defaults to 0.

    Returns:
        str: start and end date
    """
    smi = get_index_of_month(sm)+1
    emi = get_index_of_month(em)+1
    start_date = np.datetime64(
        f"{int(sy)}-{int(smi):02}-{int(1):02}",
        'D')
    if em == 'Feb':
        end_day = 28
    elif em in ['Jan', 'Mar', 'May', 'Jul', 'Aug', 'Oct', 'Dec']:
        end_day = 31
    else:
        end_day = 30

    ey = copy.deepcopy(ey)
    if emi < smi:
        ey = ey + 1
    end_date = np.datetime64(
        f"{int(ey)}-{int(emi):02}-{int(end_day):02}",
        'D')

    if shift > 0:
        start_date -= np.timedelta64(int(shift), 'D')
        end_date += np.timedelta64(int(shift), 'D')
    return start_date, end_date


def get_time_range(ds):
    time = ds.time.data
    sd = np.datetime64(time[0], 'D')
    ed = np.datetime64(time[-1], 'D')
    return sd, ed


def get_dates_of_time_range(time_range, dtype='datetime64[D]'):
    date_arr = np.arange(time_range[0], time_range[-1], dtype=dtype)
    # Include as well last time point
    date_arr = np.concatenate([date_arr,
                               [date_arr[-1] + np.timedelta64(1, 'D')]],
                              axis=0)
    return date_arr


def get_dates_of_time_ranges(time_ranges, dtype='datetime64[D]'):
    arr = np.array([], dtype=dtype)
    for time_range in time_ranges:
        arr = np.concatenate(
            [arr, get_dates_of_time_range(time_range, dtype=dtype)], axis=0)

    return arr


def is_full_year(ds, get_missing_dates=False):
    sd, ed = get_time_range(ds)
    ds_time = ds.time.data
    all_days = np.arange(sd, ed, np.timedelta64(1, 'D'))

    if len(ds_time) < len(all_days):
        if get_missing_dates:
            return np.setdiff1d(all_days, ds_time)
        else:
            return False
    else:
        return True


def apply_timemean(ds, timemean, sm=None, em=None):
    """Computes the monmean average on a given xr.dataset

    Args:
        ds (xr.dataset): xr dataset for the dataset

    Returns:
        xr.dataset: monthly average dataset
    """
    if timemean == 'day':
        tm = '1D'
    elif timemean == 'week':
        tm = '1W'
    elif timemean == 'month':
        tm = '1MS'
    elif timemean == 'season':
        tm = '3MS'
    elif timemean == 'year':
        tm = '1Y'
    else:
        raise ValueError(
            f'This time mean {timemean} does not exist! Please choose week, month, season or year!')

    print(f'Compute {timemean}ly means of all variables!')
    ds = ds.resample(time=tm).mean(dim="time", skipna=True)
    if sm is not None or em is not None:
        ds = get_month_range_data(ds, start_month=sm, end_month=em)
    return ds


def averge_out_nans_ts(ts, av_range=1):

    num_nans = np.count_nonzero(np.isnan(ts) == True)
    if num_nans == 0:
        return ts
    else:
        len_ts = len(ts)
        if num_nans/len_ts > 0.1:
            print('Warning! More than 10 % nans in ts')
        idx_nans = np.where(np.isnan(ts) == True)[0]

        for idx in idx_nans:
            if idx == 0:
                ts[0] = np.nanmean(ts[:10])
            else:
                ts[idx] = np.mean(ts[idx-av_range:idx])

        num_nans = np.count_nonzero(np.isnan(ts) == True)
        if num_nans > 0:
            idx_nans = np.where(np.isnan(ts) == True)[0]
            raise ValueError(f"Still nans in ts {idx_nans}")

        return ts


def compute_anomalies(dataarray, group='dayofyear',
                      base_period=None,
                      verbose=True):
    """Calculate anomalies.

    Parameters:
    -----
    dataarray: xr.DataArray
        Dataarray to compute anomalies from.
    group: str
        time group the anomalies are calculated over, i.e. 'month', 'day', 'dayofyear'
    base_period (list, None): period to calculate climatology over. Default None.

    Return:
    -------
    anomalies: xr.dataarray
    """
    if base_period is None:
        base_period = np.array([dataarray.time.data.min(), dataarray.time.data.max()])

    if group in ['dayofyear', 'month', 'season']:
        climatology = dataarray.sel(time=slice(base_period[0], base_period[1])
                                   ).groupby(f'time.{group}').mean(dim='time')
        anomalies = (dataarray.groupby(f"time.{group}")
                     - climatology)
    elif group == 'JJAS':
        monthly_groups = dataarray.groupby("time.month")

        month_ids = []
        for month in ['Jun', 'Jul', 'Aug', 'Sep']:
            month_ids.append(get_index_of_month(month))
        climatology = monthly_groups.mean(dim='time').sel(
            month=month_ids).mean(dim='month')

        anomalies = dataarray - climatology
    if verbose:
        print(f'Created {group}ly anomalies!')

    return anomalies


def detrend_dim(da, dim='time', deg=1):
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)

    return da - fit


def correlation_per_timeperiod(x, y, time_period):
    """Correlation per time period.

    Args:
        x ([type]): [description]
        y ([type]): [description]
        time_period ([type]): [description]

    Returns:
        [type]: [description]
    """
    corr = []
    for tp in time_period:
        corr.append(
            np.corrcoef(x.sel(time=slice(tp[0], tp[1])),
                        y.sel(time=slice(tp[0], tp[1])))[0, 1]
        )

    return xr.DataArray(corr,
                        dims=['time'],
                        coords={'time': time_period[:, 0]})


def get_ymd_date(date):
    date = np.datetime64(date)
    d = date.astype('M8[D]')
    m = date.astype('M8[M]')
    y = date.astype('M8[Y]')

    return y, m, d


def add_time_window(date, time_step=1, time_unit='D'):
    y, m, d = get_ymd_date(date)

    ad = time_step if time_unit == 'D' else 0
    am = time_step if time_unit == 'M' else 0
    ay = time_step if time_unit == 'Y' else 0
    if time_unit == 'D':
        next_date = (d + ad) + (date - d)
    elif time_unit == 'M':
        next_date = (m + am) + (date - m)
    elif time_unit == 'Y':
        next_date = (y + ay) + (date - y)
    return np.datetime64(next_date, 'D')


def get_tw_periods(sd, ed,
                   tw_length=1,
                   tw_unit='Y',
                   sliding_length=1,
                   sliding_unit='M'):

    ep = sd
    all_time_periods = []
    all_tps = []
    while ep < ed:
        ep = add_time_window(sd,
                             time_step=tw_length,
                             time_unit=tw_unit)
        if ep < ed:
            tw_range = get_dates_of_time_range([sd, ep]
                                               )
            all_time_periods.append(tw_range)
            all_tps.append(ep)
            sd = add_time_window(sd,
                                 time_step=sliding_length,
                                 time_unit=sliding_unit)

    return {'range': all_time_periods,
            'tps': all_tps
            }


def sliding_time_window(da, corr_method='spearman',
                        tw_length=1,
                        tw_unit='Y',
                        sliding_length=1,
                        sliding_unit='M',
                        source_ids=None,
                        target_ids=None):
    """Computes a sliding time window approach for a given dataset.

    Args:
        da (xr.dataarray): dataarray that contains the time series of
        points (list, optional): list of spatial points to applay the method.
                                 Defaults to None.
    """
    reload(sut)
    corr_function = sut.get_corr_function(corr_method=corr_method)
    sids = source_ids
    tids = target_ids
    tids = gut.remove_tids_sids(sids=sids, tids=tids)
    comb_ids = np.concatenate([sids, tids])
    num_nodes = len(comb_ids)

    # Get all different time periods
    sd, ed = get_time_range(ds=da)

    tw_periods = get_tw_periods(sd, ed, tw_length=tw_length, tw_unit=tw_unit,
                                sliding_length=sliding_length,
                                sliding_unit=sliding_unit)

    corr_time_dict = dict(
        sids=sids,
        tids=tids,
        tps=tw_periods['tps']
    )

    # Apply sliding window over all time periods
    for idx, tw_period in enumerate(tqdm(tw_periods['range'])):
        # Select data only for specified source - target points
        this_tp_data = da.sel(time=tw_period, points=comb_ids)
        corr, pvalue = corr_function(data=this_tp_data)
        if corr.shape != (num_nodes, num_nodes):
            raise ValueError(
                f'Wrong dimension of corr matrix {corr.shape} != {(num_nodes, num_nodes)}!')

        # Define source - correlations, target correlations and source-target correlations
        # in correlation matrix
        st_dict = gut.get_source_target_corr(corr=corr,
                                             sids=sids)

        corr_time_dict[idx] = dict(
            corr=corr,
            source_corr=st_dict['source'],
            target_corr=st_dict['target'],
            st_corr=st_dict['source_target']
        )

    return corr_time_dict


def mean_slw(corr_time_dict, corr_key='st_corr'):
    """Computes the mean correlation time of a dictionary of different times
    to get the time evolution of a correlation.

    Args:
        corr_time_dict (dict): dict that contains the time points and cross correlations
        corr_key (str, optional): Which correlation to use: source correlations, targetcorr
                                  or st_corr. Defaults to 'st_corr'.

    Returns:
        [type]: [description]
    """
    tps = corr_time_dict['tps']
    mean_arr = []
    std_arr = []
    ts_pd = pd.DataFrame(index=pd.DatetimeIndex(tps),
                         columns=['mean', 'std'])

    for idx, tp in enumerate(tps):
        st_corr = corr_time_dict[idx][corr_key]
        mean_arr.append(np.mean(st_corr))
        std_arr.append(np.std(st_corr))

    ts_pd.loc[tps, 'mean'] = mean_arr
    ts_pd.loc[tps, 'std'] = std_arr
    xr_ts = xr.Dataset.from_dataframe(ts_pd)

    return xr_ts.rename({'index': 'time'})


def local_cross_degree_slw(corr_time_dict,
                           corr_key='st_corr',
                           th=0.1):

    tps = corr_time_dict['tps']
    mean_arr = []
    ts_pd = pd.DataFrame(index=pd.DatetimeIndex(tps),
                         columns=['lcd'])

    for idx, tp in enumerate(tps):
        st_corr = corr_time_dict[idx][corr_key]
        adj_st = np.where(np.abs(st_corr) > th, 1, 0)
        mean_arr.append(np.sum(adj_st))

    ts_pd.loc[tps, 'lcd'] = mean_arr
    xr_ts = xr.Dataset.from_dataframe(ts_pd)

    return xr_ts.rename({'index': 'time'})


def get_tp_corr(corr_time_dict, tps, corr_method='st_corr'):
    dict_tps = corr_time_dict['tps']

    corr_array = []
    for tp in tps:
        if tp not in dict_tps:
            raise ValueError(f'This tp does not exist {tp}')

        idx = np.where(tp == dict_tps)[0][0]
        # print(tp, idx)
        corr = corr_time_dict[idx][corr_method]
        corr_array.append(corr)

    return np.array(corr_array)


def corr_distribution_2_region(corr_arr):
    all_corrs = np.array(corr_arr).flatten()

    return all_corrs


def get_corr_full_ts(ds, time_periods,
                     source_ids=None,
                     target_ids=None,
                     var_name='anomalies',
                     corr_type='spearman'):
    reload(gut)
    da = ds.ds[var_name]
    corr_function = sut.get_corr_function(corr_method=corr_type)
    sids = source_ids
    tids = target_ids
    tids = gut.remove_tids_sids(sids, tids)
    comb_ids = np.concatenate([sids, tids])
    num_nodes = len(comb_ids)
    tps = get_dates_of_time_ranges(time_ranges=time_periods)
    corr_time_dict = dict(
        sids=sids,
        tids=tids,
        tps=time_periods
    )

    this_tp_data = da.sel(time=tps, points=comb_ids)
    corr, pvalue = corr_function(data=this_tp_data)
    if corr.shape != (num_nodes, num_nodes):
        raise ValueError(
            f'Wrong dimension of corr matrix {corr.shape} != {(num_nodes, num_nodes)}!')

    # Define source - correlations, target correlations and source-target correlations
    # in correlation matrix
    st_dict = gut.get_source_target_corr(corr=corr,
                                         sids=sids)

    corr_time_dict.update(dict(
        corr=corr,
        source_corr=st_dict['source'],
        target_corr=st_dict['target'],
        st_corr=st_dict['source_target']
    )
    )

    return corr_time_dict
