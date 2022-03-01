"""
File that contains files for the analysis of different time series.
Often using event synchronization
"""
import pandas as pd
import climnet.event_synchronization as es
import numpy as np
import xarray as xr
from importlib import reload
import copy
import climnet.utils.time_utils as tu

def get_yearly_ts(data_t, times,
                  sm='Jan',
                  em='Dec',
                  name='data'):
    """Gets a yearly seperated time series where the columns are the years and the
    indices the time points in the yearly data

    Args:
        data_t (np.array): time series of dataset
        times (np.datetime): array of dates to
        sm (str, optional): Start Months. Defaults to 'Jan'.
        em (str, optional): end months. Defaults to 'Dec'.

    Returns:
        pd.DataFrame: DataFrame that contains the years seperated as columns
    """
    xr_ts = xr.DataArray(data=data_t,
                         name='data',
                         coords={'time': np.array(times)},
                         dims=['time'])

    sy, ey = tu.get_sy_ey_time(xr_ts.time, sm=sm, em=em)

    for idx_y, year in enumerate(range(sy, ey+1, 1)):
        start_date, end_date = tu.get_start_end_date(sm, em, year, year)
        data = xr_ts.sel(time=slice(start_date,
                                    end_date)
                         )

        if idx_y == 0:
            ts_yearly = pd.DataFrame(data.data,
                                     columns=[year],
                                     )
        elif idx_y > 0:
            ts_yearly_new = pd.DataFrame(data.data,
                                         columns=[year],
                                         )
            ts_yearly = pd.concat([ts_yearly, ts_yearly_new], axis=1)

    return ts_yearly


def count_all_events_series(ds, evs_ind_arr, plot=True, savepath=None, label_arr=None):
    """
    Get all events in array of cluster indices.
    """
    tps_arr = []

    for ts_evs in evs_ind_arr:
        tps_arr_ts = []
        for evs in ts_evs:
            ts = np.where(evs == 1)[0]
            tps_arr_ts.append(ts)

        tps = np.concatenate(np.array(tps_arr_ts, dtype=object), axis=0)
        # print(tps)
        tps_arr.append(np.array(tps, dtype=int))

    return tps_arr


def count_tps_occ(ds, tps_idx_arr, times=None):
    """Counts the number of events per Month for a given list of list of time
    point indices.

    Args:
        ds (BaseDataset): BaseDateset or derived datesets
        tps_idx_arr (2dlist): list of list of indices
        times (xr.times, optional): the time array to be used. Defaults to None.

    Returns:
        list: list of length 12 with rel freq of occurence for every month
    """
    if times is None:
        times = ds.ds['evs'].time
    res_c_occ_arr = np.zeros((len(tps_idx_arr), len(ds.months)))

    for idx, tps_idx in enumerate(tps_idx_arr):
        m_c_occ = res_c_occ_arr[idx]
        tot_num = len(tps_idx)
        times_tps = np.array(times[tps_idx])
        tpspd = pd.DatetimeIndex(times_tps)
        u, count = np.unique(tpspd.month,  return_counts=True)
        u = np.array(u-1, dtype=int)  # -1 because January has index 1
        m_c_occ[u] = count
        m_c_occ /= tot_num

    res_c_occ = np.mean(res_c_occ_arr, axis=0)

    return res_c_occ


def count_tps_occ_evs(ds, tps_arr, times=None):
    """Counts the number of occurence in every month for a given array of binary event
    series

    Args:
        ds (BaseDataset): [description]
        tps_arr (list): list of event series
        times (xr.time, optional): Xarray datetime. Defaults to None.

    Raises:
        ValueError:

    Returns:
        list: list of freq of occurence per month
    """
    if times is None:
        times = ds.ds['evs'].time

    for idx, tps in enumerate(tps_arr):
        if len(times) != len(tps):
            raise ValueError(
                f"Idx {idx} Not same length of event series {len(tps)} and times{len(times)}!")

    tps_idx_arr = es.get_vector_list_index_of_extreme_events(tps_arr)

    res_c_occ = count_tps_occ(ds=ds,
                              tps_idx_arr=tps_idx_arr,
                              times=times)

    return res_c_occ


def get_sync_times_ES(ds,
                      net,
                      region_1_dict,
                      region_2_dict=None,
                      taumax=10,
                      use_adj=False,
                      ):
    """Function that computes the time series of synchronous events for
    a given Network (via adjacency) and certain nodes (nodes can be the same!)


    """
    reload(es)
    time = ds.ds.time.data
    ts_pd = pd.DataFrame(index=pd.DatetimeIndex(time),
                         columns=['t', 't12', 't21'])
    if region_2_dict is None:
        region_2_dict = region_1_dict
    ts_1 = es.prepare_es_input_data(region_1_dict['data']['evs'].T)
    ts_2 = es.prepare_es_input_data(region_2_dict['data']['evs'].T)
    num_tp = len(region_1_dict['data'].time)
    if num_tp != len(region_2_dict['data'].time):
        raise ValueError(f'lenght of time series {num_tp} not equal!')

    if use_adj:
        ind_ts_dict1 = dict(zip(region_1_dict['ids'], ts_1))
        ind_ts_dict2 = dict(zip(region_2_dict['ids'], ts_2))
        t, t12, t21, dyn_delay = es.es_reg_network(
            ind_ts_dict1, ind_ts_dict2, taumax,
            adjacency=net.adjacency,
            num_tp=num_tp)
    else:
        t, t12, t21, dyn_delay = es.es_reg_network(
            es_idx_1=ts_1,
            es_idx_2=ts_2,
            taumax=taumax,
            num_tp=num_tp)

    if len(t12) != len(time):
        raise ValueError(f"Times are not of same length {len(t12)}!")

    ts_pd.loc[time, 't'] = t
    ts_pd.loc[time, 't12'] = t12
    ts_pd.loc[time, 't21'] = t21

    xr_ts = xr.Dataset.from_dataframe(ts_pd)
    xr_ts = xr_ts.rename({'index': 'time'})

    return xr_ts


def get_sync_times_ES_yearly(ds,
                             net,
                             region_1_dict,
                             region_2_dict=None,
                             sy=None,
                             ey=None,
                             sm=None,
                             em=None,
                             taumax=10,
                             use_adj=True,
                             ):

    reload(es)
    if region_2_dict is None:
        region_2_dict = region_1_dict
    times = ds.ds['time']
    start_year, end_year = tu.get_sy_ey_time(times, sy=sy, ey=ey, sm=sm, em=em)

    for idx_y, year in enumerate(np.arange(start_year, end_year)):
        print(f"Year {year}")
        ts_idx_1 = es.prepare_es_input_data(region_1_dict[year]['ts'])
        ts_idx_2 = es.prepare_es_input_data(region_2_dict[year]['ts'])
        num_tp = len(region_1_dict[year]['ts'][0])
        times = region_1_dict[year]['times']
        if num_tp != len(times):
            raise ValueError(f'lenght of time series {num_tp} not equal!')

        ind_ts_dict1 = dict(zip(region_1_dict['ids'], ts_idx_1))
        ind_ts_dict2 = dict(zip(region_2_dict['ids'], ts_idx_2))
        if use_adj is True:
            t, t12, t21, dyn_delay = es.es_reg_network(
                ind_ts_dict1, ind_ts_dict2, taumax,
                adjacency=net.adjacency,
                num_tp=num_tp)
        else:
            t, t12, t21, dyn_delay = es.es_reg(
                ts_idx_1, ts_idx_2, taumax,
                num_tp=num_tp)
        if len(times) != len(t12):
            raise ValueError(
                f"Time not of same length {len(times)} vs {len(t12)}!")
        pd_array_new = pd.DataFrame(np.vstack([t12, t21]).transpose(),
                                    columns=['t12', 't21'],
                                    index=times.data)
        if idx_y == 0:
            pd_array = copy.deepcopy(pd_array_new)
            ts12_yearly = pd.DataFrame(t12,
                                       columns=[year],
                                       )
            ts21_yearly = pd.DataFrame(t21,
                                       columns=[year],
                                       )
        elif idx_y > 0:
            pd_array = pd_array.append(pd_array_new)
            ts12_yearly_new = pd.DataFrame(t12,
                                           columns=[year],
                                           )
            ts12_yearly = pd.concat([ts12_yearly, ts12_yearly_new], axis=1)

            ts21_yearly_new = pd.DataFrame(t21,
                                           columns=[year],
                                           )
            ts21_yearly = pd.concat([ts21_yearly, ts21_yearly_new], axis=1)

    return pd_array, ts12_yearly, ts21_yearly


def get_quantile_yearly_ts(ts_yearly, q_arr=0):
    """Yearly pandas array of time series.

    Args:
        ts_yearly (pd.DataFrame):  Rows are time steps of year, columns
    are specific year.
        q_arr ([type]): [description]
        label_arr ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    pd_res = ts_yearly.quantile()

    return pd_res



