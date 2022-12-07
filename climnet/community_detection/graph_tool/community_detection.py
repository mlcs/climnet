#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 08:53:08 2020
Functions for Community Detection
@author: Felix Strnad
"""
from climnet.graph_tool.dendrograms import Dendrogram_ES
import climnet.tsa.time_series_analysis as tsa
import climnet.utils.time_utils as tu
import climnet.plots as cplt
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from importlib import reload
import xarray as xr


def get_density_cluster(ds, loc_arr, num_runs=30,
                        abs_th=4, rel_th=0,
                        graph_folder=None, graph_file=None):
    """
    This function returns the main cluster in a band like structure for selected lat.
    """

    cluster_idx_arr = np.zeros((num_runs, len(ds.indices_flat)))

    sel_ids = ds.get_ids_loc_arr(loc_arr=loc_arr)

    counter = 0
    for idx, job_id in tqdm(enumerate(range(0, num_runs))):
        if graph_folder is None:
            raise ValueError("Graph path folder not provided!")
        else:
            sbm_filepath = graph_folder + f"{job_id}_" + graph_file

        if not os.path.exists(sbm_filepath + '_group_levels.npy'):
            print(
                f"WARNING file {sbm_filepath +'_group_levels.npy'} does not exist!")
            cluster_idx_arr = np.delete(cluster_idx_arr, counter, 0)
            continue
        else:
            group_levels = np.load(
                sbm_filepath+'_group_levels.npy',  allow_pickle=True)
            d_es = Dendrogram_ES(group_levels,)

            # Compute the cluster which is given by the returned leaf node ids
            leaf_nodes = ds.get_cluster_sel_ids(
                sel_ids=sel_ids, d_es=d_es, abs_th=abs_th, rel_th=rel_th)

            if len(leaf_nodes) == len(ds.indices_flat):
                print(f"JobID {job_id}: Warning all nodes in one cluster!")
            for id in np.concatenate(sel_ids):
                if id not in leaf_nodes:
                    print(
                        f"JobId {job_id}: Warning monsoon id {id} not in cluster ids!")

            cluster_idx_arr[counter, :] = ds.flat_idx_array(leaf_nodes)
            counter += 1

    mean = np.mean(cluster_idx_arr, axis=0)
    std = np.std(cluster_idx_arr, axis=0)

    return mean, std


def get_density_map_loc_arr(ds, loc_arr, num_runs=30,
                            den_th=0.8,  abs_th=4, rel_th=0,
                            plot=False,
                            savepath=None, title=None,
                            graph_folder=None,
                            graph_file=None):
    reload(cplt)
    mean_cluster, std_cluster = get_density_cluster(ds=ds,
                                                    loc_arr=loc_arr,
                                                    num_runs=num_runs,
                                                    abs_th=abs_th,
                                                    graph_folder=graph_folder,
                                                    graph_file=graph_file)

    eps = 0.2
    mean_cluster = np.where(mean_cluster > eps, mean_cluster, np.nan)
    mi_indices = np.where(mean_cluster > den_th)[0]
    if plot is True:

        im = cplt.plot_map(ds=ds,
                           dmap=ds.get_map(mean_cluster),
                           plot_type='contourf',
                           significant_mask=True,
                           plt_grid=True,
                           orientation='vertical',
                           label='Membership Likelihood',
                           title=title,
                           vmin=0, vmax=1,
                           levels=10,
                           cmap='Reds')

        cluster_map = ds.get_map(ds.flat_idx_array(mi_indices))
        cplt.plot_map(ds, cluster_map,
                      ax=im['ax'],
                      plot_type='contour',
                      color='blue',
                      significant_mask=False,
                      levels=2,
                      bar=False)

        if savepath is not None:
            plt.savefig(savepath, bbox_inches='tight')

    return mi_indices


def compute_member_idx_dict(ds, region_names, num_runs=1,
                            den_th=0.99, graph_path_folder=None,
                            graph_file=None, savepath=None):

    loc_arr = []
    for idx, region_name in enumerate(region_names):
        loc = ds.monsoon_dictionary[region_name]['loc']
        loc_arr.append(loc)
    print(loc_arr)

    c_indices = get_density_map_loc_arr(ds=ds,
                                        loc_arr=loc_arr,
                                        den_th=den_th,
                                        num_runs=num_runs,
                                        plot=True,
                                        savepath=savepath,
                                        graph_folder=graph_path_folder,
                                        graph_file=graph_file,
                                        )

    return c_indices


def get_c_indices_m_region_bak(ds, c_indices, mname,
                               sm='Jan',
                               em='Dec',
                               sy=None,
                               ey=None,
                               get_map=True,
                               an=True):
    """Gets for a specific monsoon regions these indices that
    belong to this region

    Args:
        c_indices (list): list of indices
        mname (str): Name of the monsoon region
        sm (str, optional): start month for time Series. Defaults to 'Jan'.
        em (str, optional): end month for time Series. Defaults to 'Dec'.

    Returns:
        dict: dictonary of monsoon containing the time series, the ids and the respective
        maps
    """
    reload(tsa)
    reload(tu)
    m_ts_dict = dict()
    data_evs = ds.ds['evs']
    data = ds.ds['pr']
    if an is True:
        data_an = ds.ds['an']
    if sy is not None or ey is not None:
        start_date, end_date = tsa.get_start_end_date_year(
            sm=sm,
            em=em,
            sy=sy,
            ey=ey)
        data_evs = data_evs.sel(time=slice(start_date, end_date))
        data = data.sel(time=slice(start_date, end_date))
        if an is True:
            data_an = data_an.sel(time=slice(start_date, end_date))
    if sm != 'Jan' or em != 'Dec':
        data = tu.get_month_range_data(
            data, start_month=sm, end_month=em)
        data_evs = tu.get_month_range_data(
            data_evs, start_month=sm, end_month=em)
        if an is True:
            data_an = ds.get_month_range_data(
                data_an, start_month=sm, end_month=em)

    times = data_evs.time
    es_time_series = ds.flatten_array(
        dataarray=data_evs, check=False).T
    pr_time_series = ds.flatten_array(
        dataarray=data, check=False).T
    if an is True:
        an_time_series = ds.flatten_array(
            dataarray=data_an, check=False).T

    monsoon_dict = ds.monsoon_dictionary
    mi_map = ds.get_map(ds.flat_idx_array(c_indices))
    this_indices, this_map = ds.get_idx_monsoon(
        monsoon_dict[mname], mi_map)
    points = ds.get_points_for_idx(this_indices)

    if get_map is True:
        m_ts_dict = {
            'pr': pr_time_series[points],
            'an': an_time_series[points] if an is True else None,
            'ts': es_time_series[points],
            'ids': this_indices,
            'map': this_map,
            'times': times}
    else:
        m_ts_dict = {
            'ts': es_time_series[points],
            'pr': pr_time_series[points],
            'an': an_time_series[points] if an is True else None,
            'times': times}

    return m_ts_dict


def get_c_indices_m_region(ds, c_indices, mname,
                           sm='Jan',
                           em='Dec',
                           sy=None,
                           ey=None,
                           an=True):
    """Gets for a specific monsoon regions these indices that
    belong to this region

    Args:
        c_indices (list): list of indices
        mname (str): Name of the monsoon region
        sm (str, optional): start month for time Series. Defaults to 'Jan'.
        em (str, optional): end month for time Series. Defaults to 'Dec'.

    Returns:
        dict: dictonary of monsoon containing the time series, the ids and the respective
        maps
    """
    reload(tsa)
    reload(tu)
    data_evs = ds.ds['evs']
    data = ds.ds['pr']
    if an is True:
        data_an = ds.ds['an']
    if sy is not None or ey is not None:
        start_date, end_date = tsa.get_start_end_date_year(
            sm=sm,
            em=em,
            sy=sy,
            ey=ey)
        data_evs = data_evs.sel(time=slice(start_date, end_date))
        data = data.sel(time=slice(start_date, end_date))
        if an is True:
            data_an = data_an.sel(time=slice(start_date, end_date))
    if sm != 'Jan' or em != 'Dec':
        data = tu.get_month_range_data(
            data, start_month=sm, end_month=em)
        data_evs = tu.get_month_range_data(
            data_evs, start_month=sm, end_month=em)
        if an is True:
            data_an = ds.get_month_range_data(
                data_an, start_month=sm, end_month=em)

    mi_map = ds.get_map(ds.flat_idx_array(c_indices))
    monsoon_dict = ds.monsoon_dictionary

    this_indices, this_map = ds.get_idx_monsoon(
        monsoon_dict[mname], mi_map)
    points = ds.get_points_for_idx(this_indices)   # This is important!
    if an:
        xr_ts = xr.merge([data_evs.sel(points=points),
                         data.sel(points=points),
                         data_an.sel(points=points)]
                         )
    else:
        xr_ts = xr.merge([data_evs.sel(points=points),
                         data.sel(points=points)]
                         )
    return {'data': xr_ts,
            'map': this_map,
            'ids': this_indices,
            }


def get_monsoon_communities_dict(ds, m_arr_names, node_indices,
                                 sm='Jan',
                                 em='Dec',
                                 sy=None,
                                 ey=None,
                                 an=True):
    new__monsoon_dict = dict()
    monsoon_dict = ds.monsoon_dictionary
    for mname, mregion in monsoon_dict.items():
        if mname in m_arr_names:
            this_m_dict = dict()
            this_m_dict.update(mregion)
            this_m_dict.update(get_c_indices_m_region(ds,
                                                      node_indices,
                                                      mname,
                                                      sm=sm,
                                                      em=em,
                                                      sy=sy,
                                                      ey=ey,
                                                      an=an)
                               )
            new__monsoon_dict[mname] = this_m_dict

    return new__monsoon_dict


def get_yearly_monsoon_communities_dict(ds,
                                        m_arr_names,
                                        node_indices,
                                        sm='Jan',
                                        em='Dec',
                                        sy=None,
                                        ey=None):
    times = ds.ds['time']
    start_year, end_year = tsa.get_sy_ey_time(times, sy=sy, ey=ey)
    print('Create full directory!')
    all_year_monsoon_dict = get_monsoon_communities_dict(ds,
                                                         m_arr_names,
                                                         node_indices,
                                                         sm=sm,
                                                         em=em
                                                         )
    print('Now start yearly dictionary!')
    for idx_y, year in enumerate(np.arange(start_year, end_year)):
        print(f"Year: {year}")
        monsoon_dict = get_monsoon_communities_dict(ds, m_arr_names, node_indices,
                                                    sm=sm,
                                                    em=em,
                                                    sy=year,
                                                    ey=year,
                                                    get_map=False)
        for mname in m_arr_names:
            y_m_dict = {k: monsoon_dict[mname][k] for k in monsoon_dict[mname].keys() & {
                'ts', 'times', 'pr', 'an'}}
            all_year_monsoon_dict[mname][year] = y_m_dict

    return all_year_monsoon_dict


def get_yearly_sum(dict,
                   sy,
                   ey,):
    sum_arr = []
    for year in range(sy, ey+1, 1):
        y_sum = np.sum(dict[year]['pr'])
        sum_arr.append(y_sum)

    return np.array(sum_arr)


def get_yearly_av(dict,
                  sy,
                  ey,
                  ):
    av_arr = []
    for year in range(sy, ey+1, 1):
        y_av = np.mean(dict[year])
        av_arr.append(y_av)

    return np.array(av_arr)


def get_yearly_av_var(dict,
                      sy,
                      ey,
                      var='pr'):
    av_arr = []
    for year in range(sy, ey+1, 1):
        y_av = np.mean(dict[year][var])
        av_arr.append(y_av)

    return np.array(av_arr)


def get_av_region(dict,
                  var='pr',):
    val_ts = dict[var]
    mean_ts = np.mean(val_ts, axis=0)
    if len(mean_ts) != len(dict['times']):
        raise ValueError('Times and value ts not of same length')

    return mean_ts
