#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:09:03 2020

@author: Felix Strnad
"""
import sys, os
import numpy as np
from math import ceil
import xarray as xr
import multiprocessing as mpi
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import scipy.stats as st
from scipy.signal import filtfilt, cheby1, argrelmax,  find_peaks

def get_vector_list_index_of_extreme_events(event_data):
    extreme_event_index_matrix=[]
    for i, e in enumerate(event_data):
        ind_list= np.where(e>0)[0]
        extreme_event_index_matrix.append(ind_list)
    return np.array(extreme_event_index_matrix, dtype=object)


def remove_consecutive_days(event_data, event_data_idx):
    """
    Consecutive days with rainfall above the threshold are considered as single events
    and placed on the first day of occurrence.

    Example:
        event_series_matrix = compute_event_time_series(fully_proccessed_data, var)
        all_event_series=flatten_lat_lon_array(event_series_matrix)
        extreme_event_index_matrix=es.get_vector_list_index_of_extreme_events(all_event_series)
        this_series_idx=extreme_event_index_matrix[index]
        print(this_series_idx)

        corr_all_event_series=es.remove_consecutive_days(all_event_series, extreme_event_index_matrix)
        corr_extreme_event_index_matrix=es.get_vector_list_index_of_extreme_events(corr_all_event_series)
        this_series_idx=corr_extreme_event_index_matrix[index]
        print(this_series_idx)

    Parameters
    ----------
    event_data : Array
        Array containing event_data
    event_data_idx : Array
        Array containing indices of all events in event_data

    Returns
    -------
    event_data : Array
        Corrected array of event data.
    """
    if len(event_data) != len(event_data_idx):
        raise ValueError("ERROR! Event data and list of idx event data are not of the same length!")


    for i, e in enumerate(event_data):

        this_series_idx = event_data_idx[i]
        this_series_idx_1nb = event_data_idx[i] + 1
        this_series_idx_2nb = event_data_idx[i] + 2
        # this_series_idx_3nb=extreme_event_index_matrix[i] +3

        intersect_1nb=np.intersect1d(this_series_idx, this_series_idx_1nb )
        intersect_2nb=np.intersect1d(intersect_1nb, this_series_idx_2nb )
        # intersect_3nb=np.intersect1d(intersect_2nb,this_series_idx_3nb )

        e[intersect_1nb]=0
        e[intersect_2nb]=0

    return event_data


def randomize_e_matrix(e_matrix):
    for idx, ts in enumerate(e_matrix):
        e_matrix[idx] = np.random.permutation(ts)
    return e_matrix


def event_synchronization(event_data, taumax=10, min_num_sync_events=10, randomize_ts=False):
    num_time_series = len(event_data)
    adj_matrix = np.zeros((num_time_series,num_time_series),dtype=int)
    double_taumax = 2*taumax
    extreme_event_index_matrix = get_vector_list_index_of_extreme_events(event_data)
    event_data = remove_consecutive_days(event_data, extreme_event_index_matrix)
    extreme_event_index_matrix = get_vector_list_index_of_extreme_events(event_data)

    if randomize_ts is True:
        extreme_event_index_matrix=randomize_e_matrix(extreme_event_index_matrix)

    start=time.time()
    print(f"Start computing event synchronization!")

    for i, ind_list_e1 in enumerate(extreme_event_index_matrix):
        # Get indices of event series 1
        #ind_list_e1= np.where(e1>0)[0]
        for j, ind_list_e2 in enumerate(extreme_event_index_matrix):
            if i == j:
                continue

            sync_event=0

            for m, e1_ind in enumerate(ind_list_e1[1:-1], start=1):
                d_11_past = e1_ind-ind_list_e1[m-1]
                d_11_next = ind_list_e1[m+1]-e1_ind

                for n,e2_ind in enumerate(ind_list_e2[1:-1], start=1):
                    d_12_now = (e1_ind-e2_ind)
                    if d_12_now > taumax:
                        continue

                    d_22_past = e2_ind-ind_list_e2[n-1]
                    d_22_next = ind_list_e2[n+1]-e2_ind

                    tau = min(d_11_past, d_11_next, d_22_past, d_22_next, double_taumax) / 2
                    #print(tau, d_11_past, d_11_next, d_22_past, d_22_next, double_taumax)
                    if d_12_now <= tau and d_12_now >= 0:
                        sync_event += 1
                        #print("Sync: ", d_12_now, e1_ind, e2_ind, sync_event,n)

                    if d_12_now < -taumax:
                        #print('break!',  d_12_now, e1_ind, e2_ind, )
                        break

            # Createria if number of synchron events is relevant
            if sync_event >= min_num_sync_events:
                #print(i,j, sync_event)
                adj_matrix[i, j] = 1
    end = time.time()
    print(end - start)
    np.save('adj_matrix_gpcp.npy', adj_matrix)
    print(adj_matrix)

    return adj_matrix


def event_synchronization_one_series(extreme_event_index_matrix, ind_list_e1, i, taumax=10, min_num_sync_events=10):
    double_taumax = 2*taumax
    sync_time_series_indicies = []
    # Get indices of event series 1
    # ind_list_e1= np.where(e1>0)[0]

    for j, ind_list_e2 in enumerate(extreme_event_index_matrix):
        if i == j:
            continue

        sync_events = event_sync(ind_list_e1, ind_list_e2, taumax, double_taumax)


        # Createria if number of synchron events is relevant
        if sync_events >= min_num_sync_events:
            # print(i,j, sync_event)
            num_events_i = len(ind_list_e1)
            num_events_j = len(ind_list_e2)
            sync_time_series_indicies.append((j, num_events_i, num_events_j, sync_events))

    return (i, sync_time_series_indicies)


def event_sync(ind_list_e1, ind_list_e2, taumax, double_taumax):
    # Get indices of event series 2
    # ind_list_e2=np.where(e2>0)[0]
    sync_events = 0
    #print(ind_list_e1)
    #print(ind_list_e2)
    for m, e1_ind in enumerate(ind_list_e1[1:-1], start=1):
        d_11_past = e1_ind-ind_list_e1[m-1]
        d_11_next = ind_list_e1[m+1]-e1_ind

        for n, e2_ind in enumerate(ind_list_e2[1:-1], start=1):
            d_12_now = (e1_ind-e2_ind)
            if d_12_now > taumax:
                continue

            d_22_past = e2_ind-ind_list_e2[n-1]
            d_22_next = ind_list_e2[n+1]-e2_ind

            tau = min(d_11_past, d_11_next, d_22_past, d_22_next, double_taumax) / 2
            #print(tau, d_11_past, d_11_next, d_22_past, d_22_next, double_taumax)
            if d_12_now <= tau and d_12_now >= 0:
                sync_events += 1
                # print("Sync: ", d_12_now, e1_ind, e2_ind, sync_event,n)

            if d_12_now < -taumax:
                #print('break!',  d_12_now, e1_ind, e2_ind, )
                break

    return sync_events


def prepare_es_input_data(event_data, rcd=True):
    """
    Creates array of list, where events take place and removes consecutive days
    """
    extreme_event_index_matrix = get_vector_list_index_of_extreme_events(event_data)
    if rcd is True:
        print("Start removing consecutive days...")
        event_data = remove_consecutive_days(event_data, extreme_event_index_matrix)
        extreme_event_index_matrix = get_vector_list_index_of_extreme_events(event_data)
        print("End removing consecutive days!")

    return extreme_event_index_matrix

def parallel_event_synchronization(event_data, taumax=10, min_num_sync_events=1, job_id=0, num_jobs=1, savepath="./E_matrix.npy", null_model=None,
                                   ):
    num_time_series = len(event_data)
    one_array_length = int(num_time_series/num_jobs) +1

    extreme_event_index_matrix = prepare_es_input_data(event_data)

    start_arr_idx = job_id*one_array_length
    end_arr_idx = (job_id+1)*one_array_length
    print(f"Start computing event synchronization for event data from {start_arr_idx} to {end_arr_idx}!")

    # For parallel Programming
    num_cpus_avail = mpi.cpu_count()
    print(f"Number of available CPUs: {num_cpus_avail}")

    parallelArray = []

    start = time.time()

    # Parallelizing by using joblib
    backend = 'multiprocessing'
    # backend='loky'
    #backend='threading'
    parallelArray = (Parallel(n_jobs=num_cpus_avail, backend=backend)
                     (delayed(event_synchronization_one_series)
                      (extreme_event_index_matrix, e1, start_arr_idx + i, taumax, min_num_sync_events)
                      for i, e1 in enumerate(tqdm(extreme_event_index_matrix[start_arr_idx:end_arr_idx]))
                      )
                    )

    # Store output of parallel processes in adjecency matrix
    adj_matrix_edge_list = []
    print("Now store results in numpy array to hard drive!")
    for process in tqdm(parallelArray):
        i, list_sync_event_series = process
        for sync_event in list_sync_event_series:
            j, num_events_i, num_events_j, num_sync_events_ij = sync_event
            thresh_null_model = null_model[num_events_i, num_events_j]
            # Check if number of synchronous events is significant according to the null model
            # Threshold needs to be larger (non >= !)
            if num_sync_events_ij > thresh_null_model:
                # print(
                #     f'i {i} {num_events_i}, j {j} {num_events_j} Sync_events {num_sync_events_ij} > {int(thresh_null_model)}')
                if weighted is True:
                    if np.abs(hq - lq) < 0.001:
                        print(f'WARNING, hq{hq}=lq{lq}')
                        weight = 0
                    else:
                        weight = (num_sync_events_ij - med) / (hq - lq)
                else:
                    weight = 1  # All weights are set to 1
                adj_matrix_edge_list.append((int(i), int(j), weight))

        # print(i, list_sync_event_series)
    end = time.time()
    print(end - start)
    np.save(savepath, adj_matrix_edge_list)
    print(f'Finished for job ID {job_id}')

    return adj_matrix_edge_list



def event_sync_reg(ind_list_e1, ind_list_e2, taumax, double_taumax):
    """
    ES for regional analysis that delivers specific timings.
    It returns the
    """
    sync_events = 0
    t12_lst = []
    t21_lst = []
    t_lst = []
    dyn_delay_lst = []

    for m, e1_ind in enumerate(ind_list_e1[1:-1], start=1):
        d_11_past = e1_ind-ind_list_e1[m-1]
        d_11_next = ind_list_e1[m+1]-e1_ind

        for n, e2_ind in enumerate(ind_list_e2[1:-1], start=1):
            d_12_now = (e1_ind-e2_ind)
            if d_12_now > taumax:
                continue

            d_22_past = e2_ind-ind_list_e2[n-1]
            d_22_next = ind_list_e2[n+1]-e2_ind

            tau = min(d_11_past, d_11_next, d_22_past, d_22_next, double_taumax) / 2
            if abs(d_12_now) <= tau:
                sync_events += 1
                dyn_delay_lst.append(d_12_now)
                if d_12_now < 0:
                    t12_lst.append(e1_ind)
                    t_lst.append(e1_ind)
                elif d_12_now > 0:
                    t21_lst.append(e2_ind)
                    t_lst.append(e2_ind)
                else:
                    t12_lst.append(e1_ind)
                    t21_lst.append(e2_ind)
                    t_lst.append(e2_ind)
            if d_12_now < -taumax:
                # print('break!',  d_12_now, e1_ind, e2_ind, )
                break

    return (t_lst, t12_lst, t21_lst, dyn_delay_lst)


def es_reg(es_r1, es_r2, taumax ):
    """
    """
    from itertools import product
    if es_r1.shape[1] != es_r2.shape[1]:
        raise ValueError("The number of time points of ts1 and ts 2 are not identical!")

    num_tp = es_r1.shape[1]
    es_r1 = prepare_es_input_data(es_r1)
    es_r2 = prepare_es_input_data(es_r2)



    comb_e12 = np.array(list(product(es_r1, es_r2)),dtype=object)
    backend = 'multiprocessing'
    # backend='loky'
    # backend='threading'
    num_cpus_avail = mpi.cpu_count()
    print(f"Number of available CPUs: {num_cpus_avail}")
    parallelArray = (Parallel(n_jobs=num_cpus_avail, backend=backend)
                     (delayed(event_sync_reg)
                      (e1,  e2,  taumax, 2*taumax)
                      for (e1, e2) in tqdm(comb_e12)
                      )
                     )

    t12 = np.zeros(num_tp, dtype=int)
    t21 = np.zeros(num_tp, dtype=int)
    t = np.zeros(num_tp)
    for (t_e, t12_e, t21_e, _) in parallelArray:
        t[t_e] += 1
        t12[t12_e] += 1
        t21[t21_e] += 1
    return t, t12, t21


def get_network_comb(c_indices1, c_indices2, adjacency=None):
    from itertools import product
    comb_c12 = np.array(list(product(c_indices1, c_indices2)), dtype=object)

    if adjacency is None:
        return comb_c12
    else:
        comb_c12_in_network = []
        for (c1, c2) in tqdm(comb_c12) :
            if adjacency[c1][c2] == 1 or adjacency[c2][c1] == 1:
                comb_c12_in_network.append([c1, c2])
        if len(comb_c12) == len(comb_c12_in_network):
            print("WARNING! All links in network seem to be connected!")
        return np.array(comb_c12_in_network, dtype=object)

def get_network_comb_es(c_indices1, c_indices2, ind_ts_dict1, ind_ts_dict2, adjacency=None):
    comb_c12_in_network = get_network_comb(c_indices1, c_indices2, adjacency=adjacency)
    print("Get combinations!")
    comb_e12 = []
    for (c1, c2) in comb_c12_in_network:
        e1 = ind_ts_dict1[c1]
        e2 = ind_ts_dict2[c2]
        comb_e12.append([e1, e2])
    comb_e12 = np.array(comb_e12, dtype=object)
    return comb_e12

def es_reg_network(ind_ts_dict1, ind_ts_dict2, taumax, adjacency=None):
    """
    ES between 2 regions. However, only links are considered that are found to be statistically significant
    """
    from itertools import product

    c_indices1 = ind_ts_dict1.keys()
    c_indices2 = ind_ts_dict2.keys()

    es1 = np.array(list(ind_ts_dict1.values()))
    es2 = np.array(list(ind_ts_dict2.values()))

    if es1.shape[1] != es2.shape[1]:
        raise ValueError("The number of time points of ts1 and ts 2 are not identical!")

    num_tp = es1.shape[1]
    es_r1 = prepare_es_input_data(es1)
    es_r2 = prepare_es_input_data(es2)

    ind_ts_dict1 = dict(zip(c_indices1, es_r1))
    ind_ts_dict2 = dict(zip(c_indices2, es_r2))

    backend = 'multiprocessing'

    comb_c12_in_network = get_network_comb(c_indices1, c_indices2, adjacency=adjacency)
    print("Get combinations!")
    comb_e12 = []
    for (c1, c2) in comb_c12_in_network:
        e1 = ind_ts_dict1[c1]
        e2 = ind_ts_dict2[c2]
        comb_e12.append([e1, e2])
    comb_e12 = np.array(comb_e12, dtype=object)
    # print(comb_e12)
    num_cpus_avail = mpi.cpu_count()
    print(f"Number of available CPUs: {num_cpus_avail}")
    parallelArray = (
                    Parallel(n_jobs=num_cpus_avail, backend=backend)
                            (delayed(event_sync_reg)
                             (e1, e2, taumax, 2*taumax)
                             for(e1, e2) in tqdm(comb_e12)
                             )
                    )

    t12 = np.zeros(num_tp, dtype=int)
    t21 = np.zeros(num_tp, dtype=int)
    t = np.zeros(num_tp)
    # dyn_delay_arr=np.array([])
    dyn_delay_arr = []
    for (t_e, t12_e, t21_e, dyn_delay) in tqdm(parallelArray):
        t[t_e] += 1
        t12[t12_e] += 1
        t21[t21_e] += 1
        dyn_delay_arr.append(dyn_delay)
        # dyn_delay_arr=np.concatenate([dyn_delay_arr, np.array(dyn_delay)], axis=0 )
    dyn_delay_arr = np.concatenate(dyn_delay_arr, axis=0)

    return t, t12, t21, dyn_delay_arr


# %%  Null model

def get_null_model_adj_matrix_from_E_files(E_matrix_folder, num_time_series,
                                           savepath=None):
    if os.path.exists(E_matrix_folder):
        path = E_matrix_folder
        E_matrix_files = [os.path.join(path, fn) for fn in next(os.walk(path))[2]]
    else:
        raise ValueError(f"E_matrix Folder {E_matrix_folder} does not exist!")
    adj_matrix = np.zeros((num_time_series, num_time_series), dtype=int)
    weight_matrix = np.zeros((num_time_series, num_time_series))

    for filename in tqdm(E_matrix_files):
        print(f"Read Matrix with name {filename}")
        if os.path.isfile(filename):
            this_E_matrix = np.load(filename)
        else:
            raise ValueError(f"WARNING! File does not exist {filename}!")

        for adj_list in tqdm(this_E_matrix):
            i, j = adj_list
            adj_matrix[i, j] = 1
    if savepath is not None:
        np.save(savepath, adj_matrix)
    print(f'Finished computing Adjency Matrix for Null model with {num_time_series} time series!')

    return adj_matrix


def null_model_one_series(i, min_num_events, l, num_permutations, taumax, double_taumax):
    list_thresholds_i = []
    for j in range(min_num_events, i + 1):
        season1 = np.zeros(l, dtype="bool")
        season2 = np.zeros(l, dtype="bool")
        season1[:i] = 1
        season2[:j] = 1
        dat = np.zeros((2, l), dtype="bool")
        cor = np.zeros(num_permutations)
        for k in range(num_permutations):
            dat[0] = np.random.permutation(season1)
            dat[1] = np.random.permutation(season2)
            ind_list_e1, ind_list_e2 = get_vector_list_index_of_extreme_events(dat)
            cor[k] = event_sync(ind_list_e1, ind_list_e2, taumax, double_taumax)
        th05 = np.quantile(cor, 0.95)
        th02 = np.quantile(cor, 0.98)
        th01 = np.quantile(cor, 0.99)
        th005 = np.quantile(cor, 0.995)
        th001 = np.quantile(cor, 0.999)

        list_thresholds_i.append([j, th05, th02, th01, th005, th001])

    return i, list_thresholds_i


def null_model_distribution(length_time_series, taumax=10,
                            min_num_events=10, max_num_events=1000,
                            num_permutations=3000, savepath=None):
    print("Start creating Null model of Event time series!")
    print(f"Model distribution size: {num_permutations}")
    l = length_time_series
    double_taumax = 2*taumax

    size = max_num_events-min_num_events
    # num_ij_pairs = ceil(size*(size + 1) / 2) #  "Kleiner Gauss"
    print(f"Size of Null_model Matrix: {size}")

    size = max_num_events
    P1 = np.zeros((size, size))
    P2 = np.zeros((size, size))
    P3 = np.zeros((size, size))
    P4 = np.zeros((size, size))
    P5 = np.zeros((size, size))

    # For parallel Programming
    num_cpus_avail = mpi.cpu_count()
    # num_cpus_avail=1
    print(f"Number of available CPUs: {num_cpus_avail}")
    backend = 'multiprocessing'
    # backend='loky'
    # backend='threading'

    # Parallelizing by using joblib
    parallelArray = (Parallel(n_jobs=num_cpus_avail, backend=backend)
                     (delayed(null_model_one_series)
                     (i, min_num_events, l, num_permutations, taumax, double_taumax)
                      for i in tqdm(range(min_num_events, max_num_events))
                      )
                     )

    print("Now store results in numpy array to hard drive!")
    for process in tqdm(parallelArray):
        i, list_thresholds_i = process
        for j_thresholds in list_thresholds_i:
            j, th05, th02, th01, th005, th001 = j_thresholds
            P1[i, j] = P1[j, i] = th05
            P2[i, j] = P2[j, i] = th02
            P3[i, j] = P3[j, i] = th01
            P4[i, j] = P4[j, i] = th005
            P5[i, j] = P5[j, i] = th001

    # Fill P for events smaller thresholds
    for i in range(0, min_num_events):
        for j in range(0, max_num_events):
            P1[i, j] = P1[j, i] = np.nan
            P2[i, j] = P2[j, i] = np.nan
            P3[i, j] = P3[j, i] = np.nan
            P4[i, j] = P4[j, i] = np.nan
            P5[i, j] = P5[j, i] = np.nan


    np.save(savepath + '_threshold_05.npy', P1)
    np.save(savepath + '_threshold_02.npy', P2)
    np.save(savepath + '_threshold_01.npy', P3)
    np.save(savepath + '_threshold_005.npy', P4)
    np.save(savepath + '_threshold_001.npy', P5)

    return P1, P2, P3, P4, P5


def null_model_cdf_one_series(i, min_num_events, l, num_permutations, taumax, double_taumax):
    list_thresholds_i = []
    for j in range(min_num_events, i + 1):
        season1 = np.zeros(l, dtype="bool")
        season2 = np.zeros(l, dtype="bool")
        season1[:i] = 1
        season2[:j] = 1
        dat = np.zeros((2, l), dtype="bool")
        cor = np.zeros(num_permutations)
        for k in range(num_permutations):
            dat[0] = np.random.permutation(season1)
            dat[1] = np.random.permutation(season2)
            ind_list_e1, ind_list_e2 = get_vector_list_index_of_extreme_events(dat)
            cor[k] = event_sync(ind_list_e1, ind_list_e2, taumax, double_taumax)

        norm_cdf = st.norm.cdf(cor)

        list_thresholds_i.append([j, norm_cdf])

    return i, list_thresholds_i


# %% Past processing
def construct_full_E(num_jobs, filename, savepath=None):
    # Load matrix for jobid 0
    print(f"Read data from {filename}")

    if os.path.exists(savepath):
        full_adj_matrix = np.load(savepath)
    else:
        full_adj_matrix = np.load(filename+'0.npy')
        for job_id in tqdm(range(1, num_jobs)):
            print(f"Read Matrix with ID {job_id}")
            this_filename = filename+str(job_id) + '.npy'
            if os.path.isfile(this_filename):
                this_adj_matrix = np.load(this_filename)
            else:
                continue
            full_adj_matrix = np.concatenate((full_adj_matrix, this_adj_matrix), axis=0)
            del this_adj_matrix

        print("Full length E_matrix: ", len(full_adj_matrix))
        if savepath is not None:
            np.save(savepath, full_adj_matrix)
    return full_adj_matrix


# %%
def cheby_lowpass(cutoff, fs, order, rp):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = cheby1(order, rp, normal_cutoff, btype='low', analog=False)
    return b, a


def cheby_lowpass_filter(x, cutoff, fs, order, rp):
    b, a = cheby_lowpass(cutoff, fs, order, rp)
    y = filtfilt(b, a, x)
    return y


# def get_locmax_of_score(ts, q=0.9):
#     locmax = np.array(argrelmax(ts)[0])
#     pscore= np.where(ts >= np.quantile(ts, q))[0]
#     sync_times=np.intersect1d(locmax, pscore)
#     return sync_times
def get_locmax_of_score(ts, q=0.9):
        q_value = np.quantile(ts, q)
        peaks, _ = find_peaks(ts, height=q_value, distance=1, prominence=1)
        sync_times = peaks
        return sync_times