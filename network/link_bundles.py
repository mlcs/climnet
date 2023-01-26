#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 10:33:56 2021

@author: Felix Strnad
"""

# python libraries
import numpy as np
import os
import xarray as xr
from joblib import Parallel, delayed
import multiprocessing as mpi
from tqdm import tqdm
import geoutils.utils.general_utils as gut
import geoutils.utils.spatial_utils as sput
import climnet.grid.grid as grid


def obtain_kde_bandwidth(grid_step, nn_points_bw=1):
    """KDE bandwidth is set to nn_points_bw*max_dist_of_points.

    Parameters:
    ----------
    nn_points_bw: int
        Number of next-neighbor points of KDE bandwidth.
    """
    dist_eq = grid.degree2distance_equator(
        grid_step=grid_step, radius=grid.RADIUS_EARTH
    )
    bandwidth = nn_points_bw * dist_eq / grid.RADIUS_EARTH

    return bandwidth


def compute_stats(runs):
    mean = np.mean(runs, axis=0)
    std = np.std(runs, axis=0)
    perc90 = np.quantile(runs, 0.9, axis=0)
    perc95 = np.quantile(runs, 0.95, axis=0)
    perc99 = np.quantile(runs, 0.99, axis=0)
    perc995 = np.quantile(runs, 0.995, axis=0)
    perc999 = np.quantile(runs, 0.999, axis=0)

    return np.array([mean, std, perc90, perc95, perc99, perc995, perc999])


def link_bundle_null_model_link_number(coord_rad, num_links,
                                       folder, filename,
                                       bw=None,
                                       num_rand_permutations=1000):
    """
    Args:
    -----
    coord_rad: np.ndarray (num_nodes, 2)

    num_link: int

    folder: str

    filename: str

    bw: float

    num_rand_permutations: int

    """
    if num_links < 2:
        print("No number of links!")
        return None
    num_points = len(coord_rad)

    filename += f'_num_links_{num_links}_num_points_{num_points}.npy'

    if os.path.exists(folder+filename):
        return None

    null_model_bundles = np.zeros((num_rand_permutations, coord_rad.shape[0]))
    for s in range(num_rand_permutations):
        all_links_rand = np.vstack([np.random.choice(coord_rad[:, 0], num_links),
                                    np.random.choice(coord_rad[:, 1], num_links)]).T
        if np.isnan(all_links_rand).any():
            raise ValueError(
                f'Nan in rand links for num links {num_links}, bw={bw}')
        null_model_bundles[s, :] = sput.spherical_kde(
            all_links_rand, coord_rad, bw_opt=bw)

    stats = compute_stats(runs=null_model_bundles)
    np.save(folder + filename, stats)

    return None


def link_bundle_null_model(adj_matrix, coord_rad,
                           link_bundle_folder, filename,
                           bw=None,
                           num_rand_permutations=2000,
                           num_cpus=mpi.cpu_count(),
                           job_array=False):
    """Create link bundle null model.

    Args:
        adj_matrix (np.ndarray): Adjacency matrix of shape (num_nodes, num_nodes)
        coord_rad (np.ndarray): Map coordinates of nodes in rad [lat, lon] of shape (num_nodes, 2)
        link_bundle_folder (str): _description_
        filename (_type_): _description_
        bw (float, optional): KDE Bandwidth. Defaults to None.
        num_rand_permutations (int, optional): _description_. Defaults to 2000.
        num_cpus (_type_, optional): _description_. Defaults to mpi.cpu_count().
        job_array (bool, optional): When True the slurm job array is used to compute the
            different null model configurations. Defaults to False.
    """

    buff = []
    for i in range(0, adj_matrix.shape[0]):
        count_row = np.count_nonzero(adj_matrix[i, :])
        count_col = np.count_nonzero(adj_matrix[:, i])
        buff.append(count_row)
        buff.append(count_col)

    # shuffle link numbers to have fairer job times
    with gut.temp_seed():
        link_numbers = np.unique(buff)
        np.random.shuffle(link_numbers)

    if not os.path.exists(link_bundle_folder):
        os.makedirs(link_bundle_folder)
        gut.myprint(f"Created folder: {link_bundle_folder}!")
    else:
        gut.myprint(f"Save to folder: {link_bundle_folder}!")

    diff_links = len(link_numbers)
    gut.myprint(f'Compute {diff_links} number of different KDE tests...')

    # Compute Null model using slurm job-array
    if job_array:
        gut.myprint("Use job-array for parallelization of null model creation.")
        job_id, num_jobs = gut.get_job_array_ids()

        one_array_length = int(diff_links/num_jobs) + 1

        start_arr_idx = job_id * one_array_length
        end_arr_idx = (job_id + 1) * one_array_length
        link_numbers = link_numbers[start_arr_idx:end_arr_idx]
        gut.myprint(
            f"Job id {job_id}: Number of different links count {len(link_numbers)}.")

    # For parallel Programming
    gut.myprint(f"Number of available CPUs: {num_cpus} for link bundeling!")
    backend = 'multiprocessing'
    (Parallel(n_jobs=num_cpus, backend=backend)
             (delayed(link_bundle_null_model_link_number)
              (coord_rad, num_links,
               link_bundle_folder, filename, bw, num_rand_permutations)
              for num_links in tqdm(link_numbers))
     )

    return None


def link_bundle_one_location(adj_matrix, idx_node, coord_rad,
                             folder, filename,
                             bw=None,
                             perc=999,
                             plot=False,
                             verbose=False):
    """
    Args:
    -----
    adj_matrix: np.ndarray (num_nodes, num_nodes)

    coord_rad: np.ndarray (num_nodes, 2)

    Return:
    -------
    Dictionary with significant links

    """
    result_dic = {'idx_node': idx_node,
                  'significant_links': [],
                  'density': []}
    # Get links of node
    link_indices_node = np.where(adj_matrix[idx_node, :] > 0)[0]
    num_links = len(link_indices_node)
    num_points = len(coord_rad)
    link_coord = coord_rad[link_indices_node]
    if num_links < 2:  # 1 Link is by definition random
        if verbose:
            gut.myprint(f'Node with index {idx_node} has <2 links!')
        return result_dic

    # KDE
    Z_node = sput.spherical_kde(link_coord, coord_rad, bw_opt=bw)

    # read null model
    filename += f'_num_links_{num_links}_num_points_{num_points}.npy'
    if not os.path.exists(folder+filename):
        raise ValueError(
            f"Warning {folder}/{filename} does not exist, even though #links={num_links}>1!")

    mean, std, perc90, perc95, perc99, perc995, perc999 = np.load(
        folder + filename)

    if perc == 999:
        Z_rand = perc999
    elif perc == 995:
        Z_rand = perc995
    elif perc == 99:
        Z_rand = perc99
    elif perc == 95:
        Z_rand = perc95
    elif perc == 90:
        Z_rand = perc90
    else:
        raise ValueError("Choosen percentile does not exist!")

    # Check if density is significant
    significant_indices = np.intersect1d(
        np.where(Z_node > Z_rand)[0], link_indices_node
    )
    result_dic['significant_links'] = significant_indices

    if plot:
        sigdat = Z_node.copy()
        sigdat[Z_node > mean + 5 * std] = 5.5
        sigdat[Z_node <= mean + 5 * std] = 4.5
        sigdat[Z_node <= mean + 4 * std] = 3.5
        sigdat[Z_node <= mean + 3 * std] = 2.5
        sigdat[Z_node <= mean + 2 * std] = 1.5
        sigdat[-1] = 1.5
        sigdat[0] = 1.5

        result_dic['density'] = sigdat
    else:
        result_dic['density'] = Z_node

    return result_dic


def link_bundle_adj_matrix(adj_matrix, coord_rad,
                           null_model_folder, null_model_filename,
                           bw=None,
                           perc=999,
                           num_cpus=mpi.cpu_count()):
    """
    """
    gut.myprint(f"Number of available CPUs: {num_cpus}")

    backend = 'multiprocessing'
    results = (
        Parallel(n_jobs=num_cpus, backend=backend)
        (delayed(link_bundle_one_location)
            (adj_matrix, idx_node, coord_rad,
             null_model_folder, null_model_filename, bw, perc)
            for idx_node in tqdm(range(0, adj_matrix.shape[0]))
         )
    )

    # Now update Adjacency Matrix
    adj_matrix_corrected = np.zeros_like(adj_matrix)
    for result_dic in results:
        idx_node = result_dic['idx_node']
        links = result_dic['significant_links']

        adj_matrix_corrected[idx_node, links] = 1

    return adj_matrix_corrected
