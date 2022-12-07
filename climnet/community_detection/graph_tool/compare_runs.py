#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 08:53:08 2020
Class for network of rainfall events
@author: Felix Strnad
"""
# %%
from climnet.graph_tool.dendrograms import Dendrogram_ES
from climnet.monsoon.monsoon_region_es import Monsoon_Region_ES
import sys
import os
import numpy as np
import pandas as pd

import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH + "/../")  # Adds higher directory

# %%
"""Class of monsoon regions, containing the
monsoon definitions, the node ids and the regional monsoons."""


class Compare_Runs(Monsoon_Region_ES):
    """ Dataset for analysing the network output and
    comparing multiple cluster runs from graph tool.

    Args:
    ----------
    nc_file: str
        filename
    var_name: str
        Variable name of interest
    """

    def __init__(self,
                 var_name=None,
                 data_nc=None,
                 load_nc=None,
                 time_range=None,
                 month_range=None,
                 lon_range=[-180, 180],
                 lat_range=[-90, 90],
                 grid_step=1,
                 grid_type='gaussian',
                 lsm=False,
                 large_ds=False,
                 name='',
                 abs_th_wang=2, abs_th_ee=50, rel_th=0.55,
                 full_year=False,
                 **kwargs):
        super().__init__(data_nc=data_nc, load_nc=load_nc,
                         var_name=var_name, time_range=time_range,
                         lon_range=lon_range, lat_range=lat_range,
                         grid_step=grid_step, grid_type=grid_type,
                         month_range=month_range, large_ds=large_ds,
                         name=name, lsm=lsm,
                         abs_th_wang=abs_th_wang, abs_th_ee=abs_th_ee, rel_th=rel_th,
                         full_year=full_year,
                         **kwargs
                         )
        self.PATH = os.path.dirname(os.path.abspath(__file__))

    ############ Cluster Analysis ############
    def get_ids_loc_arr(self, loc_arr):
        sel_ids = []
        for loc in loc_arr:
            sel_ids.append(self.get_n_ids(loc))
        return sel_ids

    def get_density_cluster(self, loc_arr, num_runs=30,  abs_th=4, rel_th=0,
                            graph_folder=None, graph_file=None):
        """
        This function returns the main cluster in a band like structure for selected lat.
        """

        cluster_idx_arr = np.zeros((num_runs, len(self.indices_flat)))

        sel_ids = self.get_ids_loc_arr(loc_arr=loc_arr)

        for idx, job_id in tqdm(enumerate(range(0, num_runs))):
            if graph_folder is None:
                raise ValueError("Graph path folder not provided!")
            else:
                sbm_filepath = graph_folder + f"{job_id}_" + graph_file

            if not os.path.exists(sbm_filepath + '_group_levels.npy'):
                print(
                    f"WARNING file {sbm_filepath +'_group_levels.npy'} does not exist!")
                continue
            group_levels = np.load(
                sbm_filepath+'_group_levels.npy',  allow_pickle=True)
            d_es = Dendrogram_ES(group_levels,)

            # Compute the cluster which is given by the returned leaf node ids
            leaf_nodes = self.get_cluster_sel_ids(
                sel_ids=sel_ids, d_es=d_es, abs_th=abs_th, rel_th=rel_th)

            if len(leaf_nodes) == len(self.indices_flat):
                print(f"JobID {job_id}: Warning all nodes in one cluster!")
            for id in np.concatenate(sel_ids):
                if id not in leaf_nodes:
                    print(
                        f"JobId {job_id}: Warning monsoon id {id} not in cluster ids!")

            cluster_idx_arr[idx, :] = self.flat_idx_array(leaf_nodes)

        mean = np.mean(cluster_idx_arr, axis=0)
        std = np.std(cluster_idx_arr, axis=0)

        return mean, std

    def get_cluster_sel_ids(self, sel_ids, d_es=None, abs_th=0, rel_th=1):
        """
        Returns the node ids of the an array of selected node ids.
        """
        if d_es is None:
            d_es = self.d_es
        g_Zid = d_es.d_tree.get_split_groups(
            sel_ids, abs_th=abs_th, rel_th=rel_th)
        leaf_nodes = d_es.d_tree.get_leaf_nodes_of_is_node(g_Zid)

        return leaf_nodes

    def compare_grouping(self, loc, num_last_levels=5, num_runs=10, scott_factor=1,  graph_file=None):
        """
        Compares the groupings of group_levels for different runs

        Args
        ------
        loc: location as (lon, lat)
        """
        coord_deg, coord_rad, map_idx = self.get_coordinates_flatten()
        cluster_idx_arr = np.empty(
            (num_last_levels, num_runs, coord_rad.shape[0]))
        av_num_links = np.count_nonzero(
            self.climnet.adjacency)/self.climnet.adjacency.shape[0]

        # Scott's rule of thumb
        bw_opt = scott_factor * av_num_links**(-1./(2+4))
        for idx, job_id in enumerate(range(0, num_runs)):
            if graph_file is None:
                sbm_filepath = (self.PATH +
                                f"/graphs/{self.dataset_name}_{self.grid_step}/{job_id}_{self.dataset_name}_graph_tool_ES_{self.grid_step}")
            else:
                sbm_filepath = self.PATH + graph_file

            if not os.path.exists(sbm_filepath + '_group_levels.npy'):
                print(
                    f"WARNING file {sbm_filepath +'_group_levels.npy'} does not exist!")
                continue
            group_levels = np.load(
                sbm_filepath+'_group_levels.npy',  allow_pickle=True)
            d_es = Dendrogram_ES(group_levels,)

            node_levels = d_es.node_levels
            # ordered_node_levels, nl_lat_dict=self.parallel_ordered_nl_loc(node_levels)
            num_levels = len(node_levels)
            for lidx, lid in enumerate(range(num_levels - num_last_levels, num_levels)):
                # lat_this_level=nl_lat_dict[lid]['lat']
                # lon_this_level=nl_lat_dict[lid]['lon']
                # loc_this_level=list(zip(lon_this_level, lat_this_level))
                # g_id, _= self.find_min_distance(loc_this_level, loc)
                # g_id,_=self.find_nearest(lat_this_level, loc)

                idx_loc = self.get_index_for_coord(lon=loc[0], lat=loc[1])
                # Get the group number in which the location occurs
                g_id = node_levels[lid][idx_loc]

                leaf_nodes = np.where(node_levels[lid] == g_id)[0]

                c_coord = coord_rad[leaf_nodes]
                occ = lb.spherical_kde(c_coord, coord_rad, bw_opt=bw_opt)
                den = occ/max(occ)
                cluster_idx_arr[lidx, idx, :] = den

        mean_arr = []
        std_arr = []
        for lidx, lid in enumerate(range(num_levels - num_last_levels, num_levels)):

            mean, std, perc90, perc95, perc99, perc995, perc999 = lb.compute_stats(
                runs=cluster_idx_arr[lidx])
            mean_arr.append(mean)
            std_arr.append(std)
        return mean_arr, std_arr

    def compare_entropy(self, num_runs=10, max_num_levels=14, plot=False, savepath=None, graph_file=None, ax=None):

        sbm_entropy_arr = np.zeros((num_runs, max_num_levels))
        sbm_num_groups_arr = np.zeros((num_runs, max_num_levels))

        for idx, job_id in enumerate(range(0, num_runs)):
            if graph_file is None:
                sbm_filepath = self.PATH + \
                    f"/graphs/{self.dataset_name}_{self.grid_step}/{job_id}_{self.dataset_name}_graph_tool_ES_{self.grid_step}"
            else:
                sbm_filepath = self.PATH + graph_file

            if not os.path.exists(sbm_filepath + '_group_levels.npy'):
                print(
                    f"WARNING file {sbm_filepath +'_group_levels.npy'} does not exist!")
                continue
            sbm_entropy = np.load(
                sbm_filepath+'_entropy.npy',  allow_pickle=True)
            sbm_num_groups = np.load(
                sbm_filepath+'_num_groups.npy',  allow_pickle=True)

            sbm_entropy_arr[idx, :len(sbm_entropy)] = sbm_entropy
            sbm_num_groups_arr[idx, :len(sbm_num_groups)] = sbm_num_groups

            if plot is True:
                self.plot_entropy_groups(
                    entropy_arr=sbm_entropy, groups_arr=sbm_num_groups)

        mean_entropy, std_entropy, _, _, _, _, _ = lb.compute_stats(
            runs=sbm_entropy_arr)
        mean_num_groups, std_num_groups, _, _, _, _, _ = lb.compute_stats(
            runs=sbm_num_groups_arr)
        # -1 Because last level is trivial!
        mean_entropy = mean_entropy[:-1]
        std_entropy = std_entropy[:-1]
        mean_num_groups = mean_num_groups[:-1]
        std_num_groups = std_num_groups[:-1]

        # Now plot
        from matplotlib.ticker import MaxNLocator
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        num_levels = len(mean_entropy)
        ax.set_xlabel('Level')

        # Entropy
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(rf'Description Length $\Gamma$')

        x_data = np.arange(1, num_levels+1)
        ax.errorbar(x_data, (mean_entropy), yerr=(std_entropy),
                    color='tab:blue', elinewidth=2, label='Descr. Length')
        ax.fill_between(x_data, mean_entropy - std_entropy, mean_entropy + std_entropy,
                        color='tab:blue', alpha=0.3)
        ax.set_yscale('log')
        ax.yaxis.label.set_color('tab:blue')
        ax.tick_params(axis='y', colors='tab:blue')

        # Number of Groups
        ax1_2 = ax.twinx()
        ax1_2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax1_2.set_ylabel('Number of groups')
        ax1_2.errorbar(x_data, mean_num_groups, yerr=std_num_groups,
                       color='tab:green', label='Groups')
        ax1_2.fill_between(x_data, mean_num_groups - std_num_groups, mean_num_groups + std_num_groups,
                           color='tab:green', alpha=0.3)
        ax1_2.set_yscale('log')
        ax1_2.yaxis.label.set_color('tab:green')
        ax1_2.tick_params(axis='y', colors='tab:green')

        ax.legend(loc='upper right', bbox_to_anchor=(.25, 1),
                  bbox_transform=ax.transAxes)
        ax1_2.legend(loc='upper right', bbox_to_anchor=(
            1, 1), bbox_transform=ax.transAxes)

        # fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)

        if savepath is not None:
            print(f"Store files to: {savepath}")
            fig.savefig(savepath)

        return mean_entropy, std_entropy, mean_num_groups, std_num_groups
