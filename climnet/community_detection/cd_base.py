#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:11:04 2020

@author: Felix Strnad
"""
import numpy as np
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import climnet.network.network_functions as nwf
import climnet.community_detection.cd_functions as cdf
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH + "/../")  # Adds higher directory


# %%
""" Create a class for community detection algorithms object
that is applicable on the precipitation ES dataset
"""


class BaseCommunityDetection:
    """
    Dataset for Creating Clusters provided by the graph_tool package.
    """

    def __init__(self, network=None, weighted=False, dense_net=True, **kwargs):

        self.weighted = weighted
        self.theta_arr = []
        if network is not None:
            self.net = network
            is_nodes = nwf.get_isolated_nodes(adjacency=self.net.adjacency)
            if len(is_nodes) > 0:
                raise ValueError(
                    f"ERROR! The network contains {len(is_nodes)} isolated nodes!"
                )
            if self.weighted:
                self.corr = network.corr
            self.ds = self.net.ds
        else:
            raise ValueError("ERROR no network!")

        if self.weighted is True and self.net.weighted is False:
            print("WARNING! The network file is only provided unweighted!")
            self.weighted = False
        else:
            print("The graph for graph tool is unweighted")

    """ ################## Load arrays ###############################"""
    def load_sp_arr(self, sp_arr):
        self.theta_arr = []
        for run, sp_theta in enumerate(sp_arr):
            if os.path.exists(sp_theta):
                print(f'Use {sp_theta}...')
                theta, hard_cluster = self.load_communities(sp_theta)
                self.theta_arr.append(dict(theta=theta, hard_cluster=hard_cluster))
            else:
                print(f'File does not exist: {sp_theta}')
    """################### Groupings of nodes and clusters ############"""

    def node_level_dict(self, node_levels):
        """
        Gives for each level, for each group number which leaf nodes are in it.
        """
        node_level_dict = dict()
        for lid, level_ids in enumerate(node_levels):
            group_ids = np.unique(level_ids)
            this_level = []
            for idx, gid in enumerate(group_ids):
                node_idx = np.where(level_ids == gid)[0]
                if idx != int(gid):
                    raise ValueError(
                        f"Attention group ID missing: {gid} for idx {idx}!"
                    )
                this_level.append(node_idx)
            node_level_dict[lid] = this_level

        return node_level_dict

    def reduce_node_levels(self, node_levels):
        """
        Graph_tool with MCMC search does sometimes skip certain group numbers.
        This function brings back the ordering to numbers from 0 to len(level).
        """
        red_hierach_data = []
        trans_dict = dict()
        level_dict = cdf.level_dict(node_levels)
        node_red_dict = dict()
        for l_id, this_level_dict in enumerate(level_dict.values()):
            this_trans_dict = dict()
            for i, (group_id, group_count) in enumerate(this_level_dict.items()):
                this_trans_dict[group_id] = i
            trans_dict[l_id] = this_trans_dict

        for l_id, level_ids in enumerate(node_levels):
            this_level = []
            for level_id in level_ids:
                this_level.append(trans_dict[l_id][level_id])
                if l_id == 0:
                    node_red_dict[level_id] = trans_dict[l_id][level_id]
            red_hierach_data.append(this_level)

        return np.array(red_hierach_data)

    def get_upper_level_group_number(self, arr):
        """
        Returns for one level which groups belong to which group number
        in the upper level.
        """
        unique_sorted = np.unique(arr)
        orig_dict = dict()
        for i in unique_sorted:
            occ_in_arr = np.where(arr == i)
            orig_dict[i] = occ_in_arr[0]
        return orig_dict

    def get_hierarchical_data_from_nodes(self, node_through_levels):
        new_hierarchical_data = []
        node_group_dict = dict()
        for l_id, level in enumerate(node_through_levels):
            upper_level_groups = self.get_upper_level_group_number(level)
            this_level_arr = np.zeros(len(level), dtype=int)
            if l_id == 0:
                for (group_id, group_count) in upper_level_groups.items():
                    for i in group_count:
                        this_level_arr[i] = group_id
                        node_group_dict[i] = group_id
            else:
                lower_level = self.get_upper_level_group_number(
                    node_through_levels[l_id - 1]
                )
                this_level_arr = np.zeros(len(lower_level), dtype=int)

                for (group_id, group_count) in upper_level_groups.items():
                    for i in group_count:
                        this_group = node_group_dict[i]
                        this_level_arr[this_group] = group_id
                        node_group_dict[i] = group_id
            new_hierarchical_data.append(this_level_arr)

        return np.array(new_hierarchical_data, dtype=object)

    def reduce_group_levels(self, group_levels):
        node_levels, _ = cdf.node_level_arr(group_levels)
        new_node_levels = self.reduce_node_levels(node_levels)
        red_group_levels = self.get_hierarchical_data_from_nodes(new_node_levels)

        return red_group_levels

    def foreward_node_levels(self, node_levels, node_id):
        foreward_path = []
        for level in node_levels:
            group_num = level[node_id]
            foreward_path.append(group_num)
        return np.array(foreward_path)

    def get_sorted_loc_gid(self, group_level_ids, lid):
        mean_lat_arr = []
        mean_lon_arr = []
        result_dict = dict()
        for gid, node_ids in enumerate(group_level_ids):
            lon_arr = []
            lat_arr = []
            for nid in node_ids:
                map_idx = self.ds.get_map_index(nid)
                lon_arr.append(map_idx["lon"])
                lat_arr.append(map_idx["lat"])
            # loc_dict[gid]=[np.mean(lat_arr), np.mean(lon_arr)]
            mean_lat_arr.append(np.mean(lat_arr))
            mean_lon_arr.append(np.mean(lon_arr))
        # sorted_ids=np.argsort(mean_lat_arr)  # sort by arg
        sorted_ids = [
            sorted(mean_lat_arr).index(i) for i in mean_lat_arr
        ]  # relative sorting
        if len(sorted_ids) != len(mean_lat_arr):
            raise ValueError("Error! two lats with the exact same mean!")
        sorted_lat = np.sort(mean_lat_arr)  # sort by latitude
        sorted_lon = np.array(mean_lon_arr)[sorted_ids]
        result_dict["lid"] = lid
        result_dict["sorted_ids"] = sorted_ids
        result_dict["sorted_lat"] = sorted_lat
        result_dict["sorted_lon"] = sorted_lon
        return result_dict

    def parallel_ordered_nl_loc(self, node_levels):
        import multiprocessing as mpi

        nl_dict = self.node_level_dict(node_levels)
        new_node_levels = np.empty_like(node_levels)

        # For parallel Programming
        num_cpus_avail = mpi.cpu_count()
        backend = "multiprocessing"
        parallelSortedLoc = Parallel(n_jobs=num_cpus_avail, backend=backend)(
            delayed(self.get_sorted_loc_gid)(group_level_ids, lid)
            for lid, group_level_ids in tqdm(nl_dict.items())
        )
        loc_dict = dict()

        for result_dict in parallelSortedLoc:
            lid = result_dict["lid"]
            sorted_ids = result_dict["sorted_ids"]
            loc_dict[lid] = {
                "lat": np.array(result_dict["sorted_lat"]),
                "lon": np.array(result_dict["sorted_lon"]),
                "ids": np.array(result_dict["sorted_ids"]),
            }
            for gid, node_ids in enumerate(nl_dict[lid]):
                new_node_levels[lid][node_ids] = sorted_ids[gid]

        return new_node_levels, loc_dict

    def ordered_nl_loc(self, node_levels):
        nl_dict = self.node_level_dict(node_levels)
        new_node_levels = np.empty_like(node_levels)
        for lid, group_level_ids in tqdm(nl_dict.items()):
            res_sorted_dict = self.get_sorted_loc_gid(group_level_ids, lid)
            sorted_loc = res_sorted_dict["sorted_loc"]  # sort by latitude
            for gid, node_ids in enumerate(group_level_ids):
                new_node_levels[lid][node_ids] = sorted_loc[gid]

        return new_node_levels

    def compute_link_density(self, n, num_edges):
        pot_con = n * (n - 1) / 2  # Kleiner Gauss
        den = num_edges / pot_con
        return den

    def get_main_gr(self, region_dict, names, theta=None, hard_cluster=None):
        if theta is None:
            theta = self.theta
            hard_cluster = self.get_hard_cluster(theta)

        all_gr_numbers = np.array([])
        max_cnts = 0
        for region in names:
            rep_ids = region_dict[region]["rep_ids"]
            gr_numbers = hard_cluster[rep_ids]
            all_gr_numbers = np.concatenate((all_gr_numbers, gr_numbers), axis=0)
            max_cnts += len(rep_ids)

        un_gr_nums, cnts = np.unique(all_gr_numbers, return_counts=True)
        this_gr = int(un_gr_nums[np.argmax(cnts)])

        rel_freq = np.max(cnts) / max_cnts
        if rel_freq < 0.5:
            print(
                f"WARNING! Less than half of all rep ids is in community: \
                {np.max(cnts)}/{max_cnts}!"
            )

        return this_gr, hard_cluster

    def get_hc_for_gr_num(self, gr_num, theta=None):
        hard_cluster = self.get_hard_cluster(theta=theta)
        this_gr = np.where(hard_cluster == gr_num, 1, 0)

        return this_gr
