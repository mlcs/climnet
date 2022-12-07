#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:11:04 2020

@author: Felix Strnad
"""
from climnet.community_detection.cd_base import BaseCommunityDetection
import climnet.community_detection.cd_functions as cdf
import climnet.utils.general_utils as gut
import graph_tool.all as gt
import numpy as np
import sys
import os

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH + "/../")  # Adds higher directory


# %%
""" Create a class for a graph tool object that is applicable on the precipitation ES dataset """


class ES_Graph_tool(BaseCommunityDetection):
    """
    Dataset for Creating Clusters provided by the graph_tool package.
    """

    def __init__(
        self, network=None, graph_file=None, rcg=False, weighted=False, **kwargs
    ):

        super().__init__(network=network, weight=weighted, **kwargs)

        if graph_file is not None:
            self.graph = self.construct_graph_from_adj_matrix(
                savepath=graph_file, rcg=rcg, weighted=self.weighted
            )
            # Check if graph is consistent with network file!
            g_N = self.graph.num_vertices()
            if g_N != len(self.ds.indices_flat):
                raise ValueError(
                    f"Too many indices in graph: {g_N} vs {len(self.ds.indices_flat)}!"
                )
        else:
            print("WARNING! No Graph file submitted!")

    def construct_graph_from_adj_matrix(self, savepath=None, rcg=False, weighted=False):
        # Preprocessing

        # ensure square matrix
        adj_matrix = self.net.adjacency
        M, N = adj_matrix.shape
        if os.path.isfile(savepath) and rcg is False:
            print(f"File already exists! Take file {savepath}")
            g = gt.load_graph(savepath)
            return g
        else:
            g_folder_path = os.path.dirname(savepath)
            if not os.path.exists(g_folder_path):
                os.makedirs(g_folder_path)
        if M != N:
            raise ValueError("Adjacency must be square!")

        # We start with an empty, directed graph
        g = gt.Graph()
        # Add N nodes
        g.add_vertex(N)

        edge_list = self.net.get_edgelist(weighted=weighted)
        self.net.check_network_dim()
        B = len(edge_list)
        print(f"Graph N {N}, B {B}")
        if weighted:
            print("Attention! Create weighted graph!")
            eweight = g.new_ep("double")
            g.properties[("e", "weight")] = eweight
            g.add_edge_list(edge_list, eprops=[eweight])
        else:
            g.add_edge_list(edge_list)
        print("Finished creating graph")

        print("Finished creating graph! Summary:")
        print(g)
        if savepath is not None:
            g.save(savepath)
            print(f"Graph File saved to {savepath}!")
        return g

    def get_sbm_matrix(self, e):
        """
        This functions stores the Matrix of edge counts between groups for the SBM

        Parameters
        ----------
        e : graph
            graph of the hierarchical level.

        Returns
        -------
        np.array
            Array of 2d-Matrices for each level.

        """

        matrix = e.todense()
        M, N = matrix.shape
        if M != N:
            print(f"Shape SB Matrix: {M},{N}")
            print(f"ERROR! SB matrix is not square! {M}, {N}")
            sys.exit(1)

        return np.array(matrix)

    def collect_node_membership(self, nodes):
        """Gives for every node the group idx

        Args:
            nodes (list): array of nodes to

        Returns:
            list: list of groups for node
        """
        group_membership_nodes = []
        for node in nodes:
            group_membership_nodes.append(node)

        return group_membership_nodes

    def apply_SBM(
        self,
        g,
        B_min=None,
        B_max=None,
        epsilon=None,
        equilibrate=False,
        multi_level=True,
        savepath=None,
        add_sbm=False,
    ):
        import time

        import warnings

        warnings.filterwarnings("ignore", category=FutureWarning)

        start = time.time()

        print("Start computing SBM on graph...")
        multilevel_mcmc_args = dict()
        if B_min is not None:
            multilevel_mcmc_args['B_min'] = B_min
        if B_max is not None:
            multilevel_mcmc_args['B_max'] = B_max
        # if epsilon is not None:
        #     multilevel_mcmc_args["epsilon"] = epsilon

        state_args = {}
        if self.weighted:
            # Compare
            # https://graph-tool.skewed.de/static/doc/demos/inference/inference.html#edge-weights-and-covariates
            weight_map = self.graph.edge_properties["weight"]
            state_args = dict(recs=[weight_map], rec_types=["real-normal"])
        if multi_level:
            print("Compute Nested Blockmodel with mulit levels...")
            state = gt.minimize_nested_blockmodel_dl(
                g,
                multilevel_mcmc_args=multilevel_mcmc_args,
                # deg_corr=True,
                # state_args=state_args,
            )
            print("Finished minimize Nested Blockmodel!")
            levels = state.get_levels()

            state.print_summary()
        else:
            print("Compute Blockmodel on single level...")
            state = gt.minimize_blockmodel_dl(
                g,
                # deg_corr=True,  # Changed in version 2.45
                multilevel_mcmc_args=multilevel_mcmc_args,
                # mcmc_args=mcmc_args,
                # mcmc_equilibrate_args=mcmc_equilibrate_args,
                # state_args=state_args,
            )
            levels = [state]
            print("Finished minimize 1 - level Blockmodel!")
        S1 = state.entropy()

        # we will pad the hierarchy with another xx empty levels, to give it room to potentially increase
        # TODO maybe try out first equilibration and then minimize nested blockmodel!
        if equilibrate is True:
            state = state.copy(bs=state.get_bs() +
                               [np.zeros(1)] * 4, sampling=True)
            state = state.copy(bs=state.get_bs(), sampling=True)
            print("Do further MCMC sweeps...")
            for i in range(2000):
                # ret = state.multiflip_mcmc_sweep(niter=20, beta=np.inf)
                state.multiflip_mcmc_sweep(niter=20, beta=np.inf)

            # Now equilibrate
            print("Now equilibrate results...")
            # gt.mcmc_equilibrate(state, nbreaks=2, wait=100, mcmc_args=dict(niter=10))

        S2 = state.entropy()
        print(f" Finished MCMC search! Improvement: {S2-S1}")

        # Now we collect the marginal distribution for exactly 100,000 sweeps
        niter = 1000
        N = g.num_vertices() + 1
        # E = g.num_edges()

        def collect_num_groups(s):
            num_levels = len(state.get_levels())
            h = np.zeros((num_levels, N))
            for li, sl in enumerate(s.get_levels()):
                B = sl.get_nonempty_B()
                h[li][B] += 1

        print(f"Sample from the posterior in {niter} samples!")
        # gt.mcmc_equilibrate(state, force_niter=niter, mcmc_args=dict(niter=10),
        #                 callback=collect_num_groups)
        print("Finished sampling from the posterior.")
        # gt.mcmc_anneal(state, beta_range=(1, 10), niter=1000, mcmc_equilibrate_args=dict(force_niter=10))

        # The hierarchical levels themselves are represented by individual BlockState instances
        # obtained via the get_levels() method:
        group_levels = []
        entropy_arr = []
        num_groups_arr = []
        sbm_matrix_arr = []
        num_groups_new = 0
        num_groups_old = N
        for s in levels:
            e_here = s.get_matrix()
            nodes = s.get_blocks()
            group_membership_nodes = self.collect_node_membership(nodes)
            num_groups_new = s.get_nonempty_B()

            if num_groups_new < num_groups_old:
                # Group numbers
                print(
                    f"New number of groups: {num_groups_new} < previous: {num_groups_old}"
                )
                group_levels.append(group_membership_nodes)
                # SBM Matrx
                sbm_matrix = self.get_sbm_matrix(e_here)
                sbm_matrix_arr.append(sbm_matrix)
                # Total number of groups
                num_groups_arr.append(num_groups_new)
                # Minimum description length
                entropy = s.entropy()
                entropy_arr.append(entropy)

                num_groups_old = num_groups_new

        group_levels = np.array(group_levels, dtype=object)
        group_levels = self.reduce_group_levels(group_levels)
        ng_last_level = gut.count_elements(group_levels[-1])
        ng = len(ng_last_level)
        # To conclude always with one cluster!
        if ng > 1:
            group_levels = np.array(group_levels.tolist() + [np.zeros(ng)])

        entropy_arr = np.array(entropy_arr)
        sbm_matrix_arr = np.array(sbm_matrix_arr, dtype=object)
        num_groups_arr = np.array(num_groups_arr)

        end = time.time()
        run_time = end - start
        print(f"Elapsed time for SBM: {run_time:.2f}")

        node_levels, _ = cdf.node_level_arr(group_levels)

        ordered_node_levels, _ = self.parallel_ordered_nl_loc(node_levels)
        if add_sbm:
            result_dict = dict(
                node_levels=ordered_node_levels,
                sbm=sbm_matrix_arr,
                group_levels=group_levels,
                entropy=entropy_arr,
            )
        else:
            result_dict = dict(
                node_levels=ordered_node_levels,
                group_levels=group_levels,
                entropy=entropy_arr,
            )

        self.theta = result_dict

        if savepath is not None:
            self.save_communities(savepath)

        return result_dict

    def save_communities(self, savepath, theta=None):
        if theta is None:
            theta = self.theta
        np.save(savepath, self.theta, allow_pickle=True)

    def load_communities(self, savepath):
        theta = np.load(savepath, allow_pickle=True).item()
        hard_cluster = self.get_hard_cluster(theta)

        return theta, hard_cluster

    def get_hard_cluster(self, theta=None, level=0):
        if theta is None:
            theta = self.theta

        hard_cluster = theta["node_levels"][level]

        return hard_cluster

    def compute_prob_map(self, all_region_dict, arr_regions, sp_arr=None, level=0):

        if len(self.theta_arr) < 1:
            if sp_arr is None:
                raise ValueError(
                    f'Please specify array of locations for theta {sp_arr}!')
            self.load_sp_arr(sp_arr)

        collect_res = []
        gr_maps = []
        all_comm = []
        for run, theta_dict in enumerate(self.theta_arr):
            theta = theta_dict['theta']
            hard_cluster = theta_dict['hard_cluster']
            gr_num, hard_cluster = self.get_main_gr(
                all_region_dict, arr_regions,
                theta=theta, hard_cluster=hard_cluster)
            this_prob_map = self.get_hc_for_gr_num(gr_num, theta=theta)
            collect_res.append(this_prob_map)

            gr_map = self.ds.get_map(this_prob_map)
            gr_maps.append(gr_map)

            all_hc = self.ds.get_map(hard_cluster)
            all_comm.append(all_hc)

        if len(all_comm) > 0:

            prob_map = np.mean(collect_res, axis=0)
            prob_map_std = np.std(collect_res, axis=0)
        else:
            raise ValueError(f'No file was existing!')

        return prob_map, prob_map_std, gr_maps, all_comm
