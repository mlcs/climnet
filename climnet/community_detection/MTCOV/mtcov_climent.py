#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:11:04 2020

@author: Felix Strnad
"""
import os
import time
from climnet.community_detection.cd_base import BaseCommunityDetection
import climnet.community_detection.MTCOV.MTCOV as mtcov
import climnet.community_detection.cd_functions as cdf
import numpy as np
import networkx as nx
import sktensor as skt

# %%
""" Create a class for a graph tool object that is applicable on the precipitation ES dataset """


class MTCOV_Climnet(BaseCommunityDetection):
    """
    Dataset for Creating Clusters provided by the graph_tool package.
    """

    def __init__(self, network=None, weighted=False, dense_net=False, **kwargs):

        super().__init__(
            network=network, weight=weighted, dense_net=dense_net, **kwargs
        )

        self.A, self.B, self.nodes = self.prepare_graph(
            adjacency=self.net.adjacency, weights=self.net.corr, weighted=self.weighted
        )
        self.Xs = self.prepare_covariates(nodes=self.nodes)

    def run_mtcov(
        self,
        C,
        gamma=0.0,
        rseed=0,
        N_real=20,
        maxit=1000,
        flag_conv="deltas",
        undirected=True,
        batch_size=None,
    ):

        conf = dict(
            initialization=0,
            rseed=rseed,
            inf=10000000000.0,
            err_max=0.0000001,
            err=0.1,
            N_real=N_real,
            tolerance=0.0001,
            decision=10,
            maxit=maxit,
            out_inference=False,  # Don't use MTCOV storing function
            end_file="",
            assortative=True,
            verbose=True,
            files=None,
            out_folder=None,
            plot_loglik=False,
        )

        valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
        assert any(isinstance(self.B, vt) for vt in valid_types)

        print("\n### Run MTCOV ###")
        print(f"Using: \n number of communities = {C}\n gamma = {gamma}\n")

        time_start = time.time()
        MTCOV = mtcov.MTCOV(
            N=self.A[0].number_of_nodes(),
            L=len(self.A),
            C=C,
            Z=self.Xs.shape[1],
            gamma=gamma,
            undirected=undirected,
            **conf,
        )
        u, v, w, beta, maxL = MTCOV.fit(
            data=self.B,
            data_X=self.Xs,
            flag_conv=flag_conv,
            nodes=self.nodes,
            batch_size=batch_size,
        )

        print(
            "\nTime elapsed:",
            np.round(time.time() - time_start, 2),
            " seconds.",
            flush=True,
        )

        theta = {"u": u, "v": v, "w": w, "beta": beta, "maxL": maxL}
        self.theta = self.remove_empty_cluster(theta)
        self.hard_cluster = self.get_hard_cluster(
            theta=self.theta, ordered=True)

        return self.theta

    def save_communities(self, savepath, theta=None):
        if theta is None:
            theta = self.theta
        np.save(savepath, self.theta, allow_pickle=True)

    def prepare_graph(
        self,
        adjacency,
        weights,
        weighted=False,
        undirected=False,
        force_dense=True,
        noselfloop=True,
        verbose=True,
    ):
        """
            Import data, i.e. the adjacency tensor and the design matrix, from a given folder.

            Return the NetworkX graph, its numpy adjacency tensor and the dummy version of the design matrix.

            Parameters
            ----------
            in_folder : str
                        Path of the folder containing the input files.
            adj_name : str
                    Input file name of the adjacency tensor.
            cov_name : str
                    Input file name of the design matrix.
            ego : str
                Name of the column to consider as source of the edge.
            egoX : str
                Name of the column to consider as node IDs in the design matrix-attribute dataset.
            alter : str
                    Name of the column to consider as target of the edge.
            attr_name : str
                        Name of the attribute to consider in the analysis.
            undirected : bool
                        If set to True, the algorithm considers an undirected graph.
            force_dense : bool
                        If set to True, the algorithm is forced to consider a dense adjacency tensor.
            noselfloop : bool
                        If set to True, the algorithm removes the self-loops.
            verbose : bool
                    Flag to print details.

            Returns
            -------
            A : list
                List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
            B : ndarray/sptensor
                Graph adjacency tensor.
            X_attr : DataFrame
                    Pandas DataFrame object representing the one-hot encoding version of the design matrix.
            nodes : list
                    List of nodes IDs.
        """

        # df_adj = pd.read_csv(in_folder + adj_name, index_col=0) # read adjacency file
        print("\nAdjacency shape: {0}".format(adjacency.shape), flush=True)

        # create the graph adding nodes and edges
        A = self.read_graph(
            adj=adjacency,
            weights=weights,
            weighted=weighted,
            undirected=undirected,
            noselfloop=noselfloop,
            verbose=verbose,
        )

        nodes = list(A[0].nodes)
        print("\nNumber of nodes =", len(nodes), flush=True)
        print("Number of layers =", len(A), flush=True)
        if verbose:
            self.print_graph_stat(A)

        # save the multilayer network in a tensor with all layers
        if force_dense:
            B = self.build_B_from_A(A, nodes=nodes)
        else:
            B = self.build_sparse_B_from_A(A)

        return A, B, nodes

    def read_graph(
        self,
        adj,
        weights=None,
        weighted=False,
        undirected=False,
        noselfloop=True,
        verbose=True,
    ):
        """
            Create the graph by adding edges and nodes.

            Return the list MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.

            Parameters
            ----------
            df_adj : DataFrame
                    Pandas DataFrame object containing the edges of the graph.
            ego : str
                Name of the column to consider as source of the edge.
            alter : str
                    Name of the column to consider as target of the edge.
            undirected : bool
                        If set to True, the algorithm considers an undirected graph.
            noselfloop : bool
                        If set to True, the algorithm removes the self-loops.
            verbose : bool
                    Flag to print details.

            Returns
            -------
            A : list
                List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
        """

        # build the multilayer NetworkX graph: create a list of graphs, as many graphs as there are layers
        if verbose:
            print("Creating the network ...", end=" ", flush=True)

        L = 1  # FIXME adjust for multilayer networks

        if weighted:
            if weights is not None:
                if weights.shape == adj.shape:
                    # Taking abs to only have positive weights
                    corr = np.abs(np.where(adj == 1, weights, 0))
                else:
                    raise ValueError(f"adj shape not equal weights shape!")
                if undirected:
                    A = [nx.MultiGraph(corr) for _ in range(L)]
                else:
                    A = [nx.MultiDiGraph(corr) for _ in range(L)]
            else:
                raise ValueError(
                    "ERROR! You specified weighted but the network file is unweighted!"
                )
        else:
            if undirected:
                A = [nx.MultiGraph(adj) for _ in range(L)]
            else:
                A = [nx.MultiDiGraph(adj) for _ in range(L)]

        if verbose:
            print("done!", flush=True)

        # remove self-loops
        if noselfloop:
            if verbose:
                print("Removing self loops")
            for l in range(L):
                A[l].remove_edges_from(list(nx.selfloop_edges(A[l])))

        return A

    def prepare_covariates(self, nodes):
        """So far dummy covariates matrix

        Args:
            X ([type]): [description]

        Returns:
            [type]: [description]
        """
        size = len(nodes)
        X = np.zeros((size, 2))
        print("Indiv shape: ", X.shape)
        return np.array(X)

    def build_B_from_A(self, A, nodes=None):
        """
        Create the numpy adjacency tensor of a networkX graph.

        Parameters
        ----------
        A : list
            List of MultiDiGraph NetworkX objects.
        nodes : list
                List of nodes IDs.

        Returns
        -------
        B : ndarray
            Graph adjacency tensor.
        """

        N = A[0].number_of_nodes()
        if nodes is None:
            nodes = list(A[0].nodes())
        B = np.empty(shape=[len(A), N, N])
        for l in range(len(A)):
            B[l, :, :] = nx.to_numpy_matrix(
                A[l], weight="weight", dtype=int, nodelist=nodes
            )

        return B

    def build_sparse_B_from_A(self, A):
        """
            Create the sptensor adjacency tensor of a networkX graph.

            Parameters
            ----------
            A : list
                List of MultiDiGraph NetworkX objects.

            Returns
            -------
            data : sptensor
                Graph adjacency tensor.
        """

        N = A[0].number_of_nodes()
        L = len(A)

        d1 = np.array((), dtype="int64")
        d2 = np.array((), dtype="int64")
        d3 = np.array((), dtype="int64")
        v = np.array(())
        for l in range(L):
            b = nx.to_scipy_sparse_matrix(A[l])
            nz = b.nonzero()
            d1 = np.hstack((d1, np.array([l] * len(nz[0]))))
            d2 = np.hstack((d2, nz[0]))
            d3 = np.hstack((d3, nz[1]))
            v = np.hstack((v, np.array([b[i, j] for i, j in zip(*nz)])))
        subs_ = (d1, d2, d3)
        data = skt.sptensor(subs_, v, shape=(L, N, N), dtype=v.dtype)

        return data

    def print_graph_stat(self, A):
        """
            Print the statistics of the graph A.

            Parameters
            ----------
            A : list
                List of MultiGraph (or MultiDiGraph if undirected=False) NetworkX objects.
        """

        L = len(A)
        N = A[0].number_of_nodes()
        print("Number of edges and average degree in each layer:")
        avg_edges = 0
        avg_density = 0
        avg_M = 0
        avg_densityW = 0
        unweighted = True
        for l in range(L):
            E = A[l].number_of_edges()
            k = 2 * float(E) / float(N)
            avg_edges += E
            avg_density += k
            print(f"E[{l}] = {E} - <k> = {np.round(k, 3)}")

            weights = [d["weight"] for u, v, d in list(A[l].edges(data=True))]
            if not np.array_equal(weights, np.ones_like(weights)):
                unweighted = False
                M = np.sum([d["weight"]
                            for u, v, d in list(A[l].edges(data=True))])
                kW = 2 * float(M) / float(N)
                avg_M += M
                avg_densityW += kW
                print(f"M[{l}] = {M} - <k_weighted> = {np.round(kW, 3)}")

            print(f"Sparsity [{l}] = {np.round(E / (N * N), 3)}")

        print("\nAverage edges over all layers:", np.round(avg_edges / L, 3))
        print("Average degree over all layers:", np.round(avg_density / L, 2))
        print("Total number of edges:", avg_edges)
        if not unweighted:
            print("Average edges over all layers (weighted):",
                  np.round(avg_M / L, 3))
            print(
                "Average degree over all layers (weighted):",
                np.round(avg_densityW / L, 2),
            )
            print("Total number of edges (weighted):", avg_M)
        print(f"Sparsity = {np.round(avg_edges / (N * N * L), 3)}")

    def get_hard_cluster(self, theta, ordered=True):
        u = theta["u"]
        first_clustering = np.argmax(u, axis=1)
        un_clusters = np.unique(first_clustering)
        all_clusters = np.arange(u.shape[1])
        empty_groups = np.setdiff1d(all_clusters, un_clusters)
        if len(empty_groups) > 0:
            print(
                "WARNING! One cluster group has to low probability to be counted in argmax!"
            )
            theta = self.remove_empty_cluster(theta, idx_empty=empty_groups)

        hard_cluster = cdf.reduce_node_levels([first_clustering])
        num_diff_clusters = len(np.unique(hard_cluster))
        lu = theta["u"].shape[1]
        if num_diff_clusters != theta["u"].shape[1]:
            raise ValueError(
                f"Different length of hard_cluster array {num_diff_clusters} != {lu}"
            )
        if ordered:
            hard_cluster, loc_dict = self.parallel_ordered_nl_loc(hard_cluster)
            # Rearange order in theta
            print("reorder as well theta")
            self.theta = self.reorder_theta(
                theta, loc_dict[0]["ids"]
            )  # Zero for level id
        print(f"Number of different cluster {num_diff_clusters}")
        return np.array(hard_cluster[0])

    def remove_empty_cluster(self, theta, idx_empty=None):
        u = theta["u"]
        v = theta["v"]
        w = theta["w"]
        beta = theta["beta"]
        maxL = theta["maxL"]

        if idx_empty is None:
            idx_empty = np.where(w[0] < 0.001)[0]
        u = np.delete(u, idx_empty, axis=1)
        v = np.delete(v, idx_empty, axis=1)
        w = np.delete(w, idx_empty, axis=1)
        beta = np.delete(beta, idx_empty, axis=0)

        theta = {"u": u, "v": v, "w": w, "beta": beta, "maxL": maxL}

        return theta

    def reorder_theta(self, theta, permutation):

        u = theta["u"]
        v = theta["v"]
        w = theta["w"]
        beta = theta["beta"]
        maxL = theta["maxL"]

        lu = theta["u"].shape[1]
        lp = len(permutation)
        if len(permutation) != lu:
            print(permutation, theta["u"].shape)
            raise ValueError(
                f"Permutation array not of same length as node array {lp} != {lu}"
            )
        idx = np.empty_like(permutation)
        idx[permutation] = np.arange(lp)
        for a in [u, v, w]:
            a[:] = a[:, idx]
        beta = beta[idx, :]
        theta = {"u": u, "v": v, "w": w, "beta": beta, "maxL": maxL}

        return theta

    def load_communities(self, sp_theta):
        theta = np.load(sp_theta, allow_pickle=True).item()
        self.theta = self.remove_empty_cluster(theta)
        self.hard_cluster = self.get_hard_cluster(self.theta)
        return self.theta, self.hard_cluster

    def compute_prob_map(self, all_region_dict, arr_regions, sp_arr=None):
        collect_res = []
        gr_maps = []
        all_comm = []

        if len(self.theta_arr) < 1:
            if sp_arr is None:
                raise ValueError(
                    f'Please specify array of locations for theta {sp_arr}!')
            self.load_sp_arr(sp_arr)

        for run, theta_dict in enumerate(self.theta_arr):
            theta = theta_dict['theta']
            hard_cluster = theta_dict['hard_cluster']
            gr_num, hard_cluster = self.get_main_gr(all_region_dict, arr_regions,
                                                    theta=theta, hard_cluster=hard_cluster)

            this_prob_map = theta["u"][:, int(gr_num)]
            collect_res.append(this_prob_map)
            gr_map = self.ds.get_map(this_prob_map)
            gr_maps.append(gr_map)

            hc_map = self.net.ds.get_map(hard_cluster)
            all_comm.append(hc_map)

        prob_map = np.mean(collect_res, axis=0)
        prob_map_std = np.std(collect_res, axis=0)

        return prob_map, prob_map_std, gr_maps, all_comm
