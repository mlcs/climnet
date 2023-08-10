#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:11:04 2020

@author: Felix Strnad
"""
from importlib import reload
import networkx as nx
import networkit as nk
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
import scipy.stats as st
import numpy as np
import xarray as xr
import sys
import os
from tqdm import tqdm
import multiprocessing as mpi
from joblib import Parallel, delayed
import climnet.grid.grid as grid
import climnet.network.network_functions as nwf
import climnet.network.corr_network as cnet
import climnet.network.es_network as esnet
import climnet.network.link_bundles as lb
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut


PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH + "/../")  # Adds higher directory


# %%
""" Create a class for a graph tool object that is applicable on the precipitation ES dataset """


class Clim_NetworkX:
    """
    Dataset for Creating Clusters provided by the graph_tool package.
    """

    def __init__(
        self, dataset, nx_path_file=None, network=None, **kwargs
    ):

        self.ds = dataset
        self.adjacency = None
        self.corr = None
        self.reset_is_points_arr()
        if network is not None and nx_path_file is not None:
            raise ValueError(
                "ERROR both nx_path_file and network are provided, but only 1 can be used! Please choose one of both!")
        if network is not None:
            self.create(network=network)
        elif nx_path_file is None:
            return
        else:
            self.cnx = self.load(nx_path_file)

    def create(self, method='corr', network=None, weighted=False, **kwargs):

        self.weighted = weighted
        if network is not None:
            print(f'Create Network based on input network', flush=True)
            self.adjacency = network.adjacency
            self.corr = np.where(self.adjacency == 1, network.corr, 0)
        else:
            print(f'Create Network based on method {method}', flush=True)
            if method == 'corr':
                reload(cnet)
                self.adjacency, self.corr = cnet.create(
                    self.ds, **kwargs
                )
            elif method == 'es':
                reload(esnet)
                self.adjacency, self.corr = esnet.create(
                    self.ds, **kwargs
                )

        if weighted is False:
            self.corr = None
        # ensure square matrix
        M, N = self.adjacency.shape

        if M != N:
            raise ValueError("Adjacency must be square!")
        self.cnx = self.init_cnx()
        self.remove_isolated_nodes()
        # Check if graph is consistent with network file!
        self.check_network_dim()
        self.get_nx_info()

        return self.cnx

    def init_cnx(self):
        if self.corr is not None:
            gut.myprint(
                f'Init the network by correlation matrix {self.corr.shape}')
            self.weighted = True
            self.corr = np.where(self.adjacency == 1, self.corr, 0)
            # Taking abs to only have positive weights
            cnx = nx.DiGraph(np.abs(self.corr))
            cnx = self.set_edge_corr(cnx)
        else:
            gut.myprint(
                f'Init the network by adjacency matrix {self.adjacency.shape}')
            cnx = nx.DiGraph(self.adjacency)

        self.cnx = self.post_process_cnx(cnx)
        return self.cnx

    def print_properties(self):
        print(self.cnx)
        self.get_nx_info()
        return None

    def post_process_cnx(self, cnx):
        gut.myprint(f'Set dataset attributes to network....')
        for n in tqdm(cnx.nodes):
            # important to use the correct point ids!
            pid = self.ds.get_points_for_idx([n])[0]
            cnx.nodes[n]["lon"] = float(self.ds.ds.points[pid].lon)
            cnx.nodes[n]["lat"] = float(self.ds.ds.points[pid].lat)
        # for the curvature computation to work, we need to convert the labels first
        cnx = nx.convert_node_labels_to_integers(cnx)
        # self.adjacency = np.where(nx.to_numpy_array(cnx) > 0, 1, 0)  # Check again why this is wrong!
        self.cnx = cnx
        self.check_network_dim()

        if self.corr is not None:
            self.set_corr_edges()

        return cnx

    def set_edge_corr(self, cnx):
        if self.corr is None:
            raise ValueError(f'ERROR correlation is None!')
        if self.add_attr2ds is None:
            raise ValueError(f'ERROR adjacency is None!')
        print('Set Edge correlation!', flush=True)
        sids, tids = np.where(self.adjacency == 1)
        pairs = list(zip(sids, tids))
        corr_lst = [self.corr[p] for p in pairs]
        corr_dict = gut.mk_dict_2_lists(pairs, corr_lst)

        nx.set_edge_attributes(cnx, corr_dict, "corr")
        return cnx

    def make_network_undirected(self, dense=True):
        # Make sure that adjacency is symmetric (ie. in-degree = out-degree)
        gut.myprint(f"Make network undirected with dense={dense}")
        self.adjacency, self.corr = nwf.make_network_undirected(
            adjacency=self.adjacency, corr=self.corr, dense=dense
        )
        self.init_cnx()

        return

    def load(self, nx_path_file):
        # Load networkX file
        self.cnx = fut.load_nx(filepath=nx_path_file)

        self.is_degraph()
        self.adjacency = nwf.nx_to_adjacency(graph=self.cnx)
        # self.corr = self.get_attr_array(attr='corr')  # Not all networks have corr
        # if self.adjacency.shape != self.corr.shape:
        #     gut.myprint(self.adjacency.shape, self.corr.shape)
        #     raise ValueError(f'ERROR Corr and adjacency not of same shape!')
        node_attr = self.get_node_attributes()
        if 'rempoints' in node_attr:
            gut.myprint('Load removal of isolated points...')
            self.is_points = []
            # Iterate over all nodes that might contain a removed point number
            for nd, ndata in self.cnx.nodes(data=True):
                if 'rempoints' in list(ndata.keys()):
                    self.is_points.append(int(ndata['rempoints']))
                else:
                    break
            self.is_points = np.array(self.is_points)
            self.ds.mask_point_ids(points=self.is_points)
            self.ds.re_init()
        is_nodes = nwf.get_isolated_nodes(adjacency=self.adjacency)
        if len(is_nodes) > 0:
            raise ValueError(
                f"ERROR! The network contains {len(is_nodes)} isolated nodes!"
            )
        # self.cnx = nx.relabel.convert_node_labels_to_integers(self.cnx)  # Otherwise node labels are strings!
        self.create_ds()

        # Check if graph is consistent with network file!
        self.check_network_dim()
        self.weighted = self.is_weighted()
        self.get_nx_info()

        return self.cnx

    def is_degraph(self):
        if isinstance(self.cnx, nx.classes.graph.Graph):
            gut.myprint('WARNING! Undirected graph object!')
            self.cnx = nx.DiGraph(self.cnx)
            gut.myprint('Casted to DiGraph object!')

    def is_weighted(self):
        all_weights = self.get_edge_attr('weight')
        if int(np.sum(all_weights)) == len(all_weights):  # if all weights are 1 == num edges!
            weighted = False
            print('Network is unweighted!')
        else:
            weighted = True
            print('Network is weighted!')

        return weighted

    def save(self, savepath, ds_savepath=None, cnx=None):
        """Stores the network as gml file. Note, to save memory space, it is
        useful to store it as .gml.gz files (compressed files).

        Args:
            savepath (str): file path. good to end with .gml.gz.
            cnx (climNetworkX, optional): climNetworkx. By default the self.cnx is stored.
        """
        if cnx is None:
            cnx = self.cnx
        gut.myprint(f" Start saving ClimNetworkx File to {savepath}!")

        self.set_removed_points()
        cnx = nx.convert_node_labels_to_integers(cnx)
        nx.write_gml(cnx, savepath)
        gut.myprint(f"ClimNetworkx File saved to {savepath}!")
        if ds_savepath is not None:
            gut.myprint(
                f" Start saving as well dataset File to {ds_savepath}!")
            self.ds.save(filepath=ds_savepath)

    # ########################### Link bundling ########################################
    def link_bundles(
        self,
        num_rand_permutations,
        num_cpus=mpi.cpu_count(),
        nn_points_bw=None,
        link_bundle_folder=None,
        job_array=False,
    ):
        """Significant test for adjacency.

        Args:
            num_rand_permutations (_type_): _description_
            num_cpus (_type_, optional): _description_. Defaults to mpi.cpu_count().
            nn_points_bw (_type_, optional): _description_. Defaults to None.
            link_bundle_folder (_type_, optional): _description_. Defaults to None.
            job_array (bool, optional): When True the slurm job array is used to compute the
                different null model configurations. Defaults to False.
        """
        reload(lb)
        # Get coordinates of all nodes
        coord_deg, coord_rad, map_idx = self.ds.get_coordinates_flatten()

        # First compute Null Model of old adjacency matrix
        if link_bundle_folder is None:
            link_bundle_folder = PATH + f"/link_bundles/{self.ds.var_name}/"
        else:
            link_bundle_folder = PATH + f"/link_bundles/{link_bundle_folder}/"
        null_model_filename = f"link_bundle_null_model_{self.ds.var_name}"

        # Set KDE bandwidth to 2*max_dist_of_points
        if nn_points_bw is not None:
            dist_eq = grid.degree2distance_equator(
                self.ds.grid_step, radius=grid.RADIUS_EARTH
            )
            bandwidth = nn_points_bw * dist_eq / grid.RADIUS_EARTH
        else:
            bandwidth = None  # Is computed later based on Scott's rule of thumb!

        gut.myprint(
            f"Start computing null model of link bundles using {bandwidth}!")
        lb.link_bundle_null_model(
            self.adjacency,
            coord_rad,
            link_bundle_folder=link_bundle_folder,
            filename=null_model_filename,
            num_rand_permutations=num_rand_permutations,
            num_cpus=num_cpus,
            bw=bandwidth,
            job_array=job_array,
        )

        # Now compute again adjacency corrected by the null model of the link bundles
        # try:
        gut.myprint("Now compute new adjacency matrix!")
        self.adjacency = lb.link_bundle_adj_matrix(
            adj_matrix=self.adjacency,
            coord_rad=coord_rad,
            null_model_folder=link_bundle_folder,
            null_model_filename=null_model_filename,
            bw=bandwidth,
            perc=999,
            num_cpus=num_cpus,
        )
        # except:
        #     print(
        #         "Other jobs for link bundling are not finished yet! Last job will do the rest!"
        #     )
        #     sys.exit()
        # Correct the weight matrix accordingly to the links
        if self.corr is not None:
            self.corr = np.where(self.adjacency == 1, self.corr, 0)

        self.lb = True
        self.remove_isolated_nodes()

    # ########### General Functions for better handeling networkx ###################

    def check_network_dim(self):
        g_N = self.cnx.number_of_nodes()
        g_E = self.cnx.number_of_edges()
        if g_N != len(self.ds.indices_flat):
            raise ValueError(
                f"Too many indices in graph: {g_N} vs {len(self.ds.indices_flat)}!"
            )

        adj_len = self.adjacency.shape[0]
        if g_N != adj_len:
            raise ValueError(
                f"Different number of nodes {g_N} than adjacency shape {adj_len}!"
            )

        ed_num_net = np.count_nonzero(self.adjacency)
        if g_E != ed_num_net:
            if int(2 * g_E) != ed_num_net:
                raise ValueError(
                    f"Different number of edges: graph file: {g_E} vs net file:{ed_num_net}!"
                )
            else:
                gut.myprint("WARNING! Adjacency is undirected and symmetric!")

    def set_corr_edges(self):
        """Setting the edges values to the corresponding correlation values
        """
        if self.corr is None:
            raise ValueError(
                f"ERROR, no correlation values provided {self.corr}!")
        for i, row in enumerate(self.adjacency):
            idx_links = np.where(row == 1)[0]
            for j in idx_links:
                val = self.corr[i, j]
                self.cnx.edges[i, j]["corr"] = float(val)
        gut.myprint("Finished setting edge corr values!", flush=True)

    def set_weight_edges(self):
        """
        Setting the edges correlation values to the corresponding edge!
        """
        if self.corr is None:
            raise ValueError(
                f"ERROR, no correlation values provided {self.corr}!")
        for i, row in enumerate(self.adjacency):
            idx_links = np.where(row == 1)[0]
            for j in idx_links:
                val = self.corr[i, j]
                self.cnx.edges[i, j]["weight"] = np.abs(float(val))
        gut.myprint("Finished setting edge weight values!", flush=True)

    def get_sparsity(self):
        """Obtain sparsity of adjacency matrix."""
        sparsity = nwf.get_sparsity(M=self.adjacency)
        return sparsity

    def get_nx_info(self):
        reload(nwf)
        # print(nx.info(self.cnx))  # depracted in networkx3
        num_nodes = self.cnx.number_of_nodes()
        num_edges = self.cnx.number_of_edges()
        sparsity = nwf.get_sparsity(M=self.adjacency, verbose=False)
        attrs = self.get_node_attributes()
        ed_attrs = self.get_edge_attributes()

        nx_info_dict = {'Num_nodes': num_nodes, 'Num_edges': num_edges,
                        'sparsity': sparsity,
                        'Node_attrs': attrs, 'Edge_attrs': ed_attrs}

        gut.myprint(f'Network Info: {nx_info_dict}')

        return nx_info_dict

    def get_threshold(self):
        all_e_vals = self.get_edge_attr("weight")
        min_val = np.min(all_e_vals)
        return min_val

    def remove_nodes_from_adjacency(self, idx_list):
        """Removes list of indices row and column wise from the adjacency matrix
        and also the correlation matrix.

        Args:
            idx_list (list): list of indices

        Raises:
            ValueError:
            ValueError:
        """
        frac_is = len(idx_list) / len(self.adjacency)
        gut.myprint(
            f"WARNING! Removed nodes from network! Fraction: {frac_is:.4f}")
        # This removes unconnected nodes as well from the network
        adj_rem_rows = np.delete(
            self.adjacency, idx_list, axis=0
        )  # Delete rows
        adj_rem_cols_rows = np.delete(
            adj_rem_rows, idx_list, axis=1
        )  # Delete columns
        M, N = adj_rem_cols_rows.shape
        if M != N:
            raise ValueError(
                f"Deleted different number of rows{M} and columns {N}")
        else:
            self.adjacency = adj_rem_cols_rows

            if self.corr is not None:
                corr_rows = np.delete(
                    self.corr, idx_list, axis=0
                )  # Delete rows
                self.corr = np.delete(
                    corr_rows, idx_list, axis=1
                )  # Delete columns

                if self.corr.shape != self.adjacency.shape:
                    raise ValueError(
                        f"Corr shape {self.corr.shape} != Adj shape {self.adjacency.shape}!"
                    )
        return

    def reset_is_points_arr(self):
        self.is_points = np.array([])

    def remove_isolated_nodes(self):
        isolated_nodes = nwf.get_isolated_nodes(adjacency=self.adjacency)
        # While because maybe remove of lines will create further empty lines
        is_points = self.ds.get_points_for_idx(
            isolated_nodes) if len(isolated_nodes) > 0 else []
        while len(isolated_nodes) > 0:
            # Remove nodes from adjacency
            self.remove_nodes_from_adjacency(idx_list=isolated_nodes)

            # This masks the nodes as well in the dataset
            gut.myprint("Update Dataset as well and remove unconnected nodes")
            self.ds.mask_node_ids(isolated_nodes)
            if self.corr is None:
                isolated_nodes = nwf.get_isolated_nodes(
                    adjacency=self.adjacency)
            else:
                # Maybe some weights are 0 even though adjacency is 1
                isolated_nodes = nwf.get_isolated_nodes(adjacency=self.corr)
            is_points = np.append(
                is_points, self.ds.get_points_for_idx(isolated_nodes))

        self.ds.re_init()
        self.is_points = np.append(
            self.is_points, is_points)
        self.set_removed_points()
        # Re-Initialize the network with the the new adjacency!
        gut.myprint('Re-Init the new network')
        self.init_cnx()

        return None

    def remove_nodes_from_network(self, node_list):
        self.remove_nodes_from_adjacency(idx_list=node_list)
        self.init_cnx()  # This reinitializes the nx object with nx.Graph()
        return

    def set_removed_points(self, points=None):
        rem_points = self.is_points
        # Consistency check
        max_num_points = len(self.ds.ds.points)

        max_rem_pid = np.max(rem_points) if len(rem_points) > 0 else 0
        if max_rem_pid >= max_num_points:
            raise ValueError(
                f'To remove ids {max_rem_pid} >= number of datapoints {max_num_points}!')

        if points is not None:
            rem_points = np.append(points, rem_points)
        rem_points = np.array(rem_points, dtype=int)
        rem_nodes_dict = gut.mk_dict_2_lists(key_lst=np.arange(len(rem_points)),
                                             val_lst=rem_points.tolist())  # To list important np.int64 will create Error!
        nx.set_node_attributes(
            self.cnx, rem_nodes_dict, name="rempoints")
        return

    def cut_network(self, lon_range=None, lat_range=None, dateline=False):
        gut.myprint(f'Cut the dataset first...')
        rem_dict = self.ds.cut_ds(lon_range=lon_range, lat_range=lat_range,
                                  dateline=dateline)
        rem_node_ids = rem_dict['remove_idx']
        self.remove_nodes_from_network(node_list=rem_node_ids)
        gut.myprint('Reset isolated points...')
        # Reset previously removed point number
        self.reset_is_points_arr()

        # Remove nodes that are maybe now isolated
        gut.myprint('Remove possibly newly isolated nodes')
        self.remove_isolated_nodes()
        self.check_network_dim()
        gut.myprint('Finished setting up network!')
        self.get_nx_info()
        return

    # ####################  Compute network attributes ############

    def compute_network_attrs(self, *attr, rc_attr=False):
        reload(nwf)
        if len(attr) == 0:
            attr = ["degree", "betweenness", "clustering"]
            if 'weight' in self.get_node_attributes():
                attr += "weight"
        if "degree" in attr:
            if not self.node_attr_exists(attr="degree") or rc_attr is True:
                self.cnx = nwf.degree(netx=self.cnx, weighted=False)
                if self.weighted:
                    self.cnx = nwf.degree(netx=self.cnx, weighted=True)
                self.cnx = nwf.in_degree(netx=self.cnx)
                self.cnx = nwf.out_degree(netx=self.cnx)
                self.cnx = nwf.divergence(netx=self.cnx)

        if "weighted_degree" in attr:
            if not self.node_attr_exists(attr="weighted_degree") or rc_attr is True:
                self.cnx = nwf.weighted_degree(netx=self.cnx)

        if "betweenness" in attr:
            if not self.node_attr_exists(attr="betweenness") or rc_attr is True:
                self.cnx = nwf.betweenness(netx=self.cnx)

        if "clustering" in attr:
            if not self.node_attr_exists(attr="clustering") or rc_attr is True:
                self.cnx = nwf.clustering_coeff(netx=self.cnx)

        if 'transitivity' in attr:
            # Sets all nodes to the same value!
            if not self.node_attr_exists(attr="transitivity") or rc_attr is True:
                self.cnx = nwf.transitivity(netx=self.cnx)

        if 'triangles' in attr:
            if not self.node_attr_exists(attr="triangles") or rc_attr is True:
                self.cnx = nwf.triangles(netx=self.cnx)

        self.create_ds()

        return self.cnx

    def node_attr_exists(self, attr):
        node_attr = self.get_node_attributes()
        if attr == "forman":
            attr = "formanCurvature"
        if attr == "ollivier":
            attr = "ricciCurvature"
        if attr in node_attr:
            print(f"Attribute {attr} already exists!", flush=True)
            return True
        else:
            return False

    def edge_attr_exists(self, attr):
        edge_attr = self.get_edge_attributes()
        if attr in edge_attr:
            return True
        else:
            return False

    def compute_curvature(self, c_type="forman",
                          rc_curv=False):
        """Creates Networkx with Forman or Ollivier curvatures

        Args:
            c_type (str, optional): curvature type (Forman or Ollivier). Defaults to 'forman'.

        Raises:
            ValueError: If wrong curvature type is given.py

        Returns:
            nx network: nx file of network that contains previous properties of cnx
        """
        import time

        tt = time.time()
        if self.weighted:
            print("ATTENTION! Weighted is True!", flush=True)
        # compute the Forman Ricci curvature of the given graph cnx
        if self.node_attr_exists(attr=c_type) and rc_curv is False:
            return self.cnx
        else:
            if c_type == "forman":
                print(
                    "\n===== Compute the Forman Ricci curvature of the given graph =====",
                    flush=True
                )
                rc = FormanRicci(self.cnx, verbose="INFO")
            elif c_type == "ollivier":
                print(
                    "\n===== Compute the Ollivier Ricci curvature of the given graph =====",
                    flush=True
                )
                rc = OllivierRicci(
                    self.cnx,
                    alpha=0.5,
                    verbose="TRACE",
                    proc=mpi.cpu_count(),  # Is as well default value in OllivierRicci mix
                    method="OTD",  # Sinkhorn does not work very well
                )
            else:
                raise ValueError(f"Curvature {c_type} does not exist!")

            rc.compute_ricci_curvature()
            # digraph = nx.DiGraph(rc.G)  # Set Ricci Curvature as DiGraph
            self.cnx = rc.G  # sets the node and edge attributes!
            print("time elapsed: %s" % (time.time() - tt), flush=True)

            return self.cnx

    # ################## Network attributes #####################

    def is_undirected(self):
        """
        Returns True if the the graph is undirected
        """
        return np.array_equal(self.adjacency, self.adjacency.T)

    def get_node_attributes(self):
        all_attrs = list(self.cnx.nodes[0].keys())
        return all_attrs

    def get_edge_attributes(self):
        all_attrs = list(list(self.cnx.edges(0, data=True))[0][2].keys())
        return all_attrs

    def get_node_attr(self, attr):
        return np.array(list(self.get_node_attr_dict(attr).values()))

    def get_single_node(self, node_number):
        return self.cnx.nodes[node_number]

    def get_node_attr_dict(self, attr):
        return nx.get_node_attributes(self.cnx, attr)

    def get_edge_attr(self, attr):
        return np.array(list(self.get_edge_attr_dict(attr).values()))

    def get_edge_attr_dict(self, attr):
        return nwf.get_edge_attr_dict(graph=self.cnx, attr=attr)

    def get_attr_array(self, attr):
        return np.array(nx.attr_matrix(self.cnx, edge_attr=attr))

    def create_ds(self):
        print("Create ds for all attributes present in node 0.", flush=True)
        all_attrs = self.get_node_attributes()
        all_attrs.remove("lat")
        all_attrs.remove("lon")

        da_list = []
        for key in all_attrs:
            node_attr = nx.get_node_attributes(self.cnx, key)
            # Set NaNs if node attribute is not set
            if len(node_attr.values()) != len(self.cnx.nodes):
                for ne in self.cnx.nodes:
                    if ne not in node_attr:
                        node_attr[ne] = np.NaN

            # Sort dicionary
            sort_dic = dict(sorted(node_attr.items()))

            da_list.append(self.ds.get_map(list(sort_dic.values()), name=key))

        self.ds_nx = xr.merge(da_list)
        return self.ds_nx

    def add_attr2ds(self, new_attr):
        """Add new attribute to self.ds_nx.

        Args:
            new_attr (str): Name of new attribute.

        Returns:
            ds_nx (xr.Dataset): Dataset of map with all attributes
        """
        if not self.ds_nx:
            self.create_ds()

        node_attr = nx.get_node_attributes(self.cnx, new_attr)
        # Set NaNs if node attribute is not set
        if len(node_attr.values()) != len(self.cnx.nodes):
            for ne in self.cnx.nodes:
                if ne not in node_attr:
                    node_attr[ne] = np.NaN

        # Sort dicionary
        sort_dic = dict(sorted(node_attr.items()))

        self.ds_nx[new_attr] = self.ds.get_map(
            list(sort_dic.values()), name=new_attr)

        return self.ds_nx

    def get_edgelist(self, weighted=False, sort=False):
        reload(nwf)
        edge_list = nwf.get_edgelist(
            net=self.cnx, weighted=weighted, sort=sort)

        return edge_list

    def get_edgelist_th(self, attr, th, upper=True, edge_list=None, sort=True):
        """Gets an edge list for edge values that are above (below) a certain
        threshold.

        Args:
            attr (str): Edge Attribute Name
            th (float): threshold value
            upper (bool, optional): Below (lower) than threhold. Defaults to True.
            edge_list (2d list, optional): list of edge within edges are selected. Defaults to None.

        Returns:
            list: list of source-target edges
        """
        print(attr, flush=True)
        el = []
        if upper:
            print(f"Higher than threshold = {th}")
            for ne in self.cnx.nodes:
                for u, v, e in self.cnx.edges(ne, data=True):
                    if e[attr] >= th:
                        el.append((u, v))
        else:
            print(f"Lower than threshold < {th}")
            for ne in self.cnx.nodes:
                for u, v, e in self.cnx.edges(ne, data=True):
                    if e[attr] <= th:
                        el.append((u, v))

        el = np.array(el)
        if edge_list is not None:
            el = nwf.get_intersect_2el(el1=el, el2=edge_list)
            # keep structure of input edge list
            sources = edge_list[:, 0]
            for i, e in enumerate(el):
                if e[0] not in sources:
                    el[i] = (e[1], e[0])

        el = nwf.remove_dublicates_el(np.array(list(el)))
        if sort:
            el, _ = nwf.sort_el_lon_lat(el=el, netx=self.cnx)

        return el

    def get_q_edge_list(self, attr, q=None, edge_list=None):
        print(f'Edge list for {attr}')

        if q is not None:
            edge_attr = self.get_edge_attr(attr=attr)
            q_val = np.quantile(edge_attr, q=q)
            el = []
            if q < 0.5:
                print(f"most negative {q} <= {q_val}")
                for ne in self.cnx.nodes:
                    for u, v, e in self.cnx.edges(ne, data=True):
                        if e[attr] <= q_val:
                            el.append((u, v))
            else:
                print(f"most positive {q} >= {q_val}")
                for ne in self.cnx.nodes:
                    for u, v, e in self.cnx.edges(ne, data=True):
                        if e[attr] >= q_val:
                            el.append((u, v))
        else:
            # el = self.get_edgelist()  # this only counts each link once!
            el = []
            for ne in self.cnx.nodes:
                for u, v, e in self.cnx.edges(ne, data=True):
                    el.append((u, v))
        el = np.array(el)
        if edge_list is not None:
            # el = el[(el[:, None] == edge_list).all(2).any(1)]  very slow
            el = nwf.get_intersect_2el(el1=el, el2=edge_list)

            # keep structure of input edge list
            sources = edge_list[:, 0]
            for i, e in enumerate(el):
                if e[0] not in sources:
                    el[i] = (e[1], e[0])

        return el

    def get_edges_between_nodes(self, ids1, ids2=None):

        if ids2 is None:
            ids2 = ids1

        el_1 = self.get_edges_node_ids(*ids1)
        el_2 = self.get_edges_node_ids(*ids2)

        el_12 = []
        for e in el_1:
            tid1 = e[1]
            if tid1 in ids2:
                el_12.append(e)

        el_12 = np.array(el_12)
        if len(el_12) < 1:
            raise ValueError(f'No links between nodes in ids1 and ids2!')

        return el_12

    # #################  Spatial attributes #######################

    def ll_one_row(
        self, row, i,
    ):
        idx_links = np.where(row == 1)[0]
        ll_i = None
        if len(idx_links) > 0:
            coord_i = self.get_single_node(i)

            lon_links = []
            lat_links = []
            for j in idx_links:
                coord_j = self.get_single_node(j)
                lon_links.append(coord_j["lon"])
                lat_links.append(coord_j["lat"])

            ll_i = grid.haversine(
                coord_i["lon"],
                coord_i["lat"],
                np.array(lon_links),
                np.array(lat_links),
                radius=grid.RADIUS_EARTH,
            )

        return ll_i, i, idx_links

    def compute_link_lengths_edges(
        self, backend="multiprocessing",
        num_cpus=1  # Check why parallel is not so fast anymore?
    ):
        """Attribute to each link its spatial length"""
        gut.myprint(
            "===== Compute spatial link lenghts of all edges in parallel ===========")
        undirected = self.is_undirected()
        if undirected:
            # Use only 1 half of the adjecency
            adj_ll = np.tril(self.adjacency)
            gut.myprint('WARNING! Only use half of adjacency!')
        else:
            adj_ll = self.adjacency

        link_length = Parallel(n_jobs=num_cpus, backend=backend)(
            delayed(self.ll_one_row)(row, i)
            for i, row in enumerate(tqdm(adj_ll))
        )

        for ll in link_length:
            # Some array entries are None, these need to be removed!
            if ll is not None:
                ll_i, i, j_indices = ll
                for idx, j in enumerate(j_indices):
                    self.cnx.edges[i, j]["length"] = float(ll_i[idx])
                    if undirected:
                        self.cnx.edges[j,
                                       i]["length"] = self.cnx.edges[i, j]["length"]

        return self.cnx

    def get_link_length_distribution(self,
                                     var=None,
                                     q=None,
                                     edge_list=None):
        """Gets link length distribution for a given variable.txt

        Args:
            var (str, optional): Variable on which to apply link length distribution.
                                 Defaults to None.
        """

        if "length" not in self.get_edge_attributes():
            self.compute_link_lengths_edges()

        ll = []
        if var is None:
            ll = self.get_edge_attr(attr='length')
        elif (
            var == "formanCurvature" or var == "betweenness" or var == "ricciCurvature"
        ):
            # q = None every link is counted twice as incoming and outgoing link
            el = self.get_q_edge_list(attr=var, q=q, edge_list=edge_list)
            # adj = self.net.get_adj_from_edge_list(edge_list=edge_list)
            # ll = self.net.get_link_length_distribution(adjacency=adj)
            for u, v in el:
                try:
                    ed = self.cnx[u][v]["length"]
                except:
                    ed = self.cnx[v][u]["length"]
                    gut.myprint(
                        f'WARNING: {u}, {v} only exists ll as {v}, {u}!')
                ll.append(ed)

        ll = np.array(ll)
        return ll

    def get_el_min_length(self, el, min_length=0):
        red_el = []
        min_length = 2000
        for e in el:
            i, j = e
            ll = self.cnx[i][j]['length']
            if ll >= min_length:
                red_el.append(e)
        return red_el

    def get_long_range_edges(self, min_length, attr=None, q=None):
        el = self.get_q_edge_list(attr=attr, q=q)
        edge_list = self.get_el_min_length(el=el, min_length=min_length)

        return edge_list

    def get_link_idx_lst(self, idx_lst):
        """Get list of links for a list of nodes.

        Parameters:
        ----------
        idx_lst: np.ndarray
            List of indices for the adjacency.

        Returns:
        --------
        List of link indices.
        """
        link_flat = np.zeros_like(self.adjacency[0, :], dtype=bool)
        for idx in idx_lst:
            link_flat = np.logical_or(
                link_flat, np.where(self.adjacency[idx, :] == 1, True, False)
            )
        return link_flat

    def get_links_for_coord(self, lon, lat):
        idx_coord = self.ds.get_index_for_coord(lon=lon, lat=lat)
        link_list = self.get_link_idx_lst(idx_lst=[idx_coord])

        return link_list

    def get_target_ids_for_coord(self, lon, lat):
        source_id = self.ds.get_index_for_coord(lon=lon, lat=lat)
        target_ids = self.get_edges_node_ids(source_id)[1, :]
        return target_ids

    def get_spec_loc(self, lon_range, lat_range, attr=None, th=None, dateline=False):
        """Get locations and values of an attribute in a given lon-lat range.

        Args:
            lon_range (list): Range of longitudes.
            lat_range (list): Range of latitudes
            attr (str): Attribute name stored in the networkx class.
            th (int, optional): Threshold on the attribute,
                i.e. only grid points with values larger than the threshold are returned.
                Defaults to 0.

        Raises:
            ValueError: [description]

        Returns:
            ids: [description]
            loc_map: [description]
        """
        if attr is not None:
            all_maps = self.ds_nx
            var_names = list(all_maps.keys())
            if attr not in var_names:
                raise ValueError(
                    f"Error! This attr {attr} does not exist in dataset!")
            def_map = all_maps[attr]
            if th is not None:
                def_map = xr.where(def_map > th, def_map, np.nan)
        else:
            def_map = self.ds.mask

        loc_dict = self.ds.get_locations_in_range(
            lon_range=lon_range, lat_range=lat_range, def_map=def_map,
            dateline=dateline,
        )
        ids = loc_dict['idx']
        mmap = loc_dict['mmap']
        return ids, mmap

    def get_edges_node_ids(self, *ids_lst):
        """Gets the outgoing edges to a list of node ids (i.e. the node id is the source).

        Raises:
            ValueError:
            ValueError:

        Returns:
            np.array: 2d numpy array of source-target ids.
        """
        if len(ids_lst) == 0:
            raise ValueError("Node id list ist empty")

        el = []
        for nid in ids_lst:
            if isinstance(nid, (int, np.int64)) is False:
                raise ValueError(f"Id {nid} is not int!")

            el_tmp = list(self.cnx.edges(nid))
            el.append(el_tmp)

        return np.concatenate(el, axis=0)

    def get_target_ids_for_node_ids(self, *ids_lst):
        if len(ids_lst) == 0:
            raise ValueError("Node id list ist empty")

        tids = []
        for nid in ids_lst:
            if isinstance(nid, (int, np.int64)) is False:
                raise ValueError(f"Id {nid} is not int!")

            el_tmp = np.array(list(self.cnx.edges(nid)))[:, 1]
            tids.append(el_tmp)
        tids, cnts = np.unique(np.concatenate(
            tids, axis=0), return_counts=True)
        return tids, cnts

    def get_edge_attr_for_edge_list(self, edge_list, attr):
        """Get attributes to specified edges.

        Args:
            edge_list ([type]): [description]
            attr ([type]): [description]

        Returns:
            el_attr (dict): Dictionary with edges as keys
                            and attributes as values
        """
        edge_attr = nx.get_edge_attributes(self.cnx, attr)
        el_attr = {}
        for e in edge_list:
            if tuple(e) in edge_attr:
                el_attr[tuple(e)] = edge_attr[tuple(e)]
            elif (e[1], e[0]) in edge_attr:
                el_attr[tuple(e)] = edge_attr[(e[1], e[0])]

        return el_attr

    def get_node_attr_for_edge_list(
        self, edge_list, attribute, q=None, attr_suffix="loc", normalize=True
    ):
        """Get node attribute to speciefied edges,
            i.e. sum of edge attributes at all nodes specified in edge list
            normalized by node degree

        Args:
            edge_list ([type]): [description]
            attr ([type]): [description]
            attr_suffix (str, optional): [description]. Defaults to "loc".

        Return:
            node_attr (dict): Dictionary of node attributes corresponding to edge list
        """
        edge_attr = nx.get_edge_attributes(self.cnx, attribute)

        # Initialize node attr
        node_attr = {}
        for ne in np.unique(edge_list):
            node_attr[ne] = 0

        # Sum edge attributes of node
        for e in edge_list:
            if tuple(e) in edge_attr:
                node_attr[e[1]] += edge_attr[tuple(e)]
            elif (e[1], e[0]) in edge_attr:
                node_attr[e[1]] += edge_attr[(e[1], e[0])]
            else:
                print(f"Edge {e} has no assigned attribute.", flush=True)

        # Add to networkx
        attr = attribute if q is None else f"{attribute}_q{q}"
        for ne, value in node_attr.items():
            if normalize:
                value = value / self.cnx.degree(ne)
            self.cnx.nodes[ne][f"{attr}_{attr_suffix}"] = value

        self.add_attr2ds(new_attr=f"{attr}_{attr_suffix}")

        return node_attr

    def get_edges_nodes_for_region(
        self, lon_range, lat_range, attribute=None, binary=False, q=None,
        dateline=False,
    ):
        """Get edges for a lon-lat range

        Args:
            lon_range (list): min lon, max lon_range
            lat_range (list): min lat, max lat_range
            attribute (str, optional): specific attribute to filter. Defaults to None.
            binary (bool, optional): if counts of edges are considered as well. Defaults to False.
            q (float, optional): in which quantile of attribute. Defaults to None.
            dateline (bool, optional): If region is over the dateline. Defaults to False.

        Returns:
            Returns:
            el_loc (np.ndarray): Edge list of links from the region
                                which have the specified attribute
            source_map (xr.dataarray): Map containing the attribute in the region
            target_link_map (xr.dataarray): Map containing the target links as counts
                                            if binary=False in the region
        """
        if attribute is not None:
            attr = attribute if q is None else f"{attribute}_q{q}"
        else:
            attr = None
            q = None
        # select region
        ids, source_map = self.get_spec_loc(
            lon_range=lon_range, lat_range=lat_range, attr=attr, th=None,
            dateline=dateline
        )

        # Get all edges and nodes connected to the selected region
        el_loc = self.get_edges_node_ids(*ids)
        if q is not None:
            el_loc = self.get_q_edge_list(
                attr=attribute, q=q, edge_list=el_loc)

        num_link_arr = self.ds.count_indices_to_array(
            np.array(el_loc).flatten())  # All edges are counted, source-target does not matter
        if binary:
            num_link_arr = np.where(num_link_arr == 0, np.NaN, 1)
        else:
            num_link_arr[np.where(num_link_arr == 0)[0]] = np.NaN
        tids = np.sort(np.unique(el_loc[:, 1]))
        target_link_map = self.ds.get_map(num_link_arr)

        return dict(el=el_loc,
                    sids=ids,
                    tids=tids,
                    lon_range=lon_range,
                    lat_range=lat_range,
                    source_map=source_map,
                    target_map=target_link_map)

    def get_edges_between_regions(
        self, lon_range_s, lat_range_s, lon_range_t, lat_range_t, attribute=None, binary=False, q=None
    ):
        """Get edge and node properties for a specified region.

        Args:
            lon_range ([type]): [description]
            lat_range ([type]): [description]
            attribute ([type]): [description]
            q ([type], optional): [description]. Defaults to None.

        Returns:
            el_loc (np.ndarray): Edge list of links from the region
                                which have the specified attribute
            source_map (xr.dataarray): Map containing the attribute in the region
            target_link_map (xr.dataarray): Map containing the target links as counts
                                            if binary=False in the region
        """

        link_dict_s = self.get_edges_nodes_for_region(
            lon_range=lon_range_s,
            lat_range=lat_range_s, binary=binary,
            attribute=attribute, q=q)

        link_dict_t = self.get_edges_nodes_for_region(
            lon_range=lon_range_t,
            lat_range=lat_range_t, binary=binary,
            attribute=attribute, q=q)

        el_s = link_dict_s['el']
        source_map = link_dict_s['source_map']

        el_t = link_dict_t['el']

        el_st = nwf.get_intersect_2el(el1=el_s, el2=el_t)

        num_link_arr = self.ds.count_indices_to_array(
            np.array(el_st).flatten())  # All edges are counted, source-target does not matter
        if binary:
            num_link_arr = np.where(num_link_arr == 0, np.NaN, 1)
        else:
            num_link_arr[np.where(num_link_arr == 0)[0]] = np.NaN
        target_map = self.ds.get_map(num_link_arr)

        return dict(el=el_st,
                    source_map=source_map,
                    target_map=target_map)

    def get_edges_for_loc(
        self, lon, lat, binary=False,
    ):
        """Get edge map and list for a specified location.

        Args:
            lon (float): longitude
            lat_range (float): latitude
        Returns:
            el_loc (np.ndarray): Edge list of links from the region
                                which have the specified attribute
            target_link_map (xr.dataarray): Map containing the target links as counts
                                            if binary=False in the region
        """

        # Get all edges and nodes connected to the selected region

        idx_coord = self.ds.get_index_for_coord(lon=lon, lat=lat)
        el_loc = self.get_edges_node_ids(idx_coord)
        target_link_map = self.get_el_map(el=el_loc, binary=binary)

        return el_loc, target_link_map

    def get_el_map(self, el, binary=False):
        num_link_arr = self.ds.count_indices_to_array(np.array(el).flatten())
        if binary:  # Only if a link exists is important not the number of incoming links
            num_link_arr = np.where(num_link_arr == 0, np.NaN, 1)
        else:  # Count all incoming links
            num_link_arr[np.where(num_link_arr == 0)[0]] = np.NaN
        target_link_map = self.ds.get_map(num_link_arr)
        return target_link_map

    def normalize_edge_attr(self, attributes=["formanCurvature"]):
        """Normalize edge attribute between -1 and 1 to keep the notion of positive and
        negative curvatures.
        See https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
        Args:
            attributes (list): List of edge attribute to normalize, e.g. ['formanCurvature']
        """
        for attr in attributes:
            edge_attr = self.get_edge_attr_dict(attr=attr)
            edge_attr_val = self.get_edge_attr(attr=attr)
            edge_attr_norm = (
                2 * (edge_attr_val - np.min(edge_attr_val))
                / (np.max(edge_attr_val) - np.min(edge_attr_val))
                - 1
            )
            # Add to networkx
            print(f"Store normalized {attr} in network.", flush=True)
            for n, (i, j) in enumerate(edge_attr):
                self.cnx.edges[i, j][f"{attr}_norm"] = edge_attr_norm[n]

        self.set_node_attr(edge_attrs=attributes, attr_name="norm")

        return

    def rank_edge_attr(self, attributes=["formanCurvature"]):
        """Rank edge attribute between 1 and # edges to k
        Args:
            attributes (list): List of edge attribute to rank, e.g. ['formanCurvature']
        """
        for attr in attributes:
            edge_attr = self.get_edge_attr_dict(attr=attr)
            edge_attr_val = self.get_edge_attr(attr=attr)
            edge_attr_rk = st.rankdata(
                edge_attr_val, axis=0) / len(edge_attr_val)
            # Add to networkx
            print(f"Store rank {attr} in network.", flush=True)
            for n, (i, j) in enumerate(edge_attr):
                self.cnx.edges[i, j][f"{attr}_rank"] = edge_attr_rk[n]

        self.set_node_attr(edge_attrs=attributes, attr_name="rank")

        return

    def set_node_attr(
        self,
        edge_attrs=["formanCurvature", "ricciCurvature"],
        norm=True,
        attr_name="norm",
    ):
        all_attrs = self.get_edge_attributes()

        for attr in edge_attrs:
            if attr not in all_attrs:
                raise ValueError(f"Edge attribute {attr} does not exist!")
            print(attr, flush=True)
            new_node_attr = f"{attr}_{attr_name}"
            print(f"Get node attr {new_node_attr}...", flush=True)
            self.cnx = nwf.set_node_attr(
                G=self.cnx, attr=new_node_attr, norm=norm)

            self.correct_node_attr(attr=new_node_attr, q=0.5)

        self.ds_nx = self.create_ds()
        return

    def get_node_attr_q(
        self,
        edge_attrs=["formanCurvature", "ricciCurvature"],
        norm=True,
        q_values=[0.05, 0.1, 0.9, 0.95],
    ):
        """Compute quantiles of edge attributes and store normalized sum of quantiles
            at each node.

        Args:
            all_attrs (list, optional): List of attributes to obtain the quantiles for.
                If None quantiles are obtained for all attributes of the network.
                Default None.
            q_values (list, optional): List of quantiles. Defaults to [0.1, 0.2, 0.9, 0.8].
        """
        all_attrs = self.get_edge_attributes()

        for attr in edge_attrs:
            if attr not in all_attrs:
                raise ValueError(f"Edge attribute {attr} does not exist!")
            gut.myprint(attr)
            edge_attr = self.get_edge_attr(attr)
            for q in q_values:
                gut.myprint(q)
                q_val = np.quantile(edge_attr, q=q)
                gut.myprint(f"Get values {q} <= {q_val}")
                for ne in self.cnx.nodes:
                    node_sum = 0.0
                    node_cnt = 0
                    for u, v, e in self.cnx.edges(ne, data=True):
                        if (q < 0.5 and e[attr] <= q_val) | (
                            q >= 0.5 and e[attr] >= q_val
                        ):
                            node_sum += e[attr]
                            node_cnt += 1
                    # Normalized attribute by degree
                    new_attr = f"{attr}_q{q}"
                    if node_cnt > 0:
                        if norm is True:
                            # Devide by local node degree, alternative:/self.cnx.degree(ne)
                            self.cnx.nodes[ne][new_attr] = node_sum / node_cnt
                        else:
                            # Only sum up
                            self.cnx.nodes[ne][new_attr] = node_sum
                    else:
                        # This needs to be later adjusted
                        self.cnx.nodes[ne][new_attr] = np.nan

                # q_corr = 1 if q < 0.5 else 0
                # self.correct_node_attr(attr=new_attr, q=q_corr)

        ds_nx = self.create_ds()

        return ds_nx

    def correct_node_attr(self, attr, q=0.5):
        node_vals = self.get_node_attr(attr=attr)
        num_nans = np.count_nonzero(np.isnan(node_vals))
        if num_nans > 0:
            q_med = np.nanquantile(node_vals, q=q)
            for ne in self.cnx.nodes:
                if np.isnan(self.cnx.nodes[ne][attr]):
                    # print(self.cnx.nodes[ne][attr])
                    self.cnx.nodes[ne][attr] = q_med
            node_vals = self.get_node_attr(attr=attr)
            num_nansnow = np.count_nonzero(np.isnan(node_vals))
            if num_nansnow > 0:
                raise ValueError(
                    f"{attr}: Still {num_nans} nans remained in nodes!", flush=True)
        else:
            print(f"{attr}: No nans in nodes, nothing to correct!", flush=True)

    # ##################### NetworkIt #######################
    def get_nk_graph(self):
        self.cnk = nk.nxadapter.nx2nk(self.cnx.to_undirected())
        return self.cnk
