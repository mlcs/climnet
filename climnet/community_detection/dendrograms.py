#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 20:26:53 2021

@author: Felix Strnad
"""
from climnet.community_detection.graph_tool.es_graph_tool import ES_Graph_tool

from scipy.cluster.hierarchy import dendrogram, to_tree
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH + "/../")  # Adds higher directory


# %%

class Dendro_node():
    def __init__(self, id=0, left=None, right=None, parent=None, is_leaf=True, level=0, num_groups=0):
        self.id = id
        self.parent = parent
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.level = level
        self.num_groups = num_groups


class Dendro_tree():
    def __init__(self, Z, node_levels, icr=False):
        self.node_dict, self.node_list = self.get_node_dict(Z)
        self.dict_node_Z = self.get_node_level_Z_dict(node_levels)

        if icr is True:
            sys.setrecursionlimit(10000)

    def get_node_dict(self, Z):
        parent_dict, node_list = self.get_dict_parent_nodes(Z)
        node_dict = dict()
        for node in node_list:
            this_id = node.id
            if node.is_leaf():
                this_node = Dendro_node(id=this_id,
                                        parent=parent_dict[this_id],
                                        is_leaf=node.is_leaf(),
                                        level=int(node.dist),
                                        num_groups=node.count
                                        )
            else:
                this_node = Dendro_node(id=this_id,
                                        parent=parent_dict[this_id],
                                        left=node.left.id,
                                        right=node.right.id,
                                        is_leaf=node.is_leaf(),
                                        level=int(node.dist),
                                        num_groups=node.count
                                        )
            node_dict[this_id] = this_node
        return node_dict, node_list

    def get_dict_parent_nodes(self, Z):

        root_node, node_list = to_tree(Z, rd=True)

        dict_parent_nodes = dict()

        for node in node_list:
            parent = node.id
            if node.is_leaf():
                continue
            else:
                child_left = node.left.id
                child_right = node.right.id
                dict_parent_nodes[child_left] = parent
                dict_parent_nodes[child_right] = parent
        dict_parent_nodes[root_node.id] = None
        return dict_parent_nodes, node_list

    def foreward_path(self, node_id):
        single_node_path = []

        old_level = -1
        while node_id is not None:
            node = self.node_dict[node_id]
            parent = node.parent
            level = node.level
            if old_level == level:
                single_node_path[-1] = node
            else:
                single_node_path.append(node)
            old_level = level
            node_id = parent

        return single_node_path

    def foreward_path_full(self, node_id):
        """
        This function captures all nodes on the way, even if they are at the same level!
        Needed for group_id dictionary.
        """
        single_node_path = []

        while node_id is not None:
            node = self.node_dict[node_id]
            parent = node.parent
            single_node_path.append(node)
            node_id = parent

        return single_node_path

    def tree_traversal(self, root):
        """"
        Using preorder tree traversal.
        Find here: https://www.tutorialspoint.com/python_data_structure/python_tree_traversal_algorithms.htm
        """
        res = []
        res.append(root.id)

        if root.is_leaf is False:
            # print(root.id, root.left, root.right)
            left_node = self.node_dict[root.left]
            right_node = self.node_dict[root.right]
            res = res + self.tree_traversal(left_node)
            res = res + self.tree_traversal(right_node)
        return res

    def backward_path(self, node_id):

        start_node = self.node_dict[node_id]
        tree_ids = self.tree_traversal(start_node)

        leaf_nodes = []
        for tree_id in tree_ids:
            node = self.node_dict[tree_id]
            if node.is_leaf:
                leaf_nodes.append(tree_id)

        return tree_ids, leaf_nodes

    def find_common_root(self, node_id1, node_id2):

        node1 = self.node_dict[node_id1]
        li = 0
        back_path, leaf_nodes = self.backward_path(node1.id)
        while node_id2 not in back_path:
            li += 1
            node1 = self.node_dict[node1.parent]
            back_path, leaf_nodes = self.backward_path(node1.id)

        intersect_level = node1.level
        intersect_node_id = node1.id

        return intersect_level, intersect_node_id

    def get_node_level_Z_dict(self, node_levels):
        """
        Creates Dictionary that returns the group number and the level where two groups merge.
        """

        ground_level = np.arange(len(node_levels[0]), )
        full_node_levels = np.concatenate(
            ([ground_level], node_levels), axis=0)
        top_node = max(self.node_dict)
        back_path, leaf_nodes = self.backward_path(node_id=top_node)
        dict_node_Z = dict()

        for leaf_id in leaf_nodes:
            foreward_path = self.foreward_path_full(leaf_id)
            for node in foreward_path[:]:
                Z_id = node.id
                nl = node.level
                g_id_level = int(full_node_levels[nl][leaf_id])
                if Z_id not in dict_node_Z:
                    dict_node_Z[Z_id] = {'level': nl,
                                         'group_id': g_id_level}

        return dict_node_Z

    def get_intersect_node_ids(self, sel_node_ids):
        from itertools import combinations
        u_node_ids = np.unique(sel_node_ids)
        all_combintions = list(combinations(u_node_ids, 2))
        level_arr = []
        node_ids = []
        is_node_ids = []

        for node1, node2 in all_combintions:

            level, intersect_node = self.find_common_root(node1, node2)
            level_arr.append(level)
            is_node_ids.append(intersect_node)
            node_ids.append((node1, node2))

        result_dict = {'level': level_arr,
                       'is_node': is_node_ids,
                       'loc_id': node_ids}
        return result_dict

    def get_split_groups(self, node_ids_arr, abs_th=2, rel_th=0.1):

        all_node_ids = np.concatenate(np.array(node_ids_arr)).ravel()

        Z_ids = self.get_all_Z_ids_leaf_ids(all_node_ids)
        u_Z_ids = np.unique(Z_ids)
        num = len(all_node_ids)
        for node_ids in node_ids_arr:
            Z_ids = self.get_all_Z_ids_leaf_ids(node_ids)
            # sorted unique array of Z_ids
            u_Z_ids = np.intersect1d(Z_ids, u_Z_ids)

        split_Z_id = max(u_Z_ids)
        level = int(self.node_dict[split_Z_id].level)
        for idx, Z_id in enumerate(u_Z_ids):
            leaf_nodes = self.get_leaf_nodes_of_is_node(Z_id)
            c = sum(ln in leaf_nodes for ln in all_node_ids)
            # Get cluster in which are still enough node ids
            if ((num-c) <= abs_th) and (c/num) >= rel_th:
                new_level = self.node_dict[Z_id].level
                if new_level >= level:
                    break
                else:
                    split_Z_id = self.node_dict[Z_id].id
                    level = new_level

        return split_Z_id

    def get_split_groups_bak(self, node_ids_arr, abs_th=10, rel_th=1):

        all_node_ids = np.concatenate(np.array(node_ids_arr)).ravel()

        Z_ids = self.get_all_Z_ids_leaf_ids(all_node_ids)
        u_Z_ids = np.unique(Z_ids)
        num = len(all_node_ids)
        for node_ids in node_ids_arr:
            Z_ids_1 = self.get_all_Z_ids_leaf_ids(node_ids)
            # sorted unique array of Z_ids
            u_Z_ids = np.intersect1d(Z_ids_1, u_Z_ids)

        split_Z_id = max(u_Z_ids)
        level = int(self.node_dict[split_Z_id].level)
        for idx, Z_id in enumerate(u_Z_ids):
            leaf_nodes = self.get_leaf_nodes_of_is_node(Z_id)
            c = sum(ln in leaf_nodes for ln in all_node_ids)
            # Get cluster in which are still enough node ids
            if ((num-c) <= abs_th) and (c/num) >= rel_th:
                csucess = 0
                for node_ids in node_ids_arr:
                    num_sarr = len(node_ids)
                    c_sarr = sum(ln in leaf_nodes for ln in node_ids)
                    if ((num_sarr-c_sarr) <= abs_th) and (c_sarr/num_sarr) >= rel_th:
                        csucess += 1
                if csucess == len(node_ids_arr):
                    # Get the one cluster above
                    new_level = self.node_dict[Z_id].level
                    if new_level > level:
                        break
                    else:
                        split_Z_id = self.node_dict[Z_id].id
                        level = new_level

        return split_Z_id

    def get_leaf_nodes_of_is_node(self, Z_is):
        _, leaf_nodes = self.backward_path(Z_is)

        return leaf_nodes

    def get_hist_sel_nodes(self, sel_node_ids, th=2):
        """
        Gets distribution where most ids of sel_node_ids merge.
        """
        from collections import Counter

        i_node_dict = self.get_intersect_node_ids(sel_node_ids)

        is_nodes = i_node_dict['is_node']
        counter = np.array(Counter(is_nodes).most_common())

        frequencies = np.array(counter[:, 1]/np.sum(counter[:, 1]))
        frequencies = np.array(counter[:, 1])

        Z_ids = np.array(counter[:, 0])

        # neglect values smaller th

        use_index = [frequencies > th]

        frequencies = frequencies[use_index]
        Z_ids = Z_ids[use_index]
        Z_most_occ = Z_ids[np.argmax(frequencies)]

        return Z_ids, frequencies,  Z_most_occ

    def get_Z_ids_of_level(self, level):
        dict_node_Z = self.dict_node_Z
        Z_ids = []
        for (Z_id, val) in dict_node_Z.items():
            if val['level'] == level:
                Z_ids.append(Z_id)
        return np.array(Z_ids)

    def get_all_Z_ids_leaf_ids(self, leaf_ids, min_level=4):
        Z_ids = []
        for leaf_id in leaf_ids:
            foreward_path = self.foreward_path_full(leaf_id)
            for path_id in foreward_path:
                Z_id = path_id.id
                level = self.dict_node_Z[Z_id]['level']
                if level >= min_level and Z_id not in Z_ids:
                    Z_ids.append(Z_id)

        return Z_ids

    """################ Plotting of tree parts and histograms ##################"""

    def plot_hist_is_nodes(self, sel_node_ids, th=2, ax=None, title='', savepath=None):
        """"
        This function creates a bar plot from a counter.

        :param counter: This is a counter object, a dictionary with the item as the key
        and the frequency as the value
        :param ax: an axis of matplotlib
        :return: the axis wit the object in it
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

        Z_ids, frequencies, Z_most_occ = self.get_hist_sel_nodes(
            sel_node_ids, th=th)
        names = []
        for Z_id in Z_ids:
            level = self.dict_node_Z[Z_id]['level']
            group_id = self.dict_node_Z[Z_id]['group_id']
            names.append(f"L{level}\n#{group_id}\nZ{Z_id}")

        x_coordinates = np.arange(len(frequencies))
        ax.bar(x_coordinates, frequencies, align='center')
        ax.set_ylabel('# Occurences')
        ax.set_title(title)

        ax.xaxis.set_major_locator(plt.FixedLocator(x_coordinates))
        ax.xaxis.set_major_formatter(plt.FixedFormatter(names))
        if savepath is not None and fig is not None:
            fig.savefig(savepath)

        return ax


class Dendrogram_ES (ES_Graph_tool):
    def __init__(self, group_levels, ):

        self.dflt_col = "#D3D3D3"   # Unclustered gray
        self.group_levels = group_levels
        self.node_levels, self.node_counts = self.node_level_arr(group_levels)
        # print("Create Linkage Matrix Z...")
        self.Z = self.create_Z_linkage(group_levels)

        """Tree objects """
        # print("Create Dendrogram Tree Object for Matrix Z...")
        self.d_tree = Dendro_tree(Z=self.Z, node_levels=self.node_levels)
        # print("Finished preparing Dendrogram objects!")

    def create_Z_linkage(self, group_levels):
        """
        This function creates a linkage array.
        Each leaf node and each linkage (!) btw. nodes gets a new index.
        """
        num_nodes = len(group_levels[0])

        z_index = num_nodes-1  # 0 is counted as number
        group_idx_Z = dict()
        n_nodes_z = dict()
        Z = []
        for li, level in enumerate(group_levels):
            # print(f"Level {li}...")
            group_idx = li+1
            if li == 0:
                Z, group_idx_Z, n_nodes_z, z_index = self.linkage_first_level(
                    level, group_idx)
            else:
                this_l_Z, group_idx_Z, n_nodes_z, z_index = self.Z_on_upper_level(level,
                                                                                  group_idx_Z,
                                                                                  n_nodes_z,
                                                                                  group_idx,
                                                                                  z_index)

                if len(this_l_Z) > 0:
                    if len(Z) < 1:
                        Z = this_l_Z
                    else:
                        Z = np.concatenate((Z, this_l_Z), axis=0)

        return Z.astype(float)

    def linkage_first_level(self, group_levels, group_idx):
        Z = []
        z_index = len(group_levels)-1  # 0 is counted as number
        clustered_nodes = self.get_upper_level_group_number(group_levels)
        group_idx_Z = dict()
        num_nodes_cluster_z = dict()
        cg = 0
        for group_number, node_ids in clustered_nodes.items():
            if len(node_ids) < 2:
                # print(f"Single node group: {node_ids}!")
                id_1 = node_ids[0]
                group_idx_Z[cg] = id_1
                num_nodes_cluster_z[cg] = len(node_ids)   # len(node_ids) ==1
            else:
                group_count = 2
                this_link = [node_ids[0], node_ids[1], group_idx, group_count]
                Z.append(this_link)
                z_index += 1

                for ci, node_id in enumerate(node_ids[2:]):
                    group_count += 1
                    this_link = [node_id, z_index, group_idx, group_count]
                    Z.append(this_link)
                    z_index += 1
                group_idx_Z[cg] = z_index
                num_nodes_cluster_z[cg] = len(node_ids)
            cg += 1
        if len(Z) > 1:
            Z = np.vstack(Z)
        return Z, group_idx_Z, num_nodes_cluster_z, z_index

    def Z_on_upper_level(self, group_levels, ind_translate, num_nodes, group_idx, z_index):
        Z = []
        clustered_nodes = self.get_upper_level_group_number(group_levels)

        group_idx_Z = dict()
        num_nodes_cluster_z = dict()
        cg = 0

        for group_number, node_ids in clustered_nodes.items():
            if len(node_ids) < 2:
                # print(f"Single node group: {node_ids}!")
                id_1 = ind_translate[node_ids[0]]
                group_idx_Z[cg] = id_1
                # len(node_ids)
                num_nodes_cluster_z[cg] = num_nodes[node_ids[0]]
            else:
                id_1 = ind_translate[node_ids[0]]
                id_2 = ind_translate[node_ids[1]]
                ng_1 = num_nodes[node_ids[0]]
                ng_2 = num_nodes[node_ids[1]]
                group_count = ng_1 + ng_2
                this_link = [id_1, id_2, group_idx, group_count]
                Z.append(this_link)
                z_index += 1

                for ci, node_id in enumerate(node_ids[2:]):
                    this_id = ind_translate[node_id]
                    group_count += num_nodes[node_id]

                    this_link = [this_id, z_index, group_idx, group_count]
                    Z.append(this_link)
                    z_index += 1
                group_idx_Z[cg] = z_index
                num_nodes_cluster_z[cg] = group_count
            cg += 1
        if len(Z) > 1:
            Z = np.vstack(Z)

        return Z, group_idx_Z, num_nodes_cluster_z, z_index

    """
    ################# Reduced Dendrograms to relevant paths #######################
    """

    def sel_node_levels(self, sel_node_ids):
        foreward_paths = []
        leaf_node_dict = dict()
        for i, leaf_id in enumerate(sel_node_ids):
            foreward_path = self.foreward_node_levels(
                self.node_levels, leaf_id)
            foreward_paths.append(foreward_path)
            leaf_node_dict[leaf_id] = i
        sel_node_levels = np.array(foreward_paths).T

        return sel_node_levels, leaf_node_dict

    def sel_group_levels(self, sel_node_ids):
        node_levels, leaf_node_dict = self.sel_node_levels(sel_node_ids)
        red_node_levels = self.reduce_node_levels(node_levels)
        # print(red_node_levels)
        sel_group_levels = self.get_hierarchical_data_from_nodes(
            red_node_levels)

        return sel_group_levels

    """
    ################# Plotting ###########################
    """

    def get_link_colors(self, Z, group_levels, cut_level, color_branch=True):
        """
        Logic:
        * rows in Z correspond to "inverted U" links that connect clusters
        * rows are ordered by increasing distance
        * if the colors of the connected clusters match, use that color for link

        """
        from matplotlib.colors import rgb2hex
        if cut_level < 1:
            print(f"ERROR! Cut Level (={cut_level}) has to be >=1!")
            sys.exit(1)
        level = cut_level-1
        num_clusters = len(self.count_elements(group_levels[level]))
        # Attention check again if this is true!
        ground_level_ids = group_levels[0]

        cmap, norm = self.discrete_cmap(0, num_clusters)
        colors = np.array([rgb2hex(cmap(norm(rgb)))
                           for rgb in range(num_clusters)])

        color_ground_cluster = colors[ground_level_ids]
        # Color mapping

        link_cols = {}
        for i, link in enumerate(Z[:].astype(int)):
            if not color_branch:
                link_cols[i+1+len(Z)] = self.dflt_col
            else:
                idx_0, idx_1, cluster_level, num_groups = link
                c1, c2 = (link_cols[x] if x > len(Z) else color_ground_cluster[x]
                          for x in (idx_0, idx_1))
                if cluster_level > cut_level:
                    link_cols[i+1+len(Z)] = self.dflt_col
                else:
                    if c1 == c2:
                        link_cols[i+1+len(Z)] = c1

        return link_cols, colors, num_clusters

    def get_id_to_coord(self, Z, ddata, ax=None):
        def flatten(lt):
            return [item for sublist in lt for item in sublist]
        X = flatten(ddata['icoord'])
        Y = flatten(ddata['dcoord'])
        # get leave coordinates, which are at y == 0
        leave_coords = [(x, y) for x, y in zip(X, Y) if y == 0]

        # in the dendogram data structure,
        # leave ids are listed in ascending order according to their x-coordinate
        order = np.argsort([x for x, y in leave_coords])
        # <- main data structure
        id_to_coord = dict(
            zip(ddata['leaves'], [leave_coords[idx] for idx in order]))

        # ----------------------------------------
        # get coordinates of other nodes

        # map endpoint of each link to coordinates of parent node
        children_to_parent_coords = dict()
        for i, d in zip(ddata['icoord'], ddata['dcoord']):
            x = (i[1] + i[2]) / 2
            y = d[1]  # or d[2]
            parent_coord = (x, y)
            left_coord = (i[0], d[0])
            right_coord = (i[-1], d[-1])
            children_to_parent_coords[(left_coord, right_coord)] = parent_coord
        # traverse tree from leaves upwards and populate mapping ID -> (x,y)
        root_node, node_list = to_tree(Z, rd=True)
        ids_left = range(len(ddata['leaves']), len(node_list))

        while len(ids_left) > 0:

            for ii, node_id in enumerate(ids_left):
                node = node_list[node_id]
                if not node.is_leaf():
                    if (node.left.id in id_to_coord) and (node.right.id in id_to_coord):
                        left_coord = id_to_coord[node.left.id]
                        right_coord = id_to_coord[node.right.id]
                        id_to_coord[node_id] = children_to_parent_coords[(
                            left_coord, right_coord)]

            ids_left = [node_id for node_id in range(
                len(node_list)) if node_id not in id_to_coord]

        # plot result on top of dendrogram
        if ax is not None:
            for node_id, (x, y) in id_to_coord.items():
                if not node_list[node_id].is_leaf():
                    if node_id > 16700:
                        ax.plot(x, y, 'ro')
                        ax.annotate(str(node_id), (x, y), xytext=(0, -8),
                                    textcoords='offset points',
                                    va='top', ha='center')
        return id_to_coord

    def fancy_dendrogram(*args, **kwargs):
        """
        Inspired from
        https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

        Returns
        -------
        Dendogramm data as scipy dataset.

        """
        from matplotlib import ticker

        ax = kwargs['ax']

        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)
        lw = kwargs.pop('lw', None)
        if lw is None:
            lw = 1
        with plt.rc_context({'lines.linewidth': lw}):
            ddata = dendrogram(*args, **kwargs)
        if not kwargs.get('no_plot', False):

            ax.set_xlabel('sample index (cluster size) ')
            ax.set_ylabel('Cluster Level')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    ax.plot(x, y, 'o', c=c)
                    ax.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                textcoords='offset points',
                                va='top', ha='center')
            if max_d:
                ax.axhline(y=max_d, c='k', lw=4, ls='--')
        for axis in [ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))

        return ddata

    def plot_path(self, ddata, node_path, ax, color, label=None):
        lw = 2
        node_id_to_coord = ddata['node_id_to_coord']
        xcoord = None
        ycoord = None
        for i, node_id in enumerate(node_path):
            this_node_coord = node_id_to_coord[node_id]
            if i < len(node_path)-1:
                next_id = node_path[i+1]
                next_node_coord = node_id_to_coord[next_id]
                xcoord = [this_node_coord[0],
                          this_node_coord[0], next_node_coord[0]]
                ycoord = [this_node_coord[1],
                          next_node_coord[1], next_node_coord[1]]
                ax.plot(xcoord, ycoord, lw=lw, ls='dashed',
                        color=color, label=None)
        if xcoord is not None:
            ax.plot(xcoord, ycoord, lw=lw, ls='dashed',
                    color=color, label=label)

    def plot_dendrogram(self, Z=None, group_levels=None, cut_level=None, title=None,
                        node_ids=None, fig=None, ax=None, colors=None, labels=None,
                        color_branch=True, plot_Z=True, savepath=None,):
        SMALL_SIZE = 16
        MEDIUM_SIZE = 18
        BIGGER_SIZE = 20
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
        # fontsize of the x and y labels
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        if Z is None:
            Z = self.Z

        if group_levels is None:
            group_levels = self.group_levels
        node_levels, _ = self.node_level_arr(group_levels)
        this_d_tree = Dendro_tree(Z, node_levels)

        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 8))
        else:
            savepath = None

        if cut_level is not None and plot_Z:
            link_colors, cluster_colors, num_clusters = self.get_link_colors(Z,
                                                                             group_levels,
                                                                             cut_level,
                                                                             color_branch=color_branch
                                                                             )

            ddata = self.fancy_dendrogram(Z,
                                          ax=ax,
                                          leaf_rotation=90.,  # rotates the x axis labels
                                          annotate_above=10,
                                          max_d=cut_level+0.5,
                                          link_color_func=lambda x: link_colors[x],
                                          lw=1,
                                          no_plot=False,
                                          )
        else:
            num_clusters = len(self.count_elements(group_levels[0]))
            ddata = dendrogram(Z)

            if not plot_Z:
                # use [:] to get a copy, since we're adding to the same list
                for c in ax.collections[:]:
                    # Remove the original LineCollection
                    ax.collections.remove(c)

        if node_ids is not None:

            # if ax is not None, id numbers are plotted
            id_to_coord = self.get_id_to_coord(Z, ddata, ax=None)

            ddata['node_id_to_coord'] = id_to_coord
            for i, node_id in enumerate(node_ids):
                if colors is not None:
                    color = colors[i]
                else:
                    color = 'red'
                if labels is not None:
                    label = labels[i]
                else:
                    label = None
                node_path = this_d_tree.foreward_path(node_id)
                node_id_path = [this_node.id for this_node in node_path]

                self.plot_path(ddata, node_id_path, ax, color, label)

            if labels is not None:
                ax.legend(bbox_to_anchor=(1, 1), loc='upper left',
                          fancybox=True, shadow=True, ncol=1)

        y_title = 1.05
        if title is None:
            ax.set_title(
                f'Hierarchical Clustering Dendrogram with {num_clusters} groups', y=y_title)
        else:
            ax.set_title(title, y=y_title)

        if savepath is not None:
            fig.tight_layout()

            plt.savefig(savepath)

        return ddata
