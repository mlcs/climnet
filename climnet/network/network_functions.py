from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import numpy as np
import xarray as xr
import networkx as nx


def get_sparsity(M=None):
    """Obtain sparsity of adjacency matrix."""
    sparsity = (
        np.count_nonzero(M.flatten())
        / M.shape[0]**2
    )
    print("Sparsity of adjacency: ", sparsity)
    return sparsity


def get_threshold(adjacency, corr):
    if corr is not None:
        min_val = np.nanmin(
            np.where(adjacency == 1, np.abs(corr), np.nan))
    else:
        min_val = 0
        print('WARNING: No treshold defined. Set to default = 0.')
    return min_val


def get_adj_from_edge_list(self, edge_list, len_adj=None):
    """Gets the adjacency for a given edge list of size len_adj

    Args:
        edge_list (list): list of tuples u, values
        len_adj (int, optional): length of adjacency. Defaults to None.

    Returns:
        np.ndarray: 2d array of adjacency
    """
    if len_adj is None:
        len_adj = max(edge_list)
    adj = np.zeros((len_adj, len_adj))

    for u, v in edge_list:
        adj[u, v] = 1

    get_sparsity(M=adj)

    return adj


def get_intersect_2el(el1, el2):
    """Gets the intersection of two edge lists. Regardless if the edge list is provided
    as (i,j) or (j,i)

    Args:
        el1 (np.ndarray): 2d array of (u,v) entries
        el2 (np.ndarray): 2d array of (u,v) entries

    Returns:
        el: 2d array of elements in both edge lists.
    """
    # To account that links can be stored as i,j or j,i
    sorted_1 = map(sorted, el1)
    sorted_2 = map(sorted, el2)
    tuple_1 = map(tuple, sorted_1)
    tuple_2 = map(tuple, sorted_2)
    el = np.array(list(map(list, set(tuple_1).intersection(tuple_2))))

    return el


def remove_dublicates_el(el):
    return get_intersect_2el(el1=el, el2=el)


def sort_el_lon_lat(el, netx):
    data = []
    el_sort = []
    for e in el:
        u, v = e
        u_lon = netx.nodes[u]['lon']
        u_lat = netx.nodes[u]['lat']
        v_lon = netx.nodes[v]['lon']
        v_lat = netx.nodes[v]['lat']
        if u_lon <= v_lon:
            el_sort.append([u, v])
            data.append([u_lon, u_lat, v_lon, v_lat])
        else:
            el_sort.append([v, u])
            data.append([v_lon, v_lat, u_lon, u_lat])

    return np.array(el_sort), np.array(data)


def get_lon_lat_el(el, netx):
    data = []
    for e in el:
        u, v = e
        u_lon = netx.nodes[u]['lon']
        u_lat = netx.nodes[u]['lat']
        v_lon = netx.nodes[v]['lon']
        v_lat = netx.nodes[v]['lat']

        data.append([v_lon, v_lat, u_lon, u_lat])

    return np.array(data)


def degree(netx):
    degs = {node: val for (node, val) in netx.degree()}
    nx.set_node_attributes(netx, degs, "degree")
    return netx


def weighted_degree(netx):
    print('Compute Weighted Node degree...')
    # Node attbs have to be dict like
    degs = {node: val for (node, val)
            in netx.degree(weight='weight')}
    nx.set_node_attributes(netx, degs, "weighted_degree")

    return netx


def betweenness(netx):
    print('Compute Betweenness...')
    btn = nx.betweenness_centrality(netx)
    nx.set_node_attributes(netx, btn, "betweenness")

    # Computes as well edge values
    btn = nx.edge_betweenness_centrality(netx, normalized=True)
    nx.set_edge_attributes(netx, btn, "betweenness")

    return netx


def clustering_coeff(netx):
    print('Compute Clustering Coefficient...')
    clust_coff = nx.clustering(netx)
    nx.set_node_attributes(netx, clust_coff, "clustering")

    return netx


# ################## Clustering


def apply_K_means_el(data, n, el=None):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(data)
    cluster_numbers = kmeans.predict(data)
    coord_centers = kmeans.cluster_centers_
    num_clusters = np.max(cluster_numbers)
    cluster_dict = dict()
    if el is not None:
        min_num_elem = 2
        clcnt = 0
        for cn in range(num_clusters+1):
            el_ind_cl = np.where(cluster_numbers == cn)[0]
            if len(el_ind_cl) > min_num_elem:
                el_cl = el[el_ind_cl]
                cluster_dict[clcnt] = el_cl
                clcnt += 1
    return {'number': cluster_numbers,
            'center': coord_centers,
            'cluster': cluster_dict
            }



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    sch.dendrogram(linkage_matrix, **kwargs)


def apply_complete_linkage(data, el=None,
                           method='ward',
                           metric='euclidean',
                           n=None):

    dist_th = 0 if n is None else None
    cluster = AgglomerativeClustering(n_clusters=n,
                                      distance_threshold=dist_th,
                                      affinity=metric,
                                      linkage=method)
    if n is None:
        plot_dendrogram(cluster.fit(data))

    cluster_numbers = cluster.fit_predict(data)
    num_clusters = np.max(cluster_numbers)
    cluster_dict = dict()
    if el is not None:
        min_num_elem = 2
        clcnt = 0
        for cn in range(num_clusters+1):
            el_ind_cl = np.where(cluster_numbers == cn)[0]
            if len(el_ind_cl) > min_num_elem:
                el_cl = el[el_ind_cl]
                cluster_dict[clcnt] = el_cl
            clcnt += 1
    return {'number': cluster_numbers,
            'cluster': cluster_dict,
            }
