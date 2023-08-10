# %%
import networkx as nx
import networkit as nk
import numpy as np
import climnet.community_detection.cd_functions as cdf
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut


def get_hard_cluster(nk_partition):
    return nk_partition.getVector()


def apply_nk_cluster(cnk, cd_name="PLM"):
    if cd_name == "PLM":
        nkCommunities = nk.community.detectCommunities(
            cnk, algo=nk.community.PLM(cnk, True)
        )
    elif cd_name == "PLP":
        nkCommunities = nk.community.detectCommunities(
            cnk, algo=nk.community.PLP(cnk))
    else:
        raise ValueError(f"Algorithm {cd_name} is not known!")

    groupIds = np.array(get_hard_cluster(nk_partition=nkCommunities))
    redGroupIds = cdf.reduce_node_levels([groupIds])

    return np.array(redGroupIds[0])


def save_hard_cluster(cluster, savepath):
    result_dict = dict(
        node_levels=np.array([cluster]),
        hard_cluster=cluster
    )
    fut.save_np_dict(result_dict, savepath)
