import numpy as np
import climnet.utils.general_utils as gut


def node_level_arr(level_arr):
    buff = np.array(level_arr[0])
    x = [buff]
    count_nodes = [gut.count_elements(buff)]
    for li in range(1, len(level_arr)):
        buff = np.array(level_arr[li])[buff]
        this_node_count = gut.count_elements(buff)
        x.append(buff)
        count_nodes.append(this_node_count)

    return np.array(x), count_nodes


def level_dict(arr_levels):
    """
    Gives dict for which nodes before are merged into this node.
    Works for group_levels and node_levels
    """
    level_dict = dict()
    for l_id, level_ids in enumerate(arr_levels):
        this_node_count = gut.count_elements(level_ids)
        level_dict[l_id] = this_node_count
    return level_dict


def reduce_node_levels(node_levels):
    """
    Graph_tool with MCMC search does sometimes skip certain group numbers.
    This function brings back the ordering to numbers from 0 to len(level).
    """
    red_hierach_data = []
    trans_dict = dict()
    all_level_dict = level_dict(node_levels)
    node_red_dict = dict()
    for l_id, this_level_dict in enumerate(all_level_dict.values()):
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
