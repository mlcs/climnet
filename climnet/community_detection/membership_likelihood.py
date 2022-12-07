import climnet.utils.general_utils as gut
import climnet.utils.spatial_utils as sput
import xarray as xr
from importlib import reload
import climnet.utils.time_utils as tu
import climnet.tsa.time_series_analysis as tsa
from climnet.community_detection.dendrograms import Dendrogram_ES

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_indices_prob_map(prob_map, sig_th=0.95):
    indices = np.where(prob_map > sig_th)[0]
    return indices


def get_sign_prob_map(ds, prob_map, sig_th=0.95):
    indices = get_indices_prob_map(prob_map=prob_map, sig_th=sig_th)
    link_map = ds.get_map_for_idx(idx_lst=indices)

    return link_map


def get_idx_region(ds, def_map,
                   exclude_outlayers=False):
    """
    Gets the indices for a specific dictionary that has lon/lat_range as keys.
    E.g. can be applied to get all indices of the South American monsoon defined by Wang/EE.
    """
    reload(sput)
    if def_map is None:
        def_map = ds.mask
        if def_map is None:
            raise ValueError(
                "ERROR mask is None! Check if mask is computed properly!")

    points = np.where(def_map > 0)[0]
    ids = ds.get_idx_point_lst(point_lst=points)
    mmap = def_map

    if exclude_outlayers:
        gut.myprint(f'Exclude as well outlayers!')
        locs = ds.get_locs_for_indices(idx_lst=ids)
        mean_std_dict = sput.get_mean_std_loc(locs=locs)
        std_lon = mean_std_dict['std_lon']
        std_lat = mean_std_dict['std_lat']
        mean_lon = mean_std_dict['mean_lon']
        mean_lat = mean_std_dict['mean_lat']

        radius_lon = np.abs(2.5*std_lon)
        radius_lat = np.abs(2.5*std_lat)

        ids_include = []
        for idx, (lon, lat) in enumerate(locs):
            if np.abs(lon - mean_lon) < radius_lon and np.abs(lat - mean_lat) < radius_lat:
                ids_include.append(idx)
        # print(ids_include)
        ids = ids[np.array(ids_include)]
        mmap = ds.get_map_for_idx(idx_lst=ids)

    return ids, mmap


def get_data_prob_map(
    ds, prob_map, region_dict,
    sig_th=0.95, sm="Jan", em="Dec", an=False,
    inrange=False,
    exclude_outlayers=False,
    lon_range=None,
    lat_range=None
):
    """Gets for a specific monsoon regions these indices that
    belong to this region

    Args:
        c_indices (list): list of indices
        mname (str): Name of the monsoon region
        sm (str, optional): start month for time Series. Defaults to 'Jan'.
        em (str, optional): end month for time Series. Defaults to 'Dec'.

    Returns:
        dict: dictonary of monsoon containing the time series, the ids and the respective
        maps
    """
    reload(tsa)
    reload(tu)
    data_evs = ds.ds["evs"]
    data = ds.ds["pr"]
    sd, ed = tu.get_time_range(ds.ds)
    print(f"Use data from {sd} to {ed}!")
    if an is True:
        data_an = ds.ds["an"]

    if sm != "Jan" or em != "Dec":
        data = tu.get_month_range_data(data, start_month=sm, end_month=em)
        data_evs = tu.get_month_range_data(
            data_evs, start_month=sm, end_month=em)
        if an is True:
            data_an = ds.get_month_range_data(
                data_an, start_month=sm, end_month=em)

    this_map = get_sign_prob_map(ds=ds, prob_map=prob_map, sig_th=sig_th)
    region_indices, region_map = get_idx_region(
        ds, def_map=this_map,
        exclude_outlayers=exclude_outlayers
    )
    points = ds.get_points_for_idx(region_indices)  # This is important!
    if an:
        xr_ts = xr.merge(
            [
                data_evs.sel(points=points),
                data.sel(points=points),
                data_an.sel(points=points),
            ]
        )
    else:
        xr_ts = xr.merge(
            [data_evs.sel(points=points), data.sel(points=points)])

    if inrange:
        lon_range = region_dict["lon_range"] if lon_range is None else lon_range
        lat_range = region_dict["lat_range"] if lat_range is None else lat_range
        gut.myprint(f'Include only {lon_range}, {lat_range}...')
        def_map_range = sput.get_locations_in_range(
            lon_range=lon_range, lat_range=lat_range, def_map=region_map,
            dateline=False
        )
        points_range = np.where(def_map_range > 0)[0]
        region_indices_range = ds.get_idx_point_lst(point_lst=points_range)
        xr_ts_range = xr.merge(
            [data_evs.sel(points=points_range),
             data.sel(points=points_range)])
    else:
        def_map_range = None
        points_range = None
        region_indices_range = None
        xr_ts_range = None

    return {
        "data": xr_ts,
        "map": region_map,
        "ids": region_indices,
        "pids": points,
        "map_range": def_map_range,
        "points_range": points_range,
        "region_indices_range": region_indices_range,
        "data_range": xr_ts_range
    }


def get_regions_data_prob_map(
    ds_cd,
    all_region_dict,
    arr_names=[],
    sig_th=0.95,
    sm="Jan",
    em="Dec",
    an=False,
    inrange=False,
    exclude_outlayers=False,  # So far only based on spatial location
    lon_range=None,
    lat_range=None
):
    new_dict = dict()
    for region_name, region in all_region_dict.items():
        if region_name in arr_names or len(arr_names) == 0:
            print(f'{region_name}')
            this_m_dict = dict()

            prob_map, prob_map_std, gr_maps, all_comm = ds_cd.compute_prob_map(
                all_region_dict=all_region_dict, arr_regions=[region_name]
            )
            this_m_dict.update(region)
            this_m_dict['prob_map'] = prob_map
            this_m_dict['prob_map_std'] = prob_map_std
            this_m_dict['gr_maps'] = gr_maps
            this_m_dict['all_comm'] = all_comm

            new_data = get_data_prob_map(
                ds=ds_cd.ds,
                prob_map=prob_map,
                region_dict=region,
                sig_th=sig_th,
                sm=sm,
                em=em,
                an=an,
                inrange=inrange,
                exclude_outlayers=exclude_outlayers,
                lon_range=lon_range,
                lat_range=lat_range
            )
            this_m_dict.update(new_data)
            new_dict[region_name] = this_m_dict

    return new_dict


def get_yearly_sum(
    dict, sy, ey,
):
    sum_arr = []
    for year in range(sy, ey + 1, 1):
        y_sum = np.sum(dict[year]["pr"])
        sum_arr.append(y_sum)

    return np.array(sum_arr)


def get_yearly_av(
    dict, sy, ey,
):
    av_arr = []
    for year in range(sy, ey + 1, 1):
        y_av = np.mean(dict[year])
        av_arr.append(y_av)

    return np.array(av_arr)


def get_yearly_av_var(dict, sy, ey, var="pr"):
    av_arr = []
    for year in range(sy, ey + 1, 1):
        y_av = np.mean(dict[year][var])
        av_arr.append(y_av)

    return np.array(av_arr)


def get_av_region(
    dict, var="pr",
):
    val_ts = dict[var]
    mean_ts = np.mean(val_ts, axis=0)
    if len(mean_ts) != len(dict["times"]):
        raise ValueError("Times and value ts not of same length")

    return mean_ts
