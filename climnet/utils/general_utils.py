"""General Util functions."""
import pickle
from itertools import combinations_with_replacement
from scipy.signal import find_peaks
import contextlib
import os
import numpy as np
import xarray as xr
SEED = 42


@contextlib.contextmanager
def temp_seed(seed=SEED):
    """Set seed locally.
    Usage:
    ------
    with temp_seed(42):
        np.random.randn(3)

    Parameters:
    ----------
    seed: int
        Seed of function, Default: SEED
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_all_combs(arr):
    return list(combinations_with_replacement(arr, r=2))


def count_elements(arr):
    """Returns a dictionary of elements in array and their
    corresponding counts.py

    Args:
        arr (list): list of elements

    Returns:
        dict: dictionary (element: Counter)
    """
    unique, counts = np.unique(arr, return_counts=True)
    this_arr_count = dict(zip(unique, counts))
    return this_arr_count


def tryToCreateDir(d):
    dirname = os.path.dirname(d)
    try:
        os.makedirs(dirname)
    except FileExistsError:
        print("Directory ", d,  " already exists")

    return None


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def check_range(arr_range, compare_arr_range):
    min_arr = min(arr_range)
    max_arr = max(arr_range)

    min_carr = min(compare_arr_range)
    max_carr = max(compare_arr_range)

    if min_arr <= min_carr or max_arr >= max_carr:
        return False
    else:
        return True


def check_range_val(val, arr_range):
    min_arr = min(arr_range)
    max_arr = max(arr_range)

    if min_arr < val and max_arr > val:
        return True
    else:
        return False


def find_local_min(data_series):

    peak_idx, _ = find_peaks(data_series*(-1))
    peak_val = data_series[peak_idx]

    return {'idx': peak_idx, 'val': peak_val}


def find_local_max(data_series):

    peak_idx, _ = find_peaks(data_series)
    peak_val = data_series[peak_idx]
    if len(peak_idx) == 0:
        peak_idx = []
        peak_val = []
    return {'idx': peak_idx, 'val': peak_val}


def get_locmax_of_score(ts, q=0.95):
    q_value = np.quantile(ts, q)
    peak_idx, _ = find_peaks(ts, height=q_value, distance=1, prominence=1)
    peak_val = ts[peak_idx]
    return peak_val, peak_idx


def get_locmax_of_ts(ts, q=0.95):
    """Gets the maximum timepoints of xr.Dataarray time series for given quantile.

    Args:
        ts (xr.Dataarray): dataarray that contains the different time series.
        q (float, optional): qunatile value above the values are chosen. Defaults to 0.95.

    Returns:
        xr.Dataarray: returns the time points as np.datetime64
    """
    q_value = np.quantile(ts, q)
    peak_idx, _ = find_peaks(ts, height=q_value,
                             distance=1,
                             #  prominence=1
                             )
    peak_ts = ts[peak_idx]

    return peak_ts


def save_np_dict(arr_dict, sp):
    print(f'Store to {sp}')
    np.save(sp, arr_dict, allow_pickle=True)
    return None


def save_pkl_dict(arr_dict, sp):
    with open(sp, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(arr_dict, f, pickle.HIGHEST_PROTOCOL)
    return None


def load_np_dict(sp):
    print(f'Load {sp}')
    return np.load(sp, allow_pickle=True).item()


def load_npy(fname):
    """Load .npy files and convert dict to xarray object.

    Args:
        fname (str): Filename

    Returns:
        converted_dic [dict]: dictionary of stored objects
    """
    dic = np.load(fname,
                  allow_pickle=True).item()
    converted_dic = {}
    for key, item in dic.items():
        # convert dict to xarray object
        if isinstance(item, dict):
            if 'data_vars' in item.keys():
                item = xr.Dataset.from_dict(item)
            elif 'data' in item.keys():
                item = xr.DataArray.from_dict(item)
        # store object to new dict
        converted_dic[key] = item

    return converted_dic


def get_source_target_corr(corr, sids):
    source_corr = corr[0:len(sids), 0:len(sids)]
    target_corr = corr[np.ix_(np.arange(len(sids), len(corr)),
                              np.arange(len(sids), len(corr))
                              )
                       ]
    source_target_corr = corr[np.ix_(np.arange(0, len(sids)),
                                     np.arange(len(sids), len(corr))
                                     )
                              ]
    return {'source': source_corr,
            'target': target_corr,
            'source_target': source_target_corr}


def mk_grid_array(data, x_coords=None, y_coords=None,
                  x_coord_name='x', y_coord_name='y',
                  name='data',
                  **kwargs):

    if x_coords is None:
        x_coords = np.arange(0, data.shape[1])
    if y_coords is None:
        y_coords = np.arange(0, data.shape[0])

    xr_2d = xr.DataArray(
        data=np.array(data),
        dims=[y_coord_name, x_coord_name],  # x and y are in data.shape(y,x)
        coords={
            x_coord_name: x_coords,
            y_coord_name: y_coords,
        },
        name=name
    )

    return xr_2d


def remove_tids_sids(sids, tids):
    t_in_s = np.in1d(tids, sids)
    num_s_in_t = np.count_nonzero(t_in_s)
    if num_s_in_t > 0:
        print(f'Remove {num_s_in_t} targets that are in source!')
        # remove target links that are as well in source
        tids = tids[~np.in1d(tids, sids)]
    return tids

# These functions below are taken from
# https://stackoverflow.com/questions/50299172/python-range-or-numpy-arange-with-end-limit-include


def cust_range(*args, rtol=1e-05, atol=1e-08,
               include=[True, False],
               dtype=float):
    """
    Combines numpy.arange and numpy.isclose to mimic
    open, half-open and closed intervals.
    Avoids also floating point rounding errors as with
    >>> numpy.arange(1, 1.3, 0.1)
    array([1. , 1.1, 1.2, 1.3])

    args: [start, ]stop, [step, ]
        as in numpy.arange
    rtol, atol: floats
        floating point tolerance as in numpy.isclose
    include: boolean list-like, length 2
        if start and end point are included
    """
    # process arguments
    if len(args) == 1:
        start = 0
        stop = args[0]
        step = 1
    elif len(args) == 2:
        start, stop = args
        step = 1
    else:
        assert len(args) == 3
        start, stop, step = tuple(args)

    # determine number of segments
    n = (stop-start)/step + 1

    # do rounding for n
    if np.isclose(n, np.round(n), rtol=rtol, atol=atol):
        n = np.round(n)

    # correct for start/end is exluded
    if not include[0]:
        n -= 1
        start += step
    if not include[1]:
        n -= 1
        stop -= step
    return np.linspace(start, stop, int(n), dtype=dtype)


def crange(*args, **kwargs):
    return cust_range(*args, **kwargs, include=[True, True])


def orange(*args, **kwargs):
    return cust_range(*args, **kwargs, include=[True, False])


def get_random_numbers_no_neighboring_elems(min_num, max_num, amount):
    """Generates amount random numbers in [min_num,..,max_num] that do not
    include neighboring numbers."""

    # this is far from exact - it is best to have about 5+ times the amount
    # of numbers to choose from - if the margin is too small you might take
    # very long to get all your "fitting numbers" as only about 1/4 of the range
    # is a viable candidate (worst case):
    #   [1 2 3 4 5 6 7 8 9 10]: draw 2 then 5 then 8 and no more are possible
    if (max_num-min_num) // 5 < amount:
        raise ValueError(f"Range too small - increase given range.")

    l = set()
    while len(l) != amount:
        a = np.random.randint(min_num, max_num)
        if not {a-1, a, a+1} & l:  # set intersection: empty == False == no commons
            l.add(a)
    return np.array(sorted(l))


def get_job_array_ids(def_id=0):
    # for job array on slurm cluster
    try:
        min_job_id = int(os.environ['SLURM_ARRAY_TASK_MIN'])
        max_job_id = int(os.environ['SLURM_ARRAY_TASK_MAX'])
        job_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        num_jobs = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        num_jobs = max_job_id
        print(
            f"job_id: {job_id}/{num_jobs}, Min Job ID: {min_job_id}, Max Job ID: {max_job_id}",
            flush=True)

    except KeyError:
        job_id = 0
        num_jobs = 1
        print("Not running with SLURM job arrays, but with manual id: ", job_id,
              flush=True)

    return job_id, num_jobs


def get_job_id(def_id=0):
    # for job array
    job_id, _ = get_job_array_ids(def_id=def_id)
    return job_id
