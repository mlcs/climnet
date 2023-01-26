# %%
import numpy as np
from tqdm import tqdm
import os
import sys
import climnet.network.network_functions as nwf
import geoutils.utils.general_utils as gut
import geoutils.tsa.event_synchronization as es
from importlib import reload
# %%
PATH = os.path.dirname(os.path.abspath(__file__))
es_filespath = f"{PATH}/es_files/"
null_model_folder = f"{es_filespath}/null_model/"
# %%


def create(
    ds,
    taumax=10,
    E_matrix_folder=None,
    null_model_file=None,
    q_med=0.5,
    lq=0.25,
    hq=0.75,
    q_sig=0.95,
    min_num_sync_events=2,
    weighted=False,
    num_cpus=None,
):
    """
    This function has to be called twice, once, to compute the exact numbers of synchronous
    events between two time series, second again to compute the adjacency matrix
    and the link bundles.
    Attention: The number of parrallel jobs that were used for the E_matrix needs to be
    passed correctly to the function.
    """
    # Test if ES data is computed
    if ds.ds["evs"] is None:
        raise ValueError(
            "ERROR Event Synchronization data is not computed yet")

    num_time_series = len(ds.indices_flat)

    if E_matrix_folder is None:
        E_matrix_folder = (
            es_filespath +
            f"/E_matrix/{ds.var_name}_{ds.grid_step}/"
        )
    else:
        E_matrix_folder = es_filespath + \
            f"/E_matrix/{E_matrix_folder}"

    if null_model_file is None:
        raise ValueError("ERROR Please provide null_model_file!")
    else:
        null_model_file = f"{null_model_folder}/{null_model_file}"
        if not os.path.exists(null_model_file):
            raise ValueError(f"File {null_model_file} does not exist!")
    print(f"Load null model: {null_model_file}", flush=True)
    null_model = np.load(null_model_file, allow_pickle=True).item()

    event_synchronization_run(
        ds=ds,
        E_matrix_folder=E_matrix_folder,
        taumax=taumax,
        null_model=null_model,
        min_num_sync_events=min_num_sync_events,
        num_cpus=num_cpus,
    )

    adjacency, corr = compute_es_adjacency(
        E_matrix_folder=E_matrix_folder,
        num_time_series=num_time_series,
        weighted=weighted,
        q_dict=null_model,
        q_med=q_med,
        lq=lq,
        hq=hq,
        q_sig=q_sig,
    )

    nwf.get_sparsity(M=adjacency)
    return adjacency, corr


def event_synchronization_run(
    ds,
    E_matrix_folder=None,
    taumax=10,
    min_num_sync_events=2,
    null_model=None,
    num_cpus=None,
):
    """
    Event synchronization definition

    Parameters
    ----------
    E_matrix_folder: str
        Folder where E-matrix files are to be stored
    taumax: int
        Maximum time delay between two events
    min_num_sync_events: int
        Minimum number of sychronous events between two time series
    null_model_file: str
        File where null model for Adjacency is stored. If None, it will be computed again (costly!)
    Return
    ------
    None
    """
    reload(es)
    # for job array
    job_id, num_jobs = gut.get_job_array_ids()
    event_series_matrix = ds.get_defined_ds().evs.transpose('points', 'time').compute()  # transpose to get array of timeseries!
    event_series_matrix = event_series_matrix.data
    gut.myprint('Computed Event Series Matrix')

    if not os.path.exists(E_matrix_folder):
        os.makedirs(E_matrix_folder)
    E_matrix_filename = (
        f"E_matrix_{ds.var_name}_"
        f"q_{ds.q}_min_num_events_{ds.min_evs}_"
        f"taumax_{taumax}_jobid_{job_id}.npy"
    )
    gut.myprint(f'Store E-matrix as file {E_matrix_filename}!')
    print(
        f"JobID {job_id}: Start comparing all time series with taumax={taumax}!")
    if not os.path.exists(E_matrix_folder + E_matrix_filename):
        es.parallel_event_synchronization(
            event_series_matrix,
            taumax=taumax,
            min_num_sync_events=min_num_sync_events,
            job_id=job_id,
            num_jobs=num_jobs,
            savepath=E_matrix_folder + E_matrix_filename,
            q_dict=null_model,
            q_min=0.5,  # This is not the sign threshold for later tests, it is only to store less data
            num_cpus=num_cpus,
        )
    else:
        print(
            f"File {E_matrix_folder+E_matrix_filename} does already exist!")

    path = E_matrix_folder
    E_matrix_files = [os.path.join(path, fn)
                      for fn in next(os.walk(path))[2]]
    if len(E_matrix_files) < num_jobs:
        gut.myprint(
            f"JobId {job_id}: Finished. Missing {num_jobs} - {len(E_matrix_files)}!",
        )
        sys.exit(0)
    else:
        gut.myprint(f'Last job (id: {job_id}) finished. Now computing full E-matrix!')

    return None


def compute_es_adjacency(
    E_matrix_folder,
    num_time_series,
    weighted=False,
    q_dict=None,
    q_med=0.5,
    lq=0.25,
    hq=0.75,
    q_sig=0.95,
):
    if not os.path.exists(E_matrix_folder):
        raise ValueError("ERROR! The parallel ES is not computed yet!")

    adj_null_model, weights = es.get_adj_from_E(
        E_matrix_folder,
        num_time_series,
        weighted=weighted,
        q_dict=q_dict,
        q_med=q_med,
        lq=lq,
        hq=hq,
        q_sig=q_sig,
    )

    if adj_null_model.shape != weights.shape:
        raise ValueError("Adjacency and weights not of the same shape!")

    return adj_null_model, weights
