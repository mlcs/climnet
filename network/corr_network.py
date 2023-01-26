# %%
from importlib import reload
import geoutils.utils.general_utils as gut
import numpy as np
import geoutils.utils.statistic_utils as sut
import multiprocessing as mpi
from joblib import Parallel, delayed
from tqdm import tqdm
import geoutils.tsa.iaaft as iaaft


def create(
        ds,
        corr_method="spearman",
        threshold=None,
        density=None,
        stat_test="twosided",
        confidence=0.99,
        significance_test="bonf",
        **kwargs):
    """Creates the climate network using a given correlation method.

    Parameters:
    ----------
    **kwargs: dict
        Arguments for get_adjacency().

    """
    corr, pvalue = get_corr(dataset=ds,
                            corr_method=corr_method,
                            stat_test=stat_test,)

    adjacency = get_adjacency_corr(
        ds=ds,
        corr=corr,
        pvalue=pvalue,
        threshold=threshold,
        density=density,
        confidence=confidence,
        significance_test=significance_test,
        **kwargs,
    )

    return adjacency, corr


def get_corr(dataset, corr_method="spearman", stat_test="twosided"):
    if corr_method == "spearman":
        corr, pvalue = calc_spearman(
            dataset, stat_test)

    # pearson correlation
    elif corr_method == "pearson":
        corr, pvalue = calc_pearson(dataset)
    else:
        raise ValueError(
            f"Choosen correlation method {corr_method} does not exist!")

    return corr, pvalue


def calc_spearman(dataset, test="twosided"):
    """Spearman correlation of the flattened and remove NaNs object.
    """
    data = dataset.flatten_array()  # This is a dataset object of the dataset class!
    print(data.shape)
    if test == 'fully_connected':
        mytest = 'twosided'
    else:
        mytest = test
    corr, pvalue = sut.calc_spearman(data=data, test=mytest, verbose=True)

    return corr, pvalue


def calc_pearson(dataset):
    """Pearson correlation of the flattened array."""
    data = dataset.flatten_array()  # This is a dataset object of the dataset class!
    print(data.shape)
    # Pearson correlation
    corr, pvalue = sut.calc_pearson(data=data, verbose=True)
    return corr, pvalue


def get_idx_local_corr(idx, ds, num_nn, corr):
    nidx_lst = ds.get_n_ids(nid=idx, num_nn=num_nn)

    # Start from 1 as 0 is idx itself
    local_corr_vals = corr[(idx, nidx_lst[1:])]
    local_corr_idx = np.mean(local_corr_vals)

    return idx, local_corr_idx


def local_corr(ds,
               corr_method='spearman',
               stat_test='twosided',
               num_nn=10,
               idx_lst=None
               ):
    reload(gut)
    if idx_lst is None:
        idx_lst = ds.indices_flat
    corr, pvalue = get_corr(dataset=ds,  # this is the dataset of the dataset class
                            corr_method=corr_method,
                            stat_test=stat_test)

    local_corr = gut.nans_array(len(ds.indices_flat))

    backend = 'multiprocessing'
    # backend='loky'
    # backend='threading'
    num_cpus_avail = mpi.cpu_count()
    print(f"Number of available CPUs: {num_cpus_avail}")

    parallelArray = (Parallel(n_jobs=num_cpus_avail, backend=backend)
                     (delayed(get_idx_local_corr)
                      (idx, ds, num_nn, corr)
                      for idx in tqdm(idx_lst)
                      )
                     )

    for idx, local_corr_idx in tqdm(parallelArray):
        local_corr[idx] = local_corr_idx

    # get a map of the local correlations
    loc_corr_map = ds.get_map(local_corr)

    return loc_corr_map


def get_adjacency_corr(
    ds,
    corr,
    pvalue,
    threshold=None,
    density=None,
    confidence=0.95,
    significance_test="bonf",
    corr_method=None,
    **iaaft_kwargs,
):
    """Create adjacency matrix from spearman correlation.

    Args:
    -----
    corr: np.ndarray (N x N)
        Spearman correlation matrix
    pvalue: np.ndarray (N x N)
        Pairwise pvalues of correlation matrix
    threshold: float
        Threshold to cut correlation
    confidence: float
        Confidence level
    significance_test: str
        type of statistical p-value testing
    Returns:
    --------
    adjacency: np.ndarray (N x N)
    """
    # Significance test on p-values
    if significance_test == "bonf" or significance_test == "dunn":
        pval_flat = pvalue.flatten()
        indices = sut.holm(
            pval_flat, alpha=(1 - confidence), corr_type=significance_test
        )
        mask_list = np.zeros_like(pval_flat)
        mask_list[indices] = 1
        mask_confidence = np.reshape(mask_list, pvalue.shape)
    elif significance_test == "standard":
        mask_confidence = np.where(pvalue <= (
            1 - confidence), 1, 0)  # p-value test
    elif significance_test == "iaaft":
        # self for read out TODO: delete in the future
        corr_distribution = get_iaaft_surrogates(
            ds, corr_method=corr_method, **iaaft_kwargs
        )
        q_corr = np.quantile(
            np.abs(corr_distribution), q=confidence, axis=0)
        mask_confidence = np.where(np.abs(corr) >= q_corr, 1, 0)
    elif significance_test == 'fully_connected':
        gut.myprint('WARNING! Network is fully connected!')
        mask_confidence = np.ones_like(corr)
    else:
        raise ValueError(f"Method {significance_test} does not exist!")

    # Threhold test on correlation values
    if threshold is not None and density is not None:
        raise ValueError("Threshold and density cannot both be specified!")

    if threshold is not None:
        mask_correlation = np.where(np.abs(corr) >= threshold, 1, 0)
    elif density is not None:
        if density < 0 or density > 1:
            raise ValueError(
                f"Density must be between 0 and 1 but is {density}!")
        num_links = int(density * len(corr) ** 2)
        buff = np.abs(corr)
        for i in range(corr.shape[0]):
            buff[i, i] = 0  # set diagonale to 0
        # Sort the 2 d correlations
        sorted_values = np.sort(buff.flatten())
        threshold = sorted_values[-num_links]
        mask_correlation = np.where(buff >= threshold, 1, 0)
        print(f"Minimum Correlation values: {threshold}")
    elif threshold is None and density is None:
        gut.myprint('WARNING! No threshold set, all correlation values are links!')
        mask_correlation = 1

    adjacency = mask_confidence * mask_correlation
    # set diagonal to zero
    for i in range(len(adjacency)):
        adjacency[i, i] = 0
    print("Created adjacency matrix.")

    adjacency = adjacency * adjacency.transpose()

    return adjacency


def get_iaaft_surrogates(
    dataset,
    num_surrogates=1000,
    corr_method="pearson",
    tol_pc=5.0,
    verbose=True,
    maxiter=1e6,
    sorttype="quicksort",
    num_cpus=mpi.cpu_count(),
):
    """Applies Iterative adjusted amplitude Fourier transform for all spatial locations
    of time series in

    Args:
        dataset (BaseDataset): BaseDateset containing the time series
        num_surrogates (int): Number of surrogates, default is 1000
        corr_method (str, optional): Correlation method. Defaults to 'pearson'.
        tol_pc (float, optional): tolerance values. Defaults to 5..
        verbose (bool, optional): verbose outputs. Defaults to False.
        maxiter (int, optional): maximum iteration times. Defaults to 1E6.
        sorttype (str, optional): are arguments sorted. Defaults to "quicksort".

    Returns:
        [type]: [description]
    """
    print(
        "Applies Iterative adjusted amplitude Fourier transform for all pairs"
        " of spatial locations may take a while!"
    )

    data = dataset.flatten_array()
    print(data.shape)

    print(f"Number of available CPUs: {num_cpus}")
    backend = "loky"
    corr_matrices = Parallel(n_jobs=num_cpus, backend=backend)(
        delayed(iaaft_surrogate)(
            data, corr_method, tol_pc, maxiter, sorttype)
        for s in tqdm(range(num_surrogates))
    )

    return np.array(corr_matrices)


def iaaft_surrogate(
    data, corr_method="pearson", tol_pc=5.0, maxiter=1e6, sorttype="quicksort"
):
    import scipy.stats as stat
    """Applies Iterative adjusted amplitude Fourier transform for all pairs of
    spatial locations.

    Args:
        data (np.ndarray): Array of time series (N,M)
        corr_method (str, optional): Correlation method. Defaults to 'pearson'.
        tol_pc (float, optional): tolerance values. Defaults to 5..
        verbose (bool, optional): verbose outputs. Defaults to False.
        maxiter (int, optional): maximum iteration times. Defaults to 1E6.
        sorttype (str, optional): are arguments sorted. Defaults to "quicksort".

    Returns:
        [type]: [description]
    """
    surr_arr = []
    for ts in data:
        ts_iaaft = iaaft.surrogates(
            ts,
            ns=1,
            verbose=False,
            maxiter=maxiter,
            tol_pc=tol_pc,
            sorttype=sorttype,
        )[0]
        surr_arr.append(ts_iaaft)

    assert np.array(surr_arr).shape == data.shape

    if corr_method == "pearson":
        corr = np.corrcoef(np.array(surr_arr).T)
    elif corr_method == "spearman":
        corr, _ = stat.spearmanr(data, axis=0, nan_policy="propagate")
    else:
        raise ValueError(f"Correlation method {corr_method} not known!")

    return corr
