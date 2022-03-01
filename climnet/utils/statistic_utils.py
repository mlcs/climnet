from scipy.stats import skew
import scipy.special as special
import scipy.stats as st
import numpy as np


def holm(pvals, alpha=0.05, corr_type="bonf"):
    """
    Returns indices of p-values using Holm's method for multiple testing.

    Args:
    ----
    pvals: list
        list of p-values
    alpha: float
        TODO
    corr_type: str
        TODO
    """
    n = len(pvals)
    sortidx = np.argsort(pvals)
    p_ = pvals[sortidx]
    j = np.arange(1, n + 1)
    if corr_type == "bonf":
        corr_factor = alpha / (n - j + 1)
    elif corr_type == "dunn":
        corr_factor = 1. - (1. - alpha) ** (1. / (n - j + 1))
    try:
        idx = np.where(p_ <= corr_factor)[0][-1]
        lst_idx = sortidx[:idx]
    except IndexError:
        lst_idx = []
    return lst_idx


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v/norm


def standardize(dataset, axis=0):
    return (dataset - np.average(dataset, axis=axis)) / (np.std(dataset, axis=axis))


def normalize_minmax(dataarray, axis=None):
    """Normalize dataarray between 0, 1

    Args:
        dataarray ([type]): [description]
    """
    if axis is None:
        flatten = dataarray.stack(z=dataarray.dims)
    else:
        flatten = dataarray.stack(z=dataarray.dims[axis])
    norm_data = (
        (flatten - flatten.min(skipna=True)) /
        (flatten.max(skipna=True) - flatten.min(skipna=True))
    )
    return norm_data.unstack('z')


# Spearman's Correlation
def calc_spearman(data, test='onesided', verbose=False):
    """Spearman correlation of the flattened and without NaNs object.
    """

    corr, pvalue_twosided = st.spearmanr(
        data, axis=0, nan_policy='propagate')

    if test == 'onesided':
        pvalue, zscore = onesided_test(corr)
    elif test == 'twosided':
        pvalue = pvalue_twosided
    else:
        raise ValueError('Choosen test statisics does not exist. Choose "onesided" '
                         + 'or "twosided" test.')
    if verbose:
        print(f"Created spearman correlation matrix of shape {np.shape(corr)}")

    return corr, pvalue


# Pearson's Correlation
def calc_pearson(data, verbose=False):
    """Pearson correlation of the flattened array."""
    # Pearson correlation
    corr = np.corrcoef(data.T)
    assert corr.shape[0] == data.shape[1]

    # get p-value matrix
    # https://stackoverflow.com/questions/24432101/correlation-coefficients-and-p-values-for-all-pairs-of-rows-of-a-matrix
    # TODO: Understand and check the implementation
    rf = corr[np.triu_indices(corr.shape[0], 1)]
    df = data.shape[1] - 2
    ts = rf * rf * (df / (1 - rf * rf))
    pf = special.betainc(0.5 * df, 0.5, df / (df + ts))
    p = np.zeros(shape=corr.shape)
    p[np.triu_indices(p.shape[0], 1)] = pf
    p[np.tril_indices(p.shape[0], -1)
      ] = p.T[np.tril_indices(p.shape[0], -1)]
    p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])

    if verbose:
        print(f"Created pearson correlation matrix of shape {np.shape(corr)}")

    return corr, p


def get_corr_function(corr_method):
    # spearman correlation
    if corr_method == 'spearman':
        return calc_spearman
    # pearson correlation
    elif corr_method == 'pearson':
        return calc_pearson
    else:
        raise ValueError("Choosen correlation method does not exist!")


def onesided_test(corr):
    """P-values of one sided t-test of spearman correlation.
    Following: https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    """
    n = corr.shape[0]
    f = np.arctanh(corr)
    zscore = np.sqrt((n-3)/1.06) * f
    pvalue = 1 - st.norm.cdf(zscore)

    return pvalue, zscore


def loghist(arr, nbins=None, density=True):
    """
    Returns the histogram counts on a logarithmic binning.
    """
    if nbins is None:
        nbins = __doane(np.log(arr))
    bins = np.logspace(np.log10(arr.min()),
                       np.log10(arr.max() + 0.01),
                       nbins + 1)
    hc, be = np.histogram(arr, bins=bins, density=density)
    bc = 0.5 * (be[1:] + be[:-1])

    return hc, bc, be


def hist(arr, nbins=None, bw=None,
         min_bw=None, max_bw=None,
         density=True):
    if nbins is None:
        nbins = __doane(arr)

    if bw is None:
        bins = np.linspace(arr.min(),
                           arr.max() + 0.01,
                           nbins + 1
                           )
    else:
        minb = min_bw if min_bw is not None else arr.min()
        maxb = max_bw if max_bw is not None else arr.max()
        bins = np.arange(minb,
                         maxb + 0.01,
                         bw)
        # print(bw, min_bw, max_bw, bins)
    hc, be = np.histogram(arr, bins=bins, density=density)
    bc = 0.5 * (be[1:] + be[:-1])

    return hc, bc, be


def __doane(arr):
    """
    Returns the number of bins according to Doane's formula.

    More info:
    https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
    """
    n = float(len(arr))
    g1 = skew(arr)
    sig_g1 = np.sqrt((6. * (n - 2)) / ((n + 1) * (n + 3)))
    nbins = int(np.ceil(1. + np.log2(n) + np.log2(1 + np.abs(g1) / sig_g1)))

    return nbins


def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (st.entropy(p, m) + st.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def KS_test(data1, data2=None, test='norm'):
    if data2 is not None:
        KS_st, p_val = st.ks_2samp(data1, data2)
    else:
        KS_st, p_val = st.kstest(data1, test)

    return KS_st, p_val
