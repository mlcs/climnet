from scipy.signal import filtfilt, cheby1, argrelmax,  find_peaks
import numpy as np
from scipy.fft import fft, fftfreq
import scipy.ndimage as ndim


def cheby_lowpass(cutoff, fs, order, rp):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = cheby1(order, rp, normal_cutoff, btype='low', analog=False)
    return b, a


def cheby_lowpass_filter(x, cutoff, fs, order, rp):
    b, a = cheby_lowpass(cutoff, fs, order, rp)
    y = filtfilt(b, a, x)
    return y


def apply_cheby_filter(ts,
                       cutoff=4,
                       order=8,
                       fs=1,
                       rp=.05):
    fcutoff = .95 * 1. / cutoff
    ts_lp = cheby_lowpass_filter(ts, fcutoff, fs, order, rp)
    return ts_lp


def apply_Savitzky_Golay_filter(ts, wl=1, order=3):
    from scipy.signal import savgol_filter

    ts_sav_gol = savgol_filter(ts,
                               window_length=wl,
                               polyorder=order
                               )

    return ts_sav_gol


def apply_butter_filter(ts,
                        cutoff,
                        order=3,
                        fs=50):
    from scipy.signal import butter, filtfilt
    # nyq = 0.5 * fs
    # normal_cutoff = cutoff / nyq  # Nyquist Frequency
    normal_cutoff = 1 / cutoff
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, ts)
    return y


def apply_med_filter(ts,
                     size=7):
    """This function performs a median average of size 7 on a given ts

    Args:
        ts (ndarray): nd array, eg. a time series
        size (int, optional): Size of the rolling median average. Defaults to 7.

    Returns:
        ndarray: median filtered time series
    """

    ts_med_fil = ndim.median_filter(ts, size=size)

    return ts_med_fil


def compute_fft(ts, dt=1):

    n = len(ts)
    F = fft(ts)
    w = fftfreq(n, dt)
    t = np.linspace(1, n, n)
    indices = np.where(w > 0)[0]
    w_pos = abs(w[indices])
    F_pos = abs(F[indices])
    # T = n/t[:len(w_pos)]
    T = 1/w_pos
    return w_pos, F_pos, T


def compute_lead_lag_corr(ts1, ts2, lag=0, corr_method='spearman'):
    import scipy.stats as st
    Nx = len(ts1)
    if Nx != len(ts2):
        raise ValueError('x and y must be equal length')
    if lag != 0:
        print('WARNING! Input 2 is shifted!')
        nts2_shift = np.roll(ts2, lag, axis=0)
    else:
        nts2_shift = ts2
    if corr_method == 'spearman':
        corr, p_val = st.spearmanr(ts1, nts2_shift, axis=0, nan_policy='propagate')
    elif corr_method == 'pearson':
        corr = np.corrcoef(ts1.T, nts2_shift)

    return corr
