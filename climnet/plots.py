"""Basic plotting functions for maps"""
# import matplotlib.cm as cm
import seaborn as sns
import copy
import climnet.utils.statistic_utils as sut
import climnet.utils.general_utils as gut
import os
import string
import numpy as np
import scipy.interpolate as interp
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
# plt.style.use('./src/matplotlib_style.py')
import cartopy.crs as ccrs
import cartopy as ctp
import matplotlib.ticker
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from importlib import reload

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20
plt.rcdefaults()

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
# fontsize of the x and y labels
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rcParams['pcolor.shading'] ='nearest' # For pcolormesh
colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:orange',
          'm', 'c', 'y', 'tab:purple', 'darkviolet', 'slategray']


def plot_entropy(num_runs=1,
                 max_num_levels=10,
                 graph_folder=None,
                 graph_file=None,
                 ylim=(1e5, 1e7),
                 ax=None):

    sbm_entropy_arr = np.zeros((num_runs, max_num_levels))
    sbm_num_groups_arr = np.zeros((num_runs, max_num_levels))

    counter = 0
    for idx, job_id in enumerate(range(0, num_runs)):
        if graph_folder is None:
            raise ValueError("Graph path folder not provided!")
        else:
            sbm_filepath = graph_folder + f"{job_id}_" + graph_file

        if not os.path.exists(sbm_filepath + '_group_levels.npy'):
            print(
                f"WARNING file {sbm_filepath +'_group_levels.npy'} does not exist!")
            sbm_entropy_arr = np.delete(sbm_entropy_arr, counter, 0)
            sbm_num_groups_arr = np.delete(sbm_num_groups_arr, counter, 0)
            continue
        sbm_entropy = np.load(sbm_filepath+'_entropy.npy',  allow_pickle=True)
        sbm_num_groups = np.load(
            sbm_filepath+'_num_groups.npy',  allow_pickle=True)

        sbm_entropy_arr[idx, :len(sbm_entropy)] = sbm_entropy
        sbm_num_groups_arr[idx, :len(sbm_num_groups)] = sbm_num_groups

    mean_entropy = np.mean(sbm_entropy_arr, axis=0)
    std_entropy = np.std(sbm_entropy_arr, axis=0)
    mean_num_groups = np.mean(sbm_num_groups_arr, axis=0)
    std_num_groups = np.std(sbm_num_groups_arr, axis=0)

    # -1 Because last level is trivial!
    mean_entropy = mean_entropy[:-1]
    std_entropy = std_entropy[:-1]
    mean_num_groups = mean_num_groups[:-1]
    std_num_groups = std_num_groups[:-1]

    # Now plot
    from matplotlib.ticker import MaxNLocator
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        prepare_axis(ax)
    num_levels = len(mean_entropy)
    ax.set_xlabel('Level')

    # Entropy
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(r'Description Length $\Gamma$')

    x_data = np.arange(1, num_levels+1)
    ax.errorbar(x_data, (mean_entropy), yerr=(std_entropy),
                color='tab:blue', elinewidth=2, label='Entropy')
    ax.fill_between(x_data, mean_entropy - std_entropy, mean_entropy + std_entropy,
                    color='tab:blue', alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim(ylim)

    ax.yaxis.label.set_color('tab:blue')
    y_major = matplotlib.ticker.LogLocator(base=10.0, numticks=10)
    ax.yaxis.set_major_locator(y_major)
    y_minor = matplotlib.ticker.LogLocator(
        base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.tick_params(axis='y', colors='tab:blue')

    # Number of Groups
    ax1_2 = ax.twinx()
    ax1_2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1_2.set_ylabel('Number of groups')
    ax1_2.errorbar(x_data, mean_num_groups, yerr=std_num_groups,
                   color='tab:green', label='Groups')
    ax1_2.fill_between(x_data, mean_num_groups - std_num_groups, mean_num_groups + std_num_groups,
                       color='tab:green', alpha=0.3)
    ax1_2.set_yscale('log')
    ax1_2.yaxis.label.set_color('tab:green')
    ax1_2.tick_params(axis='y', colors='tab:green')

    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 1),
              bbox_transform=ax.transAxes)

    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1_2.legend(loc='upper right', bbox_to_anchor=(
        1, 1), bbox_transform=ax.transAxes)

    # fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)

    return ax


"""
##########   x - y plots
"""


def place_legend(ax, label_arr=[], **kwargs):
    if label_arr is not None:
        loc = kwargs.pop('loc', 'upper right')
        fsize = kwargs.pop('fontsize', MEDIUM_SIZE)
        if loc != 'outside':
            leg = ax.legend(bbox_to_anchor=None, loc=loc,
                            fancybox=True,
                            shadow=False,
                            ncol=1,
                            framealpha=.8,
                            frameon=True,
                            fontsize=fsize,
                            )
            leg.get_frame().set_linewidth(0.0)
        else:
            ax.legend(bbox_to_anchor=(.95, 1), loc='upper left',
                      fancybox=True, shadow=False,
                      ncol=1,
                      framealpha=0,
                      frameon=True)
    return ax


def plot_xy(x_arr, y_arr,
            x_err_arr=[],
            y_err_arr=[],
            label_arr=None,
            lw_arr=None,
            mk_arr=None,
            ls_arr=None,
            color_arr=None,
            lcmap=None,
            ax=None,
            all_x=False,
            norm=False,
            stdize=False,
            ts_axis=False,
            kde=False,
            **kwargs):
    reload(sut)
    reload(gut)
    if ax is None:
        fig_size = kwargs.pop('fig_size', (8, 5))
        fig, ax = plt.subplots(figsize=(fig_size), nrows=1, ncols=1)
    if ts_axis:
        ax = prepare_ts_x_axis(ax, dates=np.array(x_arr[0]),
                               **kwargs)
    else:
        ax = prepare_axis(ax, **kwargs)

    num_items = len(y_arr) if len(y_arr) >= len(x_arr) else len(x_arr)
    if lcmap is not None:
        evenly_spaced_interval = np.linspace(0, 1, num_items)
        lcmap = plt.get_cmap(lcmap)
        ccolors = [lcmap(x) for x in evenly_spaced_interval]
    for idx in range(num_items):
        x = x_arr[idx] if len(x_arr) > 1 else x_arr[0]
        y = y_arr[idx] if len(y_arr) > 1 else y_arr[0]
        if norm is True:
            y = sut.normalize(y)
        if stdize is True:
            y = sut.standardize(y)
            if len(y_err_arr) > 0:
                y_err_arr[idx] = sut.normalize(y_err_arr[idx])
        lw = None if lw_arr is None else lw_arr[idx] if len(
            lw_arr) > 1 else lw_arr[0]
        mk = "" if mk_arr is None else mk_arr[idx] if len(
            mk_arr) > 1 else mk_arr[0]
        ls = "-" if ls_arr is None else ls_arr[idx] if len(
            ls_arr) > 1 else ls_arr[0]
        label = label_arr[idx] if label_arr is not None else None
        if lcmap is None:
            c = colors[idx] if color_arr is None else color_arr[idx]
        else:
            c = ccolors[idx]
        if ts_axis:
            x_0 = np.array(x[0], dtype='datetime64[D]')
            x_end = np.array(x[-1], dtype='datetime64[D]') + \
                np.timedelta64(int(1), 'D')
            x_ts = np.arange(x_0, x_end, dtype='datetime64[D]')
            y_ids = np.nonzero(
                np.in1d(x_ts, np.array(x, dtype='datetime64[D]')))[0]

            y_ts = np.empty(x_ts.shape)
            y_ts[:] = np.nan
            y_ts[y_ids] = y
            ax.plot(x_ts, y_ts, label=label, lw=lw, marker=mk, ls=ls, color=c)
        else:
            if kde:
                levels = kwargs.pop('levels', 10)
                cmap = kwargs.pop('cmap', 'viridis')
                cbar = kwargs.pop('cbar', True)
                sns.kdeplot(x=x,
                            y=y,
                            ax=ax,
                            fill=True,
                            levels=levels,
                            label=label,
                            cmap=cmap,
                            cbar=cbar)

            elif len(x_err_arr) == 0 and len(y_err_arr) == 0:
                ax.plot(x, y, label=label, lw=lw, marker=mk, ls=ls, color=c)
            else:
                print('Errorbars')
                y_err = y_err_arr[idx] if len(y_err_arr) > 0 else None
                x_err = x_err_arr[idx] if len(x_err_arr) > 0 else None

                ax.plot(x, y, label=label, lw=lw, marker=mk, ls=ls, color=c)
                ax.fill_between(x,
                                np.array(y - y_err/2, dtype=float),
                                np.array(y + y_err/2, dtype=float),
                                color=c, alpha=0.5)

                # ax.errorbar(x, y, xerr=x_err, yerr=y_err,
                #             label=label, lw=lw, marker=mk, ls=ls, color=c,
                #             capsize=2)

    if lcmap is not None:
        norm = mpl.colors.Normalize(
            vmin=evenly_spaced_interval.min(), vmax=evenly_spaced_interval.max())
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
        cmap.set_array([])
        fig.colorbar(cmap, ticks=evenly_spaced_interval)

    if all_x is True:
        ax.set_xticks(x_arr[0])
    sci = kwargs.pop('sci', None)
    if sci is not None:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(sci, sci))

    ax = place_legend(ax, label_arr, **kwargs)

    return {'ax': ax}


def plot_hist(data,
              ax=None,
              label_arr=None,
              log=False,
              ylog=False,
              color_arr=None,
              **kwargs):
    reload(sut)
    if ax is None:
        figsize = kwargs.pop('figsize', (6, 4))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax = prepare_axis(ax, log=log, **kwargs)

    if ylog is True:
        ax.set_yscale('log')
    density = kwargs.pop('density', True)
    bar = kwargs.pop('bar', True)
    nbins = kwargs.pop('nbins', None)
    bw = kwargs.pop('bw', None)
    if len(np.shape(data)) > 1:
        data_tmp = np.concatenate(data, axis=0)
        data_arr = data
    else:
        if isinstance(data[0], (list, np.ndarray)):
            data_tmp = data[0]
            data_arr = data
        else:
            data_tmp = data
            data_arr = [data]
    if nbins is None and bw is None:
        nbins = sut.__doane(data_tmp)
    for idx, arr in enumerate(data_arr):
        if log:
            hc, bc, be = sut.loghist(arr, nbins=nbins,
                                     density=density)
        else:
            hc, bc, be = sut.hist(arr,
                                  nbins=nbins,
                                  bw=bw,
                                  min_bw=ax.get_xlim()[0],
                                  max_bw=ax.get_xlim()[1],
                                  density=density)

        label = label_arr[idx] if label_arr is not None else None
        c = colors[idx] if color_arr is None else color_arr[idx]
        if bar:
            width = (bc[1]-bc[0]) * 0.4
            x_pos = bc + width*idx
            ax.bar(x_pos, hc, yerr=None,
                   ecolor=c, capsize=2, width=width,
                   color=c, label=label)
        else:
            ax.plot(bc, hc, 'x',
                    color=c, label=label)

    ax = place_legend(ax, label_arr, **kwargs)

    sci = kwargs.pop('sci', None)
    if sci is not None:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(sci, sci))
    return ax


def prepare_axis(ax, log=False, **kwargs):
    """Prepares an axis for any type x to y plot

    Args:
        ax (ax-object): axis object from matplotlib

    Returns:
        ax: ax matplotlib object
    """
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction='out', length=SMALL_SIZE/2, width=1, colors='k',
                   grid_alpha=0.5)
    ylabel = kwargs.pop('ylabel', None)
    xlabel = kwargs.pop('xlabel', None)
    xpos = kwargs.pop('xlabel_pos', None)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xpos is not None:
        if xpos == 'right':
            ax.xaxis.set_label_coords(1., -0.2)
        else:
            raise ValueError(f'{xpos} does not exist.')
    xlim = kwargs.pop('xlim', None)
    ylim = kwargs.pop('ylim', None)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    rot = kwargs.pop('rot', 0)
    ax.tick_params(axis='x', labelrotation=rot)

    title = kwargs.pop('title', None)
    ax.set_title(title)

    ylog = kwargs.pop('ylog', False)
    if ylog is True:
        ax.set_yscale('log')

    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')

        x_major = matplotlib.ticker.LogLocator(base=10.0, numticks=10)
        ax.xaxis.set_major_locator(x_major)
        x_minor = matplotlib.ticker.LogLocator(
            base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    unset_ticks = kwargs.pop('unset_ticks', False)
    if unset_ticks:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    return ax


def prepare_ts_x_axis(ax, dates, **kwargs):

    ax = prepare_axis(ax, **kwargs)
    ax.tick_params(axis='x', labelrotation=45)
    # Text in the x axis will be displayed in 'YYYY' format.
    fmt_form = mdates.DateFormatter('%m-%Y')
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_form))
    fmt_loc = mdates.YearLocator()
    ax.xaxis.set_major_locator(fmt_loc)
    ax.tick_params(direction='out', length=SMALL_SIZE/2, width=1, colors='k',
                   grid_alpha=0.5)

    return ax


#  ################# Histogramms  #################


def plot_cnt_occ_ensemble(ds, mean_cnt_arr, std_cnt_arr=None, savepath=None, label_arr=None,
                          polar=False, **kwargs):
    figsize = (10, 6)
    if polar is True:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={
                               'projection': 'polar'})
        ax.margins(y=0)
        x_pos = np.deg2rad(np.linspace(0, 360, 13))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ds.months+[''], )
        ax.set_rlabel_position(60)  # get radial labels away from plotted line
        ax.set_rticks([0., 0.1, 0.2, .3, 0.4])  # Less radial ticks
        # rotate the axis arbitrarily, just replace pi with the angle you want.
        ax.set_theta_offset(np.pi)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        prepare_axis(ax, **kwargs)
        ax.set_xlabel('Month')
        ax.set_ylabel('Relative Frequency')
        x_pos = ds.months

    if std_cnt_arr is None:
        std_cnt_arr = np.zeros_like(mean_cnt_arr)
    if len(mean_cnt_arr) != len(std_cnt_arr):
        raise ValueError(
            f"Mean len {len(mean_cnt_arr)} != Std len {len(std_cnt_arr)}")

    sum_mean_cnt = np.zeros_like(mean_cnt_arr[0])
    for idx in range(len(mean_cnt_arr)):
        mean_cnt = np.array(mean_cnt_arr[idx], dtype=float)
        std_cnt = np.array(std_cnt_arr[idx], dtype=float)
        if polar is True:
            mean_cnt = np.append(mean_cnt, np.array([mean_cnt[0]]), axis=0)
            std_cnt = np.append(std_cnt, np.array([std_cnt[0]]), axis=0)
        if label_arr is None:
            label = None
        else:
            label = label_arr[idx]

        width = 1/len(mean_cnt_arr)-0.1
        x_pos = np.arange(len(mean_cnt)) + width*idx
        ax.bar(x_pos, mean_cnt, yerr=(std_cnt), ecolor=colors[idx], capsize=10, width=width,
               color=colors[idx], label=label)
        sum_mean_cnt += mean_cnt
        # print(sum_mean_cnt)
    # off_set = len(mean_cnt_arr)
    ax.set_xticks(x_pos)  # + width/off_set
    ax.set_xticklabels(ds.months)
    ax.grid(True)

    if label_arr is not None:
        place_legend(ax, label_arr, **kwargs)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')

    return ax


#  ########################  Null Model #########################

def plot_null_model(arr,
                    title='Null Model for Event Synchronization', ):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax = prepare_axis(ax)
    ax.set_title(title)
    im = ax.imshow(arr,
                   interpolation='None',
                   cmap='coolwarm',
                   origin='lower')
    # ax.set_aspect('equal')
    ax.set_xlabel('number of events j')
    ax.set_ylabel('number of events i')
    make_colorbar(ax, im, label='# Sync Events')
    return ax

#  ########################  Maps ###############################


def set_grid(ax, alpha=0.5):
    gl = ax.gridlines(draw_labels=True,
                      xlocs=np.arange(-180, 181, 60),
                      ylocs=range(-90, 90, 30),
                      crs=ccrs.PlateCarree(),
                      x_inline=False, y_inline=False,
                      alpha=alpha)
    # gl = ax.gridlines(draw_labels=True, dms=True,
    #                   x_inline=False, y_inline=False, )
    # gl.rotate_labels = False
    gl.left_labels = True
    gl.right_labels = False
    # gl.right_labels=True
    gl.bottom_labels = False
    # gl.ylabel_style = {'rotation': -0, 'color': 'black'}
    gl.top_labels = True
    return ax


def make_colorbar_descrete(ax, CS,
                           vmin=None,
                           vmax=None,
                           label=None, **kwargs):
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig = ax.get_figure()
    if vmin is None:
        vmin = CS.get_clim()[0]
    if vmax is None:
        vmax = CS.get_clim()[1]
    m = ScalarMappable(cmap=CS.get_cmap())
    m.set_array(CS.get_array())
    m.set_clim(CS.get_clim())
    step = CS.levels[1] - CS.levels[0]
    num_levels = len(CS.levels)
    cliplower = CS.zmin < vmin
    clipupper = CS.zmax > vmax
    noextend = 'extend' in kwargs.keys() and kwargs['extend'] == 'neither'
    # set the colorbar boundaries

    boundaries = np.linspace(vmin - (cliplower and noextend)*step,
                             vmax + (clipupper and noextend)*(step+1),
                             num_levels, endpoint=True)
    boundaries_map = copy.deepcopy(boundaries)
    if cliplower:
        boundaries_map = np.insert(
            boundaries_map, 0, vmin-abs(vmin/100), axis=0)
    if clipupper:
        boundaries_map = np.insert(boundaries_map, len(
            boundaries_map), vmax+abs(vmax/100), axis=0)
    # print(boundaries_map, boundaries)
    kwargs['boundaries'] = boundaries_map
    # if the z-values are outside the colorbar range, add extend marker(s)
    # This behavior can be disabled by providing extend='neither' to the function call
    if not('extend' in kwargs.keys()) or kwargs['extend'] in ['min', 'max']:
        extend_min = cliplower or (
            'extend' in kwargs.keys() and kwargs['extend'] == 'min')
        extend_max = clipupper or (
            'extend' in kwargs.keys() and kwargs['extend'] == 'max')
        if extend_min and extend_max:
            kwargs['extend'] = 'both'
        elif extend_min:
            kwargs['extend'] = 'min'
        elif extend_max:
            kwargs['extend'] = 'max'
    divider = make_axes_locatable(ax)
    orientation = kwargs.pop('orientation', 'horizontal')
    if orientation == 'vertical':
        loc = 'right'
        pad = kwargs.pop('pad', '3%')
    elif orientation == 'horizontal':
        loc = 'bottom'
        pad = kwargs.pop('pad', '8%')
    cax = divider.append_axes(
        loc, '5%', pad=pad, axes_class=mpl.pyplot.Axes)
    tick_step = int(kwargs.pop('tick_step', 1))
    round_dec = kwargs.pop('round_dec', None)
    ticks = boundaries[::tick_step] if round_dec is None else np.round_(boundaries[::tick_step],
                                                                        round_dec)

    sci = kwargs.pop('sci', None)
    if sci is not None:
        fmt = mpl.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((sci, sci))
        cbar = fig.colorbar(m, cax=cax,
                            orientation=orientation,
                            label=label,
                            format=fmt,
                            ticks=ticks,
                            **kwargs)
    else:
        cbar = fig.colorbar(m, cax=cax,
                            orientation=orientation,
                            label=label,
                            ticks=ticks,
                            **kwargs
                            )
        cbar.set_ticklabels(ticks)

    return cbar


def make_colorbar(ax, cmap, **kwargs):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    orientation = kwargs.pop('orientation', 'vertical')

    if orientation == 'vertical':
        loc = 'right'
        cax = divider.append_axes(
            loc, size='5%', pad='3%',
            axes_class=mpl.pyplot.Axes)
    elif orientation == 'horizontal':
        loc = 'bottom'
        cax = divider.append_axes(
            loc, size='5%',
            pad=0.2,
            axes_class=mpl.pyplot.Axes)

    label = kwargs.pop('label', None)
    ticks = kwargs.pop('ticks', None)
    extend = kwargs.pop('extend', 'neither')

    return ax.get_figure().colorbar(cmap, cax=cax, orientation=orientation,
                                    label=label, shrink=0.8, ticks=ticks,
                                    extend=extend)


def discrete_cmap(vmin, vmax, colormap=None, num_ticks=None, shift_ticks=False):
    import matplotlib as mpl

    # colormap=pt.Spectral_11.mpl_colormap
    if colormap is None:
        # import palettable.colorbrewer.diverging as pt
        import palettable.colorbrewer.qualitative as pt
        colormap = pt.Paired_12.mpl_colormap
    cmap = plt.get_cmap(colormap)

    normticks = discrete_norm_ticks(
        vmin, vmax, num_ticks=num_ticks, shift_ticks=shift_ticks)

    norm = mpl.colors.BoundaryNorm(normticks, cmap.N)
    return cmap, norm


def discrete_norm_ticks(vmin, vmax, shift_ticks=False, num_ticks=None):
    if vmin is None or vmax is None:
        return None
    if num_ticks is None:
        num_ticks = 10

    if shift_ticks is True:
        # +1.1 to account for start and end
        normticks = np.arange(vmin, vmax+1.1, dtype=int)
    else:
        normticks = np.linspace(vmin, vmax, num_ticks+1)

    return normticks


def set_extent(ds, ax, central_longitude=0, **kwargs):
    import math
    projection = ccrs.PlateCarree(central_longitude=central_longitude)

    extent = kwargs.pop('extent', None)
    if extent is None:
        ax.set_global()
    else:
        print(extent)
        lon_range, lat_range = extent
        min_ext_lon = math.floor(np.min(lon_range))
        max_ext_lon = math.floor(np.max(lon_range))
        min_ext_lat = math.floor(np.min(lat_range))
        max_ext_lat = math.ceil(np.max(lat_range))

        # min_ext_lon = math.floor(np.min(ds.ds.coords['lon']))
        # max_ext_lon = math.floor(np.max(ds.ds.coords['lon']))
        # min_ext_lat = math.floor(np.min(ds.ds.coords['lat']))
        # max_ext_lat = math.ceil(np.max(ds.ds.coords['lat']))

        if abs(min_ext_lon) > 179 or abs(max_ext_lon) > 179:
            min_ext_lon = -179.9
            max_ext_lon = 179.9
        if abs(min_ext_lat) > 89 or abs(max_ext_lat) > 89:
            min_ext_lat = -89.9
            max_ext_lat = 89.9
        ax.set_extent([min_ext_lon,
                       max_ext_lon,
                       min_ext_lat,
                       max_ext_lat],
                      crs=projection)
    # print(
    #     [min_ext_lon,
    #      max_ext_lon,
    #      min_ext_lat,
    #      max_ext_lat]
    # )

    return ax


def create_map(ds=None, ax=None, fig=None,
               projection="EqualEarth", central_longitude=0, alpha=1,
               plt_grid=False, **kwargs):
    proj = get_projection(projection, central_longitude)

    # create figure
    if ax is None:
        fig_size = kwargs.pop('fig_size', (10, 7))
        fig, ax = plt.subplots(figsize=(fig_size))
        ax = plt.axes(projection=proj)

    ax = set_extent(ds=ds,
                    ax=ax,
                    central_longitude=central_longitude,
                    **kwargs,
                    )

    # axes properties
    coast_color = kwargs.pop('coast_color', 'k')
    ax.coastlines(alpha=alpha, color=coast_color)
    ax.add_feature(ctp.feature.BORDERS, linestyle=':', color='grey',
                   alpha=alpha)
    land_ocean = kwargs.pop('land_ocean', False)
    if land_ocean:
        ax.add_feature(ctp.feature.OCEAN, alpha=alpha, zorder=-1)
        ax.add_feature(ctp.feature.LAND, alpha=alpha, zorder=-1)
    if plt_grid is True:
        ax = set_grid(ax, alpha=alpha)

    return ax, fig


def get_projection(projection, central_longitude):
    if projection == 'Mollweide':
        proj = ccrs.Mollweide(central_longitude=central_longitude)
    elif projection == 'EqualEarth':
        proj = ccrs.EqualEarth(central_longitude=central_longitude)
    elif projection == 'Robinson':
        proj = ccrs.Robinson(central_longitude=central_longitude)
    elif projection == 'PlateCarree':
        proj = ccrs.PlateCarree(central_longitude=central_longitude)
    else:
        raise ValueError(
            f'This projection {projection} is not available yet!')

    return proj


def plot_map(ds, dmap,
             fig=None, ax=None,
             intpol=True,
             plot_type='scatter',
             central_longitude=0,
             vmin=None, vmax=None,
             cmap=None, bar=True,
             projection="EqualEarth",
             label=None, title=None,
             significant_mask=False,
             **kwargs):
    """Simple map plotting using xArray.

    Parameters:
    -----------
    dmap: xarray.Dataarray
        Dataarray containing data, and coordinates lat, lon
    plot_type: str
        Plot type, currently supported "scatter" and "colormesh". Default: 'scatter'
    central_longitude: int
        Set central longitude of the plot. Default: 0
    vmin: float
        Lower limit for colorplot. Default: None
    vmax: float
        Upper limit for colorplot. Default: None
    color: str
        Colormap supported by matplotlib. Default: "RdBu"
    bar: bool
        If True colorbar is shown. Default: True
    ax: matplotlib.axes
        Axes object for plotting on. Default: None
    ctp_projection: str
        Cartopy projection type. Default: "Mollweide"
    label: str
        Label of the colorbar. Default: None
    grid_step: float
        Grid step for interpolation on Gaussian grid. Only required for plot_type='colormesh'. Default: 2.5

    Returns:
    --------
    Dictionary including {'ax', 'projection'}
    """
    plt.rcParams['pcolor.shading'] = 'nearest'  # For pcolormesh

    plt_grid = kwargs.pop('plt_grid', False)
    ax, fig = create_map(ds=ds, ax=ax,
                         projection=projection,
                         central_longitude=central_longitude,
                         plt_grid=plt_grid)

    projection = ccrs.PlateCarree()  # nicht: central_longitude=central_longitude!

    # set colormap
    if cmap is not None:
        cmap = plt.get_cmap(cmap)
    kwargs_pl = dict()  # kwargs plot function
    kwargs_cb = dict()  # kwargs colorbar
    if bar == 'discrete':
        normticks = np.arange(0, dmap.max(skipna=True)+2, 1)
        kwargs_pl['norm'] = mpl.colors.BoundaryNorm(normticks, cmap.N)
        kwargs_cb['ticks'] = normticks + 0.5

    mask = xr.where(ds.mask == 0, 1, np.nan)
    # interpolate grid of points to regular grid
    if intpol:
        lon_interp = gut.crange(ds.ds.coords['lon'].min(),
                                ds.ds.coords['lon'].max() + ds.grid_step,
                                ds.grid_step)
        lat_interp = gut.crange(ds.ds.coords['lat'].min(),
                                ds.ds.coords['lat'].max() + ds.grid_step,
                                ds.grid_step)
        lon_mesh, lat_mesh = np.meshgrid(lon_interp, lat_interp)
        new_points = np.array([lon_mesh.flatten(), lat_mesh.flatten()]).T
        origin_points = np.array(
            [ds.ds.coords['lon'], ds.ds.coords['lat']]).T
        # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
        new_values = interp.griddata(origin_points, dmap.data, new_points,
                                     method='nearest')
        mesh_values = new_values.reshape(
            len(lat_interp), len(lon_interp))
        new_values_msk = interp.griddata(origin_points, mask, new_points,
                                         method='nearest')
        mesh_values_mask = new_values_msk.reshape(
            len(lat_interp), len(lon_interp))
    else:
        lon_mesh, lat_mesh = dmap.coords['lon'], dmap.coords['lat']
        mesh_values = dmap
        mesh_values_mask = mask
    # plotting
    color = kwargs.pop('color', None)
    levels = kwargs.pop('levels', 8)
    alpha = kwargs.pop('alpha', 1.0)
    lw = kwargs.pop('lw', 1)
    size = kwargs.pop('size', 1)
    marker = kwargs.pop('marker', 's')
    fillstyle = kwargs.pop('fillstyle', 'full')

    ticks = None
    if plot_type == 'scatter':
        im = ax.scatter(x=ds.ds.coords['lon'], y=ds.ds.coords['lat'],
                        c=dmap.data, vmin=vmin, vmax=vmax, cmap=cmap,
                        transform=projection, alpha=alpha,
                        marker=mpl.markers.MarkerStyle(
                            marker=marker, fillstyle=fillstyle),
                        s=size)
    elif plot_type == 'colormesh':
        # print(lon_mesh.shape, lat_mesh.shape, mesh_values.shape, )
        im = ax.pcolor(
            lon_mesh, lat_mesh,
            mesh_values,
            cmap=cmap, vmin=vmin,
            vmax=vmax, transform=projection,
            shading='auto',
            # **kwargs_pl
        )
    elif plot_type == 'contourf':
        if vmin is not None and vmax is not None:
            levels = np.linspace(vmin, vmax, levels+1, endpoint=True)

        im = ax.contourf(lon_mesh, lat_mesh, mesh_values,
                         levels=levels, vmin=vmin, vmax=vmax,
                         cmap=cmap, transform=projection,
                         extend='both', alpha=alpha
                         )
    elif plot_type == 'contour':
        im = ax.contour(lon_mesh, lat_mesh, mesh_values,
                        levels=levels,
                        # cmap=cmap,
                        transform=projection,
                        colors=color,
                        linewidths=lw  # maybe linewidth=
                        )
    elif plot_type == 'points':
        flat_idx_lst = ds.flatten_array(
            dmap, time=False, check=False)
        flat_idx = np.where(np.abs(flat_idx_lst) > 0.01)[0]
        xp = []
        yp = []
        for idx in flat_idx:
            map_idx = ds.get_map_index(idx)
            xp.append(map_idx['lon'])
            yp.append(map_idx['lat'])

        im = ax.plot(xp, yp, c=color,
                     linewidth=0,
                     markersize=size, marker=marker, fillstyle=fillstyle,
                     transform=projection, alpha=alpha)
    elif plot_type == 'discrete':
        vmin = np.nanmin(dmap)
        vmax = np.nanmax(dmap)
        cmap, norm = discrete_cmap(vmin, vmax, colormap=color, num_ticks=levels,
                                   shift_ticks=True)
        normticks = discrete_norm_ticks(
            vmin, vmax,  num_ticks=levels, shift_ticks=True)
        ticks = normticks[:-1] + 0.5

        im = ax.pcolor(lon_mesh, lat_mesh, mesh_values,
                       vmin=vmin, vmax=vmax,
                       cmap=cmap, transform=projection,
                       norm=norm)
    else:
        raise ValueError(f"Plot type {plot_type} does not exist!")
    # areas which are dotted are mask
    if significant_mask:
        ax.pcolor(
            lon_mesh, lat_mesh, mesh_values_mask,  hatch='///', alpha=0.,
            transform=projection)

    if bar:
        label = dmap.name if label is None else label

        if plot_type == 'contourf':
            make_colorbar_descrete(ax, im, label=label,
                                   vmin=vmin, vmax=vmax,
                                   **kwargs)

        else:
            cbar = make_colorbar(ax, im, label=label,
                                 **kwargs)
            if plot_type == 'discrete':
                cbar.set_ticks(ticks)
                cbar.ax.set_xticklabels(ticks, rotation=45)
                cbar.set_ticklabels(normticks)

    if title is not None:
        # y_title = 1.1
        ax.set_title(title)

    return {"ax": ax, 'fig': fig, "projection": projection, "im": im}


def plot_edges(ds, edges, weights=None,
               central_longitude=0,
               fig=None, ax=None,
               projection="EqualEarth",
               plot_points=False,
               vmin=None, vmax=None,
               **kwargs):

    plt_grid = kwargs.pop('plt_grid', False)
    ax, fig = create_map(ds=ds, ax=ax,
                         projection=projection,
                         central_longitude=central_longitude,
                         plt_grid=plt_grid)

    counter = 0
    lw = kwargs.pop('lw', 1)
    alpha = kwargs.pop('alpha', 1)
    color = kwargs.pop('color', 'k')

    if vmin is None:
        vmin = np.min(weights)
    if vmax is None:
        vmax = np.max(weights)

    if weights is not None:
        cmap = plt.get_cmap(color)
        norm = mpl.colors.Normalize(vmin=vmin,
                                    vmax=vmax)

    for i, (u, v) in enumerate(edges):
        counter += 1
        map_idx_u = ds.get_map_index(u)
        map_idx_v = ds.get_map_index(v)
        lon_u = map_idx_u['lon']
        lat_u = map_idx_u['lat']
        lon_v = map_idx_v['lon']
        lat_v = map_idx_v['lat']
        if plot_points is True:
            ax.scatter([lon_u, lon_v], [lat_u, lat_v],
                       c='k', transform=ccrs.PlateCarree(), s=4)
        if weights is not None:
            c = cmap(norm(weights[i]))
        else:
            c = color

        ax.plot([lon_u, lon_v], [lat_u, lat_v],
                c=c,
                linewidth=lw,
                alpha=alpha,
                transform=ccrs.Geodetic(),
                zorder=-1)  # zorder = -1 to always set at the background

    print(f"number of edges: {counter}")
    return {"ax": ax, 'fig': fig, "projection": projection}


def plot_wind_field(ax, u, v,
                    lons=None,
                    lats=None,
                    lonsteps=4,
                    latsteps=4,
                    key_loc=(0.95, 0.05),
                    **kwargs):
    if lons is None:
        lons = u.coords['lon']
    if lats is None:
        lats = u.coords['lat']
    u_dat = u.data[::latsteps, ::lonsteps]
    v_dat = v.data[::latsteps, ::lonsteps]

    lw = kwargs.pop('lw', 1)
    # headwidth = kwargs.pop('headwidth', 1)
    # width = kwargs.pop('width', 0.005)

    Q = ax.quiver(lons[::lonsteps], lats[::latsteps],
                  u=u_dat, v=v_dat,
                  pivot='middle',
                  transform=ccrs.PlateCarree(),
                  linewidths=lw,
                  )
    ax.quiverkey(Q, key_loc[0], key_loc[1], 1, r'$1 \frac{m}{s}$',
                 labelpos='W',
                 coordinates='axes')


def create_multi_map_plot(nrows, ncols,
                          projection='EqualEarth',
                          **kwargs):
    central_longitude = kwargs.pop('central_longitude', 0)
    proj = get_projection(projection=projection,
                          central_longitude=central_longitude)
    figsize = kwargs.pop('figsize', (9, 6))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols,
                                                               figsize[1]*nrows),
                            subplot_kw=dict(projection=proj),
                            )

    for ax in axs.flatten():
        ax.set_global()

    return {"ax": axs.flatten(),
            "fig": fig,
            "projection": projection}


def create_multi_plot(nrows, ncols,
                      **kwargs):

    figsize = kwargs.pop('figsize', (9, 6))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols,
                                                               figsize[1]*nrows),
                            **kwargs
                            )

    return {"ax": axs.flatten(),
            "fig": fig,
            }


def plt_text_map(ax, lon_pos, lat_pos, text, color='k'):
    ax.text(lon_pos, lat_pos, text,
            horizontalalignment='center',
            transform=ccrs.Geodetic(),
            color=color)

    return ax


def plt_text(ax, lon_pos, lat_pos, text, color='k'):
    ax.text(lon_pos, lat_pos, text,
            horizontalalignment='right',
            transform=ax.transAxes,
            color=color)

    return ax


def mk_plot_dir(savepath):
    if os.path.exists(savepath):
        return
    else:
        dirname = os.path.dirname(savepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        return


def enumerate_subplots(axs, pos_x=-0.05, pos_y=1.01, fontsize=20):
    """Adds letters to subplots of a figure.

    Args:
        axs (list): List of plt.axes.
        pos_x (float, optional): x position of label. Defaults to 0.02.
        pos_y (float, optional): y position of label. Defaults to 0.85.
        fontsize (int, optional): Defaults to 18.

    Returns:
        axs (list): List of plt.axes.
    """

    if type(pos_x) == float:
        pos_x = [pos_x] * len(axs.flatten())
    if type(pos_y) == float:
        pos_y = [pos_y] * len(axs.flatten())

    for n, ax in enumerate(axs.flatten()):
        ax.text(pos_x[n], pos_y[n], f"{string.ascii_lowercase[n]}.",
                transform=ax.transAxes,
                size=fontsize, weight='bold')
    plt.tight_layout()

    return axs



def plot_corr_matrix(mat_corr, pick_x=None, pick_y=None,
                     label_x=None, label_y=None,
                     ax=None, vmin=-1, vmax=1,
                     color='BrBG', bar_title='correlation'):
    """Plot correlation matrix.

    Args:
        mat_corr ([type]): [description]
        pick_x ([type], optional): [description]. Defaults to None.
        pick_y ([type], optional): [description]. Defaults to None.
        label_x ([type], optional): [description]. Defaults to None.
        label_y ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.
        vmin (int, optional): [description]. Defaults to -1.
        vmax (int, optional): [description]. Defaults to 1.
        color (str, optional): [description]. Defaults to 'BrBG'.
        bar_title (str, optional): [description]. Defaults to 'correlation'.

    Returns:
        im (plt.imshow): [description]
    """
    if ax is None:
        fig, ax = plt.subplots()

    if pick_y is not None and pick_x is not None:
        corr = mat_corr[pick_x, :].copy()
        corr = corr[:, pick_y]
    elif pick_x is not None:
        corr = mat_corr[pick_x, :]
    elif pick_y is not None:
        corr = mat_corr[:, pick_y]
    else:
        corr = mat_corr

    cmap = plt.get_cmap(color)
    im = ax.imshow(corr, vmin=vmin, vmax=vmax, aspect='auto', cmap=cmap)

    cbar = plt.colorbar(im, extend='both', orientation='horizontal',
                        label=bar_title, shrink=1.0, ax=ax)

    if label_x is not None:
        ax.set_xticks(np.arange(0, len(label_x)))
        ax.set_xticklabels(label_x)
    if label_y is not None:
        ax.set_yticks(np.arange(0, len(label_y)))
        ax.set_yticklabels(label_y)

    return im


def plot_rectangle(ax, lon_range, lat_range, **kwargs):
    """Plots a rectangle on a cartopy map

    Args:
        ax (geoaxis): Axis of cartopy object
        lon_range (list): list of min and max longitude
        lat_range (list): list of min and max lat

    Returns:
        geoaxis: axis with rectangle plotted
    """
    import matplotlib.patches as mpatches
    from shapely.geometry.polygon import LinearRing
    shortest = kwargs.pop('shortest', True)
    if (max(lon_range) - min(lon_range) <= 180) or shortest is False:
        cl = 0
        lons = [max(lon_range), min(lon_range), min(lon_range), max(lon_range)]
    else:
        cl = 180
        lons = [max(lon_range)-180,
                180+min(lon_range),
                180+min(lon_range),
                max(lon_range)-180]
    lats = [min(lat_range), min(lat_range), max(lat_range), max(lat_range)]

    ring = LinearRing(list(zip(lons, lats)))
    lw = kwargs.pop('lw', 1)
    color = kwargs.pop('color', 'k')
    fill = kwargs.pop('fill', False)
    facecolor = color if fill else 'none'
    ax.add_geometries([ring], ccrs.PlateCarree(central_longitude=cl),
                      facecolor=facecolor,
                      edgecolor=color,
                      linewidth=lw,
                      zorder=10)

    return ax
