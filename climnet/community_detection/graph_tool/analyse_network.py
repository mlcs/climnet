#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 08:53:08 2020
Class for network of rainfall events
@author: Felix Strnad
"""
# %%
from climnet.community_detection.graph_tool.compare_runs import Compare_Runs
import geoutils.tsa.event_synchronization as es
import sys
import os
import numpy as np
import pandas as pd

import xarray as xr
import matplotlib.pyplot as plt
from importlib import reload
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH + "/../")  # Adds higher directory

"""Class of monsoon regions, containing the
monsoon definitions, the node ids and the regional monsoons."""


class Analyse_Network(Compare_Runs):
    """ Dataset for analysing the network output and
    comparing multiple cluster runs from graph tool.

    Args:
    ----------
    nc_file: str
        filename
    var_name: str
        Variable name of interest
    """

    def __init__(self,
                 var_name=None,
                 data_nc=None,
                 load_nc=None,
                 time_range=None,
                 month_range=None,
                 lon_range=[-180, 180],
                 lat_range=[-90, 90],
                 grid_step=1,
                 grid_type='gaussian',
                 lsm=False,
                 large_ds=False,
                 abs_th_wang=2, abs_th_ee=50, rel_th=0.55,
                 full_year=False,
                 can=False,
                 **kwargs):
        super().__init__(data_nc=data_nc, load_nc=load_nc,
                         var_name=var_name, time_range=time_range,
                         lon_range=lon_range, lat_range=lat_range,
                         grid_step=grid_step, grid_type=grid_type,
                         month_range=month_range, large_ds=large_ds,
                         lsm=lsm,
                         abs_th_wang=abs_th_wang, abs_th_ee=abs_th_ee, rel_th=rel_th,
                         full_year=full_year,
                         **kwargs
                         )
        self.can = can
        # Compute Anomalies if needed
        if self.can is True:
            print('Compute additionally Anomalies...!')
            if 'an' not in self.vars:
                self.ds['an'] = self.compute_anomalies(self.ds[self.var_name])

    def get_density_map_loc_arr(self, loc_arr, num_runs=30, den_th=0.8,  abs_th=4, rel_th=0,
                                plot=False, projection='Mollweide', savepath=None, ax=None, title=None, graph_folder=None,
                                graph_file=None, num_ticks=10, extent=None, plt_grid=True):

        mean_cluster, std_cluster = self.get_density_cluster(loc_arr=loc_arr, num_runs=num_runs,
                                                             abs_th=abs_th, graph_folder=graph_folder,
                                                             graph_file=graph_file)

        eps = 0.2
        mean_cluster = np.where(mean_cluster > eps, mean_cluster, np.nan)
        mi_indices = np.where(mean_cluster > den_th)[0]
        if plot is True:

            cluster_map = self.get_map(self.flat_idx_array(mi_indices))

            ax = self.plot_map(dmap=self.get_map(mean_cluster), ax=ax, label='Membership Likelihood',
                               title=title, color='Reds', extend='neither', num_ticks=num_ticks, dcbar=True,
                               vmin=0, vmax=None, projection=projection, plt_grid=plt_grid, extent=extent)
            # self.plot_contour_map(cluster_map, ax=ax, fill_out=False,
            #             cmap='Greys', vmin=0, vmax=1, projection=projection )#
            self.plot_map(cluster_map, ax=ax, fill_out=False, contour=True,
                          color='blue', vmin=0, vmax=1, projection=projection, bar=False, extent=extent)
            if savepath is not None:
                plt.savefig(savepath, bbox_inches='tight')

        return mi_indices

    def get_c_indices_m_region(self, c_indices, mname):
        mi_map = self.get_map(self.flat_idx_array(c_indices))
        m_ts_dict = dict()
        self.es_time_series = self.flatten_array(
            dataarray=self.data_evs, check=False).T
        monsoon_dict = self.monsoon_dictionary
        this_indices = self.get_idx_monsoon(monsoon_dict[mname], mi_map)
        this_map = self.get_map(self.flat_idx_array(this_indices))
        m_ts_dict = {
            'ts': self.es_time_series[this_indices], 'ids': this_indices, 'map': this_map}

        return m_ts_dict

    def count_ev_area(self, es_r, lp_filter=True):
        num_tp = es_r.shape[1]
        tp_es = es.prepare_es_input_data(es_r)
        ts_e = np.zeros(num_tp)
        for evs in tp_es:
            ts_e[evs] += 1

        if lp_filter is True:
            cutoff = 4  # Attention this changes dramatically the smearing out of local extreme!
            fcutoff = .95 * 1. / cutoff
            order = 8
            fs = 1
            rp = .05
            ts_e = es.cheby_lowpass_filter(ts_e, fcutoff, fs, order, rp)

        return ts_e

    def rolling_mean_ts(self, ts, rm=10):
        ts = pd.Series(ts)
        ts_rm = ts.rolling(rm).mean()
        ts_rm[np.isnan(ts_rm)] = 0
        ts_rm = np.array(ts_rm)
        # ts_rm=np.roll(np.array(ts_rm), rm)

        return ts_rm

    def get_sync_times_2mregions(self, es_r1, es_r2, taumax=10, q=0.99, lp_filter=False, rm=None):
        """
        """

        t, t12, t21 = es.es_reg(es_r1, es_r2, taumax)

        cutoff = 4
        fcutoff = .95 * 1. / cutoff
        order = 8
        fs = 1
        rp = .05
        if lp_filter:
            ts12 = es.cheby_lowpass_filter(t12, fcutoff, fs, order, rp)
            ts21 = es.cheby_lowpass_filter(t21, fcutoff, fs, order, rp)
        else:
            ts12 = t12
            ts21 = t21
        if rm is not None:
            ts12 = self.rolling_mean_ts(ts12, rm=rm)
            ts21 = self.rolling_mean_ts(ts21, rm=rm)

        sync_times12 = es.get_locmax_of_score(ts12, q=q)
        sync_times21 = es.get_locmax_of_score(ts21, q=q)

        return sync_times12, sync_times21, ts12, ts21

    def get_sync_times_monsoon(self, sel_m, m_ts_dict,  taumax,
                               q=0.99,
                               lp_filter=True,
                               rm=None):
        """
        Get sync times for two regions within sel_m
        """
        times = self.data_evs['time']
        es_r1 = m_ts_dict[sel_m[0]]['ts']
        es_r2 = m_ts_dict[sel_m[1]]['ts']

        sync_times12, sync_times21, ts12, ts21 = self.get_sync_times_2mregions(es_r1, es_r2, taumax=taumax, q=q,
                                                                               lp_filter=lp_filter, rm=rm)

        return ts12, ts21, sync_times12, sync_times21

    def get_sync_times(self, es_r1, es_r2,
                       taumax, q=0.99, lp_filter=False, rm=None
                       ):
        """
        Get sync times for two regions within sel_m
        -----
        Return
        sync_times12: int array of synchronous times in from 1 to 2
        sync_times21: int array of synchronous times in from 2 to 1
        ts12 : time series of 1 with events in 2
        ts21 : time series of 2 with events in 1
        """
        sync_times12, sync_times21, ts12, ts21 = self.get_sync_times_2mregions(es_r1, es_r2, taumax=taumax, q=q,
                                                                               lp_filter=lp_filter, rm=rm)

        return sync_times12, sync_times21, ts12, ts21

    def count_tps_occ(self, tps_arr, plot=True, label_arr=None, savepath=None):
        if plot is True:
            fig, ax = plt.subplots(figsize=(7, 4))

        res_c_occ = np.zeros((len(tps_arr), len(self.months)))
        for idx, tps in enumerate(tps_arr):
            tot_num = len(tps)
            count_occ = []

            for midx, month in enumerate(self.months):
                tp_month = self.get_month_range_data(
                    tps, start_month=month, end_month=month)
                counts = tp_month.shape[0]
                count_occ.append(counts/tot_num)
                res_c_occ[idx][midx] = counts/tot_num

            if plot is True:
                if label_arr is not None:
                    label = label_arr[idx]
                else:
                    label = None
                ax.plot(self.months, count_occ, label=label)
                ax.set_xlabel('Month')
                ax.set_ylabel('Relative Frequency')
                if label_arr is not None:
                    ax.legend(bbox_to_anchor=(1, 1), loc='upper left',
                              fancybox=True, shadow=True, ncol=1)
                if savepath is not None:
                    fig.savefig(savepath, bbox_inches='tight')

        return res_c_occ

    def count_tps_per_year(self, tps_arr, plot=True, label_arr=None, savepath=None):
        from matplotlib.ticker import MaxNLocator

        if plot is True:
            fig, ax = plt.subplots(figsize=(7, 4))

        times = self.dataarray['time']
        all_years = np.arange(
            int(times[0].time.dt.year), int(times[-1].time.dt.year), 1)

        for idx, tps in enumerate(tps_arr):
            year_arr = tps.time.dt.year.data
            years, y_counts = np.unique(year_arr, return_counts=True)
            all_year_counts = np.zeros_like(all_years)
            for yidx, year in enumerate(all_years):
                if year in years:
                    all_year_counts[yidx] = y_counts[np.where(years == year)[
                        0][0]]

            if plot is True:
                if label_arr is not None:
                    label = label_arr
                else:
                    label = None
                ax.plot(all_years, all_year_counts, label=label[idx])
                ax.set_xlabel('Year')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_ylabel('Number of Occurence')
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                if label_arr is not None:
                    ax.legend(bbox_to_anchor=(1, 1), loc='upper left',
                              fancybox=True, shadow=True, ncol=1)
                if savepath is not None:
                    fig.savefig(savepath,  bbox_inches='tight')

        return None

    def count_tps_occ_day(self, tps, plot=True, label_arr=None, savepath=None, sm='Jan', em='Dec'):
        """
        TODO unfinished needs more exact extraction of days!
        """
        if plot is True:
            fig, ax = plt.subplots(figsize=(7, 4))

        tot_num = len(tps)
        count_occ = []

        day_in_month_arr = np.zeros(380)
        day_idx = 0
        for midx, month in enumerate(self.months):
            tp_month = self.get_month_range_data(
                tps, start_month=month, end_month=month)

            if len(tp_month.data) > 0:
                # print(month, tp_month.data)
                days_in_month = int(tp_month.time.dt.daysinmonth[0])
                for day in range(days_in_month+1):
                    counts = np.count_nonzero(tp_month.time.dt.day.isin(day))
                    day_in_month_arr[day_idx] = counts
                    day_idx += 1
            else:
                day_idx += 30

        if plot is True:
            if label_arr is not None:
                label = label_arr
            else:
                label = None
            ax.plot(np.arange(len(day_in_month_arr)),
                    day_in_month_arr, label=label)
            ax.set_xlabel('Day of the year')
            ax.set_ylabel('Number of Occurence')
            if label_arr is not None:
                ax.legend(bbox_to_anchor=(1, 1), loc='upper left',
                          fancybox=True, shadow=True, ncol=1)
            if savepath is not None:
                fig.savefig(savepath)

        return day_in_month_arr

    def normalize_input(self, a, b):
        a = (a - np.mean(a)) / (np.std(a) * len(a))
        b = (b - np.mean(b)) / (np.std(b))
        return a, b

    def compute_lead_lag_corr(self, ts1, ts2, maxlags=20, savepath=None, title=None):
        Nx = len(ts1)
        if Nx != len(ts2):
            raise ValueError('x and y must be equal length')
        nts1, nts2 = self.normalize_input(ts1, ts2)
        corr = np.correlate(nts1, nts2, mode='full')
        if maxlags >= Nx or maxlags < 1:
            raise ValueError(
                f'maxlags must be None or strictly positive < {Nx}')

        corr_range = corr[Nx - 1 - maxlags:Nx + maxlags]

        fig, ax = plt.subplots(figsize=(8, 5))
        # because last value in arange is not counted
        x_lag = np.arange(-maxlags, maxlags+1)
        ax.plot(x_lag, corr_range)
        ax.grid()
        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('Correlation')
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()
        if savepath is not None:
            fig.savefig(savepath)
        return corr_range
