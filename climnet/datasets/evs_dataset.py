import numpy as np
import xarray as xr
import os
from importlib import reload
import climnet.datasets.dataset as cds
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu
reload(cds)


class EvsDataset(cds.BaseDataset):
    def __init__(
        self,
        data_nc=None,
        load_nc=None,
        var_name=None,
        time_range=None,
        lon_range=[-180, 180],
        lat_range=[-90, 90],
        grid_step=1,
        grid_type="gaussian",
        large_ds=False,
        lsm=False,
        q=0.95,
        min_evs=20,
        min_treshold=1,
        th_eev=15,
        rrevs=False,
        can=False,
        timemean=None,
        month_range=None,
        **kwargs,
    ):

        super().__init__(
            data_nc=data_nc,
            load_nc=load_nc,
            var_name=var_name,
            time_range=time_range,
            lon_range=lon_range,
            lat_range=lat_range,
            grid_step=grid_step,
            grid_type=grid_type,
            large_ds=large_ds,
            timemean=timemean,
            lsm=lsm,
            init_indices=False,  # Indices are later initialized with evs mask
            **kwargs,
        )

        # Event synchronization
        if data_nc is not None or rrevs:

            self.q = q
            self.min_evs = min_evs
            self.min_treshold = min_treshold
            self.th_eev = th_eev
        else:
            self.load_evs_attrs()
        if "rrevs" in kwargs:
            rrevs = kwargs.pop("rrevs")
        # compute event synch if not given in nc file
        if "evs" in self.vars:
            gut.myprint("Evs are already stored in dataset.")
        elif var_name is None:
            raise ValueError("Specify varname to compute event sync.")
        else:
            gut.myprint(
                f"Compute Event synchronization for variable {self.var_name}.",
            )
            rrevs = True
        if rrevs is True:
            if var_name is None:
                var_name = self.var_name

            self.ds = self.create_evs_ds(
                var_name=var_name,
                th=self.min_treshold,
                q=self.q,
                min_evs=self.min_evs,
                th_eev=self.th_eev,
                month_range=month_range
            )
        else:
            self.mask = self.get_es_mask(self.ds["evs"], min_evs=self.min_evs)
            self.init_map_indices()
        self.vars = self.get_vars()

    def create_evs_ds(
        self, var_name, q=0.95, th=1, th_eev=15, min_evs=20, month_range=None
    ):
        """Genereates an event time series of the variable of the dataset.
        Attention, if month range is provided all values not in the month range are
        set to 0, not deleted, therefore the number of dates is retained

        Args:
            var_name (str): variable name
            q (float, optional): Quantile that defines extreme events. Defaults to 0.95.
            th (float, optional): threshold of minimum value in a time series. Defaults to 1.
            th_eev (float, optional): Threshold of minimum value of an extreme event. Defaults to 15.
            min_evs (int, optional): Minimum number of extreme events in the whole time Series. Defaults to 20.
            month_range (list, optional): list of strings as [start_month, end_month]. Defaults to None.

        Returns:
            xr.Dataset: Dataset with new values of variable and event series
        """
        self.q = q
        self.th = th
        self.th_eev = th_eev
        self.min_evs = min_evs
        gut.myprint(f'Create EVS with EE defined by q > {q}')
        if month_range is None:
            da_es, self.mask = self.compute_event_time_series(
                var_name=var_name,)
        else:
            da_es, self.mask = self.compute_event_time_series_month_range(
                start_month=month_range[0], end_month=month_range[1]
            )
        da_es.attrs = {"var_name": var_name}

        self.set_ds_attrs(ds=da_es)
        self.ds["evs"] = da_es

        return self.ds

    def get_q_maps(self, var_name):

        if var_name is None:
            var_name = self.var_name
        gut.myprint(f"Apply Event Series on variable {var_name}")

        dataarray = self.ds[var_name]

        q_val_map, ee_map, data_above_quantile, rel_frac_q_map = tu.get_ee_ds(
            dataarray=dataarray, q=self.q, th=self.th, th_eev=self.th_eev
        )

        return q_val_map, ee_map, data_above_quantile

    def compute_event_time_series(
        self, var_name=None,
    ):
        reload(tu)
        if var_name is None:
            var_name = self.var_name
        gut.myprint(f"Apply Event Series on variable {var_name}")

        dataarray = self.ds[var_name]

        event_series, mask = tu.compute_evs(
            dataarray=dataarray,
            q=self.q,
            th=self.th,
            th_eev=self.th_eev,
            min_evs=self.min_evs,
        )

        return event_series, mask

    def compute_event_time_series_month_range(
        self, start_month="Jan", end_month="Dec",
    ):
        """
        This function generates data within a given month range.
        It can be from smaller month to higher (eg. Jul-Sep) but as well from higher month
        to smaller month (eg. Dec-Feb)

        Parameters
        ----------
        start_month : string, optional
            Start month. The default is 'Jan'.
        end_month : string, optional
            End Month. The default is 'Dec'.

        Returns
        -------
        seasonal_data : xr.dataarray
            array that contains only data within month-range.

        """
        reload(tu)
        times = self.ds["time"]
        start_year, end_year = tu.get_sy_ey_time(times, sy=None, ey=None)
        gut.myprint(
            f"Get month range data from year {start_year} to {end_year}!")

        da = self.ds[self.var_name]
        # Sets the data outside the month range to 0, but retains the dates
        da_mr = self.get_month_range_data(
            dataarray=da, start_month=start_month, end_month=end_month, set_zero=True
        )
        # Computes the Event Series
        evs_mr, mask = tu.compute_evs(
            dataarray=da_mr,
            q=self.q,
            th=self.th,
            th_eev=self.th_eev,
            min_evs=self.min_evs,
        )

        return evs_mr, mask

    def create_evs_var_q(self, var_name=None, *qs):
        """Creates event time series for a list of passed q values.

        Raises:
            ValueError: if q is not float

        Returns:
            dataset: xr.DataSet with new q time series stored as variables!
        """
        for q in qs:
            if not isinstance(q, float):
                raise ValueError(f"{q} has to be float")
            self.q = q
            da_es, _ = self.compute_event_time_series(
                var_name=var_name,
                th=self.min_treshold,
                q=q,
                min_evs=self.min_evs,
                th_eev=self.th_eev,
            )
            self.set_ds_attrs(ds=da_es)
            self.ds[f"evs_q{q}"] = da_es

        return self.ds

    def get_es_mask(self, data_evs, min_evs):
        num_non_nan_occurence = data_evs.where(data_evs == 1).count(dim="time")
        self.mask = xr.where(num_non_nan_occurence > min_evs, 1, 0)
        self.min_evs = min_evs
        return self.mask

    def set_ds_attrs(self, ds):
        param_class = {
            "grid_step": self.grid_step,
            "grid_type": self.grid_type,
            "lsm": int(self.lsm),
            "q": self.q,
            "min_evs": self.min_evs,
            "min_threshold": self.min_treshold,
            "th_eev": self.th_eev,
            "th": self.th,
            "an": int(self.can),
        }
        ds.attrs = param_class
        return ds

    def save(self, file):
        """Save the dataset class object to file.
        Args:
        ----
        filepath: str
        """
        filepath = os.path.dirname(file)

        if os.path.exists(file):
            gut.myprint("File" + file + " already exists!")
            os.rename(file, file + "_backup")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        gut.myprint(f"Save file {file}")
        ds_temp = self.set_ds_attrs(self.ds)
        ds_temp.to_netcdf(file)

        return None

    def load_evs_attrs(self):
        self.q = self.ds.attrs["q"]
        self.min_evs = self.ds.attrs["min_evs"]
        self.min_treshold = self.ds.attrs["min_threshold"]
        self.th_eev = self.ds.attrs["th_eev"]
        if "an" in self.ds.attrs:  # To account for old version saved files!
            self.can = bool(self.ds.attrs["an"])
        else:
            self.can = False

    def randomize_spatio_temporal_data_yearly(
        self,
        data,
        var=None,
        start_year=None,
        end_year=None,
        sm_arr=["Jan"],
        em_arr=["Dec"],
        set_rest_zero=False,
        full_rnd=False,
        seed=0,
    ):
        """
        Permutates randomly time series for every grid location.
        Keeps the year in which the events did occur.
        """

        if len(sm_arr) != len(em_arr):
            raise ValueError(
                "ERROR! Start month array and end month array not of the same length!"
            )
        if len(sm_arr) > 1 and set_rest_zero is True:
            raise ValueError(
                "Set Time Zeros 0 and shuffle for to time periods is not possible!"
            )
        times = self.ds["time"]
        if start_year is None:
            start_year = int(times[0].time.dt.year)
        if end_year is None:
            end_year = int(times[-1].time.dt.year) + 1
        if var is None:
            var = self.var

        with gut.temp_seed():
            if full_rnd is True:
                gut.myprint(f"WARNING! Fully randomized time Series of {var}!")

                start_date = times.data[0]
                end_date = times.data[-1]
                arr_data = data.sel(time=slice(start_date, end_date))
                arr_rnd = self.randomize_spatio_temporal_data_full(
                    arr_data.data, axis=0
                )
                data.loc[dict(time=slice(start_date, end_date))] = arr_rnd
            else:
                for idx in range(len(sm_arr)):
                    sm = sm_arr[idx]
                    em = em_arr[idx]
                    gut.myprint(
                        f"WARNING! Time Series of {var} are for {sm} to {em} randomized!"
                    )

                    for idx, year in enumerate(np.arange(start_year, end_year)):

                        gut.myprint(
                            f"Shuffle Year {year} for months {sm}, {em}")
                        smi = self._get_index_of_month(sm) + 1
                        emi = self._get_index_of_month(em) + 1
                        start_date = f"{smi}-01-{year}"
                        if em == "Feb":
                            end_day = 28
                        elif em in ["Jan", "Mar", "May", "Jul", "Aug", "Oct", "Dec"]:
                            end_day = 31
                        else:
                            end_day = 30

                        ey = copy.deepcopy(year)
                        if emi < smi:
                            ey = year + 1
                        end_date = f"{emi}-{end_day}-{ey}"
                        if emi < 10:
                            end_date = f"0{emi}-{end_day}-{ey}"

                        arr_1_year = data.sel(time=slice(start_date, end_date))
                        # arr_1_year_rnd=np.random.permutation(arr_1_year.data)
                        arr_1_year_rnd = self.randomize_spatio_temporal_data_full(
                            arr_1_year.data, axis=0
                        )

                        arr_1_year.data = arr_1_year_rnd
                        # if idx == 0:
                        #     all_year = arr_1_year
                        # else:
                        #     all_year = xr.merge([all_year, arr_1_year])
                        data.loc[
                            dict(time=slice(start_date, end_date))
                        ] = arr_1_year_rnd

                        if set_rest_zero is True:
                            gut.myprint("Warning: Set Rest to Zero!")
                            if emi >= smi:  # E.g. for Jun-Sep
                                start_date_before = f"01-01-{year}"
                                end_data_after = f"12-31-{year}"
                                rest_data_before = data.sel(
                                    time=slice(start_date_before, start_date)
                                )
                                rest_data_after = data.sel(
                                    time=slice(end_date, end_data_after)
                                )
                                data_before = xr.zeros_like(rest_data_before)
                                data_after = xr.zeros_like(rest_data_after)

                                data.loc[
                                    dict(time=slice(start_date_before, start_date))
                                ] = data_before
                                data.loc[
                                    dict(time=slice(end_date, end_data_after))
                                ] = data_after
                            else:
                                # Year not ey!
                                end_date = f"{emi}-{end_day}-{year}"
                                if emi < 10:
                                    end_date = f"0{emi}-{end_day}-{year}"
                                rest_data_between = data.sel(
                                    time=slice(end_date, start_date)
                                )
                                data_between = xr.zeros_like(rest_data_between)
                                data.loc[
                                    dict(time=slice(end_date, start_date))
                                ] = data_between

                # only_data = all_year[var]
        return data

    def randomize_spatio_temporal_data_full(self, a, axis=0, seed=None):
        """
        Shuffle `a` in-place along the given axis.
        Code mainly from
        https://stackoverflow.com/questions/26310346/quickly-calculate-randomized-3d-numpy-array-from-2d-numpy-array/
        Apply numpy.random.shuffle to the given axis of `a`.
        Each one-dimensional slice is shuffled independently.
        """
        if seed is not None:
            np.random.seed(seed)
        b = a.swapaxes(axis, -1)
        # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
        # so `a` is shuffled in place, too.
        shp = b.shape[:-1]
        for ndx in np.ndindex(shp):
            np.random.shuffle(b[ndx])
        return a
