# %%
"""
Base class for the different dataset classes of the multilayer climate network.
"""

import os
import numpy as np
import scipy.interpolate as interp
import xarray as xr
import climnet.grid.grid as grid
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu
import geoutils.utils.spatial_utils as sput
from importlib import reload
from tqdm import tqdm
import geoutils.geodata.base_dataset as bds
reload(bds)


class BaseDataset(bds.BaseDataset):
    """ Base Dataset.
    Args:

    ----------
    data_nc: str
        Filename of input data, Default: None
    load_nc: str
        Already processed data file. Default: None
        If specified all other attritbutes are loaded.
    time_range: list
        List of time range, e.g. ['1997-01-01', '2019-01-01']. Default: None
    lon_range: list
        Default: [-180, 180],
    lat_range: list
        Default: [-90,90],
    grid_step: int
        Default: 1
    grid_type: str
        Default: 'gaussian',
    lsm: bool
        Default:False,
    **kwargs
    """

    def __init__(
        self,
        var_name=None,
        data_nc=None,
        load_nc=None,
        time_range=None,
        lon_range=[-180, 180],
        lat_range=[-90, 90],
        grid_step=1,
        grid_type="fekete",
        lsm=False,
        can=False,
        large_ds=False,
        timemean=None,
        detrend=False,
        **kwargs,
    ):

        if data_nc is not None and load_nc is not None:
            raise ValueError("Specify either data or load file.")

        # initialized dataset
        elif data_nc is not None:
            # check if file exists
            if not os.path.exists(data_nc):
                PATH = os.path.dirname(os.path.abspath(__file__))
                gut.myprint(f"You are here: {PATH}!")
                raise ValueError(f"File does not exist {data_nc}!")

            if large_ds is True:
                ds = self.open_large_ds(
                    var_name=var_name,
                    data_nc=data_nc,
                    time_range=time_range,
                    grid_step=grid_step,
                    **kwargs,
                )
            else:
                ds = xr.open_dataset(data_nc)

            ds = self.check_dimensions(ds, **kwargs)  # check dimensions
            ds = self.rename_var(ds)  # rename specific variable names
            self.grid_step = grid_step
            self.grid_type = grid_type
            self.lsm = lsm
            self.info_dict = kwargs

            # choose time range
            if time_range is not None:
                ds = self.get_data_timerange(ds, time_range)

            if timemean is not None:
                ds = tu.apply_timemean(ds, timemean=timemean)

            # regridding
            self.GridClass = self.create_grid(
                grid_type=self.grid_type, **kwargs,)
            if lon_range != [-180, 180] and lat_range != [-90, 90]:
                ds = self.cut_map(ds=ds, lon_range=lon_range,
                                  lat_range=lat_range)

            self.grid = self.GridClass.cut_grid(
                lat_range=[ds["lat"].min().data, ds["lat"].max().data],
                lon_range=[ds["lon"].min().data, ds["lon"].max().data],
            )

            da = ds[var_name]
            # Bring always in the form (time, lat, lon)
            # much less memory consuming than for dataset!
            gut.myprint("transpose data!")
            da = da.transpose("time", "lat", "lon")
            da = self.interp_grid(da, self.grid)

            if large_ds is True:
                ds.unify_chunks()

            if self.lsm is True:
                mask_lsm, _ = self.get_land_sea_mask_data(da)
            else:
                mask_lsm = None
            self.ds = da.to_dataset(name=var_name)
            (
                self.time_range,
                self.lon_range,
                self.lat_range,
            ) = self.get_spatio_temp_range(ds)
            self.mask = self.init_mask(da=da, mask_ds=mask_lsm)

        # load dataset object from file
        elif load_nc is not None:
            self.load(load_nc)
            if timemean is not None:
                self.ds = self.apply_timemean(timemean=timemean)
            if time_range is not None:
                self.ds = self.get_data_timerange(
                    self.ds, time_range=time_range)
                self.time_range = time_range

        # select a main var name
        self.vars = list(self.ds.keys())
        self.var_name = var_name if var_name is not None else self.vars[0]

        # detrending
        if detrend is True:
            detrend_from = kwargs.pop('detrend_from', None)
            self.detrend(dim="time", startyear=detrend_from)

        # Compute Anomalies if needed
        self.can = can
        if self.can is True:
            self.compute_anomalies_ds(kwargs)

        # Flatten index in map
        # Predefined variables set to None
        init_indices = kwargs.pop('init_indices', True)
        if init_indices:
            self.indices_flat, self.idx_map = self.init_map_indices()

        if load_nc is None:
            self.ds = self.ds.assign_coords(
                idx_flat=("points", self.idx_map.data))

        self.loc_dict = dict()

    def re_init(self):
        return None

    def open_large_ds(self, var_name, data_nc, time_range, grid_step, **kwargs):
        sp_large_ds = kwargs.pop("sp_large_ds", None)
        if not os.path.exists(sp_large_ds):
            ds = self.preprocess_large_ds(
                nc_file=data_nc,
                var_name=var_name,
                time_range=time_range,
                grid_step=grid_step,
                sp_large_ds=sp_large_ds,
                **kwargs,
            )
            gut.myprint("Finished preprocessing large dataset...")
        else:
            gut.myprint(
                f"Compressed file {sp_large_ds} already exists! Read now!"
            )
        ds = xr.open_dataset(sp_large_ds)

        return ds

    def load(self, load_nc):
        """Load dataset object from file.

        Parameters:
        ----------
        ds: xr.Dataset
            Dataset object containing the preprocessed dataset

        """
        # check if file exists
        gut.myprint("Loading Data...")

        if not os.path.exists(load_nc):
            PATH = os.path.dirname(os.path.abspath(__file__))
            gut.myprint(f"You are here: {PATH}!")
            raise ValueError(f"File does not exist {load_nc}!")

        gut.myprint(f"Load Dataset: {load_nc}")
        # set lock to false to allow running in parallel
        with xr.open_dataset(load_nc, lock=False) as ds:
            (
                self.time_range,
                self.lon_range,
                self.lat_range,
            ) = self.get_spatio_temp_range(ds)

            self.grid_step = ds.attrs["grid_step"]
            self.grid_type = ds.attrs["grid_type"]
            self.lsm = bool(ds.attrs["lsm"])
            self.info_dict = ds.attrs  # TODO
            # Read and create grid class
            self.grid = dict(lat=ds.lat.data, lon=ds.lon.data)
            for name, da in ds.data_vars.items():
                gut.myprint(f"Variables in dataset: {name}")

            # points which are always NaN will be NaNs in mask
            mask = np.ones_like(ds[name][0].data, dtype=bool)
            for idx, t in enumerate(ds.time):
                mask *= np.isnan(ds[name].sel(time=t).data)

            self.mask = xr.DataArray(
                data=xr.where(mask == 0, 1, np.NaN),  # or mask == False
                dims=da.sel(time=da.time[0]).dims,
                coords=da.sel(time=da.time[0]).coords,
                name="lsm",
            )

            ds = self.check_time(ds)

            self.ds = ds

        return self.ds

    def save(self, filepath):
        """Save the dataset class object to file.
        Args:
        ----
        filepath: str
        """
        if os.path.exists(filepath):
            gut.myprint("File" + filepath + " already exists!")
            os.rename(filepath, filepath + "_backup")

        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        param_class = {
            "grid_step": self.grid_step,
            "grid_type": self.grid_type,
            "lsm": int(self.lsm),
            **self.info_dict,
        }
        ds_temp = self.ds
        ds_temp.attrs = param_class
        ds_temp.to_netcdf(filepath)
        gut.myprint(f"File {filepath} written!")
        return None

    def preprocess_large_ds(
        self,
        nc_file,
        var_name,
        time_range=None,
        grid_step=1,
        sp_large_ds=None,
        **kwargs,
    ):
        gut.myprint("Start preprocessing data!")

        ds_large = xr.open_dataset(nc_file, chunks={"time": 100})
        ds_large = self.check_dimensions(ds_large, **kwargs)
        ds_large = self.rename_var(ds_large)
        da = ds_large[var_name]
        da = da.transpose("time", "lat", "lon")
        da = self.get_data_timerange(da, time_range)
        if max(da.lon) > 180:
            gut.myprint("Shift longitude in Preprocessing!")
            da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
        da = self.common_grid(dataarray=da, grid_step=grid_step)
        ds_large.unify_chunks()

        ds = da.to_dataset(name=var_name)
        gut.myprint("Finished preprocessing data")

        if sp_large_ds is not None:
            dirname = os.path.dirname(sp_large_ds)
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            ds.to_netcdf(sp_large_ds)
            gut.myprint(f"Save processed data to file {sp_large_ds}!")

        return ds

    def create_grid(self, grid_type="fibonacci", num_iter=1000, **kwargs):
        """Common grid for all datasets.

        ReturnL
        -------
        Grid: grid.BaseGrid
        """
        reload(grid)
        dist_equator = grid.degree2distance_equator(self.grid_step)
        sp_grid = kwargs.pop("sp_grid", None)
        gut.myprint(f"Start create grid {grid_type}...")
        if grid_type == "gaussian":
            Grid = grid.GaussianGrid(self.grid_step, self.grid_step)
        elif grid_type == "fibonacci":
            Grid = grid.FibonacciGrid(dist_equator)
        elif grid_type == "fekete":
            num_points = grid.get_num_points(dist_equator)
            Grid = grid.FeketeGrid(
                num_points=num_points,
                num_iter=num_iter,
                pre_proccess_type=None,
                load_grid=sp_grid,
            )
        else:
            raise ValueError(f"Grid type {grid_type} does not exist.")

        return Grid

    def interp_grid(self, dataarray, new_grid):
        """Interpolate dataarray on new grid.
        dataarray: xr.DataArray
            Dataarray to interpolate.
        new_grid: dict
            Grid we want to interpolate on.
        """
        new_points = np.array([new_grid["lon"], new_grid["lat"]]).T
        lon_mesh, lat_mesh = np.meshgrid(dataarray.lon, dataarray.lat)
        origin_points = np.array([lon_mesh.flatten(), lat_mesh.flatten()]).T
        # for one timestep
        if len(dataarray.data.shape) < 3:
            origin_values = dataarray.data.flatten()
            assert len(origin_values) == origin_points.shape[0]
            new_values = interp.griddata(
                origin_points, origin_values, new_points, method="nearest"
            )
            new_values = np.array(new_values).T
            coordinates = dict(
                points=np.arange(0, len(new_points), 1),
                lon=("points", new_points[:, 0]),
                lat=("points", new_points[:, 1]),
            )
            dims = ["points"]
        else:
            new_values = []
            pb_fmt = "{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}"
            pb_desc = "Intpol Grid points in time..."
            for idx, t in enumerate(
                tqdm(dataarray.time, bar_format=pb_fmt, desc=pb_desc)
            ):
                origin_values = dataarray.sel(time=t.data).data.flatten()
                assert len(origin_values) == origin_points.shape[0]
                new_values.append(
                    interp.griddata(
                        origin_points, origin_values, new_points, method="nearest"
                    )
                )
            # coordinates = dict(time=dataarray.time.data,
            #                    points=np.arange(0, len(new_points), 1),
            #                    lon=("points", new_points[:, 0]),
            #                    lat=("points", new_points[:, 1]))
            coordinates = sput.create_new_coordinates(
                times=dataarray.time.data,
                lons=new_points[:, 0], lats=new_points[:, 1]
            )
            dims = ["time", "points"]
            new_values = np.array(new_values)

        new_dataarray = xr.DataArray(
            data=new_values, dims=dims, coords=coordinates, name=dataarray.name
        )

        return new_dataarray

    def get_land_sea_mask_data(self, dataarray):
        """
        Compute a land-sea-mask for the dataarray,
        based on an input file for the land-sea-mask.
        """
        PATH = os.path.dirname(os.path.abspath(
            __file__))  # Adds higher directory
        lsm_mask_ds = xr.open_dataset(PATH + "/../input/land-sea-mask_era5.nc")
        lsm_mask = self.interp_grid(lsm_mask_ds["lsm"], self.grid)

        land_dataarray = xr.where(np.array([lsm_mask]) == 1, dataarray, np.nan)
        return lsm_mask, land_dataarray

    def mask_point_ids(self, points):
        """In the index list the indices are delivered as eg. nodes of a network.
        This is not yet the point number! In the mask the corresponding point numbers
        are set to 0 and the new mask is reinitialized

        Args:
            points (list): list of points.
        """
        if len(points) > 0:
            self.mask[points] = int(0)
            self.init_map_indices()
        return

    def mask_node_ids(self, idx_list):
        """In the index list the indices are delivered as eg. nodes of a network.
        This is not yet the point number! In the mask the corresponding point numbers
        are set to 0 and the new mask is reinitialized

        Args:
            idx_list (list): list of indices.
        """
        if len(idx_list) > 0:
            points = self.get_points_for_idx(idx_list)
            self.mask_point_ids(points=points)
        return

    def get_points_for_idx(self, idx_lst):
        """Returns the point number of the map for a given index list.
        Important eg. to transform node ids to points of the network
        Args:
            idx_lst (list): list of indices of the network.

        Returns:
            np.array: array of the points of the index list
        """
        point_lst = []

        for idx in idx_lst:
            # map_idx = self.get_map_index(idx)
            # point = int(map_idx["point"])
            # point_lst.append(point)
            point_lst.append(self.key_val_idx_point_dict[idx])

        return np.array(point_lst, dtype=int)

    def get_idx_for_point(self, point):
        """Gets for a point its corresponding indices

        Args:
            point (int): point number (is a dimension of)

        Raises:
            ValueError:

        Returns:
            int: index number, must be <= point
        """
        flat_idx = self.idx_map.sel(points=point).data
        if np.isnan(flat_idx) is True:
            raise ValueError(f"Error the point {point} is not defined!")

        return int(flat_idx)

    def get_idx_point_lst(self, point_lst):
        idx_lst = []
        for point in point_lst:
            idx_lst.append(self.get_idx_for_point(point=point))

        return np.array(idx_lst)

    def get_coordinates_flatten(self):
        """Get coordinates of flatten array with removed NaNs.

        Return:
        -------
        coord_deg:
        coord_rad:
        map_idx:
        """
        # length of the flatten array with NaNs removed
        # length = self.flatten_array().shape[1]
        length = len(self.indices_flat)
        coord_deg = []
        map_idx = []
        for i in range(length):
            buff = self.get_map_index(i)
            coord_deg.append([buff["lat"], buff["lon"]])
            map_idx.append(buff["point"])

        coord_rad = np.radians(coord_deg)  # transforms to np.array

        return np.array(coord_deg), coord_rad, np.array(map_idx)

    def count_indices_to_array(self, idx_list):
        """
        Returns a flattened list of indices where the idx_list is at the correct position.
        It counts the occurrence of each index in the idx_lst.
        """
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        len_index = np.count_nonzero(mask_arr)
        full_idx_lst = np.zeros(len_index)
        # counts as number of occurences
        u_index, counts = np.unique(idx_list, return_counts=True)
        full_idx_lst[u_index] = counts

        return full_idx_lst

    # ####################### Spatial Location functions specific for climnet #####################

    def get_locations_in_range(
        self, def_map=None, lon_range=None, lat_range=None,
        dateline=False
    ):
        """
        Returns a map with the location within certain range.

        Parameters:
        -----------
        lon_range: list
            Range of longitudes, i.e. [min_lon, max_lon]
        lat_range: list
            Range of latitudes, i.e. [min_lat, max_lat]
        def_map: xr.Dataarray
            Map of data, i.e. the mask.

        Returns:
        --------
        idx_lst: np.array
            List of indices of the flattened map.
        mmap: xr.Dataarray
            Dataarray including ones at the location and NaNs everywhere else
        """
        if lon_range is None:
            lon_range = self.lon_range
        if lat_range is None:
            lat_range = self.lat_range
        if def_map is None:
            def_map = self.mask
        mmap = sput.get_locations_in_range(
            def_map=def_map, lon_range=lon_range, lat_range=lat_range,
            dateline=dateline
        )

        defined_points = np.nonzero(self.mask.data)[0]
        point_lst_range = np.where(~np.isnan(mmap.data))[0]
        def_point_list = np.intersect1d(defined_points, point_lst_range)
        idx_lst = self.get_idx_point_lst(point_lst=def_point_list)

        return idx_lst, mmap


class AnomalyDataset(BaseDataset):
    """Anomaly Dataset.

    Parameters:
    ----------
    data_nc: str
        Filename of input data, Default: None
    load_nc: str
        Already processed data file. Default: None
        If specified all other attritbutes are loaded.
    time_range: list
        Default: ['1997-01-01', '2019-01-01'],
    lon_range: list
        Default: [-180, 180],
    lat_range: list
        Default: [-90,90],
    grid_step: int
        Default: 1
    grid_type: str
        Default: 'gaussian',
    start_month: str
        Default: 'Jan'
    end_month: str
        Default: 'Dec'
    lsm: bool
        Default:False
    detrend: bool
        Default: True
    climatology: str
        Specified climatology the anomalies are computed over. Default: "dayofyear"
    **kwargs
    """

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
        timemean=None,
        lsm=False,
        climatology="dayofyear",
        detrend=False,
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
            timemean=timemean,
            lsm=lsm,
            detrend=detrend,
            **kwargs,
        )

        # compute anomalies if not given in nc file
        if "anomalies" in self.vars:
            gut.myprint("Anomalies are already stored in dataset.")
        elif var_name is None:
            raise ValueError("Specify varname to compute anomalies.")
        else:
            gut.myprint(f"Compute anomalies for variable {self.var_name}.")
            da = self.ds[self.var_name]
            da = self.compute_anomalies(da, group=climatology)
            da.attrs = {"var_name": self.var_name}
            self.ds["anomalies"] = da

        # set var name to "anomalies" in order to run network on anomalies
        self.var_name = "anomalies"
