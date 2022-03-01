# %%
"""
Base class for the different dataset classes of the multilayer climate network.
"""
import sys
import os
import numpy as np
import scipy.interpolate as interp
import xarray as xr
import copy
import tqdm
import climnet.grid as grid
import climnet.utils.general_utils as gut
import climnet.utils.time_utils as tu
import climnet.utils.spatial_utils as sput
from importlib import reload


class BaseDataset:
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
    month_range: list
        List of month range, e.g. ["Jan", "Dec"]. Default: None
    lsm: bool
        Default:False,
    **kwargs
    """

    def __init__(self,
                 var_name,
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
                 timemean=None,
                 can=False,
                 detrend=False,
                 **kwargs
                 ):

        if data_nc is not None and load_nc is not None:
            raise ValueError("Specify either data or load file.")

        # initialized dataset
        elif data_nc is not None:
            # check if file exists
            if not os.path.exists(data_nc):
                PATH = os.path.dirname(os.path.abspath(__file__))
                print(f"You are here: {PATH}!", flush=True)
                raise ValueError(f"File does not exist {data_nc}!")

            if large_ds is True:
                ds = self.open_large_ds(var_name=var_name,
                                        data_nc=data_nc,
                                        time_range=time_range,
                                        grid_step=grid_step,
                                        **kwargs)
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
            if month_range is not None:
                ds = self.get_month_range_data(ds, start_month=month_range[0],
                                               end_month=month_range[1])

            # regridding
            self.GridClass = self.create_grid(grid_type=self.grid_type,
                                              )
            if lon_range != [-180, 180] and lat_range != [-90, 90]:
                ds = self.cut_map(ds, lon_range, lat_range)
            self.grid = self.GridClass.cut_grid(
                [ds['lat'].min().data, ds['lat'].max().data],
                [ds['lon'].min().data, ds['lon'].max().data]
            )

            da = ds[var_name]
            # Bring always in the form (time, lat, lon)
            # much less memory consuming than for dataset!
            print('transpose data!', flush=True)
            da = da.transpose('time', 'lat', 'lon')
            da = self.interp_grid(da, self.grid)

            if large_ds is True:
                ds.unify_chunks()

            if self.lsm is True:
                self.mask, da = self.get_land_sea_mask_data(da)
            else:
                self.mask = xr.DataArray(
                    data=np.ones_like(da[0].data),
                    dims=da.sel(time=da.time[0]).dims,
                    coords=da.sel(time=da.time[0]).coords,
                    name='mask')

            self.ds = da.to_dataset(name=var_name)
            self.time_range, self.lon_range, self.lat_range = self.get_spatio_temp_range(
                ds)


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
        self.vars = []
        for name, da in self.ds.data_vars.items():
            self.vars.append(name)
        self.var_name = var_name if var_name is not None else self.vars[0]

        # detrending
        if detrend is True:
            self.detrend(dim='time')

        # Flatten index in map
        self.indices_flat, self.idx_map = self.init_mask_idx()

        if load_nc is None:
            self.ds = self.ds.assign_coords(idx_flat=("points", self.idx_map.data))


    def re_init(self):
        return None

    def months(self):
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        return months

    def open_large_ds(self, var_name,
                      data_nc,
                      time_range,
                      grid_step,
                      **kwargs):
        sp_large_ds = kwargs.pop('sp_large_ds', 'None')
        if not os.path.exists(sp_large_ds):
            ds = self.preprocess_large_ds(nc_file=data_nc,
                                          var_name=var_name,
                                          time_range=time_range,
                                          grid_step=grid_step,
                                          sp_large_ds=sp_large_ds,
                                          **kwargs,
                                          )
            # sys.exit(0)  # Just to ensure that the file is created!
        else:
            print(f'Compressed file {sp_large_ds} already exists! Read now!',
                  flush=True)
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
        if not os.path.exists(load_nc):
            PATH = os.path.dirname(os.path.abspath(__file__))
            print(f"You are here: {PATH}!")
            raise ValueError(f"File does not exist {load_nc}!")

        print(f'Load Dataset: {load_nc}', flush=True)
        # set lock to false to allow running in parallel
        with xr.open_dataset(load_nc, lock=False) as ds:
            self.time_range, self.lon_range, self.lat_range = self.get_spatio_temp_range(
                ds)

            self.grid_step = ds.attrs['grid_step']
            self.grid_type = ds.attrs['grid_type']
            self.lsm = bool(ds.attrs['lsm'])
            self.info_dict = ds.attrs  # TODO
            # Read and create grid class
            self.grid = dict(lat=ds.lat.data, lon=ds.lon.data)
            for name, da in ds.data_vars.items():
                print("Variables in dataset: ", name, flush=True)

            # points which are always NaN will be NaNs in mask
            mask = np.ones_like(ds[name][0].data, dtype=bool)
            for idx, t in enumerate(ds.time):
                mask *= np.isnan(ds[name].sel(time=t).data)

            self.mask = xr.DataArray(
                data=xr.where(mask == False, 1, np.NaN),
                dims=da.sel(time=da.time[0]).dims,
                coords=da.sel(time=da.time[0]).coords,
                name='lsm')

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
            print("File" + filepath + " already exists!", flush=True)
            os.rename(filepath, filepath + "_backup")

        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        param_class = {
            "grid_step": self.grid_step,
            "grid_type": self.grid_type,
            "lsm": int(self.lsm),
            **self.info_dict
        }
        ds_temp = self.ds
        ds_temp.attrs = param_class
        ds_temp.to_netcdf(filepath)
        print(f"File {filepath} written!", flush=True)
        return None

    def check_dimensions(self, ds, **kwargs):
        """
        Checks whether the dimensions are the correct ones for xarray!
        """
        lon_lat_names = ['longitude', 'latitude']
        xr_lon_lat_names = ['lon', 'lat']
        dims = list(ds.dims)

        for idx, lon_lat in enumerate(lon_lat_names):
            if lon_lat in dims:
                print(dims)
                print(f'Rename:{lon_lat} : {xr_lon_lat_names[idx]} ')
                ds = ds.rename({lon_lat: xr_lon_lat_names[idx]})
                dims = list(ds.dims)
                print(dims)
        clim_dims = ['time', 'lat', 'lon']
        for dim in clim_dims:
            if dim not in dims:
                raise ValueError(
                    f"The dimension {dim} not consistent with required dims {clim_dims}!")

        # If lon from 0 to 360 shift to -180 to 180
        if max(ds.lon) > 180:
            print("Shift longitude!", flush=True)
            ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))

        # Transpose always to form time x lat x lon
        # ds = ds.transpose('time', 'lat', 'lon')
        # Set time series to days
        ds = self.check_time(ds, **kwargs)

        return ds

    def check_time(self, ds, **kwargs):
        ts_days = kwargs.pop('ts_days', True)
        if ts_days:
            ds = ds.assign_coords(time=ds.time.data.astype('datetime64[D]'))

        return ds

    def preprocess_large_ds(self,
                            nc_file,
                            var_name,
                            time_range=None,
                            grid_step=1,
                            sp_large_ds=None,
                            **kwargs,
                            ):
        print("Start preprocessing data!", flush=True)

        ds = xr.open_dataset(nc_file, chunks={"time": 100})
        ds = self.check_dimensions(ds, **kwargs)
        ds = self.rename_var(ds)
        da = ds[var_name]
        da = da.transpose('time', 'lat', 'lon')
        da = self.get_data_timerange(da, time_range)
        if max(da.lon) > 180:
            print("Shift longitude in Preprocessing!")
            da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
        da = self.common_grid(dataarray=da,
                              grid_step=grid_step)
        ds.unify_chunks()

        ds = da.to_dataset(name=var_name)
        print('Finished preprocessing data', flush=True)

        if sp_large_ds is not None:
            dirname = os.path.dirname(sp_large_ds)
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            da.to_netcdf(sp_large_ds)
            print(f"Save processed data to file {sp_large_ds}!", flush=True)

        return ds

    def rename_var(self, ds):
        names = []
        for name, da in ds.data_vars.items():
            names.append(name)

        if 'precipitation' in names:
            ds = ds.rename({'precipitation': 'pr'})
            print('Rename precipitation: pr!')
        if 'tp' in names:
            ds = ds.rename({'tp': 'pr'})
            print('Rename tp: pr!')

        if 'p86.162' in names:
            ds = ds.rename({'p86.162': 'vidtef'})
            print(
                'Rename vertical integral of divergence of total energy flux to: vidtef!')
        if 'p84.162' in names:
            ds = ds.rename({'p84.162': 'vidmf'})
            print(
                'Rename vertical integral of divergence of moisture flux to: vidmf!')

        return ds

    def create_grid(self,
                    grid_type='fibonacci',
                    num_iter=1000):
        """Common grid for all datasets.

        ReturnL
        -------
        Grid: grid.BaseGrid
        """
        dist_equator = grid.degree2distance_equator(self.grid_step)

        print(f'Start create grid {grid_type}...', flush=True)
        if grid_type == 'gaussian':
            Grid = grid.GaussianGrid(self.grid_step, self.grid_step)
        elif grid_type == 'fibonacci':
            Grid = grid.FibonacciGrid(dist_equator)
        elif grid_type == 'fekete':
            num_points = grid.get_num_points(dist_equator)
            Grid = grid.FeketeGrid(num_points=num_points,
                                   num_iter=num_iter,
                                   pre_proccess_type=None)
        else:
            raise ValueError(f'Grid type {grid_type} does not exist.')

        return Grid

    def interp_grid(self, dataarray, new_grid):
        """Interpolate dataarray on new grid.
        dataarray: xr.DataArray
            Dataarray to interpolate.
        new_grid: dict
            Grid we want to interpolate on.
        """
        print('Interpolate data to new grid:')
        new_points = np.array([new_grid['lon'], new_grid['lat']]).T

        lon_mesh, lat_mesh = np.meshgrid(dataarray.lon, dataarray.lat)
        origin_points = np.array([lon_mesh.flatten(), lat_mesh.flatten()]).T
        # for one timestep
        if len(dataarray.shape) < 3:
            origin_values = dataarray.data.flatten()
            assert len(origin_values) == origin_points.shape[0]
            new_values = interp.griddata(origin_points, origin_values, new_points,
                                         method='nearest')
            new_values = np.array(new_values).T
            coordinates = dict(points=np.arange(0, len(new_points), 1),
                               lon=("points", new_points[:, 0]),
                               lat=("points", new_points[:, 1]))
            dims = ['points']
        else:
            new_values = []
            for idx, t in enumerate(dataarray.time):
                origin_values = dataarray.sel(time=t.data).data.flatten()
                assert len(origin_values) == origin_points.shape[0]
                new_values.append(
                    interp.griddata(origin_points, origin_values,
                                    new_points,
                                    method='nearest')
                )
            coordinates = dict(time=dataarray.time.data,
                               points=np.arange(0, len(new_points), 1),
                               lon=("points", new_points[:, 0]),
                               lat=("points", new_points[:, 1]))
            dims = ['time', 'points']
            new_values = np.array(new_values)

        new_dataarray = xr.DataArray(
            data=new_values,
            dims=dims,
            coords=coordinates,
            name=dataarray.name)

        return new_dataarray

    def common_grid(self, dataarray,
                    grid_step=1):
        """Common grid for all datasets.
        """
        # min_lon = min(lon_range)
        # min_lat = min(lat_range)
        # Use minimum of original dataset because other lower variables aren't defined
        min_lat = min(dataarray['lat'])
        min_lon = min(dataarray['lon'])

        max_lat = max(dataarray['lat'])
        max_lon = max(dataarray['lon'])

        # init_lat = np.arange(min_lat, max_lat, grid_step, dtype=float)
        # init_lon = np.arange(min_lon, max_lon, grid_step, dtype=float)
        init_lat = gut.crange(min_lat, max_lat, grid_step)
        init_lon = gut.crange(min_lon, max_lon, grid_step)

        nlat = len(init_lat)
        if nlat % 2:
            # Odd number of latitudes includes the poles.
            print('WARNING: Poles might be included: {min_lat} and {min_lat}!', flush=True)

        grid = {'lat': init_lat, 'lon': init_lon}

        print(
            f"Interpolte grid from {float (min_lon)} to {float(max_lon)}, {float(min_lat)} to {float(max_lat)}!", flush=True)
        da = dataarray.interp(grid, method='nearest')

        return da

    def cut_map(self, ds, lon_range, lat_range, shortest=True):
        """Cut an area in the map. Use always smallest range as default.
        It lon ranges accounts for regions (eg. Pacific) that are around the -180/180 region.

        Args:
        ----------
        lon_range: list [min, max]
            range of longitudes
        lat_range: list [min, max]
            range of latitudes
        shortest: boolean
            use shortest range in longitude (eg. -170, 170 range) contains all points from
            170-180, -180- -170, not all between -170 and 170. Default is True.
        Return:
        -------
        ds_area: xr.dataset
            Dataset cut to range
        """
        ds_cut = sput.cut_map(ds=ds,
                              lon_range=lon_range,
                              lat_range=lat_range,
                              shortest=shortest)

        return ds_cut

    def get_spatio_temp_range(self, ds):
        time_range = [ds.time.data[0], ds.time.data[-1]]
        lon_range = [float(ds.lon.min()), float(ds.lon.max())]
        lat_range = [float(ds.lat.min()), float(ds.lat.max())]

        return time_range, lon_range, lat_range

    def get_land_sea_mask_data(self, dataarray):
        """
        Compute a land-sea-mask for the dataarray,
        based on an input file for the land-sea-mask.
        """
        PATH = os.path.dirname(os.path.abspath(
            __file__))  # Adds higher directory
        lsm_mask_ds = xr.open_dataset(PATH + "/../input/land-sea-mask_era5.nc")
        lsm_mask = self.interp_grid(lsm_mask_ds['lsm'], self.grid)

        land_dataarray = xr.where(np.array([lsm_mask]) == 1, dataarray, np.nan)
        return lsm_mask, land_dataarray

    def flatten_array(self, dataarray=None, time=True, check=True):
        """Flatten and remove NaNs.
        """
        if dataarray is None:
            dataarray = self.ds[self.var_name]

        idx_land = np.where(self.mask.data.flatten() == 1)[0]
        if time is False:
            buff = dataarray.data.flatten()
            buff[np.isnan(buff)] = 0.0  # set missing data to climatology
            data = buff[idx_land]
        else:
            data = []
            for idx, t in enumerate(dataarray.time):
                buff = dataarray.sel(time=t.data).data.flatten()
                buff[np.isnan(buff)] = 0.0  # set missing data to climatology
                data.append(buff[idx_land])

        # check
        if check is True:
            num_nonzeros = np.count_nonzero(data[-1])
            num_landpoints = sum(~np.isnan(self.mask.data.flatten()))
            print(f"The number of non-zero datapoints {num_nonzeros} "
                  + f"should approx. be {num_landpoints}.")

        return np.array(data)

    def init_mask_idx(self):
        """
        Initializes the flat indices of the map.
        Usefule if get_map_index is called multiple times.
        """
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        self.indices_flat = np.arange(
            0, np.count_nonzero(mask_arr), 1, dtype=int)

        self.idx_map = self.get_map(self.indices_flat, name='idx_flat')
        def_locs = []
        for idx in self.indices_flat:
            slon, slat = self.get_coord_for_idx(idx)
            def_locs.append([slon, slat])

        self.def_locs = np.array(def_locs)

        return self.indices_flat, self.idx_map

    def mask_node_ids(self, idx_list):
        """In the index list the indices are delivered as eg. nodes of a network.
        This is not yet the point number! In the mask the corresponding point numbers
        are set to 0 and the new mask is reinitialized

        Args:
            idx_list (list): list of indices.
        """
        points = self.get_points_for_idx(idx_list)
        self.mask[points] = int(0)
        self.init_mask_idx()

    def get_map(self, data, name=None):
        """Restore dataarray map from flattened array.

        TODO: So far only a map at one time works, extend to more than one time

        This also includes adding NaNs which have been removed.
        Args:
        -----
        data: np.ndarray (n,0)
            flatten datapoints without NaNs
        mask_nan: xr.dataarray
            Mask of original dataarray containing True for position of NaNs
        name: str
            naming of xr.DataArray

        Return:
        -------
        dmap: xr.dataArray
            Map of data
        """
        if name is None:
            name = self.var_name
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        # Number of non-NaNs should be equal to length of data
        assert np.count_nonzero(mask_arr) == len(data)

        # create array with NaNs
        data_map = np.empty(len(mask_arr))
        data_map[:] = np.NaN

        # fill array with sample
        data_map[mask_arr] = data

        # dmap = xr.DataArray(
        #     data=data_map,
        #     dims=['points'],
        #     coords=dict(points=self.ds.points.data,
        #                 lon=("points", self.ds.lon.data),
        #                 lat=("points", self.ds.lat.data)),
        #     name=name)

        dmap = xr.DataArray(
            data=np.reshape(data_map, self.mask.data.shape),
            dims=self.mask.dims,
            coords=self.mask.coords,
            name=name)

        return dmap

    def get_map_index(self, idx_flat):
        """Get lat, lon and index of map from index of flatten array
           without Nans.

        # Attention: Mask has to be initialised

        Args:
        -----
        idx_flat: int, list
            index or list of indices of the flatten array with removed NaNs

        Return:
        idx_map: dict
            Corresponding indices of the map as well as lat and lon coordinates
        """

        indices_flat = self.indices_flat

        idx_map = self.idx_map

        buff = idx_map.where(idx_map == idx_flat, drop=True)
        if idx_flat > len(indices_flat):
            raise ValueError("Index doesn't exist.")
        map_idx = {
            'lat': buff.lat.data,
            'lon': buff.lon.data,
            'idx': np.argwhere(idx_map.data == idx_flat)
        }
        return map_idx

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
            map_idx = self.get_map_index(idx)
            point = int(map_idx['idx'])
            point_lst.append(point)

        return np.array(point_lst, dtype=int)

    def get_idx_for_point(self, point):
        """Gets for a point its corresponding indices

        Args:
            point (int): point number (is a dimension of)

        Raises:
            ValueError:

        Returns:
            int: index number, must be < point
        """
        flat_idx = self.idx_map.sel(points=point).data
        if np.isnan(flat_idx) is True:
            raise ValueError(
                f"Error the point {point} is not defined!")

        return int(flat_idx)

    def get_idx_point_lst(self, point_lst):
        idx_lst = []
        for point in point_lst:
            idx_lst.append(self.get_idx_for_point(point=point))

        return idx_lst

    def get_coord_for_idx(self, idx):
        map_dict = self.get_map_index(idx)
        slon = float(map_dict['lon'])
        slat = float(map_dict['lat'])

        return slon, slat

    def get_coordinates_flatten(self):
        """Get coordinates of flatten array with removed NaNs.

        Return:
        -------
        coord_deg:
        coord_rad:
        map_idx:
        """
        # length of the flatten array with NaNs removed
        length = self.flatten_array().shape[1]

        coord_deg = []
        map_idx = []
        for i in range(length):
            buff = self.get_map_index(i)
            coord_deg.append([buff['lat'][0], buff['lon'][0]])
            map_idx.append(buff['idx'][0])

        coord_rad = np.radians(coord_deg)

        return np.array(coord_deg), coord_rad, map_idx

    def get_index_for_coord(self, lon, lat):
        """Get index of flatten array for specific lat, lon."""
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        indices_flat = np.arange(0, np.count_nonzero(mask_arr), 1, dtype=int)

        idx_map = self.get_map(indices_flat, name='idx_flat')

        # idx = idx_map.sel(lat=lat, lon=lon, method='nearest')
        lon, lat, idx_all = grid.find_nearest_lat_lon(lon=lon,
                                                      lat=lat,
                                                      lon_arr=idx_map['lon'],
                                                      lat_arr=idx_map['lat'])

        idx = self.idx_map.sel(points=idx_all)
        if np.isnan(idx.data) is True:
            raise ValueError(
                f"Error the lon {lon} lat {lat} is not defined!")

        return int(idx)

    def flat_idx_array(self, idx_list):
        """
        Returns a flattened list of indices where the idx_list is at the correct position.
        """
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        len_index = np.count_nonzero(mask_arr)
        max_idx = np.max(idx_list)
        if max_idx > len_index:
            raise ValueError(f'Error: index {max_idx} higher than #nodes {len_index}!')
        full_idx_lst = np.zeros(len_index)
        full_idx_lst[idx_list] = 1

        return full_idx_lst

    def count_indices_to_array(self, idx_list):
        """
        Returns a flattened list of indices where the idx_list is at the correct position.
        It counts the occurrence of each index in the idx_lst.
        """
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        len_index = np.count_nonzero(mask_arr)
        full_idx_lst = np.zeros(len_index)
        u_index, counts = np.unique(idx_list, return_counts=True)  # counts as number of occurences
        full_idx_lst[u_index] = counts

        return full_idx_lst

    def find_nearest(self, a, a0):
        """
        Element in nd array `a` closest to the scalar value `a0`
        ----
        Args a: nd array
             a0: scalar value
        Return
            idx, value
        """
        idx = np.abs(a - a0).argmin()
        return idx, a.flat[idx]

    def interp_times(self, dataset, time_range):
        """Interpolate time in time range in steps of days.
        TODO: So far only days works.
        """
        time_grid = np.arange(
            time_range[0], time_range[1], dtype='datetime64[D]'
        )
        ds = dataset.interp(time=time_grid, method='nearest')
        return ds

    def get_data_timerange(self, data, time_range=None):
        """Gets data in a certain time range.
        Checks as well if time range exists in file!

        Args:
            data (xr.Dataarray): xarray dataarray
            time_range (list, optional): List dim 2 that contains the time interval. Defaults to None.

        Raises:
            ValueError: If time range is not in time range of data

        Returns:
            xr.Dataarray: xr.Dataarray in seleced time range.
        """

        start_time, end_time = tu.get_sy_ey_time(data.time)

        td = data.time.data
        if time_range is not None:
            if (td[0] > np.datetime64(time_range[0])) or (td[-1] < np.datetime64(time_range[1])):
                raise ValueError(
                    f"Please select time array within {td[0]} - {td[-1]}!")
            else:
                print(f"Time steps within {time_range} selected!")
            # da = data.interp(time=t, method='nearest')
            da = data.sel(time=slice(time_range[0], time_range[1]))

            print("Time steps selected!")
        else:
            da = data
        return da

    def _get_index_of_month(self, month):
        idx = -1
        idx = self.months().index(month)
        if idx == -1:
            raise ValueError("This month does not exist: ", month)
        return idx

    def _is_in_month_range(self, month, start_month, end_month):
        start_month_idx = self._get_index_of_month(start_month)+1
        end_month_idx = self._get_index_of_month(end_month)+1

        if start_month_idx <= end_month_idx:
            mask = (month >= start_month_idx) & (month <= end_month_idx)
        else:
            mask = (month >= start_month_idx) | (month <= end_month_idx)
        return mask

    def get_month_range_data(self, dataset=None,
                             start_month='Jan', end_month='Dec',
                             set_ds=False):
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
        if dataset is None:
            dataset = self.ds
        seasonal_data = tu.get_month_range_data(dataset,
                                                start_month=start_month,
                                                end_month=end_month)
        if set_ds is True:
            self.ds = seasonal_data
        return seasonal_data

    def get_mean_loc(self, idx_lst):
        """
        Gets a mean location for a list of indices
        that is defined!
        """
        lon_arr = []
        lat_arr = []
        if len(idx_lst) == 0:
            raise ValueError('ERROR! List of points is empty!')

        for idx in idx_lst:
            map_idx = self.get_map_index(idx)
            lon_arr.append(map_idx['lon'])
            lat_arr.append(map_idx['lat'])
        mean_lat = np.mean(lat_arr)

        if max(lon_arr)-min(lon_arr) > 180:
            lon_arr = np.array(lon_arr)
            lon_arr[lon_arr < 0] = lon_arr[lon_arr < 0]+360

        mean_lon = np.mean(lon_arr)
        if mean_lon > 180:
            mean_lon -= 360

        nearest_locs = grid.haversine(mean_lon, mean_lat,
                                      self.def_locs[:, 0],
                                      self.def_locs[:, 1],
                                      radius=1)
        idx_min = np.argmin(nearest_locs)
        mean_lon = self.def_locs[idx_min, 0]
        mean_lat = self.def_locs[idx_min, 1]

        return (mean_lon, mean_lat)

    def get_locations_in_range(self, def_map,
                               lon_range=None,
                               lat_range=None,
                               rect_grid=False):
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

        mmap = sput.get_locations_in_range(def_map=def_map,
                                           lon_range=lon_range,
                                           lat_range=lat_range)

        # Return these indices (NOT points!!!) that are defined
        if rect_grid:
            idx_lst = np.where(self.flatten_array(
                mmap, time=False, check=False) > 0)[0]  # TODO check for better solution!
        else:
            defined_points = np.nonzero(self.mask.data)[0]
            point_lst_range = np.where(~np.isnan(mmap.data))[0]
            def_point_list = np.intersect1d(defined_points, point_lst_range)
            idx_lst = self.get_idx_point_lst(point_lst=def_point_list)

        return idx_lst, mmap

    def get_n_ids(self, loc, num_nn=3):
        """
        Gets for a specific location, the neighboring lats and lons ids.
        ----
        Args:
        loc: (float, float) provided as lon, lat values
        """
        slon, slat = loc
        # lon = self.grid['lon']
        # lat = self.grid['lat']
        # sidx = self.get_index_for_coord(lon=slon, lat=slat)
        # sidx_r = self.get_index_for_coord(lon=slon + self.grid_step, lat=slat)
        # sidx_t = self.get_index_for_coord(lon=slon, lat=slat + self.grid_step)

        nearest_locs = grid.haversine(slon, slat, self.def_locs[:, 0],
                                      self.def_locs[:, 1], radius=1)
        idx_sort = np.argsort(nearest_locs)
        n_idx = []
        for idx in range(num_nn):
            sidx_t = self.get_index_for_coord(lon=self.def_locs[idx_sort[idx], 0],
                                              lat=self.def_locs[idx_sort[idx], 1])
            n_idx.append(sidx_t)
        return n_idx

    def use_time_snippets(self, time_snippets):
        """Cut time snippets from dataset and concatenate them.

        Parameters:
        -----------
        time_snippets: np.datetime64  (n,2)
            Array of n time snippets with dimension (n,2).

        Returns:
        --------
        xr.Dataset with concatenate times
        """
        ds_lst = []
        for time_range in time_snippets:
            ds_lst.append(self.ds.sel(
                time=slice(time_range[0], time_range[1])))

        self.ds = xr.merge(ds_lst)

        return self.ds

    def apply_timemean(self, timemean=None):
        self.ds = tu.apply_timemean(ds=self.ds,
                                    timemean=timemean)
        return self.ds

    def get_max(self, var_name=None):
        if var_name is None:
            var_name = self.var_name
        maxval = self.ds[var_name].where(self.ds[var_name] == self.ds[var_name].max(),
                                         drop=True).squeeze()
        lon = float(maxval.lon)
        lat = float(maxval.lat)
        tp = maxval.time

        return {'lon': lon, 'lat': lat, 'tp': tp}

    def get_min(self, var_name=None):
        if var_name is None:
            var_name = self.var_name
        maxval = self.ds[var_name].where(self.ds[var_name] == self.ds[var_name].min(),
                                         drop=True).squeeze()
        lon = float(maxval.lon)
        lat = float(maxval.lat)
        tp = maxval.time

        return {'lon': lon, 'lat': lat, 'tp': tp}

    def compute_anomalies(self, dataarray=None, group='dayofyear'):
        """Calculate anomalies.

        Parameters:
        -----
        dataarray: xr.DataArray
            Dataarray to compute anomalies from.
        group: str
            time group the anomalies are calculated over, i.e. 'month', 'day', 'dayofyear'

        Return:
        -------
        anomalies: xr.dataarray
        """
        reload(tu)
        if dataarray is None:
            dataarray = self.ds[self.var_name]
        anomalies = tu.compute_anomalies(dataarray=dataarray, group=group)

        return anomalies

    def get_idx_region(self, region_dict, def_map=None):
        """
        Gets the indices for a specific dictionary that has lon/lat_range as keys.
        E.g. can be applied to get all indices of the South American monsoon defined by Wang/EE.
        """
        if def_map is None:
            def_map = self.mask
            if def_map is None:
                raise ValueError(
                    "ERROR mask is None! Check if mask is computed properly!")

        lon_range = region_dict['lon_range']
        lat_range = region_dict['lat_range']
        ids, mmap = self.get_locations_in_range(lon_range=lon_range,
                                                lat_range=lat_range,
                                                def_map=def_map
                                                )
        return ids, mmap

    def get_sel_tps_ds(self, tps):
        ds_sel = tu.get_sel_tps_ds(ds=self.ds, tps=tps)

        return ds_sel

    def detrend(self, dim='time', deg=1):
        """Detrend dataarray.
        Args:
            dim (str, optional): [description]. Defaults to 'time'.
            deg (int, optional): [description]. Defaults to 1.
        """
        print('Detrending data...')
        da_detrend = tu.detrend_dim(self.ds[self.var_name], dim=dim, deg=deg)
        self.ds[self.var_name] = da_detrend
        self.ds.attrs['detrended'] = 'True'
        print('... finished!')
        return


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

    def __init__(self, data_nc=None, load_nc=None,
                 var_name=None, time_range=None,
                 lon_range=[-180, 180], lat_range=[-90, 90],
                 grid_step=1, grid_type='gaussian',
                 month_range=None, timemean=None,
                 lsm=False, climatology="dayofyear",
                 detrend=False, **kwargs):

        super().__init__(data_nc=data_nc, load_nc=load_nc,
                         var_name=var_name, time_range=time_range,
                         lon_range=lon_range, lat_range=lat_range,
                         grid_step=grid_step, grid_type=grid_type,
                         month_range=month_range, timemean=timemean,
                         lsm=lsm, detrend=detrend, **kwargs)

        # compute anomalies if not given in nc file
        if "anomalies" in self.vars:
            print("Anomalies are already stored in dataset.")
        elif var_name is None:
            raise ValueError("Specify varname to compute anomalies.")
        else:
            print(f"Compute anomalies for variable {self.var_name}.")
            da = self.ds[self.var_name]
            da = self.compute_anomalies(da, group=climatology)
            da.attrs = {"var_name": self.var_name}
            self.ds['anomalies'] = da

        # set var name to "anomalies" in order to run network on anomalies
        self.var_name = 'anomalies'


class EvsDataset(BaseDataset):

    def __init__(self, data_nc=None, load_nc=None,
                 var_name=None,
                 time_range=None,
                 lon_range=[-180, 180], lat_range=[-90, 90],
                 grid_step=1,
                 grid_type='gaussian',
                 month_range=None,
                 large_ds=False,
                 lsm=False,
                 q=0.95,
                 min_evs=20,
                 min_treshold=1,
                 th_eev=15,
                 rrevs=False,
                 can=False,
                 timemean=None,
                 **kwargs):

        super().__init__(data_nc=data_nc, load_nc=load_nc,
                         var_name=var_name, time_range=time_range,
                         lon_range=lon_range, lat_range=lat_range,
                         grid_step=grid_step, grid_type=grid_type,
                         month_range=month_range, large_ds=large_ds,
                         timemean=timemean, can=can,
                         lsm=lsm, **kwargs)

        # Event synchronization
        if data_nc is not None:

            self.q = q
            self.min_evs = min_evs
            self.min_treshold = min_treshold
            self.th_eev = th_eev
            self.can = can

        else:
            self.load_evs_attrs()

        if 'rrevs' in kwargs:
            rrevs = kwargs.pop('rrevs')
        # compute event synch if not given in nc file
        if "evs" in self.vars:
            print("Evs are already stored in dataset.")
        elif var_name is None:
            raise ValueError("Specify varname to compute event sync.")
        else:
            print(
                f"Compute Event synchronization for variable {self.var_name}.")
            rrevs = True

        if rrevs is True:
            if self.can:
                var_name = 'an'
            else:
                var_name = self.var_name
            self.ds = self.create_evs_ds(
                var_name=var_name,
                th=self.min_treshold,
                q=self.q,
                min_evs=self.min_evs,
                th_eev=self.th_eev)
        else:
            self.mask = self.get_es_mask(self.ds['evs'], min_evs=self.min_evs)

        self.vars = []
        for name, da in self.ds.data_vars.items():
            self.vars.append(name)

        # Flatten index in map
        self.indices_flat, self.idx_map = self.init_mask_idx()

    def create_evs_ds(self,
                      var_name,
                      th=1,
                      q=0.95,
                      th_eev=15,
                      min_evs=20):

        da_es, self.mask = self.compute_event_time_series(
            var_name=var_name,
            th=th,
            q=q,
            min_evs=min_evs,
            th_eev=th_eev)
        da_es.attrs = {"var_name": var_name}

        self.set_ds_attrs(ds=da_es)
        self.ds['evs'] = da_es

        return self.ds

    def get_q_maps(self, var_name, q=0.95):

        if var_name is None:
            var_name = self.var_name
        print(f'Apply Event Series on variable {var_name}')
        dataarray = self.ds[var_name]

        dataarray = self.ds[var_name]
        # Remove days without rain
        data_above_th = dataarray.where(dataarray > self.th)
        # Gives the quanile value for each cell
        q_val_map = data_above_th.quantile(self.q, dim='time')
        # Set values below quantile to 0
        data_above_quantile = xr.where(
            data_above_th > q_val_map[:], data_above_th, np.nan)
        # Set values to 0 that have not at least the value th_eev
        data_above_quantile = xr.where(
            data_above_quantile > self.th_eev, data_above_quantile, np.nan)
        ee_map = data_above_quantile.count(dim='time')

        rel_frac_q_map = data_above_quantile.sum(
            dim='time') / dataarray.sum(dim='time')

        return q_val_map, ee_map, data_above_quantile, rel_frac_q_map

    def get_month_range_data_evs(self, dataset,
                                 start_month='Jan',
                                 end_month='Dec',
                                 shift=0):
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
        times = dataset['time']
        start_year, end_year = tu.get_sy_ey_time(times,
                                                 sy=None,
                                                 ey=None)
        print(f'Get month range data from year {start_year} to {end_year}!')

        data = dataset[self.var_name]

        for idx_y, year in enumerate(np.arange(start_year, end_year)):

            print(f'Compute Year {year} for months {start_month}, {end_month}')
            start_date, end_date = tu.get_start_end_date_year_shift(
                sm=start_month,
                em=end_month,
                sy=year,
                ey=year,
                shift=shift
            )

            arr_1_year = data.sel(time=slice(start_date, end_date))

            # Set shift days before and after to zero
            start_date_before = start_date + np.timedelta64(int(shift), 'D')
            end_data_after = end_date - np.timedelta64(int(shift), 'D')
            rest_data_before = arr_1_year.sel(
                time=slice(start_date_before, start_date))
            rest_data_after = arr_1_year.sel(
                time=slice(end_date, end_data_after))
            data_0_before = xr.zeros_like(rest_data_before)
            data_0_after = xr.zeros_like(rest_data_after)

            arr_1_year.loc[dict(time=slice(
                start_date_before, start_date))] = data_0_before
            arr_1_year.loc[dict(time=slice(
                end_date, end_data_after))] = data_0_after

            if idx_y == 0:
                all_year = arr_1_year
            else:
                all_year = xr.merge([all_year, arr_1_year])

        return all_year

    def compute_event_time_series(self,
                                  var_name=None,
                                  th=1,
                                  q=0.95,
                                  th_eev=15,
                                  min_evs=20):
        self.th = th
        self.q = q
        self.th_eev = th_eev
        self.min_evs = min_evs

        if self.q > 1 or self.q < 0:
            raise ValueError(f'ERROR! q = {self.q} has to be in range [0, 1]!')

        # Compute percentile data, remove all values below percentile, but with a minimum of threshold q
        print(
            f"Start remove values below q={self.q} and at least with q_value >= {self.th_eev} ...")
        q_mask, num_non_nan_occurence, data_above_quantile, rel_frac_q_map = self.get_q_maps(var_name=var_name,
                                                                                             q=self.q)

        # Get relative amount of q rainfall to total yearly rainfall

        # Create mask for which cells are left out
        print(f"Remove cells without min number of events: {self.min_evs}")
        mask = (num_non_nan_occurence > self.min_evs)
        final_data = data_above_quantile.where(mask, np.nan)

        print("Now create binary event series!")
        event_series = xr.where(final_data[:] > 0, 1, 0)
        print("Done!")
        event_series = event_series.rename('evs')
        # Create new mask for dataset
        self.q_mask = q_mask

        self.num_eev_map = num_non_nan_occurence
        self.rel_frac_q_map = rel_frac_q_map

        # Masked values are areas with no events!
        mask = xr.where(num_non_nan_occurence > self.min_evs, 1, 0)

        return event_series, mask

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
                th_eev=self.th_eev)
            self.set_ds_attrs(ds=da_es)
            self.ds[f'evs_q{q}'] = da_es

        return self.ds

    def get_es_mask(self, data_evs, min_evs):
        num_non_nan_occurence = data_evs.where(data_evs == 1).count(dim='time')
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
            "an": int(self.can)
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
            print("File" + file + " already exists!")
            os.rename(file, file + "_backup")
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        ds_temp = self.set_ds_attrs(self.ds)
        ds_temp.to_netcdf(file)

        return None

    def load_evs_attrs(self):
        self.q = self.ds.attrs['q']
        self.min_evs = self.ds.attrs['min_evs']
        self.min_treshold = self.ds.attrs['min_threshold']
        self.th_eev = self.ds.attrs['th_eev']
        if 'an' in self.ds.attrs:  # To account for old version saved files!
            self.can = bool(self.ds.attrs['an'])
        else:
            self.can = False

    def randomize_spatio_temporal_data_yearly(self, data,
                                              var=None,
                                              start_year=None,
                                              end_year=None,
                                              sm_arr=['Jan'],
                                              em_arr=['Dec'],
                                              set_rest_zero=False,
                                              full_rnd=False,
                                              seed=0):
        """
        Permutates randomly time series for every grid location.
        Keeps the year in which the events did occur.
        """

        if len(sm_arr) != len(em_arr):
            raise ValueError(
                "ERROR! Start month array and end month array not of the same length!")
        if len(sm_arr) > 1 and set_rest_zero is True:
            raise ValueError(
                "Set Time Zeros 0 and shuffle for to time periods is not possible!")
        times = self.ds['time']
        if start_year is None:
            start_year = int(times[0].time.dt.year)
        if end_year is None:
            end_year = int(times[-1].time.dt.year) + 1
        if var is None:
            var = self.var

        with gut.temp_seed():
            if full_rnd is True:
                print(
                    f"WARNING! Fully randomized time Series of {var}!")

                start_date = times.data[0]
                end_date = times.data[-1]
                arr_data = data.sel(time=slice(start_date, end_date))
                arr_rnd = self.randomize_spatio_temporal_data_full(
                    arr_data.data, axis=0)
                data.loc[dict(time=slice(start_date, end_date))] = arr_rnd
            else:
                for idx in range(len(sm_arr)):
                    sm = sm_arr[idx]
                    em = em_arr[idx]
                    print(
                        f"WARNING! Time Series of {var} are for {sm} to {em} randomized!")

                    for idx, year in enumerate(np.arange(start_year, end_year)):

                        print(f'Shuffle Year {year} for months {sm}, {em}')
                        smi = self._get_index_of_month(sm)+1
                        emi = self._get_index_of_month(em)+1
                        start_date = f'{smi}-01-{year}'
                        if em == 'Feb':
                            end_day = 28
                        elif em in ['Jan', 'Mar', 'May', 'Jul', 'Aug', 'Oct', 'Dec']:
                            end_day = 31
                        else:
                            end_day = 30

                        ey = copy.deepcopy(year)
                        if emi < smi:
                            ey = year + 1
                        end_date = f'{emi}-{end_day}-{ey}'
                        if emi < 10:
                            end_date = f'0{emi}-{end_day}-{ey}'

                        arr_1_year = data.sel(time=slice(start_date, end_date))
                        # arr_1_year_rnd=np.random.permutation(arr_1_year.data)
                        arr_1_year_rnd = self.randomize_spatio_temporal_data_full(
                            arr_1_year.data, axis=0)

                        arr_1_year.data = arr_1_year_rnd
                        # if idx == 0:
                        #     all_year = arr_1_year
                        # else:
                        #     all_year = xr.merge([all_year, arr_1_year])
                        data.loc[dict(time=slice(
                            start_date, end_date))] = arr_1_year_rnd

                        if set_rest_zero is True:
                            print("Warning: Set Rest to Zero!")
                            if emi >= smi:  # E.g. for Jun-Sep
                                start_date_before = f'01-01-{year}'
                                end_data_after = f'12-31-{year}'
                                rest_data_before = data.sel(
                                    time=slice(start_date_before, start_date))
                                rest_data_after = data.sel(
                                    time=slice(end_date, end_data_after))
                                data_before = xr.zeros_like(rest_data_before)
                                data_after = xr.zeros_like(rest_data_after)

                                data.loc[dict(time=slice(
                                    start_date_before, start_date))] = data_before
                                data.loc[dict(time=slice(
                                    end_date, end_data_after))] = data_after
                            else:
                                # Year not ey!
                                end_date = f'{emi}-{end_day}-{year}'
                                if emi < 10:
                                    end_date = f'0{emi}-{end_day}-{year}'
                                rest_data_between = data.sel(
                                    time=slice(end_date, start_date))
                                data_between = xr.zeros_like(rest_data_between)
                                data.loc[dict(time=slice(
                                    end_date, start_date))] = data_between

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


class BaseRectDataset(BaseDataset):
    """Class that defines a classical rectangular dataset, that is stored as as classical
    nc file. It has however the same functions of BaseDataset but is not defined of the
    grid of BaseDataset, but on the standard lon, lat grid that is by default used in nc-files.
    i.e. for different longitude and latitude boxes

    Args:
        BaseDataset (class): Base Dataset

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
                 large_ds=False,
                 can=False,
                 **kwargs
                 ):

        if data_nc is not None and load_nc is not None:
            raise ValueError("Specify either data or load file.")
        # initialize dataset
        elif data_nc is not None:
            # check if file exists
            if not os.path.exists(data_nc):
                PATH = os.path.dirname(os.path.abspath(__file__))
                print(f"You are here: {PATH}!")
                raise ValueError(f"File does not exist {data_nc}!")
            self.var_name = var_name
            self.grid_step = grid_step
            ds = self.open_ds(nc_file=data_nc,
                              var_name=var_name,
                              lat_range=lat_range, lon_range=lon_range,
                              time_range=time_range,
                              grid_step=grid_step,
                              large_ds=large_ds,
                              **kwargs)
            self.time_range, self.lon_range, self.lat_range = self.get_spatio_temp_range(
                ds)
            if month_range is not None:
                ds = self.get_month_range_data(ds,
                                               start_month=month_range[0],
                                               end_month=month_range[1])
            self.ds = ds
        else:
            self.load(load_nc=load_nc)
        # select a main var name
        self.vars = []
        for name, da in self.ds.data_vars.items():
            self.vars.append(name)
        self.var_name = var_name if var_name is not None else self.vars[0]
        self.indices_flat, self.idx_map = self.init_mask_idx()

        # Compute Anomalies if needed
        self.can = can
        if self.can is True:
            if 'an' not in self.vars:
                an_types = kwargs.pop('an_types', ['dayofyear'])
                for an_type in an_types:
                    self.ds[f'an_{an_type}'] = self.compute_anomalies(self.ds[self.var_name],
                                                                      group=an_type)

    def open_ds(self,
                nc_file,
                var_name,
                time_range=None,
                grid_step=1,
                large_ds=True,
                lon_range=[-180, 180],
                lat_range=[-90, 90],
                **kwargs,
                ):
        print("Start processing data!")
        if large_ds:
            ds = xr.open_dataset(nc_file, chunks={"time": 100})
        else:
            ds = xr.open_dataset(nc_file)
        ds = self.check_dimensions(ds, **kwargs)
        ds = self.rename_var(ds)
        da = ds[var_name]
        da = da.transpose('time', 'lat', 'lon')
        da = self.get_data_timerange(da, time_range)
        if max(da.lon) > 180:
            print("Shift longitude in Preprocessing!")
            da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
        da = self.common_grid(dataarray=da,
                              grid_step=grid_step)

        ds.unify_chunks()
        if lon_range != [-180, 180] and lat_range != [-90, 90]:
            ds = self.cut_map(ds, lon_range, lat_range)

        ds = da.to_dataset(name=var_name)

        # For the moment, all values are defined! TODO implement lsm
        self.mask = xr.DataArray(
            data=np.ones_like(da[0].data),
            dims=da.sel(time=da.time[0]).dims,
            coords=da.sel(time=da.time[0]).coords,
            name='mask')

        print('Finished processing data')

        return ds

    def load(self, load_nc):
        """Load dataset object from file.

        Parameters:
        ----------
        ds: xr.Dataset
            Dataset object containing the preprocessed dataset

        """
        # check if file exists
        if not os.path.exists(load_nc):
            PATH = os.path.dirname(os.path.abspath(__file__))
            print(f"You are here: {PATH}!")
            raise ValueError(f"File does not exist {load_nc}!")

        print(f'Load Dataset: {load_nc}')
        ds = xr.open_dataset(load_nc)

        self.time_range, self.lon_range, self.lat_range = self.get_spatio_temp_range(
            ds)

        self.grid_step = ds.attrs['grid_step']
        self.info_dict = ds.attrs  # TODO
        # Read and create grid class

        for name, da in ds.data_vars.items():
            print("Variables in dataset: ", name)

        mask = np.ones_like(ds[name][0].data, dtype=bool)
        for idx, t in enumerate(ds.time):
            mask *= np.isnan(ds[name].sel(time=t).data)

        self.mask = xr.DataArray(
            data=xr.where(mask == False, 1, np.NaN),
            dims=da.sel(time=da.time[0]).dims,
            coords=da.sel(time=da.time[0]).coords,
            name='lsm')

        ds = self.check_time(ds)

        self.ds = ds

        return None

    def save(self, filepath):
        """Save the dataset class object to file.
        Args:
        ----
        filepath: str
        """
        if os.path.exists(filepath):
            print("File" + filepath + " already exists!")
            os.rename(filepath, filepath + "_backup")

        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        param_class = {
            "grid_step": self.grid_step,
        }
        ds_temp = self.ds
        ds_temp.attrs = param_class
        ds_temp.to_netcdf(filepath)
        print(f"File {filepath} written!", flush=True)
        return None

    def get_index_for_coord(self, lon, lat):
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        indices_flat = np.arange(0, np.count_nonzero(mask_arr), 1, dtype=int)

        idx_map = self.get_map(indices_flat, name='idx_flat')

        idx = idx_map.sel(lat=lat, lon=lon, method='nearest')
        if np.isnan(idx):
            print(
                f"The lon {lon} lat {lat} index is not within mask, i.e. NaN!",
                flush=True)
            return idx
        else:
            return int(idx)
