import climnet.utils.time_utils as tu
import numpy as np
import pandas as pd
import xarray as xr
import scipy.interpolate as interp
import copy
from importlib import reload
import climnet.utils.general_utils as ut
import climnet.utils.spatial_mean as spm
import climnet.tsa.filters as flt


def compute_rm(da, rm_val, dim='time', sm='Jan', em='Dec'):
    """This function computes a rolling mean on an input
    xarray dataarray. The first and last rm_val elements
    of the time series are deleted.py

    Args:
        da (xr.dataarray): dataarray of the input dataset
        rm_val (int): size of the rolling mean

    Returns:
        [type]: [description]
    """
    reload(tu)
    times = da.time
    if dim != 'time':
        da_rm = da.rolling(time=rm_val, center=True).mean(
            skipna=True).dropna(dim, how='all')
    else:
        start_year, end_year = tu.get_sy_ey_time(times)
        all_year_arr = []
        for idx, year in enumerate(np.arange(start_year, end_year)):
            print(f'Compute RM Year {year}')
            start_date, end_date = tu.get_start_end_date(
                sy=year,
                ey=year,
                sm=sm,
                em=em)
            arr_1_year = da.sel(time=slice(start_date, end_date))
            arr_1_year_rm = arr_1_year.rolling(
                time=rm_val, center=True, min_periods=1).mean(skipna=True).dropna(dim=dim,
                                                                                  how='all')
            all_year_arr.append(arr_1_year_rm)
        da_rm_ds = xr.merge(all_year_arr)

        for name, da in da_rm_ds.data_vars.items():
            var_name = name
        da_rm = da_rm_ds[var_name]

    return da_rm


def get_map_for_def_map(data_map, def_map):
    if data_map.shape[1:] != def_map.shape[:]:
        raise ValueError("Error! Not same shape of def map and data map!")
    nan_map = xr.where(def_map > 0, data_map, np.nan)
    return nan_map


def interp_fib2gaus(dataarray, grid_step=2.5):
    """Interpolate dataarray on Fibonacci grid to Gaussian grid.

    Args:
        dataarray (xr.DataArray): Dataarray with fibonacci grid
        grid_step (float, optional): Grid step of new Gaussian grid. Defaults to 2.5.

    Returns:
        dataarray_gaus: Dataarray interpolated on Gaussian grid.
    """
    # Create gaussian grid
    lon_gaus = np.arange(np.round_(dataarray.coords['lon'].min()),
                         dataarray.coords['lon'].max(),
                         grid_step)
    lat_gaus = np.arange(np.round_(dataarray.coords['lat'].min()),
                         dataarray.coords['lat'].max(),
                         grid_step)
    lon_mesh, lat_mesh = np.meshgrid(lon_gaus, lat_gaus)
    new_points = np.array([lon_mesh.flatten(), lat_mesh.flatten()]).T
    origin_points = np.array(
        [dataarray.coords['lon'], dataarray.coords['lat']]).T

    # Interpolate
    if len(dataarray.data.shape) < 2:  # If there is not time dimension
        origin_values = dataarray.data.flatten()
        assert len(origin_values) == origin_points.shape[0]
        new_values = interp.griddata(origin_points, origin_values, new_points,
                                     method='nearest')
        new_data = new_values.reshape(len(lat_gaus), len(lon_gaus))
        coordinates = dict(lon=lon_gaus, lat=lat_gaus)
        dims = ['lat', 'lon']
    else:  # with time dimension
        new_data = []
        for idx, t in enumerate(dataarray.time):
            origin_values = dataarray.sel(time=t.data).data.flatten()
            assert len(origin_values) == origin_points.shape[0]
            new_values = interp.griddata(origin_points, origin_values,
                                         new_points,
                                         method='nearest')
            new_data.append(
                new_values.reshape(len(lat_gaus), len(lon_gaus))
            )
        coordinates = dict(time=dataarray.time.data,
                           lon=lon_gaus, lat=lat_gaus)
        dims = ['time', 'lat', 'lon']
        new_data = np.array(new_data)

    da_gaus = xr.DataArray(
        data=new_data,
        dims=dims,
        coords=coordinates,
        name=dataarray.name)

    return da_gaus


def cut_map(ds, lon_range=[-180, 180], lat_range=[-90, 90], shortest=True):
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
    if (max(lon_range) - min(lon_range) <= 180) or shortest is False:
        ds_cut = ds.sel(
            lon=slice(np.min(lon_range), np.max(lon_range)),
            lat=slice(np.min(lat_range), np.max(lat_range))
        )
    else:
        # To account for areas that lay at the border of -180 to 180
        ds_cut = ds.sel(
            lon=ds.lon[(ds.lon < min(lon_range)) | (ds.lon > max(lon_range))],
            lat=slice(np.min(lat_range), np.max(lat_range))
        )

    return ds_cut


def compute_zonal_mean(ds, ):
    zonal_mean = ds.mean(dim='lat', skipna=True)

    return zonal_mean


def compute_meridional_mean(ds):
    lats_rad = np.deg2rad(ds['lat'])
    weighted_meridional_mean = ds.mean(
        dim='lon', skipna=True) # * np.cos(lats_rad)

    weighted_meridional_std = ds.std(
        dim='lon', skipna=True)
    return weighted_meridional_mean, weighted_meridional_std


def compute_meridional_quantile(ds, q):
    lats_rad = np.deg2rad(ds['lat'])
    weighted_meridional_mean = ds.quantile(
        q=q, dim='lon', skipna=True)  # * np.cos(lats_rad)

    return weighted_meridional_mean


def get_map4indices(ds, indices):
    da = xr.ones_like(ds.ds['anomalies']) * np.NAN
    da[:, indices] = ds.ds['anomalies'].sel(points=indices).data

    return da


def get_locations_in_range(def_map, lon_range, lat_range):
    """Returns a map with nans at the positions that are not within lon_range-lat_range

    Args:
        def_map (xr.DataArray): dataarray, must contain lat and lon definition
        lon_range (list): list of lons
        lat_range (list): list of lats

    Returns:
        xr.DataArray: masked xr.DataArray.
    """
    if (max(lon_range) - min(lon_range) < 180):
        mask = (
            (def_map['lat'] >= min(lat_range))
            & (def_map['lat'] <= max(lat_range))
            & (def_map['lon'] >= min(lon_range))
            & (def_map['lon'] <= max(lon_range))
        )
    else:   # To account for areas that lay at the border of -180 to 180
        mask = (
            (def_map['lat'] >= min(lat_range))
            & (def_map['lat'] <= max(lat_range))
            & ((def_map['lon'] <= min(lon_range)) | (def_map['lon'] >= max(lon_range)))
        )
    mmap = xr.where(mask, def_map, np.nan)
    return mmap
