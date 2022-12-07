
import numpy as np
import xarray as xr
import climnet.utils.spatial_utils as sput
from climnet.datasets.dataset import BaseDataset


class EmptyDataset(BaseDataset):
    def __init__(
        self,
        load_nc=None,
        time_range=None,
        lon_range=[-180, 180],
        lat_range=[-90, 90],
        grid_step=1,
        grid_type="gaussian",
        **kwargs,
    ):

        # Event synchronization
        if load_nc is None:
            self.grid_step = grid_step
            self.grid_type = grid_type
            self.info_dict = kwargs

            # get Grid
            self.GridClass = self.create_grid(
                grid_type=self.grid_type, **kwargs,)
            ds = self.add_ds()
            if lon_range != [-180, 180] and lat_range != [-90, 90]:
                ds = self.cut_map(ds=ds, lon_range=lon_range,
                                  lat_range=lat_range)

            self.grid = self.GridClass.cut_grid(
                lon_range=lon_range,
                lat_range=lat_range,
            )

        else:
            self.load()

    def add_ds(self, data, var_name, times):
        """Adds a dataset to an empty dataset

        Args:
            data (np.array): array (num_tps, num_points)
            var_name (str): name of the variables.
            times (np.datetime Array): array of times for the dataset.

        Raises:
            ValueError:
            ValueError:

        Returns:
            ds: xr.Dataset
        """
        dims = ["time", "points"]
        new_values = np.array(data)
        new_points = np.array([self.grid["lon"], self.grid["lat"]]).T
        num_tps, num_points = data.shape
        if len(times) != num_tps:
            raise ValueError(f'Number of tps {num_tps} != times {len(times)}!')
        if len(new_points) != num_points:
            raise ValueError(
                f'Number of points {num_points} != grid points {len(new_points)}!')

        coordinates = sput.create_new_coordinates(
            times=times,
            lons=new_points[:, 0], lats=new_points[:, 1]
        )
        new_dataarray = xr.DataArray(
            data=new_values, dims=dims, coords=coordinates, name=var_name
        )

        self.ds = new_dataarray.to_dataset()
        self.init_mask(da=new_dataarray)
        self.indices_flat, self.idx_map = self.init_map_indices()

        return self.ds
