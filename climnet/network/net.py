"""Climate network class."""
import sys
import os
import numpy as np
import xarray as xr
import scipy.stats as stat
import scipy.sparse as sparse
import scipy.special as special
from joblib import Parallel, delayed
import multiprocessing as mpi
from tqdm import tqdm
import copy
import climnet.network.link_bundles as lb
import climnet.tsa.event_synchronization as es
import climnet.grid as grid
from climnet.utils.statistic_utils import holm
import climnet.utils.statistic_utils as sut
import climnet.utils.time_utils as tu
import climnet.network.network_functions as nwf
import climnet.utils.general_utils as gut
from climnet.tsa import iaaft

PATH = os.path.dirname(os.path.abspath(__file__))


def load(dataset, fname):
    """Load stored climate networks from .npz file.

    Parameters:
    -----------
    dataset: climnet.dataset.BaseDataset
        Dataset object.
    fname: str
        Filename of .npz file

    Returns:
    --------
    Net: climnet.net.BaseClimNet
        Climate network object.
    """
    print(f"Load network: {fname}")
    with np.load(fname, allow_pickle=True) as data:
        class_type = data["type"][0]

        # correlation based climate network
        if class_type == "correlation":
            Net = CorrClimNet(
                dataset,
                corr_method=data["corr_method"][0],
                confidence=data["confidence"][0],
                stat_test=data["stat_test"][0],
                significance_test=data["significance_test"][0],
            )
            Net.corr = data["corr"]
            Net.pvalue = data["pvalue"]

        # event synchronization network
        elif class_type == "evs":
            if "weighted" in list(data.keys()):
                weighted = data["weighted"][0]
            else:
                weighted = False
            Net = EventSyncClimNet(dataset, taumax=data["taumax"][0], weighted=weighted)
            if Net.weighted:
                corr = data["corr"]
                Net.corr = np.where(corr == np.inf, 0, corr)
        else:
            raise ValueError(f"Class type {class_type} not implemented!")
        Net.adjacency = data["adjacency"]
        Net.lb = bool(data["lb"][0])

        g_N = len(Net.adjacency)
        num_ds_nodes = len(Net.ds.indices_flat)
        if g_N != num_ds_nodes:
            raise ValueError(
                f"Error network and dataset not of the same dimension net: {g_N} != ds {num_ds_nodes}!"
            )
    return Net


class BaseClimNet:
    """ Climate Network class.
    Args:
    ----------
    dataset: BaseDataset object
        Dataset
    """

    def __init__(self, dataset):
        self.ds = copy.deepcopy(dataset)
        self.lb = False
        self.adjacency = None
        self.type = "Base"
        self.sparsity = None

    def create(self):
        """Abstract create function."""
        return

    def save(self, fname):
        """Abstract save function."""
        return

    def check_densityApprox(self, adjacency, idx, nn_points_bw=1):
        """KDE approximation of the given adjacency and node idx."""
        coord_deg, coord_rad, map_idx = self.ds.get_coordinates_flatten()
        all_link_idx_this_node = np.where(adjacency[idx, :] > 0)[0]
        if len(all_link_idx_this_node) == 0:
            raise ValueError("ERROR! This idx does not contain any link!")

        link_coord = coord_rad[all_link_idx_this_node]
        bandwidth = self.obtain_kde_bandwidth(nn_points_bw=nn_points_bw)
        Z = lb.spherical_kde(link_coord, coord_rad, bandwidth)

        return {"z": Z, "link_coord": link_coord, "all_link": all_link_idx_this_node}

    def obtain_kde_bandwidth(self, nn_points_bw=1):
        """KDE bandwidth is set to nn_points_bw*max_dist_of_points.

        Parameters:
        ----------
        nn_points_bw: int
            Number of next-neighbor points of KDE bandwidth.
        """
        dist_eq = grid.degree2distance_equator(
            self.ds.grid_step, radius=grid.RADIUS_EARTH
        )
        bandwidth = nn_points_bw * dist_eq / grid.RADIUS_EARTH

        return bandwidth

    def _link_bundles(
        self,
        num_rand_permutations,
        num_cpus=mpi.cpu_count(),
        nn_points_bw=None,
        link_bundle_folder=None,
    ):
        """Significant test for adjacency. """
        # Get coordinates of all nodes
        coord_deg, coord_rad, map_idx = self.ds.get_coordinates_flatten()

        # First compute Null Model of old adjacency matrix
        if link_bundle_folder is None:
            link_bundle_folder = PATH + f"/link_bundles/{self.ds.var_name}/"
        else:
            link_bundle_folder = PATH + f"/link_bundles/{link_bundle_folder}/"
        null_model_filename = f"link_bundle_null_model_{self.ds.var_name}"

        # Set KDE bandwidth to 2*max_dist_of_points
        if nn_points_bw is not None:
            dist_eq = grid.degree2distance_equator(
                self.ds.grid_step, radius=grid.RADIUS_EARTH
            )
            bandwidth = nn_points_bw * dist_eq / grid.RADIUS_EARTH
        else:
            bandwidth = None  # Is computed later based on Scott's rule of thumb!

        print(f"Start computing null model of link bundles using {bandwidth}!")
        lb.link_bundle_null_model(
            self.adjacency,
            coord_rad,
            link_bundle_folder=link_bundle_folder,
            filename=null_model_filename,
            num_rand_permutations=num_rand_permutations,
            num_cpus=num_cpus,
            bw=bandwidth,
        )

        # Now compute again adjacency corrected by the null model of the link bundles
        # try:
        print("Now compute new adjacency matrix!")
        adjacency = lb.link_bundle_adj_matrix(
            adj_matrix=self.adjacency,
            coord_rad=coord_rad,
            null_model_folder=link_bundle_folder,
            null_model_filename=null_model_filename,
            bw=bandwidth,
            perc=999,
            num_cpus=num_cpus,
        )
        # except:
        #     print(
        #         "Other jobs for link bundling are not finished yet! Last job will do the rest!"
        #     )
        #     sys.exit()
        self.adjacency = adjacency

        self.lb = True
        return adjacency

    def get_node_degree(self, weighted=False):
        node_degree = []
        corr = self.corr
        if corr.shape != self.adjacency.shape:
            raise ValueError("ERROR! Adjacency and weights not of the same shape!")
        for node in self.adjacency:
            if weighted is True:
                node_degree.append(np.sum(node))
            else:
                node_degree.append(np.count_nonzero(node))

        return np.array(node_degree)

    def get_edgelist(self, weighted=False):
        """Gets the edge list for a given adjacency matrix.

        Args:
            adj_matrix (np.ndarray): 2darray for adjacency
            weighted (bool, optional): Is the network weighted. Defaults to False.

        Returns:
            np.array: list of edges (source, target, (weight)) if weighted=True
        """
        if weighted:
            edges = np.transpose(self.adjacency.nonzero())
            edge_list = []
            for ed in tqdm(edges):
                i, j = ed
                w = self.corr[i, j]
                edge_list.append((i, j, w))
            # edge_list = np.array(edge_list)
        else:
            edge_list = np.transpose(self.adjacency.nonzero())
        B = np.count_nonzero(self.adjacency)
        assert B == len(edge_list)

        return edge_list


class CorrClimNet(BaseClimNet):
    """Correlation based climate network.

    Parameters:
    -----------
    corr_method: str
        Correlation method of network ['spearman', 'pearson'], default: 'spearman'
    threshold: float
        Default: -1.0
    stat_test: str
        Default: 'twosided'
    confidence: float
        Default: 0.99
    significance_test: str
        Method for statistical p-value testing. Default is 'bonf' (Holm-Bonferroni method)
    """

    def __init__(
        self,
        dataset,
        corr_method="spearman",
        threshold=None,
        density=None,
        stat_test="twosided",
        confidence=0.99,
        significance_test="bonf",
    ):
        super().__init__(dataset)
        # set to class variables
        self.type = "correlation"
        self.corr_method = corr_method
        self.threshold = threshold
        self.density = density
        self.confidence = confidence
        self.stat_test = stat_test
        self.corr = None
        self.pvalue = None
        self.significance_test = significance_test
        self.weighted = True

    def create(self, **kwargs):
        """Creates the climate network using a given correlation method.

        Parameters:
        ----------
        **kwargs: dict
            Arguments for self.get_adjacency().

        """
        # spearman's rank correlation
        if self.corr_method == "spearman":
            self.corr, self.pvalue = self.calc_spearman(self.ds, self.stat_test)

        # pearson correlation
        elif self.corr_method == "pearson":
            self.corr, self.pvalue = self.calc_pearson(self.ds)
        else:
            raise ValueError("Choosen correlation method does not exist!")

        self.adjacency = self.get_adjacency(
            self.corr,
            self.pvalue,
            threshold=self.threshold,
            density=self.density,
            confidence=self.confidence,
            significance_test=self.significance_test,
            **kwargs,
        )

        return None

    def save(self, fname):
        """Store adjacency, correlation and pvalues to an .npz file.

        Parameters:
        -----------
        fname: str
            Filename to store .npz
        """
        if os.path.exists(fname):
            print("Warning File" + fname + " already exists! No over writing!")
            os.rename(fname, fname + "_bak")

        np.savez(
            fname,
            corr=self.corr,
            pvalue=self.pvalue,
            adjacency=self.adjacency,
            lb=np.array([self.lb]),
            corr_method=np.array([self.corr_method]),
            threshold=np.array([self.threshold]),
            density=np.array([self.density]),
            confidence=np.array([self.confidence]),
            stat_test=np.array([self.stat_test]),
            significance_test=np.array([self.significance_test]),
            type=np.array([self.type]),
        )
        print(f"Network stored to {fname}!")
        return None

    def calc_spearman(self, dataset, test="onesided"):
        """Spearman correlation of the flattened and remove NaNs object.
        """
        data = dataset.flatten_array()
        print(data.shape)

        corr, pvalue = sut.calc_spearman(data=data, test=test, verbose=True)

        print(f"Created spearman correlation matrix of shape {np.shape(corr)}")
        return corr, pvalue

    def calc_pearson(self, dataset):
        """Pearson correlation of the flattened array."""
        data = dataset.flatten_array()
        print(data.shape)
        # Pearson correlation
        corr, pvalue = sut.calc_pearson(data=data, verbose=True)
        return corr, pvalue

    def get_adjacency(
        self,
        corr,
        pvalue,
        threshold=None,
        density=None,
        confidence=0.95,
        significance_test="bonf",
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
            indices = holm(
                pval_flat, alpha=(1 - confidence), corr_type=significance_test
            )
            mask_list = np.zeros_like(pval_flat)
            mask_list[indices] = 1
            mask_confidence = np.reshape(mask_list, pvalue.shape)
        elif significance_test == "standard":
            mask_confidence = np.where(pvalue <= (1 - confidence), 1, 0)  # p-value test
        elif significance_test == "iaaft":
            # self for read out TODO: delete in the future
            self.corr_distribution = self.get_iaaft_surrogates(
                self.ds, corr_method=self.corr_method, **iaaft_kwargs
            )
            q_corr = np.quantile(np.abs(self.corr_distribution), q=confidence, axis=0)
            mask_confidence = np.where(np.abs(corr) >= q_corr, 1, 0)
        else:
            raise ValueError(f"Method {significance_test} does not exist!")

        # Threhold test on correlation values
        if threshold is not None and density is not None:
            raise ValueError("Threshold and density cannot both be specified!")

        if threshold is not None:
            mask_correlation = np.where(np.abs(corr) >= threshold, 1, 0)
        elif density is not None:
            if density < 0 or density > 1:
                raise ValueError(f"Density must be between 0 and 1 but is {density}!")
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
            mask_correlation = 1

        adjacency = mask_confidence * mask_correlation
        # set diagonal to zero
        for i in range(len(adjacency)):
            adjacency[i, i] = 0
        print("Created adjacency matrix.")

        self.threshold = threshold
        adjacency = adjacency * adjacency.transpose()

        return adjacency

    def get_iaaft_surrogates(
        self,
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
            delayed(self.iaaft_surrogate)(data, corr_method, tol_pc, maxiter, sorttype)
            for s in tqdm(range(num_surrogates))
        )

        return np.array(corr_matrices)

    @staticmethod
    def iaaft_surrogate(
        data, corr_method="pearson", tol_pc=5.0, maxiter=1e6, sorttype="quicksort"
    ):
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

    def link_bundles(
        self,
        num_rand_permutations,
        num_cpus=mpi.cpu_count(),
        nn_points_bw=None,
        link_bundle_folder=None,
    ):
        self.adjacency = self._link_bundles(
            num_rand_permutations=num_rand_permutations,
            num_cpus=num_cpus,
            nn_points_bw=nn_points_bw,
            link_bundle_folder=link_bundle_folder,
        )

        # self.adjacency = nwf.make_network_undirected()

        return self.adjacency


class EventSyncClimNet(BaseClimNet):
    """Correlation based climate network.

    Parameters:
    -----------
    corr_method: str
        Correlation method of network ['spearman', 'pearson'], default: 'spearman'
    threshold: float
        Default: -1.0
    stat_test: str
        Default: 'twosided'
    confidence: float
        Default: 0.99
    significance_test: str
        Method for statistical p-value testing. Default is 'bonf' (Holm-Bonferroni method)
    """

    def __init__(self, dataset, taumax=10, min_num_sync_events=2, weighted=False):
        vars = dataset.vars
        if "evs" not in vars:
            raise ValueError(
                "For EventSyncNet a dataset has to be provided that contains event series!"
            )
        super().__init__(dataset)
        # set to class variables
        self.type = "evs"
        self.taumax = taumax
        self.weighted = weighted
        self.min_num_sync_events = min_num_sync_events
        self.es_filespath = f"{PATH}/../es_files/"
        self.null_model_folder = self.es_filespath + "null_model/"

    def save(self, file):
        """Store adjacency, correlation and pvalues to an .npz file.

        Parameters:
        -----------
        fname: str
            Filename to store .npz
        """
        filepath = os.path.dirname(file)

        if os.path.exists(file):
            print("File" + file + " already exists!")
            os.rename(file, file + "_backup")
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        np.savez_compressed(
            file,
            corr=self.corr,
            #  pvalue=self.pvalue,
            adjacency=self.adjacency,
            lb=np.array([self.lb]),
            type=np.array([self.type]),
            weighted=np.array([self.weighted]),
            taumax=np.array([self.taumax]),
        )
        print(f"Network stored to {file}!")
        return None

    def create(
        self,
        E_matrix_folder=None,
        null_model_file=None,
        q_med=0.5,
        lq=0.25,
        hq=0.75,
        q_sig=0.95,
    ):
        """
        This function has to be called twice, once, to compute the exact numbers of synchronous
        events between two time series, second again to compute the adjacency matrix
        and the link bundles.
        Attention: The number of parrallel jobs that were used for the E_matrix needs to be
        passed correctly to the function.
        """
        # Test if ES data is computed
        if self.ds.ds["evs"] is None:
            raise ValueError("ERROR Event Synchronization data is not computed yet")
        else:
            data_evs = self.ds.ds["evs"]
            num_time_series = self.ds.flatten_array().shape[1]
            if num_time_series == data_evs.shape[1]:
                print("WARNING: The Mask did not remove any values!")
        if E_matrix_folder is None:
            E_matrix_folder = (
                self.es_filespath + f"/E_matrix/{self.ds.var_name}_{self.ds.grid_step}/"
            )
        else:
            E_matrix_folder = self.es_filespath + f"/E_matrix/{E_matrix_folder}"

        if null_model_file is None:
            raise ValueError("ERROR Please provide null_model_file!")
        else:
            null_model_file = self.es_filespath + f"null_model/{null_model_file}"
            if not os.path.exists(null_model_file):
                raise ValueError(f"File {null_model_file} does not exist!")
        print(f"Load null model: {null_model_file}")
        null_model = np.load(null_model_file, allow_pickle=True).item()

        self.event_synchronization_run(
            data_evs=data_evs,
            E_matrix_folder=E_matrix_folder,
            taumax=self.taumax,
            null_model=null_model,
            min_num_sync_events=self.min_num_sync_events,
        )

        self.adjacency, self.corr = self.compute_es_adjacency(
            E_matrix_folder=E_matrix_folder,
            num_time_series=num_time_series,
            weighted=self.weighted,
            q_dict=null_model,
            q_med=q_med,
            lq=lq,
            hq=hq,
            q_sig=q_sig,
        )

        nwf.get_sparsity(M=self.adjacency)
        return None

    def event_synchronization_run(
        self,
        data_evs,
        E_matrix_folder=None,
        taumax=10,
        min_num_sync_events=2,
        null_model=None,
    ):
        """
        Event synchronization definition

        Parameters
        ----------
        data_evs: np.ndarray (lon, lat, timesteps)
            binary time series
        E_matrix_folder: str
            Folder where E-matrix files are to be stored
        taumax: int
            Maximum time delay between two events
        min_num_sync_events: int
            Minimum number of sychronous events between two time series
        null_model_file: str
            File where null model for Adjacency is stored. If None, it will be computed again (costly!)
        Return
        ------
        None
        """

        # for job array
        job_id, num_jobs = gut.get_job_array_ids()

        event_series_matrix = self.ds.flatten_array(
            dataarray=data_evs, check=False
        ).T  # transpose to get array of timeseries!

        if not os.path.exists(E_matrix_folder):
            os.makedirs(E_matrix_folder)
        E_matrix_filename = (
            f"E_matrix_{self.ds.var_name}_"
            f"q_{self.ds.q}_min_num_events_{self.ds.min_evs}_"
            f"taumax_{taumax}_jobid_{job_id}.npy"
        )

        print(f"JobID {job_id}: Start comparing all time series with taumax={taumax}!")
        if not os.path.exists(E_matrix_folder + E_matrix_filename):
            es.parallel_event_synchronization(
                event_series_matrix,
                taumax=taumax,
                min_num_sync_events=min_num_sync_events,
                job_id=job_id,
                num_jobs=num_jobs,
                savepath=E_matrix_folder + E_matrix_filename,
                q_dict=null_model,
                q_min=0.5,  # This is not the sign threshold for later tests, it is only to store less data
            )
        else:
            print(f"File {E_matrix_folder+E_matrix_filename} does already exist!")

        path = E_matrix_folder
        E_matrix_files = [os.path.join(path, fn) for fn in next(os.walk(path))[2]]
        if len(E_matrix_files) < num_jobs:
            print(
                f"JobId {job_id}: Finished. Missing {num_jobs} - {len(E_matrix_files)}!",
                flush=True,
            )
            sys.exit(0)

        return None

    def compute_es_adjacency(
        self,
        E_matrix_folder,
        num_time_series,
        weighted=False,
        q_dict=None,
        q_med=0.5,
        lq=0.25,
        hq=0.75,
        q_sig=0.95,
    ):
        if not os.path.exists(E_matrix_folder):
            raise ValueError("ERROR! The parallel ES is not computed yet!")

        adj_null_model, weights = es.get_adj_from_E(
            E_matrix_folder,
            num_time_series,
            weighted=weighted,
            q_dict=q_dict,
            q_med=q_med,
            lq=lq,
            hq=hq,
            q_sig=q_sig,
        )

        if adj_null_model.shape != weights.shape:
            raise ValueError("Adjacency and weights not of the same shape!")

        return adj_null_model, weights

    def compute_es_null_model(
        self,
        n_pmts=3000,
        null_model_file=None,
        q=[0.25, 0.5, 0.75, 0.95, 0.98, 0.99, 0.995, 0.999],
        rc_null_model=False,
    ):
        """Generates a null model for the given dataset timeseries

        Args:
            n_pmts (int, optional): Number of permutations. Defaults to 3000.
            null_model_file (str, optional): Filename of the null_model. Defaults to None.

        Returns:
            None
        """
        print(f"Create ES Null Model with {n_pmts} permutations!")

        time_steps = tu.get_num_tps(ds=self.ds.ds)
        max_num_events = tu.get_max_num_tps(ds=self.ds.ds, q=self.q)

        if not os.path.exists(self.null_model_folder):
            print(f"Created folder {self.null_model_folder}!")
            os.makedirs(self.null_model_folder)

        savepath = self.null_model_folder + null_model_file
        if not os.path.exists(savepath):
            rc_null_model = True
        if rc_null_model:
            print(f"Create file {savepath}!")
            q_dict = es.null_model_distribution(
                length_time_series=time_steps,
                min_num_events=1,
                max_num_events=max_num_events,
                num_permutations=n_pmts,
                savepath=savepath,
                q=q,
            )
        else:
            print(f"Dict already exists: {savepath}! Load file...")
            # q_dict is dictionary
            q_dict = np.load(savepath, allow_pickle=True).item()

        return q_dict

    def link_bundles(
        self,
        num_rand_permutations,
        num_cpus=mpi.cpu_count(),
        nn_points_bw=None,
        link_bundle_folder=None,
    ):

        self.adjacency = self._link_bundles(
            num_rand_permutations=num_rand_permutations,
            num_cpus=num_cpus,
            nn_points_bw=nn_points_bw,
            link_bundle_folder=link_bundle_folder,
        )

        # Make sure that adjacency is symmetric (ie. in-degree = out-degree)
        # self.adjacency = nwf.make_network_undirected()
        # Correct the weight matrix accordingly to the links
        if self.corr is not None:
            self.corr = np.where(self.adjacency == 1, self.corr, 0)

        return self.adjacency


class EventSyncClimNetRandom(EventSyncClimNet):
    """Correlation based climate network.

    Parameters:
    -----------
    corr_method: str
        Correlation method of network ['spearman', 'pearson'], default: 'spearman'
    threshold: float
        Default: -1.0
    stat_test: str
        Default: 'twosided'
    confidence: float
        Default: 0.99
    significance_test: str
        Method for statistical p-value testing. Default is 'bonf' (Holm-Bonferroni method)
    """

    def __init__(self, dataset, taumax=10, min_num_sync_events=2):
        vars = dataset.vars
        if "evs" not in vars:
            raise ValueError(
                "For EventSyncNet a dataset has to be provided that contains event series!"
            )
        super().__init__(
            dataset=dataset, taumax=taumax, min_num_sync_events=min_num_sync_events
        )

    def create(
        self,
        E_matrix_folder=None,
        null_model_file=None,
        sy=None,
        ey=None,
        sm_arr=["Jan"],
        em_arr=["Dec"],
        set_rest_zero=False,
        full_rnd=False,
    ):
        """
        This function has to be called twice, once, to compute the exact numbers of synchronous
        events between two time series, second again to compute the adjacency matrix
        and the link bundles.
        Attention: The number of parrallel jobs that were used for the E_matrix needs to be
        passed correctly to the function.
        """
        # Test if ES data is computed
        if self.ds.ds["evs"] is None:
            raise ValueError("ERROR Event Synchronization data is not computed yet")
        else:
            data_evs = self.ds.ds["evs"]
            num_time_series = self.ds.flatten_array().shape[1]
            if num_time_series == data_evs.shape[1]:
                print("WARNING: The Mask did not remove any values!")
        if E_matrix_folder is None:
            E_matrix_folder = (
                self.es_filespath
                + f"/E_matrix/{self.ds.var_name}_{self.ds.grid_step}_rnd/"
            )
        else:
            E_matrix_folder = self.es_filespath + f"/E_matrix_rnd/{E_matrix_folder}"

        if null_model_file is None:
            null_model_file = self.compute_es_null_model()
            print("finished computing ES null model. End all jobs!")
            sys.exit()
        else:
            null_model_file = self.es_filespath + f"null_model/{null_model_file}"
            if not os.path.exists(null_model_file):
                raise ValueError(f"File {null_model_file} does not exist!")

        # Randomize time series
        data_evs = self.ds.randomize_spatio_temporal_data_yearly(
            data=data_evs,
            start_year=sy,
            end_year=ey,
            sm_arr=sm_arr,
            em_arr=em_arr,
            var="evs",
            set_rest_zero=set_rest_zero,
            full_rnd=full_rnd,
            seed=0,
        )

        self.event_synchronization_run(
            data_evs=data_evs,
            E_matrix_folder=E_matrix_folder,
            taumax=self.taumax,
            null_model_file=null_model_file,
        )

        self.adjacency = self.compute_es_adjacency(
            E_matrix_folder=E_matrix_folder, num_time_series=num_time_series
        )

        return None
