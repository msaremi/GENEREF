import os
import pandas as pd
import numpy as np
import scipy.sparse as sprs
from io import StringIO
import scipy.stats as stats
from typing import List, Tuple, Union
from itertools import count


class DataIO:
    @staticmethod
    def load_network(file_path, triplet=False, header=False, dtype='f', ftype='tsv'):
        if triplet:
            with open(file_path, 'r') as file:
                str_data = file.read().replace('G', '')

            stream_data = StringIO(str_data)
            stream_data.seek(0)
            d = np.genfromtxt(stream_data, dtype=dtype)
            w = np.max(d[:, 0:-1])
            return sprs.coo_matrix((d[:, -1], (d[:, 0] - 1, d[:, 1] - 1)), shape=(w, w)).todense()
        elif ftype == 'tsv':
            data_frame = pd.read_csv(file_path, sep="\t", header=int(header))
            return data_frame.values
        else:
            return np.load(file_path)

    @staticmethod
    def load_experiment_set(file_path, header=False):
        header_index = 0 if header is True else None
        data_frame = pd.read_csv(file_path, sep="\t", header=header_index)
        return data_frame.values

    @staticmethod
    def save_network(file_path, data, triplet=False, rewrite=True, ftype='tsv'):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if triplet:
            s = sprs.coo_matrix(data)
            triplet_data = np.array([a for a in zip(s.row + 1, s.col + 1, s.data)])
            np.savetxt(file_path, triplet_data, fmt=["G%d", "G%d", "%d"], delimiter="\t")
        elif ftype == 'tsv':
            if rewrite or not os.path.exists(file_path):
                df = pd.DataFrame(data)
                df.to_csv(file_path, sep="\t", index=False)
        else:
            np.save(file_path, data)


class Loader:
    def __init__(self, path, network, preds_path=None):
        self._path = path
        self._network = network
        self._preds_path = os.path.join(path, "preds") if preds_path is None else preds_path

    def load_multifactorial(self, key=None):
        """
        Load multifactorial.tsv into numpy array
        :return: numpy array of multifactorial data
        """
        key = "" if key is None else "[%s]" % str(key)
        file_name = "%s_multifactorial%s.tsv" % (self._network, key)
        file_path = os.path.join(self._path, file_name)
        steadystate_data = DataIO.load_experiment_set(file_path, header=True)
        return steadystate_data

    def load_knockout(self, key=None):
        """
        Load knockout.tsv into numpy array
        :return: numpy array of knockout data
        """
        key = "" if key is None else "[%s]" % str(key)
        file_name = "%s_knockouts%s.tsv" % (self._network, key)
        file_path = os.path.join(self._path, file_name)
        steadystate_data = DataIO.load_experiment_set(file_path, header=True)
        return steadystate_data

    def load_goldstandard(self):
        """
        Load goldstandard.tsv into numpy array
        :return: dense numpy array of goldstandard data
        """
        file_name = "%s_goldstandard.tsv" % self._network
        file_path = os.path.join(self._path, file_name)
        goldstandard_data = DataIO.load_network(file_path, triplet=True, dtype='i')
        return goldstandard_data

    def load_prediction(self, key=None):
        key = "" if key is None else "[%s]" % str(key)
        file_name = "%s_prediction%s.npy" % (self._network, key)
        file_path = os.path.join(self._preds_path, file_name)
        goldstandard_data = DataIO.load_network(file_path, triplet=False, dtype='f', ftype='npy')
        return goldstandard_data

    def load_timeseries(self, key=None):
        """
        Load timeseries.tsv into a list
        :return: list of numpy arrays of timeseries measurements
        """
        key = "" if key is None else "[%s]" % str(key)
        file_name = "%s_timeseries%s.tsv" % (self._network, key)
        # file_name = "%s_dream4_timeseries%s.tsv" % (self._network, key)
        file_path = os.path.join(self._path, file_name)
        timeseries_data = DataIO.load_experiment_set(file_path, header=True)
        idx, = np.where(timeseries_data[:, 0] == 0)
        time_points_list = np.split(timeseries_data[:, 0], idx[1:])
        timeseries_list = np.split(timeseries_data[:, 1:], idx[1:], axis=0)

        try:
            file_name = "%s_timeseries_perturbations.tsv" % self._network
            # file_name = "%s_dream4_timeseries_perturbations.tsv" % self._network
            file_path = os.path.join(self._path, file_name)
            perturbation_data = DataIO.load_experiment_set(file_path, header=True)
        except FileNotFoundError:
            perturbation_data = [None] * len(timeseries_list)

        return time_points_list, timeseries_list, perturbation_data


class Saver:
    def __init__(self, path, network, preds_path=None):
        self._path = path
        self._network = network
        self._preds_path = os.path.join(path, "preds") if preds_path is None else preds_path

    def save_prediction(self, prediction, key=None, rewrite=True):
        """
        Save predicted matrix of edge confidence values into prediction.tsv
        :param rewrite:
        :param prediction: dense numpy matrix to be saved
        :param key: if None the file will be prediction.tsv
                        otherwise, the file will be prediction[index].tsv
        :return:
        """
        key = "" if key is None else "[%s]" % str(key)
        file_name = "%s_prediction%s.npy" % (self._network, key)
        file_path = os.path.join(self._preds_path, file_name)
        DataIO.save_network(file_path, prediction, triplet=False, rewrite=rewrite, ftype='npy')

    def save_candidate(self, candidate, key=None):
        """
        Save candidate network as a sparse matrix into candidate.tsv
        :param candidate: a matrix of 0's and 1's
        :param key: if None the file will be candidate.tsv
                        otherwise, the file will be candidate[index].tsv
        :return:
        """
        key = "" if key is None else "[%s]" % str(key)
        file_name = "%s_candidate%s.tsv" % (self._network, key)
        file_path = os.path.join(self._path, "cands", file_name)
        DataIO.save_network(file_path, candidate, triplet=True)
        # s = sprs.coo_matrix(candidate)
        # ones_sparse_data = np.array([a + (1,) for a in zip(s.row + 1, s.col + 1)])
        # s = sprs.coo_matrix(1 - np.eye(candidate.shape[0]) - candidate)
        # zeros_sparse_data = np.array([a + (0,) for a in zip(s.row + 1, s.col + 1)])
        # sparse_data = np.concatenate((ones_sparse_data, zeros_sparse_data))
        #
        # np.savetxt(file_path, sparse_data, fmt=["G%d", "G%d", "%d"], delimiter="\t")


class Network:
    def __init__(self, data: np.ndarray):
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise ValueError("data must be a square matrix. Its current shape is %s" % str(data.shape))

        self._selection = np.arange(data.shape[0])
        self._data: np.ndarray = data

    @property
    def data(self):
        return self._data


class UnweightedNetwork(Network):
    def __init__(self, data: np.ndarray):
        if not np.array_equal(data, data.astype(bool)):
            raise ValueError("data must be a boolean matrix")

        super().__init__(data)


class WeightedNetwork(Network):
    def _get_ranked_data(self, method='average') -> np.ndarray:
        """
        Returns data that is distributed evenly
        :return: Redistributed data
        """
        # TODO : How the data must be resorted
        idx = stats.rankdata(self._data.ravel(), method) - 1
        idx = idx / (len(idx) - 1)
        return idx.reshape(self._data.shape)

    def redistribute(self, method='average'):
        """
        Redistributes the data evenly
        """
        self._data = self._get_ranked_data(method)

    def get_redistributed_network(self, method='average'):
        """
        Get redistributed Network
        """
        return WeightedNetwork(self._get_ranked_data(method))


class Experiment:
    def __init__(self, data: np.ndarray):
        self.data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if data.ndim != 2:
            raise ValueError("data must be a 2D matrix")

        self._data = data

    @property
    def num_measerments(self):
        return self.data.shape[0]

    @property
    def num_genes(self):
        return self.data.shape[1]

    @property
    def _normalized_data(self) -> np.ndarray:
        return self._data / np.sqrt(np.var(self._data, axis=0))

    def normalize(self):
        self._data = self._normalized_data

    def get_normalized_experiment(self):
        return Experiment(self._normalized_data)


class SteadyStateExperiment(Experiment):
    pass


class TimeseriesExperiment(Experiment):
    def __init__(self, time_points: np.ndarray, data: np.ndarray, perturbation: np.ndarray = None):
        super().__init__(data)

        if time_points.ndim != 1:
            raise ValueError("data must be a 1D array")

        if time_points.shape[0] != data.shape[0]:
            raise ValueError("number of time_points must equal number of measurements")

        self._time_points = time_points

        if perturbation is not None:
            if perturbation.shape[0] != data.shape[1]:
                raise ValueError("number of perturbations must be equal to number of genes")

            self._perturbation = perturbation
        else:
            self._perturbation = np.full(data.shape, np.nan)

    @property
    def time_points(self):
        return self._time_points


class ExperimentSet:
    def __init__(self, data: List[Experiment]):
        self._data = data

    def __getitem__(self, item) -> Union[Experiment, 'ExperimentSet']:
        if item is slice:
            return ExperimentSet(self._data[item])
        else:
            return self._data[item]

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        return self._data

    def normalize_each_experiment(self):
        for experiment in self._data:
            experiment.normalize()

    def normalize(self):
        std = np.std(np.vstack([experiment.data for experiment in self._data]), axis=0)

        for experiment in self._data:
            experiment.data /= std


class TimeseriesExperimentSet(ExperimentSet):
    def __init__(self, data: List[TimeseriesExperiment]):
        super().__init__(data)


# noinspection PyBroadException
class DataManager:
    # noinspection PyBroadException
    # noinspection PyProtectedMember
    class Predictions:
        def __init__(self, parent):
            self._parent = parent
            self._predictions = {}
            self._autosave = True
            self._new_keys = set()

        def __setitem__(self, key, prediction):
            self._predictions[key] = prediction

            if self._autosave:
                self._parent._saver.save_prediction(prediction.data, key)
            else:
                self._new_keys.add(key)

        def __getitem__(self, key) -> WeightedNetwork:
            if key not in self._predictions:
                try:
                    self._predictions[key] = WeightedNetwork(self._parent._loader.load_prediction(key))
                except:
                    raise KeyError("Key %s does not exist" % str(key))

            return self._predictions[key]

        def __contains__(self, key):
            try:
                self[key]
            except:
                return False

            return True

        def __len__(self):
            return len(self._predictions)

        def __str__(self):
            return str(self._predictions)

        def flush(self):
            for key in self._new_keys:
                self._parent._saver.save_prediction(self._predictions[key].data, key)

    def __init__(self, path: str, network: str, preds_path: str = None):
        """
        :param path: ex: [GENEREF DIR]\data\datasets\0\
        :param network: ex: insilico_size100_1
        """
        self._goldstandard = None

        self._predictions = DataManager.Predictions(self)
        self._loader = Loader(path, network, preds_path)
        self._saver = Saver(path, network, preds_path)

        experiments = []

        for i in count(0, 1):  # infinite loop
            try:
                steadystate_data = self._loader.load_multifactorial(i)
                steadystate_data = SteadyStateExperiment(steadystate_data)
                experiments.append(steadystate_data)
            except:
                break

        for i in count(0, 1):  # infinite loop
            try:
                steadystate_data = self._loader.load_knockout(i)
                steadystate_data = SteadyStateExperiment(steadystate_data)
                experiments.append(steadystate_data)
            except:
                break

        for i in count(0, 1):  # infinite loop
            try:
                time_points_list, timeseries_list, perturbation_data = self._loader.load_timeseries(i)
                timeseries_data = [TimeseriesExperiment(time_points, timeseries, perturbation)
                                   for time_points, timeseries, perturbation
                                   in zip(time_points_list, timeseries_list, perturbation_data)]
                experiments.append(TimeseriesExperimentSet(timeseries_data))
            except:
                break

        self._experiments = experiments

    def __enter__(self):
        self._predictions._autosave = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._predictions.flush()

    @property
    def goldstandard(self):
        if self._goldstandard is None:
            goldstandard_data = self._loader.load_goldstandard()
            self._goldstandard = UnweightedNetwork(goldstandard_data)

        return self._goldstandard

    def get_steadystates(self):
        return filter(lambda x: x is SteadyStateExperiment, self._experiments)

    def get_timeseries_sets(self):
        return filter(lambda x: x is TimeseriesExperimentSet, self._experiments)

    @property
    def experiments(self):
        return self._experiments

    @property
    def predictions(self):
        return self._predictions

    # def set_prediction(self, key: Union[str, tuple], prediction: WeightedNetwork):
    #     self._predictions[key] = prediction
    #
    # def get_prediction(self, key: Union[str, tuple]) -> WeightedNetwork:
    #     if key not in self._predictions:
    #         self._predictions[key] = WeightedNetwork(self._loader.load_prediction(key))
    #
    #     return self._predictions[key]

    # def flush(self):
    #     for key, prediction in self._predictions:
    #         self._saver.save_prediction(prediction.data, key)
