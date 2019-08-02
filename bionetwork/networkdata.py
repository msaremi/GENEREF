import os
import pandas as pd
import numpy as np
import scipy.sparse as sprs
from io import StringIO
import scipy.stats as stats
# from _utils import methoddispatch


class DataIO:
    @staticmethod
    def load_network(file_path, triplet=False, header=False, dtype='f'):
        if triplet:
            with open(file_path, 'r') as file:
                str_data = file.read().replace('G', '')

            stream_data = StringIO(str_data)
            stream_data.seek(0)
            d = np.genfromtxt(stream_data, dtype=dtype)
            return sprs.coo_matrix((d[:, -1], (d[:, 0] - 1, d[:, 1] - 1))).todense()
        else:
            data_frame = pd.read_csv(file_path, sep="\t", header=int(header))
            return data_frame.values

    @staticmethod
    def load_experiment_set(file_path, header=False):
        header_index = 0 if header is True else None
        data_frame = pd.read_csv(file_path, sep="\t", header=header_index)
        return data_frame.values

    @staticmethod
    def save_network(file_path, data, triplet=False):
        if triplet:
            s = sprs.coo_matrix(data)
            triplet_data = np.array([a for a in zip(s.row + 1, s.col + 1, s.data)])
            np.savetxt(file_path, triplet_data, fmt=["G%d", "G%d", "%d"], delimiter="\t")
        else:
            df = pd.DataFrame(data)
            df.to_csv(file_path)


class Loader:
    def __init__(self, path, network):
        self._path = path
        self._network = network

    def load_multifactorial(self):
        """
        Load multifactorial.tsv into numpy array
        :return: numpy array of multifactorial data
        """
        file_name = "%s_multifactorial.tsv" % self._network
        file_path = os.path.join(self._path, file_name)
        multifactorial_data = DataIO.load_experiment_set(file_path, header=True)
        return multifactorial_data

    def load_goldstandard(self):
        """
        Load goldstandard.tsv into numpy array
        :return: dense numpy array of goldstandard data
        """
        file_name = "%s_goldstandard.tsv" % self._network
        file_path = os.path.join(self._path, file_name)
        goldstandard_data = DataIO.load_network(file_path, triplet=True, dtype='i')
        return goldstandard_data

    def load_timeseries(self):
        """
        Load timeseries.tsv into a list
        :return: list of numpy arrays of timeseries measurements
        """
        file_name = "%s_dream4_timeseries.tsv" % self._network
        file_path = os.path.join(self._path, file_name)
        timeseries_data = DataIO.load_experiment_set(file_path, header=True)
        idx, = np.where(timeseries_data[:, 0] == 0)
        snapshots = np.split(timeseries_data[:, 0], idx[1:])
        timeseries_list = np.split(timeseries_data[:, 1:], idx[1:], axis=0)

        try:
            file_name = "%s_dream4_timeseries_perturbations.tsv" % self._network
            file_path = os.path.join(self._path, file_name)
            perturbation_data = DataIO.load_experiment_set(file_path, header=True)
        except FileNotFoundError:
            perturbation_data = None

        return snapshots, timeseries_list, perturbation_data


class Saver:
    def __init__(self, path, network):
        self._path = path
        self._network = network

    def save_prediction(self, prediction, index=None):
        """
        Save predicted matrix of edge confidence values into prediction.tsv
        :param prediction: dense numpy matrix to be saved
        :param index: if None the file will be prediction.tsv
                        otherwise, the file will be prediction[index].tsv
        :return:
        """
        file_name = "%s_prediction.tsv" % self._network if index is None \
            else "%s_prediction[%d].tsv" % (self._network, index)
        file_path = os.path.join(self._path, file_name)
        DataIO.save_network(file_path, prediction, triplet=False)

    def save_candidate(self, candidate, index=None):
        """
        Save candidate network as a sparse matrix into candidate.tsv
        :param candidate: a matrix of 0's and 1's
        :param index: if None the file will be candidate.tsv
                        otherwise, the file will be candidate[index].tsv
        :return:
        """
        file_name = "%s_candidate.tsv" % self._network if index is None \
            else "%s_candidate[%d].tsv" % (self._network, index)
        file_path = os.path.join(self._path, file_name)
        DataIO.save_network(file_path, candidate, triplet=True)
        # s = sprs.coo_matrix(candidate)
        # ones_sparse_data = np.array([a + (1,) for a in zip(s.row + 1, s.col + 1)])
        # s = sprs.coo_matrix(1 - np.eye(candidate.shape[0]) - candidate)
        # zeros_sparse_data = np.array([a + (0,) for a in zip(s.row + 1, s.col + 1)])
        # sparse_data = np.concatenate((ones_sparse_data, zeros_sparse_data))
        #
        # np.savetxt(file_path, sparse_data, fmt=["G%d", "G%d", "%d"], delimiter="\t")


class Laboratory:
    def __init__(self, path: str, network: str):
        self._goldstandard = None
        self._multifactorial = None
        self._timeseries = None
        self._prediction: PredictionNetwork = None
        self._loader = Loader(path, network)
        self._saver = Saver(path, network)

    # @methoddispatch
    # def __init__(self):
    #     self._goldstandard = None
    #     self._multifactorial = None
    #
    # @__init__.register(str)
    # def _(self, path: str):
    #     self._loader = Loader(path)
    #     self._saver = Saver(path)
    #
    # @__init__.register(Loader)
    # def _(self, loader: Loader, saver: Saver):
    #     self._loader = loader
    #     self._saver = saver

    @property
    def goldstandard(self):
        if self._goldstandard is None:
            goldstandard_data = self._loader.load_goldstandard()
            self._goldstandard = ActualNetwork(goldstandard_data)

        return self._goldstandard

    @property
    def multifactorial(self):
        if self._multifactorial is None:
            multifactorial_data = self._loader.load_multifactorial()
            self._multifactorial = MultifactorialExperiment(multifactorial_data)

        return self._multifactorial

    @property
    def timeseries(self):
        if self._timeseries is None:
            snapshots, timeseries_data, perturbation_data = self._loader.load_timeseries()
            self._timeseries = TimeseriesExperimentSet(snapshots, timeseries_data, perturbation_data)

        return self._timeseries

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, value):
        self._prediction = value

    def flush(self):
        if self._prediction is not None:
            self._saver.save_prediction(self._prediction.data)


class Network:
    def __init__(self, data):
        if data.shape[0] != data.shape[1]:
            raise ValueError("data must be a square matrix")

        self._selection = np.arange(data.shape[0])
        self._data: np.ndarray = data

    @property
    def data(self):
        return self._data

    @property
    def genes(self):
        return type(self).GeneGroup(self, self._selection)

    class GeneGroup:
        def __init__(self, parent, selection):
            self._parent: Network = parent
            self._selection: np.ndarray = selection

        def __getitem__(self, item):
            selection = self._selection[item]

            if type(item) is slice:
                return Network.GeneGroup(self._parent, selection)
            else:
                return type(self._parent).Gene(self._parent, selection)

        def __len__(self):
            return len(self._selection)

    class Gene:
        def __init__(self, parent, selection):
            self._parent: Network = parent
            self._selection = selection

        @property
        def in_data(self):
            return np.ravel(self._parent._data[:, self._selection])

        @property
        def out_data(self):
            return np.ravel(self._parent._data[self._selection, :])


class ActualNetwork(Network):
    class Gene(Network.Gene):
        @property
        def in_degree(self):
            return np.sum(self._parent._data[:, self._selection])

        @property
        def out_degree(self):
            return np.sum(self._parent._data[self._selection, :])


class PredictionNetwork(Network):
    @property
    def ranked_data(self):
        idx = stats.rankdata(self._data.ravel()) - 1
        idx = idx / (len(idx) - 1)
        return idx.reshape(self._data.shape)

    def distribute_evenly(self):
        self._data = self.ranked_data


class MeasurementSet:
    def __init__(self, data):
        self._data: np.ndarray = data
        self._measurement_selection = np.arange(data.shape[0])
        self._gene_selection = np.arange(data.shape[1])

    @property
    def data(self):
        return self._data

    @property
    def genes(self):
        return type(self).GeneGroup(self, self._gene_selection)

    @property
    def measurements(self):
        return type(self).MeasurementGroup(self, self._data)

    @property
    def normalized_data(self):
        return self._data / np.sqrt(np.var(self._data, axis=0))

    def normalize(self):
        self._data = self.normalized_data

    class GeneGroup:
        def __init__(self, parent, selection):
            self._parent: MeasurementSet = parent
            self._selection: np.ndarray = selection

        def __getitem__(self, item):
            selection = self._selection[item]

            if type(item) is slice:
                return type(self._parent).GeneGroup(self._parent, selection)
            else:
                return type(self._parent).Gene(self._parent, selection)

        def __len__(self):
            return len(self._selection)

        @property
        def data(self):
            return self._parent._data[:, self._selection]

    class Gene:
        def __init__(self, parent, item):
            self._parent: MeasurementSet = parent
            self._item = item

        @property
        def data(self):
            return self._parent._data[:, self._item]

    class MeasurementGroup:
        def __init__(self, parent, selection):
            self._parent: MeasurementSet = parent
            self._selection: np.ndarray = selection

        def __getitem__(self, item):
            selection = self._selection[item]

            if type(item) is slice:
                return type(self._parent).MeasurementGroup(self._parent, selection)
            else:
                return type(self._parent).Measurement(self._parent, selection)

        def __len__(self):
            return len(self._selection)

        @property
        def data(self):
            return self._parent._data[self._selection, :]

    class Measurement:
        def __init__(self, parent, item):
            self._parent: MeasurementSet = parent
            self._item = item

        @property
        def data(self):
            return self._parent._data[self._item, :]


class ExperimentSet:
    def __init__(self, data):
        self._data: list = data

    def __getitem__(self, item):
        if type(item) is slice:
            return type(self)(self._data[item])
        else:
            return type(self).Experiment(self._data[item])

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        return self._data

    @property
    def normalized_data(self):
        std = np.std(np.vstack(self._data), axis=0)
        return [d / std for d in self._data]

    def normalize(self):
        self._data = self.normalized_data

    class Experiment(MeasurementSet):
        pass


class MultifactorialExperiment(MeasurementSet):
    pass


class TimeseriesExperimentSet(ExperimentSet):
    def __init__(self, snapshots, data, perturbation=None):
        super().__init__(data)
        self._snapshots = snapshots

        if perturbation is not None:
            self._perturbation: np.ndarray = perturbation
        else:
            self._perturbation: np.ndarray = np.full((len(data), data[0].shape[1]), np.nan)

    def __getitem__(self, item):
        if type(item) is slice:
            return type(self)(self._snapshots[item], self._data[item], self._perturbation[item, :])
        else:
            return type(self).Experiment(self._snapshots[item], self._data[item], self._perturbation[item, :])

    @property
    def snapshots(self):
        return self._snapshots

    class Experiment(ExperimentSet.Experiment):
        def __init__(self, snapshots, data, perturbations):
            super().__init__(data)
            self._snapshots = snapshots
            self._perturbations = perturbations

        @property
        def snapshots(self):
            return self._snapshots

        class GeneGroup(ExperimentSet.Experiment.GeneGroup):
            @property
            def perturbations(self):
                parent: TimeseriesExperimentSet.Experiment = self._parent
                return parent._perturbations[self._selection]

        class Gene(ExperimentSet.Experiment.Gene):
            @property
            def perturbation(self):
                parent: TimeseriesExperimentSet.Experiment = self._parent
                return parent._perturbations[self._item]
