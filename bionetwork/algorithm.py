import numpy as np
from joblib import Parallel, delayed
from networkdata import Experiment, SteadyStateExperiment, TimeseriesExperimentSet, WeightedNetwork
from rainforest.ensemble import RegressionForest
from typing import Union

class Predictor:
    def __init__(self, num_of_jobs: int = 8, n_trees: int = 100, trunk_size: int = None, max_features: float = 1/7,
                 callback=None, parallel_mode='multiprocessing'):
        self._network = None
        self._num_of_jobs = num_of_jobs
        self._n_trees = n_trees
        self._max_features = max_features
        self._trunk_size = trunk_size if trunk_size else num_of_jobs
        self._callback = callback
        self._parallel_mode = parallel_mode

    @staticmethod
    def _get_timelagged_subproblem(timeseries: TimeseriesExperimentSet, target_gene, regularizations=None):
        x_list = []
        y_list = []
        weights = None

        for experiment in timeseries:
            data = experiment.data
            x_list.append(np.delete(data[:-1, :], target_gene, axis=1))
            y_list.append(data[1:, target_gene:target_gene+1])

        x = np.vstack(x_list)
        y = np.vstack(y_list)

        if regularizations is not None:
            weights = np.delete(regularizations[:, target_gene], target_gene)

        return x, y, weights

    @staticmethod
    def _get_multifactorial_subproblem(experiment: Experiment, target_gene, importances=None):
        x = np.delete(experiment.data, target_gene, axis=1)
        y = experiment.data[:, target_gene:target_gene + 1]
        weights = None

        if importances is not None:
            weights = np.delete(importances[:, target_gene], target_gene)

        return x, y, weights

    def _get_all_subproblems(self, experiment: Union[SteadyStateExperiment, TimeseriesExperimentSet], regularizations=None):
        data = []

        if isinstance(experiment, TimeseriesExperimentSet):
            get_subproblem_func = Predictor._get_timelagged_subproblem
            num_genes = experiment[0].num_genes
        elif isinstance(experiment, SteadyStateExperiment):
            get_subproblem_func = Predictor._get_multifactorial_subproblem
            num_genes = experiment.num_genes
        else:
            get_subproblem_func = None
            num_genes = 0

        for j in range(num_genes):
            x, y, w = get_subproblem_func(experiment, j, regularizations)
            data.append((x, y, w))

            if j % self._trunk_size == self._trunk_size - 1:
                if self._callback:
                    self._callback(j - (self._trunk_size - 1), num_genes)

                yield data
                data = []

        if data:
            if self._callback:
                self._callback(num_genes - (num_genes % self._trunk_size), num_genes)

            yield data

    @staticmethod
    def _solve_subproblem(n_trees, max_features, x, y, feature_weight=None):
        regr = RegressionForest(n_trees, max_features=max_features)
        regr.fit(x, np.ravel(y), feature_weight=feature_weight)
        return regr.feature_importances_

    def _solve_all_problems_parallely(self, data):
        require = None if self._parallel_mode == 'multiprocessing' else 'sharedmem'
        parallel = Parallel(n_jobs=self._num_of_jobs, require=require)
        cls = type(self)
        results = \
            parallel(delayed(cls._solve_subproblem)(self._n_trees, self._max_features, x, y, w) for x, y, w in data)
        return results

    @staticmethod
    def _make_network(results):
        num_genes = len(results)
        network = np.zeros((num_genes, num_genes))

        for j in range(num_genes):
            imp = np.insert(results[j], j, 0)
            network[:, j] = imp

        return network

    def fit(self, experiment: Union[SteadyStateExperiment, TimeseriesExperimentSet], regularization=None):
        experiment.normalize()
        results = []

        for data in self._get_all_subproblems(experiment, regularization):
            results += self._solve_all_problems_parallely(data)

        pred_network = Predictor._make_network(results)
        self._network = WeightedNetwork(pred_network)

    @property
    def network(self) -> WeightedNetwork:
        return self._network
