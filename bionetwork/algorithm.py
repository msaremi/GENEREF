import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import scale
from networkdata import MeasurementSet, MultifactorialExperiment, TimeseriesExperimentSet, PredictionNetwork
from rainforest.tree import RegressionTree
from rainforest.ensemble import RegressionForest
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from scipy.stats import beta, norm
import xgboost as xgb
from dynGENIE3 import dynGENIE3


class Predictor:
    def __init__(self, timeseries_method='dynamic'):
        self._network = None
        self._timeseries_method = timeseries_method

    @staticmethod
    def estimate_degradation_rates(timeseries: TimeseriesExperimentSet):
        ngenes = timeseries[0].data.shape[1]
        nexp = len(timeseries)

        C_min = np.array([experiment.data.min() for experiment in timeseries]).min()
        alphas = np.zeros((nexp, ngenes))

        for (i, experiment) in enumerate(timeseries):
            time_points = experiment.snapshots
            data = experiment.data

            for j in range(ngenes):
                idx_min = np.argmin(data[:, j])
                idx_max = np.argmax(data[:, j])
                xmin = data[idx_min, j]
                xmax = data[idx_max, j]
                tmin = time_points[idx_min]
                tmax = time_points[idx_max]
                xmin = max(xmin - C_min, 1e-6)
                xmax = max(xmax - C_min, 1e-6)
                xmin = np.log(xmin)
                xmax = np.log(xmax)
                alphas[i, j] = (xmax - xmin) / abs(tmin - tmax)

        alphas = alphas.max(axis=0)

        return alphas

    @staticmethod
    def _get_timelagged_subproblem(timeseries: TimeseriesExperimentSet, target_gene, alpha=None, importances=None):
        x_ = [None] * len(timeseries)
        y_ = x_.copy()
        weights = None

        for i, experiment in enumerate(timeseries):
            data = experiment.data
            x_[i] = np.delete(data[:-1, :], target_gene, axis=1)
            y_[i] = data[1:, target_gene:target_gene+1]

        x = np.vstack(x_)
        y = np.vstack(y_)

        if importances is not None:
            weights = np.delete(importances[:, target_gene], target_gene)

        return x, y, weights

    @staticmethod
    def _get_timeseries_subproblem(timeseries: TimeseriesExperimentSet, target_gene, alpha, importances=None):
        x_ = [None] * len(timeseries)
        y_ = x_.copy()
        weights = None

        for i, experiment in enumerate(timeseries):
            data = experiment.data
            snapshots = experiment.snapshots
            time_diff = snapshots[1:] - snapshots[:-1]
            x_[i] = np.delete(data[:-1, :], target_gene, axis=1)
            y_[i] = (data[1:, target_gene] - data[:-1, target_gene]) / time_diff + alpha[target_gene] * data[:-1, target_gene]

        x = np.vstack(x_)
        y = np.vstack(y_)

        if importances is not None:
            weights = np.delete(importances[:, target_gene], target_gene)
        return x, y, weights

    @staticmethod
    def _get_subproblem(experiment: MeasurementSet, target_gene, alpha=None, importances=None):
        x = np.delete(experiment.data, target_gene, axis=1)
        y = experiment.data[:, target_gene:target_gene+1]
        weights = None

        if importances is not None:
            weights = np.delete(importances[:, target_gene], target_gene)

        return x, y, weights

    @staticmethod
    def _solve_subproblem_timeseries(X, y, feature_weight=None):
        regr = xgb.XGBRegressor()
        regr.fit(X, np.ravel(y))
        return regr.feature_importances_

    @staticmethod
    def _solve_subproblem(X, y, feature_weight=None):
        regr = RegressionForest(100, max_features=1/7)
        regr.fit(X, np.ravel(y), feature_weight=feature_weight)
        return regr.feature_importances_

    @property
    def network(self):
        return self._network

    @staticmethod
    def _parallel_solve_problem_timeseries(data):
        parallel = Parallel(n_jobs=8)
        results = parallel(delayed(Predictor._solve_subproblem_timeseries)(x, y, w) for x, y, w in data)
        return results

    @staticmethod
    def _parallel_solve_problem(data):
        parallel = Parallel(n_jobs=8)
        results = parallel(delayed(Predictor._solve_subproblem)(x, y, w) for x, y, w in data)
        return results

    @staticmethod
    def _make_network(results):
        num_genes = len(results)
        network = np.zeros((num_genes, num_genes))

        for j in range(num_genes):
            imp = np.insert(results[j], j, 0)
            network[:, j] = imp

        return network

    def _get_all_subproblems(self, experiment, importances=None):
        data = []

        if isinstance(experiment, TimeseriesExperimentSet):
            func = Predictor._get_timeseries_subproblem if self._timeseries_method == 'dynamic' else Predictor._get_timelagged_subproblem
            num_genes = len(experiment[0].genes)
            alpha = Predictor.estimate_degradation_rates(experiment)
        else:
            func = Predictor._get_subproblem
            num_genes = len(experiment.genes)
            alpha = None

        for j in range(num_genes):
            x, y, w = func(experiment, j, alpha, importances)
            data.append((x, y, w))

        return data

    @staticmethod
    def solve_dynGENIE3(networks):
        ts_network = networks[0].data
        ss_network = networks[1].data
        time_points = networks[0].snapshots
        VIM, _, _, _, _ = dynGENIE3(ts_network, time_points, SS_data=ss_network)
        return VIM

    def fit(self, network_list, **kwargs):
        param = kwargs['param']
        penalty = None
        pred_network = None

        for network in network_list:
            if isinstance(network, tuple):
                pred_network = Predictor.solve_dynGENIE3(network)
            else:
                network.normalize()
                data = self._get_all_subproblems(network, penalty)
                results = Predictor._parallel_solve_problem(data)
                pred_network = Predictor._make_network(results)

            penalty = PredictionNetwork(pred_network).ranked_data
            penalty = penalty ** (2 ** param)

        self._network = PredictionNetwork(pred_network)




        # multifactorial.normalize()
        #
        #
        #
        # data = Predictor._get_all_subproblems(multifactorial)
        # results = Predictor._parallel_solve_problem(data)
        # network = Predictor._make_network(results)
        # penalty = PredictionNetwork(network).ranked_data
        # penalty = penalty ** (2 ** param)
        #
        # # data = Predictor._get_all_subproblems(multifactorial, penalty)
        # # results = Predictor._parallel_solve_problem(data)
        # # network = Predictor._make_network(results)
        # # penalty = PredictionNetwork(network).ranked_data
        # # penalty = penalty ** (2 ** param)
        #
        # # timeseries_data = np.vstack(timeseries.data)
        # # timeseries_snapshots = np.vstack(timeseries.snapshots)
        # # timeseries = TimeseriesExperimentSet([timeseries_snapshots], [timeseries_data])
        # timeseries.normalize()
        #
        # data = Predictor._get_all_subproblems(timeseries, penalty)
        # results = Predictor._parallel_solve_problem(data)
        # network = Predictor._make_network(results)


    # def fit(self, multifactorial: MultifactorialExperiment, timeseries: TimeseriesExperimentSet, **kwargs):
    #     param = kwargs['param']
    #     multifactorial.normalize()
    #     # num_genes = len(multifactorial.genes)
    #     # network = np.zeros((num_genes, num_genes))
    #
    #     data = Predictor._get_all_subproblems(multifactorial)
    #     results = Predictor._parallel_solve_problem(data)
    #     network = Predictor._make_network(results)
    #     penalty = PredictionNetwork(network).ranked_data
    #
    #     # penalty = network.copy()
    #     # # penalty = penalty / np.max(penalty, axis=0)
    #     # # penalty = np.asarray(penalty > param, dtype=float)
    #     #
    #     penalty = penalty ** (2 ** param)
    #     # # penalty = norm.pdf(penalty, 1, param)
    #     # # penalty = np.exp(-(penalty - 1) ** 2 / param ** 2)
    #     # # penalty = 1 / (1 + (penalty / (1 - penalty)) ** -param)
    #     # # penalty = np.minimum(penalty ** param, .7) / .7
    #     #
    #     timeseries_data = np.vstack(timeseries.data)
    #     timeseries_snapshots = np.vstack(timeseries.data)
    #     timeseries = TimeseriesExperimentSet([timeseries_snapshots], [timeseries_data])
    #     timeseries.normalize()
    #
    #     data = Predictor._get_all_subproblems(timeseries[0], penalty)
    #     results = Predictor._parallel_solve_problem(data)
    #     network = Predictor._make_network(results)
    #
    #     self._network = PredictionNetwork(network)
    #
    #     # for j in range(num_genes):
    #     #     x, y, weights = Predictor._get_subproblem(experiment_set, j, penalty)
    #     #     imp2 = Predictor._solve_subproblem(x, y, feature_weight=weights)
    #     #     imp2 = np.insert(imp2, j, 0)
    #     #     network[:, j] = imp2
    #     #
    #     # self._network = PredictionNetwork(network)

