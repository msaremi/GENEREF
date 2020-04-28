import numpy as np
import pandas as pd
import os
from scipy.stats import norm, beta
from scipy.special import gammainc
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc


class Evaluator:
    """Abstract class. Use one of the derived classes"""
    def __init__(self, network: str):
        self._network = int(network[-1])
        self._labels = None
        self._scores = None

    def fit(self, labels, scores):
        idx = ~np.eye(labels.shape[0], dtype=bool)
        self._labels = np.asarray(labels[idx], dtype=np.int32).ravel()
        self._scores = np.asarray(scores[idx], dtype=np.float64).ravel()

    @property
    def network(self):
        return self._network

    @property
    def auroc(self):
        return roc_auc_score(self._labels, self._scores)

    @property
    def aupr(self):
        precision, recall, _ = precision_recall_curve(self._labels, self._scores)
        return auc(recall, precision)

    @property
    def auroc_p_value(self):
        raise Exception("This property is abstract")

    @property
    def aupr_p_value(self):
        raise Exception("This property is abstract")

    @property
    def score(self):
        return -np.mean([np.log10(self.auroc_p_value), np.log10(self.aupr_p_value)])

    @property
    def score_aupr(self):
        return -np.log10(self.aupr_p_value)


class DREAM5Evaluator(Evaluator):
    _data = {}

    def fit(self, labels, scores):
        labels[np.eye(labels.shape[0], dtype=bool)] = 0
        regulators = np.any(labels, axis=1)
        targets = np.any((np.any(labels, axis=0).T, regulators), axis=0)
        regulators = np.where(regulators)
        targets = np.where(targets)
        targets, regulators = np.meshgrid(targets, regulators)

        gs_labels = labels[regulators, targets]
        gs_scores = scores[regulators, targets]
        self._labels = np.asarray(gs_labels, dtype=np.int32).ravel()
        self._scores = np.asarray(gs_scores, dtype=np.float64).ravel()

    @classmethod
    def _load_data(cls, network, metric):
        if (network, metric) not in cls._data:
            path = os.path.abspath(os.path.join(
                os.getcwd(), "resources", cls.__name__, "Network%d_%s.npy" % (network, metric)
            ))
            cls._data[(network, metric)] = np.load(path)

        return cls._data[(network, metric)]

    @staticmethod
    def _probability(x, y, a):
        dx = x[1] - x[0]
        return np.sum((x >= a) * y * dx)

    @property
    def auroc_p_value(self):
        data = self._load_data(self._network, "AUROC")
        auroc = self.auroc
        return self._probability(data[0, :], data[1, :], auroc)

    @property
    def aupr_p_value(self):
        data = self._load_data(self._network, "AUPR")
        aupr = self.aupr
        return self._probability(data[0, :], data[1, :], aupr)


class DREAM4Evaluator(Evaluator):
    """Abstract class. Use either of DREAM4S10Evaluator and DREAM4S100Evaluator"""
    _data = None

    @classmethod
    def _load_data(cls):
        if cls._data is None:
            path = os.path.abspath(os.path.join(
                os.getcwd(), "resources", cls.__name__, "metadata.csv"
            ))
            cls._data = pd.read_csv(path, index_col=0)

        return cls._data

    @property
    def auroc_p_value(self):
        data = type(self)._load_data()
        mu = data['auroc_mean'][self._network - 1]
        sigma = data['auroc_std'][self._network - 1]
        return norm.cdf(1 - self.auroc, mu, sigma)

    @property
    def aupr_p_value(self):
        data = type(self)._load_data()
        b = data['aupr_b'][self._network - 1]
        c = data['aupr_c'][self._network - 1]
        x_max = data['aupr_x_max'][self._network - 1]
        h_max = data['aupr_h_max'][self._network - 1]
        x = self.aupr
        x_por = 0.835
        x2 = x_max * x_por + x * (1 - x_por)
        w = h_max * (b ** (-1 / c)) / c
        return w * gammainc(b * (x2 - x_max) ** c, 1 / c)


class DREAM4S10Evaluator(DREAM4Evaluator):
    """DREAM4 Size 10 networks evaluator"""
    pass


class DREAM4S100Evaluator(DREAM4Evaluator):
    """DREAM4 Size 100 networks evaluator"""
    pass
