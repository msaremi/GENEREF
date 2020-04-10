import numpy as np
import os
from scipy.stats import norm, beta
from scipy.special import gammainc
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc


class Evaluator:
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
            path = os.path.abspath(os.path.join(os.getcwd(), "resources", "Network%d_%s.npy" % (network, metric)))
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
    _auroc_means = [0.5, 0.5, 0.5, 0.5, 0.5]
    _auroc_stds = [0.021921, 0.018493, 0.02086, 0.02004, 0.020945]
    _aupr_x_maxs = [0.0193143, 0.0277462, 0.02165965, 0.02343955, 0.02152605]
    _aupr_h_maxs = [263.620981387474, 222.364217252398, 255.935098206657, 240.841777084961, 261.668070766632]
    _aupr_bs = [31700.820267609, 8719.82765760784, 8568.45635982301, 10659.2365536708, 5082.12445262662]
    _aupr_cs = [1.74599609375, 1.53232421875, 1.5033203125, 1.55517578125, 1.39970703125]

    @property
    def auroc_p_value(self):
        mu = type(self)._auroc_means[self._network - 1]
        sigma = type(self)._auroc_stds[self._network - 1]
        return norm.cdf(1 - self.auroc, mu, sigma)

    @property
    def aupr_p_value(self):
        b = type(self)._aupr_bs[self._network - 1]
        c = type(self)._aupr_cs[self._network - 1]
        x_max = type(self)._aupr_x_maxs[self._network - 1]
        h_max = type(self)._aupr_h_maxs[self._network - 1]
        x = self.aupr
        x_por = 0.835
        x2 = x_max * x_por + x * (1 - x_por)
        w = h_max * (b ** (-1 / c)) / c
        return w * gammainc(b * (x2 - x_max) ** c, 1 / c)
