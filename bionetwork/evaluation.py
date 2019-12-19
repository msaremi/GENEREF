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


# class Evaluator:
#     def __init__(self):
#         self._labels = None
#         self._scores = None
#         self._fn = None
#         self._tn = None
#         self._x = None
#         self._y = None
#         self._m = None
#         self._n = None
#
#     def fit(self, labels, scores):
#         self._labels = labels = np.asarray(labels, dtype=np.int32).ravel()
#         self._scores = scores = np.asarray(scores, dtype=np.float64).ravel()
#
#         z = np.stack((scores, labels)).T
#         z = z[z[:, 0].argsort(), ]
#         c = z.shape[0]
#         _, ind = np.unique(z[:, 0], return_index=True)
#         w = ind.size
#         ind = np.append(ind, c)
#         xy = np.zeros((c,))
#         fn = np.zeros((w+1,))
#         tn = np.zeros((w+1,))
#         m = n = 0
#
#         for i in range(w):
#             inds = np.array(range(ind[i], ind[i + 1]))
#             labels = z[inds, 1]
#             pos_labels = labels == 1
#             neg_labels = labels == 0
#             mdupl = sum(pos_labels)
#             ndupl = sum(neg_labels)
#             xy[inds[pos_labels]] = n + ndupl / 2
#             xy[inds[neg_labels]] = m + mdupl / 2
#             n = n + ndupl
#             m = m + mdupl
#             fn[i + 1] = m
#             tn[i + 1] = n
#
#         self._fn = fn
#         self._tn = tn
#         self._x = xy[z[:, 1] == 1] / n
#         self._y = xy[z[:, 1] == 0] / m
#         self._m = m
#         self._n = n
#         return self
#
#     @property
#     def tpr(self):
#         return 1 - self._fn / self._m  # / self._fn[-1] #sum(self._z[:, 1])
#
#     @property
#     def fpr(self):
#         return 1 - self._tn / self._n  # / sum(1 - self._z[:, 1])
#
#     @property
#     def ppv(self):
#         with np.errstate(divide='ignore', invalid='ignore'):
#             return (self._m - self._fn) / ((self._m - self._fn) + (self._n - self._tn))
#
#     @property
#     def auc(self):
#         return np.mean(self._x)  # sum(self._xy[self._z[:, 1] == 1]) / self._fn[-1] / self._tn[-1]
#
#     @property
#     def auc_p_value(self):
#         sigma = np.sqrt(np.var(self._x)/self._m + np.var(self._y)/self._n)
#         mu = 0.5
#         return norm.cdf(1 - self.auc, mu, sigma)
#
#     @property
#     def aupr(self):
#         pass
#
#     @property
#     def aupr_p_value(self):
#         mu = self._m / (self._m + self._n)
#         sigma = 500
#         # TODO: compute sigma
#         return norm.cdf(1 - self.auc, mu, sigma)
#
#     @property
#     def labels(self):
#         return self._labels
#
#     @property
#     def scores(self):
#         return self._scores
#
#     # Old values
#
#     @property
#     def auroc(self):
#         return roc_auc_score(self.labels, self.scores)
#
#     @property
#     def roc(self):
#         return roc_curve(self.labels, self.scores, drop_intermediate=False)
#
#     @property
#     def prc(self):
#         return precision_recall_curve(self.labels, self.scores)
#
#     @property
#     def auprval(self):
#         precision, recall, _ = self.prc
#         return auc(recall, precision)