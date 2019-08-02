import numpy as np
from scipy.stats import norm, beta
from scipy.special import gammainc
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

class DREAM4Evaluator:
    _auroc_means = [0.5, 0.5, 0.5, 0.5, 0.5]
    _auroc_stds = [0.021921, 0.018493, 0.02086, 0.02004, 0.020945]
    # _aupr_means = [0.018313, 0.025685, 0.020232, 0.021846, 0.02003]
    # _aupr_stds = [0.0016999, 0.0018413, 0.0017348, 0.0017632, 0.0017279]
    _aupr_x_maxs = [0.0193143, 0.0277462, 0.02165965, 0.02343955, 0.02152605]
    _aupr_h_maxs = [263.620981387474, 222.364217252398, 255.935098206657, 240.841777084961, 261.668070766632]
    _aupr_bs = [31700.820267609, 8719.82765760784, 8568.45635982301, 10659.2365536708, 5082.12445262662]
    _aupr_cs = [1.74599609375, 1.53232421875, 1.5033203125, 1.55517578125, 1.39970703125]

    def __init__(self, network: str):
        self._network = int(network[-1]) - 1
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
    def auroc_p_value(self):
        mu = type(self)._auroc_means[self._network]
        sigma = type(self)._auroc_stds[self._network]
        return norm.cdf(1 - self.auroc, mu, sigma)

    @property
    def aupr(self):
        precision, recall, _ = precision_recall_curve(self._labels, self._scores)
        return auc(recall, precision)

    @property
    def aupr_p_value(self):
        b = type(self)._aupr_bs[self._network]
        c = type(self)._aupr_cs[self._network]
        x_max = type(self)._aupr_x_maxs[self._network]
        h_max = type(self)._aupr_h_maxs[self._network]
        x = self.aupr
        x_por = 0.835
        x2 = x_max * x_por + x * (1 - x_por)
        w = h_max * (b ** (-1 / c)) / c
        return w * gammainc(b * (x2 - x_max) ** c, 1 / c)

    @property
    def score(self):
        return -np.mean([np.log10(self.auroc_p_value), np.log10(self.aupr_p_value)])

class Evaluator:
    def __init__(self):
        self._labels = None
        self._scores = None
        self._fn = None
        self._tn = None
        self._x = None
        self._y = None
        self._m = None
        self._n = None

    def fit(self, labels, scores):
        self._labels = labels = np.asarray(labels, dtype=np.int32).ravel()
        self._scores = scores = np.asarray(scores, dtype=np.float64).ravel()

        z = np.stack((scores, labels)).T
        z = z[z[:, 0].argsort(), ]
        c = z.shape[0]
        _, ind = np.unique(z[:, 0], return_index=True)
        w = ind.size
        ind = np.append(ind, c)
        xy = np.zeros((c,))
        fn = np.zeros((w+1,))
        tn = np.zeros((w+1,))
        m = n = 0

        for i in range(w):
            inds = np.array(range(ind[i], ind[i + 1]))
            labels = z[inds, 1]
            pos_labels = labels == 1
            neg_labels = labels == 0
            mdupl = sum(pos_labels)
            ndupl = sum(neg_labels)
            xy[inds[pos_labels]] = n + ndupl / 2
            xy[inds[neg_labels]] = m + mdupl / 2
            n = n + ndupl
            m = m + mdupl
            fn[i + 1] = m
            tn[i + 1] = n

        self._fn = fn
        self._tn = tn
        self._x = xy[z[:, 1] == 1] / n
        self._y = xy[z[:, 1] == 0] / m
        self._m = m
        self._n = n
        return self

    @property
    def tpr(self):
        return 1 - self._fn / self._m  # / self._fn[-1] #sum(self._z[:, 1])

    @property
    def fpr(self):
        return 1 - self._tn / self._n  # / sum(1 - self._z[:, 1])

    @property
    def ppv(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            return (self._m - self._fn) / ((self._m - self._fn) + (self._n - self._tn))

    @property
    def auc(self):
        return np.mean(self._x)  # sum(self._xy[self._z[:, 1] == 1]) / self._fn[-1] / self._tn[-1]

    @property
    def auc_p_value(self):
        sigma = np.sqrt(np.var(self._x)/self._m + np.var(self._y)/self._n)
        mu = 0.5
        return norm.cdf(1 - self.auc, mu, sigma)

    @property
    def aupr(self):
        pass

    @property
    def aupr_p_value(self):
        mu = self._m / (self._m + self._n)
        sigma = 500
        # TODO: compute sigma
        return norm.cdf(1 - self.auc, mu, sigma)

    @property
    def labels(self):
        return self._labels

    @property
    def scores(self):
        return self._scores

    # Old values

    @property
    def auroc(self):
        return roc_auc_score(self.labels, self.scores)

    @property
    def roc(self):
        return roc_curve(self.labels, self.scores, drop_intermediate=False)

    @property
    def prc(self):
        return precision_recall_curve(self.labels, self.scores)

    @property
    def auprval(self):
        precision, recall, _ = self.prc
        return auc(recall, precision)