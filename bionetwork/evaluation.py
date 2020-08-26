import os
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from typing import Union


class Evaluator:
    """Abstract class. Use one of the derived classes"""

    _probs = {}

    def __init__(self, network: str = None, p_values_path: str = None, self_loops: bool = False):
        self._self_loops = self_loops
        self._p_values_path = p_values_path
        self._network = network
        self._labels = None
        self._scores = None

    def _load_probs(self, metric):
        if metric not in self._probs:
            path = os.path.join(self._p_values_path, "%s_%s.npy" % (self._network, metric.lower()))
            data = np.squeeze(np.load(path))

            if data.ndim == 1:
                x = np.linspace(0, 1, data.shape[0])
                data = np.vstack((x, data))

            self._probs[metric] = data

        return self._probs[metric][0, :], self._probs[metric][1, :]

    def fit(self, labels, scores):
        if not self._self_loops:
            idx = ~np.eye(labels.shape[0], dtype=bool)
            labels = labels[idx]
            scores = scores[idx]

        self._labels = np.asarray(labels, dtype=np.int32).ravel()
        self._scores = np.asarray(scores, dtype=np.float64).ravel()

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
    def score(self):
        return -np.mean([np.log10(self.auroc_p_value), np.log10(self.aupr_p_value)])

    @property
    def score_aupr(self):
        return -np.log10(self.aupr_p_value)

    @property
    def auroc_p_value(self):
        if self._p_values_path is None:
            raise Exception("AUROC p-value cannot be computed for this network. "
                            "This network does not have the p-value distribution.")

        try:
            xs, probs = self._load_probs("AUROC")
        except FileNotFoundError:
            raise Exception("AUROC p-value cannot be computed for this network")

        return np.interp(self.auroc, xs, probs)

    @property
    def aupr_p_value(self):
        if self._p_values_path is None:
            raise Exception("AUPR p-value cannot be computed for this network. "
                            "This network does not have the p-value distribution.")

        try:
            xs, probs = self._load_probs("AUPR")
        except FileNotFoundError:
            raise Exception("AUPR p-value cannot be computed for this network")

        return np.interp(self.aupr, xs, probs)


class DREAM5Evaluator(Evaluator):
    """Dream5 Network Evaluation"""
    def fit(self, labels, scores):
        # Sine not all genes are in the goldstandards, extra genes are removed
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
