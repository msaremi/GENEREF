import os
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


class Evaluator:
    """Abstract class. Use one of the derived classes"""

    _probs = {}
    _xs = None

    def __init__(self, network: str):
        self._network = int(network[-1])
        self._labels = None
        self._scores = None

    @classmethod
    def _load_probs(cls, metric):
        if metric not in cls._probs:
            resource_path = os.path.abspath(os.path.join(os.getcwd(), "resources", cls.__name__))
            path = os.path.join(resource_path, "%s.npy" % metric.upper())
            cls._probs[metric] = np.load(path)

            if cls._xs is None:
                path = os.path.join(resource_path, "Xs.npy")
                cls._xs = np.load(path) \
                    if os.path.exists(path) \
                    else np.linspace(0, 1, cls._probs[metric].shape[1])

        return cls._probs[metric], cls._xs

    def fit(self, labels, scores):
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
        probs, xs = type(self)._load_probs("AUROC")
        return np.interp(self.auroc, xs, probs[self._network - 1, :])

    @property
    def aupr_p_value(self):
        probs, xs = type(self)._load_probs("AUPR")
        return np.interp(self.aupr, xs, probs[self._network - 1, :])


class SimpleEvaluator(Evaluator):
    """Evaluator for Networks without self-loop"""
    def fit(self, labels, scores):
        idx = ~np.eye(labels.shape[0], dtype=bool)
        super().fit(labels[idx], scores[idx])


class GenericEvaluator(Evaluator):
    """Evaluator for networks whose null-distribution is unknown"""
    @property
    def auroc_p_value(self):
        raise Exception("The auroc_p_value method can not be called on generic evaluators")

    @property
    def aupr_p_value(self):
        raise Exception("The aupr_p_value method can not be called on generic evaluators")


class GenericSimpleEvaluator(SimpleEvaluator):
    """Evaluator for networks without self-loops whose null-distribution is unknown"""
    @property
    def auroc_p_value(self):
        raise Exception("The auroc_p_value method can not be called on generic evaluators")

    @property
    def aupr_p_value(self):
        raise Exception("The aupr_p_value method can not be called on generic evaluators")


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


class DREAM4Evaluator(SimpleEvaluator):
    """Abstract class. Use either of DREAM4S10Evaluator and DREAM4S100Evaluator"""
    pass


class DREAM4S10Evaluator(DREAM4Evaluator):
    """DREAM4 Size 10 networks evaluator"""
    pass


class DREAM4S100Evaluator(DREAM4Evaluator):
    """DREAM4 Size 100 networks evaluator"""
    pass
