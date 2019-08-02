from .tree import RegressionTree
from sklearn.utils import check_random_state
import numpy as np


class RegressionForest:
    def __init__(self, n_estimators=100, max_features=None, random_state=None):
        self._max_features = max_features
        random_state = check_random_state(random_state)
        rand_seeds = random_state.randint(0, 0x7FFFFFFF, n_estimators)
        self._estimators = [RegressionTree(rand_seeds[i], max_features=max_features) for i in range(n_estimators)]

    @staticmethod
    def _generate_sample_indices(random_state, n_samples):
        """Private function used to _parallel_build_trees function."""
        random_instance = check_random_state(random_state)
        sample_indices = random_instance.randint(0, n_samples, n_samples)

        return sample_indices

    def fit(self, X, y, sample_weight=None, feature_weight=None):
        n_samples = X.shape[0]
        X_idx_sorted = np.asfortranarray(np.argsort(X, axis=0), dtype=np.int32)

        if sample_weight is None:
            sample_weight = np.ones((n_samples,), dtype=np.float64)

        for estimator in self._estimators:
            indices = type(self)._generate_sample_indices(estimator.random_state, n_samples)
            sample_counts = np.bincount(indices, minlength=n_samples)
            curr_sample_weight = sample_weight * sample_counts
            estimator.fit(X, y, sample_weight=curr_sample_weight, X_idx_sorted=X_idx_sorted, feature_weight=feature_weight)
            # TODO: Check if estimators are initialized with different random states

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).
        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """

        all_importances = [estimator.feature_importances_ for estimator in self._estimators]

        return sum(all_importances) / len(self._estimators)
