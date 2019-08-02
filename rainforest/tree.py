from abc import ABCMeta
from abc import abstractmethod

from sklearn.utils import check_array
from sklearn.utils import check_random_state
# import _tree
import numpy as np
import numbers
from ._criterion import MSE
from ._splitter import Splitter
from ._tree import Tree, TreeBuilder
from ._tree import DTYPE, DOUBLE

# DTYPE = _tree.DTYPE
# DOUBLE = _tree.DOUBLE


class RegressionTree:#(BaseEstimator, metaclass=ABCMeta):
    def __init__(self,
                 criterion="mse",
                 max_features=None,
                 random_state=None):
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None, feature_weight=None):
        random_state = check_random_state(self.random_state)

        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse=False)
            y = check_array(y, ensure_2d=False, dtype=None)

        # Determine output settings
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        # self.classes_ = [None]
        # self.n_classes_ = [1]

        # self.n_classes_ = np.array([1], dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        # max_depth = ((2 ** 31) - 1)
        # max_leaf_nodes = (-1)
        # min_samples_leaf = 1
        # min_samples_split = 2

        if self.max_features is None:
            max_features = self.n_features_
        elif self.max_features == "sqrt":
            max_features = max(1, int(np.sqrt(self.n_features_)))
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features
        # print(max_features)

        # if len(y) != n_samples:
        #     raise ValueError("Number of labels=%d does not match "
        #                      "number of samples=%d" % (len(y), n_samples))
        # if not 0 <= self.min_weight_fraction_leaf <= 0.5:
        #     raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        # if max_depth <= 0:
        #     raise ValueError("max_depth must be greater than zero. ")
        # if not (0 < max_features <= self.n_features_):
        #     raise ValueError("max_features must be in (0, n_features]")
        # if not isinstance(max_leaf_nodes, (numbers.Integral, np.integer)):
        #     raise ValueError("max_leaf_nodes must be integral number but was "
        #                      "%r" % max_leaf_nodes)
        # if -1 < max_leaf_nodes < 2:
        #     raise ValueError(("max_leaf_nodes {0} must be either None "
        #                       "or larger than 1").format(max_leaf_nodes))

        if sample_weight is not None:
            if sample_weight.dtype != DOUBLE or not sample_weight.flags.contiguous:
                sample_weight = np.ascontiguousarray(sample_weight, dtype=DOUBLE)
            # if len(sample_weight.shape) > 1:
            #     raise ValueError("Sample weights array has more "
            #                      "than one dimension: %d" %
            #                      len(sample_weight.shape))
            # if len(sample_weight) != n_samples:
            #     raise ValueError("Number of weights=%d does not match "
            #                      "number of samples=%d" %
            #                      (len(sample_weight), n_samples))

        if feature_weight is not None:
            if feature_weight.dtype != DOUBLE or not feature_weight.flags.contiguous:
                feature_weight = np.ascontiguousarray(feature_weight, dtype=DOUBLE)

        # Set min_weight_leaf from min_weight_fraction_leaf
        # min_weight_leaf = 0
        # min_impurity_split = 1e-7
        #
        # presort = True

        # # If multiple trees are built on the same dataset, we only want to
        # # presort once. Splitters now can accept presorted indices if desired,
        # # but do not handle any presorting themselves. Ensemble algorithms
        # # which desire presorting must do presorting themselves and pass that
        # # matrix into each tree.
        # if X_idx_sorted is None and presort:
        #     X_idx_sorted = np.asfortranarray(np.argsort(X, axis=0),
        #                                      dtype=np.int32)
        #
        # if presort and X_idx_sorted.shape != X.shape:
        #     raise ValueError("The shape of X (X.shape = {}) doesn't match "
        #                      "the shape of X_idx_sorted (X_idx_sorted"
        #                      ".shape = {})".format(X.shape,
        #                                            X_idx_sorted.shape))

        # Build tree
        criterion = MSE(n_samples)
        splitter = Splitter(criterion,
                            self.max_features_,
                            random_state)
        self.tree_ = Tree(self.n_features_)
        builder = TreeBuilder(splitter)
        builder.build(self.tree_, X, y, sample_weight, feature_weight, X_idx_sorted)

        # self.n_classes_ = self.n_classes_[0]
        # self.classes_ = self.classes_[0]

        return self

    @property
    def feature_importances_(self):

        return self.tree_.compute_feature_importances()