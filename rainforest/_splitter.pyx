from ._criterion cimport RegressionCriterion

from libc.stdlib cimport free
from libc.stdlib cimport qsort
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdio cimport printf

import numpy as np
cimport numpy as np

import cython
cimport cython

from ._utils cimport RAND_R_MAX
from ._utils cimport safe_realloc
from ._utils cimport rand_int

cdef double INFINITY = np.inf
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7

cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY

cdef class Splitter:
    """Abstract splitter class.
    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """
    def __cinit__(self, RegressionCriterion criterion, SIZE_t max_features, object random_state):
        self.criterion = criterion

        self.samples = NULL
        self.n_samples = 0
        self.features = NULL
        self.feature_weight = NULL
        self.feature_wight_map = NULL
        self.n_features = 0
        self.feature_values = NULL

        self.y = NULL
        self.sample_weight = NULL

        self.max_features = max_features
        self.random_state = random_state

        self.X = None
        self.X_idx_sorted = None
        self.sample_mask = NULL

    def __dealloc__(self):
        free(self.sample_mask)
        free(self.samples)
        free(self.features)
        free(self.constant_features)
        free(self.feature_values)

    cdef int init(self, np.ndarray X,
                   np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                   DOUBLE_t* sample_weight, DOUBLE_t* feature_weight,
                   np.ndarray X_idx_sorted) except -1:
        # printf("initcalled\n")
        """Initialize the splitter.
        Take in the input data X, the target Y, and optional sample weights.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.
        y : numpy.ndarray, dtype=DOUBLE_t
            This is the vector of targets, or true labels, for the samples
        sample_weight : numpy.ndarray, dtype=DOUBLE_t (optional)
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight.
        """

        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef SIZE_t n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)
        cdef SIZE_t i
        cdef SIZE_t j = 0
        cdef double weighted_n_samples = 0.0

        if sample_weight == NULL:
            for i in range(n_samples):
                samples[i] = i

            self.n_samples = n_samples
            self.weighted_n_samples = <double>n_samples
        else:
            for i in range(n_samples):
                if sample_weight[i] != 0.0:
                    weighted_n_samples += sample_weight[i]
                    samples[j] = i
                    j += 1

            self.n_samples = j
            self.weighted_n_samples = weighted_n_samples

        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t* features = safe_realloc(&self.features, n_features)
        j = 0

        if feature_weight == NULL:
            for i in range(n_features):
                features[i] = i

            self.n_features = n_features
        else:
            for i in range(n_features):
                if feature_weight[i] != 0.0:
                    features[j] = i
                    j += 1

            self.n_features = j

        safe_realloc(&self.feature_values, n_samples)
        safe_realloc(&self.constant_features, n_features)

        self.y = <DOUBLE_t*> y.data

        self.sample_weight = sample_weight
        self.feature_weight = feature_weight

        # cdef np.ndarray X_ndarray = X
        self.X = X

        self.X_idx_sorted = X_idx_sorted

        self.n_total_samples = X.shape[0]
        safe_realloc(&self.sample_mask, self.n_total_samples)
        memset(self.sample_mask, 0, self.n_total_samples*sizeof(SIZE_t))

        return 0

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) nogil except -1:
        """Reset splitter on node samples[start:end].
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        Parameters
        ----------
        start : SIZE_t
            The index of the first sample to consider
        end : SIZE_t
            The index of the last sample to consider
        weighted_n_node_samples : numpy.ndarray, dtype=double pointer
            The total weight of those samples
        """

        self.start = start
        self.end = end

        self.criterion.init(self.y,
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples,
                            start,
                            end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0

    @cython.boundscheck(False)
    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Find the best split on node samples[start:end]
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features
        cdef double current_feature_weight = 1.0

        cdef DTYPE_t[:,:] X = self.X
        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t max_features = self.max_features
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef INT32_t[:,:] X_idx_sorted = self.X_idx_sorted
        cdef SIZE_t* sample_mask = self.sample_mask

        cdef SplitRecord best, current
        cdef double current_proxy_improvement = -INFINITY
        cdef double best_proxy_improvement = -INFINITY

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j
        cdef SIZE_t tmp
        cdef SIZE_t p
        cdef SIZE_t feature_idx_offset
        cdef SIZE_t feature_offset
        cdef SIZE_t i
        cdef SIZE_t j

        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef DTYPE_t current_feature_value
        cdef SIZE_t partition_end

        _init_split(&best, end)

        # if self.presort == 1:
        for p in range(start, end):
            sample_mask[samples[p]] = 1

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1
            # TODO: max features is not sqrt(features) yet, fix it in the ensemble class
            # why n_drawn_constants? according to sklearn documentation for decision tree, the search for a split does
            # not stop until at least one valid partition of the node samples is found, even if it requires to
            # effectively inspect more than max_features features.

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                features[f_j], features[n_drawn_constants] = features[n_drawn_constants], features[f_j]
                n_drawn_constants += 1
            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[
                current.feature = features[f_j]

                # feature_offset = self.X_feature_stride * current.feature
                # printf("current_Feature_weight: %g\n", current_feature_weight)
                # Sort samples along that feature; either by utilizing
                # presorting, or by copying the values into an array and
                # sorting the array in a manner which utilizes the cache more
                # effectively.
                # Always presort available
                # if self.presort == 1:
                p = start

                for i in range(self.n_total_samples):
                    j = X_idx_sorted[i, current.feature] #[i + feature_idx_offset]
                    if sample_mask[j] == 1:
                        samples[p] = j
                        Xf[p] = X[j, current.feature]
                        p += 1

                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    if self.feature_weight != NULL:
                        current_feature_weight = self.feature_weight[current.feature]

                    # Evaluate all splits
                    self.criterion.reset(current_feature_weight)
                    p = start

                    while p < end:
                        while (p + 1 < end and
                               Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD):
                            p += 1

                        # (p + 1 >= end) or (X[samples[p + 1], current.feature] >
                        #                    X[samples[p], current.feature])
                        p += 1
                        # (p >= end) or (X[samples[p], current.feature] >
                        #                X[samples[p - 1], current.feature])

                        if p < end:
                            current.pos = p

                            # Reject if min_samples_leaf is not guaranteed
                            if (((current.pos - start) < 1) or
                                    ((end - current.pos) < 1)):
                                continue

                            self.criterion.update(current.pos)

                            # Reject if min_weight_leaf is not satisfied
                            # if ((self.criterion.weighted_n_left < min_weight_leaf) or
                            #         (self.criterion.weighted_n_right < min_weight_leaf)):
                            #     continue

                            current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                            if current_proxy_improvement > best_proxy_improvement:
                                best_proxy_improvement = current_proxy_improvement
                                # sum of halves is used to avoid infinite value
                                current.threshold = Xf[p - 1] / 2.0 + Xf[p] / 2.0

                                if ((current.threshold == Xf[p]) or
                                    (current.threshold == INFINITY) or
                                    (current.threshold == -INFINITY)):
                                    current.threshold = Xf[p - 1]

                                best = current  # copy

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            # feature_offset = X_feature_stride * best.feature
            partition_end = end
            p = start

            while p < partition_end:
                if X[samples[p], best.feature] <= best.threshold:
                    p += 1

                else:
                    partition_end -= 1

                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp

            self.criterion.reset(current_feature_weight)
            self.criterion.update(best.pos)
            best.improvement = self.criterion.impurity_improvement(impurity)
            self.criterion.children_impurity(&best.impurity_left,
                                             &best.impurity_right)

        # Reset sample mask
        # if self.presort == 1:
        for p in range(start, end):
            sample_mask[samples[p]] = 0

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants
        return 0

    cdef double node_value(self) nogil:
        """Copy the value of node samples[start:end] into dest."""

        return self.criterion.node_value()

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node."""
        return self.criterion.node_impurity()

