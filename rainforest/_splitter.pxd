import numpy as np
cimport numpy as np

from ._criterion cimport RegressionCriterion

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cdef struct SplitRecord:
        # Data to track sample split
        SIZE_t feature         # Which feature to split on.
        SIZE_t pos             # Split samples array at the given position,
                               # i.e. count of samples below threshold for feature.
                               # pos is >= end if the node is a leaf.
        double threshold       # Threshold to split at.
        double improvement     # Impurity improvement given parent node.
        double impurity_left   # Impurity of the left split.
        double impurity_right  # Impurity of the right split.

cdef class Splitter:
    # The splitter searches in the input space for a feature and a threshold
    # to split the samples samples[start:end].
    #
    # The impurity computations are delegated to a criterion object.

    # Internal structures
    cdef public RegressionCriterion criterion      # Impurity criterion
    cdef public SIZE_t max_features      # Number of features to test
    # cdef public SIZE_t min_samples_leaf  # Min samples in a leaf
    # cdef public double min_weight_leaf   # Minimum weight in a leaf

    cdef object random_state             # Random state
    cdef UINT32_t rand_r_state           # sklearn_rand_r random number state

    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t n_samples                # X.shape[0]
    cdef double weighted_n_samples       # Weighted number of samples
    cdef SIZE_t* features                # Feature indices in X
    cdef SIZE_t* constant_features       # Constant features indices
    cdef SIZE_t n_features               # X.shape[1]
    cdef DTYPE_t* feature_values         # temp. array holding feature values
    cdef DOUBLE_t* feature_weight
    cdef SIZE_t* feature_wight_map

    cdef SIZE_t start                    # Start position for the current node
    cdef SIZE_t end                      # End position for the current node

    cdef DOUBLE_t* y
    cdef DOUBLE_t* sample_weight

    cdef DTYPE_t[:,:] X

    cdef INT32_t[:,:] X_idx_sorted
    cdef SIZE_t n_total_samples
    cdef SIZE_t* sample_mask

    # The samples vector `samples` is maintained by the Splitter object such
    # that the samples contained in a node are contiguous. With this setting,
    # `node_split` reorganizes the node samples `samples[start:end]` in two
    # subsets `samples[start:pos]` and `samples[pos:end]`.

    # The 1-d  `features` array of size n_features contains the features
    # indices and allows fast sampling without replacement of features.

    # The 1-d `constant_features` array of size n_features holds in
    # `constant_features[:n_constant_features]` the feature ids with
    # constant values for all the samples that reached a specific node.
    # The value `n_constant_features` is given by the parent node to its
    # child nodes.  The content of the range `[n_constant_features:]` is left
    # undefined, but preallocated for performance reasons
    # This allows optimization with depth-based tree building.

    # Methods
    cdef int init(self, np.ndarray X, np.ndarray y,
                  DOUBLE_t* sample_weight, DOUBLE_t* feature_weight,
                  np.ndarray X_idx_sorted) except -1

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) nogil except -1

    cdef int node_split(self,
                        double impurity,   # Impurity of the node
                        SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1

    cdef double node_value(self) nogil

    cdef double node_impurity(self) nogil