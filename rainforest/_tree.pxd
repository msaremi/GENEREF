import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

from ._splitter cimport Splitter
from ._splitter cimport SplitRecord

cdef struct Node:
    # Base storage structure for the nodes in a Tree object

    SIZE_t left_child                    # id of the left child of the node
    SIZE_t right_child                   # id of the right child of the node
    SIZE_t feature                       # Feature used for splitting the node
    DOUBLE_t threshold                   # Threshold value at the node
    DOUBLE_t impurity                    # Impurity of the node (i.e., the value of the criterion)
    SIZE_t n_node_samples                # Number of samples at the node
    DOUBLE_t weighted_n_node_samples     # Weighted number of samples at the node

cdef class Tree:
    cdef public SIZE_t n_features        # Number of features in X
    cdef public SIZE_t node_count        # Counter for node IDs
    cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
    cdef Node* nodes                     # Array of nodes

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples,
                          double weighted_n_node_samples) nogil except -1
    cdef int _resize_c(self, SIZE_t capacity=*) nogil except -1
    cpdef compute_feature_importances(self, normalize=*)

cdef class TreeBuilder:
    # The TreeBuilder recursively builds a Tree object from training samples,
    # using a Splitter object for splitting internal nodes and assigning
    # values to leaves.
    #
    # This class controls the various stopping criteria and the node splitting
    # evaluation order, e.g. depth-first or best-first.

    cdef Splitter splitter              # Splitting algorithm

    cpdef build(self, Tree tree, np.ndarray X, np.ndarray y,
                np.ndarray sample_weight=*,
                np.ndarray feature_weight=*,
                np.ndarray X_idx_sorted=*)
    cdef _check_input(self, np.ndarray X, np.ndarray y, np.ndarray sample_weight)