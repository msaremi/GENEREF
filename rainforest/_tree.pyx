from libc.stdlib cimport free
from libc.stdio cimport printf
from libc.stdlib cimport realloc
from libc.math cimport fabs
from libc.string cimport memcpy
from libc.string cimport memset

from ._utils cimport Stack
from ._utils cimport StackRecord
from ._utils cimport safe_realloc

import numpy as np
cimport numpy as np

cdef double INFINITY = np.inf

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef class Tree:
    def __cinit__(self, int n_features):
        self.n_features = n_features
        self.node_count = 0
        self.capacity = 0
        self.nodes = NULL

    def __dealloc__(self):
        free(self.nodes)

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples,
                          double weighted_n_node_samples) nogil except -1:
        """Add a node to the tree.
        The new node registers itself as the child of its parent.
        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return <SIZE_t>(-1)

        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED

        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold

        self.node_count += 1

        return node_id

    cdef int _resize_c(self, SIZE_t capacity=<SIZE_t>(-1)) nogil except -1:
        """Guts of _resize
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == <SIZE_t>(-1):
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        safe_realloc(&self.nodes, capacity)

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    cpdef compute_feature_importances(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        cdef Node* left
        cdef Node* right
        cdef Node* nodes = self.nodes
        cdef Node* node = nodes
        cdef Node* end_node = node + self.node_count

        cdef double normalizer = 0.

        cdef np.ndarray[np.float64_t, ndim=1] importances
        importances = np.zeros((self.n_features,))
        cdef DOUBLE_t* importance_data = <DOUBLE_t*>importances.data

        with nogil:
            while node != end_node:
                if node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    left = &nodes[node.left_child]
                    right = &nodes[node.right_child]

                    # printf("%g\n", (
                    #     node.impurity))

                    importance_data[node.feature] += (
                        node.weighted_n_node_samples * node.impurity
                        - left.weighted_n_node_samples * left.impurity
                        - right.weighted_n_node_samples * right.impurity)
                node += 1

        importances /= nodes[0].weighted_n_node_samples

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                importances /= normalizer

        return importances


cdef class TreeBuilder:
    cdef inline _check_input(self, np.ndarray X, np.ndarray y,
                             np.ndarray sample_weight):
        """Check input dtype, layout and format"""
        if X.dtype != DTYPE:
            X = np.asfortranarray(X, dtype=DTYPE)

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if (sample_weight is not None and
            (sample_weight.dtype != DOUBLE or
            not sample_weight.flags.contiguous)):
                sample_weight = np.asarray(sample_weight, dtype=DOUBLE, order="C")

        return X, y, sample_weight

    def __cinit__(self, Splitter splitter):
        self.splitter = splitter

    cpdef build(self, Tree tree, np.ndarray X, np.ndarray y, np.ndarray sample_weight=None, np.ndarray feature_weight=None, np.ndarray X_idx_sorted=None):
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL if sample_weight is None else <DOUBLE_t*> sample_weight.data
        cdef DOUBLE_t* feature_weight_ptr = NULL if feature_weight is None else <DOUBLE_t*> feature_weight.data

        cdef int init_capacity = 2047
        tree._resize_c(init_capacity)

        cdef Splitter splitter = self.splitter
        splitter.init(X, y, sample_weight_ptr, feature_weight_ptr, X_idx_sorted)

        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef double weighted_n_samples = splitter.weighted_n_samples
        cdef double weighted_n_node_samples
        cdef SplitRecord split
        cdef SIZE_t node_id

        cdef double threshold
        cdef double impurity = INFINITY
        cdef SIZE_t n_constant_features
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0)

        with nogil:
            while not stack.is_empty():
                stack.pop(&stack_record)

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features
                # printf("info %d %d %d %d\n", depth, splitter.features[0], splitter.features[1], splitter.features[2])
                n_node_samples = end - start
                splitter.node_reset(start, end, &weighted_n_node_samples)

                is_leaf = (n_node_samples < 2)

                # printf("isleaf: %d, %d\n", n_node_samples, is_leaf)

                if first:
                    impurity = splitter.node_impurity()
                    first = 0

                if not is_leaf:
                    splitter.node_split(impurity, &split, &n_constant_features)
                    is_leaf = (is_leaf or split.pos >= end)

                # printf("Hel: %d, %g, %g\n", split.pos, split.impurity_left, split.impurity_right)

                node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                         split.threshold, impurity, n_node_samples,
                                         weighted_n_node_samples)

                if node_id == <SIZE_t>(-1):
                    rc = -1
                    break

                # splitter.node_value(tree.value + node_id * tree.value_stride)

                if not is_leaf:
                    # Push right child on stack
                    rc = stack.push(split.pos, end, depth + 1, node_id, 0,
                                    split.impurity_right, n_constant_features)
                    if rc == -1:
                        break

                    rc = stack.push(start, split.pos, depth + 1, node_id, 1,
                                    split.impurity_left, n_constant_features)
                    if rc == -1:
                        break

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

