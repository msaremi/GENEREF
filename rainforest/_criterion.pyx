import numpy as np
cimport numpy as np
from libc.stdio cimport printf
np.import_array()

cdef class RegressionCriterion:
    r"""Abstract regression criterion.
    This handles cases where the target is a continuous value, and is
    evaluated by computing the variance of the target values left and right
    of the split point. The computation takes linear time with `n_samples`
    by using ::
        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
    """

    # def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples):
    def __cinit__(self, SIZE_t n_samples):
        """Initialize parameters for this criterion.
        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted
        n_samples : SIZE_t
            The total number of samples to fit on
        """

        # Default values
        self.y = NULL
        self.sample_weight = NULL
        self.feature_weight = 0.0

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        # self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.sq_sum_total = 0.0

        # Allocate memory for the accumulators
        self.sum_total = 0.0 # <double*> calloc(n_outputs, sizeof(double))
        self.sum_left = 0.0 # <double*> calloc(n_outputs, sizeof(double))
        self.sum_right = 0.0 # <double*> calloc(n_outputs, sizeof(double))

    cdef int init(self, DOUBLE_t* y, DOUBLE_t* sample_weight, #DOUBLE_t* feature_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        # self.feature_weight = feature_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_i
        cdef DOUBLE_t w_y_i
        cdef DOUBLE_t w = 1.0

        self.sq_sum_total = 0.0
        self.sum_total = 0.0

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            y_i = y[i]
            w_y_i = w * y_i
            self.sum_total += w_y_i
            self.sq_sum_total += w_y_i * y_i
            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset(1.0)
        return 0

    cdef int reset(self, DOUBLE_t feature_weight) nogil except -1:
        """Reset the criterion at pos=start."""
        self.sum_left = 0.0
        self.sum_right = self.sum_total

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        self.feature_weight = feature_weight
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef double sum_left = self.sum_left

        cdef double* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef DOUBLE_t* y = self.y
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t y_i

        for p in range(pos, new_pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            y_i = y[i]
            sum_left += w * y_i

            self.weighted_n_left += w

        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        self.sum_left = sum_left
        self.sum_right = self.sum_total - sum_left
        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left, double* impurity_right) nogil:
        pass

    cdef double node_value(self) nogil:
        return self.sum_total / self.weighted_n_node_samples

    cdef double proxy_impurity_improvement(self) nogil:
        cdef double impurity_left
        cdef double impurity_right
        self.children_impurity(&impurity_left, &impurity_right)

        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)

    cdef double impurity_improvement(self, double impurity) nogil:
        cdef double impurity_left
        cdef double impurity_right

        self.children_impurity(&impurity_left, &impurity_right)

        return self.feature_weight * ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity - (self.weighted_n_right /
                             self.weighted_n_node_samples * impurity_right)
                          - (self.weighted_n_left /
                             self.weighted_n_node_samples * impurity_left)))


cdef class MSE(RegressionCriterion):
    """Mean squared error impurity criterion.
        MSE = var_left + var_right
    """

    cdef double node_impurity(self) nogil:
        return (self.feature_weight) * (self.sq_sum_total / self.weighted_n_node_samples
                - (self.sum_total / self.weighted_n_node_samples)**2.0)

    cdef double proxy_impurity_improvement(self) nogil:
        cdef double proxy_impurity_left = self.sum_left * self.sum_left
        cdef double proxy_impurity_right = self.sum_right * self.sum_right

        return (self.feature_weight) * (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""


        cdef DOUBLE_t* y = self.y
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef double sum_left = self.sum_left
        cdef double sum_right = self.sum_right

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t y_i

        for p in range(start, pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            y_i = y[i]
            sq_sum_left += w * y_i * y_i

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left[0] = (sq_sum_left / self.weighted_n_left
                            - (sum_left / self.weighted_n_left) ** 2.0)
        impurity_right[0] = (sq_sum_right / self.weighted_n_right
                             - (sum_right / self.weighted_n_right) ** 2.0)