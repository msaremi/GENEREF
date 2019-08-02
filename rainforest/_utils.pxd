import numpy as np
cimport numpy as np

from _tree cimport Node

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (DTYPE_t*)
    (SIZE_t*)
    (unsigned char*)
    # (WeightedPQueueRecord*)
    (DOUBLE_t*)
    (DOUBLE_t**)
    (Node*)
    # (Cell*)
    (Node**)
    (StackRecord*)
    # (PriorityHeapRecord*)

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *

cdef SIZE_t rand_int(SIZE_t low, SIZE_t high, UINT32_t* random_state) nogil

cdef struct StackRecord:
    SIZE_t start
    SIZE_t end
    SIZE_t depth
    SIZE_t parent
    bint is_left
    double impurity
    SIZE_t n_constant_features

cdef class Stack:
    cdef SIZE_t capacity
    cdef SIZE_t top
    cdef StackRecord* stack_

    cdef bint is_empty(self) nogil
    cdef int push(self, SIZE_t start, SIZE_t end, SIZE_t depth, SIZE_t parent,
                  bint is_left, double impurity,
                  SIZE_t n_constant_features) nogil except -1
    cdef int pop(self, StackRecord* res) nogil

