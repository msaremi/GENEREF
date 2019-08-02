from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.math cimport log as ln

import numpy as np
cimport numpy as np

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *:
    # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    # 0.20.1 to crash.
    ptr = p[0]
    cdef size_t nbytes = nelems * sizeof(ptr[0])

    if nbytes / sizeof(ptr[0]) != nelems:
        # Overflow in the multiplication
        with gil:
            raise MemoryError("could not allocate (%d * %d) bytes" % (nelems, sizeof(ptr[0])))

    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)

    if tmp == NULL:
        with gil:
            raise MemoryError("could not allocate %d bytes" % nbytes)

    p[0] = tmp
    return tmp  # for convenience

cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>0x80000000)
    # return seed[0] % (<UINT32_t>RAND_R_MAX + 1)

cdef inline SIZE_t rand_int(SIZE_t low, SIZE_t high,
                            UINT32_t* random_state) nogil:
    """Generate a random integer in [low; end)."""
    return low + our_rand_r(random_state) % (high - low)

cdef class Stack:
    def __cinit__(self, SIZE_t capacity):
        self.capacity = capacity
        self.top = 0
        self.stack_ = <StackRecord*> malloc(capacity * sizeof(StackRecord))

    def __dealloc__(self):
        free(self.stack_)

    cdef bint is_empty(self) nogil:
        return self.top <= 0

    cdef int push(self, SIZE_t start, SIZE_t end, SIZE_t depth, SIZE_t parent,
                  bint is_left, double impurity,
                  SIZE_t n_constant_features) nogil except -1:
        """Push a new element onto the stack.
        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef SIZE_t top = self.top
        cdef StackRecord* stack = NULL

        # Resize if capacity not sufficient
        if top >= self.capacity:
            self.capacity *= 2
            # Since safe_realloc can raise MemoryError, use `except -1`
            safe_realloc(&self.stack_, self.capacity)

        stack = self.stack_
        stack[top].start = start
        stack[top].end = end
        stack[top].depth = depth
        stack[top].parent = parent
        stack[top].is_left = is_left
        stack[top].impurity = impurity
        stack[top].n_constant_features = n_constant_features

        # Increment stack pointer
        self.top = top + 1
        return 0

    cdef int pop(self, StackRecord* res) nogil:
        """Remove the top element from the stack and copy to ``res``.
        Returns 0 if pop was successful (and ``res`` is set); -1
        otherwise.
        """
        cdef SIZE_t top = self.top
        cdef StackRecord* stack = self.stack_

        if top <= 0:
            return -1

        res[0] = stack[top - 1]
        self.top = top - 1

        return 0