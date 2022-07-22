# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
#

ctypedef struct SortedSet:
    double val
    SortedSet* next
    SortedSet* last

cdef void insert(SortedSet** head, double val) nogil
cdef int count(SortedSet* head, double val) nogil
cdef void free_ss(SortedSet** head) nogil
