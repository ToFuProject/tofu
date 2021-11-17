# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
#

ctypedef struct ChainedList:
    double val
    int size
    ChainedList* next
    ChainedList* last

cdef ChainedList* create_ordered(int N) nogil
cdef void pop_at_pos(ChainedList** head, int n) nogil
cdef double get_at_pos(ChainedList* head, int n) nogil
cdef void push_back(ChainedList** head, double val) nogil
cdef void free_cl(ChainedList** head) nogil
