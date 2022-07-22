# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
#
from libc.stdlib cimport malloc, free


cdef void insert(SortedSet** head, double val) nogil:
  # insert the value at the right position so all elements
  # are in increasing order, don't add if already present
  cdef SortedSet* adding
  cdef SortedSet* current
  cdef SortedSet* previous

  # set is empty
  if head[0] == NULL:
      adding = <SortedSet*>malloc(sizeof(SortedSet))
      adding.val = val
      adding.next = NULL
      adding.last = adding
      head[0] = adding
      return

  # adding on top
  if head[0].val > val:
      adding = <SortedSet*>malloc(sizeof(SortedSet))
      adding.val = val
      adding.next = head[0]
      adding.last = head[0].last
      head[0] = adding
      return

  # adding on bottom
  if head[0].last.val < val:
      adding = <SortedSet*>malloc(sizeof(SortedSet))
      adding.val = val
      adding.next = NULL
      head[0].last.next = adding
      head[0].last = adding
      return

  # if same as one of extremities, do nothing
  if head[0].val == val or head[0].last.val == val:
      return

  current = head[0].next
  previous = head[0]
  while current.val < val:
      previous = current
      current = current.next

  # already in set, nothing to do
  if current.val == val:
      return

  adding = <SortedSet*>malloc(sizeof(SortedSet))
  adding.val = val
  adding.next = current
  previous.next = adding

  return


cdef int count(SortedSet* head, double val) nogil:
  # returns 1 if val in set, else 0
  cdef SortedSet* current

  # outside of scope
  if head.val > val or head.last.val < val:
      return 0

  # is one of extremities
  if  head.val == val or head.last.val == val:
      return 1

  current = head[0].next
  while current.val <= val:
      if current.val == val:
          return 1
      current = current.next

  # we ended and didnt find a match
  return 0

cdef void free_ss(SortedSet** head) nogil:
  cdef SortedSet* current

  if head[0] == NULL :
      return

  previous = head[0]
  current = head[0].next
  while current != NULL:
      free(previous)
      previous = current
      current = current.next
  free(previous)
  return
