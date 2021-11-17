# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
#
from libc.stdlib cimport malloc, free

cdef ChainedList* create_ordered(int N) nogil:
   # Create a vector of size N, s.t. vecotr = {0, 1, 2, ..., N-1}
   cdef int ii
   cdef ChainedList* head
   cdef ChainedList* current
   cdef ChainedList* new_node

   # empty chained list
   if N==0:
       return <ChainedList*>NULL

   # there is at least one:
   head = <ChainedList*>malloc(sizeof(ChainedList))
   head.val = 0
   head.size = 1
   head.next = NULL
   head.last = head

   if N==1:
       return head

   # there are at least 2
   current = head

   for ii in range(1, N):
       new_node = <ChainedList*>malloc(sizeof(ChainedList))
       new_node.val = ii
       new_node.next = NULL
       current.next = new_node
       current = new_node
       head.size += 1

   head.last = current
   return head


cdef void pop_at_pos(ChainedList** head, int n) nogil:
  # Pop the element at the n-th position
  cdef ChainedList* to_pop
  cdef ChainedList* previous

  if n==0:
      to_pop = head[0]
      head[0] = head[0].next
      head[0].size = to_pop.size - 1
      head[0].last = to_pop.last
      free(to_pop)
      return

  to_pop = head[0].next
  previous = head[0]
  for _ in range(1, n):
      previous = to_pop
      to_pop = to_pop.next

  previous.next = to_pop.next
  if to_pop.next == NULL:
      head[0].last = previous

  head[0].size-=1
  free(to_pop)
  return


cdef double get_at_pos(ChainedList* head, int n) nogil:
  # Get the n-th element
  cdef ChainedList* current

  current = head
  for _ in range(0, n):
      current = current.next

  return current.val


cdef void push_back(ChainedList** head, double val) nogil:
  # adding element at the end of the chain
  cdef ChainedList* new_node

  new_node = <ChainedList*>malloc(sizeof(ChainedList))
  new_node.val = val
  new_node.next = NULL

  if head[0] == NULL:
      new_node.size = 1
      new_node.last = new_node
      head[0] = new_node
      return

  head[0].last.next = new_node
  head[0].last = new_node
  head[0].size += 1
  return


cdef void free_cl(ChainedList** head) nogil:
  cdef ChainedList* current

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
