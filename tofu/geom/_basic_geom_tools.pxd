# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
################################################################################
# Utility functions for basic geometry (vector calculus, path, ...)
################################################################################
cimport cython
from cpython.array cimport array, clone
from libc.math cimport cos as Ccos, sin as Csin
from libc.math cimport atan2 as Catan2
from libc.math cimport sqrt as Csqrt
from libc.math cimport fabs as Cabs

# ==============================================================================
# =  Geometry global variables
# ==============================================================================

cdef double _VSMALL
cdef double _SMALL

# ==============================================================================
# =  Point in path
# ==============================================================================

cdef inline bint is_point_in_path(const int nvert,
                                  const double* vertx,
                                  const double* verty,
                                  const double testx,
                                  const double testy) nogil:
    """
    Computes if a point of coordiates (testx, testy) is in the polygon defined
    by nvert vertices of coordinates (vertx, verty).
    WARNING: the poly should be CLOSED
    Params
    ======
        nvert : int
           number of vertices in polygon
        vertx : double array
           x-coordinates of polygon
        verty : double array
           y-coordinate of polygon
        testx : double
           x-coordinate of point to be tested if in or out of polygon
        testy : double
           y-coordinate of point to be tested if in or out of polygon
    Returns
    =======
        bool : True if point is in the polygon, else False
    """
    cdef int i
    cdef bint c = 0
    for i in range(nvert):
        if ( ((verty[i]>testy) != (verty[i+1]>testy)) and
            (testx < (vertx[i+1]-vertx[i]) * (testy-verty[i]) \
             / (verty[i+1]-verty[i]) + vertx[i]) ):
            c = not c
    return c


cdef inline int is_point_in_path_vec(const int nvert,
                                     const double* vertx,
                                     const double* verty,
                                     const int npts,
                                     const double* testx,
                                     const double* testy,
                                     bint* is_in_path) nogil:
    """
    Computes if a series of points of coordiates (testx[ii], testy[ii]) is in
    the polygon defined by nvert vertices of coordinates (vertx, verty)
    Params
    ======
        nvert : int
           number of vertices in polygon
        vertx : double array
           x-coordinates of polygon
        verty : double array
           y-coordinate of polygon
        npts : int
           number of points to test if in poly or not
        testx : double array
           x-coordinates of points to be tested if in or out of polygon
        testy : double array
           y-coordinates of points to be tested if in or out of polygon
        is_in_path : bint array
           True if point is in the polygon, else False
    Returns
    =======
     The number of "true"
    """
    cdef int ii
    cdef int tot_true = 0
    cdef bint c = 0
    for ii in range(npts):
        c = is_point_in_path(nvert, vertx, verty, testx[ii], testy[ii])
        is_in_path[ii] = c
        tot_true += c*1
    return tot_true


# ==============================================================================
# =  Computing inverse of vector and sign of each element
# ==============================================================================

cdef inline void compute_inv_and_sign(const double[3] ray_vdir,
                                      int[3] sign,
                                      double[3] inv_direction) nogil:
    """
    Computes the inverse direction and sign of each coordinate of a vector.
    Params
    ======
    ray_vdir : (3) double array
       direction of the LOS
    sign : (3) int array <INOUT>
       for each coordinate of the direction, indicates if negative or not
       If sign[i] = 1, ray_vdir[i] < 0, else sign[i] = 0
    inv_direction : (3) double array
       Inverse on each axis of direction of LOS
    """
    cdef int t0 = 1000000
    # computing sign and direction
    for  ii in range(3):
        if ray_vdir[ii] * ray_vdir[ii] < _VSMALL:
            inv_direction[ii] = 1. / _VSMALL
        else:
            inv_direction[ii] = 1. / ray_vdir[ii]
        if ray_vdir[ii] < 0.:
            sign[ii] = 1
        else:
            sign[ii] = 0
    return

# ==============================================================================
# =  Computing Hypothenus
# =============================================================================

cdef inline array compute_hypot(double[::1] xpts, double[::1] ypts,
                                int npts=-1):
    cdef int ii
    cdef array hypot
    if npts == -1:
        npts  = xpts.shape[0]
    hypot = clone(array('d'), npts, False)
    for ii in range(npts):
        hypot[ii] = Csqrt(xpts[ii]*xpts[ii] + ypts[ii]*ypts[ii])
    return hypot

cdef inline double comp_min_hypot(double[::1] xpts, double[::1] ypts,
                                  int npts=-1):
    cdef int ii
    cdef double tmp
    cdef double hypot = xpts[0]*xpts[0] + ypts[0]*ypts[0]
    if npts == -1:
        npts  = xpts.shape[0]
    for ii in range(npts):
        tmp = xpts[ii]*xpts[ii] + ypts[ii]*ypts[ii]
        if tmp < hypot:
            hypot = tmp
    return Csqrt(hypot)
