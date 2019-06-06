# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
################################################################################
# Utility functions for basic geometry (vector calculus, path, ...)
################################################################################
cimport cython
from cython.parallel import prange
from cpython.array cimport array, clone
from libc.math cimport cos as Ccos, sin as Csin
from libc.math cimport atan2 as Catan2
from libc.math cimport sqrt as Csqrt
from libc.math cimport fabs as Cabs


# ==============================================================================
# =  Geometry global variables
# ==============================================================================
# Values defined in the *.pyx file
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
    cdef double* ptr_hypot
    if npts == -1:
        npts  = xpts.shape[0]
    hypot = clone(array('d'), npts, False)
    ptr_hypot = hypot.data.as_doubles
    with nogil:
        for ii in range(npts):
            ptr_hypot[ii] = Csqrt(xpts[ii]*xpts[ii] + ypts[ii]*ypts[ii])
    return hypot

cdef inline double comp_min_hypot(double[::1] xpts, double[::1] ypts,
                                  int npts=-1) nogil:
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

# ==============================================================================
# == Vector Calculus Helpers
# ==============================================================================
cdef inline void compute_cross_prod(const double[3] vec_a,
                                    const double[3] vec_b,
                                    double[3] res) nogil:
    res[0] = vec_a[1]*vec_b[2] - vec_a[2]*vec_b[1]
    res[1] = vec_a[2]*vec_b[0] - vec_a[0]*vec_b[2]
    res[2] = vec_a[0]*vec_b[1] - vec_a[1]*vec_b[0]
    return

cdef inline double compute_dot_prod(const double[3] vec_a,
                                    const double[3] vec_b) nogil:
    return vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1] + vec_a[2] * vec_b[2]

cdef inline double compute_norm(const double[3] vec) nogil:
    return Csqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])

cdef inline double compute_g(double s, double m2b2, double rm0sqr,
                             double m0sqr, double b1sqr) nogil:
    return s + m2b2 - rm0sqr*s / Csqrt(m0sqr*s*s + b1sqr)

cdef inline double compute_bisect(double m2b2, double rm0sqr,
                                  double m0sqr, double b1sqr,
                                  double smin, double smax) nogil:
    cdef int maxIterations = 10000
    cdef double root = 0.
    root = compute_find(m2b2, rm0sqr, m0sqr, b1sqr,
                smin, smax, -1.0, 1.0, maxIterations, root)
    gmin = compute_g(root, m2b2, rm0sqr, m0sqr, b1sqr)
    return root

cdef inline double compute_find(double m2b2, double rm0sqr,
                                double m0sqr, double b1sqr,
                                double t0, double t1, double f0, double f1,
                                int maxIterations, double root) nogil:
    cdef double fm, product
    if (t0 < t1):
        # Test the endpoints to see whether F(t) is zero.
        if f0 == 0.:
            root = t0
            return root
        if f1 == 0.:
            root = t1
            return root
        if f0*f1 > 0.:
            # It is not known whether the interval bounds a root.
            return root
        for i in range(2, maxIterations+1):
            root = (0.5) * (t0 + t1)
            if (root == t0 or root == t1):
                # The numbers t0 and t1 are consecutive floating-point
                # numbers.
                break
            fm = compute_g(root, m2b2, rm0sqr, m0sqr, b1sqr)
            product = fm * f0
            if (product < 0.):
                t1 = root
                f1 = fm
            elif (product > 0.):
                t0 = root
                f0 = fm
            else:
                break
        return root
    else:
        return root
