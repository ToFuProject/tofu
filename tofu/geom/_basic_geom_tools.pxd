# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
################################################################################
# Utility functions for basic geometry (vector calculus, path, ...)
################################################################################
cimport cython
from cpython.array cimport array, clone

# ==============================================================================
# =  Geometry global variables
# ==============================================================================
# Values defined in the *.pyx file
cdef double _VSMALL
cdef double _SMALL

cdef bint is_point_in_path(const int nvert,
                           const double* vertx,
                           const double* verty,
                           const double testx,
                           const double testy) nogil

cdef  int is_point_in_path_vec(const int nvert,
                                     const double* vertx,
                                     const double* verty,
                                     const int npts,
                                     const double* testx,
                                     const double* testy,
                                     bint* is_in_path) nogil

cdef  void compute_inv_and_sign(const double[3] ray_vdir,
                                      int[3] sign,
                                      double[3] inv_direction) nogil

cdef  array compute_hypot(double[::1] xpts, double[::1] ypts,
                                int npts=*)

cdef  double comp_min_hypot(double[::1] xpts, double[::1] ypts,
                                  int npts=*) nogil

cdef  void compute_cross_prod(const double[3] vec_a,
                                    const double[3] vec_b,
                                    double[3] res) nogil

cdef  double compute_dot_prod(const double[3] vec_a,
                                    const double[3] vec_b) nogil

cdef  double compute_g(double s, double m2b2, double rm0sqr,
                             double m0sqr, double b1sqr) nogil

cdef  double compute_bisect(double m2b2, double rm0sqr,
                                  double m0sqr, double b1sqr,
                                  double smin, double smax) nogil

cdef  double compute_find(double m2b2, double rm0sqr,
                                double m0sqr, double b1sqr,
                                double t0, double t1, double f0, double f1,
                                int maxIterations, double root) nogil
