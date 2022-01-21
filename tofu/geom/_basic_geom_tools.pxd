# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
#
################################################################################
# Utility functions for basic geometry :
#   - vector calculus (cross product, dot product, norm, ...)
#   - cythonization of matplotlib path functions (is point in a path?)
#   - cythonization of some numpy functions (hypotenus, tile, sum)
################################################################################
cimport cython
from cpython.array cimport array, clone

# ==============================================================================
# ==  Geometry global variables
# ==============================================================================
# Values defined in the *.pyx file
cdef double _VSMALL
cdef double _SMALL
cdef double _TWOPI

# ==============================================================================
# == Redifinition of functions
# ==============================================================================
cdef bint is_point_in_path(const int nvert,
                           const double* vertx,
                           const double* verty,
                           const double testx,
                           const double testy) nogil

cdef int is_point_in_path_vec(const int nvert,
                              const double* vertx,
                              const double* verty,
                              const int npts,
                              const double* testx,
                              const double* testy,
                              int* is_in_path) nogil

cdef void compute_inv_and_sign(const double[3] ray_vdir,
                               int[3] sign,
                               double[3] inv_direction) nogil

cdef array compute_hypot(const double[::1] xpts, const double[::1] ypts,
                         int npts=*)

cdef double comp_min_hypot(const double[::1] xpts, const double[::1] ypts,
                           int npts=*) nogil

cdef double comp_min(double[::1] vec, int npts) nogil

cdef void tile_3_to_2d(double v0, double v1, double v2, int npts,
                       double[:,::1] res) nogil
# ==============================================================================
# =  Polygon helpers
# ==============================================================================
cdef int find_ind_lowerright_corner(const double[::1] xpts,
                                    const double[::1] ypts,
                                    int npts) nogil

# ==============================================================================
# == Vector Calculus Helpers
# ==============================================================================
cdef void compute_cross_prod(const double[3] vec_a,
                             const double[3] vec_b,
                             double[3] res) nogil

cdef void compute_dot_cross_vec(const double[:, ::1] lvec_a,
                                const double[:, ::1] lvec_b,
                                double[:, ::1] cross_p,
                                double[::1] dot_p,
                                const int npts,
                                const int num_threads,
                                ) nogil

cdef void find_centroids_ltri(const double[:, :, ::1] poly_coords,
                              const long** ltri,
                              const long* lnvert,
                              const int npoly,
                              const int num_threads,
                              double[:, ::1] centroid) nogil

cdef void find_centroids_GB_GC_ltri(const double** poly_coords,
                                    const long** ltri,
                                    const long* lnvert,
                                    const int npoly,
                                    const int num_threads,
                                    double[:, ::1] centroid,
                                    double[:, ::1] vec_GB,
                                    double[:, ::1] vec_GC,
                                    ) nogil

cdef void compute_vec_ass_tri(const double pt0, const double pt1,
                              const double pt2, int npts,
                              const double[:, ::1] ptG,
                              const double[:, ::1] poly_norm,
                              const double[:, ::1] cross_bc,
                              const double[:, ::1] vecb,
                              const double[:, ::1] vecc,
                              double* side_of_poly,
                              double* num,
                              double* dot_Gb,
                              double* dot_Gc,
                              double* normG2) nogil

cdef void compute_dist_pt_arr(const double pt0, const double pt1,
                              const double pt2, int npts,
                              const double* vec,
                              double* dist) nogil

cdef void compute_dist_pt_vec(const double pt0, const double pt1,
                              const double pt2, const int npts,
                              const double[:, ::1] vec,
                              double* dist) nogil

cdef void compute_dist_vec_vec(const int npts1, const int npts2,
                               const double[:, ::1] vec1,
                               const double[:, ::1] vec2,
                               double[:, ::1] dist) nogil

cdef double compute_norm(const double[3] vec) nogil

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

cdef void compute_diff_div(const double[:, ::1] vec1,
                           const double[:, ::1] vec2,
                           const double* div,
                           const int npts,
                           double[:, ::1] res) nogil

# ==============================================================================
# == Matrix sum (np.sum)
# ==============================================================================
cdef long sum_naive_int(long* orig, int n_cols) nogil
