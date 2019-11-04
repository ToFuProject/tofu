# distutils: language=c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
################################################################################
# Utility functions for the earclipping techinque for triangulation of a
# polygon. This is really useful when vignetting, as the computation to know
# if a ray intersected a polygon, is easier if we check if the polygon is
# discretized in triangles and then we check if the ray intersected each
# triangle.
################################################################################
cdef  void compute_diff3d(double* orig,
                          int nvert,
                          double* diff) nogil

cdef  void are_points_reflex(int nvert,
                             double* diff,
                             bint* are_reflex) nogil

cdef  bint is_pt_in_tri(double[3] v0, double[3] v1,
                        double ax, double ay, double az,
                        double px, double py, double pz) nogil

cdef  void earclipping_poly(double* vignett,
                            long* ltri,
                            double* diff,
                            bint* lref,
                            int nvert) nogil

cdef int triangulate_polys(double** vignett_poly,
                            long* lnvert,
                            int nvign,
                            long** ltri,
                            int num_threads) nogil except -1

cdef void vignetting_core(double[:, ::1] ray_orig,
                          double[:, ::1] ray_vdir,
                          double** vignett,
                          long* lnvert,
                          double* lbounds,
                          long** ltri,
                          int nvign,
                          int nlos,
                          bint* goes_through,
                          int num_threads) nogil
