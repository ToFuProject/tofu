# distutils: language=c++
# cython: language_level=3
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
cdef void compute_diff2d(
    double* orig,
    int nvert,
    double* diff,
) nogil

# DEPRECATED ?
cdef void compute_diff3d(double* orig,
                         int nvert,
                         double* diff) nogil

cdef bint is_reflex_2d(
    const double[1] u0,
    const double[1] u1,
    const double[1] v0,
    const double[1] v1,
) nogil

# DEPRECATED
cdef bint is_reflex_3d(
    const double[3] u,
    const double[3] v,
    double[3] vect_cc
) nogil

cdef void are_points_reflex_2d(
    int nvert,
    double* diff,
    bint* are_reflex,
) nogil

# DEPRECATED ?
cdef  void are_points_reflex_3d(int nvert,
                                double* diff,
                                bint* are_reflex,
                                double[3] vect_cc) nogil

cdef bint is_pt_in_tri_2d(
    double Ax,
    double Ay,
    double Bx,
    double By,
    double Cx,
    double Cy,
    double px,
    double py,
) nogil

# DEPRECATED
cdef bint is_pt_in_tri_3d(
    double[3] v0, double[3] v1,
    double Ax, double Ay, double Az,
    double px, double py, double pz,
) nogil

# cdef int get_one_ear(
    # double* polygon,
    # bint* lref,
    # _cl.ChainedList* working_index,
    # int nv,
    # int nvert,
# ) nogil

cdef void earclipping_poly_2d(
    double* vignett,
    long* ltri,
    double* diff,
    bint* lref,
    int nvert,
) nogil

cdef void triangulate_poly(double* vignett_poly,
                          long nvert,
                          long** ltri) nogil

cdef int triangulate_polys(double** vignett_poly,
                            long* lnvert,
                            int nvign,
                            long** ltri,
                            int num_threads) nogil except -1


# ===============================================================
#               Vignetting
# ===============================================================


cdef bint inter_ray_poly(
    const double[3] ray_orig,
    const double[3] ray_vdir,
    double* vignett,
    int nvert,
    long* ltri,
) nogil

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

cdef int vignetting_vmesh_vpoly(int npts, int sz_r,
                                bint is_cart,
                                double[:, ::1] vpoly,
                                double[:, ::1] pts,
                                double[::1] vol_resol,
                                double[::1] r_on_phi,
                                double* disc_r,
                                long[::1] lind,
                                double** res_x,
                                double** res_y,
                                double** res_z,
                                double** res_vres,
                                double** res_rphi,
                                long** res_lind,
                                long* sz_rphi,
                                int num_threads) nogil

cdef int are_in_vignette(int sz_r, int sz_z,
                         double[:, ::1] vpoly,
                         int npts_vpoly,
                         double* disc_r,
                         double* disc_z,
                         long[:, ::1] is_in_vignette) nogil
