# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
################################################################################
# Utility functions related to the distance between a LOS and a VPoly:
#    - Compute distance between LOS and a circle (unit and vectorial)
#    - Test if a LOS and a circle are close (unit and vectorial) to a epsilon
#    - Compute distance between LOS and VPoly (unit and vectorial)
#    - Test if a LOS and a VPoly are close (unit and vectorial) to a epsilon
#    - Compute which LOS is closer to a VPoly
#    - Compute which VPoly is closer to a LOS
################################################################################

# ==============================================================================
# == DISTANCE CIRCLE - LOS
# ==============================================================================
cdef void dist_los_circle_core(const double[3] direct,
                                      const double[3] origin,
                                      const double radius, const double circ_z,
                                      double norm_dir,
                                      double[2] result) nogil

cdef void comp_dist_los_circle_vec_core(int num_los, int num_cir,
                                               double* los_directions,
                                               double* los_origins,
                                               double* circle_radius,
                                               double* circle_z,
                                               double* norm_dir_tab,
                                               double[::1] res_k,
                                               double[::1] res_dist,
                                               int num_threads) nogil

# ==============================================================================
# == TEST CLOSENESS CIRCLE - LOS
# ==============================================================================
cdef bint is_close_los_circle_core(const double[3] direct,
                                          const double[3] origin,
                                          double radius, double circ_z,
                                          double norm_dir, double eps) nogil

cdef void is_close_los_circle_vec_core(int num_los, int num_cir,
                                       double eps,
                                       double* los_directions,
                                       double* los_origins,
                                       double* circle_radius,
                                       double* circle_z,
                                       double* norm_dir_tab,
                                       int[::1] res, int num_threads) nogil

# ==============================================================================
# == DISTANCE BETWEEN LOS AND EXT-POLY
# ==============================================================================
cdef void comp_dist_los_vpoly_vec_core(int num_poly, int nlos,
                                              double* ray_orig,
                                              double* ray_vdir,
                                              double[:,:,::1] ves_poly,
                                              double eps_uz,
                                              double eps_a,
                                              double eps_vz,
                                              double eps_b,
                                              double eps_plane,
                                              int ves_type,
                                              int algo_type,
                                              double* res_k,
                                              double* res_dist,
                                              double disc_step,
                                              int num_threads) nogil

cdef void simple_dist_los_vpoly_core(const double[3] ray_orig,
                                            const double[3] ray_vdir,
                                            const double* lpolyx,
                                            const double* lpolyy,
                                            const int nvert,
                                            const double upscaDp,
                                            const double upar2,
                                            const double dpar2,
                                            const double invuz,
                                            const double crit2,
                                            const double eps_uz,
                                            const double eps_vz,
                                            const double eps_a,
                                            const double eps_b,
                                            double* res_final) nogil

# ==============================================================================
# == ARE LOS AND EXT-POLY CLOSE
# ==============================================================================
cdef void is_close_los_vpoly_vec_core(int num_poly, int nlos,
                                             double* ray_orig,
                                             double* ray_vdir,
                                             double[:,:,::1] ves_poly,
                                             double eps_uz,
                                             double eps_a,
                                             double eps_vz,
                                             double eps_b,
                                             double eps_plane,
                                             double epsilon,
                                             int[::1] are_close,
                                             int num_threads) nogil

# ==============================================================================
# == WHICH LOS/VPOLY IS CLOSER
# ==============================================================================
cdef void which_los_closer_vpoly_vec_core(int num_poly, int nlos,
                                                 double* ray_orig,
                                                 double* ray_vdir,
                                                 double[:,:,::1] ves_poly,
                                                 double eps_uz,
                                                 double eps_a,
                                                 double eps_vz,
                                                 double eps_b,
                                                 double eps_plane,
                                                 int[::1] ind_close_tab,
                                                 int num_threads) nogil

cdef void which_vpoly_closer_los_vec_core(int num_poly, int nlos,
                                                 double* ray_orig,
                                                 double* ray_vdir,
                                                 double[:,:,::1] ves_poly,
                                                 double eps_uz,
                                                 double eps_a,
                                                 double eps_vz,
                                                 double eps_b,
                                                 double eps_plane,
                                                 int[::1] ind_close_tab,
                                                 int num_threads) nogil
