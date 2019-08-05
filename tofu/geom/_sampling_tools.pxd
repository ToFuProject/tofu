# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
# ==============================================================================
# =  LINEAR MESHING
# ==============================================================================
cdef long discretize_line1d_core(double* LMinMax, double dstep,
                                         double[2] DL, bint Lim,
                                         int mode, double margin,
                                         double** ldiscret_arr,
                                         double[1] resolution,
                                         long** lindex_arr, long[1] N) nogil

cdef void first_discretize_line1d_core(double* LMinMax,
                                              double dstep,
                                              double[1] resolution,
                                              long[1] num_cells,
                                              long[1] Nind,
                                              int[1] nL0,
                                              double[2] DL,
                                              bint Lim,
                                              int mode,
                                              double margin) nogil

cdef void second_discretize_line1d_core(double* LMinMax,
                                               double* ldiscret,
                                               long* lindex,
                                               int nL0,
                                               double resolution,
                                               long Nind) nogil

cdef void simple_discretize_line1d(double[2] LMinMax, double dstep,
                                          int mode, double margin,
                                          double** ldiscret_arr,
                                          double[1] resolution,
                                          long[1] N) nogil

cdef void cythonize_subdomain_dl(DL, double[2] dl_array) # uses gil
# ==============================================================================
# =  Vessel's poloidal cut discretization
# ==============================================================================

cdef void discretize_vpoly_core(double[:, ::1] VPoly, double dstep,
                                       int mode, double margin, double DIn,
                                       double[:, ::1] VIn,
                                       double** XCross, double** YCross,
                                       double** reso, long** ind,
                                       long** numcells, double** Rref,
                                       double** XPolybis, double** YPolybis,
                                       int[1] tot_sz_vb, int[1] tot_sz_ot,
                                       int NP) nogil

# ------------------------------------------------------------------------------
# - Simplified version of previous algo
# ------------------------------------------------------------------------------
cdef void simple_discretize_vpoly_core(double[:, ::1] VPoly,
                                              int num_pts,
                                              double dstep,
                                              double** XCross,
                                              double** YCross,
                                              int[1] new_nb_pts,
                                              int mode,
                                              double margin) nogil

# ==============================================================================
# == LOS sampling
# ==============================================================================

# -- Quadrature Rules : Middle Rule --------------------------------------------
cdef void middle_rule_rel(int num_los, int num_raf,
                                 double* los_lims_x,
                                 double* los_lims_y,
                                 double* los_resolution,
                                 double* los_coeffs,
                                 long* los_ind,
                                 int num_threads) nogil

cdef void middle_rule_abs_1(int num_los, double resol,
                                   double* los_lims_x,
                                   double* los_lims_y,
                                   double* los_resolution,
                                   long* ind_cum,
                                   int num_threads) nogil

cdef void middle_rule_abs_2(int num_los,
                                   double* los_lims_x,
                                   long* ind_cum,
                                   double* los_resolution,
                                   double* los_coeffs,
                                   int num_threads) nogil

cdef void middle_rule_abs_var(int num_los, double* resolutions,
                                     double* los_lims_x,
                                     double* los_lims_y,
                                     double* los_resolution,
                                     double** los_coeffs,
                                     long* los_ind,
                                     int num_threads) nogil

cdef void middle_rule_rel_var(int num_los, double* resolutions,
                                     double* los_lims_x,
                                     double* los_lims_y,
                                     double* los_resolution,
                                     double** los_coeffs,
                                     long* los_ind,
                                     int num_threads) nogil

# -- Quadrature Rules : Left Rule ----------------------------------------------
cdef void left_rule_rel(int num_los, int num_raf,
                               double* los_lims_x,
                               double* los_lims_y,
                               double* los_resolution,
                               double* los_coeffs,
                               long* los_ind, int num_threads) nogil

cdef void simps_left_rule_abs(int num_los, double resol,
                                     double* los_lims_x,
                                     double* los_lims_y,
                                     double* los_resolution,
                                     double** los_coeffs,
                                     long* los_ind,
                                     int num_threads) nogil

cdef void romb_left_rule_abs(int num_los, double resol,
                                    double* los_lims_x,
                                    double* los_lims_y,
                                    double* los_resolution,
                                    double** los_coeffs,
                                    long* los_ind, int num_threads) nogil

cdef void simps_left_rule_rel_var(int num_los, double* resolutions,
                                         double* los_lims_x,
                                         double* los_lims_y,
                                         double* los_resolution,
                                         double** los_coeffs,
                                         long* los_ind,
                                         int num_threads) nogil

cdef void simps_left_rule_abs_var(int num_los, double* resolutions,
                                         double* los_lims_x,
                                         double* los_lims_y,
                                         double* los_resolution,
                                         double** los_coeffs,
                                         long* los_ind,
                                         int num_threads) nogil

cdef void romb_left_rule_rel_var(int num_los, double* resolutions,
                                        double* los_lims_x,
                                        double* los_lims_y,
                                        double* los_resolution,
                                        double** los_coeffs,
                                        long* los_ind,
                                        int num_threads) nogil

cdef void romb_left_rule_abs_var(int num_los, double* resolutions,
                                        double* los_lims_x,
                                        double* los_lims_y,
                                        double* los_resolution,
                                        double** los_coeffs,
                                        long* los_ind,
                                        int num_threads) nogil

# -- LOS sampling for a single ray ---------------------------------------------
cdef int get_nb_imode(str imode)

cdef int get_nb_dmode(str dmode)
