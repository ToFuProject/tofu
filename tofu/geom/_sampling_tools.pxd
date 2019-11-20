# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
cimport numpy as cnp
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

# -- LOS sampling for a single ray ---------------------------------------------
cdef int get_nb_imode(str imode)

cdef int get_nb_dmode(str dmode)

cdef int los_get_sample_single(double los_kmin, double los_kmax,
                               double resol, int imethod, int imode,
                               double[1] eff_res, double** coeffs) nogil

# -- Calc signal utility function ---------------------------------------------
cdef call_get_sample_single_ani(double los_kmin, double los_kmax,
                                double resol,
                                int n_dmode, int n_imode,
                                double[1] eff_res,
                                long[1] nb_rows,
                                double[:, ::1] ray_orig,
                                double[:, ::1] ray_vdir)

cdef cnp.ndarray[double,
                 ndim=2,
                 mode='c'] call_get_sample_single(double los_kmin,
                                                  double los_kmax,
                                                  double resol,
                                                  int n_dmode,
                                                  int n_imode,
                                                  double[1] eff_res,
                                                  long[1] nb_rows,
                                                  double[:, ::1] ray_orig,
                                                  double[:, ::1] ray_vdir)

cdef int los_get_sample_core_const_res(int nlos,
                                       double* los_lim_min,
                                       double* los_lim_max,
                                       int n_dmode, int n_imode,
                                       double val_resol,
                                       double** coeff_ptr,
                                       double* dLr,
                                       long* los_ind,
                                       int num_threads) nogil

cdef void los_get_sample_core_var_res(int nlos,
                                     double* los_lim_min,
                                     double* los_lim_max,
                                     int n_dmode, int n_imode,
                                     double* resol,
                                     double** coeff_ptr,
                                     double* dLr,
                                     long* los_ind,
                                     int num_threads) nogil

cdef void los_get_sample_pts(int nlos,
                             double* ptx,
                             double* pty,
                             double* ptz,
                             double* usx,
                             double* usy,
                             double* usz,
                             double[:,::1] ray_orig,
                             double[:,::1] ray_vdir,
                             double* coeff_ptr,
                             long* los_ind,
                             int num_threads) nogil
