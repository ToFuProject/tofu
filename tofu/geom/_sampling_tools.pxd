# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#

from libc.stdint cimport int64_t
cimport numpy as cnp
# ==============================================================================
# =  LINEAR MESHING
# ==============================================================================
cdef int64_t discretize_line1d_core(double* LMinMax, double dstep,
                                 double[2] DL, bint Lim,
                                 int mode, double margin,
                                 double** ldiscret_arr,
                                 double[1] resolution,
                                 int64_t** lindex_arr, int64_t[1] N) nogil

cdef void first_discretize_line1d_core(double* LMinMax,
                                       double dstep,
                                       double[1] resolution,
                                       int64_t[1] num_cells,
                                       int64_t[1] Nind,
                                       int[1] nL0,
                                       double[2] DL,
                                       bint Lim,
                                       int mode,
                                       double margin) nogil

cdef void second_discretize_line1d_core(double* LMinMax,
                                        double* ldiscret,
                                        int64_t* lindex,
                                        int nL0,
                                        double resolution,
                                        int64_t Nind) nogil

cdef void simple_discretize_line1d(double[2] LMinMax, double dstep,
                                   int mode, double margin,
                                   double** ldiscret_arr,
                                   double[1] resolution,
                                   int64_t[1] N) nogil

cdef void cythonize_subdomain_dl(DL, double[2] dl_array) # uses gil


# ==============================================================================
# =  Vessel's poloidal cut discretization
# ==============================================================================
cdef void discretize_vpoly_core(double[:, ::1] VPoly, double dstep,
                                int mode, double margin, double DIn,
                                double[:, ::1] VIn,
                                double** XCross, double** YCross,
                                double** reso, int64_t** ind,
                                int64_t** numcells, double** Rref,
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
cpdef int get_nb_imode(str imode)

cpdef int get_nb_dmode(str dmode)

cdef int los_get_sample_single(double los_kmin, double los_kmax,
                               double resol, int imethod, int imode,
                               double[1] eff_res, double** coeffs) nogil

# -- Calc signal utility function ---------------------------------------------
cdef call_get_sample_single_ani(double los_kmin, double los_kmax,
                                double resol,
                                int n_dmode, int n_imode,
                                double[1] eff_res,
                                int64_t[1] nb_rows,
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
                                                  int64_t[1] nb_rows,
                                                  double[:, ::1] ray_orig,
                                                  double[:, ::1] ray_vdir)

cdef int los_get_sample_core_const_res(int nlos,
                                       double* los_lim_min,
                                       double* los_lim_max,
                                       int n_dmode, int n_imode,
                                       double val_resol,
                                       double** coeff_ptr,
                                       double* dLr,
                                       int64_t* los_ind,
                                       int num_threads) nogil

cdef void los_get_sample_core_var_res(int nlos,
                                     double* los_lim_min,
                                     double* los_lim_max,
                                     int n_dmode, int n_imode,
                                     double* resol,
                                     double** coeff_ptr,
                                     double* dLr,
                                     int64_t* los_ind,
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
                             int64_t* los_ind,
                             int num_threads) nogil

# ==============================================================================
# == Vmesh sampling
# ==============================================================================

# -- Vmesh utility functions --------------------------------------------------
cdef int  vmesh_disc_phi(int sz_r, int sz_z,
                         int64_t* ncells_rphi,
                         double phistep,
                         int ncells_rphi0,
                         double* disc_r,
                         double* disc_r0,
                         double* step_rphi,
                         double[::1] reso_phi_mv,
                         int64_t* tot_nc_plane,
                         int ind_loc_r0,
                         int ncells_r0,
                         int ncells_z,
                         int* max_sz_phi,
                         double min_phi,
                         double max_phi,
                         int64_t* sz_phi,
                         int64_t[:,::1] indi_mv,
                         double margin,
                         int num_threads) nogil

cdef int vmesh_get_index_arrays(int64_t[:, :, ::1] lnp,
                                int64_t[:, ::1] is_in_vignette,
                                int sz_r,
                                int sz_z,
                                int64_t* sz_phi) nogil

cdef void vmesh_assemble_arrays(int64_t[::1] first_ind_mv,
                                int64_t[:, ::1] indi_mv,
                                int64_t[:, ::1] is_in_vignette,
                                bint is_cart,
                                int sz_r,
                                int sz_z,
                                int64_t* lindex_z,
                                int64_t* ncells_rphi,
                                int64_t* tot_nc_plane,
                                double reso_r_z,
                                double* step_rphi,
                                double* disc_r,
                                double* disc_z,
                                int64_t[:,:,::1] lnp,
                                int64_t* phin,
                                double[::1] dv_mv,
                                double[::1] r_on_phi_mv,
                                double[:, ::1] pts_mv,
                                int64_t[::1] ind_mv,
                                int num_threads) nogil

cdef void vmesh_ind_init_tabs(int* ncells_rphi,
                              double* disc_r,
                              int sz_r, int sz_z,
                              double twopi_over_dphi,
                              double[::1] dRPhirRef,
                              int64_t* tot_nc_plane,
                              double** phi_mv,
                              int num_threads) nogil

cdef void vmesh_ind_cart_loop(int np,
                              int sz_r,
                              int64_t[::1] ind,
                              int64_t* tot_nc_plane,
                              int* ncells_rphi,
                              double* phi_tab,
                              double* disc_r,
                              double* disc_z,
                              double[:,::1] pts,
                              double[::1] res3d,
                              double reso_r_z,
                              double[::1] dRPhirRef,
                              int[::1] Ru,
                              double[::1] dRPhir,
                              int num_threads) nogil

cdef void vmesh_ind_polr_loop(int np,
                              int sz_r,
                              int64_t[::1] ind,
                              int64_t* tot_nc_plane,
                              int* ncells_rphi,
                              double* phi_tab,
                              double* disc_r,
                              double* disc_z,
                              double[:,::1] pts,
                              double[::1] res3d,
                              double reso_r_z,
                              double[::1] dRPhirRef,
                              int[::1] Ru,
                              double[::1] dRPhir,
                              int num_threads) nogil


# ==============================================================================
# == Solid Angles
# ==============================================================================
cdef int  sa_disc_phi(int sz_r, int sz_z,
                      int64_t* ncells_rphi,
                      double phistep,
                      double* disc_r,
                      double* disc_r0,
                      double* step_rphi,
                      int ind_loc_r0,
                      int ncells_r0,
                      int ncells_z,
                      int* max_sz_phi,
                      double min_phi,
                      double max_phi,
                      int64_t* sz_phi,
                      int64_t[:, ::1] indi_mv,
                      double margin,
                      int num_threads) nogil


cdef int sa_get_index_arrays(int64_t[:, ::1] lnp,
                             int64_t[:, ::1] is_in_vignette,
                             int sz_r,
                             int sz_z) nogil

cdef void sa_assemble_arrays(int block,
                             int use_approx,
                             double[:, ::1] part_coords,
                             double[::1] part_rad,
                             int64_t[:, ::1] is_in_vignette,
                             double[:, ::1] sa_map,
                             double[:, ::1] ves_poly,
                             double[:, ::1] ves_norm,
                             double[::1] ves_lims,
                             # int64_t[::1] lstruct_nlim,
                             int64_t[::1] lstruct_nlim,
                             double[::1] lstruct_polyx,
                             double[::1] lstruct_polyy,
                             double[::1] lstruct_lims,
                             double[::1] lstruct_normx,
                             double[::1] lstruct_normy,
                             # int64_t[::1] lnvert,
                             int64_t[::1] lnvert,
                             int nstruct_tot,
                             int nstruct_lim,
                             double rmin,
                             double eps_uz, double eps_a,
                             double eps_vz, double eps_b,
                             double eps_plane,
                             bint forbid,
                             int64_t[::1] first_ind_mv,
                             int64_t[:, ::1] indi_mv,
                             int sz_p,
                             int sz_r, int sz_z,
                             int64_t* ncells_rphi,
                             double reso_r_z,
                             double* disc_r,
                             double* step_rphi,
                             double* disc_z,
                             int64_t[:, ::1] ind_rz2pol,
                             int64_t* sz_phi,
                             double[::1] reso_rdrdz,
                             double[:, ::1] pts_mv,
                             int64_t[::1] ind_mv,
                             int num_threads)


# ##################################################################################
# ##################################################################################
#               Solid angle of a polygon
# ##################################################################################


cdef double comp_sa_tri(
    double A_x,
    double A_y,
    double A_z,
    double B_x,
    double B_y,
    double B_z,
    double C_x,
    double C_y,
    double C_z,
    double pt_x,
    double pt_y,
    double pt_z,
) nogil