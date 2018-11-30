# cimport
cimport cython
cimport numpy as cnp
from cpython cimport bool
from libc.math cimport sqrt as Csqrt, ceil as Cceil, abs as Cabs
from libc.math cimport floor as Cfloor, round as Cround, log2 as Clog2
from libc.math cimport cos as Ccos, acos as Cacos, sin as Csin, asin as Casin
from libc.math cimport atan2 as Catan2, pi as Cpi
from cpython.array cimport array, clone

import line_profiler

import numpy as np
#from matplotlib.path import Path

from tofu.geom._poly_utils import get_bbox_poly_extruded, get_bbox_poly_limited



__all__ = ['LOS_Calc_PInOut_VesStruct']



########################################################
########################################################
#       PIn POut
########################################################

# TODO : @LM > recall
def LOS_Calc_PInOut_VesStruct(cnp.ndarray Ds, cnp.ndarray dus,
                              cnp.ndarray[double, ndim=2,mode='c'] VPoly,
                              cnp.ndarray[double, ndim=2,mode='c'] VIn,
                              Lim=None, nLim=None,
                              LSPoly=None, LSLim=None, lSnLim=None, LSVIn=None,
                              RMin=None, Forbid=True,
                              double EpsUz=1.e-6, double EpsVz=1.e-9, double EpsA=1.e-9,
                              double EpsB=1.e-9, double EpsPlane=1.e-9,
                              str VType='Tor', bool Test=True):
    """ Compute the entry and exit point of all provided LOS for the provided
    vessel polygon (toroidal or linear), also return the normal vector at
    impact point and the index of the impact segment

    For each LOS,

    Parameters
    ----------



    Return
    ------
    PIn :       np.ndarray
        Point of entry (if any) of the LOS into the vessel, returned in (X,Y,Z)
        cartesian coordinates as:
            1 LOS => (3,) array or None if there is no entry point
            NL LOS => (3,NL), with NaNs when there is no entry point
    POut :      np.ndarray
        Point of exit of the LOS from the vessel, returned in (X,Y,Z) cartesian
        coordinates as:
            1 LOS => (3,) array or None if there is no entry point
            NL LOS => (3,NL), with NaNs when there is no entry point
    VOut :      np.ndarray

    IOut :      np.ndarray

    """
    if Test:
        assert type(Ds) is cnp.ndarray and type(dus) is cnp.ndarray and \
            Ds.ndim in [1,2] and Ds.shape[0]==dus.shape[0] and \
            Ds.shape[0]==3 and Ds.shape[Ds.ndim-1] == dus.shape[Ds.ndim-1], (
                "Args Ds and dus must be of the same shape (3,) or (3,NL)!")
        assert VPoly.shape[0]==2 and VIn.shape[0]==2 and \
            VIn.shape[1]==VPoly.shape[1]-1, (
                "Args VPoly and VIn must be of the same shape (2,NS)!")
        C1 = all([pp is None for pp in [LSPoly,LSLim,LSVIn]])
        C2 = all([hasattr(pp,'__iter__') and len(pp)==len(LSPoly) for pp
                  in [LSPoly,LSLim,LSVIn]])
        assert C1 or C2, "Args LSPoly,LSLim,LSVIn must be None or lists of same len()!"
        assert RMin is None or type(RMin) in [float,int,np.float64,np.int64], (
            "Arg RMin must be None or a float!")
        assert type(Forbid) is bool, "Arg Forbid must be a bool!"
        assert all([type(ee) in [int,float,np.int64,np.float64] and ee<1.e-4
                    for ee in [EpsUz,EpsVz,EpsA,EpsB,EpsPlane]]), \
                        "Args [EpsUz,EpsVz,EpsA,EpsB] must be floats < 1.e-4!"
        assert type(VType) is str and VType.lower() in ['tor','lin'], (
            "Arg VType must be a str in ['Tor','Lin']!")

    cdef int ii, jj
    cdef bool v
    cdef bool found_new_kout
    cdef double kpin_jj
    cdef double kpout_jj
    cdef double L0, L1
    cdef Py_ssize_t ind_tmp
    cdef Py_ssize_t len_lim
    cdef Py_ssize_t num_los = Ds.shape[1]
    cdef cnp.uint8_t bool1, bool2
    cdef cnp.uint8_t nonan_jj, mask2_jj
    cdef long iin_jj
    cdef long [:,:] iout_view
    cdef double [:] kpin_view, kpout_view
    cdef double [3] struct_vperpin_view
    cdef double [:,:] vperp_out_view
    cdef double[3] last_pout_view
    cdef double[6] bounds
    cdef double[1] kpin_loc = clone(array('d'), 1, True)
    cdef int[1] indin_loc = clone(array('i'), 1, True)
    cdef cnp.ndarray[long,ndim=2] IOut=np.zeros((3, num_los), dtype=int)

    if nLim==0:
        Lim = None
    elif nLim==1:
        Lim = [Lim[0,0],Lim[0,1]]
    if lSnLim is not None:
        for ii in range(0,len(lSnLim)):
            if lSnLim[ii]==0:
                LSLim[ii] = None
            elif lSnLim[ii]==1:
                LSLim[ii] = [LSLim[ii][0,0],LSLim[ii][0,1]]

    v = Ds.ndim==2
    if not v:
        Ds, dus = Ds.reshape((3,1)), dus.reshape((3,1))

    if VType.lower()=='tor':
        # RMin is necessary to avoid looking on the other side of the tokamak
        if RMin is None:
            RMin = 0.95*min(np.min(VPoly[0,:]),
                            np.min(np.hypot(Ds[0,:],Ds[1,:])))

        # Main function to compute intersections with Vessel
        kPIn, kPOut, \
          VperpOut, \
          IOut[2,:] = Calc_LOS_PInOut_Tor(Ds, dus, VPoly, VIn, Lim=Lim,
                                               Forbid=Forbid, RMin=RMin,
                                               EpsUz=EpsUz, EpsVz=EpsVz,
                                               EpsA=EpsA, EpsB=EpsB,
                                               EpsPlane=EpsPlane)

        kpout_view = kPOut

        # If there are Struct, call the same function
        # Structural optimzation : do everything in one big for loop and only
        # keep the relevant points (to save memory)
        if LSPoly is not None:

            kpout_view = kPOut
            vperp_out_view = VperpOut
            iout_view = IOut

            for ii in range(0,len(LSPoly)):

                if LSLim[ii] is None or not all([hasattr(ll,'__iter__') for ll in LSLim[ii]]):
                    lslim = [LSLim[ii]]
                else:
                    lslim = LSLim[ii]
                len_lim = len(lslim)

                for jj in range(len_lim):
                    # # Warning: for "full" (aka "In") Structures we get kpin, vperpIn, iIn
                    # # and not kpout, vperpOut, iOut
                    # kpin_view, vperpIn,\
                    #   iIn  = Calc_LOS_PInOut_Tor_Lim(Ds, dus, LSPoly[ii],
                    #                                       LSVIn[ii],
                    #                                       Lim=lslim[jj],
                    #                                       Forbid=Forbid, RMin=RMin,
                    #                                       EpsUz=EpsUz, EpsVz=EpsVz,
                    #                                       EpsA=EpsA, EpsB=EpsB,
                    #                                       EpsPlane=EpsPlane)

                    # # kpin_view = kpIn
                    # # struct_pin_view = pIn
                    # struct_vperpin_view = vperpIn
                    # struct_iin_view = iIn


                    # We compute the structure's bounding box:
                    if lslim[jj] is not None:
                        L0 = Catan2(Csin(lslim[jj][0]),Ccos(lslim[jj][0]))
                        L1 = Catan2(Csin(lslim[jj][1]),Ccos(lslim[jj][1]))
                        bounds = get_bbox_poly_limited(np.asarray(LSPoly[ii]), [L0, L1])
                    else:
                        bounds = get_bbox_poly_extruded(np.asarray(LSPoly[ii]))

                    for ind_tmp in range(num_los):
                        # We get the last kpout:
                        kpout_jj = kpout_view[ind_tmp]
                        kpin_loc[0] = kpout_jj
                        indin_loc[0] = iout_view[2,ind_tmp]
                        last_pout_view[0] = kpout_jj * dus[0, ind_tmp] + Ds[0, ind_tmp]
                        last_pout_view[1] = kpout_jj * dus[1, ind_tmp] + Ds[1, ind_tmp]
                        last_pout_view[2] = kpout_jj * dus[2, ind_tmp] + Ds[2, ind_tmp]
                        # We compute new values
                        found_new_kout = compute_kout_los_on_filled(Ds[:,ind_tmp], dus[:,ind_tmp],
                                                              LSPoly[ii],
                                                              LSVIn[ii], LSVIn[ii].shape[1],
                                                              bounds,
                                                              last_pout_view,
                                                              RMin,
                                                              Lim==None, L0, L1,
                                                              kpin_loc, indin_loc, struct_vperpin_view,
                                                              Forbid=Forbid,
                                                              EpsUz=EpsUz, EpsVz=EpsVz,
                                                              EpsA=EpsA, EpsB=EpsB,
                                                              EpsPlane=EpsPlane)
                        if found_new_kout :
                            kpout_view[ind_tmp] = kpin_loc[0]
                            vperp_out_view[0,ind_tmp] = struct_vperpin_view[0]
                            vperp_out_view[1,ind_tmp] = struct_vperpin_view[1]
                            vperp_out_view[2,ind_tmp] = struct_vperpin_view[2]
                            iout_view[2,ind_tmp] = indin_loc[0]
                            iout_view[0,ind_tmp] = 1+ii
                            iout_view[1,ind_tmp] = jj

    if not v:
        kPIn, kPOut, \
          VperpOut, IOut = kPIn[0], kPOut[0], VperpOut.flatten(), IOut.flatten()
    return kPIn, kPOut, VperpOut, IOut

@cython.profile(True)
@cython.linetrace(True)
@cython.binding(True)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef Calc_LOS_PInOut_Tor_Lim(double [:,::1] Ds, double [:,::1] us,
                             double [:,::1] VPoly, double [:,::1] vIn,
                             Lim=None,
                             bool Forbid=True, RMin=None, double EpsUz=1.e-6,
                             double EpsVz=1.e-9, double EpsA=1.e-9,
                             double EpsB=1.e-9, double EpsPlane=1.e-9):

    cdef Py_ssize_t ii, jj, Nl=Ds.shape[1], Ns=vIn.shape[1]
    cdef double Rmin, upscaDp, upar2, Dpar2, Crit2, kout, kin
    cdef int indin=0, Done=0
    cdef double L, S1X=0., S1Y=0., S2X=0., S2Y=0., sca, sca0, sca1, sca2
    cdef double q, C, delta, sqd, k, sol0, sol1, phi=0., L0=0., L1=0.
    cdef double v0, v1, A, B, ephiIn0, ephiIn1
    cdef double SOut1, SOut0
    cdef int Forbidbis, Forbid0
    cdef cnp.ndarray[double,ndim=1] kIn=np.nan*np.ones((Nl,))
    cdef cnp.ndarray[double,ndim=2] VPerpIn=np.nan*np.ones((3,Nl))
    cdef cnp.ndarray[long,ndim=1] indIn = np.zeros((Nl,), dtype=long)

    cdef bool inter_bbox
    cdef double[:] bounds

    if Lim is not None:
        L0 = Catan2(Csin(Lim[0]),Ccos(Lim[0]))
        L1 = Catan2(Csin(Lim[1]),Ccos(Lim[1]))
        bounds = get_bbox_poly_limited(np.asarray(VPoly), [L0, L1])
    else:
        bounds = get_bbox_poly_extruded(np.asarray(VPoly))

    ################
    # Prepare input
    if RMin is None:
        Rmin = 0.95*min(np.min(VPoly[0,:]), np.min(np.hypot(Ds[0,:],Ds[1,:])))
    else:
        Rmin = RMin

    ################
    # Compute
    if Forbid:
        Forbid0, Forbidbis = 1, 1
    else:
        Forbid0, Forbidbis = 0, 0

    for ii in range(0,Nl):

        if Lim is not None:
            inter_bbox = ray_intersects_abba_bbox(bounds,
                                                  Ds[:,ii], us[:,ii])
            if not inter_bbox:
                continue

        upscaDp = us[0,ii]*Ds[0,ii] + us[1,ii]*Ds[1,ii]
        upar2 = us[0,ii]**2 + us[1,ii]**2
        Dpar2 = Ds[0,ii]**2 + Ds[1,ii]**2
        # Prepare in case Forbid is True
        if Forbid0 and not Dpar2>0:
            Forbidbis = 0
        if Forbidbis:
            # Compute coordinates of the 2 points where the tangents touch
            # the inner circle
            L = Csqrt(Dpar2-Rmin**2)
            S1X = (Rmin**2*Ds[0,ii]+Rmin*Ds[1,ii]*L)/Dpar2
            S1Y = (Rmin**2*Ds[1,ii]-Rmin*Ds[0,ii]*L)/Dpar2
            S2X = (Rmin**2*Ds[0,ii]-Rmin*Ds[1,ii]*L)/Dpar2
            S2Y = (Rmin**2*Ds[1,ii]+Rmin*Ds[0,ii]*L)/Dpar2

        # Compute all solutions
        # Set tolerance value for us[2,ii]
        # EpsUz is the tolerated DZ across 20m (max Tokamak size)
        Crit2 = EpsUz**2*upar2/400.
        kout, kin, Done = 1.e12, 1e12, 0
        # Case with horizontal semi-line
        if us[2,ii]**2<Crit2:
            for jj in range(0,Ns):
                # Solutions exist only in the case with non-horizontal
                # segment (i.e.: cone, not plane)
                # TODO : @LM : is this faster than checking abs(diff)>eps ?
                if (VPoly[1,jj+1] - VPoly[1,jj])**2 > EpsVz**2:
                    # TODO : @LM this probably can done matrix wise (qmatrix)
                    q = (Ds[2,ii]-VPoly[1,jj]) / (VPoly[1,jj+1]-VPoly[1,jj])
                    # The intersection must stand on the segment
                    if q>=0 and q<1:
                        C = q**2*(VPoly[0,jj+1]-VPoly[0,jj])**2 + \
                            2.*q*VPoly[0,jj]*(VPoly[0,jj+1]-VPoly[0,jj]) + \
                            VPoly[0,jj]**2
                        delta = upscaDp**2 - upar2*(Dpar2-C)
                        if delta>0.:
                            sqd = Csqrt(delta)
                            # The intersection must be on the semi-line
                            # (i.e.: k>=0)
                            # First solution
                            if -upscaDp - sqd >=0:
                                # TODO : @LM - est-ce que c'est possible de le mat ?
                                # ou le sortir d'ici
                                k = (-upscaDp - sqd)/upar2
                                sol0, sol1 = Ds[0,ii] + k*us[0,ii], \
                                             Ds[1,ii] + k*us[1,ii]
                                if Forbidbis:
                                    sca0 = (sol0-S1X)*Ds[0,ii] + \
                                           (sol1-S1Y)*Ds[1,ii]
                                    sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                                    sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                                if not Forbidbis or (Forbidbis and not
                                                     (sca0<0 and sca1<0 and
                                                      sca2<0)):
                                    # Get the normalized perpendicular vector
                                    # at intersection
                                    phi = Catan2(sol1,sol0)
                                    # Check sol inside the Lim
                                    if Lim is None or (Lim is not None and
                                                       ((L0<L1 and L0<=phi and
                                                         phi<=L1)
                                                        or (L0>L1 and
                                                            (phi>=L0 or
                                                             phi<=L1)))):
                                        # Get the scalar product to determine
                                        # entry or exit point
                                        sca = Ccos(phi)*vIn[0,jj]*us[0,ii] + \
                                              Csin(phi)*vIn[0,jj]*us[1,ii] + \
                                              vIn[1,jj]*us[2,ii]
                                        if sca<=0 and k<kout:
                                            kout = k
                                            Done = 1
                                            #print(1, k)
                                        elif sca>=0 and k<min(kin,kout):
                                            kin = k
                                            indin = jj
                                            #print(2, k)

                            # Second solution
                            if -upscaDp + sqd >=0:
                                k = (-upscaDp + sqd)/upar2
                                sol0, sol1 = Ds[0,ii] + k*us[0,ii], Ds[1,ii] \
                                             + k*us[1,ii]
                                if Forbidbis:
                                    sca0 = (sol0-S1X)*Ds[0,ii] + \
                                           (sol1-S1Y)*Ds[1,ii]
                                    sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                                    sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                                if not Forbidbis or (Forbidbis and not
                                                     (sca0<0 and sca1<0 and
                                                      sca2<0)):
                                    # Get the normalized perpendicular vector
                                    # at intersection
                                    phi = Catan2(sol1,sol0)
                                    if Lim is None or (Lim is not None and
                                                       ((L0<L1 and L0<=phi and
                                                         phi<=L1) or
                                                        (L0>L1 and
                                                         (phi>=L0 or phi<=L1))
                                                       )):
                                        # Get the scalar product to determine
                                        # entry or exit point
                                        sca = Ccos(phi)*vIn[0,jj]*us[0,ii] + \
                                              Csin(phi)*vIn[0,jj]*us[1,ii] + \
                                              vIn[1,jj]*us[2,ii]
                                        if sca<=0 and k<kout:
                                            kout = k
                                            Done = 1
                                            #print(3, k)
                                        elif sca>=0 and k<min(kin,kout):
                                            kin = k
                                            indin = jj
                                            #print(4, k)

        # More general non-horizontal semi-line case
        else:
            for jj in range(Ns):
                v0, v1 = VPoly[0,jj+1]-VPoly[0,jj], VPoly[1,jj+1]-VPoly[1,jj]
                A = v0**2 - upar2*(v1/us[2,ii])**2
                B = VPoly[0,jj]*v0 + v1*(Ds[2,ii]-VPoly[1,jj])*upar2/us[2,ii]**2 - upscaDp*v1/us[2,ii]
                C = -upar2*(Ds[2,ii]-VPoly[1,jj])**2/us[2,ii]**2 + 2.*upscaDp*(Ds[2,ii]-VPoly[1,jj])/us[2,ii] - Dpar2 + VPoly[0,jj]**2

                if A**2<EpsA**2 and B**2>EpsB**2:
                    q = -C/(2.*B)
                    if q>=0. and q<1.:
                        k = (q*v1 - (Ds[2,ii]-VPoly[1,jj]))/us[2,ii]
                        if k>=0:
                            sol0, sol1 = Ds[0,ii] + k*us[0,ii], Ds[1,ii] + k*us[1,ii]
                            if Forbidbis:
                                sca0 = (sol0-S1X)*Ds[0,ii] + (sol1-S1Y)*Ds[1,ii]
                                sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                                sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                                #print 1, k, kout, sca0, sca1, sca2
                                if sca0<0 and sca1<0 and sca2<0:
                                    continue
                            # Get the normalized perpendicular vector at intersection
                            phi = Catan2(sol1,sol0)
                            if Lim is None or (Lim is not None and ((L0<L1 and L0<=phi and phi<=L1) or (L0>L1 and (phi>=L0 or phi<=L1)))):
                                # Get the scalar product to determine entry or exit point
                                sca = Ccos(phi)*vIn[0,jj]*us[0,ii] + Csin(phi)*vIn[0,jj]*us[1,ii] + vIn[1,jj]*us[2,ii]
                                if sca<=0 and k<kout:
                                    kout = k
                                    Done = 1
                                    #print(5, k)
                                elif sca>=0 and k<min(kin,kout):
                                    kin = k
                                    indin = jj
                                    #print(6, k)

                elif A**2>=EpsA**2 and B**2>A*C:
                    sqd = Csqrt(B**2-A*C)
                    # First solution
                    q = (-B + sqd)/A
                    if q>=0. and q<1.:
                        k = (q*v1 - (Ds[2,ii]-VPoly[1,jj]))/us[2,ii]
                        if k>=0.:
                            sol0, sol1 = Ds[0,ii] + k*us[0,ii], Ds[1,ii] + k*us[1,ii]
                            if Forbidbis:
                                sca0 = (sol0-S1X)*Ds[0,ii] + (sol1-S1Y)*Ds[1,ii]
                                sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                                sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                                #print 2, k, kout, sca0, sca1, sca2
                            if not Forbidbis or (Forbidbis and not (sca0<0 and sca1<0 and sca2<0)):
                                # Get the normalized perpendicular vector at intersection
                                phi = Catan2(sol1,sol0)
                                if Lim is None or (Lim is not None and ((L0<L1 and L0<=phi and phi<=L1) or (L0>L1 and (phi>=L0 or phi<=L1)))):
                                    # Get the scalar product to determine entry or exit point
                                    sca = Ccos(phi)*vIn[0,jj]*us[0,ii] + Csin(phi)*vIn[0,jj]*us[1,ii] + vIn[1,jj]*us[2,ii]
                                    if sca<=0 and k<kout:
                                        kout = k
                                        Done = 1
                                        #print(7, k, q, A, B, C, sqd)
                                    elif sca>=0 and k<min(kin,kout):
                                        kin = k
                                        indin = jj
                                        #print(8, k, jj)

                    # Second solution
                    q = (-B - sqd)/A
                    if q>=0. and q<1.:
                        k = (q*v1 - (Ds[2,ii]-VPoly[1,jj]))/us[2,ii]

                        if k>=0.:
                            sol0, sol1 = Ds[0,ii] + k*us[0,ii], Ds[1,ii] + k*us[1,ii]
                            if Forbidbis:
                                sca0 = (sol0-S1X)*Ds[0,ii] + (sol1-S1Y)*Ds[1,ii]
                                sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                                sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                                #print 3, k, kout, sca0, sca1, sca2
                            if not Forbidbis or (Forbidbis and not (sca0<0 and sca1<0 and sca2<0)):
                                # Get the normalized perpendicular vector at intersection
                                phi = Catan2(sol1,sol0)
                                if Lim is None or (Lim is not None and ((L0<L1 and L0<=phi and phi<=L1) or (L0>L1 and (phi>=L0 or phi<=L1)))):
                                    # Get the scalar product to determine entry or exit point
                                    sca = Ccos(phi)*vIn[0,jj]*us[0,ii] + Csin(phi)*vIn[0,jj]*us[1,ii] + vIn[1,jj]*us[2,ii]
                                    if sca<=0 and k<kout:
                                        kout = k
                                        Done = 1
                                        #print(9, k, jj)
                                    elif sca>=0 and k<min(kin,kout):
                                        kin = k
                                        indin = jj
                                        #print(10, k, q, A, B, C, sqd, v0, v1, jj)

        if Lim is not None:
            ephiIn0, ephiIn1 = -Csin(L0), Ccos(L0)
            if Cabs(us[0,ii]*ephiIn0+us[1,ii]*ephiIn1)>EpsPlane:
                k = -(Ds[0,ii]*ephiIn0+Ds[1,ii]*ephiIn1)/(us[0,ii]*ephiIn0+us[1,ii]*ephiIn1)
                if k>=0:
                    # Check if in VPoly
                    sol0, sol1 = (Ds[0,ii]+k*us[0,ii])*Ccos(L0) + (Ds[1,ii]+k*us[1,ii])*Csin(L0), Ds[2,ii]+k*us[2,ii]
                    #if path_poly_t.contains_point([sol0,sol1], transform=None, radius=0.0):
                    #if ray_tracing(VPoly, sol0, sol1):
                    inter_bbox = pnpoly(Ns, VPoly[0,:], VPoly[1,:], sol0, sol1)
                    if inter_bbox:
                        # Check PIn (POut not possible for limited torus)
                        sca = us[0,ii]*ephiIn0 + us[1,ii]*ephiIn1
                        if sca<=0 and k<kout:
                            kout = k
                            Done = 1
                        elif sca>=0 and k<min(kin,kout):
                            kin = k
                            indin = -1

            ephiIn0, ephiIn1 = Csin(L1), -Ccos(L1)
            if Cabs(us[0,ii]*ephiIn0+us[1,ii]*ephiIn1)>EpsPlane:
                k = -(Ds[0,ii]*ephiIn0+Ds[1,ii]*ephiIn1)/(us[0,ii]*ephiIn0+us[1,ii]*ephiIn1)
                if k>=0:
                    sol0, sol1 = (Ds[0,ii]+k*us[0,ii])*Ccos(L1) + (Ds[1,ii]+k*us[1,ii])*Csin(L1), Ds[2,ii]+k*us[2,ii]
                    # Check if in VPoly
                    #if path_poly_t.contains_point([sol0,sol1], transform=None, radius=0.0):
                    # if ray_tracing(VPoly, sol0, sol1):
                    inter_bbox = pnpoly(Ns, VPoly[0,:], VPoly[1,:], sol0, sol1)
                    if inter_bbox:
                        # Check PIn (POut not possible for limited torus)
                        sca = us[0,ii]*ephiIn0 + us[1,ii]*ephiIn1
                        if sca<=0 and k<kout:
                            kout = k
                            Done = 1
                        elif sca>=0 and k<min(kin,kout):
                            kin = k
                            indin = -2

        # print("  For Line ", ii, "  test = ", inter_bbox, " and kout = ", Done, kin, kout)
        if Done==1:
            if kin<kout:
                kIn[ii] = kin
                if indin==-1:
                    VPerpIn[0,ii] = Csin(L0)
                    VPerpIn[1,ii] = -Ccos(L0)
                    VPerpIn[2,ii] = 0.
                elif indin==-2:
                    VPerpIn[0,ii] = -Csin(L1)
                    VPerpIn[1,ii] = Ccos(L1)
                    VPerpIn[2,ii] = 0.
                else:
                    SIn0 = Ds[0,ii] + kin*us[0,ii]
                    SIn1 = Ds[1,ii] + kin*us[1,ii]
                    phi = Catan2(SIn1,SIn0)
                    VPerpIn[0,ii] = -Ccos(phi)*vIn[0,indin]
                    VPerpIn[1,ii] = -Csin(phi)*vIn[0,indin]
                    VPerpIn[2,ii] = -vIn[1,indin]
                indIn[ii] = indin

    return np.asarray(kIn), np.asarray(VPerpIn), np.asarray(indIn)



@cython.profile(True)
@cython.linetrace(True)
@cython.binding(True)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline bool compute_kout_los_on_filled(double [:] Ds, double [:] us,
                                double [:,::1] VPoly, double [:,::1] vIn,
                                int vin_shape,
                                double [:] bounds,
                                double [:] last_pout_coords, double Rmin,
                                bool lim_is_none, double L0, double L1,
                                double[1] kpin_loc, int[1] indin_loc, double[3] vperpin,
                                bool Forbid=True, double EpsUz=1.e-6,
                                double EpsVz=1.e-9, double EpsA=1.e-9,
                                double EpsB=1.e-9, double EpsPlane=1.e-9):
    cdef Py_ssize_t jj#, Ns=vIn.shape[1]
    cdef double upscaDp, upar2, Dpar2, Crit2, kout, kin
    cdef int indin=0, Done=0
    cdef double L, S1X=0., S1Y=0., S2X=0., S2Y=0., sca, sca0, sca1, sca2
    cdef double q, C, delta, sqd, k, sol0, sol1, phi=0.
    cdef double v0, v1, A, B, ephiIn0, ephiIn1
    cdef double SOut1, SOut0
    cdef int Forbidbis, Forbid0
    cdef double res_kin = kpin_loc[0]
    # cdef double[3] VPerpIn=np.nan*np.ones((3,))
    # cdef long res_indin = -1
    # cdef array VPerpIn = clone(array('d'), 3, False)
    cdef cnp.ndarray[double, ndim=1] VPerpIn = np.nan*np.ones((3,)) 
    cdef bool inter_bbox

    # VPerpIn[0] = np.nan
    # VPerpIn[1] = np.nan
    # VPerpIn[2] = np.nan

    # We check if the ray intersects it 
    inter_bbox = ray_intersects_abba_bbox(bounds, Ds, us)
    if not inter_bbox:
        return False

    # We check if the bounding box is not actually "behind"
    # the last object intersected
    inter_bbox = ray_intersects_abba_bbox(bounds, last_pout_coords, us)
    # in this case we dont want the ray to intersect the bbox, else the
    # last k found is in front of the box.
    if inter_bbox:
        return False


    ################
    # Compute
    if Forbid:
        Forbid0, Forbidbis = 1, 1
    else:
        Forbid0, Forbidbis = 0, 0    

    upscaDp = us[0]*Ds[0] + us[1]*Ds[1]
    upar2   = us[0]*us[0] + us[1]*us[1]
    Dpar2   = Ds[0]*Ds[0] + Ds[1]*Ds[1]
    # Prepare in case Forbid is True
    if Forbid0 and not Dpar2>0:
        Forbidbis = 0
    if Forbidbis:
        # Compute coordinates of the 2 points where the tangents touch
        # the inner circle
        L = Csqrt(Dpar2-Rmin**2)
        S1X = (Rmin**2*Ds[0]+Rmin*Ds[1]*L)/Dpar2
        S1Y = (Rmin**2*Ds[1]-Rmin*Ds[0]*L)/Dpar2
        S2X = (Rmin**2*Ds[0]-Rmin*Ds[1]*L)/Dpar2
        S2Y = (Rmin**2*Ds[1]+Rmin*Ds[0]*L)/Dpar2

    # Compute all solutions
    # Set tolerance value for us[2,ii]
    # EpsUz is the tolerated DZ across 20m (max Tokamak size)
    Crit2 = EpsUz**2*upar2/400.
    kout, kin, Done = 1.e12, 1e12, 0
    # Case with horizontal semi-line
    if us[2]*us[2]<Crit2:
        for jj in range(0,vin_shape):
            # Solutions exist only in the case with non-horizontal
            # segment (i.e.: cone, not plane)
            # TODO : @LM : is this faster than checking abs(diff)>eps ?
            if (VPoly[1,jj+1] - VPoly[1,jj])**2 > EpsVz**2:
                # TODO : @LM this probably can done matrix wise (qmatrix)
                q = (Ds[2]-VPoly[1,jj]) / (VPoly[1,jj+1]-VPoly[1,jj])
                # The intersection must stand on the segment
                if q>=0 and q<1:
                    C = q**2*(VPoly[0,jj+1]-VPoly[0,jj])**2 + \
                        2.*q*VPoly[0,jj]*(VPoly[0,jj+1]-VPoly[0,jj]) + \
                        VPoly[0,jj]**2
                    delta = upscaDp**2 - upar2*(Dpar2-C)
                    if delta>0.:
                        sqd = Csqrt(delta)
                        # The intersection must be on the semi-line
                        # (i.e.: k>=0)
                        # First solution
                        if -upscaDp - sqd >=0:
                            # TODO : @LM - est-ce que c'est possible de le mat ?
                            # ou le sortir d'ici
                            k = (-upscaDp - sqd)/upar2
                            sol0, sol1 = Ds[0] + k*us[0], \
                                         Ds[1] + k*us[1]
                            if Forbidbis:
                                sca0 = (sol0-S1X)*Ds[0] + \
                                       (sol1-S1Y)*Ds[1]
                                sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                                sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                            if not Forbidbis or (Forbidbis and not
                                                 (sca0<0 and sca1<0 and
                                                  sca2<0)):
                                # Get the normalized perpendicular vector
                                # at intersection
                                phi = Catan2(sol1,sol0)
                                # Check sol inside the Lim
                                if lim_is_none or (not lim_is_none and
                                                   ((L0<L1 and L0<=phi and
                                                     phi<=L1)
                                                    or (L0>L1 and
                                                        (phi>=L0 or
                                                         phi<=L1)))):
                                    # Get the scalar product to determine
                                    # entry or exit point
                                    sca = Ccos(phi)*vIn[0,jj]*us[0] + \
                                          Csin(phi)*vIn[0,jj]*us[1] + \
                                          vIn[1,jj]*us[2]
                                    if sca<=0 and k<kout:
                                        kout = k
                                        Done = 1
                                        #print(1, k)
                                    elif sca>=0 and k<min(kin,kout):
                                        kin = k
                                        indin = jj
                                        #print(2, k)

                        # Second solution
                        if -upscaDp + sqd >=0:
                            k = (-upscaDp + sqd)/upar2
                            sol0, sol1 = Ds[0] + k*us[0], Ds[1] \
                                         + k*us[1]
                            if Forbidbis:
                                sca0 = (sol0-S1X)*Ds[0] + \
                                       (sol1-S1Y)*Ds[1]
                                sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                                sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                            if not Forbidbis or (Forbidbis and not
                                                 (sca0<0 and sca1<0 and
                                                  sca2<0)):
                                # Get the normalized perpendicular vector
                                # at intersection
                                phi = Catan2(sol1,sol0)
                                if lim_is_none or (not lim_is_none and
                                                   ((L0<L1 and L0<=phi and
                                                     phi<=L1) or
                                                    (L0>L1 and
                                                     (phi>=L0 or phi<=L1))
                                                   )):
                                    # Get the scalar product to determine
                                    # entry or exit point
                                    sca = Ccos(phi)*vIn[0,jj]*us[0] + \
                                          Csin(phi)*vIn[0,jj]*us[1] + \
                                          vIn[1,jj]*us[2]
                                    if sca<=0 and k<kout:
                                        kout = k
                                        Done = 1
                                        #print(3, k)
                                    elif sca>=0 and k<min(kin,kout):
                                        kin = k
                                        indin = jj
                                        #print(4, k)

    # More general non-horizontal semi-line case
    else:
        for jj in range(vin_shape):
            v0, v1 = VPoly[0,jj+1]-VPoly[0,jj], VPoly[1,jj+1]-VPoly[1,jj]
            A = v0**2 - upar2*(v1/us[2])**2
            B = VPoly[0,jj]*v0 + v1*(Ds[2]-VPoly[1,jj])*upar2/us[2]**2 - upscaDp*v1/us[2]
            C = -upar2*(Ds[2]-VPoly[1,jj])**2/us[2]**2 + 2.*upscaDp*(Ds[2]-VPoly[1,jj])/us[2] - Dpar2 + VPoly[0,jj]**2

            if A**2<EpsA**2 and B**2>EpsB**2:
                q = -C/(2.*B)
                if q>=0. and q<1.:
                    k = (q*v1 - (Ds[2]-VPoly[1,jj]))/us[2]
                    if k>=0:
                        sol0, sol1 = Ds[0] + k*us[0], Ds[1] + k*us[1]
                        if Forbidbis:
                            sca0 = (sol0-S1X)*Ds[0] + (sol1-S1Y)*Ds[1]
                            sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                            sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                            #print 1, k, kout, sca0, sca1, sca2
                            if sca0<0 and sca1<0 and sca2<0:
                                continue
                        # Get the normalized perpendicular vector at intersection
                        phi = Catan2(sol1,sol0)
                        if lim_is_none or (not lim_is_none and ((L0<L1 and L0<=phi and phi<=L1) or (L0>L1 and (phi>=L0 or phi<=L1)))):
                            # Get the scalar product to determine entry or exit point
                            sca = Ccos(phi)*vIn[0,jj]*us[0] + Csin(phi)*vIn[0,jj]*us[1] + vIn[1,jj]*us[2]
                            if sca<=0 and k<kout:
                                kout = k
                                Done = 1
                                #print(5, k)
                            elif sca>=0 and k<min(kin,kout):
                                kin = k
                                indin = jj
                                #print(6, k)

            elif A**2>=EpsA**2 and B**2>A*C:
                sqd = Csqrt(B**2-A*C)
                # First solution
                q = (-B + sqd)/A
                if q>=0. and q<1.:
                    k = (q*v1 - (Ds[2]-VPoly[1,jj]))/us[2]
                    if k>=0.:
                        sol0, sol1 = Ds[0] + k*us[0], Ds[1] + k*us[1]
                        if Forbidbis:
                            sca0 = (sol0-S1X)*Ds[0] + (sol1-S1Y)*Ds[1]
                            sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                            sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                            #print 2, k, kout, sca0, sca1, sca2
                        if not Forbidbis or (Forbidbis and not (sca0<0 and sca1<0 and sca2<0)):
                            # Get the normalized perpendicular vector at intersection
                            phi = Catan2(sol1,sol0)
                            if lim_is_none or (not lim_is_none and ((L0<L1 and L0<=phi and phi<=L1) or (L0>L1 and (phi>=L0 or phi<=L1)))):
                                # Get the scalar product to determine entry or exit point
                                sca = Ccos(phi)*vIn[0,jj]*us[0] + Csin(phi)*vIn[0,jj]*us[1] + vIn[1,jj]*us[2]
                                if sca<=0 and k<kout:
                                    kout = k
                                    Done = 1
                                    #print(7, k, q, A, B, C, sqd)
                                elif sca>=0 and k<min(kin,kout):
                                    kin = k
                                    indin = jj
                                    #print(8, k, jj)

                # Second solution
                q = (-B - sqd)/A
                if q>=0. and q<1.:
                    k = (q*v1 - (Ds[2]-VPoly[1,jj]))/us[2]

                    if k>=0.:
                        sol0, sol1 = Ds[0] + k*us[0], Ds[1] + k*us[1]
                        if Forbidbis:
                            sca0 = (sol0-S1X)*Ds[0] + (sol1-S1Y)*Ds[1]
                            sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                            sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                            #print 3, k, kout, sca0, sca1, sca2
                        if not Forbidbis or (Forbidbis and not (sca0<0 and sca1<0 and sca2<0)):
                            # Get the normalized perpendicular vector at intersection
                            phi = Catan2(sol1,sol0)
                            if lim_is_none or (not lim_is_none and ((L0<L1 and L0<=phi and phi<=L1) or (L0>L1 and (phi>=L0 or phi<=L1)))):
                                # Get the scalar product to determine entry or exit point
                                sca = Ccos(phi)*vIn[0,jj]*us[0] + Csin(phi)*vIn[0,jj]*us[1] + vIn[1,jj]*us[2]
                                if sca<=0 and k<kout:
                                    kout = k
                                    Done = 1
                                    #print(9, k, jj)
                                elif sca>=0 and k<min(kin,kout):
                                    kin = k
                                    indin = jj
                                    #print(10, k, q, A, B, C, sqd, v0, v1, jj)

    if not lim_is_none:
        ephiIn0, ephiIn1 = -Csin(L0), Ccos(L0)
        if Cabs(us[0]*ephiIn0+us[1]*ephiIn1)>EpsPlane:
            k = -(Ds[0]*ephiIn0+Ds[1]*ephiIn1)/(us[0]*ephiIn0+us[1]*ephiIn1)
            if k>=0:
                # Check if in VPoly
                sol0, sol1 = (Ds[0]+k*us[0])*Ccos(L0) + (Ds[1]+k*us[1])*Csin(L0), Ds[2]+k*us[2]
                #if path_poly_t.contains_point([sol0,sol1], transform=None, radius=0.0):
                #if ray_tracing(VPoly, sol0, sol1):
                inter_bbox = pnpoly(vin_shape, VPoly[0,:], VPoly[1,:], sol0, sol1)
                if inter_bbox:
                    # Check PIn (POut not possible for limited torus)
                    sca = us[0]*ephiIn0 + us[1]*ephiIn1
                    if sca<=0 and k<kout:
                        kout = k
                        Done = 1
                    elif sca>=0 and k<min(kin,kout):
                        kin = k
                        indin = -1

        ephiIn0, ephiIn1 = Csin(L1), -Ccos(L1)
        if Cabs(us[0]*ephiIn0+us[1]*ephiIn1)>EpsPlane:
            k = -(Ds[0]*ephiIn0+Ds[1]*ephiIn1)/(us[0]*ephiIn0+us[1]*ephiIn1)
            if k>=0:
                sol0, sol1 = (Ds[0]+k*us[0])*Ccos(L1) + (Ds[1]+k*us[1])*Csin(L1), Ds[2]+k*us[2]
                # Check if in VPoly
                #if path_poly_t.contains_point([sol0,sol1], transform=None, radius=0.0):
                # if ray_tracing(VPoly, sol0, sol1):
                inter_bbox = pnpoly(vin_shape, VPoly[0,:], VPoly[1,:], sol0, sol1)
                if inter_bbox:
                    # Check PIn (POut not possible for limited torus)
                    sca = us[0]*ephiIn0 + us[1]*ephiIn1
                    if sca<=0 and k<kout:
                        kout = k
                        Done = 1
                    elif sca>=0 and k<min(kin,kout):
                        kin = k
                        indin = -2

    # print("  For Line ", ii, "  test = ", inter_bbox, " and kout = ", Done, kin, kout)
    if Done==1:
        if kin<kout:
            kpin_loc[0] = kin
            if indin==-1:
                vperpin[0] = Csin(L0)
                vperpin[1] = -Ccos(L0)
                vperpin[2] = 0.
            elif indin==-2:
                vperpin[0] = -Csin(L1)
                vperpin[1] = Ccos(L1)
                vperpin[2] = 0.
            else:
                SIn0 = Ds[0] + kin*us[0]
                SIn1 = Ds[1] + kin*us[1]
                phi = Catan2(SIn1,SIn0)
                vperpin[0] = -Ccos(phi)*vIn[0,indin]
                vperpin[1] = -Csin(phi)*vIn[0,indin]
                vperpin[2] = -vIn[1,indin]
            indin_loc[0] = indin

                
    return res_kin == kpin_loc[0]

# et creer vecteurs
#    return np.asarray(kIn), np.asarray(kOut), np.asarray(vPerpOut), np.asarray(indOut)

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline bool ray_intersects_abba_bbox(double[:] bounds, double [:] ds, double [:] us):
    """
    bounds = [3d coords of lowerleftback point of bounding box,
              3d coords of upperrightfront point of bounding box]
    ds = [3d coords of origin of ray]
    us = [3d coords of direction of ray]
    returns True if ray intersects bounding box, else False
    """
    cdef int[3] sign
    cdef double[3] inv_direction
    cdef double tmin, tmax, tymin, tymax
    cdef double tzmin, tzmax
    cdef int t0 = 1000000
    cdef bool res
    cdef Py_ssize_t ii
    # computing sing and direction
    for  ii in range(3):
        if us[ii]*us[ii] < 1.e-9:
            inv_direction[ii] = t0
        else:
            inv_direction[ii] = 1./us[ii]
        if us[ii] < 0.:
            sign[ii] = 1
        else:
            sign[ii] = 0
    # computing intersection
    tmin = (bounds[sign[0]*3] - ds[0]) * inv_direction[0];
    tmax = (bounds[(1-sign[0])*3] - ds[0]) * inv_direction[0];
    tymin = (bounds[(sign[1])*3 + 1] - ds[1]) * inv_direction[1];
    tymax = (bounds[(1-sign[1])*3+1] - ds[1]) * inv_direction[1];
    if ( (tmin > tymax) or (tymin > tmax) ):
        return False
    if (tymin > tmin):
        tmin = tymin
    if (tymax < tmax):
        tmax = tymax
    tzmin = (bounds[(sign[2])*3+2] - ds[2]) * inv_direction[2]
    tzmax = (bounds[(1-sign[2])*3+2] - ds[2]) * inv_direction[2]
    if ( (tmin > tzmax) or (tzmin > tmax) ):
        return False
    if (tzmin > tmin):
        tmin = tzmin
    if (tzmax < tmax):
        tmax = tzmax
    res = (tmin < t0) and (tmax > -t0)
    return  res


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline bool ray_tracing(double[:,::1] poly, double x, double y):
    cdef Py_ssize_t n = poly.shape[1]
    cdef Py_ssize_t ii
    cdef bool inside = False
    cdef double p2x, p2y
    cdef double xints =0.
    cdef double p1x, p1y

    p1x, p1y = poly[:,0]
    for ii in range(n+1):
        p2x,p2y = poly[:,ii]
        if y > min(p1y,p2y) and y <= max(p1y,p2y) and x <= max(p1x,p2x):
            if p1y != p2y:
                xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
            if p1x == p2x or x <= xints:
                inside = not inside
        p1x,p1y = p2x,p2y

    return inside

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline bool pnpoly(int nvert, double[:] vertx, double[:] verty, double testx, double testy):
    cdef int i
    cdef bool c = False
    for i in range(nvert):
        if ( ((verty[i]>testy) != (verty[i+1]>testy)) and
            (testx < (vertx[i+1]-vertx[i]) * (testy-verty[i]) / (verty[i+1]-verty[i]) + vertx[i]) ):
            c = not c
    return c

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef Calc_LOS_PInOut_Tor(double [:,::1] Ds, double [:,::1] us,
                         double [:,::1] VPoly, double [:,::1] vIn, Lim=None,
                         bool Forbid=True, RMin=None, double EpsUz=1.e-6,
                         double EpsVz=1.e-9, double EpsA=1.e-9,
                         double EpsB=1.e-9, double EpsPlane=1.e-9):

    cdef int ii, jj, Nl=Ds.shape[1], Ns=vIn.shape[1]
    cdef double Rmin, upscaDp, upar2, Dpar2, Crit2, kout, kin
    cdef int indout=0, Done=0
    cdef double L, S1X=0., S1Y=0., S2X=0., S2Y=0., sca, sca0, sca1, sca2
    cdef double q, C, delta, sqd, k, sol0, sol1, phi=0., L0=0., L1=0.
    cdef double v0, v1, A, B, ephiIn0, ephiIn1
    cdef double SOut1, SOut0
    cdef int Forbidbis, Forbid0
    cdef cnp.ndarray[double,ndim=1] kIn_=np.nan*np.ones((Nl,))
    cdef cnp.ndarray[double,ndim=1] kOut_=np.nan*np.ones((Nl,))
    cdef cnp.ndarray[double,ndim=2] VPerp_Out=np.nan*np.ones((3,Nl))
    cdef cnp.ndarray[long,ndim=1] indOut= np.zeros((Nl,), dtype=long)

    cdef double[:] kIn=kIn_, kOut=kOut_
    cdef double[:,::1] VPerpOut=VPerp_Out

    if Lim is not None:
        L0 = Catan2(Csin(Lim[0]),Ccos(Lim[0]))
        L1 = Catan2(Csin(Lim[1]),Ccos(Lim[1]))

    #path_poly_t = Path(VPoly.T)
    ################
    # Prepare input
    if RMin is None:
        Rmin = 0.95*min(np.min(VPoly[0,:]), np.min(np.hypot(Ds[0,:],Ds[1,:])))
    else:
        Rmin = RMin

    ################
    # Compute
    if Forbid:
        Forbid0, Forbidbis = 1, 1
    else:
        Forbid0, Forbidbis = 0, 0
    for ii in range(0,Nl):
        upscaDp = us[0,ii]*Ds[0,ii] + us[1,ii]*Ds[1,ii]
        upar2 = us[0,ii]**2 + us[1,ii]**2
        Dpar2 = Ds[0,ii]**2 + Ds[1,ii]**2
        # Prepare in case Forbid is True
        if Forbid0 and not Dpar2>0:
            Forbidbis = 0
        if Forbidbis:
            # Compute coordinates of the 2 points where the tangents touch
            # the inner circle
            L = Csqrt(Dpar2-Rmin**2)
            S1X = (Rmin**2*Ds[0,ii]+Rmin*Ds[1,ii]*L)/Dpar2
            S1Y = (Rmin**2*Ds[1,ii]-Rmin*Ds[0,ii]*L)/Dpar2
            S2X = (Rmin**2*Ds[0,ii]-Rmin*Ds[1,ii]*L)/Dpar2
            S2Y = (Rmin**2*Ds[1,ii]+Rmin*Ds[0,ii]*L)/Dpar2

        # Compute all solutions
        # Set tolerance value for us[2,ii]
        # EpsUz is the tolerated DZ across 20m (max Tokamak size)
        Crit2 = EpsUz**2*upar2/400.
        kout, kin, Done = 1.e12, 1e12, 0
        # Case with horizontal semi-line
        if us[2,ii]**2<Crit2:
            for jj in range(0,Ns):
                # Solutions exist only in the case with non-horizontal
                # segment (i.e.: cone, not plane)
                # TODO : @LM : is this faster than checking abs(diff)>eps ?
                if (VPoly[1,jj+1] - VPoly[1,jj])**2 > EpsVz**2:
                    # TODO : @LM this probably can done matrix wise (qmatrix)
                    q = (Ds[2,ii]-VPoly[1,jj]) / (VPoly[1,jj+1]-VPoly[1,jj])
                    # The intersection must stand on the segment
                    # TODO : @LM why is q==1 rejected ?
                    if q>=0 and q<1:
                        C = q**2*(VPoly[0,jj+1]-VPoly[0,jj])**2 + \
                            2.*q*VPoly[0,jj]*(VPoly[0,jj+1]-VPoly[0,jj]) + \
                            VPoly[0,jj]**2
                        delta = upscaDp**2 - upar2*(Dpar2-C)
                        if delta>0.:
                            sqd = Csqrt(delta)
                            # The intersection must be on the semi-line
                            # (i.e.: k>=0)
                            # First solution
                            if -upscaDp - sqd >=0:
                                # TODO : @LM - est-ce que c'est possible de le mat ?
                                # ou le sortir d'ici
                                k = (-upscaDp - sqd)/upar2
                                sol0, sol1 = Ds[0,ii] + k*us[0,ii], \
                                             Ds[1,ii] + k*us[1,ii]
                                if Forbidbis:
                                    sca0 = (sol0-S1X)*Ds[0,ii] + \
                                           (sol1-S1Y)*Ds[1,ii]
                                    sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                                    sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                                if not Forbidbis or (Forbidbis and not
                                                     (sca0<0 and sca1<0 and
                                                      sca2<0)):
                                    # Get the normalized perpendicular vector
                                    # at intersection
                                    phi = Catan2(sol1,sol0)
                                    # Check sol inside the Lim
                                    if Lim is None or (Lim is not None and
                                                       ((L0<L1 and L0<=phi and
                                                         phi<=L1)
                                                        or (L0>L1 and
                                                            (phi>=L0 or
                                                             phi<=L1)))):
                                        # Get the scalar product to determine
                                        # entry or exit point
                                        sca = Ccos(phi)*vIn[0,jj]*us[0,ii] + \
                                              Csin(phi)*vIn[0,jj]*us[1,ii] + \
                                              vIn[1,jj]*us[2,ii]
                                        if sca<=0 and k<kout:
                                            kout = k
                                            indout = jj
                                            Done = 1
                                            #print(1, k)
                                        elif sca>=0 and k<min(kin,kout):
                                            kin = k
                                            #print(2, k)

                            # Second solution
                            if -upscaDp + sqd >=0:
                                k = (-upscaDp + sqd)/upar2
                                sol0, sol1 = Ds[0,ii] + k*us[0,ii], Ds[1,ii] \
                                             + k*us[1,ii]
                                if Forbidbis:
                                    sca0 = (sol0-S1X)*Ds[0,ii] + \
                                           (sol1-S1Y)*Ds[1,ii]
                                    sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                                    sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                                if not Forbidbis or (Forbidbis and not
                                                     (sca0<0 and sca1<0 and
                                                      sca2<0)):
                                    # Get the normalized perpendicular vector
                                    # at intersection
                                    phi = Catan2(sol1,sol0)
                                    if Lim is None or (Lim is not None and
                                                       ((L0<L1 and L0<=phi and
                                                         phi<=L1) or
                                                        (L0>L1 and
                                                         (phi>=L0 or phi<=L1))
                                                       )):
                                        # Get the scalar product to determine
                                        # entry or exit point
                                        sca = Ccos(phi)*vIn[0,jj]*us[0,ii] + \
                                              Csin(phi)*vIn[0,jj]*us[1,ii] + \
                                              vIn[1,jj]*us[2,ii]
                                        if sca<=0 and k<kout:
                                            kout = k
                                            indout = jj
                                            Done = 1
                                            #print(3, k)
                                        elif sca>=0 and k<min(kin,kout):
                                            kin = k
                                            #print(4, k)

        # More general non-horizontal semi-line case
        else:
            for jj in range(Ns):
                v0, v1 = VPoly[0,jj+1]-VPoly[0,jj], VPoly[1,jj+1]-VPoly[1,jj]
                A = v0**2 - upar2*(v1/us[2,ii])**2
                B = VPoly[0,jj]*v0 + v1*(Ds[2,ii]-VPoly[1,jj])*upar2/us[2,ii]**2 - upscaDp*v1/us[2,ii]
                C = -upar2*(Ds[2,ii]-VPoly[1,jj])**2/us[2,ii]**2 + 2.*upscaDp*(Ds[2,ii]-VPoly[1,jj])/us[2,ii] - Dpar2 + VPoly[0,jj]**2

                if A**2<EpsA**2 and B**2>EpsB**2:
                    q = -C/(2.*B)
                    if q>=0. and q<1.:
                        k = (q*v1 - (Ds[2,ii]-VPoly[1,jj]))/us[2,ii]
                        if k>=0:
                            sol0, sol1 = Ds[0,ii] + k*us[0,ii], Ds[1,ii] + k*us[1,ii]
                            if Forbidbis:
                                sca0 = (sol0-S1X)*Ds[0,ii] + (sol1-S1Y)*Ds[1,ii]
                                sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                                sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                                #print 1, k, kout, sca0, sca1, sca2
                                if sca0<0 and sca1<0 and sca2<0:
                                    continue
                            # Get the normalized perpendicular vector at intersection
                            phi = Catan2(sol1,sol0)
                            if Lim is None or (Lim is not None and ((L0<L1 and L0<=phi and phi<=L1) or (L0>L1 and (phi>=L0 or phi<=L1)))):
                                # Get the scalar product to determine entry or exit point
                                sca = Ccos(phi)*vIn[0,jj]*us[0,ii] + Csin(phi)*vIn[0,jj]*us[1,ii] + vIn[1,jj]*us[2,ii]
                                if sca<=0 and k<kout:
                                    kout = k
                                    indout = jj
                                    Done = 1
                                    #print(5, k)
                                elif sca>=0 and k<min(kin,kout):
                                    kin = k
                                    #print(6, k)

                elif A**2>=EpsA**2 and B**2>A*C:
                    sqd = Csqrt(B**2-A*C)
                    # First solution
                    q = (-B + sqd)/A
                    if q>=0. and q<1.:
                        k = (q*v1 - (Ds[2,ii]-VPoly[1,jj]))/us[2,ii]
                        if k>=0.:
                            sol0, sol1 = Ds[0,ii] + k*us[0,ii], Ds[1,ii] + k*us[1,ii]
                            if Forbidbis:
                                sca0 = (sol0-S1X)*Ds[0,ii] + (sol1-S1Y)*Ds[1,ii]
                                sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                                sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                                #print 2, k, kout, sca0, sca1, sca2
                            if not Forbidbis or (Forbidbis and not (sca0<0 and sca1<0 and sca2<0)):
                                # Get the normalized perpendicular vector at intersection
                                phi = Catan2(sol1,sol0)
                                if Lim is None or (Lim is not None and ((L0<L1 and L0<=phi and phi<=L1) or (L0>L1 and (phi>=L0 or phi<=L1)))):
                                    # Get the scalar product to determine entry or exit point
                                    sca = Ccos(phi)*vIn[0,jj]*us[0,ii] + Csin(phi)*vIn[0,jj]*us[1,ii] + vIn[1,jj]*us[2,ii]
                                    if sca<=0 and k<kout:
                                        kout = k
                                        indout = jj
                                        Done = 1
                                        #print(7, k, q, A, B, C, sqd)
                                    elif sca>=0 and k<min(kin,kout):
                                        kin = k
                                        #print(8, k, jj)

                    # Second solution
                    q = (-B - sqd)/A
                    if q>=0. and q<1.:
                        k = (q*v1 - (Ds[2,ii]-VPoly[1,jj]))/us[2,ii]

                        if k>=0.:
                            sol0, sol1 = Ds[0,ii] + k*us[0,ii], Ds[1,ii] + k*us[1,ii]
                            if Forbidbis:
                                sca0 = (sol0-S1X)*Ds[0,ii] + (sol1-S1Y)*Ds[1,ii]
                                sca1 = (sol0-S1X)*S1X + (sol1-S1Y)*S1Y
                                sca2 = (sol0-S2X)*S2X + (sol1-S2Y)*S2Y
                                #print 3, k, kout, sca0, sca1, sca2
                            if not Forbidbis or (Forbidbis and not (sca0<0 and sca1<0 and sca2<0)):
                                # Get the normalized perpendicular vector at intersection
                                phi = Catan2(sol1,sol0)
                                if Lim is None or (Lim is not None and ((L0<L1 and L0<=phi and phi<=L1) or (L0>L1 and (phi>=L0 or phi<=L1)))):
                                    # Get the scalar product to determine entry or exit point
                                    sca = Ccos(phi)*vIn[0,jj]*us[0,ii] + Csin(phi)*vIn[0,jj]*us[1,ii] + vIn[1,jj]*us[2,ii]
                                    if sca<=0 and k<kout:
                                        kout = k
                                        indout = jj
                                        Done = 1
                                        #print(9, k, jj)
                                    elif sca>=0 and k<min(kin,kout):
                                        kin = k
                                        #print(10, k, q, A, B, C, sqd, v0, v1, jj)

        if Lim is not None:
            ephiIn0, ephiIn1 = -Csin(L0), Ccos(L0)
            if Cabs(us[0,ii]*ephiIn0+us[1,ii]*ephiIn1)>EpsPlane:
                k = -(Ds[0,ii]*ephiIn0+Ds[1,ii]*ephiIn1)/(us[0,ii]*ephiIn0+us[1,ii]*ephiIn1)
                if k>=0:
                    # Check if in VPoly
                    sol0, sol1 = (Ds[0,ii]+k*us[0,ii])*Ccos(L0) + (Ds[1,ii]+k*us[1,ii])*Csin(L0), Ds[2,ii]+k*us[2,ii]
                    #if path_poly_t.contains_point([sol0,sol1], transform=None, radius=0.0):
                    # if ray_tracing(VPoly, sol0, sol1):
                    if pnpoly(Ns, VPoly[0,:], VPoly[1,:], sol0, sol1):
                        # Check PIn (POut not possible for limited torus)
                        sca = us[0,ii]*ephiIn0 + us[1,ii]*ephiIn1
                        if sca<=0 and k<kout:
                            kout = k
                            indout = -1
                            Done = 1
                        elif sca>=0 and k<min(kin,kout):
                            kin = k

            ephiIn0, ephiIn1 = Csin(L1), -Ccos(L1)
            if Cabs(us[0,ii]*ephiIn0+us[1,ii]*ephiIn1)>EpsPlane:
                k = -(Ds[0,ii]*ephiIn0+Ds[1,ii]*ephiIn1)/(us[0,ii]*ephiIn0+us[1,ii]*ephiIn1)
                if k>=0:
                    sol0, sol1 = (Ds[0,ii]+k*us[0,ii])*Ccos(L1) + (Ds[1,ii]+k*us[1,ii])*Csin(L1), Ds[2,ii]+k*us[2,ii]
                    # Check if in VPoly
                    #if path_poly_t.contains_point([sol0,sol1], transform=None, radius=0.0):
                    # if ray_tracing(VPoly, sol0, sol1):
                    if pnpoly(Ns, VPoly[0,:], VPoly[1,:], sol0, sol1):
                        # Check PIn (POut not possible for limited torus)
                        sca = us[0,ii]*ephiIn0 + us[1,ii]*ephiIn1
                        if sca<=0 and k<kout:
                            kout = k
                            indout = -2
                            Done = 1
                        elif sca>=0 and k<min(kin,kout):
                            kin = k

        if Done==1:
            kOut[ii] = kout
            if indout==-1:
                VPerpOut[0,ii] = -Csin(L0)
                VPerpOut[1,ii] = Ccos(L0)
                VPerpOut[2,ii] = 0.
            elif indout==-2:
                VPerpOut[0,ii] = Csin(L1)
                VPerpOut[1,ii] = -Ccos(L1)
                VPerpOut[2,ii] = 0.
            else:
                SOut0 = Ds[0,ii] + kout*us[0,ii]
                SOut1 = Ds[1,ii] + kout*us[1,ii]
                phi = Catan2(SOut1,SOut0)
                VPerpOut[0,ii] = Ccos(phi)*vIn[0,indout]
                VPerpOut[1,ii] = Csin(phi)*vIn[0,indout]
                VPerpOut[2,ii] = vIn[1,indout]
            indOut[ii] = indout
            if kin<kout:
                kIn[ii] = kin

    return np.asarray(kIn), np.asarray(kOut), \
      np.asarray(VPerpOut), \
      np.asarray(indOut)
    # return SIn, SOut, \
    #   VPerpIn,VPerpOut, \
    #   indIn, indOut
