# cimport
cimport cython
cimport numpy as cnp
from cpython cimport bool
from libc.math cimport sqrt as Csqrt, ceil as Cceil, abs as Cabs
from libc.math cimport floor as Cfloor, round as Cround, log2 as Clog2
from libc.math cimport cos as Ccos, acos as Cacos, sin as Csin, asin as Casin
from libc.math cimport atan2 as Catan2, pi as Cpi
# import line_profiler
# import
import sys
import numpy as np
import scipy.integrate as scpintg
from matplotlib.path import Path
from ray_box_test import check_inter_bbox_ray
from ray_box_test import Box, Ray

from tofu.geom._poly_utils import get_bbox_poly_extruded
if sys.version[0]=='3':
    from inspect import signature as insp
elif sys.version[0]=='2':
    from inspect import getargspec as insp



__all__ = ['LOS_Calc_PInOut_VesStruct']



########################################################
########################################################
#       PIn POut
########################################################

# TODO : @LM > recall
def LOS_Calc_PInOut_VesStruct(Ds, dus,
                              cnp.ndarray[double, ndim=2,mode='c'] VPoly,
                              cnp.ndarray[double, ndim=2,mode='c'] VIn,
                              Lim=None, LSPoly=None, LSLim=None, LSVIn=None,
                              RMin=None, Forbid=True,
                              EpsUz=1.e-6, EpsVz=1.e-9, EpsA=1.e-9,
                              EpsB=1.e-9, EpsPlane=1.e-9,
                              VType='Tor', Test=True):
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
        assert type(Ds) is np.ndarray and type(dus) is np.ndarray and \
            Ds.ndim in [1,2] and Ds.shape==dus.shape and \
            Ds.shape[0]==3, (
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

    v = Ds.ndim==2
    if not v:
        Ds, dus = Ds.reshape((3,1)), dus.reshape((3,1))
    NL = Ds.shape[1]
    IOut = np.zeros((3,Ds.shape[1]))
    if VType.lower()=='tor':
        # RMin is necessary to avoid looking on the other side of the tokamak
        if RMin is None:
            RMin = 0.95*min(np.min(VPoly[0,:]),
                            np.min(np.hypot(Ds[0,:],Ds[1,:])))

        # Main function to compute intersections with Vessel
        PIn, POut, \
            VperpIn, VperpOut, \
            IIn, IOut[2,:] = Calc_LOS_PInOut_Tor(Ds, dus, VPoly, VIn, Lim=Lim,
                                                 Forbid=Forbid, RMin=RMin,
                                                 EpsUz=EpsUz, EpsVz=EpsVz,
                                                 EpsA=EpsA, EpsB=EpsB,
                                                 EpsPlane=EpsPlane)

        # k = coordinate (in m) along the line from D
        kPOut = np.sqrt(np.sum((POut-Ds)**2,axis=0))
        kPIn = np.sqrt(np.sum((PIn-Ds)**2,axis=0))
        assert np.allclose(kPOut,np.sum((POut-Ds)*dus,axis=0),equal_nan=True)
        assert np.allclose(kPIn,np.sum((PIn-Ds)*dus,axis=0),equal_nan=True)

        # If there are Struct, call the same function
        # Structural optimzation : do everything in one big for loop and only
        # keep the relevant points (to save memory)
        if LSPoly is not None:
            Ind = np.zeros((2,NL))
            for ii in range(0,len(LSPoly)):
                if LSLim[ii] is None or not all([hasattr(ll,'__iter__') for ll in LSLim[ii]]):
                    lslim = [LSLim[ii]]
                else:
                    lslim = LSLim[ii]
                for jj in range(0,len(lslim)):
                    pIn, pOut,\
                        vperpIn, vperpOut,\
                        iIn, iOut = Calc_LOS_PInOut_Tor(Ds, dus, LSPoly[ii],
                                                        LSVIn[ii], Lim=lslim[jj],
                                                        Forbid=Forbid, RMin=RMin,
                                                        EpsUz=EpsUz, EpsVz=EpsVz,
                                                        EpsA=EpsA, EpsB=EpsB,
                                                        EpsPlane=EpsPlane)
                    kpin = np.sqrt(np.sum((Ds-pIn)**2,axis=0))
                    indNoNan = (~np.isnan(kpin)) & (~np.isnan(kPOut))
                    indout = np.zeros((NL,),dtype=bool)
                    indout[indNoNan] = kpin[indNoNan]<kPOut[indNoNan]
                    indout[(~np.isnan(kpin)) & np.isnan(kPOut)] = True
                    if np.any(indout):
                        kPOut[indout] = kpin[indout]
                        POut[:,indout] = pIn[:,indout]
                        VperpOut[:,indout] = vperpIn[:,indout]
                        IOut[2,indout] = iIn[indout]
                        IOut[0,indout] = 1+ii
                        IOut[1,indout] = jj

    if not v:
        PIn, POut, \
            kPIn, kPOut, \
            VperpIn, VperpOut, \
            IIn, IOut = PIn.flatten(), POut.flatten(), kPIn[0], kPOut[0], \
                        VperpIn.flatten(), VperpOut.flatten(), IIn[0], \
                        IOut.flatten()
    return PIn, POut, kPIn, kPOut, VperpIn, VperpOut, IIn, IOut


@cython.profile(True)
@cython.linetrace(True)
@cython.binding(True)
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
    cdef int indin=0, indout=0, Done=0
    cdef double L, S1X=0., S1Y=0., S2X=0., S2Y=0., sca, sca0, sca1, sca2
    cdef double q, C, delta, sqd, k, sol0, sol1, phi=0., L0=0., L1=0.
    cdef double v0, v1, A, B, ephiIn0, ephiIn1
    cdef int Forbidbis, Forbid0
    cdef cnp.ndarray[double,ndim=2] SIn_=np.nan*np.ones((3,Nl))
    cdef cnp.ndarray[double,ndim=2] SOut_=np.nan*np.ones((3,Nl))
    cdef cnp.ndarray[double,ndim=2] VPerp_In=np.nan*np.ones((3,Nl))
    cdef cnp.ndarray[double,ndim=2] VPerp_Out=np.nan*np.ones((3,Nl))
    cdef cnp.ndarray[double,ndim=1] indIn_=np.nan*np.ones((Nl,))
    cdef cnp.ndarray[double,ndim=1] indOut_=np.nan*np.ones((Nl,))

    cdef double[:,::1] SIn=SIn_, SOut=SOut_
    cdef double[:,::1] VPerpIn=VPerp_In, VPerpOut=VPerp_Out
    cdef double[::1] indIn=indIn_, indOut=indOut_
    if Lim is not None:
        L0 = Catan2(Csin(Lim[0]),Ccos(Lim[0]))
        L1 = Catan2(Csin(Lim[1]),Ccos(Lim[1]))

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

    # print("")
    # print("New box................................")
    bounds = get_bbox_poly_extruded(np.asarray(VPoly))

    #bbox = Box(bounds)
    #ax = bbox.plot()
    for ii in range(0,Nl):

        # Let us first check if the line intersects the bounding box of structure
        inter_bbox = check_inter_bbox_ray(bounds, Ds[:,ii], us[:,ii])
        if not inter_bbox:
            continue
        # ray = Ray(Ds[:,ii], us[:,ii], notVec=True)
        #ray.plot(ax=ax,block=ii==(Nl-1))
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
                                            indout = jj
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
                                    indout = jj
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
                                        indout = jj
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
                                        indout = jj
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
                    if Path(VPoly.T).contains_point([sol0,sol1], transform=None, radius=0.0):
                        # Check PIn (POut not possible for limited torus)
                        sca = us[0,ii]*ephiIn0 + us[1,ii]*ephiIn1
                        if sca<=0 and k<kout:
                            kout = k
                            indout = -1
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
                    if Path(VPoly.T).contains_point([sol0,sol1], transform=None, radius=0.0):
                        # Check PIn (POut not possible for limited torus)
                        sca = us[0,ii]*ephiIn0 + us[1,ii]*ephiIn1
                        if sca<=0 and k<kout:
                            kout = k
                            indout = -2
                            Done = 1
                        elif sca>=0 and k<min(kin,kout):
                            kin = k
                            indin = -2

        # print("  For Line ", ii, "  test = ", inter_bbox, " and kout = ", Done, kin, kout)
        if Done==1:
            SOut[0,ii] = Ds[0,ii] + kout*us[0,ii]
            SOut[1,ii] = Ds[1,ii] + kout*us[1,ii]
            SOut[2,ii] = Ds[2,ii] + kout*us[2,ii]
            phi = Catan2(SOut[1,ii],SOut[0,ii])
            if indout==-1:
                VPerpOut[0,ii] = -Csin(L0)
                VPerpOut[1,ii] = Ccos(L0)
                VPerpOut[2,ii] = 0.
            elif indout==-2:
                VPerpOut[0,ii] = Csin(L1)
                VPerpOut[1,ii] = -Ccos(L1)
                VPerpOut[2,ii] = 0.
            else:
                VPerpOut[0,ii] = Ccos(phi)*vIn[0,indout]
                VPerpOut[1,ii] = Csin(phi)*vIn[0,indout]
                VPerpOut[2,ii] = vIn[1,indout]
            indOut[ii] = indout
            if kin<kout:
                SIn[0,ii] = Ds[0,ii] + kin*us[0,ii]
                SIn[1,ii] = Ds[1,ii] + kin*us[1,ii]
                SIn[2,ii] = Ds[2,ii] + kin*us[2,ii]
                phi = Catan2(SIn[1,ii],SIn[0,ii])
                if indin==-1:
                    VPerpIn[0,ii] = Csin(L0)
                    VPerpIn[1,ii] = -Ccos(L0)
                    VPerpIn[2,ii] = 0.
                elif indin==-2:
                    VPerpIn[0,ii] = -Csin(L1)
                    VPerpIn[1,ii] = Ccos(L1)
                    VPerpIn[2,ii] = 0.
                else:
                    VPerpIn[0,ii] = -Ccos(phi)*vIn[0,indin]
                    VPerpIn[1,ii] = -Csin(phi)*vIn[0,indin]
                    VPerpIn[2,ii] = -vIn[1,indin]
                indIn[ii] = indin

    return np.asarray(SIn), np.asarray(SOut), np.asarray(VPerpIn), np.asarray(VPerpOut), np.asarray(indIn), np.asarray(indOut)
# et creer vecteurs
#    return np.asarray(kIn), np.asarray(kOut), np.asarray(vPerpOut), np.asarray(indOut)

