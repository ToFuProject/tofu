
# cimport
cimport cython
cimport numpy as cnp
from cpython cimport bool
from libc.math cimport sqrt as Csqrt, ceil as Cceil, abs as Cabs, floor as Cfloor, round as Cround
from libc.math cimport cos as Ccos, sin as Csin, atan2 as Catan2, pi as Cpi


# import
import numpy as np
from matplotlib.path import Path




__all__ = ['CoordShift','_Ves_isInside',
           '_Ves_mesh_dlfromL_cython', '_Ves_mesh_CrossPoly',
           '_Ves_Vmesh_Tor_SubFromD_cython', '_Ves_Vmesh_Tor_SubFromInd_cython',
           '_Ves_Vmesh_Lin_SubFromD_cython', '_Ves_Vmesh_Lin_SubFromInd_cython',
           '_Ves_Smesh_Tor_SubFromD_cython', '_Ves_Smesh_Tor_SubFromInd_cython',
           '_Ves_Smesh_TorStruct_SubFromD_cython', '_Ves_Smesh_TorStruct_SubFromInd_cython',
           '_Ves_Smesh_Lin_SubFromD_cython', '_Ves_Smesh_Lin_SubFromInd_cython']






########################################################
########################################################
#       Coordinates handling
########################################################

def CoordShift(Pts, In='(X,Y,Z)', Out='(R,Z)', CrossRef=None):
    """ Check the shape of an array of points coordinates and/or converts from 2D to 3D, 3D to 2D, cylindrical to cartesian... (CrossRef is an angle (Tor) or a distance (X for Lin))"""
    assert all([type(ff) is str and ',' in ff for ff in [In,Out]]), "Arg In and Out (coordinate format) must be comma-separated  !"
    assert type(Pts) is np.ndarray and Pts.ndim in [1,2] and Pts.shape[0] in (2,3), "Points must be a 1D or 2D np.ndarray of 2 or 3 coordinates !"
    assert CrossRef is None or type(CrossRef) in [int,float,np.int64,np.float64], "Arg CrossRef must be a float !"
    
    # Pre-format inputs
    In, Out = In.lower(), Out.lower()

    # Get order
    Ins = In.replace('(','').replace(')','').split(',')
    Outs = Out.replace('(','').replace(')','').split(',')
    assert all([ss in ['x','y','z','r','phi'] for ss in Ins]), "Non-valid In !"
    assert all([ss in ['x','y','z','r','phi'] for ss in Outs]), "Non-valid Out !"
    InT = 'cyl' if any([ss in Ins for ss in ['r','phi']]) else 'cart'
    OutT = 'cyl' if any([ss in Outs for ss in ['r','phi']]) else 'cart'

    ndim = Pts.ndim
    if ndim==1:
        Pts = np.copy(Pts.reshape((Pts.shape[0],1)))

    # Compute
    if InT==OutT:
        assert all([ss in Ins for ss in Outs])
        pts = []
        for ii in Outs:
            if ii=='phi':
                pts.append(np.arctan2(np.sin(Pts[Ins.index(ii),:]),np.cos(Pts[Ins.index(ii),:])))
            else:
                pts.append(Pts[Ins.index(ii),:])
    elif InT=='cart':
        pts = []
        for ii in Outs:
            if ii=='r':
                assert all([ss in Ins for ss in ['x','y']])
                pts.append(np.hypot(Pts[Ins.index('x'),:],Pts[Ins.index('y'),:]))
            elif ii=='z':
                assert 'z' in Ins                
                pts.append(Pts[Ins.index('z'),:])
            elif ii=='phi':
                if all([ss in Ins for ss in ['x','y']]):
                    pts.append(np.arctan2(Pts[Ins.index('y'),:],Pts[Ins.index('x'),:]))
                elif CrossRef is not None:
                    pts.append( CrossRef*np.ones((Pts.shape[1],)) )
                else:
                    raise Exception("There is no phi value available !")
    else:
        pts = []
        for ii in Outs:
            if ii=='x':
                if all([ss in Ins for ss in ['r','phi']]):
                    pts.append(Pts[Ins.index('r'),:]*np.cos(Pts[Ins.index('phi'),:]))
                elif CrossRef is not None:
                    pts.append( CrossRef*np.ones((Pts.shape[1],)) )
                else:
                    raise Exception("There is no x value available !")
            elif ii=='y':
                assert all([ss in Ins for ss in ['r','phi']])
                pts.append(Pts[Ins.index('r'),:]*np.sin(Pts[Ins.index('phi'),:]))
            elif ii=='z':
                assert 'z' in Ins
                pts.append(Pts[Ins.index('z'),:])

    # Format output
    pts = np.vstack(pts)
    if ndim==1:
        pts = pts.flatten()
    return pts







"""
########################################################
########################################################
########################################################
#                       Ves-specific
########################################################
########################################################
########################################################
"""




########################################################
########################################################
#       isInside
########################################################

def _Ves_isInside(Pts, VPoly, VLong=None, VType='Tor', In='(X,Y,Z)', Test=True):
    if Test:
        assert type(Pts) is np.ndarray and Pts.ndim in [1,2], "Arg Pts must be a 1D or 2D np.ndarray !"
        assert type(VPoly) is np.ndarray and VPoly.ndim==2 and VPoly.shape[0]==2, "Arg VPoly must be a (2,N) np.ndarray !"
        assert VLong is None or (hasattr(VLong,'__iter__') and len(VLong)==2), "Arg VLong must be a len()==2 iterable !"
        assert type(VType) is str and VType.lower() in ['tor','lin'], "Arg VType must be a str in ['Tor','Lin'] !"

    path = Path(VPoly.T)
    if VType.lower()=='tor':
        pts = CoordShift(Pts, In=In, Out='(R,Z,Phi)')
        ind = Path(VPoly.T).contains_points(pts[:2,:].T, transform=None, radius=0.0)
        if VLong is not None:
            VLong = [Catan2(Csin(VLong[0]),Ccos(VLong[0])), Catan2(Csin(VLong[1]),Ccos(VLong[1]))]
            if VLong[0]<VLong[1]:
                ind = ind & (pts[2,:]>=VLong[0]) & (pts[2,:]<=VLong[1])
            else:
                ind = ind & ((pts[2,:]>=VLong[0]) | (pts[2,:]<=VLong[1]))
    else:
        pts = CoordShift(Pts, In=In, Out='(X,Y,Z)')
        ind = Path(VPoly.T).contains_points(pts[1:,:].T, transform=None, radius=0.0)
        ind = ind & (pts[0,:]>=VLong[0]) & (pts[0,:]<=VLong[1])
    return ind





########################################################
########################################################
#       Meshing - Common - Linear
########################################################


# Preliminary function to get optimal resolution from input resolution
@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_mesh_dlfromL_cython(double[::1] LMinMax, double dL, DL=None, Lim=True, double margin=1.e-9):
    """ Get the actual reolution from the desired resolution and MinMax and limits """
    # Get the number of mesh elements in LMinMax
    cdef double N = Cceil((LMinMax[1]-LMinMax[0])/dL)
    # Derive the real (effective) resolution
    cdef double dLr = (LMinMax[1]-LMinMax[0])/N
    # Get desired limits if any
    cdef double[::1] DLc, L
    cdef long [::1] indL
    #cdef cnp.ndarray[double,ndim=1] indL, L
    cdef double abs0, abs1, A
    cdef int nL0, nL1, Nind, ii, jj
    cdef list dl
    if DL is None:
        DLc = LMinMax
    else:
        dl = list(DL)
        if dl[0] is None:
            dl[0] = LMinMax[0]
        if dl[1] is None:
            dl[1] = LMinMax[1]
        if Lim and dl[0]<=LMinMax[0]:
            dl[0] = LMinMax[0]
        if Lim and dl[1]>=LMinMax[1]:
            dl[1] = LMinMax[1]
        DLc = np.array(dl)       
 
    # Get the extreme indices of the mesh elements that really need to be created within those limits
    abs0 = Cabs(DLc[0]-LMinMax[0])
    if abs0-dLr*Cfloor(abs0/dLr)<margin*dLr:
        nL0 = int(Cround((DLc[0]-LMinMax[0])/dLr))
    else:
        nL0 = int(Cfloor((DLc[0]-LMinMax[0])/dLr))
    abs1 = Cabs(DLc[1]-LMinMax[0])
    if abs1-dLr*Cfloor(abs1/dLr)<margin*dLr:
        nL1 = int(Cround((DLc[1]-LMinMax[0])/dLr)-1)
    else:
        nL1 = int(Cfloor((DLc[1]-LMinMax[0])/dLr))
    # Get the corresponding indices
    Nind = nL1+1-nL0
    indL = np.empty((Nind,),dtype=int)#np.linspace(nL0,nL1,Nind,endpoint=True)
    L = np.empty((Nind,))
    for ii in range(0,Nind):
        jj = nL0+ii
        indL[ii] = jj
        L[ii] = LMinMax[0] + (0.5 + (<double>jj))*dLr
    return np.asarray(L), dLr, np.asarray(indL), <long>N


########################################################
########################################################
#       Meshing - Common - Polygon face
########################################################


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_mesh_CrossPoly(double[:,::1] VPoly, double dL, D1=None, D2=None, double margin=1.e-9, double DIn=0., VIn=None):
    cdef int ii, jj, nn=0, NP=VPoly.shape[1]
    cdef double[::1] LMinMax, L
    cdef double v0, v1, dlr
    cdef long[::1] indL
    cdef cnp.ndarray[long,ndim=1] N, ind
    cdef cnp.ndarray[double,ndim=1] dLr, Rref
    cdef cnp.ndarray[double,ndim=2] PtsCross
    cdef list LPtsCross=[], LdLr=[], Lind=[], LRref=[], VPolybis=[]
    
    LMinMax = np.array([0.,1.],dtype=float)
    N = np.empty((NP-1,),dtype=int)
    if DIn==0.:
        for ii in range(0,NP-1):
            v0, v1 = VPoly[0,ii+1]-VPoly[0,ii], VPoly[1,ii+1]-VPoly[1,ii]
            LMinMax[1] = Csqrt(v0**2 + v1**2)
            L, dlr, indL, N[ii] = _Ves_mesh_dlfromL_cython(LMinMax, dL, DL=None, Lim=True, margin=margin)
            VPolybis.append((VPoly[0,ii],VPoly[1,ii]))
            v0, v1 = v0/LMinMax[1], v1/LMinMax[1]
            for jj in range(0,N[ii]):
                LdLr.append(dlr)
                LRref.append(VPoly[0,ii] + L[jj]*v0)
                LPtsCross.append((VPoly[0,ii] + L[jj]*v0, VPoly[1,ii] + L[jj]*v1))
                Lind.append(nn)
                nn += 1
                VPolybis.append((VPoly[0,ii] + jj*dlr*v0, VPoly[1,ii] + jj*dlr*v1))
        VPolybis.append((VPoly[0,0],VPoly[1,0]))
    else:
        for ii in range(0,NP-1):
            v0, v1 = VPoly[0,ii+1]-VPoly[0,ii], VPoly[1,ii+1]-VPoly[1,ii]
            LMinMax[1] = Csqrt(v0**2 + v1**2)
            L, dlr, indL, N[ii] = _Ves_mesh_dlfromL_cython(LMinMax, dL, DL=None, Lim=True, margin=margin)
            VPolybis.append((VPoly[0,ii],VPoly[1,ii]))
            v0, v1 = v0/LMinMax[1], v1/LMinMax[1]
            for jj in range(0,N[ii]):
                LdLr.append(dlr)
                LRref.append(VPoly[0,ii] + L[jj]*v0)
                LPtsCross.append((VPoly[0,ii] + L[jj]*v0 + DIn*VIn[0,ii], VPoly[1,ii] + L[jj]*v1 + DIn*VIn[1,ii]))
                Lind.append(nn)
                nn += 1
                VPolybis.append((VPoly[0,ii] + jj*dlr*v0, VPoly[1,ii] + jj*dlr*v1))
        VPolybis.append((VPoly[0,0],VPoly[1,0]))
    
    PtsCross, dLr, ind, Rref = np.array(LPtsCross).T, np.array(LdLr), np.array(Lind,dtype=int), np.array(LRref)
    if D1 is not None:
        indin = (PtsCross[0,:]>=D1[0]) & (PtsCross[0,:]<=D1[1])
        PtsCross = PtsCross[:,indin]
        dLr, ind = dLr[indin], ind[indin]
    if D2 is not None:
        indin = (PtsCross[1,:]>=D2[0]) & (PtsCross[1,:]<=D2[1])
        PtsCross = PtsCross[:,indin]
        dLr, ind = dLr[indin], ind[indin]   
    
    return PtsCross, dLr, ind, N, Rref, np.array(VPolybis).T


########################################################
########################################################
#       Meshing - Volume - Tor
########################################################


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Vmesh_Tor_SubFromD_cython(double dR, double dZ, double dRPhi, 
                                   double[::1] RMinMax, double[::1] ZMinMax,
                                   DR=None, DZ=None, DPhi=None, VPoly=None,
                                   str Out='(X,Y,Z)', double margin=1.e-9):
    " Return the desired submesh indicated by the limits (DR,DZ,DPhi), for the desired resolution (dR,dZ,dRphi) "
    
    cdef double[::1] R0, R, Z, dRPhir, dPhir, NRPhi#, dPhi, NRZPhi_cum0, indPhi, phi
    cdef double dRr0, dRr, dZr, DPhi0, DPhi1
    cdef double abs0, abs1, phi, indiijj
    cdef long[::1] indR0, indR, indZ, Phin, NRPhi0
    cdef int NR0, NR, NZ, Rn, Zn, nRPhi0, indR0ii, ii, jj, nPhi0, nPhi1, zz, NP, NRPhi_int, Rratio
    cdef cnp.ndarray[double,ndim=2] Pts, indI
    cdef cnp.ndarray[double,ndim=1] iii, dV, ind
    
    # Get the actual R and Z resolutions and mesh elements
    R0, dRr0, indR0, NR0 = _Ves_mesh_dlfromL_cython(RMinMax, dR, None, Lim=True, margin=margin)
    R, dRr, indR, NR = _Ves_mesh_dlfromL_cython(RMinMax, dR, DR, Lim=True, margin=margin)
    Z, dZr, indZ, NZ = _Ves_mesh_dlfromL_cython(ZMinMax, dZ, DZ, Lim=True, margin=margin)
    Rn = len(R)
    Zn = len(Z)
    
    # Get the limits if any (and make sure to replace them in the proper quadrants)
    if DPhi is None:
        DPhi0, DPhi1 = -Cpi, Cpi
    else:
        DPhi0, DPhi1 = Catan2(Csin(DPhi[0]),Ccos(DPhi[0])), Catan2(Csin(DPhi[1]),Ccos(DPhi[1]))
    
    dRPhir, dPhir = np.empty((Rn,)), np.empty((Rn,))
    Phin = np.empty((Rn,),dtype=int)
    NRPhi = np.empty((Rn,))
    NRPhi0 = np.zeros((Rn,),dtype=int)
    nRPhi0, indR0ii = 0, 0
    NP, NPhimax = 0, 0
    Rratio = int(Cceil(R[Rn-1]/R[0]))
    for ii in range(0,Rn):
        # Get the actual RPhi resolution and Phi mesh elements (! depends on R !)
        NRPhi[ii] = Cceil(2.*Cpi*R[ii]/dRPhi)
        NRPhi_int = int(NRPhi[ii])
        dPhir[ii] = 2.*Cpi/NRPhi[ii]
        dRPhir[ii] = dPhir[ii]*R[ii]
        # Get index and cumulated indices from background
        for jj in range(indR0ii,NR0):
            if R0[jj]==R[ii]:
                indR0ii = jj
                break
            else:
                nRPhi0 += <long>Cceil(2.*Cpi*R0[jj]/dRPhi)
                NRPhi0[ii] = nRPhi0*NZ
        # Get indices of phi
        # Get the extreme indices of the mesh elements that really need to be created within those limits
        abs0 = Cabs(DPhi0+Cpi)
        if abs0-dPhir[ii]*Cfloor(abs0/dPhir[ii])<margin*dPhir[ii]:
            nPhi0 = int(Cround((DPhi0+Cpi)/dPhir[ii]))
        else:
            nPhi0 = int(Cfloor((DPhi0+Cpi)/dPhir[ii]))
        abs1 = Cabs(DPhi1+Cpi)
        if abs1-dPhir[ii]*Cfloor(abs1/dPhir[ii])<margin*dPhir[ii]:
            nPhi1 = int(Cround((DPhi1+Cpi)/dPhir[ii])-1)
        else:
            nPhi1 = int(Cfloor((DPhi1+Cpi)/dPhir[ii]))
            
        if DPhi0<DPhi1:
            #indI.append(list(range(nPhi0,nPhi1+1)))
            Phin[ii] = nPhi1+1-nPhi0
            if ii==0:
                indI = np.nan*np.ones((Rn,Phin[ii]*Rratio+1))
            for jj in range(0,Phin[ii]):
                indI[ii,jj] = <double>( nPhi0+jj )
        else:
            #indI.append(list(range(nPhi0,NRPhi_int)+list(range(0,nPhi1+1))))
            Phin[ii] = nPhi1+1+NRPhi_int-nPhi0
            if ii==0:
                indI = np.nan*np.ones((Rn,Phin[ii]*Rratio+1))
            for jj in range(0,NRPhi_int-nPhi0):
                indI[ii,jj] = <double>( nPhi0+jj )
            for jj in range(NRPhi_int-nPhi0,Phin[ii]):
                indI[ii,jj] = <double>( jj- (NRPhi_int-nPhi0) )
        NP += Zn*Phin[ii]   
    
    Pts = np.empty((3,NP))
    ind = np.empty((NP,))
    dV = np.empty((NP,))
    # Compute Pts, dV and ind
    # This triple loop is the longest part, it takes ~90% of the CPU time
    NP = 0
    if Out.lower()=='(x,y,z)':
        for ii in range(0,Rn):
            iii = np.sort(indI[ii,~np.isnan(indI[ii,:])]) # To make sure the indices are in increasing order
            for zz in range(0,Zn):
                for jj in range(0,Phin[ii]):
                    indiijj = iii[jj]
                    phi = -Cpi + (0.5+indiijj)*dPhir[ii]
                    Pts[0,NP] = R[ii]*Ccos(phi)
                    Pts[1,NP] = R[ii]*Csin(phi)
                    Pts[2,NP] = Z[zz]
                    ind[NP] = NRPhi0[ii] + indZ[zz]*NRPhi[ii] + indiijj
                    dV[NP] = dRr*dZr*dRPhir[ii]
                    NP += 1
    else:
        for ii in range(0,Rn):
            iii = np.sort(indI[ii,~np.isnan(indI[ii,:])])
            #assert iii.size==Phin[ii] and np.all(np.unique(iii)==iii)
            for zz in range(0,Zn):
                for jj in range(0,Phin[ii]):
                    indiijj = iii[jj] #indI[ii,iii[jj]]
                    Pts[0,NP] = R[ii]
                    Pts[1,NP] = Z[zz]
                    Pts[2,NP] = -Cpi + (0.5+indiijj)*dPhir[ii]
                    ind[NP] = NRPhi0[ii] + indZ[zz]*NRPhi[ii] + indiijj
                    dV[NP] = dRr*dZr*dRPhir[ii]
                    NP += 1

    if VPoly is not None:
        if Out.lower()=='(x,y,z)':
            R = np.hypot(Pts[0,:],Pts[1,:])
            indin = Path(VPoly.T).contains_points(np.array([R,Pts[2,:]]).T, transform=None, radius=0.0)
            Pts, dV, ind = Pts[:,indin], dV[indin], ind[indin]
            Ru = np.unique(R)
        else:
            indin = Path(VPoly.T).contains_points(Pts[:-1,:].T, transform=None, radius=0.0)        
            Pts, dV, ind = Pts[:,indin], dV[indin], ind[indin]
            Ru = np.unique(Pts[0,:])
        if not np.all(Ru==R):
            dRPhir = np.array([dRPhir[ii] for ii in range(0,len(R)) if R[ii] in Ru])
    return Pts, dV, ind.astype(int), dRr, dZr, np.asarray(dRPhir)




@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Vmesh_Tor_SubFromInd_cython(double dR, double dZ, double dRPhi, 
                                     double[::1] RMinMax, double[::1] ZMinMax, long[::1] ind, 
                                     str Out='(X,Y,Z)', double margin=1.e-9):
    """ Return the desired submesh indicated by the (numerical) indices, for the desired resolution (dR,dZ,dRphi) """
    cdef double[::1] R, Z, dRPhirRef, dPhir, Ru, dRPhir
    cdef double dRr, dZr, phi
    cdef long[::1] indR, indZ, NRPhi0, NRPhi
    cdef long NR, NZ, Rn, Zn, NP=len(ind), Rratio
    cdef int ii=0, jj=0, iiR, iiZ, iiphi
    cdef double[:,::1] Phi
    cdef cnp.ndarray[double,ndim=2] Pts=np.empty((3,NP))
    cdef cnp.ndarray[double,ndim=1] dV=np.empty((NP,))
    
    # Get the actual R and Z resolutions and mesh elements
    R, dRr, indR, NR = _Ves_mesh_dlfromL_cython(RMinMax, dR, None, Lim=True, margin=margin)
    Z, dZr, indZ, NZ = _Ves_mesh_dlfromL_cython(ZMinMax, dZ, None, Lim=True, margin=margin)
    Rn, Zn = len(R), len(Z)
    
    # Number of Phi per R
    dRPhirRef, dPhir = np.empty((NR,)), np.empty((NR,))
    Ru, dRPhir = np.zeros((NR,)), np.nan*np.ones((NR,))
    NRPhi, NRPhi0 = np.empty((NR,),dtype=int), np.empty((NR+1,),dtype=int)
    Rratio = int(Cceil(R[NR-1]/R[0]))
    for ii in range(0,NR):
        NRPhi[ii] = <long>(Cceil(2.*Cpi*R[ii]/dRPhi))
        dRPhirRef[ii] = 2.*Cpi*R[ii]/<double>(NRPhi[ii])
        dPhir[ii] = 2.*Cpi/<double>(NRPhi[ii])
        if ii==0:
            NRPhi0[ii] = 0
            Phi = np.empty((NR,NRPhi[ii]*Rratio+1))
        else:
            NRPhi0[ii] = NRPhi0[ii-1] + NRPhi[ii-1]*NZ
        for jj in range(0,NRPhi[ii]):
            Phi[ii,jj] = -Cpi + (0.5+<double>jj)*dPhir[ii]
            
    if Out.lower()=='(x,y,z)':
        for ii in range(0,NP):
            for jj in range(0,NR+1):
                if ind[ii]-NRPhi0[jj]<0.:
                    break
            iiR = jj-1
            iiZ = (ind[ii] - NRPhi0[iiR])//NRPhi[iiR]
            iiphi = ind[ii] - NRPhi0[iiR] - iiZ*NRPhi[iiR]
            phi = Phi[iiR,iiphi]
            Pts[0,ii] = R[iiR]*Ccos(phi)
            Pts[1,ii] = R[iiR]*Csin(phi)
            Pts[2,ii] = Z[iiZ]
            dV[ii] = dRr*dZr*dRPhirRef[iiR]
            if Ru[iiR]==0.:
                dRPhir[iiR] = dRPhirRef[iiR]
                Ru[iiR] = 1.
    else: 
        for ii in range(0,NP):
            for jj in range(0,NR+1):
                if ind[ii]-NRPhi0[jj]<0.:
                    break
            iiR = jj-1
            iiZ = (ind[ii] - NRPhi0[iiR])//NRPhi[iiR]
            iiphi = ind[ii] - NRPhi0[iiR] - iiZ*NRPhi[iiR]
            Pts[0,ii] = R[iiR]
            Pts[1,ii] = Z[iiZ]
            Pts[2,ii] = Phi[iiR,iiphi]
            dV[ii] = dRr*dZr*dRPhirRef[iiR]
            if Ru[iiR]==0.:
                dRPhir[iiR] = dRPhirRef[iiR]
                Ru[iiR] = 1.
    
    return Pts, dV, dRr, dZr, np.asarray(dRPhir)[~np.isnan(dRPhir)]



########################################################
########################################################
#       Meshing - Volume - Lin
########################################################


def _Ves_Vmesh_Lin_SubFromD_cython(double dX, double dY, double dZ, 
                                   double[::1] XMinMax, double[::1] YMinMax, double[::1] ZMinMax,
                                   DX=None, DY=None, DZ=None, VPoly=None,
                                   double margin=1.e-9):
    " Return the desired submesh indicated by the limits (DX,DY,DZ), for the desired resolution (dX,dY,dZ) "
    
    cdef double[::1] X, Y, Z
    cdef double dXr, dYr, dZr, dV
    cdef cnp.ndarray[long,ndim=1] indX, indY, indZ
    cdef int NX, NY, NZ, Xn, Yn, Zn
    cdef cnp.ndarray[double,ndim=2] Pts
    cdef cnp.ndarray[long,ndim=1] ind
    
    # Get the actual X, Y and Z resolutions and mesh elements
    X, dXr, indX, NX = _Ves_mesh_dlfromL_cython(XMinMax, dX, DX, Lim=True, margin=margin)
    Y, dYr, indY, NY = _Ves_mesh_dlfromL_cython(YMinMax, dY, DY, Lim=True, margin=margin)
    Z, dZr, indZ, NZ = _Ves_mesh_dlfromL_cython(ZMinMax, dZ, DZ, Lim=True, margin=margin)
    Xn, Yn, Zn = len(X), len(Y), len(Z)
        
    Pts = np.array([np.tile(X,(Yn*Zn,1)).flatten(), np.tile(np.repeat(Y,Xn),(Zn,1)).flatten(), np.repeat(Z,Xn*Yn)])
    ind = np.repeat(NX*NY*indZ,Xn*Yn) + np.tile(np.repeat(NX*indY,Xn),(Zn,1)).flatten() + np.tile(indX,(Yn*Zn,1)).flatten()
    dV = dXr*dYr*dZr
    
    if VPoly is not None:
        indin = Path(VPoly.T).contains_points(Pts[1:,:].T, transform=None, radius=0.0)
        Pts, ind = Pts[:,indin], ind[indin]
    
    return Pts, dV, ind.astype(int), dXr, dYr, dZr
    

def _Ves_Vmesh_Lin_SubFromInd_cython(double dX, double dY, double dZ, 
                                     double[::1] XMinMax, double[::1] YMinMax, double[::1] ZMinMax,
                                     cnp.ndarray[long,ndim=1] ind, double margin=1.e-9):
    " Return the desired submesh indicated by the limits (DX,DY,DZ), for the desired resolution (dX,dY,dZ) "
    
    cdef cnp.ndarray[double,ndim=1] X, Y, Z
    cdef double dXr, dYr, dZr, dV
    cdef long[::1] bla
    cdef cnp.ndarray[long,ndim=1] indX, indY, indZ
    cdef int NX, NY, NZ, Xn, Yn, Zn
    cdef cnp.ndarray[double,ndim=2] Pts
    
    # Get the actual X, Y and Z resolutions and mesh elements
    X, dXr, bla, NX = _Ves_mesh_dlfromL_cython(XMinMax, dX, None, Lim=True, margin=margin)
    Y, dYr, bla, NY = _Ves_mesh_dlfromL_cython(YMinMax, dY, None, Lim=True, margin=margin)
    Z, dZr, bla, NZ = _Ves_mesh_dlfromL_cython(ZMinMax, dZ, None, Lim=True, margin=margin)
    
    indZ = ind // (NX*NY)
    indY = (ind - NX*NY*indZ) // NX
    indX = ind - NX*NY*indZ - NX*indY
    Pts = np.array([X[indX.astype(int)], Y[indY.astype(int)], Z[indZ.astype(int)]])
    dV = dXr*dYr*dZr
    
    return Pts, dV, dXr, dYr, dZr







########################################################
########################################################
#       Meshing - Surface - Tor
########################################################

def _getBoundsInter2AngSeg(bool Full, double Phi0, double Phi1, double DPhi0, double DPhi1):
    """ Return Inter=True if an intersection exist (all angles in radians in [-pi;pi])

    If Inter, return Bounds, a list of tuples indicating the segments defining the intersection, with
    The intervals are ordered from lowest index to highst index (with respect to [Phi0,Phi1]) 
    """
    if Full:
        Bounds = [[DPhi0,DPhi1]] if DPhi0<=DPhi1 else [[-Cpi,DPhi1],[DPhi0,Cpi]]
        Inter = True
        Faces = [None, None]

    else:
        Inter, Bounds, Faces = False, None, [False,False]
        if Phi0<=Phi1:
            if DPhi0<=DPhi1:
                if DPhi0<=Phi1 and DPhi1>=Phi0:
                    Inter = True
                    Bounds = [[None,None]]
                    Bounds[0][0] = Phi0 if DPhi0<=Phi0 else DPhi0
                    Bounds[0][1] = Phi1 if DPhi1>=Phi1 else DPhi1
                    Faces[0] = DPhi0<=Phi0
                    Faces[1] = DPhi1>=Phi1
            else:
                if DPhi0<=Phi1 or DPhi1>=Phi0:
                    Inter = True
                    if DPhi0<=Phi1 and DPhi1>=Phi0:
                        Bounds = [[Phi0,DPhi1],[DPhi0,Phi1]]
                        Faces = [True,True] 
                    else:
                        Bounds = [[None,None]]
                        if DPhi0<=Phi1:
                            Bounds[0][0] = Phi0 if DPhi0<=Phi0 else DPhi0
                            Bounds[0][1] = Phi1
                            Faces[0] = DPhi0<=Phi0
                            Faces[1] = True
                        else:
                            Bounds[0][0] = Phi0
                            Bounds[0][1] = Phi1 if DPhi1>=Phi1 else DPhi1
                            Faces[0] = True
                            Faces[1] = DPhi1>=Phi1
        else:
            if DPhi0<=DPhi1:
                if DPhi0<=Phi1 or DPhi1>=Phi0:
                    Inter = True
                    if DPhi0<=Phi1 and DPhi1>=Phi0:
                        Bounds = [[Phi0,DPhi1],[DPhi0,Phi1]]
                        Faces = [True,True]
                    else:
                        Bounds = [[None,None]]
                        if DPhi0<=Phi1:
                            Bounds[0][0] = DPhi0
                            Bounds[0][1] = Phi1 if DPhi1>=Phi1 else DPhi1
                            Faces[1] = DPhi1>=Phi1
                        else:
                            Bounds[0][0] = Phi0 if DPhi0<=Phi0 else DPhi0
                            Bounds[0][1] = DPhi1
                            Faces[0] = DPhi0<=Phi0
            else:
                Inter = True
                if DPhi0>=Phi0 and DPhi1>=Phi0:
                    Bounds = [[Phi0,DPhi1],[DPhi0,Cpi],[-Cpi,Phi1]]
                    Faces = [True,True]
                elif DPhi0<=Phi1 and DPhi1<=Phi1:
                    Bounds = [[Phi0,Cpi],[-Cpi,DPhi1],[DPhi0,Phi1]]
                    Faces = [True,True]
                else:
                    Bounds = [[None,Cpi],[-Cpi,None]]
                    Bounds[0][0] = Phi0 if DPhi0<=Phi0 else DPhi0
                    Bounds[1][1] = Phi1 if DPhi1>=Phi1 else DPhi1
                    Faces[0] = DPhi0<=Phi0
                    Faces[1] = DPhi1>=Phi1
    return Inter, Bounds, Faces




@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Smesh_Tor_SubFromD_cython(double dL, double dRPhi, 
                                   double[:,::1] VPoly,
                                   DR=None, DZ=None, DPhi=None,
                                   double DIn=0., VIn=None, PhiMinMax=None,
                                   str Out='(X,Y,Z)', double margin=1.e-9):
    " Return the desired surfacic submesh indicated by the limits (DR,DZ,DPhi), for the desired resolution (dR,dZ,dRphi) "
    cdef double[::1] R, Z, dPhir, NRPhi#, dPhi, NRZPhi_cum0, indPhi, phi
    cdef double dRr0, dRr, dZr, DPhi0, DPhi1, DDPhi, DPhiMinMax
    cdef double abs0, abs1, phi, indiijj
    cdef long[::1] indR0, indR, indZ, Phin, NRPhi0, Indin
    cdef int NR0, NR, NZ, Rn, Zn, nRPhi0, indR0ii, ii, jj0=0, jj, nPhi0, nPhi1, zz, NP, NRPhi_int, Rratio, Ln
    cdef cnp.ndarray[double,ndim=2] Pts, indI, PtsCross, VPbis
    cdef cnp.ndarray[double,ndim=1] R0, dS, ind, dLr, Rref, dRPhir, iii
    cdef cnp.ndarray[long,ndim=1] indL, NL, indok
    
    # Pre-format input
    if PhiMinMax is None:
        PhiMinMax = [-Cpi,Cpi]
        DPhiMinMax = 2.*Cpi
        Full = True
    else:
        PhiMinMax = [Catan2(Csin(PhiMinMax[0]),Ccos(PhiMinMax[0])), Catan2(Csin(PhiMinMax[1]),Ccos(PhiMinMax[1]))]
        DPhiMinMax = PhiMinMax[1]-PhiMinMax[0] if PhiMinMax[1]>=PhiMinMax[0] else 2.*Cpi + PhiMinMax[1] - PhiMinMax[0]
        Full = False

    # Get the limits if any (and make sure to replace them in the proper quadrants)
    if DPhi is None:
        DPhi0, DPhi1 = PhiMinMax[0], PhiMinMax[1]
    else:
        DPhi0 = PhiMinMax[0] if DPhi[0] is None else Catan2(Csin(DPhi[0]),Ccos(DPhi[0]))
        DPhi1 = PhiMinMax[1] if DPhi[1] is None else Catan2(Csin(DPhi[1]),Ccos(DPhi[1]))
    DDPhi = DPhi1-DPhi0 if DPhi1>DPhi0 else 2.*Cpi+DPhi1-DPhi0

    Inter, Bounds, Faces = _getBoundsInter2AngSeg(Full, PhiMinMax[0], PhiMinMax[1], DPhi0, DPhi1)
    BC = list(Bounds)
    nBounds = len(Bounds)
    for ii in range(0,nBounds):
        if BC[ii][0]<PhiMinMax[0]:
            BC[ii][0] += 2.*Cpi
        if BC[ii][1]<=PhiMinMax[0]:
            BC[ii][1] += 2.*Cpi

    if Inter:

        # Get the actual R and Z resolutions and mesh elements
        PtsCross, dLr, indL, NL, Rref, VPbis = _Ves_mesh_CrossPoly(VPoly, dL, D1=None, D2=None, margin=margin, DIn=DIn, VIn=VIn)
        R0 = np.copy(Rref)
        NR0 = R0.size
        indin = np.ones((PtsCross.shape[1],),dtype=bool)
        if DR is not None:
            indin = indin & (R0>=DR[0]) & (R0<=DR[1])
        if DZ is not None:
            indin = indin & (PtsCross[1,:]>=DZ[0]) & (PtsCross[1,:]<=DZ[1])
        PtsCross, dLr, indL, Rref = PtsCross[:,indin], dLr[indin], indL[indin], Rref[indin]
        Ln = indin.sum()
        Indin = indin.nonzero()[0]

        dRPhir, dPhir = np.empty((Ln,)), np.empty((Ln,))
        Phin = np.zeros((Ln,),dtype=int)
        NRPhi = np.empty((Ln,))
        NRPhi0 = np.zeros((Ln,),dtype=int)
        nRPhi0, indR0ii = 0, 0
        NP, NPhimax = 0, 0
        Rratio = int(Cceil(np.max(Rref)/np.min(Rref)))
        indBounds = np.empty((2,nBounds),dtype=int)
        for ii in range(0,Ln):
            # Get the actual RPhi resolution and Phi mesh elements (! depends on R !)
            NRPhi[ii] = Cceil(DPhiMinMax*Rref[ii]/dRPhi)
            NRPhi_int = int(NRPhi[ii])
            dPhir[ii] = DPhiMinMax/NRPhi[ii]
            dRPhir[ii] = dPhir[ii]*Rref[ii]
            # Get index and cumulated indices from background
            for jj0 in range(indR0ii,NR0):
                if jj0==Indin[ii]:
                    indR0ii = jj0
                    break
                else:
                    nRPhi0 += <long>Cceil(DPhiMinMax*R0[jj0]/dRPhi)
                    NRPhi0[ii] = nRPhi0
            # Get indices of phi
            # Get the extreme indices of the mesh elements that really need to be created within those limits
            for kk in range(0,nBounds):
                abs0 = BC[kk][0]-PhiMinMax[0]                    
                if abs0-dPhir[ii]*Cfloor(abs0/dPhir[ii])<margin*dPhir[ii]:
                    nPhi0 = int(Cround(abs0/dPhir[ii]))
                else:
                    nPhi0 = int(Cfloor(abs0/dPhir[ii]))
                abs1 = BC[kk][1]-PhiMinMax[0]
                if abs1-dPhir[ii]*Cfloor(abs1/dPhir[ii])<margin*dPhir[ii]:
                    nPhi1 = int(Cround(abs1/dPhir[ii])-1)
                else:
                    nPhi1 = int(Cfloor(abs1/dPhir[ii]))
                indBounds[0,kk] = nPhi0
                indBounds[1,kk] = nPhi1
                Phin[ii] += nPhi1+1-nPhi0                    

            if ii==0:
                indI = np.nan*np.ones((Ln,Phin[ii]*Rratio+1))
            jj = 0
            for kk in range(0,nBounds):
                for kkb in range(indBounds[0,kk],indBounds[1,kk]+1):
                    indI[ii,jj] = <double>( kkb )
                    jj += 1
            NP += Phin[ii] 
        
        # Finish counting to get total number of points
        if jj0<=NR0-1:
            for jj0 in range(indR0ii,NR0):
                nRPhi0 += <long>Cceil(DPhiMinMax*R0[jj0]/dRPhi)
            
        # Compute Pts, dV and ind
        Pts = np.nan*np.ones((3,NP))
        ind = np.nan*np.ones((NP,))
        dS = np.nan*np.ones((NP,))
        # This triple loop is the longest part, it takes ~90% of the CPU time
        NP = 0
        if Out.lower()=='(x,y,z)':
            for ii in range(0,Ln):
                iii = np.unique(indI[ii,~np.isnan(indI[ii,:])]) # Some rare cases with doubles have to be eliminated
                for jj in range(0,len(iii)):
                    indiijj = iii[jj]
                    phi = PhiMinMax[0] + (0.5+indiijj)*dPhir[ii]
                    Pts[0,NP] = PtsCross[0,ii]*Ccos(phi)
                    Pts[1,NP] = PtsCross[0,ii]*Csin(phi)
                    Pts[2,NP] = PtsCross[1,ii]
                    ind[NP] = NRPhi0[ii] + indiijj
                    dS[NP] = dLr[ii]*dRPhir[ii]
                    NP += 1
        else:
            for ii in range(0,Ln):
                iii = np.unique(indI[ii,~np.isnan(indI[ii,:])])
                for jj in range(0,len(iii)):
                    indiijj = iii[jj]
                    Pts[0,NP] = PtsCross[0,ii]
                    Pts[1,NP] = PtsCross[1,ii]
                    Pts[2,NP] = PhiMinMax[0] + (0.5+indiijj)*dPhir[ii]
                    ind[NP] = NRPhi0[ii] + indiijj
                    dS[NP] = dLr[ii]*dRPhir[ii]
                    NP += 1
        indok = (~np.isnan(ind)).nonzero()[0]
        ind = ind[indok]
        dS = dS[indok]
        if len(indok)==1:
            Pts = Pts[:,indok].reshape((3,1)) 
        else:
            Pts = Pts[:,indok]
    else:
        Pts, dS, ind, NL, Rref, dRPhir, nRPhi0 = np.ones((3,0)), np.ones((0,)), np.ones((0,)), np.nan*np.ones((VPoly.shape[1]-1,)), np.ones((0,)), np.ones((0,)), 0
    return Pts, dS, ind.astype(int), NL, dLr, Rref, dRPhir, nRPhi0, VPbis



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Smesh_Tor_SubFromInd_cython(double dL, double dRPhi, 
                                     double[:,::1] VPoly, long[::1] ind, 
                                     double DIn=0., VIn=None, PhiMinMax=None,
                                     str Out='(X,Y,Z)', double margin=1.e-9):
    """ Return the desired submesh indicated by the (numerical) indices, for the desired resolution (dR,dZ,dRphi) """
    cdef double[::1] dRPhirRef, dPhir
    cdef long[::1] indL, NRPhi0, NRPhi
    cdef long NR, NZ, Rn, Zn, NP=len(ind), Rratio
    cdef int ii=0, jj=0, iiL, iiphi, Ln, nn=0, kk=0, nRPhi0
    cdef double[:,::1] Phi
    cdef cnp.ndarray[double,ndim=2] Pts=np.empty((3,NP)), indI, PtsCross, VPbis
    cdef cnp.ndarray[double,ndim=1] R0, dS=np.empty((NP,)), dLr, dRPhir, Rref
    cdef cnp.ndarray[long,ndim=1] NL

    # Pre-format input
    if PhiMinMax is None:
        PhiMinMax = [-Cpi,Cpi]
        DPhiMinMax = 2.*Cpi
    else:
        PhiMinMax = [Catan2(Csin(PhiMinMax[0]),Ccos(PhiMinMax[0])), Catan2(Csin(PhiMinMax[1]),Ccos(PhiMinMax[1]))]
        if PhiMinMax[1]>=PhiMinMax[0]:
            DPhiMinMax = PhiMinMax[1]-PhiMinMax[0]
        else:
            DPhiMinMax = 2.*Cpi + PhiMinMax[1] - PhiMinMax[0]
    
    
    # Get the actual R and Z resolutions and mesh elements
    PtsCross, dLrRef, indL, NL, RrefRef, VPbis = _Ves_mesh_CrossPoly(VPoly, dL, D1=None, D2=None, margin=margin, DIn=DIn, VIn=VIn)
    Ln = dLrRef.size
    # Number of Phi per R
    dRPhirRef, dPhir, dRPhir = np.empty((Ln,)), np.empty((Ln,)), -np.ones((Ln,))
    dLr, Rref = -np.ones((Ln,)), -np.ones((Ln,))
    NRPhi, NRPhi0 = np.empty((Ln,),dtype=int), np.empty((Ln,),dtype=int)
    Rratio = int(Cceil(np.max(RrefRef)/np.min(RrefRef)))
    for ii in range(0,Ln):
        NRPhi[ii] = <long>(Cceil(DPhiMinMax*RrefRef[ii]/dRPhi))
        dRPhirRef[ii] = DPhiMinMax*RrefRef[ii]/<double>(NRPhi[ii])
        dPhir[ii] = DPhiMinMax/<double>(NRPhi[ii])
        if ii==0:
            NRPhi0[ii] = 0
            Phi = np.empty((Ln,NRPhi[ii]*Rratio+1))
        else:
            NRPhi0[ii] = NRPhi0[ii-1] + NRPhi[ii-1]
        for jj in range(0,NRPhi[ii]):
            Phi[ii,jj] = PhiMinMax[0] + (0.5+<double>jj)*dPhir[ii]
    nRPhi0 = NRPhi0[Ln-1]+NRPhi[Ln-1]
            
    if Out.lower()=='(x,y,z)':
        for ii in range(0,NP):
            for jj in range(0,Ln+1):
                if ind[ii]-NRPhi0[jj]<0.:
                    break
            iiL = jj-1
            iiphi = ind[ii] - NRPhi0[iiL]
            Pts[0,ii] = PtsCross[0,iiL]*Ccos(Phi[iiL,iiphi])
            Pts[1,ii] = PtsCross[0,iiL]*Csin(Phi[iiL,iiphi])
            Pts[2,ii] = PtsCross[1,iiL]
            dS[ii] = dLrRef[iiL]*dRPhirRef[iiL]
            if dRPhir[iiL]==-1.:
                dRPhir[iiL] = dRPhirRef[iiL]
                dLr[iiL] = dLrRef[iiL]
                Rref[iiL] = RrefRef[iiL]
            
    else: 
        for ii in range(0,NP):
            for jj in range(0,Ln+1):
                if ind[ii]-NRPhi0[jj]<0.:
                    break
            iiL = jj-1
            iiphi = ind[ii] - NRPhi0[iiL]
            Pts[0,ii] = PtsCross[0,iiL]
            Pts[1,ii] = PtsCross[1,iiL]
            Pts[2,ii] = Phi[iiL,iiphi]
            dS[ii] = dLrRef[iiL]*dRPhirRef[iiL]
            if dRPhir[iiL]==-1.:
                dRPhir[iiL] = dRPhirRef[iiL]
                dLr[iiL] = dLrRef[iiL]
                Rref[iiL] = RrefRef[iiL]
    return Pts, dS, NL, dLr[dLr>-0.5], Rref[Rref>-0.5], dRPhir[dRPhir>-0.5], <long>nRPhi0, VPbis




########################################################
########################################################
#       Meshing - Surface - TorStruct
########################################################


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Smesh_TorStruct_SubFromD_cython(double[::1] PhiMinMax, double dL, double dRPhi, 
                                         double[:,::1] VPoly,
                                         DR=None, DZ=None, DPhi=None,
                                         double DIn=0., VIn=None,
                                         str Out='(X,Y,Z)', double margin=1.e-9):
    " Return the desired surfacic submesh indicated by the limits (DR,DZ,DPhi), for the desired resolution (dR,dZ,dRphi) "
    cdef double Dphi, dR0r=0., dZ0r=0.
    cdef int NR0=0, NZ0=0, R0n, Z0n, NRPhi0
    cdef double[::1] phiMinMax = np.array([Catan2(Csin(PhiMinMax[0]),Ccos(PhiMinMax[0])), Catan2(Csin(PhiMinMax[1]),Ccos(PhiMinMax[1]))])
    cdef cnp.ndarray[double, ndim=1] R0, Z0, dsF, dSM, dLr, Rref, dRPhir, dS
    cdef cnp.ndarray[long,ndim=1] indR0, indZ0, iind, iindF, indM, NL, ind
    cdef cnp.ndarray[double,ndim=2] ptsrz, pts, PtsM, VPbis, Pts
    cdef list LPts=[], LdS=[], Lind=[] 

    # Pre-format input
    if PhiMinMax is None:
        PhiMinMax = np.array([-Cpi,Cpi])
        DPhiMinMax = 2.*Cpi
        Full = True
    else:
        PhiMinMax = np.array([Catan2(Csin(PhiMinMax[0]),Ccos(PhiMinMax[0])), Catan2(Csin(PhiMinMax[1]),Ccos(PhiMinMax[1]))])
        DPhiMinMax = PhiMinMax[1]-PhiMinMax[0] if PhiMinMax[1]>=PhiMinMax[0] else 2.*Cpi + PhiMinMax[1] - PhiMinMax[0]
        Full = False

    # Get the limits if any (and make sure to replace them in the proper quadrants)
    if DPhi is None:
        DPhi0, DPhi1 = PhiMinMax[0], PhiMinMax[1]
    else:
        DPhi0 = PhiMinMax[0] if DPhi[0] is None else Catan2(Csin(DPhi[0]),Ccos(DPhi[0]))
        DPhi1 = PhiMinMax[1] if DPhi[1] is None else Catan2(Csin(DPhi[1]),Ccos(DPhi[1]))
    DDPhi = DPhi1-DPhi0 if DPhi1>DPhi0 else 2.*Cpi+DPhi1-DPhi0

    Inter, Bounds, Faces = _getBoundsInter2AngSeg(Full, PhiMinMax[0], PhiMinMax[1], DPhi0, DPhi1)
    BC = list(Bounds)
    nBounds = len(Bounds)
    for ii in range(0,nBounds):
        if BC[ii][0]<PhiMinMax[0]:
            BC[ii][0] += 2.*Cpi
        if BC[ii][1]<=PhiMinMax[0]:
            BC[ii][1] += 2.*Cpi   

    Dphi = DIn/np.max(VPoly[0,:]) if DIn!=0. else 0. # Required distance effective at max R

    # Get the mesh for the faces
    if any(Faces) :
        R0, dR0r, indR0, NR0 = _Ves_mesh_dlfromL_cython(np.array([np.min(VPoly[0,:]),np.max(VPoly[0,:])]), dL, DL=DR, Lim=True, margin=margin)
        Z0, dZ0r, indZ0, NZ0 = _Ves_mesh_dlfromL_cython(np.array([np.min(VPoly[1,:]),np.max(VPoly[1,:])]), dL, DL=DZ, Lim=True, margin=margin)
        R0n, Z0n = len(R0), len(Z0)
        ptsrz = np.array([np.tile(R0,Z0n),np.repeat(Z0,R0n)])
        iind = NR0*np.repeat(indZ0,R0n) + np.tile(indR0,Z0n)
        indin = Path(VPoly.T).contains_points(ptsrz.T, transform=None, radius=0.0)
        if np.any(indin):
            ptsrz = ptsrz[:,indin] if indin.sum()>1 else ptsrz[:,indin].reshape((2,1))
            iindF = iind[indin]
            dsF = dR0r*dZ0r*np.ones((indin.sum(),))

    # First face
    if Faces[0]:
        if Out.lower()=='(x,y,z)':
            pts = np.array([ptsrz[0,:]*Ccos(phiMinMax[0]+Dphi), ptsrz[0,:]*Csin(phiMinMax[0]+Dphi), ptsrz[1,:]])
        else:
            pts = np.array([ptsrz[0,:],ptsrz[1,:],(phiMinMax[0]+Dphi)*np.ones((indin.sum(),))])
        LPts.append( pts )
        Lind.append( iindF )
        LdS.append( dsF )
  
    # Main body
    PtsM, dSM, indM, NL, dLr, Rref, dRPhir, nRPhi0, VPbis = _Ves_Smesh_Tor_SubFromD_cython(dL, dRPhi, VPoly, 
                                                                                           DR=DR, DZ=DZ, DPhi=[DPhi0,DPhi1],
                                                                                           DIn=DIn, VIn=VIn, PhiMinMax=phiMinMax,
                                                                                           Out=Out, margin=margin)      

    if PtsM.shape[1]>=1:
        if PtsM.shape[1]==1:
            LPts.append(PtsM.reshape((3,1)))
        else:
            LPts.append(PtsM)
        Lind.append( indM + NR0*NZ0 )
        LdS.append( dSM )

    # Second face
    if Faces[1]:
        if Out.lower()=='(x,y,z)':
            pts = np.array([ptsrz[0,:]*Ccos(phiMinMax[1]-Dphi), ptsrz[0,:]*Csin(phiMinMax[1]-Dphi), ptsrz[1,:]])
        else:
            pts = np.array([ptsrz[0,:],ptsrz[1,:],(phiMinMax[1]-Dphi)*np.ones((indin.sum(),))])
        LPts.append( pts )
        Lind.append( iindF + NR0*NZ0 + nRPhi0 )
        LdS.append( dsF )

    # Aggregate
    if len(LPts)==1:
        Pts = LPts[0]
        ind = Lind[0]
        dS = LdS[0]
    else:
        Pts = np.concatenate(tuple(LPts),axis=1)
        ind = np.concatenate(tuple(Lind)).astype(int)
        dS = np.concatenate(tuple(LdS))
        
    return Pts, dS, ind, NL, dLr, Rref, dR0r, dZ0r, dRPhir, VPbis



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Smesh_TorStruct_SubFromInd_cython(double[::1] PhiMinMax, double dL, double dRPhi, 
                                           double[:,::1] VPoly, cnp.ndarray[long,ndim=1] ind,
                                           double DIn=0., VIn=None,
                                           str Out='(X,Y,Z)', double margin=1.e-9):
    " Return the desired surfacic submesh indicated by the limits (DR,DZ,DPhi), for the desired resolution (dR,dZ,dRphi) "
    cdef double Dphi, dR0r, dZ0r
    cdef int NR0, NZ0, R0n, Z0n, NRPhi0
    cdef double[::1] phiMinMax = np.array([Catan2(Csin(PhiMinMax[0]),Ccos(PhiMinMax[0])), Catan2(Csin(PhiMinMax[1]),Ccos(PhiMinMax[1]))])
    cdef cnp.ndarray[double, ndim=1] R0, Z0, dsF, dSM, dLr, Rref, dRPhir, dS
    cdef cnp.ndarray[long,ndim=1] bla, indR0, indZ0, iind, iindF, indM, NL
    cdef cnp.ndarray[double,ndim=2] ptsrz, pts, PtsM, VPbis, Pts
    cdef list LPts=[], LdS=[], Lind=[] 

    # Pre-format input
    Dphi = DIn/np.max(VPoly[0,:]) if DIn!=0. else 0. # Required distance effective at max R

    # Get the basic meshes for the faces
    R0, dR0r, bla, NR0 = _Ves_mesh_dlfromL_cython(np.array([np.min(VPoly[0,:]),np.max(VPoly[0,:])]), dL, DL=None, Lim=True, margin=margin)
    Z0, dZ0r, bla, NZ0 = _Ves_mesh_dlfromL_cython(np.array([np.min(VPoly[1,:]),np.max(VPoly[1,:])]), dL, DL=None, Lim=True, margin=margin)
    
    PtsM, dSM, indM, NL, dLr, Rref, dRPhir, nRPhi0, VPbis = _Ves_Smesh_Tor_SubFromD_cython(dL, dRPhi, VPoly, 
                                                                                           DR=None, DZ=None, DPhi=None,
                                                                                           DIn=DIn, VIn=VIn, PhiMinMax=phiMinMax,
                                                                                           Out=Out, margin=margin)
    # First face
    ii = (ind<NR0*NZ0).nonzero()[0]
    nii = len(ii)
    if nii>0:
        indZ0 = ind[ii] // NR0
        indR0 = (ind[ii]-indZ0*NR0)
        if Out.lower()=='(x,y,z)':
            pts = np.array([R0[indR0]*Ccos(phiMinMax[0]+Dphi), R0[indR0]*Csin(phiMinMax[0]+Dphi), Z0[indZ0]])
        else:
            pts = np.array([R0[indR0], Z0[indZ0], (phiMinMax[0]+Dphi)*np.ones((nii,))])
        pts = pts if nii>1 else pts.reshape((3,1))
        LPts.append( pts )
        LdS.append( dR0r*dZ0r*np.ones((nii,)) )
 
    # Main body
    ii = (ind>=NR0*NZ0) & (ind<NR0*NZ0+PtsM.shape[1])
    nii = len(ii)
    if nii>0:
        pts = PtsM[:,ind[ii]-NR0*NZ0] if nii>1 else PtsM[:,ind[ii]-NR0*NZ0].reshape((3,1))
        LPts.append( PtsM[:,ind[ii]-NR0*NZ0] )
        LdS.append( dSM[ind[ii]-NR0*NZ0] )
        
    # Second face
    ii = (ind >= NR0*NZ0+PtsM.shape[1] ).nonzero()[0]
    nii = len(ii)
    if nii>0:
        indZ0 = (ind[ii]-(NR0*NZ0+PtsM.shape[1])) // NR0
        indR0 = ind[ii]-(NR0*NZ0+PtsM.shape[1]) - indZ0*NR0
        if Out.lower()=='(x,y,z)':
            pts = np.array([R0[indR0]*Ccos(phiMinMax[1]-Dphi), R0[indR0]*Csin(phiMinMax[1]-Dphi), Z0[indZ0]])
        else:
            pts = np.array([R0[indR0], Z0[indZ0], (phiMinMax[1]-Dphi)*np.ones((nii,))])
        pts = pts if nii>1 else pts.reshape((3,1))
        LPts.append( pts )
        LdS.append( dR0r*dZ0r*np.ones((nii,)) )
        
    # Aggregate
    if len(LPts)==1:
        Pts = LPts[0]
        dS = LdS[0]
    elif len(LPts)>1:
        Pts = np.concatenate(tuple(LPts),axis=1)
        dS = np.concatenate(tuple(LdS))
        
    return Pts, dS, NL, dLr, Rref, dR0r, dZ0r, dRPhir, VPbis






########################################################
########################################################
#       Meshing - Surface - Lin
########################################################


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Smesh_Lin_SubFromD_cython(double[::1] XMinMax, double dL, double dX, 
                                   double[:,::1] VPoly,
                                   DX=None, DY=None, DZ=None,
                                   double DIn=0., VIn=None, double margin=1.e-9):
    " Return the desired surfacic submesh indicated by the limits (DX,DY,DZ), for the desired resolution (dX,dL) "
    cdef cnp.ndarray[double,ndim=1] X, Y0, Z0
    cdef double dXr, dY0r, dZ0r
    cdef int NY0, NZ0, Y0n, Z0n, NX, Xn, Ln, NR0
    cdef cnp.ndarray[double,ndim=2] Pts, PtsCross, VPbis
    cdef cnp.ndarray[double,ndim=1] dS, dLr, Rref
    cdef cnp.ndarray[long,ndim=1] indX, indY0, indZ0, indL, NL, ind
    
    # Preformat
    
    if DX is not None:
        DX = list(DX)
        if DX[0]<=XMinMax[0]:
            DX[0] = None
        if DX[1]>=XMinMax[1]:
            DX[1] = None
    if DY is not None:
        DY = list(DY)
        if DY[0]<=np.min(VPoly[0,:]):
            DY[0] = None
        if DY[1]>=np.max(VPoly[0,:]):
            DY[1] = None
    if DZ is not None:
        DZ = list(DZ)
        if DZ[0]<=np.min(VPoly[1,:]):
            DZ[0] = None
        if DZ[1]>=np.max(VPoly[1,:]):
            DZ[1] = None
    
    # Get the mesh for the faces
    Y0, dY0r, indY0, NY0 = _Ves_mesh_dlfromL_cython(np.array([np.min(VPoly[0,:]),np.max(VPoly[0,:])]), dL, DL=DY, Lim=True, margin=margin)
    Z0, dZ0r, indZ0, NZ0 = _Ves_mesh_dlfromL_cython(np.array([np.min(VPoly[1,:]),np.max(VPoly[1,:])]), dL, DL=DZ, Lim=True, margin=margin)
    Y0n, Z0n = len(Y0), len(Z0)
    
    # Get the actual R and Z resolutions and mesh elements
    X, dXr, indX, NX = _Ves_mesh_dlfromL_cython(XMinMax, dX, DL=DX, Lim=True, margin=margin)
    Xn = len(X)
    PtsCross, dLr, indL, NL, Rref, VPbis = _Ves_mesh_CrossPoly(VPoly, dL, D1=None, D2=None, margin=margin, DIn=DIn, VIn=VIn)
    NR0 = Rref.size
    indin = np.ones((PtsCross.shape[1],),dtype=bool)
    if DY is not None:
        if DY[0] is not None:
            indin = indin & (PtsCross[0,:]>=DY[0])
        if DY[1] is not None:
            indin = indin & (PtsCross[0,:]<=DY[1])
    if DZ is not None:
        if DZ[0] is not None:
            indin = indin & (PtsCross[1,:]>=DZ[0])
        if DZ[1] is not None:
            indin = indin & (PtsCross[1,:]<=DZ[1])
    PtsCross, dLr, indL, Rref = PtsCross[:,indin], dLr[indin], indL[indin], Rref[indin]
    Ln = indin.sum() 
    # Agregating
    Pts = np.array([np.repeat(X,Ln), np.tile(PtsCross[0,:],Xn), np.tile(PtsCross[1,:],Xn)])
    ind = NY0*NZ0 + np.repeat(indX*NR0,Ln) + np.tile(indL,Xn)
    dS = np.tile(dLr*dXr,Xn)
    if DX is None or DX[0] is None:
        pts = np.array([(XMinMax[0]+DIn)*np.ones((Y0n*Z0n,)), np.tile(Y0,Z0n), np.repeat(Z0,Y0n)])
        iind = NY0*np.repeat(indZ0,Y0n) + np.tile(indY0,Z0n)
        indin = Path(VPoly.T).contains_points(pts[1:,:].T, transform=None, radius=0.0)
        if np.any(indin):
            pts = pts[:,indin].reshape((3,1)) if indin.sum()==1 else pts[:,indin]
            Pts = np.concatenate((pts,Pts),axis=1)
            ind = np.concatenate((iind[indin], ind))
            dS = np.concatenate((dY0r*dZ0r*np.ones((indin.sum(),)),dS))
    if DX is None or DX[1] is None:
        pts = np.array([(XMinMax[1]-DIn)*np.ones((Y0n*Z0n,)), np.tile(Y0,Z0n), np.repeat(Z0,Y0n)])
        iind = NY0*NZ0 + NX*NR0 + NY0*np.repeat(indZ0,Y0n) + np.tile(indY0,Z0n)
        indin = Path(VPoly.T).contains_points(pts[1:,:].T, transform=None, radius=0.0)
        if np.any(indin):
            pts = pts[:,indin].reshape((3,1)) if indin.sum()==1 else pts[:,indin]
            Pts = np.concatenate((Pts,pts),axis=1)
            ind = np.concatenate((ind,iind[indin]))
            dS = np.concatenate((dS,dY0r*dZ0r*np.ones((indin.sum(),))))
    return Pts, dS, ind, NL, dLr, Rref, dXr, dY0r, dZ0r, VPbis



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def _Ves_Smesh_Lin_SubFromInd_cython(double[::1] XMinMax, double dL, double dX, 
                                     double[:,::1] VPoly, cnp.ndarray[long,ndim=1] ind,
                                     double DIn=0., VIn=None, double margin=1.e-9):
    " Return the desired surfacic submesh indicated by ind, for the desired resolution (dX,dL) "
    cdef double dXr, dY0r, dZ0r
    cdef int NX, NY0, NZ0, Ln, NR0, nii
    cdef list LPts, LdS
    cdef cnp.ndarray[double,ndim=2] Pts, PtsCross, VPbis
    cdef cnp.ndarray[double,ndim=1] X, Y0, Z0, dS, dLr, Rref
    cdef cnp.ndarray[long,ndim=1] indX, indY0, indZ0, indL, NL, ii
    
    # Get the mesh for the faces
    Y0, dY0r, bla, NY0 = _Ves_mesh_dlfromL_cython(np.array([np.min(VPoly[0,:]),np.max(VPoly[0,:])]), dL, DL=None, Lim=True, margin=margin)
    Z0, dZ0r, bla, NZ0 = _Ves_mesh_dlfromL_cython(np.array([np.min(VPoly[1,:]),np.max(VPoly[1,:])]), dL, DL=None, Lim=True, margin=margin)
    
    # Get the actual R and Z resolutions and mesh elements
    X, dXr, bla, NX = _Ves_mesh_dlfromL_cython(XMinMax, dX, DL=None, Lim=True, margin=margin)
    PtsCross, dLr, bla, NL, Rref, VPbis = _Ves_mesh_CrossPoly(VPoly, dL, D1=None, D2=None, margin=margin, DIn=DIn, VIn=VIn)
    Ln = PtsCross.shape[1] 

    LPts, LdS = [], []
    # First face
    ii = (ind<NY0*NZ0).nonzero()[0]
    nii = len(ii)
    if nii>0:
        indZ0 = ind[ii] // NY0
        indY0 = (ind[ii]-indZ0*NY0)
        if nii==1:
            LPts.append( np.array([[XMinMax[0]+DIn], [Y0[indY0]], [Z0[indZ0]]]) )
        else:
            LPts.append( np.array([(XMinMax[0]+DIn)*np.ones((nii,)), Y0[indY0], Z0[indZ0]]) )
        LdS.append( dY0r*dZ0r*np.ones((nii,)) )

    # Cylinder
    ii = ((ind>=NY0*NZ0) & (ind<NY0*NZ0+NX*Ln)).nonzero()[0]
    nii = len(ii)
    if nii>0:
        indX = (ind[ii]-NY0*NZ0) // Ln
        indL = (ind[ii]-NY0*NZ0 - Ln*indX)
        if nii==1:
            LPts.append( np.array([[X[indX]], [PtsCross[0,indL]], [PtsCross[1,indL]]]) )
            LdS.append( np.array([dXr*dLr[indL]]) )
        else:
            LPts.append( np.array([X[indX], PtsCross[0,indL], PtsCross[1,indL]]) )
            LdS.append( dXr*dLr[indL] )

    # End face
    ii = (ind >= NY0*NZ0+NX*Ln).nonzero()[0]
    nii = len(ii)
    if nii>0:
        indZ0 = (ind[ii]-NY0*NZ0-NX*Ln) // NY0
        indY0 = ind[ii]-NY0*NZ0-NX*Ln - NY0*indZ0
        if nii==1:
            LPts.append( np.array([[XMinMax[1]-DIn], [Y0[indY0]], [Z0[indZ0]]]) )
        else:
            LPts.append( np.array([(XMinMax[1]-DIn)*np.ones((nii,)), Y0[indY0], Z0[indZ0]]) )
        LdS.append( dY0r*dZ0r*np.ones((nii,)) )

    # Format output
    if len(LPts)==1:
        Pts, dS = LPts[0], LdS[0]
    else:
        Pts = np.concatenate(tuple(LPts),axis=1)
        dS = np.concatenate(tuple(LdS))
        
    return Pts, dS, NL, dLr, Rref, dXr, dY0r, dZ0r, VPbis






########################################################
########################################################
#       n
########################################################





