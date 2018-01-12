"""
This module is the computational part of the geometrical module of ToFu
"""

# General common libraries
import sys
import numpy as np
import scipy.interpolate as scpinterp
import scipy.integrate as scpintg
if sys.version[0]=='3':
    from inspect import signature as insp
elif sys.version[0]=='2':
    from inspect import getargspec as insp

# Less common libraries
import Polygon as plg

# ToFu-specific
try:
    import tofu.geom._def as _def
    import tofu.geom._GG as _GG
except Exception:
    from . import _def as _def
    from . import _GG as _GG



"""
###############################################################################
###############################################################################
                        Ves functions
###############################################################################
"""


############################################
#####       Ves sub-functions
############################################



def _Ves_set_Poly(Poly, arrayorder='C', Type='Tor', Lim=None, Clock=False):
    """ Prepare, format and compute all key geometrical attributes of a Ves object based on its Poly  """
    # Make Poly closed, counter-clockwise, with '(cc,N)' layout and good arrayorder
    Poly = _GG.Poly_Order(Poly, order='C', Clock=False, close=True, layout='(cc,N)', Test=True)
    assert Poly.shape[0]==2, "Arg Poly must be a 2D polygon !"
    NP = Poly.shape[1]
    P1Max = Poly[:,np.argmax(Poly[0,:])]
    P1Min = Poly[:,np.argmin(Poly[0,:])]
    P2Max = Poly[:,np.argmax(Poly[1,:])]
    P2Min = Poly[:,np.argmin(Poly[1,:])]
    BaryP = np.sum(Poly[:,:-1],axis=1,keepdims=False)/(Poly.shape[1]-1)
    BaryL = np.array([(P1Max[0]+P1Min[0])/2., (P2Max[1]+P2Min[1])/2.])
    TorP = plg.Polygon(Poly.T)
    Surf = TorP.area()
    BaryS = np.array(TorP.center()).flatten()
    Multi = False
    if Type.lower()=='lin':
        if all([hasattr(ll,'__iter__') for ll in Lim]):
            assert all([len(ll)==2 and not hasattr(ll[0],'__iter__') and not hasattr(ll[1],'__iter__') and ll[0]<ll[1] for ll in Lim])
            Multi = True
        else:
            assert len(Lim)==2 and not hasattr(Lim[0],'__iter__') and not hasattr(Lim[1],'__iter__') and Lim[0]<Lim[1]
        Lim = np.asarray(Lim)
        Vol, BaryV = None, None
    else:
        if Lim is not None:
            if all([hasattr(ll,'__iter__') for ll in Lim]):
                assert all([len(ll)==2 and not hasattr(ll[0],'__iter__') and not hasattr(ll[1],'__iter__') for ll in Lim])
                Multi = True
            else:
                assert len(Lim)==2 and not hasattr(Lim[0],'__iter__') and not hasattr(Lim[1],'__iter__')
            Lim = np.asarray(Lim)
        Vol, BaryV = _GG.Poly_VolAngTor(Poly)
        assert Vol > 0., "Pb. with volume computation for Ves object of type 'Tor' !"
    # Compute the non-normalized vector of each side of the Poly
    Vect = np.diff(Poly,n=1,axis=1)
    Vect = np.ascontiguousarray(Vect) if arrayorder=='C' else np.asfortranarray(Vect)
    # Compute the normalised vectors directed inwards
    Vin = np.array([Vect[1,:],-Vect[0,:]]) if _GG.Poly_isClockwise(Poly) else np.array([-Vect[1,:],Vect[0,:]])
    Vin = Vin/np.tile(np.hypot(Vin[0,:],Vin[1,:]),(2,1))
    Vin = np.ascontiguousarray(Vin) if arrayorder=='C' else np.asfortranarray(Vin)
    poly = _GG.Poly_Order(Poly, order=arrayorder, Clock=Clock, close=True, layout='(cc,N)', Test=True)
    return poly, NP, P1Max, P1Min, P2Max, P2Min, BaryP, BaryL, Surf, BaryS, Lim, Vol, BaryV, Vect, Vin, Multi


def _Ves_get_InsideConvexPoly(Poly, P2Min, P2Max, BaryS, RelOff=_def.TorRelOff, ZLim='Def', Spline=True, Splprms=_def.TorSplprms, NP=_def.TorInsideNP, Plot=False, Test=True):
    if Test:
        assert type(RelOff) is float, "Arg RelOff must be a float"
        assert ZLim is None or ZLim=='Def' or type(ZLim) in [tuple,list], "Arg ZLim must be a tuple (ZlimMin, ZLimMax)"
        assert type(Spline) is bool, "Arg Spline must be a bool !"
    if not ZLim is None:
        if ZLim=='Def':
            ZLim = (P2Min[1]+0.1*(P2Max[1]-P2Min[1]), P2Max[1]-0.05*(P2Max[1]-P2Min[1]))
        indZLim = (Poly[1,:]<ZLim[0]) | (Poly[1,:]>ZLim[1])
        Poly = np.delete(Poly, indZLim.nonzero()[0], axis=1)
    if np.all(Poly[:,0]==Poly[:,-1]):
        Poly = Poly[:,:-1]
    Np = Poly.shape[1]
    if Spline:
        BarySbis = np.tile(BaryS,(Np,1)).T
        Ptemp = (1.-RelOff)*(Poly-BarySbis)
        #Poly = BarySbis + Ptemp
        Ang = np.arctan2(Ptemp[1,:],Ptemp[0,:])
        Ang, ind = np.unique(Ang, return_index=True)
        Ptemp = Ptemp[:,ind]
        # spline parameters
        ww = Splprms[0]*np.ones((Np+1,))
        ss = Splprms[1]*(Np+1) # smoothness parameter
        kk = Splprms[2] # spline order
        nest = int((Np+1)/2.) # estimate of number of knots needed (-1 = maximal)
        # Find the knot points

        #tckp,uu = scpinterp.splprep([np.append(Ptemp[0,:],Ptemp[0,0]),np.append(Ptemp[1,:],Ptemp[1,0]),np.append(Ang,Ang[0]+2.*np.pi)], w=ww, s=ss, k=kk, nest=nest)
        tckp,uu = scpinterp.splprep([np.append(Ptemp[0,:],Ptemp[0,0]),np.append(Ptemp[1,:],Ptemp[1,0])], u=np.append(Ang,Ang[0]+2.*np.pi), w=ww, s=ss, k=kk, nest=nest, full_output=0)
        xnew,ynew = scpinterp.splev(np.linspace(-np.pi,np.pi,NP),tckp)
        Poly = np.array([xnew+BaryS[0],ynew+BaryS[1]])
        Poly = np.concatenate((Poly,Poly[:,0:1]),axis=1)
    if Plot:
        f = plt.figure(facecolor='w',figsize=(8,10))
        ax = f.add_axes([0.1,0.1,0.8,0.8])
        ax.plot(Poly[0,:], Poly[1,:],'-k', Poly[0,:],Poly[1,:],'-r')
        ax.set_aspect(aspect="equal",adjustable='datalim'), ax.set_xlabel(r"R (m)"), ax.set_ylabel(r"Z (m)")
        f.canvas.draw()
    return Poly



def _Ves_get_sampleEdge(VPoly, dL, DS=None, dLMode='abs', DIn=0., VIn=None, margin=1.e-9):
    types =[int,float,np.int32,np.int64,np.float32,np.float64]
    assert type(dL) in types and type(DIn) in types
    assert DS is None or (hasattr(DS,'__iter__') and len(DS)==2)
    if DS is None:
        DS = [None,None]
    else:
        assert all([ds is None or (hasattr(ds,'__iter__') and len(ds)==2 and all([ss is None or type(ss) in types for ss in ds])) for ds in DS])
    assert type(dLMode) is str and dLMode.lower() in ['abs','rel'], "Arg dLMode must be in ['abs','rel'] !" 
    #assert ind is None or (type(ind) is np.ndarray and ind.ndim==1 and ind.dtype in ['int32','int64'] and np.all(ind>=0)), "Arg ind must be None or 1D np.ndarray of positive int !"
    Pts, dLr, ind, N, Rref, VPolybis = _GG._Ves_Smesh_Cross(VPoly, float(dL), dLMode=dLMode.lower(), D1=DS[0], D2=DS[1], margin=margin, DIn=float(DIn), VIn=VIn)
    return Pts, dLr, ind



def _Ves_get_sampleCross(VPoly, Min1, Max1, Min2, Max2, dS, DS=None, dSMode='abs', ind=None, margin=1.e-9):
    types =[int,float,np.int32,np.int64,np.float32,np.float64]
    assert type(dS) in types or (hasattr(dS,'__iter__') and len(dS)==2 and all([type(ds) in types for ds in dS])), "Arg dS must be a float or a list 2 floats !"
    dS = [float(dS),float(dS)] if type(dS) in types else [float(dS[0]),float(dS[1])]
    assert DS is None or (hasattr(DS,'__iter__') and len(DS)==2)
    if DS is None:
        DS = [None,None]
    else:
        assert all([ds is None or (hasattr(ds,'__iter__') and len(ds)==2 and all([ss is None or type(ss) in types for ss in ds])) for ds in DS])
    assert type(dSMode) is str and dSMode.lower() in ['abs','rel'], "Arg dSMode must be in ['abs','rel'] !"
    assert ind is None or (type(ind) is np.ndarray and ind.ndim==1 and ind.dtype in ['int32','int64'] and np.all(ind>=0)), "Arg ind must be None or 1D np.ndarray of positive int !"
    
    MinMax1 = np.array([Min1,Max1])
    MinMax2 = np.array([Min2,Max2])
    if ind is None:
        Pts, dS, ind, d1r, d2r = _GG._Ves_meshCross_FromD(MinMax1, MinMax2, dS[0], dS[1], D1=DS[0], D2=DS[1], dSMode=dSMode, VPoly=VPoly, margin=margin)
    else:
        assert type(ind) is np.ndarray and ind.ndim==1 and ind.dtype in ['int32','int64'] and np.all(ind>=0), "Arg ind must be a np.ndarray of int !"
        Pts, dS, d1r, d2r = _GG._Ves_meshCross_FromInd(MinMax1, MinMax2, dS[0], dS[1], ind, dSMode=dSMode, margin=margin)
    return Pts, dS, ind, (d1r,d2r)


def _Ves_get_sampleV(VPoly, Min1, Max1, Min2, Max2, dV, DV=None, dVMode='abs', ind=None, VType='Tor', VLim=None, Out='(X,Y,Z)', margin=1.e-9):
    types =[int,float,np.int32,np.int64,np.float32,np.float64]
    assert type(dV) in types or (hasattr(dV,'__iter__') and len(dV)==3 and all([type(ds) in types for ds in dV])), "Arg dV must be a float or a list 3 floats !"
    dV = [float(dV),float(dV),float(dV)] if type(dV) in types else [float(dV[0]),float(dV[1]),float(dV[2])]
    assert DV is None or (hasattr(DV,'__iter__') and len(DV)==3)
    if DV is None:
        DV = [None,None,None]
    else:
        assert all([ds is None or (hasattr(ds,'__iter__') and len(ds)==2 and all([ss is None or type(ss) in types for ss in ds])) for ds in DV]), "Arg DV must be a list of 3 lists of 2 floats !"
    assert type(dVMode) is str and dVMode.lower() in ['abs','rel'], "Arg dVMode must be in ['abs','rel'] !"
    assert ind is None or (type(ind) is np.ndarray and ind.ndim==1 and ind.dtype in ['int32','int64'] and np.all(ind>=0)), "Arg ind must be None or 1D np.ndarray of positive int !"

    MinMax1 = np.array([Min1,Max1])
    MinMax2 = np.array([Min2,Max2])
    VLim = None if VType.lower()=='tor' else np.array(VLim)
    dVr = [None,None,None]
    if ind is None:
        if VType.lower()=='tor':
            Pts, dV, ind, dVr[0], dVr[1], dVr[2] = _GG._Ves_Vmesh_Tor_SubFromD_cython(dV[0], dV[1], dV[2], MinMax1, MinMax2, DR=DV[0], DZ=DV[1], DPhi=DV[2], VPoly=VPoly, Out=Out, margin=margin)
        else:
            Pts, dV, ind, dVr[0], dVr[1], dVr[2] = _GG._Ves_Vmesh_Lin_SubFromD_cython(dV[0], dV[1], dV[2], VLim, MinMax1, MinMax2, DX=DV[0], DY=DV[1], DZ=DV[2], VPoly=VPoly, margin=margin)
    else:
        if VType.lower()=='tor':
            Pts, dV, dVr[0], dVr[1], dVr[2] = _GG._Ves_Vmesh_Tor_SubFromInd_cython(dV[0], dV[1], dV[2], MinMax1, MinMax2, ind, Out=Out, margin=margin)
        else:
            Pts, dV, dVr[0], dVr[1], dVr[2] = _GG._Ves_Vmesh_Lin_SubFromInd_cython(dV[0], dV[1], dV[2], VLim, MinMax1, MinMax2, ind, margin=margin)
    return Pts, dV, ind, dVr


def _Ves_get_sampleS(VPoly, Min1, Max1, Min2, Max2, dS, DS=None, dSMode='abs', ind=None, DIn=0., VIn=None, VType='Tor', VLim=None, Out='(X,Y,Z)', margin=1.e-9, Multi=False, Ind=None):
    types =[int,float,np.int32,np.int64,np.float32,np.float64]
    assert type(dS) in types or (hasattr(dS,'__iter__') and len(dS)==2 and all([type(ds) in types for ds in dS])), "Arg dS must be a float or a list of 2 floats !"
    dS = [float(dS),float(dS),float(dS)] if type(dS) in types else [float(dS[0]),float(dS[1]),float(dS[2])]
    assert DS is None or (hasattr(DS,'__iter__') and len(DS)==3)
    if DS is None:
        DS = [None,None,None]
    else:
        assert all([ds is None or (hasattr(ds,'__iter__') and len(ds)==2 and all([ss is None or type(ss) in types for ss in ds])) for ds in DS]), "Arg DS must be a list of 3 lists of 2 floats !"
    assert type(dSMode) is str and dSMode.lower() in ['abs','rel'], "Arg dSMode must be in ['abs','rel'] !"
    assert type(Multi) is bool, "Arg Multi must be a bool !"

    VLim = None if VLim is None else np.array(VLim)
    MinMax1 = np.array([Min1,Max1])
    MinMax2 = np.array([Min2,Max2])

    if Multi:
        assert VLim is not None, "For multiple Struct, Lim cannot be None !"
        assert all([hasattr(ll,'__iter__') and len(ll)==2 for ll in VLim])
        if Ind is None:
            Ind = np.arange(0,len(VLim))
        else:
            Ind = [Ind] if not hasattr(Ind,'__iter__') else Ind
            Ind = np.asarray(Ind).astype(int)
        if ind is not None:
            assert hasattr(ind,'__iter__') and len(ind)==len(Ind), "For multiple Struct, ind must be a list of len() = len(Ind) !"
            assert all([type(ind[ii]) is np.ndarray and ind[ii].ndim==1 and ind[ii].dtype in ['int32','int64'] and np.all(ind[ii]>=0) for ii in range(0,len(ind))]), "For multiple Struct, ind must be a list of index arrays !"

    else:
        VLim = [VLim]
        assert ind is None or (type(ind) is np.ndarray and ind.ndim==1 and ind.dtype in ['int32','int64'] and np.all(ind>=0)), "Arg ind must be None or 1D np.ndarray of positive int !"
        Ind = [0]

    if ind is None:
        Pts, dS, ind, dSr = [0 for ii in Ind], [dS for ii in Ind], [0 for ii in Ind], [[0,0] for ii in Ind]
        if VType.lower()=='tor':
            for ii in range(0,len(Ind)):
                if VLim[Ind[ii]] is None:
                    Pts[ii], dS[ii], ind[ii], NL, dSr[ii][0], Rref, dSr[ii][1], nRPhi0, VPbis = _GG._Ves_Smesh_Tor_SubFromD_cython(dS[ii][0], dS[ii][1], VPoly, DR=DS[0], DZ=DS[1], DPhi=DS[2], DIn=DIn, VIn=VIn, PhiMinMax=None, Out=Out, margin=margin)
                else:
                    Pts[ii], dS[ii], ind[ii], NL, dSr[ii][0], Rref, dR0r, dZ0r, dSr[ii][1], VPbis = _GG._Ves_Smesh_TorStruct_SubFromD_cython(VLim[Ind[ii]], dS[ii][0], dS[ii][1], VPoly, DR=DS[0], DZ=DS[1], DPhi=DS[2], DIn=DIn, VIn=VIn, Out=Out, margin=margin)
                    dSr[ii] += [dR0r, dZ0r]
        else:
            for ii in range(0,len(Ind)):
                Pts[ii], dS[ii], ind[ii], NL, dSr[ii][0], Rref, dSr[ii][1], dY0r, dZ0r, VPbis = _GG._Ves_Smesh_Lin_SubFromD_cython(VLim[Ind[ii]], dS[ii][0], dS[ii][1], VPoly, DX=DS[0], DY=DS[1], DZ=DS[2], DIn=DIn, VIn=VIn, margin=margin)
                dSr[ii] += [dY0r, dZ0r]
    else:
        ind = ind if Multi else [ind]
        Pts, dS, dSr = [np.ones((3,0)) for ii in Ind], [dS for ii in Ind], [[0,0] for ii in Ind]
        if VType.lower()=='tor':
            for ii in range(0,len(Ind)):
                if ind[Ind[ii]].size>0:
                    if VLim[Ind[ii]] is None:
                        Pts[ii], dS[ii], NL, dSr[ii][0], Rref, dSr[ii][1], nRPhi0, VPbis = _GG._Ves_Smesh_Tor_SubFromInd_cython(dS[ii][0], dS[ii][1], VPoly, ind[Ind[ii]], DIn=DIn, VIn=VIn, PhiMinMax=None, Out=Out, margin=margin)
                    else:
                        Pts[ii], dS[ii], NL, dSr[ii][0], Rref, dR0r, dZ0r, dSr[ii][1], VPbis = _GG._Ves_Smesh_TorStruct_SubFromInd_cython(VLim[Ind[ii]], dS[ii][0], dS[ii][1], VPoly, ind[Ind[ii]], DIn=DIn, VIn=VIn, Out=Out, margin=margin)
                        dSr[ii] += [dR0r, dZ0r]
        else:
            for ii in range(0,len(Ind)):
                if ind[Ind[ii]].size>0:
                    Pts[ii], dS[ii], NL, dSr[ii][0], Rref, dSr[ii][1], dY0r, dZ0r, VPbis = _GG._Ves_Smesh_Lin_SubFromInd_cython(VLim[Ind[ii]], dS[ii][0], dS[ii][1], VPoly, ind[Ind[ii]], DIn=DIn, VIn=VIn, margin=margin)
                    dSr[ii] += [dY0r, dZ0r]

    if len(VLim)==1:
        Pts, dS, ind, dSr = Pts[0], dS[0], ind[0], dSr[0]
    return Pts, dS, ind, dSr




"""
###############################################################################
###############################################################################
                        LOS functions
###############################################################################
"""

def LOS_PRMin(Ds, dus, kPOut=None, Eps=1.e-12, Test=True):
    """  Compute the point on the LOS where the major radius is minimum """
    if Test:
        assert Ds.ndim in [1,2] and 3 in Ds.shape and Ds.shape==dus.shape
        assert kPOut is None or (Ds.ndim==1 and not hasattr(kPOut,'__iter__')) or (Ds.ndim==2 and kPOut.shape==(Ds.size/3,))

    v = Ds.ndim==1
    if v:
        Ds = Ds.reshape((3,1))
        dus = dus.reshape((3,1))
        if kPOut is not None:
            kPOut = np.array([kPOut])

    kRMin = np.nan*np.ones((Ds.shape[1],))
    uparN = np.sqrt(dus[0,:]**2 + dus[1,:]**2)

    # Case with u vertical
    ind = uparN>Eps
    kRMin[~ind] = 0.

    # Else
    kRMin[ind] = -(dus[0,ind]*Ds[0,ind]+dus[1,ind]*Ds[1,ind])/uparN[ind]**2

    # Check
    kRMin[kRMin<=0.] = 0.
    if kPOut is not None:
        kRMin[kRMin>kPOut] = kPOut[kRMin>kPOut]

    # Derive
    PRMin = Ds + kRMin[np.newaxis,:]*dus
    RMin = np.sqrt(PRMin[0,:]**2+PRMin[1,:]**2)

    if v:
        PRMin = PRMin.flatten()
        kRMin, RMin = kRMin[0], RMin[0]
    return PRMin, kRMin, RMin


def LOS_CrossProj(VType, Ds, us, kPIns, kPOuts, kRMins,
                  Lplot='In', Proj='All', multi=False):
    """ Compute the parameters to plot the poloidal projection of the LOS  """
    assert type(VType) is str and VType.lower() in ['tor','lin']
    assert Lplot.lower() in ['tot','in']
    assert Proj.lower() in ['cross','hor','all','3d']
    assert Ds.ndim==2 and Ds.shape==us.shape
    nL = Ds.shape[1]
    k0 = kPIns if Lplot.lower()=='in' else np.zeros((nL,))

    if VType.lower()=='tor' and Proj.lower() in ['cross','all']:
        CrossProjAng = np.arccos(np.sqrt(us[0,:]**2+us[1,:]**2)
                                 /np.sqrt(np.sum(us**2,axis=0)))
        nkp = np.ceil(25.*(1 - (CrossProjAng/(np.pi/4)-1)**2) + 2)
        ks = np.max([kRMins,kPIns],axis=0) if Lplot.lower()=='in' else kRMins
        pts0 = []
        if multi:
            for ii in range(0,nL):
                k = np.linspace(k0[ii],kPOuts[ii],nkp[ii],endpoint=True)
                k = np.unique(np.append(k,ks[ii]))
                pp = Ds[:,ii:ii+1] + k[np.newaxis,:]*us[:,ii:ii+1]
                pts0.append( np.array([np.hypot(pp[0,:],pp[1,:]),pp[2,:]])  )
        else:
            for ii in range(0,nL):
                k = np.linspace(k0[ii],kPOuts[ii],nkp[ii],endpoint=True)
                k = np.append(np.unique(np.append(k,ks[ii])),np.nan)
                pts0.append( Ds[:,ii:ii+1] + k[np.newaxis,:]*us[:,ii:ii+1] )
            pts0 = np.concatenate(tuple(pts0),axis=1)
            pts0 = np.array([np.hypot(pts0[0,:],pts0[1,:]),pts0[2,:]])

    if not (VType.lower()=='tor' and Proj.lower()=='cross'):
        pts = []
        if multi:
            for ii in range(0,nL):
                k = np.array([k0[ii],kPOuts[ii]])
                pts.append( Ds[:,ii:ii+1] + k[np.newaxis,:]*us[:,ii:ii+1] )
        else:
            for ii in range(0,nL):
                k = np.array([k0[ii],kPOuts[ii],np.nan])
                pts.append( Ds[:,ii:ii+1] + k[np.newaxis,:]*us[:,ii:ii+1] )
            pts = np.concatenate(tuple(pts),axis=1)

    if Proj.lower()=='hor':
        pts = [pp[:2,:] for pp in pts] if multi else pts[:2,:]
    elif Proj.lower()=='cross':
        if VType.lower()=='tor':
            pts = pts0
        else:
            pts = [pp[1,:] for pp in pts] if multi else pts[1:,:]
    elif Proj.lower()=='all':
        if multi:
            if VType.lower()=='tor':
                pts = [(p0,pp[:2,:]) for (p0,pp) in zip(*[pts0,pts])]
            else:
                pts = (pts[1,:],pts[:2,:])
        else:
            pts = (pts0,pts[:2,:]) if VType.lower()=='tor' else (pts[1,:],pts[:2,:])
    return pts


##############################################
#       Selection
##############################################

def Rays_touch(VesOk, StructOk, IndOut, StructNames, touch='Ves'):
    assert type(touch) in [str,list,tuple], "Arg touch must be a str or list !"
    if VesOk and touch=='Ves':
        ind = IndOut[0,:]==0
    elif StructOk:
       if type(touch) is str:
            ii = [ii for ii in range(0,len(StructNames))
                  if StructNames[ii]==touch]
            assert len(ii)==1, "Required Struct Name not found !"
            ind = IndOut[0,:]==ii[0]+1
       else:
            assert len(touch)==2, "Arg touch must be a list of len()==2 !"
            assert type(touch[0]) is str, "touch[0] must be str (Struct Name)"
            assert type(touch[1]) is int, "touch[1] must be int (Struct index)"
            ii = [ii for ii in range(0,len(StructNames))
                  if StructNames[ii]==touch[0]]
            assert len(ii)==1, "Required Struct Name not found !"
            ind = (IndOut[0,:]==ii[0]+1) & (IndOut[1,:]==touch[1])
    return ind





##############################################
#       Meshing & signal
##############################################

def LOS_get_sample(D, u, dL, DL=None, dLMode='abs', method='sum', Test=True):
    """ Return the sampled line, with the specified method

    'linspace': return the N+1 edges, including the first and last point
    'sum' : return the N middle of the segments
    'simps': return the N+1 egdes, where N has to be even (scipy.simpson requires an even number of intervals)
    'romb' : return the N+1 edges, where N+1 = 2**k+1 (fed to scipy.romb for integration)
    """
    if Test:
        assert all([type(dd) is np.ndarray and dd.shape==(3,) for dd in [D,u]])
        assert not hasattr(dL,'__iter__')
        assert DL is None or all([hasattr(DL,'__iter__'), len(DL)==2, all([not hasattr(dd,'__iter__') for dd in DL])])
        assert dLMode in ['abs','rel']
        assert type(method) is str and method in ['linspace','sum','simps','romb']
    # Compute the minimum number of intervals to satisfy the specified resolution
    N = int(np.ceil((DL[1]-DL[0])/dL)) if dLMode=='abs' else int(np.ceil(1./dL))
    # Modify N according to the desired method
    if method=='simps':
        N = N if N%2==0 else N+1
    elif method=='romb':
        N = 2**int(np.ceil(np.log(N)/np.log(2.)))

    # Derive k and dLr
    if method=='sum':
        dLr = (DL[1]-DL[0])/N
        k = DL[0] + (0.5+np.arange(0,N))*dLr
    else:
        k, dLr = np.linspace(DL[0], DL[1], N+1, endpoint=True, retstep=True, dtype=float)

    Pts = D[:,np.newaxis] + k[np.newaxis,:]*u[:,np.newaxis]
    return Pts, k, dLr


def LOS_calc_signal(ff, D, u, dL, DL=None, dLMode='abs', method='romb', Test=True):
    assert hasattr(ff,'__call__'), "Arg ff must be a callable (function) taking at least 1 positional Pts (a (3,N) np.ndarray of cartesian (X,Y,Z) coordinates) !"
    assert not method=='linspace'
    Pts, k, dLr = LOS_get_sample(D, u, dL, DL=DL, dLMode=dLMode, method=method, Test=Test)
    out = insp(ff)
    if sys.version[0]=='3':
        N = np.sum([(pp.kind==pp.POSITIONAL_OR_KEYWORD and pp.default is pp.empty) for pp in out.parameters.values()])
    else:
        N = len(out.args)

    if N==1:
        Vals = ff(Pts)
    elif N==2:
        Vals = ff(Pts, np.tile(-u,(Pts.shape[1],1)).T)
    else:
        raise ValueError("The function (ff) assessing the emissivity loccaly must take a single positional argument: Pts, a (3,N) np.ndarray of (X,Y,Z) cartesian coordinates !")

    Vals[np.isnan(Vals)] = 0.
    if method=='sum':
        Int = np.sum(Vals)*dLr
    elif method=='simps':
        Int = scpintg.simps(Vals, x=None, dx=dLr)
    elif method=='romb':
        Int = scpintg.romb(Vals, dx=dLr, show=False)
    return Int
