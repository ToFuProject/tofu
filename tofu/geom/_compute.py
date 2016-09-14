"""
This module is the geometrical part of the ToFu general package
It includes all functions and object classes necessary for tomography on Tokamaks
"""

#import matplotlib
#matplotlib.use('WxAgg')
#matplotlib.interactive(True)

import matplotlib.pyplot as plt
from matplotlib import _cntr as cntr
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon as mplg
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as scpinteg
import scipy.interpolate as scpinterp
import inspect

import Polygon as plg
import Polygon.Utils as plgut
import itertools as itt
import warnings
import math

import numpy as np
#import scipy.interpolate as scinterp

#from mayavi import mlab
import datetime as dtm
#import scipy as sp
import time as time


# ToFu specific
import tofu.defaults as TFD
from tofu.pathfile import ID as tfpfID
from . import General_Geom_cy as GG


"""
###############################################################################
###############################################################################
                        Ves functions
###############################################################################
"""


############################################
#####       Ves sub-functions
############################################

def _Ves_set_Poly(Poly, arrayorder, Type, DLong=None, Clock=False):
    # Make Poly closed, counter-clockwise, with '(cc,N)' layout and good arrayorder
    Poly = GG.PolyOrder(Poly, order='C', Clock=False, close=True, layout='(cc,N)', Test=True)
    assert Poly.shape[0]==2, "Arg Poly must be a planar polygon !"
    Poly = Poly
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
    if Type=='Lin':
        assert hasattr(DLong,'__iter__') and len(DLong)==2 and DLong[1]>DLong[0], "Arg DLong must be a iterable of len()==2 sorted in increasing order !"
        DLong = list(DLong)
        Vol, BaryV = None, None
    else:
        DLong = None
        Vol, BaryV = GG.Calc_VolBaryV_CylPol(Poly)
        assert Vol > 0., "Pb. with volume computation for Ves object of type 'Tor' !"
    # Compute the non-normalized vector of each side of the Poly
    Vect = np.diff(Poly,n=1,axis=1)
    Vect = np.ascontiguousarray(Vect) if arrayorder=='C' else np.asfortranarray(Vect)
    # Compute the normalised vectors directed inwards
    Vin = np.array([Vect[1,:],-Vect[0,:]]) if GG.PolyTestClockwise(Poly) else np.array([-Vect[1,:],Vect[0,:]])
    Vin = Vin/np.tile(np.hypot(Vin[0,:],Vin[1,:]),(2,1))
    Vin = np.ascontiguousarray(Vin) if arrayorder=='C' else np.asfortranarray(Vin)
    Poly = GG.PolyOrder(Poly, order=arrayorder, Clock=Clock, close=True, layout='(cc,N)', Test=True)
    return Poly, NP, P1Max, P1Min, P2Max, P2Min, BaryP, BaryL, Surf, BaryS, DLong, Vol, BaryV, Vect, Vin


def _Ves_isInside(Poly, Type, DLong, Points, In='(X,Y,Z)'):
    PathPol = Path(Poly.T)
    if Type=='Tor':
        Points = GG.CoordShift(Points,In=In,Out='(R,Z)')
        ind = PathPol.contains_points(Points.T, transform=None, radius=0.0)
    elif Type=='Lin':
        Pts = GG.CoordShift(Points,In=In,Out='(Y,Z)')
        ind = PathPol.contains_points(Pts.T, transform=None, radius=0.0)
        if In=='(X,Y,Z)':
            ind = ind & (Points[0,:]>=DLong[0]) & (Points[0,:]<=DLong[1])
    return ind


def _Ves_get_InsideConvexPoly(Poly, P2Min, P2Max, BaryS, RelOff=TFD.TorRelOff, ZLim='Def', Spline=True, Splprms=TFD.TorSplprms, NP=TFD.TorInsideNP, Plot=False, Test=True):
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


def _Ves_get_MeshCrossSection(P1Min, P1Max, P2Min, P2Max, Poly, Type, DLong=None, CrossMesh=[0.01,0.01], CrossMeshMode='abs', Test=True):
    if Test:
        assert all([type(pp) is np.ndarray and pp.shape==(2,) for pp in [P1Min,P1Max,P2Min,P2Max]]), "Args P1Min, P1Max, P2Min, P2Max must be 1-dim array of size==2 !"
        assert P1Min[0]<P1Max[0] and P2Min[1]<P2Max[1], "Arg P1Min should have smaller first coordinate than P1Max (respectively 2nd coordinate of P2Min and P2Max) !"
        assert type(Poly) is np.ndarray and Poly.ndim==2 and Poly.shape[0]==2, "Arg Poly must be a (2,N) array of 2D points coordinates !"
        assert Type in ['Tor','Lin'], "Arg Type must be in ['Tor','Lin'] !"
        assert hasattr(CrossMesh,'__iter__') and len(CrossMesh)==2, "Arg CrossMesh must be an iterable of len()==2 !"
        assert CrossMeshMode in ['abs','rel'], "Arg CrossMeshMode must be in ['abs','rel'] !"
    DX1 = [P1Min[0],P1Max[0]]
    DX2 = [P2Min[1],P2Max[1]]
    NumX1 = int(np.ceil(np.diff(DX1)/CrossMesh[0])) if CrossMeshMode=='abs' else int(1./CrossMesh[0])
    NumX2 = int(np.ceil(np.diff(DX2)/CrossMesh[1])) if CrossMeshMode=='abs' else int(1./CrossMesh[1])
    X1 = np.linspace(DX1[0],DX1[1],NumX1)
    X2 = np.linspace(DX2[0],DX2[1],NumX2)
    XX1 = np.tile(X1,(NumX2,1)).flatten()
    XX2 = np.tile(X2,(NumX1,1)).T.flatten()
    Pts = np.array([XX1, XX2])
    In = '(R,Z)' if Type=='Tor' else '(Y,Z)'
    ind = _Ves_isInside(Poly, Type, DLong, Pts, In=In)
    return Pts[:,ind], X1, X2, NumX1, NumX2







"""
###############################################################################
###############################################################################
                        LOS class and functions
###############################################################################
"""


############################################
##### Computing functions
############################################



def _LOS_calc_InOutPolProj(Type, Poly, Vin, DLong, D, uu, Name):
    if Type=='Tor':
        PIn, POut = GG.Calc_InOut_LOS_PIO(D.reshape((3,1)), uu.reshape((3,1)), np.ascontiguousarray(Poly), np.ascontiguousarray(Vin))
    else:
        PIn, POut = GG.Calc_InOut_LOS_PIO_Lin(D.reshape((3,1)), uu.reshape((3,1)), np.ascontiguousarray(Poly), np.ascontiguousarray(Vin), DLong)
    if np.any(np.isnan(PIn)):
        warnings.warn(Name+" seems to have no PIn (possible if LOS start point already inside Vessel), PIn is set to self.D !")
        PIn = D
    Err = np.any(np.isnan(POut))
    PIn, POut = PIn.flatten(), POut.flatten()
    kPIn = (PIn-D).dot(uu)
    kPOut = (POut-D).dot(uu)
    return PIn, POut, kPIn, kPOut, Err


def _LOS_set_CrossProj(Type, D, uu, kPIn, kPOut):
    if Type=='Tor':
        PRMin, RMin, kRMin, PolProjAng = GG.Calc_PolProj_LOS_cy(D.reshape((3,1)), uu.reshape((3,1)), kmax=kPOut)
        PRMin = PRMin.flatten()
        nkp = TFD.kpVsPolProjAng(PolProjAng)
        kplotTot = np.insert(np.linspace(0.,kPOut,nkp,endpoint=True),1,kRMin)
        kplotIn = np.insert(np.linspace(kPIn,kPOut,nkp,endpoint=True),1,max(kRMin,kPIn))
        kplotTot, kplotIn = np.unique(kplotTot), np.unique(kplotIn)
        LTot, LIn = kplotTot.size, kplotIn.size
        kplotTot = np.reshape(kplotTot,(LTot,1))
        kplotIn  = np.reshape(kplotIn,(LIn,1))
        PplotOut = (np.tile(D,(LTot,1)) + kplotTot*uu).T
        PplotIn  = (np.tile(D,(LIn,1)) + kplotIn*uu).T
    elif Type=='Lin':
        PRMin, RMin, kRMin, PolProjAng = np.nan*np.ones((3,)), np.nan, np.nan, np.nan
        PplotOut = np.array([D,D+uu*kPOut]).T
        PplotIn  = np.array([D+uu*kPIn,D+uu*kPOut]).T
    return PRMin, RMin, kRMin, PolProjAng, PplotOut, PplotIn


def Calc_DiscretiseLOS(L, SLim=TFD.LOSDiscrtSLim, SLMode=TFD.LOSDiscrtSLMode, DS=TFD.LOSDiscrtDS, SMode=TFD.LOSDiscrtSMode, Test=True):
# Function used to discretise a LOS as a series of points
# The discretisation can be done in a relative mode (=> specify fraction of the total length) or in absolute mode (=> specify absolute length of segments)
    if Test:
        assert isinstance(L,LOS), "Arg L should be LOS instance !"
        assert type(SLim) is tuple and len(SLim)==2 and SLim[0]<SLim[1], "Arg SLim should be a len=2 tuple with increasing numeric values !"
        assert type(DS) is float, "Arg DS should be a float value !"
        assert (SMode=='m' or SMode=='norm') and (SLMode=='m' or SLMode=='norm'), "Args SLMode and SMode should be 'm' (for meters) or 'norm' (for normalised) !"

    if SLMode == 'm':
        P0 = L.PIn + SLim[0]*L.u
        P1 = L.PIn + SLim[1]*L.u
    else:
        P0 = L.PIn + SLim[0]*(L.POut-L.PIn)
        P1 = L.PIn + SLim[1]*(L.POut-L.PIn)

    if SMode=='norm':
        DS = DS*np.linalg.norm(L.POut-L.PIn)

    D = np.linalg.norm(P1-P0)
    NP = int(np.round(D/DS))
    Points = np.dot(P0,np.ones((1,NP))) + np.dot(L.u,np.linspace(0,D,NP).reshape((1,NP)))
    return Points





"""
###############################################################################
###############################################################################
                   Lens class and functions
###############################################################################
"""



###############################################
###############################################
#       Lens and other-specific computation
###############################################


def Calc_nInFromTor_Poly(BaryS, nIn, VBaryS):
    """ Returns nIn such that it points inward the torus """
    assert isinstance(BaryS,np.ndarray) and isinstance(nIn,np.ndarray) and BaryS.size==nIn.size==3, "Args BaryS and nIn should be ndarrays of size 3 !"
    assert isinstance(VBaryS,np.ndarray) and VBaryS.size==2, "Arg VBaryS and nIn should be ndarrays of size 2 !"
    BaryS, nIn = BaryS.flatten(), nIn.flatten()
    R = np.hypot(BaryS[0],BaryS[1])
    BaryTor = np.array([VBaryS[0]*BaryS[0]/R,VBaryS[0]*BaryS[1]/R,VBaryS[1]])
    vect = (BaryTor-BaryS)/np.linalg.norm(BaryTor-BaryS)
    sca = np.sum(vect*nIn)
    if sca < 0:
        nIn = -nIn
    return nIn


def Calc_nInFromLin_Poly(BaryS, nIn, VBaryS):
    """ Returns nIn such that it points inward the torus """
    assert isinstance(BaryS,np.ndarray) and isinstance(nIn,np.ndarray) and BaryS.size==nIn.size==3, "Args BaryS and nIn should be ndarrays of size 3 !"
    assert isinstance(VBaryS,np.ndarray) and VBaryS.size==2, "Arg VBaryS and nIn should be ndarrays of size 2 !"
    BaryS, nIn = BaryS.flatten(), nIn.flatten()
    thet = np.arctan2(BaryS[2]-VBaryS[1],BaryS[1]-VBaryS[0])
    er = np.array([0., np.cos(thet), np.sin(thet)])
    sca = np.sum(-er*nIn)
    if sca < 0:
        nIn = -nIn
    return nIn



##################################
##################################
#       Lens-specific computation
##################################

def _Lens_set_geom_reduced(O, nIn, Rad, F1, F2=np.inf, Type='Sph'):
    if Type=='Sph':
        O = np.asarray(O).flatten()
        nIn = np.asarray(nIn).flatten()
        nIn = nIn/np.linalg.norm(nIn)
        return O, nIn, float(Rad), float(F1), float(F2)

def _Lens_set_geom_full(R1=None, R2=None, dd=None, O=None, Rad=None, nIn=None, Type='Sph'):
    if Type=='Sph':
        assert all([aa is None for aa in [R1,R2,dd]]) or all([type(aa) in [float,np.float64] for aa in [R1,R2,dd]]), "Args R1, R2 and dd must be floats !"
        Full = not R1 is None
        C1, C2, Angmax1, Angmax2 = None, None, None, None
        if Full:
            R1, R2, dd = float(R1), float(R2), float(dd)
            C1 = O - (R1-dd/2.)*nIn
            C2 = O + (R2-dd/2.)*nIn
            Angmax1 = float(np.abs(np.arcsin(Rad/R1)))
            Angmax2 = float(np.abs(np.arcsin(Rad/R2)))
        return Full, R1, R2, dd, C1, C2, Angmax1, Angmax2




"""
###############################################################################
###############################################################################
                   Detector and Aperture classes and functions
###############################################################################
"""

############################################
##### Input checking  functions
############################################


def _ApDetect_check_inputs(Id=None, Poly=None, Optics=None, Ves=None, Sino_RefPt=None, CalcEtend=None, CalcSpanImp=None, CalcCone=None, CalcPreComp=None, Calc=None, Verb=None,
            RelErr=None, dX12=None, dX12Mode=None, Ratio=None, Colis=None, LOSRef=None, Method=None,
            arrayorder=None, Clock=None, Type=None, Exp=None, Diag=None, shot=None, dtime=None, dtimeIn=None, SavePath=None):    # Used
    if not Id is None:
        assert type(Id) in [str,tfpfID], "Arg Id must be a str or a tofu.pathfile..ID object !"
    if not Poly is None:
        assert type(Poly) is dict or (hasattr(Poly,'__getitem__') and np.asarray(Poly).ndim==2 and 3 in np.asarray(Poly).shape), "Arg Poly must be a dict or an iterable with 3D cartesian coordinates of points !"
        if type(Poly) is dict:
            assert all([aa in Poly.keys() for aa in ['O','nn','Rad']]), "Arg Poly must be a dict with keys ['O','nn','Rad'] !"
            assert type(Poly['Rad']) in [float,np.float64], "Arg Poly['Rad'] must be a float !"
            assert all([hasattr(aa,'__getitem__') and np.asarray(aa).shape==(3,) for aa in [Poly['O'],Poly['nn']]]), "Args Poly['O'] and Poly['nn'] must be iterables with 3D cartesian coordinates (center and vector) !"
    if not Optics is None and not Exp is None:
        if type(Optics) is list:
            assert Exp==Optics[0].Id.Exp, "Arg Exp must be the same as the Optics[0].Id.Exp !"
        else:
            assert Exp==Optics.Id.Exp, "Arg Exp must be the same as the Optics.Id.Exp !"
    if not Ves is None and not Exp is None:
        assert Exp==Ves.Id.Exp, "Arg Exp must be the same as the Ves.Id.Exp !"
    if not Sino_RefPt is None:
        assert hasattr(Sino_RefPt,'__getitem__') and np.asarray(Sino_RefPt).ndim==1 and np.asarray(Sino_RefPt).size==2, "Arg Sino_RefPt must be an iterable with the 2D Cross-section coordinates of a reference point !"
    bools = [CalcEtend,CalcSpanImp,CalcCone,CalcPreComp,Calc,Verb,Clock,dtimeIn,Colis]
    if any([not aa is None for aa in bools]):
        assert all([aa is None or type(aa) is bool for aa in bools]), " Args [CalcEtend,CalcSpanImp,CalcCone,CalcPreComp,Calc,Verb,Clock,dtimeIn] must all be bool !"
    if not arrayorder is None:
        assert arrayorder in ['C','F'], "Arg arrayorder must be in ['C','F'] !"
    strs = [Type,Exp,Diag,SavePath,LOSRef]
    if any([not aa is None for aa in strs]):
        assert all([aa is None or type(aa) is str for aa in strs]), "Args [Type,Exp,Diag,SavePath] must all be str !"
    if not shot is None:
        assert type(shot) is int, "Arg shot must be a int !"
    if not dtime is None:
        assert type(dtime) is dtm.datetime, "Arg dtime must be a dtm.datetime !"
    floats = [RelErr,Ratio]
    if any([not aa is None for aa in floats]):
        assert all([aa is None or type(aa) is float for aa in floats]), "Args [RelErr,Ratio] must all be str !"
    if not Method is None:
        assert Method in ['quad','trapz','simps'], "Arg Method must be in ['quad','trapz','simps'] !"
    if not dX12Mode is None:
        assert dX12Mode in ['rel','abs'], "Arg dX12Mode must be in ['rel','abs'] !"



############################################
##### Checking points
############################################


def _Detect_isOnGoodSide(Pts, DP, DnIn, LPs, LnIn, NbPoly=None, Log='all'):    # Used
    assert type(Pts) is np.ndarray and Pts.ndim==2 and Pts.shape[0]==3, "Pts must be a (3,N) np.ndarray in cartesian coordinates !"
    assert NbPoly is None or type(NbPoly) is int, "Arg NbPoly must be a int indicating the number of Poly (from Detect=1 to last Apert=None) to be considered !"
    assert Log in ['each','all','any'], "Arg Log must be in ['each','all','any'] !"
    Barys, ns = [DP]+LPs, [DnIn]+LnIn
    NPolys, NP = len(Barys), Pts.shape[1]
    Sides = np.empty((NPolys,NP),dtype=bool)
    for ii in range(0,NPolys):
        Sides[ii,:] = (Pts[0,:]-Barys[ii][0])*ns[ii][0] + (Pts[1,:]-Barys[ii][1])*ns[ii][1] + (Pts[2,:]-Barys[ii][2])*ns[ii][2] > 0.
    if Log=='each':
        ind = Sides
    elif Log=='all':
        ind = np.all(Sides[:NbPoly,:],axis=0)
    elif Log=='any':
        ind = np.any(Sides[:NbPoly,:],axis=0)
    return ind



def _Detect_isInsideConeWidthLim(Pts, LOSD, LOSu, ConeWidth_k, ConeWidth_X1, ConeWidth_X2):
    assert isinstance(Pts, np.ndarray) and Pts.ndim==2 and Pts.shape[0]==3, "Arg Pts must be a (3,N) np.ndarray in cartesian coordinates !"
    e1, e2 = GG.Calc_DefaultCheck_e1e2_PLane_1D(LOSD, LOSu)
    Ss = (Pts[0,:]-LOSD[0])*LOSu[0] + (Pts[1,:]-LOSD[1])*LOSu[1] + (Pts[2,:]-LOSD[2])*LOSu[2]
    X1 = (Pts[0,:]-LOSD[0])*e1[0] + (Pts[1,:]-LOSD[1])*e1[1] + (Pts[2,:]-LOSD[2])*e1[2]
    X2 = (Pts[0,:]-LOSD[0])*e2[0] + (Pts[1,:]-LOSD[1])*e2[1] + (Pts[2,:]-LOSD[2])*e2[2]
    MinX1, MaxX1 = np.interp(Ss,ConeWidth_k,ConeWidth_X1[0,:]), np.interp(Ss,ConeWidth_k,ConeWidth_X1[1,:])
    MinX2, MaxX2 = np.interp(Ss,ConeWidth_k,ConeWidth_X2[0,:]), np.interp(Ss,ConeWidth_k,ConeWidth_X2[1,:])
    return (X1>=MinX1) & (X1<=MaxX1) & (X2>=MinX2) & (X2<=MaxX2)



############################################
##### Computing functions
############################################


def _ApDetect_set_Poly(Poly, Type=None, arrayorder='C', Clock=False, NP=100):    # Used
    # Make Poly closed, counter-clockwise, with '(cc,N)' layout and good arrayorder
    if Type=='Circ':
        BaryS = np.asarray(Poly['O']).flatten()
        Rad = float(Poly['Rad'])
        nIn = np.asarray(Poly['nIn']).flatten()
        nIn = nIn/np.linalg.norm(nIn)
        BaryP = BaryS
        Surf = np.pi*Rad**2
        Poly, NP = None, NP
    else:
        Poly = GG.PolyOrder(Poly, order=arrayorder, Clock=Clock, close=True, layout='(cc,N)', Test=True)
        assert Poly.shape[0]==3, "Arg Poly must be a 3D polygon in cartesian coordinates !"
        NP = Poly.shape[1]-1
        # Compute barycenter and vectro perpendicular to plane of polygon
        BaryP, nIn = GG.Calc_BaryNorm_3DPoly_1D(Poly)
        # Compute area and surfacic center of mass
        Poly2, P, en, e1, e2 = GG.Calc_2DPolyFrom3D_1D(Poly, P=BaryP, en=nIn)
        TorP = plg.Polygon(Poly2.T)
        Surf, BaryS = TorP.area(),  np.array(TorP.center())
        BaryS = P + e1*BaryS[0] + e2*BaryS[1]
        Rad = None
    return Poly, NP, nIn, BaryP, Surf, BaryS, Rad


def _Detect_set_LOS(Name, LSurfs, LBaryS, LnIn, LPolys, BaryS, Poly, OpType='Apert', Verb=True, Test=True):    # Used
    if Test:
        assert type(Name) is str, "Arg Name must be a str !"
        assert type(LSurfs) is list and all([type(aa) is float for aa in LSurfs]), "Arg LSurfs must be a list of floats !"
        assert type(LBaryS) is list and all([hasattr(aa,'__getitem__') for aa in LBaryS]), "Arg LBaryS must be a list of iterables !"
        assert type(LnIn) is list and all([hasattr(aa,'__getitem__') for aa in LnIn]), "Arg LnIn must be a list of floats !"
        assert type(LPolys) is list and all([hasattr(aa,'__getitem__') for aa in LPolys]), "Arg LPolys must be a list of floats !"
        assert type(BaryS) is np.ndarray and BaryS.shape==(3,), "Arg BaryS must be a (3,) np.ndarray !"

    if Verb:
        print "    "+Name+" : Computing LOS..."

    if OpType=="Lens":
        LOS_ApertPolyInt = LPolys[0]
        LOS_ApertPolyInt_S = LSurfs[0]
        LOS_ApertPolyInt_BaryS = LBaryS[0]
        du = LnIn[0]
    else:
        # Select the plane (smallest surface) and get local coordinate system
        ind = np.nanargmin(LSurfs)
        P, nP = LBaryS[ind], LnIn[ind]
        e1, e2 = GG.Calc_DefaultCheck_e1e2_PLane_1D(P, nP)
        # Get projections of all polygons on this plane, as seen from BaryS
        try:
            PolyInt = GG.Calc_PolysProjPlanePoint_Fast(LPolys, BaryS, P, nP, e1, e2)
        except Exception:
            PolyInt = GG.Calc_PolysProjPlanePoint(LPolys,BaryS,P,nP,e1P=e1,e2P=e2,Test=True)[1]

        # If several polygons, compute their intersection
        if type(PolyInt) is list:
            PolyInt = GG.Calc_PolyInterLPoly2D(PolyInt, Test=True)
        # If remaining polygon has non-zero area, a LOS can be computed from BaryS towards the plasma, through the surfacic center of mass of this intersection polygon
        assert PolyInt.size>0, "!!!!!! Detect "+ Name +" : calculation of LOS impossible !!!!!!!"
        LOS_ApertPolyInt = np.array([P[0]+e1[0]*PolyInt[0,:]+e2[0]*PolyInt[1,:],P[1]+e1[1]*PolyInt[0,:]+e2[1]*PolyInt[1,:],P[2]+e1[2]*PolyInt[0,:]+e2[2]*PolyInt[1,:]])
        PolyInt = plg.Polygon(PolyInt.T)
        LOS_ApertPolyInt_S, LOS_ApertPolyInt_BaryS = PolyInt.area(), np.array([PolyInt.center()]).T
        B = GG.Calc_3DPolyfrom2D_1D(LOS_ApertPolyInt_BaryS.reshape((2,1)), P, nP, e1, e2, Test=True).flatten()
        du = (B-BaryS)/np.linalg.norm(B-BaryS)
        LOS_ApertPolyInt_BaryS = B

    return LOS_ApertPolyInt, LOS_ApertPolyInt_S, LOS_ApertPolyInt_BaryS, du



def _Detect_SAngVect_Points(Pts, DPoly=None, DBaryS=None, DnIn=None, LOBaryS=None, LOnIns=None, LOPolys=None, SAngPlane=None, Lens_ConeTip=None,Lens_ConeHalfAng=None, RadL=None, RadD=None, F1=None, thet=np.linspace(0.,2.*np.pi,100),
        OpType='Apert', VPoly=None, VVin=None, DLong=None, VType='Tor', Cone_PolyCrossbis=None, Cone_PolyHorbis=None, TorAngRef=None, Colis=True, Test=True):   # Used
    """
    Not usable for etendue because uses Cone_Poly (not yet computed for etendue) !
    """
    if Test:
        assert type(Pts) is np.ndarray and Pts.ndim==2 and Pts.shape[0]==3, "Arg Pts must be an np.ndarray with the 3D cartesian coordinates of N>1 points"

    NP = Pts.shape[1]
    SAng, Vect = np.zeros((NP,),dtype=float), np.nan*np.ones((3,NP),dtype=float)
    ind = _Detect_isOnGoodSide(Pts, DBaryS, DnIn, LOBaryS, LOnIns, NbPoly=None, Log='all')
    if Colis:
        ind_Cone = _Detect_isInside(Cone_PolyCrossbis, Cone_PolyHorbis, Pts, In='(X,Y,Z)', VType=VType, TorAngRef=TorAngRef, Test=Test)   # Cone Poly
        ind = ind & ind_Cone

    if np.any(ind):
        Ptsind = Pts[:,ind] if ind.sum()>1 else Pts[:,ind].reshape((Pts.shape[0],1))
        if OpType=='Apert':
            SAng[ind], Vect[:,ind] = GG.Calc_SAngVect_LPolysPoints_Flex([DPoly]+LOPolys, Ptsind,  SAngPlane[0], SAngPlane[1], SAngPlane[2], SAngPlane[3])
        else:

            SAng[ind], Vect[:,ind] = GG.Calc_SAngVect_LPolysPoints_Flex_Lens(LOBaryS[0][0],LOBaryS[0][1],LOBaryS[0][2], Lens_ConeTip[0],Lens_ConeTip[1],Lens_ConeTip[2], LOnIns[0][0],LOnIns[0][1],LOnIns[0][2],
                                                                             np.ascontiguousarray(Ptsind[0,:]), np.ascontiguousarray(Ptsind[1,:]), np.ascontiguousarray(Ptsind[2,:]), RadL, RadD, F1, np.tan(Lens_ConeHalfAng),
                                                                             np.ascontiguousarray(LOPolys[0][0,:]), np.ascontiguousarray(LOPolys[0][1,:]), np.ascontiguousarray(LOPolys[0][2,:]), thet=thet, VectReturn=True)
        indPos = SAng>0.
        if Colis and np.any(indPos):
            PtsindPos = Pts[:,indPos] if indPos.sum()>1 else Pts[:,indPos].reshape((Pts.shape[0],1))
            indC = GG.Calc_InOut_LOS_Colis(DBaryS, PtsindPos, VPoly, VVin) if VType=='Tor' else GG.Calc_InOut_LOS_Colis_Lin(DBaryS, PtsindPos, VPoly, VVin, DLong)
            indnul = indPos.nonzero()[0]
            SAng[indnul[~indC]] = 0.
            Vect[:,indnul[~indC]] = np.nan
    return SAng, Vect



def Calc_Etendue_PlaneLOS(Ps, nPs, DPoly, DBaryS, DnIn, LOPolys, LOnIns, LOSurfs, LOBaryS, SAngPlane, VPoly, VVin, DLong=None,
        Lens_ConeTip=None, Lens_ConeHalfAng=None, RadL=None, RadD=None, F1=None,
        OpType='Apert', VType='Tor', Mode='quad', epsrel=TFD.DetEtendepsrel, dX12=TFD.DetEtenddX12, dX12Mode=TFD.DetEtenddX12Mode, e1=None,e2=None, Ratio=0.02, Colis=TFD.DetCalcEtendColis, Details=False, Test=True):    # Used
    """ Computes the Etendue of a Detect with Apert on N given planes parametrised by (P,nP) with two possible methods : discrete (simps) or (dblquad)

    Inputs :
        D           A Detect instance
        P           A (3,N) np.ndarray corresponding to the cartesian coordinates of N points (one for each plane)
        nP          A (3,N) np.ndarray corresponding to the cartesian coordinates of N normal vectors (one for each plane)
        Mode        'dblquad' (default) or 'simps' or 'trapz' to choose the scipy computation method of the integral
        epsrel      For 'dblquad', a positive float defining the relative tolerance allowed
        dX12        For 'simps'/'trapz', a list of 2 floats defining the resolution of the sampling in X1 and X2
        dX12Mode    For 'simps'/'trapz', 'rel' or 'abs', if 'rel' the resolution dX12 is in dimensionless units in [0;1] (hence a value of 0.1 means 10 discretisation points between the extremes), if 'abs' dX12 is in meters
        e1          The e1 unitary vector (corresponds to X1), if unspecified, the optimal vector is used
        e2          The e2 unitary vector (corresponds to X2), must be orthogonal to e1, if unspecified, the optimal vector is used
        Ratio       A float specifying the relative margin to be taken for integration boundaries
    Outputs :
        Etendue     A positive float, the calculated etendue (in )
        e1          The e1 unitary vector (corresponds to X1) used
        e2          The e2 unitary vector (corresponds to X2) used
        err         An array containing error flags for the integral computation
    """

    if Test:
        assert isinstance(Ps,np.ndarray) and Ps.ndim==2 and Ps.shape[0]==3, "Arg Ps should be a (3,N) ndarray !"
        assert isinstance(nPs,np.ndarray) and nPs.shape==Ps.shape, "Arg nPs should be a (3,N) ndarray !"
        assert Mode in ['quad','simps','trapz'], "Arg Mode should be 'quad' or 'simps' or 'trapz' !"

    nPlans = nPs.shape[1]
    nPs = nPs/np.tile(np.sqrt(np.sum(nPs**2,axis=0)),(3,1))
    e1, e2 = GG.Calc_DefaultCheck_e1e2_PLanes_2D(Ps, nPs, e1, e2)

    if OpType=='Apert':
        LnPtemp = np.asarray(LOnIns)*np.tile(LOSurfs,(3,1)).T
        Out = Calc_ViewConePointsMinMax_PlanesDetectApert_2Steps(DPoly, LOPolys, LnPtemp, LOSurfs, LOBaryS[0], Ps, nPs, e1=e1, e2=e2, Test=False)
    elif OpType=='Lens':
        Out = Calc_ViewConePointsMinMax_PlanesDetectLens(LOPolys[0], Lens_ConeTip, Ps, nPs, e1=e1, e2=e2, Test=False)

    MinX1, MinX2, MaxX1, MaxX2 = Out[0], Out[1], Out[2], Out[3]
    MinX1 = MinX1 - Ratio*(MaxX1-MinX1)
    MaxX1 = MaxX1 + Ratio*(MaxX1-MinX1)
    MinX2 = MinX2 - Ratio*(MaxX2-MinX2)
    MaxX2 = MaxX2 + Ratio*(MaxX2-MinX2)

    Etend = np.nan*np.ones((nPlans,))
    err = np.nan*np.ones((nPlans,))
    NOPoly = len(LOPolys)

    PBary, nPtemp, e1bis, e2bis = SAngPlane
    LPolys = [DPoly] + LOPolys

    if Mode=='quad' and not Details:
        LBaryS = np.ascontiguousarray(np.vstack(tuple([DBaryS]+LOBaryS)).T)
        LnIn = np.ascontiguousarray(np.vstack(tuple([DnIn]+LOnIns)).T)
        nPolys = NOPoly+1
        for ii in range(0,nPlans):
            if not np.any(np.isnan(np.array([MinX1[ii], MinX2[ii], MaxX1[ii], MaxX2[ii]]))):
                def X2Min(x):
                    return MinX2[ii]
                def X2Max(x):
                    return MaxX2[ii]
                if OpType=='Apert':
                    FuncSA = GG.FuncSAquad_Apert(LPolys, nPtemp, e1bis, e2bis, Ps[:,ii], e1[:,ii], e2[:,ii], LBaryS, LnIn, nPolys, PBary, VPoly, VVin, DLong=DLong, VType=VType, Colis=Colis)
                else:
                    thet = np.linspace(0.,2.*np.pi,LOPolys[0].shape[1])
                    FuncSA = GG.FuncSAquad_Lens(LOBaryS[0], Lens_ConeTip, Ps[:,ii], e1[:,ii], e2[:,ii], LBaryS, LnIn, nPolys,
                            RadL, RadD, F1, np.tan(Lens_ConeHalfAng), LOPolys[0], VPoly, VVin, DLong=DLong, VType=VType, thet=thet, VectReturn=False, Colis=Colis)

                Etend[ii], err[ii] = GG.dblquad_custom(FuncSA, MinX1[ii], MaxX1[ii], X2Min, X2Max, epsabs=0., epsrel=epsrel)

    else:
        if dX12Mode=='rel':
            NumX1 = [int(1./dX12[0])]*MinX1.size
            NumX2 = [int(1./dX12[1])]*MinX2.size
        else:
            NumX1 = [int((MaxX1[zz]-MinX1[zz])/dX12[0]) for zz in range(0,nPlans)]
            NumX2 = [int((MaxX2[zz]-MinX2[zz])/dX12[1]) for zz in range(0,nPlans)]
        NumP = [NumX1[zz]*NumX2[zz] for zz in range(0,nPlans)]
        for ii in range(0,nPlans):
            SA = np.zeros((NumP[ii],))
            X1 = np.linspace(MinX1[ii],MaxX1[ii],NumX1[ii],endpoint=True).reshape((NumX1[ii],1))
            X2 = np.linspace(MinX2[ii],MaxX2[ii],NumX2[ii],endpoint=True).reshape((1,NumX2[ii]))
            x1 = np.dot(X1,np.ones((1,NumX2[ii]))).reshape((1,NumP[ii]))
            x2 = np.dot(np.ones((NumX1[ii],1)),X2).reshape((1,NumP[ii]))

            Pps = np.dot(Ps[:,ii:ii+1],np.ones((1,NumP[ii]))) + np.dot(e1[:,ii:ii+1],x1) + np.dot(e2[:,ii:ii+1],x2)
            ind = _Detect_isOnGoodSide(Pps, DBaryS, DnIn, LOBaryS, LOnIns, NbPoly=None, Log='all')
            if np.any(ind):
                Ppsind = Pps[:,ind] if ind.sum()>1 else Pps[:,ind].reshape((Pps.shape[0],1))
                if OpType=='Apert':
                    SA[ind] = GG.Calc_SAngVect_LPolysPoints_Flex(LPolys, Ppsind,  PBary, nPtemp, e1bis, e2bis)[0]      # Cython
                else:
                    SA[ind] = GG.Calc_SAngVect_LPolysPoints_Flex_Lens(LOBaryS[0][0],LOBaryS[0][1],LOBaryS[0][2], Lens_ConeTip[0],Lens_ConeTip[1],Lens_ConeTip[2], LOnIns[0][0],LOnIns[0][1],LOnIns[0][2],
                                                                      np.ascontiguousarray(Ppsind[0,:]),np.ascontiguousarray(Ppsind[1,:]),np.ascontiguousarray(Ppsind[2,:]), RadL, RadD, F1, np.tan(Lens_ConeHalfAng),
                                                                      np.ascontiguousarray(LOPolys[0][0,:]),np.ascontiguousarray(LOPolys[0][1,:]),np.ascontiguousarray(LOPolys[0][2,:]),
                                                                      thet=np.linspace(0.,2.*np.pi,LOPolys[0].shape[1]), VectReturn=False)
            indPos = SA>0.
            if Colis and np.any(indPos):
                PpsindPos = Pps[:,indPos] if indPos.sum()>1 else Pps[:,indPos].reshape((Pps.shape[0],1))
                indC = GG.Calc_InOut_LOS_Colis(PBary, PpsindPos, VPoly, VVin) if VType=='Tor' else GG.Calc_InOut_LOS_Colis_Lin(PBary, PpsindPos, VPoly, VVin, DLong)
                indnul = indPos.nonzero()[0]
                SA[indnul[~indC]] = 0.

            if Mode=='simps':
                Etend[ii] = GG.dblsimps_custom(SA.reshape((NumX1[ii],NumX2[ii])),x=X1.flatten(),y=X2.flatten())
            elif Mode=='trapz':
                Etend[ii] = GG.dbltrapz_custom(SA.reshape((NumX1[ii],NumX2[ii])),x=X1.flatten(),y=X2.flatten())
    if Details:
        return Etend, e1, e2, err, SA, X1, X2, NumX1, NumX2
    else:
        return Etend, e1, e2, err





def Calc_SpanImpBoth_2Steps(DPoly, DNP, DBaryS, LOPolys, LOBaryS, LOSD, LOSu, RefPt, P, nP, VPoly, VVin, DLong=None, VType='Tor', e1=None,e2=None, OpType='Apert', Lens_ConeTip=None, NEdge=TFD.DetSpanNEdge, NRad=TFD.DetSpanNRad, Test=True):    # Used
    """ Computes the span in (R,Theta,Z,k) coordinates of the viewing cone of a detector by sampling it with a multitude of LOS

    Inputs :
        RefPt       A (2,) np.ndarray indicating the (R,Z) coordinates to be used for computing the poloidal projection of the viewing span in projection space
        P           A (3,) np.ndarray corresponding to the cartesian coordinates of a point on a plane
        nP          A (3,) np.ndarray corresponding to the cartesian coordinates of a normal vector to a plane
        e1          The e1 unitary vector (corresponds to X1), automatically determined if unspecified, the optimal vector is used
        e2          The e2 unitary vector (corresponds to X2), must be orthogonal to e1, automatically determined if if unspecified
        NEdge       A int indicating in how many segments each edge of a polygon must be divided
        NRad        A int indicating in how many segments each radius of a polygon must be divided
        Test        A boolean to know if tests for inputs should be performed or not, if False make sure that all inputs are perfect !
    Outputs :
        (MinR,MaxR)         Tuple containing the minimum and maximum values of R for the detector whole viewing cone
        (MinTheta,MaxTheta) Tuple containing the minimum and maximum values of Theta for the detector whole viewing cone
        (MinZ,MaxZ)         Tuple containing the minimum and maximum values of Z for the detector whole viewing cone
        (Mink,Maxk)         Tuple containing the minimum and maximum values of k (i.e.: length along the LOS) for the detector whole viewing cone
        PProj               (2,N) np.ndarray representing the (X1,X2) coordinates of the projected polygon on plane (P,nP,e1,e2)
        NEdge               int, value of NEdge used
        NRad                int, value of NRad used
    """

    if Test:
        assert all([vv is None or (isinstance(vv,np.ndarray) and vv.shape==(3,)) for vv in [P,nP,e1,e2]]), "Arg [P,nP,e1,e2] should be (3,) ndarrays !"
        assert all([type(nn) is int and nn>=0 for nn in [NEdge,NRad]]), "Args [NEdge,NRad] must be a positive ints !"
    if e1 is None or e2 is None:
        e1, e2 = GG.Calc_DefaultCheck_e1e2_PLane_1D(P, nP)

    def getallpointsfromPoly(Poly, PolyN, BaryS, NEdge, NRad):
        PP = Poly[:,:PolyN]
        if NEdge>0:
            aa = np.linspace(1,NEdge,NEdge)/(NEdge+1)
            Extra = []
            for ii in range(0,PolyN):
                Extra += [Poly[:,ii:ii+1]*np.ones((1,NEdge)) + (Poly[:,ii+1:ii+2]-Poly[:,ii:ii+1])*aa.reshape((1,NEdge))]
            Extra = np.concatenate(tuple(Extra),axis=1)
        PP = np.concatenate((PP,Extra),axis=1)
        if NRad>0:
            aa = np.linspace(1,NRad,NRad)/(NRad+1)
            Extra = []
            for ii in range(0,PP.shape[1]):
                Extra += [BaryS*np.ones((1,NRad)) + (PP[:,ii:ii+1]-BaryS)*aa.reshape((1,NRad))]
            Extra = np.concatenate(tuple(Extra),axis=1)
        PP = np.concatenate((PP, Extra, BaryS),axis=1)
        return PP

    if OpType=='Apert':
        Corners = getallpointsfromPoly(DPoly, DNP, DBaryS.reshape((3,1)), NEdge, NRad)
    elif OpType=='Lens':
        Corners = Lens_ConeTip.reshape((3,1))
    NCorners = Corners.shape[1]
    NPoly = len(LOPolys)

    SIn, SOut, RMin, pp, TT = [], [], [], [], []
    if NPoly==1:
        PolyRef = LOPolys[0]
        PolyRef = getallpointsfromPoly(PolyRef, PolyRef.shape[1]-1, LOBaryS[0].reshape((3,1)), NEdge, NRad)
        NC = PolyRef.shape[1]*Corners.shape[1]
        Ds = np.resize(Corners.T,(PolyRef.shape[1], Corners.shape[1], 3)).T.reshape((3,NC))
        Lus = np.resize(PolyRef.T,(NC,3)).T
        Lus = Lus - Ds
        Lus = Lus/(np.ones((3,1)).dot(np.sqrt(np.sum(Lus**2,axis=0,keepdims=True))))
        if VType=='Tor':
            SIn, SOut = GG.Calc_InOut_LOS_PIO(Ds, Lus, VPoly, VVin)
        elif VType=='Lin':
            SIn, SOut = GG.Calc_InOut_LOS_PIO_Lin(Ds, Lus, VPoly, VVin, DLong)

        indnonan = ~np.any(np.isnan(SOut),axis=0)
        Nnan = np.sum(indnonan)
        if Nnan==1:
            Ds, Lus = np.ascontiguousarray(Ds[:,indnonan].reshape((3,Nnan))), np.ascontiguousarray(Lus[:,indnonan].reshape((3,Nnan)))
            RMin = GG.Calc_PolProj_LOS_cy(Ds, Lus, kmax=np.sum((SOut[:,indnonan]-Ds)*Lus,axis=0))[1]
        else:
            Ds, Lus = np.ascontiguousarray(Ds[:,indnonan]), np.ascontiguousarray(Lus[:,indnonan])
            RMin = GG.Calc_PolProj_LOS_cy(Ds, Lus, kmax=np.sum((SOut[:,indnonan]-Ds)*Lus,axis=0))[1]

        kOut = np.sum((SOut[:,indnonan]-Ds)*Lus,axis=0)
        if VType=='Tor':
            AA = GG.Calc_Impact_LineMulti(Ds, Lus, RefPt, kOut)
        elif VType=='Lin':
            AA = GG.Calc_Impact_LineMulti_Lin(Ds, Lus, RefPt, kOut)
        pp, TT = list(AA[4]), list(AA[5])

    else:
        #t1, t2, t3, t4, t5, t6, t7 = 0., 0., 0., 0., 0., 0., 0. # DB
        for ii in range(NCorners):
            #tt = dtm.datetime.now() # DB
            Out = GG.Calc_PolysProjPlanesPoint(LOPolys, Corners[:,ii:ii+1], P.reshape((3,1)), nP.reshape((3,1)), e1P=e1.reshape((3,1)), e2P=e2.reshape((3,1)), Test=True)
            PolyRef = [np.concatenate((Out[3][jj],Out[4][jj]),axis=0) for jj in range(0,NPoly)]
            PolyRef = GG.Calc_PolyInterLPoly2D(PolyRef,Test=False)
            #t1 += (dtm.datetime.now() - tt).total_seconds() # DB

            if PolyRef.shape[1]>0:
                #tt = dtm.datetime.now() # DB
                BaryS = plg.Polygon(PolyRef.T).center()
                BaryS = P + e1*BaryS[0] + e2*BaryS[1]
                PolyRef = GG.Calc_3DPolyfrom2D_1D(PolyRef, P, nP, e1, e2, Test=True)
                #t2 += (dtm.datetime.now()-tt).total_seconds() # DB
                #tt = dtm.datetime.now() # DB
                PolyRef = getallpointsfromPoly(PolyRef, PolyRef.shape[1]-1, BaryS.reshape((3,1)), NEdge, NRad)
                Ds = Corners[:,ii:ii+1].dot(np.ones((1,PolyRef.shape[1])))
                Lus = PolyRef - Ds
                Lus = Lus/(np.ones((3,1)).dot(np.sqrt(np.sum(Lus**2,axis=0,keepdims=True))))
                #t3 += (dtm.datetime.now()-tt).total_seconds() # DB
                #tt = dtm.datetime.now() # DB
                if VType=='Tor':
                    Sin, Sout = GG.Calc_InOut_LOS_PIO(Ds, Lus, VPoly, VVin)
                elif VType=='Lin':
                    Sin, Sout = GG.Calc_InOut_LOS_PIO_Lin(Ds, Lus, VPoly, VVin, DLong)
                #t4 += (dtm.datetime.now()-tt).total_seconds() # DB
                #tt = dtm.datetime.now() # DB
                indnonan = ~np.any(np.isnan(Sout),axis=0)
                Nnan = np.sum(indnonan)
                #t5 += (dtm.datetime.now()-tt).total_seconds() # DB
                #tt = dtm.datetime.now() # DB
                if Nnan==1:
                    Ds, Lus = np.ascontiguousarray(Ds[:,indnonan].reshape((3,Nnan))), np.ascontiguousarray(Lus[:,indnonan].reshape((3,Nnan)))
                    RMin.append(GG.Calc_PolProj_LOS_cy(Ds, Lus, kmax=np.sum((Sout[:,indnonan]-Ds)*Lus,axis=0))[1])
                else:
                    Ds, Lus = np.ascontiguousarray(Ds[:,indnonan]), np.ascontiguousarray(Lus[:,indnonan])
                    RMin.append(GG.Calc_PolProj_LOS_cy(Ds, Lus, kmax=np.sum((Sout[:,indnonan]-Ds)*Lus,axis=0))[1])    #  10x faster than Python list comprehension
                SIn.append(Sin)
                SOut.append(Sout)

                kOut = np.sum((Sout[:,indnonan]-Ds)*Lus,axis=0)
                #t6 += (dtm.datetime.now()-tt).total_seconds() # DB
                #tt = dtm.datetime.now() # DB
                if VType=='Tor':
                    AA = GG.Calc_Impact_LineMulti(Ds, Lus, RefPt, kOut)
                elif VType=='Lin':
                    AA = GG.Calc_Impact_LineMulti_Lin(Ds, Lus, RefPt, kOut)
                pp, TT = list(AA[4]), list(AA[5])
                #t7 += (dtm.datetime.now()-tt).total_seconds() # DB

        SIn = np.concatenate(tuple(SIn),axis=1)
        SOut = np.concatenate(tuple(SOut),axis=1)
        RMin = np.concatenate(tuple(RMin))
    if SIn.size==0:
        sca = [(bb-LOSD)*LOSu for bb in LOBaryS]
        #sca = [(aa.BaryS-D.LOS[LOSRef]['LOS'].D)*D.LOS[LOSRef]['LOS'].u for aa in D.LApert]
        SIn = LOBaryS[np.argmax(sca)].reshape((3,1))
        #SIn = D.LApert[np.argmax(sca)].BaryS.reshape((3,1))
    AllS = np.concatenate((SIn,SOut), axis=1)

    Z = AllS[2,:]
    k = (AllS[0,:]-LOSD[0])*LOSu[0] + (AllS[1,:]-LOSD[1])*LOSu[1] + (AllS[2,:]-LOSD[2])*LOSu[2]
    pp, TT = np.array(pp), np.array(TT)
    PProj = ConvexHull(np.array([TT,pp]).T)
    PProj = np.array([TT[PProj.vertices],pp[PProj.vertices]])       # .vertices ony availabe from scipy 0.13.0 !
    PDiff = PProj-np.mean(PProj,axis=1,keepdims=True).dot(np.ones((1,PProj.shape[1])))
    ind = np.argsort(np.arctan2(PDiff[1,:],PDiff[0,:]))
    PProj = PProj[:,ind]
    #print t1, t2, t3, t4, t5, t6, t7 # DB
    if VType=='Tor':
        R = np.concatenate((np.sqrt(AllS[0,:]**2 + AllS[1,:]**2), RMin))
        Theta = np.arctan2(AllS[1,:], AllS[0,:])
        return np.array([np.nanmin(R),np.nanmax(R)]), np.array([np.nanmin(Theta),np.nanmax(Theta)]), np.array([np.nanmin(Z),np.nanmax(Z)]), np.array([np.nanmin(k),np.nanmax(k)]), np.concatenate((PProj,PProj[:,0:1]),axis=1), NEdge, NRad
    elif VType=='Lin':
        return np.array([np.nanmin(AllS[0,:]),np.nanmax(AllS[0,:])]), np.array([np.nanmin(AllS[1,:]),np.nanmax(AllS[1,:])]), np.array([np.nanmin(Z),np.nanmax(Z)]), np.array([np.nanmin(k),np.nanmax(k)]), np.concatenate((PProj,PProj[:,0:1]),axis=1), NEdge, NRad




def Calc_ViewConePointsMinMax_PlanesDetectApert_2Steps(Poly, LPolys, LnPs, LSurfs, BaryS, Ps, nPs, e1=None, e2=None, Test=True):    # Used
    """ Compute the Points which are projections of each Detect.Poly corners through all its apertures on a set of N planes defined by (P,nP)

    Inputs :
        D           A Detect instance
        P           A (3,N) np.ndarray corresponding to the cartesian coordinates of N points (one for each plane)
        nP          A (3,N) np.ndarray corresponding to the cartesian coordinates of N normal vectors (one for each plane)
        e1          The e1 unitary vector (corresponds to X1), if unspecified, the optimal vector is used
        e2          The e2 unitary vector (corresponds to X2), must be orthogonal to e1, if unspecified, the optimal vector is used
        Test        A boolean to know if tests for inputs should be performed or not, if False make sure that all inputs are perfect !
    Outputs :
        MinX1       A (NPlans,1) np.ndarray containing the minimum value of X1 (cf. e1) of the intersection of the projected polygons (on each plan for each corner)
        MinX2       A (NPlans,1) np.ndarray containing the minimum value of X2 (cf. e2) of the intersection of the projected polygons (on each plan for each corner)
        MaxX1       A (NPlans,1) np.ndarray containing the maximum value of X1 (cf. e1) of the intersection of the projected polygons (on each plan for each corner)
        MaxX2       A (NPlans,1) np.ndarray containing the maximum value of X2 (cf. e2) of the intersection of the projected polygons (on each plan for each corner)
    """

    if Test:
        assert isinstance(Ps,np.ndarray) and Ps.shape[0] == 3, "Arg Ps should be a (3,N) ndarray !"
        assert isinstance(nPs,np.ndarray) and nPs.shape == Ps.shape, "Arg nPs should be a (3,N) ndarray !"
        assert e1 is None or (isinstance(e1,np.ndarray) and e1.shape == Ps.shape), "Arg e1 should be a (3,N) ndarray !"
        assert e2 is None or (isinstance(e2,np.ndarray) and e2.shape == Ps.shape), "Arg e2 should be a (3,N) ndarray !"
    e1, e2 = GG.Calc_DefaultCheck_e1e2_PLanes_2D(Ps, nPs, e1, e2)
    Corners = Poly[:,:-1]
    NCorners, NPlans = Corners.shape[1], Ps.shape[1]
    MinX1, MinX2, MaxX1, MaxX2 = np.nan*np.ones((NPlans,NCorners)), np.nan*np.ones((NPlans,NCorners)), np.nan*np.ones((NPlans,NCorners)), np.nan*np.ones((NPlans,NCorners))
    #LPolys = [D.LApert[jj].Poly for jj in range(len(D.LApert))]
    NPoly = len(LPolys)

    if NPoly==1:
        for ii in range(NCorners):
            Out = GG.Calc_PolysProjPlanesPoint(LPolys, Corners[:,ii:ii+1], Ps, nPs, e1P=e1, e2P=e2, Test=False)
            MinX1[:,ii] = np.min(Out[3],axis=1)
            MinX2[:,ii] = np.min(Out[4],axis=1)
            MaxX1[:,ii] = np.max(Out[3],axis=1)
            MaxX2[:,ii] = np.max(Out[4],axis=1)
    else:
        #nPtemp = [D.LApert[ii].nIn*D.LApert[ii].Surf for ii in range(0,NPoly)]
        nPtemp = LnPs
        #S = [D.LApert[ii].Surf for ii in range(0,NPoly)]
        S = LSurfs
        nPtemp = np.sum(np.array(nPtemp).T.reshape((3,NPoly)), axis=1, keepdims=False)/np.sum(np.array(S))
        nPtemp = nPtemp/np.linalg.norm(nPtemp)
        #Ptemp = D.LApert[0].BaryS
        Ptemp = BaryS
        e1temp, e2temp = GG.Calc_DefaultCheck_e1e2_PLane_1D(Ptemp, nPtemp)
        for ii in range(NCorners):
            Out = GG.Calc_PolysProjPlanesPoint(LPolys, Corners[:,ii:ii+1], Ptemp.reshape((3,1)), nPtemp.reshape((3,1)), e1P=e1temp.reshape((3,1)), e2P=e2temp.reshape((3,1)), Test=False)
            PolyRef = [np.concatenate((Out[3][jj],Out[4][jj]),axis=0) for jj in range(0,NPoly)]
            PolyRef = GG.Calc_PolyInterLPoly2D(PolyRef,Test=False)
            if PolyRef.shape[1]>0:
                PolyRef = GG.Calc_3DPolyfrom2D_1D(PolyRef, Ptemp, nPtemp, e1temp, e2temp, Test=False)
                Out = GG.Calc_PolysProjPlanesPoint(PolyRef, Corners[:,ii:ii+1], Ps, nPs, e1P=e1,e2P=e2,Test=False)
                MinX1[:,ii] = np.min(Out[3],axis=1)
                MinX2[:,ii] = np.min(Out[4],axis=1)
                MaxX1[:,ii] = np.max(Out[3],axis=1)
                MaxX2[:,ii] = np.max(Out[4],axis=1)
    assert ~np.any(np.isnan(MinX1)) and ~np.any(np.isnan(MinX2)) and ~np.any(np.isnan(MaxX1)) and ~np.any(np.isnan(MaxX2))
    return np.nanmin(MinX1,axis=1), np.nanmin(MinX2,axis=1), np.nanmax(MaxX1,axis=1), np.nanmax(MaxX2,axis=1), e1, e2



def Calc_ViewConePointsMinMax_PlanesDetectLens(PolyL, ConeTip, Ps, nPs, e1=None, e2=None, Test=True):    # Used
    """ Compute the Points which are projections of each Detect.Poly corners through all its apertures on a set of N planes defined by (P,nP)

    Inputs :
        D           A Detect instance
        P           A (3,N) np.ndarray corresponding to the cartesian coordinates of N points (one for each plane)
        nP          A (3,N) np.ndarray corresponding to the cartesian coordinates of N normal vectors (one for each plane)
        e1          The e1 unitary vector (corresponds to X1), if unspecified, the optimal vector is used
        e2          The e2 unitary vector (corresponds to X2), must be orthogonal to e1, if unspecified, the optimal vector is used
        Test        A boolean to know if tests for inputs should be performed or not, if False make sure that all inputs are perfect !
    Outputs :
        MinX1       A (NPlans,1) np.ndarray containing the minimum value of X1 (cf. e1) of the intersection of the projected polygons (on each plan for each corner)
        MinX2       A (NPlans,1) np.ndarray containing the minimum value of X2 (cf. e2) of the intersection of the projected polygons (on each plan for each corner)
        MaxX1       A (NPlans,1) np.ndarray containing the maximum value of X1 (cf. e1) of the intersection of the projected polygons (on each plan for each corner)
        MaxX2       A (NPlans,1) np.ndarray containing the maximum value of X2 (cf. e2) of the intersection of the projected polygons (on each plan for each corner)
    """

    if Test:
        assert isinstance(Ps,np.ndarray) and Ps.shape[0] == 3, "Arg Ps should be a (3,N) ndarray !"
        assert isinstance(nPs,np.ndarray) and nPs.shape == Ps.shape, "Arg nPs should be a (3,N) ndarray !"
        assert e1 is None or (isinstance(e1,np.ndarray) and e1.shape == Ps.shape), "Arg e1 should be a (3,N) ndarray !"
        assert e2 is None or (isinstance(e2,np.ndarray) and e2.shape == Ps.shape), "Arg e2 should be a (3,N) ndarray !"
    e1, e2 = GG.Calc_DefaultCheck_e1e2_PLanes_2D(Ps, nPs, e1, e2)

    NPlans = Ps.shape[1]
    Out = GG.Calc_PolysProjPlanesPoint(PolyL, ConeTip, Ps, nPs, e1P=e1, e2P=e2, Test=False)
    MinX1 = np.nanmin(Out[3],axis=1)
    MinX2 = np.nanmin(Out[4],axis=1)
    MaxX1 = np.nanmax(Out[3],axis=1)
    MaxX2 = np.nanmax(Out[4],axis=1)

    assert ~np.any(np.isnan(MinX1)) and ~np.any(np.isnan(MinX2)) and ~np.any(np.isnan(MaxX1)) and ~np.any(np.isnan(MaxX2))
    return MinX1, MinX2, MaxX1, MaxX2, e1, e2




def _get_CrossHorMesh(SingPoints=None, LSpan_R=None, LSpan_Theta=None, LSpan_X=None, LSpan_Y=None, LSpan_Z=None,
        DX=TFD.DetConeDX, DY=TFD.DetConeDRY, DR=TFD.DetConeDRY, DTheta=TFD.DetConeDTheta, DZ=TFD.DetConeDZ, VType='Tor', Proj='Cross', ReturnPts=True):
    """  """
    assert SingPoints is None or type(SingPoints) is np.ndarray and SingPoints.ndim in [1,2] and 3 in SingPoints.shape, "Arg SingPoints must contain the 3D cartesian coordiantes of Singular points !"

    # Create all points
    LSpan_R = LSpan_R if type(LSpan_R) is list else [LSpan_R]
    LSpan_Theta = LSpan_Theta if type(LSpan_Theta) is list else [LSpan_Theta]
    LSpan_X = LSpan_X if type(LSpan_X) is list else [LSpan_X]
    LSpan_Y = LSpan_Y if type(LSpan_Y) is list else [LSpan_Y]
    LSpan_Z = LSpan_Z if type(LSpan_Z) is list else [LSpan_Z]

    Span_Z = [min([sr[0] for sr in LSpan_Z]), max([sr[1] for sr in LSpan_Z])]
    if SingPoints is None:
        SingPoints = SingPoints.reshape((3,1)) if SingPoints.ndim==1 else SingPoints
        SingPoints = SingPoints if SingPoints.shape[0]==3 else SingPoints.T

    Z = np.linspace(Span_Z[0],Span_Z[1],np.ceil((Span_Z[1]-Span_Z[0])/DZ))

    Pts, Out = None, None
    if VType=='Tor':
        Span_R = [min([sr[0] for sr in LSpan_R]), max([sr[1] for sr in LSpan_R])]
        Span_Theta = [min([sr[0] for sr in LSpan_Theta]), max([sr[1] for sr in LSpan_Theta])]
        R = np.linspace(Span_R[0],Span_R[1],np.ceil((Span_R[1]-Span_R[0])/DR))
        Theta = np.linspace(Span_Theta[0], Span_Theta[1], np.ceil(np.abs(Span_Theta[1]-Span_Theta[0])/DTheta))
        if not SingPoints is None:
            RSing = np.hypot(SingPoints[0,:],SingPoints[1,:])
            ThetaSing = np.arctan2(SingPoints[1,:],SingPoints[0,:])
            R = np.unique(np.concatenate((R,RSing)))
            Theta = np.unique(np.concatenate((Theta, ThetaSing)))
            Z = np.unique(np.concatenate((Z, SingPoints[2,:])))
        NR, NTheta, NZ = R.size, Theta.size, Z.size
        if ReturnPts:
            if Proj=='Cross':
                RR = np.tile(R,(NZ,1))
                ZZ = np.tile(Z,(NR,1)).T
                Pts = np.array([RR.flatten(), ZZ.flatten()])
                Out = '(R,Z)'
            elif Proj=='Hor':
                RR = np.tile(R,(NTheta,1))
                TT = np.tile(Theta,(NR,1)).T
                Pts = np.array([RR.flatten()*np.cos(TT.flatten()), RR.flatten()*np.sin(TT.flatten())])
                Out = '(X,Y)'
            elif Proj is None:
                RR = np.tile(np.tile(R,(NZ,1)).flatten(),(NTheta,1))
                ZZ = np.tile((np.tile(Z,(NR,1)).T).flatten(),(NTheta,1))
                TT = np.tile(Theta,(NR*NZ,1))
                Pts = np.array([RR.flatten()*np.cos(TT.flatten()), RR.flatten()*np.sin(TT.flatten()), ZZ.flatten()])
                Out = '(X,Y,Z)'
        return R, Theta, Z, NR, NTheta, NZ, Pts, Out
    elif VType=='Lin':
        Span_X = [min([sr[0] for sr in LSpan_X]), max([sr[1] for sr in LSpan_X])]
        Span_Y = [min([sr[0] for sr in LSpan_Y]), max([sr[1] for sr in LSpan_Y])]
        X = np.linspace(Span_X[0],Span_X[1],np.ceil((Span_X[1]-Span_X[0])/DX))
        Y = np.linspace(Span_Y[0],Span_Y[1],np.ceil((Span_Y[1]-Span_Y[0])/DY))
        if not SingPoints is None:
            X = np.unique(np.concatenate((X,SingPoints[0,:])))
            Y = np.unique(np.concatenate((Y,SingPoints[1,:])))
            Z = np.unique(np.concatenate((Z,SingPoints[2,:])))
        NX, NY, NZ = X.size, Y.size, Z.size
        if ReturnPts:
            if Proj=='Cross':
                YY = np.tile(Y,(NZ,1))
                ZZ = np.tile(Z,(NY,1)).T
                Pts = np.array([YY.flatten(), ZZ.flatten()])
                Out = '(Y,Z)'
            elif Proj=='Hor':
                XX = np.tile(X,(NY,1))
                YY = np.tile(Y,(NX,1)).T
                Pts = np.array([XX.flatten(), YY.flatten()])
                Out = '(X,Y)'
            elif Proj is None:
                XX = np.tile(X,(NY*NZ,1))
                YY = np.tile((np.tile(Y,(NZ,1))).flatten(),(NX,1))
                ZZ = np.tile((np.tile(Z,(NY,1)).T).flatten(),(NX,1))
                Pts = np.array([XX.flatten(), YY.flatten(), ZZ.flatten()])
                Out = '(X,Y,Z)'
        return X, Y, Z, NX, NY, NZ, Pts, Out






def _Detect_set_ConePoly(DPoly, DBaryS, DnIn, LOPolys, LOnIns, LSurfs, LOBaryS, SAngPlane, LOSD, LOSu, LOSPIn, LOSPOut, Span_k, Span_R=None, Span_Theta=None, Span_X=None, Span_Y=None, Span_Z=None,
            ConeWidth_k=None, ConeWidth_X1=None, ConeWidth_X2=None, Lens_ConeTip=None, Lens_ConeHalfAng=None, RadD=None, RadL=None, F1=None, VPoly=None, VVin=None, DLong=None,
            VType='Tor', OpType='Apert', NPsi=20, Nk=60, thet=np.linspace(0.,2.*np.pi,100),
            DXTheta=None, DRY=TFD.DetConeDRY, DZ=TFD.DetConeDZ, Test=True):       # Used

    LPolys = [DPoly] + LOPolys
    BaryS, nP, e1, e2 = SAngPlane

    # Prepare input
    if DXTheta is None:
        DXTheta = TFD.DetConeDTheta if VType=='Tor' else TFD.DetConeDX


    # Create all points
    SingPoints = np.vstack((LOSPIn, LOSPIn+0.002*LOSu, 0.5*(LOSPOut+LOSPIn), LOSPOut-0.002*LOSu , LOSPOut)).T
    if VType=='Tor':
        R, Theta, Z, NR, NTheta, NZ, foo, bar = _get_CrossHorMesh(SingPoints=SingPoints, LSpan_R=Span_R, LSpan_Theta=Span_Theta, LSpan_Z=Span_Z, DR=DRY, DTheta=DXTheta, DZ=DZ, VType=VType, ReturnPts=False)
    elif VType=='Lin':
        X, Y, Z, NX, NY, NZ, foo, bar = _get_CrossHorMesh(SingPoints=SingPoints, LSpan_X=Span_X, LSpan_Y=Span_Y, LSpan_Z=Span_Z, DX=DXTheta, DY=DRY, DZ=DZ, VType=VType, ReturnPts=False)

    # Compute
    _SAngCross_Reg, _SAngCross_Reg_Int, _SAngCross_Reg_K, _SAngCross_Reg_Psi = False, None, None, None
    if VType=='Tor':
        if np.abs(np.arctan2(LOSPIn[1],LOSPIn[0])-np.arctan2(LOSPOut[1],LOSPOut[0])) < 1.e-6:
            print "        Poloidal detector => computing SAng Int on regular grid !"
            # Create all points in a Poloidal cross-section
            psiMin, psiMax = Calc_PolProj_ConePsiMinMax_2Steps(DPoly, DBaryS, LOPolys, LOnIns, LSurfs, LOBaryS, Span_k, LOSu, LOSPOut, Nk=Nk, Test=True)
            psiMin, psiMax = psiMin.min(), psiMax.max()
            K = np.linspace(0.95*Span_k[0],1.05*Span_k[1],Nk)
            Psi = np.linspace(psiMin,psiMax,NPsi)
            KK, PPsi = np.tile(K,(NPsi,1)).T, np.tile(Psi,(Nk,1))
            RD = np.hypot(DBaryS[0],DBaryS[1])
            uR = np.hypot(LOSPOut[0],LOSPOut[1])-RD
            uZ = LOSPOut[2]-DBaryS[2]
            uN = np.hypot(uR,uZ)
            uR, uZ = uR/uN, uZ/uN
            V = np.array([uR*np.cos(PPsi.flatten()) - uZ*np.sin(PPsi.flatten()), uZ*np.cos(PPsi.flatten()) + uR*np.sin(PPsi.flatten())])
            PtsRZ = np.array([RD+KK.flatten()*V[0,:], DBaryS[2]+KK.flatten()*V[1,:]])
            Vis = np.zeros((Nk*NPsi, NTheta))
            ind1 = _Ves_isInside(VPoly, VType, DLong, PtsRZ, In='(R,Z)')
            for ii in range(0,NTheta):
                Points = np.array([PtsRZ[0,:]*np.cos(Theta[ii]), PtsRZ[0,:]*np.sin(Theta[ii]), PtsRZ[1,:]])
                Ind = np.zeros((Nk*NPsi,),dtype=float)
                ind2 = _Detect_isOnGoodSide(Points, DBaryS, DnIn, LOBaryS, LOnIns, NbPoly=None, Log='all')
                ind3 = _Detect_isInsideConeWidthLim(Points, LOSD, LOSu, ConeWidth_k, ConeWidth_X1, ConeWidth_X2)
                indSide = ind1 & ind2 & ind3
                if np.any(indSide):
                    if OpType=='Apert':
                        Ind[indSide] = GG.Calc_SAngVect_LPolysPoints_Flex(LPolys, Points[:,indSide], BaryS, nP, e1, e2)[0]
                    elif OpType=='Lens':
                        Ind[indSide] = GG.Calc_SAngVect_LPolysPoints_Flex_Lens(LOBaryS[0][0],LOBaryS[0][1],LOBaryS[0][2], Lens_ConeTip[0],Lens_ConeTip[1],Lens_ConeTip[2], DnIn[0],DnIn[1],DnIn[2],
                                Points[0,indSide],Points[1,indSide],Points[2,indSide], RadL, RadD, F1, np.tan(Lens_ConeHalfAng), LOPolys[0][0,:],LOPolys[0][1,:],LOPolys[0][2,:], thet=thet, VectReturn=False)
                indPos = Ind>0
                if np.any(indPos):
                    indC = GG.Calc_InOut_LOS_Colis(BaryS, Points[:,indPos], VPoly, VVin)
                    Ind[indPos.nonzero()[0][~indC]] = 0
                Vis[:,ii] = Ind
            _SAngCross_Reg = True
            _SAngCross_Reg_Int = np.zeros((Nk*NPsi))
            for ii in range(0,Nk*NPsi):
                _SAngCross_Reg_Int[ii] = scpinteg.trapz(Vis[ii,:], x = PtsRZ[0,ii]*Theta, axis=0)      # !!! Integral * R included here !!! (important for later in ToFu_MatComp)
            _SAngCross_Reg_Int = _SAngCross_Reg_Int.reshape((Nk,NPsi))
            _SAngCross_Reg_K = K
            _SAngCross_Reg_Psi = Psi

        Vis = np.zeros((NR, NTheta, NZ))
        RR,  ZZ  = np.tile(R,(NZ,1)).T,  np.tile(Z,(NR,1))
        RRf, ZZf = RR.flatten(),         ZZ.flatten()
        ind1 = _Ves_isInside(VPoly, VType, DLong, np.array([RRf,ZZf]), In='(R,Z)')
        NRef = round(NTheta/5.)
        for ii in range(0,NTheta):
            Points = np.array([RRf*np.cos(Theta[ii]), RRf*np.sin(Theta[ii]), ZZf])
            Ind = np.zeros((NR*NZ,),dtype=float)
            ind2 = _Detect_isOnGoodSide(Points, DBaryS, DnIn, LOBaryS, LOnIns, NbPoly=None, Log='all')
            ind3 = _Detect_isInsideConeWidthLim(Points, LOSD, LOSu, ConeWidth_k, ConeWidth_X1, ConeWidth_X2)
            indSide = ind1 & ind2 & ind3
            if np.any(indSide):
                if OpType=='Apert':
                    Ind[indSide] = GG.Calc_SAngVect_LPolysPoints_Flex(LPolys, Points[:,indSide], BaryS, nP, e1, e2)[0]
                elif OpType=='Lens':
                    Ind[indSide] = GG.Calc_SAngVect_LPolysPoints_Flex_Lens(LOBaryS[0][0],LOBaryS[0][1],LOBaryS[0][2], Lens_ConeTip[0],Lens_ConeTip[1],Lens_ConeTip[2], DnIn[0],DnIn[1],DnIn[2],
                                Points[0,indSide],Points[1,indSide],Points[2,indSide], RadL, RadD, F1, np.tan(Lens_ConeHalfAng), LOPolys[0][0,:],LOPolys[0][1,:],LOPolys[0][2,:], thet=thet, VectReturn=False)
            indPos = Ind>0
            if np.any(indPos):
                indC = GG.Calc_InOut_LOS_Colis(BaryS, Points[:,indPos], VPoly, VVin)
                Ind[indPos.nonzero()[0][~indC]] = 0
            Vis[:,ii,:] = Ind.reshape((NR,NZ))
            if ii==0 or (ii+1)%NRef==0:
                print "        Computing cross slice", ii+1, "/", NTheta, "  with ", np.sum(indSide), "/", RRf.size," points"

    elif VType=='Lin':
        if np.abs(LOSPIn[0]-LOSPOut[0]) < 1.e-6:
            print "        Cross detector => computing SAng Int on regular grid !"
            # Create all points in a cross-section
            psiMin, psiMax = Calc_PolProj_ConePsiMinMax_2Steps_Lin(DPoly, DBaryS, LOPolys, LOnIns, LSurfs, LOBaryS, Span_k, LOSu, LOSPOut, Nk=Nk, Test=True)
            psiMin, psiMax = psiMin.min(), psiMax.max()
            K = np.linspace(0.95*Span_k[0],1.05*Span_k[1],Nk)
            Psi = np.linspace(psiMin,psiMax,NPsi)
            KK, PPsi = np.tile(K,(NPsi,1)).T, np.tile(Psi,(Nk,1))
            uY = LOSPOut[1]-DBaryS[1]
            uZ = LOSPOut[2]-DBaryS[2]
            uN = np.hypot(uY,uZ)
            uY, uZ = uY/uN, uZ/uN
            V = np.array([uY*np.cos(PPsi.flatten()) - uZ*np.sin(PPsi.flatten()), uZ*np.cos(PPsi.flatten()) + uY*np.sin(PPsi.flatten())])
            PtsRZ = np.array([DBaryS[1]+KK.flatten()*V[0,:], DBaryS[2]+KK.flatten()*V[1,:]])
            Vis = np.zeros((Nk*NPsi, NX))
            for ii in range(0,NX):
                Points = np.array([X[ii]*np.ones((Nk*NPsi,)), PtsRZ[0,:], PtsRZ[1,:]])
                Ind = np.zeros((Nk*NPsi,),dtype=float)
                ind1 = _Ves_isInside(VPoly, VType, DLong, Points, In='(X,Y,Z)')
                ind2 = _Detect_isOnGoodSide(Points, DBaryS, DnIn, LOBaryS, LOnIns, NbPoly=None, Log='all')
                ind3 = _Detect_isInsideConeWidthLim(Points, LOSD, LOSu, ConeWidth_k, ConeWidth_X1, ConeWidth_X2)
                indSide = ind1 & ind2 & ind3
                if np.any(indSide):
                    if OpType=='Apert':
                        Ind[indSide] = GG.Calc_SAngVect_LPolysPoints_Flex(LPolys, Points[:,indSide], BaryS, nP, e1, e2)[0]
                    elif OpType=='Lens':
                        Ind[indSide] = GG.Calc_SAngVect_LPolysPoints_Flex_Lens(LOBaryS[0][0],LOBaryS[0][1],LOBaryS[0][2], Lens_ConeTip[0],Lens_ConeTip[1],Lens_ConeTip[2], DnIn[0],DnIn[1],DnIn[2],
                                Points[0,indSide],Points[1,indSide],Points[2,indSide], RadL, RadD, F1, np.tan(Lens_ConeHalfAng), LOPolys[0][0,:],LOPolys[0][1,:],LOPolys[0][2,:], thet=thet, VectReturn=False)
                indPos = Ind>0
                if np.any(indPos):
                    indC = GG.Calc_InOut_LOS_Colis_Lin(BaryS, Points[:,indPos], VPoly, VVin, DLong)
                    Ind[indPos.nonzero()[0][~indC]] = 0
                Vis[:,ii] = Ind
            _SAngCross_Reg = True
            _SAngCross_Reg_Int = np.zeros((Nk*NPsi))
            for ii in range(0,Nk*NPsi):
                _SAngCross_Reg_Int[ii] = scpinteg.trapz(Vis[ii,:], x=X, axis=0)
            _SAngCross_Reg_Int = _SAngCross_Reg_Int.reshape((Nk,NPsi))
            _SAngCross_Reg_K = K
            _SAngCross_Reg_Psi = Psi

        Vis = np.zeros((NX, NY, NZ))
        YY,  ZZ  = np.tile(Y,(NZ,1)).T,  np.tile(Z,(NY,1))
        YYf, ZZf = YY.flatten(),         ZZ.flatten()
        NRef = round(NX/5.)
        for ii in range(0,NX):
            Points = np.array([X[ii]*np.ones((NY*NZ,)), YYf, ZZf])
            Ind = np.zeros((NY*NZ,),dtype=float)
            ind1 = _Ves_isInside(VPoly, VType, DLong, Points, In='(X,Y,Z)')
            ind2 = _Detect_isOnGoodSide(Points, DBaryS, DnIn, LOBaryS, LOnIns, NbPoly=None, Log='all')
            ind3 = _Detect_isInsideConeWidthLim(Points, LOSD, LOSu, ConeWidth_k, ConeWidth_X1, ConeWidth_X2)
            indSide = ind1 & ind2 & ind3
            if np.any(indSide):
                if OpType=='Apert':
                    Ind[indSide] = GG.Calc_SAngVect_LPolysPoints_Flex(LPolys, Points[:,indSide], BaryS, nP, e1, e2)[0]
                elif OpType=='Lens':
                        Ind[indSide] = GG.Calc_SAngVect_LPolysPoints_Flex_Lens(LOBaryS[0][0],LOBaryS[0][1],LOBaryS[0][2], Lens_ConeTip[0],Lens_ConeTip[1],Lens_ConeTip[2], DnIn[0],DnIn[1],DnIn[2],
                                Points[0,indSide],Points[1,indSide],Points[2,indSide], RadL, RadD, F1, np.tan(Lens_ConeHalfAng), LOPolys[0][0,:],LOPolys[0][1,:],LOPolys[0][2,:], thet=thet, VectReturn=False)
            indPos = Ind>0
            if np.any(indPos):
                indC = GG.Calc_InOut_LOS_Colis_Lin(BaryS, Points[:,indPos], VPoly, VVin, DLong)
                Ind[indPos.nonzero()[0][~indC]] = 0
            Vis[ii,:,:] = Ind.reshape((NY,NZ))
            if ii==0 or (ii+1)%NRef==0:
                print "        Computing cross slice", ii+1, "/", NX, "  with ", np.sum(indSide), "/", YYf.size," points"

    # Cone_PolyCross
    if VType=='Tor':
        ind = np.any(Vis>0,axis=1).flatten()
        _SAngCross_Points = np.array([RRf[ind], ZZf[ind]])
        _SAngCross_Max = np.max(Vis,axis=1).flatten()[ind]
        _SAngCross_Int = np.zeros((NR,NZ))
        for ii in range(0,NR):
            _SAngCross_Int[ii,:] = scpinteg.trapz(Vis[ii,:,:], x=R[ii]*Theta, axis=0)    # Multiplication by R, important !!!
        _SAngCross_Int = _SAngCross_Int.flatten()[ind]
        Visbis = np.max(Vis,axis=1)
        Visbis[Visbis>0] = 1
                # Don't forget the edge...
        dR, dZ = np.diff(np.unique(R)).min(), np.diff(np.unique(Z)).min()
        RRbis = np.concatenate((RR[:,0:1],RR,RR[:,-1:]),axis=1)
        RRbis = np.concatenate((RRbis[0:1,:]-dR,RRbis,RRbis[-1:,:]+dR),axis=0)
        ZZbis = np.concatenate((ZZ[0:1,:],ZZ,ZZ[-1:,:]),axis=0)
        ZZbis = np.concatenate((ZZbis[:,0:1]-dZ,ZZbis,ZZbis[:,-1:]+dZ),axis=1)
        Visbis = np.concatenate((np.zeros((NR,1)),Visbis,np.zeros((NR,1))),axis=1)
        Visbis = np.concatenate((np.zeros((1,NZ+2)),Visbis,np.zeros((1,NZ+2))),axis=0)
        CNb = cntr.Cntr(RRbis,ZZbis,Visbis)
    elif VType=='Lin':
        ind = np.any(Vis>0,axis=0).flatten()
        _SAngCross_Points = np.array([YYf[ind], ZZf[ind]])
        _SAngCross_Max = np.max(Vis,axis=0).flatten()[ind]
        _SAngCross_Int = np.zeros((NY,NZ))
        for ii in range(0,NY):
            _SAngCross_Int[ii,:] = scpinteg.trapz(Vis[:,ii,:], x=X, axis=0)
        _SAngCross_Int = _SAngCross_Int.flatten()[ind]
        Visbis = np.max(Vis,axis=0)
        Visbis[Visbis>0] = 1
                # Don't forget the edge...
        dY, dZ = np.diff(np.unique(Y)).min(), np.diff(np.unique(Z)).min()
        YYbis = np.concatenate((YY[:,0:1],YY,YY[:,-1:]),axis=1)
        YYbis = np.concatenate((YYbis[0:1,:]-dY,YYbis,YYbis[-1:,:]+dY),axis=0)
        ZZbis = np.concatenate((ZZ[0:1,:],ZZ,ZZ[-1:,:]),axis=0)
        ZZbis = np.concatenate((ZZbis[:,0:1]-dZ,ZZbis,ZZbis[:,-1:]+dZ),axis=1)
        Visbis = np.concatenate((np.zeros((NY,1)),Visbis,np.zeros((NY,1))),axis=1)
        Visbis = np.concatenate((np.zeros((1,NZ+2)),Visbis,np.zeros((1,NZ+2))),axis=0)
        CNb = cntr.Cntr(YYbis,ZZbis,Visbis)
    Cone_Poly = CNb.trace(0)
    NPol = len(Cone_Poly)/2
    _Cone_PolyCross = [np.asarray(Cone_Poly[ii].T,dtype=float) for ii in range(0,NPol)]

    # Cone_PolyHor
    if VType=='Tor':
        RR, TT = np.tile(R,(NTheta,1)).T, np.tile(Theta,(NR,1))
        XX, YY = RR*np.cos(TT), RR*np.sin(TT)
        XXf, YYf = XX.flatten(), YY.flatten()
        ind = np.any(Vis>0,axis=2).flatten()
        _SAngHor_Points = np.array([XXf[ind], YYf[ind]])
        _SAngHor_Max = np.max(Vis,axis=2).flatten()[ind]
        _SAngHor_Int = scpinteg.trapz(Vis, x=Z, axis=2).flatten()[ind]
        Visbis = np.max(Vis,axis=2)
        Visbis[Visbis>0] = 1
                # Don't forget the edge...
        dTheta = np.min(np.diff(np.unique(Theta)))
        RRbis = np.concatenate((RR[:,0:1],RR,RR[:,-1:]),axis=1)
        RRbis = np.concatenate((RRbis[0:1,:]-dR,RRbis,RRbis[-1:,:]+dR),axis=0)
        TTbis = np.concatenate((TT[0:1,:],TT,TT[-1:,:]),axis=0)
        TTbis = np.concatenate((TTbis[:,0:1]-dTheta,TTbis,TTbis[:,-1:]+dTheta),axis=1)
        XX, YY = RRbis*np.cos(TTbis), RRbis*np.sin(TTbis)
        Visbis = np.concatenate((np.zeros((NR,1)),Visbis,np.zeros((NR,1))),axis=1)
        Visbis = np.concatenate((np.zeros((1,NTheta+2)),Visbis,np.zeros((1,NTheta+2))),axis=0)
        CNb = cntr.Cntr(XX,YY,Visbis)
    elif VType=='Lin':
        XX, YY = np.tile(X,(NY,1)).T, np.tile(Y,(NX,1))
        XXf, YYf = XX.flatten(), YY.flatten()
        ind = np.any(Vis>0,axis=2).flatten()
        _SAngHor_Points = np.array([XXf[ind], YYf[ind]])
        _SAngHor_Max = np.max(Vis,axis=2).flatten()[ind]
        _SAngHor_Int = scpinteg.trapz(Vis, x=Z, axis=2).flatten()[ind]
        Visbis = np.max(Vis,axis=2)
        Visbis[Visbis>0] = 1
                # Don't forget the edge...
        dX = np.min(np.diff(np.unique(X)))
        XXbis = np.concatenate((XX[:,0:1],XX,XX[:,-1:]),axis=1)
        XXbis = np.concatenate((XXbis[0:1,:]-dX,XXbis,XXbis[-1:,:]+dX),axis=0)
        YYbis = np.concatenate((YY[0:1,:],YY,YY[-1:,:]),axis=0)
        YYbis = np.concatenate((YYbis[:,0:1]-dY,YYbis,YYbis[:,-1:]+dY),axis=1)
        Visbis = np.concatenate((np.zeros((NX,1)),Visbis,np.zeros((NX,1))),axis=1)
        Visbis = np.concatenate((np.zeros((1,NY+2)),Visbis,np.zeros((1,NY+2))),axis=0)
        CNb = cntr.Cntr(XXbis,YYbis,Visbis)
    Cone_Poly = CNb.trace(0)
    NPol = len(Cone_Poly)/2
    _Cone_PolyHor = [np.asarray(Cone_Poly[ii].T,dtype=float) for ii in range(0,NPol)]

    _Cone_PolyCrossbis = [np.copy(pp) for pp in _Cone_PolyCross]
    _Cone_PolyHorbis = [np.copy(pp) for pp in _Cone_PolyHor]
    if VType=='Tor':
        _Cone_Poly_DR, _Cone_Poly_DZ, _Cone_Poly_DTheta = DRY, DZ, DXTheta
        _Cone_Poly_DX, _Cone_Poly_DY = None, None
    elif VType=='Lin':
        _Cone_Poly_DX, _Cone_Poly_DY, _Cone_Poly_DZ = DXTheta, DRY, DZ
        _Cone_Poly_DR, _Cone_Poly_DTheta = None, None

    return _SAngCross_Reg, _SAngCross_Reg_Int, _SAngCross_Reg_K, _SAngCross_Reg_Psi, _SAngCross_Points, _SAngCross_Max, _SAngCross_Int, _SAngHor_Points, _SAngHor_Max, _SAngHor_Int, _Cone_PolyCross, _Cone_PolyHor, _Cone_PolyCrossbis, _Cone_PolyHorbis, _Cone_Poly_DX, _Cone_Poly_DY, _Cone_Poly_DR, _Cone_Poly_DTheta, _Cone_Poly_DZ




def Calc_PolProj_ConePsiMinMax_2Steps(DPoly, DBaryS, LOPolys, LOnIns, LSurfs, LOBaryS, Span_k, LOSu, LOSPOut, Nk=20, Test=True):      # Used
    """ Compute the min, max of the angular width in poloidal projection of the viewing cone

    Inputs :
    --------
        Test        A boolean to know if tests for inputs should be performed or not, if False make sure that all inputs are perfect !

    Outputs :
    ---------
        PsiMin
        PsiMax
    """

    if Test:
        assert type(Nk) is int, "Arg Nk must be a int !"
        assert all([type(aa) is list for aa in [LOPolys, LOnIns, LSurfs, LOBaryS]]), "Args LOPolys, LOnIns, LSurfs and LOBaryS must be list !"

    ks = np.linspace(Span_k[0],Span_k[1],Nk)
    Ps = np.array([DBaryS[0]+ks*LOSu[0], DBaryS[1]+ks*LOSu[1], DBaryS[2]+ks*LOSu[2]])
    RPs = np.hypot(Ps[0,:],Ps[1,:])
    nPs = np.tile(LOSu, (Nk,1)).T
    RD = np.hypot(DBaryS[0],DBaryS[1])
    uR = np.hypot(LOSPOut[0],LOSPOut[1])-RD
    uZ = LOSPOut[2]-DBaryS[2]
    uN = np.hypot(uR,uZ)
    uR, uZ = uR/uN, uZ/uN
    e1, e2 = GG.Calc_DefaultCheck_e1e2_PLanes_2D(Ps, nPs)
    Corners = DPoly[:,:-1]
    NCorners, NPlans = Corners.shape[1], Ps.shape[1]
    NPoly = len(LOPolys)
    PsiMin, PsiMax = np.nan*np.ones((NPlans,NCorners)), np.nan*np.ones((NPlans,NCorners))
    if NPoly==1:
        for ii in range(NCorners):
            Out = GG.Calc_PolysProjPlanesPoint(LOPolys, Corners[:,ii:ii+1], Ps, nPs, e1P=e1,e2P=e2,Test=False)
            Rps = np.hypot(Out[0],Out[1])
            VectR, VectZ = Rps-RD, Out[2]-DBaryS[2]
            VectN = np.hypot(VectR,VectZ)
            sinpsi = (uR*VectZ-uZ*VectR)/VectN
            psi = np.arcsin(sinpsi)
            PsiMin[:,ii] = np.min(psi,axis=1)
            PsiMax[:,ii] = np.max(psi,axis=1)
    else:
        nPtemp = np.asarray([LOnIns[ii]*LSurfs[ii] for ii in range(0,len(LOPolys))])
        nPtemp = np.sum(np.array(nPtemp).T.reshape((3,NPoly)), axis=1, keepdims=False)/np.sum(np.array(LSurfs))
        nPtemp = nPtemp/np.linalg.norm(nPtemp)
        Ptemp = LOBaryS[0]
        e1temp, e2temp = GG.Calc_DefaultCheck_e1e2_PLane_1D(Ptemp, nPtemp)
        for ii in range(NCorners):
            Out = GG.Calc_PolysProjPlanesPoint(LOPolys, Corners[:,ii:ii+1], Ptemp.reshape((3,1)), nPtemp.reshape((3,1)), e1P=e1temp.reshape((3,1)), e2P=e2temp.reshape((3,1)),Test=False)
            PolyRef = [np.concatenate((Out[3][jj],Out[4][jj]),axis=0) for jj in range(0,NPoly)]
            PolyRef = GG.Calc_PolyInterLPoly2D(PolyRef,Test=False)
            if PolyRef.shape[1]>0:
                PolyRef = GG.Calc_3DPolyfrom2D_1D(PolyRef, Ptemp, nPtemp, e1temp, e2temp, Test=False)
                Out = GG.Calc_PolysProjPlanesPoint(PolyRef, Corners[:,ii:ii+1], Ps, nPs, e1P=e1,e2P=e2,Test=False)
                Rps = np.hypot(Out[0],Out[1])
                VectR, VectZ = Rps-RD, Out[2]-DBaryS[2]
                VectN = np.hypot(VectR,VectZ)
                sinpsi = (uR*VectZ-uZ*VectR)/VectN
                psi = np.arcsin(sinpsi)
                PsiMin[:,ii] = np.min(psi,axis=1)
                PsiMax[:,ii] = np.max(psi,axis=1)
    assert ~np.any(np.isnan(PsiMin)) and ~np.any(np.isnan(PsiMax))
    return np.nanmin(PsiMin,axis=1), np.nanmax(PsiMax,axis=1)


def Calc_PolProj_ConePsiMinMax_2Steps_Lin(DPoly, DBaryS, LOPolys, LOnIns, LSurfs, LOBaryS, Span_k, LOSu, LOSPOut, Nk=20, Test=True):      # Used
    """ Compute the min, max of the angular width in poloidal projection of the viewing cone

    Inputs :
        D           A Detect instance
        P           A (3,N) np.ndarray corresponding to the cartesian coordinates of N points (one for each plane)
        nP          A (3,N) np.ndarray corresponding to the cartesian coordinates of N normal vectors (one for each plane)
        e1          The e1 unitary vector (corresponds to X1), if unspecified, the optimal vector is used
        e2          The e2 unitary vector (corresponds to X2), must be orthogonal to e1, if unspecified, the optimal vector is used
        Test        A boolean to know if tests for inputs should be performed or not, if False make sure that all inputs are perfect !
    Outputs :
        MinX1       A (NPlans,1) np.ndarray containing the minimum value of X1 (cf. e1) of the intersection of the projected polygons (on each plan for each corner)
        MinX2       A (NPlans,1) np.ndarray containing the minimum value of X2 (cf. e2) of the intersection of the projected polygons (on each plan for each corner)
        MaxX1       A (NPlans,1) np.ndarray containing the maximum value of X1 (cf. e1) of the intersection of the projected polygons (on each plan for each corner)
        MaxX2       A (NPlans,1) np.ndarray containing the maximum value of X2 (cf. e2) of the intersection of the projected polygons (on each plan for each corner)
    """

    if Test:
        assert isinstance(LOSu,np.ndarray), "Arg LOSu should be a np.ndarray !"
    ks = np.linspace(Span_k[0],Span_k[1],Nk)
    Ps = np.array([DBaryS[0]+ks*LOSu[0], DBaryS[1]+ks*LOSu[1], DBaryS[2]+ks*LOSu[2]])
    nPs = np.tile(LOSu, (Nk,1)).T
    uY = LOSPOut[1]-DBaryS[1]
    uZ = LOSPOut[2]-DBaryS[2]
    uN = np.hypot(uY,uZ)
    uY, uZ = uY/uN, uZ/uN
    e1, e2 = GG.Calc_DefaultCheck_e1e2_PLanes_2D(Ps, nPs)
    Corners = DPoly[:,:-1]
    NCorners, NPlans = Corners.shape[1], Ps.shape[1]
    NPoly = len(LOPolys)
    PsiMin, PsiMax = np.nan*np.ones((NPlans,NCorners)), np.nan*np.ones((NPlans,NCorners))
    if NPoly==1:
        for ii in range(NCorners):
            Out = GG.Calc_PolysProjPlanesPoint(LOPolys, Corners[:,ii:ii+1], Ps, nPs, e1P=e1,e2P=e2,Test=False)
            VectY, VectZ = Out[1]-DBaryS[1], Out[2]-DBaryS[2]
            VectN = np.hypot(VectY,VectZ)
            sinpsi = (uY*VectZ-uZ*VectY)/VectN
            psi = np.arcsin(sinpsi)
            PsiMin[:,ii] = np.min(psi,axis=1)
            PsiMax[:,ii] = np.max(psi,axis=1)
    else:
        nPtemp = np.asarray([LOnIns[ii]*LSurfs[ii] for ii in range(0,len(LOPolys))])
        nPtemp = np.sum(np.array(nPtemp).T.reshape((3,NPoly)), axis=1, keepdims=False)/np.sum(np.array(LSurfs))
        nPtemp = nPtemp/np.linalg.norm(nPtemp)
        Ptemp = LOBaryS[0]
        e1temp, e2temp = GG.Calc_DefaultCheck_e1e2_PLane_1D(Ptemp, nPtemp)
        for ii in range(NCorners):
            Out = GG.Calc_PolysProjPlanesPoint(LOPolys, Corners[:,ii:ii+1], Ptemp.reshape((3,1)), nPtemp.reshape((3,1)), e1P=e1temp.reshape((3,1)), e2P=e2temp.reshape((3,1)),Test=False)
            PolyRef = [np.concatenate((Out[3][jj],Out[4][jj]),axis=0) for jj in range(0,NPoly)]
            PolyRef = GG.Calc_PolyInterLPoly2D(PolyRef,Test=False)
            if PolyRef.shape[1]>0:
                PolyRef = GG.Calc_3DPolyfrom2D_1D(PolyRef, Ptemp, nPtemp, e1temp, e2temp, Test=False)
                Out = GG.Calc_PolysProjPlanesPoint(PolyRef, Corners[:,ii:ii+1], Ps, nPs, e1P=e1,e2P=e2,Test=False)
                #Rps = np.hypot(Out[0],Out[1])
                VectY, VectZ = Out[1]-DBaryS[1], Out[2]-DBaryS[2]
                VectN = np.hypot(VectY,VectZ)
                sinpsi = (uY*VectZ-uZ*VectY)/VectN
                psi = np.arcsin(sinpsi)
                PsiMin[:,ii] = np.min(psi,axis=1)
                PsiMax[:,ii] = np.max(psi,axis=1)
    assert ~np.any(np.isnan(PsiMin)) and ~np.any(np.isnan(PsiMax))
    return np.nanmin(PsiMin,axis=1), np.nanmax(PsiMax,axis=1)


def Refine_ConePoly_All(Poly, dMax=TFD.DetConeRefdMax):      # Used
    """
    Return a lighter (i.e.: resampled) version of the input polygon, assuming concavity is limited to an input dMax
    """
    diffp = np.sqrt(np.sum((Poly[:,1:]-Poly[:,:-1])**2,axis=0))
    ind = (diffp>1.e-6).nonzero()[0]
    Poly = Poly[:,ind]
    Poly = GG.MakeClockwise(Poly)
    ConvPoly = np.array(plgut.convexHull(plg.Polygon(Poly.T))[0]).T
    ConvPoly = GG.MakeClockwise(np.append(ConvPoly,ConvPoly[:,0:1],axis=1))
    Polybis = np.copy(Poly)
    for ii in range(0,ConvPoly.shape[1]-1):
        p1, p2 = ConvPoly[:,ii], ConvPoly[:,ii+1]
        ind0 = np.argmin(np.hypot(Polybis[0,:]-p1[0],Polybis[1,:]-p1[1]))
        Polybis = np.concatenate((Polybis[:,ind0:], Polybis[:,:ind0]),axis=1)  # Make sure it starts from p1
        ind1, ind2 = np.argmin(np.hypot(Polybis[0,:]-p1[0],Polybis[1,:]-p1[1])), np.argmin(np.hypot(Polybis[0,:]-p2[0],Polybis[1,:]-p2[1]))
        if ind2==ind1+1:
            indtemp = [ind1,ind2]
        else:
            u12 = (p2-p1)/np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
            lp = Polybis[:,ind1+1:ind2]
            if np.any(np.abs(u12[0]*(lp[1,:]-p1[1]) - u12[1]*(lp[0,:]-p1[0])) > dMax):
                indtemp = range(ind1,ind2+1)
            else:
                indtemp = [ind1,ind2]
        PointsIn = Polybis[:,indtemp]
        Polybis = np.concatenate((PointsIn, Polybis[:,ind2+1:]),axis=1)
    if not np.all(Polybis[:,0]==Polybis[:,-1]):
        Polybis = np.append(Polybis,Polybis[:,0:1],axis=1)
    return Polybis



def _Detect_get_KPsiCrossInt(PtsRZ, SAngCross_Reg=None, LOSPOut=None, DBaryS=None, VType='Tor'):        # Used
    assert SAngCross_Reg is True, "Only possible for cross-section detectors !"
    assert isinstance(PtsRZ,np.ndarray) and PtsRZ.ndim==2 and PtsRZ.shape[0]==2, "Arg PtsRZ must be a (2,N) np.ndarray !"
    uZ = LOSPOut[2]-DBaryS[2]
    if VType=='Tor':
        RD = np.hypot(DBaryS[0],DBaryS[1])
        u1 = np.hypot(LOSPOut[0],LOSPOut[1])-RD
        Vect1, VectZ = PtsRZ[0,:]-RD, PtsRZ[1,:]-DBaryS[2]
    elif VType=='Lin':
        u1 = LOSPOut[1]-DBaryS[1]
        Vect1, VectZ = PtsRZ[0,:]-DBaryS[1], PtsRZ[1,:]-DBaryS[2]
    uN = np.hypot(u1,uZ)
    u1, uZ = u1/uN, uZ/uN
    VectN = np.hypot(Vect1,VectZ)
    sinpsi = (u1*VectZ-uZ*Vect1)/VectN
    Psi = np.arcsin(sinpsi)
    return VectN, Psi




def _Detect_isInside(Cone_PolyCrossbis, Cone_PolyHorbis, Points, In='(X,Y,Z)', VType='Tor', TorAngRef=None, Test=True):        # Used
    if Test:
        assert isinstance(Points,np.ndarray) and Points.ndim==2 and Points.shape[0] in [2,3], "Arg Points must be a 2D np.ndarray !"
        assert In in ['(R,Z)','(Y,Z)','(X,Y)','(X,Y,Z)','(R,phi,Z)'], "Arg In must be in ['(R,Z)','(Y,Z)','(X,Y)','(X,Y,Z)','(R,phi,Z)'] !"
    NP = Points.shape[1]
    NPol, NHor = len(Cone_PolyCrossbis), len(Cone_PolyHorbis)
    if In=='(R,Z)' and VType=='Tor':
        indPol = np.zeros((NPol,NP),dtype=bool)
        for ii in range(0,NPol):
            pp = Path(Cone_PolyCrossbis[ii].T)
            indPol[ii,:] = pp.contains_points(Points.T, transform=None, radius=0.0)
        indHor = np.ones((NHor,NP),dtype=bool)
    elif In=='(Y,Z)' and VType=='Lin':
        indPol = np.zeros((NPol,NP),dtype=bool)
        for ii in range(0,NPol):
            pp = Path(Cone_PolyCrossbis[ii].T)
            indPol[ii,:] = pp.contains_points(Points.T, transform=None, radius=0.0)
        indHor = np.ones((NHor,NP),dtype=bool)
    elif In=='(X,Y)':
        indHor = np.zeros((NHor,NP),dtype=bool)
        for ii in range(0,NPol):
            pp = Path(Cone_PolyHorbis[ii].T)
            indHor[ii,:] = pp.contains_points(Points.T, transform=None, radius=0.0)
        indPol = np.ones((NPol,NP),dtype=bool)
    else:
        Points = GG.CoordShift(Points, In=In, Out='(X,Y,Z)', CrossRef=TorAngRef) if VType=='Tor' else GG.CoordShift(Points, In=In, Out='(X,Y,Z)')
        PointsCross = np.array([np.hypot(Points[0,:],Points[1,:]),Points[2,:]]) if VType=='Tor' else Points[1:,:]
        PointsHor = Points[:2,:]
        indPol, indHor = np.zeros((NPol,NP),dtype=bool), np.zeros((NHor,NP),dtype=bool)
        for ii in range(0,NPol):
            pp = Path(Cone_PolyCrossbis[ii].T)
            indPol[ii,:] = pp.contains_points(PointsCross.T, transform=None, radius=0.0)
        for ii in range(0,NHor):
            pp = Path(Cone_PolyHorbis[ii].T)
            indHor[ii,:] = pp.contains_points(PointsHor.T, transform=None, radius=0.0)
    return np.logical_and(np.any(indPol,axis=0), np.any(indHor,axis=0))



def _Detect_get_SAngIntMax(SAngCross_Reg=True, SAngCross_Points=None, SAngCross_Reg_K=None, SAngCross_Reg_Psi=None, SAngCross_Reg_Int=None, SAngCross_Int=None, SAngCross_Max=None, SAngHor_Points=None, SAngHor_Int=None, SAngHor_Max=None,
        Cone_PolyCrossbis=None, Cone_PolyHorbis=None, TorAngRef=None, DBaryS=None, LOSPOut=None, Proj='Cross', SAng='Int', VType='Tor', Test=True):        # Used
    assert Proj in ['Cross','Hor'], "Arg Proj must be in ['Cross','Hor'] !"
    assert SAng in ['Int','Max'], "Arg SAng must be in ['Int','Max'] !"

    if Proj=='Cross' and SAngCross_Reg and SAng=='Int':
        InRef = '(R,Z)' if VType=='Tor' else '(Y,Z)'
        ff = scpinterp.RectBivariateSpline(SAngCross_Reg_K, SAngCross_Reg_Psi, SAngCross_Reg_Int, bbox=[None, None, None, None], kx=1, ky=1, s=0)
        def FF(Pts, In=None, ff=ff, InRef=InRef):
            assert In==InRef, "Arg Pts must be provided in "+InRef
            SA = np.zeros((Pts.shape[1],))
            ind = _Detect_isInside(Cone_PolyCrossbis, Cone_PolyHorbis, Pts, In=In, VType=VType, TorAngRef=TorAngRef, Test=Test)
            if np.any(ind):
                Pts = Pts[:,ind].reshape((2,1)) if ind.sum()==1 else Pts[:,ind]
                K, psi = _Detect_get_KPsiCrossInt(Pts, SAngCross_Reg=SAngCross_Reg, LOSPOut=LOSPOut, DBaryS=DBaryS, VType=VType)
                SA[ind] = ff(K,psi,grid=False)
            return SA
    else:
        if Proj=='Cross':
            InRef = '(R,Z)' if VType=='Tor' else '(Y,Z)'
            val = SAngCross_Int if SAng=='Int' else SAngCross_Max
            #ff = scpinterp.interp2d(SAngCross_Points[0,:], SAngCross_Points[1,:], val, kind='linear', copy=True, bounds_error=False, fill_value=0.)
            [val], X, Y, nx, ny = GG.Remap_2DFromFlat(SAngCross_Points, [val])
            val[np.isnan(val)] = 0.
            ff = scpinterp.RectBivariateSpline(X, Y, val, bbox=[None, None, None, None], kx=1, ky=1, s=0)    # Faster and less prone to errors than scpinter.interp2d :-)
        else:
            InRef = '(X,Y)'
            val = SAngHor_Int if SAng=='Int' else SAngHor_Max
            #ff = scpinterp.interp2d(SAngHor_Points[0,:], SAngHor_Points[1,:], val, kind='linear', copy=True, bounds_error=False, fill_value=0.)
            [val], X, Y, nx, ny = GG.Remap_2DFromFlat(SAngHor_Points, [val])
            val[np.isnan(val)] = 0.
            ff = scpinterp.RectBivariateSpline(X, Y, val, bbox=[None, None, None, None], kx=1, ky=1, s=0)    # Faster and less prone to errors than scpinter.interp2d :-)


        def FF(Pts, In=None, ff=ff, InRef=InRef):
            assert In==InRef, "Arg Pts must be provided in "+InRef
            SA = np.zeros((Pts.shape[1],))
            ind = _Detect_isInside(Cone_PolyCrossbis, Cone_PolyHorbis, Pts, In=In, VType=VType, TorAngRef=TorAngRef, Test=Test)
            if np.any(ind):
                Pts = Pts[:,ind].reshape((2,1)) if ind.sum()==1 else Pts[:,ind]
                Ind = ind.nonzero()[0]
                for ii in range(0,Ind.size):
                    SA[Ind[ii]] = ff(Pts[0,ii],Pts[1,ii])
            return SA
    return FF


def _Detect_set_SigPrecomp(DPoly, DBaryS, DnIn, LOPolys, LOBaryS, LOnIns, SAngPlane, LOSD=None, LOSu=None, Span_k=None, ConeWidth_k=None, ConeWidth_X1=None, ConeWidth_X2=None, Cone_PolyCrossbis=None, Cone_PolyHorbis=None,
        Lens_ConeTip=None, Lens_ConeHalfAng=None, RadL=None, RadD=None, F1=None, thet=np.linspace(0.,2.*np.pi,100), VPoly=None, VVin=None, DLong=None, CrossRef=None,
        dX12=TFD.DetSynthdX12, dX12Mode=TFD.DetSynthdX12Mode, ds=TFD.DetSynthds, dsMode=TFD.DetSynthdsMode, MarginS=TFD.DetSynthMarginS, VType='Tor', OpType='Apert', Colis=True, Test=True):        # Used

    Points, dV = Calc_SynthDiag_SampleVolume(LOSD=LOSD, LOSu=LOSu, Span_k=Span_k, ConeWidth_k=ConeWidth_k, ConeWidth_X1=ConeWidth_X1, ConeWidth_X2=ConeWidth_X2,
                                             dX12=dX12, dX12Mode=dX12Mode, ds=ds, dsMode=dsMode, MarginS=MarginS, Detail=False)

    SAng, Vect = _Detect_SAngVect_Points(Points, DPoly=DPoly, DBaryS=DBaryS, DnIn=DnIn, LOBaryS=LOBaryS, LOnIns=LOnIns, LOPolys=LOPolys, SAngPlane=SAngPlane, Lens_ConeTip=Lens_ConeTip, Lens_ConeHalfAng=Lens_ConeHalfAng,
                                         RadL=RadL, RadD=RadD, F1=F1, thet=thet, OpType=OpType, VPoly=VPoly, VVin=VVin, DLong=DLong, VType=VType,
                                         Cone_PolyCrossbis=Cone_PolyCrossbis, Cone_PolyHorbis=Cone_PolyHorbis, TorAngRef=CrossRef,Colis=Colis,Test=Test)
    indPos = SAng>0.
    assert np.any(indPos), "There seems to be no visible point in the plasma... !"

    _SynthDiag_Points, _SynthDiag_SAng, _SynthDiag_Vect, _SynthDiag_dV = Points[:,indPos], SAng[indPos], Vect[:,indPos], dV
    _SynthDiag_ds, _SynthDiag_dsMode, _SynthDiag_MarginS, _SynthDiag_dX12, _SynthDiag_dX12Mode = ds, dsMode, MarginS, dX12, dX12Mode
    _SynthDiag_Colis = Colis

    return _SynthDiag_Points, _SynthDiag_SAng, _SynthDiag_Vect, _SynthDiag_dV, _SynthDiag_ds, _SynthDiag_dsMode, _SynthDiag_MarginS, _SynthDiag_dX12, _SynthDiag_dX12Mode, _SynthDiag_Colis



def Calc_SynthDiag_SampleVolume(LOSD=None, LOSu=None, Span_k=None, ConeWidth_k=None, ConeWidth_X1=None, ConeWidth_X2=None,
        dX12=TFD.DetSynthdX12, dX12Mode=TFD.DetSynthdX12Mode, ds=TFD.DetSynthds, dsMode=TFD.DetSynthdsMode, MarginS=TFD.DetSynthMarginS, Detail=False):   # Used
    """
    Return a (X,Y,Z) mesh of the viewing volume of a detector
    """
    e1, e2 = GG.Calc_DefaultCheck_e1e2_PLane_1D(LOSD, LOSu)
    DS = Span_k[1]-Span_k[0]
    if dsMode=='rel':
        Nums = int(np.ceil(1./ds))
    else:
        Nums = int(np.ceil(DS/ds))
    if dX12Mode=='rel':
        NumX1 = int(np.ceil(1./dX12[0]))
        NumX2 = int(np.ceil(1./dX12[1]))
    else:
        NumX1 = int(np.ceil(np.diff(ConeWidth_X1[:,-1])/dX12[0]))
        NumX2 = int(np.ceil(np.diff(ConeWidth_X2[:,-1])/dX12[1]))
    X1 = np.linspace(ConeWidth_X1[0,-1], ConeWidth_X1[1,-1], NumX1)
    X2 = np.linspace(ConeWidth_X2[0,-1], ConeWidth_X2[1,-1], NumX2)
    Ss = np.linspace(Span_k[0]+MarginS, Span_k[1], Nums)     # was mistake : np.linspace(MarginS, DS, Nums)
    dX1, dX2, ds = np.mean(np.diff(X1)), np.mean(np.diff(X2)), np.mean(np.diff(Ss))
    dV = dX1*dX2*ds
    NumP = NumX1*NumX2*Nums
    X1f = (np.resize(X1,(Nums,NumX2,NumX1)).T).flatten()
    X2f = (np.resize(X2,(NumX1,Nums,NumX2)).swapaxes(1,2)).flatten()
    Ssf = np.tile(Ss,(NumX1,NumX2,1)).flatten()
    if Detail:
        Points = np.array([LOSD[0] + LOSu[0]*Ssf + e1[0]*X1f + e2[0]*X2f, LOSD[1] + LOSu[1]*Ssf + e1[1]*X1f + e2[1]*X2f, LOSD[2] + LOSu[2]*Ssf + e1[2]*X1f + e2[2]*X2f])
        return Points, dV, X1, X2, Ss
    else:
        MinX1, MaxX1 = np.interp(Ss,ConeWidth_k,ConeWidth_X1[0,:]), np.interp(Ss,ConeWidth_k,ConeWidth_X1[1,:])
        MinX2, MaxX2 = np.interp(Ss,ConeWidth_k,ConeWidth_X2[0,:]), np.interp(Ss,ConeWidth_k,ConeWidth_X2[1,:])
        del NumP, Nums, dX1, dX2, ds, DS, X1, X2, Ss, dsMode, dX12Mode, dX12
        MinX1, MaxX1 = np.tile(MinX1,(NumX1,NumX2,1)).flatten(), np.tile(MaxX1,(NumX1,NumX2,1)).flatten()
        MinX2, MaxX2 = np.tile(MinX2,(NumX1,NumX2,1)).flatten(), np.tile(MaxX2,(NumX1,NumX2,1)).flatten()
        ind = (X1f>=MinX1) & (X1f<=MaxX1) & (X2f>=MinX2) & (X2f<=MaxX2)
        X1f, X2f, Ssf = X1f[ind], X2f[ind], Ssf[ind]
        Points = np.array([LOSD[0] + LOSu[0]*Ssf + e1[0]*X1f + e2[0]*X2f, LOSD[1] + LOSu[1]*Ssf + e1[1]*X1f + e2[1]*X2f, LOSD[2] + LOSu[2]*Ssf + e1[2]*X1f + e2[2]*X2f])
        return Points, dV





def _Detect_SigSynthDiag(ff, extargs={}, Method='Vol', Mode='simps', PreComp=True,
        DPoly=None, DBaryS=None, DnIn=None, LOPolys=None, LOBaryS=None, LOnIn=None, Lens_ConeTip=None, Lens_ConeHalfAng=None, RadL=None, RadD=None, F1=None, thet=np.linspace(0.,2.*np.pi,100), OpType='Apert',
        LOSD=None, LOSu=None, LOSkPIn=None, LOSkPOut=None, LOSEtend=None, Span_k=None, ConeWidth_X1=None, ConeWidth_X2=None, SAngPlane=None, CrossRef=None,
        Cone_PolyCrossbis=None, Cone_PolyHorbis=None, VPoly=None,  VVin=None, VType='Tor',
        SynthDiag_Points=None, SynthDiag_SAng=None, SynthDiag_Vect=None, SynthDiag_dV=None,
        SynthDiag_dX12=None, SynthDiag_dX12Mode=None, SynthDiag_ds=None, SynthDiag_dsMode=None, SynthDiag_MarginS=None, SynthDiag_Colis=None,
        epsrel=None, dX12=None, dX12Mode=None, ds=None, dsMode=None, MarginS=None, Colis=True, Test=True):        # Used
    if Test:
        assert hasattr(ff, '__call__'), "Arg ff must be a callable (function of one or two arguments) !"
        assert type(extargs) is dict, "Arg extargs must be a dict of keyword args for ff !"

    insargs = inspect.getargspec(ff)
    if len(insargs[0])-len(insargs[3])==1:
        Ani = False
    elif len(insargs[0])-len(insargs[3])==2:
        Ani = True

    Sig = 0.
    if Method=='Vol':
        if PreComp or (not SynthDiag_dX12 is None and np.all(dX12==SynthDiag_dX12) and dX12Mode==SynthDiag_dX12Mode and ds==SynthDiag_ds and dsMode==SynthDiag_dsMode and MarginS==SynthDiag_MarginS and Colis==SynthDiag_Colis):
            Points, SAng, Vect, dV = SynthDiag_Points, SynthDiag_SAng, SynthDiag_Vect, SynthDiag_dV
            Sig = dV*np.sum(ff(Points,Vect,**extargs)*SAng) if Ani else dV*np.sum(ff(Points,**extargs)*SAng)
        elif Mode=='quad':
            print "Calc_Sig quad => to be checked !!!!"
            LPolys = [DPoly]+LOPolys
            P, u = self.LOS[self._LOSRef]['LOS'].D, self.LOS[self._LOSRef]['LOS'].u
            e1, e2 = GG.Calc_DefaultCheck_e1e2_PLane_1D(P, u)
            (MinX1, MaxX1), (MinX2, MaxX2) = ConeWidth_X1.min(axis=1), ConeWidth_X2.min(axis=1)
            PSA, nPSA, e1SA, e2SA = SAngPlane
            if Ani:
                if Colis:
                    if OpType=='Apert':
                        def FF(x1,x2,s,**eargs):
                            Pt = P + s*nP + x1*e1 + x2*e2
                            if GG.Calc_InOut_LOS_Colis_1D(P[0],P[1],P[2], Pt[0],Pt[1],Pt[2], VPoly, VVin, DLong=DLong, VType=VType):
                                SAng, Vect = GG.Calc_SAngVect_LPolys1Point_Flex(LPolys, Pt, PSA, nPSA, e1SA, e2SA, VectReturn=True)
                                return ff(Pt,Vect,**eargs)*SAng
                            else:
                                return 0.
                    elif OpType=='Lens':
                        tanthetmax = np.tan(Lens_ConeHalfAng)
                        def FF(x1,x2,s,**eargs):
                            Pt = P + s*nP + x1*e1 + x2*e2
                            if GG.Calc_InOut_LOS_Colis_1D(P[0],P[1],P[2], Pt[0],Pt[1],Pt[2], VPoly, VVin, DLong=DLong, VType=VType):
                                SAng, Vect = GG.Calc_SAngVect_LPolys1Point_Flex_Lens(LOBaryS[0][0],LOBaryS[0][1],LOBaryS[0][2], Lens_ConeTip[0],Lens_ConeTip[1],Lens_ConeTip[2], LOnIn[0][0],LOnIn[0][1],LOnIn[0][2],
                                        Pt[0],Pt[1],Pt[2], RadL, RadD, F1, tanthetmax, LOPolys[0][0,:],LOPolys[0][1,:],LOPolys[0][2,:], thet=thet, VectReturn=True)
                                return ff(Pt,Vect,**eargs)*SAng
                            else:
                                return 0.
                else:
                    if OpType=='Apert':
                        def FF(x1,x2,s,**eargs):
                            Pt = P + s*nP + x1*e1 + x2*e2
                            SAng, Vect = GG.Calc_SAngVect_LPolys1Point_Flex(LPolys, Pt, PSA, nPSA, e1SA, e2SA, VectReturn=True)
                            return ff(Pt,Vect,**eargs)*SAng
                    elif OpType=='Lens':
                        tanthetmax = np.tan(Lens_ConeHalfAng)
                        def FF(x1,x2,s,**eargs):
                            Pt = P + s*nP + x1*e1 + x2*e2
                            SAng, Vect = GG.Calc_SAngVect_LPolys1Point_Flex_Lens(LOBaryS[0][0],LOBaryS[0][1],LOBaryS[0][2], Lens_ConeTip[0],Lens_ConeTip[1],Lens_ConeTip[2], LOnIn[0][0],LOnIn[0][1],LOnIn[0][2],
                                    Pt[0],Pt[1],Pt[2], RadL, RadD, F1, tanthetmax, LOPolys[0][0,:],LOPolys[0][1,:],LOPolys[0][2,:], thet=thet, VectReturn=True)
                            return ff(Pt,Vect,**eargs)*SAng

            else:
                if Colis:
                    if OpType=='Apert':
                        def FF(x1,x2,s,**eargs):
                            Pt = P + s*nP + x1*e1 + x2*e2
                            if GG.Calc_InOut_LOS_Colis_1D(P[0],P[1],P[2], Pt[0],Pt[1],Pt[2], VPoly, VVin, DLong=DLong, VType=VType):
                                SAng = GG.Calc_SAngVect_LPolys1Point_Flex(LPolys, Pt, PSA, nPSA, e1SA, e2SA, VectReturn=False)
                                return ff(Pt,**eargs)*SAng
                            else:
                                return 0.
                    elif OpType=='Lens':
                        tanthetmax = np.tan(Lens_ConeHalfAng)
                        def FF(x1,x2,s,**eargs):
                            Pt = P + s*nP + x1*e1 + x2*e2
                            if GG.Calc_InOut_LOS_Colis_1D(P[0],P[1],P[2], Pt[0],Pt[1],Pt[2], VPoly, VVin, DLong=DLong, VType=VType):
                                SAng = GG.Calc_SAngVect_LPolys1Point_Flex_Lens(LOBaryS[0][0],LOBaryS[0][1],LOBaryS[0][2], Lens_ConeTip[0],Lens_ConeTip[1],Lens_ConeTip[2], LOnIn[0][0],LOnIn[0][1],LOnIn[0][2],
                                        Pt[0],Pt[1],Pt[2], RadL, RadD, F1, tanthetmax, LOPolys[0][0,:],LOPolys[0][1,:],LOPolys[0][2,:], thet=thet, VectReturn=False)
                                return ff(Pt,**eargs)*SAng
                            else:
                                return 0.
                else:
                    if OpType=='Apert':
                        def FF(x1,x2,s,**eargs):
                            Pt = P + s*nP + x1*e1 + x2*e2
                            SAng = GG.Calc_SAngVect_LPolys1Point_Flex(LPolys, Pt, PSA, nPSA, e1SA, e2SA, VectReturn=False)
                            return ff(Pt,**eargs)*SAng
                    elif OpType=='Lens':
                        tanthetmax = np.tan(Lens_ConeHalfAng)
                        def FF(x1,x2,s,**eargs):
                            Pt = P + s*nP + x1*e1 + x2*e2
                            SAng = GG.Calc_SAngVect_LPolys1Point_Flex_Lens(LOBaryS[0][0],LOBaryS[0][1],LOBaryS[0][2], Lens_ConeTip[0],Lens_ConeTip[1],Lens_ConeTip[2], LOnIn[0][0],LOnIn[0][1],LOnIn[0][2],
                                    Pt[0],Pt[1],Pt[2], RadL, RadD, F1, tanthetmax, LOPolys[0][0,:],LOPolys[0][1,:],LOPolys[0][2,:], thet=thet, VectReturn=False)
                            return ff(Pt,**eargs)*SAng
            minx1 = lambda x: MinX1
            maxx1 = lambda x: MaxX1
            minx2 = lambda x: MinX2
            maxx2 = lambda x: MaxX2
            aa = []
            for kk in extargs.keys:
                aa.append(extargs[kk])
            Sig = GG.tplquad_custom(FF, Span_k[0], Span_k[1], minx1, maxx1, minx2, maxx2, args=tuple(aa), epsrel=epsrel)

        else:
            if Mode=='sum':
                Pts, dV = Calc_SynthDiag_SampleVolume(LOSD=LOSD, LOSu=LOSu, Span_k=Span_k, ConeWidth_k=ConeWidth_k, ConeWidth_X1=ConeWidth_X1, ConeWidth_X2=ConeWidth_X2,
                        dX12=dX12, dX12Mode=dX12Mode, ds=ds, dsMode=dsMode, MarginS=MarginS, Detail=False)
            else:
                Pts, dV, X1, X2, Ss = Calc_SynthDiag_SampleVolume(LOSD=LOSD, LOSu=LOSu, Span_k=Span_k, ConeWidth_k=ConeWidth_k, ConeWidth_X1=ConeWidth_X1, ConeWidth_X2=ConeWidth_X2,
                        dX12=dX12, dX12Mode=dX12Mode, ds=ds, dsMode=dsMode, MarginS=MarginS, Detail=True)
                NumX1, NumX2, Nums = X1.size, X2.size, Ss.size

            SAng, Vect = _Detect_SAngVect_Points(Pts, DPoly=DPoly, DBaryS=DBaryS, DnIn=DnIn, LOBaryS=LOBaryS, LOnIns=LOnIns, LOPolys=LOPolys, SAngPlane=SAngPlane,
                    Lens_ConeTip=Lens_ConeTip,Lens_ConeHalfAng=Lens_ConeHalfAng, RadL=RadL, RadD=RadD, F1=F1, thet=thet, OpType=OpType, VPoly=VPoly, VVin=VVin, DLong=DLong, VType=VType,
                    Cone_PolyCrossbis=Cone_PolyCrossbis, Cone_PolyHorbis=Cone_PolyHorbis, TorAngRef=CrossRef, Colis=True,Test=Test)
            Emiss = ff(Pts,Vect,**extargs)*SAng if Ani else ff(Pts,**extargs)*SAng
            if Mode=='sum':
                Sig = dV * np.sum(Emiss)
            elif Mode=='simps':
                Sig = GG.tplsimps_custom(Emiss.reshape((NumX1,NumX2,Nums)),x=X1,y=X2,z=Ss)
            elif Mode=='trapz':
                Sig = GG.tpltrapz_custom(Emiss.reshape((NumX1,NumX2,Nums)),x=X1,y=X2,z=Ss)
            elif Mode=='nptrapz':
                Sig = np.trapz(np.trapz(np.trapz(Emiss.reshape((NumX1,NumX2,Nums)),x=Ss),x=X2),x=X1)

    # If LOS, no computation of SAng, Vect=-LOSu, Information about OpType is in Etend, Colis determines the length of the LOS
    else:
        if Mode=='quad':
            LPolys = [DPoly]+LOPolys
            P, nP, e1, e2 = SAngPlane
            s1 = LOSkPIn+MarginS
            s2 = LOSkPOut if Colis else Span_k[1]
            D, u = LOSD.reshape((3,1)), LOSu.reshape((3,1))
            Neargs = len(extargs.keys())
            assert Neargs<=1, "With quad method, only one extra argument can be passed !"
            if Ani:
                if Neargs==0:
                    def FF(s):
                        return ff(D+s*u,-u)
                else:
                    def FF(s, earg):
                        return ff(D+s*u,-u,earg)
            else:
                if Neargs==0:
                    def FF(s):
                        return ff(D+s*u)
                else:
                    def FF(s, earg):
                        return ff(D+s*u,earg)
            aa = [] if Neargs==0 else [extargs[extargs.keys()[0]]]
            Sig = scpinteg.quad(FF, s1, s2, args=tuple(aa), epsrel=epsrel)[0]*LOSEtend
        else:
            k2 = LOSkPOut if Colis else Span_k[1]
            Nums = int(np.ceil(1./ds)) if dsMode=='rel' else int(np.ceil((k2-LOSkPIn)/ds))
            Ss = np.linspace(LOSkPIn+MarginS, k2, Nums)
            D, u = LOSD, LOSu
            Pts = np.array([D[0]+u[0]*Ss, D[1]+u[1]*Ss, D[2]+u[2]*Ss])
            Emiss = ff(Pts,-np.tile(u,(Nums,1)).T,**extargs) if Ani else ff(Pts,**extargs)
            if Mode=='simps':
                Sig = scpinteg.simps(Emiss,x=Ss)*LOSEtend
            elif Mode=='trapz':
                Sig = scpinteg.trapz(Emiss,x=Ss)*LOSEtend
            elif Mode=='nptrapz':
                Sig = np.trapz(Emiss,x=Ss)*LOSEtend
            elif Mode=='sum':
                ds = np.mean(np.diff(Ss))
                Sig = np.sum(Emiss)*ds*LOSEtend
    return Sig









"""  Not used ?
def Calc_SynthDiag_GDetect(GD, ff, Method='Vol', Mode='simps', PreComp=True, epsrel=TFD.DetSynthEpsrel, dX12=TFD.DetSynthdX12, dX12Mode=TFD.DetSynthdX12Mode, ds=TFD.DetSynthds, dsMode=TFD.DetSynthdsMode, MarginS=TFD.DetSynthMarginS, Colis=TFD.DetCalcSAngVectColis, LOSRef='Cyl', Test=True):
    if Test:
        assert isinstance(GD,GDetect) or isinstance(GD,Detect) or (type(GD) is list and all([isinstance(dd,Detect) for dd in GD])), "Arg GD must be a GDetect or a Detect instance or a list of Detect instances !"
        assert hasattr(ff, '__call__'), "Arg ff must be a callable (function of one or two arguments) !"
        assert type(Colis) is bool, "Arg Colis must be a bool !"
        assert Mode in ['quad','simps','trapz','nptrapz','sum'], "Arg Mode must be in ['quad','simps','trapz'] !"
        assert dX12Mode in ['rel','abs'], "Arg dX12Mode must be in ['rel','abs'] !"
        assert dsMode in ['rel','abs'], "Arg dsMode must be in ['rel','abs'] !"
        assert (type(dX12) in [tuple,list] and len(dX12)==2) or (type(dX12) is np.ndarray and dX12.size==2), "Arg dX12 must be a list, tuple or np.ndarray of 2 floats !"
        assert type(ds) is float, "Arg ds must be a float !"
        assert Method in ['Vol','LOS']

    if isinstance(GD,GDetect):
        GD = GD.LDetect
    elif isinstance(GD,Detect):
        GD = [GD]
    nD = len(GD)

    try:
        Emiss = ff(np.ones((3,2)),np.ones((3,2)))
        Ani = True
    except Exception:
        Ani = False

    Sig = np.nan*np.ones((nD,))
    if Method=='Vol':
        if PreComp or (not GD[0]._SynthDiag_dX12 is None and all([np.all(dX12==dd._SynthDiag_dX12) and dX12Mode==dd._SynthDiag_dX12Mode and ds==dd._SynthDiag_ds and dsMode==dd._SynthDiag_dsMode and Colis==dd._SynthDiag_Colis and MarginS==dd._SynthDiag_MarginS for dd in GD])):
            for ii in range(0,nD):
                Points, SAng, Vect, dV = GD[ii]._SynthDiag_Points, GD[ii]._SynthDiag_SAng, GD[ii]._SynthDiag_Vect, GD[ii]._SynthDiag_dV
                Sig[ii] = dV*np.sum(ff(Points,Vect)*SAng) if Ani else dV*np.sum(ff(Points)*SAng)
        elif Mode=='quad':
            print "Calc_Sig quad => to be checked !!!!"
            for ii in range(0,nD):
                LPolys = [GD[ii].Poly]+[aa.Poly for aa in GD[ii].LApert]
                P, nP, e1, e2 = P, u = self.LOS[self._LOSRef]['LOS'].D, self.LOS[self._LOSRef]['LOS'].u
                e1, e2 = GG.Calc_DefaultCheck_e1e2_PLane_1D(P, u)
                (MinX1, MaxX1), (MinX2, MaxX2) = self._ConeWidth_X1.min(axis=1), self._ConeWidth_X2.min(axis=1)
                PSA, nPSA, e1SA, e2SA = GD[ii]._SAngPlane
                if Ani:
                    if Colis:
                        def FF(x1,x2,s,ii=ii):
                            Point = P + s*nP + x1*e1 + x2*e2
                            if GG.Calc_InOut_LOS_Colis_1D(self._SAngPlane[0], Point, GD[ii].Ves.Poly, GD[ii].Ves._Vin):
                                SAng, Vect = GG.Calc_SAngVect_LPolysPoints_Flex(LPolys, Point, PSA, nPSA, e1SA, e2SA)
                                return ff(Point,Vect)*SAng
                            else:
                                return 0.
                                SAng, Vect = GG.Calc_SAngVect_LPolysPoints_Flex(LPolys, Point, PSA, nPSA, e1SA, e2SA)
                    else:
                        def FF(x1,x2,s,ii=ii):
                            Point = P + s*nP + x1*e1 + x2*e2
                            SAng, Vect = GG.Calc_SAngVect_LPolysPoints_Flex(LPolys, Point, PSA, nPSA, e1SA, e2SA)
                            return ff(Point,Vect)*SAng
                else:
                    if Colis:
                        def FF(x1,x2,s,ii=ii):
                            Point = D + s*nP + x1*e1 + x2*e2
                            if GG.Calc_InOut_LOS_Colis_1D(self._SAngPlane[0], Point, GD[ii].Ves.Poly, GD[ii].Ves._Vin):
                                return ff(Point) * GG.Calc_SAngVect_LPolysPoints_Flex(LPolys, Point, PSA, nPSA, e1SA, e2SA)[0]
                            else:
                                return 0.
                    else:
                        def FF(x1,x2,s,ii=ii):
                            Point = D + s*nP + x1*e1 + x2*e2
                            return ff(Point) * GG.Calc_SAngVect_LPolysPoints_Flex(LPolys, Point, PSA, nPSA, e1SA, e2SA)[0]
                Sig[ii] = GG.tblquad_custom(FF, self._Span_k[0], self._Span_k[1], x11, x12, x21, x22, epsrel=epsrel)
        else:
            for ii in range(0,nD):
                if Mode=='sum':
                    Points, dV = Calc_SynthDiag_SampleVolume(GD[ii],dX12=dX12,ds=ds,dX12Mode=dX12Mode,dsMode=dsMode, Detail=False)
                else:
                    Points, dV, X1, X2, Ss = Calc_SynthDiag_SampleVolume(GD[ii],dX12=dX12,ds=ds,dX12Mode=dX12Mode,dsMode=dsMode, Detail=True)
                NumX1, NumX2, Nums = X1.size, X2.size, Ss.size
                SAng, Vect = np.zeros((NumX1*NumX2*Nums,)), np.zeros((3,NumX1*NumX2*Nums))
                LPolys = [GD[ii].Poly]+[aa.Poly for aa in GD[ii].LApert]
                ind = GD[ii]._isOnGoodSide(Points) & GD[ii].isInside(Points) if Colis else GD[ii]._isOnGoodSide(Points)
                if np.any(ind):
                    SAng[ind], Vect[:,ind] = GG.Calc_SAngVect_LPolysPoints_Flex(LPolys, Points[:,ind], GD[ii]._SAngPlane[0], GD[ii]._SAngPlane[1], GD[ii]._SAngPlane[2], GD[ii]._SAngPlane[3])
                    indPos = SAng>0.
                    if Colis and np.any(indPos):
                        indC = GG.Calc_InOut_LOS_Colis(GD[ii]._SAngPlane[0], Points[:,indPos], GD[ii].Ves.Poly, GD[ii].Ves._Vin)
                        indnul = indPos.nonzero()[0]
                        SAng[indnul[~indC]] = 0.
                Emiss = ff(Points,Vect)*SAng if Ani else ff(Points)*SAng
                if Mode=='sum':
                    Sig[ii] = dV * np.sum(Emiss)
                elif Mode=='simps':
                    Sig[ii] = GG.tplsimps_custom(Emiss.reshape((NumX1,NumX2,Nums)),x=X1,y=X2,z=Ss)
                elif Mode=='trapz':
                    Sig[ii] = GG.tpltrapz_custom(Emiss.reshape((NumX1,NumX2,Nums)),x=X1,y=X2,z=Ss)
                elif Mode=='nptrapz':
                    Sig[ii] = np.trapz(np.trapz(np.trapz(Emiss.reshape((NumX1,NumX2,Nums)),x=Ss),x=X2),x=X1)

    else:
        if Mode=='quad':
            for ii in range(0,nD):
                LPolys = [GD[ii].Poly]+[aa.Poly for aa in GD[ii].LApert]
                P, nP, e1, e2 = GD[ii]._SAngPlane
                if Colis:
                    s1, s2 = GD[ii].LOS[LOSRef]['LOS'].kPIn+MarginS, GD[ii].LOS[LOSRef]['LOS'].kPOut
                else:
                    s1, s2 = GD[ii].LOS[LOSRef]['LOS'].kPIn+MarginS, GD[ii]._Span_k[1]
                D, u = GD[ii].LOS[LOSRef]['LOS'].D.reshape((3,1)), GD[ii].LOS[LOSRef]['LOS'].u.reshape((3,1))
                if Ani:
                    def FF(s,D=D,u=u):
                        return ff(D+s*u,-u)
                else:
                    def FF(s,D=D,u=u):
                        return ff(D + s*u)
                Sig[ii] = scpinteg.quad(FF, s1, s2, epsrel=epsrel)[0]*GD[ii].LOS[LOSRef]['Etend']
        else:
            for ii in range(0,nD):
                k2 = GD[ii].LOS[LOSRef]['LOS']kPOut if Colis else GD[ii]._Span_k[1]
                Nums = int(np.ceil(1./ds)) if dsMode=='rel' else int(np.ceil((k2-GD[ii].LOS[LOSRef]['LOS'].kPIn)/ds))
                Ss = np.linspace(GD[ii].LOS[LOSRef]['LOS'].kPIn+MarginS, k2, Nums)
                D, u = GD[ii].LOS[LOSRef]['LOS'].D, GD[ii].LOS[LOSRef]['LOS'].u
                Points = np.array([D[0]+u[0]*Ss, D[1]+u[1]*Ss, D[2]+u[2]*Ss])
                Emiss = ff(Points,-np.tile(GD[ii].LOS[LOSRef]['LOS'].u,(Nums,1)).T) if Ani else ff(Points)
                if Mode=='simps':
                    Sig[ii] = scpinteg.simps(Emiss,x=Ss)*GD[ii].LOS[LOSRef]['Etend']
                elif Mode=='trapz':
                    Sig[ii] = scpinteg.trapz(Emiss,x=Ss)*GD[ii].LOS[LOSRef]['Etend']
                elif Mode=='nptrapz':
                    Sig[ii] = np.trapz(Emiss,x=Ss)*GD[ii].LOS[LOSRef]['Etend']
                elif Mode=='sum':
                    ds = np.mean(np.diff(Ss))
                    Sig[ii] = np.sum(Emiss)*ds*GD[ii].LOS[LOSRef]['Etend']
    return Sig
"""




def Calc_Etendue_AlongLOS(D, NP=5, Length='POut', Mode='quad', kMode='rel', Colis=TFD.DetSAngColis, epsrel=TFD.DetEtendepsrel, dX12=TFD.DetEtenddX12, dX12Mode=TFD.DetEtenddX12Mode, LOSRef=None): # Used - to be corrected !
    assert isinstance(D,Detect), "Arg D should be Detect instance !"
    assert type(NP) is int, "Arg NP must be an int !"
    assert Mode == 'simps' or Mode == 'trapz' or Mode == 'quad', "Arg Mode should be 'simps', 'trapz' or 'quad' for indicating the integration method to use !"
    assert Length in ['POut','kMax'], "Arg Length must be in ['POut','kMax'] !"
    assert kMode in ['rel','abs'], "Arg kMode must be in ['norm','abs'] !"
    assert LOSRef is None or LOSRef in D.LOS.keys(), "Arg LOSRef must be one of the LOS keys of D !"
    LOSRef = D._LOSRef if LOSRef is None else LOSRef
    DS = 1./(NP+2)
    PIN, POUT, U = np.copy(D.LOS[LOSRef]['LOS'].PIn), np.copy(D.LOS[LOSRef]['LOS'].POut), np.copy(D.LOS[LOSRef]['LOS'].u)
    if Length=='kMax':
        POUT = np.copy(D.LOS[LOSRef]['LOS'].D + D._Span_k[1]*U)
    Dk = np.linalg.norm(POUT-PIN)
    kPoints = np.linspace(0,Dk,NP+2)
    Points = np.array([PIN[0]+np.linspace(0,Dk,NP+2)*U[0], PIN[1]+np.linspace(0,Dk,NP+2)*U[1], PIN[2]+np.linspace(0,Dk,NP+2)*U[2]])
    Points = np.delete(Points,[0,NP+1],axis=1)
    kPoints = np.delete(kPoints,[0,NP+1])
    if kMode=='rel':
        kPoints = kPoints/Dk
    nPbis = np.tile(U,(Points.shape[1],1)).T
    Etend, e1, e2, err = Calc_Etendue_PlaneLOS(D, Points, nPbis, Mode=Mode, dX12=dX12, dX12Mode=dX12Mode, epsrel=epsrel, e1=None,e2=None, Colis=Colis, Test=True)



    Calc_Etendue_PlaneLOS(Ps, nPs, DPoly, DBaryS, DnIn, LOPolys, LOnIns, LOSurfs, LOBaryS, SAngPlane, VPoly, VVin, DLong=None, Lens_ConeTip=None, Lens_ConeHalfAng=None, RadL=None, RadD=None, F1=None,
        OpType='Apert', VType='Tor', Mode='quad', epsrel=TFD.DetEtendepsrel, dX12=TFD.DetEtenddX12, dX12Mode=TFD.DetEtenddX12Mode, e1=None,e2=None, Ratio=0.02, Colis=TFD.DetCalcEtendColis, Test=True)






    return Points, kPoints, Etend





def Calc_SAngNb_ProjSlice_LDetect(Pts=None, Proj='Cross', Slice='Int', DRY=TFD.DetSliceAGdR, DXTheta=TFD.DetSliceAGdTheta, DZ=TFD.DetSliceAGdZ, Colis=TFD.DetSAngColis,
        LLOSPIn=None, LOSu=None, LOSPOut=None, LSpan_R=None, LSpan_Theta=None, LSpan_X=None, LSpan_Y=None, LSpan_Z=None,
        LSAngCross_Reg=None, LSAngCross_Points=None, LSAngCross_Reg_K=None, LSAngCross_Reg_Psi=None, LSAngCross_Reg_Int=None, LSAngCross_Max=None, LSAngHor_Points=None, LSAngHor_Int=None, LSAngHor_Max=None,
        LCone_PolyCrossbis=None, LCone_PolyHorbis=None, LLens_ConeTip=None, LLens_ConeHalfAng=None, LRadL=None, LRadD=None, LF1=None, Lthet=None, LOpType=None,
        LDPoly=None, LDBaryS=None, LDnIn=None, LLOBaryS=None, LLOnIns=None, LLOPolys=None, LSAngPlane=None, VPoly=None, VVin=None, VType='Tor', Test=True):  # Used
    assert Slice in ['Int','Max'] or type(Slice) is float, "Arg Slice must be in ['Int','Max'] or a float !"
    ND = len(LSAngCross_Reg)

    # Get the mesh if Pts not provided
    if Pts is None:
        SingPts = np.vstack(tuple([np.vstack((LLOSPIn[ii], LOSPIn[ii]+0.002*LOSu[ii], 0.5*(LOSPOut[ii]+LOSPIn[ii]), LOSPOut[ii]-0.002*LOSu[ii] , LOSPOut[ii])) for ii in range(0,ND)])).T
        if VType=='Tor':
            X1, XIgn, Z, NX1, NIgn, NZ, Pts, out = _get_CrossHorMesh(SingPoints=SingPts, LSpan_R=LSpan_R, LSpan_Theta=LSpan_Theta, LSpan_Z=LSpan_Z, DR=DRY, DTheta=DXTheta, DZ=DZ, VType=VType, Proj=Proj, ReturnPts=True)
        elif VType=='Lin':
            XIgn, X1, Z, NIgn, NX1, NZ, Pts, out = _get_CrossHorMesh(SingPoints=SingPts, LSpan_X=LSpan_X, LSpan_Y=LSpan_Y, LSpan_Z=LSpan_Z, DX=DXTheta, DY=DRY, DZ=DZ, VType=VType, Proj=Proj, ReturnPts=True)

    # Compute solid angle and Nb
    SA = np.zeros((ND,Pts.shape[1]))
    if Slice in ['Int','Max']:
        for ii in range(0,ND):
            FF = _Detect_get_SAngIntMax(SAngCross_Reg=LSAngCross_Reg[ii], SAngCross_Points=LSAngCross_Points[ii], SAngCross_Reg_K=LSAngCross_Reg_K[ii], SAngCross_Reg_Psi=LSAngCross_Reg_Psi[ii],
                    SAngCross_Reg_Int=LSAngCross_Reg_Int[ii], SAngCross_Max=LSAngCross_Max[ii], SAngHor_Points=LSAngHor_Points[ii], SAngHor_Int=LSAngHor_Int[ii], SAngHor_Max=LSAngHor_Max[ii],
                    Cone_PolyCrossbis=LCone_PolyCrossbis[ii], Cone_PolyHorbis=LCone_PolyHorbis[ii], TorAngRef=None, DBaryS=LDBaryS[ii], LOSPOut=LLOSPOut[ii], Proj=Proj, SAng=Slice, VType=VType)
            SA[ii,:] = FF(Pts, In=out)
    else:
        if Proj=='Hor':
            LSpan = LSpan_Z
        else:
            LSpan = LSpan_Theta if VType=='Tor' else LSpan_X
        DXIgn = [min([LSpan[ii][0] for ii in range(0,ND)]), max([LSpan[ii][1] for ii in range(0,ND)])]
        assert Slice>=DXIgn[0] and Slice<=DXIgn[1], "Arg Slice is outside of the intervall were non-zeros values can be found !"
        Pts = GG.CoordShift(Pts, In=out, Out='(X,Y,Z)', CrossRef=Slice)
        for ii in range(0,ND):
            SA[ii,:] = _Detect_SAngVect_Points(Pts, DPoly=LDPoly[ii], DBaryS=LDBaryS[ii], DnIn=LDnIn[ii], LOBaryS=LLOBaryS[ii], LOnIns=LLOnIns[ii], LOPolys=LLOPolys[ii], SAngPlane=LSAngPlane[ii], Lens_ConeTip=LLens_ConeTip[ii],
                    Lens_ConeHalfAng=LLens_ConeHalfAng[ii], RadL=LRadL[ii], RadD=LRadD[ii], F1=LF1[ii], thet=Lthet[ii], OpType=LOpType[ii],
                    VPoly=VPoly, VVin=VVin, DLong=DLong, VType=VType, Cone_PolyCrossbis=LCone_PolyCrossbis[ii], Cone_PolyHorbis=LCone_PolyHorbis[ii], TorAngRef=Slice, Colis=Colis, Test=Test)

    Nb = np.sum(SA>0.,axis=0)
    return SAng, Nb, Pts, XIgn, X1, Z, NIgn, NX1, NZ








































































def _Calc_BaryCylFromCart(Poly, BaryS, NP1=TFD.DetBaryCylNP1, NP2=TFD.DetBaryCylNP2):
    assert isinstance(Poly,np.ndarray) and Poly.ndim==2 and Poly.shape[0]==3 and np.all(Poly[:,0]==Poly[:,-1]), "Arg pPoly must be a (3,N) np.ndarray in cartesian coordinates with first = last point !"
    BaryCyl = np.copy(BaryS)
    PX, PY, PZ, nP = Poly[0,:], Poly[1,:], Poly[2,:], Poly.shape[1]
    px, py = np.empty(((nP-1)*NP1,)), np.empty(((nP-1)*NP1,))
    for ii in range(0,nP-1):
        px[ii*NP1:(ii+1)*NP1] = np.linspace(PX[ii],PX[ii+1],NP1, endpoint=False)
        py[ii*NP1:(ii+1)*NP1] = np.linspace(PY[ii],PY[ii+1],NP1, endpoint=False)
    px, py = np.append(px,px[0]), np.append(py,py[0])
    R, Theta = np.hypot(px,py), np.arctan2(py,px)
    RMin, RMax, ThetaMin, ThetaMax = R.min(), R.max(), Theta.min(), Theta.max()
    R1, theta1 = np.linspace(RMin,RMax,NP2), np.linspace(ThetaMin,ThetaMax,NP2)
    PR, Ptheta = np.mean(np.array([R1[:-1],R1[1:]]),axis=0), np.mean(np.array([theta1[:-1],theta1[1:]]),axis=0)
    DR, DTheta = np.diff(np.array([R1[:-1],R1[1:]])**2,axis=0)/2., np.diff(np.array([theta1[:-1],theta1[1:]]),axis=0)
    (Rgrid, thetagrid), (DRgrid, DThetagrid) = np.meshgrid(PR,Ptheta), np.meshgrid(DR,DTheta)
    Points = np.array([Rgrid.flatten(),thetagrid.flatten()])
    Surfs = DRgrid.flatten()*DThetagrid.flatten()
    ind = Path(np.array([R, Theta]).T).contains_points(Points.T, transform=None, radius=0.0)
    if np.any(ind):
        Points, Surfs = Points[:,ind], Surfs[ind]
        RAv = np.sum(Points[0,:]*Surfs)/np.sum(Surfs)
        thetaAv = np.sum(Points[1,:]*Surfs)/np.sum(Surfs)
        BaryCyl = np.array([RAv*np.cos(thetaAv), RAv*np.sin(thetaAv), BaryS[2]])
    return BaryCyl




"""
def Calc_SolAngVect_DetectApert(D, Points, PP='None', nP='None', e1='None', e2='None', Colis=TFD.DetSAngColis, Test=True):      # Not used ?
    if Test:
        assert isinstance(D,Detect), "Arg D should a GDetect instance or a list of Detect instances or a Detect instance !"
        assert isinstance(Points,np.ndarray) and Points.shape[0]==3, "Arg Points should be a (3,N) ndarray representing points !"
        if nP=='None':
            NPoly = len(D.LApert)
            nP = [D.LApert[kk].nIn*D.LApert[kk].Surf for kk in range(0,len(D.LApert))]
            S = [D.LApert[kk].Surf for kk in range(0,NPoly)]
            nP = np.sum(np.array(nP).T.reshape((3,NPoly)), axis=1, keepdims=True)/np.sum(np.array(S))
            nP = nP/np.linalg.norm(nP)

        e1, e2 = GG.Calc_DefaultCheck_e1e2_PLanes(D.LApert[0].BaryS, nP, e1, e2)
        #e1, e2 = Calc_DefaultCheck_e1e2_LOSPLanes(D.LApert[0].BaryS, D.LApert[0].nIn, e1, e2)

    NP = Points.shape[1]
    AngS = np.zeros((1,NP))
    Vect = np.nan*np.ones((3,NP))
    LApert = D.LApert
    if PP=='None':
        PP = LApert[0].BaryS

    NPoly = len(LApert)+1
    if nP=='None':
        nP = [D.LApert[ii].nIn for ii in range(0,NPoly-1)]
        nP = np.mean(np.array(nP).T.reshape((3,NPoly-1)), axis=1, keepdims=True)
        nP = nP/np.linalg.norm(nP)

    Polys = [D.Poly] + [LApert[jj].Poly for jj in range(0,NPoly-1)]
    Out = GG.Calc_PolysProjPlanePoints(Polys,Points,PP,nP,e1P=e1,e2P=e2, Test=False)

    indok, indwrg = ~Out[5], Out[5]
    IndOK, IndWRG, nok, nwrg = indok.nonzero()[0], indwrg.nonzero()[0], np.sum(indok), np.sum(indwrg)
    #print "Number of points requiring specific planes for solid angle computation for "+D.Id.Name+" = "+str(nwrg)+" / "+str(NP)

    if not Colis:
        for kk in range(0,nok):
            PolyInt = [np.concatenate((Out[3][jj][IndOK[kk]:IndOK[kk]+1,:],Out[4][jj][IndOK[kk]:IndOK[kk]+1,:]),axis=0) for jj in range(0,NPoly)]
            PolyRef = plg.Polygon(PolyInt[0].T)
            for ii in range(1,NPoly):
                PolyRef = PolyRef & plg.Polygon(PolyInt[ii].T)
            if PolyRef.area()>1e-20:
                PolyInt = np.array(PolyRef[0]).T
                PolyInt = np.concatenate((PolyInt,PolyInt[:,0:1]),axis=1)
                BaryS = PolyRef.center()
                BaryS = PP + e1*BaryS[0] + e2*BaryS[1]
                PolyInt = np.dot(PP,np.ones((1,PolyInt.shape[1]))) + np.dot(e1,PolyInt[0:1,:]) + np.dot(e2,PolyInt[1:2,:])
                AngS[0,IndOK[kk]], Vect[:,IndOK[kk]:IndOK[kk]+1] = GG.Calc_SolAngVect_PointsOnePoly(Points[:,IndOK[kk]:IndOK[kk]+1], PolyInt, G=BaryS, Test=False)      # Replace by cython routines ?

        for kk in range(0,nwrg):
            nPbis = Points[:,IndWRG[kk]:IndWRG[kk]+1]-PP
            nPbis = nPbis/np.linalg.norm(nPbis)
            e1bis, e2bis = GG.Calc_DefaultCheck_e1e2_PLanes(D.LApert[0].BaryS, nPbis)
            out = GG.Calc_PolysProjPlanePoints(Polys,Points[:,IndWRG[kk]:IndWRG[kk]+1], PP, nPbis, e1P=e1bis, e2P=e2bis, Test=False)
            if np.any(out[5]):#, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"                                                                               # DB
                print "Points are :", Points[:,IndWRG[kk]:IndWRG[kk]+1], np.sqrt(Points[0,IndWRG[kk]:IndWRG[kk]+1]**2+Points[1,IndWRG[kk]:IndWRG[kk]+1]**2) # DB
            PolyInt = [np.concatenate((out[3][jj],out[4][jj]),axis=0) for jj in range(0,NPoly)]
            PolyRef = plg.Polygon(PolyInt[0].T)
            for ii in range(1,NPoly):
                PolyRef = PolyRef & plg.Polygon(PolyInt[ii].T)
            if PolyRef.area()>1e-20:
                PolyInt = np.array(PolyRef[0]).T
                PolyInt = np.concatenate((PolyInt,PolyInt[:,0:1]),axis=1)
                BaryS = PolyRef.center()
                BaryS = PP + e1bis*BaryS[0] + e2bis*BaryS[1]
                PolyInt = np.dot(PP,np.ones((1,PolyInt.shape[1]))) + np.dot(e1bis,PolyInt[0:1,:]) + np.dot(e2bis,PolyInt[1:2,:])
                AngS[0,IndWRG[kk]], Vect[:,IndWRG[kk]:IndWRG[kk]+1] = GG.Calc_SolAngVect_PointsOnePoly(Points[:,IndWRG[kk]:IndWRG[kk]+1], PolyInt, G=BaryS, Test=False)
    else:
        for kk in range(0,nok):
            PolyInt = [np.concatenate((Out[3][jj][IndOK[kk]:IndOK[kk]+1,:],Out[4][jj][IndOK[kk]:IndOK[kk]+1,:]),axis=0) for jj in range(0,NPoly)]

            PolyRef = plg.Polygon(PolyInt[0].T)
            for ii in range(1,NPoly):
                PolyRef = PolyRef & plg.Polygon(PolyInt[ii].T)

            if PolyRef.area()>1e-20:
                PolyInt = np.array(PolyRef[0]).T
                PolyInt = np.concatenate((PolyInt,PolyInt[:,0:1]),axis=1)
                BaryS = PolyRef.center()
                BaryS = PP + e1*BaryS[0] + e2*BaryS[1]
                du = (BaryS - Points[:,IndOK[kk]:IndOK[kk]+1])
                if GG.Calc_InOut_LOS(Points[:,IndOK[kk]:IndOK[kk]+1], du/np.linalg.norm(du), D.Ves, Test=True)[0].shape[1]==0:     # Update with GG.Calc_InOut_LOS !!!
                    PolyInt = np.dot(PP,np.ones((1,PolyInt.shape[1]))) + np.dot(e1,PolyInt[0:1,:]) + np.dot(e2,PolyInt[1:2,:])
                    AngS[0,IndOK[kk]], Vect[:,IndOK[kk]:IndOK[kk]+1] = GG.Calc_SolAngVect_PointsOnePoly(Points[:,IndOK[kk]:IndOK[kk]+1],PolyInt, G=BaryS, Test=False)

        for kk in range(0,nwrg):
            nPbis = Points[:,IndWRG[kk]:IndWRG[kk]+1]-PP
            nPbis = nPbis/np.linalg.norm(nPbis)
            e1bis, e2bis = GG.Calc_DefaultCheck_e1e2_PLanes(D.LApert[0].BaryS, nPbis)
            out = GG.Calc_PolysProjPlanePoints(Polys,Points[:,IndWRG[kk]:IndWRG[kk]+1], PP, nPbis, e1P=e1bis, e2P=e2bis, Test=False)
            assert not np.any(out[5]), "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            PolyInt = [np.concatenate((out[3][jj],out[4][jj]),axis=0) for jj in range(0,NPoly)]
            PolyRef = plg.Polygon(PolyInt[0].T)
            for ii in range(1,NPoly):
                PolyRef = PolyRef & plg.Polygon(PolyInt[ii].T)
            if PolyRef.area()>1e-20:
                PolyInt = np.array(PolyRef[0]).T
                PolyInt = np.concatenate((PolyInt,PolyInt[:,0:1]),axis=1)
                BaryS = PolyRef.center()
                BaryS = PP + e1bis*BaryS[0] + e2bis*BaryS[1]
                du = (BaryS - Points[:,IndWRG[kk]:IndWRG[kk]+1])
                if GG.Calc_InOut_LOS(Points[:,IndWRG[kk]:IndWRG[kk]+1], du/np.linalg.norm(du), D.Ves, Test=True)[0].shape[1]==0:     # Update with GG.Calc_InOut_LOS !!!
                    PolyInt = np.dot(PP,np.ones((1,PolyInt.shape[1]))) + np.dot(e1bis,PolyInt[0:1,:]) + np.dot(e2bis,PolyInt[1:2,:])
                    AngS[0,IndWRG[kk]], Vect[:,IndWRG[kk]:IndWRG[kk]+1] = GG.Calc_SolAngVect_PointsOnePoly(Points[:,IndWRG[kk]:IndWRG[kk]+1], PolyInt, G=BaryS, Test=False)

    return AngS, Vect

def Calc_DefaultCheck_e1e2_LOSPLanes(PIn, nP, e1, e2, CrossNTHR=0.01,EPS=1.e-10):       # Not used ? (replaced by GG....)

    NPlans = nP.shape[1]
    if e1=='None':
        RPL = np.linalg.norm(PIn[0:2,0])
        thetaPL = np.arccos(PIn[0,0]/RPL)
        if PIn[1,0]<0:
            thetaPL = -thetaPL
        e1 = np.array([[-np.sin(thetaPL)],[np.cos(thetaPL)],[0.]])

        CrossNorm = np.array([e1[1]*nP[2,:] - e1[2]*nP[1,:], e1[2]*nP[0,:] - e1[0]*nP[2,:], e1[0]*nP[1,:] - e1[1]*nP[0,:]])
        CrossNorm = np.sum(CrossNorm**2,axis=0)
        e1 = np.dot(e1,np.ones((1,NPlans)))
        ind = CrossNorm < CrossNTHR
        if np.any(ind):
            e1[:,ind] = np.dot(np.array([[np.cos(thetaPL)],[np.sin(thetaPL)],[0.]]),np.ones((1,np.sum(ind))))
        e1 = e1 - np.dot(np.ones((3,1)),np.sum(nP*e1,axis=0,keepdims=True))*nP
        e1 = e1/np.dot(np.ones((3,1)),np.sqrt(np.sum(e1*e1,axis=0,keepdims=True)))
    else:
        assert np.all(np.abs(np.sqrt(np.sum(e1*e1,axis=0))-1) < EPS) and np.all(np.sum(nP*e1,axis=0)<EPS), "Arg e1 should be normalised and perpendicular to nP !"

    if e2=='None':
        e2 = np.array([nP[1,:]*e1[2,:] - nP[2,:]*e1[1,:], nP[2]*e1[0,:] - nP[0]*e1[2,:], nP[0]*e1[1,:] - nP[1]*e1[0,:]])
        e2 = e2/np.dot(np.ones((3,1)),np.sqrt(np.sum(e2**2,axis=0,keepdims=True)) )
    else:
        assert np.all(np.abs(np.sqrt(np.sum(e2*e2,axis=0))-1) < EPS) and np.all(np.sum(nP*e2,axis=0)<EPS) and np.all(np.sum(e2*e1,axis=0)<EPS), "Arg e2 should be normalised, perp. to nP and to e1 !"
    return e1, e2
"""





def Calc_SAngOnPlane(D,P,nP, dX12=[0.01,0.01], dX12Mode='rel', e1=None,e2=None, Ratio=0.01, Colis=TFD.DetSAngColis, Test=True):
    if Test:
        assert isinstance(D,Detect), "Arg D should be Detect instance !"
        assert isinstance(P,np.ndarray) and P.shape==(3,), "Arg P should be a (3,N) ndarray !"
        assert isinstance(nP,np.ndarray) and nP.shape==P.shape, "Arg nP should be a (3,N) ndarray !"
    F = lambda VV: VV.reshape((3,1)) if VV.ndim==1 else VV
    Ps, nPs = F(P), F(nP)
    nPnorm = np.sum(nPs**2,axis=0)
    nPs = np.array([nPs[0,:]/nPnorm, nPs[1,:]/nPnorm, nPs[2,:]/nPnorm])
    nPlans = nPs.shape[1]
    e1, e2 = GG.Calc_DefaultCheck_e1e2_PLanes_2D(Ps, nPs, e1, e2)

    Out = Calc_ViewConePointsMinMax_PlanesDetectApert_2Steps(D,Ps,nPs,e1=e1,e2=e2,Test=False)
    MinX1, MinX2, MaxX1, MaxX2 = Out[0], Out[1], Out[2], Out[3]
    MinX1 = MinX1 - Ratio*(MaxX1-MinX1)
    MaxX1 = MaxX1 + Ratio*(MaxX1-MinX1)
    MinX2 = MinX2 - Ratio*(MaxX2-MinX2)
    MaxX2 = MaxX2 + Ratio*(MaxX2-MinX2)
    NPoly = len(D.LApert)
    PBary, nPtemp, e1bis, e2bis = D._SAngPlane
    Polys = [D.Poly] + [D.LApert[jj].Poly for jj in range(0,NPoly)]

    if dX12Mode=='rel':
        NumX1 = np.ceil(1./dX12[0])*np.ones(MinX1.shape)
        NumX2 = np.ceil(1./dX12[1])*np.ones(MinX2.shape)
    else:
        NumX1 = np.ceil((MaxX1-MinX1)/dX12[0])
        NumX2 = np.ceil((MaxX2-MinX2)/dX12[1])
    NumP = NumX1*NumX2
    SA, X1L, X2L = [], [], []
    for ii in range(0,nPlans):
        sa = np.zeros((NumP[ii],))
        X1 = np.linspace(MinX1[ii],MaxX1[ii],NumX1[ii],endpoint=True)
        X2 = np.linspace(MinX2[ii],MaxX2[ii],NumX2[ii],endpoint=True)
        x1, x2 = np.tile(X1,(NumX2[ii],1)).T.flatten(), np.tile(X2,(NumX1[ii],1)).flatten()
        Pps = np.array([Ps[0,ii]+e1[0,ii]*x1+e2[0,ii]*x2, Ps[1,ii]+e1[1,ii]*x1+e2[1,ii]*x2, Ps[2,ii]+e1[2,ii]*x1+e2[2,ii]*x2])
        ind = D._isOnGoodSide(Pps) & D.isInside(Pps) if Colis else D._isOnGoodSide(Pps)
        sa[ind] = GG.Calc_SAngVect_LPolysPoints_Flex(Polys, Pps[:,ind],  PBary, nPtemp, e1bis, e2bis)[0]
        indPos = sa>0.
        if Colis and np.any(indPos):
            if np.sum(indPos)==1:
                indC = GG.Calc_InOut_LOS_Colis(PBary, Pps[:,indPos].reshape((3,np.sum(indPos))), D.Ves.Poly, D.Ves._Vin)
            else:
                indC = GG.Calc_InOut_LOS_Colis(PBary, Pps[:,indPos], D.Ves.Poly, D.Ves._Vin)
            indPos = indPos.nonzero()[0]
            sa[indPos[~indC]] = 0.
        SA.append(sa)
        X1L.append(X1)
        X2L.append(X2)
    return SA, X1L, X2L, NumX1, NumX2


