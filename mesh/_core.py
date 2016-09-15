# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:03:58 2014

@author: didiervezinet
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Polygon as plg
import scipy.sparse as scpsp
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection as PcthColl
import datetime as dtm
import scipy.interpolate as scpinterp

#from mayavi import mlab



# ToFu-specific
import tofu.defaults as tfd
import tofu.pathfile as tfpf
from . import _bsplines_cy as _tfm_bs
from . import _compute as _tfm_c
from . import _plot as _tfm_p


__all__ = ['LinMesh','LinMesh_List','Mesh1D','Mesh2D','LBF1D']



"""
###############################################################################
###############################################################################
                        Default
###############################################################################
"""

BS2Dict_Def = {'linewidth':0.,'rstride':1,'cstride':1, 'antialiased':False}
Tot2Dict_Def = {'color':(0.7,0.7,0.7),'linewidth':0.,'rstride':1,'cstride':1, 'antialiased':False}
SuppDict_Def = {'facecolor':(0.8,0.8,0.8), 'lw':0.}
PMaxDict_Def = {'color':'g', 'marker':'s', 'markersize':8, 'linestyle':'None', 'linewidth':1.}


"""
###############################################################################
###############################################################################
                        Mesh definitions
###############################################################################
"""

############################################
#####     Helper functions
############################################



def LinMesh(X1,X2,Res1,Res2, DRes=tfd.L1DRes, Mode=tfd.L1Mode, Test=True):
    """
    Create a linearly variable-size 1D mesh between between 2 points, with specified resolution (i.e. mesh size) in the neighbourhood of each point

    Inputs:
    -------
        X1      float / np.float64  First point of the interval to be meshed
        X2      float / np.float64  Last point of the interval to be meshed
        Res1    float / np.float64  Mesh size to be achieved in the vicinity of X1
        Res2    float / np.float64  Mesh size to be achieved in the vicinity of X2
        DRes    float / np.float64  Tolerable variation on the mesh size (i.e.: if abs(Res1-Res2)<DRes they are considered identical)
        Mode    str                 Flag indicating how the linear variation should be computed:
                                        'Larger':   the necessary margin is applied to side with the larger mesh size
                                        'Both':     the margin is applied to both sides
        Test    bool                Flag indicating whether the inputs should be tested for conformity

    Outputs:
    --------
        xn      np.ndarray          Array of knots of the computed 1D mesh
        Res1    float               Effectively achieved mesh size in the vicinity of X1
        Res2    float               Effectively achieved mesh size in the vicinity of X2

    """
    if Test:
        assert type(X1) in [float, np.float, np.float64], "Arg X1 must be a float !"
        assert type(X2) in [float, np.float, np.float64], "Arg X2 must be a float !"
        assert type(Res1) in [float, np.float, np.float64], "Arg Res1 must be a strictly positive float !"
        assert type(Res2) in [float, np.float, np.float64], "Arg Res2 must be a strictly positive float !"
        assert type(DRes) in [float, np.float, np.float64] and DRes>0., "Arg DRes must be a strictly positive float !"
        assert Mode=='Larger' or Mode=='Both', "Arg Mode must be 'Larger' or 'Both' !"
    Delta = abs(X2-X1)
    Sum = Res1+Res2
    if Delta < Res2 or Delta < Res1:
        xn = np.array([X1, X2])
    elif Delta < Sum:
        xn = np.array([X1, X1*Res2/Sum + X2*Res1/Sum, X2])
    elif abs(Res2 - Res1)<DRes:
        Nb = int(round(Delta/Res1))
        xn = np.linspace(X1,X2,num=Nb+1,endpoint=True)
    else:
        N = round(2.*Delta/Sum-1)
        n = np.arange(0,N+2)
        if Mode == "Larger":
            if Res1 > Res2:
                Eps = 2.*Delta/(Res1*(N+1)) - Sum/Res1
                Res1 = Res1*(1+Eps)
            else:
                Eps = 2.*Delta/(Res2*(N+1)) - Sum/Res2
                Res2 = Res2*(1+Eps)
        else:
            Eps = 2.*Delta/((N+1)*Sum) - 1
            Res1, Res2 = Res1*(1+Eps), Res2*(1+Eps)
        xn = X1 + Res1*n + (Res2-Res1)*n*(n-1)/(2*N)
    return xn, float(Res1), float(Res2)


def LinMesh_List(KnotsL, ResL, DRes=tfd.L1DRes, Mode=tfd.L1Mode, Tol=tfd.L1Tol, Concat=True, Test=True):
    """
    Create a linearly variable-size 1D mesh by concatenating several such meshes computed with LinMesh() from a list of adjacent intervals

    Inputs:
    -------
        KnotsL  list                List of intervals (tuples or lists of len()==2)
        ResL    list                List of associated resolution objectives (tuples or lists of len()==2)
        DRes    float / np.float64  Tolerable variation on the mesh size (see LinMesh())
        Mode    str                 Flag indicating how the linear variation should be computed (see LinMesh())
        Tol     float               Tolerance on the final concatenated array (to decide whether points are different or identical)
        Concat  bool                Flag indicating whether the final list of meshes should be concatenated into a unique array
        Test    bool                Flag indicating whether the inputs should be tested for conformity

    Outputs:
    --------
        X       list / np.ndarray   Final list of meshes or cancatenated array
        Res     list                List of final acheived resolutions / mesh sizes

    """
    if Test:
        assert type(KnotsL) is list and [len(K)==2 for K in KnotsL], "Arg KnotsL must be a list of intervals !"
        assert type(ResL) is list and [len(R)==2 for R in ResL], "Arg ResL must be a list of resolutions !"
        assert type(DRes) is float and DRes>0., "Arg DRes must be a strictly positive float !"
        assert Mode=='Larger' or Mode=='Both', "Arg Mode must be 'Larger' or 'Both' !"
    X, Res = [], []
    for ii in range(0,len(KnotsL)):
        x, res1, res2 = LinMesh(float(KnotsL[ii][0]), float(KnotsL[ii][1]), float(ResL[ii][0]), float(ResL[ii][1]), DRes=DRes, Mode=Mode, Test=True)
        X.append(x)
        Res.append((res1,res2))
    if Concat:
        X = np.unique(np.concatenate(tuple(X)))
        dX = np.diff(X)
        indout = (dX <= Tol).nonzero()[0]
        X = np.delete(X,indout)
    return X, Res




############################################
#####     Objects definitions
############################################


class Mesh1D(object):
    """ A class defining a 1D mesh (knot vector, with associated lengths, centers and correspondace between them, as well as plotting routines)

    Inputs:
    -------
        Knots       iterable                    The knots of the mesh
        Id          None or str or tfpf.ID      A name or tfpf.ID class, for identification
        Type        None or str                 The type of Mesh1D object (default: None)
        Exp         None or str                 The experiment to which this object relates (e.g.: "ITER", "AUG", "JET"...)
        shot        None or int                 A shot number from which this object can be used (i.e.: in case of changing geometry)
        dtime       None or dtm.datetime        A time reference to be used to identify this particular instance (used for debugging mostly), default: None
        dtimeIn     bool                        Flag indicating whether dtime should be included in the SaveName (used for debugging mostly), default: False
        SavePath    None or str                 Absolute path where the object would be saved if necessary
    """

    def __init__(self, Id, Knots, Type=None, Exp=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):

        self._Done = False
        self._set_Id(Id, Type=Type, Exp=Exp, shot=shot, dtime=dtime, dtimeIn=dtimeIn, SavePath=SavePath)
        self._set_Knots(Knots)
        self._Done = True


    @property
    def Id(self):
        "" "Return the Id """
        return self._Id
    @property
    def Knots(self):
        """ Return the knots """
        return self._Knots
    @property
    def NKnots(self):
        return self._NKnots
    @property
    def Cents(self):
        """ Return the centers of the mesh """
        return self._Cents
    @property
    def NCents(self):
        """ Return the centers of the mesh """
        return self._NCents
    @property
    def Bary(self):
        """ Return the barycenter """
        return self._Bary
    @property
    def Lengths(self):
        """ Return the vector of length of each mesh element """
        return self._Lengths
    @property
    def Length(self):
        """ Return the total length of the Mesh """
        return self._Length

    #@property
    #def Cents_Knotsind(self):
    #    """ Return the indices of all knots surrounding each center """
    #    return self._Cents_Knotsind
    #@property
    #def Knots_Centsind(self):
    #    """ Return the indices of all centers surrounding each knot """
    #    return self._Knots_Centsind


    def _check_inputs(self, Id=None, Knots=None, Type=None, Exp=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):
        _Mesh1D_check_inputs(Id=Id, Knots=Knots, Type=Type, Exp=Exp, shot=shot, dtime=dtime, dtimeIn=dtimeIn, SavePath=SavePath)

    def _set_Id(self, Id, Type=None, Exp=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):
        if self._Done:
            Out = tfpf._get_FromItself(self.Id,{'Type':Type, 'Exp':Exp, 'shot':shot, 'dtime':dtime, '_dtimeIn':dtimeIn, 'SavePath':SavePath})
            Type, Exp, shot, dtime, dtimeIn, SavePath = Out['Type'], Out['Exp'], Out['shot'], Out['dtime'], Out['dtimeIn'], Out['SavePath']
        tfpf._check_NotNone({'Id':Id})
        self._check_inputs(Id=Id)
        if type(Id) is str:
            Exp = 'Test' if Exp is None else Exp
            tfpf._check_NotNone({'Exp':Exp, 'dtimeIn':dtimeIn})
            self._check_inputs(Type=Type, Exp=Exp, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
            Id = tfpf.ID('Mesh1D', Id, Type=Type, Exp=Exp, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._Id = Id

    def _set_Knots(self, Knots):
        """ Set the knots and computes all subsequent attributes """
        tfpf._check_NotNone({'Knots':Knots})
        self._check_inputs(Knots=Knots)
        self._NKnots, self._Knots, self._NCents, self._Cents, self._Lengths, self._Length, self._Bary, self._Cents_Knotsind, self._Knots_Centsind = _tfm_c._Mesh1D_set_Knots(Knots)

    def sample(self, Sub=tfd.BF1Sub, SubMode=tfd.BF1SubMode, Test=True):
        """ Return a sampling of Mesh using the provided Sub resolution in 'rel' (relative to the size of each mesh element) or 'abs' mode (in distance unit), useful for plotting basis functions on the mesh """
        xx = _tfm_c._Mesh1D_sample(self.Knots, Sub=Sub, SubMode=SubMode, Test=True)
        return xx


    # plotting routines
    def plot(self, y=0., Elt='KCN', ax=None, Kdict=tfd.M1Kd, Cdict=tfd.M1Cd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
        """ Return a plt.Axes instance to the plot of Mesh1D """
        return _tfm_p.Mesh1D_Plot(self._Knots, y=y, Elt=Elt, Leg=self._Id.NameLTX, ax=ax, Kdict=Kdict, Cdict=Cdict, LegDict=LegDict, draw=draw, a4=a4, Test=Test)

    def plot_Res(self, ax=None, Dict=tfd.M1Resd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
        """ Return a plt.Axes instance to the plot of the resolution of Mesh1D """
        return _tfm_p.Mesh1D_Plot_Res(self._Knots, Leg=self._Id.NameLTX, ax=ax, Dict=Dict, LegDict=LegDict, draw=draw, a4=a4, Test=Test)

    # Saving routine
    def save(self, SaveName=None, Path=None, Mode='npz'):
        """
        Save the object in folder Name, under file name SaveName, using specified mode

        Inputs:
        ------
            SaveName    str     The name to be used to for the saved file, if None (recommended) uses Ves.Id.SaveName (default: None)
            Path        str     Path specifying where to save the file, if None (recommended) uses Ves.Id.SavePath (default: None)
            Mode        str     Flag specifying whether to save the object as a numpy array file ('.npz', recommended) or an object using cPickle (not recommended, may cause retro-compatibility issues with later versions)
        """
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path, Mode=Mode)


def _Mesh1D_check_inputs(Id=None, Knots=None, Type=None, Exp=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):
    if Id is not None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID object !"
    if Knots is not None:
        assert hasattr(Knots,'__iter__') and np.asarray(Knots).ndim==1 and np.all(Knots==np.unique(Knots)) and not np.any(np.isnan(Knots)), "Arg Knots must be an iterable of increasing non-NaN knots coordinates !"
    assert Type is None, "Arg Type must be None for a 1D mesh and LBF !"
    if not Exp is None:
        assert Exp in tfd.AllowedExp, "Ar Exp must be in "+str(tfd.AllowedExp)+" !"
    Ints = [shot]
    if any([not aa is None for aa in Ints]):
        assert all([aa is None or type(aa) is int for aa in Ints]), "Args [shot] must be int !"
    strs = [SavePath]
    if any([not aa is None for aa in strs]):
        assert all([aa is None or type(aa) is str for aa in strs]), "Args [SavePath] must all be str !"
    if not dtime is None:
        assert type(dtime) is dtm.datetime, "Arg dtime must be a dtm.datetime !"
    bools = [dtimeIn]
    if any([not aa is None for aa in bools]):
        assert all([aa is None or type(aa) is bool for aa in bools]), " Args [dtimeIn] must all be bool !"








class Mesh2D(object):
    """ A class defining a 2D mesh (knot vector, with associated surfaces, centers and correspondace between them, as well as plotting routines)

    Inputs:
    -------
        Id          None or str or tfpf.ID      A name or tfpf.ID class, for identification
        Knots       iterable or Mesh2D          An iterable of len()==2, containing either 2 Mesh1D instances or 2 iterables of knots, or a Mesh2D instance, of which only some elements (indicated by ind) are to be kept
        ind         iterable                    An iterable
        Type        None or str                 The type of Mesh1D object (default: None)
        Exp         None or str                 The experiment to which this object relates (e.g.: "ITER", "AUG", "JET"...)
        shot        None or int                 A shot number from which this object can be used (i.e.: in case of changing geometry)
        dtime       None or dtm.datetime        A time reference to be used to identify this particular instance (used for debugging mostly), default: None
        dtimeIn     bool                        Flag indicating whether dtime should be included in the SaveName (used for debugging mostly), default: False
        SavePath    None or str                 Absolute path where the object would be saved if necessary
    """

    def __init__(self, Id, Knots, ind=None, Type='Tor', Exp=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):

        self._Done = False
        self._set_Id(Id, Type=Type, Exp=Exp, shot=shot, dtime=dtime, dtimeIn=dtimeIn, SavePath=SavePath)
        self._set_Knots(Knots, ind=ind)
        self._Done = True

    @property
    def Id(self):
        return self._Id
    @property
    def Type(self):
        return self.Id.Type
    @property
    def Knots(self):
        return self._Knots
    @property
    def NKnots(self):
        return self._NKnots
    @property
    def Cents(self):
        return self._Cents
    @property
    def NCents(self):
        return self._NCents
    @property
    def MeshX1(self):
        return self._MeshX1
    @property
    def MeshX2(self):
        return self._MeshX2
    @property
    def SubMesh(self):
        return self._SubMesh
    @property
    def BaryS(self):
        return self._BaryS
    @property
    def Surfs(self):
        return self._Surfs
    @property
    def Surf(self):
        return self._Surf
    @property
    def VolAngs(self):
        return self._VolAngs
    @property
    def VolAng(self):
        return self._VolAng
    @property
    def BaryV(self):
        return self._BaryV
    @property
    def CentsV(self):
        return self._CentsV


    def _check_inputs(self, Id=None, Knots=None, ind=None, Type=None, Exp=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):
        _Mesh2D_check_inputs(Id=Id, Knots=Knots, ind=ind, Type=Type, Exp=Exp, shot=shot, dtime=dtime, dtimeIn=dtimeIn, SavePath=SavePath)

    def _set_Id(self, Id, Type=None, Exp=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):
        if self._Done:
            Out = tfpf._get_FromItself(self.Id,{'Type':Type, 'Exp':Exp, 'shot':shot, 'dtime':dtime, '_dtimeIn':dtimeIn, 'SavePath':SavePath})
            Type, Exp, shot, dtime, dtimeIn, SavePath = Out['Type'], Out['Exp'], Out['shot'], Out['dtime'], Out['dtimeIn'], Out['SavePath']
        tfpf._check_NotNone({'Id':Id})
        self._check_inputs(Id=Id)
        if type(Id) is str:
            Exp = 'Test' if Exp is None else Exp
            tfpf._check_NotNone({'Exp':Exp, 'dtimeIn':dtimeIn})
            self._check_inputs(Type=Type, Exp=Exp, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
            Id = tfpf.ID('Mesh2D', Id, Type=Type, Exp=Exp, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._Id = Id

    def _set_Knots(self, Knots, ind=None):
        tfpf._check_NotNone({'Knots':Knots})
        self._check_inputs(Knots=Knots, ind=ind)
        if ind is None:
            self._MeshX1 = Knots[0] if type(Knots[0]) is Mesh1D else Mesh1D(self.Id.Name+'_MeshX1', np.asarray(Knots[0]))
            self._MeshX2 = Knots[1] if type(Knots[1]) is Mesh1D else Mesh1D(self.Id.Name+'_MeshX2', np.asarray(Knots[1]))
            self._Knots = np.array([np.tile(self.MeshX1.Knots,(1,self.MeshX2.NKnots)).flatten(), np.tile(self.MeshX2.Knots,(self.MeshX1.NKnots,1)).T.flatten()])
            self._Cents = np.array([np.tile(self.MeshX1.Cents,(1,self.MeshX2.NCents)).flatten(), np.tile(self.MeshX2.Cents,(self.MeshX1.NCents,1)).T.flatten()])
        else:
            # ind should refer to the centers of 'Knots', which should be a Mesh2D object
            ind = np.asarray(ind)
            ind = ind.nonzero()[0] if ind.dtype.name is 'bool' else ind
            self._Cents = Knots._Cents[:,ind]
            indKnots = np.unique(Knots._Cents_Knotsind[:,ind].flatten())
            self._Knots = Knots.Knots[:,indKnots]
            self._MeshX1 = Mesh1D(self.Id.Name+'_MeshX1',np.unique(self.Knots[0,:]))
            self._MeshX2 = Mesh1D(self.Id.Name+'_MeshX2',np.unique(self.Knots[1,:]))
        self._NCents = self.Cents.shape[1]
        self._NKnots = self.Knots.shape[1]
        self._set_Attribs()

    def _set_Attribs(self):
        self._set_Cents_Knotsind()
        self._set_Knots_Centsind()
        self._set_SurfVolBary()
        self._set_BoundPoly()
        self._SubMesh = {}

    def _set_Cents_Knotsind(self):
        self._Cents_Knotsind = _tfm_c._Mesh2D_set_Cents_Knotsind(self.NCents, self.MeshX1.Knots, self.MeshX2.Knots, self.Cents, self.Knots)

    def _set_Knots_Centsind(self):
        self._Knots_Centsind = _tfm_c._Mesh2D_set_Knots_Centsind(self.NKnots, self.MeshX1.Cents, self.MeshX2.Cents, self.Knots, self.Cents, self.NCents)

    def _set_SurfVolBary(self):
        self._Surfs, self._Surf, self._VolAngs, self._VolAng, self._BaryS, self._BaryV, self._CentsV = _tfm_c._Mesh2D_set_SurfVolBary(self.Knots, self._Cents_Knotsind, self.Cents, VType=self.Id.Type)

    def _set_BoundPoly(self):
        self._BoundPoly = _tfm_c._Mesh2D_set_BoundPoly(self.Knots, self._Cents_Knotsind, self.NCents)

    def _get_SubMeshInPolygon(self, Poly, InMode='Cents', NLim=1, Id=None, Out='Mesh2D'):
        """ Get the indices (int or bool) of the Cents which lie inside the input Poly (InMode='Cents') or of the Cents of which at least NLim Knots lie inside the input Poly (InMode='Knots'), can also return  Mesh2D object """
        assert Out in ['Mesh2D', int, bool], "Arg Out must be in ['Mesh2D', int, bool] !"
        indIn = _tfm_c._Mesh2D_get_SubMeshPolygon(self.Cents, self.Knots, self._Cents_Knotsind, Poly, InMode=InMode, NLim=NLim)
        if Out=='Mesh2D':
            Id = self.Id.Name+'_SubMesh' if Id is None else Id
            indIn = Mesh2D(Id, self, ind=indIn, Type=self.Id.Type, Exp=self.Exp, shot=self.shot, SavePath=self.SavePath, dtime=self.dtime, dtimeIn=self.dtimeIn)
            indIn.Id.set_USRdict({'SubMeshPoly':Poly})
        else:
            indIn = indIn if Out==bool else indIn.nonzero()[0]
        return indIn

    def add_SubMesh(self, Name, ind=None, Poly=None, InMode='Cents', NLim=1):
        """
        Add a submesh, defined by the indices of some centers or by the elements lying inside a polygon

        Inputs:
        -------
            Name    str         The name to be given to the created submesh
            ind     None /      Indices of the mesh elements to be used for creating the submesh
            Poly    None /      Polygon inside which the elements should lie to belong to the submesh
            InMode  str         Flag indicating how to determine whether an element lies inside the polygon, by its center ('Cents') or knots ('Knots')
            NLim    int         If InMode=='Knots', number of knots that must lie inside the polygon for the mesh element to be considered inside too

        The created submesh is added to the self.SubMesh dictionary under the key Name
        """
        assert not (ind is None) == (Poly is None), "Either ind or Poly must be None !"
        assert Name is None or type(Name) is str, "Arg name must be str !"
        if ind is not None:
            ind = np.asarray(ind) if hasattr(ind,'__getitem__') else np.asarray([ind])
            assert ind.ndim==1 and ind.dtype.name in ['bool','int64'], "Arg ind must be an index or a iterable of indices in int or bool format !"
            if ind.dtype.name=='int64':
                iind = np.zeros((self.NCents,),dtype=bool)
                iind[ind] = True
                ind = iind
            assert ind.size==self.NCents and ind.dtype.name=='bool', "Arg ind must be an array of bool of size==self.NCents !"
        else:
            ind = self._get_SubMeshInPolygon(Poly, InMode='Cents', Out=bool)
        Name = r"Sub{0:02.0f}".format(len(self.SubMesh.keys())) if Name is None else Name
        Sub = {'Name':Name, 'ind':ind, 'Mesh2D':Mesh2D(Name, self, ind=ind, Type=self.Id.Type, Exp=self.Id.Exp, shot=self.Id.shot, SavePath=self.Id.SavePath, dtime=self.Id.dtime, dtimeIn=self.Id._dtimeIn)}
        self._SubMesh[Name] = Sub

    def _get_CentBckg(self):
        return _tfm_c._Mesh2D_get_CentBckg(self.MeshX1.NCents, self.MeshX2.NCents, self.MeshX1.Cents, self.MeshX2.Cents, self.Cents, self.NCents)

    def _get_KnotsBckg(self):
        """ Return the knots of the background full rectangular mesh, and the indices of those background knots which belong to the mesh """
        return _tfm_c._Mesh2D_get_KnotsBckg(self.MeshX1.NKnots, self.MeshX2.NKnots, self.MeshX1.Knots, self.MeshX2.Knots, self.Knots, self.NKnots)

    def isInside(self, Pts2D):
        """ Return a 1d bool array indicating which points (in (R,Z) or (Y,Z) coordinates) are inside the mesh support (i.e. inside the bounding polygon) """
        return _tfm_c._Mesh2D_isInside(Pts2D, self._BoundPoly)


    def sample(self, Sub=tfd.BF2Sub, SubMode=tfd.BF2SubMode, Test=True):
        """ Return a sampling of Mesh using the provided Sub resolution in 'rel' (relative to the size of each mesh element) or 'abs' mode (in distance unit), useful for plotting basis functions on the mesh """
        Pts = _tfm_c._Mesh2D_sample(self.Knots, Sub=Sub, SubMode=SubMode, BoundPoly=self._BoundPoly, Test=Test)
        return Pts



    def plot(self, ax=None, Elt='MBgKCBsBv', indKnots=None, indCents=None, SubMesh=None, Bckdict=tfd.M2Bckd, Mshdict=tfd.M2Mshd, Kdict=tfd.M2Kd, Cdict=tfd.M2Cd, LegDict=tfd.TorLegd, draw=True, a4=False, Test=True):
        """
        Plot the mesh

        Inputs:
        -------
            ax          None / plt.Axes         Axes to be used for plotting, if None a new figure with axes is created
            Elt         str                     Flag specifying which elements to plot, each capital letter corresponds to an element
                                                    'M':    the mesh itself
                                                    'Bg':   the background mesh on which it lies (if this mesh was extracted from a larger mesh)
                                                    'K':    some specific knots of the mesh (identified by kwdarg indKnots)
                                                    'C':    some specific centers of the mesh (identified by kwdarg indCents)
                                                    'Bs':   plot the surfacic center of mass
                                                    'Bv':   plot the volumic center of mass
            indKnots    None / str / iterable   Flag indicating which knots to plot specifically, None, 'all' or those indicated by the int indices in a list/tuple/array
            indCents    None / str / iterable   Flag indicating which centers to plot specifically, None, 'all' or those indicated by the int indices in a list/tuple/array
            SubMesh     None / str              Flag indicating which submesh to plot, if any (provide the str key)
            Bckdict     dict                    Dictionary of properties used for the background mesh, if plotted (fed to plt.plot())
            Mshdict     dict                    Dictionary of properties used for the mesh, if plotted (fed to plt.plot())
            Kdict       dict                    Dictionary of properties used for the knots, if plotted (fed to plt.plot())
            Cdict       dict                    Dictionary of properties used for the centers, if plotted (fed to plt.plot())
            LegDict     dict                    Dictionary of properties used for plotting the legend, fed to plt.legend(), the legend is not plotted if None
            draw        bool                    Flag indicating whether the fig.canvas.draw() shall be called automatically
            a4          bool                    Flag indicating whether the figure should be plotted in a4 dimensions for printing
            Test        bool                    Flag indicating whether the inputs should be tested for conformity

        Outputs:
        --------
            ax          plt.Axes

        """
        ax = _tfm_p.Mesh2D_Plot(self, ax=ax, Elt=Elt, indKnots=indKnots, indCents=indCents, SubMesh=SubMesh, Bckdict=Bckdict, Mshdict=Mshdict, Kdict=Kdict, Cdict=Cdict, LegDict=LegDict, draw=draw, a4=a4, Test=Test)
        return ax


    def plot_Res(self, ax1=None, ax2=None, ax3=None, axcb=None, Elt='MBgS', SubMesh=None, Leg=None, Bckdict=tfd.M2Bckd, Mshdict=tfd.M2Mshd, LegDict=tfd.M2Legd, draw=True, a4=False, Test=True):
        """
        Plot the resolution of the mesh

        Inputs:
        -------
            ax1         None or plt.Axes    Axes to be used for plotting the mesh, if None a new figure with axes is created
            ax2         None or plt.Axes    Axes to be used for plotting the , if None a new figure with axes is created
            ax3         None or plt.Axes    Axes to be used for plotting the, if None a new figure with axes is created
            axcb        None or plt.Axes    Axes to be used for the colorbar of surfaces, if None a new figure with axes is created
            Elt         str                 Flag specifying which elements to plot, each capital letter corresponds to an element
                                                'M':    the mesh itself
                                                'Bg':   the background mesh on which it lies (if this mesh was extracted from a larger mesh)
                                                'S':    the surface of each mesh element (color-coded)
            SubMesh     None or str         Flag indicating which submesh to plot, if any (provide the str key)
            Leg         None or str         String to be used as legend
            Bckdict     dict                Dictionary of properties used for the background mesh, if plotted (fed to plt.plot())
            Mshdict     dict                Dictionary of properties used for the mesh, if plotted (fed to plt.plot())
            LegDict     dict                Dictionary of properties used for plotting the legend, fed to plt.legend(), the legend is not plotted if None
            draw        bool                Flag indicating whether the fig.canvas.draw() shall be called automatically
            a4          bool                Flag indicating whether the figure should be plotted in a4 dimensions for printing
            Test        bool                Flag indicating whether the inputs should be tested for conformity


        Outputs:
        --------
            ax1         plt.Axes
            ax2         plt.Axes
            ax3         plt.Axes
            axcb        plt.Axes

        """
        ax1, ax2, ax3, axcb = _tfm_p.Mesh2D_Plot_Res(self, ax1=ax1, ax2=ax2, ax3=ax3, axcb=axcb, Elt=Elt, SubMesh=SubMesh, Leg=Leg, Bckdict=Bckdict, Mshdict=Mshdict, LegDict=LegDict, draw=draw, a4=a4, Test=Test)
        return ax1, ax2, ax3, axcb

    def save(self, SaveName=None, Path=None, Mode='npz'):
        """
        Save the object in folder Name, under file name SaveName, using specified mode

        Inputs:
        ------
            SaveName    str     The name to be used to for the saved file, if None (recommended) uses Ves.Id.SaveName (default: None)
            Path        str     Path specifying where to save the file, if None (recommended) uses Ves.Id.SavePath (default: None)
            Mode        str     Flag specifying whether to save the object as a numpy array file ('.npz', recommended) or an object using cPickle (not recommended, may cause retro-compatibility issues with later versions)
        """
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path, Mode=Mode)



def _Mesh2D_check_inputs(Id=None, Knots=None, ind=None, Type=None, Exp=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):
    if Id is not None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID object !"
    if Knots is not None:
        assert type(Knots) is Mesh2D or (hasattr(Knots,'__iter__') and len(Knots)==2), "Arg Knots must be an iterable of 2 Knots vectors or 2 Mesh1D objects, or a Mesh2D object !"
        if type(Knots) is not Mesh2D:
            assert all([type(kk) is Mesh1D or (hasattr(kk,'__iter__') and np.asarray(kk).ndim==1 and np.all(kk==np.unique(kk)) and not np.any(np.isnan(kk))) for kk in Knots]), "Each element of Knots must be Mesh1D or knot vector !"
    if not ind is None:
        assert type(Knots) is Mesh2D, "Arg Knots must be a Mesh2D instance when ind is not None !"
        assert hasattr(ind,'__iter__') and np.asarray(ind).ndim==1 and np.asarray(ind).dtype.name in ['bool',int,'int32','int64'], "Arg ind must be an iterable of bool or int indices !"
        if np.asarray(ind).dtype.name is 'bool':
            assert len(ind)==Knots.NCents, "Arg ind must be of len()==Knots.Cents if indices of type bool !"
        if np.asarray(ind).dtype.name in [int,'int32','int64']:
            assert max(ind)<Knots.NCents, "Arg ind must not contain indices higher than Knots.NCents !"
    if not Type is None:
        assert Type in ['Tor','Lin'], "Arg Type must be in ['Tor','Lin'] !"
    if not Exp is None:
        assert Exp in tfd.AllowedExp, "Ar Exp must be in "+str(tfd.AllowedExp)+" !"
    Ints = [shot]
    if any([not aa is None for aa in Ints]):
        assert all([aa is None or type(aa) is int for aa in Ints]), "Args [shot] must be int !"
    strs = [SavePath]
    if any([not aa is None for aa in strs]):
        assert all([aa is None or type(aa) is str for aa in strs]), "Args [SavePath] must all be str !"
    if not dtime is None:
        assert type(dtime) is dtm.datetime, "Arg dtime must be a dtm.datetime !"
    bools = [dtimeIn]
    if any([not aa is None for aa in bools]):
        assert all([aa is None or type(aa) is bool for aa in bools]), " Args [dtimeIn] must all be bool !"



#class Mesh3D(object):
#    def __init__(self, Id, KnR, KnZ, KnThet):
#        assert type(Id) is str or isinstance(Id,ID), "Arg Id should be string or an ID instance !"
#        if type(Id) is str:
#            Id = tfpf.ID('Mesh3D',Id)
#            Id.Time = str(dtm.datetime.now())
#        self.Id = Id






"""
###############################################################################
###############################################################################
                   LBF Objects and properties
###############################################################################
"""


class LBF1D(object):
    def __init__(self, Id, MeshKnts, Deg, Type=None, Exp=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):
        self._Done = False
        self._set_Id(Id, Deg=Deg, Type=Type, Exp=Exp, shot=shot, dtime=dtime, dtimeIn=dtimeIn, SavePath=SavePath)
        self._set_Mesh(MeshKnts, Deg=Deg)
        self._Done = True


    @property
    def Id(self):
        """Return the Id"""
        return self._Id
    @property
    def Type(self):
        return self.Id.Type

    @property
    def Mesh(self):
        """Return the Mesh1D"""
        return self._Mesh
    @property
    def Deg(self):
        return self._Deg
    @property
    def LFunc(self):
        return self._LFunc
    @property
    def NFunc(self):
        return self._NFunc


    def _check_inputs(self, Id=None, MeshKnts=None, Deg=None, Type=None, Exp=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):
        _LBF1D_check_inputs(Id=Id, MeshKnts=MeshKnts, Deg=Deg, Type=Type, Exp=Exp, shot=shot, dtime=dtime, dtimeIn=dtimeIn, SavePath=SavePath)

    def _set_Id(self, Id, Deg=None, Type=None, Exp=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):
        if self._Done:
            Out = tfpf._get_FromItself(self.Id,{'Deg':Deg, 'Type':Type, 'Exp':Exp, 'shot':shot, 'dtime':dtime, '_dtimeIn':dtimeIn, 'SavePath':SavePath})
            Type, Exp, Deg, shot, dtime, dtimeIn, SavePath = Out['Type'], Out['Exp'], Out['Deg'], Out['shot'], Out['dtime'], Out['dtimeIn'], Out['SavePath']
        tfpf._check_NotNone({'Id':Id,'Deg':Deg})
        self._check_inputs(Id=Id,Deg=Deg)
        if type(Id) is str:
            Exp = 'Test' if Exp is None else Exp
            tfpf._check_NotNone({'Exp':Exp, 'Deg':Deg, 'dtimeIn':dtimeIn})
            self._check_inputs(Type=Type, Deg=Deg, Exp=Exp, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
            Id = tfpf.ID('LBF1D', Id, Type=Type, Deg=Deg, Exp=Exp, shot=shot, SavePath=SavePath, dtime=dtime, dtimeIn=dtimeIn)
        self._Id = Id

    def _set_Mesh(self, MeshKnts, Deg=None):
        self._check_inputs(MeshKnts=MeshKnts, Deg=Deg)
        if type(MeshKnts) is not Mesh1D:
            MeshKnts = np.unique(MeshKnts)
            MeshKnts = Mesh1D(self.Id.Name, MeshKnts, Type=self.Type, Exp=self.Id.Exp, shot=self.Id.shot, dtime=self.Id.dtime, dtimeIn=self.Id._dtimeIn, SavePath=self.Id.SavePath)
        self._Mesh = MeshKnts
        self._set_BF(Deg=Deg)

    def _set_BF(self,Deg=None, Mode='scp'):
        tfpf._check_NotNone({'Deg':Deg})
        self._check_inputs(Deg=Deg)
        self._Deg = Deg
        self._LFunc, self._Func_Knotsind, self._Func_Centsind, self._Knots_Funcind, self._Cents_Funcind, self._Func_MaxPos, scp_Lkntsf, scp_Lcoeff = _tfm_bs.BSpline_LFunc(self.Deg, self.Mesh.Knots, Deriv=0, Mode=Mode, Test=True)
        self._LFunc_scp_Lkntsf, self._LFunc_scp_Lcoeff = scp_Lkntsf, scp_Lcoeff
        self._NFunc = len(self._LFunc)
        self._LFunc_Mode = Mode

    def _get_Func_Supps(self):
        Knots = self.Mesh.Knots[self._Func_Knotsind]
        return np.array([np.nanmin(Knots,axis=0), np.nanmax(Knots,axis=0)])

    def _get_Func_InterFunc(self):
        indF = np.nan*np.ones((2*self.Deg,self.NFunc))
        if self.Deg>0:
            for ii in range(0,self.NFunc):
                ind = self._Cents_Funcind[:,self._Func_Centsind[:,ii]].flatten()
                ind = np.unique(ind[~np.isnan(ind)])
                ind = np.delete(ind,(ind==ii).nonzero()[0])
                indF[:ind.size,ii] = ind.astype(int)
        return indF

    def get_TotFunc(self, Deriv=0, Coefs=1., thr=1.e-8, thrmode='rel', Abs=True, Test=True):
        """
        Return the function or list of functions ff such that ff(Pts) give the total value of the basis functions for each time step provided in Coefs

        Inputs:
        -------
            Deriv       int / str           Flag indicating which quantity should be computed
                                                0 / 'D0':   The function itself
                                                1 / 'D1':   Its first derivative
                                                2 / 'D2':   Its first derivative
                                                3 / 'D3':   Its first derivative
                                                'D0N2':     Its squared norm
                                                'D0ME':     Its entropy
                                                'D1N2':     The squared norm of its first derivative
                                                'D1FI':     Its Fisher information
                                                'D2N2':     The squared norm of its second derivative
                                                'D3N2':     The squared norm of its third derivative
            Coefs       float / np.ndarray  Values of the coefficients to be used for each basis function, common to all if float, if 2-dim the first dimension should be the number of time steps
            thr         None / float        If provided, value used as a lower threshold to the function value for computing non-linear diverging quantities such as 'D0ME' or 'D1FI'
            thrmode     str                 Flag indicating whether the provided thr value shall be considered as an absolute value ('abs') or relative to the maximum of the funcion ('rel')
            Abs         bool                Flag indicating whether the absolute value (True) of the function should be considered for computing 'D0ME' and 'D1FI'
            Test        bool                Flag indicating whether the inputs should be tested for conformity

        Outputs:
        --------
            Func        callable / list     Function issuing the required quantity for any provided points, list of such if several time steps were provided

        """
        return _tfm_bs.BSpline_TotFunc(self.Deg, self.Mesh.Knots, Deriv=Deriv, Coefs=Coefs, thr=thr, thrmode=thrmode, Abs=Abs, Test=Test)

    def get_TotVal(self, Pts, Deriv=0, Coefs=1., thr=1.e-8, thrmode='rel', Test=True):
        """
        Return the total value the required quantity computed using the provided coefficients at provided points

        Inputs: (see self.get_TotFunc() for documentation of common inputs)
        -------
            Pts     np.ndarray      1-dim array of pts coordinates where the quantity should be evaluated

        Outputs:
        --------
            Val     np.ndarray      1 or 2-dim array of evaluated values (2-dim if the provided coefficients were also 2-dim, i.e.: if they spanned several time steps)

        """
        TF = _tfm_bs.BSpline_TotFunc(self.Deg, self.Mesh.Knots, Deriv=Deriv, Coefs=Coefs, thr=thr, thrmode=thrmode, Test=Test)
        Val = np.vstack([ff(Pts) for ff in TF]) if type(TF) is list else TF(Pts)
        return Val

    def get_Coefs(self, xx=None, yy=None, ff=None, Sub=tfd.BF1Sub, SubMode=tfd.BF1SubMode, Test=True):
        """
        Return the coefficients and residue obtained by a lest square fit of the basis functions to the provided data data or function

        Inputs:
        -------
            xx          None / np.ndarray   If provided, the points at which the data is provided, 1-dim
            yy          None / np.ndarray   If provided, the data provided at points xx, 1-dim
            ff          None / callable     If provided, the function to be used for the fit (xx will be computed from a sampling of the underlying mesh, using parameters Sub and SubMode, and yy=ff(xx))
            Sub         None / float        Needed if ff is used instead of (xx,yy), resolution to be used for the sampling of the mesh, fed to self.Mesh.sample()
            SubMode     None / str          Flag indcating whether Sub should be understood as an absolute distance ('abs') or a fracion of each mesh element ('rel')
            Test        bool                Flag indicating whether the inputs should be tested for conformity

        Outputs:
        --------
            Coefs       np.ndarray          Array of coefficients (1-dim) found by least-square fit
            res                             Residue of the fit (from np.linalg.lstsq())

        """
        Coefs, res = _tfm_c._LBF1D_get_Coefs(self.LFunc, self.NFunc, self.Mesh.Knots, xx=xx, yy=yy, ff=ff, Sub=Sub, SubMode=SubMode, Test=Test)
        return Coefs, res

    def get_IntOp(self, Deriv=0, Method=None, Mode='Vol', Sparse=tfd.L1IntOpSpa, SpaFormat=tfd.L1IntOpSpaFormat, N=None, Test=True):
        """
        Return the matrix operator necessary for computing the chosen integral, only possible for linear or quadratic functionals like derivatives or squared norm of derivatives

        Inputs:
        -------
            Deriv       int / str      Flag indicating which integral operator is to be computed, notice that since exact derivations are used (i.e.: no discretization), you can only access a derivative <= degree of the functions
                                            0 or 'D0': integral of the function
                                            1 or 'D1': integral of the first derivative
                                            2 or 'D2': integral of the second derivative
                                            3 or 'D3': integral of the third derivative
                                            'D0N2': integral of the squared norm of the function
                                            'D1N2': integral of the squared norm of the first derivative
                                            'D2N2': integral of the squared norm of the second derivative
                                            'D3N2': integral of the squared norm of the third derivative
            Method      None / str     Flag indicating the method to be used for computing the operator, if None switches to 'exact' whenever possible
                                            'exact': uses pre-derived exact analytical formulas (prefered because faster and more accurate)
                                            'quad': uses Gauss-Legendre quadrature formulas (exact too if the good number of points, but slower, implemented to prepare for CAID and non-linear operators)
            Mode        str             Flag indicating whether the integral shall be computed assuming as is or by multiplying by the abscissa prior to integration
                                            'Surf': Normal integration
                                            'Vol': The integrand is multiplied by x prioir to integration, useful later for volume integrals for 2D toiroidal geometries
            Sparse      bool            Flag indicting whether the operator should be returned as a scipy sparse matrix
            SpaFormat   str             Flag indicating which format to use for the operator if it is to be returned as a scipy sparse matrix, in ['dia','bsr','coo','csc','csr']
            N           None / int      If int provided, forces the number of quadrature points to N, if None N is computed automatically
            Test        bool            Flag indicating whether the inputs should be tested for conformity

        Outputs:
        --------
            A           np.ndarray / scipy.sparse.spmatrix  Operator
            m           int                                 Flag indicating how to get the integral from the operator, if coefs is the vector of coefficients of the basis functions
                                                                0: integral = A.dot(coefs) or np.sum(A*coefs)
                                                                1: integral = coefs.dot(A.dot(coefs))
        """
        return _tfm_bs.Calc_1D_LinIntOp(Knots=self.Mesh.Knots, Deg=self.Deg, Deriv=Deriv, Method=Method, Mode=Mode, LFunc=self.LFunc, quad_pts=None, quad_w=None, quad_aa=None, Sparse=Sparse, SpaFormat=SpaFormat, N=N, Test=Test)


    def get_IntQuadPts(self, Deriv=0, Mode='Vol', N=None, Test=True):   # To be finished
        """
        Return the Gauss-Legendre quadrature points and weights for computing the desired integral

        Inputs:
        -------
            Deriv   int / str       Flag indicating which integral operator is to be computed, notice that since exact derivations are used (i.e.: no discretization), you can only access a derivative <= degree of the functions
                                        0 or 'D0': integral of the function
                                        1 or 'D1': integral of the first derivative
                                        2 or 'D2': integral of the second derivative
                                        3 or 'D3': integral of the third derivative
                                        'D0N2': integral of the squared norm of the function
                                        'D1N2': integral of the squared norm of the first derivative
                                        'D2N2': integral of the squared norm of the second derivative
                                        'D3N2': integral of the squared norm of the third derivative
            Mode    str             Flag indicating whether the integral shall be computed assuming as is or by multiplying by the abscissa prior to integration
                                        'Surf': Normal integration
                                        'Vol': The integrand is multiplied by x prioir to integration, useful later for volume integrals for 2D toiroidal geometries
            N       None / int      If int provided, forces the number of quadrature points to N, if None N is computed automatically
            Test    bool            Flag indicating whether the inputs should be tested for conformity

        Outputs:
        --------
            pts     np.ndarray      2D array of shape (N,NKnots-1) of points coordinates in each mesh element, where N is the number of points per mesh element and NKnots-1 is the number of mesh elements
            w       np.ndarray      2D array of weights associated to each point, same shape as pts
            A       np.ndarray      1D array of rescaling coefficients (i.e.: (b-a)/2) to multiply to the sum in each mesh element
            N       int             Number of points per mesh element (i.e.: = pts.shape[0])


        """
        intDeriv = Deriv if type(Deriv) is int else int(Deriv[1])
        return _tfm_bs.get_IntQuadPts(self.Knots, self.Deg, Deriv, intDeriv, Mode=Mode, N=N)


    def get_IntVal(self, Coefs=1., Deriv=0, Method=None, Mode='Vol', Abs=True, thr=1.e-8, thrmode='rel', N=None, Test=True):   # To be finished
        """
        Return the value of the desired integral (chosen from Deriv in [0,1,2,3,'D0','D1','D2','D3','D0N2','D1N2','D2N2','D3N2'] for linear functionals and ['D0ME','D1FI'] for non-linear ones), with the provided coefficients

        Inputs:
        -------
            Coefs       float / np.ndarray
            Deriv       int / str           Flag indicating which integral operator is to be computed (see doc of self.IntOp() for details)
            Method      None / str          Flag indicating the method to be used for computing the operator, if None switches to 'exact' whenever possible (see doc of self.IntOp() for details)
            Mode        str                 Flag indicating whether the integral shall be computed assuming as is or by multiplying by the abscissa prior to integration (see doc of self.IntOp() for details)
            Abs         bool                Flag indicating whether the absolute value (True) of the function should be considered for computing 'D0ME' and 'D1FI'
            thr         None / float        If provided, value used as a lower threshold to the function value for computing non-linear diverging quantities such as 'D0ME' or 'D1FI'
            thrmode     str                 Flag indicating whether the provided thr value shall be considered as an absolute value ('abs') or relative to the maximum of the funcion ('rel')
            N           None / int          If int provided, forces the number of quadrature points to N, if None N is computed automatically
            Test        bool                Flag indicating whether the inputs should be tested for conformity

        Outputs:
        --------
            Int         float / np.ndarray  The computed integral or array of integral values if a 2D array of Coefs was provided (each line corresponding to a time step)

        """
        return _tfm_c.get_IntVal(Coefs=Coefs, Knots=self.Mesh.Knots, Deg=self.Deg, Deriv=Deriv, LFunc=self.LFunc, LFunc_Mode=self._LFunc_Mode, Method=Method, Mode=Mode, N=N, Test=Test)


    def plot(self, ax='None', Coefs=1., Deriv=0, Elt='TL', Sub=tfd.BF1Sub, SubMode=tfd.BF1SubMode, LFdict=tfd.BF1Fd, Totdict=tfd.BF1Totd, LegDict=tfd.TorLegd, Test=True):
        """

        """
        if type(Coefs) is float:
            Coefs = Coefs*np.ones((self.NFunc,))
        if 'T' in Elt:
            TotF = BSpline_get_TotFunc(self.Deg, self.Mesh.Knots, Deriv=Deriv, Coefs=Coefs, Test=Test)
        else:
            TotF = None
        if (type(Deriv) is int or Deriv in ['D0','D1','D2','D3']) and 'L' in Elt:
            if not type(Deriv) is int:
                Deriv = int(Deriv[1])
            LF1 = BSplineDeriv(self.Deg, self.Mesh.Knots, Deriv=Deriv, Test=Test)
            LF = [lambda x,Coefs=Coefs,ii=ii: Coefs[ii]*LF1[ii](x) for ii in range(0,len(LF1))]
        else:
            LF = None
        return Plot_BSpline1D(self.Mesh.Knots, TotF, LF, ax=ax, Elt=Elt, Name=self.Id.Name+' '+str(Deriv), Sub=Sub, SubMode=SubMode, LFdict=LFdict, Totdict=Totdict, LegDict=LegDict, Test=Test)

    def plot_Ind(self, ax='None', Ind=0, Coefs=1., Elt='LCK', y=0., Sub=tfd.BF1Sub, SubMode=tfd.BF1SubMode, LFdict=tfd.BF1Fd, Kdict=tfd.M2Kd, Cdict=tfd.M2Cd, LegDict=tfd.TorLegd, Test=True):
        """

        """
        assert type(Ind) in [int,list,np.ndarray], "Arg Ind must be a int, a list of int or a np.ndarray of int or booleans !"
        if type(Ind) is int:
            Ind = [Ind]
            NInd = len(Ind)
        elif type(Ind) is list:
            NInd = len(Ind)
        elif type(Ind) is np.ndarray:
            assert (np.issubdtype(Ind.dtype,bool) and Ind.size==self.NFunc) or np.issubdtype(Ind.dtype,int), "Arg Ind must be a np.ndarray of boolenas with size==self.NFunc or a np.ndarray of int !"
            NInd = Ind.size

        if type(Coefs) is float:
            Coefs = Coefs*np.ones((self.NFunc,))
        if 'L' in Elt:
            LF1 = BSplineDeriv(self.Deg, self.Mesh.Knots, Deriv=0, Test=Test)
            LF = [lambda x,Coefs=Coefs,ii=ii: Coefs[Ind[ii]]*LF1[Ind[ii]](x) for ii in range(0,NInd)]
            ax = Plot_BSpline1D(self.Mesh.Knots, None, LF, ax=ax, Elt=Elt, Name=self.Id.Name, Sub=Sub, SubMode=SubMode, LFdict=LFdict, LegDict=LegDict, Test=Test)
        if 'C' in Elt or 'K' in Elt:
            Cents = self.Mesh.Cents[self.Func_Centsind[:,Ind]].flatten()
            Knots = self.Mesh.Knots[self.Func_Knotsind[:,Ind]].flatten()
            ax = Plot_Mesh1D(Knots, Cents=Cents, y=y, Elt=Elt, Name=self.Id.NameLTX, ax=ax, Kdict=Kdict, Cdict=Cdict, LegDict=LegDict, Test=Test)
        return ax

    def save(self,SaveName=None,Path=None,Mode='npz'):
        """
        Save the object in folder Name, under file name SaveName, using specified mode

        Inputs:
        ------
            SaveName    str     The name to be used to for the saved file, if None (recommended) uses Ves.Id.SaveName (default: None)
            Path        str     Path specifying where to save the file, if None (recommended) uses Ves.Id.SavePath (default: None)
            Mode        str     Flag specifying whether to save the object as a numpy array file ('.npz', recommended) or an object using cPickle (not recommended, may cause retro-compatibility issues with later versions)
        """
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path, Mode=Mode)



def _LBF1D_check_inputs(Id=None, MeshKnts=None, Deg=None, Type=None, Exp=None, shot=None, dtime=None, dtimeIn=False, SavePath=None):
    if Id is not None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID object !"
    if MeshKnts is not None:
        assert type(MeshKnts) is Mesh1D or (hasattr(MeshKnts,'__iter__') and np.asarray(MeshKnts).ndim==1 and np.asarray(MeshKnts).dtype.name=='float64'), "Arg MeshKnts must be Mesh1D object, or a 1-dim iterable of floats (knots) !"
    if not Deg is None:
        assert type(Deg) is int and Deg in [0,1,2,3], "Arg Knots must be a Mesh2D instance when ind is not None !"
    assert Type is None, "Arg Type must be None for a 1D mesh and LBF !"
    if not Exp is None:
        assert Exp in tfd.AllowedExp, "Ar Exp must be in "+str(tfd.AllowedExp)+" !"
    Ints = [shot]
    if any([not aa is None for aa in Ints]):
        assert all([aa is None or type(aa) is int for aa in Ints]), "Args [shot] must be int !"
    strs = [SavePath]
    if any([not aa is None for aa in strs]):
        assert all([aa is None or type(aa) is str for aa in strs]), "Args [SavePath] must all be str !"
    if not dtime is None:
        assert type(dtime) is dtm.datetime, "Arg dtime must be a dtm.datetime !"
    bools = [dtimeIn]
    if any([not aa is None for aa in bools]):
        assert all([aa is None or type(aa) is bool for aa in bools]), " Args [dtimeIn] must all be bool !"



















































class BF2D(object):

    def __init__(self, Id, Mesh, Deg, dtime=None):
        self.set_Id(Id,Deg=Deg, dtime=dtime)
        self.set_Mesh(Mesh,Deg=Deg)

    @property
    def Id(self):
        return self._Id
    @property
    def Mesh(self):
        return self._Mesh
    @Mesh.setter
    def Mesh(self,Val):
        self.set_Mesh(Val)
    @property
    def Deg(self):
        return self._Deg
    @Deg.setter
    def Deg(self,Val):
        self.set_BF(Deg=Val)
    @property
    def NFunc(self):
        return self._NFunc

    @property
    def Surf(self):
        indin = np.any(~np.isnan(self._Cents_Funcind),axis=0)
        return np.sum(self.Mesh._Surfs[indin])

    def set_Id(self,Id,Deg=Deg, dtime=None):
        assert type(Id) is str or isinstance(Id,tfpf.ID), "Arg Id should be string or an tfpf.ID instance !"
        if type(Id) is str:
            Id = tfpf.ID('BF2D',Id+'_D{0:01.0f}'.format(Deg), dtime=dtime)
        self._Id = Id

    def set_Mesh(self,Mesh,Deg=None):
        assert isinstance(Mesh,Mesh2D), "Arg Mesh must be a Mesh2D instance !"
        self._Mesh = Mesh
        self.set_BF(Deg=Deg)

    def set_BF(self,Deg=None):
        assert Deg is None or type(Deg) is int, "Arg Deg must be a int !"
        if not Deg is None:
            self._Deg = Deg
        BSR, RF_Kind, RF_Cind, RK_Find, RC_Find, RMaxPos = BSpline_LFunc(self._Deg, self._Mesh._MeshR._Knots)
        BSZ, ZF_Kind, ZF_Cind, ZK_Find, ZC_Find, ZMaxPos = BSpline_LFunc(self._Deg, self._Mesh._MeshZ._Knots)
        nBR, nBZ = len(BSR), len(BSZ)
        Func, F_Kind, F_Cind, F_MaxPos = [], [], [], []

        CentBckg, indCentBckInMesh, NumCentBck = self._Mesh._get_CentBckg()
        NCperF, NKperF = (self._Deg+1)**2, (self._Deg+2)**2
        for ii in range(0,nBZ):
            for jj in range(0,nBR):
                inds = ZF_Cind[:,ii].reshape((ZF_Cind.shape[0],1))*self._Mesh._MeshR._NCents + RF_Cind[:,jj]
                if np.all(indCentBckInMesh[inds]):
                    Func.append(lambda RZ, ii=ii,jj=jj: BSR[jj](RZ[0,:])*BSZ[ii](RZ[1,:]))
                    F_Cind.append(NumCentBck[inds].reshape(NCperF,1).astype(int))
                    F_Kind.append(np.unique(self._Mesh._Cents_Knotsind[:,F_Cind[-1][:,0]]).reshape(NKperF,1))
                    F_MaxPos.append(np.array([[RMaxPos[jj]],[ZMaxPos[ii]]]))
        self._LFunc = Func
        self._NFunc = len(Func)
        self._Func_Knotsind = np.concatenate(tuple(F_Kind),axis=1)
        self._Func_Centsind = np.concatenate(tuple(F_Cind),axis=1)
        self._Func_MaxPos = np.concatenate(tuple(F_MaxPos),axis=1)
        self._Cents_Funcind = self._get_Cents_Funcind(Init=True)
        self._Knots_Funcind = self._get_Knots_Funcind(Init=True)
        self._Func_InterFunc = self._get_Func_InterFunc(Init=True)

    def _get_Cents_Funcind(self,Init=True):                  # To be updated with ability to select fraction of BF
        Cent_indFunc = np.nan*np.ones(((self._Deg+1)**2,self._Mesh._NCents))
        for ii in range(0,self._Mesh._NCents):
            inds = np.any(self._Func_Centsind==ii,axis=0)
            Cent_indFunc[:inds.sum(),ii] = inds.nonzero()[0]
        return Cent_indFunc

    def _get_Knots_Funcind(self,Init=True):                  # To be updated with ability to select fraction of BF
        Knots_indFunc = np.nan*np.ones(((self._Deg+2)**2,self._Mesh._NKnots))
        for ii in range(0,self._Mesh._NKnots):
            inds = np.any(self._Func_Knotsind==ii,axis=0)
            Knots_indFunc[:inds.sum(),ii] = inds.nonzero()[0]
        return Knots_indFunc

    def _get_Func_InterFunc(self,Init=True):                  # To be updated with ability to select fraction of BF
        Func_InterFunc = np.nan*np.ones(((2*self._Deg+1)**2-1,self._NFunc))
        for ii in range(0,self.NFunc):
            ind = self._Cents_Funcind[:,self._Func_Centsind[:,ii]].flatten()
            ind = np.unique(ind[~np.isnan(ind)])
            ind = np.delete(ind,(ind==ii).nonzero()[0])
            Func_InterFunc[:ind.size,ii] = ind.astype(int)
        return Func_InterFunc

    def _get_Func_InterFunc(self, Init=False, indFin=None): # To be updated with ability to select fraction of BF
        assert indFin is None or (isnstance(indFin,np.ndarray) and indFin.dtype.name in ['bool','int32','int64','int']), "Arg indFin must be None or a np.naddary of bool or int !"
        if indFin is None and not Init:
            return self._Func_InterFunc
        elif indFin is None:
            indFin = np.arange(0,self._NFunc)
        if indFin.dtype.name=='bool':
            indFin = indFin.nonzero()[0]
        NF = indFin.size
        Func_InterFunc = np.nan*np.ones(((2*self._Deg+1)**2-1,NF))      # Update in progress from here...
        for ii in range(0,NF):
            ind = self._Cents_Funcind[:,self._Func_Centsind[:,ii]].flatten()
            ind = np.unique(ind[~np.isnan(ind)])
            ind = np.delete(ind,(ind==ii).nonzero()[0])
            Func_InterFunc[:ind.size,ii] = ind.astype(int)
        return Func_InterFunc

    def _get_quadPoints(self):
        R = self._Mesh._Knots[0,self._Func_Knotsind]
        Z = self._Mesh._Knots[1,self._Func_Knotsind]
        QuadR, QuadZ = np.zeros((self._Deg+2,self._NFunc)), np.zeros((self._Deg+2,self._NFunc))
        for ii in range(0,self._NFunc):
            QuadR[:,ii] = np.unique(R[:,ii])
            QuadZ[:,ii] = np.unique(Z[:,ii])
        return QuadR, QuadZ

    def _get_Func_SuppBounds(self):
        Func_SuppRZ = np.nan*np.ones((4,self._NFunc))
        RKnots = self._Mesh._Knots[0,self._Func_Knotsind]
        ZKnots = self._Mesh._Knots[1,self._Func_Knotsind]
        Func_SuppRZ = np.concatenate((RKnots.min(axis=0,keepdims=True), np.max(RKnots,axis=0,keepdims=True), np.min(ZKnots,axis=0,keepdims=True), np.max(ZKnots,axis=0,keepdims=True)),axis=0)
        return Func_SuppRZ

    def _get_Func_Supps(self):
        R = self._Mesh._Knots[0,self._Func_Knotsind]
        Z = self._Mesh._Knots[1,self._Func_Knotsind]
        R = np.array([np.min(R,axis=0),np.max(R,axis=0)])
        Z = np.array([np.min(Z,axis=0),np.max(Z,axis=0)])
        return [np.array([[R[0,ii],R[1,ii],R[1,ii],R[0,ii],R[0,ii]],[Z[0,ii],Z[0,ii],Z[1,ii],Z[1,ii],Z[0,ii]]]) for ii in range(0,self._NFunc)]

    def get_SubBFPolygon_indin(self, Poly, NLim=3, Out=bool):
        assert Out in [bool,int], "Arg Out must be in [bool,int] !"
        indMeshout = ~self.Mesh.get_SubMeshPolygon(Poly, NLim=NLim, Out=bool)
        indFout = self._Cents_Funcind[:,indMeshout]
        indFout = np.unique(indFout[~np.isnan(indFout)]).astype(int)
        indF = np.ones((self.NFunc,),dtype=bool)
        indF[indFout] = False
        if Out==int:
            indF = indF.nonzero()[0]
        return indF

    def get_SubBFPolygonind(self, Poly=None, NLim=3, indFin=None):
        assert indFin is None or (isinstance(indFin,np.ndarray) and indFin.dtype.name in ['bool','int','int32','int64']), "Arg indFin must be None or a np.ndarray of bool or int !"
        assert (not indFin is None and Poly is None) or (indFin is None and isinstance(Poly,np.ndarray) and Poly.ndim==2 and Poly.shape[0]==2), "If arg indFin is None, arg Poly must be a 2D np.ndarray instance !"
        if indFin is None:
            indFin = self.get_SubBFPolygon_indin(Poly, NLim=NLim, Out=int)
        if indFin.dtype.name=='bool':
            indFin = indFin.nonzero()[0]
        IndMin = np.unique(self._Func_Centsind[:,indFin].flatten())
        Id, IdM = self.Id, self.Mesh.Id
        Id._Name, IdM._Name = Id._Name+'_SubBF', Id._Name+'_SubMesh'
        M = Mesh2D(IdM, self.Mesh, IndMin)
        return BF2D(Id, M, self.Deg, self.Id.dtime)

    def get_TotFunc_FixPoints(self, Points, Deriv=0, Test=True):
        assert Deriv in [0,1,2,'D0','D0N2','D1','D1N2','D1FI','D2','D2-Gauss','D2-Mean','D2N2-Lapl','D2N2-Vect'], "Arg Deriv must be in [0,1,2,'D0','D0N2','D1','D1N2','D1FI','D2','D2-Gauss','D2-Mean','D2N2-Lapl','D2N2-Vect'] !"
        if Deriv in [0,'D0']:
            AA, m = BF2D_get_Op(self, Points, Deriv=Deriv, Test=Test)
            def FF(Coefs,AA=AA):
                return AA.dot(Coefs)
        elif Deriv in [1,2,'D1','D2']:
            AA, m = BF2D_get_Op(self, Points, Deriv=Deriv, Test=Test)
            def FF(Coefs, DVect,AA=AA):
                dvR = np.hypot(DVect[0,:],DVect[1,:])
                return dvR*AA[0].dot(Coefs) + DVect[2,:]*AA[1].dot(Coefs)
        elif Deriv == 'D0N2':
            try:
                AA, m = BF2D_get_Op(self, Points, Deriv=Deriv, Test=Test)
                def FF(Coefs,AA=AA):
                    return Coefs.dot(AA.dot(Coefs))
            except MemoryError:
                AA, m = BF2D_get_Op(self, Points, Deriv=0, Test=Test)
                def FF(Coefs,AA=AA):
                    return AA.dot(Coefs)*AA.dot(Coefs)
        elif Deriv=='D2-Gauss':
            try:
                AA, m = BF2D_get_Op(self, Points, Deriv=2, Test=Test)
                BB, n = BF2D_get_Op(self, Points, Deriv='D2N2', Test=Test)
                CC, n = BF2D_get_Op(self, Points, Deriv='D1N2', Test=Test)
                AA, BB, CC = (AA[0],AA[1]), BB[2], CC[0]+CC[1]
                def FF(Coefs, AA=AA, BB=BB, CC=CC):
                    return (AA[0].dot(Coefs) * AA[1].dot(Coefs) - Coefs.dot(BB.dot(Coefs))) / (1. + Coefs.dot(CC.dot(Coefs)))**2
            except MemoryError:
                AA, m = BF2D_get_Op(self, Points, Deriv=2, Test=Test)
                BB, n = BF2D_get_Op(self, Points, Deriv=1, Test=Test)
                def FF(Coefs, AA=AA, BB=BB):
                    return (AA[0].dot(Coefs) * AA[1].dot(Coefs) - (AA[2].dot(Coefs))**2) / (1. + (BB[0].dot(Coefs))**2+(BB[1].dot(Coefs))**2)**2
        elif Deriv=='D2-Mean':
            try:
                AA, m = BF2D_get_Op(self, Points, Deriv=1, Test=Test)
                BB, n = BF2D_get_Op(self, Points, Deriv='D1N2', Test=Test)
                CC, p = BF2D_get_Op(self, Points, Deriv=2, Test=Test)
                CC = (CC[0],CC[1],CC[2])
                def FF(Coefs,AA=AA,BB=BB,CC=CC):
                    return ( (1.+Coefs.dot(BB[0].dot(Coefs)))*CC[1].dot(Coefs) - 2.*AA[0].dot(Coefs)*AA[1].dot(Coefs)*CC[2].dot(Coefs) + (1.+Coefs.dot(BB[1].dot(Coefs)))*CC[0].dot(Coefs) ) / (2.*(1. + Coefs.dot((BB[0]+BB[1]).dot(Coefs)))**1.5)
            except MemoryError:
                AA, m = BF2D_get_Op(self, Points, Deriv=2, Test=Test)
                BB, n = BF2D_get_Op(self, Points, Deriv=1, Test=Test)
                def FF(Coefs,AA=AA,BB=BB):
                    return ( (1.+(BB[0].dot(Coefs))**2)*AA[1].dot(Coefs) - 2.*BB[0].dot(Coefs)*BB[1].dot(Coefs)*AA[2].dot(Coefs) + (1.+(BB[1].dot(Coefs))**2)*AA[0].dot(Coefs) ) / (2.*(1. + BB[0].dot(Coefs)**2+(BB[1].dot(Coefs))**2)**1.5)
        else:
            if 'D1' in Deriv:
                try:
                    AA, m = BF2D_get_Op(self, Points, Deriv='D1N2', Test=Test)
                    AA = AA[0]+AA[1]
                    if Deriv=='D1N2':
                        def FF(Coefs,AA=AA):
                            return Coefs.dot(AA.dot(Coefs))
                    elif Deriv=='D1FI':
                        B, n = BF2D_get_Op(self, Points, Deriv='D0', Test=Test)
                        def FF(Coefs,AA=AA,B=B):
                            return Coefs.dot(AA.dot(Coefs))/B.dot(Coefs)
                except MemoryError:
                    AA, m = BF2D_get_Op(self, Points, Deriv='D1', Test=Test)
                    if Deriv=='D1N2':
                        def FF(Coefs,AA=AA):
                            return (AA[0].dot(Coefs))**2 + (AA[1].dot(Coefs))**2
                    elif Deriv=='D1FI':
                        B, n = BF2D_get_Op(self, Points, Deriv='D0', Test=Test)
                        def FF(Coefs,AA=AA,B=B):
                            return ((AA[0].dot(Coefs))**2 + (AA[1].dot(Coefs))**2)/B.dot(Coefs)
            elif 'D2' in Deriv:
                try:
                    AA, m = BF2D_get_Op(self, Points, Deriv='D2N2', Test=Test)
                    if Deriv=='D2N2-Lapl':
                        AA, B = AA[0]+AA[1], AA[3]
                        def FF(Coefs,AA=AA,B=B):
                            return Coefs.dot(AA.dot(Coefs)) + 2.*B.dot(Coefs)
                    elif Deriv=='D2N2-Vect':
                        AA = AA[0]+AA[1]
                        def FF(Coefs,AA=AA):
                            return Coefs.dot(AA.dot(Coefs))
                except MemoryError:
                    AA, m = BF2D_get_Op(self, Points, Deriv='D2', Test=Test)
                    AA = (AA[0],AA[1])
                    if Deriv=='D2N2-Lapl':
                        def FF(Coefs,AA=AA):
                            return (AA[0].dot(Coefs))**2 + (AA[1].dot(Coefs))**2 + 2.*AA[0].dot(Coefs)*AA[1].dot(Coefs)
                    elif Deriv=='D2N2-Vect':
                        AA = (AA[0],AA[1])
                        def FF(Coefs,AA=AA):
                            return (AA[0].dot(Coefs))**2 + (AA[1].dot(Coefs))**2
        return FF

    def get_TotFunc_FixCoefs(self, Deriv=0, DVect=tfd.BF2_DVect_DefR, Coefs=1., Test=True):
        assert type(Coefs) in [int,float] or (isinstance(Coefs,np.ndarray) and Coefs.ndim==1 and Coefs.size==self._NFunc), "Arg Coefs must be a float or a (NF,) np.ndarray !"
        return BF2D_get_TotFunc(self, Deriv=Deriv, DVect=DVect, Coefs=Coefs, Test=Test)

    def get_TotVal(self, Points, Deriv=0, DVect=tfd.BF2_DVect_DefR, Coefs=1., Test=True):
        assert Deriv in [0,1,2,'D0','D0N2','D1','D1N2','D1FI','D2','D2-Gauss','D2-Mean','D2N2-Lapl','D2N2-Vect'], "Arg Deriv must be in [0,1,2,'D0','D0N2','D1','D1N2','D1FI','D2','D2-Gauss','D2-Mean','D2N2-Lapl','D2N2-Vect'] !"
        if type(Coefs) in [int,float] or (isinstance(Coefs,np.ndarray) and Coefs.ndim==1):
            Coefs = Coefs*np.ones((self._NFunc,))
        if Points.shape==2:
            Points = np.array([Points[0,:],np.zeros((Points.shape[1],)), Points[1,:]])
        FF = self.get_TotFunc_FixPoints(Points, Deriv=Deriv, Test=Test)
        if Coefs.ndim==1:
            if Deriv in [1,2,'D1','D2']:
                dvect = DVect(Points)
                return FF(Coefs,dvect)
            else:
                return FF(Coefs)
        else:
            Nt = Coefs.shape[0]
            Vals = np.empty((Nt,Points.shape[1]))
            if Deriv in [1,2,'D1','D2']:
                dvect = DVect(Points)
                for ii in range(0,Nt):
                    Vals[ii,:] = FF(Coefs[ii,:], dvect)
            else:
                for ii in range(0,Nt):
                    Vals[ii,:] = FF(Coefs[ii,:])
            return Vals

    def get_Coefs(self,xx=None,yy=None, zz=None, ff=None, SubP=tfd.BF2Sub, SubMode=tfd.BF1SubMode, indFin=None, Test=True):         # To be updated to take into account fraction of BF
        assert not all([xx is None or yy is None, ff is None]), "You must provide either a function (ff) or sampled data (xx and yy) !"
        assert indFin is None or (isnstance(indFin,np.ndarray) and indFin.dtype.name in ['bool','int32','int64','int']), "Arg indFin must be None or a np.naddary of bool or int !"
        print self.Id.Name+" : Getting fit coefficients..."
        if indFin is None:
            indFin = np.arange(0,self.NFunc)
        if indFin.dtype.name=='bool':
            indFin = indFin.nonzero()[0]

        if not ff is None:
            xx, yy, nx, ny = self.get_XYplot(SubP=SubP, SubMode=SubMode)        # To update
            xx, yy = xx.flatten(), yy.flatten()
            zz = ff(RZ2Points(np.array([xx,yy])))
        else:
            assert xx.shape==yy.shape==zz.shape, "Args xx, yy and zz must have same shape !"
            xx, yy, zz = xx.flatten(), yy.flatten(), zz.flatten()
        # Keep only points inside the Boundary
        ind = self.Mesh.isInside(np.array([xx,yy]))
        xx, yy, zz = xx[ind], yy[ind], zz[ind]

        AA, m = BF2D_get_Op(self, np.array([xx.flatten(),yy.flatten()]), Deriv=0, indFin=indFin, Test=Test)
        Coefs, res, rank, sing = np.linalg.lstsq(AA,zz)
        if rank < indFin.size:
            xx1, yy1, nx1, ny1 = self.get_XYplot(SubP=SubP/2., SubMode=SubMode)
            xx1, yy1 = xx1.flatten(), yy1.flatten()
            ind = self._Mesh.isInside(np.array([xx1,yy1]))
            xx1, yy1 = xx1[ind], yy1[ind]
            zz1 = scpint.interp2d(xx, yy, zzz, kind='linear', bounds_error=False, fill_value=0)(xx1,yy1)
            xx, yy, zz = np.concatenate((xx,xx1)), np.concatenate((yy,yy1)), np.concatenate((zz,zz1))
            AA, m = BF2D_get_Op(self, np.array([xx.flatten(),yy.flatten()]), Deriv=0, Test=Test)
            Coefs, res, rank, sing = np.linalg.lstsq(AA,zz)
        return Coefs, res

    def get_XYplot(self, SubP=tfd.BF2PlotSubP, SubMode=tfd.BF2PlotSubMode):
        Xplot, Yplot = Calc_SumMeshGrid2D(self.Mesh._MeshR.Knots, self.Mesh._MeshZ.Knots, SubP=SubP, SubMode=SubMode, Test=True)
        nx, ny = Xplot.size, Yplot.size
        Xplot, Yplot = np.dot(np.ones((ny,1)),Xplot.reshape((1,nx))), np.dot(Yplot.reshape((ny,1)),np.ones((1,nx)))
        return Xplot, Yplot, nx, ny

    def get_IntOp(self, Deriv=0, Mode=tfd.BF2IntMode, Sparse=tfd.L1IntOpSpa, SpaFormat=tfd.L1IntOpSpaFormat, Test=True):
        print self.Id.Name+" : Getting integral operator "+str(Deriv)
        return Calc_IntOp_BSpline2D(self, Deriv=Deriv, Mode=Mode, Sparse=Sparse, SpaFormat=SpaFormat, Test=Test)

    def get_IntVal(self, Deriv=0, Mode=tfd.BF2IntMode, Coefs=1., Test=True):
        A, m = Calc_IntOp_BSpline2D(self, Deriv=Deriv, Mode=Mode, Sparse=True, Test=Test)
        if type(Coefs) is float:
            Coefs = Coefs*np.ones((self.NFunc,),dtype=float)
        if Coefs.ndim==1:
            if m==0:
                Int = A.dot(Coefs)
            elif m==1:
                Int = Coefs.dot(A.dot(Coefs))
            elif m==2:
                A = A[0]+A[1]
                Int = Coefs.dot(A.dot(Coefs))
            else:
                print 'Not coded yet !'
        else:
            Int = np.nan*np.ones((Coefs.shape[0],))
            if m==0:
                for ii in range(0,Coefs.shape[0]):
                    Int[ii] = A.dot(Coefs[ii,:])
            elif m==1:
                for ii in range(0,Coefs.shape[0]):
                    Int[ii] = Coefs[ii,:].dot(A.dot(Coefs[ii,:]))
            elif m==2:
                A = A[0]+A[1]
                for ii in range(0,Coefs.shape[0]):
                    Int[ii] =Coefs[ii,:].dot(A.dot(Coefs[ii,:]))
            else:
                print 'Not coded yet !'
        return Int

    def get_MinMax(self, Coefs=1., Ratio=0.05, SubP=0.004, SubP1=0.015, SubP2=0.001, TwoSteps=False, SubMode='abs', Deriv='D0', Test=True):                                  # To be deprecated by get_Extrema() when implemented
        return Get_MinMax(self, Coefs=Coefs, Ratio=Ratio, SubP=SubP, SubP1=SubP1, SubP2=SubP2, TwoSteps=TwoSteps, SubMode=SubMode, Deriv=Deriv, Test=Test)

    def get_Extrema(self, Coefs=1., Ratio=0.95, SubP=0.002, SubMode='abs', Deriv='D0', D1N2=True, D2N2=True):
        return Get_Extrema(self, Coefs=Coefs, Ratio=Ratio, SubP=SubP, SubMode=SubMode, D1N2=D1N2, D2N2=D2N2)


    def plot(self, ax='None', Coefs=1., Deriv=0, Elt='T', NC=tfd.BF2PlotNC, DVect=tfd.BF2_DVect_DefR, PlotMode='contourf', SubP=tfd.BF1Sub, SubMode=tfd.BF1SubMode, Totdict=tfd.BF1Totd, LegDict=tfd.TorLegd, Test=True):
        return Plot_BSpline2D(self, ax=ax, Elt=Elt, Deriv=Deriv, Coefs=Coefs, DVect=DVect, NC=NC, PlotMode=PlotMode, Name=self.Id.Name+' '+str(Deriv), SubP=SubP, SubMode=SubMode, Totdict=Totdict, LegDict=LegDict, Test=Test)

    def plot_fit(self, ax1='None', ax2='None', xx=None, yy=None, zz=None, ff=None, NC=tfd.BF2PlotNC, PlotMode='contourf', Name='', SubP=tfd.BF2Sub, SubMode=tfd.BF1SubMode, Totdict=tfd.BF1Totd, LegDict=tfd.TorLegd, Test=True):
        Coefs, res = self.get_Coefs(xx=xx,yy=yy, zz=zz, ff=ff, SubP=SubP, SubMode=SubMode, Test=Test)
        if xx is None:
            xx, yy, nxg, nyg = self.get_XYplot(SubP=SubP, SubMode=SubMode)
            zz = ff(RZ2Points(np.array([xx.flatten(),yy.flatten()])))
            zz = zz.reshape((xx.shape))
        else:
            assert xx.ndim==2, "Args xx, yy and zz must be (N,M) plot-friendly !"
        if Name=='':
            Name = self.Id.Name

        Xplot, Yplot, nx, ny = self.get_XYplot(SubP=SubP, SubMode=SubMode)
        PointsRZ = np.array([Xplot.flatten(),Yplot.flatten()])
        ind = self.Mesh.isInside(PointsRZ)
        Val = self.get_TotVal(PointsRZ, Deriv=0, Coefs=Coefs, Test=True)
        Val[~ind] = np.nan
        Val = Val.reshape(Xplot.shape)
        if PlotMode=='surf':
            if ax1=='None' or ax2=='None':
                ax1, ax2 = tfd.Plot_BSplineFit_DefAxes('3D')
            ax1.plot_surface(xx,yy,zz, label='Model', **Totdict)
            ax2.plot_surface(Xplot,Yplot,Val, label=Name, **Totdict)
        else:
            if ax1=='None' or ax2=='None':
                ax1, ax2  = tfd.Plot_BSplineFit_DefAxes('2D')
            if PlotMode=='contour':
                ax1.contour(xx,yy,zz, NC, label='Model', **Totdict)
                ax2.contour(Xplot,Yplot,Val, NC, label=Name, **Totdict)
            elif PlotMode=='contourf':
                ax1.contourf(xx,yy,zz, NC, label='Model', **Totdict)
                ax2.contourf(Xplot,Yplot,Val, NC, label=Name, **Totdict)
        if not LegDict is None:
            ax1.legend(**LegDict)
            ax2.legend(**LegDict)
        ax2.set_title(r"$\chi^2 = "+str(res)+"$")
        ax1.figure.canvas.draw()
        return ax1, ax2

    def plot_Ind(self, Ind=0, Elt='LSP', EltM='BMCK', ax='None', Coefs=1., NC=tfd.BF2PlotNC, PlotMode='contourf', SubP=tfd.BF1Sub, SubMode=tfd.BF1SubMode, Totdict=tfd.BF2PlotTotd,
            Cdict=tfd.M2Cd, Kdict=tfd.M2Kd, Bckdict=tfd.M2Bckd, Mshdict=tfd.M2Mshd, LegDict=tfd.TorLegd, Colorbar=True, Test=True):
        assert type(Ind) in [int,list,np.ndarray], "Arg Ind must be a int, a list of int or a np.ndarray of int or booleans !"
        if type(Ind) is int:
            Ind = np.array([Ind],dtype=int)
        elif type(Ind) is list:
            Ind = np.array(tuple(Ind),dtype=int)
        elif type(Ind) is np.ndarray:
            assert (np.issubdtype(Ind.dtype,bool) and Ind.size==self._NFunc) or np.issubdtype(Ind.dtype,int), "Arg Ind must be a np.ndarray of boolenas with size==self.NFunc or a np.ndarray of int !"
            if np.issubdtype(Ind.dtype,bool):
                Ind = Ind.nonzero()[0]
        NInd = Ind.size
        if type(Coefs) is float:
            Coefs = Coefs*np.ones((self._NFunc,))
        indC, indK = self._Func_Centsind[:,Ind].flatten().astype(int), self._Func_Knotsind[:,Ind].flatten().astype(int)
        ax = self._Mesh.plot(indKnots=indK, indCents=indC, ax=ax, Elt=EltM, Cdict=Cdict, Kdict=Kdict, Bckdict=Bckdict, Mshdict=Mshdict, LegDict=LegDict, Test=Test)
        ax = Plot_BSpline2D_Ind(self, Ind, ax=ax, Coefs=Coefs, Elt=Elt, NC=NC, PlotMode=PlotMode, Name=self._Id.Name, SubP=SubP, SubMode=SubMode, Totdict=Totdict, LegDict=LegDict, Colorbar=Colorbar, Test=Test)
        return ax

    def save(self,SaveName=None,Path=None,Mode='npz'):
        if Path is None:
            Path = self.Id.SavePath
        else:
            assert type(Path) is str, "Arg Path must be a str !"
            self._Id.SavePath = Path
        if SaveName is None:
            SaveName = self.Id.SaveName
        else:
            assert type(SaveName) is str, "Arg SaveName must be a str !"
            self.Id.SaveName = SaveName
        Ext = '.npz' if 'npz' in Mode else '.pck'
        tfpf.save(self, Path+SaveName+Ext)





############################################
#####     Computing functions
############################################


def RZ2Points(PointsRZ, Theta=0.):
        return np.array([PointsRZ[0,:]*np.cos(Theta), PointsRZ[0,:]*np.sin(Theta),PointsRZ[1,:]])

def Points2RZ(Points):
    return np.array([np.sqrt(Points[0,:]**2+Points[1,:]**2),Points[2,:]])


def Calc_BF1D_Weights(LFunc, Points, Test=True):
    if Test:
        assert type(LFunc) is list, "Arg LFunc must be a list of functions !"
        assert isinstance(Points,np.ndarray) and Points.ndim==1, "Arg Points must be a (N,) np.ndarray !"
    NFunc = len(LFunc)
    Wgh = np.zeros((Points.size, NFunc))
    for ii in range(0,NFunc):
        Wgh[:,ii] = LFunc[ii](Points)
    return Wgh

def Calc_BF2D_Weights(LFunc, PointsRZ, Test=True):
    if Test:
        assert type(LFunc) is list, "Arg LFunc must be a list of functions !"
        assert isinstance(PointsRZ,np.ndarray) and PointsRZ.ndim==2, "Arg Points must be a (2,N) np.ndarray !"
    NFunc = len(LFunc)
    Wgh = np.zeros((PointsRZ.shape[1], NFunc))
    for ii in range(0,NFunc):
        Wgh[:,ii] = LFunc[ii](PointsRZ)
    return Wgh


def BF2D_get_Op(BF2, Points, Deriv=0, indFin=None, Test=True):            # To be updated to take into account only fraction of BF
    """ Return the operator to compute the desired quantity on NP points (to be multiplied by Coefs) Y = Op(Coefs) for NF basis functions
    Input :
        BF2         A BF2D instance
        Points      A (3,NP) np.ndarray indicating (X,Y,Z) cylindrical coordinates of points at which to evaluate desired quantity (automatically converted to (R,Z) coordinates)
        Deriv       A flag indicating the desired quantity in [0,1,2,'D0','D0N2','D1','D1N2','D1FI','D2','D2-Lapl','D2N2-Lapl']
        Test        A bool Flag indicating whether inputs shall be tested for conformity
    Output :
        A           The operator itself as a 2D or 3D numpy array
        m           Flag indicating the kind of operation necessary
            = 0         Y = A.dot(C)                            (Y = AC, with A.shape=(NP,NF))
            = 1         Y = C.dot(A.dot(C))                     (Y = tCAC, with A.shape=(NF,NP,NF))
            = 2         Y = sum( C.dot(A[:].dot(C)) )           (e.g.: Y = tCArC + tCAzC with Ar.shape==Az.shape=(NF,NP,NF), you can compute a scalar product with each component if necessary)
    """
    if Test:
        assert isinstance(BF2,BF2D), "Arg BF2 must be a BF2D instance !"
        assert isinstance(Points,np.ndarray) and Points.ndim==2 and Points.shape[0] in [2,3], "Arg Points must be a (2-3,NP) np.ndarray !"
        assert Deriv in [0,1,2,'D0','D0N2','D1','D1N2','D2','D2N2'], "Arg Deriv must be in [0,1,2,'D0','D0N2','D1','D1N2','D2','D2N2'] !"
        assert indFin is None or (isinstance(indFin,np.ndarray) and indFin.dtype.name in ['bool','int32','int64','int']), "Arg indFin must be None or a np.naddary of bool or int !"

    if indFin is None:
        indFin = np.arange(0,BF2.NFunc)
    if indFin.dtype.name=='bool':
        indFin = indFin.nonzero()[0]

    if type(Deriv) is int:
        Dbis = Deriv
        Deriv = 'D'+str(Dbis)
    else:
        Dbis = int(Deriv[1])
    NF = indFin.size
    NP = Points.shape[1]
    if Points.shape[0]==3:
        RZ = np.array([np.hypot(Points[0,:],Points[1,:]),Points[2,:]])
    else:
        RZ = np.copy(Points)

    QuadR, QuadZ = BF2._get_quadPoints()
    QuadR, QuadZ = QuadR[:,indFin], QuadZ[:,indFin]
    if Deriv=='D0':
        m = 0
        A = np.zeros((NP,NF))
        for ii in range(0,NF):
            A[:,ii] = BF2._LFunc[indFin[ii]](RZ)
        return A, m
    elif Deriv=='D0N2':                 # Update in progress from here...
        m = 1
        A = np.zeros((NF,NP,NF))
        for ii in range(0,NF):
            Yii = BF2._LFunc[ii](RZ)
            A[ii,:,ii] = Yii**2
            Ind = BF2._Func_InterFunc[:,ii]
            Ind = np.unique(Ind[~np.isnan(Ind)])
            for jj in range(0,Ind.size):
                A[ii,:,Ind[jj]] = Yii*BF2._LFunc[Ind[jj]](RZ)
        return A, m
    elif Deriv=='D1' and BF2.Deg>=1:
        m = 2
        Ar, Az = np.zeros((NP,NF)), np.zeros((NP,NF))
        for ii in range(0,NF):
            Ar[:,ii] = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=1)[0][0](RZ[0,:])*BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=0)[0][0](RZ[1,:])
            Az[:,ii] = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=0)[0][0](RZ[0,:])*BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=1)[0][0](RZ[1,:])
        return (Ar,Az), m
    elif Deriv=='D1N2' and BF2.Deg>=1:
        m = 2
        AR, AZ = np.zeros((NF,NP,NF)), np.zeros((NF,NP,NF))
        for ii in range(0,NF):
            rii, zii = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=0)[0][0](RZ[0,:]), BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=0)[0][0](RZ[1,:])
            Rii, Zii = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=1)[0][0](RZ[0,:]), BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=1)[0][0](RZ[1,:])
            AR[ii,:,ii] = (Rii*zii)**2
            AZ[ii,:,ii] = (rii*Zii)**2
            Ind = BF2._Func_InterFunc[:,ii]
            Ind = np.unique(Ind[~np.isnan(Ind)])
            for jj in range(0,Ind.size):
                AR[ii,:,Ind[jj]] = Rii * BSpline_LFunc(BF2.Deg,QuadR[:,Ind[jj]],Deriv=1)[0][0](RZ[0,:]) * zii * BSpline_LFunc(BF2.Deg,QuadZ[:,Ind[jj]],Deriv=0)[0][0](RZ[1,:])
                AZ[ii,:,Ind[jj]] = rii * BSpline_LFunc(BF2.Deg,QuadR[:,Ind[jj]],Deriv=0)[0][0](RZ[0,:]) * Zii * BSpline_LFunc(BF2.Deg,QuadZ[:,Ind[jj]],Deriv=1)[0][0](RZ[1,:])
        return (AR,AZ), m
    elif Deriv=='D2' and BF2.Deg>=2:
        m = 2
        Arr, Azz, Arz = np.zeros((NP,NF)), np.zeros((NP,NF)), np.zeros((NP,NF))
        for ii in range(0,NF):
            Arr[:,ii] = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=2)[0][0](RZ[0,:])*BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=0)[0][0](RZ[1,:])
            Azz[:,ii] = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=0)[0][0](RZ[0,:])*BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=2)[0][0](RZ[1,:])
            Arz[:,ii] = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=1)[0][0](RZ[0,:])*BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=1)[0][0](RZ[1,:])
        return (Arr,Azz,Arz), m
    elif Deriv=='D2N2' and BF2.Deg>=2:
        m = 2
        ARR, AZZ, ARZ, Arrzz = np.zeros((NF,NP,NF)), np.zeros((NF,NP,NF)), np.zeros((NF,NP,NF)), np.zeros((NF,NP,NF))
        for ii in range(0,NF):
            rii, zii = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=0)[0][0](RZ[0,:]), BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=0)[0][0](RZ[1,:])
            Rii, Zii = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=1)[0][0](RZ[0,:]), BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=1)[0][0](RZ[1,:])
            RRii, ZZii = BSpline_LFunc(BF2.Deg, QuadR[:,ii], Deriv=2)[0][0](RZ[0,:]), BSpline_LFunc(BF2.Deg, QuadZ[:,ii], Deriv=2)[0][0](RZ[1,:])
            ARR[ii,:,ii] = (RRii*zii)**2
            AZZ[ii,:,ii] = (rii*ZZii)**2
            ARZ[ii,:,ii] = (Rii*Zii)**2
            Arrzz[ii,:,ii] = RRii*ZZii
            Ind = BF2._Func_InterFunc[:,ii]
            Ind = np.unique(Ind[~np.isnan(Ind)])
            for jj in range(0,Ind.size):
                ARR[ii,:,Ind[jj]] = RRii * BSpline_LFunc(BF2.Deg,QuadR[:,Ind[jj]],Deriv=2)[0][0](RZ[0,:]) * zii * BSpline_LFunc(BF2.Deg,QuadZ[:,Ind[jj]],Deriv=0)[0][0](RZ[1,:])
                AZZ[ii,:,Ind[jj]] = rii * BSpline_LFunc(BF2.Deg,QuadR[:,Ind[jj]],Deriv=0)[0][0](RZ[0,:]) * ZZii * BSpline_LFunc(BF2.Deg,QuadZ[:,Ind[jj]],Deriv=2)[0][0](RZ[1,:])
                ARZ[ii,:,Ind[jj]] = Rii * BSpline_LFunc(BF2.Deg,QuadR[:,Ind[jj]],Deriv=1)[0][0](RZ[0,:]) * Zii * BSpline_LFunc(BF2.Deg,QuadZ[:,Ind[jj]],Deriv=1)[0][0](RZ[1,:])
        return (ARR,AZZ,ARZ, Arrzz), m


def BF2D_get_TotFunc(BF2, Deriv=0, DVect=tfd.BF2_DVect_DefR, Coefs=1., Test=True):
    if Test:
        assert isinstance(BF2,BF2D), "Arg BF2 must be a BF2D instance !"
        assert type(Deriv) in [int,str], "Arg Deriv must be a int or a str !"
        assert type(Coefs) is float or (isinstance(Coefs,np.ndarray) and Coefs.size==BF2.NFunc), "Arg Coefs must be a float or a np.ndarray !"

    if type(Deriv) is int:
        Dbis = Deriv
        Deriv = 'D'+str(Dbis)
    else:
        Dbis = int(Deriv[1])

    if type(Coefs) is float:
        Coefs = Coefs.np.ones((BF2.NFunc,))

    NF = BF2.NFunc
    QuadR, QuadZ = BF2._get_quadPoints()
    if Deriv in [0,'D0']:
        LF = BF2._LFunc
        return lambda Points, Coefs=Coefs, LF=LF, NF=NF: np.sum(np.concatenate(tuple([Coefs[jj]*LF[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
    elif Deriv=='D0N2':
        LF = BF2._LFunc
        return lambda Points, Coefs=Coefs, LF=LF, NF=NF: np.sum(np.concatenate(tuple([Coefs[jj]*LF[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)**2

    elif Dbis==1:
        LDR = []
        LDZ = []
        for ii in range(0,NF):
            LDR.append(lambda PointsRZ, ii=ii: BSplineDeriv(BF2.Deg, QuadR[:,ii], Deriv=1, Test=Test)[0](PointsRZ[0,:])*BSplineDeriv(BF2.Deg, QuadZ[:,ii], Deriv=0, Test=Test)[0](PointsRZ[1,:]))
            LDZ.append(lambda PointsRZ, ii=ii: BSplineDeriv(BF2.Deg, QuadR[:,ii], Deriv=0, Test=Test)[0](PointsRZ[0,:])*BSplineDeriv(BF2.Deg, QuadZ[:,ii], Deriv=1, Test=Test)[0](PointsRZ[1,:]))
        if Deriv=='D1':
            def FTot(Points, Coefs=Coefs, DVect=DVect, NF=NF, LDR=LDR, LDZ=LDZ):
                DVect = DVect(Points)
                Theta = np.arctan2(Points[1,:],Points[0,:])
                eR = np.array([np.cos(Theta),np.sin(Theta),np.zeros((Theta.size,))])
                DVect = np.array([np.sum(DVect*eR,axis=0), DVect[2,:]])
                ValR = np.sum(np.concatenate(tuple([Coefs[jj]*LDR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)*DVect[0,:]
                ValZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)*DVect[1,:]
                return ValR+ValZ
        elif Deriv=='D1N2':
            def FTot(Points, Coefs=Coefs, NF=NF, LDR=LDR, LDZ=LDZ):
                ValR = np.sum(np.concatenate(tuple([Coefs[jj]*LDR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                ValZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                return ValR**2+ValZ**2
        elif Deriv=='D1FI':
            LF = BF2._LFunc
            def FTot(Points, Coefs=Coefs, NF=NF, LF=LF, LDR=LDR, LDZ=LDZ):
                ValR = np.sum(np.concatenate(tuple([Coefs[jj]*LDR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                ValZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                Val = np.sum(np.concatenate(tuple([Coefs[jj]*LF[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                return (ValR**2+ValZ**2)/Val

    elif Dbis==2:
        LDRR, LDZZ = [], []
        for ii in range(0,NF):
            LDRR.append(lambda PointsRZ, ii=ii: BSplineDeriv(BF2.Deg, QuadR[:,ii], Deriv=2, Test=Test)[0](PointsRZ[0,:])*BSplineDeriv(BF2.Deg, QuadZ[:,ii], Deriv=0, Test=Test)[0](PointsRZ[1,:]))
            LDZZ.append(lambda PointsRZ, ii=ii: BSplineDeriv(BF2.Deg, QuadR[:,ii], Deriv=0, Test=Test)[0](PointsRZ[0,:])*BSplineDeriv(BF2.Deg, QuadZ[:,ii], Deriv=2, Test=Test)[0](PointsRZ[1,:]))
        if Deriv=='D2-Lapl':
            def FTot(Points, Coefs=Coefs, NF=NF, LDRR=LDRR, LDZZ=LDZZ):
                ValR = np.sum(np.concatenate(tuple([Coefs[jj]*LDRR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                ValZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                return ValR+ValZ
        elif Deriv=='D2N2-Lapl':
            def FTot(Points, Coefs=Coefs, NF=NF, LDRR=LDRR, LDZZ=LDZZ):
                ValR = np.sum(np.concatenate(tuple([Coefs[jj]*LDRR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                ValZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                return (ValR+ValZ)**2
        else:
            LDR, LDZ, LDRZ= [], [], []
            for ii in range(0,NF):
                LDR.append(lambda PointsRZ, ii=ii: BSplineDeriv(BF2.Deg, QuadR[:,ii], Deriv=1, Test=Test)[0](PointsRZ[0,:])*BSplineDeriv(BF2.Deg, QuadZ[:,ii], Deriv=0, Test=Test)[0](PointsRZ[1,:]))
                LDZ.append(lambda PointsRZ, ii=ii: BSplineDeriv(BF2.Deg, QuadR[:,ii], Deriv=0, Test=Test)[0](PointsRZ[0,:])*BSplineDeriv(BF2.Deg, QuadZ[:,ii], Deriv=1, Test=Test)[0](PointsRZ[1,:]))
                LDRZ.append(lambda PointsRZ, ii=ii: BSplineDeriv(BF2.Deg, QuadR[:,ii], Deriv=1, Test=Test)[0](PointsRZ[0,:])*BSplineDeriv(BF2.Deg, QuadZ[:,ii], Deriv=1, Test=Test)[0](PointsRZ[1,:]))
            if Deriv=='D2-Gauss':
                def FTot(Points, Coefs=Coefs, NF=NF, LDR=LDR, LDZ=LDZ, LDRR=LDRR, LDZZ=LDZZ, LDRZ=LDRZ):
                    ValRR = np.sum(np.concatenate(tuple([Coefs[jj]*LDRR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    ValZZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    ValRZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDRZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    ValR = np.sum(np.concatenate(tuple([Coefs[jj]*LDR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    ValZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    return (ValRR*ValZZ-ValRZ**2)/(1. + ValR**2 + ValZ**2)**2
            elif Deriv=='D2-Mean':
                def FTot(Points, Coefs=Coefs, NF=NF, LDR=LDR, LDZ=LDZ, LDRR=LDRR, LDZZ=LDZZ, LDRZ=LDRZ):
                    ValRR = np.sum(np.concatenate(tuple([Coefs[jj]*LDRR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    ValZZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    ValRZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDRZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    ValR = np.sum(np.concatenate(tuple([Coefs[jj]*LDR[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    ValZ = np.sum(np.concatenate(tuple([Coefs[jj]*LDZ[jj](Points2RZ(Points)).reshape((1,Points.shape[1])) for jj in range(0,NF)]),axis=0), axis=0)
                    return ((1.+ValR**2)*ValZZ - 2.*ValR*ValZ*ValRZ + (1.+ValZ**2)*ValRR)/(2.*(1. + ValR**2 + ValZ**2)**(1.5))

    return FTot


def Calc_BF2D_Val(LFunc, Points, Coef=1., Test=True):
    if Test:
        assert type(LFunc) is list, "Arg LFunc must be a list of functions !"
        assert isinstance(Points,np.ndarray) and (Points.shape[0]==2 or Points.shape[0]==3), "Arg Points must be a (2,N) or (3,N) np.ndarray !"
        assert (type(Coef) is float and Coef==1.) or (isinstance(Coef,np.ndarray) and Coef.shape[0]==len(LFunc)), "Arg Coef must be a (BF2.NFunc,1) np.ndarray !"

    if Points.shape[0]==3:
        R = np.sqrt(np.sum(Points[0:2,:]**2,axis=0,keepdims=False))
        Points = np.array([[R],[Points[2,:]]])

    NFunc = len(LFunc)
    Val = np.zeros((Points.shape[1],))
    Coef = Coef*np.ones((NFunc,1))
    for ii in range(0,NFunc):
        Val = Val + Coef[ii]*LFunc[ii](Points)

    return Val









def Calc_Integ_BF2(BF2, Coefs=1., Mode='Vol', Test=True):
    if Test:
        assert isinstance(BF2,BF2D), "Arg BF2 must be a BF2D instance !"
        assert Mode=='Vol' or Mode=='Surf', "Arg Mode must be 'Vol' or 'Surf' !"
        assert type(Coefs) is float or (isinstance(Coefs,np.ndarray) and Coefs.size==BF2.NFunc), "Arg Coefs must be a (BF2.NFunc,) np.ndarray !"
        assert BF2.Deg <= 3, "Arg BF2 should not have Degree > 3 !"

    if type(Coefs) is float:
        Coefs = Coefs*np.ones((BF2.NFunc,))
    if Mode=='Surf':
        if BF2.Deg==0:
            Supp = BF2.get_Func_SuppRZ()
            Int = np.sum(Coefs.flatten()*(Supp[1,:]-Supp[0,:])*(Supp[3,:]-Supp[2,:]))
        elif BF2.Deg==1:
            Supp = BF2.get_Func_SuppRZ()
            Int = np.sum(Coefs.flatten()*0.25*(Supp[1,:]-Supp[0,:])*(Supp[3,:]-Supp[2,:]))
        elif BF2.Deg==2:
            QR, QZ = BF2._get_quadPoints()
            IntR1 = (QR[1,:]-QR[0,:])**2/(3.*(QR[2,:]-QR[0,:]))
            IntR21 = (QR[2,:]**2 -2.*QR[1,:]**2+QR[1,:]*QR[2,:]+3.*QR[0,:]*(QR[1,:]-QR[2,:]))/(6*(QR[2,:]-QR[0,:]))
            IntR22 = (-2.*QR[2,:]**2+QR[1,:]**2+QR[1,:]*QR[2,:]+3.*QR[3,:]*(QR[2,:]-QR[1,:]))/(6.*(QR[3,:]-QR[1,:]))
            IntR3 = (QR[3,:]-QR[2,:])**2/(3.*(QR[3,:]-QR[1,:]))
            IntZ1 = (QZ[1,:]-QZ[0,:])**2/(3.*(QZ[2,:]-QZ[0,:]))
            IntZ21 = (QZ[2,:]**2 -2*QZ[1,:]**2+QZ[1,:]*QZ[2,:]+3.*QZ[0,:]*(QZ[1,:]-QZ[2,:]))/(6*(QZ[2,:]-QZ[0,:]))
            IntZ22 = (-2.*QZ[2,:]**2+QZ[1,:]**2+QZ[1,:]*QZ[2,:]+3.*QZ[3,:]*(QZ[2,:]-QZ[1,:]))/(6.*(QZ[3,:]-QZ[1,:]))
            IntZ3 = (QZ[3,:]-QZ[2,:])**2/(3.*(QZ[3,:]-QZ[1,:]))
            Int = np.sum(Coefs.flatten() * (IntR1+IntR21+IntR22+IntR3) * (IntZ1+IntZ21+IntZ22+IntZ3))
        elif BF2.Deg==3:
            print "NOT CODED YET !"

    else:
        if BF2.Deg==0:
            Supp = BF2.get_Func_SuppRZ()
            Int = np.sum( Coefs.flatten() * 2.*np.pi * 0.5*(Supp[1,:]**2-Supp[0,:]**2)*(Supp[3,:]-Supp[2,:]))
        elif BF2.Deg==1:
            Supp = BF2.get_Func_SuppRZ()
            QuadR, QuadZ = BF2._get_quadPoints()
            Int = np.sum( Coefs.flatten() * 2.*np.pi * 0.5*(QuadR[2,:]**2-QuadR[0,:]**2 + QuadR[1,:]*(QuadR[2,:]-QuadR[0,:]))*(Supp[3,:]-Supp[2,:])/6.)
        elif BF2.Deg==2:
            QR, QZ = BF2._get_quadPoints()
            IntR1 = (3.*QR[1,:]**3+QR[0,:]**3 -5.*QR[0,:]*QR[1,:]**2+QR[0,:]**2*QR[1,:])/(12.*(QR[2,:]-QR[0,:]))
            IntR21 = (QR[2,:]**3 -3.*QR[1,:]**3+QR[1,:]**2*QR[2,:]+QR[1,:]*QR[2,:]**2 -2.*QR[0,:]*QR[2,:]**2 -2.*QR[0,:]*QR[1,:]*QR[2,:] +4.*QR[0,:]*QR[1,:]**2)/(12.*(QR[2,:]-QR[0,:]))
            IntR22 = ( -3.*QR[2,:]**3+QR[1,:]**3+QR[1,:]*QR[2,:]**2+QR[1,:]**2*QR[2,:]+4.*QR[2,:]**2*QR[3,:]-2.*QR[1,:]*QR[2,:]*QR[3,:]-2.*QR[1,:]**2*QR[3,:] )/(12.*(QR[3,:]-QR[1,:]))
            IntR3 = ( QR[3,:]**3 +3.*QR[2,:]**3 -5.*QR[2,:]**2*QR[3,:]+QR[2,:]*QR[3,:]**2)/(12.*(QR[3,:]-QR[1,:]))
            IntZ1 = (QZ[1,:]-QZ[0,:])**2/(3.*(QZ[2,:]-QZ[0,:]))
            IntZ21 = (QZ[2,:]**2 -2*QZ[1,:]**2+QZ[1,:]*QZ[2,:]+3.*QZ[0,:]*(QZ[1,:]-QZ[2,:]))/(6*(QZ[2,:]-QZ[0,:]))
            IntZ22 = (-2.*QZ[2,:]**2+QZ[1,:]**2+QZ[1,:]*QZ[2,:]+3.*QZ[3,:]*(QZ[2,:]-QZ[1,:]))/(6.*(QZ[3,:]-QZ[1,:]))
            IntZ3 = (QZ[3,:]-QZ[2,:])**2/(3.*(QZ[3,:]-QZ[1,:]))
            Int = np.sum(Coefs.flatten() * 2.*np.pi * (IntR1+IntR21+IntR22+IntR3) * (IntZ1+IntZ21+IntZ22+IntZ3))
        elif BF2.Deg==3:
            print "NOT CODED YET !"

    return Int







def Calc_IntOp_BSpline2D(BF2, Deriv=0, Mode='Vol', Sparse=tfd.L1IntOpSpa, SpaFormat=tfd.L1IntOpSpaFormat, Test=True):
    if Test:
        assert isinstance(BF2,BF2D), "Arg BF2 must be a BF2D instance !"
        assert Mode in ['Surf','Vol'], "Arg Mode must be in ['Surf','Vol'] !"
        assert (type(Deriv) is int and Deriv<=BF2.Deg) or (type(Deriv) is str and Deriv in ['D0','D1','D2','D3','D0N2','D1N2','D2N2','D3N2','D1FI'] and int(Deriv[1])<=BF2.Deg), "Arg Deriv must be a int or a str !"

    if type(Deriv) is int:
        Dbis = Deriv
        Deriv = 'D'+str(Dbis)
    else:
        Dbis = int(Deriv[1])

    if Deriv=='D0':
        m = 0
        if BF2.Deg==0:
            Supp = BF2._get_Func_SuppBounds()
            if Mode=='Surf':
                A = (Supp[1,:]-Supp[0,:]) * (Supp[3,:]-Supp[2,:])
            else:
                A = 0.5*(Supp[1,:]**2-Supp[0,:]**2) * (Supp[3,:]-Supp[2,:])
        elif BF2.Deg==1:
            Supp = BF2._get_Func_SuppBounds()
            if Mode=='Surf':
                A = 0.25*(Supp[1,:]-Supp[0,:])*(Supp[3,:]-Supp[2,:])
            else:
                QuadR, QuadZ = BF2._get_quadPoints()
                A = 0.5*(QuadR[2,:]**2-QuadR[0,:]**2 + QuadR[1,:]*(QuadR[2,:]-QuadR[0,:])) * (Supp[3,:]-Supp[2,:])/6.
        elif BF2.Deg==2:
            QR, QZ = BF2._get_quadPoints()

            IntZ1 = (QZ[1,:]-QZ[0,:])**2/(3.*(QZ[2,:]-QZ[0,:]))
            IntZ21 = (QZ[2,:]**2 -2*QZ[1,:]**2+QZ[1,:]*QZ[2,:]+3.*QZ[0,:]*(QZ[1,:]-QZ[2,:]))/(6*(QZ[2,:]-QZ[0,:]))
            IntZ22 = (-2.*QZ[2,:]**2+QZ[1,:]**2+QZ[1,:]*QZ[2,:]+3.*QZ[3,:]*(QZ[2,:]-QZ[1,:]))/(6.*(QZ[3,:]-QZ[1,:]))
            IntZ3 = (QZ[3,:]-QZ[2,:])**2/(3.*(QZ[3,:]-QZ[1,:]))
            IntZ = IntZ1+IntZ21+IntZ22+IntZ3

            if Mode=='Surf':
                IntR1 = (QR[1,:]-QR[0,:])**2/(3.*(QR[2,:]-QR[0,:]))
                IntR21 = (QR[2,:]**2 -2.*QR[1,:]**2+QR[1,:]*QR[2,:]+3.*QR[0,:]*(QR[1,:]-QR[2,:]))/(6*(QR[2,:]-QR[0,:]))
                IntR22 = (-2.*QR[2,:]**2+QR[1,:]**2+QR[1,:]*QR[2,:]+3.*QR[3,:]*(QR[2,:]-QR[1,:]))/(6.*(QR[3,:]-QR[1,:]))
                IntR3 = (QR[3,:]-QR[2,:])**2/(3.*(QR[3,:]-QR[1,:]))
            else:
                IntR1 = (3.*QR[1,:]**3+QR[0,:]**3 -5.*QR[0,:]*QR[1,:]**2+QR[0,:]**2*QR[1,:])/(12.*(QR[2,:]-QR[0,:]))
                IntR21 = (QR[2,:]**3 -3.*QR[1,:]**3+QR[1,:]**2*QR[2,:]+QR[1,:]*QR[2,:]**2 -2.*QR[0,:]*QR[2,:]**2 -2.*QR[0,:]*QR[1,:]*QR[2,:] +4.*QR[0,:]*QR[1,:]**2)/(12.*(QR[2,:]-QR[0,:]))
                IntR22 = ( -3.*QR[2,:]**3+QR[1,:]**3+QR[1,:]*QR[2,:]**2+QR[1,:]**2*QR[2,:]+4.*QR[2,:]**2*QR[3,:]-2.*QR[1,:]*QR[2,:]*QR[3,:]-2.*QR[1,:]**2*QR[3,:] )/(12.*(QR[3,:]-QR[1,:]))
                IntR3 = ( QR[3,:]**3 +3.*QR[2,:]**3 -5.*QR[2,:]**2*QR[3,:]+QR[2,:]*QR[3,:]**2)/(12.*(QR[3,:]-QR[1,:]))
            IntR = IntR1+IntR21+IntR22+IntR3
            A = IntR*IntZ
        elif BF2.Deg==3:
            print "NOT CODED YET !"
            A = 0

    elif Deriv=='D0N2':
        m = 1
        if BF2.Deg==0:
            Supp = BF2._get_Func_SuppBounds()
            if Mode=='Surf':
                A = scpsp.diags([(Supp[1,:]-Supp[0,:]) * (Supp[3,:]-Supp[2,:])],[0],shape=None,format=SpaFormat)
            else:
                A = scpsp.diags([0.5*(Supp[1,:]**2-Supp[0,:]**2) * (Supp[3,:]-Supp[2,:])],[0],shape=None,format=SpaFormat)
        elif BF2.Deg==1:
            Supp = BF2._get_Func_SuppBounds()
            QR, QZ = BF2._get_quadPoints()
            if Mode=='Surf':
                d0R, d0Z = (Supp[1,:]-Supp[0,:])/3., (Supp[3,:]-Supp[2,:])/3.
                LL = []
                for ii in range(0,BF2.NFunc):
                    ll = np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intR = d0R[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            intR = (kR[1]-kR[0])/6.
                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZ = d0Z[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            intZ = (kZ[1]-kZ[0])/6.
                        ll[0,Ind[jj]] = intR*intZ
                    ll[0,ii] = d0R[ii]*d0Z[ii]
                    LL.append(scpsp.coo_matrix(ll))
            else:
                d0R, d0Z = (QR[2,:]**2-QR[0,:]**2 + 2.*QR[1,:]*(QR[2,:]-QR[0,:]))/12., (Supp[3,:]-Supp[2,:])/3.
                LL = []
                for ii in range(0,BF2.NFunc):
                    ll = np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intR = d0R[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            intR = (kR[1]**2-kR[0]**2)/12.
                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZ = d0Z[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            intZ = (kZ[1]-kZ[0])/6.
                        ll[0,Ind[jj]] = intR*intZ
                    ll[0,ii] = d0R[ii]*d0Z[ii]
                    LL.append(scpsp.coo_matrix(ll))
            A = scpsp.vstack(LL,format='csr')
        elif BF2.Deg==2:
            Supp = BF2._get_Func_SuppBounds()
            QR, QZ = BF2._get_quadPoints()
            if Mode=='Surf':
                d0R21 = (QR[2,:]-QR[1,:])*(10.*QR[0,:]**2+6.*QR[1,:]**2+3.*QR[1,:]*QR[2,:]+QR[2,:]**2 - 5.*QR[0,:]*(3.*QR[1,:]+QR[2,:]))/(30.*(QR[2,:]-QR[0,:])**2)
                d0R22 = (QR[2,:]-QR[1,:])*(10.*QR[3,:]**2+6.*QR[2,:]**2+3.*QR[1,:]*QR[2,:]+QR[1,:]**2 - 5.*QR[3,:]*(3.*QR[2,:]+QR[1,:]))/(30.*(QR[3,:]-QR[1,:])**2)
                d0R23 = ( 3.*QR[1,:]**2 + 4.*QR[1,:]*QR[2,:] + 3.*QR[2,:]**2 - 5.*QR[0,:]*(QR[1,:]+QR[2,:]-2.*QR[3,:]) -5.*QR[3,:]*(QR[1,:]+QR[2,:]) )*(QR[1,:]-QR[2,:])/(60.*(QR[2,:]-QR[0,:])*(QR[3,:]-QR[1,:]))
                d0Z21 = (QZ[2,:]-QZ[1,:])*(10.*QZ[0,:]**2+6.*QZ[1,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[2,:]**2 - 5.*QZ[0,:]*(3.*QZ[1,:]+QZ[2,:]))/(30.*(QZ[2,:]-QZ[0,:])**2)
                d0Z22 = (QZ[2,:]-QZ[1,:])*(10.*QZ[3,:]**2+6.*QZ[2,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[1,:]**2 - 5.*QZ[3,:]*(3.*QZ[2,:]+QZ[1,:]))/(30.*(QZ[3,:]-QZ[1,:])**2)
                d0Z23 = ( 3.*QZ[1,:]**2 + 4.*QZ[1,:]*QZ[2,:] + 3.*QZ[2,:]**2 - 5.*QZ[0,:]*(QZ[1,:]+QZ[2,:]-2.*QZ[3,:]) -5.*QZ[3,:]*(QZ[1,:]+QZ[2,:]) )*(QZ[1,:]-QZ[2,:])/(60.*(QZ[2,:]-QZ[0,:])*(QZ[3,:]-QZ[1,:]))
                d0R = (QR[1,:]-QR[0,:])**3/(5.*(QR[2,:]-QR[0,:])**2) + d0R21 + d0R22 + 2.*d0R23 + (QR[3,:]-QR[2,:])**3/(5.*(QR[3,:]-QR[1,:])**2)
                d0Z = (QZ[1,:]-QZ[0,:])**3/(5.*(QZ[2,:]-QZ[0,:])**2) + d0Z21 + d0Z22 + 2.*d0Z23 + (QZ[3,:]-QZ[2,:])**3/(5.*(QZ[3,:]-QZ[1,:])**2)
                LL = []
                for ii in range(0,BF2.NFunc):
                    ll = np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intR = d0R[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            if kR.size==2 and np.all(kR==QR[0:2,ii]):
                                intR = (QR[1,ii]-QR[0,ii])**3/(30.*(QR[2,ii]-QR[0,ii])*(QR[3,Ind[jj]]-QR[1,Ind[jj]]))
                            elif kR.size==3 and np.all(kR==QR[0:3,ii]):
                                intR = ( 6.*QR[1,ii]**2 - QR[0,ii]*(9.*QR[1,ii]+QR[2,ii]-10.*QR[3,ii]) + QR[2,ii]*(QR[2,ii]-4.*QR[3,ii]) +3.*QR[1,ii]*(QR[2,ii]-2.*QR[3,ii]) )*(QR[2,ii]-QR[1,ii])**2/(30.*(QR[1,ii]-QR[3,ii])*(QR[2,ii]-QR[0,ii])**2)  +  ( QR[0,ii]**2 + 3.*QR[0,ii]*QR[1,ii] + 6.*QR[1,ii]**2 - 2.*QR[0,Ind[jj]]*(2.*QR[0,ii]+3.*QR[1,ii] - 5.*QR[2,ii]) - QR[0,ii]*QR[2,ii] -9.*QR[1,ii]*QR[2,ii] )*(QR[1,ii]-QR[0,ii])**2/(30.*(QR[0,Ind[jj]]-QR[2,Ind[jj]])*(QR[2,ii]-QR[0,ii])**2)
                            elif kR.size==2 and np.all(kR==QR[-2:,ii]):
                                intR = (QR[3,ii]-QR[2,ii])**3/(30.*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[0,Ind[jj]]))
                            elif kR.size==3 and np.all(kR==QR[-3:,ii]):
                                intR = ( 6.*QR[2,ii]**2 - QR[1,ii]*(9.*QR[2,ii]+QR[3,ii]-10.*QR[3,Ind[jj]]) + QR[3,ii]*(QR[3,ii]-4.*QR[3,Ind[jj]]) +3.*QR[2,ii]*(QR[3,ii]-2.*QR[3,Ind[jj]]) )*(QR[3,ii]-QR[2,ii])**2/(30.*(QR[2,ii]-QR[3,Ind[jj]])*(QR[3,ii]-QR[1,ii])**2)  +  ( QR[1,ii]**2 + 3.*QR[1,ii]*QR[2,ii] + 6.*QR[2,ii]**2 - 2.*QR[0,ii]*(2.*QR[1,ii]+3.*QR[2,ii] - 5.*QR[3,ii]) - QR[1,ii]*QR[3,ii] -9.*QR[2,ii]*QR[3,ii] )*(QR[2,ii]-QR[1,ii])**2/(30.*(QR[0,ii]-QR[2,ii])*(QR[3,ii]-QR[1,ii])**2)
                            else:
                                assert np.all(kR==QR[0:3,ii]), "Something wrong with common R knots..."
                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZ = d0Z[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            if kZ.size==2 and np.all(kZ==QZ[0:2,ii]):
                                intZ = (QZ[1,ii]-QZ[0,ii])**3/(30.*(QZ[2,ii]-QZ[0,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]]))
                            elif kZ.size==3 and np.all(kZ==QZ[0:3,ii]):
                                intZ = ( 6.*QZ[1,ii]**2 - QZ[0,ii]*(9.*QZ[1,ii]+QZ[2,ii]-10.*QZ[3,ii]) + QZ[2,ii]*(QZ[2,ii]-4.*QZ[3,ii]) +3.*QZ[1,ii]*(QZ[2,ii]-2.*QZ[3,ii]) )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[1,ii]-QZ[3,ii])*(QZ[2,ii]-QZ[0,ii])**2)  +  ( QZ[0,ii]**2 + 3.*QZ[0,ii]*QZ[1,ii] + 6.*QZ[1,ii]**2 - 2.*QZ[0,Ind[jj]]*(2.*QZ[0,ii]+3.*QZ[1,ii] - 5.*QZ[2,ii]) - QZ[0,ii]*QZ[2,ii] -9.*QZ[1,ii]*QZ[2,ii] )*(QZ[1,ii]-QZ[0,ii])**2/(30.*(QZ[0,Ind[jj]]-QZ[2,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2)
                            elif kZ.size==2 and np.all(kZ==QZ[-2:,ii]):
                                intZ = (QZ[3,ii]-QZ[2,ii])**3/(30.*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]]))
                            elif kZ.size==3 and np.all(kZ==QZ[-3:,ii]):
                                intZ = ( 6.*QZ[2,ii]**2 - QZ[1,ii]*(9.*QZ[2,ii]+QZ[3,ii]-10.*QZ[3,Ind[jj]]) + QZ[3,ii]*(QZ[3,ii]-4.*QZ[3,Ind[jj]]) +3.*QZ[2,ii]*(QZ[3,ii]-2.*QZ[3,Ind[jj]]) )*(QZ[3,ii]-QZ[2,ii])**2/(30.*(QZ[2,ii]-QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2)  +  ( QZ[1,ii]**2 + 3.*QZ[1,ii]*QZ[2,ii] + 6.*QZ[2,ii]**2 - 2.*QZ[0,ii]*(2.*QZ[1,ii]+3.*QZ[2,ii] - 5.*QZ[3,ii]) - QZ[1,ii]*QZ[3,ii] -9.*QZ[2,ii]*QZ[3,ii] )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[0,ii]-QZ[2,ii])*(QZ[3,ii]-QZ[1,ii])**2)
                            else:
                                assert np.all(kZ==QZ[0:3,ii]), "Something wrong with common Z knots..."
                        ll[0,Ind[jj]] = intR*intZ
                    ll[0,ii] = d0R[ii]*d0Z[ii]
                    LL.append(scpsp.coo_matrix(ll))
                A = scpsp.vstack(LL,format='csr')

            elif Mode=='Vol':
                d0R21 = (10*QR[1,:]**3 + 6.*QR[1,:]**2*QR[2,:] + 3.*QR[1,:]*QR[2,:]**2 + QR[2,:]**3 + 5.*QR[0,:]**2*(3.*QR[1,:]+QR[2,:]) - 4.*QR[0,:]*(6.*QR[1,:]**2+3.*QR[1,:]*QR[2,:]+QR[2,:]**2))*(QR[2,:]-QR[1,:])/(60.*(QR[2,:]-QR[0,:])**2)
                d0R22 = (10*QR[2,:]**3 + 6.*QR[2,:]**2*QR[1,:] + 3.*QR[2,:]*QR[1,:]**2 + QR[1,:]**3 + 5.*QR[3,:]**2*(3.*QR[2,:]+QR[1,:]) - 4.*QR[3,:]*(6.*QR[2,:]**2+3.*QR[2,:]*QR[1,:]+QR[1,:]**2))*(QR[2,:]-QR[1,:])/(60.*(QR[3,:]-QR[1,:])**2)
                d0R23 = (2.*QR[1,:]**3 + QR[1,:]*QR[2,:]*(3.*QR[2,:]-4.*QR[3,:]) +(2.*QR[2,:]-3.*QR[3,:])*QR[2,:]**2 +3.*(QR[2,:]-QR[3,:])*QR[1,:]**2 + QR[0,:]*(-3.*QR[1,:]**2-4.*QR[1,:]*QR[2,:]-3.*QR[2,:]**2+5.*QR[1,:]*QR[3,:]+5.*QR[2,:]*QR[3,:]) )*(QR[1,:]-QR[2,:])/(60.*(QR[3,:]-QR[1,:])*(QR[2,:]-QR[0,:]))
                d0Z21 = (QZ[2,:]-QZ[1,:])*(10.*QZ[0,:]**2+6.*QZ[1,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[2,:]**2 - 5.*QZ[0,:]*(3.*QZ[1,:]+QZ[2,:]))/(30.*(QZ[2,:]-QZ[0,:])**2)
                d0Z22 = (QZ[2,:]-QZ[1,:])*(10.*QZ[3,:]**2+6.*QZ[2,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[1,:]**2 - 5.*QZ[3,:]*(3.*QZ[2,:]+QZ[1,:]))/(30.*(QZ[3,:]-QZ[1,:])**2)
                d0Z23 = ( 3.*QZ[1,:]**2 + 4.*QZ[1,:]*QZ[2,:] + 3.*QZ[2,:]**2 - 5.*QZ[0,:]*(QZ[1,:]+QZ[2,:]-2.*QZ[3,:]) -5.*QZ[3,:]*(QZ[1,:]+QZ[2,:]) )*(QZ[1,:]-QZ[2,:])/(60.*(QZ[2,:]-QZ[0,:])*(QZ[3,:]-QZ[1,:]))
                d0R = (5.*QR[1,:]+QR[0,:])*(QR[1,:]-QR[0,:])**3/(30.*(QR[2,:]-QR[0,:])**2) + d0R21 + d0R22 + 2.*d0R23 + (5.*QR[2,:]+QR[3,:])*(QR[3,:]-QR[2,:])**3/(30.*(QR[3,:]-QR[1,:])**2)
                d0Z = (QZ[1,:]-QZ[0,:])**3/(5.*(QZ[2,:]-QZ[0,:])**2) + d0Z21 + d0Z22 + 2.*d0Z23 + (QZ[3,:]-QZ[2,:])**3/(5.*(QZ[3,:]-QZ[1,:])**2)
                LL = []
                for ii in range(0,BF2.NFunc):
                    ll = np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intR = d0R[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            if kR.size==2 and np.all(kR==QR[0:2,ii]):
                                intR = (QR[1,ii]+QR[0,ii])*(QR[1,ii]-QR[0,ii])**3/(60.*(QR[1,ii]-QR[1,Ind[jj]])*(QR[2,ii]-QR[0,ii]))
                            elif kR.size==3 and np.all(kR==QR[0:3,ii]):
                                intR1A = ( -2.*QR[0,Ind[jj]]*QR[0,ii] + QR[0,ii]**2 - 3.*QR[0,Ind[jj]]*QR[1,ii] + 2.*QR[0,ii]*QR[1,ii] + 2.*QR[1,ii]**2 )*(QR[1,ii]-QR[0,ii])**2/(60.*(QR[1,ii]-QR[0,Ind[jj]])*(QR[2,ii]-QR[0,ii]))
                                intR1B = - ( QR[0,ii]**2 + 2.*QR[1,ii]*(5.*QR[1,ii]-6.*QR[2,ii]) + QR[0,ii]*(4.*QR[1,ii]-3.*QR[2,ii]) )*(QR[1,ii]-QR[0,ii])**2/(60.*(QR[2,ii]-QR[0,ii])**2)
                                intR2A = ( 10.*QR[1,ii]**2 + 4.*QR[1,ii]*QR[2,ii] + QR[2,ii]**2 - 3.*QR[0,ii]*(4.*QR[1,ii]+QR[2,ii]) )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])**2)
                                intR2B = - ( 2.*QR[1,ii]**2 + 2.*QR[1,ii]*QR[2,ii] + QR[2,ii]**2 - 3.*QR[1,ii]*QR[3,ii] -2.*QR[2,ii]*QR[3,ii] )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii]))
                                intR = intR1A + intR1B + intR2A + intR2B

                            elif kR.size==2 and np.all(kR==QR[-2:,ii]):
                                intR = (QR[3,ii]+QR[2,ii])*(QR[3,ii]-QR[2,ii])**3/(60.*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[2,ii]))
                            elif kR.size==3 and np.all(kR==QR[-3:,ii]):
                                intR1A = ( -2.*QR[0,ii]*QR[1,ii] + QR[1,ii]**2 - 3.*QR[0,ii]*QR[2,ii] + 2.*QR[1,ii]*QR[2,ii] + 2.*QR[2,ii]**2 )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii]))
                                intR1B = - ( QR[1,ii]**2 + 2.*QR[2,ii]*(5.*QR[2,ii]-6.*QR[3,ii]) + QR[1,ii]*(4.*QR[2,ii]-3.*QR[3,ii]) )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[3,ii]-QR[1,ii])**2)
                                intR2A = ( 10.*QR[2,ii]**2 + 4.*QR[2,ii]*QR[3,ii] + QR[3,ii]**2 - 3.*QR[1,ii]*(4.*QR[2,ii]+QR[3,ii]) )*(QR[3,ii]-QR[2,ii])**2/(60.*(QR[3,ii]-QR[1,ii])**2)
                                intR2B = - ( 2.*QR[2,ii]**2 + 2.*QR[2,ii]*QR[3,ii] + QR[3,ii]**2 - 3.*QR[2,ii]*QR[3,Ind[jj]] -2.*QR[3,ii]*QR[3,Ind[jj]] )*(QR[3,ii]-QR[2,ii])**2/(60.*(QR[3,ii]-QR[1,ii])*(QR[3,Ind[jj]]-QR[2,ii]))
                                intR = intR1A + intR1B + intR2A + intR2B
                            else:
                                assert np.all(kR==QR[0:3,ii]), "Something wrong with common R knots..."
                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZ = d0Z[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            if kZ.size==2 and np.all(kZ==QZ[0:2,ii]):
                                intZ = (QZ[1,ii]-QZ[0,ii])**3/(30.*(QZ[2,ii]-QZ[0,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]]))
                            elif kZ.size==3 and np.all(kZ==QZ[0:3,ii]):
                                intZ = ( 6.*QZ[1,ii]**2 - QZ[0,ii]*(9.*QZ[1,ii]+QZ[2,ii]-10.*QZ[3,ii]) + QZ[2,ii]*(QZ[2,ii]-4.*QZ[3,ii]) +3.*QZ[1,ii]*(QZ[2,ii]-2.*QZ[3,ii]) )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[1,ii]-QZ[3,ii])*(QZ[2,ii]-QZ[0,ii])**2)  +  ( QZ[0,ii]**2 + 3.*QZ[0,ii]*QZ[1,ii] + 6.*QZ[1,ii]**2 - 2.*QZ[0,Ind[jj]]*(2.*QZ[0,ii]+3.*QZ[1,ii] - 5.*QZ[2,ii]) - QZ[0,ii]*QZ[2,ii] -9.*QZ[1,ii]*QZ[2,ii] )*(QZ[1,ii]-QZ[0,ii])**2/(30.*(QZ[0,Ind[jj]]-QZ[2,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2)
                            elif kZ.size==2 and np.all(kZ==QZ[-2:,ii]):
                                intZ = (QZ[3,ii]-QZ[2,ii])**3/(30.*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]]))
                            elif kZ.size==3 and np.all(kZ==QZ[-3:,ii]):
                                intZ = ( 6.*QZ[2,ii]**2 - QZ[1,ii]*(9.*QZ[2,ii]+QZ[3,ii]-10.*QZ[3,Ind[jj]]) + QZ[3,ii]*(QZ[3,ii]-4.*QZ[3,Ind[jj]]) +3.*QZ[2,ii]*(QZ[3,ii]-2.*QZ[3,Ind[jj]]) )*(QZ[3,ii]-QZ[2,ii])**2/(30.*(QZ[2,ii]-QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2)  +  ( QZ[1,ii]**2 + 3.*QZ[1,ii]*QZ[2,ii] + 6.*QZ[2,ii]**2 - 2.*QZ[0,ii]*(2.*QZ[1,ii]+3.*QZ[2,ii] - 5.*QZ[3,ii]) - QZ[1,ii]*QZ[3,ii] -9.*QZ[2,ii]*QZ[3,ii] )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[0,ii]-QZ[2,ii])*(QZ[3,ii]-QZ[1,ii])**2)
                            else:
                                assert np.all(kZ==QZ[0:3,ii]), "Something wrong with common Z knots..."
                        ll[0,Ind[jj]] = intR*intZ
                    ll[0,ii] = d0R[ii]*d0Z[ii]
                    LL.append(scpsp.coo_matrix(ll))
                A = scpsp.vstack(LL,format='csr')

    elif Deriv=='D1N2':
        m = 2
        if BF2.Deg==1:
            Supp = BF2._get_Func_SuppBounds()
            QR, QZ = BF2._get_quadPoints()
            LLR, LLZ = [], []

            if Mode=='Surf':
                d0R, d0Z = (Supp[1,:]-Supp[0,:])/3., (Supp[3,:]-Supp[2,:])/3.
                d0DR, d0DZ = (QR[2,:]-QR[0,:])/((QR[2,:]-QR[1,:])*(QR[1,:]-QR[0,:])), (QZ[2,:]-QZ[0,:])/((QZ[2,:]-QZ[1,:])*(QZ[1,:]-QZ[0,:]))
                for ii in range(0,BF2.NFunc):
                    llR,llZ = np.zeros((1,BF2.NFunc)), np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intRDR = d0DR[ii]
                            intRDZ = d0R[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            intRDR = -1./(kR[1]-kR[0])
                            intRDZ = (kR[1]-kR[0])/6.
                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZDR = d0Z[ii]
                            intZDZ = d0DZ[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            intZDR = (kZ[1]-kZ[0])/6.
                            intZDZ = -1./(kZ[1]-kZ[0])
                        llR[0,Ind[jj]] = intRDR*intZDR
                        llZ[0,Ind[jj]] = intRDZ*intZDZ
                    llR[0,ii] = d0DR[ii]*d0Z[ii]
                    llZ[0,ii] = d0R[ii]*d0DZ[ii]
                    LLR.append(scpsp.coo_matrix(llR))
                    LLZ.append(scpsp.coo_matrix(llZ))
            else:
                d0R, d0Z = (QR[2,:]**2-QR[0,:]**2 + 2.*QR[1,:]*(QR[2,:]-QR[0,:]))/12., (Supp[3,:]-Supp[2,:])/3.
                d0DR, d0DZ = 0.5*(QR[1,:]+QR[0,:])/(QR[1,:]-QR[0,:]) + 0.5*(QR[2,:]+QR[1,:])/(QR[2,:]-QR[1,:]), (QZ[2,:]-QZ[0,:])/((QZ[2,:]-QZ[1,:])*(QZ[1,:]-QZ[0,:]))
                for ii in range(0,BF2.NFunc):
                    llR,llZ = np.zeros((1,BF2.NFunc)), np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intRDR = d0DR[ii]
                            intRDZ = d0R[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            intRDR = -0.5*(kR[1]+kR[0])/(kR[1]-kR[0])
                            intRDZ = (kR[1]**2-kR[0]**2)/12.
                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZDR = d0Z[ii]
                            intZDZ = d0DZ[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            intZDR = (kZ[1]-kZ[0])/6.
                            intZDZ = -1./(kZ[1]-kZ[0])
                        llR[0,Ind[jj]] = intRDR*intZDR
                        llZ[0,Ind[jj]] = intRDZ*intZDZ
                    llR[0,ii] = d0DR[ii]*d0Z[ii]
                    llZ[0,ii] = d0R[ii]*d0DZ[ii]
                    LLR.append(scpsp.coo_matrix(llR))
                    LLZ.append(scpsp.coo_matrix(llZ))
            AR, AZ = scpsp.vstack(LLR,format='csr'), scpsp.vstack(LLZ,format='csr')
            A = (AR,AZ)

        elif BF2.Deg==2:
            Supp = BF2._get_Func_SuppBounds()
            QR, QZ = BF2._get_quadPoints()
            LLR, LLZ = [], []
            if Mode=='Surf':
                d0R21 = (QR[2,:]-QR[1,:])*(10.*QR[0,:]**2+6.*QR[1,:]**2+3.*QR[1,:]*QR[2,:]+QR[2,:]**2 - 5.*QR[0,:]*(3.*QR[1,:]+QR[2,:]))/(30.*(QR[2,:]-QR[0,:])**2)
                d0R22 = (QR[2,:]-QR[1,:])*(10.*QR[3,:]**2+6.*QR[2,:]**2+3.*QR[1,:]*QR[2,:]+QR[1,:]**2 - 5.*QR[3,:]*(3.*QR[2,:]+QR[1,:]))/(30.*(QR[3,:]-QR[1,:])**2)
                d0R23 = ( 3.*QR[1,:]**2 + 4.*QR[1,:]*QR[2,:] + 3.*QR[2,:]**2 - 5.*QR[0,:]*(QR[1,:]+QR[2,:]-2.*QR[3,:]) -5.*QR[3,:]*(QR[1,:]+QR[2,:]) )*(QR[1,:]-QR[2,:])/(60.*(QR[2,:]-QR[0,:])*(QR[3,:]-QR[1,:]))
                d0Z21 = (QZ[2,:]-QZ[1,:])*(10.*QZ[0,:]**2+6.*QZ[1,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[2,:]**2 - 5.*QZ[0,:]*(3.*QZ[1,:]+QZ[2,:]))/(30.*(QZ[2,:]-QZ[0,:])**2)
                d0Z22 = (QZ[2,:]-QZ[1,:])*(10.*QZ[3,:]**2+6.*QZ[2,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[1,:]**2 - 5.*QZ[3,:]*(3.*QZ[2,:]+QZ[1,:]))/(30.*(QZ[3,:]-QZ[1,:])**2)
                d0Z23 = ( 3.*QZ[1,:]**2 + 4.*QZ[1,:]*QZ[2,:] + 3.*QZ[2,:]**2 - 5.*QZ[0,:]*(QZ[1,:]+QZ[2,:]-2.*QZ[3,:]) -5.*QZ[3,:]*(QZ[1,:]+QZ[2,:]) )*(QZ[1,:]-QZ[2,:])/(60.*(QZ[2,:]-QZ[0,:])*(QZ[3,:]-QZ[1,:]))
                d0R = (QR[1,:]-QR[0,:])**3/(5.*(QR[2,:]-QR[0,:])**2) + d0R21 + d0R22 + 2.*d0R23 + (QR[3,:]-QR[2,:])**3/(5.*(QR[3,:]-QR[1,:])**2)
                d0Z = (QZ[1,:]-QZ[0,:])**3/(5.*(QZ[2,:]-QZ[0,:])**2) + d0Z21 + d0Z22 + 2.*d0Z23 + (QZ[3,:]-QZ[2,:])**3/(5.*(QZ[3,:]-QZ[1,:])**2)

                d0DR = 4.*(QR[1,:]-QR[0,:])/(3.*(QR[2,:]-QR[0,:])**2) + 4.*( QR[0,:]**2 + QR[1,:]**2 + QR[2,:]**2 + (QR[2,:]-2.*QR[3,:])*QR[1,:] - QR[2,:]*QR[3,:] + QR[3,:]**2 + (-QR[1,:]-2.*QR[2,:]+QR[3,:])*QR[0,:] )*(QR[2,:]-QR[1,:])/(3.*(QR[2,:]-QR[0,:])**2*(QR[3,:]-QR[1,:])**2) + 4.*(QR[3,:]-QR[2,:])/(3.*(QR[3,:]-QR[1,:])**2)
                d0DZ = 4.*(QZ[1,:]-QZ[0,:])/(3.*(QZ[2,:]-QZ[0,:])**2) + 4.*( QZ[0,:]**2 + QZ[1,:]**2 + QZ[2,:]**2 + (QZ[2,:]-2.*QZ[3,:])*QZ[1,:] - QZ[2,:]*QZ[3,:] + QZ[3,:]**2 + (-QZ[1,:]-2.*QZ[2,:]+QZ[3,:])*QZ[0,:] )*(QZ[2,:]-QZ[1,:])/(3.*(QZ[2,:]-QZ[0,:])**2*(QZ[3,:]-QZ[1,:])**2) + 4.*(QZ[3,:]-QZ[2,:])/(3.*(QZ[3,:]-QZ[1,:])**2)

                for ii in range(0,BF2.NFunc):
                    llR,llZ = np.zeros((1,BF2.NFunc)), np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intRDZ = d0R[ii]
                            intRDR = d0DR[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            if kR.size==2 and np.all(kR==QR[0:2,ii]):
                                intRDZ = (QR[1,ii]-QR[0,ii])**3/(30.*(QR[2,ii]-QR[0,ii])*(QR[3,Ind[jj]]-QR[1,Ind[jj]]))
                                intRDR = 2.*(QR[0,ii]-QR[1,ii])/(3.*(QR[1,ii]-QR[1,Ind[jj]])*(QR[2,ii]-QR[0,ii]))
                            elif kR.size==3 and np.all(kR==QR[0:3,ii]):
                                intRDZ = ( 6.*QR[1,ii]**2 - QR[0,ii]*(9.*QR[1,ii]+QR[2,ii]-10.*QR[3,ii]) + QR[2,ii]*(QR[2,ii]-4.*QR[3,ii]) +3.*QR[1,ii]*(QR[2,ii]-2.*QR[3,ii]) )*(QR[2,ii]-QR[1,ii])**2/(30.*(QR[1,ii]-QR[3,ii])*(QR[2,ii]-QR[0,ii])**2)  +  ( QR[0,ii]**2 + 3.*QR[0,ii]*QR[1,ii] + 6.*QR[1,ii]**2 - 2.*QR[0,Ind[jj]]*(2.*QR[0,ii]+3.*QR[1,ii] - 5.*QR[2,ii]) - QR[0,ii]*QR[2,ii] -9.*QR[1,ii]*QR[2,ii] )*(QR[1,ii]-QR[0,ii])**2/(30.*(QR[0,Ind[jj]]-QR[2,Ind[jj]])*(QR[2,ii]-QR[0,ii])**2)
                                intRDR = 2.*(2.*QR[0,Ind[jj]]-QR[0,ii]-2.*QR[1,ii]+QR[2,ii])*(QR[1,ii]-QR[0,ii])/(3.*(QR[1,ii]-QR[0,Ind[jj]])*(QR[2,ii]-QR[0,ii])**2) + 2.*(QR[0,ii]-2.*QR[1,ii]-QR[2,ii]+2.*QR[3,ii])*(QR[2,ii]-QR[1,ii])/(3.*(QR[1,ii]-QR[3,ii])*(QR[2,ii]-QR[0,ii])**2)
                            elif kR.size==2 and np.all(kR==QR[-2:,ii]):
                                intRDZ = (QR[3,ii]-QR[2,ii])**3/(30.*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[0,Ind[jj]]))
                                intRDR = 2.*(QR[2,ii]-QR[3,ii])/(3.*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[2,ii]))
                            elif kR.size==3 and np.all(kR==QR[-3:,ii]):
                                intRDZ = ( 6.*QR[2,ii]**2 - QR[1,ii]*(9.*QR[2,ii]+QR[3,ii]-10.*QR[3,Ind[jj]]) + QR[3,ii]*(QR[3,ii]-4.*QR[3,Ind[jj]]) +3.*QR[2,ii]*(QR[3,ii]-2.*QR[3,Ind[jj]]) )*(QR[3,ii]-QR[2,ii])**2/(30.*(QR[2,ii]-QR[3,Ind[jj]])*(QR[3,ii]-QR[1,ii])**2)  +  ( QR[1,ii]**2 + 3.*QR[1,ii]*QR[2,ii] + 6.*QR[2,ii]**2 - 2.*QR[0,ii]*(2.*QR[1,ii]+3.*QR[2,ii] - 5.*QR[3,ii]) - QR[1,ii]*QR[3,ii] -9.*QR[2,ii]*QR[3,ii] )*(QR[2,ii]-QR[1,ii])**2/(30.*(QR[0,ii]-QR[2,ii])*(QR[3,ii]-QR[1,ii])**2)
                                intRDR = 2.*(2.*QR[0,ii]-QR[1,ii]-2.*QR[2,ii]+QR[3,ii])*(QR[2,ii]-QR[1,ii])/(3.*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii])**2) + 2.*(QR[1,ii]-2.*QR[2,ii]-QR[3,ii]+2.*QR[3,Ind[jj]])*(QR[3,ii]-QR[2,ii])/(3.*(QR[2,ii]-QR[3,Ind[jj]])*(QR[3,ii]-QR[1,ii])**2)
                            else:
                                assert np.all(kR==QR[0:3,ii]), "Something wrong with common R knots..."
                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZDR = d0Z[ii]
                            intZDZ = d0DZ[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            if kZ.size==2 and np.all(kZ==QZ[0:2,ii]):
                                intZDR = (QZ[1,ii]-QZ[0,ii])**3/(30.*(QZ[2,ii]-QZ[0,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]]))
                                intZDZ = 2.*(QZ[0,ii]-QZ[1,ii])/(3.*(QZ[1,ii]-QZ[1,Ind[jj]])*(QZ[2,ii]-QZ[0,ii]))
                            elif kZ.size==3 and np.all(kZ==QZ[0:3,ii]):
                                intZDR = ( 6.*QZ[1,ii]**2 - QZ[0,ii]*(9.*QZ[1,ii]+QZ[2,ii]-10.*QZ[3,ii]) + QZ[2,ii]*(QZ[2,ii]-4.*QZ[3,ii]) +3.*QZ[1,ii]*(QZ[2,ii]-2.*QZ[3,ii]) )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[1,ii]-QZ[3,ii])*(QZ[2,ii]-QZ[0,ii])**2)  +  ( QZ[0,ii]**2 + 3.*QZ[0,ii]*QZ[1,ii] + 6.*QZ[1,ii]**2 - 2.*QZ[0,Ind[jj]]*(2.*QZ[0,ii]+3.*QZ[1,ii] - 5.*QZ[2,ii]) - QZ[0,ii]*QZ[2,ii] -9.*QZ[1,ii]*QZ[2,ii] )*(QZ[1,ii]-QZ[0,ii])**2/(30.*(QZ[0,Ind[jj]]-QZ[2,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2)
                                intZDZ = 2.*(2.*QZ[0,Ind[jj]]-QZ[0,ii]-2.*QZ[1,ii]+QZ[2,ii])*(QZ[1,ii]-QZ[0,ii])/(3.*(QZ[1,ii]-QZ[0,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2) + 2.*(QZ[0,ii]-2.*QZ[1,ii]-QZ[2,ii]+2.*QZ[3,ii])*(QZ[2,ii]-QZ[1,ii])/(3.*(QZ[1,ii]-QZ[3,ii])*(QZ[2,ii]-QZ[0,ii])**2)
                            elif kZ.size==2 and np.all(kZ==QZ[-2:,ii]):
                                intZDR = (QZ[3,ii]-QZ[2,ii])**3/(30.*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]]))
                                intZDZ = 2.*(QZ[2,ii]-QZ[3,ii])/(3.*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[2,ii]))
                            elif kZ.size==3 and np.all(kZ==QZ[-3:,ii]):
                                intZDR = ( 6.*QZ[2,ii]**2 - QZ[1,ii]*(9.*QZ[2,ii]+QZ[3,ii]-10.*QZ[3,Ind[jj]]) + QZ[3,ii]*(QZ[3,ii]-4.*QZ[3,Ind[jj]]) +3.*QZ[2,ii]*(QZ[3,ii]-2.*QZ[3,Ind[jj]]) )*(QZ[3,ii]-QZ[2,ii])**2/(30.*(QZ[2,ii]-QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2)  +  ( QZ[1,ii]**2 + 3.*QZ[1,ii]*QZ[2,ii] + 6.*QZ[2,ii]**2 - 2.*QZ[0,ii]*(2.*QZ[1,ii]+3.*QZ[2,ii] - 5.*QZ[3,ii]) - QZ[1,ii]*QZ[3,ii] -9.*QZ[2,ii]*QZ[3,ii] )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[0,ii]-QZ[2,ii])*(QZ[3,ii]-QZ[1,ii])**2)
                                intZDZ = 2.*(2.*QZ[0,ii]-QZ[1,ii]-2.*QZ[2,ii]+QZ[3,ii])*(QZ[2,ii]-QZ[1,ii])/(3.*(QZ[2,ii]-QZ[0,ii])*(QZ[3,ii]-QZ[1,ii])**2) + 2.*(QZ[1,ii]-2.*QZ[2,ii]-QZ[3,ii]+2.*QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[2,ii])/(3.*(QZ[2,ii]-QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2)
                            else:
                                assert np.all(kZ==QZ[0:3,ii]), "Something wrong with common Z knots..."

                        llR[0,Ind[jj]] = intRDR*intZDR
                        llZ[0,Ind[jj]] = intRDZ*intZDZ
                    llR[0,ii] = d0DR[ii]*d0Z[ii]
                    llZ[0,ii] = d0R[ii]*d0DZ[ii]
                    LLR.append(scpsp.coo_matrix(llR))
                    LLZ.append(scpsp.coo_matrix(llZ))

            elif Mode=='Vol':
                d0R21 = (10*QR[1,:]**3 + 6.*QR[1,:]**2*QR[2,:] + 3.*QR[1,:]*QR[2,:]**2 + QR[2,:]**3 + 5.*QR[0,:]**2*(3.*QR[1,:]+QR[2,:]) - 4.*QR[0,:]*(6.*QR[1,:]**2+3.*QR[1,:]*QR[2,:]+QR[2,:]**2))*(QR[2,:]-QR[1,:])/(60.*(QR[2,:]-QR[0,:])**2)
                d0R22 = (10*QR[2,:]**3 + 6.*QR[2,:]**2*QR[1,:] + 3.*QR[2,:]*QR[1,:]**2 + QR[1,:]**3 + 5.*QR[3,:]**2*(3.*QR[2,:]+QR[1,:]) - 4.*QR[3,:]*(6.*QR[2,:]**2+3.*QR[2,:]*QR[1,:]+QR[1,:]**2))*(QR[2,:]-QR[1,:])/(60.*(QR[3,:]-QR[1,:])**2)
                d0R23 = (2.*QR[1,:]**3 + QR[1,:]*QR[2,:]*(3.*QR[2,:]-4.*QR[3,:]) +(2.*QR[2,:]-3.*QR[3,:])*QR[2,:]**2 +3.*(QR[2,:]-QR[3,:])*QR[1,:]**2 + QR[0,:]*(-3.*QR[1,:]**2-4.*QR[1,:]*QR[2,:]-3.*QR[2,:]**2+5.*QR[1,:]*QR[3,:]+5.*QR[2,:]*QR[3,:]) )*(QR[1,:]-QR[2,:])/(60.*(QR[3,:]-QR[1,:])*(QR[2,:]-QR[0,:]))
                d0Z21 = (QZ[2,:]-QZ[1,:])*(10.*QZ[0,:]**2+6.*QZ[1,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[2,:]**2 - 5.*QZ[0,:]*(3.*QZ[1,:]+QZ[2,:]))/(30.*(QZ[2,:]-QZ[0,:])**2)
                d0Z22 = (QZ[2,:]-QZ[1,:])*(10.*QZ[3,:]**2+6.*QZ[2,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[1,:]**2 - 5.*QZ[3,:]*(3.*QZ[2,:]+QZ[1,:]))/(30.*(QZ[3,:]-QZ[1,:])**2)
                d0Z23 = ( 3.*QZ[1,:]**2 + 4.*QZ[1,:]*QZ[2,:] + 3.*QZ[2,:]**2 - 5.*QZ[0,:]*(QZ[1,:]+QZ[2,:]-2.*QZ[3,:]) -5.*QZ[3,:]*(QZ[1,:]+QZ[2,:]) )*(QZ[1,:]-QZ[2,:])/(60.*(QZ[2,:]-QZ[0,:])*(QZ[3,:]-QZ[1,:]))
                d0R = (5.*QR[1,:]+QR[0,:])*(QR[1,:]-QR[0,:])**3/(30.*(QR[2,:]-QR[0,:])**2) + d0R21 + d0R22 + 2.*d0R23 + (5.*QR[2,:]+QR[3,:])*(QR[3,:]-QR[2,:])**3/(30.*(QR[3,:]-QR[1,:])**2)
                d0Z = (QZ[1,:]-QZ[0,:])**3/(5.*(QZ[2,:]-QZ[0,:])**2) + d0Z21 + d0Z22 + 2.*d0Z23 + (QZ[3,:]-QZ[2,:])**3/(5.*(QZ[3,:]-QZ[1,:])**2)

                d0DR = (QR[1,:]-QR[0,:])*(QR[0,:]+3.*QR[1,:])/(3.*(QR[2,:]-QR[0,:])**2) + ( (3.*QR[1,:]+QR[2,:])/(QR[2,:]-QR[0,:])**2 + (QR[1,:]+3.*QR[2,:])/(QR[3,:]-QR[1,:])**2 - 2.*(QR[1,:]+QR[2,:])/((QR[2,:]-QR[0,:])*(QR[3,:]-QR[1,:])) )*(QR[2,:]-QR[1,:])/3. + (QR[3,:]-QR[2,:])*(QR[3,:]+3.*QR[2,:])/(3.*(QR[3,:]-QR[1,:])**2)
                d0DZ = 4.*(QZ[1,:]-QZ[0,:])/(3.*(QZ[2,:]-QZ[0,:])**2) + 4.*( QZ[0,:]**2 + QZ[1,:]**2 + QZ[2,:]**2 + (QZ[2,:]-2.*QZ[3,:])*QZ[1,:] - QZ[2,:]*QZ[3,:] + QZ[3,:]**2 + (-QZ[1,:]-2.*QZ[2,:]+QZ[3,:])*QZ[0,:] )*(QZ[2,:]-QZ[1,:])/(3.*(QZ[2,:]-QZ[0,:])**2*(QZ[3,:]-QZ[1,:])**2) + 4.*(QZ[3,:]-QZ[2,:])/(3.*(QZ[3,:]-QZ[1,:])**2)

                for ii in range(0,BF2.NFunc):
                    llR,llZ = np.zeros((1,BF2.NFunc)), np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intRDZ = d0R[ii]
                            intRDR = d0DR[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            if kR.size==2 and np.all(kR==QR[0:2,ii]):
                                intRDZ = (QR[1,ii]+QR[0,ii])*(QR[1,ii]-QR[0,ii])**3/(60.*(QR[1,ii]-QR[1,Ind[jj]])*(QR[2,ii]-QR[0,ii]))
                                intRDR = (QR[0,ii]+QR[1,ii])*(QR[0,ii]-QR[1,ii])/(3.*(QR[1,ii]-QR[1,Ind[jj]])*(QR[2,ii]-QR[0,ii]))
                            elif kR.size==3 and np.all(kR==QR[0:3,ii]):
                                intR1A = ( -2.*QR[0,Ind[jj]]*QR[0,ii] + QR[0,ii]**2 - 3.*QR[0,Ind[jj]]*QR[1,ii] + 2.*QR[0,ii]*QR[1,ii] + 2.*QR[1,ii]**2 )*(QR[1,ii]-QR[0,ii])**2/(60.*(QR[1,ii]-QR[0,Ind[jj]])*(QR[2,ii]-QR[0,ii]))
                                intR1B = - ( QR[0,ii]**2 + 2.*QR[1,ii]*(5.*QR[1,ii]-6.*QR[2,ii]) + QR[0,ii]*(4.*QR[1,ii]-3.*QR[2,ii]) )*(QR[1,ii]-QR[0,ii])**2/(60.*(QR[2,ii]-QR[0,ii])**2)
                                intR2A = ( 10.*QR[1,ii]**2 + 4.*QR[1,ii]*QR[2,ii] + QR[2,ii]**2 - 3.*QR[0,ii]*(4.*QR[1,ii]+QR[2,ii]) )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])**2)
                                intR2B = - ( 2.*QR[1,ii]**2 + 2.*QR[1,ii]*QR[2,ii] + QR[2,ii]**2 - 3.*QR[1,ii]*QR[3,ii] -2.*QR[2,ii]*QR[3,ii] )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii]))
                                intRDZ = intR1A + intR1B + intR2A + intR2B
                                intRDR = ( -QR[0,ii]**2 + (QR[0,ii]+3.*QR[1,ii])*QR[0,Ind[jj]] + (-3.*QR[1,ii]+QR[2,ii])*QR[1,ii] +(-2.*QR[1,ii]+QR[2,ii])*QR[0,ii] )*(QR[1,ii]-QR[0,ii])/(3.*(QR[1,ii]-QR[0,Ind[jj]])*(QR[2,ii]-QR[0,ii])**2) + ( -3.*QR[1,ii]**2 + (QR[1,ii]+QR[2,ii])*QR[0,ii] + (-QR[2,ii]+QR[3,ii])*QR[2,ii] + (-2.*QR[2,ii]+3.*QR[3,ii])*QR[1,ii] )*(QR[2,ii]-QR[1,ii])/(3.*(QR[1,ii]-QR[3,ii])*(QR[2,ii]-QR[0,ii])**2)
                            elif kR.size==2 and np.all(kR==QR[-2:,ii]):
                                intRDZ = (QR[3,ii]+QR[2,ii])*(QR[3,ii]-QR[2,ii])**3/(60.*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[2,ii]))
                                intRDR = (QR[2,ii]+QR[3,ii])*(QR[2,ii]-QR[3,ii])/(3.*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[2,ii]))
                            elif kR.size==3 and np.all(kR==QR[-3:,ii]):
                                intR1A = ( -2.*QR[0,ii]*QR[1,ii] + QR[1,ii]**2 - 3.*QR[0,ii]*QR[2,ii] + 2.*QR[1,ii]*QR[2,ii] + 2.*QR[2,ii]**2 )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii]))
                                intR1B = - ( QR[1,ii]**2 + 2.*QR[2,ii]*(5.*QR[2,ii]-6.*QR[3,ii]) + QR[1,ii]*(4.*QR[2,ii]-3.*QR[3,ii]) )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[3,ii]-QR[1,ii])**2)
                                intR2A = ( 10.*QR[2,ii]**2 + 4.*QR[2,ii]*QR[3,ii] + QR[3,ii]**2 - 3.*QR[1,ii]*(4.*QR[2,ii]+QR[3,ii]) )*(QR[3,ii]-QR[2,ii])**2/(60.*(QR[3,ii]-QR[1,ii])**2)
                                intR2B = - ( 2.*QR[2,ii]**2 + 2.*QR[2,ii]*QR[3,ii] + QR[3,ii]**2 - 3.*QR[2,ii]*QR[3,Ind[jj]] -2.*QR[3,ii]*QR[3,Ind[jj]] )*(QR[3,ii]-QR[2,ii])**2/(60.*(QR[3,ii]-QR[1,ii])*(QR[3,Ind[jj]]-QR[2,ii]))
                                intRDZ = intR1A + intR1B + intR2A + intR2B
                                intRDR = ( -QR[1,ii]**2 + (QR[1,ii]+3.*QR[2,ii])*QR[0,ii] + (-3.*QR[2,ii]+QR[3,ii])*QR[2,ii] +(-2.*QR[2,ii]+QR[3,ii])*QR[1,ii] )*(QR[2,ii]-QR[1,ii])/(3.*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii])**2) + ( -3.*QR[2,ii]**2 + (QR[2,ii]+QR[3,ii])*QR[1,ii] + (-QR[3,ii]+QR[3,Ind[jj]])*QR[3,ii] + (-2.*QR[3,ii]+3.*QR[3,Ind[jj]])*QR[2,ii] )*(QR[3,ii]-QR[2,ii])/(3.*(QR[2,ii]-QR[3,Ind[jj]])*(QR[3,ii]-QR[1,ii])**2)
                            else:
                                assert np.all(kR==QR[0:3,ii]), "Something wrong with common R knots..."
                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZDR = d0Z[ii]
                            intZDZ = d0DZ[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            if kZ.size==2 and np.all(kZ==QZ[0:2,ii]):
                                intZDR = (QZ[1,ii]-QZ[0,ii])**3/(30.*(QZ[2,ii]-QZ[0,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]]))
                                intZDZ = 2.*(QZ[0,ii]-QZ[1,ii])/(3.*(QZ[1,ii]-QZ[1,Ind[jj]])*(QZ[2,ii]-QZ[0,ii]))
                            elif kZ.size==3 and np.all(kZ==QZ[0:3,ii]):
                                intZDR = ( 6.*QZ[1,ii]**2 - QZ[0,ii]*(9.*QZ[1,ii]+QZ[2,ii]-10.*QZ[3,ii]) + QZ[2,ii]*(QZ[2,ii]-4.*QZ[3,ii]) +3.*QZ[1,ii]*(QZ[2,ii]-2.*QZ[3,ii]) )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[1,ii]-QZ[3,ii])*(QZ[2,ii]-QZ[0,ii])**2)  +  ( QZ[0,ii]**2 + 3.*QZ[0,ii]*QZ[1,ii] + 6.*QZ[1,ii]**2 - 2.*QZ[0,Ind[jj]]*(2.*QZ[0,ii]+3.*QZ[1,ii] - 5.*QZ[2,ii]) - QZ[0,ii]*QZ[2,ii] -9.*QZ[1,ii]*QZ[2,ii] )*(QZ[1,ii]-QZ[0,ii])**2/(30.*(QZ[0,Ind[jj]]-QZ[2,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2)
                                intZDZ = 2.*(2.*QZ[0,Ind[jj]]-QZ[0,ii]-2.*QZ[1,ii]+QZ[2,ii])*(QZ[1,ii]-QZ[0,ii])/(3.*(QZ[1,ii]-QZ[0,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2) + 2.*(QZ[0,ii]-2.*QZ[1,ii]-QZ[2,ii]+2.*QZ[3,ii])*(QZ[2,ii]-QZ[1,ii])/(3.*(QZ[1,ii]-QZ[3,ii])*(QZ[2,ii]-QZ[0,ii])**2)
                            elif kZ.size==2 and np.all(kZ==QZ[-2:,ii]):
                                intZDR = (QZ[3,ii]-QZ[2,ii])**3/(30.*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]]))
                                intZDZ = 2.*(QZ[2,ii]-QZ[3,ii])/(3.*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[2,ii]))
                            elif kZ.size==3 and np.all(kZ==QZ[-3:,ii]):
                                intZDR = ( 6.*QZ[2,ii]**2 - QZ[1,ii]*(9.*QZ[2,ii]+QZ[3,ii]-10.*QZ[3,Ind[jj]]) + QZ[3,ii]*(QZ[3,ii]-4.*QZ[3,Ind[jj]]) +3.*QZ[2,ii]*(QZ[3,ii]-2.*QZ[3,Ind[jj]]) )*(QZ[3,ii]-QZ[2,ii])**2/(30.*(QZ[2,ii]-QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2)  +  ( QZ[1,ii]**2 + 3.*QZ[1,ii]*QZ[2,ii] + 6.*QZ[2,ii]**2 - 2.*QZ[0,ii]*(2.*QZ[1,ii]+3.*QZ[2,ii] - 5.*QZ[3,ii]) - QZ[1,ii]*QZ[3,ii] -9.*QZ[2,ii]*QZ[3,ii] )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[0,ii]-QZ[2,ii])*(QZ[3,ii]-QZ[1,ii])**2)
                                intZDZ = 2.*(2.*QZ[0,ii]-QZ[1,ii]-2.*QZ[2,ii]+QZ[3,ii])*(QZ[2,ii]-QZ[1,ii])/(3.*(QZ[2,ii]-QZ[0,ii])*(QZ[3,ii]-QZ[1,ii])**2) + 2.*(QZ[1,ii]-2.*QZ[2,ii]-QZ[3,ii]+2.*QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[2,ii])/(3.*(QZ[2,ii]-QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2)
                            else:
                                assert np.all(kZ==QZ[0:3,ii]), "Something wrong with common Z knots..."

                        llR[0,Ind[jj]] = intRDR*intZDR
                        llZ[0,Ind[jj]] = intRDZ*intZDZ
                    llR[0,ii] = d0DR[ii]*d0Z[ii]
                    llZ[0,ii] = d0R[ii]*d0DZ[ii]
                    LLR.append(scpsp.coo_matrix(llR))
                    LLZ.append(scpsp.coo_matrix(llZ))

            AR, AZ = scpsp.vstack(LLR,format='csr'), scpsp.vstack(LLZ,format='csr')
            A = (AR,AZ)


    elif Deriv=='D1FI':
        m = 3
        if BF2.Deg==1:
            Supp = BF2._get_Func_SuppBounds()
            QR, QZ = BF2._get_quadPoints()
            LLR, LLZ = [], []




    elif 'D2N2' in Deriv:
        m = 2
        if BF2.Deg==2:
            Supp = BF2._get_Func_SuppBounds()
            QR, QZ = BF2._get_quadPoints()
            LLR, LLZ = [], []

            if Mode=='Surf':
                d0R21 = (QR[2,:]-QR[1,:])*(10.*QR[0,:]**2+6.*QR[1,:]**2+3.*QR[1,:]*QR[2,:]+QR[2,:]**2 - 5.*QR[0,:]*(3.*QR[1,:]+QR[2,:]))/(30.*(QR[2,:]-QR[0,:])**2)
                d0R22 = (QR[2,:]-QR[1,:])*(10.*QR[3,:]**2+6.*QR[2,:]**2+3.*QR[1,:]*QR[2,:]+QR[1,:]**2 - 5.*QR[3,:]*(3.*QR[2,:]+QR[1,:]))/(30.*(QR[3,:]-QR[1,:])**2)
                d0R23 = ( 3.*QR[1,:]**2 + 4.*QR[1,:]*QR[2,:] + 3.*QR[2,:]**2 - 5.*QR[0,:]*(QR[1,:]+QR[2,:]-2.*QR[3,:]) -5.*QR[3,:]*(QR[1,:]+QR[2,:]) )*(QR[1,:]-QR[2,:])/(60.*(QR[2,:]-QR[0,:])*(QR[3,:]-QR[1,:]))
                d0Z21 = (QZ[2,:]-QZ[1,:])*(10.*QZ[0,:]**2+6.*QZ[1,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[2,:]**2 - 5.*QZ[0,:]*(3.*QZ[1,:]+QZ[2,:]))/(30.*(QZ[2,:]-QZ[0,:])**2)
                d0Z22 = (QZ[2,:]-QZ[1,:])*(10.*QZ[3,:]**2+6.*QZ[2,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[1,:]**2 - 5.*QZ[3,:]*(3.*QZ[2,:]+QZ[1,:]))/(30.*(QZ[3,:]-QZ[1,:])**2)
                d0Z23 = ( 3.*QZ[1,:]**2 + 4.*QZ[1,:]*QZ[2,:] + 3.*QZ[2,:]**2 - 5.*QZ[0,:]*(QZ[1,:]+QZ[2,:]-2.*QZ[3,:]) -5.*QZ[3,:]*(QZ[1,:]+QZ[2,:]) )*(QZ[1,:]-QZ[2,:])/(60.*(QZ[2,:]-QZ[0,:])*(QZ[3,:]-QZ[1,:]))
                d0R = (QR[1,:]-QR[0,:])**3/(5.*(QR[2,:]-QR[0,:])**2) + d0R21 + d0R22 + 2.*d0R23 + (QR[3,:]-QR[2,:])**3/(5.*(QR[3,:]-QR[1,:])**2)
                d0Z = (QZ[1,:]-QZ[0,:])**3/(5.*(QZ[2,:]-QZ[0,:])**2) + d0Z21 + d0Z22 + 2.*d0Z23 + (QZ[3,:]-QZ[2,:])**3/(5.*(QZ[3,:]-QZ[1,:])**2)

                d0DDR = 4./((QR[1,:]-QR[0,:])*(QR[2,:]-QR[0,:])**2) + 4.*(QR[3,:]+QR[2,:]-QR[1,:]-QR[0,:])**2/((QR[2,:]-QR[1,:])*(QR[3,:]-QR[1,:])**2*(QR[2,:]-QR[0,:])**2) + 4./((QR[3,:]-QR[2,:])*(QR[3,:]-QR[1,:])**2)
                d0DDZ = 4./((QZ[1,:]-QZ[0,:])*(QZ[2,:]-QZ[0,:])**2) + 4.*(QZ[3,:]+QZ[2,:]-QZ[1,:]-QZ[0,:])**2/((QZ[2,:]-QZ[1,:])*(QZ[3,:]-QZ[1,:])**2*(QZ[2,:]-QZ[0,:])**2) + 4./((QZ[3,:]-QZ[2,:])*(QZ[3,:]-QZ[1,:])**2)
                for ii in range(0,BF2.NFunc):
                    llR,llZ = np.zeros((1,BF2.NFunc)), np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intRDDR = d0DDR[ii]
                            intRDDZ = d0R[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            if kR.size==2 and np.all(kR==QR[0:2,ii]):
                                intRDDR = 4./((kR[1]-kR[0])*(QR[2,ii]-QR[0,ii])*(QR[3,Ind[jj]]-QR[1,Ind[jj]]))
                                intRDDZ = (QR[1,ii]-QR[0,ii])**3/(30.*(QR[2,ii]-QR[0,ii])*(QR[3,Ind[jj]]-QR[1,Ind[jj]]))
                            elif kR.size==3 and np.all(kR==QR[0:3,ii]):
                                intRDDR = -4.*(QR[2,ii]+QR[1,ii]-QR[1,Ind[jj]]-QR[0,Ind[jj]])/((QR[2,Ind[jj]]-QR[1,Ind[jj]])*(QR[2,Ind[jj]]-QR[0,Ind[jj]])*(QR[2,ii]-QR[0,ii])**2) - 4.*(QR[3,ii]+QR[2,ii]-QR[1,ii]-QR[0,ii])/((QR[2,ii]-QR[1,ii])*(QR[3,ii]-QR[1,ii])*(QR[2,ii]-QR[0,ii])**2)
                                intRDDZ = ( 6.*QR[1,ii]**2 - QR[0,ii]*(9.*QR[1,ii]+QR[2,ii]-10.*QR[3,ii]) + QR[2,ii]*(QR[2,ii]-4.*QR[3,ii]) +3.*QR[1,ii]*(QR[2,ii]-2.*QR[3,ii]) )*(QR[2,ii]-QR[1,ii])**2/(30.*(QR[1,ii]-QR[3,ii])*(QR[2,ii]-QR[0,ii])**2)  +  ( QR[0,ii]**2 + 3.*QR[0,ii]*QR[1,ii] + 6.*QR[1,ii]**2 - 2.*QR[0,Ind[jj]]*(2.*QR[0,ii]+3.*QR[1,ii] - 5.*QR[2,ii]) - QR[0,ii]*QR[2,ii] -9.*QR[1,ii]*QR[2,ii] )*(QR[1,ii]-QR[0,ii])**2/(30.*(QR[0,Ind[jj]]-QR[2,Ind[jj]])*(QR[2,ii]-QR[0,ii])**2)
                            elif kR.size==2 and np.all(kR==QR[-2:,ii]):
                                intRDDR = 4./((kR[1]-kR[0])*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[0,Ind[jj]]))
                                intRDDZ = (QR[3,ii]-QR[2,ii])**3/(30.*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[0,Ind[jj]]))
                            elif kR.size==3 and np.all(kR==QR[-3:,ii]):
                                intRDDR = -4.*(QR[3,Ind[jj]]+QR[2,Ind[jj]]-QR[2,ii]-QR[1,ii])/((QR[3,ii]-QR[2,ii])*(QR[3,Ind[jj]]-QR[1,Ind[jj]])*(QR[3,ii]-QR[1,ii])**2) - 4.*(QR[3,ii]+QR[2,ii]-QR[1,ii]-QR[0,ii])/((QR[2,ii]-QR[1,ii])*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii])**2)
                                intRDDZ = ( 6.*QR[2,ii]**2 - QR[1,ii]*(9.*QR[2,ii]+QR[3,ii]-10.*QR[3,Ind[jj]]) + QR[3,ii]*(QR[3,ii]-4.*QR[3,Ind[jj]]) +3.*QR[2,ii]*(QR[3,ii]-2.*QR[3,Ind[jj]]) )*(QR[3,ii]-QR[2,ii])**2/(30.*(QR[2,ii]-QR[3,Ind[jj]])*(QR[3,ii]-QR[1,ii])**2)  +  ( QR[1,ii]**2 + 3.*QR[1,ii]*QR[2,ii] + 6.*QR[2,ii]**2 - 2.*QR[0,ii]*(2.*QR[1,ii]+3.*QR[2,ii] - 5.*QR[3,ii]) - QR[1,ii]*QR[3,ii] -9.*QR[2,ii]*QR[3,ii] )*(QR[2,ii]-QR[1,ii])**2/(30.*(QR[0,ii]-QR[2,ii])*(QR[3,ii]-QR[1,ii])**2)
                            else:
                                assert np.all(kR==QR[0:3,ii]), "Something wrong with common knots..."

                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZDDR = d0Z[ii]
                            intZDDZ = d0DDZ[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            if kZ.size==2 and np.all(kZ==QZ[0:2,ii]):
                                intZDDZ = 4./((kZ[1]-kZ[0])*(QZ[2,ii]-QZ[0,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]]))
                                intZDDR = (QZ[1,ii]-QZ[0,ii])**3/(30.*(QZ[2,ii]-QZ[0,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]]))
                            elif kZ.size==3 and np.all(kZ==QZ[0:3,ii]):
                                intZDDZ = -4.*(QZ[2,ii]+QZ[1,ii]-QZ[1,Ind[jj]]-QZ[0,Ind[jj]])/((QZ[2,Ind[jj]]-QZ[1,Ind[jj]])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2) - 4.*(QZ[3,ii]+QZ[2,ii]-QZ[1,ii]-QZ[0,ii])/((QZ[2,ii]-QZ[1,ii])*(QZ[3,ii]-QZ[1,ii])*(QZ[2,ii]-QZ[0,ii])**2)
                                intZDDR = ( 6.*QZ[1,ii]**2 - QZ[0,ii]*(9.*QZ[1,ii]+QZ[2,ii]-10.*QZ[3,ii]) + QZ[2,ii]*(QZ[2,ii]-4.*QZ[3,ii]) +3.*QZ[1,ii]*(QZ[2,ii]-2.*QZ[3,ii]) )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[1,ii]-QZ[3,ii])*(QZ[2,ii]-QZ[0,ii])**2)  +  ( QZ[0,ii]**2 + 3.*QZ[0,ii]*QZ[1,ii] + 6.*QZ[1,ii]**2 - 2.*QZ[0,Ind[jj]]*(2.*QZ[0,ii]+3.*QZ[1,ii] - 5.*QZ[2,ii]) - QZ[0,ii]*QZ[2,ii] -9.*QZ[1,ii]*QZ[2,ii] )*(QZ[1,ii]-QZ[0,ii])**2/(30.*(QZ[0,Ind[jj]]-QZ[2,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2)
                            elif kZ.size==2 and np.all(kZ==QZ[-2:,ii]):
                                intZDDZ = 4./((kZ[1]-kZ[0])*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]]))
                                intZDDR = (QZ[3,ii]-QZ[2,ii])**3/(30.*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]]))
                            elif kZ.size==3 and np.all(kZ==QZ[-3:,ii]):
                                intZDDZ = -4.*(QZ[3,Ind[jj]]+QZ[2,Ind[jj]]-QZ[2,ii]-QZ[1,ii])/((QZ[3,ii]-QZ[2,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2) - 4.*(QZ[3,ii]+QZ[2,ii]-QZ[1,ii]-QZ[0,ii])/((QZ[2,ii]-QZ[1,ii])*(QZ[2,ii]-QZ[0,ii])*(QZ[3,ii]-QZ[1,ii])**2)
                                intZDDR = ( 6.*QZ[2,ii]**2 - QZ[1,ii]*(9.*QZ[2,ii]+QZ[3,ii]-10.*QZ[3,Ind[jj]]) + QZ[3,ii]*(QZ[3,ii]-4.*QZ[3,Ind[jj]]) +3.*QZ[2,ii]*(QZ[3,ii]-2.*QZ[3,Ind[jj]]) )*(QZ[3,ii]-QZ[2,ii])**2/(30.*(QZ[2,ii]-QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2)  +  ( QZ[1,ii]**2 + 3.*QZ[1,ii]*QZ[2,ii] + 6.*QZ[2,ii]**2 - 2.*QZ[0,ii]*(2.*QZ[1,ii]+3.*QZ[2,ii] - 5.*QZ[3,ii]) - QZ[1,ii]*QZ[3,ii] -9.*QZ[2,ii]*QZ[3,ii] )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[0,ii]-QZ[2,ii])*(QZ[3,ii]-QZ[1,ii])**2)
                            else:
                                assert np.all(kZ==QZ[0:3,ii]), "Something wrong with common knots..."

                        llR[0,Ind[jj]] = intRDDR*intZDDR
                        llZ[0,Ind[jj]] = intRDDZ*intZDDZ
                    llR[0,ii] = d0DDR[ii]*d0Z[ii]
                    llZ[0,ii] = d0R[ii]*d0DDZ[ii]
                    LLR.append(scpsp.coo_matrix(llR))
                    LLZ.append(scpsp.coo_matrix(llZ))

            elif Mode=='Vol':
                d0R21 = (10*QR[1,:]**3 + 6.*QR[1,:]**2*QR[2,:] + 3.*QR[1,:]*QR[2,:]**2 + QR[2,:]**3 + 5.*QR[0,:]**2*(3.*QR[1,:]+QR[2,:]) - 4.*QR[0,:]*(6.*QR[1,:]**2+3.*QR[1,:]*QR[2,:]+QR[2,:]**2))*(QR[2,:]-QR[1,:])/(60.*(QR[2,:]-QR[0,:])**2)
                d0R22 = (10*QR[2,:]**3 + 6.*QR[2,:]**2*QR[1,:] + 3.*QR[2,:]*QR[1,:]**2 + QR[1,:]**3 + 5.*QR[3,:]**2*(3.*QR[2,:]+QR[1,:]) - 4.*QR[3,:]*(6.*QR[2,:]**2+3.*QR[2,:]*QR[1,:]+QR[1,:]**2))*(QR[2,:]-QR[1,:])/(60.*(QR[3,:]-QR[1,:])**2)
                d0R23 = (2.*QR[1,:]**3 + QR[1,:]*QR[2,:]*(3.*QR[2,:]-4.*QR[3,:]) +(2.*QR[2,:]-3.*QR[3,:])*QR[2,:]**2 +3.*(QR[2,:]-QR[3,:])*QR[1,:]**2 + QR[0,:]*(-3.*QR[1,:]**2-4.*QR[1,:]*QR[2,:]-3.*QR[2,:]**2+5.*QR[1,:]*QR[3,:]+5.*QR[2,:]*QR[3,:]) )*(QR[1,:]-QR[2,:])/(60.*(QR[3,:]-QR[1,:])*(QR[2,:]-QR[0,:]))
                d0Z21 = (QZ[2,:]-QZ[1,:])*(10.*QZ[0,:]**2+6.*QZ[1,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[2,:]**2 - 5.*QZ[0,:]*(3.*QZ[1,:]+QZ[2,:]))/(30.*(QZ[2,:]-QZ[0,:])**2)
                d0Z22 = (QZ[2,:]-QZ[1,:])*(10.*QZ[3,:]**2+6.*QZ[2,:]**2+3.*QZ[1,:]*QZ[2,:]+QZ[1,:]**2 - 5.*QZ[3,:]*(3.*QZ[2,:]+QZ[1,:]))/(30.*(QZ[3,:]-QZ[1,:])**2)
                d0Z23 = ( 3.*QZ[1,:]**2 + 4.*QZ[1,:]*QZ[2,:] + 3.*QZ[2,:]**2 - 5.*QZ[0,:]*(QZ[1,:]+QZ[2,:]-2.*QZ[3,:]) -5.*QZ[3,:]*(QZ[1,:]+QZ[2,:]) )*(QZ[1,:]-QZ[2,:])/(60.*(QZ[2,:]-QZ[0,:])*(QZ[3,:]-QZ[1,:]))
                d0R = (5.*QR[1,:]+QR[0,:])*(QR[1,:]-QR[0,:])**3/(30.*(QR[2,:]-QR[0,:])**2) + d0R21 + d0R22 + 2.*d0R23 + (5.*QR[2,:]+QR[3,:])*(QR[3,:]-QR[2,:])**3/(30.*(QR[3,:]-QR[1,:])**2)
                d0Z = (QZ[1,:]-QZ[0,:])**3/(5.*(QZ[2,:]-QZ[0,:])**2) + d0Z21 + d0Z22 + 2.*d0Z23 + (QZ[3,:]-QZ[2,:])**3/(5.*(QZ[3,:]-QZ[1,:])**2)

                d0DDR = 2.*(QR[1,:]+QR[0,:])/((QR[1,:]-QR[0,:])*(QR[2,:]-QR[0,:])**2) + 2.*(QR[2,:]+QR[1,:])*(QR[3,:]+QR[2,:]-QR[1,:]-QR[0,:])**2/((QR[2,:]-QR[1,:])*(QR[3,:]-QR[1,:])**2*(QR[2,:]-QR[0,:])**2) + 2.*(QR[3,:]+QR[2,:])/((QR[3,:]-QR[2,:])*(QR[3,:]-QR[1,:])**2)
                d0DDZ = 4./((QZ[1,:]-QZ[0,:])*(QZ[2,:]-QZ[0,:])**2) + 4.*(QZ[3,:]+QZ[2,:]-QZ[1,:]-QZ[0,:])**2/((QZ[2,:]-QZ[1,:])*(QZ[3,:]-QZ[1,:])**2*(QZ[2,:]-QZ[0,:])**2) + 4./((QZ[3,:]-QZ[2,:])*(QZ[3,:]-QZ[1,:])**2)
                for ii in range(0,BF2.NFunc):
                    llR,llZ = np.zeros((1,BF2.NFunc)), np.zeros((1,BF2.NFunc))
                    Ind = BF2._Func_InterFunc[:,ii]
                    Ind = np.unique(Ind[~np.isnan(Ind)])
                    Ind = np.asarray([int(xxx) for xxx in Ind], dtype=int)
                    for jj in range(0,Ind.size):
                        if np.all(QR[:,Ind[jj]]==QR[:,ii]):
                            intRDDR = d0DDR[ii]
                            intRDDZ = d0R[ii]
                        else:
                            kR = np.intersect1d(QR[:,Ind[jj]], QR[:,ii], assume_unique=True)
                            if kR.size==2 and np.all(kR==QR[0:2,ii]):
                                intRDDR = 2.*(kR[1]+kR[0])/((kR[1]-kR[0])*(QR[2,ii]-QR[0,ii])*(QR[3,Ind[jj]]-QR[1,Ind[jj]]))
                                intRDDZ = (QR[1,ii]+QR[0,ii])*(QR[1,ii]-QR[0,ii])**3/(60.*(QR[1,ii]-QR[1,Ind[jj]])*(QR[2,ii]-QR[0,ii]))
                            elif kR.size==3 and np.all(kR==QR[0:3,ii]):
                                intR1A = ( -2.*QR[0,Ind[jj]]*QR[0,ii] + QR[0,ii]**2 - 3.*QR[0,Ind[jj]]*QR[1,ii] + 2.*QR[0,ii]*QR[1,ii] + 2.*QR[1,ii]**2 )*(QR[1,ii]-QR[0,ii])**2/(60.*(QR[1,ii]-QR[0,Ind[jj]])*(QR[2,ii]-QR[0,ii]))
                                intR1B = - ( QR[0,ii]**2 + 2.*QR[1,ii]*(5.*QR[1,ii]-6.*QR[2,ii]) + QR[0,ii]*(4.*QR[1,ii]-3.*QR[2,ii]) )*(QR[1,ii]-QR[0,ii])**2/(60.*(QR[2,ii]-QR[0,ii])**2)
                                intR2A = ( 10.*QR[1,ii]**2 + 4.*QR[1,ii]*QR[2,ii] + QR[2,ii]**2 - 3.*QR[0,ii]*(4.*QR[1,ii]+QR[2,ii]) )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])**2)
                                intR2B = - ( 2.*QR[1,ii]**2 + 2.*QR[1,ii]*QR[2,ii] + QR[2,ii]**2 - 3.*QR[1,ii]*QR[3,ii] -2.*QR[2,ii]*QR[3,ii] )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii]))
                                intRDDZ = intR1A + intR1B + intR2A + intR2B
                                intRDDR = -2.*(QR[1,ii]+QR[0,ii])*(QR[2,ii]+QR[1,ii]-QR[0,ii]-QR[0,Ind[jj]])/((QR[1,ii]-QR[0,ii])*(QR[1,ii]-QR[0,Ind[jj]])*(QR[2,ii]-QR[0,ii])**2) - 2.*(QR[2,ii]+QR[1,ii])*(QR[3,ii]+QR[2,ii]-QR[1,ii]-QR[0,ii])/((QR[2,ii]-QR[1,ii])*(QR[3,ii]-QR[1,ii])*(QR[2,ii]-QR[0,ii])**2)
                            elif kR.size==2 and np.all(kR==QR[-2:,ii]):
                                intRDDR = 2.*(kR[1]+kR[0])/((kR[1]-kR[0])*(QR[2,Ind[jj]]-QR[0,Ind[jj]])*(QR[3,ii]-QR[1,ii]))
                                intRDDZ = (QR[3,ii]+QR[2,ii])*(QR[3,ii]-QR[2,ii])**3/(60.*(QR[3,ii]-QR[1,ii])*(QR[2,Ind[jj]]-QR[2,ii]))
                            elif kR.size==3 and np.all(kR==QR[-3:,ii]):
                                intR1A = ( -2.*QR[0,ii]*QR[1,ii] + QR[1,ii]**2 - 3.*QR[0,ii]*QR[2,ii] + 2.*QR[1,ii]*QR[2,ii] + 2.*QR[2,ii]**2 )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii]))
                                intR1B = - ( QR[1,ii]**2 + 2.*QR[2,ii]*(5.*QR[2,ii]-6.*QR[3,ii]) + QR[1,ii]*(4.*QR[2,ii]-3.*QR[3,ii]) )*(QR[2,ii]-QR[1,ii])**2/(60.*(QR[3,ii]-QR[1,ii])**2)
                                intR2A = ( 10.*QR[2,ii]**2 + 4.*QR[2,ii]*QR[3,ii] + QR[3,ii]**2 - 3.*QR[1,ii]*(4.*QR[2,ii]+QR[3,ii]) )*(QR[3,ii]-QR[2,ii])**2/(60.*(QR[3,ii]-QR[1,ii])**2)
                                intR2B = - ( 2.*QR[2,ii]**2 + 2.*QR[2,ii]*QR[3,ii] + QR[3,ii]**2 - 3.*QR[2,ii]*QR[3,Ind[jj]] -2.*QR[3,ii]*QR[3,Ind[jj]] )*(QR[3,ii]-QR[2,ii])**2/(60.*(QR[3,ii]-QR[1,ii])*(QR[3,Ind[jj]]-QR[2,ii]))
                                intRDDZ = intR1A + intR1B + intR2A + intR2B
                                intRDDR = -2.*(QR[2,ii]+QR[1,ii])*(QR[3,ii]+QR[2,ii]-QR[1,ii]-QR[0,ii])/((QR[2,ii]-QR[1,ii])*(QR[2,ii]-QR[0,ii])*(QR[3,ii]-QR[1,ii])**2) - 2.*(QR[3,ii]+QR[2,ii])*(QR[3,Ind[jj]]+QR[3,ii]-QR[2,ii]-QR[1,ii])/((QR[3,ii]-QR[2,ii])*(QR[3,Ind[jj]]-QR[2,ii])*(QR[3,ii]-QR[1,ii])**2)
                            else:
                                assert np.all(kR==QR[0:3,ii]), "Something wrong with common knots..."

                        if np.all(QZ[:,Ind[jj]]==QZ[:,ii]):
                            intZDDR = d0Z[ii]
                            intZDDZ = d0DDZ[ii]
                        else:
                            kZ = np.intersect1d(QZ[:,Ind[jj]], QZ[:,ii], assume_unique=True)
                            if kZ.size==2 and np.all(kZ==QZ[0:2,ii]):
                                intZDDZ = 4./((kZ[1]-kZ[0])*(QZ[2,ii]-QZ[0,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]]))
                                intZDDR = (QZ[1,ii]-QZ[0,ii])**3/(30.*(QZ[2,ii]-QZ[0,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]]))
                            elif kZ.size==3 and np.all(kZ==QZ[0:3,ii]):
                                intZDDZ = -4.*(QZ[2,ii]+QZ[1,ii]-QZ[1,Ind[jj]]-QZ[0,Ind[jj]])/((QZ[2,Ind[jj]]-QZ[1,Ind[jj]])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2) - 4.*(QZ[3,ii]+QZ[2,ii]-QZ[1,ii]-QZ[0,ii])/((QZ[2,ii]-QZ[1,ii])*(QZ[3,ii]-QZ[1,ii])*(QZ[2,ii]-QZ[0,ii])**2)
                                intZDDR = ( 6.*QZ[1,ii]**2 - QZ[0,ii]*(9.*QZ[1,ii]+QZ[2,ii]-10.*QZ[3,ii]) + QZ[2,ii]*(QZ[2,ii]-4.*QZ[3,ii]) +3.*QZ[1,ii]*(QZ[2,ii]-2.*QZ[3,ii]) )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[1,ii]-QZ[3,ii])*(QZ[2,ii]-QZ[0,ii])**2)  +  ( QZ[0,ii]**2 + 3.*QZ[0,ii]*QZ[1,ii] + 6.*QZ[1,ii]**2 - 2.*QZ[0,Ind[jj]]*(2.*QZ[0,ii]+3.*QZ[1,ii] - 5.*QZ[2,ii]) - QZ[0,ii]*QZ[2,ii] -9.*QZ[1,ii]*QZ[2,ii] )*(QZ[1,ii]-QZ[0,ii])**2/(30.*(QZ[0,Ind[jj]]-QZ[2,Ind[jj]])*(QZ[2,ii]-QZ[0,ii])**2)
                            elif kZ.size==2 and np.all(kZ==QZ[-2:,ii]):
                                intZDDZ = 4./((kZ[1]-kZ[0])*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]]))
                                intZDDR = (QZ[3,ii]-QZ[2,ii])**3/(30.*(QZ[3,ii]-QZ[1,ii])*(QZ[2,Ind[jj]]-QZ[0,Ind[jj]]))
                            elif kZ.size==3 and np.all(kZ==QZ[-3:,ii]):
                                intZDDZ = -4.*(QZ[3,Ind[jj]]+QZ[2,Ind[jj]]-QZ[2,ii]-QZ[1,ii])/((QZ[3,ii]-QZ[2,ii])*(QZ[3,Ind[jj]]-QZ[1,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2) - 4.*(QZ[3,ii]+QZ[2,ii]-QZ[1,ii]-QZ[0,ii])/((QZ[2,ii]-QZ[1,ii])*(QZ[2,ii]-QZ[0,ii])*(QZ[3,ii]-QZ[1,ii])**2)
                                intZDDR = ( 6.*QZ[2,ii]**2 - QZ[1,ii]*(9.*QZ[2,ii]+QZ[3,ii]-10.*QZ[3,Ind[jj]]) + QZ[3,ii]*(QZ[3,ii]-4.*QZ[3,Ind[jj]]) +3.*QZ[2,ii]*(QZ[3,ii]-2.*QZ[3,Ind[jj]]) )*(QZ[3,ii]-QZ[2,ii])**2/(30.*(QZ[2,ii]-QZ[3,Ind[jj]])*(QZ[3,ii]-QZ[1,ii])**2)  +  ( QZ[1,ii]**2 + 3.*QZ[1,ii]*QZ[2,ii] + 6.*QZ[2,ii]**2 - 2.*QZ[0,ii]*(2.*QZ[1,ii]+3.*QZ[2,ii] - 5.*QZ[3,ii]) - QZ[1,ii]*QZ[3,ii] -9.*QZ[2,ii]*QZ[3,ii] )*(QZ[2,ii]-QZ[1,ii])**2/(30.*(QZ[0,ii]-QZ[2,ii])*(QZ[3,ii]-QZ[1,ii])**2)
                            else:
                                assert np.all(kZ==QZ[0:3,ii]), "Something wrong with common knots..."

                        llR[0,Ind[jj]] = intRDDR*intZDDR
                        llZ[0,Ind[jj]] = intRDDZ*intZDDZ
                    llR[0,ii] = d0DDR[ii]*d0Z[ii]
                    llZ[0,ii] = d0R[ii]*d0DDZ[ii]
                    LLR.append(scpsp.coo_matrix(llR))
                    LLZ.append(scpsp.coo_matrix(llZ))

            AR, AZ = scpsp.vstack(LLR,format='csr'), scpsp.vstack(LLZ,format='csr')
            A = (AR,AZ)

        if not Sparse:
            if m in [0,1]:
                A = A.toarray()
            elif m==2:
                A = (A[0].toarray(),A[1].toarray())

    return A, m



















def Calc_BF2D_DerivFunc(BF2, Deriv, Test=True):
    if Test:
        assert isinstance(BF2, BF2D), "Arg BF2 must be a MeshBase2D instance !"
        assert type(Deriv) is int, "Arg Deriv must be a int !"

    KnR, KnZ = BF2._get_quadPoints()
    if Deriv==1:
        LFuncR, LFuncZ = [], []
        for ii in range(0,BF2.NFunc):
            LFuncR.append(lambda X, ii=ii: BSplineDeriv(BF2.Deg, KnZ[:,ii], Deriv=0, Test=False)[0](X[1,:])*BSplineDeriv(BF2.Deg, KnR[:,ii], Deriv=Deriv, Test=False)[0](X[0,:]))
            LFuncZ.append(lambda X, ii=ii: BSplineDeriv(BF2.Deg, KnR[:,ii], Deriv=0, Test=False)[0](X[0,:])*BSplineDeriv(BF2.Deg, KnZ[:,ii], Deriv=Deriv, Test=False)[0](X[1,:]))
        return LFuncR, LFuncZ
    elif Deriv==2:
        # Formulas for Gauss and Mean curvature were found on http://en.wikipedia.org/wiki/Differential_geometry_of_surfaces
        DRR, DRZ, DZZ = [], [], []
        for ii in range(0,BF2.NFunc):
            DRR.append(lambda X, ii=ii: BSplineDeriv(BF2.Deg, KnR[:,ii], Deriv=Deriv, Test=False)[0](X[0,:])*BSplineDeriv(BF2.Deg, KnZ[:,ii], Deriv=0, Test=False)[0](X[1,:]))
            DRZ.append(lambda X, ii=ii: BSplineDeriv(BF2.Deg, KnR[:,ii], Deriv=1, Test=False)[0](X[0,:])*BSplineDeriv(BF2.Deg, KnZ[:,ii], Deriv=1, Test=False)[0](X[1,:]))
            DZZ.append(lambda X, ii=ii: BSplineDeriv(BF2.Deg, KnR[:,ii], Deriv=0, Test=False)[0](X[0,:])*BSplineDeriv(BF2.Deg, KnZ[:,ii], Deriv=Deriv, Test=False)[0](X[1,:]))
        return DRR, DRZ, DZZ




def Get_MinMax(BF2, Coefs=1., Ratio=0.05, SubP=0.004, SubP1=0.015, SubP2=0.001, TwoSteps=False, SubMode='abs', Deriv='D0', Margin=0.2, Test=True):
    assert Ratio is None or (type(Ratio) is float and Ratio>0 and Ratio<1), "Arg Ratio must be None or a float in ]0;1[ !"
    if TwoSteps:
        X, Y = Calc_SumMeshGrid2D(BF2.Mesh.MeshR.Knots, BF2.Mesh.MeshZ.Knots, SubP=SubP1, SubMode='abs', Test=Test)
    else:
        X, Y = Calc_SumMeshGrid2D(BF2.Mesh.MeshR.Knots, BF2.Mesh.MeshZ.Knots, SubP=SubP, SubMode='abs', Test=Test)
    nx, ny = X.size, Y.size
    dS = np.mean(np.diff(X))*np.mean(np.diff(Y))
    Xplot, Yplot = np.tile(X,(ny,1)), np.tile(Y,(nx,1)).T
    Points = np.array([Xplot.flatten(), Yplot.flatten()])
    Vals = BF2.get_TotVal(Points, Deriv=Deriv, Coefs=Coefs, Test=Test)

    def getminmaxRatioFloat(valsmax, valsmin, indmax, indmin, Ratio, Ptsmin, Ptsmax):
        DV = valsmax[indmax]-valsmin[indmin]
        imin, imax = valsmin<=valsmin[indmin]+Ratio*DV, valsmax>=valsmax[indmax]-Ratio*DV
        PMin = np.array([np.sum(Ptsmin[0,imin]*valsmin[imin])/np.sum(valsmin[imin]), np.sum(Ptsmin[1,imin]*valsmin[imin])/np.sum(valsmin[imin])])
        PMax = np.array([np.sum(Ptsmax[0,imax]*valsmax[imax])/np.sum(valsmax[imax]), np.sum(Ptsmax[1,imax]*valsmax[imax])/np.sum(valsmax[imax])])
        VMin, VMax = np.mean(valsmin[imin]), np.mean(valsmax[imax])
        return PMin, PMax, VMin, VMax

    def get_XYgridFine(DXmin, DYmin, DXmax, DYmax, Margin, subp):
        DXmin, DYmin = [DXmin[0]-Margin*(DXmin[1]-DXmin[0]), DXmin[1]+Margin*(DXmin[1]-DXmin[0])], [DYmin[0]-Margin*(DYmin[1]-DYmin[0]), DYmin[1]+Margin*(DYmin[1]-DYmin[0])]
        DXmax, DYmax = [DXmax[0]-Margin*(DXmax[1]-DXmax[0]), DXmax[1]+Margin*(DXmax[1]-DXmax[0])], [DYmax[0]-Margin*(DYmax[1]-DYmax[0]), DYmax[1]+Margin*(DYmax[1]-DYmax[0])]
        Nxmin, Nymin = (DXmin[1]-DXmin[0])/subp, (DYmin[1]-DYmin[0])/subp
        Nxmax, Nymax = (DXmax[1]-DXmax[0])/subp, (DYmax[1]-DYmax[0])/subp
        Xmin, Ymin = np.linspace(DXmin[0],DXmin[1], Nxmin), np.linspace(DYmin[0],DYmin[1], Nymin)
        Xmax, Ymax = np.linspace(DXmax[0],DXmax[1], Nxmax), np.linspace(DYmax[0],DYmax[1], Nymax)
        Ptsmin = np.array([np.tile(Xmin,(Nymin,1)).flatten(), np.tile(Ymin,(Nxmin,1)).T.flatten()])
        Ptsmax = np.array([np.tile(Xmax,(Nymax,1)).flatten(), np.tile(Ymax,(Nxmax,1)).T.flatten()])
        return Ptsmin, Ptsmax

    def get_minmaxFine(vals, indmaxi, indmini, coefs, Points=Points, ratio=0.02, SubP2=SubP2, Ratio=Ratio, Test=Test):
        DV = vals[indmaxi]-vals[indmini]
        imin, imax = vals<=vals[indmini]+ratio*DV, vals>=vals[indmaxi]-ratio*DV
        xminmin, xminmax = np.min(Points[0,imin]), np.max(Points[0,imin])
        xmaxmin, xmaxmax = np.min(Points[0,imax]), np.max(Points[0,imax])
        yminmin, yminmax = np.min(Points[1,imin]), np.max(Points[1,imin])
        ymaxmin, ymaxmax = np.min(Points[1,imax]), np.max(Points[1,imax])
        Ptsmin, Ptsmax = get_XYgridFine([xminmin,xminmax], [yminmin,yminmax], [xmaxmin, xmaxmax], [ymaxmin, ymaxmax], Margin, SubP2)
        valsmin, valsmax = BF2.get_TotVal(Ptsmin, Deriv=Deriv, Coefs=coefs, Test=Test), BF2.get_TotVal(Ptsmax, Deriv=Deriv, Coefs=coefs, Test=Test)
        indmin, indmax = np.nanargmin(valsmin), np.nanargmax(valsmax)
        if Ratio is None:
            return Ptsmin[:,indmin], Ptsmax[:,indmax], valsmin[indmin], valsmax[indmax]
        else:
            return getminmaxRatioFloat(valsmax, valsmin, indmax, indmin, Ratio, Ptsmin, Ptsmax)

    if not hasattr(Coefs,'__getitem__') or Coefs.ndim==1:
        indmin, indmax = np.nanargmin(Vals), np.nanargmax(Vals)
        if TwoSteps:
            PMin, PMax, VMin, VMax = get_minmaxFine(Vals, indmax, indmin, Coefs, Points=Points, ratio=0.02, SubP2=SubP2, Ratio=Ratio, Test=Test)
        else:
            if Ratio is None:
                PMin, PMax = Points[:,indmin], Points[:,indmax]
                VMin, VMax = Vals[indmin], Vals[indmax]
            else:
                PMin, PMax, VMin, VMax = getminmaxRatioFloat(Vals, Vals, indmax, indmin, Ratio, Points, Points)
        Surf = (Vals >= Vals(np.nanargmax(Vals))*0.5).sum()*dS
    else:
        indmin, indmax = np.nanargmin(Vals,axis=1), np.nanargmax(Vals,axis=1)
        if TwoSteps:
            ratio = 0.02 if Ratio is None else Ratio+0.02
            mmin, mmax = np.nanmin(Vals,axis=1).max(), np.nanmax(Vals,axis=1).min()
            DV = np.max(np.nanmax(Vals,axis=1)-np.nanmin(Vals,axis=1))
            assert mmin+ratio*DV <= mmax-ratio*DV, "Profile changes too much !"
            imin, imax = np.any(Vals<=mmin+ratio*DV,axis=0), np.any(Vals>=mmax-ratio*DV,axis=0)
            DXmin, DYmin = [Points[0,imin].min(), Points[0,imin].max()], [Points[1,imin].min(), Points[1,imin].max()]
            DXmax, DYmax = [Points[0,imax].min(), Points[0,imax].max()], [Points[1,imax].min(), Points[1,imax].max()]
            Ptsmin, Ptsmax = get_XYgridFine(DXmin, DYmin, DXmax, DYmax, Margin, SubP2)
            Valsmin, Valsmax = BF2.get_TotVal(Ptsmin, Deriv=Deriv, Coefs=Coefs, Test=Test), BF2.get_TotVal(Ptsmax, Deriv=Deriv, Coefs=Coefs, Test=Test)
        else:
            Ptsmin, Ptsmax = Points, Points
            Valsmin, Valsmax = Vals, Vals
        indmin, indmax = np.nanargmin(Valsmin,axis=1), np.nanargmax(Valsmax,axis=1)
        if Ratio is None:
            PMin, PMax = Ptsmin[:,indmin].T, Ptsmax[:,indmax].T
            VMin, VMax = np.nanmin(Valsmin,axis=1), np.nanmax(Valsmax,axis=1)
        else:
            Nt = Coefs.shape[0]
            PMin, PMax = np.empty((Nt,2)), np.empty((Nt,2))
            VMin, VMax = np.empty((Nt,)), np.empty((Nt,))
            for ii in range(0,Nt):
                PMin[ii,:], PMax[ii,:], VMin[ii], VMax[ii] = getminmaxRatioFloat(Valsmax[ii,:], Valsmin[ii,:], indmax[ii], indmin[ii], Ratio, Ptsmin, Ptsmax)
        Surf = np.sum(Vals >= np.tile(np.nanmax(Vals,axis=1)*0.5,(Vals.shape[1],1)).T,axis=1)*dS

    return PMin, PMax, VMin, VMax, Surf








def Calc_GetRoots(BF2, Deriv=0., Coefs=1., Test=True):
    if Test:
        assert isinstance(BF2, BF2D), "Arg BF2 must be a BF2D instance !"
        assert Deriv in [0,1,2,3,'D0','D1','D2','D3','D0N2','D1N2','D1FI'], "Arg Deriv must be in [0,1,2,3,'D0','D1','D2','D3','D0N2','D1N2','D1FI'] !"
    if type(Deriv) is str:
        intDeriv = int(Deriv[1])
    else:
        intDeriv = Deriv
    assert BF2.Deg > 0 and intDeriv<BF2.Deg, "Cannot find roots for Deg=0 and Deriv=Deg (root-finding only for continuous functions)"
    NCents = BF2.Mesh.NCents
    NKnots = BF2.Mesh.NKnots
    NbF = BF2.NFunc
    if type(Coefs) in [int,float]:
        Coefs = float(Coefs)*np.ones((NbF,))
    Inds = np.zeros((NCents,),dtype=bool)
    Pts = np.nan*np.zeros((2,NCents))
    Shape = [['',[]] for ii in range(0,NCents)]
    if intDeriv==0:
        if BF2.Deg==1:
            for ii in range(0,NCents):
                ind = BF2._Cents_Funcind[:,ii]
                C4 = Coefs[np.unique(ind[~np.isnan(ind)]).astype(int)]
                Kts = BF2.Mesh.Knots[:,np.unique(BF2.Mesh._Cents_Knotsind[:,ii])]
                if not (np.all(C4<0) or np.all(C4>0)) and C4.size==4:
                    Inds[ii] = True
                    A = C4[0]+C4[3] - C4[1]+C4[2]
                    B = C4[1]*Kts[1,1] - C4[3]*Kts[1,0] - C4[0]*Kts[1,1] + C4[2]*Kts[1,0]
                    C = C4[1]*Kts[0,0] - C4[3]*Kts[0,0] - C4[0]*Kts[0,1] + C4[2]*Kts[0,1]
                    D = Kts[0,1]*(C4[0]*Kts[1,1]-C4[2]*Kts[1,0]) - Kts[0,0]*(C4[1]*Kts[1,1]-C4[3]*Kts[1,0])
                    if A==0.:
                        if C==0.:
                            if B==0.:
                                Shape[ii] = ['all',[]] if D==0. else ['',[]] # Else not possible
                            else:
                                Shape[ii] = ['x',[-D/B]]
                        else:
                            Shape[ii] = ['yx',[-B/C,-D/C]]
                    else:
                        if -C/A>Kts[0,1] or -C/A<Kts[0,0]:
                            Shape[ii] = ['y/x',[(B*C-D)/A,C/A,-B/C]]
                        else:
                            print "        Singularity"
        elif BF2.Deg==2:
            for ii in range(0,NCents):
                ind = BF2._Cents_Funcind[:,ii]
                C9 = Coefs[np.unique(ind[~np.isnan(ind)]).astype(int)]
                Kts = BF2.Mesh.Knots[:,np.unique(BF2.Mesh._Cents_Knotsind[:,ii])]
                if not (np.all(C9<0) or np.all(C9>0)) and C9.size==9:
                    Inds[ii] = True
                    print "        Not finished yet !"




    if intDeriv==1:
        if Deriv[1]=='D1N2':
            if BF2.Deg==2:
                for ii in range(0,NCents):
                    ind = BF2._Cents_Funcind[:,ii]
                    C9 = Coefs[np.unique(ind[~np.isnan(ind)]).astype(int)]
                    Kts = BF2.Mesh.Knots[:,np.unique(BF2.Mesh._Cents_Knotsind[:,ii])]

                    #A0, B0 =
                    #A1, B1 =
                    #A2, B2 =
                    #alpha0, beta0, gam0 =
                    #alpha1, beta1, gam1 =
                    #alpha2, beta2, gam2 =
    return Inds, Pts, Shape






############################################
#####     Plotting functions
############################################



def Plot_BF2D(BF2, Coef=1., ax='None',SubP=0.1, SubMode='Rel', Name='Tot', TotDict=Tot2Dict_Def):
    assert isinstance(BF2,BF2D), "Arg BF2 must be a BF2D instance !"
    assert ax=='None' or isinstance(ax,plt.Axes), "Arg ax must be a plt.Axes instance !"
    assert type(Name) is str, "Arg Name must be a str !"

    assert type(TotDict) is dict, "Arg TotDict must be a dict !"

    X, Y = Calc_SumMeshGrid2D(BF2.Mesh.MeshR.Knots, BF2.Mesh.MeshZ.Knots, SubP=SubP, SubMode=SubMode, Test=True)
    nx, ny = X.size, Y.size
    Xplot, Yplot = np.dot(np.ones((ny,1)),X.reshape((1,nx))), np.dot(Y.reshape((ny,1)),np.ones((1,nx)))
    Points = np.concatenate((Xplot.reshape((1,nx*ny)), Yplot.reshape((1,nx*ny))),axis=0)
    Z = Calc_BF2D_Val(BF2.LFunc, Points, Coef=Coef, Test=True)
    Zplot= Z.reshape((ny,nx))

    if ax=='None':
        ax = tfd.Plot_BSpline_DefAxes('2D')
    ax.plot_surface(Xplot,Yplot,Zplot, label=Name, **TotDict)

    ax.figure.canvas.draw()
    return ax




def Plot_BF2D_BFuncMesh(BF2, ind, Coef=1., ax1='None', ax2='None',SubP=0.25, SubMode='Rel', Name='', TotDict=Tot2Dict_Def):
    assert isinstance(BF2,BF2D), "Arg BF2 must be a BF2D instance !"
    assert type(ind) is int, "Arg ind must be a int !"
    assert ax1=='None' or isinstance(ax1,plt.Axes), "Arg ax1 must be a plt.Axes instance !"
    assert ax2=='None' or isinstance(ax2,plt.Axes), "Arg ax2 must be a plt.Axes instance !"
    assert type(Name) is str, "Arg Name must be a str !"
    assert type(TotDict) is dict, "Arg TotDict must be a dict !"

    X, Y = Calc_SumMeshGrid2D(BF2.Mesh.MeshR.Knots, BF2.Mesh.MeshZ.Knots, SubP=SubP, SubMode=SubMode, Test=True)
    nx, ny = X.size, Y.size
    Xplot, Yplot = np.dot(np.ones((ny,1)),X.reshape((1,nx))), np.dot(Y.reshape((ny,1)),np.ones((1,nx)))
    Points = np.concatenate((Xplot.reshape((1,nx*ny)), Yplot.reshape((1,nx*ny))),axis=0)
    Z = Calc_BF2D_Val([BF2.LFunc[ind]], Points, Coef=Coef, Test=True)
    Zplot= Z.reshape((ny,nx))

    if ax1=='None' or ax2=='None':
        ax1, ax2 = tfd.Plot_BF2D_BFuncMesh_DefAxes()

    BF2.Mesh.plot(ax=ax1)
    BF2.Mesh.plot_Cents(ax=ax1,Ind=BF2.Func_Centsind[:,ind], Knots=False)
    BF2.Mesh.plot_Knots(ax=ax1,Ind=BF2.Func_Knotsind[:,ind], Cents=False)

    ax2.plot_surface(Xplot,Yplot,Zplot, label=Name, **TotDict)
    ax1.figure.canvas.draw()
    return ax1, ax2


def Plot_BFunc_SuppMax_PolProj(BF2, ind, ax='None', Supp=True,PMax=True,SuppDict=SuppDict_Def, PMaxDict=PMaxDict_Def):
    assert type(ind) is int, "Arg ind must be a int !"
    assert type(Supp) is bool, "Arg Supp must be a bool !"
    assert type(PMax) is bool, "Arg Supp must be a bool !"
    assert type(SuppDict) is dict, "Arg SuppDict must be a dict !"
    assert type(PMaxDict) is dict, "Arg SuppDict must be a dict !"
    assert isinstance(ax,plt.Axes) or ax=='None', "Arg ax must be a plt.Axes instance !"


    if ax=='None':
        ax = tfd.Plot_BFunc_SuppMax_PolProj_DefAxes()

    if Supp:
        RZsupp = BF2.get_Func_SuppRZ()[:,ind]
        verts = [(RZsupp[0], RZsupp[2]), # left, bottom
                (RZsupp[1], RZsupp[2]), # left, top
                (RZsupp[1], RZsupp[3]), # right, top
                (RZsupp[0], RZsupp[3]), # right, bottom
                (RZsupp[0], RZsupp[2])]

        codes = [Path.MOVETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.CLOSEPOLY]

        patch = patches.PathPatch(Path(verts, codes), **SuppDict)
        ax.add_patch(patch)
    if PMax:
        PMax = BF2.Func_PMax[:,ind]
        ax.plot(PMax[0],PMax[1], **PMaxDict)

    return ax



"""
###############################################################################
###############################################################################
                        Testing ground
###############################################################################
"""
"""
Deg = 2
KnotsMult1 = np.array([0.,1.,2.,3.,4.,5., 5., 5.])
KnotsMult2 = np.array([0.,0.,1.,2.,3.,4.,5.])
BS1 = BSpline(Deg,KnotsMult1)[0]
BS2 = BSpline(Deg,KnotsMult2)[0]
#ax = Plot_BSpline1D(KnotsMult1,BS1,ax='None',SubP=0.05,SubMode='Rel')
ax1, ax2 = Plot_BSpline2D(KnotsMult1, KnotsMult2, BS1, BS2, ax1=None, ax2='None',SubP=0.05,SubMode='Rel')
"""

