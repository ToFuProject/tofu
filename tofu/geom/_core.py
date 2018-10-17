"""
This module is the geometrical part of the ToFu general package
It includes all functions and object classes necessary for tomography on Tokamaks
"""

# Built-in
import os
import warnings
from abc import ABCMeta, abstractmethod
import copy

# Common
import numpy as np
import matplotlib as mpl
import datetime as dtm
try:
    import pandas as pd
except Exception:
    lm = ['tf.geom.Config.get_description()']
    msg = "Could not import pandas, "
    msg += "the following may not work :"
    msg += "\n    - ".join(lm)
    warnings.warn(msg)


# ToFu-specific
import tofu.pathfile as tfpf
import tofu.utils as utils
try:
    import tofu.geom._def as _def
    import tofu.geom._GG as _GG
    import tofu.geom._comp as _comp
    import tofu.geom._plot as _plot
except Exception:
    from . import _def as _def
    from . import _GG as _GG
    from . import _comp as _comp
    from . import _plot as _plot

__all__ = ['Ves', 'PFC', 'CoilPF', 'CoilCS',
           'Rays','LOSCam1D','LOSCam2D']


_arrayorder = 'C'
_Clock = False
_Type = 'Tor'




"""
###############################################################################
###############################################################################
                        Ves class and functions
###############################################################################
"""



class Struct(utils.ToFuObject):
    """ A class defining a Linear or Toroidal vaccum vessel (i.e. a 2D polygon representing a cross-section and assumed to be linearly or toroidally invariant)

    A Ves object is mostly defined by a close 2D polygon, which can be understood as a poloidal cross-section in (R,Z) cylindrical coordinates if Type='Tor' (toroidal shape) or as a straight cross-section through a cylinder in (Y,Z) cartesian coordinates if Type='Lin' (linear shape).
    Attributes such as the surface, the angular volume (if Type='Tor') or the center of mass are automatically computed.
    The instance is identified thanks to an attribute Id (which is itself a tofu.ID class object) which contains informations on the specific instance (name, Type...).

    Parameters
    ----------
    Id :            str / tfpf.ID
        A name string or a pre-built tfpf.ID class to be used to identify this particular instance, if a string is provided, it is fed to tfpf.ID()
    Poly :          np.ndarray
        An array (2,N) or (N,2) defining the contour of the vacuum vessel in a cross-section, if not closed, will be closed automatically
    Type :          str
        Flag indicating whether the vessel will be a torus ('Tor') or a linear device ('Lin')
    Lim :         list / np.ndarray
        Array or list of len=2 indicating the limits of the linear device volume on the x axis
    Sino_RefPt :    None / np.ndarray
        Array specifying a reference point for computing the sinogram (i.e. impact parameter), if None automatically set to the (surfacic) center of mass of the cross-section
    Sino_NP :       int
        Number of points in [0,2*pi] to be used to plot the vessel sinogram envelop
    Clock :         bool
        Flag indicating whether the input polygon should be made clockwise (True) or counter-clockwise (False)
    arrayorder:     str
        Flag indicating whether the attributes of type=np.ndarray (e.g.: Poly) should be made C-contiguous ('C') or Fortran-contiguous ('F')
    Exp :           None / str
        Flag indicating which experiment the object corresponds to, allowed values are in [None,'AUG','MISTRAL','JET','ITER','TCV','TS','Misc']
    shot :          None / int
        Shot number from which this Ves is usable (in case of change of geometry)
    SavePath :      None / str
        If provided, forces the default saving path of the object to the provided value

    Returns
    -------
    Ves :        Ves object
        The created Ves object, with all necessary computed attributes and methods

    """

    __metaclass__ = ABCMeta


    # Fixed (class-wise) dictionary of default properties
    _ddef = {'Id':{'shot':0,
                   'include':['Mod','Cls','Exp','Diag','Name','shot']},
             'dgeom':{'Type':'Tor', 'Lim':[], 'arrayorder':'C'},
             'dsino':{},
             'dphys':{},
             'dmisc':{'color':(0.8,0.8,0.8,0.8)}}

    def __init_subclass__(cls, color=_ddef['dmisc']['color'], **kwdargs):
        super().__init_subclass__(**kwdargs)
        cls._ddef = copy.deepcopy(Struct._ddef)
        cls._ddef['dmisc']['color'] = mpl.colors.to_rgba(color)

    def __init__(self, Poly=None, Type=None, Lim=None, mobile=False,
                 Id=None, Name=None, Exp=None, shot=None,
                 sino_RefPt=None, sino_nP=_def.TorNP,
                 Clock=False, arrayorder='C', fromdict=None,
                 SavePath=os.path.abspath('./'),
                 SavePath_Include=tfpf.defInclude, color=None):

        kwdargs = locals()
        del kwdargs['self']
        super().__init__(**kwdargs)

    def _reset(self):
        super()._reset()
        self._dgeom = dict.fromkeys(self._get_keys_dgeom())
        self._dsino = dict.fromkeys(self._get_keys_dsino())
        self._dphys = dict.fromkeys(self._get_keys_dphys())
        self._dmisc = dict.fromkeys(self._get_keys_dmisc())

    @classmethod
    def _checkformat_inputs_Id(cls, Id=None, Name=None,
                               Exp=None, shot=None, Type=None,
                               include=None,
                               **kwdargs):
        if Id is not None:
            assert isinstance(Id,utils.ID)
            Name, Exp, shot, Type = Id.Name, Id.Exp, Id.shot, Id.Type
        assert type(Name) is str
        assert type(Exp) is str
        if shot is None:
            shot = cls._ddef['Id']['shot']
        assert type(shot) is int
        if Type is None:
            Type = cls._ddef['dgeom']['Type']
        assert Type in ['Tor','Lin']
        if include is None:
            include = cls._ddef['Id']['include']
        kwdargs.update({'Name':Name, 'Exp':Exp, 'shot':shot, 'Type':Type,
                        'include':include})
        return kwdargs

    ###########
    # Get largs
    ###########

    @staticmethod
    def _get_largs_dgeom(sino=True):
        largs = ['Poly','Lim','mobile','Clock','arrayorder']
        if sino:
            lsino = Struct._get_largs_dsino()
            largs += ['sino_{0}'.format(s) for s in lsino]
        return largs

    @staticmethod
    def _get_largs_dsino():
        largs = ['RefPt','nP']
        return largs

    @staticmethod
    def _get_largs_dphys():
        largs = ['lSymbols']
        return largs

    @staticmethod
    def _get_largs_dmisc():
        largs = ['color']
        return largs

    ###########
    # Get check and format inputs
    ###########

    @staticmethod
    def _checkformat_inputs_dgeom(Poly=None, Lim=None, mobile=False,
                                  Type=None, Clock=False, arrayorder=None):
        assert type(Clock) is bool
        assert type(mobile) is bool
        if arrayorder is None:
            arrayorder = Struct._ddef['dgeom']['arrayorder']
        assert arrayorder in ['C','F']
        assert Poly is not None and hasattr(Poly,'__iter__')
        Poly = np.asarray(Poly).astype(float)
        assert Poly.ndim==2 and 2 in Poly.shape
        if Poly.shape[0]!=2:
            Poly = Poly.T
        if Type is None:
            Type = Struct._ddef['dgeom']['Type']
        assert Type in ['Tor','Lin']
        if Lim is None:
            Lim = Struct._ddef['dgeom']['Lim']
        assert hasattr(Lim,'__iter__')
        Lim = np.asarray(Lim).astype(float)
        assert Lim.ndim in [1,2] and (2 in Lim.shape or 0 in Lim.shape)
        if Lim.ndim==1:
            assert Lim.size in [0,2]
            if Lim.size==2:
                Lim = Lim.reshape((2,1))
        else:
            if Lim.shape[0]!=2:
                Lim = Lim.T
        if Type=='Lin':
            assert Lim.size>0
        return Poly, Lim, Type, arrayorder

    def _checkformat_inputs_dsino(self, RefPt=None, nP=None):
        assert type(nP) is int and nP>0
        assert RefPt is None or hasattr(RefPt,'__iter__')
        if RefPt is None:
            RefPt = self._dgeom['BaryS']
        RefPt = np.asarray(RefPt,dtype=float).flatten()
        return RefPt

    @staticmethod
    def _checkformat_inputs_dphys(lSymbols=None):
        if lSymbols is not None:
            assert type(lSymbols) in [list,str]
            if type(lSymbols) is list:
                assert all([type(ss) is str for ss in lSymbols])
            else:
                lSymbols = [lSymbols]
            lSymbols = np.asarray(lSymbols,dtype=str)
        return lSymbols

    @classmethod
    def _checkformat_inputs_dmisc(cls, color=None):
        if color is None:
            color = cls._ddef['dmisc']['color']
        assert mpl.colors.is_color_like(color)
        color = mpl.colors.to_rgba(color)
        return color

    ###########
    # Get keys of dictionnaries
    ###########

    @staticmethod
    def _get_keys_dgeom():
        lk = ['Poly','Lim','nLim','Multi','nP',
              'P1Max','P1Min','P2Max','P2Min',
              'BaryP','BaryL','BaryS','BaryV',
              'Surf','VolAng','Vect','VIn','mobile',
              'circ-C','circ-r','Clock','arrayorder']
        return lk

    @staticmethod
    def _get_keys_dsino():
        lk = ['RefPt','nP','EnvTheta','EnvMinMax']
        return lk

    @staticmethod
    def _get_keys_dphys():
        lk = ['lSymbols']
        return lk

    @staticmethod
    def _get_keys_dmisc():
        lk = ['color']
        return lk

    ###########
    # _init
    ###########

    def _init(self, Poly=None, Type=_Type, Lim=None,
              Clock=_Clock, arrayorder=_arrayorder,
              sino_RefPt=None, sino_nP=_def.TorNP, **kwdargs):
        largs = self._get_largs_dgeom(sino=True)
        kwdgeom = self._extract_kwdargs(locals(), largs)
        largs = self._get_largs_dphys()
        kwdphys = self._extract_kwdargs(locals(), largs)
        largs = self._get_largs_dmisc()
        kwdmisc = self._extract_kwdargs(locals(), largs)
        self._set_dgeom(**kwdgeom)
        self.set_dphys(**kwdphys)
        self._set_dmisc(**kwdmisc)
        self._dstrip['strip'] = 0

    ###########
    # set dictionaries
    ###########

    def _set_dgeom(self, Poly=None, Lim=None, mobile=False,
                   Clock=False, arrayorder='C',
                   sino_RefPt=None, sino_nP=_def.TorNP, sino=True):
        out = self._checkformat_inputs_dgeom(Poly=Poly, Lim=Lim, mobile=mobile,
                                             Type=self.Id.Type, Clock=Clock)
        Poly, Lim, Type, arrayorder = out
        dgeom = _comp._Struct_set_Poly(Poly, Lim=Lim,
                                       arrayorder=arrayorder,
                                       Type=self.Id.Type, Clock=Clock)
        dgeom['arrayorder'] = arrayorder
        dgeom['mobile'] = mobile
        self._dgeom = dgeom
        if sino:
            self.set_dsino(sino_RefPt, nP=sino_nP)

    def set_dsino(self, RefPt=None, nP=_def.TorNP):
        RefPt = self._checkformat_inputs_dsino(RefPt=RefPt, nP=nP)
        EnvTheta, EnvMinMax = _GG.Sino_ImpactEnv(RefPt, self.Poly,
                                                 NP=nP, Test=False)
        self._dsino = {'RefPt':RefPt, 'nP':nP,
                       'EnvTheta':EnvTheta, 'EnvMinMax':EnvMinMax}

    def set_dphys(self, lSymbols=None):
        lSymbols = self._checkformat_inputs_dphys(lSymbols)
        self._dphys['lSymbols'] = lSymbols

    def _set_dmisc(self, color=None):
        color = self._checkformat_inputs_dmisc(color=color)
        self._dmisc['color'] = color

    ###########
    # strip dictionaries
    ###########

    def _strip_dgeom(self, lkeep=['Poly','Lim','mobile','Clock','arrayorder']):
        utils.ToFuObject._strip_dict(self._dgeom, lkeep=lkeep)

    def _strip_dsino(self, lkeep=['RefPt','nP']):
        utils.ToFuObject._strip_dict(self._dsino, lkeep=lkeep)

    def _strip_dphys(self, lkeep=['lSymbols']):
        utils.ToFuObject._strip_dict(self._dphys, lkeep=lkeep)

    def _strip_dmisc(self, lkeep=['color']):
        utils.ToFuObject._strip_dict(self._dmisc, lkeep=lkeep)

    ###########
    # rebuild dictionaries
    ###########

    def _rebuild_dgeom(self, lkeep=['Poly','Lim','mobile','Clock','arrayorder']):
        reset = utils.ToFuObject._test_Rebuild(self._dgeom, lkeep=lkeep)
        if reset:
            utils.ToFuObject._check_Fields4Rebuild(self._dgeom,
                                                   lkeep=lkeep, dname='dgeom')
            self._set_dgeom(self.Poly, Lim=self.Lim,
                            Clock=self.dgeom['Clock'],
                            arrayorder=self.dgeom['arrayorder'],
                            sino=False)

    def _rebuild_dsino(self, lkeep=['RefPt','nP']):
        reset = utils.ToFuObject._test_Rebuild(self._dsino, lkeep=lkeep)
        if reset:
            utils.ToFuObject._check_Fields4Rebuild(self._dsino,
                                                   lkeep=lkeep, dname='dsino')
            self.set_dsino(RefPt=self.dsino['RefPt'], nP=self.dsino['nP'])

    def _rebuild_dphys(self, lkeep=['lSymbols']):
        reset = utils.ToFuObject._test_Rebuild(self._dphys, lkeep=lkeep)
        if reset:
            utils.ToFuObject._check_Fields4Rebuild(self._dphys,
                                                   lkeep=lkeep, dname='dphys')
            self.set_dphys(lSymbols=self.dphys['lSymbols'])

    def _rebuild_dmisc(self, lkeep=['color']):
        reset = utils.ToFuObject._test_Rebuild(self._dmisc, lkeep=lkeep)
        if reset:
            utils.ToFuObject._check_Fields4Rebuild(self._dmisc,
                                                   lkeep=lkeep, dname='dmisc')
            self._set_dmisc(color=self.dmisc['color'])

    ###########
    # _strip and get/from dict
    ###########

    @classmethod
    def _strip_init(cls):
        cls._dstrip['allowed'] = [0,1,2]
        nMax = max(cls._dstrip['allowed'])
        doc = """
                 1: Remove dsino expendables
                 2: Remove also dgeom, dphys and dmisc expendables"""
        doc = utils.ToFuObjectBase.strip.__doc__.format(doc,nMax)
        cls.strip.__doc__ = doc

    def strip(self, strip=0):
        super().strip(strip=strip)

    def _strip(self, strip=0):
        if strip==0:
            self._rebuild_dgeom()
            self._rebuild_dsino()
            self._rebuild_dphys()
            self._rebuild_dmisc()
        elif strip==1:
            self._strip_dsino()
            self._rebuild_dgeom()
            self._rebuild_dphys()
            self._rebuild_dmisc()
        else:
            self._strip_dsino()
            self._strip_dgeom()
            self._strip_dphys()
            self._strip_dmisc()

    def _to_dict(self):
        dout = {'dgeom':{'dict':self.dgeom, 'lexcept':None},
                'dsino':{'dict':self.dsino, 'lexcept':None},
                'dphys':{'dict':self.dphys, 'lexcept':None},
                'dmisc':{'dict':self.dmisc, 'lexcept':None}}
        return dout

    def _from_dict(self, fd):
        self._dgeom.update(**fd['dgeom'])
        self._dsino.update(**fd['dsino'])
        self._dphys.update(**fd['dphys'])
        self._dmisc.update(**fd['dmisc'])


    ###########
    # Properties
    ###########

    @property
    def Type(self):
        """Return the type of structure """
        return self._Id.Type
    @property
    def dgeom(self):
        return self._dgeom
    @property
    def Poly(self):
        """Return the polygon defining the structure cross-section"""
        return self._dgeom['Poly']
    @property
    def Poly_closed(self):
        """ Returned the closed polygon """
        return np.hstack((self._dgeom['Poly'],self._dgeom['Poly'][:,0:1]))
    @property
    def Lim(self):
        return self._dgeom['Lim']
    @property
    def nLim(self):
        return self._dgeom['nLim']
    @property
    def dsino(self):
        return self._dsino
    @property
    def dphys(self):
        return self._dphys
    @property
    def dmisc(self):
        return self._dmisc
    @property
    def color(self):
        return self._dmisc['color']
    @color.setter
    def color(self, val):
        self._dmisc['color'] = mpl.colors.to_rgba(val)


    ###########
    # public methods
    ###########


    @abstractmethod
    def isMobile(self):
        return self._dgeom['mobile']

    def isInside(self, pts, In='(X,Y,Z)'):
        """ Return an array of booleans indicating whether each point lies
        inside the Struct volume

        Tests for each point whether it lies inside the Struct object.
        The points coordinates can be provided in 2D or 3D
        You must specify which coordinate system is used with 'In' kwdarg.
        An array of boolean flags is returned.

        Parameters
        ----------
        pts :   np.ndarray
            (2,N) or (3,N) array, coordinates of the points to be tested
        In :    str
            Flag indicating the coordinate system in which pts are provided
            e.g.: '(X,Y,Z)' or '(R,Z)'

        Returns
        -------
        ind :   np.ndarray
            (N,) array of booleans, True if a point is inside the volume

        """
        ind = _GG._Ves_isInside(pts, self.Poly, Lim=self.Lim,
                                VType=self.Id.Type, In=In, Test=True)
        return ind


    def get_InsideConvexPoly(self, RelOff=_def.TorRelOff, ZLim='Def',
                             Spline=True, Splprms=_def.TorSplprms,
                             NP=_def.TorInsideNP, Plot=False, Test=True):
        """ Return a polygon that is a smaller and smoothed approximation of Ves.Poly, useful for excluding the divertor region in a Tokamak

        For some uses, it can be practical to approximate the polygon defining the Ves object (which can be non-convex, like with a divertor), by a simpler, sligthly smaller and convex polygon.
        This method provides a fast solution for computing such a proxy.

        Parameters
        ----------
        RelOff :    float
            Fraction by which an homothetic polygon should be reduced (1.-RelOff)*(Poly-BaryS)
        ZLim :      None / str / tuple
            Flag indicating what limits shall be put to the height of the polygon (used for excluding divertor)
        Spline :    bool
            Flag indiating whether the reduced and truncated polygon shall be smoothed by 2D b-spline curves
        Splprms :   list
            List of 3 parameters to be used for the smoothing [weights,smoothness,b-spline order], fed to scipy.interpolate.splprep()
        NP :        int
            Number of points to be used to define the smoothed polygon
        Plot :      bool
            Flag indicating whether the result shall be plotted for visual inspection
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        Poly :      np.ndarray
            (2,N) polygon resulting from homothetic transform, truncating and optional smoothing

        """
        return _comp._Ves_get_InsideConvexPoly(self.Poly, self.dgeom['P2Min'],
                                               self.dgeom['P2Max'],
                                               self.dgeom['BaryS'],
                                               RelOff=RelOff, ZLim=ZLim,
                                               Spline=Spline, Splprms=Splprms,
                                               NP=NP, Plot=Plot, Test=Test)

    def get_sampleEdge(self, res, DS=None, resMode='abs', offsetIn=0.):
        """ Sample the polygon edges, with resolution res

        Sample each segment of the 2D polygon
        Sampling can be limited to a subdomain defined by DS
        """
        pts, dlr, ind = _comp._Ves_get_sampleEdge(self.Poly, res, DS=DS,
                                                  dLMode=resMode, DIn=offsetIn,
                                                  VIn=self.dgeom['VIn'],
                                                  margin=1.e-9)
        return pts, dlr, ind

    def get_sampleCross(self, res, DS=None, resMode='abs', ind=None):
        """ Sample, with resolution res, the 2D cross-section

        The sampling domain can be limited by DS or ind
        """
        args = [self.Poly, self.dgeom['P1Min'][0], self.dgeom['P1Max'][0],
                self.dgeom['P2Min'][1], self.dgeom['P2Max'][1], res]
        kwdargs = dict(DS=DS, dsMode=resMode, ind=ind, margin=1.e-9)
        pts, dS, ind, reseff = _comp._Ves_get_sampleCross(*args, **kwdargs)
        return pts, dS, ind, reseff

    def get_sampleS(self, res, DS=None, dSMode='abs',
                    ind=None, offsetIn=0., Out='(X,Y,Z)', Ind=None):
        """ Sample, with resolution res, the surface defined by DS or ind

        An optionnal offset perpendicular to the surface can be used
        (offsetIn>0 => inwards)

        Parameters
        ----------
        res     :   float / list of 2 floats
            Desired resolution of the surfacic sample
                float   : same resolution for all directions of the sample
                list    : [dl,dXPhi] where:
                    dl      : res. along polygon contours (cross-section)
                    dXPhi   : res. along axis (toroidal/linear direction)
        DS      :   None / list of 3 lists of 2 floats
            Limits of the domain in which the sample should be computed
                None : whole surface of the object
                list : [D1,D2,D3], where Di is a len()=2 list
                       (increasing floats, setting limits along coordinate i)
                    [DR,DZ,DPhi]: in toroidal geometry (self.Id.Type=='Tor')
                    [DX,DY,DZ]  : in linear geometry (self.Id.Type=='Lin')
        resMode  :   str
            Flag, specifies if res is absolute or relative to element sizes
                'abs'   :   res is an absolute distance
                'rel'   :   if res=0.1, each polygon segment is divided in 10,
                            as is the toroidal/linear length
        ind     :   None / np.ndarray of int
            If provided, DS is ignored and the sample points corresponding to
            the provided indices are returned
            Example (assuming obj is a Ves object)
                > # We create a 5x5 cm2 sample of the whole surface
                > pts, dS, ind, reseff = obj.get_sample(0.05)
                > # Perform operations, save only the points indices (save space)
                > ...
                > # Retrieve the points from their indices (requires same res)
                > pts2, dS2, ind2, reseff2 = obj.get_sample(0.05, ind=ind)
                > np.allclose(pts,pts2)
                True
        offsetIn:   float
            Offset distance from the actual surface of the object
            Inwards if positive
            Useful to avoid numerical errors
        Out     :   str
            Flag indicating the coordinate system of returned points
            e.g. : '(X,Y,Z)' or '(R,Z,Phi)'
        Ind     :   None / iterable of ints
            Array of indices of the entities to be considered
            (only when multiple entities, i.e.: self.nLim>1)

        Returns
        -------
        pts     :   np.ndarray / list of np.ndarrays
            Sample points coordinates, as a (3,N) array.
            A list is returned if the object has multiple entities
        dS      :   np.ndarray / list of np.ndarrays
            The surface (in m^2) associated to each point
        ind     :   np.ndarray / list of np.ndarrays
            The index of each point
        reseff  :   np.ndarray / list of np.ndarrays
            Effective resolution in both directions after sample computation
        """
        if Ind is not None:
            assert self.dgeom['Multi']
        kwdargs = dict(DS=DS, dSMode=resMode, ind=ind, DIn=offsetIn,
                       VIn=self.dgeom['VIn'], VType=self.Id.Type,
                       VLim=self.Lim, Out=Out, margin=1.e-9,
                       Multi=self.dgeom['Multi'], Ind=Ind)
        args = [self.Poly, self.dgeom['P1Min'][0], self.dgeom['P1Max'][0],
                self.dgeom['P2Min'][1], self.dgeom['P2Max'][1], res]
        pts, dS, ind, reseff = _comp._Ves_get_sampleS(*args, **kwdargs)
        return pts, dS, ind, reseff

    def get_sampleV(self, res, DV=None, resMode='abs', ind=None, Out='(X,Y,Z)'):
        """ Sample, with resolution res, the volume defined by DV or ind """

        args = [self.Poly, self.dgeom['P1Min'][0], self.dgeom['P1Max'][0],
                self.dgeom['P2Min'][1], self.dgeom['P2Max'][1], res]
        kwdargs = dict(DV=res, dVMode=resMode, ind=ind, VType=self.Id.Type,
                      VLim=self.Lim, Out=Out, margin=1.e-9)
        pts, dV, ind, reseff = _comp._Ves_get_sampleV(*args, **kwdargs)
        return pts, dV, ind, reseff


    def plot(self, lax=None, proj='All', Elt='PIBsBvV',
             dP=None, dI=_def.TorId, dBs=_def.TorBsd, dBv=_def.TorBvd,
             dVect=_def.TorVind, dIHor=_def.TorITord, dBsHor=_def.TorBsTord,
             dBvHor=_def.TorBvTord, Lim=None, Nstep=_def.TorNTheta,
             dLeg=_def.TorLegd, draw=True, fs=None, wintit='tofu', Test=True):
        """ Plot the polygon defining the vessel, in chosen projection

        Generic method for plotting the Ves object
        The projections to be plotted, the elements to plot can be specified
        Dictionaries of properties for each elements can also be specified
        If an ax is not provided a default one is created.

        Parameters
        ----------
        Lax :       list or plt.Axes
            The axes to be used for plotting
            Provide a list of 2 axes if Proj='All'
            If None a new figure with axes is created
        Proj :      str
            Flag specifying the kind of projection
                - 'Cross' : cross-section projection
                - 'Hor' : horizontal projection
                - 'All' : both
                - '3d' : a 3d matplotlib plot
        Elt  :      str
            Flag specifying which elements to plot
            Each capital letter corresponds to an element:
                * 'P': polygon
                * 'I': point used as a reference for impact parameters
                * 'Bs': (surfacic) center of mass
                * 'Bv': (volumic) center of mass for Tor type
                * 'V': vector pointing inward perpendicular to each segment
        dP :        dict / None
            Dict of properties for plotting the polygon
            Fed to plt.Axes.plot() or plt.plot_surface() if Proj='3d'
        dI :        dict / None
            Dict of properties for plotting point 'I' in Cross-section projection
        dIHor :     dict / None
            Dict of properties for plotting point 'I' in horizontal projection
        dBs :       dict / None
            Dict of properties for plotting point 'Bs' in Cross-section projection
        dBsHor :    dict / None
            Dict of properties for plotting point 'Bs' in horizontal projection
        dBv :       dict / None
            Dict of properties for plotting point 'Bv' in Cross-section projection
        dBvHor :    dict / None
            Dict of properties for plotting point 'Bv' in horizontal projection
        dVect :     dict / None
            Dict of properties for plotting point 'V' in cross-section projection
        dLeg :      dict / None
            Dict of properties for plotting the legend, fed to plt.legend()
            The legend is not plotted if None
        Lim :       list or tuple
            Array of a lower and upper limit of angle (rad.) or length for plotting the '3d' Proj
        Nstep :     int
            Number of points for sampling in ignorable coordinate (toroidal angle or length)
        draw :      bool
            Flag indicating whether the fig.canvas.draw() shall be called automatically
        a4 :        bool
            Flag indicating whether the figure should be plotted in a4 dimensions for printing
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        La          list / plt.Axes
            Handles of the axes used for plotting (list if several axes where used)

        """
        kwdargs = locals()
        lout = ['self']
        lrepl = [('lax','Lax'), ('proj','Proj'),
                 ('dP','Pdict'),('dI','Idict'),('dBs','Bsdict'),
                 ('dBv','Bvdict'),('dVect','Vdict'),('dIHor','IdictHor'),
                 ('dBsHor','BsdictHor'),('dBvHor','BvdictHor'),
                 ('dLeg','LegDict')]
        for k in lout:
            del kwdargs[k]
        for k in lrepl:
            kwdargs[k[1]] = kwdargs[k[0]]
            del kwdargs[k[0]]
        return _plot.Ves_plot(self, **kwdargs)


    def plot_sino(self, Proj='Cross', ax=None, Ang=_def.LOSImpAng,
                  AngUnit=_def.LOSImpAngUnit, Sketch=True, Pdict=None,
                  LegDict=_def.TorLegd, draw=True, fs=None, wintit='tofu',
                  Test=True):
        """ Plot the sinogram of the vessel polygon, by computing its envelopp in a cross-section, can also plot a 3D version of it

        The envelop of the polygon is computed using self.Sino_RefPt as a reference point in projection space, and plotted using the provided dictionary of properties.
        Optionaly a smal sketch can be included illustrating how the angle and the impact parameters are defined (if the axes is not provided).

        Parameters
        ----------
        Proj :      str
            Flag indicating whether to plot a classic sinogram ('Cross') from the vessel cross-section (assuming 2D), or an extended 3D version '3d' of it with additional angle
        ax   :      None or plt.Axes
            The axes on which the plot should be done, if None a new figure and axes is created
        Ang  :      str
            Flag indicating which angle to use for the impact parameter, the angle of the line itself (xi) or of its impact parameter (theta)
        AngUnit :   str
            Flag for the angle units to be displayed, 'rad' for radians or 'deg' for degrees
        Sketch :    bool
            Flag indicating whether a small skecth showing the definitions of angles 'theta' and 'xi' should be included or not
        Pdict :     dict
            Dictionary of properties used for plotting the polygon envelopp, fed to plt.plot() if Proj='Cross' and to plt.plot_surface() if Proj='3d'
        LegDict :   None or dict
            Dictionary of properties used for plotting the legend, fed to plt.legend(), the legend is not plotted if None
        draw :      bool
            Flag indicating whether the fig.canvas.draw() shall be called automatically
        a4 :        bool
            Flag indicating whether the figure should be plotted in a4 dimensions for printing
        Test :      bool
            Flag indicating whether the inputs shall be tested for conformity

        Returns
        -------
        ax :        plt.Axes
            The axes used to plot

        """
        if Test:
            msg = "The impact parameters must be set ! (self.set_dsino())"
            assert not self.dsino['RefPt'] is None, msg
            msg = "Arg Proj must be in ['Cross','3d'] !"
            assert Proj in ['Cross','3d'], msg
        if Proj=='Cross':
            Pdict = _def.TorPFilld if Pdict is None else Pdict
            ax = _plot.Plot_Impact_PolProjPoly(self, ax=ax, Ang=Ang,
                                               AngUnit=AngUnit, Sketch=Sketch,
                                               Leg=self.Id.NameLTX, Pdict=Pdict,
                                               dLeg=LegDict, draw=False,
                                               fs=fs, wintit=wintit, Test=Test)
        else:
            Pdict = _def.TorP3DFilld if Pdict is None else Pdict
            ax = _plot.Plot_Impact_3DPoly(self, ax=ax, Ang=Ang, AngUnit=AngUnit,
                                          Pdict=Pdict, dLeg=LegDict, draw=False,
                                          fs=fs, wintit=wintit, Test=Test)
        if draw:
            ax.figure.canvas.draw()
        return ax



"""
###############################################################################
###############################################################################
                      Effective Struct subclasses
###############################################################################
"""

class Ves(Struct, color='k'):

    def __init__(self, Poly=None, Type=None, Lim=None,
                 Id=None, Name=None, Exp=None, shot=None,
                 sino_RefPt=None, sino_nP=_def.TorNP,
                 Clock=False, arrayorder='C', fromdict=None,
                 SavePath=os.path.abspath('./'),
                 SavePath_Include=tfpf.defInclude, color=None):
        kwdargs = locals()
        del kwdargs['self'], kwdargs['__class__']
        super(Ves,self).__init__(mobile=False, **kwdargs)


    @staticmethod
    def _checkformat_inputs_dgeom(Poly=None, Lim=None, mobile=False,
                                  Type=None, Clock=False, arrayorder=None):
        kwdargs = locals()
        out = Struct._checkformat_inputs_dgeom(**kwdargs)
        Poly, Lim, Type, arrayorder = out
        msg = "Ves instances cannot be mobile !"
        assert mobile is False, msg
        msg = "There cannot be Lim if Type='Tor' !"
        if Type=='Tor':
            assert Lim.size==0, msg
        return out

    ######
    # Overloading of asbtract methods

    def isMobile(self):
        assert self._dgeom['mobile'] is False
        return False


class PFC(Struct, color=(0.8,0.8,0.8,0.8)):

    def move(self):
        """ To be overriden at object-level after instance creation

        To do so:
            1/ create the instance:
                >> S = tfg.Struct('test', poly, Exp='Test')
            2/ Define a moving function f taking the instance as first argument
                >> def f(self, Delta=1.):
                       Polynew = self.Poly
                       Polynew[0,:] = Polynew[0,:] + Delta
                       self._set_geom(Polynew, Lim=self.Lim)
            3/ Bound your custom function to the self.move() method
               using types.MethodType() found in the types module
                >> import types
                >> S.move = types.MethodType(f, S)

            See the following page for info and details on method-patching:
            https://tryolabs.com/blog/2013/07/05/run-time-method-patching-python/
        """
        print(self.move.__doc__)


    def get_sampleV(self, *args, **kwdargs):
        msg = "class cannot use get_sampleV() method !"
        raise Exception(msg)

    ######
    # Overloading of asbtract methods

    def isMobile(self):
        super(PFC, self).isMobile()


class CoilPF(Ves, color='r'):

    def __init_subclass__(cls, **kwdargs):
        super().__init_subclass__(**kwdargs)

    def __init__(self, nturns=None, superconducting=None, **kwdargs):
        super().__init__(**kwdargs)

    def _reset(self):
        super()._reset()
        self._dmag = dict.fromkeys(self._get_keys_dmag())
        self._dmag['nI'] = 0

    ###########
    # Get largs
    ###########

    @staticmethod
    def _get_largs_dmag():
        largs = ['nturns','superconducting']
        return largs

    ###########
    # Get check and format inputs
    ###########

    @staticmethod
    def _checkformat_inputs_dmag(nturns=None, superconducting=None):
        C0 = nturns is None
        C1 = type(nturns) in [int,float,np.int64,np.float64] and nturns>0
        assert C0 or C1
        if C1:
            nturns = int(nturns)
        C0 = superconducting is None
        C1 = type(superconducting) is bool
        assert C0 or C1
        return nturns

    ###########
    # Get keys of dictionnaries
    ###########

    @staticmethod
    def _get_keys_dmag():
        lk = ['nturns','superconducting','I','nI']
        return lk

    ###########
    # _init
    ###########

    def _init(self, nturns=None, superconducting=None, **kwdargs):
        super()._init(**kwdargs)
        self.set_dmag(nturns=nturns, superconducting=superconducting)


    ###########
    # set dictionaries
    ###########

    def set_dmag(self, superconducting=None, nturns=0):
        nturns = self._checkformat_inputs_dmag(nturns=nturns,
                                                superconducting=superconducting)
        self._dmag.update({'superconducting':superconducting,
                           'nturns':nturns})

    ###########
    # strip dictionaries
    ###########

    def _strip_dmag(self, lkeep=['nturns','superconducting']):
        utils.ToFuObject._strip_dict(self._dmag, lkeep=lkeep)
        self._dmag['nI'] = 0

    ###########
    # rebuild dictionaries
    ###########

    def _rebuild_dmag(self, lkeep=['nturns','superconducting']):
        reset = utils.ToFuObject._test_Rebuild(self._dmag, lkeep=lkeep)
        if reset:
            utils.ToFuObject._check_Fields4Rebuild(self._dmag,
                                                   lkeep=lkeep, dname='dmag')
            self.set_dmag(nturns=self.nturns,
                          superconducting=self.dmag['superconducting'])

    ###########
    # _strip and get/from dict
    ###########

    @classmethod
    def _strip_init(cls):
        cls._dstrip['allowed'] = [0,1,2]
        nMax = max(cls._dstrip['allowed'])
        doc = """
                 1: Remove dsino and dmag expendables
                 2: Remove also dgeom, dphys and dmisc expendables"""
        doc = utils.ToFuObjectBase.strip.__doc__.format(doc,nMax)
        cls.strip.__doc__ = doc

    def strip(self, strip=0):
        super().strip(strip=strip)

    def _strip(self, strip=0):
        out = super()._strip(strip=strip)
        if strip==0:
            self._rebuild_dmag()
        else:
            self._strip_dmag()
        return out

    def _to_dict(self):
        dout = super()._to_dict()
        dout.update({'dmag':{'dict':self.dmag, 'lexcept':None}})
        return dout

    def _from_dict(self, fd):
        super()._from_dict(fd)
        self._dmag.update(**fd['dmag'])


    ###########
    # Properties
    ###########

    @property
    def dmag(self):
        return self._dmag

    @property
    def nturns(self):
        return self._dmag['nturns']

    @property
    def I(self):
        return self._dmag['I']

    ###########
    # public methods
    ###########

    def set_I(self, I=None):
        """ Set the current circulating on the coil (A) """
        C0 = I is None
        C1 = type(I) in [int,float,np.int64,np.float64]
        C2 = type(I) in [list,tuple,np.ndarray]
        msg = "Arg I must be None, a float or an 1D np.ndarray !"
        assert C0 or C1 or C2, msg
        if C1:
            I = np.array([I],dtype=float)
        elif C2:
            I = np.asarray(I,dtype=float).ravel()
        self._dmag['I'] = I
        if C0:
            self._dmag['nI'] = 0
        else:
            self._dmag['nI'] = I.size


class CoilCS(CoilPF, color='r'): pass



"""
###############################################################################
###############################################################################
                        Overall Config object
###############################################################################
"""

class Config(utils.ToFuObject):


    # Special dict subclass with attr-like value access


    # Fixed (class-wise) dictionary of default properties
    _ddef = {'Id':{'shot':0},
             'dstruct':{'order':['Ves','PFC','CoilPF','CoilCS']}}

    def __init__(self, lStruct=None,
                 Id=None, Name=None, Exp=None, shot=None,
                 SavePath=os.path.abspath('./'),
                 SavePath_Include=tfpf.defInclude,
                 fromdict=None):

        kwdargs = locals()
        del kwdargs['self']
        super().__init__(**kwdargs)

    def _reset(self):
        super()._reset()
        self._dstruct = dict.fromkeys(self._get_keys_dstruct())

    @classmethod
    def _checkformat_inputs_Id(cls, Id=None, Name=None,
                               shot=None, **kwdargs):
        if Id is not None:
            assert isinstance(Id,utils.ID)
            Name, shot = Id.Name, Id.shot
        assert type(Name) is str
        if shot is None:
            shot = cls._ddef['Id']['shot']
        assert type(shot) is int
        kwdargs.update({'Name':Name, 'shot':shot})
        return kwdargs

    ###########
    # Get largs
    ###########

    @staticmethod
    def _get_largs_dstruct():
        largs = ['lStruct']
        return largs

    ###########
    # Get check and format inputs
    ###########

    @staticmethod
    def _checkformat_inputs_dstruct(lStruct=None):
        msg = "Arg lStruct must be"
        msg += " a tofu.geom.Struct subclass or a list of such !"
        msg += "\nValid subclasses include:"
        lsub = ['Ves','PFC','CoilPF','CoilCS']
        for ss in lsub:
            msg += "\n    - tf.geom.{0}".format(ss)
        assert type(lStruct) is not None, msg
        C0 = isinstance(lStruct,list) or isinstance(lStruct,tuple)
        C1 = issubclass(lStruct.__class__,Struct)
        assert C0 or C1, msg
        if C0:
            Ci = [issubclass(ss.__class__,Struct) for ss in lStruct]
            assert all(Ci), msg
            lStruct = list(lStruct)
        else:
            lStruct = [lStruct]
        C = all([ss.Id.Exp==lStruct[0].Id.Exp for ss in lStruct])
        msg = "All Struct objects must have the same Exp !"
        msg += "\nCurrently we have:"
        ls = ["{0}: {1}".format(ss.Id.SaveName, ss.Id.Exp)for ss in lStruct]
        msg += "\n    - " + "\n    - ".join(ls)
        assert C, msg
        return lStruct

    ###########
    # Get keys of dictionnaries
    ###########

    @staticmethod
    def _get_keys_dstruct():
        lk = ['dStruct','dvisible',
              'nStruct','lorder','lCls']
        return lk

    ###########
    # _init
    ###########

    def _init(self, lStruct=None, **kwdargs):
        largs = self._get_largs_dstruct()
        kwdstruct = self._extract_kwdargs(locals(), largs)
        self._set_dstruct(**kwdstruct)

    ###########
    # set dictionaries
    ###########

    def _set_dstruct(self, lStruct=None):
        lStruct = self._checkformat_inputs_dstruct(lStruct=lStruct)
        # Make sure to kill the link to the mutable being provided
        nStruct = len(lStruct)
        # Get extra info
        lCls = list(set([ss.Id.Cls for ss in lStruct]))
        lorder = [ss.Id.SaveName_Conv(Cls=ss.Id.Cls,
                                      Name=ss.Id.Name,
                                      include=['Cls','Name']) for ss in lStruct]

        msg = "There is an ambiguity in the names :"
        msg += "\n    - " + "\n    - ".join(lorder)
        msg += "\n => Please clarify (choose unique Cls/Names)"
        assert len(list(set(lorder)))==nStruct, msg

        self._dstruct = {'dStruct':dict([(k,{}) for k in lCls]),
                         'dvisible':dict([(k,{}) for k in lCls])}
        for k in lCls:
            lk = [ss for ss in lStruct if ss.Id.Cls==k]
            for ss in lk:
                self._dstruct['dStruct'][k].update({ss.Id.Name:ss.copy()})
                self._dstruct['dvisible'][k].update({ss.Id.Name:True})

        self._dstruct.update({'nStruct':nStruct,
                              'lorder':lorder, 'lCls':lCls})
        self._dstruct_dynamicattr()

    def _dstruct_dynamicattr(self):
        # get (key, val) pairs
        def set_vis(obj, k0, k1, val):
            assert type(val) is bool
            obj._dstruct['dvisible'][k0][k1] = val
        def get_vis(obj, k0, k1):
            return obj._dstruct['dvisible'][k0][k1]
        for k in self._ddef['dstruct']['order']:
            if k in self._dstruct['lCls']:
                # Find a way to programmatically add dynamic properties to the
                # instances , like visible
                # In the meantime use a simple functions
                for kk in self._dstruct['dStruct'][k].keys():
                    setattr(self._dstruct['dStruct'][k][kk],
                            'set_visible',
                            lambda vis, k0=k, k1=kk: set_vis(self, k0, k1, vis))
                    setattr(self._dstruct['dStruct'][k][kk],
                            'get_visible',
                            lambda k0=k, k1=kk: get_vis(self, k0, k1))
                dd = utils.dictattr(self._dstruct['dStruct'][k])
                setattr(self, k, dd)
            elif hasattr(self,k):
                exec('del self.{0}'.format(k))

    ###########
    # strip dictionaries
    ###########

    def _strip_dstruct(self, lkeep=['dStruct'], strip=1, force=False):
        if strip in [1,2]:
            for k in self._dstruct['lCls']:
                for kk, v  in self._dstruct['dStruct'][k].items():
                    self._dstruct['dStruct'][k][kk].strip(strip=strip)
        elif strip>2:
            for k in self._dstruct['lCls']:
                for kk, v  in self._dstruct['dStruct'][k].items():
                    pathfile = os.path.join(v.Id.SavePath, v.Id.SaveName)
                    # --- Check !
                    lf = os.listdir(v.Id.SavePath)
                    lf = [ff for ff in lf
                          if all([ss in ff for ss in [v.Id.SaveName,'.npz']])]
                    exist = len(lf)==1
                    # ----------
                    if not exist:
                        msg = """BEWARE:
                            You are about to delete the Struct objects
                            Only the path/name to saved objects will be kept

                            But it appears that the following object has no
                            saved file were specified (obj.Id.SavePath)
                            Thus it won't be possible to retrieve it
                            (unless available in the current console:"""
                        msg += "\n    - {0}".format(pathfile+'.npz')
                        if force:
                            warning.warn(msg)
                        else:
                            raise Exception(msg)
                    self._dstruct['dStruct'][k][kk] = pathfile
        utils.ToFuObject._strip_dict(self._dstruct, lkeep=lkeep)

    ###########
    # rebuild dictionaries
    ###########

    def _rebuild_dstruct(self, lkeep=['dStruct']):
        reset = utils.ToFuObject._test_Rebuild(self._dstruct, lkeep=lkeep)
        if reset:
            utils.ToFuObject._check_Fields4Rebuild(self._dstruct,
                                                   lkeep=lkeep, dname='dstruct')
            self.set_dstruct(lStruct=self.lStruct)

    ###########
    # _strip and get/from dict
    ###########

    @classmethod
    def _strip_init(cls):
        cls._dstrip['allowed'] = [0,1,2]
        nMax = max(cls._dstrip['allowed'])
        doc = """
                 1: apply strip(1) to objects in self.lStruct
                 2: apply strip(2) to objects in self.lStruct
                 3: replace objects in self.lStruct by their SavePath+SaveName"""
        doc = utils.ToFuObjectBase.strip.__doc__.format(doc,nMax)
        cls.strip.__doc__ = doc

    def strip(self, strip=0):
        super().strip(strip=strip)

    ### To be revised !!!
    def _strip(self, strip=0, force=False):
        if strip==0:
            self._rebuild_dstruct()
        else:
            self._strip_dstruct(strip=strip, force=force)
        return out

    def _to_dict(self):
        dout = {'dstruct':{'dict':self.dstruct, 'lexcept':None}}
        return dout

    def _from_dict(self, fd):
        self._dstruct.update(**fd['dstruct'])


    ###########
    # Properties
    ###########

    @property
    def dstruct(self):
       return self._dstruct
    @property
    def lStruct(self):
        """ Return the list of Struct that was used for creation

        As tofu objects of SavePath+SaveNames (according to strip status)
        """
        lStruct = []
        for k in self._dstruct['lorder']:
            k0, k1 = k.split('_')
            lStruct.append(self._dstruct['dStruct'][k0][k1])
        return lStruct
    @property
    def visible(self):
        """ Return the array of visible bool (same order as lStruct) """
        vis = np.ones((self._dstruct['nStruct'],),dtype=bool)
        ii = 0
        for k in self._dstruct['lorder']:
            k0, k1 = k.split('_')
            vis[ii] = self._dstruct['dvisible'][k0][k1]
            ii += 1
        return vis
    @property
    def color(self):
        """ Return the array of rgba colors (same order as lStruct) """
        col = np.full((self._dstruct['nStruct'],4), np.nan)
        ii = 0
        for k in self._dstruct['lorder']:
            k0, k1 = k.split('_')
            col[ii,:] = self._dstruct['dStruct'][k0][k1].color
            ii += 1
        return col

    ###########
    # public methods
    ###########

    def get_description(self, verb=False, max_columns=100, width=1000):
        """ Summary description of the object content as a pandas DataFrame """
        # Make sure the data is accessible
        msg = "The data is not accessible because self.strip(2) was used !"
        assert self._dstrip['strip']<2, msg

        # Build the list
        d = self._dstruct['dStruct']
        data = []
        for k in self._ddef['dstruct']['order']:
            if k not in d.keys():
                continue
            for kk in d[k].keys():
                tu = (k,
                      self._dstruct['dStruct'][k][kk]._Id._dall['SaveName'],
                      self._dstruct['dStruct'][k][kk]._dgeom['nP'],
                      self._dstruct['dStruct'][k][kk]._dgeom['nLim'],
                      self._dstruct['dStruct'][k][kk]._dgeom['mobile'],
                      self._dstruct['dvisible'][k][kk],
                      self._dstruct['dStruct'][k][kk]._dmisc['color'])
                data.append(tu)

        # Build the pandas DataFrame
        col = ['class', 'SaveName', 'nP', 'nLim',
               'mobile', 'visible', 'color']
        df = pd.DataFrame(data, columns=col)
        pd.set_option('display.max_columns',max_columns)
        pd.set_option('display.width',width)

        if verb:
            print(df)
        return df

    def plot(self, lax=None, proj='All', Elt='P', dLeg=_def.TorLegd,
             draw=True, fs=None, wintit=None, Test=True):
        kwdargs = locals()
        del kwdargs['self'], kwdargs['lax']
        del kwdargs['dLeg'], kwdargs['draw']
        for k in self._dstruct['lCls']:
            for v in self.dStruct[k]:
                lax = v.plot(lax=lax, dLeg=None, draw=False, **kwdargs)
        if dLeg is not None:
            lax[0].legend(**dLeg)
        if draw:
            lax[0].figure.canvas.draw()
        return lax

    def plot_sino(self, **kwdargs):
        pass


"""
###############################################################################
###############################################################################
                        Rays-derived classes and functions
###############################################################################
"""


class Rays(object):
    """ Parent class of rays (ray-tracing), LOS, LOSCam1D and LOSCam2D

    Focused on optimizing the computation time for many rays.

    Each ray is defined by a starting point (D) and a unit vector(u).
    If a vessel (Ves) and structural elements (LStruct) are provided,
    the intersection points are automatically computed.

    Methods for plootting, computing synthetic signal are provided.

    Parameters
    ----------
    Id :            str  / :class:`~tofu.pathfile.ID`
        A name string or a :class:`~tofu.pathfile.ID` to identify this instance,
        if a string is provided, it is fed to :class:`~tofu.pathfile.ID`
    Du :            iterable
        Iterable of len=2, containing 2 np.ndarrays represnting, for N rays:
            - Ds: a (3,N) array of the (X,Y,Z) coordinates of starting points
            - us: a (3,N) array of the (X,Y,Z) coordinates of the unit vectors
    Ves :           None / :class:`~tofu.geom.Ves`
        A :class:`~tofu.geom.Ves` instance to be associated to the rays
    LStruct:        None / :class:`~tofu.geom.Struct` / list
        A :class:`~tofu.geom.Struct` instance or list of such, for obstructions
    Sino_RefPt :    None / np.ndarray
        Iterable of len=2 with the coordinates of the sinogram reference point
            - (R,Z) coordinates if the vessel is of Type 'Tor'
            - (Y,Z) coordinates if the vessel is of Type 'Lin'
    Type :          None
        (not used in the current version)
    Exp        :    None / str
        Experiment to which the LOS belongs:
            - if both Exp and Ves are provided: Exp==Ves.Id.Exp
            - if Ves is provided but not Exp: Ves.Id.Exp is used
    Diag       :    None / str
        Diagnostic to which the LOS belongs
    shot       :    None / int
        Shot number from which this LOS is valid
    SavePath :      None / str
        If provided, default saving path of the object

    """

    def __init__(self, Id=None, Du=None, Ves=None, LStruct=None,
                 Etendues=None, Surfaces=None, Sino_RefPt=None,
                 fromdict=None, Exp=None, Diag=None, shot=0, dchans=None,
                 SavePath=os.path.abspath('./'), plotdebug=True):

        self._Done = False
        if fromdict is None:
            self._check_inputs(Id=Id, Du=Du, Ves=Ves, LStruct=LStruct,
                               Sino_RefPt=Sino_RefPt, Exp=Exp, Diag=Diag,
                               shot=shot, dchans=dchans, SavePath=SavePath)
            if Ves is not None:
                Exp = Ves.Id.Exp if Exp is None else Exp
            self._set_Id(Id, Exp=Exp, Diag=Diag, shot=shot, SavePath=SavePath)
            self._set_Ves(Ves, LStruct=LStruct, Du=Du, dchans=dchans,
                          plotdebug=plotdebug)
            self.set_Etendues(Entendues)
            self.set_Surfaces(Surfaces)
            self.set_sino(RefPt=Sino_RefPt)
        else:
            self._fromdict(fromdict)
        self._Done = True

    def _fromdict(self, fd, lvl=0):
        allowed = [0,1]
        assert lvl in [-1]+allowed
        lvl = allowed[lvl]

        if lvl==0:
            _Rays_check_fromdict(fd)
            self._Id = tfpf.ID(fromdict=fd['Id'])
            self._dchans = fd['dchans']
            if fd['Ves'] is None:
                self._Ves = None
            else:
                self._Ves = Ves(fromdict=fd['Ves'])
            if fd['LStruct'] is None:
                self._LStruct = None
            else:
                self._LStruct = [Struct(fromdict=ds) for ds in fd['LStruct']]
            self._geom = fd['geom']
            self._sino = fd['sino']
            if 'extra' in fd.keys():
                self._extra = fd['extra']
            else:
                self._extra = {'Etendues':None, 'Surfaces':None}
        elif lvl==1:
            fd = utils.reshapedict(fd, )
            self._Id = tfpf.ID(fromdict=fd['Id'])
            self._dchans = fd['dchans']

            # Get Ves and LStruct here to get VType

            geom = fd['geom']
            geom['PIn'] = geom['D'] + geom['kPIn'][np.newaxis,:]*geom['u']
            geom['POut'] = geom['D'] + geom['kPOut'][np.newaxis,:]*geom['u']
            geom['PRMin'] = geom['D'] + geom['kPRMin'][np.newaxis,:]*geom['u']
            geom['RMin'] = np.hypot(geom['PRMin'][0,:],geom['PRMin'][1,:])
            self._geom = geom

            self._extra = fd['extra']

    def _todict(self, lvl=0):
        allowed = [0,1]
        assert lvl in [-1]+allowed
        lvl = allowed[lvl]

        if lvl==0:
            out = {'Id':self.Id._todict(),
                   'dchans':self.dchans,
                   'geom':self.geom,
                   'extra': self._extra,
                   'sino':self.sino}
            out['Ves'] = None if self.Ves is None else self.Ves._todict()
            if self.LStruct is None:
                out['LStruct'] = None
            else:
                out['LStruct'] = [ss._todict() for ss in self.LStruct]
        elif lvl==1:
            Id = self.Id._todict(-1)
            dchans = utils.flattendict(self.dchans)
            lexcept = ['PIn','POut','PRMin','RMin']
            geom = utils.flattendict(self.geom, lexcept=lexcept)
            extra = utils.flattendict(self._extra)
            out = utils.get_todictfields([Id,   dchans,  geom],
                                         ['Id','dchans','geom'])
        return out

    @property
    def Id(self):
        return self._Id
    @property
    def geom(self):
        return self._geom
    @property
    def nRays(self):
        return self.geom['nRays']
    @property
    def D(self):
        return self.geom['D']
    @property
    def u(self):
        return self.geom['u']
    @property
    def PIn(self):
        return self.geom['PIn']
    @property
    def POut(self):
        return self.geom['POut']
    @property
    def dchans(self):
        return self._dchans
    @property
    def Ves(self):
        return self._Ves
    @property
    def LStruct(self):
        return self._LStruct
    @property
    def Etendues(self):
        return self._extra['Etendues']
    @property
    def Surfaces(self):
        return self._extra['Surfaces']
    @property
    def sino(self):
        return self._sino

    def _check_inputs(self, Id=None, Du=None, Ves=None, LStruct=None,
                      Sino_RefPt=None, Exp=None, shot=None, Diag=None,
                      SavePath=None, ind=None,
                      dchans=None, fromdict=None):
        _Rays_check_inputs(Id=Id, Du=Du, Vess=Ves, LStruct=LStruct,
                          Sino_RefPt=Sino_RefPt, Exp=Exp, shot=shot, ind=ind,
                          Diag=Diag, SavePath=SavePath,
                          dchans=dchans, fromdict=fromdict)

    def _set_Id(self, Val,
                Exp=None, Diag=None, shot=None,
                SavePath=os.path.abspath('./')):
        dd = {'Exp':Exp, 'shot':shot, 'Diag':Diag, 'SavePath':SavePath}
        if self._Done:
            tfpf._get_FromItself(self.Id, dd)
        tfpf._check_NotNone({'Id':Val})
        self._check_inputs(Id=Val)
        if type(Val) is str:
            Val = tfpf.ID(self.__class__, Val, **dd)
        self._Id = Val

    def _set_Ves(self, Ves=None, LStruct=None, Du=None, dchans=None,
                 plotdebug=True):
        self._check_inputs(Ves=Ves, Exp=self.Id.Exp, LStruct=LStruct, Du=Du)
        LObj = []
        if not Ves is None:
            LObj.append(Ves.Id)
        if not LStruct is None:
            LStruct = [LStruct] if type(LStruct) is Struct else LStruct
            LObj += [ss.Id for ss in LStruct]
        if len(LObj)>0:
            self.Id.set_LObj(LObj)
        self._Ves = Ves
        self._LStruct = LStruct
        Du = Du if Du is not None else (self.D,self.u)
        self._set_geom(Du, dchans=dchans, plotdebug=plotdebug)

    def _set_geom(self, Du, dchans=None,
                  plotdebug=True, fs=None, wintit='tofu', draw=True):
        """ Compute all geometrical attributes

        Du is a tuple with D (start points) and u (unit vectors)
        D and u can be (3,) arrays or (3,N) arrays in (X,Y,Z) coordinates

        """

        # Check and format inputs
        tfpf._check_NotNone({'Du':Du})
        self._check_inputs(Du=Du, dchans=dchans)

        # D = start point
        # u = unit vector
        D, u = np.asarray(Du[0]), np.asarray(Du[1])
        msg = "D and u must be arrays of (X,Y,Z) coordinates !"
        assert D.size%3==0 and u.size%3==0, msg
        nRays = int(max(D.size/3, u.size/3))
        if D.ndim==2:
            if D.shape[1]==3 and not D.shape[0]==3:
                D = D.T
        else:
            D = D.reshape((3,1))
        assert D.shape[1] in [1,nRays]
        if D.shape[1]<nRays:
            D = np.repeat(D, nRays, axis=1)
        if u.ndim==2:
            if u.shape[1]==3 and not u.shape[0]==3:
                u = u.T
        else:
            assert u.size==3
            u = u.reshape((3,1))
        assert u.shape[1] in [1,nRays]
        if u.shape[1]<nRays:
            u = np.repeat(u, nRays, axis=1)
        u = u/np.sqrt(np.sum(u**2,axis=0))
        D = np.ascontiguousarray(D)
        u = np.ascontiguousarray(u)

        # Prepare the output
        kPIn, kPOut = np.full((nRays,),np.nan), np.full((nRays,),np.nan)
        PIn, POut = np.full((3,nRays),np.nan), np.full((3,nRays),np.nan)
        VPerpIn, VPerpOut = np.full((3,nRays),np.nan), np.full((3,nRays),np.nan)
        IndIn, IndOut = np.full((nRays,),np.nan), np.full((nRays,),np.nan)

        # Only compute is Ves was provided
        if self.Ves is not None:
            if self.LStruct is not None:
                lSPoly = [ss.Poly for ss in self.LStruct]
                lSLim = [ss.Lim for ss in self.LStruct]
                lSVIn = [ss.geom['VIn'] for ss in self.LStruct]
            else:
                lSPoly, lSLim, lSVIn = None, None, None

            kargs = dict(RMin=None, Forbid=True, EpsUz=1.e-6, EpsVz=1.e-9,
                         EpsA=1.e-9, EpsB=1.e-9, EpsPlane=1.e-9, Test=True)

            #####################
            # call the dedicated function (Laura)
            out = _GG.LOS_Calc_PInOut_VesStruct(D, u, self.Ves.Poly,
                                                self.Ves.geom['VIn'],
                                                Lim=self.Ves.Lim, LSPoly=lSPoly,
                                                LSLim=lSLim, LSVIn=lSVIn,
                                                VType=self.Ves.Type, **kargs)
            ######################

            PIn, POut, kPIn, kPOut, VPerpIn, VPerpOut, IndIn, IndOut = out
            ind = (np.isnan(kPOut) | np.isinf(kPOut)
                   | np.any(np.isnan(POut),axis=0))
            kPOut[ind] = np.nan
            if np.any(ind):
                warnings.warn("Some LOS have no visibility inside the vessel !")
                if plotdebug:
                    _plot._LOS_calc_InOutPolProj_Debug(self.Ves, D[:,ind], u[:,ind],
                                                       PIn[:,ind], POut[:,ind],
                                                       fs=fs, wintit=wintit,
                                                       draw=draw)
            ind = np.isnan(kPIn)
            PIn[:,ind], kPIn[ind] = D[:,ind], 0.

        PRMin, kRMin, RMin = _comp.LOS_PRMin(D, u, kPOut=kPOut, Eps=1.e-12)
        self._geom = {'D':D, 'u':u, 'nRays':nRays,
                      'PIn':PIn, 'POut':POut, 'kPIn':kPIn, 'kPOut':kPOut,
                      'VPerpIn':VPerpIn, 'VPerpOut':VPerpOut,
                      'IndIn':IndIn, 'IndOut':IndOut,
                      'PRMin':PRMin, 'kRMin':kRMin, 'RMin':RMin}

        # Get basics of 2D geometry
        if self.Id.Cls=='LOSCam2D':
            C = np.nanmean(D,axis=1)
            CD0 = D[:,:-1] - C[:,np.newaxis]
            CD1 = D[:,1:] - C[:,np.newaxis]
            cross = np.array([CD1[1,1:]*CD0[2,:-1]-CD1[2,1:]*CD0[1,:-1],
                              CD1[2,1:]*CD0[0,:-1]-CD1[0,1:]*CD0[2,:-1],
                              CD1[0,1:]*CD0[1,:-1]-CD1[1,1:]*CD0[0,:-1]])
            crossn2 = np.sum(cross**2,axis=0)
            if np.all(np.abs(crossn2)<1.e-12):
                msg = "Is %s really a 2D camera ? (LOS aligned?)"%self.Id.Name
                warnings.warn(msg)
            cross = cross[:,np.nanargmax(crossn2)]
            cross = cross / np.linalg.norm(cross)
            nIn = cross if np.sum(cross*np.nanmean(u,axis=1))>0. else -cross
            # Find most relevant e1 (for pixels alignment), without a priori info
            D0D = D-D[:,0][:,np.newaxis]
            dist = np.sqrt(np.sum(D0D**2,axis=0))
            dd = np.min(dist[1:])
            e1 = (D[:,1]-D[:,0])/np.linalg.norm(D[:,1]-D[:,0])
            cross = np.sqrt((D0D[1,:]*e1[2]-D0D[2,:]*e1[1])**2
                            + (D0D[2,:]*e1[0]-D0D[0,:]*e1[2])**2
                            + (D0D[0,:]*e1[1]-D0D[1,:]*e1[0])**2)
            D0D = D0D[:,cross<dd/3.]
            sca = np.sum(D0D*e1[:,np.newaxis],axis=0)
            e1 = D0D[:,np.argmax(np.abs(sca))]
            nIn, e1, e2 = utils.get_nIne1e2(C, nIn=nIn, e1=e1)
            if np.abs(np.abs(nIn[2])-1.)>1.e-12:
                if np.abs(e1[2])>np.abs(e2[2]):
                    e1, e2 = e2, e1
            e2 = e2 if e2[2]>0. else -e2
            self._geom.update({'C':C, 'nIn':nIn, 'e1':e1, 'e2':e2})

        if dchans is None:
            self._dchans = dchans
        else:
            lK = list(dchans.keys())
            self._dchans = dict([(kk,np.asarray(dchans[kk]).ravel()) for kk in lK])


    def set_Etendues(self, E=None):
        C0 = E is None
        C1 = type(E) in [int,float,np.int64,np.float64]
        C2 = type(E) in [list,tuple,np.ndarray]
        msg = "Arg E must be None, a float or a np.ndarray !"
        assert C0 or C1 or C2, msg
        if not C0:
            if  C1:
                E = [E]
            E = np.asarray(E, dtype=float).ravel()
            msg = "E must be an iterable of size == 1 or self.nRays !"
            assert E.size in [1, self.nRays], msg
        self._extra['Etendues'] = E

    def set_Surfaces(self, S=None):
        C0 = S is None
        C1 = type(S) in [int,float,np.int64,np.float64]
        C2 = type(S) in [list,tuple,np.ndarray]
        msg = "Arg S must be None, a float or a np.ndarray !"
        assert C0 or C1 or C2, msg
        if not C0:
            if C1:
                S = [S]
            S = np.asarray(S, dtype=float).ravel()
            msg = "S must be an iterable of size == 1 or self.nRays !"
            assert S.size in [1, self.nRays], msg
        self._extra['Surfaces'] = S

    def set_sino(self, RefPt=None):
        self._check_inputs(Sino_RefPt=RefPt)
        if RefPt is None and self.Ves is None:
            self._sino = None
        else:
            if RefPt is None:
                RefPt = self.Ves.sino['RefPt']
            if RefPt is None:
                if self.Ves.Type=='Tor':
                    RefPt = self.Ves.geom['BaryV']
                else:
                    RefPt = self.Ves.geom['BaryS']
            RefPt = np.asarray(RefPt).ravel()
            if self.Ves is not None:
                self._Ves.set_sino(RefPt)
                VType = self.Ves.Type
            else:
                VType = 'Lin'
            kMax = np.copy(self.geom['kPOut'])
            kMax[np.isnan(kMax)] = np.inf
            try:
                out = _GG.LOS_sino(self.D, self.u, RefPt, kMax,
                                   Mode='LOS', VType=VType)
                Pt, kPt, r, Theta, p, theta, Phi = out
            except Exception as err:
                msg = "Could not compute sinogram !\n"
                msg += str(err)
                warnings.warn(msg)
                Pt, kPt, r = None, None, None
                Theta, p, theta, Phi = None, None, None, None
            self._sino = {'RefPt':RefPt, 'Pt':Pt, 'kPt':kPt, 'r':r,
                          'Theta':Theta, 'p':p, 'theta':theta, 'Phi':Phi}

    def select(self, key=None, val=None, touch=None, log='any', out=int):
        assert out in [int,bool]
        assert log in ['any','all','not']
        C = [key is None,touch is None]
        assert np.sum(C)>=1
        if np.sum(C)==2:
            ind = np.ones((self.nRays,),dtype=bool)
        else:
            if key is not None:
                assert type(key) is str and key in self._dchans.keys()
                ltypes = [str,int,float,np.int64,np.float64]
                C0 = type(val) in ltypes
                C1 = type(val) in [list,tuple,np.ndarray]
                assert C0 or C1
                if C0:
                    val = [val]
                else:
                    assert all([type(vv) in ltypes for vv in val])
                ind = np.vstack([self._dchans[key]==ii for ii in val])
                if log=='any':
                    ind = np.any(ind,axis=0)
                elif log=='all':
                    ind = np.all(ind,axis=0)
                else:
                    ind = ~np.any(ind,axis=0)
            elif touch is not None:
                VesOk, StructOk = self.Ves is not None, self.LStruct is not None
                StructNames = [ss.Id.Name for ss in self.LStruct] if StructOk else None
                ind = _comp.Rays_touch(VesOk, StructOk, self.geom['IndOut'],
                                       StructNames,touch=touch)
                ind = ~ind if log=='not' else ind
        if out is int:
            ind = ind.nonzero()[0]
        return ind

    def get_subset(self, indch=None):
        if indch is None:
            return self
        else:
            assert type(indch) is np.ndarray and indch.ndim==1
            assert indch.dtype in [np.int64,np.bool_]
            d = self._todict()
            d['Id']['Name'] = d['Id']['Name']+'-subset'
            d['dchans'] = dict([(vv,vv[indch]) for vv in d['chans'].keys()])

            # Geom
            for kk in d['geom']:
                C0 = type(d['geom'][kk]) is np.ndarray
                C1 = d['geom'][kk].ndim==1 and d['geom'][kk].size==self.nRays
                C2 = d['geom'][kk].ndim==2 and d['geom'][kk].shape[1]==self.nRays
                C3 = kk in ['C','nIn','e1','e2']
                if C0 and C1 and not C3:
                    d['geom'][kk] = d['geom'][kk][indch]
                elif C0 and C2 and not C3:
                    d['geom'][kk] = d['geom'][kk][:,indch]

            # Sino
            for kk in d['sino'].keys():
                if d['sino'][kk].ndim==2:
                    d['sino'][kk] = d['sino'][kk][:,indch]
                elif d['sino'][kk].ndim==1:
                    d['sino'][kk] = d['sino'][kk][indch]

            if 'Rays' in self.Id.Cls:
                c = tfg.Rays(fromdict=d)
            elif 'LOSCam1D' in self.Id.Cls:
                c = tfg.LOSCam1D(fromdict=d)
            elif 'LOSCam2D' in self.Id.Cls:
                c = tfg.LOSCam2D(fromdict=d)
            elif 'Cam1D' in self.Id.Cls:
                c = tfg.Cam1D(fromdict=d)
            elif 'Cam2D' in self.Id.Cls:
                c = tfg.Cam2D(fromdict=d)

        return c

    def _get_plotL(self, Lplot='Tot', Proj='All', ind=None, multi=False):
        self._check_inputs(ind=ind)
        if ind is not None:
            ind = np.asarray(ind)
            if ind.dtype in [bool,np.bool_]:
                ind = ind.nonzero()[0]
        else:
            ind = np.arange(0,self.nRays)
        if len(ind)>0:
            Ds, us = self.D[:,ind], self.u[:,ind]
            if len(ind)==1:
                Ds, us = Ds.reshape((3,1)), us.reshape((3,1))
            kPIn, kPOut = self.geom['kPIn'][ind], self.geom['kPOut'][ind]
            kRMin = self.geom['kRMin'][ind]
            pts = _comp.LOS_CrossProj(self.Ves.Type, Ds, us, kPIn, kPOut,
                                      kRMin, Proj=Proj,Lplot=Lplot,multi=multi)
        else:
            pts = None
        return pts

    def get_sample(self, dl, dlMode='abs', DL=None, method='sum', ind=None):
        """ Return a linear sampling of the LOS

        The LOS is sampled into a series a points and segments lengths
        The resolution (segments length) is <= dl
        The sampling can be done according to different methods
        It is possible to sample only a subset of the LOS

        Parameters
        ----------
        dl:     float
            Desired resolution
        dlMode: str
            Flag indicating dl should be understood as:
                - 'abs':    an absolute distance in meters
                - 'rel':    a relative distance (fraction of the LOS length)
        DL:     None / iterable
            The fraction [L1;L2] of the LOS that should be sampled, where
            L1 and L2 are distances from the starting point of the LOS (LOS.D)
        method: str
            Flag indicating which to use for sampling:
                - 'sum':    the LOS is sampled into N segments of equal length,
                            where N is the smallest int such that:
                                * segment length <= resolution(dl,dlMode)
                            The points returned are the center of each segment
                - 'simps':  the LOS is sampled into N segments of equal length,
                            where N is the smallest int such that:
                                * segment length <= resolution(dl,dlMode)
                                * N is even
                            The points returned are the egdes of each segment
                - 'romb':   the LOS is sampled into N segments of equal length,
                            where N is the smallest int such that:
                                * segment length <= resolution(dl,dlMode)
                                * N = 2^k + 1
                            The points returned are the egdes of each segment

        Returns
        -------
        Pts:    np.ndarray
            A (3,NP) array of NP points along the LOS in (X,Y,Z) coordinates
        kPts:   np.ndarray
            A (NP,) array of the points distances from the LOS starting point
        dl:     float
            The effective resolution (<= dl input), as an absolute distance

        """
        self._check_inputs(ind=ind)
        # Preformat ind
        if ind is None:
            ind = np.arange(0,self.nRays)
        if np.asarray(ind).dtype is bool:
            ind = ind.nonzero()[0]
        # Preformat DL
        if DL is None:
            DL = np.array([self.geom['kPIn'][ind],self.geom['kPOut'][ind]])
        elif np.asarray(DL).size==2:
            DL = np.tile(np.asarray(DL).ravel(),(len(ind),1)).T
        DL = np.ascontiguousarray(DL).astype(float)
        assert type(DL) is np.ndarray and DL.ndim==2
        assert DL.shape==(2,len(ind)), "Arg DL has wrong shape !"
        ii = DL[0,:]<self.geom['kPIn'][ind]
        DL[0,ii] = self.geom['kPIn'][ind][ii]
        ii = DL[0,:]>=self.geom['kPOut'][ind]
        DL[0,ii] = self.geom['kPOut'][ind][ii]
        ii = DL[1,:]>self.geom['kPOut'][ind]
        DL[1,ii] = self.geom['kPOut'][ind][ii]
        ii = DL[1,:]<=self.geom['kPIn'][ind]
        DL[1,ii] = self.geom['kPIn'][ind][ii]
        # Preformat Ds, us
        Ds, us = self.D[:,ind], self.u[:,ind]
        if len(ind)==1:
            Ds, us = Ds.reshape((3,1)), us.reshape((3,1))
        Ds, us = np.ascontiguousarray(Ds), np.ascontiguousarray(us)
        # Launch    # NB : find a way to exclude cases with DL[0,:]>=DL[1,:] !!
        Pts, kPts, dlr = _GG.LOS_get_sample(Ds, us, dl, DL,
                                            dLMode=dlMode, method=method)
        return Pts, kPts, dlr


    def get_kInkOut(self, lPoly, Lim=None):
        msg = "Arg lPoly must be a list or np.ndarray !"
        assert type(lPoly) in [list, np.ndarray], msg

        # Preformat input
        if type(lPoly) is np.ndarray:
            lPoly = [lPoly]
        lPoly = [np.ascontiguousarray(pp) for pp in lPoly]
        msg = "Arg lPoly must be a list of (2,N) or (N,2) np.ndarrays !"
        assert all([pp.ndim==2 and 2 in pp.shape for pp in lPoly]), msg
        nPoly = len(lPoly)
        for ii in range(0,nPoly):
            if lPoly[ii].shape[0]!=2:
                lPoly[ii] = lPoly[ii].T

        if Lim is None and self.Ves is not None and self.Ves.Type=='Lin':
            Lim = self.Ves.Lim

        # Prepare output
        kIn = np.full((self.nRays,nPoly), np.nan)
        kOut = np.full((self.nRays,nPoly), np.nan)

        # Compute intersections
        for ii in range(0,nPoly):
            VIn = np.diff(lPoly[ii],axis=1)
            VIn = VIn/(np.sqrt(np.sum(VIn**2,axis=0))[np.newaxis,:])
            VIn = np.ascontiguousarray([-VIn[1,:],VIn[0,:]])
            sca = VIn[0,0]*(isoR[0,0]-VP[0,0]) + VIn[1,0]*(isoZ[0,0]-VP[1,0])
            if sca<0:
                VIn = -VIn
            Out = tf.geom._GG.LOS_Calc_PInOut_VesStruct(self.D, self.u,
                                                        lPoly[ii], VIn, Lim=Lim)
            kPIn[:,ii], kPOut[:,ii] = Out[2:4]

        return kIn, kOut


    def calc_signal(self, ff, t=None, Ani=None, fkwdargs={}, Brightness=True,
                    dl=0.005, DL=None, dlMode='abs', method='sum',
                    ind=None, out=object, plot=True, plotmethod='imshow',
                    fs=None, dmargin=None, wintit='tofu', invert=True,
                    units=None, draw=True, connect=True):
        """ Return the line-integrated emissivity

        Beware, by default, Brightness=True and it is only a line-integral !

        Indeed, to get the received power, you need an estimate of the Etendue
        (previously set using self.set_Etendues()) and use Brightness=False.

        Hence, if Brightness=True and if
        the emissivity is provided in W/m3 (resp. W/m3/sr),
        => the method returns W/m2 (resp. W/m2/sr)
        The line is sampled using :meth:`~tofu.geom.LOS.get_sample`,

        The integral can be computed using three different methods:
            - 'sum':    A numpy.sum() on the local values (x segments lengths)
            - 'simps':  using :meth:`scipy.integrate.simps`
            - 'romb':   using :meth:`scipy.integrate.romb`

        Except ff, arguments common to :meth:`~tofu.geom.LOS.get_sample`

        Parameters
        ----------
        ff :    callable
            The user-provided

        Returns
        -------
        sig :   np.ndarray
            The computed signal, a 1d or 2d array depending on whether a time
            vector was provided.
        units:  str
            Units of the result

        """
        self._check_inputs(ind=ind)
        assert type(Brightness) is bool, "Arg Brightness must be a bool !"
        if Brightness is False:
            msg = "Etendue must be set if Brightness is False !"
            assert self.Etendues is not None, msg

        # Preformat ind
        if ind is None:
            ind = np.arange(0,self.nRays)
        if np.asarray(ind).dtype is bool:
            ind = ind.nonzero()[0]
        # Preformat DL
        if DL is None:
            DL = np.array([self.geom['kPIn'][ind],self.geom['kPOut'][ind]])
        elif np.asarray(DL).size==2:
            DL = np.tile(np.asarray(DL).ravel(),(len(ind),1)).T
        DL = np.ascontiguousarray(DL).astype(float)
        assert type(DL) is np.ndarray and DL.ndim==2
        assert DL.shape==(2,len(ind)), "Arg DL has wrong shape !"
        ii = DL[0,:]<self.geom['kPIn'][ind]
        DL[0,ii] = self.geom['kPIn'][ind][ii]
        ii = DL[0,:]>=self.geom['kPOut'][ind]
        DL[0,ii] = self.geom['kPOut'][ind][ii]
        ii = DL[1,:]>self.geom['kPOut'][ind]
        DL[1,ii] = self.geom['kPOut'][ind][ii]
        ii = DL[1,:]<=self.geom['kPIn'][ind]
        DL[1,ii] = self.geom['kPIn'][ind][ii]
        # Preformat Ds, us and Etendue
        Ds, us = self.D[:,ind], self.u[:,ind]
        if Brightness is False:
            E = self.Etendues
            if self.Etendues.size==self.nRays:
                E = E[ind]
        if len(ind)==1:
            Ds, us = Ds.reshape((3,1)), us.reshape((3,1))
        if t is None or len(t)==1:
            sig = np.full((Ds.shape[1],),np.nan)
        else:
            sig = np.full((len(t),Ds.shape[1]),np.nan)
        indok = ~(np.any(np.isnan(DL),axis=0) | np.any(np.isinf(DL),axis=0)
                  | ((DL[1,:]-DL[0,:])<=0.))
        if np.any(indok):
            Ds, us, DL = Ds[:,indok], us[:,indok], DL[:,indok]
            if indok.sum()==1:
                Ds, us = Ds.reshape((3,1)), us.reshape((3,1))
                DL = DL.reshape((2,1))
            Ds, us = np.ascontiguousarray(Ds), np.ascontiguousarray(us)
            DL = np.ascontiguousarray(DL)
            # Launch    # NB : find a way to exclude cases with DL[0,:]>=DL[1,:] !!
            # Exclude Rays not seeing the plasma
            s = _GG.LOS_calc_signal(ff, Ds, us, dl, DL,
                                    dLMode=dlMode, method=method,
                                    t=t, Ani=Ani, fkwdargs=fkwdargs, Test=True)
            if t is None or len(t)==1:
                sig[indok] = s
            else:
                sig[:,indok] = s
        if Brightness is False:
            if t is None or len(t)==1 or E.size==1:
                sig = sig*E
            else:
                sig = sig*E[np.newaxis,:]
            if units is None:
                units = r"origin x $m^3.sr$"
        elif units is None:
            units = r"origin x m"

        if plot or out is object:
            assert '1D' in self.Id.Cls or '2D' in self.Id.Cls, "Set Cam type!!"
            import tofu.data as tfd
            if '1D' in self.Id.Cls:
                osig = tfd.Data1D(data=sig, t=t, LCam=self,
                                  Id=self.Id.Name, dunits={'data':units},
                                  Exp=self.Id.Exp, Diag=self.Id.Diag)
            else:
                osig = tfd.Data2D(data=sig, t=t, LCam=self,
                                  Id=self.Id.Name, dunits={'data':units},
                                  Exp=self.Id.Exp, Diag=self.Id.Diag)
            if plot:
                KH = osig.plot(fs=fs, dmargin=dmargin, wintit=wintit,
                               plotmethod=plotmethod, invert=invert,
                               draw=draw, connect=connect)
            if out is object:
                sig = osig
        return sig, units

    def plot(self, Lax=None, Proj='All', Lplot=_def.LOSLplot, Elt='LDIORP',
             EltVes='', EltStruct='', Leg='', dL=None, dPtD=_def.LOSMd,
             dPtI=_def.LOSMd, dPtO=_def.LOSMd, dPtR=_def.LOSMd,
             dPtP=_def.LOSMd, dLeg=_def.TorLegd, dVes=_def.Vesdict,
             dStruct=_def.Structdict, multi=False, ind=None,
             fs=None, wintit='tofu', draw=True, Test=True):
        """ Plot the Rays / LOS, in the chosen projection(s)

        Optionnally also plot associated :class:`~tofu.geom.Ves` and Struct
        The plot can also include:
            - special points
            - the unit directing vector

        Parameters
        ----------
        Lax :       list / plt.Axes
            The axes for plotting (list of 2 axes if Proj='All')
            If None a new figure with new axes is created
        Proj :      str
            Flag specifying the kind of projection:
                - 'Cross' : cross-section
                - 'Hor' : horizontal
                - 'All' : both cross-section and horizontal (on 2 axes)
                - '3d' : a (matplotlib) 3d plot
        Elt :       str
            Flag specifying which elements to plot
            Each capital letter corresponds to an element:
                * 'L': LOS
                * 'D': Starting point of the LOS
                * 'I': Input point (i.e.: where the LOS enters the Vessel)
                * 'O': Output point (i.e.: where the LOS exits the Vessel)
                * 'R': Point of minimal major radius R (only if Ves.Type='Tor')
                * 'P': Point of used for impact parameter (i.e.: with minimal
                        distance to reference point Sino_RefPt)
        Lplot :     str
            Flag specifying the length to plot:
                - 'Tot': total length, from starting point (D) to output point
                - 'In' : only the in-vessel fraction (from input to output)
        EltVes :    str
            Flag for Ves elements to plot (:meth:`~tofu.geom.Ves.plot`)
        EltStruct : str
            Flag for Struct elements to plot (:meth:`~tofu.geom.Struct.plot`)
        Leg :       str
            Legend, if Leg='' the LOS name is used
        dL :     dict / None
            Dictionary of properties for plotting the lines
            Fed to plt.Axes.plot(), set to default if None
        dPtD :      dict
            Dictionary of properties for plotting point 'D'
        dPtI :      dict
            Dictionary of properties for plotting point 'I'
        dPtO :      dict
            Dictionary of properties for plotting point 'O'
        dPtR :      dict
            Dictionary of properties for plotting point 'R'
        dPtP :      dict
            Dictionary of properties for plotting point 'P'
        dLeg :      dict or None
            Dictionary of properties for plotting the legend
            Fed to plt.legend(), the legend is not plotted if None
        dVes :      dict
            Dictionary of kwdargs to fed to :meth:`~tofu.geom.Ves.plot`
            And 'EltVes' is used instead of 'Elt'
        dStruct:    dict
            Dictionary of kwdargs to fed to :meth:`~tofu.geom.Struct.plot`
            And 'EltStruct' is used instead of 'Elt'
        draw :      bool
            Flag indicating whether fig.canvas.draw() shall be called
        a4 :        bool
            Flag indicating whether to plot the figure in a4 dimensions
        Test :      bool
        a4 :        bool
            Flag indicating whether to plot the figure in a4 dimensions
        Test :      bool
        a4 :        bool
            Flag indicating whether to plot the figure in a4 dimensions
        Test :      bool
        a4 :        bool
            Flag indicating whether to plot the figure in a4 dimensions
        Test :      bool
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        La :        list / plt.Axes
            Handles of the axes used for plotting (list if Proj='All')

        """

        return _plot.Rays_plot(self, Lax=Lax, Proj=Proj, Lplot=Lplot, Elt=Elt,
                               EltVes=EltVes, EltStruct=EltStruct, Leg=Leg,
                               dL=dL, dPtD=dPtD, dPtI=dPtI, dPtO=dPtO, dPtR=dPtR,
                               dPtP=dPtP, dLeg=dLeg, dVes=dVes, dStruct=dStruct,
                               multi=multi, ind=ind,
                               fs=fs, wintit=wintit, draw=draw, Test=Test)


    def plot_sino(self, Proj='Cross', ax=None, Elt=_def.LOSImpElt, Sketch=True,
                  Ang=_def.LOSImpAng, AngUnit=_def.LOSImpAngUnit, Leg=None,
                  dL=_def.LOSMImpd, dVes=_def.TorPFilld, dLeg=_def.TorLegd,
                  ind=None, multi=False,
                  fs=None, wintit='tofu', draw=True, Test=True):
        """ Plot the LOS in projection space (sinogram)

        Plot the Rays in projection space (cf. sinograms) as points.
        Can also optionnally plot the associated :class:`~tofu.geom.Ves`

        Can plot the conventional projection-space (in 2D in a cross-section),
        or a 3D extrapolation of it, where the third coordinate is provided by
        the angle that the LOS makes with the cross-section plane
        (useful in case of multiple LOS with a partially tangential view)

        Parameters
        ----------
        Proj :      str
            Flag indicating whether to plot:
                - 'Cross':  a classic sinogram (vessel cross-section)
                - '3d': an extended 3D version ('3d'), with an additional angle
        ax :        None / plt.Axes
            The axes on which to plot, if None a new figure is created
        Elt :       str
            Flag indicating which elements to plot (one per capital letter):
                * 'L': LOS
                * 'V': Vessel
        Ang  :      str
            Flag indicating which angle to use for the impact parameter:
                - 'xi': the angle of the line itself
                - 'theta': its impact parameter (theta)
        AngUnit :   str
            Flag for the angle units to be displayed:
                - 'rad': for radians
                - 'deg': for degrees
        Sketch :    bool
            Flag indicating whether to plot a skecth with angles definitions
        dL :        dict
            Dictionary of properties for plotting the Rays points
        dV :        dict
            Dictionary of properties for plotting the vessel envelopp
        dLeg :      None / dict
            Dictionary of properties for plotting the legend
            The legend is not plotted if None
        draw :      bool
            Flag indicating whether to draw the figure
        a4 :        bool
            Flag indicating whether the figure should be a4
        Test :      bool
            Flag indicating whether the inputs shall be tested for conformity

        Returns
        -------
        ax :        plt.Axes
            The axes used to plot

        """
        assert self.sino is not None, "The sinogram ref. point is not set !"
        return _plot.GLOS_plot_Sino(self, Proj=Proj, ax=ax, Elt=Elt, Leg=Leg,
                                    Sketch=Sketch, Ang=Ang, AngUnit=AngUnit,
                                    dL=dL, dVes=dVes, dLeg=dLeg,
                                    ind=ind, fs=fs, wintit=wintit,
                                    draw=draw, Test=Test)

    def plot_touch(self, key=None, invert=None, plotmethod='imshow',
                   lcol=['k','r','b','g','y','m','c'],
                   fs=None, wintit='tofu', draw=True):
        assert self.Id.Cls in ['LOSCam1D','LOSCam2D'], "Specify camera type !"
        assert self.Ves is not None, "self.Ves should not be None !"
        out = _plot.Rays_plot_touch(self, key=key, invert=invert,
                                    lcol=lcol, plotmethod=plotmethod,
                                    fs=fs, wintit=wintit, draw=draw)
        return out

    def save(self, SaveName=None, Path=None,
             Mode='npz', compressed=False, Print=True):
        """ Save the object in folder Name, under SaveName

        Parameters
        ----------
        SaveName :  None / str
            The name to be used for the saved file
            If None (recommended) uses self.Id.SaveName
        Path :      None / str
            Path specifying where to save the file
            If None (recommended) uses self.Id.SavePath
        Mode :      str
            Flag specifying how to save the object:
                'npz': as a numpy array file (recommended)
        compressed :    bool
            Flag, used when Mode='npz', indicates whether to use:
                - False : np.savez
                - True :  np.savez_compressed (slower but smaller files)

        """
        tfpf.Save_Generic(self, SaveName=SaveName, Path=Path,
                          Mode=Mode, compressed=compressed, Print=Print)







def _Rays_check_inputs(Id=None, Du=None, Vess=None, LStruct=None,
                       Sino_RefPt=None, Exp=None, shot=None, Diag=None,
                       SavePath=None, Calc=None, ind=None,
                       dchans=None, fromdict=None):
    if not Id is None:
        assert type(Id) in [str,tfpf.ID], "Arg Id must be a str or a tfpf.ID !"
    if not Du is None:
        C0 = hasattr(Du,'__iter__') and len(Du)==2
        C1 = 3 in np.asarray(Du[0]).shape and np.asarray(Du[0]).ndim in [1,2]
        C2 = 3 in np.asarray(Du[1]).shape and np.asarray(Du[1]).ndim in [1,2]
        #C3 = np.asarray(Du[0]).shape==np.asarray(Du[1]).shape
        assert C0, "Arg Du must be an iterable of len()=2 !"
        assert C1, "Du[0] must contain 3D coordinates of all starting points !"
        assert C2, "Du[1] must contain 3D coordinates of all unit vectors !"
        #assert C3, "Du[0] and Du[1] must be of same shape !"
    if not Vess is None:
        assert type(Vess) is Ves, "Arg Ves must be a Ves instance !"
        if Exp is not None and Vess.Id.Exp is not None:
            assert Exp==Vess.Id.Exp, "Arg Exp must be the same as Ves.Id.Exp !"
    if LStruct is not None:
        assert type(LStruct) in [list,Struct], "LStruct = %s !"%(type(LStruct))
        if type(LStruct) is list:
            for ss in LStruct:
                assert type(ss) is Struct, "LStruct = list of Struct !"
                if Exp is not None and ss.Id.Exp is not None:
                    assert Exp==ss.Id.Exp, "Struct elements for a different Exp"
        else:
            if Exp is not None and LStruct.Id.Exp is not None:
                assert Exp==LStruct.Id.Exp, "Struct element for a different Exp"
    bools = [Calc]
    if any([not aa is None for aa in bools]):
        C = all([aa is None or type(aa) is bool for aa in bools])
        assert C, " Args [Calc] must all be bool !"
    strs = [Exp,Diag,SavePath]
    if any([not aa is None for aa in strs]):
        C = all([aa is None or type(aa) is str for aa in strs])
        assert C, "Args [Exp,Diag,SavePath] must all be str !"
    Iter2 = [Sino_RefPt]
    for aa in Iter2:
        if aa is not None:
            C0 = np.asarray(aa).shape==(2,)
            assert C0, "Args [Sino_RefPt] must be an iterable with len()=2 !"
    Ints = [shot]
    for aa in Ints:
        if aa is not None:
            assert type(aa) is int, "Args [shot] must be int !"
    if ind is not None:
        assert np.asarray(ind).ndim==1
        assert np.asarray(ind).dtype in [bool,np.int64]
    if dchans is not None:
        assert type(dchans) is dict
        if Du is not None:
            nch = Du[0].shape[1]
            assert all([len(dchans[kk])==nch for kk in dchans.keys()])



def _Rays_check_fromdict(fd):
    assert type(fd) is dict, "Arg from dict must be a dict !"
    k0 = {'Id':dict,'geom':dict,'sino':dict,
          'dchans':[None,dict],
          'Ves':[None,dict], 'LStruct':[None,list]}
    keys = list(fd.keys())
    for kk in k0:
        assert kk in keys, "%s must be a key of fromdict"%kk
        typ = type(fd[kk])
        C = typ is k0[kk] or typ in k0[kk] or fd[kk] in k0[kk]
        assert C, "Wrong type of fromdict[%s]: %s"%(kk,str(typ))







class LOSCam1D(Rays):
    def __init__(self, Id=None, Du=None, Ves=None, LStruct=None,
                 Sino_RefPt=None, fromdict=None,
                 Exp=None, Diag=None, shot=0, Etendues=None,
                 dchans=None, SavePath=os.path.abspath('./'),
                 plotdebug=True):
        Rays.__init__(self, Id=Id, Du=Du, Ves=Ves, LStruct=LStruct,
                 Sino_RefPt=Sino_RefPt, fromdict=fromdict, Etendues=Etendues,
                 Exp=Exp, Diag=Diag, shot=shot, plotdebug=plotdebug,
                 dchans=dchans, SavePath=SavePath)

class LOSCam2D(Rays):
    def __init__(self, Id=None, Du=None, Ves=None, LStruct=None,
                 Sino_RefPt=None, fromdict=None, Etendues=None,
                 Exp=None, Diag=None, shot=0, X12=None,
                 dchans=None, SavePath=os.path.abspath('./'),
                 plotdebug=True):
        Rays.__init__(self, Id=Id, Du=Du, Ves=Ves, LStruct=LStruct,
                 Sino_RefPt=Sino_RefPt, fromdict=fromdict, Etendues=Etendues,
                 Exp=Exp, Diag=Diag, shot=shot, plotdebug=plotdebug,
                 dchans=dchans, SavePath=SavePath)
        self.set_X12(X12)

    def set_e12(self, e1=None, e2=None):
        assert e1 is None or (hasattr(e1,'__iter__') and len(e1)==3)
        assert e2 is None or (hasattr(e2,'__iter__') and len(e2)==3)
        if e1 is None:
            e1 = self._geom['e1']
        else:
            e1 = np.asarray(e1).astype(float).ravel()
        e1 = e1 / np.linalg.norm(e1)
        if e2 is None:
            e2 = self._geom['e2']
        else:
            e2 = np.asarray(e1).astype(float).ravel()
        e2 = e2 / np.linalg.norm(e2)
        assert np.abs(np.sum(e1*self._geom['nIn']))<1.e-12
        assert np.abs(np.sum(e2*self._geom['nIn']))<1.e-12
        assert np.abs(np.sum(e1*e2))<1.e-12
        self._geom['e1'] = e1
        self._geom['e2'] = e2

    def set_X12(self, X12=None):
        if X12 is not None:
            X12 = np.asarray(X12)
            assert X12.shape==(2,self.Ref['nch'])
        self._X12 = X12

    def get_X12(self, out='1d'):
        if self._X12 is None:
            Ds = self.D
            C = np.mean(Ds,axis=1)
            X12 = Ds-C[:,np.newaxis]
            X12 = np.array([np.sum(X12*self.geom['e1'][:,np.newaxis],axis=0),
                            np.sum(X12*self.geom['e2'][:,np.newaxis],axis=0)])
        else:
            X12 = self._X12
        if X12 is None or out.lower()=='1d':
            DX12 = None
        else:
            x1u, x2u, ind, DX12 = utils.get_X12fromflat(X12)
            if out.lower()=='2d':
                X12 = [x1u, x2u, ind]
        return X12, DX12







    """ Return the indices or instances of all LOS matching criteria

    The selection can be done according to 2 different mechanisms

    Mechanism (1): provide the value (Val) a criterion (Crit) should match
    The criteria are typically attributes of :class:`~tofu.pathfile.ID`
    (i.e.: name, or user-defined attributes like the camera head...)

    Mechanism (2): (used if Val=None)
    Provide a str expression (or a list of such) to be fed to eval()
    Used to check on quantitative criteria.
        - PreExp: placed before the criterion value (e.g.: 'not ' or '<=')
        - PostExp: placed after the criterion value
        - you can use both

    Other parameters are used to specify logical operators for the selection
    (match any or all the criterion...) and the type of output.

    Parameters
    ----------
    Crit :      str
        Flag indicating which criterion to use for discrimination
        Can be set to:
            - any attribute of :class:`~tofu.pathfile.ID`
        A str (or list of such) expression to be fed to eval()
        Placed after the criterion value
        Used for selection mechanism (2)
    Log :       str
        Flag indicating whether the criterion shall match:
            - 'all': all provided values
            - 'any': at least one of them
    InOut :     str
        Flag indicating whether the returned indices are:
            - 'In': the ones matching the criterion
            - 'Out': the ones not matching it
    Out :       type / str
        Flag indicating in which form to return the result:
            - int: as an array of integer indices
            - bool: as an array of boolean indices
            - 'Name': as a list of names
            - 'LOS': as a list of :class:`~tofu.geom.LOS` instances

    Returns
    -------
    ind :       list / np.ndarray
        The computed output, of nature defined by parameter Out

    Examples
    --------
    >>> import tofu.geom as tfg
    >>> VPoly, VLim = [[0.,1.,1.,0.],[0.,0.,1.,1.]], [-1.,1.]
    >>> V = tfg.Ves('ves', VPoly, Lim=VLim, Type='Lin', Exp='Misc', shot=0)
    >>> Du1 = ([0.,-0.1,-0.1],[0.,1.,1.])
    >>> Du2 = ([0.,-0.1,-0.1],[0.,0.5,1.])
    >>> Du3 = ([0.,-0.1,-0.1],[0.,1.,0.5])
    >>> l1 = tfg.LOS('l1', Du1, Ves=V, Exp='Misc', Diag='A', shot=0)
    >>> l2 = tfg.LOS('l2', Du2, Ves=V, Exp='Misc', Diag='A', shot=1)
    >>> l3 = tfg.LOS('l3', Du3, Ves=V, Exp='Misc', Diag='B', shot=1)
    >>> gl = tfg.GLOS('gl', [l1,l2,l3])
    >>> Arg1 = dict(Val=['l1','l3'],Log='any',Out='LOS')
    >>> Arg2 = dict(Val=['l1','l3'],Log='any',InOut='Out',Out=int)
    >>> Arg3 = dict(Crit='Diag', Val='A', Out='Name')
    >>> Arg4 = dict(Crit='shot', PostExp='>=1')
    >>> gl.select(**Arg1)
    [l1,l3]
    >>> gl.select(**Arg2)
    array([1])
    >>> gl.select(**Arg3)
    ['l1','l2']
    >>> gl.select(**Arg4)
    array([False, True, True], dtype=bool)

    """
