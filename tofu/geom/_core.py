"""
This module is the geometrical part of the ToFu general package
It includes all functions and object classes necessary for tomography on Tokamaks
"""

# Built-in
import os
import sys
import warnings
#from abc import ABCMeta, abstractmethod
import copy
if sys.version[0]=='2':
    import re, tokenize, keyword
    import funcsigs as inspect
else:
    import inspect


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

__all__ = ['PlasmaDomain', 'Ves', 'PFC', 'CoilPF', 'CoilCS', 'Config',
           'Rays','CamLOS1D','CamLOS2D']


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
        A name string or a pre-built tfpf.ID class to be used to identify this
        particular instance, if a string is provided, it is fed to tfpf.ID()
    Poly :          np.ndarray
        An array (2,N) or (N,2) defining the contour of the vacuum vessel in a
        cross-section, if not closed, will be closed automatically
    Type :          str
        Flag indicating whether the vessel will be a torus ('Tor') or a linear
        device ('Lin')
    Lim :         list / np.ndarray
        Array or list of len=2 indicating the limits of the linear device volume
        on the x axis
    Sino_RefPt :    None / np.ndarray
        Array specifying a reference point for computing the sinogram (i.e.
        impact parameter), if None automatically set to the (surfacic) center of
        mass of the cross-section
    Sino_NP :       int
        Number of points in [0,2*pi] to be used to plot the vessel sinogram
        envelop
    Clock :         bool
        Flag indicating whether the input polygon should be made clockwise
        (True) or counter-clockwise (False)
    arrayorder:     str
        Flag indicating whether the attributes of type=np.ndarray (e.g.: Poly)
        should be made C-contiguous ('C') or Fortran-contiguous ('F')
    Exp :           None / str
        Flag indicating which experiment the object corresponds to, allowed
        values are in [None,'AUG','MISTRAL','JET','ITER','TCV','TS','Misc']
    shot :          None / int
        Shot number from which this Ves is usable (in case of change of
        geometry)
    SavePath :      None / str
        If provided, forces the default saving path of the object to the
        provided value

    Returns
    -------
    Ves :        Ves object
        The created Ves object, with all necessary computed attributes and
        methods

    """

    #__metaclass__ = ABCMeta


    # Fixed (class-wise) dictionary of default properties
    _ddef = {'Id':{'shot':0,
                   'include':['Mod','Cls','Exp','Diag',
                              'Name','shot','version']},
             'dgeom':{'Type':'Tor', 'Lim':[], 'arrayorder':'C'},
             'dsino':{},
             'dphys':{},
             'dmisc':{'color':'k'}}
    _dplot = {'cross':{'Elt':'P',
                       'dP':{'color':'k','lw':2},
                       'dI':{'color':'k','ls':'--','marker':'x','ms':8,'mew':2},
                       'dBs':{'color':'b','ls':'--','marker':'x','ms':8,'mew':2},
                       'dBv':{'color':'g','ls':'--','marker':'x','ms':8,'mew':2},
                       'dVect':{'color':'r','scale':10}},
              'hor':{'Elt':'P',
                     'dP':{'color':'k','lw':2},
                     'dI':{'color':'k','ls':'--'},
                     'dBs':{'color':'b','ls':'--'},
                     'dBv':{'color':'g','ls':'--'},
                     'Nstep':50},
              '3d':{'Elt':'P',
                    'dP':{'color':(0.8,0.8,0.8,1.),
                          'rstride':1,'cstride':1,
                          'linewidth':0., 'antialiased':False},
                    'Lim':None,
                    'Nstep':50}}


    # Does not exist beofre Python 3.6 !!!
    def __init_subclass__(cls, color='k', **kwdargs):
        # Python 2
        super(Struct,cls).__init_subclass__(**kwdargs)
        # Python 3
        #super().__init_subclass__(**kwdargs)
        cls._ddef = copy.deepcopy(Struct._ddef)
        cls._dplot = copy.deepcopy(Struct._dplot)
        cls._set_color_ddef(cls._color)

    @classmethod
    def _set_color_ddef(cls, color):
        cls._ddef['dmisc']['color'] = mpl.colors.to_rgba(color)

    def __init__(self, Poly=None, Type=None,
                 Lim=None, pos=None, extent=None, mobile=False,
                 Id=None, Name=None, Exp=None, shot=None,
                 sino_RefPt=None, sino_nP=_def.TorNP,
                 Clock=False, arrayorder='C', fromdict=None,
                 SavePath=os.path.abspath('./'),
                 SavePath_Include=tfpf.defInclude, color=None):

        # To replace __init_subclass__ for Python 2
        if sys.version[0]=='2':
            self._dstrip = utils.ToFuObjectBase._dstrip.copy()
            self.__class__._strip_init()

        # Create a dplot at instance level
        self._dplot = copy.deepcopy(self.__class__._dplot)

        kwdargs = locals()
        del kwdargs['self']
        # super()
        super(Struct,self).__init__(**kwdargs)

    def _reset(self):
        # super()
        super(Struct,self)._reset()
        self._dgeom = dict.fromkeys(self._get_keys_dgeom())
        self._dsino = dict.fromkeys(self._get_keys_dsino())
        self._dphys = dict.fromkeys(self._get_keys_dphys())
        self._dmisc = dict.fromkeys(self._get_keys_dmisc())
        #self._dplot = copy.deepcopy(self.__class__._ddef['dplot'])

    @classmethod
    def _checkformat_inputs_Id(cls, Id=None, Name=None,
                               Exp=None, shot=None, Type=None,
                               include=None,
                               **kwdargs):
        if Id is not None:
            assert isinstance(Id,utils.ID)
            Name, Exp, shot, Type = Id.Name, Id.Exp, Id.shot, Id.Type
        if shot is None:
            shot = cls._ddef['Id']['shot']
        if Type is None:
            Type = cls._ddef['dgeom']['Type']
        if include is None:
            include = cls._ddef['Id']['include']

        dins = {'Name':{'var':Name, 'cls':str},
                'Exp': {'var':Exp, 'cls':str},
                'shot': {'var':shot, 'cls':int},
                'Type': {'var':Type, 'in':['Tor','Lin']},
                'include':{'var':include, 'listof':str}}
        dins, err, msg = cls._check_InputsGeneric(dins)
        if err:
            raise Exception(msg)

        kwdargs.update({'Name':Name, 'Exp':Exp, 'shot':shot, 'Type':Type,
                        'include':include})
        return kwdargs

    ###########
    # Get largs
    ###########

    @staticmethod
    def _get_largs_dgeom(sino=True):
        largs = ['Poly','Lim','pos','extent','mobile','Clock','arrayorder']
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
    def _checkformat_Lim(Lim, Type='Tor'):
        if Lim is None:
            Lim = np.array([],dtype=float)
        else:
            assert hasattr(Lim,'__iter__')
            Lim = np.asarray(Lim,dtype=float)
            assert Lim.ndim in [1,2]
            if Lim.ndim==1:
                assert Lim.size in [0,2]
                if Lim.size==2:
                    Lim = Lim.reshape((2,1))
            else:
                if Lim.shape[0]!=2:
                    Lim = Lim.T
            if Type=='Lin':
                if not np.all(Lim[0,:]<Lim[1,:]):
                    msg = "All provided Lim must be increasing !"
                    raise Exception(msg)
            else:
                Lim = np.arctan2(np.sin(Lim),np.cos(Lim))
            assert np.all(~np.isnan(Lim))
        return Lim

    @staticmethod
    def _checkformat_posextent(pos, extent, Type='Tor'):
        lC = [pos is None, extent is None]
        if any(lC):
            if not all(lC):
                msg = ""
                raise Exception(msg)
            pos = np.array([],dtype=float)
            extent = np.array([],dtype=float)
        else:
            lfloat = [int, float, np.int64, np.float64]
            assert type(pos) in lfloat or hasattr(pos,'__iter__')
            if type(pos) in lfloat:
                pos = np.array([pos],dtype=float)
            else:
                pos = np.asarray(pos,dtype=float).ravel()
            if Type=='Tor':
                pos = np.arctan2(np.sin(pos),np.cos(pos))
            assert type(extent) in lfloat or hasattr(extent,'__iter__')
            if type(extent) in lfloat:
                extent = float(extent)
            else:
                extent = np.asarray(extent,dtype=float).ravel()
                assert extent.size==pos.size
            if not np.all(extent>0.):
                msg = "All provided extent values must be >0 !"
                raise Exception(msg)
            if Type=='Tor':
                if not np.all(extent<2.*np.pi):
                    msg = "Provided extent must be in ]0;2pi[ (radians)!"
                    raise Exception(msg)
            assert np.all(~np.isnan(pos)) and np.all(~np.isnan(extent))
        return pos, extent

    @staticmethod
    def _get_LimFromPosExtent(pos, extent, Type='Tor'):
        if pos.size>0:
            Lim = pos[np.newaxis,:] + np.array([[-0.5],[0.5]])*extent
            if Type=='Tor':
                Lim = np.arctan2(np.sin(Lim),np.cos(Lim))
        else:
            Lim = np.asarray([],dtype=float)
        return Lim

    @staticmethod
    def _get_PosExtentFromLim(Lim, Type='Tor'):
        if Lim.size>0:
            pos, extent = np.mean(Lim,axis=0), Lim[1,:]-Lim[0,:]
            if Type=='Tor':
                ind = Lim[0,:]>Lim[1,:]
                pos[ind] = pos[ind] + np.pi
                extent[ind] = 2.*np.pi + extent[ind]
                pos = np.arctan2(np.sin(pos),np.cos(pos))
                assert np.all(extent>0.)
            if np.std(extent)<np.mean(extent)*1.e-9:
                extent = np.mean(extent)
        else:
            pos = np.array([],dtype=float)
            extent = np.array([],dtype=float)
        return pos, extent

    @classmethod
    def _checkformat_inputs_dgeom(cls, Poly=None,
                                  Lim=None, pos=None, extent=None, mobile=False,
                                  Type=None, Clock=False, arrayorder=None):
        if arrayorder is None:
            arrayorder = Struct._ddef['dgeom']['arrayorder']
        if Type is None:
            Type = Struct._ddef['dgeom']['Type']

        dins = {'Poly':{'var':Poly, 'iter2array':float, 'ndim':2, 'inshape':2},
                'Clock':{'var':Clock, 'cls':bool},
                'mobile':{'var':mobile, 'cls':bool},
                'arrayorder':{'var':arrayorder, 'in':['C','F']},
                'Type':{'var':Type, 'in':['Tor','Lin']}}
        dins, err, msg = cls._check_InputsGeneric(dins, tab=0)
        if err:
            raise Exception(msg)
        Poly = dins['Poly']['var']
        if Poly.shape[0]!=2:
            Poly = Poly.T

        # Elimininate any double identical point
        ind = np.sum(np.diff(Poly,axis=1)**2, axis=0) < 1.e-12
        if np.any(ind):
            npts = Poly.shape[1]
            msg = "%s instance: double identical points in Poly\n"%cls.__name__
            msg += "  => %s points removed\n"%ind.sum()
            msg += "  => Poly goes from %s to %s points"%(npts,npts-ind.sum())
            warnings.warn(msg)
            Poly = Poly[:,~ind]


        lC = [Lim is None, pos is None]
        if not any(lC):
            msg = "Please provide either Lim xor pos/extent pair!\n"
            msg += "Lim should be an array of limits\n"
            msg += "pos should be an array of centers and extent a float / array"
            raise Exception(msg)
        if all(lC):
            pos = np.asarray([],dtype=float)
            extent = np.asarray([],dtype=float)
            #Lim = np.asarray([],dtype=float)
        elif lC[0]:
            pos, extent =  cls._checkformat_posextent(pos, extent, Type)
            #Lim = cls._get_LimFromPosExtent(pos, extent, Type)
        else:
            Lim =  cls._checkformat_Lim(Lim, Type)
            pos, extent = cls._get_PosExtentFromLim(Lim, Type)

        return Poly, pos, extent, Type, arrayorder

    def _checkformat_inputs_dsino(self, RefPt=None, nP=None):
        assert type(nP) is int and nP>0
        assert RefPt is None or hasattr(RefPt,'__iter__')
        if RefPt is None:
            RefPt = self._dgeom['BaryS']
        RefPt = np.asarray(RefPt,dtype=float).flatten()
        assert RefPt.size==2, "RefPt must be of size=2 !"
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
            color = mpl.colors.to_rgba(cls._ddef['dmisc']['color'])
        assert mpl.colors.is_color_like(color)
        return tuple(mpl.colors.to_rgba(color))

    ###########
    # Get keys of dictionnaries
    ###########

    @staticmethod
    def _get_keys_dgeom():
        lk = ['Poly','pos','extent','noccur','Multi','nP',
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

    def _init(self, Poly=None, Type=_Type,
              Lim=None, pos=None, extent=None, mobile=False,
              Clock=_Clock, arrayorder=_arrayorder,
              sino_RefPt=None, sino_nP=_def.TorNP, color=None, **kwdargs):
        allkwds = dict(locals(), **kwdargs)
        largs = self._get_largs_dgeom(sino=True)
        kwdgeom = self._extract_kwdargs(allkwds, largs)
        largs = self._get_largs_dphys()
        kwdphys = self._extract_kwdargs(allkwds, largs)
        largs = self._get_largs_dmisc()
        kwdmisc = self._extract_kwdargs(allkwds, largs)
        self._set_dgeom(**kwdgeom)
        self.set_dphys(**kwdphys)
        self._set_dmisc(**kwdmisc)
        self._dstrip['strip'] = 0

    ###########
    # set dictionaries
    ###########

    def _set_dgeom(self, Poly=None,
                   Lim=None, pos=None, extent=None, mobile=False,
                   Clock=False, arrayorder='C',
                   sino_RefPt=None, sino_nP=_def.TorNP, sino=True):
        out = self._checkformat_inputs_dgeom(Poly=Poly, Lim=Lim, pos=pos,
                                             extent=extent, mobile=mobile,
                                             Type=self.Id.Type, Clock=Clock)
        Poly, pos, extent, Type, arrayorder = out
        dgeom = _comp._Struct_set_Poly(Poly, pos=pos, extent=extent,
                                       arrayorder=arrayorder,
                                       Type=self.Id.Type, Clock=Clock)
        dgeom['arrayorder'] = arrayorder
        dgeom['mobile'] = mobile
        self._dgeom = dgeom
        if sino:
            self.set_dsino(sino_RefPt, nP=sino_nP)

    def set_dsino(self, RefPt=None, nP=_def.TorNP):
        RefPt = self._checkformat_inputs_dsino(RefPt=RefPt, nP=nP)
        EnvTheta, EnvMinMax = _GG.Sino_ImpactEnv(RefPt, self.Poly_closed,
                                                 NP=nP, Test=False)
        self._dsino = {'RefPt':RefPt, 'nP':nP,
                       'EnvTheta':EnvTheta, 'EnvMinMax':EnvMinMax}

    def set_dphys(self, lSymbols=None):
        lSymbols = self._checkformat_inputs_dphys(lSymbols)
        self._dphys['lSymbols'] = lSymbols

    def _set_color(self, color=None):
        color = self._checkformat_inputs_dmisc(color=color)
        self._dmisc['color'] = color
        self._dplot['cross']['dP']['color'] = color
        self._dplot['hor']['dP']['color'] = color
        self._dplot['3d']['dP']['color'] = color

    def _set_dmisc(self, color=None):
        self._set_color(color)

    ###########
    # strip dictionaries
    ###########

    def _strip_dgeom(self, lkeep=['Poly','pos', 'extent','mobile','Clock','arrayorder']):
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

    def _rebuild_dgeom(self, lkeep=['Poly','pos','extent','mobile','Clock','arrayorder']):
        reset = utils.ToFuObject._test_Rebuild(self._dgeom, lkeep=lkeep)
        if reset:
            utils.ToFuObject._check_Fields4Rebuild(self._dgeom,
                                                   lkeep=lkeep, dname='dgeom')
            self._set_dgeom(self.Poly, pos=self.pos, extent=self.extent,
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
        if sys.version[0]=='2':
            cls.strip.__func__.__doc__ = doc
        else:
            cls.strip.__doc__ = doc

    def strip(self, strip=0):
        # super()
        super(Struct,self).strip(strip=strip)

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
                'dmisc':{'dict':self.dmisc, 'lexcept':None},
                'dplot':{'dict':self._dplot, 'lexcept':None}}
        return dout

    def _from_dict(self, fd):
        self._dgeom.update(**fd['dgeom'])
        self._dsino.update(**fd['dsino'])
        self._dphys.update(**fd['dphys'])
        self._dmisc.update(**fd['dmisc'])
        if 'dplot' in fd.keys():
            self._dplot.update(**fd['dplot'])

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
    def pos(self):
        return self._dgeom['pos']
    @property
    def extent(self):
        if hasattr(self._dgeom['extent'],'__iter__'):
            extent = self._dgeom['extent']
        else:
            extent = np.full(self._dgeom['pos'].shape,self._dgeom['extent'])
        return extent
    @property
    def noccur(self):
        return self._dgeom['noccur']
    @property
    def Lim(self):
        Lim = self._get_LimFromPosExtent(self._dgeom['pos'],
                                         self._dgeom['extent'],
                                         Type=self.Id.Type)
        return Lim.T
    @property
    def dsino(self):
        return self._dsino
    @property
    def dphys(self):
        return self._dphys
    @property
    def dmisc(self):
        return self._dmisc


    ###########
    # public methods
    ###########

    def set_color(self, col):
        self._set_color(col)

    def get_color(self):
        return self._dmisc['color']

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
                                nLim=self._dgeom['noccur'],
                                VType=self.Id.Type,
                                In=In, Test=True)
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
        return _comp._Ves_get_InsideConvexPoly(self.Poly_closed,
                                               self.dgeom['P2Min'],
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

    def get_sampleCross(self, res, DS=None, resMode='abs', ind=None, mode='flat'):
        """ Sample, with resolution res, the 2D cross-section

        The sampling domain can be limited by DS or ind

        Depending on the value of mode, the method returns:
            - 'flat': (tuned for integrals computing)
                pts   : (2,npts) array of points coordinates
                dS    : (npts,) array of surfaces
                ind   : (npts,) array of integer indices
                reseff: (2,) array of effective resolution (R and Z)
            - 'imshow' : (tuned for imshow plotting)
                pts : (2,n1,n2) array of points coordinates
                x1  : (n1,) vector of unique x1 coordinates
                x2  : (n2,) vector of unique x2 coordinates
                extent : the extent to be fed to mpl.pyplot.imshow()

        """
        args = [self.Poly, self.dgeom['P1Min'][0], self.dgeom['P1Max'][0],
                self.dgeom['P2Min'][1], self.dgeom['P2Max'][1], res]
        kwdargs = dict(DS=DS, dSMode=resMode, ind=ind, margin=1.e-9, mode=mode)
        out = _comp._Ves_get_sampleCross(*args, **kwdargs)
        return out

    def get_sampleS(self, res, DS=None, resMode='abs',
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
                       VLim=np.ascontiguousarray(self.Lim), nVLim=self.noccur,
                       Out=Out, margin=1.e-9,
                       Multi=self.dgeom['Multi'], Ind=Ind)
        args = [self.Poly, self.dgeom['P1Min'][0], self.dgeom['P1Max'][0],
                self.dgeom['P2Min'][1], self.dgeom['P2Max'][1], res]
        pts, dS, ind, reseff = _comp._Ves_get_sampleS(*args, **kwdargs)
        return pts, dS, ind, reseff

    def get_sampleV(self, res, DV=None, resMode='abs', ind=None, Out='(X,Y,Z)'):
        """ Sample, with resolution res, the volume defined by DV or ind """

        args = [self.Poly, self.dgeom['P1Min'][0], self.dgeom['P1Max'][0],
                self.dgeom['P2Min'][1], self.dgeom['P2Max'][1], res]
        kwdargs = dict(DV=DV, dVMode=resMode, ind=ind, VType=self.Id.Type,
                      VLim=self.Lim, Out=Out, margin=1.e-9)
        pts, dV, ind, reseff = _comp._Ves_get_sampleV(*args, **kwdargs)
        return pts, dV, ind, reseff


    def plot(self, lax=None, proj='all', element='PIBsBvV',
             dP=None, dI=_def.TorId, dBs=_def.TorBsd, dBv=_def.TorBvd,
             dVect=_def.TorVind, dIHor=_def.TorITord, dBsHor=_def.TorBsTord,
             dBvHor=_def.TorBvTord, Lim=None, Nstep=_def.TorNTheta,
             dLeg=_def.TorLegd, indices=False,
             draw=True, fs=None, wintit=None, Test=True):
        """ Plot the polygon defining the vessel, in chosen projection

        Generic method for plotting the Ves object
        The projections to be plotted, the elements to plot can be specified
        Dictionaries of properties for each elements can also be specified
        If an ax is not provided a default one is created.

        Parameters
        ----------
        Lax :       list or plt.Axes
            The axes to be used for plotting
            Provide a list of 2 axes if proj='All'
            If None a new figure with axes is created
        proj :      str
            Flag specifying the kind of projection
                - 'Cross' : cross-section projection
                - 'Hor' : horizontal projection
                - 'All' : both
                - '3d' : a 3d matplotlib plot
        element :   str
            Flag specifying which elements to plot
            Each capital letter corresponds to an element:
                * 'P': polygon
                * 'I': point used as a reference for impact parameters
                * 'Bs': (surfacic) center of mass
                * 'Bv': (volumic) center of mass for Tor type
                * 'V': vector pointing inward perpendicular to each segment
        dP :        dict / None
            Dict of properties for plotting the polygon
            Fed to plt.Axes.plot() or plt.plot_surface() if proj='3d'
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
            Array of a lower and upper limit of angle (rad.) or length for
            plotting the '3d' proj
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
        for k in lout:
            del kwdargs[k]
        return _plot.Struct_plot(self, **kwdargs)


    def plot_sino(self, ax=None, Ang=_def.LOSImpAng,
                  AngUnit=_def.LOSImpAngUnit, Sketch=True, dP=None,
                  dLeg=_def.TorLegd, draw=True, fs=None, wintit=None,
                  Test=True):
        """ Plot the sinogram of the vessel polygon, by computing its envelopp in a cross-section, can also plot a 3D version of it

        The envelop of the polygon is computed using self.Sino_RefPt as a reference point in projection space,
        and plotted using the provided dictionary of properties.
        Optionaly a small sketch can be included illustrating how the angle
        and the impact parameters are defined (if the axes is not provided).

        Parameters
        ----------
        proj :      str
            Flag indicating whether to plot a classic sinogram ('Cross') from the vessel cross-section (assuming 2D)
            or an extended 3D version '3d' of it with additional angle
        ax   :      None or plt.Axes
            The axes on which the plot should be done, if None a new figure and axes is created
        Ang  :      str
            Flag indicating which angle to use for the impact parameter, the angle of the line itself (xi) or of its impact parameter (theta)
        AngUnit :   str
            Flag for the angle units to be displayed, 'rad' for radians or 'deg' for degrees
        Sketch :    bool
            Flag indicating whether a small skecth showing the definitions of angles 'theta' and 'xi' should be included or not
        Pdict :     dict
            Dictionary of properties used for plotting the polygon envelopp,
            fed to plt.plot() if proj='Cross' and to plt.plot_surface() if proj='3d'
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

        # Only plot cross sino, from version 1.4.0
        dP = _def.TorPFilld if dP is None else dP
        ax = _plot.Plot_Impact_PolProjPoly(self, ax=ax, Ang=Ang,
                                           AngUnit=AngUnit, Sketch=Sketch,
                                           Leg=self.Id.NameLTX, dP=dP,
                                           dLeg=dLeg, draw=False,
                                           fs=fs, wintit=wintit, Test=Test)
        # else:
        # Pdict = _def.TorP3DFilld if Pdict is None else Pdict
        # ax = _plot.Plot_Impact_3DPoly(self, ax=ax, Ang=Ang, AngUnit=AngUnit,
                                      # Pdict=Pdict, dLeg=LegDict, draw=False,
                                      # fs=fs, wintit=wintit, Test=Test)
        if draw:
            ax.figure.canvas.draw()
        return ax

    def save_to_txt(self, path='./', name=None,
                    include=['Mod','Cls','Exp','Name'],
                    fmt='%.18e', delimiter=' ',
                    footer='', encoding=None, verb=True, return_pfe=False):
        """ Save the basic geometrical attributes only (polygon and pos/extent)

        The attributes are saved to a txt file with chosen encoding
        Usefu for easily sharing input with non-python users

        BEWARE: doesn't save all attributes !!!
        Only saves the basic geometrical inputs !!!
        Not equivalent to full tofu save (using self.save()) !!!

        The saving convention is:
            * data is saved on 2 columns
            * The first line gives 2 numbers: nP, no
                - nP = Number of points in the polygon
                    (i.e.: the number of following lines describing the polygon)
                - no = Number of occurences (toroidal if in toroidal geometry)
                    (i.e.: the nb. of pos/extent lines after the first nP lines)
            * Hence, the data is a 2D array of shape (1 + nP + no, 2)
            * The two columns of the nP lines describing the polygon represent:
                - 1st: R (resp. Y) coordinate of polygon points
                - 2nd: Z (resp. Z) coordinate of polygon points
            * The two columns of the no lines representing the occurences are:
                - 1st: pos, the tor. angle (resp. X) center of occurences
                - 2nd: extent, the tor. angle (resp. X) extension of occurences

        Hence, the polygon and pos/extent of the object can be retrieved with:
        >>> import numpy as np
        >>> out = np.loadtxt(filename)
        >>> nP, no = out[0,:]
        >>> poly = out[1:1+nP,:]
        >>> pos, extent = out[1+nP:,0], out[1+nP:,1]

        All parameters apart from path, name and include are fed to numpy.savetxt()

        Parameters
        ----------
        path:   None / str
            The path where to save the file
            If None -> self.Id.SavePath
        name:   None / str
            The name to use for the saved file
            If None -> self.Id.SaveName(include)
        include:    list
            List of attributes of to be used to built the default saving name
            Fed to tf.utils.ID.generate_SaveName()
            Recommended: ['Mod','Cls','Exp','Name']
        """
        if name is None:
            name = self.Id.generate_SaveName(include)
        if path is None:
            path = self.Id.SavePath
        path = os.path.abspath(path)
        pfe = os.path.join(path,name+'.txt')

        nPno = np.r_[self.Poly.shape[1], self.noccur]
        poly = self.Poly.T
        posext = np.vstack((self.pos, self.extent)).T
        out = np.vstack((nPno,poly,posext))

        # default standards
        newline = '\n'
        comments = '#'
        header = ' Cls = %s\n Exp = %s\n Name = %s'%(self.__class__.__name__,
                                                     self.Id.Exp, self.Id.Name)

        kwds = dict(fmt=fmt, delimiter=delimiter, newline=newline,
                   header=header, footer=footer, comments=comments)
        if 'encoding' in inspect.signature(np.savetxt).parameters:
            kwds['encoding'] = encoding
        np.savetxt(pfe, out, **kwds)
        if verb:
            print("save_to_txt in:\n", pfe)
        if return_pfe:
            return pfe

    @classmethod
    def from_txt(cls, pfe, out='object',
                 Exp=None, Name=None, shot=None, Type=None,
                 mobile=False, color=None, SavePath=os.path.abspath('./')):
        """ Return the polygon and pos/extent stored in a .txt file

        The file must have been generated by the method save_to_txt()
        All arguments appart from pfe and out are:
            - fed to the relevant tofu.geom.Struct subclass to instanciate it
            - used only if out = 'object'

        Parameters
        ----------
        pfe:    str
            Unique string containing the path and file name to be read
            The file must be formatted as if generated by self.save_to_txt():
                - Must contain a (N,2) array
                - Line 0 must contain 2 integers:
                    - npts : the nb. of points of the polygon
                    - noccur : the nb. of occurences (=0 if axisymmetric)
                - Hence the number of lines hould be N = npts + noccur + 1
                - Lines 1:npts+1 contain the polygon points
                - Lines npts+1: contain positions and extent of each occurence
        out:    str
            Flag indicating whether to return:
                - 'dict'  : a dictionnary of np.ndarrays
                - 'object': a tofu.geom.Struct subclass, using the other kwdargs

        Return
        ------
        obj:    tf.geom.Struct sublass instance  / dict
            Depending on the value of out, obj can be:
                - An instance of the relevant tofu.geom.Struct subclass
                - A dict with keys 'poly', 'pos' and 'extent'
        """

        if not out in [object,'object','dict']:
            msg = "Arg out must be either:"
            msg += "    - 'object': return a %s instance\n"%cls.__name__
            msg += "    - 'dict' : return a dict with polygon, pos and extent"
            raise Exception(msg)
        if not pfe[-4:]=='.txt':
            msg = "Only accepts .txt files (fed to np.loadtxt) !\n"
            msg += "    file:  %s"%pfe
            raise Exception(msg)
        oo = np.loadtxt(pfe)
        if not (oo.ndim==2 and oo.shape[1]==2):
            msg = "The file should contain a (N,2) array !\n"
            msg += "    file : %s\n"%pfe
            msg += "    shape: {0}".format(oo.shape)
            raise Exception(msg)
        C0 = (oo[0,0]==int(oo[0,0]) and oo[0,1]==int(oo[0,1]))
        C1 = oo.shape == (oo[0,0] + oo[0,1] + 1, 2)
        if not (C0 and C1):
            sha = (oo[0,0]+oo[0,1]+1,2)
            msg = "The shape of the array is not as expected!\n"
            msg += "    file : %s\n"%pfe
            if not C0:
                msg += "  The first line should contain integers!\n"
                msg += "    First line : {0}".format(oo[0,:])
            else:
                msg += "    Expected shape: {0}".format(sha)
                msg += " = ({0} + {1} + 1, 2)\n".format(oo[0,0], oo[0,1])
                msg += "    Observed shape: {0}".format(oo.shape)
            raise Exception(msg)
        npts, noccur = int(oo[0,0]), int(oo[0,1])
        poly = oo[1:1+npts,:]
        if noccur>0:
            pos, extent = oo[1+npts:,0], oo[1+npts:,1]
        else:
            pos, extent = None, None

         # Try reading Exp and Name if not provided
        if Exp is None:
            Exp = cls._from_txt_extract_params(pfe, 'Exp')
        if Name is None:
            Name = cls._from_txt_extract_params(pfe, 'Name')

        if out=='dict':
            return {'poly':poly, 'pos':pos, 'extent':extent}
        else:
            SavePath = os.path.abspath(SavePath)
            obj = cls(Name=Name, Exp=Exp, shot=shot,
                      Type=Type, mobile=mobile,
                      Poly=poly, pos=pos, extent=extent,
                      SavePath=SavePath, color=color)
            return obj

    @staticmethod
    def _from_txt_extract_params(pfe, param):
        p, name = os.path.split(pfe)

        # Try from file name
        lk = name.split('_')
        lind = [param in k for k in lk]
        if np.sum(lind) > 1:
            msg = "Several values form %s found in file name:\n"
            msg += "    file: %s"%pfe
            raise Exception(msg)
        if any(lind):
            paramstr = lk[np.nonzero(lind)[0][0]]
            paramstr = paramstr.replace(param,'')
            return paramstr

        # try from file content
        paramstr = None
        lout = [param, '#', ':', '=', ' ', '\n', '\t']
        with open(pfe) as fid:
            while True:
                line = fid.readline()
                if param in line:
                    for k in lout:
                        line = line.replace(k,'')
                    paramstr = line
                    break
                elif not line:
                    break
        return paramstr




"""
###############################################################################
###############################################################################
                      Effective Struct subclasses
###############################################################################
"""

class StructIn(Struct):
    _color = 'k'
    _InOut = 'in'

    @classmethod
    def _set_color_ddef(cls, color):
        # super
        color = mpl.colors.to_rgba(color)
        cls._ddef['dmisc']['color'] = color
        cls._dplot['cross']['dP']['color'] = cls._ddef['dmisc']['color']
        cls._dplot['hor']['dP']['color'] = cls._ddef['dmisc']['color']
        cls._dplot['3d']['dP']['color'] = cls._ddef['dmisc']['color']

    @staticmethod
    def _checkformat_inputs_dgeom(Poly=None, Lim=None,
                                  pos=None, extent=None, mobile=False,
                                  Type=None, Clock=False, arrayorder=None):
        kwdargs = locals()
        # super
        out = Struct._checkformat_inputs_dgeom(**kwdargs)
        Poly, pos, extent, Type, arrayorder = out
        if Type=='Tor':
            msg = "StructIn subclasses cannot have noccur>0 if Type='Tor'!"
            assert pos.size==0, msg
        return out


class StructOut(Struct):
    _color = (0.8,0.8,0.8,0.8)
    _InOut = 'out'

    @classmethod
    def _set_color_ddef(cls, color):
        color = mpl.colors.to_rgba(color)
        cls._ddef['dmisc']['color'] = color
        cls._dplot['cross']['dP'] = {'fc':color, 'ec':'k','linewidth':1}
        cls._dplot['hor']['dP'] = {'fc':color, 'ec':'none'}
        cls._dplot['3d']['dP']['color'] = color

    def _set_color(self, color=None):
        color = self._checkformat_inputs_dmisc(color=color)
        self._dmisc['color'] = color
        self._dplot['cross']['dP']['fc'] = color
        self._dplot['hor']['dP']['fc'] = color
        self._dplot['3d']['dP']['color'] = color

    def get_sampleV(self, *args, **kwdargs):
        msg = "StructOut subclasses cannot use get_sampleV()!"
        raise Exception(msg)

class PlasmaDomain(StructIn):
    _color = (0.8,0.8,0.8,1.)

class Ves(StructIn):
    _color = 'k'

class PFC(StructOut):
    _color = (0.8,0.8,0.8,0.8)


class CoilPF(StructOut):
    _color = 'r'

    def __init__(self, Poly=None, Type=None, Lim=None, pos=None, extent=None,
                 Id=None, Name=None, Exp=None, shot=None,
                 sino_RefPt=None, sino_nP=_def.TorNP,
                 Clock=False, arrayorder='C', fromdict=None,
                 nturns=None, superconducting=None, active=None,
                 SavePath=os.path.abspath('./'),
                 SavePath_Include=tfpf.defInclude, color=None):
        kwdargs = locals()
        del kwdargs['self'], kwdargs['__class__']
        # super()
        super(CoilPF,self).__init__(mobile=False, **kwdargs)

    def __init__(self, nturns=None, superconducting=None, active=None,
                 **kwdargs):
        # super()
        super(CoilPF,self).__init__(**kwdargs)

    def _reset(self):
        # super()
        super(CoilPF,self)._reset()
        self._dmag = dict.fromkeys(self._get_keys_dmag())
        self._dmag['nI'] = 0

    ###########
    # Get largs
    ###########

    @staticmethod
    def _get_largs_dmag():
        largs = ['nturns','superconducting','active']
        return largs

    ###########
    # Get check and format inputs
    ###########

    @classmethod
    def _checkformat_inputs_dmag(cls, nturns=None, superconducting=None, active=None):
        dins = {'nturn':{'var':nturns, 'NoneOrIntPos':None},
                'superconducting':{'var':superconducting, 'NoneOrCls':bool},
                'active':{'var':active, 'NoneOrCls':bool}}
        dins, err, msg = cls._check_InputsGeneric(dins, tab=0)
        if err:
            raise Exception(msg)
        nturn = dins['nturn']['var']
        return nturn

    ###########
    # Get keys of dictionnaries
    ###########

    @staticmethod
    def _get_keys_dmag():
        lk = ['nturns','superconducting','active','I','nI']
        return lk

    ###########
    # _init
    ###########

    def _init(self, nturns=None, superconducting=None, active=None, **kwdargs):
        super(CoilPF,self)._init(**kwdargs)
        self.set_dmag(nturns=nturns, superconducting=superconducting,
                      active=active)


    ###########
    # set dictionaries
    ###########

    def set_dmag(self, superconducting=None, nturns=None, active=None):
        nturns = self._checkformat_inputs_dmag(nturns=nturns, active=active,
                                                superconducting=superconducting)
        self._dmag.update({'superconducting':superconducting,
                           'nturns':nturns, 'active':active})

    ###########
    # strip dictionaries
    ###########

    def _strip_dmag(self, lkeep=['nturns','superconducting','active']):
        utils.ToFuObject._strip_dict(self._dmag, lkeep=lkeep)
        self._dmag['nI'] = 0

    ###########
    # rebuild dictionaries
    ###########

    def _rebuild_dmag(self, lkeep=['nturns','superconducting','active']):
        self.set_dmag(nturns=self.nturns, active=self._dmag['active'],
                      superconducting=self._dmag['superconducting'])

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
        if sys.version[0]=='2':
            cls.strip.__func__.__doc__ = doc
        else:
            cls.strip.__doc__ = doc

    def strip(self, strip=0):
        super(CoilPF, self).strip(strip=strip)

    def _strip(self, strip=0):
        out = super(CoilPF, self)._strip(strip=strip)
        if strip==0:
            self._rebuild_dmag()
        else:
            self._strip_dmag()
        return out

    def _to_dict(self):
        dout = super(CoilPF,self)._to_dict()
        dout.update({'dmag':{'dict':self.dmag, 'lexcept':None}})
        return dout

    def _from_dict(self, fd):
        super(CoilPF,self)._from_dict(fd)
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


class CoilCS(CoilPF): pass



"""
###############################################################################
###############################################################################
                        Overall Config object
###############################################################################
"""

class Config(utils.ToFuObject):


    # Special dict subclass with attr-like value access


    # Fixed (class-wise) dictionary of default properties
    _ddef = {'Id':{'shot':0, 'Type':'Tor', 'Exp':'Dummy',
                   'include':['Mod','Cls','Exp',
                              'Name','shot','version']},
             'dStruct':{'order':['PlasmaDomain','Ves','PFC','CoilPF','CoilCS'],
                        'dextraprop':{'visible':True}}}
    _lclsstr = ['PlasmaDomain','Ves','PFC','CoilPF','CoilCS']


    def __init__(self, lStruct=None, Lim=None, dextraprop=None,
                 Id=None, Name=None, Exp=None, shot=None, Type=None,
                 SavePath=os.path.abspath('./'),
                 SavePath_Include=tfpf.defInclude,
                 fromdict=None):

        # To replace __init_subclass__ for Python 2
        if sys.version[0]=='2':
            self._dstrip = utils.ToFuObjectBase._dstrip.copy()
            self.__class__._strip_init()

        kwdargs = locals()
        del kwdargs['self']
        super(Config,self).__init__(**kwdargs)

    def _reset(self):
        super(Config,self)._reset()
        self._dStruct = dict.fromkeys(self._get_keys_dStruct())
        self._dextraprop = dict.fromkeys(self._get_keys_dextraprop())
        self._dsino = dict.fromkeys(self._get_keys_dsino())

    @classmethod
    def _checkformat_inputs_Id(cls, Id=None, Name=None, Type=None,
                               Exp=None, shot=None, include=None, **kwdargs):
        if Id is not None:
            assert isinstance(Id,utils.ID)
            Name, shot = Id.Name, Id.shot
        if Type is None:
            Type = cls._ddef['Id']['Type']
        if Exp is None:
            Exp = cls._ddef['Id']['Exp']
        if shot is None:
            shot = cls._ddef['Id']['shot']
        if include is None:
            include = cls._ddef['Id']['include']

        dins = {'Name':{'var':Name, 'cls':str},
                'Type':{'var':Type, 'in':['Tor','Lin']},
                'Exp':{'var':Exp, 'cls':str},
                'shot':{'var':shot, 'cls':int},
                'include':{'var':include, 'listof':str}}
        dins, err, msg = cls._check_InputsGeneric(dins, tab=0)
        if err:
            raise Exception(msg)
        kwdargs.update({'Name':Name, 'Type':Type, 'Exp':Exp,
                        'include':include, 'shot':shot})

        return kwdargs

    ###########
    # Get largs
    ###########

    @staticmethod
    def _get_largs_dStruct():
        largs = ['lStruct', 'Lim']
        return largs
    @staticmethod
    def _get_largs_dextraprop():
        largs = ['dextraprop']
        return largs
    @staticmethod
    def _get_largs_dsino():
        largs = ['RefPt','nP']
        return largs

    ###########
    # Get check and format inputs
    ###########

    def _checkformat_inputs_Struct(self, struct, err=True):
        assert issubclass(struct.__class__,Struct)
        C0 = struct.Id.Exp==self.Id.Exp
        C1 = struct.Id.Type==self.Id.Type
        if sys.version[0]=='2':
            C2 = (re.match(tokenize.Name + '$', struct.Id.Name)
                  and not keyword.iskeyword(struct.Id.Name))
        else:
            C2 = struct.Id.Name.isidentifier()
        C2 = C2 and '_' not in struct.Id.Name
        msgi = None
        if not (C0 and C1 and C2):
            msgi = "\n    - {0} :".format(struct.Id.SaveName)
            if not C0:
                msgi += "\n     Exp: {0}".format(struct.Id.Exp)
            if not C1:
                msgi += "\n     Type: {0}".format(struct.Id.Type)
            if not C2:
                msgi += "\n     Name: {0}".format(struct.Id.Name)
            if err:
                msg = "Non-conform struct Id:"+msgi
                raise Exception(msg)
        return msgi


    def _checkformat_inputs_dStruct(self, lStruct=None, Lim=None):
        if lStruct is None:
            msg = "Arg lStruct must be"
            msg += " a tofu.geom.Struct subclass or a list of such !"
            msg += "\nValid subclasses include:"
            lsub = ['PlasmaDomain','Ves','PFC','CoilPF','CoilCS']
            for ss in lsub:
                msg = "\n    - tf.geom.{0}".format(ss)
            raise Exception(msg)

        C0 = isinstance(lStruct,list) or isinstance(lStruct,tuple)
        C1 = issubclass(lStruct.__class__,Struct)
        assert C0 or C1, msg
        if C0:
            Ci = [issubclass(ss.__class__,Struct) for ss in lStruct]
            assert all(Ci), msg
            lStruct = list(lStruct)
        else:
            lStruct = [lStruct]

        msg = ""
        for ss in lStruct:
            msgi = self._checkformat_inputs_Struct(ss, err=False)
            if msgi is not None:
                msg += msgi
        if msg!="":
            msg = "The following objects have non-confrom Id:" + msg
            msg += "\n  => Expected values are:"
            msg += "\n      Exp: {0}".format(self.Id.Exp)
            msg += "\n      Type: {0}".format(self.Id.Type)
            msg += "\n      Name: a valid identifier, without '_'"
            msg += " (check str.isidentifier())"
            raise Exception(msg)

        if Lim is None:
            if not self.Id.Type=='Tor':
                msg = "Issue with tf.geom.Config {0}:".format(self.Id.Name)
                msg += "\n  If input Lim is None, Type should be 'Tor':"
                msg += "\n    Type = {0}".format(self.Id.Type)
                msg += "\n    Lim = {0}".format(str(Lim))
                raise Exception(msg)
            nLim = 0
        else:
            if not self.Id.Type=='Lin':
                msg = "Issue with tf.geom.Config {0}:".format(self.Id.Name)
                msg = "  If input Lim!=None, Type should be 'Lin':"
                msg += "\n    Type = {0}".format(self.Id.Type)
                msg += "\n    Lim = {0}".format(str(Lim))
                raise Exception(msg)
            Lim = np.asarray(Lim,dtype=float).ravel()
            assert Lim.size==2 and Lim[0]<Lim[1]
            Lim = Lim.reshape((1,2))
            nLim = 1

        return lStruct, Lim, nLim

    def _checkformat_inputs_extraval(self, extraval, key='',
                                     multi=True, size=None):
        lsimple = [bool,float,int,np.int64,np.float64]
        C0 = type(extraval) in lsimple
        C1 = isinstance(extraval,np.ndarray)
        C2 = isinstance(extraval,dict)
        if multi:
            assert C0 or C1 or C2, str(type(extraval))
        else:
            assert C0, str(type(extraval))
        if multi and C1:
            size = self._dStruct['nStruct'] if size is None else size
            C = extraval.shape==((self._dStruct['nObj'],))
            if not C:
                msg = "The value for %s has wrong shape!"%key
                msg += "\n    Expected: ({0},)".format(self._dStruct['nObj'])
                msg += "\n    Got:      {0}".format(str(extraval.shape))
                raise Exception(msg)
            C = np.ndarray
        elif multi and C2:
            msg0 = "If an extra attribute is provided as a dict,"
            msg0 += " it should have the same structure as self.dStruct !"
            lk = sorted(self._dStruct['lCls'])
            c = lk==sorted(extraval.keys())
            if not c:
                msg = "\nThe value for %s has wrong keys !"%key
                msg += "\n    expected : "+str(lk)
                msg += "\n    received : "+str(sorted(extraval.keys()))
                raise Exception(msg0+msg)
            c = [isinstance(extraval[k],dict) for k in lk]
            if not all(c):
                msg = "\nThe value for %s shall be a dict of nested dict !"%key
                msg += "\n    "
                msg += "\n    ".join(['{0} : {1}'.format(lk[ii],c[ii])
                                     for ii in range(0,len(lk))])
                raise Exception(msg0+msg)
            c = [(k, sorted(v.keys()), sorted(self.dStruct['dObj'][k].keys()))
                 for k, v in extraval.items()]
            if not all([cc[1]==cc[2] for cc in c]):
                lc = [(cc[0], str(cc[1]), str(cc[2])) for cc in c if cc[1]!=cc[2]]
                msg = "\nThe value for %s has wrong nested dict !"%key
                msg += "\n    - " + '\n    - '.join([' '.join(cc)
                                                     for cc in lc])
                raise Exception(msg0+msg)
            for k in lk:
                for kk,v in extraval[k].items():
                    if not type(v) in lsimple:
                        msg = "\n    type(%s[%s][%s])"%(key,k,kk)
                        msg += " = %s"%str(type(v))
                        msg += " should be in %s"%str(lsimple)
                        raise Exception(msg)
            C = dict
        elif C0:
            C = int
        return C

    def _checkformat_inputs_dextraprop(self, dextraprop=None):
        if dextraprop is None:
            dextraprop = self._ddef['dStruct']['dextraprop']
        if dextraprop is None:
            dextraprop = {}
        assert isinstance(dextraprop,dict)
        dC = {}
        for k in dextraprop.keys():
            dC[k] = self._checkformat_inputs_extraval(dextraprop[k], key=k)
        return dextraprop, dC

    def _checkformat_inputs_dsino(self, RefPt=None, nP=None):
        assert type(nP) is int and nP>0
        assert hasattr(RefPt,'__iter__')
        RefPt = np.asarray(RefPt,dtype=float).flatten()
        assert RefPt.size==2, "RefPt must be of size=2 !"
        return RefPt

    ###########
    # Get keys of dictionnaries
    ###########

    @staticmethod
    def _get_keys_dStruct():
        lk = ['dObj', 'Lim', 'nLim',
              'nObj','lorder','lCls']
        return lk

    @staticmethod
    def _get_keys_dextraprop():
        lk = ['lprop']
        return lk

    @staticmethod
    def _get_keys_dsino():
        lk = ['RefPt','nP']
        return lk

    ###########
    # _init
    ###########

    def _init(self, lStruct=None, Lim=None, dextraprop=None, **kwdargs):
        largs = self._get_largs_dStruct()
        kwdStruct = self._extract_kwdargs(locals(), largs)
        largs = self._get_largs_dextraprop()
        kwdextraprop = self._extract_kwdargs(locals(), largs)
        self._set_dStruct(**kwdStruct)
        self._set_dextraprop(**kwdextraprop)
        self._dynamicattr()
        self._dstrip['strip'] = 0

    ###########
    # set dictionaries
    ###########


    def _set_dStruct(self, lStruct=None, Lim=None):
        lStruct, Lim, nLim = self._checkformat_inputs_dStruct(lStruct=lStruct,
                                                              Lim=Lim)
        self._dStruct.update({'Lim':Lim, 'nLim':nLim})
        self._set_dlObj(lStruct, din=self._dStruct)


    def _set_dextraprop(self, dextraprop=None):
        dextraprop, dC = self._checkformat_inputs_dextraprop(dextraprop)
        self._dextraprop['lprop'] = sorted(list(dextraprop.keys()))

        # Init dict
        lCls = self._dStruct['lCls']
        for pp in dextraprop.keys():
            dp = 'd'+pp
            dd = dict.fromkeys(lCls,{})
            for k in lCls:
                dd[k] = dict.fromkeys(self._dStruct['dObj'][k].keys())
            self._dextraprop.update({dp:dd})

        # Populate
        for pp in dextraprop.keys():
            self._set_extraprop(pp, dextraprop[pp])


    def add_extraprop(self, key, val):
        assert type(key) is str
        d, dC = self._checkformat_inputs_dextraprop({key:val})
        self._dextraprop['lprop'] = sorted(set(self.dextraprop['lprop']+[key]))

        # Init dict
        lCls = self._dStruct['lCls']
        dp = 'd'+key
        dd = dict.fromkeys(lCls,{})
        for k in lCls:
            dd[k] = dict.fromkeys(self._dStruct['dObj'][k].keys())
        self._dextraprop.update({dp:dd})

        # Populate
        self._set_extraprop(key, val)
        self._dynamicattr()

    def _set_extraprop(self, pp, val, k0=None, k1=None):
        assert not (k0 is None and k1 is not None)
        dp = 'd'+pp
        if k0 is None and k1 is None:
            C = self._checkformat_inputs_extraval(val, pp)
            if C is int:
                for k0 in self._dStruct['dObj'].keys():
                    for k1 in self._dextraprop[dp][k0].keys():
                        self._dextraprop[dp][k0][k1] = val
            elif C is np.ndarray:
                ii = 0
                for k in self._dStruct['lorder']:
                    k0, k1 = k.split('_')
                    self._dextraprop[dp][k0][k1] = val[ii]
                    ii += 1
            else:
                for k0 in self._dStruct['dObj'].keys():
                    for k1 in self._dextraprop[dp][k0].keys():
                        self._dextraprop[dp][k0][k1] = val[k0][k1]
        elif k1 is None:
            size = len(self._dextraprop[dp][k0].keys())
            C = self._checkformat_inputs_extraval(val, pp, size=size)
            assert C in [int,np.ndarray]
            if C is int:
                for k1 in self._dextraprop[dp][k0].keys():
                    self._dextraprop[dp][k0][k1] = val
            elif C is np.ndarray:
                ii = 0
                for k in self._dStruct['lorder']:
                    kk, k1 = k.split('_')
                    if k0==kk:
                        self._dextraprop[dp][k0][k1] = val[ii]
                        ii += 1
        else:
            C = self._checkformat_inputs_extraval(val, pp, multi=False)
            assert C is int
            self._dextraprop[dp][k0][k1] = val

    def _get_extraprop(self, pp, k0=None, k1=None):
        assert not (k0 is None and k1 is not None)
        dp = 'd'+pp
        if k0 is None and k1 is None:
            val = np.zeros((self._dStruct['nObj'],),dtype=bool)
            ii = 0
            for k in self._dStruct['lorder']:
                k0, k1 = k.split('_')
                val[ii] = self._dextraprop[dp][k0][k1]
                ii += 1
        elif k1 is None:
            val = np.zeros((len(self._dStruct['dObj'][k0].keys()),),dtype=bool)
            ii = 0
            for k in self._dStruct['lorder']:
                k, k1 = k.split('_')
                if k0==k:
                    val[ii] = self._dextraprop[dp][k0][k1]
                    ii += 1
        else:
            val = self._dextraprop[dp][k0][k1]
        return val

    def _set_color(self, k0, val):
        for k1 in self._dStruct['dObj'][k0].keys():
            self._dStruct['dObj'][k0][k1].set_color(val)

    def _dynamicattr(self):
        # get (key, val) pairs

        # Purge
        for k in self._ddef['dStruct']['order']:
            if hasattr(self,k):
                delattr(self,k)
                # if sys.version[0]=='2':
                    # exec "del self.{0}".format(k) in locals()
                # else:
                    # exec("del self.{0}".format(k))

        # Set
        for k in self._dStruct['dObj'].keys():
            # Find a way to programmatically add dynamic properties to the
            # instances , like visible
            # In the meantime use a simple functions
            lset = ['set_%s'%pp for pp in self._dextraprop['lprop']]
            lget = ['get_%s'%pp for pp in self._dextraprop['lprop']]
            if not type(list(self._dStruct['dObj'][k].values())[0]) is str:
                for kk in self._dStruct['dObj'][k].keys():
                    for pp in self._dextraprop['lprop']:
                        setattr(self._dStruct['dObj'][k][kk],
                                'set_%s'%pp,
                                lambda val, pk=pp, k0=k, k1=kk: self._set_extraprop(pk, val, k0, k1))
                        setattr(self._dStruct['dObj'][k][kk],
                                'get_%s'%pp,
                                lambda pk=pp, k0=k, k1=kk: self._get_extraprop(pk, k0, k1))
                dd = utils.Dictattr(['set_color']+lset+lget,
                                    self._dStruct['dObj'][k])
                for pp in self._dextraprop['lprop']:
                    setattr(dd,
                            'set_%s'%pp,
                            lambda val, pk=pp, k0=k: self._set_extraprop(pk, val, k0))
                    setattr(dd,
                            'get_%s'%pp,
                            lambda pk=pp, k0=k: self._get_extraprop(pk, k0))
                setattr(dd,
                        'set_color',
                        lambda col, k0=k: self._set_color(k0, col))
                setattr(self, k, dd)
        for pp in self._dextraprop['lprop']:
            setattr(self, 'set_%s'%pp,
                    lambda val, pk=pp: self._set_extraprop(pk,val))
            setattr(self, 'get_%s'%pp,
                    lambda pk=pp: self._get_extraprop(pk))

    def set_dsino(self, RefPt, nP=_def.TorNP):
        RefPt = self._checkformat_inputs_dsino(RefPt=RefPt, nP=nP)
        for k in self._dStruct['dObj'].keys():
            for kk in self._dStruct['dObj'][k].keys():
                self._dStruct['dObj'][k][kk].set_dsino(RefPt=RefPt, nP=nP)
        self._dsino = {'RefPt':RefPt, 'nP':nP}


    ###########
    # strip dictionaries
    ###########

    def _strip_dStruct(self, strip=0, force=False, verb=True):
        if self._dstrip['strip'] == strip:
            return

        if self._dstrip['strip'] > strip:

            # Reload if necessary
            if self._dstrip['strip'] == 3:
                for k in self._dStruct['dObj'].keys():
                    for kk in self._dStruct['dObj'][k].keys():
                        pfe = self._dStruct['dObj'][k][kk]
                        try:
                            self._dStruct['dObj'][k][kk] = utils.load(pfe,
                                                                      verb=verb)
                        except Exception as err:
                            msg = str(err)
                            msg += "\n    k = {0}".format(str(k))
                            msg += "\n    kk = {0}".format(str(kk))
                            msg += "\n    type(pfe) = {0}".format(str(type(pfe)))
                            msg += "\n    self._dstrip['strip'] = {0}".format(self._dstrip['strip'])
                            msg += "\n    strip = {0}".format(strip)
                            raise Exception(msg)

            for k in self._dStruct['dObj'].keys():
                for kk in self._dStruct['dObj'][k].keys():
                    self._dStruct['dObj'][k][kk].strip(strip=strip)

            lkeep = self._get_keys_dStruct()
            reset = utils.ToFuObject._test_Rebuild(self._dStruct, lkeep=lkeep)
            if reset:
                utils.ToFuObject._check_Fields4Rebuild(self._dStruct,
                                                       lkeep=lkeep,
                                                       dname='dStruct')
            self._set_dStruct(lStruct=self.lStruct, Lim=self._dStruct['Lim'])
            self._dynamicattr()

        else:
            if strip in [1,2]:
                for k in self._dStruct['lCls']:
                    for kk, v  in self._dStruct['dObj'][k].items():
                        self._dStruct['dObj'][k][kk].strip(strip=strip)
                lkeep = self._get_keys_dStruct()

            elif strip == 3:
                for k in self._dStruct['lCls']:
                    for kk, v  in self._dStruct['dObj'][k].items():
                        path, name = v.Id.SavePath, v.Id.SaveName
                        # --- Check !
                        lf = os.listdir(path)
                        lf = [ff for ff in lf
                              if all([s in ff for s in [name,'.npz']])]
                        exist = len(lf)==1
                        # ----------
                        pathfile = os.path.join(path, name)+'.npz'
                        if not exist:
                            msg = """BEWARE:
                                You are about to delete the Struct objects
                                Only the path/name to saved objects will be kept

                                But it appears that the following object has no
                                saved file where specified (obj.Id.SavePath)
                                Thus it won't be possible to retrieve it
                                (unless available in the current console:"""
                            msg += "\n    - {0}".format(pathfile)
                            if force:
                                warning.warn(msg)
                            else:
                                raise Exception(msg)
                        self._dStruct['dObj'][k][kk] = pathfile
                self._dynamicattr()
                lkeep = self._get_keys_dStruct()
            utils.ToFuObject._strip_dict(self._dStruct, lkeep=lkeep)

    def _strip_dextraprop(self, strip=0):
        lkeep = list(self._dextraprop.keys())
        utils.ToFuObject._strip_dict(self._dextraprop, lkeep=lkeep)

    def _strip_dsino(self, lkeep=['RefPt','nP']):
        for k in self._dStruct['dObj'].keys():
            for kk in self._dStruct['dObj'][k].keys():
                self._dStruct['dObj'][k][kk]._strip_dsino(lkeep=lkeep)

    ###########
    # _strip and get/from dict
    ###########

    @classmethod
    def _strip_init(cls):
        cls._dstrip['allowed'] = [0,1,2,3]
        nMax = max(cls._dstrip['allowed'])
        doc = """
                 1: apply strip(1) to objects in self.lStruct
                 2: apply strip(2) to objects in self.lStruct
                 3: replace objects in self.lStruct by their SavePath+SaveName"""
        doc = utils.ToFuObjectBase.strip.__doc__.format(doc,nMax)
        if sys.version[0]=='2':
            cls.strip.__func__.__doc__ = doc
        else:
            cls.strip.__doc__ = doc

    def strip(self, strip=0, force=False, verb=True):
        # super()
        super(Config,self).strip(strip=strip, force=force, verb=verb)

    def _strip(self, strip=0, force=False, verb=True):
        self._strip_dStruct(strip=strip, force=force, verb=verb)
        #self._strip_dextraprop()
        #self._strip_dsino()

    def _to_dict(self):
        dout = {'dStruct':{'dict':self.dStruct, 'lexcept':None},
                'dextraprop':{'dict':self._dextraprop, 'lexcept':None},
                'dsino':{'dict':self.dsino, 'lexcept':None}}
        return dout

    @classmethod
    def _checkformat_fromdict_dStruct(cls, dStruct):
        if dStruct['lorder'] is None:
            return None
        for clsn in dStruct['lorder']:
            c, n = clsn.split('_')
            if type(dStruct['dObj'][c][n]) is dict:
                dStruct['dObj'][c][n]\
                        = eval(c).__call__(fromdict=dStruct['dObj'][c][n])
            lC = [issubclass(dStruct['dObj'][c][n].__class__,Struct),
                  type(dStruct['dObj'][c][n]) is str]
            assert any(lC)

    def _from_dict(self, fd):
        self._checkformat_fromdict_dStruct(fd['dStruct'])

        self._dStruct.update(**fd['dStruct'])
        self._dextraprop.update(**fd['dextraprop'])
        self._dsino.update(**fd['dsino'])
        self._dynamicattr()


    ###########
    # Properties
    ###########

    @property
    def dStruct(self):
       return self._dStruct
    @property
    def nStruct(self):
       return self._dStruct['nObj']
    @property
    def lStruct(self):
        """ Return the list of Struct that was used for creation

        As tofu objects or SavePath+SaveNames (according to strip status)
        """
        lStruct = []
        for k in self._dStruct['lorder']:
            k0, k1 = k.split('_')
            lStruct.append(self._dStruct['dObj'][k0][k1])
        return lStruct

    @property
    def lStructIn(self):
        """ Return the list of StructIn contained in self.lStruct

        As tofu objects or SavePath+SaveNames (according to strip status)
        """
        lStruct = []
        for k in self._dStruct['lorder']:
            k0, k1 = k.split('_')
            if type(self._dStruct['dObj'][k0][k1]) is str:
                if any([ss in self._dStruct['dObj'][k0][k1]
                        for ss in ['Ves','PlasmaDomain']]):
                    lStruct.append(self._dStruct['dObj'][k0][k1])
            elif issubclass(self._dStruct['dObj'][k0][k1].__class__, StructIn):
                lStruct.append(self._dStruct['dObj'][k0][k1])
        return lStruct

    @property
    def Lim(self):
        return self._dStruct['Lim']
    @property
    def nLim(self):
        return self._dStruct['nLim']

    @property
    def dextraprop(self):
       return self._dextraprop
    @property
    def dsino(self):
       return self._dsino

    ###########
    # public methods
    ###########

    def add_Struct(self, struct=None,
                   Cls=None, Name=None, Poly=None,
                   mobile=False, shot=None,
                   Lim=None, Type=None,
                   dextraprop=None):
        """ Add a Struct instance to the config

        An already existing Struct subclass instance can be added
        Or it will be created from the (Cls,Name,Poly,Lim) keyword args

        """
        # Check inputs
        C0a = struct is None
        C1a = all([ss is None for ss in [Cls,Name,Poly,Lim,Type]])
        if not np.sum([C0a,C1a])==1:
            msg = "Provide either:"
            msg += "\n    - struct: a Struct subclass instance"
            msg += "\n    - the keyword args to create one"
            msg += "\n        (Cls,Name,Poly,Lim,Type)\n"
            msg += "\n You provded:"
            msg += "\n    - struct: {0}, {1}".format(str(struct),
                                                     type(struct))
            raise Exception(msg)

        # Create struct if not provided
        if C0a:
            if not (type(Cls) is str or issubclass(Cls,Struct)):
                msg = "Cls must be either:"
                msg += "\n    - a Struct subclass"
                msg += "\n    - the str Name of it (e.g.: 'PFC','CoilPF',...)"
                raise Exception(msg)
            if type(Cls) is str:
                Cls = eval('%s'%Cls)

            # Preformat Lim and Type
            if Lim is None:
                Lim = self.Lim
            if Type is None:
                Type = self.Id.Type

            # Create instance
            struct = Cls(Poly=Poly, Name=Name, Lim=Lim, Type=Type,
                         mobile=mobile, shot=shot, Exp=self.Id.Exp)

        C0b = issubclass(struct.__class__, Struct)
        assert C0b, "struct must be a Struct subclass instance !"

        # Prepare dextraprop
        dextra = self.dextraprop
        lk = sorted([k[1:] for k in dextra.keys() if k!='lprop'])
        if dextraprop is None:
            if not dextra in [None,{}]:
                msg = "The current Config instance has the following extraprop:"
                msg += "\n    - " + "\n    - ".join(lk)
                msg += "\n  => Please specify a dextraprop for struct !"
                msg += "\n     (using the same keys !)"
                raise Exception(msg)
        else:
            assert isinstance(dextraprop,dict)
            assert all([k in lk for k in dextraprop.keys()])
            assert all([k in dextraprop.keys() for k in lk])
            dx = {}
            for k in lk:
                dk = 'd'+k
                dx[k] = {}
                for k0 in dextra[dk].keys():
                    dx[k][k0] = {}
                    for k1 in dextra[dk][k0].keys():
                        dx[k][k0][k1] = dextra[dk][k0][k1]
                if not struct.Id.Cls in dx[k].keys():
                    dx[k][struct.Id.Cls] = {struct.Id.Name:dextraprop[k]}
                else:
                    dx[k][struct.Id.Cls][struct.Id.Name] = dextraprop[k]

        # Set self.lStruct
        lS = self.lStruct + [struct]
        self._init(lStruct=lS, Lim=self.Lim, dextraprop=dx)

    def remove_Struct(self, Cls=None, Name=None):
        # Check inputs
        assert type(Cls) is str
        assert type(Name) is str
        C0 = Cls in self._dStruct['lCls']
        if not C0:
            msg = "The Cls must be a class existing in self.dStruct['lCls']:"
            msg += "\n    [{0}]".format(', '.join(self._dStruct['lCls']))
            raise Exception(msg)
        C0 = Name in self._dStruct['dObj'][Cls].keys()
        if not C0:
            ln = self.dStruct['dObj'][Cls].keys()
            msg = "The Name must match an instance in"
            msg += " self.dStruct['dObj'][{0}].keys():".format(Cls)
            msg += "\n    [{0}]".format(', '.join(ln))
            raise Exception(msg)

        # Create list
        lS = self.lStruct
        if not Cls+"_"+Name in self._dStruct['lorder']:
            msg = "The desired instance is not in self.dStruct['lorder'] !"
            lord = ', '.join(self.dStruct['lorder'])
            msg += "\n    lorder = [{0}]".format(lord)
            msg += "\n    Cls_Name = {0}".format(Cls+'_'+Name)
            raise Exception(msg)

        ind = self._dStruct['lorder'].index(Cls+"_"+Name)
        del lS[ind]
        # Important : also remove from dict ! (no reset() !)
        del self._dStruct['dObj'][Cls][Name]

        # Prepare dextraprop
        dextra = self.dextraprop
        dx = {}
        for k in dextra.keys():
            if k=='lprop':
                continue
            dx[k[1:]] = {}
            for cc in dextra[k].keys():
                dx[k[1:]][cc] = dict(dextra[k][cc])
            del dx[k[1:]][Cls][Name]

        self._init(lStruct=lS, Lim=self.Lim, dextraprop=dx)


    def get_color(self):
        """ Return the array of rgba colors (same order as lStruct) """
        col = np.full((self._dStruct['nObj'],4), np.nan)
        ii = 0
        for k in self._dStruct['lorder']:
            k0, k1 = k.split('_')
            col[ii,:] = self._dStruct['dObj'][k0][k1].get_color()
            ii += 1
        return col

    def get_summary(self, verb=False, max_columns=100, width=1000):
        """ Summary description of the object content as a pandas DataFrame """
        # Make sure the data is accessible
        msg = "The data is not accessible because self.strip(2) was used !"
        assert self._dstrip['strip']<2, msg

        # Build the list
        d = self._dStruct['dObj']
        data = []
        for k in self._ddef['dStruct']['order']:
            if k not in d.keys():
                continue
            for kk in d[k].keys():
                lu = [k,
                      self._dStruct['dObj'][k][kk]._Id._dall['Name'],
                      self._dStruct['dObj'][k][kk]._Id._dall['SaveName'],
                      self._dStruct['dObj'][k][kk]._dgeom['nP'],
                      self._dStruct['dObj'][k][kk]._dgeom['noccur'],
                      self._dStruct['dObj'][k][kk]._dgeom['mobile'],
                      self._dStruct['dObj'][k][kk]._dmisc['color']]
                for pp in self._dextraprop['lprop']:
                    lu.append(self._dextraprop['d'+pp][k][kk])
                data.append(lu)

        # Build the pandas DataFrame
        col = ['class', 'Name', 'SaveName', 'nP', 'noccur',
               'mobile', 'color'] + self._dextraprop['lprop']
        df = pd.DataFrame(data, columns=col)
        pd.set_option('display.max_columns',max_columns)
        pd.set_option('display.width',width)

        if verb:
            print(df)
        return df

    def isInside(self, pts, In='(X,Y,Z)', log='any'):
        """ Return a 2D array of bool

        Equivalent to applying isInside to each Struct
        Check self.lStruct[0].isInside? for details

        Arg log determines how Struct with multiple Limits are treated
            - 'all' : True only if pts belong to all elements
            - 'any' : True if pts belong to any element
        """
        msg = "Arg pts must be a 1D or 2D np.ndarray !"
        assert isinstance(pts,np.ndarray) and pts.ndim in [1,2], msg
        msg = "Arg log must be in ['any','all']"
        assert log in ['any','all'], msg
        if pts.ndim==1:
            msg = "Arg pts must contain the coordinates of a point !"
            assert pts.size in [2,3], msg
            pts = pts.reshape((pts.size,1)).astype(float)
        else:
            msg = "Arg pts must contain the coordinates of points !"
            assert pts.shape[0] in [2,3], pts
        nP = pts.shape[1]

        ind = np.zeros((self._dStruct['nObj'],nP), dtype=bool)
        lStruct = self.lStruct
        for ii in range(0,self._dStruct['nObj']):
            indi = _GG._Ves_isInside(pts,
                                     lStruct[ii].Poly,
                                     Lim=lStruct[ii].Lim,
                                     nLim=lStruct[ii].noccur,
                                     VType=lStruct[ii].Id.Type,
                                     In=In, Test=True)
            if lStruct[ii].noccur>1:
                if log=='any':
                    indi = np.any(indi,axis=0)
                else:
                    indi = np.all(indi,axis=0)
            ind[ii,:] = indi
        return ind

    def plot(self, lax=None, proj='all', element='P', dLeg=_def.TorLegd,
             indices=False, Lim=None, Nstep=None,
             draw=True, fs=None, wintit=None, tit=None, Test=True):
        assert tit is None or isinstance(tit,str)
        vis = self.get_visible()
        lStruct, lS = self.lStruct, []
        for ii in range(0,self._dStruct['nObj']):
            if vis[ii]:
                lS.append(lStruct[ii])

        if tit is None:
            tit = self.Id.Name
        lax = _plot.Struct_plot(lS, lax=lax, proj=proj, element=element,
                                Lim=Lim, Nstep=Nstep,
                                dLeg=dLeg, draw=draw, fs=fs, indices=indices,
                                wintit=wintit, tit=tit, Test=Test)
        return lax


    def plot_sino(self, ax=None, dP=None,
                  Ang=_def.LOSImpAng, AngUnit=_def.LOSImpAngUnit,
                  Sketch=True, dLeg=_def.TorLegd,
                  draw=True, fs=None, wintit=None, tit=None, Test=True):

        msg = "Set the sino params before plotting !"
        msg += "\n    => run self.set_sino(...)"
        assert self.dsino['RefPt'] is not None, msg
        assert tit is None or isinstance(tit,str)
        # Check uniformity of sinogram parameters
        for ss in self.lStruct:
            msg = "{0} {1} has different".format(ss.Id.Cls, ss.Id.Name)
            msgf = "\n    => run self.set_sino(...)"
            msg0 = msg+" sino RefPt"+msgf
            assert np.allclose(self.dsino['RefPt'],ss.dsino['RefPt']), msg0
            msg1 = msg+" sino nP"+msgf
            assert self.dsino['nP']==ss.dsino['nP'], msg1

        if tit is None:
            tit = self.Id.Name

        vis = self.get_visible()
        lS = self.lStruct
        lS = [lS[ii] for ii in range(0,self._dStruct['nObj']) if vis[ii]]

        ax = _plot.Plot_Impact_PolProjPoly(lS,
                                           ax=ax, Ang=Ang,
                                           AngUnit=AngUnit, Sketch=Sketch,
                                           dP=dP, dLeg=dLeg, draw=draw,
                                           fs=fs, tit=tit, wintit=wintit, Test=Test)
        return ax



"""
###############################################################################
###############################################################################
                        Rays-derived classes and functions
###############################################################################
"""


class Rays(utils.ToFuObject):
    """ Parent class of rays (ray-tracing), LOS, CamLOS1D and CamLOS2D

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

    # Fixed (class-wise) dictionary of default properties
    _ddef = {'Id':{'shot':0,
                   'include':['Mod','Cls','Exp','Diag',
                              'Name','shot','version']},
             'dgeom':{'Type':'Tor', 'Lim':[], 'arrayorder':'C'},
             'dsino':{},
             'dmisc':{'color':'k'}}
    _dplot = {'cross':{'Elt':'P',
                       'dP':{'color':'k','lw':2},
                       'dI':{'color':'k','ls':'--','m':'x','ms':8,'mew':2},
                       'dBs':{'color':'b','ls':'--','m':'x','ms':8,'mew':2},
                       'dBv':{'color':'g','ls':'--','m':'x','ms':8,'mew':2},
                       'dVect':{'color':'r','scale':10}},
              'hor':{'Elt':'P',
                     'dP':{'color':'k','lw':2},
                     'dI':{'color':'k','ls':'--'},
                     'dBs':{'color':'b','ls':'--'},
                     'dBv':{'color':'g','ls':'--'},
                     'Nstep':50},
              '3d':{'Elt':'P',
                    'dP':{'color':(0.8,0.8,0.8,1.),
                          'rstride':1,'cstride':1,
                          'linewidth':0., 'antialiased':False},
                    'Lim':None,
                    'Nstep':50}}

    _dcases = {'A':{'type':tuple, 'lk':[]},
               'B':{'type':dict, 'lk':['D','u']},
               'C':{'type':dict, 'lk':['D','pinhole']},
               'D':{'type':dict, 'lk':['pinhole','F','nIn','e1','x1']},
               'E':{'type':dict, 'lk':['pinhole','F','nIn','e1','l1','n1']},
               'F':{'type':dict, 'lk':['pinhole','F','angles','x1']},
               'G':{'type':dict, 'lk':['pinhole','F','angles','l1','n1']}}

    _method = "optimized"



    # Does not exist beofre Python 3.6 !!!
    def __init_subclass__(cls, color='k', **kwdargs):
        # Python 2
        super(Rays,cls).__init_subclass__(**kwdargs)
        # Python 3
        #super().__init_subclass__(**kwdargs)
        cls._ddef = copy.deepcopy(Rays._ddef)
        cls._dplot = copy.deepcopy(Rays._dplot)
        cls._set_color_ddef(color)
        if cls._is2D():
            cls._dcases['D']['lk'] += ['e2','x2']
            cls._dcases['E']['lk'] += ['e2','l2','n2']
            cls._dcases['F']['lk'] += ['x2']
            cls._dcases['G']['lk'] += ['l2','n2']

    @classmethod
    def _set_color_ddef(cls, color):
        cls._ddef['dmisc']['color'] = mpl.colors.to_rgba(color)

    def __init__(self, dgeom=None, lOptics=None, Etendues=None, Surfaces=None,
                 config=None, dchans=None, dX12='geom',
                 Id=None, Name=None, Exp=None, shot=None, Diag=None,
                 sino_RefPt=None, fromdict=None, method='optimized',
                 SavePath=os.path.abspath('./'), color=None, plotdebug=True):

        # To replace __init_subclass__ for Python 2
        if sys.version[0]=='2':
            self._dstrip = utils.ToFuObjectBase._dstrip.copy()
            self.__class__._strip_init()

        # Create a dplot at instance level
        self._dplot = copy.deepcopy(self.__class__._dplot)

        # Extra-early fix for Exp
        # Workflow to be cleaned up later ?
        if Exp is None and config is not None:
            Exp = config.Id.Exp

        kwdargs = locals()
        del kwdargs['self']
        # super()
        super(Rays,self).__init__(**kwdargs)

    def _reset(self):
        # super()
        super(Rays,self)._reset()
        self._dgeom = dict.fromkeys(self._get_keys_dgeom())
        if self._is2D():
            self._dX12 = dict.fromkeys(self._get_keys_dX12())
        self._dOptics = dict.fromkeys(self._get_keys_dOptics())
        self._dconfig = dict.fromkeys(self._get_keys_dconfig())
        self._dsino = dict.fromkeys(self._get_keys_dsino())
        self._dchans = dict.fromkeys(self._get_keys_dchans())
        self._dmisc = dict.fromkeys(self._get_keys_dmisc())
        #self._dplot = copy.deepcopy(self.__class__._ddef['dplot'])

    @classmethod
    def _checkformat_inputs_Id(cls, Id=None, Name=None,
                               Exp=None, shot=None, Diag=None,
                               include=None,
                               **kwdargs):
        if Id is not None:
            assert isinstance(Id,utils.ID)
            Name, Exp, shot, Diag = Id.Name, Id.Exp, Id.shot, Id.Diag
        if shot is None:
            shot = cls._ddef['Id']['shot']
        if include is None:
            include = cls._ddef['Id']['include']

        dins = {'Name':{'var':Name, 'cls':str},
                'Exp':{'var':Exp, 'cls':str},
                'Diag':{'var':Diag, 'cls':str},
                'shot':{'var':shot, 'cls':int},
                'include':{'var':include, 'listof':str}}
        dins, err, msg = cls._check_InputsGeneric(dins, tab=0)
        if err:
            raise Exception(msg)
        kwdargs.update({'Name':Name, 'Exp':Exp, 'shot':shot, 'Diag':Diag,
                        'include':include})
        return kwdargs

    ###########
    # Get largs
    ###########

    @staticmethod
    def _get_largs_dgeom(sino=True):
        largs = ['dgeom']
        if sino:
            lsino = Rays._get_largs_dsino()
            largs += ['sino_{0}'.format(s) for s in lsino]
        return largs

    @staticmethod
    def _get_largs_dX12():
        largs = ['dX12']
        return largs

    @staticmethod
    def _get_largs_dOptics():
        largs = ['lOptics']
        return largs

    @staticmethod
    def _get_largs_dconfig():
        largs = ['config']
        return largs

    @staticmethod
    def _get_largs_dsino():
        largs = ['RefPt']
        return largs

    @staticmethod
    def _get_largs_dchans():
        largs = ['dchans']
        return largs

    @staticmethod
    def _get_largs_dmisc():
        largs = ['color']
        return largs

    ###########
    # Get check and format inputs
    ###########


    def _checkformat_inputs_dES(self, val=None):
        if val is not None:
            C0 = type(val) in [int,float,np.int64,np.float64]
            C1 = hasattr(val,'__iter__')
            assert C0 or C1
            if C0:
                val = np.asarray([val],dtype=float)
            else:
                val = np.asarray(val,dtype=float).ravel()
                assert val.size==self._dgeom['nRays']
        return val


    @classmethod
    def _checkformat_inputs_dgeom(cls, dgeom=None):
        assert dgeom is not None
        assert isinstance(dgeom,tuple) or isinstance(dgeom,dict)
        lC = [k for k in cls._dcases.keys()
              if (isinstance(dgeom,cls._dcases[k]['type'])
                  and all([kk in dgeom.keys() for kk in cls._dcases[k]['lk']]))]
        if not len(lC)==1:
            lstr = [v['lk'] for v in cls._dcases.values()]
            msg = "Arg dgeom must be either:\n"
            msg += "  - dict with keys:\n"
            msg += "\n    - " + "\n    - ".join(lstr)
            msg += "  - tuple of len()==2 containing (D,u)"
            raise Exception(msg)
        case = lC[0]

        def _checkformat_Du(arr, name):
            arr = np.asarray(arr,dtype=float)
            msg = "Arg %s must be an iterable convertible into either:"%name
            msg += "\n    - a 1D np.ndarray of size=3"
            msg += "\n    - a 2D np.ndarray of shape (3,N)"
            assert arr.ndim in [1,2], msg
            if arr.ndim==1:
                assert arr.size==3, msg
                arr = arr.reshape((3,1))
            else:
                assert 3 in arr.shape, msg
                if arr.shape[0]!=3:
                    arr = arr.T
            arr = np.ascontiguousarray(arr)
            return arr

        if case in ['A','B']:
            D = dgeom[0] if case == 'A' else dgeom['D']
            u = dgeom[1] if case == 'A' else dgeom['u']
            D = _checkformat_Du(D, 'D')
            u = _checkformat_Du(u, 'u')
            # Normalize u
            u = u/np.sqrt(np.sum(u**2,axis=0))[np.newaxis,:]
            nD, nu = D.shape[1], u.shape[1]
            C0 = nD==1 and nu>1
            C1 = nD>1 and nu==1
            C2 = nD==nu
            msg = "The number of rays is ambiguous from D and u shapes !"
            assert C0 or C1 or C2, msg
            nRays = max(nD,nu)
            dgeom = {'D':D, 'u':u, 'isImage':False}

        elif case == 'C':
            D = _checkformat_Du(dgeom['D'], 'D')
            dins = {'pinhole':{'var':dgeom['pinhole'], 'vectnd':3}}
            dins, err, msg = cls._check_InputsGeneric(dins)
            if err:
                raise Exception(msg)
            pinhole = dins['pinhole']['var']
            dgeom = {'D':D, 'pinhole':pinhole, 'isImage':False}
            nRays = D.shape[1]

        else:
            dins = {'pinhole':{'var':dgeom['pinhole'], 'vectnd':3},
                    'F':{'var':dgeom['F'], 'int2float':None}}
            if case in ['D','E']:
                dins['nIn'] = {'var':dgeom['nIn'], 'unitvectnd':3}
                dins['e1'] = {'var':dgeom['e1'], 'unitvectnd':3}
                if 'e2' in dgeom.keys():
                    dins['e2'] = {'var':dgeom['e2'], 'unitvectnd':3}
            else:
                dins['angles'] = {'var':dgeom['angles'], 'vectnd':3}

            if case in ['D','F']:
                dins['x1'] = {'var':dgeom['x1'], 'vectnd':None}
                if 'x2':
                    dins['x2'] = {'var':dgeom['x2'], 'vectnd':None}
            else:
                dins['l1'] = {'var':dgeom['l1'], 'int2float':None}
                dins['n1'] = {'var':dgeom['n1'], 'float2int':None}
                if 'l2' in dgeom.keys():
                    dins['l2'] = {'var':dgeom['l2'], 'int2float':None}
                    dins['n2'] = {'var':dgeom['n2'], 'float2int':None}

            dins, err, msg = cls._check_InputsGeneric(dins)
            if err:
                raise Exception(msg)
            dgeom = {'dX12':{}}
            for k in dins.keys():
                if k == 'pinhole':
                    dgeom[k] = dins[k]['var']
                else:
                    dgeom['dX12'][k] = dins[k]['var']
            if case in ['E','G']:
                x1 = dgeom['dX12']['l1']*np.linspace(-0.5, 0.5,
                                                     dgeom['dX12']['n1'],
                                                     end_point=True)
                dgeom['dX12']['x1'] = x1
                if self._is2D():
                    x2 = dgeom['dX12']['l2']*np.linspace(-0.5, 0.5,
                                                         dgeom['dX12']['n2'],
                                                         end_point=True)
                    dgeom['dX12']['x2'] = x2
            if self._is2D():
                nRays = dgeom['dX12']['n1']*dgeom['dX12']['n2']
                ind1, ind2, indr = self._get_ind12r_n12(n1=dgeom['dX12']['n1'],
                                                        n2=dgeom['dX12']['n2'])
                dgeom['dX12']['ind1'] = ind1
                dgeom['dX12']['ind2'] = ind2
                dgeom['dX12']['indr'] = indr
                dgeom['isImage'] = True
            else:
                nRays = dgeom['dX12']['n1']
                dgeom['isImage'] = False
        dgeom.update({'case':case, 'nRays':nRays})
        return dgeom

    def _checkformat_dX12(self, dX12=None):
        lc = [dX12 is None, dX12 == 'geom' or dX12 == {'from':'geom'},
              isinstance(dX12, dict)]
        if not np.sum(lc) == 1:
            msg = "dX12 must be either:\n"
            msg += "    - None\n"
            msg += "    - 'geom' : will be derived from the 3D geometry\n"
            msg += "    - dict : containing {'x1'  : array of coords.,\n"
            msg += "                         'x2'  : array of coords.,\n"
            msg += "                         'ind1': array of int indices,\n"
            msg += "                         'ind2': array of int indices}"
            raise Exception(msg)

        if lc[1]:
            ls = self._get_keys_dX12()
            c0 = isinstance(self._dgeom['dX12'],dict)
            c1 = c0 and all([ss in self._dgeom['dX12'].keys() for ss in ls])
            c2 = c1 and all([self._dgeom['dX12'][ss] is not None for ss in ls])
            if not c2:
                msg = "dX12 cannot be derived from dgeom (info not known) !"
                raise Exception(msg)
            dX12 = {'from':'geom'}

        if lc[2]:
            ls = ['x1','x2','ind1','ind2']
            assert all([ss in dX12.keys() for ss in ls])
            x1 = np.asarray(dX12['x1']).ravel()
            x2 = np.asarray(dX12['x2']).ravel()
            n1, n2 = x1.size, x2.size
            ind1, ind2, indr = self._get_ind12r_n12(ind1=dX12['ind1'],
                                                    ind2=dX12['ind2'],
                                                    n1=n1, n2=n2)
            dX12 = {'x1':x1, 'x2':x2, 'n1':n1, 'n2':n2,
                    'ind1':ind1, 'ind2':ind2, 'indr':indr, 'from':'self'}
        return dX12

    @staticmethod
    def _checkformat_dOptics(lOptics=None):
        if lOptics is None:
            lOptics = []
        assert type(lOptics) is list
        lcls = ['Apert3D','Cryst2D']
        nOptics = len(lOptics)
        for ii in range(0,nOptics):
            assert lOptics[ii].__class__.__name__ in lcls
        return lOptics

    @staticmethod
    def _checkformat_inputs_dconfig(config=None):
        C0 = isinstance(config,Config)
        msg = "Arg config must be a Config instance !"
        msg += "\n    expected : {0}".format(str(Config))
        msg += "\n    obtained : {0}".format(str(config.__class__))
        assert C0, msg
        lS = config.lStruct
        lC = [hasattr(ss,'_InOut') and ss._InOut in ['in','out']
              for ss in lS]
        msg = "All Struct in config must have self._InOut in ['in','out']"
        assert all(lC), msg
        lSIn = [ss for ss in lS if ss._InOut=='in']
        msg = "Arg config must have at least a StructIn subclass !"
        assert len(lSIn)>0, msg
        if not 'compute' in config._dextraprop['lprop']:
            config = config.copy()
            config.add_extraprop('compute',True)
        return config

    def _checkformat_inputs_dsino(self, RefPt=None):
        assert RefPt is None or hasattr(RefPt,'__iter__')
        if RefPt is not None:
            RefPt = np.asarray(RefPt,dtype=float).flatten()
            assert RefPt.size==2, "RefPt must be of size=2 !"
        return RefPt

    def _checkformat_inputs_dchans(self, dchans=None):
        assert dchans is None or isinstance(dchans,dict)
        if dchans is None:
            dchans = {}
        for k in dchans.keys():
            arr = np.asarray(dchans[k]).ravel()
            assert arr.size==self._dgeom['nRays']
            dchans[k] = arr
        return dchans


    @classmethod
    def _checkformat_inputs_dmisc(cls, color=None):
        if color is None:
            color = mpl.colors.to_rgba(cls._ddef['dmisc']['color'])
        assert mpl.colors.is_color_like(color)
        return tuple(mpl.colors.to_rgba(color))

    ###########
    # Get keys of dictionnaries
    ###########

    @staticmethod
    def _get_keys_dgeom():
        lk = ['D','u','pinhole', 'nRays',
              'kIn', 'kOut', 'PkIn', 'PkOut', 'vperp', 'indout',
              'kRMin', 'PRMin', 'RMin', 'isImage',
              'Etendues', 'Surfaces', 'dX12']
        return lk

    @staticmethod
    def _get_keys_dX12():
        lk = ['x1','x2','n1', 'n2',
              'ind1', 'ind2', 'indr']
        return lk

    @staticmethod
    def _get_keys_dOptics():
        lk = ['lorder','lCls','nObj', 'dObj']
        return lk

    @staticmethod
    def _get_keys_dsino():
        lk = ['RefPt', 'k', 'pts',
              'theta','p','phi']
        return lk

    @staticmethod
    def _get_keys_dconfig():
        lk = ['config']
        return lk

    @staticmethod
    def _get_keys_dchans():
        lk = []
        return lk

    @staticmethod
    def _get_keys_dmisc():
        lk = ['color']
        return lk

    ###########
    # _init
    ###########

    def _init(self, dgeom=None, config=None, Etendues=None, Surfaces=None,
              sino_RefPt=None, dchans=None, method='optimized', **kwargs):
        if method is not None:
            self._method = method
        kwdargs = locals()
        kwdargs.update(**kwargs)
        largs = self._get_largs_dgeom(sino=True)
        kwdgeom = self._extract_kwdargs(kwdargs, largs)
        largs = self._get_largs_dconfig()
        kwdconfig = self._extract_kwdargs(kwdargs, largs)
        largs = self._get_largs_dchans()
        kwdchans = self._extract_kwdargs(kwdargs, largs)
        largs = self._get_largs_dOptics()
        kwdOptics = self._extract_kwdargs(kwdargs, largs)
        largs = self._get_largs_dmisc()
        kwdmisc = self._extract_kwdargs(kwdargs, largs)
        self.set_dconfig(calcdgeom=False, **kwdconfig)
        self._set_dgeom(sino=True, **kwdgeom)
        if self._is2D():
            kwdX12 = self._extract_kwdargs(kwdargs, self._get_largs_dX12())
            self.set_dX12(**kwdX12)
        self._set_dOptics(**kwdOptics)
        self.set_dchans(**kwdchans)
        self._set_dmisc(**kwdmisc)
        self._dstrip['strip'] = 0

    ###########
    # set dictionaries
    ###########

    def set_dconfig(self, config=None, calcdgeom=True):
        config = self._checkformat_inputs_dconfig(config)
        self._dconfig['Config'] = config.copy()
        if calcdgeom:
            self.compute_dgeom()

    def _update_dgeom_from_TransRotFoc(self, val, key='x'):
        # To be finished for 1.4.1
        raise Exception("Not coded yet !")
        # assert False, "Not implemented yet, for future versions"
        # if key in ['x','y','z']:
            # if key == 'x':
                # trans = np.r_[val,0.,0.]
            # elif key == 'y':
                # trans = np.r_[0.,val,0.]
            # else:
                # trans = np.r_[0.,0.,val]
            # if self._dgeom['pinhole'] is not None:
                # self._dgeom['pinhole'] += trans
                # self._dgeom['D'] += trans[:,np.newaxis]
        # if key in ['nIn','e1','e2']:
            # if key == 'nIn':
                # e1 = (np.cos(val)*self._dgeom['dX12']['e1']
                      # + np.sin(val)*self._dgeom['dX12']['e2'])
                # e2 = (np.cos(val)*self._dgeom['dX12']['e2']
                      # - np.sin(val)*self._dgeom['dX12']['e1'])
                # self._dgeom['dX12']['e1'] = e1
                # self._dgeom['dX12']['e2'] = e2
            # elif key == 'e1':
                # nIn = (np.cos(val)*self._dgeom['dX12']['nIn']
                      # + np.sin(val)*self._dgeom['dX12']['e2'])
                # e2 = (np.cos(val)*self._dgeom['dX12']['e2']
                      # - np.sin(val)*self._dgeom['dX12']['nIn'])
                # self._dgeom['dX12']['nIn'] = nIn
                # self._dgeom['dX12']['e2'] = e2
            # else:
                # nIn = (np.cos(val)*self._dgeom['dX12']['nIn']
                      # + np.sin(val)*self._dgeom['dX12']['e1'])
                # e1 = (np.cos(val)*self._dgeom['dX12']['e1']
                     # - np.sin(val)*self._dgeom['dX12']['nIn'])
                # self._dgeom['dX12']['nIn'] = nIn
                # self._dgeom['dX12']['e1'] = e1
        # if key == 'F':
            # self._dgeom['F'] += val

    @classmethod
    def _get_x12_fromflat(cls, X12):
        x1, x2 = np.unique(X12[0,:]), np.unique(X12[1,:])
        n1, n2 = x1.size, x2.size
        if n1*n2 != X12.shape[1]:
            tol = np.linalg.norm(np.diff(X12[:,:2],axis=1))/100.
            tolmag = int(np.log10(tol))-1
            x1 = np.unique(np.round(X12[0,:], -tolmag))
            x2 = np.unique(np.round(X12[1,:], -tolmag))
            ind1 = np.digitize(X12[0,:], 0.5*(x1[1:]+x1[:-1]))
            ind2 = np.digitize(X12[1,:], 0.5*(x2[1:]+x2[:-1]))
            ind1u, ind2u = np.unique(ind1), np.unique(ind2)
            x1 = np.unique([np.mean(X12[0,ind1==ii]) for ii in ind1u])
            x2 = np.unique([np.mean(X12[1,ind2==ii]) for ii in ind2u])
        n1, n2 = x1.size, x2.size

        if n1*n2 != X12.shape[1]:
            msg = "The provided X12 array does not seem to correspond to"
            msg += "a n1 x n2 2D matrix, even within tolerance"
            raise Exception(msg)

        ind1 = np.digitize(X12[0,:], 0.5*(x1[1:]+x1[:-1]))
        ind2 = np.digitize(X12[1,:], 0.5*(x2[1:]+x2[:-1]))
        ind1, ind2, indr = cls._get_ind12r_n12(ind1=ind1, ind2=ind2,
                                               n1=n1, n2=n2)
        return x1, x2, n1, n2, ind1, ind2, indr


    def _complete_dX12(self, dgeom):

        # Test if unique starting point
        if dgeom['case'] in ['A','B','C']:
            # Test if pinhole
            if dgeom['case'] in ['A','B']:
                u = dgeom['u'][:,0:1]
                sca2 = np.sum(dgeom['u'][:,1:]*u,axis=0)**2
                if np.all(sca2 < 1.-1.e-9):
                    DDb = dgeom['D'][:,1:]-dgeom['D'][:,0:1]
                    k = np.sum(DDb*(u + np.sqrt(sca2)*dgeom['u'][:,1:]),axis=0)
                    k = k / (1.-sca2)
                    if k[0] > 0 and np.all(k[1:]-k[0]<1.e-9):
                        pinhole = dgeom['D'][:,0] + k[0]*u
                        dgeom['pinhole'] = pinhole

            # Test if all D are on a common plane or line
            v0 = dgeom['D'][:,1]-dgeom['D'][:,0]
            va = dgeom['D']-dgeom['D'][:,0:1]

            # critetrion of unique D
            crit = np.sum(va**2) > 1.e-9
            if not crit:
                return dgeom

            v0 = v0/np.linalg.norm(v0)
            van = np.full(va.shape, np.nan)
            van[:,1:] = va[:,1:] / np.sqrt(np.sum(va[:,1:]**2,axis=0))[np.newaxis,:]
            vect2 = ((van[1,:]*v0[2]-van[2,:]*v0[1])**2
                     + (van[2,:]*v0[0]-van[0,:]*v0[2])**2
                     + (van[0,:]*v0[1]-van[1,:]*v0[0])**2)
            # Don't forget that vect2[0] is nan
            if np.all(vect2[1:]<1.e-9):
                # All D are aligned
                e1 = van[:,1]
                x1 = np.sum(va*e1[:,np.newaxis],axis=0)
                if dgeom['pinhole'] is not None:
                    kref = -np.sum((dgeom['D'][:,0]-dgeom['pinhole'])*e1)
                    x1 = x1-kref
                l1 = np.nanmax(x1)-np.nanmin(x1)
                if dgeom['dX12'] is None:
                    dgeom['dX12'] = {}
                dgeom['dX12'].update({'e1':e1, 'x1':x1, 'n1':x1.size})
            else:
                ind = np.nanargmax(vect2)
                v1 = van[:,ind]
                nn = np.cross(v0,v1)
                nn = nn/np.linalg.norm(nn)
                scaabs = np.abs(np.sum(nn[:,np.newaxis]*va,axis=0))
                if np.all(scaabs<1.e-9):
                    assert not '1d' in self.__class__.__name__.lower()
                    # All D are in a common plane perpendicular to n, check
                    # check nIn orientation
                    sca = np.sum(self.u*nn[:,np.newaxis],axis=0)
                    lc = [np.all(sca>=0.), np.all(sca<=0.)]
                    assert any(lc)
                    nIn = nn if lc[0] else -nn
                    e1 = v0
                    e2 = v1
                    if np.sum(np.cross(e1,nIn)*e2)<0.:
                        e2 = -e2
                    if np.abs(e1[2])>np.abs(e2[2]):
                        # Try to set e2 closer to ez if possible
                        e1, e2 = -e2, e1

                    if dgeom['dX12'] is None:
                        dgeom['dX12'] = {}
                    dgeom['dX12'].update({'nIn':nIn, 'e1':e1, 'e2':e2})

                    # Test binning
                    if dgeom['pinhole'] is not None:
                        k1ref = -np.sum((dgeom['D'][:,0]-dgeom['pinhole'])*e1)
                        k2ref = -np.sum((dgeom['D'][:,0]-dgeom['pinhole'])*e2)
                    else:
                        k1ref, k2ref = 0., 0.
                    x12 = np.array([np.sum(va*e1[:,np.newaxis],axis=0)-k1ref,
                                    np.sum(va*e2[:,np.newaxis],axis=0)-k2ref])
                    try:
                        x1, x2, n1, n2, ind1, ind2, indr=self._get_x12_fromflat(x12)
                        dgeom['dX12'].update({'x1':x1, 'x2':x2,
                                              'n1':n1, 'n2':n2,
                                              'ind1':ind1, 'ind2':ind2,
                                              'indr':indr})
                        dgeom['isImage'] = True

                    except Exception as err:
                        warnings.warn(str(err))

        else:
            if dgeom['case'] in ['F','G']:
                # Get unit vectors from angles
                msg = "Not implemented yet, angles will be available for 1.4.1"
                raise Exception(msg)

            # Get D and x12 from x1, x2
            x12 = np.array([dgeom['dX12']['x1'][dgeom['dX12']['ind1']],
                            dgeom['dX12']['x2'][dgeom['dX12']['ind2']]])
            D = dgeom['pinhole'] - dgeom['F']*dgeom['dX12']['nIn']
            D = (D[:,np.newaxis]
                 + x12[0,:]*dgeom['dX12']['e1']
                 + x12[1,:]*dgeom['dX12']['e2'])
            dgeom['D'] = D
        return dgeom



    def _prepare_inputs_kInOut(self):
        if self._method=='ref':
            # Prepare input
            D = np.ascontiguousarray(self.D)
            u = np.ascontiguousarray(self.u)

            # Get reference
            lS = self.lStruct_computeInOut

            lSIn = [ss for ss in lS if ss._InOut=='in']
            if len(lSIn)==0:
                msg = "self.config must have at least a StructIn subclass !"
                assert len(lSIn)>0, msg
            elif len(lSIn)>1:
                S = lSIn[np.argmin([ss.dgeom['Surf'] for ss in lSIn])]
            else:
                S = lSIn[0]

            VPoly = S.Poly_closed
            VVIn =  S.dgeom['VIn']
            Lim = S.Lim
            nLim = S.noccur
            VType = self.config.Id.Type

            lS = [ss for ss in lS if ss._InOut=='out']
            lSPoly, lSVIn, lSLim, lSnLim = [], [], [], []
            for ss in lS:
                lSPoly.append(ss.Poly_closed)
                lSVIn.append(ss.dgeom['VIn'])
                lSLim.append(ss.Lim)
                lSnLim.append(ss.noccur)
            largs = [D, u, VPoly, VVIn]
            dkwd = dict(Lim=Lim, nLim=nLim,
                        LSPoly=lSPoly, LSLim=lSLim,
                        lSnLim=lSnLim, LSVIn=lSVIn, VType=VType,
                        RMin=None, Forbid=True, EpsUz=1.e-6, EpsVz=1.e-9,
                        EpsA=1.e-9, EpsB=1.e-9, EpsPlane=1.e-9, Test=True)

        elif self._method=='optimized':
            # Prepare input
            D = np.ascontiguousarray(self.D)
            u = np.ascontiguousarray(self.u)
            # Get reference
            lS = self.lStruct_computeInOut

            lSIn = [ss for ss in lS if ss._InOut=='in']
            if len(lSIn)==0:
                msg = "self.config must have at least a StructIn subclass !"
                assert len(lSIn)>0, msg
            elif len(lSIn)>1:
                S = lSIn[np.argmin([ss.dgeom['Surf'] for ss in lSIn])]
            else:
                S = lSIn[0]

            VPoly = S.Poly_closed
            VVIn =  S.dgeom['VIn']
            if np.size(np.shape(S.Lim)) > 1 :
                Lim = np.asarray([S.Lim[0][0], S.Lim[0][1]])
            else:
                Lim = S.Lim
            nLim = S.noccur
            VType = self.config.Id.Type

            lS = [ss for ss in lS if ss._InOut=='out']
            lSPolyx, lSVInx = [], []
            lSPolyy, lSVIny = [], []
            lSLim, lSnLim = [], []
            lsnvert = []
            num_tot_structs = 0
            num_lim_structs = 0
            for ss in lS:
                l = ss.Poly_closed[0]
                [lSPolyx.append(item) for item in l]
                l = ss.Poly_closed[1]
                [lSPolyy.append(item) for item in l]
                l = ss.dgeom['VIn'][0]
                [lSVInx.append(item) for item in l]
                l = ss.dgeom['VIn'][1]
                [lSVIny.append(item) for item in l]
                lSLim.append(ss.Lim)
                lSnLim.append(ss.noccur)
                if len(lsnvert)==0:
                    lsnvert.append(len(ss.Poly_closed[0]))
                else:
                    lsnvert.append(len(ss.Poly_closed[0]) + lsnvert[num_lim_structs-1])
                num_lim_structs += 1
                if ss.Lim is None or len(ss.Lim) == 0:
                    num_tot_structs += 1
                else:
                    num_tot_structs += len(ss.Lim)

            lsnvert = np.asarray(lsnvert, dtype=np.int64)
            lSPolyx = np.asarray(lSPolyx)
            lSPolyy = np.asarray(lSPolyy)
            lSVInx = np.asarray(lSVInx)
            lSVIny = np.asarray(lSVIny)

            largs = [D, u, VPoly, VVIn]
            dkwd = dict(ves_lims=Lim,
                        nstruct_tot=num_tot_structs,
                        nstruct_lim=num_lim_structs,
                        lstruct_polyx=lSPolyx,
                        lstruct_polyy=lSPolyy,
                        lstruct_lims=lSLim,
                        lstruct_nlim=np.asarray(lSnLim, dtype=np.int64),
                        lstruct_normx=lSVInx,
                        lstruct_normy=lSVIny,
                        lnvert=lsnvert,
                        ves_type=VType,
                        rmin=-1, forbid=True, eps_uz=1.e-6, eps_vz=1.e-9,
                        eps_a=1.e-9, eps_b=1.e-9, eps_plane=1.e-9, test=True)

        else:
            # --------------------------------
            # Here I can prepare the inputs as requested by your routine
            pass
            # --------------------------------

        return largs, dkwd

    def _compute_kInOut(self):

        # Prepare inputs
        largs, dkwd = self._prepare_inputs_kInOut()

        if self._method=='ref':
            # call the dedicated function
            out = _GG.SLOW_LOS_Calc_PInOut_VesStruct(*largs, **dkwd)
            # Currently computes and returns too many things
            PIn, POut, kIn, kOut, VperpIn, vperp, IIn, indout = out
        elif self._method=="optimized":
            # call the dedicated function
            out = _GG.LOS_Calc_PInOut_VesStruct(*largs, **dkwd)
            # Currently computes and returns too many things
            kIn, kOut, vperp, indout = out
        else:
            pass
        return kIn, kOut, vperp, indout


    def compute_dgeom(self, extra=True, plotdebug=True):
        # Can only be computed if config if provided
        if self._dconfig['Config'] is None:
            msg = "Attribute dgeom cannot be computed without a config !"
            warnings.warn(msg)
            return

        # dX12
        self._dgeom = self._complete_dX12(self._dgeom)

        # Perform computation of kIn and kOut
        kIn, kOut, vperp, indout = self._compute_kInOut()

        # Clean up (in case of nans)
        ind = np.isnan(kIn)
        kIn[ind] = 0.
        ind = np.isnan(kOut) | np.isinf(kOut)
        if np.any(ind):
            kOut[ind] = np.nan
            msg = "Some LOS have no visibility inside the plasma domain !"
            warnings.warn(msg)
            if plotdebug:
                PIn = self.D[:,ind] + kIn[np.newaxis,ind]*self.u[:,ind]
                POut = self.D[:,ind] + kOut[np.newaxis,ind]*self.u[:,ind]
                # To be updated
                _plot._LOS_calc_InOutPolProj_Debug(self.config,
                                                   self.D[:,ind],
                                                   self.u[:,ind],
                                                   PIn, POut,
                                                   Lim=[np.pi/4.,7.*np.pi/4],
                                                   Nstep=50)

        # Handle particular cases with kIn > kOut
        ind = np.zeros(kIn.shape,dtype=bool)
        ind[~np.isnan(kOut)] = True
        ind[ind] = kIn[ind] > kOut[ind]
        kIn[ind] = 0.

        # Update dgeom
        dd = {'kIn':kIn, 'kOut':kOut, 'vperp':vperp, 'indout':indout}
        self._dgeom.update(dd)

        # Run extra computations
        if extra:
            self._compute_dgeom_kRMin()
            self._compute_dgeom_extra1()

    def _compute_dgeom_kRMin(self):
        # Get RMin if Type is Tor
        if self.config.Id.Type=='Tor':
            kRMin = _comp.LOS_PRMin(self.D, self.u, kPOut=self.kOut, Eps=1.e-12)
        else:
            kRMin = None
        self._dgeom.update({'kRMin':kRMin})

    def _compute_dgeom_extra1(self):
        if self._dgeom['kRMin'] is not None:
            PRMin = self.D + self._dgeom['kRMin'][np.newaxis,:]*self.u
            RMin = np.hypot(PRMin[0,:],PRMin[1,:])
        else:
            PRMin, RMin = None, None
        PkIn = self.D + self._dgeom['kIn'][np.newaxis,:]*self.u
        PkOut = self.D + self._dgeom['kOut'][np.newaxis,:]*self.u
        dd = {'PkIn':PkIn, 'PkOut':PkOut, 'PRMin':PRMin, 'RMin':RMin}
        self._dgeom.update(dd)

    def _compute_dgeom_extra2D(self):
        if not '2d' in self.Id.Cls.lower():
            return
        D, u = self.D, self.u
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
        crossbis= np.sqrt((D0D[1,:]*e1[2]-D0D[2,:]*e1[1])**2
                          + (D0D[2,:]*e1[0]-D0D[0,:]*e1[2])**2
                          + (D0D[0,:]*e1[1]-D0D[1,:]*e1[0])**2)
        D0D = D0D[:,crossbis<dd/3.]
        sca = np.sum(D0D*e1[:,np.newaxis],axis=0)
        e1 = D0D[:,np.argmax(np.abs(sca))]
        try:
            import tofu.geom.utils as geom_utils
        except Exception:
            from . import utils as geom_utils

        nIn, e1, e2 = geom_utils.get_nIne1e2(C, nIn=nIn, e1=e1)
        if np.abs(np.abs(nIn[2])-1.)>1.e-12:
            if np.abs(e1[2])>np.abs(e2[2]):
                e1, e2 = e2, e1
        e2 = e2 if e2[2]>0. else -e2
        self._dgeom.update({'C':C, 'nIn':nIn, 'e1':e1, 'e2':e2})

    def set_Etendues(self, val):
        val = self._checkformat_inputs_dES(val)
        self._dgeom['Etendues'] = val

    def set_Surfaces(self, val):
        val = self._checkformat_inputs_dES(val)
        self._dgeom['Surfaces'] = val

    def _set_dgeom(self, dgeom=None, Etendues=None, Surfaces=None,
                   sino_RefPt=None,
                   extra=True, sino=True):
        dgeom = self._checkformat_inputs_dgeom(dgeom=dgeom)
        self._dgeom.update(dgeom)
        self.compute_dgeom(extra=extra)
        self.set_Etendues(Etendues)
        self.set_Surfaces(Surfaces)
        if sino:
            self.set_dsino(sino_RefPt)

    def set_dX12(self, dX12=None):
        dX12 = self._checkformat_dX12(dX12)
        self._dX12.update(dX12)

    def _compute_dsino_extra(self):
        if self._dsino['k'] is not None:
            pts = self.D + self._dsino['k'][np.newaxis,:]*self.u
            R = np.hypot(pts[0,:],pts[1,:])
            DR = R-self._dsino['RefPt'][0]
            DZ = pts[2,:]-self._dsino['RefPt'][1]
            p = np.hypot(DR,DZ)
            theta = np.arctan2(DZ,DR)
            ind = theta<0
            p[ind] = -p[ind]
            theta[ind] = -theta[ind]
            phipts = np.arctan2(pts[1,:],pts[0,:])
            etheta = np.array([np.cos(phipts)*np.cos(theta),
                               np.sin(phipts)*np.cos(theta),
                               np.sin(theta)])
            phi = np.arccos(np.abs(np.sum(etheta*self.u,axis=0)))
            dd = {'pts':pts, 'p':p, 'theta':theta, 'phi':phi}
            self._dsino.update(dd)

    def set_dsino(self, RefPt=None, extra=True):
        RefPt = self._checkformat_inputs_dsino(RefPt=RefPt)
        self._dsino.update({'RefPt':RefPt})
        VType = self.config.Id.Type
        if RefPt is not None:
            self._dconfig['Config'].set_dsino(RefPt=RefPt)
            kOut = np.copy(self._dgeom['kOut'])
            kOut[np.isnan(kOut)] = np.inf
            try:
                out = _GG.LOS_sino(self.D, self.u, RefPt, kOut,
                                   Mode='LOS', VType=VType)
                Pt, k, r, Theta, p, theta, Phi = out
                self._dsino.update({'k':k})
            except Exception as err:
                msg = str(err)
                msg += "\nError while computing sinogram !"
                raise Exception(msg)
        if extra:
            self._compute_dsino_extra()

    def _set_dOptics(self, lOptics=None):
        lOptics = self._checkformat_dOptics(lOptics=lOptics)
        self._set_dlObj(lOptics, din=self._dOptics)

    def set_dchans(self, dchans=None):
        dchans = self._checkformat_inputs_dchans(dchans)
        self._dchans = dchans

    def _set_color(self, color=None):
        color = self._checkformat_inputs_dmisc(color=color)
        self._dmisc['color'] = color
        self._dplot['cross']['dP']['color'] = color
        self._dplot['hor']['dP']['color'] = color
        self._dplot['3d']['dP']['color'] = color

    def _set_dmisc(self, color=None):
        self._set_color(color)

    ###########
    # strip dictionaries
    ###########

    def _strip_dgeom(self, strip=0):
        if self._dstrip['strip']==strip:
            return

        if strip<self._dstrip['strip']:
            # Reload
            if self._dstrip['strip']==1:
                self._compute_dgeom_extra1()
            elif self._dstrip['strip']>=2 and strip==1:
                self._compute_dgeom_kRMin()
            elif self._dstrip['strip']>=2 and strip==0:
                self._compute_dgeom_kRMin()
                self._compute_dgeom_extra1()
        else:
            # strip
            if strip==1:
                lkeep = ['D','u','pinhole','nRays',
                         'kIn','kOut','vperp','indout', 'kRMin',
                         'Etendues','Surfaces','isImage','dX12']
                utils.ToFuObject._strip_dict(self._dgeom, lkeep=lkeep)
            elif self._dstrip['strip']<=1 and strip>=2:
                lkeep = ['D','u','pinhole','nRays',
                         'kIn','kOut','vperp','indout',
                         'Etendues','Surfaces','isImage','dX12']
                utils.ToFuObject._strip_dict(self._dgeom, lkeep=lkeep)

    def _strip_dconfig(self, strip=0, verb=True):
        if self._dstrip['strip'] == strip:
            return

        if strip<self._dstrip['strip']:
            if self._dstrip['strip'] == 4:
                pfe = self._dconfig['Config']
                try:
                    self._dconfig['Config'] = utils.load(pfe, verb=verb)
                except Exception as err:
                    msg = str(err)
                    msg += "\n    type(pfe) = {0}".format(str(type(pfe)))
                    msg += "\n    self._dstrip['strip'] = {0}".format(self._dstrip['strip'])
                    msg += "\n    strip = {0}".format(strip)
                    raise Exception(msg)

            self._dconfig['Config'].strip(strip, verb=verb)
        else:
            if strip == 4:
                path, name = self.config.Id.SavePath, self.config.Id.SaveName
                # --- Check !
                lf = os.listdir(path)
                lf = [ff for ff in lf
                      if all([s in ff for s in [name,'.npz']])]
                exist = len(lf)==1
                # ----------
                pathfile = os.path.join(path,name)+'.npz'
                if not exist:
                    msg = """BEWARE:
                        You are about to delete the Config object
                        Only the path/name to saved a object will be kept

                        But it appears that the following object has no
                        saved file where specified (obj.Id.SavePath)
                        Thus it won't be possible to retrieve it
                        (unless available in the current console:"""
                    msg += "\n    - {0}".format(pathfile)
                    if force:
                        warning.warn(msg)
                    else:
                        raise Exception(msg)
                self._dconfig['Config'] = pathfile

            else:
                self._dconfig['Config'].strip(strip, verb=verb)


    def _strip_dsino(self, strip=0):
        if self._dstrip['strip']==strip:
            return

        if strip<self._dstrip['strip']:
            if strip<=1 and self._dsino['k'] is not None:
                self._compute_dsino_extra()
        else:
            if self._dstrip['strip']<=1:
                utils.ToFuObject._strip_dict(self._dsino, lkeep=['RefPt','k'])

    def _strip_dmisc(self, lkeep=['color']):
        utils.ToFuObject._strip_dict(self._dmisc, lkeep=lkeep)


    ###########
    # _strip and get/from dict
    ###########

    @classmethod
    def _strip_init(cls):
        cls._dstrip['allowed'] = [0,1,2,3,4]
        nMax = max(cls._dstrip['allowed'])
        doc = """
                 1: dgeom w/o pts + config.strip(1)
                 2: dgeom w/o pts + config.strip(2) + dsino empty
                 3: dgeom w/o pts + config.strip(3) + dsino empty
                 4: dgeom w/o pts + config=pathfile + dsino empty
                 """
        doc = utils.ToFuObjectBase.strip.__doc__.format(doc,nMax)
        if sys.version[0]=='2':
            cls.strip.__func__.__doc__ = doc
        else:
            cls.strip.__doc__ = doc

    def strip(self, strip=0, verb=True):
        # super()
        super(Rays,self).strip(strip=strip, verb=verb)

    def _strip(self, strip=0, verb=True):
        self._strip_dconfig(strip=strip, verb=verb)
        self._strip_dgeom(strip=strip)
        self._strip_dsino(strip=strip)

    def _to_dict(self):
        dout = {'dconfig':{'dict':self._dconfig, 'lexcept':None},
                'dgeom':{'dict':self.dgeom, 'lexcept':None},
                'dchans':{'dict':self.dchans, 'lexcept':None},
                'dsino':{'dict':self.dsino, 'lexcept':None}}
        if self._is2D():
            dout['dX12'] = {'dict':self._dX12, 'lexcept':None}
        return dout

    @classmethod
    def _checkformat_fromdict_dconfig(cls, dconfig):
        if dconfig['Config'] is None:
            return Nonei
        if type(dconfig['Config']) is dict:
            dconfig['Config'] = Config(fromdict=dconfig['Config'])
        lC = [isinstance(dconfig['Config'],Config),
              type(dconfig['Config']) is str]
        assert any(lC)

    def _from_dict(self, fd):
        self._checkformat_fromdict_dconfig(fd['dconfig'])

        self._dconfig.update(**fd['dconfig'])
        self._dgeom.update(**fd['dgeom'])
        self._dsino.update(**fd['dsino'])
        if 'dchans' in fd.keys():
            self._dchans.update(**fd['dchans'])
        if self._is2D():
            self._dX12.update(**fd['dX12'])


    ###########
    # properties
    ###########


    @property
    def dgeom(self):
        return self._dgeom
    @property
    def dchans(self):
        return self._dchans
    @property
    def dsino(self):
        return self._dsino

    @property
    def isPinhole(self):
        c0 = 'pinhole' in self._dgeom.keys()
        return c0 and self._dgeom['pinhole'] is not None

    @property
    def nRays(self):
        return self._dgeom['nRays']

    @property
    def D(self):
        if self._dgeom['D'].shape[1]<self._dgeom['nRays']:
            D = np.tile(self._dgeom['D'], self._dgeom['nRays'])
        else:
            D = self._dgeom['D']
        return D

    @property
    def u(self):
        if self.isPinhole:
            u = self._dgeom['pinhole'][:,np.newaxis]-self._dgeom['D']
            u = u/np.sqrt(np.sum(u**2,axis=0))[np.newaxis,:]
        elif self._dgeom['u'].shape[1]<self._dgeom['nRays']:
            u = np.tile(self._dgeom['u'], self._dgeom['nRays'])
        else:
            u = self._dgeom['u']
        return u

    @property
    def pinhole(self):
        if self._dgeom['pinhole'] is None:
            msg = "This is not a pinhole camera => pinhole is None"
            warnings.warn(msg)
        return self._dgeom['pinhole']

    @property
    def config(self):
        return self._dconfig['Config']

    @property
    def lStruct_computeInOut(self):
        compute = self.config.get_compute()
        lS = self.config.lStruct
        lSI, lSO = [], []
        for ii in range(0,self._dconfig['Config']._dStruct['nObj']):
            if compute[ii]:
                if lS[ii]._InOut=='in':
                    lSI.append(lS[ii])
                elif lS[ii]._InOut=='out':
                    lSO.append(lS[ii])
        return lSI+lSO

    @property
    def Etendues(self):
        if self._dgeom['Etendues'] is None:
            E = None
        elif self._dgeom['Etendues'].size==self._dgeom['nRays']:
            E = self._dgeom['Etendues']
        elif self._dgeom['Etendues'].size==1:
            E = np.repeat(self._dgeom['Etendues'], self._dgeom['nRays'])
        else:
            msg = "Stored Etendues is not conform !"
            raise Exception(msg)
        return E
    @property
    def Surfaces(self):
        if self._dgeom['Surfaces'] is None:
            S = None
        elif self._dgeom['Surfaces'].size==self._dgeom['nRays']:
            S = self._dgeom['Surfaces']
        elif self._dgeom['Surfaces'].size==1:
            S = np.repeat(self._dgeom['Surfaces'], self._dgeom['nRays'])
        else:
            msg = "Stored Surfaces not conform !"
            raise Exception(msg)
        return S

    @property
    def kIn(self):
        return self._dgeom['kIn']
    @property
    def kOut(self):
        return self._dgeom['kOut']
    @property
    def kMin(self):
        if self.isPinhole:
            kMin = (self._dgeom['pinhole'][:,np.newaxis]-self._dgeom['D'])
            kMin = np.sqrt(np.sum(kMin**2,axis=0))
        else:
            kMin = 0.
        return kMin

    @classmethod
    def _is2D(cls):
        c0 = '2d' in cls.__name__.lower()
        return c0

    @classmethod
    def _isLOS(cls):
        c0 = 'los' in cls.__name__.lower()
        return c0

    ###########
    # public methods
    ###########

    def _check_indch(self, ind, out=int):
        if ind is not None:
            ind = np.asarray(ind)
            assert ind.ndim==1
            assert ind.dtype in [np.int64,np.bool_]
            if ind.dtype == np.bool_:
                assert ind.size==self.nRays
                if out is int:
                    indch = ind.nonzero()[0]
                else:
                    indch = ind
            else:
                assert np.max(ind)<self.nRays
                if out is bool:
                    indch = np.zeros((self.nRays,),dtype=bool)
                    indch[ind] = True
                else:
                    indch = ind
        else:
            if out is int:
                indch = np.arange(0,self.nRays)
            elif out is bool:
                indch = np.ones((self.nRays,),dtype=bool)
        return indch

    def select(self, key=None, val=None, touch=None, log='any', out=int):
        """ Return the indices of the rays matching selection criteria

        The criterion can be of two types:
            - a key found in self.dchans, with a matching value
            - a touch tuple (indicating which element in self.config is touched
                by the desired rays)

        Parameters
        ----------
        key :    None / str
            A key to be found in self.dchans
        val :   int / str / float / list of such
            The value to be matched
            If a list of values is provided, the behaviour depends on log
        log :   str
            A flag indicating which behaviour to use when val is a list
                - any : Returns indices of rays matching any value in val
                - all : Returns indices of rays matching all values in val
                - not : Returns indices of rays matching None of the val
        touch:  None / str / int / tuple
            Used if key is None
            Tuple that can be of len()=1, 2 or 3
            Tuple indicating you want the rays that are touching some specific elements of self.config:
                - touch[0] : str / int or list of such
                    str : a 'Cls_Name' string indicating the element
                    int : the index of the element in self.lStruct_computeInOut
                - touch[1] : int / list of int
                    Indices of the desired segments on the polygon
                    (i.e.: of the cross-section polygon of the above element)
                - touch[2] : int / list of int
                    Indices, if relevant, of the toroidal / linear unit
                    Only relevant when the element has noccur>1
            In this case only log='not' has an effect
        out :   str
            Flag indicating whether to return:
                - bool : a (nRays,) boolean array of indices
                - int :  a (N,) array of int indices (N=number of matching rays)

        Returns
        -------
        ind :   np.ndarray
            The array of matching rays

        """
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
                lint = [int,np.int64]
                larr = [list,tuple,np.ndarray]
                touch = [touch] if not type(touch) is list else touch
                assert len(touch) in [1,2,3]

                def _check_touch(tt):
                    cS = type(tt) is str and len(tt.split('_'))==2
                    c0 = type(tt) in lint
                    c1 = type(tt) in larr and len(tt)>=0
                    c1 = c1 and all([type(t) in lint for t in tt])
                    return cS, c0, c1
                for ii in range(0,3-len(touch)):
                    touch.append([])

                ntouch = len(touch)
                assert ntouch == 3

                for ii in range(0,ntouch):
                    cS, c0, c1 = _check_touch(touch[ii])

                    if not (cS or c0 or c1):
                        msg = "Provided touch is not valid:\n"%touch
                        msg += "    - Provided: %s\n"%str(touch)
                        msg += "Please provide either:\n"
                        msg += "    - str in the form 'Cls_Name'\n"
                        msg += "    - int (index)\n"
                        msg += "    - array of int indices"
                        raise Exception(msg)

                    if cS:
                        lS = self.lStruct_computeInOut
                        k0, k1 = touch[ii].split('_')
                        ind = [jj for jj in range(0,len(lS))
                               if lS[jj].Id.Cls==k0 and lS[jj].Id.Name==k1]
                        assert len(ind)==1
                        touch[ii] = [ind[0]]
                    elif c0:
                        touch[ii] = [touch[ii]]

                # Common part
                ind = np.zeros((ntouch,self.nRays),dtype=bool)
                for i in range(0,ntouch):
                    if len(touch[i])==0:
                        ind[i,:] = True
                    else:
                        for n in range(0,len(touch[i])):
                            ind[i,:] = np.logical_or(ind[i,:],
                                                     self._dgeom['indout'][i,:]==touch[i][n])
                ind = np.all(ind,axis=0)
                if log=='not':
                    ind[:] = ~ind
        if out is int:
            ind = ind.nonzero()[0]
        return ind

    def get_subset(self, indch=None):
        if indch is None:
            return self
        else:
            indch = self._check_indch(indch)
            d = self.to_dict()
            d['dId_dall_Name'] = d['dId_dall_Name']+'-subset'
            if self.dchans!={} and self.dchans is not None:
                for k in self.dchans.keys():
                    C0 = isinstance(v,np.ndarray) and self.nRays in v.shape
                    if C0:
                        if v.ndim==1:
                            d['dchans_%s'%k] = v[indch]
                        elif v.ndim==2 and v.shape[1]==self.nRays:
                            d['dchans_%s'%k] = v[:,indch]

            # Geom
            for k in self.dgeom.keys():
                v = d['dgeom_%s'%k]
                C0 = isinstance(v,np.ndarray) and self.nRays in v.shape
                if C0:
                    if v.ndim==1:
                        d['dgeom_%s'%k] = v[indch]
                    elif v.ndim==2 and v.shape[1]==self.nRays:
                        d['dgeom_%s'%k] = v[:,indch]

            # X12
            if self._is2D():
                for k in self.dX12.keys():
                    v = d['dX12_%s'%k]
                    C0 = isinstance(v,np.ndarray) and self.nRays in v.shape
                    if C0:
                        if v.ndim==1:
                            d['dX12_%s'%k] = v[indch]
                        elif v.ndim==2 and v.shape[1]==self.nRays:
                            d['dX12_%s'%k] = v[:,indch]

            # Sino
            for k in self.dsino.keys():
                v = d['dsino_%s'%k]
                C0 = isinstance(v,np.ndarray) and self.nRays in v.shape
                if C0:
                    if v.ndim==1:
                        d['dsino_%s'%k] = v[indch]
                    elif v.ndim==2 and v.shape[1]==self.nRays:
                        d['dsino_%s'%k] = v[:,indch]

            # Recreate from dict
            obj = self.__class__(fromdict=d)
        return obj

    def _get_plotL(self, Lplot='Tot', proj='All', ind=None, multi=False):
        """ Get the (R,Z) coordinates of the cross-section projections """
        ind = self._check_indch(ind)
        if ind.size>0:
            Ds, us = self.D[:,ind], self.u[:,ind]
            if ind.size==1:
                Ds, us = Ds.reshape((3,1)), us.reshape((3,1))
            kPIn, kPOut = self.kIn[ind], self.kOut[ind]
            if self.config.Id.Type=='Tor':
                kRMin = self._dgeom['kRMin'][ind]
            else:
                kRMin = None
            pts = _comp.LOS_CrossProj(self.config.Id.Type, Ds, us,
                                      kPIn, kPOut, kRMin, proj=proj,
                                      Lplot=Lplot, multi=multi)
        else:
            pts = None
        return pts

    def get_sample(self, res, resMode='abs', DL=None, method='sum', ind=None,
                  compact=False):
        """ Return a linear sampling of the LOS

        The LOS is sampled into a series a points and segments lengths
        The resolution (segments length) is <= res
        The sampling can be done according to different methods
        It is possible to sample only a subset of the LOS

        Parameters
        ----------
        res:     float
            Desired resolution
        resMode: str
            Flag indicating res should be understood as:
                - 'abs':    an absolute distance in meters
                - 'rel':    a relative distance (fraction of the LOS length)
        DL:     None / iterable
            The fraction [L1;L2] of the LOS that should be sampled, where
            L1 and L2 are distances from the starting point of the LOS (LOS.D)
        method: str
            Flag indicating which to use for sampling:
                - 'sum':    the LOS is sampled into N segments of equal length,
                            where N is the smallest int such that:
                                * segment length <= resolution(res,resMode)
                            The points returned are the center of each segment
                - 'simps':  the LOS is sampled into N segments of equal length,
                            where N is the smallest int such that:
                                * segment length <= resolution(res,resMode)
                                * N is even
                            The points returned are the egdes of each segment
                - 'romb':   the LOS is sampled into N segments of equal length,
                            where N is the smallest int such that:
                                * segment length <= resolution(res,resMode)
                                * N = 2^k + 1
                            The points returned are the egdes of each segment

        Returns
        -------
        pts:    np.ndarray
            A (3,NP) array of NP points along the LOS in (X,Y,Z) coordinates
        k:      np.ndarray
            A (NP,) array of the points distances from the LOS starting point
        reseff: float
            The effective resolution (<= res input), as an absolute distance

        """
        ind = self._check_indch(ind)
        # preload k
        kIn = self.kIn
        kOut = self.kOut

        # Preformat DL
        if DL is None:
            DL = np.array([kIn[ind], kOut[ind]])
        elif np.asarray(DL).size==2:
            DL = np.tile(np.asarray(DL).ravel(),(len(ind),1)).T
        DL = np.ascontiguousarray(DL).astype(float)
        assert type(DL) is np.ndarray and DL.ndim==2
        assert DL.shape==(2,len(ind)), "Arg DL has wrong shape !"

        # Check consistency of limits
        ii = DL[0,:] < kIn[ind]
        DL[0,ii] = kIn[ind][ii]
        ii[:] = DL[0,:] >= kOut[ind]
        DL[0,ii] = kOut[ind][ii]
        ii[:] = DL[1,:] > kOut[ind]
        DL[1,ii] = kOut[ind][ii]
        ii[:] = DL[1,:] <= kIn[ind]
        DL[1,ii] = kIn[ind][ii]

        # Preformat Ds, us
        Ds, us = self.D[:,ind], self.u[:,ind]
        if len(ind)==1:
            Ds, us = Ds.reshape((3,1)), us.reshape((3,1))
        Ds, us = np.ascontiguousarray(Ds), np.ascontiguousarray(us)

        # Launch    # NB : find a way to exclude cases with DL[0,:]>=DL[1,:] !!
        # Todo : reverse in _GG : make compact default for faster computation !
        lpts, k, reseff = _GG.LOS_get_sample(Ds, us, res, DL,
                                             dLMode=resMode, method=method)
        if compact:
            pts = np.concatenate(lpts, axis=1)
            ind = np.array([pt.shape[1] for pt in lpts], dtype=int)
            ind = np.cumsum(ind)[:-1]
            return pts, k, reseff, ind
        else:
            return lpts, k, reseff

    def _kInOut_IsoFlux_inputs(self, lPoly, lVIn=None):

        if self._method=='ref':
            D, u = np.ascontiguousarray(self.D), np.ascontiguousarray(self.u)
            Lim = self.config.Lim
            nLim = self.config.nLim
            Type = self.config.Id.Type

            largs = [D, u, lPoly[0], lVIn[0]]
            dkwd = dict(Lim=Lim, nLim=nLim, VType=Type)
        elif self._method=="optimized":
            D = np.ascontiguousarray(self.D)
            u = np.ascontiguousarray(self.u)
            if np.size(self.config.Lim) == 0 or self.config.Lim is None:
                Lim = np.array([])
            else:
                Lim = np.asarray(self.config.Lim)
                if np.size(np.shape(Lim)) > 1 :
                    # in case self.config.Lim = [[L0, L1]]
                    Lim = np.asarray([Lim[0][0], Lim[0][1]])
            nLim = self.config.nLim
            Type = self.config.Id.Type
            largs = [D, u, lPoly[0], lVIn[0]]
            dkwd = dict(ves_lims=Lim, ves_type=Type)
        else:
            # To be adjusted later
            pass
        return largs, dkwd

    def _kInOut_IsoFlux_inputs_usr(self, lPoly, lVIn=None):

        # Check lPoly
        if type(lPoly) is np.ndarray:
            lPoly = [lPoly]
        lPoly = [np.ascontiguousarray(pp) for pp in lPoly]
        msg = "Arg lPoly must be a list of (2,N) or (N,2) np.ndarrays !"
        assert all([pp.ndim==2 and 2 in pp.shape for pp in lPoly]), msg
        nPoly = len(lPoly)
        for ii in range(0,nPoly):
            if lPoly[ii].shape[0]!=2:
                lPoly[ii] = lPoly[ii].T
                # Check closed and anti-clockwise
                lPoly[ii] = _GG.Poly_Order(lPoly[ii], Clock=False, close=True)

        # Check lVIn
        if lVIn is None:
            lVIn = []
            for pp in lPoly:
                VIn = np.diff(pp, axis=1)
                VIn = VIn/(np.sqrt(np.sum(VIn**2,axis=0))[np.newaxis,:])
                VIn = np.ascontiguousarray([-VIn[1,:],VIn[0,:]])
                lVIn.append(VIn)
        else:
            if type(lVIn) is np.ndarray:
                lVIn = [lVIn]
            assert len(lVIn)==nPoly
            lVIn = [np.ascontiguousarray(pp) for pp in lVIn]
            msg = "Arg lVIn must be a list of (2,N) or (N,2) np.ndarrays !"
            assert all([pp.ndim==2 and 2 in pp.shape for pp in lVIn]), msg
            for ii in range(0,nPoly):
                if lVIn[ii].shape[0]!=2:
                    lVIn[ii] = lVIn[ii].T
                    lVIn[ii] = lVIn[ii]/(np.sqrt(np.sum(lVIn[ii]**2,axis=0))[np.newaxis,:])
                    assert lVIn[ii].shape==(2,lPoly[ii].shape[1]-1)
                    vect = np.diff(lPoly[ii],axis=1)
                    det = vect[0,:]*lVIn[ii][1,:] - vect[1,:]*lVIn[ii][0,:]
                    if not np.allclose(np.abs(det),1.):
                        msg = "Each lVIn must be perp. to each lPoly segment !"
                        raise Exception(msg)
                    ind = np.abs(det+1)<1.e-12
                    lVIn[ii][:,ind] = -lVIn[ii][:,ind]

        return nPoly, lPoly, lVIn

    def calc_kInkOut_IsoFlux(self, lPoly, lVIn=None, Lim=None,
                             kInOut=True):
        """ Calculate the intersection points of each ray with each isoflux

        The isofluxes are provided as a list of 2D closed polygons

        The intersections are the inward and outward intersections
        They are retruned as two np.ndarrays: kIn and kOut
        Each array contains the length parameter along the ray for each isoflux

        Parameters
        ----------


        Returns
        -------

        """

        # Preformat input
        nPoly, lPoly, lVIn = self._kInOut_IsoFlux_inputs_usr(lPoly, lVIn=lVIn)

        # Prepare output
        kIn = np.full((self.nRays,nPoly), np.nan)
        kOut = np.full((self.nRays,nPoly), np.nan)

        # Compute intersections
        assert(self._method in ['ref', 'optimized'])
        if self._method=='ref':
            for ii in range(0,nPoly):
                largs, dkwd = self._kInOut_IsoFlux_inputs([lPoly[ii]],
                                                          lVIn=[lVIn[ii]])
                out = _GG.SLOW_LOS_Calc_PInOut_VesStruct(*largs, **dkwd)
                PIn, POut, kin, kout, VperpIn, vperp, IIn, indout = out
                kIn[:,ii], kOut[:,ii] = kin, kout
        elif self._method=="optimized":
            for ii in range(0,nPoly):
                largs, dkwd = self._kInOut_IsoFlux_inputs([lPoly[ii]],
                                                          lVIn=[lVIn[ii]])

                out = _GG.LOS_Calc_PInOut_VesStruct(*largs, **dkwd)
                kin, kout, _, _ = out
                kIn[:,ii], kOut[:,ii] = kin, kout
        if kInOut:
            indok = ~np.isnan(kIn)
            ind = np.zeros((self.nRays,nPoly), dtype=bool)
            kInref = np.tile(self.kIn[:,np.newaxis],nPoly)
            kOutref = np.tile(self.kOut[:,np.newaxis],nPoly)
            ind[indok] = (kIn[indok]<kInref[indok]) | (kIn[indok]>kOutref[indok])
            kIn[ind] = np.nan

            ind[:] = False
            indok[:] = ~np.isnan(kOut)
            ind[indok] = (kOut[indok]<kInref[indok]) | (kOut[indok]>kOutref[indok])
            kOut[ind] = np.nan

        return kIn, kOut


    def calc_signal(self, ff, t=None, ani=None, fkwdargs={}, Brightness=True,
                    res=0.005, DL=None, resMode='abs', method='sum',
                    ind=None, out=object, plot=True, dataname=None,
                    fs=None, dmargin=None, wintit=None, invert=True,
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
        msg = "Arg out must be in [object,np.ndarray]"
        assert out in [object,np.ndarray], msg
        assert type(Brightness) is bool, "Arg Brightness must be a bool !"
        if Brightness is False and self.Etendues is None:
            msg = "Etendue must be set if Brightness is False !"
            raise Exception(msg)

        # Preformat ind
        ind = self._check_indch(ind)
        # Preformat DL
        kIn, kOut = self.kIn, self.kOut
        if DL is None:
            DL = np.array([kIn[ind], kOut[ind]])
        elif np.asarray(DL).size==2:
            DL = np.tile(np.asarray(DL).ravel()[:,np.newaxis],len(ind))
        DL = np.ascontiguousarray(DL).astype(float)
        assert type(DL) is np.ndarray and DL.ndim==2
        assert DL.shape==(2,len(ind)), "Arg DL has wrong shape !"

        # check limits
        ii = DL[0,:] < kIn[ind]
        DL[0,ii] = kIn[ind][ii]
        ii[:] = DL[0,:] >= kOut[ind]
        DL[0,ii] = kOut[ind][ii]
        ii[:] = DL[1,:] > kOut[ind]
        DL[1,ii] = kOut[ind][ii]
        ii[:] = DL[1,:] <= kIn[ind]
        DL[1,ii] = kIn[ind][ii]

        # Preformat Ds, us and Etendue
        Ds, us = self.D[:,ind], self.u[:,ind]
        if Brightness is False:
            E = self.Etendues
            if E.size==self.nRays:
                E = E[ind]

        # Preformat signal
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
            s = _GG.LOS_calc_signal(ff, Ds, us, res, DL,
                                    dLMode=resMode, method=method,
                                    t=t, Ani=ani, fkwdargs=fkwdargs, Test=True)
            if t is None or len(t)==1:
                sig[indok] = s
            else:
                sig[:,indok] = s

        # Format output
        if Brightness is False:
            if dataname is None:
                dataname = r"LOS-integral x Etendue"
            if t is None or len(t)==1 or E.size==1:
                sig = sig*E
            else:
                sig = sig*E[np.newaxis,:]
            if units is None:
                units = r"origin x $m^3.sr$"
        else:
            if dataname is None:
                dataname = r"LOS-integral"
            if units is None:
                units = r"origin x m"

        if plot or out in [object,'object']:
            kwdargs = dict(data=sig, t=t, lCam=self, Name=self.Id.Name,
                           dlabels={'data':{'units':units, 'name':dataname}},
                           Exp=self.Id.Exp, Diag=self.Id.Diag)
            import tofu.data as tfd
            if self._is2D():
                osig = tfd.DataCam2D(**kwdargs)
            else:
                osig = tfd.DataCam1D(**kwdargs)
            if plot:
                kh = osig.plot(fs=fs, dmargin=dmargin, wintit=wintit,
                               plotmethod=plotmethod, invert=invert,
                               draw=draw, connect=connect)

        if out in [object, 'object']:
            return osig
        else:
            return sig, units

    def plot(self, lax=None, proj='all', Lplot=_def.LOSLplot, element='L',
             element_config='P', Leg='', dL=None, dPtD=_def.LOSMd,
             dPtI=_def.LOSMd, dPtO=_def.LOSMd, dPtR=_def.LOSMd,
             dPtP=_def.LOSMd, dLeg=_def.TorLegd, multi=False, ind=None,
             fs=None, wintit=None, draw=True, Test=True):
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
        element :   str
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
        element_config : str
            Fed to self.config.plot()
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

        return _plot.Rays_plot(self, Lax=lax, Proj=proj, Lplot=Lplot,
                               element=element, element_config=element_config, Leg=Leg,
                               dL=dL, dPtD=dPtD, dPtI=dPtI, dPtO=dPtO, dPtR=dPtR,
                               dPtP=dPtP, dLeg=dLeg, multi=multi, ind=ind,
                               fs=fs, wintit=wintit, draw=draw, Test=Test)


    def plot_sino(self, ax=None, element=_def.LOSImpElt, Sketch=True,
                  Ang=_def.LOSImpAng, AngUnit=_def.LOSImpAngUnit, Leg=None,
                  dL=_def.LOSMImpd, dVes=_def.TorPFilld, dLeg=_def.TorLegd,
                  ind=None, multi=False,
                  fs=None, wintit=None, draw=True, Test=True):
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
        if self._dsino['RefPt'] is None:
            msg = "The sinogram ref. point is not set !"
            msg += "\n  => run self.set_dsino()"
            raise Exception(msg)
        return _plot.GLOS_plot_Sino(self, Proj='Cross', ax=ax, Elt=element, Leg=Leg,
                                    Sketch=Sketch, Ang=Ang, AngUnit=AngUnit,
                                    dL=dL, dVes=dVes, dLeg=dLeg,
                                    ind=ind, fs=fs, wintit=wintit,
                                    draw=draw, Test=Test)


    def get_touch_dict(self, ind=None, out=bool):
        """ Get a dictionnary of Cls_Name struct with indices of Rays touching

        Only includes Struct object with compute = True
            (as returned by self.lStruct__computeInOut_computeInOut)
        Also return the associated colors
        If in is not None, the indices for each Struct are split between:
            - indok : rays touching Struct and in ind
            - indout: rays touching Struct but not in ind

        """
        if self.config is None:
            msg = "Config must be set in order to get touch dict !"
            raise Exception(msg)

        dElt = {}
        ind = self._check_indch(ind, out=bool)
        for ss in self.lStruct_computeInOut:
            kn = "%s_%s"%(ss.__class__.__name__, ss.Id.Name)
            indtouch = self.select(touch=kn, out=bool)
            if np.any(indtouch):
                indok  = indtouch & ind
                indout = indtouch & ~ind
                if np.any(indok) or np.any(indout):
                    if out == int:
                        indok  = indok.nonzero()[0]
                        indout = indout.nonzero()[0]
                    dElt[kn] = {'indok':indok, 'indout':indout,
                                'col':ss.get_color()}
        return dElt

    def get_touch_colors(self, ind=None, dElt=None,
                         cbck=(0.8,0.8,0.8), rgba=True):
        if dElt is None:
            dElt = self.get_touch_dict(ind=None, out=bool)
        else:
            assert type(dElt) is dict
            assert all([type(k) is str and type(v) is dict
                        for k,v in dElt.items()])

        if rgba:
            colors = np.tile(mpl.colors.to_rgba(cbck), (self.nRays,1)).T
            for k,v in dElt.items():
                colors[:,v['indok']] = np.r_[mpl.colors.to_rgba(v['col'])][:,None]
        else:
            colors = np.tile(mpl.colors.to_rgb(cbck), (self.nRays,1)).T
            for k,v in dElt.items():
                colors[:,v['indok']] = np.r_[mpl.colors.to_rgb(v['col'])][:,None]
        return colors

    def plot_touch(self, key=None, quant='lengths', invert=None, ind=None,
                   Bck=True, fs=None, wintit=None, tit=None,
                   connect=True, draw=True):

        out = _plot.Rays_plot_touch(self, key=key, Bck=Bck,
                                    quant=quant, ind=ind, invert=invert,
                                    connect=connect, fs=fs, wintit=wintit,
                                    tit=tit, draw=draw)
        return out




########################################
#       CamLOS subclasses
########################################

sig = inspect.signature(Rays)
params = sig.parameters





class CamLOS1D(Rays): pass

lp = [p for p in params.values() if p.name != 'dX12']
CamLOS1D.__signature__ = sig.replace(parameters=lp)



class CamLOS2D(Rays):

    def _isImage(self):
        return self._dgeom['isImage']

    @property
    def dX12(self):
        if self._dX12 is not None and self._dX12['from'] == 'geom':
            dX12 = self._dgeom['dX12']
        else:
            dX12 = self._dX12
        return dX12

    def get_X12plot(self, plot='imshow'):
        if plot == 'imshow':
            x1, x2 = self.dX12['x1'], self.dX12['x2']
            x1min, Dx1min = x1[0], 0.5*(x1[1]-x1[0])
            x1max, Dx1max = x1[-1], 0.5*(x1[-1]-x1[-2])
            x2min, Dx2min = x2[0], 0.5*(x2[1]-x2[0])
            x2max, Dx2max = x2[-1], 0.5*(x2[-1]-x2[-2])
            extent = (x1min - Dx1min, x1max + Dx1max,
                      x2min - Dx2min, x2max + Dx2max)
            indr = self.dX12['indr']
            return x1, x2, indr, extent


    """
    def set_e12(self, e1=None, e2=None):
        assert e1 is None or (hasattr(e1,'__iter__') and len(e1)==3)
        assert e2 is None or (hasattr(e2,'__iter__') and len(e2)==3)
        if e1 is None:
            e1 = self._dgeom['e1']
        else:
            e1 = np.asarray(e1).astype(float).ravel()
        e1 = e1 / np.linalg.norm(e1)
        if e2 is None:
            e2 = self._dgeom['e2']
        else:
            e2 = np.asarray(e1).astype(float).ravel()
        e2 = e2 / np.linalg.norm(e2)
        assert np.abs(np.sum(e1*self._dgeom['nIn']))<1.e-12
        assert np.abs(np.sum(e2*self._dgeom['nIn']))<1.e-12
        assert np.abs(np.sum(e1*e2))<1.e-12
        self._dgeom['e1'] = e1
        self._dgeom['e2'] = e2

    def get_ind_flatimg(self, direction='flat2img'):
        assert direction in ['flat2img','img2flat']
        assert self._dgeom['ddetails'] is not None
        assert all([ss in self._dgeom['ddetails'].keys()
                    for ss in ['x12','x1','x2']])
        x1b = 0.5*(self._dgeom['ddetails']['x1'][1:]
                   + self._dgeom['ddetails']['x1'][:-1])
        x2b = 0.5*(self._dgeom['ddetails']['x2'][1:]
                   + self._dgeom['ddetails']['x2'][:-1])
        ind = np.array([np.digitize(self._dgeom['ddetails']['x12'][0,:], x1b),
                        np.digitize(self._dgeom['ddetails']['x12'][0,:], x2b)])
        if direction == 'flat2img':
            indr = np.zeros((self._dgeom['ddetails']['x1'].size,
                             self._dgeom['ddetails']['x2'].size),dtype=int)
            indr[ind[0,:],ind[1,:]] = np.arange(0,self._dgeom['nRays'])
            ind = indr
        return ind

    def get_X12(self, out='imshow'):

        if out == 'imshow':
            x1, x2 = self._dgeom['x1'], self._dgeom['x2']
            dx1, dx2 = 0.5*(x1[1]-x1[0]), 0.5*(x2[1]-x2[0])
            extent = (x1[0]-dx1, x1[-1]+dx1, x2[0]-dx2, x2[-1]+dx2)
            return x1, x2, extent

        # TBF
        if self._X12 is None:
            Ds = self.D
            C = np.mean(Ds,axis=1)
            X12 = Ds-C[:,np.newaxis]
            X12 = np.array([np.sum(X12*self._dgeom['e1'][:,np.newaxis],axis=0),
                            np.sum(X12*self._dgeom['e2'][:,np.newaxis],axis=0)])
        else:
            X12 = self._X12
        if X12 is None or out.lower()=='1d':
            DX12 = None
        else:
            x1u, x2u, ind, DX12 = utils.get_X12fromflat(X12)
            if out.lower()=='2d':
                X12 = [x1u, x2u, ind]
        return X12, DX12
    """

lp = [p for p in params.values()]
CamLOS2D.__signature__ = sig.replace(parameters=lp)







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
