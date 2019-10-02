
"""
This module is the geometrical part of the ToFu general package
It includes all functions and object classes necessary for tomography on Tokamaks
"""

# Built-in
import os
import warnings
#import copy


# Common
import numpy as np
import datetime as dtm

# ToFu-specific
from tofu import __version__ as __version__
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

__all__ = ['Crystal']



_Type = 'Tor'
_NTHREADS = 16

"""
###############################################################################
###############################################################################
                        Ves class and functions
###############################################################################
"""



class CrystalBragg(utils.ToFuObject):
    """ A class defining crystals for Bragg diffraction

    A crystal can be of Type flat, cylindrical, spherical or elliptical
    It is characterized by its:
        - geometry (Type, dimensions, curvature radii and position/orientation)
        - Material and lattice
        - Bragg parameters (angle vs lambda)


    Parameters
    ----------
    Id :            str / tfpf.ID
        A name string or a pre-built tfpf.ID class to be used to identify this
        particular instance, if a string is provided, it is fed to tfpf.ID()
    dgeom :         dict
        An array (2,N) or (N,2) defining the contour of the vacuum vessel in a
        cross-section, if not closed, will be closed automatically
     dspectral:     str
        Flag indicating whether the vessel will be a torus ('Tor') or a linear
        device ('Lin')
    SavePath :      None / str
        If provided, forces the default saving path of the object to the
        provided value

    """

    # Fixed (class-wise) dictionary of default properties
    _ddef = {'Id':{'shot':0,
                   'include':['Mod','Cls','Exp','Diag',
                              'Name','shot','version']},
             'dgeom':{'Type':'Tor', 'Lim':[], 'arrayorder':'C'},
             'dsino':{},
             'dphys':{},
             'dreflect':{'Type':'specular'},
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
    _DREFLECT_DTYPES = {'specular':0, 'diffusive':1, 'ccube':2}

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
        self._dreflect = dict.fromkeys(self._get_keys_dreflect())
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
