
"""
This module is the geometrical part of the ToFu general package
It includes all functions and object classes necessary for tomography on Tokamaks
"""

# Built-in
import sys
import os
import warnings
import copy


# Common
import numpy as np
import datetime as dtm
import matplotlib.pyplot as plt
import matplotlib as mpl

# ToFu-specific
from tofu import __version__ as __version__
import tofu.pathfile as tfpf
import tofu.utils as utils
try:
    import tofu.geom._def as _def
    import tofu.geom._GG as _GG
    import tofu.geom._comp_optics as _comp_optics
    import tofu.geom._plot_optics as _plot_optics
except Exception:
    from . import _def as _def
    from . import _GG as _GG
    from . import _comp_optics as _comp_optics
    from . import _plot_optics as _plot_optics



__all__ = ['CrystalBragg']



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

    A crystal can be of Type flat, cylindrical or spherical
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
    _ddef = {'Id':{'shot': 0, 'Exp': 'dummy', 'Diag': 'dummy',
                   'include':['Mod', 'Cls', 'Exp', 'Diag',
                              'Name', 'shot', 'version']},
             'dgeom':{'Type': 'sph', 'Typeoutline': 'rect'},
             'dmat':{},
             'dbragg':{},
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
              '3d':{}}
    _DEFLMOVEOK = ['rotate']
    _DEFLAMB = 3.971561e-10
    _DEFNPEAKS = 12
    # _DREFLECT_DTYPES = {'specular':0, 'diffusive':1, 'ccube':2}


    # Does not exist beofre Python 3.6 !!!
    def __init_subclass__(cls, color='k', **kwdargs):
        # Python 2
        super(CrystalBragg,cls).__init_subclass__(**kwdargs)
        # Python 3
        #super().__init_subclass__(**kwdargs)
        cls._ddef = copy.deepcopy(CrystalBragg._ddef)
        cls._dplot = copy.deepcopy(CrystalBragg._dplot)
        cls._set_color_ddef(cls._color)

    @classmethod
    def _set_color_ddef(cls, color):
        cls._ddef['dmisc']['color'] = mpl.colors.to_rgba(color)

    def __init__(self, dgeom=None, dmat=None, dbragg=None,
                 Id=None, Name=None, Exp=None, Diag=None, shot=None,
                 fromdict=None, sep=None,
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
        super(CrystalBragg,self).__init__(**kwdargs)

    def _reset(self):
        # super()
        super(CrystalBragg,self)._reset()
        self._dgeom = dict.fromkeys(self._get_keys_dgeom())
        self._dmat = dict.fromkeys(self._get_keys_dmat())
        self._dbragg = dict.fromkeys(self._get_keys_dbragg())
        self._dmisc = dict.fromkeys(self._get_keys_dmisc())
        #self._dplot = copy.deepcopy(self.__class__._ddef['dplot'])

    @classmethod
    def _checkformat_inputs_Id(cls, Id=None, Name=None,
                               Exp=None, Diag=None, shot=None, Type=None,
                               include=None,
                               **kwdargs):
        if Id is not None:
            assert isinstance(Id,utils.ID)
            Name, Exp, Type = Id.Name, Id.Exp, Id.Type
        if Type is None:
            Type = cls._ddef['dgeom']['Type']
        if Exp is None:
            Exp = cls._ddef['Id']['Exp']
        if Diag is None:
            Diag = cls._ddef['Id']['Diag']
        if shot is None:
            shot = cls._ddef['Id']['shot']
        if include is None:
            include = cls._ddef['Id']['include']

        dins = {'Name':{'var':Name, 'cls':str},
                'Exp': {'var':Exp, 'cls':str},
                'Diag': {'var':Diag, 'cls':str},
                'shot': {'var':shot, 'cls':int},
                'Type': {'var':Type, 'in':['sph']},
                'include':{'var':include, 'listof':str}}
        dins, err, msg = cls._check_InputsGeneric(dins)
        if err:
            raise Exception(msg)

        kwdargs.update({'Name':Name, 'shot':shot,
                        'Exp':Exp, 'Diag':Diag, 'Type':Type,
                        'include':include})
        return kwdargs

    ###########
    # Get largs
    ###########

    @staticmethod
    def _get_largs_dgeom(sino=True):
        largs = ['dgeom']
        return largs

    @staticmethod
    def _get_largs_dmat():
        largs = ['dmat']
        return largs

    @staticmethod
    def _get_largs_dbragg():
        largs = ['dbragg']
        return largs

    @staticmethod
    def _get_largs_dmisc():
        largs = ['color']
        return largs

    ###########
    # Get check and format inputs
    ###########

    @classmethod
    def _checkformat_dgeom(cls, dgeom=None):
        if dgeom is None:
            return
        assert isinstance(dgeom, dict)
        lkok = cls._get_keys_dgeom()
        assert all([isinstance(ss, str) and ss in lkok for ss in dgeom.keys()])
        for kk in cls._ddef['dgeom'].keys():
            dgeom[kk] = dgeom.get(kk, cls._ddef['dgeom'][kk])
        for kk in lkok:
            dgeom[kk] = dgeom.get(kk, None)
        if dgeom['center'] is not None:
            dgeom['center'] = np.atleast_1d(dgeom['center']).ravel()
            assert dgeom['center'].size == 3
        else:
            assert dgeom['summit'] is not None
            assert dgeom['rcurve'] is not None
        if dgeom['summit'] is not None:
            dgeom['summit'] = np.atleast_1d(dgeom['summit']).ravel()
            assert dgeom['summit'].size == 3
        else:
            assert dgeom['center'] is not None
            assert dgeom['rcurve'] is not None
        if dgeom['extenthalf'] is not None:
            dgeom['extenthalf'] = np.atleast_1d(dgeom['extenthalf'])
            assert dgeom['extenthalf'].size == 2
        if dgeom['rcurve'] is not None:
            dgeom['rcurve'] = float(dgeom['rcurve'])
        if dgeom['nout'] is not None:
            dgeom['nout'] = np.atleast_1d(dgeom['nout'])
            dgeom['nout'] = dgeom['nout'] / np.linalg.norm(dgeom['nout'])
            assert dgeom['nout'].size == 3
        if dgeom['nin'] is not None:
            dgeom['nin'] = np.atleast_1d(dgeom['nin'])
            dgeom['nin'] = dgeom['nin'] / np.linalg.norm(dgeom['nin'])
            assert dgeom['nin'].size == 3
        if dgeom['e1'] is not None:
            dgeom['e1'] = np.atleast_1d(dgeom['e1'])
            dgeom['e1'] = dgeom['e1'] / np.linalg.norm(dgeom['e1'])
            assert dgeom['e1'].size == 3
            assert dgeom['e2'] is not None
        if dgeom['e2'] is not None:
            dgeom['e2'] = np.atleast_1d(dgeom['e2'])
            dgeom['e2'] = dgeom['e2'] / np.linalg.norm(dgeom['e2'])
            assert dgeom['e2'].size == 3
        if dgeom['e1'] is not None:
            assert np.abs(np.sum(dgeom['e1']*dgeom['e2'])) < 1.e-12
            if dgeom['nout'] is not None:
                assert np.abs(np.sum(dgeom['e1']*dgeom['nout'])) < 1.e-12
                assert np.abs(np.sum(dgeom['e2']*dgeom['nout'])) < 1.e-12
                assert np.linalg.norm(np.cross(dgeom['e1'], dgeom['e2'])
                                      - dgeom['nout']) < 1.e-12
            if dgeom['nin'] is not None:
                assert np.abs(np.sum(dgeom['e1']*dgeom['nin'])) < 1.e-12
                assert np.abs(np.sum(dgeom['e2']*dgeom['nin'])) < 1.e-12
                assert np.linalg.norm(np.cross(dgeom['e1'], dgeom['e2'])
                                      + dgeom['nin']) < 1.e-12
        if dgeom['rotateaxis'] is not None:
            rotax = np.asarray(dgeom['rotateaxis'], dtype=float)
            assert rotax.shape == (2, 3)
            rotax[1,:] = rotax[1,:] / np.linalg.norm(rotax[1,:])
            dgeom['rotateaxis'] = rotax
        if dgeom['rotateangle'] is not None:
            dgeom['rotateangle'] = float(dgeom['rotateangle'])
        if dgeom['mobile'] is None:
            if dgeom['rotateaxis'] is not None:
                dgeom['mobile'] = 'rotate'
            else:
                dgeom['mobile'] = False
        else:
            assert isinstance(dgeom['mobile'], str)
            assert dgeom['mobile'] in cls._DEFLMOVEOK
        return dgeom

    @classmethod
    def _checkformat_dmat(cls, dmat=None):
        if dmat is None:
            return
        assert isinstance(dmat, dict)
        lkok = cls._get_keys_dmat()
        assert all([isinstance(ss, str) for ss in dmat.keys()])
        assert all([ss in lkok for ss in dmat.keys()])
        for kk in cls._ddef['dmat'].keys():
            dmat[kk] = dmat.get(kk, cls._ddef['dmat'][kk])
        if dmat['d'] is not None:
            dmat['d'] = float(dmat['d'])
        if dmat['formula'] is not None:
            assert isinstance(dmat['formula'], str)
        if dmat['density'] is not None:
            dmat['density'] = float(dmat['density'])
        if dmat['lengths'] is not None:
            dmat['lengths'] = np.atleast_1d(dmat['lengths']).ravel()
        if dmat['angles'] is not None:
            dmat['angles'] = np.atleast_1d(dmat['angles']).ravel()
        if dmat['cut'] is not None:
            dmat['cut'] = np.atleast_1d(dmat['cut']).ravel().astype(int)
            assert dmat['cut'].size <= 4
        return dmat

    @classmethod
    def _checkformat_dbragg(cls, dbragg=None):
        if dbragg is None:
            return
        assert isinstance(dbragg, dict)
        lkok = ['angle']
        assert all([isinstance(ss, str) for ss in dbragg.keys()])
        assert all([ss in lkok for ss in dbragg.keys()])

        for kk in cls._ddef['dbragg'].keys():
            dbragg[kk] = dbragg.get(kk, cls._ddef['dbragg'][kk])
        return dbragg

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
        lk = ['Type', 'Typeoutline',
              'summit', 'center', 'extenthalf', 'surface',
              'nin', 'nout', 'e1', 'e2', 'rcurve',
              'mobile', 'rotateaxis', 'rotateangle']
        return lk

    @staticmethod
    def _get_keys_dmat():
        lk = ['formula', 'density', 'symmetry',
              'lengths', 'angles', 'cut', 'd']
        return lk

    @staticmethod
    def _get_keys_dbragg():
        lk = ['rockingcurve']
        return lk

    @staticmethod
    def _get_keys_dmisc():
        lk = ['color']
        return lk

    ###########
    # _init
    ###########

    def _init(self, dgeom=None, dmat=None, dbragg=None,
              color=None, **kwdargs):
        allkwds = dict(locals(), **kwdargs)
        largs = self._get_largs_dgeom()
        kwds = self._extract_kwdargs(allkwds, largs)
        self.set_dgeom(**kwds)
        largs = self._get_largs_dmat()
        kwds = self._extract_kwdargs(allkwds, largs)
        self.set_dmat(**kwds)
        largs = self._get_largs_dbragg()
        kwds = self._extract_kwdargs(allkwds, largs)
        self._set_dbragg(**kwds)
        largs = self._get_largs_dmisc()
        kwds = self._extract_kwdargs(allkwds, largs)
        self._set_dmisc(**kwds)
        self._dstrip['strip'] = 0

    ###########
    # set dictionaries
    ###########

    def set_dgeom(self, dgeom=None):
        dgeom = self._checkformat_dgeom(dgeom)
        if dgeom['e1'] is not None:
            if dgeom['nout'] is None:
                dgeom['nout'] = np.cross(dgeom['e1'], dgeom['e2'])
            if dgeom['nin'] is None:
                dgeom['nin'] = -dgeom['nout']
            if dgeom['center'] is None:
                dgeom['center'] = dgeom['summit'] + dgeom['nin']*dgeom['rcurve']
            if dgeom['summit'] is None:
                dgeom['summit'] = dgeom['center'] + dgeom['nout']*dgeom['rcurve']

        if dgeom['extenthalf'] is not None:
            if dgeom['Type'] == 'sph' and dgeom['Typeoutline'] == 'rect':
                ind = np.argmax(dgeom['extenthalf'])
                dphi = dgeom['extenthalf'][ind]
                sindtheta = np.sin(dgeom['extenthalf'][ind-1])
                dgeom['surface'] = 4.*dgeom['rcurve']**2*dphi*sindtheta
        self._dgeom = dgeom

    def set_dmat(self, dmat=None):
        dmat = self._checkformat_dmat(dmat)
        self._dmat = dmat

    def _set_dbragg(self, dbragg=None):
        dbragg = self._checkformat_dbragg(dbragg)
        self._dbragg = dbragg

    def _set_color(self, color=None):
        color = self._checkformat_inputs_dmisc(color=color)
        self._dmisc['color'] = color
        self._dplot['cross']['dP']['color'] = color
        self._dplot['hor']['dP']['color'] = color
        # self._dplot['3d']['dP']['color'] = color

    def _set_dmisc(self, color=None):
        self._set_color(color)

    ###########
    # strip dictionaries
    ###########

    def _strip_dgeom(self, lkeep=None):
        lkeep = self._get_keys_dgeom()
        utils.ToFuObject._strip_dict(self._dgeom, lkeep=lkeep)

    def _strip_dmat(self, lkeep=None):
        lkeep = self._get_keys_dmat()
        utils.ToFuObject._strip_dict(self._dmat, lkeep=lkeep)

    def _strip_dbragg(self, lkeep=None):
        lkeep = self._get_keys_dbragg()
        utils.ToFuObject._strip_dict(self._dbragg, lkeep=lkeep)

    def _strip_dmisc(self, lkeep=['color']):
        utils.ToFuObject._strip_dict(self._dmisc, lkeep=lkeep)

    ###########
    # rebuild dictionaries
    ###########

    def _rebuild_dgeom(self, lkeep=None):
        lkeep = self._get_keys_dgeom()
        reset = utils.ToFuObject._test_Rebuild(self._dgeom, lkeep=lkeep)
        if reset:
            utils.ToFuObject._check_Fields4Rebuild(self._dgeom,
                                                   lkeep=lkeep, dname='dgeom')
            self._set_dgeom(dgeom=self._dgeom)

    def _rebuild_dmat(self, lkeep=None):
        lkeep = self._get_keys_dmat()
        reset = utils.ToFuObject._test_Rebuild(self._dmat, lkeep=lkeep)
        if reset:
            utils.ToFuObject._check_Fields4Rebuild(self._dmat,
                                                   lkeep=lkeep, dname='dmat')
            self.set_dmat(self._dmat)

    def _rebuild_dbragg(self, lkeep=None):
        lkeep = self._get_keys_dbragg()
        reset = utils.ToFuObject._test_Rebuild(self._dbragg, lkeep=lkeep)
        if reset:
            utils.ToFuObject._check_Fields4Rebuild(self._dbragg,
                                                   lkeep=lkeep, dname='dbragg')
            self.set_dbragg(self._dbragg)

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
        cls._dstrip['allowed'] = [0,1]
        nMax = max(cls._dstrip['allowed'])
        doc = """
                 1: Remove nothing"""
        doc = utils.ToFuObjectBase.strip.__doc__.format(doc,nMax)
        if sys.version[0]=='2':
            cls.strip.__func__.__doc__ = doc
        else:
            cls.strip.__doc__ = doc

    def strip(self, strip=0):
        # super()
        super(CrystalBragg, self).strip(strip=strip)

    def _strip(self, strip=0):
        if strip==0:
            self._rebuild_dgeom()
            self._rebuild_dmat()
            self._rebuild_dbragg()
            self._rebuild_dmisc()
        else:
            self._strip_dgeom()
            self._strip_dmat()
            self._strip_dbragg()
            self._strip_dmisc()

    def _to_dict(self):
        dout = {'dgeom':{'dict':self._dgeom, 'lexcept':None},
                'dmat':{'dict':self._dmat, 'lexcept':None},
                'dbragg':{'dict':self._dbragg, 'lexcept':None},
                'dmisc':{'dict':self._dmisc, 'lexcept':None},
                'dplot':{'dict':self._dplot, 'lexcept':None}}
        return dout

    def _from_dict(self, fd):
        self._dgeom.update(**fd.get('dgeom', {}))
        self._dmat.update(**fd.get('dmat', {}))
        self._dbragg.update(**fd.get('dbragg', {}))
        self._dmisc.update(**fd.get('dmisc', {}))
        self._dplot.update(**fd.get('dplot', {}))

    # -----------
    # Properties
    # -----------

    @property
    def Type(self):
        """Return the type of structure """
        return self._Id.Type

    @property
    def dgeom(self):
        return self._dgeom

    @property
    def dmat(self):
        """Return the polygon defining the structure cross-section"""
        return self._dmat

    @property
    def dbragg(self):
        """Return the polygon defining the structure cross-section"""
        return self._dbragg

    @property
    def dmisc(self):
        return self._dmisc

    @property
    def nin(self):
        return self._dgeom['nin']

    @property
    def nout(self):
        return self._dgeom['nout']

    @property
    def e1(self):
        return self._dgeom['e1']

    @property
    def e2(self):
        return self._dgeom['e2']

    @property
    def summit(self):
        return self._dgeom['summit']

    @property
    def center(self):
        return self._dgeom['center']

    # -----------------
    # methods for color
    # -----------------

    def set_color(self, col):
        self._set_color(col)

    def get_color(self):
        return self._dmisc['color']

    # -----------------
    # methods for printing
    # -----------------

    def get_summary(self, sep='  ', line='-', just='l',
                    table_sep=None, verb=True, return_=False):
        """ Summary description of the object content """

        # -----------------------
        # Build material
        col0 = ['formula', 'symmetry', 'cut', 'density',
                'd (A)', 'bragg(%s A) (deg)'%str(self._DEFLAMB)]
        ar0 = [self._dmat['formula'], self._dmat['symmetry'],
               str(self._dmat['cut']), str(self._dmat['density']),
               '{0:5.3f}'.format(self._dmat['d']*1.e10),
               str(self.get_bragg_from_lamb(self._DEFLAMB)[0]*180./np.pi)]

        # -----------------------
        # Build geometry
        col1 = ['Type', 'Type outline', 'surface (cm^2)', 'rcurve',
                'half-extent', 'summit', 'center', 'nin', 'e1', 'e2']
        ar1 = [self._dgeom['Type'], self._dgeom['Typeoutline'],
               '{0:5.1f}'.format(self._dgeom['surface']*1.e4),
               '{0:5.2f}'.format(self._dgeom['rcurve']),
               str(np.round(self._dgeom['extenthalf'], decimals=3)),
               str(np.round(self._dgeom['summit'], decimals=2)),
               str(np.round(self._dgeom['center'], decimals=2)),
               str(np.round(self._dgeom['nin'], decimals=2)),
               str(np.round(self._dgeom['e1'], decimals=2)),
               str(np.round(self._dgeom['e2'], decimals=2))]

        if self._dmisc.get('color') is not None:
            col1.append('color')
            ar1.append(str(self._dmisc['color']))

        lcol = [col0, col1]
        lar = [ar0, ar1]

        # -----------------------
        # Build mobile
        if self._dgeom['mobile'] != False:
            if self._dgeom['mobile'] == 'rotate':
                col2 = ['Mov. type', 'axis pt.', 'axis vector', 'pos. (deg)']
                ar2 = [self._dgeom['mobile'],
                       str(np.round(self._dgeom['rotateaxis'][0], decimals=2)),
                       str(np.round(self._dgeom['rotateaxis'][1], decimals=2)),
                       '{0:7.3f}'.format(self._dgeom['rotateangle']*180./np.pi)]
            lcol.append(col2)
            lar.append(ar2)
        return self._get_summary(lar, lcol,
                                  sep=sep, line=line, table_sep=table_sep,
                                  verb=verb, return_=return_)
    # -----------------
    # methods for moving
    # -----------------

    @staticmethod
    def _rotate_vector(vect, dangle, u, u1, u2):
        c1 = np.sum(vect*u1)
        c2 = np.sum(vect*u2)
        return (np.sum(vect*u)*u
                + (c1*np.cos(dangle) - c2*np.sin(dangle))*u1
                + (c2*np.cos(dangle) + c1*np.sin(dangle))*u2)


    def _rotate(self, angle=None, dangle=None):
        assert 'rotateaxis' in self._dgeom.keys()
        lc = [angle is not None, dangle is not None]
        assert np.sum(lc) == 1

        if lc[0]:
            assert 'rotateangle' in self._dgeom.keys()
            dangle = (angle - self._dgeom['rotateangle'])%(2*np.pi)
        dangle = np.arctan2(np.sin(dangle), np.cos(dangle))
        angle = (self._dgeom['rotateangle'] + dangle)%(2.*np.pi)
        angle = np.arctan2(np.sin(angle), np.cos(angle))

        # Define local frame (u, u1, u2)
        OS = self._dgeom['summit'] - self._dgeom['rotateaxis'][0]
        u = self._dgeom['rotateaxis'][1]
        u = u / np.linalg.norm(u)
        Z = np.sum(OS*u)
        u1 = OS - Z*u
        u1 = u1 / np.linalg.norm(u1)
        u2 = np.cross(u, u1)
        assert np.abs(np.linalg.norm(u2) - 1.) < 1.e-9

        # Deduce constant distance from axis
        dist = np.sum(OS*u1)

        summit = (self._dgeom['rotateaxis'][0]
                  + dist*(np.cos(dangle)*u1 + np.sin(dangle)*u2) + Z*u)
        nout = self._rotate_vector(self._dgeom['nout'], dangle, u, u1, u2)
        e1 = self._rotate_vector(self._dgeom['e1'], dangle, u, u1, u2)
        e2 = self._rotate_vector(self._dgeom['e2'], dangle, u, u1, u2)
        center = summit - self._dgeom['rcurve']*nout
        assert np.abs(np.sum(nout*e1)) < 1.e-12
        assert np.abs(np.sum(nout*e2)) < 1.e-12
        assert np.abs(np.sum(e1*e2)) < 1.e-12
        self._dgeom.update({'summit': summit, 'center':center,
                            'nin': -nout, 'nout':nout, 'e1': e1, 'e2': e2,
                            'rotateangle': angle})

    def move(self, kind=None, **kwdargs):
        if self._dgeom['mobile'] == False:
            return
        if kind is None:
            kind = self._dgeom['mobile']
        assert kind in self._DEFLMOVEOK
        if kind == 'rotate':
            self._rotate(**kwdargs)

    # -----------------
    # methods for surface and contour sampling
    # -----------------

    def sample_outline_plot(self, res=None):
        if self._dgeom['Type'] == 'sph':
            if self._dgeom['Typeoutline'] == 'rect':
                func = _comp_optics.CrystBragg_sample_outline_plot_sphrect
                outline = func(self._dgeom['center'], self.dgeom['nout'],
                               self._dgeom['e1'], self._dgeom['e2'],
                               self._dgeom['rcurve'], self._dgeom['extenthalf'],
                               res)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return outline

    def sample_outline_Rays(self, res=None):
        if self._dgeom['Type'] == 'sph':
            if self._dgeom['Typeoutline'] == 'rect':
                func = _comp_optics.CrystBragg_sample_outline_Rays
                pts, phi, theta = func()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return outline



    # -----------------
    # methods for surface and contour sampling
    # -----------------

    def _checkformat_bragglamb(self, bragg=None, lamb=None, n=None):
        lc = [lamb is not None, bragg is not None]
        assert np.sum(lc) == 1, "Provide lamb xor bragg!"
        if lc[0]:
            bragg = self.get_bragg_from_lamb(np.atleast_1d(lamb),
                                             n=n)
        else:
            bragg = np.atleast_1d(bragg)
        return bragg

    def _checkformat_get_Rays_from(self, phi=None, bragg=None):
        assert phi is not None
        assert bragg is not None
        bragg = np.atleast_1d(bragg)
        phi = np.atleast_1d(phi)
        nrays = max(phi.size, bragg.size)
        if not phi.shape == bragg.shape:
            if phi.size == 1:
                phi = np.full(bragg.shape, phi[0])
            elif bragg.size == 1:
                bragg = np.full(phi.shape, bragg[0])
            else:
                msg = "phi and bragg/lamb must have the same shape!\n"
                msg += "   phi.shape:        %s\n"%str(phi.shape)
                msg += "   bragg/lamb.shape: %s\n"%str(bragg.shape)
                raise Exception(msg)
        return phi, bragg

    def get_Rays_from_summit(self, phi=None, bragg=None,
                             lamb=None, n=None,
                             returnas=object, config=None, name=None):

        # Check inputs
        bragg = self._checkformat_bragglamb(bragg=bragg, lamb=lamb)
        phi, bragg = self._checkformat_get_Rays_from(phi=phi, bragg=bragg)
        # assert phi.ndim == 1
        phi = phi[None, ...]
        bragg = bragg[None, ...]

        # Prepare
        shape = tuple([3] + [1 for ii in range(phi.ndim)])
        nin = self._dgeom['nin'].reshape(shape)
        e1 = self._dgeom['e1'].reshape(shape)
        e2 = self._dgeom['e2'].reshape(shape)

        # Compute start point (D) and unit vectors (us)
        D = self._dgeom['summit']
        us = (np.sin(bragg)*nin
              + np.cos(bragg)*(np.cos(phi)*e1 + np.sin(phi)*e2))

        # Format output
        if returnas == tuple:
            return (D, us)
        elif returnas == object:
            from ._core import CamLOS1D
            if name is None:
                name = self.Id.Name + 'ExtractCam'
                if us.ndim > 2:
                    us = us.reshape((3, phi.size))
            return CamLOS1D(dgeom=(D, us), Name=name, Diag=self.Id.Diag,
                            Exp=self.Id.Exp, shot=self.Id.shot, config=config)

    def get_Rays_envelop(self,
                         phi=None, bragg=None, lamb=None, n=None,
                         returnas=object, config=None, name=None):
        # Check inputs
        phi, bragg = self._checkformat_get_Rays_from(phi=phi, bragg=bragg,
                                                     lamb=lamb, n=n)
        assert phi.ndim == 1

        # Compute
        func = _comp_optics.CrystBragg_sample_outline_Rays
        D, us = func(self._dgeom['center'], self._dgeom['nout'],
                     self._dgeom['e1'], self._dgeom['e2'],
                     self._dgeom['rcurve'], self._dgeom['extenthalf'],
                     bragg, phi)

        # Format output
        if returnas == tuple:
            return (D, us)
        elif returnas == object:
            from ._core import CamLOS1D
            if name is None:
                name = self.Id.Name + 'ExtractCam'
            return CamLOS1D(dgeom=(D, us), Name=name, Diag=self.Id.Diag,
                            Exp=self.Id.Exp, shot=self.Id.shot, config=config)



    # -----------------
    # methods for general plotting
    # -----------------

    def plot(self, lax=None, proj=None, res=None, element=None,
             color=None,
             dP=None, dI=None, dBs=None, dBv=None,
             dVect=None, dIHor=None, dBsHor=None,
             dBvHor=None, dleg=None,
             draw=True, fs=None, wintit=None, Test=True):
        kwdargs = locals()
        lout = ['self']
        for k in lout:
            del kwdargs[k]
        return _plot_optics.CrystalBragg_plot(self, **kwdargs)

    # -----------------
    # methods for generic first-approx
    # -----------------

    def get_bragg_from_lamb(self, lamb, n=None):
        """ Braggs' law: n*lamb = 2dsin(bragg) """
        if self._dmat['d'] is None:
            msg = "Interplane distance d no set !\n"
            msg += "  => self.set_dmat({'d':...})"
            raise Exception(msg)
        return _comp_optics.get_bragg_from_lamb(np.atleast_1d(lamb),
                                                self._dmat['d'], n=n)

    def get_lamb_from_bragg(self, bragg, n=None):
        """ Braggs' law: n*lamb = 2dsin(bragg) """
        if self._dmat['d'] is None:
            msg = "Interplane distance d no set !\n"
            msg += "  => self.set_dmat({'d':...})"
            raise Exception(msg)
        return _comp_optics.get_lamb_from_bragg(np.atleast_1d(bragg),
                                                self._dmat['d'], n=n)

    def get_approx_detector_frame(self, bragg=None, lamb=None,
                                  rcurve=None, n=None, plot=False):
        """ See notes for details on notations """

        # Check / format inputs
        if rcurve is None:
            rcurve = self._dgeom['rcurve']
        bragg = self._checkformat_bragglamb(bragg=bragg, lamb=lamb, n=n)

        # Compute crystal-centered parameters in (nout, e1, e2)
        func = _comp_optics.get_approx_detector_rel
        det_dist, det_nout_rel, det_ei_rel = func(rcurve, bragg)

        # Deduce absolute position in (x, y, z)
        func = _comp_optics.get_det_abs_from_rel
        det_cent, det_nout, det_ei, det_ej = func(det_dist,
                                                  det_nout_rel, det_ei_rel,
                                                  self._dgeom['summit'],
                                                  self._dgeom['nout'],
                                                  self._dgeom['e1'],
                                                  self._dgeom['e2'])

        if plot:
            dax = self.plot()
            p0 = np.repeat(det_cent[:,None], 3, axis=1)
            vv = np.vstack((det_nout, det_ei, det_ej)).T
            dax['cross'].plot(np.hypot(det_cent[0], det_cent[1]),
                              det_cent[2], 'xb')
            dax['hor'].plot(det_cent[0], det_cent[1], 'xb')
            dax['cross'].quiver(np.hypot(p0[0, :], p0[1, :]), p0[2, :],
                                np.hypot(vv[0, :], vv[1, :]), vv[2, :],
                                units='xy', color='b')
            dax['hor'].quiver(p0[0, :], p0[1, :], vv[0, :], vv[1, :],
                              units='xy', color='b')
        return det_cent, det_nout, det_ei, det_ej

    def get_local_noute1e2(self, theta, psi):
        if np.allclose([theta, psi], [np.pi/2., 0.]):
            summit = self._dgeom['summit']
            nout = self._dgeom['nout']
            e1, e2 = self._dgeom['e1'], self._dgeom['e2']
        else:
            func = _comp_optics.CrystBragg_get_noute1e2_from_psitheta
            nout, e1, e2 = func(self._dgeom['nout'],
                                self._dgeom['e1'], self._dgeom['e2'],
                                [psi], [theta])
            nout, e1, e2 = nout[:, 0], e1[:, 0], e2[:, 0]
            summit = self._dgeom['center'] + self._dgeom['rcurve']*nout
        return summit, nout, e1, e2


    def calc_xixj_from_phibragg(self, phi=None,
                                bragg=None, lamb=None, n=None,
                                theta=None, psi=None,
                                det_cent=None, det_nout=None,
                                det_ei=None, det_ej=None,
                                data=None, plot=True, ax=None):
        """ Assuming crystal's summit as frame origin

        According to [1], this assumes a local frame centered on the crystal

        These calculations are independent from the tokamak's frame:
            The origin of the local frame is the crystal's summit
            The (O, ez) axis is the crystal's normal
            The crystal is tangent to (O, ex, ey)

        [1] tofu/Notes_Upgrades/SpectroX2D/SpectroX2D_EllipsesOnPlane.pdf

        Parameters:
        -----------
        Z:      float
            Detector's plane intersection with (O, ez) axis
        n:      np.ndarray
            (3,) array containing local (x,y,z) coordinates of the plane's
            normal vector
        """
        # Check / format inputs
        bragg = self._checkformat_bragglamb(bragg=bragg, lamb=lamb, n=n)
        phi = np.atleast_1d(phi)
        assert bragg.ndim == phi.ndim == 1
        nphi, nbragg = phi.size, bragg.size
        npts = max(nphi, nbragg)
        assert nbragg in [1, npts] and nphi in [1, npts], (nbragg, nphi)
        if nbragg < npts:
            bragg = np.full((npts,), bragg[0])
        elif nphi < npts:
            phi = np.full((npts,), phi[0])

        lc = [det_cent is None, det_nout is None,
              det_ei is None, det_ej is None]
        assert all(lc) or not any(lc)
        if all(lc):
            func = self.get_approx_detector_frame
            det_cent, det_nout, det_ei, det_ej = func(lamb=self._DEFLAMB)

        # Get local summit nout, e1, e2 if non-centered
        if theta is None:
            theta = np.pi/2.
        if psi is None:
            psi = 0.
        summit, nout, e1, e2 = self.get_local_noute1e2(theta, psi)

        # Compute
        xi, xj = _comp_optics.calc_xixj_from_braggphi(summit,
                                                      det_cent, det_nout,
                                                      det_ei, det_ej,
                                                      nout, e1, e2,
                                                      bragg, phi)
        if plot:
            func = _plot_optics.CrystalBragg_plot_approx_detector_params
            ax = func(bragg, xi, xj, data, ax)
        return xi, xj

    @staticmethod
    def _checkformat_xixj(xi, xj):
        xi = np.atleast_1d(xi)
        xj = np.atleast_1d(xj)
        if xi.shape == xj.shape:
            return xi, xj, (xi, xj)
        else:
            return xi, xj, np.meshgrid(xi, xj)

    def calc_phibragg_from_xixj(self, xi, xj, n=None,
                                det_cent=None, det_ei=None, det_ej=None,
                                theta=None, psi=None,
                                plot=True, ax=None, **kwdargs):

        # Check / format inputs
        xi, xj, (xii, xjj) = self._checkformat_xixj(xi, xj)

        lc = [det_cent is None, det_ei is None, det_ej is None]
        assert all(lc) or not any(lc)
        if all(lc):
            func = self.get_approx_detector_frame
            det_cent, _, det_ei, det_ej = func(lamb=self._DEFLAMB)

        # Get local summit nout, e1, e2 if non-centered
        if theta is None:
            theta = np.pi/2.
        if psi is None:
            psi = 0.
        summit, nout, e1, e2 = self.get_local_noute1e2(theta, psi)

        # Compute
        func = _comp_optics.calc_braggphi_from_xixj
        bragg, phi = func(xii, xjj, det_cent, det_ei, det_ej,
                          summit, -nout, e1, e2)

        if plot != False:
            func = _plot_optics.CrystalBragg_plot_braggangle_from_xixj
            lax = func(xi=xii, xj=xjj,
                       ax=ax, plot=plot,
                       bragg=bragg * 180./np.pi,
                       angle=phi * 180./np.pi,
                       braggunits='deg', angunits='deg', **kwdargs)
        return bragg, phi

    def plot_johannerror(self, lamb=None):
        raise NotImplementedError

    def plot_data_vs_lambphi(self, xi=None, xj=None, data=None,
                            det_cent=None, det_ei=None, det_ej=None,
                            theta=None, psi=None, n=None,
                            plot=True, fs=None,
                            cmap=None, vmin=None, vmax=None):
        # Check / format inputs
        assert data is not None
        xi, xj, (xii, xjj) = self._checkformat_xixj(xi, xj)
        nxi = xi.size if xi is not None else np.unique(xii).size
        nxj = xj.size if xj is not None else np.unique(xjj).size

        # Compute lamb / phi
        func = self.calc_phibragg_from_xixj
        bragg, phi = func(xii, xjj, n=n,
                          det_cent=det_cent, det_ei=det_ei, det_ej=det_ej,
                          theta=theta, psi=psi, plot=False)
        assert bragg.shape == phi.shape == data.shape
        lamb = self.get_lamb_from_bragg(bragg, n=n)

        # Compute lambfit / phifit and spectrum1d
        lambfit, phifit = _comp_optics.get_lambphifit(lamb, phi, nxi, nxj)
        lambfitbins = 0.5*(lambfit[1:] + lambfit[:-1])
        ind = np.digitize(lamb, lambfitbins)
        spect1d = np.array([np.nanmean(data[ind==ii]) for ii in np.unique(ind)])

        # plot
        func = _plot_optics.CrystalBragg_plot_data_vs_lambphi
        ax = func(xi, xj, bragg, lamb, phi, data,
                  lambfit=lambfit, phifit=phifit, spect1d=spect1d,
                  cmap=cmap, vmin=vmin, vmax=vmax, fs=fs)
        return ax

    def plot_data_fit2d(self, xi=None, xj=None, data=None, mask=None,
                        det_cent=None, det_ei=None, det_ej=None,
                        theta=None, psi=None, n=None,
                        nlamb=None, lamb0=None, forcelamb=False,
                        deg=None, knots=None, nbsplines=None,
                        method=None, max_nfev=None,
                        xtol=None, ftol=None, gtol=None,
                        loss=None, verbose=0, debug=None,
                        plot=True, fs=None, dlines=None, dmoments=None,
                        cmap=None, vmin=None, vmax=None):
        # Check / format inputs
        assert data is not None
        xi, xj, (xii, xjj) = self._checkformat_xixj(xi, xj)
        nxi = xi.size if xi is not None else np.unique(xii).size
        nxj = xj.size if xj is not None else np.unique(xjj).size

        # Compute lamb / phi
        func = self.calc_phibragg_from_xixj
        bragg, phi = func(xii, xjj, n=n,
                          det_cent=det_cent, det_ei=det_ei, det_ej=det_ej,
                          theta=theta, psi=psi, plot=False)
        assert bragg.shape == phi.shape == data.shape
        lamb = self.get_lamb_from_bragg(bragg, n=n)

        # Compute lambfit / phifit and spectrum1d
        lambfit, phifit = _comp_optics.get_lambphifit(lamb, phi, nxi, nxj)
        lambfitbins = 0.5*(lambfit[1:] + lambfit[:-1])
        ind = np.digitize(lamb, lambfitbins)
        spect1d = np.array([np.nanmean(data[ind==ii]) for ii in np.unique(ind)])

        # Compute fit for spect1d to get lamb0 if not provided
        import tofu.data._spectrafit2d as _spectrafit2d

        func = _spectrafit2d.multiplegaussianfit1d
        dfit1d = func(lambfit, spect1d,
                      nmax=nlamb, lamb0=lamb0, forcelamb=forcelamb,
                      p0=None, bounds=None,
                      max_nfev=None, xtol=xtol, verbose=0,
                      percent=20, plot_debug=False)

        # Reorder wrt lamb0
        ind = np.argsort(dfit1d['lamb0'])
        for kk in ['lamb0', 'amp', 'ampstd',
                   'sigma', 'sigmastd', 'dlamb', 'dlambstd']:
            if dfit1d[kk].ndim == 1:
                dfit1d[kk] = dfit1d[kk][ind]
            else:
                dfit1d[kk] = dfit1d[kk][0,ind]


        # Compute dfit2d
        if mask is None:
            mask = np.ones(data.shape, dtype=bool)
        func = _spectrafit2d.multigaussianfit2d
        dfit2d = func(lamb[mask].ravel(), phi[mask].ravel(), data[mask].ravel(),
                      std=dfit1d['std'],
                      lamb0=dfit1d['lamb0'], forcelamb=forcelamb,
                      deg=deg, nbsplines=nbsplines,
                      method=method, max_nfev=max_nfev,
                      xtol=xtol, ftol=ftol, gtol=gtol,
                      loss=loss, verbose=verbose, debug=debug)


        # plot
        func = _plot_optics.CrystalBragg_plot_data_vs_fit
        ax = func(xi, xj, bragg, lamb, phi, data, mask=mask,
                  lambfit=lambfit, phifit=phifit, spect1d=spect1d,
                  dfit1d=dfit1d, dfit2d=dfit2d, lambfitbins=lambfitbins,
                  cmap=cmap, vmin=vmin, vmax=vmax,
                  fs=fs, dmoments=dmoments)
        return ax, dfit1d, None
