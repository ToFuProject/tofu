
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
import scipy.interpolate as scpinterp
import scipy.stats as scpstats
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
        if dmat.get('d') is not None:
            dmat['d'] = float(dmat['d'])
        if dmat.get('formula') is not None:
            assert isinstance(dmat['formula'], str)
        if dmat.get('density') is not None:
            dmat['density'] = float(dmat['density'])
        if dmat.get('lengths') is not None:
            dmat['lengths'] = np.atleast_1d(dmat['lengths']).ravel()
        if dmat.get('angles') is not None:
            dmat['angles'] = np.atleast_1d(dmat['angles']).ravel()
        if dmat.get('cut') is not None:
            dmat['cut'] = np.atleast_1d(dmat['cut']).ravel().astype(int)
            assert dmat['cut'].size <= 4
        return dmat

    @classmethod
    def _checkformat_dbragg(cls, dbragg=None):
        if dbragg is None:
            return
        assert isinstance(dbragg, dict)
        lkok = cls._get_keys_dbragg()
        assert all([isinstance(ss, str) for ss in dbragg.keys()])
        assert all([ss in lkok for ss in dbragg.keys()])

        for kk in cls._ddef['dbragg'].keys():
            dbragg[kk] = dbragg.get(kk, cls._ddef['dbragg'][kk])
        if dbragg.get('rockingcurve') is not None:
            assert isinstance(dbragg['rockingcurve'], dict)
            drock = dbragg['rockingcurve']
            lkeyok = ['sigam', 'deltad', 'Rmax', 'dangle', 'lamb', 'value',
                      'type', 'source']
            lkout = [kk for kk in drock.keys() if kk not in lkeyok]
            if len(lkout) > 0:
                msg = ("Unauthorized keys in dbrag['rockingcurve']:\n"
                       + "\t-" + "\n\t-".join(lkout))
                raise Exception(msg)
            try:
                if drock.get('sigma') is not None:
                    dbragg['rockingcurve']['sigma'] = float(drock['sigma'])
                    dbragg['rockingcurve']['deltad'] = float(
                        drock.get('deltad', 0.))
                    dbragg['rockingcurve']['Rmax'] = float(
                        drock.get('Rmax', 1.))
                    dbragg['rockingcurve']['type'] = 'lorentz-log'
                elif drock.get('dangle') is not None:
                    c2d = (drock.get('lamb') is not None
                           and drock.get('value').ndim == 2)
                    if c2d:
                        if drock['value'].shape != (drock['dangle'].size,
                                                    drock['lamb'].size):
                            msg = ("Tabulated 2d rocking curve should be:\n"
                                   + "\tshape = (dangle.size, lamb.size)")
                            raise Exception(msg)
                        dbragg['rockingcurve']['dangle'] = np.r_[
                            drock['dangle']]
                        dbragg['rockingcurve']['lamb'] = np.r_[drock['lamb']]
                        dbragg['rockingcurve']['value'] = drock['value']
                        dbragg['rockingcurve']['type'] = 'tabulated-2d'
                    else:
                        if drock.get('lamb') is None:
                            msg = ("Please also specify the lamb for which "
                                   + "the rocking curve was tabulated!")
                            raise Exception(msg)
                        dbragg['rockingcurve']['lamb'] = float(drock['lamb'])
                        dbragg['rockingcurve']['dangle'] = np.r_[
                            drock['dangle']]
                        dbragg['rockingcurve']['value'] = np.r_[drock['value']]
                        dbragg['rockingcurve']['type'] = 'tabulated-1d'
                    if drock.get('source') is None:
                        msg = "Unknonw source for the tabulated rocking curve!"
                        warnings.warn(msg)
                    dbragg['rockingcurve']['source'] = drock.get('source')
            except Exception as err:
                msg = ("Provide the rocking curve as a dict with either:\n"
                       + "\t- parameters of a lorentzian in log10:\n"
                       + "\t\t{'sigma': float,\n"
                       + "\t\t 'deltad': float,\n"
                       + "\t\t 'Rmax': float}\n"
                       + "\t- tabulated (dangle, value), with source (url...)"
                       + "\t\t{'dangle': np.ndarray,\n"
                       + "\t\t 'value': np.ndarray,\n"
                       + "\t\t 'source': str}")
                raise Exception(msg)
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
              'move', 'move_param', 'move_kwdargs']
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
        self.set_dbragg(**kwds)
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
                dgeom['center'] = (dgeom['summit']
                                   + dgeom['nin']*dgeom['rcurve'])
            if dgeom['summit'] is None:
                dgeom['summit'] = (dgeom['center']
                                   + dgeom['nout']*dgeom['rcurve'])
        elif dgeom['center'] is not None and dgeom['summit'] is not None:
            if dgeom['nout'] is None:
                nout = (dgeom['summit'] - dgeom['center'])
                dgeom['nout'] = nout / np.linalg.norm(nout)

        if dgeom['extenthalf'] is not None:
            if dgeom['Type'] == 'sph' and dgeom['Typeoutline'] == 'rect':
                ind = np.argmax(dgeom['extenthalf'])
                dphi = dgeom['extenthalf'][ind]
                sindtheta = np.sin(dgeom['extenthalf'][ind-1])
                dgeom['surface'] = 4.*dgeom['rcurve']**2*dphi*sindtheta
        self._dgeom = dgeom
        if dgeom['move'] is not None:
            self.set_move(move=dgeom['move'], param=dgeom['move_param'],
                          **dgeom['move_kwdargs'])

    def set_dmat(self, dmat=None):
        dmat = self._checkformat_dmat(dmat)
        self._dmat = dmat

    def set_dbragg(self, dbragg=None):
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

    @property
    def rockingcurve(self):
        if self._dbragg.get('rockingcurve') is not None:
            if self._dbragg['rockingcurve'].get('type') is not None:
                return self._dbragg['rockingcurve']
        raise Exception("rockingcurve was not set!")

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
                'd (A)', 'bragg({:9.6} A) (deg)'.format(self._DEFLAMB*1e10),
                'rocking curve']
        ar0 = [self._dmat['formula'], self._dmat['symmetry'],
               str(self._dmat['cut']), str(self._dmat['density']),
               '{0:5.3f}'.format(self._dmat['d']*1.e10),
               str(self.get_bragg_from_lamb(self._DEFLAMB)[0]*180./np.pi)]
        try:
            ar0.append(self.rockingcurve['type'])
        except Exception as err:
            ar0.append('None')


        # -----------------------
        # Build geometry
        col1 = ['Type', 'outline', 'surface (cm^2)', 'rcurve',
                'half-extent', 'summit', 'center', 'nin', 'e1']
        ar1 = [self._dgeom['Type'], self._dgeom['Typeoutline'],
               '{0:5.1f}'.format(self._dgeom['surface']*1.e4),
               '{0:5.2f}'.format(self._dgeom['rcurve']),
               str(np.round(self._dgeom['extenthalf'], decimals=3)),
               str(np.round(self._dgeom['summit'], decimals=2)),
               str(np.round(self._dgeom['center'], decimals=2)),
               str(np.round(self._dgeom['nin'], decimals=2)),
               str(np.round(self._dgeom['e1'], decimals=2))]
        if self._dgeom.get('move') not in [None, False]:
            col1 += ['move', 'param']
            ar1 += [self._dgeom['move'],
                    str(np.round(self._dgeom['move_param'], decimals=5))]

        if self._dmisc.get('color') is not None:
            col1.append('color')
            ar1.append(str(self._dmisc['color']))

        lcol = [col0, col1]
        lar = [ar0, ar1]
        return self._get_summary(lar, lcol,
                                  sep=sep, line=line, table_sep=table_sep,
                                  verb=verb, return_=return_)
    # -----------------
    # methods for moving
    # -----------------

    def _update_or_copy(self, dgeom, pinhole=None,
                        return_copy=None,
                        name=None, diag=None, shot=None):
        if return_copy is None:
            return_copy = _RETURN_COPY
        for kk, vv in self._dgeom.items():
            if kk not in dgeom.keys():
                dgeom[kk] = vv
        if return_copy is True:
            if name is None:
                name = self.Id.Name + 'copy'
            if diag is None:
                diag = self.Id.Diag
            if shot is None:
                diag = self.Id.shot
            return self.__class__(dgeom=dgeom,
                                  dbragg=self._dbragg,
                                  dmat=self._dmat,
                                  color=self._dmisc['color'],
                                  Exp=self.Id.Exp,
                                  Diag=diag,
                                  Name=name,
                                  shot=shot,
                                  SavePath=self.Id.SavePath)
        else:
            dgeom0 = self.dgeom
            try:
                self.set_dgeom(dgeom=dgeom)
            except Exception as err:
                # Make sure instance does not move
                self.set_dgeom(dgeom=dgeom0)
                msg = (str(err)
                       + "\nAn exception occured during updating\n"
                       + "  => instance unmoved")
                raise Exception(msg)

    def _rotate_or_translate(self, func, **kwdargs):
        pts = np.array([self._dgeom['summit'], self._dgeom['center']]).T
        if 'rotate' in func.__name__:
            vect = np.array([self._dgeom['nout'],
                             self._dgeom['e1'], self._dgeom['e2']]).T
            pts, vect = func(pts=pts, vect=vect, **kwdargs)
            return {'summit': pts[:, 0], 'center': pts[:, 1],
                    'nout': vect[:, 0], 'nin': -vect[:, 0],
                    'e1': vect[:, 1], 'e2': vect[:, 2]}
        else:
            pts = func(pts=pts, **kwdargs)
            return {'summit': pts[:, 0], 'center': pts[:, 1]}

    def translate_in_cross_section(self, distance=None, direction_rz=None,
                                   phi=None,
                                   return_copy=None,
                                   diag=None, name=None, shot=None):
        """ Translate the instance in the cross-section """
        if phi is None:
            phi = np.arctan2(*self.summit[1::-1])
            msg = ("Poloidal plane was not explicitely specified\n"
                   + "  => phi set to self.summit's phi ({})".format(phi))
            warnings.warn(msg)
        dgeom = self._rotate_or_translate(
            self._translate_pts_poloidal_plane,
            phi=phi, direction_rz=direction_rz, distance=distance)
        return self._update_or_copy(dgeom,
                                    return_copy=return_copy,
                                    diag=diag, name=name, shot=shot)

    def translate_3d(selfi, distance=None, direction=None,
                     return_copy=None,
                     diag=None, name=None, shot=None):
        """ Translate the instance in provided direction """
        dgeom = self._rotate_or_translate(
            self._translate_pts_3d,
            direction=direction, distance=distance)
        return self._update_or_copy(dgeom,
                                    return_copy=return_copy,
                                    diag=diag, name=name, shot=shot)

    def rotate_in_cross_section(self, angle=None, axis_rz=None,
                                phi=None,
                                return_copy=None,
                                diag=None, name=None, shot=None):
        """ Rotate the instance in the cross-section """
        if phi is None:
            phi = np.arctan2(*self.summit[1::-1])
            msg = ("Poloidal plane was not explicitely specified\n"
                   + "  => phi set to self.summit's phi ({})".format(phi))
            warnings.warn(msg)
        dgeom = self._rotate_or_translate(
            self._rotate_pts_vectors_in_poloidal_plane,
            axis_rz=axis_rz, angle=angle, phi=phi)
        return self._update_or_copy(dgeom,
                                    return_copy=return_copy,
                                    diag=diag, name=name, shot=shot)

    def rotate_around_torusaxis(self, angle=None,
                                return_copy=None,
                                diag=None, name=None, shot=None):
        """ Rotate the instance around the torus axis """
        dgeom = self._rotate_or_translate(
            self._rotate_pts_vectors_around_torusaxis,
            angle=angle)
        return self._update_or_copy(dgeom,
                                    return_copy=return_copy,
                                    diag=diag, name=name, shot=shot)

    def rotate_around_3daxis(self, angle=None, axis=None,
                             return_copy=None,
                             diag=None, name=None, shot=None):
        """ Rotate the instance around the provided 3d axis """
        dgeom = self._rotate_or_translate(
            self._rotate_pts_vectors_around_3daxis,
            axis=axis, angle=angle)
        return self._update_or_copy(dgeom,
                                    return_copy=return_copy,
                                    diag=diag, name=name, shot=shot)

    def set_move(self, move=None, param=None, **kwdargs):
        """ Set the default movement parameters

        A default movement can be set for the instance, it can be any of the
        pre-implemented movement (rotations or translations)
        This default movement is the one that will be called when using
        self.move()

        Specify the type of movement via the name of the method (passed as a
        str to move)

        Specify, for the geometry of the instance at the time of defining this
        default movement, the current value of the associated movement
        parameter (angle / distance). This is used to set an arbitrary
        difference for user who want to use absolute position values
        The desired incremental movement to be performed when calling self.move
        will be deduced by substracting the stored param value to the provided
        param value. Just set the current param value to 0 if you don't care
        about a custom absolute reference.

        kwdargs must be a parameters relevant to the chosen method (axis,
        direction...)

        e.g.:
            self.set_move(move='rotate_around_3daxis',
                          param=0.,
                          axis=([0.,0.,0.], [1.,0.,0.]))
            self.set_move(move='translate_3d',
                          param=0.,
                          direction=[0.,1.,0.])
        """
        move, param, kwdargs = self._checkformat_set_move(move, param, kwdargs)
        self._dgeom['move'] = move
        self._dgeom['move_param'] = param
        if isinstance(kwdargs, dict) and len(kwdargs) == 0:
            kwdargs = None
        self._dgeom['move_kwdargs'] = kwdargs

    def move(self, param):
        """ Set new position to desired param according to default movement

        Can only be used if default movement was set before
        See self.set_move()
        """
        param = self._move(param, dictname='_dgeom')
        self._dgeom['move_param'] = param


    # -----------------
    # methods for rocking curve
    # -----------------

    def get_rockingcurve_func(self, lamb=None, n=None):
        """ Return the rocking curve function

        Also return the wavelength (lamb) for which it was computed
            and the associated reference bragg angle

        """
        drock = self.rockingcurve
        if drock['type'] == 'tabulated-1d':
            if lamb is not None and lamb != drock['lamb']:
                msg = ("rocking curve was tabulated only for:\n"
                       + "\tlamb = {} m\n".format(lamb)
                       + "  => Please let lamb=None")
                raise Exception(msg)
            lamb = drock['lamb']
            bragg = self._checkformat_bragglamb(lamb=lamb, n=n)
            func = scpinterp.interp1d(drock['dangle'] + bragg, drock['value'],
                                      kind='linear', bounds_error=False,
                                      fill_value=0, assume_sorted=True)

        elif drock['type'] == 'tabulated-2d':
            lmin, lmax = drock['lamb'].min(), drock['lamb'].max()
            if lamb is None:
                lamb = drock['lamb']
            if lamb < lmin or lamb > lmax:
                msg = ("rocking curve was tabulated only in interval:\n"
                       + "\tlamb in [{}; {}] m\n".format(lmin, lmax)
                       + "  => Please set lamb accordingly")
                raise Exception(msg)
            bragg = self._checkformat_bragglamb(lamb=lamb, n=n)

            def func(angle, lamb=lamb, bragg=bragg, drock=drock):
                return scpinterp.interp2d(drock['dangle']+bragg, drock['lamb'],
                                          drock['value'], kind='linear',
                                          bounds_error=False, fill_value=0,
                                          assume_sorted=True)(angle, lamb)

        else:
            # TBC
            raise NotImplementedError
            def func(angle, d=d, delta_bragg=delta_bragg,
                     Rmax=drock['Rmax'], sigma=drock['sigma']):
                core = sigma**2/((angle - (bragg+delta_bragg))**2 + sigma**2)
                if Rmax is None:
                    return core/(sigma*np.pi)
                else:
                    return Rmax*core
        return func, lamb, bragg

    def plot_rockingcurve(self, lamb=None, n=None, sigma=None,
                          npts=None, color=None, ang_units=None,
                          fs=None, ax=None, legend=None):
        drock = self.rockingcurve
        func, lamb, bragg = self.get_rockingcurve_func(lamb=lamb, n=n)
        axtit = 'Rocking curve for ' + self.Id.Name
        return _plot_optics.CrystalBragg_plot_rockingcurve(
            func=func, bragg=bragg, lamb=lamb,
            sigma=sigma, npts=npts,
            ang_units=ang_units, axtit=axtit, color=color,
            fs=fs, ax=ax, legend=legend)

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
                pts, phi, dtheta = func()
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
        if not any(lc):
            lamb = self._DEFLAMB
            lc[0] = True
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
             color=None, det=None,
             dP=None, dI=None, dBs=None, dBv=None,
             dVect=None, dIHor=None, dBsHor=None,
             dBvHor=None, dleg=None,
             draw=True, fs=None, wintit=None, Test=True):
        kwdargs = locals()
        lout = ['self']
        for k in lout:
            del kwdargs[k]
        if det is not None:
            kwdargs.update({'det_'+kk: vv for kk, vv in det.items()})
        del kwdargs['det']
        return _plot_optics.CrystalBragg_plot(self, **kwdargs)

    # -----------------
    # methods for generic first-approx
    # -----------------

    def get_phi_from_magaxis_summit(self, r, z, lamb=None, bragg=None, n=None):
        # Check / format input
        r = np.atleast_1d(r)
        z = np.atleast_1d(z)
        assert r.shape == z.shape
        bragg = self._checkformat_bragglamb(bragg=bragg, lamb=lamb, n=n)

        # Compute phi

        return phi



    def get_bragg_from_lamb(self, lamb=None, n=None):
        """ Braggs' law: n*lamb = 2dsin(bragg) """
        if self._dmat['d'] is None:
            msg = "Interplane distance d no set !\n"
            msg += "  => self.set_dmat({'d':...})"
            raise Exception(msg)
        if lamb is None:
            lamb = self._DEFLAMB
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

    def get_detector_approx(self, bragg=None, lamb=None,
                            rcurve=None, n=None,
                            ddist=None, di=None, dj=None,
                            dtheta=None, dpsi=None, tilt=None,
                            lamb0=None, lamb1=None, dist01=None,
                            tangent_to_rowland=None, plot=False):
        """ Return approximate ideal detector geometry

        Assumes infinitesimal and ideal crystal
        Returns a dict containing the position and orientation of a detector if
            it was placed ideally on the rowland circle, centered on the
            desired bragg angle (in rad) or wavelength (in m)
        The detector can be tangential to the Rowland circle or perpendicular
            to the line between the crystal and the detector
        Assumes detector center matching lamb (m) / bragg (rad)

        The detector can be translated towards / away from the crystal
            to make sure the distance between 2 spectral lines
            (lamb0 and lamb1) on the detector's plane matches
            a desired distance (dist01, in m)

        Finally, a desired offset (translation) can be added
            via (ddist, di, dj), in m
        Similarly, an extra rotation can be added via (dtheta, dpsi, tilt)

        Detector is described by center position
            and (nout, ei, ej) unit vectors
        By convention, nout = np.cross(ei, ej)
        Vectors (ei, ej) define an orthogonal frame in the detector's plane
        All coordinates are 3d (X, Y, Z in the tokamak's frame)

        Return:
        -------
        det: dict
            dict of detector geometrical characteristics:
                'cent': np.ndarray
                    (3,) array of (x, y, z) coordinates of detector center
                'nout': np.ndarray
                    (3,) array of (x, y, z) coordinates of unit vector
                    perpendicular to detector' surface
                    oriented towards crystal
                'ei':   np.ndarray
                    (3,) array of (x, y, z) coordinates of unit vector
                    defining first coordinate in detector's plane
                'ej':   np.ndarray
                    (3,) array of (x, y, z) coordinates of unit vector
                    defining second coordinate in detector's plane
        """

        # Check / format inputs
        if rcurve is None:
            rcurve = self._dgeom['rcurve']
        bragg = self._checkformat_bragglamb(bragg=bragg, lamb=lamb, n=n)
        if np.all(np.isnan(bragg)):
            msg = ("There is no available bragg angle!\n"
                   + "  => Check the vlue of self.dmat['d'] vs lamb")
            raise Exception(msg)

        lc = [lamb0 is not None, lamb1 is not None, dist01 is not None]
        if any(lc) and not all(lc):
            msg = ("Arg lamb0, lamb1 and dist01 must be provided together:\n"
                   + "\t- lamb0: line0 wavelength ({})\n".format(lamb0)
                   + "\t- lamb1: line1 wavelength ({})\n".format(lamb1)
                   + "\t- dist01: distance (m) on detector between lines "
                   + "({})".format(dist01)
                  )
            raise Exception(msg)
        bragg01 = None
        if all(lc):
            bragg01 = self._checkformat_bragglamb(lamb=np.r_[lamb0, lamb1],
                                                  n=n)

        lf = ['summit', 'nout', 'e1', 'e2']
        lc = [rcurve is None] + [self._dgeom[kk] is None for kk in lf]
        if any(lc):
            msg = ("Some missing fields in dgeom for computation:"
                   + "\n\t-" + "\n\t-".join(['rcurve'] + lf))
            raise Exception(msg)

        # Compute crystal-centered parameters in (nout, e1, e2)
        (det_dist, n_crystdet_rel,
         det_nout_rel, det_ei_rel) = _comp_optics.get_approx_detector_rel(
             rcurve, bragg,
             bragg01=bragg01, dist01=dist01,
             tangent_to_rowland=tangent_to_rowland)

        # Deduce absolute position in (x, y, z)
        det_cent, det_nout, det_ei, det_ej = _comp_optics.get_det_abs_from_rel(
            det_dist, n_crystdet_rel, det_nout_rel, det_ei_rel,
            self._dgeom['summit'],
            self._dgeom['nout'], self._dgeom['e1'], self._dgeom['e2'],
            ddist=ddist, di=di, dj=dj,
            dtheta=dtheta, dpsi=dpsi, tilt=tilt)

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
        return {'cent': det_cent, 'nout': det_nout,
                'ei': det_ei, 'ej': det_ej}

    def get_local_noute1e2(self, dtheta, psi):
        """ Return (nout, e1, e2) associated to pts on the crystal's surface

        All points on the spherical crystal's surface are identified
            by (dtheta, psi) coordinates, where:
                - theta  = np.pi/2 + dtheta (dtheta=0 by default) for the center
                - psi = 0 for the center
            They are the spherical coordinates from a sphere centered on the
            crystal's center of curvature.

        Return the pts themselves and the 3 perpendicular unti vectors
            (nout, e1, e2), where nout is towards the outside of the sphere and
            nout = np.cross(e1, e2)

        Return:
        -------
        summit:     np.ndarray
            (3,) array of (x, y, z) coordinates of the points on the surface
        nout:       np.ndarray
            (3,) array of (x, y, z) coordinates of outward unit vector
        e1:         np.ndarray
            (3,) array of (x, y, z) coordinates of first unit vector
        e2:         np.ndarray
            (3,) array of (x, y, z) coordinates of second unit vector

        """
        dtheta = np.atleast_1d(dtheta)
        psi = np.atleast_1d(psi)
        if psi.shape != dtheta.shape:
            msg = ("dtheta and psi should have the same shape\n"
                   + "\t- dtheta.shape = {}\n".format(dtheta.shape)
                   + "\t- psi.shape = {}".format(psi.shape))
            raise Exception(msg)
        nmax = max(psi.size, dtheta.size)
        assert psi.size in [1, nmax] and dtheta.size in [1, nmax]

        if nmax == 1 and np.allclose([dtheta, psi], [0., 0.]):
            summ = self._dgeom['summit']
            nout = self._dgeom['nout']
            e1, e2 = self._dgeom['e1'], self._dgeom['e2']
        else:
            func = _comp_optics.CrystBragg_get_noute1e2_from_psitheta
            nout, e1, e2 = func(self._dgeom['nout'],
                                self._dgeom['e1'], self._dgeom['e2'],
                                psi, dtheta)
            if nout.ndim == 2:
                cent = self._dgeom['center'][:, None]
            elif nout.ndim == 3:
                cent = self._dgeom['center'][:, None, None]
            elif nout.ndim == 4:
                cent = self._dgeom['center'][:, None, None, None]
            else:
                msg = "nout.ndim > 4!"
                raise Exception(msg)
            summ = cent + self._dgeom['rcurve']*nout
            if nmax == 1:
                summ, nout = summ[:, 0], nout[:, 0]
                e1, e2 = e1[:, 0], e2[:, 0]
        return summ, nout, e1, e2


    def calc_xixj_from_braggphi(self, phi=None,
                                bragg=None, lamb=None, n=None,
                                dtheta=None, psi=None,
                                det=None, data=None,
                                plot=True, ax=None):
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
        assert bragg.ndim == phi.ndim
        if phi.shape != bragg.shape:
            if phi.size == 1 and bragg.ndim == 1:
                phi = np.repeat(phi, bragg.size)
            else:
                msg = ("bragg and phi should have the same shape !\n"
                       + "\t- phi.shape = {}\n".format(phi.shape)
                       + "\t- bragg.shape = {}\n".format(bragg.shape))
                raise Exception(msg)

        # Check / get det
        assert all(lc) or not any(lc)
        if det is None:
            det = self.get_detector_approx(lamb=self._DEFLAMB)

        # Get local summit nout, e1, e2 if non-centered
        if dtheta is None:
            dtheta = 0.
        if psi is None:
            psi = 0.
        summit, nout, e1, e2 = self.get_local_noute1e2(dtheta, psi)

        # Compute
        xi, xj = _comp_optics.calc_xixj_from_braggphi(summit,
                                                      det['cent'], det['nout'],
                                                      det['ei'], det['ej'],
                                                      nout, e1, e2,
                                                      bragg, phi)
        if plot:
            func = _plot_optics.CrystalBragg_plot_approx_detector_params
            ax = func(bragg, xi, xj, data, ax)
        return xi, xj

    @staticmethod
    def _checkformat_pts(pts):
        pts = np.atleast_1d(pts)
        if pts.ndim == 1:
            pts = pts.reshape((3, 1))
        if 3 not in pts.shape or pts.ndim != 2:
            msg = "pts must be a (3, npts) array of (X, Y, Z) coordinates!"
            raise Exception(msg)
        if pts.shape[0] != 3:
            pts = pts.T
        return pts

    @staticmethod
    def _checkformat_xixj(xi, xj):
        xi = np.atleast_1d(xi)
        xj = np.atleast_1d(xj)

        if xi.shape == xj.shape:
            return xi, xj, (xi, xj)
        else:
            return xi, xj, np.meshgrid(xi, xj)

    @staticmethod
    def _checkformat_dthetapsi(psi=None, dtheta=None,
                               psi_def=0., dtheta_def=0.):
        if psi is None:
            psi = psi_def
        if dtheta is None:
            dtheta = dtheta_def
        psi = np.r_[psi]
        dtheta = np.r_[dtheta]
        if psi.size != dtheta.size:
            msg = "psi and dtheta must be 1d arrays fo same size!"
            raise Exception(msg)
        return dtheta, psi


    def calc_phibragg_from_xixj(self, xi, xj, n=None,
                                det=None, dtheta=None, psi=None,
                                plot=True, ax=None, **kwdargs):

        # Check / format inputs
        xi, xj, (xii, xjj) = self._checkformat_xixj(xi, xj)

        if det is None:
            det = self.get_detector_approx(lamb=self._DEFLAMB)

        # Get local summit nout, e1, e2 if non-centered
        if dtheta is None:
            dtheta = 0.
        if psi is None:
            psi = 0.
        summit, nout, e1, e2 = self.get_local_noute1e2(dtheta, psi)

        # Get bragg, phi
        bragg, phi = _comp_optics.calc_braggphi_from_xixjpts(
            det['cent'], det['ei'], det['ej'],
            summit, -nout, e1, e2,
            xi=xi, xj=xj)

        if plot != False:
            lax = _plot_optics.CrystalBragg_plot_braggangle_from_xixj(
                xi=xii, xj=xjj,
                ax=ax, plot=plot,
                bragg=bragg * 180./np.pi,
                angle=phi * 180./np.pi,
                braggunits='deg', angunits='deg', **kwdargs)
        return bragg, phi

    def plot_line_on_det_tracing(self, lamb=None, n=None,
                                 xi_bounds=None, xj_bounds=None, nphi=None,
                                 det=None, johann=False,
                                 lpsi=None, ldtheta=None,
                                 rocking=False, fs=None, dmargin=None,
                                 wintit=None, tit=None):
        """ Visualize the de-focusing by ray-tracing of chosen lamb
        """
        # Check / format inputs
        if lamb is None:
            lamb = self._DEFLAMB
        lamb = np.atleast_1d(lamb).ravel()
        nlamb = lamb.size

        detb = np.array([[xi_bounds[0], xi_bounds[1], xi_bounds[1],
                          xi_bounds[0], xi_bounds[0]],
                         [xj_bounds[0], xj_bounds[0], xj_bounds[1],
                          xj_bounds[1], xj_bounds[0]]])

        # Compute lamb / phi
        _, phi = self.calc_phibragg_from_xixj(detb[0, :], detb[1, :], n=n,
                                              det=det, dtheta=None,
                                              psi=None, plot=False)
        phimin, phimax = np.nanmin(phi), np.nanmax(phi)
        phimin, phimax = phimin-(phimax-phimin)/10, phimax+(phimax-phimin)/10
        del phi

        # Get reference ray-tracing
        if nphi is None:
            nphi = 300
        phi = np.linspace(phimin, phimax, nphi)
        bragg = self._checkformat_bragglamb(lamb=lamb, n=n)

        xi = np.full((nlamb, nphi), np.nan)
        xj = np.full((nlamb, nphi), np.nan)
        for ll in range(nlamb):
            xi[ll, :], xj[ll, :] = self.calc_xixj_from_braggphi(
                bragg=bragg[ll], phi=phi, n=n,
                det=det, plot=False)

        # Get johann-error raytracing (multiple positions on crystal)
        xi_er, xj_er = None, None
        if johann and not rocking:
            if lpsi is None or ldtheta is None:
                lpsi = np.linspace(-1., 1., 15)
                ldtheta = np.linspace(-1., 1., 15)
                lpsi, ldtheta = np.meshgrid(lpsi, ldtheta)
                lpsi = lpsi.ravel()
                ldtheta = ldtheta.ravel()
            lpsi = self._dgeom['extenthalf'][0]*np.r_[lpsi]
            ldtheta = self._dgeom['extenthalf'][1]*np.r_[ldtheta]
            npsi = lpsi.size
            assert npsi == ldtheta.size

            xi_er = np.full((nlamb, npsi*nphi), np.nan)
            xj_er = np.full((nlamb, npsi*nphi), np.nan)
            for l in range(nlamb):
                for ii in range(npsi):
                    i0 = np.arange(ii*nphi, (ii+1)*nphi)
                    xi_er[l, i0], xj_er[l, i0] = self.calc_xixj_from_braggphi(
                        phi=phi, bragg=bragg[l], lamb=None, n=n,
                        dtheta=ldtheta[ii], psi=lpsi[ii],
                        det=det, plot=False)

        # Get rocking curve error
        if rocking:
            pass

        # Plot
        ax = _plot_optics.CrystalBragg_plot_line_tracing_on_det(
            lamb, xi, xj, xi_er, xj_er,
            det_cent=det['cent'], det_nout=det['nout'],
            det_ei=det['ei'], det_ej=det['ej'],
            johann=johann, rocking=rocking,
            fs=fs, dmargin=dmargin, wintit=wintit, tit=tit)

    def calc_johannerror(self, xi=None, xj=None, err=None,
                         det=None, n=None,
                         lpsi=None, ldtheta=None,
                         plot=True, fs=None, cmap=None,
                         vmin=None, vmax=None, tit=None, wintit=None):
        """ Plot the johann error

        The johann error is the error (scattering) induced by defocalization
            due to finite crystal dimensions
        There is a johann error on wavelength (lamb => loss of spectral
            resolution) and on directionality (phi)
        If provided, lpsi and ldtheta are taken as normalized variations with
            respect to the crystal summit and to its extenthalf.
            Typical values are:
                - lpsi   = [-1, 1, 1, -1]
                - ldtheta = [-1, -1, 1, 1]
            They must have the same len()

        """

        # Check / format inputs
        xi, xj, (xii, xjj) = self._checkformat_xixj(xi, xj)
        nxi = xi.size if xi is not None else np.unique(xii).size
        nxj = xj.size if xj is not None else np.unique(xjj).size

        # Compute lamb / phi
        bragg, phi = self.calc_phibragg_from_xixj(
            xii, xjj, n=n,
            det=det, dtheta=None, psi=None, plot=False)
        assert bragg.shape == phi.shape
        lamb = self.get_lamb_from_bragg(bragg, n=n)

        if lpsi is None:
            lpsi = np.r_[-1., 0., 1., 1., 1., 0., -1, -1]
        lpsi = self._dgeom['extenthalf'][0]*np.r_[lpsi]
        if ldtheta is None:
            ldtheta = np.r_[-1., -1., -1., 0., 1., 1., 1., 0.]
        ldtheta = self._dgeom['extenthalf'][1]*np.r_[ldtheta]
        npsi = lpsi.size
        assert npsi == ldtheta.size
        lamberr = np.full(tuple(np.r_[npsi, lamb.shape]), np.nan)
        phierr = np.full(lamberr.shape, np.nan)
        braggerr, phierr = self.calc_phibragg_from_xixj(
            xii, xjj, n=n,
            det=det, dtheta=ldtheta, psi=lpsi, plot=False)
        lamberr = self.get_lamb_from_bragg(braggerr, n=n)
        err_lamb = np.nanmax(np.abs(lamb[None, ...] - lamberr), axis=0)
        err_phi = np.nanmax(np.abs(phi[None, ...] - phierr), axis=0)
        if plot is True:
            ax = _plot_optics.CrystalBragg_plot_johannerror(
                xi, xj, lamb, phi, err_lamb, err_phi, err=err,
                cmap=cmap, vmin=vmin, vmax=vmax, fs=fs, tit=tit, wintit=wintit)
        return err_lamb, err_phi

    def _calc_braggphi_from_pts(self, pts,
                                det=None, dtheta=None, psi=None):

        # Check / format pts
        pts = self._checkformat_pts(pts)
        if det is None:
            det = self.get_detector_approx(lamb=self._DEFLAMB)

        # Get local summit nout, e1, e2 if non-centered
        dtheta, psi = self._checkformat_dthetapsi(psi=psi, dtheta=dtheta)
        summit, nout, e1, e2 = self.get_local_noute1e2(dtheta, psi)

        # Compute
        bragg, phi = _comp_optics.calc_braggphi_from_xixjpts(
            det['cent'], det['ei'], det['ej'],
            summit, -nout, e1, e2, pts=pts, lambdtheta=True)
        return bragg, phi

    def get_lamb_avail_from_pts(self, pts):
        pass

    def _calc_dthetapsiphi_from_lambpts(self, pts=None,
                                        lamb=None, n=None, ndtheta=None,
                                        det=None):

        # Check / Format inputs
        pts = self._checkformat_pts(pts)
        npts = pts.shape[1]

        if det is None:
            det = self.get_detector_approx(lamb=self._DEFLAMB)

        if lamb is None:
            lamb = self._DEFLAMB
        lamb = np.r_[lamb]
        nlamb = lamb.size
        bragg = self.get_bragg_from_lamb(lamb, n=n)

        if ndtheta is None:
            ndtheta = 10

        # Compute dtheta, psi, indnan
        dtheta, psi, indnan, indout = _comp_optics.calc_dthetapsiphi_from_lambpts(
            pts, self._dgeom['center'], self._dgeom['rcurve'],
            bragg, nlamb, npts,
            self._dgeom['nout'], self._dgeom['e1'], self._dgeom['e2'],
            self._dgeom['extenthalf'], ndtheta=ndtheta)

        bragg = np.repeat(np.repeat(bragg[:, None], npts, axis=-1)[..., None],
                          ndtheta, axis=-1)
        bragg[indnan] = np.nan
        bragg2, phi = self._calc_braggphi_from_pts(pts, dtheta=dtheta,
                                                   psi=psi, det=det)
        # TBC closely !!!
        # assert np.allclose(bragg, bragg2, equal_nan=True)
        # assert indout.sum() < psi.size
        return dtheta, psi, phi, bragg

    def calc_raytracing_from_lambpts(self, lamb=None, pts=None,
                                     xi_bounds=None, xj_bounds=None, nphi=None,
                                     det=None, n=None, ndtheta=None,
                                     johann=False, lpsi=None, ldtheta=None,
                                     rocking=False, plot=None, fs=None, dmargin=None,
                                     wintit=None, tit=None, proj=None,
                                     legend=None, draw=None, returnas=None):
        """ Visualize the de-focusing by ray-tracing of chosen lamb

        If plot, 3 different plots can be produced:
            - det: plots the intersection of rays with detector plane
            - '2d': plots the geometry of the rays in 2d cross and hor
            - '3d': plots the geometry of the rays in 3d
        Specify the plotting option by setting plot to any of these (or a list)
        """
        # Check / format inputs
        if returnas is None:
            returnas = 'data'
        if lamb is None:
            lamb = self._DEFLAMB
        if plot is None or plot is True:
            plot = ['det', '3d']
        if isinstance(plot, str):
            plot = plot.split('+')
        assert all([ss in ['det', '2d', '3d'] for ss in plot])
        assert returnas in ['data', 'ax']

        # Prepare
        lamb = np.atleast_1d(lamb).ravel()
        nlamb = lamb.size

        pts = self._checkformat_pts(pts)
        npts = pts.shape[1]

        # Get dtheta, psi and phi from pts/lamb
        dtheta, psi, phi, bragg = self._calc_dthetapsiphi_from_lambpts(
            pts=pts, lamb=lamb, n=n, ndtheta=ndtheta)
        ndtheta = dtheta.shape[-1]
        # assert dtheta.shape == (nlamb, npts, ndtheta)

        # Check / get det
        if det is None:
            det = self.get_detector_approx(lamb=self._DEFLAMB)

        # Compute xi, xj of refelxion (phi -> phi + np.pi)
        xi, xj = self.calc_xixj_from_braggphi(
            bragg=bragg, phi=phi+np.pi, n=n,
            dtheta=dtheta, psi=psi,
            det=det, plot=False)

        # Plot
        if plot is not False:
            ptscryst, ptsdet = None, None
            if '2d' in plot or '3d' in plot:
                ptscryst = self.get_local_noute1e2(dtheta, psi)[0]
                ptsdet = (det['cent'][:, None, None, None]
                          + xi[None, ...]*det['ei'][:, None, None, None]
                          + xj[None, ...]*det['ej'][:, None, None, None])

            ax = _plot_optics.CrystalBragg_plot_raytracing_from_lambpts(
                xi=xi, xj=xj, lamb=lamb,
                xi_bounds=xi_bounds, xj_bounds=xj_bounds,
                pts=pts, ptscryst=ptscryst, ptsdet=ptsdet,
                det_cent=det['cent'], det_nout=det['nout'],
                det_ei=det['ei'], det_ej=det['ej'],
                cryst=self, proj=plot, fs=fs, dmargin=dmargin,
                wintit=wintit, tit=tit, legend=legend, draw=draw)
            if returnas == 'ax':
                return ax
        return xi, xj


    def _calc_spect1d_from_data2d(self, data, lamb, phi,
                                  nlambfit=None, nphifit=None,
                                  nxi=None, nxj=None,
                                  spect1d=None, mask=None, vertsum1d=None):
        if nlambfit is None:
            nlambfit = nxi
        if nphifit is None:
            nphifit = nxj
        return _comp_optics._calc_spect1d_from_data2d(data, lamb, phi,
                                                      nlambfit=nlambfit,
                                                      nphifit=nphifit,
                                                      spect1d=spect1d,
                                                      mask=mask,
                                                      vertsum1d=vertsum1d)

    @staticmethod
    def get_dinput_for_fit1d(dlines=None, dconstraints=None,
                             lambmin=None, lambmax=None,
                             same_spectrum=None, nspect=None, dlamb=None):
        """ Return a formatted dict of lines and constraints

        To be fed to _spectrafit2d.multigausfit1d_from_dlines()
        Provides a user-friendly way of defining constraints
        """
        import tofu.data._spectrafit2d as _spectrafit2d
        return _spectrafit2d.multigausfit1d_from_dlines_dinput(
            dlines=dlines, dconstraints=dconstraints,
            lambmin=lambmin, lambmax=lambmax,
            same_spectrum=same_spectrum, nspect=nspect, dlamb=dlamb)

    def plot_data_vs_lambphi(self, xi=None, xj=None, data=None, mask=None,
                             det=None, dtheta=None, psi=None, n=None,
                             nlambfit=None, nphifit=None,
                             magaxis=None, npaxis=None,
                             dlines=None, spect1d='mean',
                             lambmin=None, lambmax=None,
                             xjcut=None, dxj=None,
                             plot=True, fs=None, tit=None, wintit=None,
                             cmap=None, vmin=None, vmax=None,
                             returnas=None):
        # Check / format inputs
        assert data is not None
        if returnas is None:
            returnas = 'spect'
        lreturn = ['ax', 'spect']
        if not returnas in lreturn:
            msg = ("Arg returnas must be in {}\n:".format(lreturn)
                   + "\t- 'spect': return a 1d vertically averaged spectrum\n"
                   + "\t- 'ax'   : return a list of axes instances")
            raise Exception(msg)

        xi, xj, (xii, xjj) = self._checkformat_xixj(xi, xj)
        nxi = xi.size if xi is not None else np.unique(xii).size
        nxj = xj.size if xj is not None else np.unique(xjj).size

        # Compute lamb / phi
        bragg, phi = self.calc_phibragg_from_xixj(
            xii, xjj, n=n, det=det,
            dtheta=dtheta, psi=psi, plot=False)
        assert bragg.shape == phi.shape == data.shape
        lamb = self.get_lamb_from_bragg(bragg, n=n)

        # Compute lambfit / phifit and spectrum1d
        (spect1d, lambfit, phifit,
         vertsum1d, phiminmax) = self._calc_spect1d_from_data2d(
            data, lamb, phi,
            nlambfit=nlambfit, nphifit=nphifit, nxi=nxi, nxj=nxj,
            spect1d=spect1d, mask=mask, vertsum1d=True
        )

        # Get phiref from mag axis
        lambax, phiax = None, None
        if magaxis is not None:
            if npaxis is None:
                npaxis = 1000
            thetacryst = np.arctan2(self._dgeom['summit'][1],
                                    self._dgeom['summit'][0])
            thetaax = thetacryst + np.pi/2*np.linspace(-1, 1, npaxis)
            pts = np.array([magaxis[0]*np.cos(thetaax),
                            magaxis[0]*np.sin(thetaax),
                            np.full((npaxis,), magaxis[1])])
            braggax, phiax = self.calc_phibragg_from_pts(pts)
            lambax = self.get_lamb_from_bragg(braggax)
            phiax = np.arctan2(np.sin(phiax-np.pi), np.cos(phiax-np.pi))
            ind = ((lambax >= lambfit[0]) & (lambax <= lambfit[-1])
                   & (phiax >= phifit[0]) & (phiax <= phifit[-1]))
            lambax, phiax = lambax[ind], phiax[ind]
            ind = np.argsort(lambax)
            lambax, phiax = lambax[ind], phiax[ind]

        # Get lamb / phi for xj
        lambcut, phicut, spectcut = None, None, None
        if xjcut is not None:
            if dxj is None:
                dxj = 0.002
            xjcut = np.sort(np.atleast_1d(xjcut).ravel())
            xicutf = np.tile(xi, (xjcut.size, 1))
            xjcutf = np.repeat(xjcut[:, None], nxi, axis=1)
            braggcut, phicut = self.calc_phibragg_from_xixj(
                xicutf, xjcutf, n=1,
                dtheta=None, psi=None, plot=False, det=det)
            lambcut = self.get_lamb_from_bragg(braggcut, n=1)
            indxj = [(np.abs(xj-xjc) <= dxj).nonzero()[0] for xjc in xjcut]
            spectcut = np.array([np.nanmean(data[ixj, :], axis=0)
                                 for ixj in indxj])

        # plot
        ax = None
        if plot:
            ax = _plot_optics.CrystalBragg_plot_data_vs_lambphi(
                xi, xj, bragg, lamb, phi, data,
                lambfit=lambfit, phifit=phifit, spect1d=spect1d,
                vertsum1d=vertsum1d, lambax=lambax, phiax=phiax,
                lambmin=lambmin, lambmax=lambmax, phiminmax=phiminmax,
                xjcut=xjcut, lambcut=lambcut, phicut=phicut, spectcut=spectcut,
                cmap=cmap, vmin=vmin, vmax=vmax, dlines=dlines,
                tit=tit, wintit=wintit, fs=fs)
        if returnas == 'spect':
            return spect1d, lambfit
        elif returnas == 'ax':
            return ax

    def plot_data_fit1d_dlines(self, xi=None, xj=None, data=None, mask=None,
                               det=None, dtheta=None, psi=None, n=None,
                               nlambfit=None, nphifit=None,
                               lambmin=None, lambmax=None,
                               dlines=None, spect1d=None,
                               dconstraints=None, dx0=None,
                               same_spectrum=None, dlamb=None,
                               double=None, Ti=None, vi=None, ratio=None,
                               dscales=None, x0_scale=None, bounds_scale=None,
                               method=None, max_nfev=None,
                               xtol=None, ftol=None, gtol=None,
                               loss=None, verbose=0, chain=None,
                               jac=None, showonly=None,
                               plot=True, fs=None, dmargin=None,
                               tit=None, wintit=None, returnas=None):
        # Check / format inputs
        assert data is not None
        if showonly is None:
            showonly = False
        if returnas is None:
            returnas = 'dict'
        lreturn = ['ax', 'dict']
        if not returnas in lreturn:
            msg = ("Arg returnas must be in {}\n:".format(lreturn)
                   + "\t- 'dict': return dict of fitted spectrum\n"
                   + "\t- 'ax'  : return a list of axes instances")
            raise Exception(msg)

        xi, xj, (xii, xjj) = self._checkformat_xixj(xi, xj)
        nxi = xi.size if xi is not None else np.unique(xii).size
        nxj = xj.size if xj is not None else np.unique(xjj).size

        # Compute lamb / phi
        bragg, phi = self.calc_phibragg_from_xixj(
            xii, xjj, n=n, det=det,
            dtheta=dtheta, psi=psi, plot=False)
        assert bragg.shape == phi.shape == data.shape
        lamb = self.get_lamb_from_bragg(bragg, n=n)

        # Compute lambfit / phifit and spectrum1d
        (spect1d, lambfit, phifit,
         vertsum1d, phiminmax) = self._calc_spect1d_from_data2d(
            data, lamb, phi,
            nlambfit=nlambfit, nphifit=nphifit, nxi=nxi, nxj=nxj,
            spect1d=spect1d, mask=mask, vertsum1d=True
        )

        # Use valid data only and optionally restrict lamb
        if lambmin is not None:
            spect1d[:, lambfit<lambmin] = np.nan
        if lambmax is not None:
            spect1d[:, lambfit>lambmax] = np.nan
        indok = (~np.any(np.isnan(spect1d), axis=0)) & (~np.isnan(lambfit))
        spect1d = spect1d[:, indok]
        lambfit = lambfit[indok]

        # Get dinput for 1d fitting
        if dlamb is None:
            dlamb = 2.*(np.nanmax(lamb) - np.nanmin(lamb))
        dinput = self.get_dinput_for_fit1d(dlines=dlines,
                                           dconstraints=dconstraints,
                                           lambmin=lambmin, lambmax=lambmax,
                                           same_spectrum=same_spectrum,
                                           nspect=spect1d.shape[0],
                                           dlamb=dlamb)

        # Compute fit for spect1d to get lamb0 if not provided
        if showonly is True:
            dfit1d = {'shift': np.zeros((1, dinput['nlines'])),
                      'coefs': np.zeros((1, dinput['nlines'])),
                      'lamb': lambfit,
                      'data': spect1d,
                      'double': False,
                      'Ti': False,
                      'vi': False,
                      'ratio': None}
        else:
            import tofu.data._spectrafit2d as _spectrafit2d

            dfit1d = _spectrafit2d.multigausfit1d_from_dlines(
                spect1d, lambfit, dinput=dinput, dx0=dx0,
                lambmin=lambmin, lambmax=lambmax,
                dscales=dscales, x0_scale=x0_scale, bounds_scale=bounds_scale,
                method=method, max_nfev=max_nfev,
                chain=chain, verbose=verbose,
                xtol=xtol, ftol=ftol, gtol=gtol, loss=loss,
                ratio=ratio, jac=jac)
            dfit1d['phiminmax'] = phiminmax

        # Plot
        dax = None
        if plot is True:
            ax = _plot_optics.CrystalBragg_plot_data_fit1d(
                dfit1d, dinput=dinput, showonly=showonly,
                lambmin=lambmin, lambmax=lambmax,
                same_spectrum=same_spectrum,
                fs=fs, dmargin=dmargin,
                tit=tit, wintit=wintit)
        if returnas == 'dict':
            return dfit1d
        else:
            return ax

    def fit2d_prepare(self, data=None, xi=None, xj=None, n=None,
                      det=None, dtheta=None, psi=None,
                      mask=None, domain=None,
                      pos=None, binning=None,
                      nbsplines=None, subset=None):
        # ----------------------
        # Geometrical transform
        xi, xj, (xii, xjj) = self._checkformat_xixj(xi, xj)
        nxi = xi.size if xi is not None else np.unique(xii).size
        nxj = xj.size if xj is not None else np.unique(xjj).size

        # Compute lamb / phi
        bragg, phi = self.calc_phibragg_from_xixj(xii, xjj, n=n,
                                                  det=det, dtheta=dtheta,
                                                  psi=psi, plot=False)
        assert bragg.shape == phi.shape
        lamb = self.get_lamb_from_bragg(bragg, n=n)

        # ----------------------
        # Prepare input data (domain, binning, subset, noise...)
        import tofu.data._spectrafit2d as _spectrafit2d
        return _spectrafit2d.multigausfit2d_from_dlines_prepare(
            data, lamb, phi,
            mask=mask, domain=domain,
            pos=pos, binning=binning,
            nbsplines=nbsplines, subset=subset,
            nxi=nxi, nxj=nxj)

    @staticmethod
    def fit2d_dinput(dlines=None, dconstraints=None,
                     Ti=None, vi=None,
                     deg=None, knots=None, nbsplines=None,
                     lambmin=None, lambmax=None,
                     phimin=None, phimax=None,
                     spectvert1d=None, phi1d=None, fraction=None):
        """ Return a formatted dict of lines and constraints

        To be fed to _spectrafit2d.multigausfit1d_from_dlines()
        Provides a user-friendly way of defining constraints
        """
        import tofu.data._spectrafit2d as _spectrafit2d
        return _spectrafit2d.multigausfit2d_from_dlines_dinput(
            dlines=dlines, dconstraints=dconstraints, Ti=Ti, vi=vi,
            deg=deg, knots=knots, nbsplines=nbsplines,
            lambmin=lambmin, lambmax=lambmax,
            phimin=phimin, phimax=phimax,
            spectvert1d=spectvert1d, phi1d=phi1d, fraction=fraction)

    def fit2d(self, xi=None, xj=None, data=None, mask=None,
              det=None, dtheta=None, psi=None, n=None,
              Ti=None, vi=None, domain=None,
              dprepare=None, dinput=None,
              dlines=None, dconstraints=None, dx0=None,
              x0_scale=None, bounds_scale=None,
              deg=None, knots=None, nbsplines=None,
              method=None, max_nfev=None, chain=None,
              xtol=None, ftol=None, gtol=None,
              loss=None, verbose=0, debug=None,
              pos=None, subset=None, binning=None,
              fit1dbinning=None, npts=None, dax=None,
              plotmode=None, angunits=None, indspect=None,
              ratio=None, jac=None, plot=True, fs=None,
              cmap=None, vmin=None, vmax=None,
              spect1d=None, nlambfit=None, sparse=None,
              dmargin=None, tit=None, wintit=None,
              returnas=None, save=None, path=None, name=None):
        """ Perform 2d fitting of a 2d apectromtere image

        Fit the spectrum by a sum of gaussians
        Modulate each gaussian parameters by bsplines in the spatial direction

        data must be provided in shape (nt, nxi, nxj), where:
            - nt is the number of time steps
            - nxi is the nb. of pixels in the horizontal / spectral direction
            - nxj is the nb. of pixels in the vertical / spacial direction

        """

        # Check / format inputs
        if returnas is None:
            returnas = 'dict'
        lreturn = ['ax', 'dict']
        if not returnas in lreturn:
            msg = ("Arg returnas must be in {}\n:".format(lreturn)
                   + "\t- 'dict': return dict of fitted spectrum\n"
                   + "\t- 'ax'  : return a list of axes instances")
            raise Exception(msg)

        # ----------------------
        # Prepare input data
        # (geometrical transform, domain, binning, subset, noise...)
        import tofu.data._spectrafit2d as _spectrafit2d
        if dprepare is None:
            dprepare = self.fit2d_prepare(
                data=data, xi=xi, xj=xj, n=n,
                det=det, dtheta=dtheta, psi=psi,
                mask=mask, domain=domain,
                pos=pos, binning=binning,
                nbsplines=nbsplines, subset=subset)

        # ----------------------
        # Get dinput for 2d fitting from dlines, and dconstraints
        if dinput is None:
            dinput = self.get_dinput_for_fit2d(
                dlines=dlines, dconstraints=dconstraints,
                Ti=Ti, vi=vi,
                deg=deg, knots=knots, nbsplines=nbsplines,
                lambmin=lambmin, lambmax=lambmax, phimin=phimin, phimax=phimax,
                spectphi1d=dprepare['spectphi1d'], phi1d=dprepare['phi1d'])

        # ----------------------
        # Perform 2d fitting
        dfit2d = _spectrafit2d.multigausfit2d_from_dlines(
            dprepare=dprepare, dinput=dinput, dx0=dx0,
            x0_scale=x0_scale, bounds_scale=bounds_scale,
            method=method, max_nfev=max_nfev, sparse=sparse,
            chain=chain, verbose=verbose,
            xtol=xtol, ftol=ftol, gtol=gtol, loss=loss,
            ratio=ratio, jac=jac, npts=npts)

        # ----------------------
        # Optional plotting
        if plot is True:
            if plotmode is None:
                plotmode = 'transform'
            if indspect is None:
                indspect = 0

            if spect1d is not None:
                # Compute lambfit / phifit and spectrum1d
                if nlambfit is None:
                    nlambfit = 200
                ((spect1d, fit1d), lambfit,
                 phifit, _, phiminmax) = self._calc_spect1d_from_data2d(
                     [dataflat[indspect, :], dfit2d['sol_tot'][indspect, :]],
                     lambflat, phiflat,
                     nlambfit=nlambfit, nphifit=10,
                     spect1d=spect1d, mask=None, vertsum1d=False)
            else:
                fit1d, lambfit, phiminmax = None, None, None

            dax = _plot_optics.CrystalBragg_plot_data_fit2d(
                xi=xi, xj=xj, data=data, lamb=lamb, phi=phi, indspect=indspect,
                indok=indok, dfit2d=dfit2d,
                dax=dax, plotmode=plotmode, angunits=angunits,
                cmap=cmap, vmin=vmin, vmax=vmax,
                spect1d=spect1d, fit1d=fit1d,
                lambfit=lambfit, phiminmax=phiminmax,
                fs=fs, dmargin=dmargin,
                tit=tit, wintit=wintit)

        # ----------------------
        # Optional saving
        if save is True:
            pfe = os.path.join()
            np.save(pfe, dinput=dinput, dfit2d=dfit2d)
            msg = ("Saved in:\n"
                   + "\t{}".format(pfe))

        # ----------------------
        # return
        if returnas == 'dict':
            return dfit2d
        else:
            return dax
