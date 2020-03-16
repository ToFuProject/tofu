
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
                       '{0:8.4f}'.format(self._dgeom['rotateangle']*180./np.pi)]
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
             color=None, det_cent=None,
             det_nout=None, det_ei=None, det_ej=None,
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
                            tangent_to_rowland=None, plot=False):
        """ Return approximate ideal detector geometry

        Assumes infinitesimal and ideal crystal
        Assumes detector center tangential to Rowland circle
        Assumes detector center matching lamb (m) / bragg (rad)

        Detector described by center position, and (nout, ei, ej) unit vectors
        By convention, nout = np.cross(ei, ej)
        Vectors (ei, ej) define an orthogonal frame in the detector's plane

        Return:
        -------
        det_cent:   np.ndarray
            (3,) array of (x, y, z) coordinates of detector center
        det_nout:   np.ndarray
            (3,) array of (x, y, z) coordinates of unit vector
                perpendicular to detector' surface
                oriented towards crystal
        det_ei:     np.ndarray
            (3,) array of (x, y, z) coordinates of unit vector
                defining first coordinate in detector's plane
        det_ej:     np.ndarray
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

        lf = ['summit', 'nout', 'e1', 'e2']
        lc = [rcurve is None] + [self._dgeom[kk] is None for kk in lf]
        if any(lc):
            msg = ("Some missing fields in dgeom for computation:"
                   + "\n\t-" + "\n\t-".join(['rcurve'] + lf))
            raise Exception(msg)

        # Compute crystal-centered parameters in (nout, e1, e2)
        func = _comp_optics.get_approx_detector_rel
        (det_dist, n_crystdet_rel,
         det_nout_rel, det_ei_rel) = _comp_optics.get_approx_detector_rel(
            rcurve, bragg, tangent_to_rowland=tangent_to_rowland)

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
        return det_cent, det_nout, det_ei, det_ej

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
        lc = [det_cent is None, det_nout is None,
              det_ei is None, det_ej is None]
        assert all(lc) or not any(lc)
        if all(lc):
            func = self.get_detector_approx
            det_cent, det_nout, det_ei, det_ej = func(lamb=self._DEFLAMB)

        # Get local summit nout, e1, e2 if non-centered
        if dtheta is None:
            dtheta = 0.
        if psi is None:
            psi = 0.
        summit, nout, e1, e2 = self.get_local_noute1e2(dtheta, psi)

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
                                det_cent=None, det_ei=None, det_ej=None,
                                dtheta=None, psi=None,
                                plot=True, ax=None, **kwdargs):

        # Check / format inputs
        xi, xj, (xii, xjj) = self._checkformat_xixj(xi, xj)

        lc = [det_cent is None, det_ei is None, det_ej is None]
        assert all(lc) or not any(lc)
        if all(lc):
            det_cent, _, det_ei, det_ej = self.get_detector_approx(
                lamb=self._DEFLAMB)

        # Get local summit nout, e1, e2 if non-centered
        if dtheta is None:
            dtheta = 0.
        if psi is None:
            psi = 0.
        summit, nout, e1, e2 = self.get_local_noute1e2(dtheta, psi)

        # Get bragg, phi
        bragg, phi = _comp_optics.calc_braggphi_from_xixjpts(
            det_cent, det_ei, det_ej,
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
                                 det_cent=None, det_nout=None,
                                 det_ei=None, det_ej=None,
                                 johann=False, lpsi=None, ldtheta=None,
                                 rocking=False, fs=None, dmargin=None,
                                 wintit=None, tit=None):
        """ Visualize the de-focusing by ray-tracing of chosen lamb
        """
        # Check / format inputs
        if lamb is None:
            lamb = self._DEFLAMB
        lamb = np.atleast_1d(lamb).ravel()
        nlamb = lamb.size

        det = np.array([[xi_bounds[0], xi_bounds[1], xi_bounds[1],
                         xi_bounds[0], xi_bounds[0]],
                        [xj_bounds[0], xj_bounds[0], xj_bounds[1],
                         xj_bounds[1], xj_bounds[0]]])

        # Compute lamb / phi
        _, phi = self.calc_phibragg_from_xixj(
            det[0, :], det[1, :], n=n,
            det_cent=det_cent, det_ei=det_ei, det_ej=det_ej,
            dtheta=None, psi=None, plot=False)
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
                det_cent=det_cent, det_nout=det_nout,
                det_ei=det_ei, det_ej=det_ej, plot=False)

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
                        det_cent=det_cent, det_nout=det_nout,
                        det_ei=det_ei, det_ej=det_ej, plot=False)

        # Get rocking curve error
        if rocking:
            pass

        # Plot
        ax = _plot_optics.CrystalBragg_plot_line_tracing_on_det(
            lamb, xi, xj, xi_er, xj_er, det=det,
            johann=johann, rocking=rocking,
            fs=fs, dmargin=dmargin, wintit=wintit, tit=tit)

    def calc_johannerror(self, xi=None, xj=None, err=None,
                         det_cent=None, det_ei=None, det_ej=None, n=None,
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
            det_cent=det_cent, det_ei=det_ei, det_ej=det_ej,
            dtheta=None, psi=None, plot=False)
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
            det_cent=det_cent, det_ei=det_ei, det_ej=det_ej,
            dtheta=ldtheta, psi=lpsi, plot=False)
        lamberr = self.get_lamb_from_bragg(braggerr, n=n)
        err_lamb = np.nanmax(np.abs(lamb[None, ...] - lamberr), axis=0)
        err_phi = np.nanmax(np.abs(phi[None, ...] - phierr), axis=0)
        if plot is True:
            ax = _plot_optics.CrystalBragg_plot_johannerror(
                xi, xj, lamb, phi, err_lamb, err_phi, err=err,
                cmap=cmap, vmin=vmin, vmax=vmax, fs=fs, tit=tit, wintit=wintit)
        return err_lamb, err_phi

    def _calc_braggphi_from_pts(self, pts,
                                det_cent=None, det_ei=None, det_ej=None,
                                dtheta=None, psi=None):

        # Check / format pts
        pts = self._checkformat_pts(pts)
        lc = [det_cent is None, det_ei is None, det_ej is None]
        assert all(lc) or not any(lc)
        if all(lc):
            det_cent, _, det_ei, det_ej = self.get_detector_approx(
                lamb=self._DEFLAMB)

        # Get local summit nout, e1, e2 if non-centered
        dtheta, psi = self._checkformat_dthetapsi(psi=psi, dtheta=dtheta)
        summit, nout, e1, e2 = self.get_local_noute1e2(dtheta, psi)

        # Compute
        bragg, phi = _comp_optics.calc_braggphi_from_xixjpts(
            det_cent, det_ei, det_ej,
            summit, -nout, e1, e2, pts=pts, lambdtheta=True)
        return bragg, phi

    def get_lamb_avail_from_pts(self, pts):
        pass

    def _calc_dthetapsiphi_from_lambpts(self, pts=None,
                                        lamb=None, n=None, ndtheta=None,
                                        det_cent=None, det_ei=None, det_ej=None):

        # Check / Format inputs
        pts = self._checkformat_pts(pts)
        npts = pts.shape[1]

        lc = [det_cent is None, det_ei is None, det_ej is None]
        assert all(lc) or not any(lc)
        if all(lc):
            det_cent, _, det_ei, det_ej = self.get_detector_approx(
                lamb=self._DEFLAMB)

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
        bragg2, phi = self._calc_braggphi_from_pts(
            pts, dtheta=dtheta, psi=psi,
            det_cent=det_cent, det_ei=det_ei, det_ej=det_ej)
        # TBC closely !!!
        # assert np.allclose(bragg, bragg2, equal_nan=True)
        # assert indout.sum() < psi.size
        return dtheta, psi, phi, bragg

    def calc_raytracing_from_lambpts(self, lamb=None, pts=None,
                                     xi_bounds=None, xj_bounds=None, nphi=None,
                                     det_cent=None, det_nout=None,
                                     det_ei=None, det_ej=None, n=None,
                                     ndtheta=None,
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
        lc = [det_cent is None, det_nout is None,
              det_ei is None, det_ej is None]
        assert all(lc) or not any(lc)
        if all(lc):
            func = self.get_detector_approx
            det_cent, det_nout, det_ei, det_ej = func(lamb=self._DEFLAMB)

        # Compute xi, xj of refelxion (phi -> phi + np.pi)
        xi, xj = self.calc_xixj_from_braggphi(
            bragg=bragg, phi=phi+np.pi, n=n,
            dtheta=dtheta, psi=psi,
            det_cent=det_cent, det_nout=det_nout,
            det_ei=det_ei, det_ej=det_ej, plot=False)

        # Plot
        if plot is not False:
            ptscryst, ptsdet = None, None
            if '2d' in plot or '3d' in plot:
                ptscryst = self.get_local_noute1e2(dtheta, psi)[0]
                ptsdet = (det_cent[:, None, None, None]
                          + xi[None, ...]*det_ei[:, None, None, None]
                          + xj[None, ...]*det_ej[:, None, None, None])

            ax = _plot_optics.CrystalBragg_plot_raytracing_from_lambpts(
                xi=xi, xj=xj, lamb=lamb,
                xi_bounds=xi_bounds, xj_bounds=xj_bounds,
                pts=pts, ptscryst=ptscryst, ptsdet=ptsdet,
                det_cent=det_cent, det_nout=det_nout,
                det_ei=det_ei, det_ej=det_ej,
                cryst=self, proj=plot, fs=fs, dmargin=dmargin,
                wintit=wintit, tit=tit, legend=legend, draw=draw)
            if returnas == 'ax':
                return ax
        return xi, xj


    def _calc_spect1d_from_data2d(self, data, lamb, phi,
                                  nlambfit=None, nphifit=None,
                                  nxi=None, nxj=None,
                                  spect1d=None, mask=None):
        # Check / format inputs
        if spect1d is None:
            spect1d = 'mean'
        lc = [isinstance(spect1d, tuple) and len(spect1d) == 2,
              (isinstance(spect1d, list)
               and all([isinstance(ss, tuple) and len(ss) == 2
                        for ss in spect1d])),
              spect1d in ['mean', 'cent']]
        if lc[0]:
            spect1d = [spect1d]
        elif lc[1]:
            pass
        elif lc[2]:
            if spect1d == 'cent':
                spect1d = [(0., 0.2)]
                nspect = 1
        else:
            msg = ("spect1d must be either:\n"
                   + "\t- 'mean': the avearge spectrum\n"
                   + "\t- 'cent': the central spectrum +/- 20%\n"
                   + "\t- (target, tol); a tuple of 2 floats:\n"
                   + "\t\ttarget: the central value of the window in [-1,1]\n"
                   + "\t\ttol:    the window tolerance (width) in [0,1]\n"
                   + "\t- list of (target, tol)")
            raise Exception(msg)

        # Compute lambfit / phifit and spectrum1d
        if mask is not None:
            data[~mask] = np.nan
        if nlambfit is None:
            nlambfit = nxi
        if nphifit is None:
            nphifit = nxj
        lambfit, phifit = _comp_optics.get_lambphifit(lamb, phi,
                                                      nlambfit, nphifit)
        lambfitbins = 0.5*(lambfit[1:] + lambfit[:-1])
        ind = np.digitize(lamb, lambfitbins)

        # Get phi window
        if spect1d == 'mean':
            phiminmax = np.r_[phifit.min(), phifit.max()][None, :]
            spect1d_out = np.array([np.nanmean(data[ind == jj])
                                    for jj in np.unique(ind)])[None, :]
        else:
            nspect = len(spect1d)
            dphi = np.nanmax(phifit) - np.nanmin(phifit)
            spect1d_out = np.full((nspect, lambfit.size), np.nan)
            phiminmax = np.full((nspect, 2), np.nan)
            for ii in range(nspect):
                phicent = np.nanmean(phifit) + spect1d[ii][0]*dphi/2.
                indphi = np.abs(phi - phicent) < spect1d[ii][1]*dphi
                spect1d_out[ii, :] = [np.nanmean(data[indphi & (ind == jj)])
                                      for jj in np.unique(ind)]
                phiminmax[ii, :] = (np.nanmin(phi[indphi]),
                                    np.nanmax(phi[indphi]))

        phifitbins = 0.5*(phifit[1:] + phifit[:-1])
        ind = np.digitize(phi, phifitbins)
        vertsum1d = np.array([np.nanmean(data[ind == ii])
                              for ii in np.unique(ind)])
        return spect1d_out, lambfit, phifit, vertsum1d, phiminmax

    @staticmethod
    def get_dinput_for_fit1d(dlines=None, dconstraints=None,
                             lambmin=None, lambmax=None):
        """ Return a formatted dict of lines and constraints

        To be fed to _spectrafit2d.multigausfit1d_from_dlines()
        Provides a user-friendly way of defining constraints
        """
        import tofu.data._spectrafit2d as _spectrafit2d
        return _spectrafit2d.multigausfit1d_from_dlines_dinput(
            dlines=dlines, dconstraints=dconstraints,
            lambmin=lambmin, lambmax=lambmax)

    def plot_data_vs_lambphi(self, xi=None, xj=None, data=None, mask=None,
                             det_cent=None, det_ei=None, det_ej=None,
                             dtheta=None, psi=None, n=None,
                             nlambfit=None, nphifit=None,
                             magaxis=None, npaxis=None,
                             dlines=None, spect1d='mean',
                             lambmin=None, lambmax=None,
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
            xii, xjj, n=n,
            det_cent=det_cent, det_ei=det_ei, det_ej=det_ej,
            dtheta=dtheta, psi=psi, plot=False)
        assert bragg.shape == phi.shape == data.shape
        lamb = self.get_lamb_from_bragg(bragg, n=n)

        # Compute lambfit / phifit and spectrum1d
        (spect1d, lambfit, phifit,
         vertsum1d, phiminmax) = self._calc_spect1d_from_data2d(
            data, lamb, phi,
            nlambfit=nlambfit, nphifit=nphifit, nxi=nxi, nxj=nxj,
            spect1d=spect1d, mask=mask
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

        # plot
        ax = None
        if plot:
            ax = _plot_optics.CrystalBragg_plot_data_vs_lambphi(
                xi, xj, bragg, lamb, phi, data,
                lambfit=lambfit, phifit=phifit, spect1d=spect1d,
                vertsum1d=vertsum1d, lambax=lambax, phiax=phiax,
                lambmin=lambmin, lambmax=lambmax, phiminmax=phiminmax,
                cmap=cmap, vmin=vmin, vmax=vmax, dlines=dlines,
                tit=tit, wintit=wintit, fs=fs)
        if returnas == 'spect':
            return spect1d, lambfit
        elif returnas == 'ax':
            return ax

    def plot_data_fit1d_dlines(self, xi=None, xj=None, data=None, mask=None,
                               det_cent=None, det_ei=None, det_ej=None,
                               dtheta=None, psi=None, n=None,
                               nlambfit=None, nphifit=None,
                               lambmin=None, lambmax=None,
                               dlines=None, spect1d=None,
                               dconstraints=None, dx0=None,
                               double=None, Ti=None, vi=None, ratio=None,
                               scales=None, x0_scale=None, bounds_scale=None,
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
            xii, xjj, n=n,
            det_cent=det_cent, det_ei=det_ei, det_ej=det_ej,
            dtheta=dtheta, psi=psi, plot=False)
        assert bragg.shape == phi.shape == data.shape
        lamb = self.get_lamb_from_bragg(bragg, n=n)

        # Compute lambfit / phifit and spectrum1d
        (spect1d, lambfit, phifit,
         vertsum1d, phiminmax) = self._calc_spect1d_from_data2d(
            data, lamb, phi,
            nlambfit=nlambfit, nphifit=nphifit, nxi=nxi, nxj=nxj,
            spect1d=spect1d, mask=mask
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
        dinput = self.get_dinput_for_fit1d(dlines=dlines,
                                           dconstraints=dconstraints,
                                           lambmin=lambmin, lambmax=lambmax)

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
                scales=scales, x0_scale=x0_scale, bounds_scale=bounds_scale,
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
                fs=fs, dmargin=dmargin,
                tit=tit, wintit=wintit)

        if returnas == 'dict':
            return dfit1d
        else:
            return ax

    @staticmethod
    def _checkformat_data_fit2d_dlines(data, lamb, phi, mask=None):
        msg = ""
        if not (data.ndim in [2, 3] and lamb.shape == phi.shape):
            raise Exdeption(msg)
        if mask is not None:
            assert mask.shape == lamb.shape, msg
        if data.ndim == 2:
            lc = [data.shape == lamb.shape, data.shape[1] == lamb.size]
            if lc[0]:
                data = data.ravel()[None, :]
                lamb = lamb.ravel()
                phi = phi.ravel()
                if mask is not None:
                    mask = mask.ravel()
            elif lc[1]:
                assert lamb.ndim == 1
            else:
                raise Exception(msg)
        else:
            assert data.shape[1:] == lamb.shape
            data = data.reshape((data.shape[0], np.prod(data.shape[1:])))
            lamb = lamb.ravel()
            phi = phi.ravel()
            if mask is not None:
                mask = mask.ravel()
        return data, lamb, phi, mask

    @staticmethod
    def get_dinput_for_fit2d(dlines=None, dconstraints=None,
                             deg=None, knots=None, nbsplines=None,
                             lambmin=None, lambmax=None,
                             phimin=None, phimax=None):
        """ Return a formatted dict of lines and constraints

        To be fed to _spectrafit2d.multigausfit1d_from_dlines()
        Provides a user-friendly way of defining constraints
        """
        import tofu.data._spectrafit2d as _spectrafit2d
        return _spectrafit2d.multigausfit2d_from_dlines_dinput(
            dlines=dlines, dconstraints=dconstraints,
            deg=deg, knots=knots, nbsplines=nbsplines,
            lambmin=lambmin, lambmax=lambmax,
            phimin=phimin, phimax=phimax)

    def plot_data_fit2d_dlines(self, xi=None, xj=None, data=None, mask=None,
                               det_cent=None, det_ei=None, det_ej=None,
                               dtheta=None, psi=None, n=None,
                               lambmin=None, lambmax=None,
                               phimin=None, phimax=None,
                               dlines=None, dconstraints=None, dx0=None,
                               x0_scale=None, bounds_scale=None,
                               deg=None, knots=None, nbsplines=None,
                               method=None, max_nfev=None, chain=None,
                               xtol=None, ftol=None, gtol=None,
                               loss=None, verbose=0, debug=None,
                               pos=None, subset=None, npts=None,
                               ratio=None, jac=None, plot=True, fs=None,
                               cmap=None, vmin=None, vmax=None, returnas=None):
        # Check / format inputs
        assert data is not None
        if pos is None:
            pos = True
        if subset is not None:
            assert isinstance(subset, int)
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
            xii, xjj, n=n,
            det_cent=det_cent, det_ei=det_ei, det_ej=det_ej,
            dtheta=dtheta, psi=psi, plot=False)
        assert bragg.shape == phi.shape
        lamb = self.get_lamb_from_bragg(bragg, n=n)

        # Check shape of data (multiple time slices possible)
        data, lamb, phi, mask = self._checkformat_data_fit2d_dlines(
            data, lamb, phi, mask=mask)
        assert lamb.ndim == phi.ndim == 1
        assert data.ndim == 2 and data.shape[1] == lamb.size

        # Use valid data only and optionally restrict lamb / phi
        indok = np.ones(lamb.shape, dtype=bool)
        if mask is not None:
            indok &= mask
        if lambmin is not None:
            indok &= lamb > lambmin
        if lambmax is not None:
            indok &= lamb < lambmax
        if phimin is not None:
            indok &= phi > phimmin
        if phimax is not None:
            indok &= phi < phimax

        # Optionally fit only on subset
        if subset is None:
            data = data[:, indok]
            lamb = lamb[indok]
            phi = phi[indok]
        else:
            data = data[:, indok][:, ::subset]
            lamb = lamb[indok][::subset]
            phi = phi[indok][::subset]

        if pos is True:
            data[data < 0.] = 0.

        # Get dinput for 1d fitting
        dinput = self.get_dinput_for_fit2d(dlines=dlines,
                                           dconstraints=dconstraints,
                                           deg=deg, knots=knots,
                                           nbsplines=nbsplines,
                                           lambmin=lambmin, lambmax=lambmax,
                                           phimin=phi.min(),
                                           phimax=phi.max())

        # Perform 1d fit to be used as initial guess for 2d fitting
        import tofu.data._spectrafit2d as _spectrafit2d

        dfit2d = _spectrafit2d.multigausfit2d_from_dlines(
            data, lamb, phi, dinput=dinput, dx0=dx0,
            x0_scale=x0_scale, bounds_scale=bounds_scale,
            method=method, max_nfev=max_nfev,
            chain=chain, verbose=verbose,
            xtol=xtol, ftol=ftol, gtol=gtol, loss=loss,
            ratio=ratio, jac=jac, npts=npts)

        import pdb; pdb.set_trace()      # DB

        # Plot
        dax = None
        if plot is True:
            dax = _plot_optics.CrystalBragg_plot_data_fit2d(
                xi, xj, indok, dfit2d, dinput=dinput,
                fs=fs, dmargin=dmargin,
                tit=tit, wintit=wintit)

        if returnas == 'dict':
            return dfit2d
        else:
            return dax
