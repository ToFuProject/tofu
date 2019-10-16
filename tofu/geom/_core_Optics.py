
"""
This module is the geometrical part of the ToFu general package
It includes all functions and object classes necessary for tomography on Tokamaks
"""

# Built-in
import sys
import os
import warnings
#import copy


# Common
import numpy as np
import datetime as dtm
import matplotlib.pyplot as plt

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
    _ddef = {'Id':{'shot':0,
                   'include':['Mod','Cls','Exp','Diag',
                              'Name','shot','version']},
             'dgeom':{'Type':'sph', 'outline_Type':'rect'},
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
                     'Nstep':50}}
    _DREFLECT_DTYPES = {'specular':0, 'diffusive':1, 'ccube':2}

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
                 Id=None, Name=None, Exp=None,
                 fromdict=None,
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
                               Exp=None, Type=None,
                               include=None,
                               **kwdargs):
        if Id is not None:
            assert isinstance(Id,utils.ID)
            Name, Exp, Type = Id.Name, Id.Exp, Id.Type
        if shot is None:
            shot = cls._ddef['Id']['shot']
        if Type is None:
            Type = cls._ddef['dgeom']['Type']
        if include is None:
            include = cls._ddef['Id']['include']

        dins = {'Name':{'var':Name, 'cls':str},
                'Exp': {'var':Exp, 'cls':str},
                'Type': {'var':Type, 'in':['Tor','Lin']},
                'include':{'var':include, 'listof':str}}
        dins, err, msg = cls._check_InputsGeneric(dins)
        if err:
            raise Exception(msg)

        kwdargs.update({'Name':Name, 'Exp':Exp, 'Type':Type,
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
        lkok = ['Type', 'Type_outline', 'summit', 'extent',
                'nIn', 'e1', 'e2', 'curve_radius']
        assert all([isinstance(ss, str) for ss in dgeom.keys()])
        assert all([ss in lkok for ss in dgeom.keys()])

        for kk in cls._ddef['dgeom'].keys():
            dgeom[kk] = dgeom.get(kk, cls._ddef['dgeom'][kk])
        return dgeom

    @classmethod
    def _checkformat_dmat(cls, dmat=None):
        if dmat is None:
            return
        assert isinstance(dmat, dict)
        lkok = ['formula']
        assert all([isinstance(ss, str) for ss in dmat.keys()])
        assert all([ss in lkok for ss in dmat.keys()])

        for kk in cls._ddef['dmat'].keys():
            dmat[kk] = dmat.get(kk, cls._ddef['dmat'][kk])
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

    ###########
    # Get keys of dictionnaries
    ###########

    @staticmethod
    def _get_keys_dgeom():
        lk = ['Type', 'Type_outline',
              'summit', 'extent',
              'nIn', 'e1', 'e2',
              'curve_rad']
        return lk

    @staticmethod
    def _get_keys_dmat():
        lk = ['lattice', 'formula', 'slice']
        return lk

    @staticmethod
    def _get_keys_dbragg():
        largs = ['bragg_ang']
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
        kwd = self._extract_kwdargs(allkwds, largs)
        self._set_dgeom(**kwds)
        largs = self._get_largs_dmat()
        kwd = self._extract_kwdargs(allkwds, largs)
        self.set_dmat(**kwds)
        largs = self._get_largs_dbragg()
        kwd = self._extract_kwdargs(allkwds, largs)
        self.set_dbragg(**kwds)
        largs = self._get_largs_dmisc()
        kwd = self._extract_kwdargs(allkwds, largs)
        self._set_dmisc(**kwdmisc)
        self._dstrip['strip'] = 0

    ###########
    # set dictionaries
    ###########

    def _set_dgeom(self, dgeom=None):
        dgeom = self._checkformat_inputs_dgeom(dgeom)
        self._dgeom = dgeom

    def _set_dmat(self, dmat=None):
        dmat = self._checkformat_inputs_dgeom(dmat)
        self._dmat = dmat

    def _set_dbragg(self, dbragg=None):
        dbragg = self._checkformat_inputs_dgeom(dbragg)
        self._dbragg = dbragg

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
        self._dgeom.update(**fd['dgeom'])
        self._dmat.update(**fd['dmat'])
        self._dbragg.update(**fd['dbragg'])
        self._dmisc.update(**fd['dmisc'])
        if 'dplot' in fd.keys():
            self._dplot.update(**fd['dplot'])

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


    # -----------------
    # methods for color
    # -----------------

    def set_color(self, col):
        self._set_color(col)

    def get_color(self):
        return self._dmisc['color']

    # -----------------
    # methods for generic first-approx
    # -----------------

    @classmethod
    def _get_bragg_from_lamb(cls, lamb, d, n=1):
        lamb = np.atleast_1d(lamb).ravel()
        nord = np.atleast_1d(n).ravel()

        theta = np.full((lamb.size, nord.size), np.nan)
        sin = nord[None, :]*lamb[:, None]/(2.*d)
        indok = np.abs(sin) <= 1.
        theta[indok] = np.arcsin(sin[indok])
        return theta

    def get_bragg_from_lamb(self, lamb, n=1, d=None):
        """ Braggs' law: n*lamb = 2dsin(theta) """

        if d is None:
            if self._dmat['d'] is None:
                msg = "Instance mesh size not set !\n"
                msg += "  => please provide d !"
                raise Exception(msg)
        else:
            d = float(d)
        return self._get_bragg_from_lamb(lamb, d, n=n)

    @classmethod
    def calc_ellipses_on_plane_2d(cls,
                                  Z, nn, frame_cent, frame_ang,
                                  ang_bragg, ang_param,
                                  return_C=False, Test=True):
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

        if Test:
            # Check / format inputs
            nn = np.atleast_1d(nn).ravel()
            assert nn.size == 3
            nn = nn / np.linalg.norm(nn)
            Z = float(Z)

            frame_cent = np.atleast_1d(frame_cent).ravel()
            assert frame_cent.size == 2
            frame_ang = float(frame_ang)

            ang_bragg = np.atleast_1d(ang_bragg).ravel()
            ang_param = np.atleast_1d(ang_param).ravel()

        # By definition, here, nIn = ez
        nIn = np.array([0., 0., 1.])

        # Deduce natural plane frame (P, e1, e2)
        P = np.array([0., 0., Z])
        e1 = np.cross(nIn, nn)
        e1n = np.linalg.norm(e1)
        if e1n < 1.e-10:
            e1 = np.array([1., 0., 0.])
        else:
            e1 = e1 / e1n
        e2 = np.cross(nn, e1)
        e2 = e2 / np.linalg.norm(e2)

        # Deduce key angles
        costheta = np.cos(ang_bragg)
        sintheta = np.sin(ang_bragg)
        cospsi = np.sum(nIn*nn)
        sinpsi = np.sum(np.cross(nIn, nn)*e1)

        # Deduce ellipse parameters
        cos2sin2 = costheta**2 - sinpsi**2
        x2C = Z * sinpsi * sintheta**2 / cos2sin2
        a = Z * sintheta * cospsi / np.sqrt(cos2sin2)
        b = Z * sintheta * cospsi * costheta / cos2sin2

        # Deduce xi, xj
        rot = np.array([np.cos(frame_ang), np.sin(frame_ang)])
        rot2 = np.array([-np.sin(frame_ang), np.cos(frame_ang)])
        ellipse_trans = np.array([a[None, :]*np.cos(ang_param[:, None])
                                  - frame_cent[0],
                                  b[None, :]*np.sin(ang_param[:, None])
                                  - frame_cent[1] + x2C[None, :]])
        xi = np.sum(ellipse_trans*rot[:, None,None], axis=0)
        xj = np.sum(ellipse_trans*rot2[:, None,None], axis=0)

        if return_C:
            Ci = None
            Cj = None
            return xi, xj, Ci, Cj
        else:
            return xi, xj

    @classmethod
    def plot_ellipses_on_plane_2d(cls, Z, nn, frame_cent, frame_ang,
                                  ang_bragg, ang_param, ax=None):

        # Check / format inputs
        nn = np.atleast_1d(nn).ravel()
        assert nn.size == 3
        nn = nn / np.linalg.norm(nn)
        Z = float(Z)

        frame_cent = np.atleast_1d(frame_cent).ravel()
        ssue202_SpectroX2DCrystalassert frame_cent.size == 2
        frame_ang = float(frame_ang)

        ang_bragg = np.atleast_1d(ang_bragg).ravel()
        ang_param = np.atleast_1d(ang_param).ravel()

        # Compute
        xi, xj, Ci, Cj = cls.calc_ellipses_on_plane_2d(Z, nn,
                                                       frame_cent, frame_ang,
                                                       ang_bragg, ang_param,
                                                       return_C=True,
                                                       Test=False)
        nbragg = ang_bragg.size

        # Separate by bragg angle
        if ax is None:
            fig = plt.figure()
            ax = fig.add_axes([0.1,0.1,0.8,0.8], aspect='equal')

        for ii in range(nbragg):
            deg ='{0:07.3f}'.format(ang_bragg[ii]*180/np.pi)
            ax.plot(xi[:,ii], xj[:,ii], '.', label='bragg %s'%deg)
            #ax.plot(Ci[:,ii], Cj[:,ii], 'x', label='bragg %s - center'%deg)
        ax.set_xlabel(r'xi')
        ax.set_ylabel(r'yi')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.), frameon=False)

        return ax
