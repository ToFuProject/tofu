
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
from . import _def as _def
from . import _GG as _GG
from . import _check_optics
from . import _comp_optics as _comp_optics
from . import _plot_optics as _plot_optics


__all__ = ['CrystalBragg']


_Type = 'Tor'
_NTHREADS = 16

# rotate / translate instance
_RETURN_COPY = False
_USE_NON_PARALLELISM = True


"""
###############################################################################
###############################################################################
                        Ves class and functions
###############################################################################
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
    _ddef = {
        'Id': {
            'shot': 0, 'Exp': 'dummy', 'Diag': 'dummy',
            'include': [
                'Mod', 'Cls', 'Exp', 'Diag', 'Name', 'shot', 'version',
            ],
        },
        'dgeom': {'Type': 'sph', 'Typeoutline': 'rect'},
        'dmat': {},
        'dbragg': {'braggref': np.pi/4.},
        'dmisc': {'color': 'k'},
    }
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
    # _DEFLAMB = 3.971561e-10
    # _DEFNPEAKS = 12
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
              'lengths', 'angles', 'cut', 'd',
              'alpha', 'beta', 'nin', 'nout', 'e1', 'e2']
        return lk

    @staticmethod
    def _get_keys_dbragg():
        lk = ['rockingcurve', 'lambref', 'braggref']
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
        self._dgeom = _check_optics._checkformat_dgeom(
            dgeom=dgeom, ddef=self._ddef['dgeom'],
            valid_keys=self._get_keys_dgeom(),
        )
        if self._dgeom['move'] is not None:
            self.set_move(
                move=self._dgeom['move'],
                param=self._dgeom['move_param'],
                **self._dgeom['move_kwdargs'],
            )

    def set_dmat(self, dmat=None):
        self._dmat = _check_optics._checkformat_dmat(
            dmat=dmat, dgeom=self._dgeom,
            ddef=self._ddef['dmat'],
            valid_keys=self._get_keys_dmat()
        )

    def set_dbragg(self, dbragg=None):
        self._dbragg = _check_optics._checkformat_dbragg(
            dbragg=dbragg,
            ddef=self._ddef['dbragg'],
            valid_keys=self._get_keys_dbragg(),
            dmat=self._dmat,
        )

    def _set_color(self, color=None):
        color = _check_optics._checkformat_inputs_dmisc(
            color=color, ddef=self._ddef,
        )
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

    # @property
    # def nin(self):
        # return self._dgeom['nin']

    # @property
    # def nout(self):
        # return self._dgeom['nout']

    # @property
    # def e1(self):
        # return self._dgeom['e1']

    # @property
    # def e2(self):
        # return self._dgeom['e2']

    @property
    def summit(self):
        return self._dgeom['summit']

    @property
    def center(self):
        return self._dgeom['center']

    @property
    def ismobile(self):
        return self._dgeom['move'] not in [None, False]

    @property
    def rockingcurve(self):
        if self._dbragg.get('rockingcurve') is not None:
            if self._dbragg['rockingcurve'].get('type') is not None:
                return self._dbragg['rockingcurve']
        raise Exception("rockingcurve was not set!")

    # --------------------------------------
    # methods for getting unit vectors basis
    # --------------------------------------

    def get_unit_vectors(self, use_non_parallelism=None):
        """ Return the unit vectors (direct orthonormal basis)

        Depending on:
            use_non_parallelism: True  => return the geometrical basis
            use_non_parallelism: False  => return the mesh basis

        """
        if use_non_parallelism is None:
            use_non_parallelism = _USE_NON_PARALLELISM

        if use_non_parallelism is True:
            nout = self._dmat['nout']
            e1 = self._dmat['e1']
            e2 = self._dmat['e2']
        else:
            nout = self._dgeom['nout']
            e1 = self._dgeom['e1']
            e2 = self._dgeom['e2']
        return nout, e1, e2, use_non_parallelism

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
        col0 = [
            'formula', 'symmetry', 'cut', 'density',
            'd (A)',
            'bragg({:9.6} A) (deg)'.format(self._dbragg['lambref']*1e10),
            'Type', 'outline', 'surface (cm²)', 'rcurve', 'rocking curve',
        ]
        ar0 = [self._dmat['formula'], self._dmat['symmetry'],
               str(self._dmat['cut']), str(self._dmat['density']),
               '{0:5.3f}'.format(self._dmat['d']*1.e10),
               str(self._dbragg['braggref']*180./np.pi),
               self._dgeom['Type'], self._dgeom['Typeoutline'],
               '{0:5.1f}'.format(self._dgeom['surface']*1.e4),
               '{0:6.3f}'.format(self._dgeom['rcurve'])]
        try:
            ar0.append(self.rockingcurve['type'])
        except Exception as err:
            ar0.append('None')


        # -----------------------
        # Build geometry
        col1 = ['half-extent', 'summit', 'center', 'nout', 'e1',
                'alpha', 'beta']
        ar1 = [
            str(np.round(self._dgeom['extenthalf'], decimals=3)),
            str(np.round(self._dgeom['summit'], decimals=2)),
            str(np.round(self._dgeom['center'], decimals=2)),
            str(np.round(self._dmat['nout'], decimals=3)),
            str(np.round(self._dmat['e1'], decimals=3)),
            str(np.round(self._dmat['alpha'], decimals=6)),
            str(np.round(self._dmat['beta'], decimals=6)),
        ]
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

    def translate_3d(self, distance=None, direction=None,
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

        Also return the wavelength (lamb) (in meters) for which it was computed
            and the associated reference bragg angle (in rad)

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
                          dmargin=None, fs=None, ax=None, legend=None):
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

    def sample_outline_plot(self, use_non_parallelism=None, res=None):
        if self._dgeom['Type'] == 'sph':
            if self._dgeom['Typeoutline'] == 'rect':
                nout, e1, e2, use_non_parallelism = self.get_unit_vectors(
                    use_non_parallelism=use_non_parallelism,
                )
                outline = _comp_optics.CrystBragg_sample_outline_plot_sphrect(
                    self._dgeom['summit'] - nout*self._dgeom['rcurve'],
                    nout,
                    e1,
                    e2,
                    self._dgeom['rcurve'],
                    self._dgeom['extenthalf'],
                    res,
                )
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
            lamb = self._dbragg['lambref']
            lc[0] = True
        assert np.sum(lc) == 1, "Provide lamb xor bragg!"
        if lc[0]:
            bragg = self.get_bragg_from_lamb(
                np.atleast_1d(lamb), n=n,
            )
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

    def _get_rays_from_cryst(
        self,
        phi=None, bragg=None,
        lamb=None, n=None,
        dtheta=None, psi=None,
        ntheta=None, npsi=None,
        use_non_parallelism=None,
        include_summit=None,
        grid=None,
    ):

        # Get phi, bragg
        bragg = self._checkformat_bragglamb(bragg=bragg, lamb=lamb)
        phi, bragg = self._checkformat_get_Rays_from(phi=phi, bragg=bragg)
        # assert phi.ndim == 1

        # Get local summits, nout, e1, e2
        pts_start, nout, e1, e2 = self.get_local_noute1e2(
            dtheta=dtheta, psi=psi,
            use_non_parallelism=use_non_parallelism,
            ntheta=ntheta, npsi=npsi,
            include_summit=include_summit,
        )
        nin = -nout
        # reshape for broadcast
        if grid is True:
            nin = nin[..., None]
            e1 = e1[..., None]
            e2 = e2[..., None]
        else:
            assert bragg.shape == nin.shape[1:]

        # Compute start point (D) and unit vectors (us)
        vect = (
            np.sin(bragg)*nin
            + np.cos(bragg)*(np.cos(phi)*e1 + np.sin(phi)*e2)
        )
        return pts_start, vect

    def get_rays_from_cryst(
        self,
        phi=None, bragg=None,
        lamb=None, n=None,
        dtheta=None, psi=None,
        use_non_parallelism=None,
        ntheta=None, npsi=None,
        include_summit=None,
        det=None, config=None, length=None,
        returnas=None,
        return_xixj=None,
        grid=None,
    ):

        # -----------
        # Check input
        if returnas is None:
            returnas = 'pts'
        if return_xixj is None:
            return_xixj = False

        lret = ['(pts, vect, length)', '(pts, vect)', 'pts']    # , object]
        if returnas not in lret:
            msg = (
                "Arg returnas must be in:\n"
                + "\t- '(pts, vect, length)': starting points, unit vector,"
                + " length\n"
                + "\t- 'pts': starting and ending points\n"
                # + "\t- object: CamLOS1D instance\n"
            )
            raise Exception(msg)

        det = self._checkformat_det(det)
        if det is False:
            msg = "det is required in get_Rays_from_summit_to_det()!"
            raise Exception(msg)

        if length is None:
            length = 7.

        if grid is None:
            try:
                grid = bragg.shape != dtheta.shape
            except Exception as err:
                grid = True

        # -----------
        # Get starting point and vectors
        pts_start, vect = self._get_rays_from_cryst(
            phi=phi, bragg=bragg,
            lamb=lamb, n=n,
            dtheta=dtheta, psi=psi,
            use_non_parallelism=use_non_parallelism,
            ntheta=ntheta, npsi=npsi,
            include_summit=include_summit,
            grid=grid,
        )

        if returnas == '(pts, vect)':
            return pts_start, vect

        # -----------
        # Get length (minimum between conf, det, length)
        dk = {
            k0: np.full(vect.shape[1:], np.nan)
            for k0 in ['config', 'det', 'length']
        }
        xi, xj = None, None
        if config is not None:
            dk['config'] = None
        if det is not None:
            shape = tuple([3] + [1 for ii in range(vect.ndim-1)])
            cent = det['cent'].reshape(shape)
            nout = det['nout'].reshape(shape)
            if grid is True:
                k = (
                    np.sum((cent-pts_start[..., None])*nout, axis=0)
                    / np.sum(vect*nout, axis=0)
                )
            else:
                k = (
                    np.sum((cent-pts_start)*nout, axis=0)
                    / np.sum(vect*nout, axis=0)
                )
            dk['det'][k >= 0.] = k[k >= 0.]
            if return_xixj is True:
                if grid:
                    pts_end = pts_start[..., None] + dk['det'][None, ...]*vect
                else:
                    pts_end = pts_start + dk['det'][None, ...]*vect
                ei = det['ei'].reshape(shape)
                ej = det['ej'].reshape(shape)
                xi = np.sum((pts_end - cent)*ei, axis=0)
                xj = np.sum((pts_end - cent)*ej, axis=0)
        if length is not None:
            dk['length'][:] = length

        k = np.nanmin([vv for vv in dk.values() if vv is not None], axis=0)

        # -----------
        # return
        if returnas == 'pts':
            if grid:
                pts_end = pts_start[..., None] + k[None, ...]*vect
                if return_xixj:
                    return pts_start, pts_end, xi, xj
                else:
                    return pts_start, pts_end
            else:
                pts_end = pts_start + k[None, ...]*vect
                if return_xixj:
                    return pts_start, pts_end, xi, xj
                else:
                    return pts_start, pts_end
        elif returnas == '(pts, vect, length)':
            if return_xixj:
                return pts_start, vect, k, xi, xj
            else:
                return pts_start, vect, k

    # -----------------
    # methods for crystal splitting
    # -----------------

    def split(self, direction=None, nb=None):

        # ------------
        # check inputs
        if direction is None:
            direction = 'e1'
        if direction not in ['e1', 'e2']:
            msg = (
                "Arg direction must be either:\n"
                "\t- 'e1': split along vector 'e1' (~horizontally)\n"
                "\t- 'e2': split along vector 'e2' (~vertically)\n"
                f"You provided: {direction}"
            )
            raise Exception(msg)

        if nb is None:
            nb = 2
        if not (isinstance(nb, int) and nb > 1):
            msg = (
                "Arg nb must be a int > 1 !\n"
                "It specifies the number of equal parts desired\n"
                f"You provided: {nb}"
            )
            raise Exception(msg)

        # ---------------
        # split

        edges = np.linspace(-1, 1, nb+1)
        mid = 0.5*(edges[1:] + edges[:-1])[None, :]
        if direction == 'e2':
            dtheta = mid*self._dgeom['extenthalf'][1]
            psi = np.zeros((1, nb), dtype=float)
            extenthalf = [
                self._dgeom['extenthalf'][0],
                self._dgeom['extenthalf'][1]/nb,
            ]
        else:
            dtheta = np.zeros((1, nb), dtype=float)
            psi = mid*self._dgeom['extenthalf'][0]
            extenthalf = [
                self._dgeom['extenthalf'][0]/nb,
                self._dgeom['extenthalf'][1],
            ]

        nouts = (
            np.cos(dtheta)*(
                self._dgeom['nout'][:, None]*np.cos(psi)
                + self._dgeom['e1'][:, None]*np.sin(psi)
            )
            + np.sin(dtheta)*self._dgeom['e2'][:, None]
        )
        e1s = (
            -self._dgeom['nout'][:, None]*np.sin(psi)
            + self._dgeom['e1'][:, None]*np.cos(psi)
        )
        e2s = np.array([
            nouts[1, :]*e1s[2, :] - nouts[2, :]*e1s[1, :],
            nouts[2, :]*e1s[0, :] - nouts[0, :]*e1s[2, :],
            nouts[0, :]*e1s[1, :] - nouts[1, :]*e1s[0, :],

        ])

        # -----------
        # Construct list of instances

        lobj = [
            self.__class__(
                dgeom={
                    'rcurve': self._dgeom['rcurve'],
                    'center': self._dgeom['center'],
                    'nout': nouts[:, ii],
                    'e1': e1s[:, ii],
                    'e2': e2s[:, ii],
                    'extenthalf': extenthalf,
                },
                dmat={
                    k0: v0 for k0, v0 in self._dmat.items()
                    if k0 not in ['nin', 'nout', 'e1', 'e2']
                },
                dbragg=dict(self._dbragg),
                Name=f"{self.Id.Name}{ii}",
                Exp=self.Id.Exp,
            )
            for ii in range(nb)
        ]

        return lobj



    # -----------------
    # methods for general plotting
    # -----------------

    def plot(
        self, dcryst=None,
        phi=None, bragg=None, lamb=None, pts=None,
        n=None, config=None, det=None, length=None,
        dtheta=None, psi=None,
        ntheta=None, npsi=None,
        include_summit=None,
        dax=None, proj=None, res=None, element=None,
        color=None, ddet=None,
        dleg=None, draw=True, dmargin=None,
        use_non_parallelism=None, grid=None,
        rays_npts=None, rays_color=None,
        fs=None, wintit=None, tit=None,
    ):
        """ Plot the crystal in desired projeection

        The projection is 3d, cross-section or horizontal
        Optionaly add rays reflected on cryst at:
            - lamb / phi: desired wavelength and incidence angle
        and either:
            - psi, dtheta : desired pts on the crystal surface
            - pts: emitted from desired pts (e.g.: in the plasma)
                   (need to be refresh with get_rays_from_cryst method
                    if new pts are wanted)

        Parameters
        ----------
        dax:        None / dict
            dict of axes to be used, with keys:
                - 'cross': axe where to plot cross-section view
                - 'hor':   axe where to plot horizontal (from top) view
                - '3d':    axe where to plot 3d view
            if None, a new figure and axes are created
        proj:       None / str
            key indicating which plot to make:
                - 'cross':  cross-section projection
                - 'hor':    horizontal projection
                - 'all':    cross-section + horizontal view
                - '3d':     3d view
        element:    None / str
            char string where each letter indicates an element to plot
                - 'o': outline (edges of crystal)
                - 's': summit (geometrical center of the crystal)
                - 'c': center (of the sphere of curvature)
                - 'r': rowland circle (plotted in e1 direction)
                - 'v': local unit vectors e1, e2, nout
            If None, default to 'oscvr'
        res:        None / float
            Resolution for the discretization of the outline
        dcryst:     None / dict
            dict of dict for plotting the various elements of the crystal:
                - 'outline': dict of properties fed to plot()
                - 'cent': dict of properties fed to plot()
                - 'summit': dict of properties fed to plot()
                - 'rowland': dict of properties fed to plot()
                - 'vectors': dict of properties fed to quiver()
        ddet:       None / dict
            dict of dict for plotting the various elements of the det:
                - 'outline': dict of properties fed to plot()
                - 'cent': dict of properties fed to plot()
                - 'vectors': dict of properties fed to quiver()
        color:      None / str / tuple
            color to be used for plotting
            Overwrites all colors in dcryst and ddet
        det:        None / dict
            Optionnal associated detector to be plotted, as a dict with keys:
                - 'cent': 1d array of cartesian coordinates of the center
                - 'nout': 1d array of cartesian coordinates of unit vector
                            oriented towards the crystal
                - 'ei':   1d array of cartesian coordinates of unit vector
                - 'ej':   1d array of cartesian coordinates of unit vector
                - 'outline': 2d array of outline coordinates in (ei, ej)
        dleg:       None / dict
            dict of properties to be passed to plt.legend()
            if False legend is not plotted
        use_non_parallelism:    None / str
            Return the unit vectors (direct orthonormal basis)
            Depending on:
                - use_non_parallelism: True  => return the geometrical basis
                - use_non_parallelism: False  => return the mesh basis
        """
        if det is None:
            det = False
        det = self._checkformat_det(det)

        lc = [
            dtheta is not None or psi is not None or phi is not None,
            pts is not None
        ]
        if np.sum(lc) == 2:
            msg = (
                "For ray tracing, please provide either:\n"
                + "\t- dtheta, psi, phi, lamb/bragg\n"
                + "\t- pts, lamb/bragg\n"
            )
            raise Exception(msg)

        # Add rays?
        if lc[0]:
            # Get one way
            # pts.shape = (3, nlamb, npts, ndtheta)
            pts_summit, pts1 = self.get_rays_from_cryst(
                phi=phi, lamb=lamb, bragg=bragg,
                n=n, use_non_parallelism=use_non_parallelism,
                dtheta=dtheta, psi=psi,
                ntheta=ntheta, npsi=npsi,
                include_summit=include_summit,
                config=config, det=det,
                returnas='pts', return_xixj=False,
                grid=grid,
            )
            # Get the other way
            pts2, xi, xj = self.get_rays_from_cryst(
                phi=phi+np.pi, lamb=lamb, bragg=bragg,
                n=n, use_non_parallelism=use_non_parallelism,
                dtheta=dtheta, psi=psi,
                ntheta=ntheta, npsi=npsi,
                include_summit=include_summit,
                config=config, det=det,
                returnas='pts', return_xixj=True,
                grid=grid,
            )[1:]
        elif lc[1]:
            c0 = (
                isinstance(pts, np.ndarray)
                and pts.ndim == 2
                and pts.shape[0] == 3
            )
            if not c0:
                msg = ("Arg pts must be a (3, npts) np.array!")
                raise Exception(msg)

            # pts.shape = (nlamb, npts, ndtheta)
            dtheta, psi, phi, bragg, _, _ = self.calc_raytracing_from_lambpts(
                pts=pts, lamb=lamb,
            )
            pts_summit, pts2, xi, xj = self.get_rays_from_cryst(
                phi=phi+np.pi, lamb=None, bragg=bragg,
                n=n, use_non_parallelism=use_non_parallelism,
                dtheta=dtheta, psi=psi,
                ntheta=ntheta, npsi=npsi,
                include_summit=include_summit,
                config=config, det=det,
                returnas='pts', return_xixj=True,
                grid=grid,
            )
            pts1 = np.repeat(
                np.repeat(
                    np.repeat(
                        pts[:, None, :], dtheta.shape[0], axis=1,
                    )[..., None],
                    dtheta.shape[2],
                    axis=-1,
                )[..., None],
                2,
                axis=-1,
            )
        else:
            pts_summit, pts1, pts2, xi, xj = None, None, None, None, None
        return _plot_optics.CrystalBragg_plot(
            cryst=self, dcryst=dcryst,
            det=det, ddet=ddet,
            dax=dax, proj=proj, res=res, element=element,
            color=color,
            pts_summit=pts_summit, pts1=pts1, pts2=pts2,
            xi=xi, xj=xj,
            rays_color=rays_color, rays_npts=rays_npts,
            dleg=dleg, draw=draw, fs=fs, dmargin=dmargin,
            use_non_parallelism=use_non_parallelism,
            wintit=wintit, tit=tit,
        )

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
            lamb = self._dbragg['lambref']
        return _comp_optics.get_bragg_from_lamb(
            np.atleast_1d(lamb), self._dmat['d'], n=n,
        )

    def get_lamb_from_bragg(self, bragg=None, n=None):
        """ Braggs' law: n*lamb = 2dsin(bragg) """
        if self._dmat['d'] is None:
            msg = "Interplane distance d no set !\n"
            msg += "  => self.set_dmat({'d':...})"
            raise Exception(msg)
        if bragg is None:
            bragg = self._dbragg['braggref']
        return _comp_optics.get_lamb_from_bragg(np.atleast_1d(bragg),
                                                self._dmat['d'], n=n)

    def update_non_parallelism(self, alpha=None, beta=None):
        """ Compute new values of unit vectors nout, e1 and e2 into
        dmat basis, due to non parallelism

        Update new values into dmat dict
        """
        if alpha is None:
            alpha = 0
        if beta is None:
            beta = 0

        (self._dmat['nin'], self._dmat['nout'], self._dmat['e1'],
         self._dmat['e2']) = _comp_optics.get_vectors_from_angles(
                         alpha, beta,
                         self._dgeom['nout'], self._dgeom['e1'],
                         self._dgeom['e2'],
                         )
        self._dmat['alpha'], self._dmat['beta'] = alpha, beta

    def calc_meridional_sagital_focus(
        self,
        rcurve=None,
        bragg=None,
        alpha=None,
        use_non_parallelism=None,
        verb=None,
    ):
        """ Compute sagittal and meridional focuses distances.
        Optionnal result according to non-parallelism, using first the
        update_non_parallelism method.

        parameters
        ----------
        rcurve:     float
            in dgeom dict., curvature radius of the crystal.
        bragg:      float
            in dbragg dict., reference bragg angle of the crystal.
        alpha:      float
            in dmat dict., amplitude of the non-parallelism
            as an a angle defined by user, in radian.
        use_non_parallelism:    str
            Need to be True to use new alpha angle

        Return
        ------
        merid_ref:  float
            Distance crystal-meridional focus (m), for a perfect crystal
        sagit_ref:  float
            Distance crystal-sagital focus (m), for a perfect crystal
        merid_unp:  float
            Distance crystal-meridional focus (m), using non_parallelism
        sagit_unp:  float
            Distance crystal-sagital focus (m), using non_parallelism

        """
        # Check inputs
        if rcurve is None:
            rcurve = self._dgeom['rcurve']
        if bragg is None:
            bragg = self._dbragg['braggref']
        if use_non_parallelism is True:
            alpha = self._dmat['alpha']
        if use_non_parallelism is False:
            alpha = 0.0

        # Compute
        return _comp_optics.calc_meridional_sagital_focus(
            rcurve=rcurve,
            bragg=bragg,
            alpha=alpha,
            use_non_parallelism=use_non_parallelism,
            verb=verb,
        )

    def get_rowland_dist_from_lambbragg(self, bragg=None, lamb=None, n=None):
        """ Return the array of dist from cryst summit to pts on rowland """
        bragg = self._checkformat_bragglamb(bragg=bragg, lamb=lamb, n=n)
        if np.all(np.isnan(bragg)):
            msg = ("There is no available bragg angle!\n"
                   + "  => Check the vlue of self.dmat['d'] vs lamb")
            raise Exception(msg)
        return _comp_optics.get_rowland_dist_from_bragg(
            bragg=bragg, rcurve=self._dgeom['rcurve'],
        )

    def get_detector_approx(
        self,
        bragg=None, lamb=None,
        rcurve=None, n=None,
        ddist=None, di=None, dj=None,
        dtheta=None, dpsi=None, tilt=None,
        lamb0=None, lamb1=None, dist01=None,
        use_non_parallelism=None,
        tangent_to_rowland=None, plot=False,
    ):
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
                'outline':   np.darray
                    (2, N) array to build detector's contour
                    where the last point is identical to the first.
                    (for example for WEST X2D spectrometer:
                    x*np.r_[-1,-1,1,1,-1], y*np.r_[-1,1,1,-1,-1])
        """

        # ---------------------
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
            msg = (
                "Arg lamb0, lamb1 and dist01 must be provided together:\n"
                + "\t- lamb0: line0 wavelength ({})\n".format(lamb0)
                + "\t- lamb1: line1 wavelength ({})\n".format(lamb1)
                + "\t- dist01: distance (m) on detector between lines "
                + "({})".format(dist01)
            )
            raise Exception(msg)
        bragg01 = None
        if all(lc):
            bragg01 = self._checkformat_bragglamb(
                lamb=np.r_[lamb0, lamb1], n=n,
            )

        # split into 2 different condition because of dmat
        lc = [rcurve is None, self._dgeom['summit'] is None]
        if any(lc):
            msg = (
                "Some missing fields in dgeom for computation:"
                + "\n\t-" + "\n\t-".join(['rcurve'] + 'summit')
            )
            raise Exception(msg)

        nout, e1, e2, use_non_parallelism = self.get_unit_vectors(
            use_non_parallelism=use_non_parallelism,
        )

        lc = [cc is None for cc in [nout, e1, e2]]
        if any(lc):
            msg = (
                """
                Field 'nout', 'e1', 'e2' missing!
                """
                )
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
            self._dgeom['summit'], nout, e1, e2,
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

    def _checkformat_det(self, det=None):
        lc = [det is None, det is False, isinstance(det, dict)]
        msg = ("det must be:\n"
               + "\t- False: not det provided\n"
               + "\t- None:  use default approx det from:\n"
               + "\t           self.get_detector_approx()\n"
               + "\t- dict:  a dictionary of 3d (x,y,z) coordinates of a point"
               + " (local frame center) and 3 unit vectors forming a direct "
               + "orthonormal basis attached to the detector's frame\n"
               + "\t\t\t\t- 'cent': detector center\n"
               + "\t\t\t\t- 'nout': unit vector perpendicular to surface, "
               + "in direction of the crystal\n"
               + "\t\t\t\t- 'ei': unit vector, first coordinate on surface\n"
               + "\t\t\t\t- 'ej': unit vector, second coordinate on surfacei\n"
               + "  You provided: {}".format(det))
        if not any(lc):
            raise Exception(msg)
        if lc[0]:
            det = self.get_detector_approx(lamb=self._dbragg['lambref'])
        elif lc[2]:
            lk = ['cent', 'nout', 'ei', 'ej']
            c0 = (isinstance(det, dict)
                  and all([(kk in det.keys()
                            and hasattr(det[kk], '__iter__')
                            and np.atleast_1d(det[kk]).size == 3
                            and not np.any(np.isnan(det[kk])))
                           for kk in lk]))
            if not c0:
                raise Exception(msg)
            for k0 in lk:
                det[k0] = np.atleast_1d(det[k0]).ravel()
        return det

    def get_local_noute1e2(
        self,
        dtheta=None, psi=None,
        ntheta=None, npsi=None,
        use_non_parallelism=None,
        include_summit=None,
    ):
        """ Return (nout, e1, e2) associated to pts on the crystal's surface

        All points on the spherical crystal's surface are identified
            by (dtheta, psi) coordinates, where:
                - theta  = np.pi/2 + dtheta (dtheta=0 default) for the center
                (for the diffracted beam), from frame's basis vector ez
                - psi = 0 for the center, positive in direction of e1
            They are the spherical coordinates from a sphere centered on the
            crystal's center of curvature.

        Return the pts themselves and the 3 perpendicular unit vectors
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
        # Get local basis at crystal summit
        nout, e1, e2, use_non_parallelism = self.get_unit_vectors(
            use_non_parallelism=use_non_parallelism,
        )
        nin = -nout

        # Get vectors at any points from psi & dtheta
        vout, ve1, ve2 = _comp_optics.CrystBragg_get_noute1e2_from_psitheta(
            nout, e1, e2,
            psi=psi, dtheta=dtheta,
            e1e2=True, sameshape=False,
            extenthalf_psi=self._dgeom['extenthalf'][0],
            extenthalf_dtheta=self._dgeom['extenthalf'][1],
            ntheta=ntheta, npsi=npsi,
            include_summit=include_summit,
        )
        vin = -vout
        # cent no longer dgeom['center'] because no longer a fixed point
        cent = self._dgeom['summit'] + self._dgeom['rcurve']*nin
        if vout.ndim == 2:
            cent = cent[:, None]
        elif vout.ndim == 3:
            cent = cent[:, None, None]
        elif vout.ndim == 4:
            cent = cent[:, None, None, None]
        elif vout.ndim == 5:
            cent = cent[:, None, None, None, None]
        else:
            msg = "nout.ndim > 5!"
            raise Exception(msg)
        # Redefining summit according to nout at each point at crystal
        summ = cent + self._dgeom['rcurve']*vout
        return summ, vout, ve1, ve2

    def calc_xixj_from_braggphi(
        self,
        phi=None,
        bragg=None,
        lamb=None,
        n=None,
        dtheta=None,
        psi=None,
        det=None,
        use_non_parallelism=None,
        strict=None,
        return_strict=None,
        data=None,
        plot=True,
        dax=None,
    ):
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
        if return_strict is None:
            return_strict = False

        # Check / format inputs
        bragg = self._checkformat_bragglamb(bragg=bragg, lamb=lamb, n=n)
        phi = np.atleast_1d(phi)

        # Check / get det
        det = self._checkformat_det(det)

        # Get local summit nout, e1, e2 if non-centered
        if dtheta is None:
            dtheta = 0.
        if psi is None:
            psi = 0.

        # Probably to update with use_non_parallelism?
        # Get back summit & vectors at any point at the crystal surface,
        #  according to parallelism properties
        summit, nout, e1, e2 = self.get_local_noute1e2(
            dtheta=dtheta, psi=psi,
            use_non_parallelism=use_non_parallelism,
            ntheta=None, npsi=None,
            include_summit=False,
        )

        # Compute
        xi, xj, strict = _comp_optics.calc_xixj_from_braggphi(
            det_cent=det['cent'],
            det_nout=det['nout'], det_ei=det['ei'], det_ej=det['ej'],
            det_outline=det.get('outline'),
            summit=summit, nout=nout, e1=e1, e2=e2,
            bragg=bragg, phi=phi, strict=strict,
        )

        if plot:
            dax = _plot_optics.CrystalBragg_plot_approx_detector_params(
                bragg, xi, xj, data, dax,
            )
        if return_strict is True:
            return xi, xj, strict
        else:
            return xi, xj

    def plot_line_on_det_tracing(
        self, lamb=None, n=None,
        nphi=None,
        det=None, johann=None,
        use_non_parallelism=None,
        lpsi=None, ldtheta=None,
        strict=None,
        ax=None, dleg=None,
        rocking=None, fs=None, dmargin=None,
        wintit=None, tit=None,
    ):
        """ Visualize the de-focusing by ray-tracing of chosen lamb
        Possibility to plot few wavelength' arcs on the same plot.
        Args:
            - lamb: array of min size 1, in 1e-10 [m]
            - det: dict
            - xi_bounds: np.min & np.max of _XI
            - xj_bounds: np.min & np.max of _XJ
            (from "inputs_temp/XICS_allshots_C34.py" l.649)
            - johann: True or False
        """
        # Check / format inputs
        if lamb is None:
            lamb = self._dbragg['lambref']
        lamb = np.atleast_1d(lamb).ravel()
        nlamb = lamb.size

        if johann is None:
            johann = lpsi is not None or ldtheta is not None
        if rocking is None:
            rocking = False

        if det is None or det.get('outline') is None:
            msg = ("Please provide det as a dict with 'outline'!")
            raise Exception(msg)

        # Get local basis
        nout, e1, e2, use_non_parallelism = self.get_unit_vectors(
            use_non_parallelism=use_non_parallelism,
        )
        nin = -nout

        # Compute lamb / phi
        _, phi = self.get_lambbraggphi_from_ptsxixj_dthetapsi(
            xi=det['outline'][0, :], xj=det['outline'][1, :], det=det,
            dtheta=0, psi=0,
            use_non_parallelism=use_non_parallelism,
            n=n,
            grid=True,
            return_lamb=False,
        )
        phimin, phimax = np.nanmin(phi), np.nanmax(phi)
        phimin, phimax = phimin-(phimax-phimin)/10, phimax+(phimax-phimin)/10

        # Get reference ray-tracing
        bragg = self._checkformat_bragglamb(lamb=lamb, n=n)
        if nphi is None:
            nphi = 100
        phi = np.linspace(phimin, phimax, nphi)

        xi = np.full((nlamb, nphi), np.nan)
        xj = np.full((nlamb, nphi), np.nan)
        for ll in range(nlamb):
            xi[ll, :], xj[ll, :] = self.calc_xixj_from_braggphi(
                bragg=np.full(phi.shape, bragg[ll]),
                phi=phi,
                dtheta=0.,
                psi=0.,
                n=n,
                det=det,
                use_non_parallelism=use_non_parallelism,
                strict=strict,
                plot=False,
            )

        # Get johann-error raytracing (multiple positions on crystal)
        xi_er, xj_er = None, None
        if johann and not rocking:
            if lpsi is None:
                lpsi = np.linspace(-1., 1., 15)
            if ldtheta is None:
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
                        det=det, plot=False,
                        use_non_parallelism=use_non_parallelism,
                        strict=strict,
                    )

        # Get rocking curve error
        if rocking:
            pass

        # Plot
        return _plot_optics.CrystalBragg_plot_line_tracing_on_det(
            lamb, xi, xj, xi_er, xj_er,
            det=det, ax=ax, dleg=dleg,
            johann=johann, rocking=rocking,
            fs=fs, dmargin=dmargin, wintit=wintit, tit=tit)

    def calc_johannerror(
        self,
        xi=None, xj=None, err=None,
        det=None, n=None,
        lpsi=None, ldtheta=None,
        lambda_interval_min=None,
        lambda_interval_max=None,
        use_non_parallelism=None,
        plot=True, fs=None, cmap=None,
        vmin=None, vmax=None, tit=None, wintit=None,
    ):
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

        First affecting a reference lambda according to:
            - pixel's position
            - crystal's summit
        Then, computing error on bragg and phi angles on each pixels by
        computing lambda and phi from the crystal's outline
        Provide lambda_interval_min/max to ensure the given wavelength interval
        is detected over the whole surface area.
        A True/False boolean is then returned.
        """

        # Check xi, xj once before to avoid doing it twice
        if err is None:
            err = 'abs'
        if lambda_interval_min is None:
            lambda_interval_min = 3.93e-10
        if lambda_interval_max is None:
            lambda_interval_max = 4.00e-10

        xi, xj, (xii, xjj) = _comp_optics._checkformat_xixj(xi, xj)

        # Check / format inputs
        bragg, phi, lamb = self.get_lambbraggphi_from_ptsxixj_dthetapsi(
            xi=xii, xj=xjj, det=det,
            dtheta=0, psi=0,
            use_non_parallelism=use_non_parallelism,
            n=n,
            grid=True,
            return_lamb=True,
        )

        # Only one summit was selected
        bragg, phi, lamb = bragg[..., 0], phi[..., 0], lamb[..., 0]

        # Check lambda interval into lamb array
        c0 = (
            np.min(lamb) < lambda_interval_min
            and np.max(lamb) > lambda_interval_max
        )
        if c0:
            test_lambda_interv = True
        else:
            test_lambda_interv = False

        # Get err from multiple ldtheta, lpsi
        if lpsi is None:
            lpsi = np.r_[-1., 0., 1., 1., 1., 0., -1, -1]
        lpsi = self._dgeom['extenthalf'][0]*np.r_[lpsi]
        if ldtheta is None:
            ldtheta = np.r_[-1., -1., -1., 0., 1., 1., 1., 0.]
        ldtheta = self._dgeom['extenthalf'][1]*np.r_[ldtheta]
        npsi = lpsi.size
        assert npsi == ldtheta.size

        (
            braggerr, phierr, lamberr,
        ) = self.get_lambbraggphi_from_ptsxixj_dthetapsi(
            xi=xii, xj=xjj, det=det,
            dtheta=ldtheta, psi=lpsi,
            use_non_parallelism=use_non_parallelism,
            n=n,
            grid=True,
            return_lamb=True,
        )
        err_lamb = np.nanmax(np.abs(lamb[..., None] - lamberr), axis=-1)
        err_phi = np.nanmax(np.abs(phi[..., None] - phierr), axis=-1)

        # absolute vs relative error
        if 'rel' in err:
            if err == 'rel':
                err_lamb = 100.*err_lamb / (np.nanmax(lamb) - np.nanmin(lamb))
                err_phi = 100.*err_phi / (np.nanmax(phi) - np.nanmin(phi))
            elif err == 'rel2':
                err_lamb = 100.*err_lamb / np.mean(lamb)
                err_phi = 100.*err_phi / np.mean(phi)
            err_lamb_units = '%'
            err_phi_units = '%'
        else:
            err_lamb_units = 'm'
            err_phi_units = 'rad'

        if plot is True:
            ax = _plot_optics.CrystalBragg_plot_johannerror(
                xi, xj, lamb, phi,
                err_lamb, err_phi,
                err_lamb_units=err_lamb_units,
                err_phi_units=err_phi_units,
                cmap=cmap, vmin=vmin, vmax=vmax,
                fs=fs, tit=tit, wintit=wintit,
                )
        return (
            err_lamb, err_phi, err_lamb_units, err_phi_units,
            test_lambda_interv,
        )

    def plot_focal_error_summed(
        self,
        dist_min=None, dist_max=None,
        di_min=None, di_max=None,
        ndist=None, ndi=None,
        lamb=None, bragg=None,
        xi=None, xj=None,
        err=None,
        use_non_parallelism=None,
        tangent_to_rowland=None, n=None,
        plot=None,
        pts=None,
        det_ref=None, plot_dets=None, nsort=None,
        dcryst=None,
        lambda_interval_min=None,
        lambda_interval_max=None,
        contour=None,
        fs=None,
        ax=None,
        cmap=None,
        vmin=None,
        vmax=None,
        return_ax=None,
    ):
        """
        Using the calc_johannerror method, computing the sum of the
        focalization error over the whole detector for different positions
        characterized by the translations ddist and di in the equatorial plane
        (dist_min, dist_max, ndist) (di_min, di_max, ndi).

        Parameters:
        -----------
        - lamb/bragg :  float
            Automatically set to crystal's references
        - xi, xj :  np.ndarray
            pixelization of the detector
            (from "inputs_temp/XICS_allshots_C34.py" l.649)
        - alpha, beta : float
            Values of Non Parallelism references angles
        - use_non_parallelism : str
        - tangent_to_rowland :  str
        - plot_dets : str
            Possibility to plot the nsort- detectors with the lowest
            summed focalization error, next to the Best Approximate Real
            detector
            dict(np.load('det37_CTVD_incC4_New.npz', allow_pickle=True))
        - nsort : float
            Number of best detector's position to plot
        - lambda_interv_min/max : float
            To ensure the given wavelength interval is detected over the whole
            surface area. A True/False boolean is then returned.
        """

        # Check / format inputs
        if dist_min is None:
            dist_min = -0.15
        if dist_max is None:
            dist_max = 0.15
        if di_min is None:
            di_min = -0.40
        if di_max is None:
            di_max = 0.40
        if ndist is None:
            ndist = 21
        if ndi is None:
            ndi = 21
        if err is None:
            err = 'rel'
        if plot is None:
            plot = True
        if plot_dets is None:
            plot_dets = det_ref is not None
        if nsort is None:
            nsort = 5
        if return_ax is None:
            return_ax = True
        if lambda_interval_min is None:
            lambda_interval_min = 3.93e-10
        if lambda_interval_max is None:
            lambda_interval_max = 4.00e-10

        l0 = [dist_min, dist_max, ndist, di_min, di_max, ndi]
        c0 = any([l00 is not None for l00 in l0])
        if not c0:
            msg = (
                "Please give the ranges of ddist and di translations\n"
                "\t to compute the different detector's position\n"
                "\t Provided:\n"
                "\t\t- dist_min, dist_max, ndist: ({}, {}, {})\n".format(
                    dist_min, dist_max, ndist,
                )
                + "\t\t- di_min, di_max, ndi: ({}, {}, {})\n".format(
                    di_min, di_max, ndi,
                )
            )
            raise Exception(msg)

        # ------------
        # Compute local coordinates of det_ref
        (
            ddist0, di0, dj0,
            dtheta0, dpsi0, tilt0,
        ) = self._get_local_coordinates_of_det(
            bragg=bragg,
            lamb=lamb,
            det_ref=det_ref,
            use_non_parallelism=use_non_parallelism,
        )

        # angle between nout vectors from get_det_approx() &
        ## get_det_approx(tangent=False)

        det1 = self.get_detector_approx(
            lamb=lamb,
            bragg=bragg,
            use_non_parallelism=use_non_parallelism,
            tangent_to_rowland=True,
        )
        det2 = self.get_detector_approx(
            lamb=lamb,
            bragg=bragg,
            use_non_parallelism=use_non_parallelism,
            tangent_to_rowland=False,
        )
        cos_angle_nout = np.sum(
            det1['nout'] * det2['nout']
            ) / (
            np.linalg.norm(det1['nout'] * np.linalg.norm(det2['nout']))
            )
        angle_nout = np.arccos(cos_angle_nout)

        # Compute
        ddist = np.linspace(dist_min, dist_max, int(ndist))
        di = np.linspace(di_min, di_max, int(ndi))
        error_lambda = np.full((di.size, ddist.size), np.nan)
        test_lamb_interv = np.zeros((di.size, ddist.size), dtype='bool')
        end = '\r'
        for ii in range(ddist.size):
            for jj in range(di.size):

                # print progression
                if ii == ndist-1 and jj == ndi-1:
                    end = '\n'
                msg = (
                    "Computing mean focal error for det "
                    f"({ii+1}, {jj+1})/({ndist}, {ndi})"
                ).ljust(60)
                print(msg, end=end, flush=True)

                # Get det
                dpsi0bis = float(dpsi0)
                if tangent_to_rowland:
                    dpsi0bis = dpsi0 - angle_nout

                det = self.get_detector_approx(
                    ddist=ddist[ii],
                    di=di[jj],
                    dj=dj0,
                    dtheta=dtheta0,
                    dpsi=dpsi0bis,
                    tilt=tilt0,
                    lamb=lamb,
                    bragg=bragg,
                    use_non_parallelism=use_non_parallelism,
                    tangent_to_rowland=False,
                )

                # Integrate error
                (
                    error_lambda_temp, test_lamb_interv[jj, ii],
                ) = self.calc_johannerror(
                    xi=xi, xj=xj,
                    det=det,
                    err=err,
                    lambda_interval_min=lambda_interval_min,
                    lambda_interval_max=lambda_interval_max,
                    plot=False,
                )[::4]
                error_lambda[jj, ii] = np.nanmean(error_lambda_temp)

        if 'rel' in err:
            units = '%'
        else:
            units = 'm'

        if plot:
            ax = _plot_optics.CrystalBragg_plot_focal_error_summed(
                cryst=self, dcryst=dcryst,
                lamb=lamb, bragg=bragg,
                error_lambda=error_lambda,
                ddist=ddist, di=di,
                ddist0=ddist0, di0=di0, dj0=dj0,
                dtheta0=dtheta0, dpsi0=dpsi0, tilt0=tilt0,
                angle_nout=angle_nout,
                det_ref=det_ref,
                units=units,
                plot_dets=plot_dets, nsort=nsort,
                tangent_to_rowland=tangent_to_rowland,
                use_non_parallelism=use_non_parallelism,
                pts=pts,
                test_lamb_interv=test_lamb_interv,
                contour=contour,
                fs=fs,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
        if return_ax:
            return error_lambda, ddist, di, test_lamb_interv, ax
        else:
            return error_lambda, ddist, di, test_lamb_interv

    def _get_local_coordinates_of_det(
        self,
        bragg=None,
        lamb=None,
        det_ref=None,
        use_non_parallelism=None,
    ):
        """
        Computation of translation (ddist, di, dj) and angular
        (dtheta, dpsi, tilt) properties of an arbitrary detector choosen by
        the user.
        """

        # ------------
        # check inputs

        if det_ref is None:
            msg = (
                "You need to provide your arbitrary detector\n"
                + "\t in order to compute its spatial properties !\n"
                + "\t You provided: {}".format(det)
            )
            raise Exception(msg)

        # Checkformat det
        det_ref = self._checkformat_det(det=det_ref)

        # ------------
        # get approx detect

        det_approx = self.get_detector_approx(
            bragg=bragg, lamb=lamb,
            tangent_to_rowland=False,
            use_non_parallelism=use_non_parallelism,
        )

        # ------------
        # get vector delta between centers

        delta = det_ref['cent'] - det_approx['cent']
        ddist = np.sum(delta * (-det_approx['nout']))
        di = np.sum(delta * det_approx['ei'])
        dj = np.sum(delta * det_approx['ej'])

        # ---------------
        # get angles from unit vectors
        dtheta, dpsi, tilt = None, None, None

        # use formulas in _comp_optics.get_det_abs_from_rel() 
        sindtheta = np.sum(det_approx['ej'] * det_ref['nout'])
        costheta_cospsi = np.sum(det_approx['nout'] * det_ref['nout'])
        costheta_sinpsi = np.sum(det_approx['ei'] * det_ref['nout'])
        costheta = np.sqrt(costheta_cospsi**2 + costheta_sinpsi**2)
        dtheta = np.arctan2(sindtheta, costheta)
        dpsi = np.arctan2(
            costheta_sinpsi / costheta,
            costheta_cospsi / costheta,
        )

        # ---------
        # tilt
        det_ei2 = (
            np.cos(dpsi)*det_approx['ei'] - np.sin(dpsi)*det_approx['nout']
        )
        det_ej2 = np.cross(det_ref['nout'], det_ei2)
        costilt = np.sum(det_ref['ei']*det_ei2)
        sintilt = np.sum(det_ref['ei']*det_ej2)
        tilt = np.arctan2(sintilt, costilt)

        return ddist, di, dj, dtheta, dpsi, tilt

    def get_lambbraggphi_from_ptsxixj_dthetapsi(
        self,
        pts=None,
        xi=None, xj=None, det=None,
        dtheta=None, psi=None,
        ntheta=None, npsi=None,
        n=None,
        use_non_parallelism=None,
        grid=None,
        return_lamb=None,
    ):
        """ Return the lamb, bragg and phi for provided pts and dtheta/psi

        if grid = True:
            compute all pts / dtheta/psi comnbinations
              => return (npts, ndtheta) arrays
        else:
            each pts is associated to a single dtheta/psi
                => assumes npts == ndtheta == npsi
                => return (npts,) arrays

        """

        # Check / Format inputs
        if return_lamb is None:
            return_lamb = True
        det = self._checkformat_det(det)

        # Get local basis
        summ, vout, ve1, ve2 = self.get_local_noute1e2(
            dtheta=dtheta, psi=psi,
            ntheta=ntheta, npsi=npsi,
            use_non_parallelism=use_non_parallelism,
            include_summit=True,
        )

        # Derive bragg, phi
        bragg, phi = _comp_optics.calc_braggphi_from_xixjpts(
            pts=pts,
            xi=xi, xj=xj, det=det,
            summit=summ, nin=-vout, e1=ve1, e2=ve2,
            grid=grid,
        )

        # Derive lamb
        if return_lamb is True:
            lamb = self.get_lamb_from_bragg(bragg=bragg, n=n)
            return bragg, phi, lamb
        else:
            return bragg, phi

    def get_lamb_avail_from_pts(
        self,
        pts=None,
        n=None, ndtheta=None, npsi=None,
        det=None, nlamb=None, klamb=None,
        use_non_parallelism=None,
        strict=None,
        return_xixj=None,
    ):
        """ Return the wavelength accessible from plasma points on the crystal

        For a given plasma point, only a certain lambda interval can be
        bragg-diffracted on the crystal (due to bragg's law and the crystal's
        dimensions)

        Beware, for a given pts and lamb, there can be up to 2 sets of
        solutions
        All non-valid solutions are set to nans, such that most of the time
        there is only one

        For a set of given:
            - pts (3, npts) array, (x, y, z) coordinates
        Using:
            - nlamb: sampling of the lamb interval (default: 100)
            - ndtheta: sampling of the lamb interval (default: 20)
            - npsi: sampling of the lamb interval (default: 'envelop')
            - det: (optional) a detector dict, for xi and xj
        Returns:
            - lamb: (npts, nlamb) array of sampled valid wavelength interval
            - phi:  (npts, nlamb, ndtheta, npsi, 2) array of phi
            - dtheta:  (npts, nlamb, ndtheta, npsi, 2) array of dtheta
            - psi:  (npts, nlamb, ndtheta, npsi, 2) array of psi
        And optionally (return_xixj=True and det provided as dict):
            - xi:  (npts, nlamb, ndtheta, npsi, 2) array of xi
            - xj:  (npts, nlamb, ndtheta, npsi, 2) array of xj

        The result is computed with or w/o taking into account non-parallelism

        """
        # Check / format
        if ndtheta is None:
            ndtheta = 20
        if nlamb is None:
            nlamb = 100
        assert nlamb >= 2, "nlamb must be >= 2"
        if return_xixj is None:
            return_xixj = det is not None

        # Get lamb min / max
        bragg, phi, lamb = self.get_lambbraggphi_from_ptsxixj_dthetapsi(
            pts=pts,
            dtheta='envelop', psi='envelop',
            ntheta=None, npsi=None,
            n=n, grid=True,
            use_non_parallelism=use_non_parallelism,
            return_lamb=True,
        )
        lambmin = np.nanmin(lamb, axis=1)
        lambmax = np.nanmax(lamb, axis=1)
        if klamb is None:
            klamb = np.linspace(0, 1, nlamb)
        elif not (isinstance(klamb, np.ndarray) and klamb.ndim == 1):
            msg = "Please provide klamb as a 1d vector!"
            raise Exception(msg)
        nlamb = klamb.size
        lamb = lambmin[:, None] + (lambmax-lambmin)[:, None]*klamb
        bragg = self._checkformat_bragglamb(lamb=lamb, n=n)

        # Compute dtheta / psi for each pts and lamb
        npts = lamb.shape[0]
        dtheta = np.full((npts, nlamb, ndtheta, 2), np.nan)
        psi = np.full((npts, nlamb, ndtheta, 2), np.nan)
        phi = np.full((npts, nlamb, ndtheta, 2), np.nan)
        for ii in range(nlamb):
            (
                dtheta[:, ii, :, :], psi[:, ii, :, :],
                phi[:, ii, :, :],
            ) = self._calc_dthetapsiphi_from_lambpts(
                pts=pts, bragg=bragg[:, ii], lamb=None,
                n=n, ndtheta=ndtheta,
                use_non_parallelism=use_non_parallelism,
                grid=False,
            )[:3]

        if return_xixj is True and det is not None:
            xi, xj, strict = self.calc_xixj_from_braggphi(
                phi=phi+np.pi,
                bragg=bragg[..., None, None],
                n=n,
                dtheta=dtheta,
                psi=psi,
                det=det,
                data=None,
                use_non_parallelism=use_non_parallelism,
                strict=strict,
                return_strict=True,
                plot=False,
                dax=None,
            )
            if strict is True and np.any(np.isnan(xi)):
                indnan = np.isnan(xi)
                phi[indnan] = np.nan
                dtheta[indnan] = np.nan
                psi[indnan] = np.nan
                lamb[np.all(np.all(indnan, axis=-1), axis=-1)] = np.nan
            return lamb, phi, dtheta, psi, xi, xj
        else:
            return lamb, phi, dtheta, psi

    def _calc_dthetapsiphi_from_lambpts(
        self,
        pts=None, bragg=None, lamb=None,
        n=None, ndtheta=None,
        use_non_parallelism=None,
        grid=None,
    ):

        # Check / Format inputs
        pts = _comp_optics._checkformat_pts(pts)
        npts = pts.shape[1]

        bragg = self._checkformat_bragglamb(bragg=bragg, lamb=lamb, n=n)

        # get nout, e1, e2
        nout, e1, e2, use_non_parallelism = self.get_unit_vectors(
            use_non_parallelism=use_non_parallelism
            )

        # Compute dtheta, psi, indnan (nlamb, npts, ndtheta)
        # In general there are 2 solutions! (only close to rowland in practice)
        dtheta, psi, indok, grid = _comp_optics.calc_dthetapsiphi_from_lambpts(
            pts,
            bragg,
            summit=self._dgeom['summit'],   # To be updated (non-paralellism)?
            rcurve=self._dgeom['rcurve'],
            nout=nout, e1=e1, e2=e2,
            extenthalf=self._dgeom['extenthalf'],
            ndtheta=ndtheta,
            grid=grid,
        )

        # reshape bragg for matching dtheta.shape
        if grid is True:
            bragg = np.repeat(
                np.repeat(
                    np.repeat(bragg[:, None], npts, axis=-1)[..., None],
                    dtheta.shape[2],
                    axis=-1,
                )[..., None],
                2,
                axis=-1,
            )
            pts = pts[:, None, :, None, None]
        else:
            bragg = np.repeat(
                np.repeat(bragg[:, None], dtheta.shape[1], axis=1)[..., None],
                2,
                axis=-1,
            )
            pts = pts[..., None, None]
        bragg[~indok] = np.nan

        # Get corresponding phi and re-check bragg, for safety
        bragg2, phi = self.get_lambbraggphi_from_ptsxixj_dthetapsi(
            pts=pts,
            dtheta=dtheta, psi=psi,
            grid=False,
            use_non_parallelism=use_non_parallelism,
            return_lamb=False,
        )

        c0 = (
            bragg2.shape == bragg.shape
            and np.allclose(bragg, bragg2, equal_nan=True)
        )
        if not c0:
            try:
                plt.figure()
                plt.plot(bragg, bragg2, '.')
            except Exception as err:
                pass
            msg = (
                "Inconsistency detected in bragg angle computations:\n"
                + "\t- from the points and lamb\n"
                + "\t- from the points and (dtheta, psi)\n"
                + "\nContext:\n"
                + "\t- use_non_parallelism: {}\n".format(use_non_parallelism)
                + "\t- bragg.shape = {}\n".format(bragg.shape)
                + "\t- bragg2.shape = {}\n".format(bragg2.shape)
            )
            raise Exception(msg)

        return dtheta, psi, phi, bragg

    def calc_raytracing_from_lambpts(
        self,
        lamb=None, bragg=None, pts=None,
        xi_bounds=None, xj_bounds=None, nphi=None,
        det=None, n=None, ndtheta=None,
        johann=False, lpsi=None, ldtheta=None,
        rocking=False, strict=None, plot=None, fs=None,
        dmargin=None, wintit=None,
        tit=None, proj=None,
        legend=None, draw=None, returnas=None,
    ):
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
        if plot is None or plot is True:
            plot = ['det', '3d']
        if isinstance(plot, str):
            plot = plot.split('+')
            assert all([ss in ['det', '2d', '3d'] for ss in plot])
        assert returnas in ['data', 'ax']

        pts = _comp_optics._checkformat_pts(pts)
        npts = pts.shape[1]

        # Get dtheta, psi and phi from pts/lamb
        dtheta, psi, phi, bragg = self._calc_dthetapsiphi_from_lambpts(
            pts=pts, lamb=lamb, bragg=bragg, n=n, ndtheta=ndtheta,
        )
        ndtheta = dtheta.shape[-1]
        # assert dtheta.shape == (nlamb, npts, ndtheta)

        # Check / get det
        det = self._checkformat_det(det)

        # Compute xi, xj of reflexion (phi -> phi + np.pi)
        xi, xj = self.calc_xixj_from_braggphi(
            bragg=bragg, phi=phi+np.pi, n=n,
            dtheta=dtheta, psi=psi,
            det=det, strict=strict, plot=False,
        )

        # Plot to be checked
        plot = False
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
        return dtheta, psi, phi, bragg, xi, xj

    def _calc_spect1d_from_data2d(self, data, lamb, phi,
                                  nlambfit=None, nphifit=None,
                                  nxi=None, nxj=None,
                                  spect1d=None, mask=None, vertsum1d=None):
        if nlambfit is None:
            nlambfit = nxi
        if nphifit is None:
            nphifit = nxj
        return _comp_optics._calc_spect1d_from_data2d(
            data, lamb, phi,
            nlambfit=nlambfit,
            nphifit=nphifit,
            spect1d=spect1d,
            mask=mask,
            vertsum1d=vertsum1d,
        )

    def plot_data_vs_lambphi(
        self,
        xi=None, xj=None, data=None, mask=None,
        det=None, dtheta=None, psi=None, n=None,
        nlambfit=None, nphifit=None,
        magaxis=None, npaxis=None,
        dlines=None, spect1d='mean',
        lambmin=None, lambmax=None,
        xjcut=None, dxj=None,
        plot=True, fs=None, tit=None, wintit=None,
        cmap=None, vmin=None, vmax=None,
        returnas=None,
    ):
        # Check / format inputs
        assert data is not None
        if returnas is None:
            returnas = 'spect'
        lreturn = ['ax', 'spect']
        if returnas not in lreturn:
            msg = ("Arg returnas must be in {}\n:".format(lreturn)
                   + "\t- 'spect': return a 1d vertically averaged spectrum\n"
                   + "\t- 'ax'   : return a list of axes instances")
            raise Exception(msg)

        xi, xj, (xii, xjj) = _comp_optics._checkformat_xixj(xi, xj)
        nxi = xi.size if xi is not None else np.unique(xii).size
        nxj = xj.size if xj is not None else np.unique(xjj).size

        # Compute lamb / phi
        bragg, phi, lamb = self.get_lambbraggphi_from_ptsxixj_dthetapsi(
            xi=xii, xj=xjj, det=det,
            dtheta=dtheta, psi=psi,
            use_non_parallelism=use_non_parallelism,
            n=n,
            grid=True,
            return_lamb=True,
        )

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
            braggax, phiax = self.calc_braggphi_from_pts(pts)
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
            (
                braggcut, phicut, lambcut,
            ) = self.get_lambbraggphi_from_ptsxixj_dthetapsi(
                xi=xicutf, xj=xjcutf, det=det,
                dtheta=0, psi=0,
                use_non_parallelism=use_non_parallelism,
                n=1,
                grid=True,
                return_lamb=True,
            )
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

    @staticmethod
    def fit1d_dinput(
        dlines=None, dconstraints=None, dprepare=None,
        data=None, lamb=None,
        mask=None, domain=None, pos=None, subset=None,
        same_spectrum=None, same_spectrum_dlamb=None,
        focus=None, valid_fraction=None, valid_nsigma=None,
        focus_half_width=None, valid_return_fract=None,
    ):
        """ Return a formatted dict of lines and constraints

        To be fed to _fit12d.multigausfit1d_from_dlines()
        Provides a user-friendly way of defining constraints
        """

        import tofu.spectro._fit12d as _fit12d
        return _fit12d.fit1d_dinput(
            dlines=dlines, dconstraints=dconstraints, dprepare=dprepare,
            data=data, lamb=lamb,
            mask=mask, domain=domain, pos=pos, subset=subset,
            same_spectrum=same_spectrum,
            same_spectrum_dlamb=same_spectrum_dlamb,
            focus=focus, valid_fraction=valid_fraction,
            valid_nsigma=valid_nsigma, focus_half_width=focus_half_width,
            valid_return_fract=valid_return_fract)

    def fit1d(
        self,
        # Input data kwdargs
        data=None, lamb=None,
        dinput=None, dprepare=None, dlines=None, dconstraints=None,
        mask=None, domain=None, subset=None, pos=None,
        same_spectrum=None, same_spectrum_dlamb=None,
        focus=None, valid_fraction=None, valid_nsigma=None,
        focus_half_width=None,
        # Optimization kwdargs
        dx0=None, dscales=None, x0_scale=None, bounds_scale=None,
        method=None, tr_solver=None, tr_options=None, max_nfev=None,
        xtol=None, ftol=None, gtol=None,
        loss=None, verbose=None, chain=None, jac=None, showonly=None,
        # Results extraction kwdargs
        amp=None, coefs=None, ratio=None,
        Ti=None, width=None, vi=None, shift=None,
        pts_lamb_total=None, pts_lamb_detail=None,
        # Saving and plotting kwdargs
        save=None, name=None, path=None,
        plot=None, fs=None, dmargin=None,
        tit=None, wintit=None, returnas=None,
    ):

        # ----------------------
        # Get dinput for 1d fitting from dlines, dconstraints, dprepare...
        if dinput is None:
            dinput = self.fit1d_dinput(
                dlines=dlines, dconstraints=dconstraints, dprepare=dprepare,
                data=data, lamb=lamb,
                mask=mask, domain=domain, pos=pos, subset=subset,
                focus=focus, valid_fraction=valid_fraction,
                valid_nsigma=valid_nsigma, focus_half_width=focus_half_width,
                same_spectrum=same_spectrum,
                same_spectrum_dlamb=same_spectrum_dlamb)

        # ----------------------
        # return
        import tofu.spectro._fit12d as _fit12d
        return _fit12d.fit1d(
            # Input data kwdargs
            data=data, lamb=lamb,
            dinput=dinput, dprepare=dprepare,
            dlines=dlines, dconstraints=dconstraints,
            mask=mask, domain=domain, subset=subset, pos=pos,
            # Optimization kwdargs
            method=method, tr_solver=tr_solver, tr_options=tr_options,
            xtol=xtol, ftol=ftol, gtol=gtol,
            max_nfev=max_nfev, loss=loss, chain=chain,
            dx0=dx0, x0_scale=x0_scale, bounds_scale=bounds_scale,
            jac=jac, verbose=verbose,
            save=save, name=name, path=path,
            amp=amp, coefs=coefs, ratio=ratio,
            Ti=Ti, width=width, vi=vi, shift=shift,
            pts_lamb_total=pts_lamb_total,
            pts_lamb_detail=pts_lamb_detail,
            plot=plot, fs=fs, wintit=wintit, tit=tit)

    @staticmethod
    def fit1d_extract(
        dfit1d=None,
        amp=None, coefs=None, ratio=None,
        Ti=None, width=None,
        vi=None, shift=None,
        pts_lamb_total=None, pts_lamb_detail=None,
    ):
        import tofu.spectro._fit12d as _fit12d
        return _fit12d.fit1d_extract(
            dfit1d=dfit,
            amp=amp, coefs=coefs, ratio=ratio,
            Ti=Ti, width=width,
            vi=vi, shift=shift,
            pts_lamb_total=pts_lamb_total, pts_lamb_detail=pts_lamb_detail)

    def fit1d_from2d(self):
        """ Useful for optimizing detector or crystal position

        Given a set of 2d images on a detector
        Transform the 2d (xi, xj) image into (lamb, phi)
        Slice nphi 1d spectra
        Fit them using a dict of reference lines (dlines)
        Optionally provide constraints for the fitting
        Return the vertical profiles of the wavelength shitf of each line
        To be used as input for an cost function and optimization

        1d fitting is used instead of 2d because:
            - faster (for optimization)
            - does not require a choice of nbsplines
            - easier to understand and decide for user

        """
        # Check / format inputs
        if lphi is None:
            msg = ("Arg lphi must be provided !")
            raise Exception(msg)

        # ----------------------
        # Prepare input data
        # (geometrical transform, domain, binning, subset, noise...)
        if dprepare is None:
            dprepare = self.fit2d_prepare(
                data=data, xi=xi, xj=xj, n=n,
                det=det, dtheta=dtheta, psi=psi,
                mask=mask, domain=domain,
                pos=pos, binning=binning,
                nbsplines=False, subset=False,
                lphi=lphi, lphi_tol=lphi_tol)

        # ----------------------
        # Get dinput for 2d fitting from dlines, and dconstraints
        if dinput is None:
            dinput = self.fit2d_dinput(
                dlines=dlines, dconstraints=dconstraints,
                deg=deg, knots=knots, nbsplines=nbsplines,
                domain=dprepare['domain'],
                dataphi1d=dprepare['dataphi1d'], phi1d=dprepare['phi1d'])

        # ----------------------
        # fit
        out = self.fit1d(
            xi=None, xj=None, data=None, mask=None,
            det=None, dtheta=None, psi=None, n=None,
            nlambfit=None, nphifit=None,
            lambmin=None, lambmax=None,
            dlines=None, spect1d=None,
            dconstraints=None, dx0=None,
            same_spectrum=None, dlamb=None,
            double=None,
            dscales=None, x0_scale=None, bounds_scale=None,
            method=None, max_nfev=None,
            xtol=None, ftol=None, gtol=None,
            loss=None, verbose=0, chain=None,
            jac=None, showonly=None,
            plot=None, fs=None, dmargin=None,
            tit=None, wintit=None, returnas=None,
        )
        pass

    def fit2d_dinput(
        self, dlines=None, dconstraints=None, dprepare=None,
        data=None, xi=None, xj=None, n=None,
        det=None, dtheta=None, psi=None,
        mask=None, domain=None, pos=None, binning=None, subset=None,
        # lphi=None, lphi_tol=None,
        deg=None, knots=None, nbsplines=None,
        focus=None, valid_fraction=None, valid_nsigma=None,
        focus_half_width=None, valid_return_fract=None,
    ):
        """ Return a formatted dict of lines and constraints

        To be fed to _fit12d.multigausfit1d_from_dlines()
        Provides a user-friendly way of defining constraints
        """

        import tofu.spectro._fit12d as _fit12d
        if dprepare is None:
            # ----------------------
            # Geometrical transform
            xi, xj, (xii, xjj) = _comp_optics._checkformat_xixj(xi, xj)
            nxi = xi.size if xi is not None else np.unique(xii).size
            nxj = xj.size if xj is not None else np.unique(xjj).size

            # Compute lamb / phi
            bragg, phi, lamb = self.get_lambbraggphi_from_ptsxixj_dthetapsi(
                xi=xii, xj=xjj, det=det,
                dtheta=dtheta, psi=psi,
                use_non_parallelism=use_non_parallelism,
                n=n,
                grid=True,
                return_lamb=True,
            )

            # ----------------------
            # Prepare input data (domain, binning, subset, noise...)
            dprepare = _fit12d.multigausfit2d_from_dlines_prepare(
                data, lamb, phi,
                mask=mask, domain=domain,
                pos=pos, binning=binning,
                nbsplines=nbsplines, subset=subset,
                nxi=nxi, nxj=nxj,
            )   # , lphi=lphi, lphi_tol=lphi_tol)
        return _fit12d.fit2d_dinput(
            dlines=dlines, dconstraints=dconstraints, dprepare=dprepare,
            deg=deg, knots=knots, nbsplines=nbsplines,
            focus=focus, valid_fraction=valid_fraction,
            valid_nsigma=valid_nsigma, focus_half_width=focus_half_width,
            valid_return_fract=valid_return_fract)

    def fit2d(
        self,
        # Input data kwdargs
        data=None, xi=None, xj=None,
        det=None, dtheta=None, psi=None, n=None,
        dinput=None, dprepare=None, dlines=None, dconstraints=None,
        mask=None, domain=None, subset=None, pos=None, binning=None,
        focus=None, valid_fraction=None, valid_nsigma=None,
        focus_half_width=None,
        deg=None, knots=None, nbsplines=None,
        # Optimization kwdargs
        dx0=None, dscales=None, x0_scale=None, bounds_scale=None,
        method=None, tr_solver=None, tr_options=None, max_nfev=None,
        xtol=None, ftol=None, gtol=None,
        loss=None, verbose=None, chain=None, jac=None, showonly=None,
        predeclare=None, debug=None,
        # Results extraction kwdargs
        amp=None, coefs=None, ratio=None,
        Ti=None, width=None, vi=None, shift=None,
        pts_lamb_total=None, pts_lamb_detail=None,
        # Saving and plotting kwdargs
        save=None, name=None, path=None,
        plot=None, fs=None, dmargin=None,
        tit=None, wintit=None, returnas=None,
    ):

        # npts=None, dax=None,
        # spect1d=None, nlambfit=None,
        # plotmode=None, angunits=None, indspect=None,
        # cmap=None, vmin=None, vmax=None):
        """ Perform 2d fitting of a 2d spectrometre image

        Fit the spectrum by a sum of gaussians
        Modulate each gaussian parameters by bsplines in the spatial direction

        data must be provided in shape (nt, nxi, nxj), where:
            - nt is the number of time steps
            - nxi is the nb. of pixels in the horizontal / spectral direction
            - nxj is the nb. of pixels in the vertical / spacial direction

        """

        # ----------------------
        # Geometrical transform in dprepare
        if dinput is None:
            dinput = self.fit2d_dinput(
                dlines=dlines, dconstraints=dconstraints, dprepare=dprepare,
                data=data, xi=xi, xj=xj, n=n,
                det=det, dtheta=dtheta, psi=psi,
                mask=mask, domain=domain,
                pos=pos, binning=binning, subset=subset,
                deg=deg, knots=knots, nbsplines=nbsplines,
                focus=focus, valid_fraction=valid_fraction,
                valid_nsigma=valid_nsigma, focus_half_width=focus_half_width)

        # ----------------------
        # return
        import tofu.spectro._fit12d as _fit12d
        return _fit12d.fit2d(
            dinput=dinput, dprepare=dprepare,
            dlines=dlines, dconstraints=dconstraints,
            lamb=lamb, phi=phi, data=data, mask=mask,
            nxi=dinput['dprepare']['nxi'], nxj=dinput['dprepare']['nxj'],
            domain=domain, pos=pos, binning=binning, subset=subset,
            deg=deg, knots=knots, nbsplines=nbsplines,
            method=method, tr_solver=tr_solver, tr_options=tr_options,
            xtol=xtol, ftol=ftol, gtol=gtol,
            max_nfev=max_nfev, loss=loss, chain=chain,
            dx0=dx0, x0_scale=x0_scale, bounds_scale=bounds_scale,
            jac=jac, verbose=verbose,
            save=save, name=name, path=path,
            plot=plot)

    @staticmethod
    def fit2d_extract(dfit2d=None,
                      amp=None, Ti=None, vi=None,
                      pts_phi=None, npts_phi=None,
                      pts_lamb_phi_total=None,
                      pts_lamb_phi_detail=None):
        import tofu.spectro._fit12d as _fit12d
        return _fit12d.fit2d_extract_data(
            dfit2d=dfit2d,
            amp=amp, Ti=Ti, vi=vi,
            pts_phi=pts_phi, npts_phi=npts_phi,
            pts_lamb_phi_total=pts_lamb_phi_total,
            pts_lamb_phi_detail=pts_lamb_phi_detail)

    def fit2d_plot(self, dfit2d=None, ratio=None,
                   dax=None, plotmode=None, angunits=None,
                   cmap=None, vmin=None, vmax=None,
                   dmargin=None, tit=None, wintit=None, fs=None):
        dout = self.fit2d_extract(
            dfit2d,
            amp=amp, Ti=Ti, vi=vi,
            pts_lamb_phi_total=pts_lamb_phi_total,
            pts_lamb_phi_detail=pts_lamb_phi_detail)
        return _plot_optics.CrystalBragg_plot_data_fit2d(
            dfit2d=dfit2d, dout=dout, ratio=ratio,
            dax=dax, plotmode=plotmode, angunits=angunits,
            cmap=cmap, vmin=vmin, vmax=vmax,
            dmargin=dmargin, tit=tit, wintit=wintit, fs=fs)

    def noise_analysis(
        self, data=None, xi=None, xj=None, n=None,
        det=None, dtheta=None, psi=None,
        mask=None, valid_fraction=None, nxerrbin=None,
        margin=None, domain=None, nlamb=None,
        deg=None, knots=None, nbsplines=None,
        loss=None, max_nfev=None,
        xtol=None, ftol=None, gtol=None,
        method=None, tr_solver=None, tr_options=None,
        verbose=None, plot=None,
        ms=None, dcolor=None,
        dax=None, fs=None, dmargin=None,
        wintit=None, tit=None, sublab=None,
        save_fig=None, name_fig=None, path_fig=None,
        fmt=None, return_dax=None,
    ):

        # ----------------------
        # Geometrical transform
        bragg, phi, lamb = self.get_lambbraggphi_from_ptsxixj_dthetapsi(
            xi=xi, xj=xj, det=det,
            dtheta=dtheta, psi=psi,
            use_non_parallelism=use_non_parallelism,
            n=n,
            grid=True,
            return_lamb=True,
        )

        import tofu.spectro._fit12d as _fit12d
        return _fit12d.noise_analysis_2d(
            data, lamb, phi,
            mask=mask, valid_fraction=valid_fraction,
            margin=margin, nxerrbin=nxerrbin,
            nlamb=nlamb, deg=deg, knots=knots, nbsplines=nbsplines,
            loss=loss, max_nfev=max_nfev,
            xtol=xtol, ftol=ftol, gtol=gtol,
            method=method, tr_solver=tr_solver, tr_options=tr_options,
            verbose=verbose, plot=plot,
            ms=ms, dcolor=dcolor,
            dax=dax, fs=fs, dmargin=dmargin,
            wintit=wintit, tit=tit, sublab=sublab,
            save_fig=save_fig, name_fig=name_fig, path_fig=path_fig,
            fmt=fmt, return_dax=return_dax)

    @staticmethod
    def noise_analysis_plot(
        dnoise=None, margin=None, valid_fraction=None,
        ms=None, dcolor=None,
        dax=None, fs=None, dmargin=None,
        wintit=None, tit=None, sublab=None,
        save=None, name=None, path=None, fmt=None,
    ):
        import tofu.spectro._plot as _plot_spectro
        return _plot_spectro.plot_noise_analysis(
            dnoise=dnoise, margin=margin, valid_fraction=valid_fraction,
            ms=ms, dcolor=dcolor,
            dax=dax, fs=fs, dmargin=dmargin,
            wintit=wintit, tit=tit, sublab=sublab,
            save=save, name=name, path=path, fmt=fmt)

    def noise_analysis_scannbs(
        self, data=None, xi=None, xj=None, n=None,
        det=None, dtheta=None, psi=None,
        mask=None, nxerrbin=None,
        domain=None, nlamb=None,
        deg=None, knots=None, nbsplines=None, lnbsplines=None,
        loss=None, max_nfev=None,
        xtol=None, ftol=None, gtol=None,
        method=None, tr_solver=None, tr_options=None,
        verbose=None, plot=None,
        ms=None, dax=None, fs=None, dmargin=None,
        wintit=None, tit=None, sublab=None,
        save_fig=None, name_fig=None, path_fig=None,
        fmt=None, return_dax=None,
    ):

        # ----------------------
        # Geometrical transform
        bragg, phi, lamb = self.get_lambbraggphi_from_ptsxixj_dthetapsi(
            xi=xi, xj=xj, det=det,
            dtheta=0, psi=0,
            use_non_parallelism=use_non_parallelism,
            n=n,
            grid=True,
            return_lamb=True,
        )

        import tofu.spectro._fit12d as _fit12d
        return _fit12d.noise_analysis_2d_scannbs(
            data, lamb, phi,
            mask=mask, nxerrbin=nxerrbin, nlamb=nlamb,
            deg=deg, knots=knots, nbsplines=nbsplines, lnbsplines=lnbsplines,
            loss=loss, max_nfev=max_nfev,
            xtol=xtol, ftol=ftol, gtol=gtol,
            method=method, tr_solver=tr_solver, tr_options=tr_options,
            verbose=verbose, plot=plot,
            ms=ms, dax=dax, fs=fs, dmargin=dmargin,
            wintit=wintit, tit=tit, sublab=sublab,
            save_fig=save_fig, name_fig=name_fig, path_fig=path_fig,
            fmt=fmt, return_dax=return_dax)

    @staticmethod
    def noise_analysis_scannbs_plot(
        dnoise_scan=None, ms=None,
        dax=None, fs=None, dmargin=None,
        wintit=None, tit=None, sublab=None,
        save=None, name=None, path=None, fmt=None,
    ):
        import tofu.spectro._plot as _plot_spectro
        return _plot_spectro.plot_noise_analysis_scannbs(
                dnoise=dnoise_scan, ms=ms,
                dax=dax, fs=fs, dmargin=dmargin,
                wintit=wintit, tit=tit, sublab=sublab,
                save=save, name=name, path=path, fmt=fmt)
