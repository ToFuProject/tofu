"""
This module is the geometrical part of the ToFu general package
It includes all functions and object classes necessary for tomography
on Tokamaks
"""


# Built-in
import os
import warnings
import copy
import inspect


# Common
import numpy as np
import scipy.interpolate as scpinterp
import matplotlib as mpl
import matplotlib.pyplot as plt


# ToFu-specific
from tofu import __version__ as __version__
import tofu.pathfile as tfpf
import tofu.utils as utils


# test global import else relative
try:
    import tofu.geom._def as _def
    import tofu.geom._GG as _GG
    import tofu.geom._comp as _comp
    import tofu.geom._comp_solidangles as _comp_solidangles
    import tofu.geom._plot as _plot
except Exception as err0:
    try:
        from . import _def as _def
        from . import _GG as _GG
        from . import _comp as _comp
        from . import _comp_solidangles
        from . import _plot as _plot
    except Exception as err1:
        raise err1 from err0


__all__ = [
    "PlasmaDomain",
    "Ves",
    "PFC",
    "CoilPF",
    "CoilCS",
    "Config",
    "Rays",
    "CamLOS1D",
    "CamLOS2D",
]


_arrayorder = "C"
_Clock = False
_Type = "Tor"

# rotate / translate instance
_UPDATE_EXTENT = True
_RETURN_COPY = False

# Parallelization
_NUM_THREADS = 10
_PHITHETAPROJ_NPHI = 2000
_PHITHETAPROJ_NTHETA = 1000
_RES = 0.005
_DREFLECT = {"specular": 0, "diffusive": 1, "ccube": 2}

# Saving
_COMMENT = '#'


"""
###############################################################################
###############################################################################
                        Ves class and functions
###############################################################################
"""


class Struct(utils.ToFuObject):
    """ A class defining a Linear or Toroidal vaccum vessel (i.e. a 2D polygon
    representing a cross-section and assumed to be linearly or toroidally
    invariant)

    A Ves object is mostly defined by a close 2D polygon, which can be
    understood as a poloidal cross-section in (R,Z) cylindrical coordinates
    if Type='Tor' (toroidal shape) or as a straight cross-section through a
    cylinder in (Y,Z) cartesian coordinates if Type='Lin' (linear shape).
    Attributes such as the surface, the angular volume (if Type='Tor') or the
    center of mass are automatically computed.
    The instance is identified thanks to an attribute Id (which is itself a
    tofu.ID class object) which contains informations on the specific instance
    (name, Type...).

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
        Array or list of len=2 indicating the limits of the linear device
        volume on the x axis
    Sino_RefPt :    None / np.ndarray
        Array specifying a reference point for computing the sinogram (i.e.
        impact parameter), if None automatically set to the (surfacic) center
        of mass of the cross-section
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

    # __metaclass__ = ABCMeta

    # Fixed (class-wise) dictionary of default properties
    _ddef = {
        "Id": {
            "shot": 0,
            "include": [
                "Mod",
                "Cls",
                "Exp",
                "Diag",
                "Name",
                "shot",
                "version",
            ],
        },
        "dgeom": {"Type": "Tor", "Lim": [], "arrayorder": "C"},
        "dsino": {},
        "dphys": {},
        "dreflect": {"Type": "specular"},
        "dmisc": {"color": "k"},
    }
    _dplot = {
        "cross": {
            "Elt": "P",
            "dP": {"color": "k", "lw": 2},
            "dI": {"color": "k", "ls": "--", "marker": "x", "ms": 8, "mew": 2},
            "dBs": {
                "color": "b",
                "ls": "--",
                "marker": "x",
                "ms": 8,
                "mew": 2,
            },
            "dBv": {
                "color": "g",
                "ls": "--",
                "marker": "x",
                "ms": 8,
                "mew": 2,
            },
            "dVect": {"color": "r", "scale": 10},
        },
        "hor": {
            "Elt": "P",
            "dP": {"color": "k", "lw": 2},
            "dI": {"color": "k", "ls": "--"},
            "dBs": {"color": "b", "ls": "--"},
            "dBv": {"color": "g", "ls": "--"},
            "Nstep": 50,
        },
        "3d": {
            "Elt": "P",
            "dP": {
                "color": (0.8, 0.8, 0.8, 1.0),
                "rstride": 1,
                "cstride": 1,
                "linewidth": 0.0,
                "antialiased": False,
            },
            "Lim": None,
            "Nstep": 50,
        },
    }
    _DREFLECT_DTYPES = {"specular": 0, "diffusive": 1, "ccube": 2}

    # Does not exist beofre Python 3.6 !!!
    def __init_subclass__(cls, color="k", **kwdargs):
        # Python 2
        super(Struct, cls).__init_subclass__(**kwdargs)
        # Python 3
        # super().__init_subclass__(**kwdargs)
        cls._ddef = copy.deepcopy(Struct._ddef)
        cls._dplot = copy.deepcopy(Struct._dplot)
        cls._set_color_ddef(cls._color)

    @classmethod
    def _set_color_ddef(cls, color):
        cls._ddef['dmisc']['color'] = mpl.colors.to_rgba(color)

    def __init__(
        self,
        Poly=None,
        Type=None,
        Lim=None,
        pos=None,
        extent=None,
        Id=None,
        Name=None,
        Exp=None,
        shot=None,
        sino_RefPt=None,
        sino_nP=_def.TorNP,
        Clock=False,
        arrayorder='C',
        fromdict=None,
        sep=None,
        SavePath=os.path.abspath('./'),
        SavePath_Include=tfpf.defInclude,
        color=None,
        nturns=None,
        superconducting=None,
        active=None,
        temperature_nominal=None,
        mag_field_max=None,
        current_lim_max=None,
    ):

        # Create a dplot at instance level
        self._dplot = copy.deepcopy(self.__class__._dplot)

        kwdargs = locals()
        del kwdargs["self"]
        # super()
        super(Struct, self).__init__(**kwdargs)

    def _reset(self):
        # super()
        super(Struct, self)._reset()
        self._dgeom = dict.fromkeys(self._get_keys_dgeom())
        self._dsino = dict.fromkeys(self._get_keys_dsino())
        self._dphys = dict.fromkeys(self._get_keys_dphys())
        self._dreflect = dict.fromkeys(self._get_keys_dreflect())
        self._dmisc = dict.fromkeys(self._get_keys_dmisc())
        # self._dplot = copy.deepcopy(self.__class__._ddef['dplot'])

    @classmethod
    def _checkformat_inputs_Id(
        cls,
        Id=None,
        Name=None,
        Exp=None,
        shot=None,
        Type=None,
        include=None,
        **kwdargs
    ):
        if Id is not None:
            assert isinstance(Id, utils.ID)
            Name, Exp, shot, Type = Id.Name, Id.Exp, Id.shot, Id.Type
        if shot is None:
            shot = cls._ddef["Id"]["shot"]
        if Type is None:
            Type = cls._ddef["dgeom"]["Type"]
        if include is None:
            include = cls._ddef["Id"]["include"]

        dins = {
            "Name": {"var": Name, "cls": str},
            "Exp": {"var": Exp, "cls": str},
            "shot": {"var": shot, "cls": int},
            "Type": {"var": Type, "in": ["Tor", "Lin"]},
            "include": {"var": include, "listof": str},
        }
        dins, err, msg = cls._check_InputsGeneric(dins)
        if err:
            raise Exception(msg)

        kwdargs.update(
            {
                "Name": Name,
                "Exp": Exp,
                "shot": shot,
                "Type": Type,
                "include": include,
            }
        )
        return kwdargs

    ###########
    # Get largs
    ###########

    @staticmethod
    def _get_largs_dgeom(sino=True):
        largs = [
            "Poly",
            "Lim",
            "pos",
            "extent",
            "Clock",
            "arrayorder",
        ]
        if sino:
            lsino = Struct._get_largs_dsino()
            largs += ["sino_{0}".format(s) for s in lsino]
        return largs

    @staticmethod
    def _get_largs_dsino():
        largs = ["RefPt", "nP"]
        return largs

    @staticmethod
    def _get_largs_dphys():
        largs = ["lSymbols"]
        return largs

    @staticmethod
    def _get_largs_dreflect():
        largs = ["Types", "coefs_reflect"]
        return largs

    @staticmethod
    def _get_largs_dmisc():
        largs = ["color"]
        return largs

    ###########
    # Get check and format inputs
    ###########

    @staticmethod
    def _checkformat_Lim(Lim, Type="Tor"):
        if Lim is None:
            Lim = np.array([], dtype=float)
        else:
            assert hasattr(Lim, "__iter__")
            Lim = np.asarray(Lim, dtype=float)
            assert Lim.ndim in [1, 2]
            if Lim.ndim == 1:
                assert Lim.size in [0, 2]
                if Lim.size == 2:
                    Lim = Lim.reshape((2, 1))
            else:
                if Lim.shape[0] != 2:
                    Lim = Lim.T
            if Type == "Lin":
                if not np.all(Lim[0, :] < Lim[1, :]):
                    msg = "All provided Lim must be increasing !"
                    raise Exception(msg)
            else:
                Lim = np.arctan2(np.sin(Lim), np.cos(Lim))
            assert np.all(~np.isnan(Lim))
        return Lim

    @staticmethod
    def _checkformat_posextent(pos, extent, Type="Tor"):
        lC = [pos is None, extent is None]
        if any(lC):
            if not all(lC):
                msg = ""
                raise Exception(msg)
            pos = np.array([], dtype=float)
            extent = np.array([], dtype=float)
        else:
            lfloat = [int, float, np.int64, np.float64]
            assert type(pos) in lfloat or hasattr(pos, "__iter__")
            if type(pos) in lfloat:
                pos = np.array([pos], dtype=float)
            else:
                pos = np.asarray(pos, dtype=float).ravel()
            if Type == "Tor":
                pos = np.arctan2(np.sin(pos), np.cos(pos))
            assert type(extent) in lfloat or hasattr(extent, "__iter__")
            if type(extent) in lfloat:
                extent = float(extent)
            else:
                extent = np.asarray(extent, dtype=float).ravel()
                assert extent.size == pos.size
            if not np.all(extent > 0.0):
                msg = "All provided extent values must be >0 !"
                raise Exception(msg)
            if Type == "Tor":
                if not np.all(extent < 2.0 * np.pi):
                    msg = "Provided extent must be in ]0;2pi[ (radians)!"
                    raise Exception(msg)
            assert np.all(~np.isnan(pos)) and np.all(~np.isnan(extent))
        return pos, extent

    @staticmethod
    def _get_LimFromPosExtent(pos, extent, Type="Tor"):
        if pos.size > 0:
            Lim = pos[np.newaxis, :] + np.array([[-0.5], [0.5]]) * extent
            if Type == "Tor":
                Lim = np.arctan2(np.sin(Lim), np.cos(Lim))
        else:
            Lim = np.asarray([], dtype=float)
        return Lim

    @staticmethod
    def _get_PosExtentFromLim(Lim, Type="Tor"):
        if Lim.size > 0:
            pos, extent = np.mean(Lim, axis=0), Lim[1, :] - Lim[0, :]
            if Type == "Tor":
                ind = Lim[0, :] > Lim[1, :]
                pos[ind] = pos[ind] + np.pi
                extent[ind] = 2.0 * np.pi + extent[ind]
                pos = np.arctan2(np.sin(pos), np.cos(pos))
                assert np.all(extent > 0.0)
            if np.std(extent) < np.mean(extent) * 1.0e-9:
                extent = np.mean(extent)
        else:
            pos = np.array([], dtype=float)
            extent = np.array([], dtype=float)
        return pos, extent

    @classmethod
    def _checkformat_inputs_dgeom(
        cls,
        Poly=None,
        Lim=None,
        pos=None,
        extent=None,
        Type=None,
        Clock=False,
        arrayorder=None,
    ):
        if arrayorder is None:
            arrayorder = Struct._ddef["dgeom"]["arrayorder"]
        if Type is None:
            Type = Struct._ddef["dgeom"]["Type"]

        dins = {
            "Poly": {
                "var": Poly,
                "iter2array": float,
                "ndim": 2,
                "inshape": 2,
            },
            "Clock": {"var": Clock, "cls": bool},
            "arrayorder": {"var": arrayorder, "in": ["C", "F"]},
            "Type": {"var": Type, "in": ["Tor", "Lin"]},
        }
        dins, err, msg = cls._check_InputsGeneric(dins, tab=0)
        if err:
            raise Exception(msg)
        Poly = dins["Poly"]["var"]
        if Poly.shape[0] != 2:
            Poly = Poly.T

        # --------------------------------------
        # Elimininate any double identical point

        # Treat closed polygons seperately (no warning)
        if np.sum((Poly[:, 0] - Poly[:, -1])**2) < 1.e-12:
            Poly = Poly[:, :-1]

        # Treat other points
        ind = np.sum(np.diff(np.concatenate((Poly, Poly[:, 0:1]), axis=1),
                             axis=1) ** 2, axis=0) < 1.0e-12
        if np.any(ind):
            npts = Poly.shape[1]
            Poly = Poly[:, ~ind]
            msg = (
                "%s instance: double identical points in Poly\n" % cls.__name__
            )
            msg += "  => %s points removed\n" % ind.sum()
            msg += "  => Poly goes from %s to %s points" % (
                npts,
                Poly.shape[1],
            )
            warnings.warn(msg)
            ind = np.sum(np.diff(np.concatenate((Poly, Poly[:, 0:1]), axis=1),
                                 axis=1) ** 2, axis=0) < 1.0e-12
            assert not np.any(ind), ind

        lC = [Lim is None, pos is None]
        if not any(lC):
            msg = "Please provide either Lim xor pos/extent pair!\n"
            msg += "Lim should be an array of limits\n"
            msg += (
                "pos should be an array of centers and extent a float / array"
            )
            raise Exception(msg)
        if all(lC):
            pos = np.asarray([], dtype=float)
            extent = np.asarray([], dtype=float)
            # Lim = np.asarray([],dtype=float)
        elif lC[0]:
            pos, extent = cls._checkformat_posextent(pos, extent, Type)
            # Lim = cls._get_LimFromPosExtent(pos, extent, Type)
        else:
            Lim = cls._checkformat_Lim(Lim, Type)
            pos, extent = cls._get_PosExtentFromLim(Lim, Type)

        return Poly, pos, extent, Type, arrayorder

    def _checkformat_inputs_dsino(self, RefPt=None, nP=None):
        assert type(nP) is int and nP > 0
        assert RefPt is None or hasattr(RefPt, "__iter__")
        if RefPt is None:
            RefPt = self._dgeom["BaryS"]
        RefPt = np.asarray(RefPt, dtype=float).flatten()
        assert RefPt.size == 2, "RefPt must be of size=2 !"
        return RefPt

    @staticmethod
    def _checkformat_inputs_dphys(lSymbols=None):
        if lSymbols is not None:
            assert type(lSymbols) in [list, str]
            if type(lSymbols) is list:
                assert all([type(ss) is str for ss in lSymbols])
            else:
                lSymbols = [lSymbols]
            lSymbols = np.asarray(lSymbols, dtype=str)
        return lSymbols

    def _checkformat_inputs_dreflect(self, Types=None, coefs_reflect=None):
        if Types is None:
            Types = self._ddef["dreflect"]["Type"]

        assert type(Types) in [str, np.ndarray]
        if type(Types) is str:
            assert Types in self._DREFLECT_DTYPES.keys()
            Types = np.full(
                (self.nseg + 2,), self._DREFLECT_DTYPES[Types], dtype=np.int64
            )
        else:
            Types = Types.astype(int).ravel()
            assert Types.shape == (self.nseg + 2,)
            Typesu = np.unique(Types)
            lc = np.array([Typesu == vv
                           for vv in self._DREFLECT_DTYPES.values()])
            assert np.all(np.any(lc, axis=0))

        assert coefs_reflect is None
        return Types, coefs_reflect

    @classmethod
    def _checkformat_inputs_dmisc(cls, color=None):
        if color is None:
            color = mpl.colors.to_rgba(cls._ddef["dmisc"]["color"])
        assert mpl.colors.is_color_like(color)
        return tuple(np.array(mpl.colors.to_rgba(color), dtype=float))

    ###########
    # Get keys of dictionnaries
    ###########

    @staticmethod
    def _get_keys_dgeom():
        lk = [
            "Poly",
            "pos",
            "extent",
            "noccur",
            "Multi",
            "nP",
            "P1Max",
            "P1Min",
            "P2Max",
            "P2Min",
            "BaryP",
            "BaryL",
            "BaryS",
            "BaryV",
            "Surf",
            "VolAng",
            "Vect",
            "VIn",
            "circ-C",
            "circ-r",
            "Clock",
            "arrayorder",
            "move",
            "move_param",
            "move_kwdargs",
        ]
        return lk

    @staticmethod
    def _get_keys_dsino():
        lk = ["RefPt", "nP", "EnvTheta", "EnvMinMax"]
        return lk

    @staticmethod
    def _get_keys_dphys():
        lk = ["lSymbols"]
        return lk

    @staticmethod
    def _get_keys_dreflect():
        lk = ["Types", "coefs_reflect"]
        return lk

    @staticmethod
    def _get_keys_dmisc():
        lk = ["color"]
        return lk

    ###########
    # _init
    ###########

    def _init(
        self,
        Poly=None,
        Type=_Type,
        Lim=None,
        pos=None,
        extent=None,
        Clock=_Clock,
        arrayorder=_arrayorder,
        sino_RefPt=None,
        sino_nP=_def.TorNP,
        color=None,
        **kwdargs
    ):
        allkwds = dict(locals(), **kwdargs)
        largs = self._get_largs_dgeom(sino=True)
        kwdgeom = self._extract_kwdargs(allkwds, largs)
        largs = self._get_largs_dphys()
        kwdphys = self._extract_kwdargs(allkwds, largs)
        largs = self._get_largs_dreflect()
        kwdreflect = self._extract_kwdargs(allkwds, largs)
        largs = self._get_largs_dmisc()
        kwdmisc = self._extract_kwdargs(allkwds, largs)
        self._set_dgeom(**kwdgeom)
        self.set_dphys(**kwdphys)
        self.set_dreflect(**kwdreflect)
        self._set_dmisc(**kwdmisc)
        self._dstrip["strip"] = 0

    ###########
    # set dictionaries
    ###########

    def _set_dgeom(
        self,
        Poly=None,
        Lim=None,
        pos=None,
        extent=None,
        Clock=False,
        arrayorder="C",
        sino_RefPt=None,
        sino_nP=_def.TorNP,
        sino=True,
    ):
        out = self._checkformat_inputs_dgeom(
            Poly=Poly,
            Lim=Lim,
            pos=pos,
            extent=extent,
            Type=self.Id.Type,
            Clock=Clock,
        )
        Poly, pos, extent, Type, arrayorder = out
        dgeom = _comp._Struct_set_Poly(
            Poly,
            pos=pos,
            extent=extent,
            arrayorder=arrayorder,
            Type=self.Id.Type,
            Clock=Clock,
        )
        dgeom["arrayorder"] = arrayorder
        self._dgeom.update(dgeom)
        if sino:
            self.set_dsino(sino_RefPt, nP=sino_nP)

    def set_dsino(self, RefPt=None, nP=_def.TorNP):
        RefPt = self._checkformat_inputs_dsino(RefPt=RefPt, nP=nP)
        EnvTheta, EnvMinMax = _GG.Sino_ImpactEnv(
            RefPt, self.Poly_closed, NP=nP, Test=False
        )
        self._dsino = {
            "RefPt": RefPt,
            "nP": nP,
            "EnvTheta": EnvTheta,
            "EnvMinMax": EnvMinMax,
        }

    def set_dphys(self, lSymbols=None):
        lSymbols = self._checkformat_inputs_dphys(lSymbols)
        self._dphys["lSymbols"] = lSymbols

    def set_dreflect(self, Types=None, coefs_reflect=None):
        Types, coefs_reflect = self._checkformat_inputs_dreflect(
            Types=Types, coefs_reflect=coefs_reflect
        )
        self._dreflect["Types"] = Types
        self._dreflect["coefs_reflect"] = coefs_reflect

    def _set_color(self, color=None):
        color = self._checkformat_inputs_dmisc(color=color)
        self._dmisc["color"] = color
        self._dplot["cross"]["dP"]["color"] = color
        self._dplot["hor"]["dP"]["color"] = color
        self._dplot["3d"]["dP"]["color"] = color

    def _set_dmisc(self, color=None):
        self._set_color(color)

    ###########
    # strip dictionaries
    ###########

    def _strip_dgeom(
        self,
        lkeep=["Poly", "pos", "extent", "Clock", "arrayorder",
               "move", "move_param", "move_kwdargs"]
    ):
        utils.ToFuObject._strip_dict(self._dgeom, lkeep=lkeep)

    def _strip_dsino(self, lkeep=["RefPt", "nP"]):
        utils.ToFuObject._strip_dict(self._dsino, lkeep=lkeep)

    def _strip_dphys(self, lkeep=["lSymbols"]):
        utils.ToFuObject._strip_dict(self._dphys, lkeep=lkeep)

    def _strip_dreflect(self, lkeep=["Types", "coefs_reflect"]):
        utils.ToFuObject._strip_dict(self._dreflect, lkeep=lkeep)

    def _strip_dmisc(self, lkeep=["color"]):
        utils.ToFuObject._strip_dict(self._dmisc, lkeep=lkeep)

    ###########
    # rebuild dictionaries
    ###########

    def _rebuild_dgeom(
        self,
        lkeep=["Poly", "pos", "extent", "Clock", "arrayorder"]
    ):
        reset = utils.ToFuObject._test_Rebuild(self._dgeom, lkeep=lkeep)
        if reset:
            utils.ToFuObject._check_Fields4Rebuild(
                self._dgeom, lkeep=lkeep, dname="dgeom"
            )
            self._set_dgeom(
                self.Poly,
                pos=self.pos,
                extent=self.extent,
                Clock=self.dgeom["Clock"],
                arrayorder=self.dgeom["arrayorder"],
                sino=False,
            )

    def _rebuild_dsino(self, lkeep=["RefPt", "nP"]):
        reset = utils.ToFuObject._test_Rebuild(self._dsino, lkeep=lkeep)
        if reset:
            utils.ToFuObject._check_Fields4Rebuild(
                self._dsino, lkeep=lkeep, dname="dsino"
            )
            self.set_dsino(RefPt=self.dsino["RefPt"], nP=self.dsino["nP"])

    def _rebuild_dphys(self, lkeep=["lSymbols"]):
        reset = utils.ToFuObject._test_Rebuild(self._dphys, lkeep=lkeep)
        if reset:
            utils.ToFuObject._check_Fields4Rebuild(
                self._dphys, lkeep=lkeep, dname="dphys"
            )
            self.set_dphys(lSymbols=self.dphys["lSymbols"])

    def _rebuild_dreflect(self, lkeep=["Types", "coefs_reflect"]):
        reset = utils.ToFuObject._test_Rebuild(self._dreflect, lkeep=lkeep)
        if reset:
            utils.ToFuObject._check_Fields4Rebuild(
                self._dreflect, lkeep=lkeep, dname="dreflect"
            )
            self.set_dreflect(
                Types=self.dreflect["Types"],
                coefs_reflect=self.dreflect["coefs_reflect"]
            )

    def _rebuild_dmisc(self, lkeep=["color"]):
        reset = utils.ToFuObject._test_Rebuild(self._dmisc, lkeep=lkeep)
        if reset:
            utils.ToFuObject._check_Fields4Rebuild(
                self._dmisc, lkeep=lkeep, dname="dmisc"
            )
            self._set_dmisc(color=self.dmisc["color"])

    ###########
    # _strip and get/from dict
    ###########

    @classmethod
    def _strip_init(cls):
        cls._dstrip["allowed"] = [0, 1, 2]
        nMax = max(cls._dstrip["allowed"])
        doc = """
                 1: Remove dsino expendables
                 2: Remove also dgeom, dphys, dreflect and dmisc expendables"""
        doc = utils.ToFuObjectBase.strip.__doc__.format(doc, nMax)
        cls.strip.__doc__ = doc

    def strip(self, strip=0):
        # super()
        super(Struct, self).strip(strip=strip)

    def _strip(self, strip=0):
        if strip == 0:
            self._rebuild_dgeom()
            self._rebuild_dsino()
            self._rebuild_dphys()
            self._rebuild_dreflect()
            self._rebuild_dmisc()
        elif strip == 1:
            self._strip_dsino()
            self._rebuild_dgeom()
            self._rebuild_dphys()
            self._rebuild_dreflect()
            self._rebuild_dmisc()
        else:
            self._strip_dsino()
            self._strip_dgeom()
            self._strip_dphys()
            self._strip_dreflect()
            self._strip_dmisc()

    def _to_dict(self):
        dout = {
            "dgeom": {"dict": self.dgeom, "lexcept": None},
            "dsino": {"dict": self.dsino, "lexcept": None},
            "dphys": {"dict": self.dphys, "lexcept": None},
            "dreflect": {"dict": self.dreflect, "lexcept": None},
            "dmisc": {"dict": self.dmisc, "lexcept": None},
            "dplot": {"dict": self._dplot, "lexcept": None},
        }
        return dout

    def _from_dict(self, fd):
        self._dgeom.update(**fd["dgeom"])
        self._dsino.update(**fd["dsino"])
        self._dphys.update(**fd["dphys"])
        self._dreflect.update(**fd["dreflect"])
        self._dmisc.update(**fd["dmisc"])
        if "dplot" in fd.keys():
            self._dplot.update(**fd["dplot"])

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
        return self._dgeom["Poly"]

    @property
    def Poly_closed(self):
        """ Returned the closed polygon """
        return np.hstack((self._dgeom["Poly"], self._dgeom["Poly"][:, 0:1]))

    @property
    def nseg(self):
        """ Retunr the number of segmnents constituting the closed polygon """
        return self._dgeom["Poly"].shape[1]

    @property
    def pos(self):
        return self._dgeom["pos"]

    @property
    def extent(self):
        if hasattr(self._dgeom["extent"], "__iter__"):
            extent = self._dgeom["extent"]
        else:
            extent = np.full(self._dgeom["pos"].shape, self._dgeom["extent"])
        return extent

    @property
    def noccur(self):
        return self._dgeom["noccur"]

    @property
    def Lim(self):
        Lim = self._get_LimFromPosExtent(
            self._dgeom["pos"], self._dgeom["extent"], Type=self.Id.Type
        )
        return Lim.T

    @property
    def dsino(self):
        return self._dsino

    @property
    def dphys(self):
        return self._dphys

    @property
    def dreflect(self):
        return self._dreflect

    @property
    def dmisc(self):
        return self._dmisc

    ###########
    # public methods
    ###########

    def get_summary(
        self,
        sep="  ",
        line="-",
        just="l",
        table_sep=None,
        verb=True,
        return_=False,
    ):
        """ Summary description of the object content """

        # -----------------------
        # Build detailed view
        col0 = [
            "class",
            "Name",
            "SaveName",
            "nP",
            "noccur",
        ]
        ar0 = [
            self._Id.Cls,
            self._Id.Name,
            self._Id.SaveName,
            str(self._dgeom["nP"]),
            str(self._dgeom["noccur"]),
        ]
        if self._dgeom["move"] is not None:
            col0 += ['move', 'param']
            ar0 += [self._dgeom["move"],
                    str(round(self._dgeom["move_param"], ndigits=4))]
        col0.append('color')
        cstr = ('('
                + ', '.join(['{:4.2}'.format(cc)
                             for cc in self._dmisc["color"]])
                + ')')
        ar0.append(cstr)

        return self._get_summary(
            [ar0],
            [col0],
            sep=sep,
            line=line,
            table_sep=table_sep,
            verb=verb,
            return_=return_,
        )

    ###########
    # public methods for movement
    ###########

    def _update_or_copy(self, poly,
                        pos=None,
                        update_extent=None,
                        return_copy=None, name=None):
        if update_extent is None:
            update_extent = _UPDATE_EXTENT
        if return_copy is None:
            return_copy = _RETURN_COPY
        extent = self.extent
        if update_extent is True:
            if extent is not None:
                ratio = np.nanmin(poly[0, :]) / np.nanmin(self.Poly[0, :])
                extent = extent*ratio
        if pos is None:
            pos = self.pos
        if return_copy is True:
            if name is None:
                name = self.Id.Name + 'copy'
            return self.__class__(Poly=poly,
                                  extent=extent, pos=pos,
                                  sino_RefPt=self._dsino['RefPt'],
                                  sino_nP=self._dsino['nP'],
                                  color=self._dmisc['color'],
                                  Exp=self.Id.Exp,
                                  Name=name,
                                  shot=self.Id.shot,
                                  SavePath=self.Id.SavePath,
                                  Type=self.Id.Type)
        else:
            self._set_dgeom(poly, pos=pos, extent=extent,
                            sino_RefPt=self._dsino['RefPt'],
                            sino_nP=self._dsino['nP'])

    def translate_in_cross_section(self, distance=None, direction_rz=None,
                                   update_extent=None,
                                   return_copy=None, name=None):
        """ Translate the structure in the poloidal plane """
        poly = self._translate_pts_poloidal_plane_2D(
            pts_rz=self.Poly,
            direction_rz=direction_rz, distance=distance)
        return self._update_or_copy(poly, update_extent=update_extent,
                                    return_copy=return_copy, name=name)

    def rotate_in_cross_section(self, angle=None, axis_rz=None,
                                update_extent=True,
                                return_copy=None, name=None):
        """ Rotate the structure in the poloidal plane """
        poly = self._rotate_pts_vectors_in_poloidal_plane_2D(
            pts_rz=self.Poly,
            axis_rz=axis_rz, angle=angle)
        return self._update_or_copy(poly, update_extent=update_extent,
                                    return_copy=return_copy, name=name)

    def rotate_around_torusaxis(self, angle=None,
                                return_copy=None, name=None):
        """ Rotate the structure in the poloidal plane """
        if self.Id.Type != 'Tor':
            msg = "Movement only available for Tor configurations!"
            raise Exception(msg)
        pos = self.pos
        if pos is not None:
            pos = pos + angle
        return self._update_or_copy(self.Poly, pos=pos,
                                    update_extent=False,
                                    return_copy=return_copy, name=name)

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

    ###########
    # Other public methods
    ###########

    def set_color(self, col):
        self._set_color(col)

    def get_color(self):
        return self._dmisc["color"]

    def isInside(self, pts, In="(X,Y,Z)"):
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
        if self._dgeom["noccur"] > 0:
            ind = _GG._Ves_isInside(
                pts,
                self.Poly,
                ves_lims=np.ascontiguousarray(self.Lim),
                nlim=self._dgeom["noccur"],
                ves_type=self.Id.Type,
                in_format=In,
                test=True,
            )
        else:
            ind = _GG._Ves_isInside(
                pts,
                self.Poly,
                ves_lims=None,
                nlim=0,
                ves_type=self.Id.Type,
                in_format=In,
                test=True,
            )
        return ind

    def get_InsideConvexPoly(
        self,
        RelOff=_def.TorRelOff,
        ZLim="Def",
        Spline=True,
        Splprms=_def.TorSplprms,
        NP=_def.TorInsideNP,
        Plot=False,
        Test=True,
    ):
        """ Return a polygon that is a smaller and smoothed approximation of
        Ves.Poly, useful for excluding the divertor region in a Tokamak

        For some uses, it can be practical to approximate the polygon defining
        the Ves object (which can be non-convex, like with a divertor), by a
        simpler, sligthly smaller and convex polygon.
        This method provides a fast solution for computing such a proxy.

        Parameters
        ----------
        RelOff :    float
            Fraction by which an homothetic polygon should be reduced
            (1.-RelOff)*(Poly-BaryS)
        ZLim :      None / str / tuple
            Flag indicating what limits shall be put to the height of the
            polygon (used for excluding divertor)
        Spline :    bool
            Flag indiating whether the reduced and truncated polygon shall be
            smoothed by 2D b-spline curves
        Splprms :   list
            List of 3 parameters to be used for the smoothing
            [weights,smoothness,b-spline order], fed to
            scipy.interpolate.splprep()
        NP :        int
            Number of points to be used to define the smoothed polygon
        Plot :      bool
            Flag indicating whether the result shall be plotted for visual
            inspection
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        Poly :      np.ndarray
            (2,N) polygon resulting from homothetic transform, truncating and
            optional smoothingop

        """
        return _comp._Ves_get_InsideConvexPoly(
            self.Poly_closed,
            self.dgeom["P2Min"],
            self.dgeom["P2Max"],
            self.dgeom["BaryS"],
            RelOff=RelOff,
            ZLim=ZLim,
            Spline=Spline,
            Splprms=Splprms,
            NP=NP,
            Plot=Plot,
            Test=Test,
        )

    def get_sampleEdge(
        self,
        res=None,
        domain=None,
        resMode=None,
        offsetIn=0.0,
    ):
        """ Sample the polygon edges, with resolution res

        Sample each segment of the 2D polygon
        Sampling can be limited to a domain
        """
        if res is None:
            res = _RES
        return _comp._Ves_get_sampleEdge(
            self.Poly_closed,
            res=res,
            domain=domain,
            resMode=resMode,
            offsetIn=offsetIn,
            VIn=self.dgeom["VIn"],
            margin=1.0e-9,
        )

    def get_sampleCross(
        self,
        res=None,
        domain=None,
        resMode=None,
        ind=None,
        mode="flat",
    ):
        """ Sample, with resolution res, the 2D cross-section

        The sampling domain can be limited by domain or ind

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
        if res is None:
            res = _RES
        args = [
            self.Poly_closed,
            self.dgeom["P1Min"][0],
            self.dgeom["P1Max"][0],
            self.dgeom["P2Min"][1],
            self.dgeom["P2Max"][1],
        ]
        kwdargs = dict(
            res=res, domain=domain, resMode=resMode, ind=ind,
            margin=1.0e-9, mode=mode
        )
        return _comp._Ves_get_sampleCross(*args, **kwdargs)

    def get_sampleS(
        self,
        res=None,
        domain=None,
        resMode=None,
        ind=None,
        offsetIn=0.0,
        returnas="(X,Y,Z)",
        Ind=None,
    ):
        """ Sample, with resolution res, the surface defined by domain or ind

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
        domain :    None / list of 3 lists of 2 floats
            Limits of the domain in which the sample should be computed
                None : whole surface of the object
                list : [D1, D2, D3], where Di is a len()=2 list
                       (increasing floats, setting limits along coordinate i)
                    [DR, DZ, DPhi]: in toroidal geometry (self.Id.Type=='Tor')
                    [DX, DY, DZ]  : in linear geometry (self.Id.Type=='Lin')
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
                > # Perform operations, save only the points indices
                > # (save space)
                > ...
                > # Retrieve the points from their indices (requires same res)
                > pts2, dS2, ind2, reseff2 = obj.get_sample(0.05, ind=ind)
                > np.allclose(pts,pts2)
                True
        offsetIn:   float
            Offset distance from the actual surface of the object
            Inwards if positive
            Useful to avoid numerical errors
        returnas:   str
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
            assert self.dgeom["Multi"]
        if res is None:
            res = _RES
        kwdargs = dict(
            res=res,
            domain=domain,
            resMode=resMode,
            ind=ind,
            offsetIn=offsetIn,
            VIn=self.dgeom["VIn"],
            VType=self.Id.Type,
            VLim=np.ascontiguousarray(self.Lim),
            nVLim=self.noccur,
            returnas=returnas,
            margin=1.0e-9,
            Multi=self.dgeom["Multi"],
            Ind=Ind,
        )
        return _comp._Ves_get_sampleS(self.Poly, **kwdargs)

    def get_sampleV(
        self,
        res=None,
        domain=None,
        resMode=None,
        ind=None,
        returnas="(X,Y,Z)",
        algo="new",
        num_threads=48
    ):
        """ Sample, with resolution res, the volume defined by domain or ind

        The 3D volume is sampled in:
            - the whole volume (domain=None and ind=None)
            - a sub-domain defined by bounds on each coordinates (domain)
            - a pre-computed subdomain stored in indices (ind)

        The coordinatesd of the center of each volume elements are returned as
        pts in choosen coordinates (returnas)

        For a torus, the elementary volume is kept constant, meaning that the
        toroidal angular step is decreased as R increases

        Parameters
        ----------
        res     :   float / list of 3 floats
            Desired resolution of the surfacic sample
                float   : same resolution for all directions of the sample
                list    : [dYR, dZ, dXPhi] where:
                    dYR     : res. along in radial / Y direction
                    dZ      : res. along Z direction
                    dXPhi   : res. along axis (toroidal/linear direction)
        domain :    None / list of 3 lists of 2 floats
            Limits of the domain in which the sample should be computed
                None : whole surface of the object
                list : [D1, D2, D3], where Di is a len()=2 list
                       (increasing floats, setting limits along coordinate i)
                    [DR, DZ, DPhi]: in toroidal geometry (self.Id.Type=='Tor')
                    [DX, DY, DZ]  : in linear geometry (self.Id.Type=='Lin')
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
                > # Perform operations, save only the points indices
                > # (save space)
                > ...
                > # Retrieve the points from their indices (requires same res)
                > pts2, dS2, ind2, reseff2 = obj.get_sample(0.05, ind=ind)
                > np.allclose(pts,pts2)
                True
        returnas:   str
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
        dV      :   np.ndarray / list of np.ndarrays
            The volume (in m^3) associated to each point
        ind     :   np.ndarray / list of np.ndarrays
            The index of each point
        reseff  :   np.ndarray / list of np.ndarrays
            Effective resolution in both directions after sample computation
        """

        args = [
            self.Poly,
            self.dgeom["P1Min"][0],
            self.dgeom["P1Max"][0],
            self.dgeom["P2Min"][1],
            self.dgeom["P2Max"][1],
        ]
        kwdargs = dict(
            res=res,
            domain=domain,
            resMode=resMode,
            ind=ind,
            VType=self.Id.Type,
            VLim=self.Lim,
            returnas=returnas,
            margin=1.0e-9,
            algo=algo,
            num_threads=num_threads
        )
        return _comp._Ves_get_sampleV(*args, **kwdargs)

    def _get_phithetaproj(self, refpt=None):
        # Prepare ax
        if refpt is None:
            msg = "Please provide refpt (R,Z)"
            raise Exception(msg)
        refpt = np.atleast_1d(np.squeeze(refpt))
        assert refpt.shape == (2,)
        return _comp._Struct_get_phithetaproj(
            refpt, self.Poly, self.Lim, self.noccur
        )

    def _get_phithetaproj_dist(
        self, refpt=None, ntheta=None, nphi=None, theta=None, phi=None
    ):
        # Prepare ax
        if refpt is None:
            msg = "Please provide refpt (R,Z)"
            raise Exception(msg)
        refpt = np.atleast_1d(np.squeeze(refpt))
        assert refpt.shape == (2,)

        # Prepare theta and phi
        if theta is None and ntheta is None:
            nphi = _PHITHETAPROJ_NTHETA
        lc = [ntheta is None, theta is None]
        if np.sum(lc) != 1:
            msg = "Please provide either ntheta xor a theta vector !"
            raise Exception(msg)
        if theta is None:
            theta = np.linspace(-np.pi, np.pi, ntheta, endpoint=True)

        if phi is None and nphi is None:
            nphi = _PHITHETAPROJ_NPHI
        lc = [nphi is None, phi is None]
        if np.sum(lc) != 1:
            msg = "Please provide either nphi xor a phi vector !"
            raise Exception(msg)
        if phi is None:
            phi = np.linspace(-np.pi, np.pi, nphi, endpoint=True)

        # Get limits
        out = _comp._Struct_get_phithetaproj(
            refpt, self.Poly_closed, self.Lim, self.noccur
        )
        nDphi, Dphi, nDtheta, Dtheta = out

        # format inputs
        theta = np.atleast_1d(np.ravel(theta))
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        phi = np.atleast_1d(np.ravel(phi))
        phi = np.arctan2(np.sin(phi), np.cos(phi))
        ntheta, nphi = theta.size, phi.size

        dist = np.full((ntheta, nphi), np.nan)

        # Get dist
        dist_theta, indphi = _comp._get_phithetaproj_dist(
            self.Poly_closed,
            refpt,
            Dtheta,
            nDtheta,
            Dphi,
            nDphi,
            theta,
            phi,
            ntheta,
            nphi,
            self.noccur,
        )
        dist[:, indphi] = dist_theta[:, None]

        return dist, nDphi, Dphi, nDtheta, Dtheta

    @staticmethod
    def _get_reflections_ufromTypes(u, vperp, Types):
        indspec = Types == 0
        inddiff = Types == 1
        indcorn = Types == 2

        # Get reflected unit vectors
        u2 = np.full(u.shape, np.nan)
        if np.any(np.logical_or(indspec, inddiff)):
            vpar = np.array(
                [
                    vperp[1, :] * u[2, :] - vperp[2, :] * u[1, :],
                    vperp[2, :] * u[0, :] - vperp[0, :] * u[2, :],
                    vperp[0, :] * u[1, :] - vperp[1, :] * u[0, :],
                ]
            )
            vpar = np.array(
                [
                    vpar[1, :] * vperp[2, :] - vpar[2, :] * vperp[1, :],
                    vpar[2, :] * vperp[0, :] - vpar[0, :] * vperp[2, :],
                    vpar[0, :] * vperp[1, :] - vpar[1, :] * vperp[0, :],
                ]
            )
            vpar = vpar / np.sqrt(np.sum(vpar ** 2, axis=0))[None, :]

            if np.any(indspec):
                # Compute u2 for specular
                sca = np.sum(
                    u[:, indspec] * vperp[:, indspec], axis=0, keepdims=True
                )
                sca2 = np.sum(
                    u[:, indspec] * vpar[:, indspec], axis=0, keepdims=True
                )
                assert np.all(sca <= 0.0) and np.all(sca >= -1.0)
                assert np.all(sca2 >= 0.0) and np.all(sca <= 1.0)
                u2[:, indspec] = (
                    -sca * vperp[:, indspec] + sca2 * vpar[:, indspec]
                )

            if np.any(inddiff):
                # Compute u2 for diffusive
                sca = 2.0 * (np.random.random((1, inddiff.sum())) - 0.5)
                u2[:, inddiff] = (
                    np.sqrt(1.0 - sca**2) * vperp[:, inddiff]
                    + sca * vpar[:, inddiff]
                )

        if np.any(indcorn):
            u2[:, indcorn] = -u[:, indcorn]
        return u2

    def get_reflections(self, indout2, u=None, vperp=None):
        """ Return the reflected unit vectors from input unit vectors and vperp

        The reflected unit vector depends on the incoming LOS (u),
        the local normal unit vector (vperp), and the polygon segment hit
        (indout2)
        Future releases: dependence on lambda

        Also return per-LOS reflection Types (0:specular, 1:diffusive, 2:ccube)

        """

        # Get per-LOS reflection Types and associated indices
        Types = self._dreflect["Types"][indout2]
        u2 = None
        if u is not None:
            assert vperp is not None
            u2 = self._get_reflections_ufromTypes(u, vperp, Types)
        return Types, u2

    def plot(
        self,
        lax=None,
        proj="all",
        element="PIBsBvV",
        dP=None,
        dI=_def.TorId,
        dBs=_def.TorBsd,
        dBv=_def.TorBvd,
        dVect=_def.TorVind,
        dIHor=_def.TorITord,
        dBsHor=_def.TorBsTord,
        dBvHor=_def.TorBvTord,
        Lim=None,
        Nstep=_def.TorNTheta,
        dLeg=_def.TorLegd,
        indices=True,
        draw=True,
        fs=None,
        wintit=None,
        Test=True,
    ):
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
            Dict of properties for plotting point 'I' in Cross-section
            projection
        dIHor :     dict / None
            Dict of properties for plotting point 'I' in horizontal projection
        dBs :       dict / None
            Dict of properties for plotting point 'Bs' in Cross-section
            projection
        dBsHor :    dict / None
            Dict of properties for plotting point 'Bs' in horizontal projection
        dBv :       dict / None
            Dict of properties for plotting point 'Bv' in Cross-section
            projection
        dBvHor :    dict / None
            Dict of properties for plotting point 'Bv' in horizontal projection
        dVect :     dict / None
            Dict of properties for plotting point 'V' in cross-section
            projection
        dLeg :      dict / None
            Dict of properties for plotting the legend, fed to plt.legend()
            The legend is not plotted if None
        Lim :       list or tuple
            Array of a lower and upper limit of angle (rad.) or length for
            plotting the '3d' proj
        Nstep :     int
            Number of points for sampling in ignorable coordinate (toroidal
            angle or length)
        draw :      bool
            Flag indicating whether the fig.canvas.draw() shall be called
            automatically
        a4 :        bool
            Flag indicating whether the figure should be plotted in a4
            dimensions for printing
        Test :      bool
            Flag indicating whether the inputs should be tested for conformity

        Returns
        -------
        La          list / plt.Axes
            Handles of the axes used for plotting (list if several axes where
            used)

        """
        kwdargs = locals()
        lout = ["self"]
        for k in lout:
            del kwdargs[k]
        return _plot.Struct_plot(self, **kwdargs)

    def plot_sino(
        self,
        ax=None,
        Ang=_def.LOSImpAng,
        AngUnit=_def.LOSImpAngUnit,
        Sketch=True,
        dP=None,
        dLeg=_def.TorLegd,
        draw=True,
        fs=None,
        wintit=None,
        Test=True,
    ):
        """ Plot the sinogram of the vessel polygon, by computing its envelopp
        in a cross-section, can also plot a 3D version of it.

        The envelop of the polygon is computed using self.Sino_RefPt as a
        reference point in projection space,
        and plotted using the provided dictionary of properties.
        Optionaly a small sketch can be included illustrating how the angle
        and the impact parameters are defined (if the axes is not provided).

        Parameters
        ----------
        proj :      str
            Flag indicating whether to plot a classic sinogram ('Cross') from
            the vessel cross-section (assuming 2D)
            or an extended 3D version '3d' of it with additional angle
        ax   :      None or plt.Axes
            The axes on which the plot should be done, if None a new figure
            and axes is created
        Ang  :      str
            Flag indicating which angle to use for the impact parameter, the
            angle of the line itself (xi) or of its impact parameter (theta)
        AngUnit :   str
            Flag for the angle units to be displayed, 'rad' for radians or
            'deg' for degrees
        Sketch :    bool
            Flag indicating whether a small skecth showing the definitions of
            angles 'theta' and 'xi' should be included or not
        Pdict :     dict
            Dictionary of properties used for plotting the polygon envelopp,
            fed to plt.plot() if proj='Cross' and to plt.plot_surface()
            if proj='3d'
        LegDict :   None or dict
            Dictionary of properties used for plotting the legend, fed to
            plt.legend(), the legend is not plotted if None
        draw :      bool
            Flag indicating whether the fig.canvas.draw() shall be called
            automatically
        a4 :        bool
            Flag indicating whether the figure should be plotted in a4
            dimensions for printing
        Test :      bool
            Flag indicating whether the inputs shall be tested for conformity

        Returns
        -------
        ax :        plt.Axes
            The axes used to plot

        """
        if Test:
            msg = "The impact parameters must be set ! (self.set_dsino())"
            assert not self.dsino["RefPt"] is None, msg

        # Only plot cross sino, from version 1.4.0
        dP = _def.TorPFilld if dP is None else dP
        ax = _plot.Plot_Impact_PolProjPoly(
            self,
            ax=ax,
            Ang=Ang,
            AngUnit=AngUnit,
            Sketch=Sketch,
            Leg=self.Id.NameLTX,
            dP=dP,
            dLeg=dLeg,
            draw=False,
            fs=fs,
            wintit=wintit,
            Test=Test,
        )
        # else:
        # Pdict = _def.TorP3DFilld if Pdict is None else Pdict
        # ax = _plot.Plot_Impact_3DPoly(self, ax=ax, Ang=Ang, AngUnit=AngUnit,
        # Pdict=Pdict, dLeg=LegDict, draw=False,
        # fs=fs, wintit=wintit, Test=Test)
        if draw:
            ax.figure.canvas.draw()
        return ax

    def save_to_txt(
        self,
        path="./",
        name=None,
        fmt=None,
        include=["Mod", "Cls", "Exp", "Name"],
        fmt_num="%.18e",
        delimiter=None,
        footer="",
        encoding=None,
        verb=True,
        return_pfe=False,
    ):
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

        All parameters apart from path, name and include are fed to
        numpy.savetxt()

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

        # Check inputs
        if name is None:
            name = self.Id.generate_SaveName(include)
        if path is None:
            path = self.Id.SavePath
        path = os.path.abspath(path)
        if fmt is None:
            fmt = 'txt'
        lfmtok = ['txt', 'csv']
        if fmt not in lfmtok:
            msg = ("The only accpedted formats are: {}".format(lfmtok))
            raise Exception(msg)
        pfe = os.path.join(path, '{}.{}'.format(name, fmt))
        if delimiter is None:
            if fmt == 'txt':
                delimiter = ' '
            else:
                delimiter = ', '

        nPno = np.r_[self.Poly.shape[1], self.noccur]
        poly = self.Poly.T
        posext = np.vstack((self.pos, self.extent)).T
        out = np.vstack((nPno, poly, posext))

        # default standards
        newline = "\n"
        comments = _COMMENT
        header = " Cls = {}\n Exp = {}\n Name = {}".format(
            self.__class__.__name__,
            self.Id.Exp,
            self.Id.Name,
        )

        kwds = dict(
            fmt=fmt_num,
            delimiter=delimiter,
            newline=newline,
            header=header,
            footer=footer,
            comments=comments,
        )
        if "encoding" in inspect.signature(np.savetxt).parameters:
            kwds["encoding"] = encoding
        np.savetxt(pfe, out, **kwds)
        if verb:
            print("save_to_txt in:\n", pfe)
        if return_pfe:
            return pfe

    @classmethod
    def from_txt(
        cls,
        pfe,
        returnas='object',
        Exp=None,
        Name=None,
        shot=None,
        Type=None,
        color=None,
        SavePath=os.path.abspath("./"),
        delimiter=None,
        comments=None,
        warn=None,
    ):
        """ Return the polygon and pos/extent stored in a .txt or .csv file

        The file must have been generated by method save_to_txt() (resp. csv)
        All arguments appart from pfe and returnas are:
            - fed to the relevant tofu.geom.Struct subclass to instanciate it
            - used only if returnas = 'object'

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
        returnas:    str
            Flag indicating whether to return:
               - 'dict'  : a dictionnary of np.ndarrays
               - 'object': a tofu.geom.Struct subclass, using the other kwdargs
        warn:       None / bool
            Whether to raise a warning if the formatting of the file is
            suspicious

        Return
        ------
        obj:    tf.geom.Struct sublass instance  / dict
            Depending on the value of returnas, obj can be:
                - An instance of the relevant tofu.geom.Struct subclass
                - A dict with keys 'poly', 'pos' and 'extent'
        """

        # Check inputs
        if returnas not in [object, 'object', dict, 'dict']:
            msg = ("Arg returnas must be either:"
                   + "\t- 'object': return {} instance\n".format(cls.__name__)
                   + "\t- 'dict' : return a dict with polygon, pos and extent")
            raise Exception(msg)
        if pfe[-4:] not in ['.txt', '.csv']:
            msg = ("Only accepts .txt and .csv files (fed to np.loadtxt) !\n"
                   + "\t file: {}".format(pfe))
            raise Exception(msg)
        if warn is None:
            warn = True

        if delimiter is None:
            if pfe.endswith('.csv'):
                delimiter = ', '
            else:
                delimiter = None
        if comments is None:
            comments = _COMMENT

        # Extract polygon from file and check
        oo = np.loadtxt(pfe, delimiter=delimiter, comments=comments)
        if not (oo.ndim == 2 and oo.shape[1] == 2):
            msg = ("The file should contain a (N,2) array !\n"
                   + "  \t file : {}\n".format(pfe)
                   + "\t shape: {0}".format(oo.shape))
            raise Exception(msg)

        c0 = oo[0, 0] == int(oo[0, 0]) and oo[0, 1] == int(oo[0, 1])
        if not c0:
            # assume noccur = 0
            npts, noccur = oo.shape[0], 0
            poly = oo
        else:
            c1 = oo.shape == (oo[0, 0] + oo[0, 1] + 1, 2)
            if c1 is True:
                npts, noccur = int(oo[0, 0]), int(oo[0, 1])
                poly = oo[1:1 + npts, :]
            else:
                npts, noccur = oo.shape[0], 0
                poly = oo
                if warn is True:
                    sha = (oo[0, 0] + oo[0, 1] + 1, 2)
                    shastr = '({0} + {1} + 1, 2)'.format(oo[0, 0], oo[0, 1])
                    msg = ("The shape of the array is not as expected!\n"
                           + "\tfile: {}\n".format(pfe)
                           + "\tExpected shape: {0} = {1}".format(sha, shastr)
                           + "\tObserved shape: {0}".format(oo.shape))
                    warnings.warn(msg)

        if noccur > 0:
            pos, extent = oo[1 + npts:, 0], oo[1 + npts:, 1]
        else:
            pos, extent = None, None

        # Try reading Exp and Name if not provided
        lc = [ss for ss, vv in [('Exp', Exp), ('Name', Name)] if vv is None]
        if len(lc) > 0:
            dparam = utils.from_txt_extract_params(pfe, lc)
            if 'Exp' in lc:
                Exp = dparam['Exp']
            if 'Name' in lc:
                Name = dparam['Name']

        # Return
        if returnas in [dict, 'dict']:
            return {'Name': Name, 'Exp': Exp, 'Cls': cls,
                    "poly": poly, "pos": pos, "extent": extent}
        else:
            SavePath = os.path.abspath(SavePath)
            obj = cls(
                Name=Name,
                Exp=Exp,
                shot=shot,
                Type=Type,
                Poly=poly,
                pos=pos,
                extent=extent,
                SavePath=SavePath,
                color=color,
            )
            return obj

    def save_to_imas(
        self,
        shot=None,
        run=None,
        refshot=None,
        refrun=None,
        occ=None,
        user=None,
        database=None,
        version=None,
        dryrun=False,
        verb=True,
        description_2d=None,
        unit=0,
    ):
        import tofu.imas2tofu as _tfimas

        _tfimas._save_to_imas(
            self,
            tfversion=__version__,
            shot=shot,
            run=run,
            refshot=refshot,
            refrun=refrun,
            user=user,
            database=database,
            version=version,
            dryrun=dryrun,
            verb=verb,
            description_2d=description_2d,
            unit=unit,
        )



"""
###############################################################################
###############################################################################
                      Effective Struct subclasses
###############################################################################
"""


class StructIn(Struct):
    _color = "k"
    _InOut = "in"

    @classmethod
    def _set_color_ddef(cls, color):
        # super
        color = mpl.colors.to_rgba(color)
        cls._ddef["dmisc"]["color"] = color
        cls._dplot["cross"]["dP"]["color"] = cls._ddef["dmisc"]["color"]
        cls._dplot["hor"]["dP"]["color"] = cls._ddef["dmisc"]["color"]
        cls._dplot["3d"]["dP"]["color"] = cls._ddef["dmisc"]["color"]

    @staticmethod
    def _checkformat_inputs_dgeom(
        Poly=None,
        Lim=None,
        pos=None,
        extent=None,
        Type=None,
        Clock=False,
        arrayorder=None,
    ):
        kwdargs = locals()
        # super
        out = Struct._checkformat_inputs_dgeom(**kwdargs)
        Poly, pos, extent, Type, arrayorder = out
        if Type == "Tor":
            msg = "StructIn subclasses cannot have noccur>0 if Type='Tor'!"
            assert pos.size == 0, msg
        return out


class StructOut(Struct):
    _color = (0.8, 0.8, 0.8, 0.8)
    _InOut = "out"

    @classmethod
    def _set_color_ddef(cls, color):
        color = mpl.colors.to_rgba(color)
        cls._ddef["dmisc"]["color"] = color
        cls._dplot["cross"]["dP"] = {"fc": color, "ec": "k", "linewidth": 1}
        cls._dplot["hor"]["dP"] = {"fc": color, "ec": "none"}
        cls._dplot["3d"]["dP"]["color"] = color

    def _set_color(self, color=None):
        color = self._checkformat_inputs_dmisc(color=color)
        self._dmisc["color"] = color
        self._dplot["cross"]["dP"]["fc"] = color
        self._dplot["hor"]["dP"]["fc"] = color
        self._dplot["3d"]["dP"]["color"] = color

    def get_sampleV(self, *args, **kwdargs):
        msg = "StructOut subclasses cannot use get_sampleV()!"
        raise Exception(msg)


class PlasmaDomain(StructIn):
    _color = (0.8, 0.8, 0.8, 1.0)


class Ves(StructIn):
    _color = "k"


class PFC(StructOut):
    _color = (0.8, 0.8, 0.8, 0.8)


class CoilPF(StructOut):
    _color = "r"

    def __init__(
        self,
        nturns=None,
        superconducting=None,
        active=None,
        temperature_nominal=None,
        mag_field_max=None,
        current_lim_max=None,
        **kwdargs
    ):
        # super()
        super(CoilPF, self).__init__(
            nturns=nturns,
            superconducting=superconducting,
            active=active,
            **kwdargs,
        )

    def _reset(self):
        # super()
        super(CoilPF, self)._reset()
        self._dmag = dict.fromkeys(self._get_keys_dmag())
        self._dmag["nI"] = 0

    ###########
    # Get largs
    ###########

    @staticmethod
    def _get_largs_dmag():
        largs = ["nturns", "superconducting", "active"]
        return largs

    ###########
    # Get check and format inputs
    ###########

    @classmethod
    def _checkformat_inputs_dmag(
        cls,
        nturns=None,
        superconducting=None,
        temperature_nominal=None,
        mag_field_max=None,
        current_lim_max=None,
        active=None,
    ):
        dins = {
            "nturns": {"var": nturns, "NoneOrFloatPos": None},
            "superconducting": {"var": superconducting, "NoneOrCls": bool},
            "active": {"var": active, "NoneOrCls": bool},
            "temperature_nominal": {"var": temperature_nominal,
                                    "NoneOrFloatPos": None},
            "mag_field_max": {"var": mag_field_max,
                              "NoneOrFloatPos": None},
            "current_lim_max": {"var": current_lim_max,
                                "NoneOrFloatPos": None},
        }
        dins, err, msg = cls._check_InputsGeneric(dins, tab=0)
        if err:
            raise Exception(msg)
        return [dins[dd]['var']
                for dd in ['nturns', 'superconducting', 'active']]

    ###########
    # Get keys of dictionnaries
    ###########

    @staticmethod
    def _get_keys_dmag():
        lk = ["nturns", "superconducting", "active", "current", "nI"]
        return lk

    ###########
    # _init
    ###########

    def _init(self, nturns=None, superconducting=None, active=None, **kwdargs):
        super(CoilPF, self)._init(**kwdargs)
        self.set_dmag(
            nturns=nturns, superconducting=superconducting, active=active
        )

    ###########
    # set dictionaries
    ###########

    def set_dmag(self, superconducting=None, nturns=None, active=None):
        out = self._checkformat_inputs_dmag(
            nturns=nturns, active=active, superconducting=superconducting
        )
        self._dmag.update(
            {
                "nturns": out[0],
                "superconducting": out[1],
                "active": out[2],
            }
        )

    ###########
    # strip dictionaries
    ###########

    def _strip_dmag(self, lkeep=["nturns", "superconducting", "active"]):
        utils.ToFuObject._strip_dict(self._dmag, lkeep=lkeep)
        self._dmag["nI"] = 0

    ###########
    # rebuild dictionaries
    ###########

    def _rebuild_dmag(self, lkeep=["nturns", "superconducting", "active"]):
        self.set_dmag(
            nturns=self.nturns,
            active=self._dmag["active"],
            superconducting=self._dmag["superconducting"],
        )

    ###########
    # _strip and get/from dict
    ###########

    @classmethod
    def _strip_init(cls):
        cls._dstrip["allowed"] = [0, 1, 2]
        nMax = max(cls._dstrip["allowed"])
        doc = """
                 1: Remove dsino and dmag expendables
                 2: Remove also dgeom, dphys and dmisc expendables"""
        doc = utils.ToFuObjectBase.strip.__doc__.format(doc, nMax)
        cls.strip.__doc__ = doc

    def strip(self, strip=0):
        super(CoilPF, self).strip(strip=strip)

    def _strip(self, strip=0):
        out = super(CoilPF, self)._strip(strip=strip)
        if strip == 0:
            self._rebuild_dmag()
        else:
            self._strip_dmag()
        return out

    def _to_dict(self):
        dout = super(CoilPF, self)._to_dict()
        dout.update({"dmag": {"dict": self.dmag, "lexcept": None}})
        return dout

    def _from_dict(self, fd):
        super(CoilPF, self)._from_dict(fd)
        self._dmag.update(**fd["dmag"])

    ###########
    # Properties
    ###########

    @property
    def dmag(self):
        return self._dmag

    @property
    def nturns(self):
        return self._dmag["nturns"]

    @property
    def current(self):
        return self._dmag["current"]

    ###########
    # public methods
    ###########

    def get_summary(
        self,
        sep="  ",
        line="-",
        just="l",
        table_sep=None,
        verb=True,
        return_=False,
    ):
        """ Summary description of the object content """

        # -----------------------
        # Build detailed view
        col0 = [
            "class",
            "Name",
            "SaveName",
            "nP",
            "noccur",
            "nturns",
            "active",
            "superconducting",
        ]
        ar0 = [
            self._Id.Cls,
            self._Id.Name,
            self._Id.SaveName,
            str(self._dgeom["nP"]),
            str(self._dgeom["noccur"]),
            str(self._dmag['nturns']),
            str(self._dmag['active']),
            str(self._dmag['superconducting']),
        ]
        if self._dgeom["move"] is not None:
            col0 += ['move', 'param']
            ar0 += [self._dgeom["move"],
                    str(round(self._dgeom["move_param"], ndigits=4))]
        col0.append('color')
        cstr = ('('
                + ', '.join(['{:4.2}'.format(cc)
                             for cc in self._dmisc["color"]])
                + ')')
        ar0.append(cstr)

        return self._get_summary(
            [ar0],
            [col0],
            sep=sep,
            line=line,
            table_sep=table_sep,
            verb=verb,
            return_=return_,
        )

    def set_current(self, current=None):
        """ Set the current circulating on the coil (A) """
        C0 = current is None
        C1 = type(current) in [int, float, np.int64, np.float64]
        C2 = type(current) in [list, tuple, np.ndarray]
        msg = "Arg current must be None, a float or an 1D np.ndarray !"
        assert C0 or C1 or C2, msg
        if C1:
            current = np.array([current], dtype=float)
        elif C2:
            current = np.asarray(current, dtype=float).ravel()
        self._dmag["current"] = current
        if C0:
            self._dmag["nI"] = 0
        else:
            self._dmag["nI"] = current.size


class CoilCS(CoilPF):
    pass


"""
###############################################################################
###############################################################################
                        Overall Config object
###############################################################################
"""


class Config(utils.ToFuObject):

    # Special dict subclass with attr-like value access

    # Fixed (class-wise) dictionary of default properties
    _ddef = {'Id': {'shot': 0, 'Type': 'Tor', 'Exp': 'Dummy',
                    'include': ['Mod', 'Cls', 'Exp',
                                'Name', 'shot', 'version']},
             'dStruct': {'order': ['PlasmaDomain',
                                   'Ves',
                                   'PFC',
                                   'CoilPF',
                                   'CoilCS'],
                         'dextraprop': {'visible': True}}}
    _lclsstr = ['PlasmaDomain', 'Ves', 'PFC', 'CoilPF', 'CoilCS']

    def __init__(self, lStruct=None, Lim=None, dextraprop=None,
                 Id=None, Name=None, Exp=None, shot=None, Type=None,
                 SavePath=os.path.abspath('./'),
                 SavePath_Include=tfpf.defInclude,
                 fromdict=None, sep=None):
        kwdargs = locals()
        del kwdargs["self"]
        super(Config, self).__init__(**kwdargs)

    def _reset(self):
        super(Config, self)._reset()
        self._dStruct = dict.fromkeys(self._get_keys_dStruct())
        self._dextraprop = dict.fromkeys(self._get_keys_dextraprop())
        self._dsino = dict.fromkeys(self._get_keys_dsino())

    @classmethod
    def _checkformat_inputs_Id(
        cls,
        Id=None,
        Name=None,
        Type=None,
        Exp=None,
        shot=None,
        include=None,
        **kwdargs
    ):
        if Id is not None:
            assert isinstance(Id, utils.ID)
            Name, shot = Id.Name, Id.shot
        if Type is None:
            Type = cls._ddef["Id"]["Type"]
        if Exp is None:
            Exp = cls._ddef["Id"]["Exp"]
        if shot is None:
            shot = cls._ddef["Id"]["shot"]
        if include is None:
            include = cls._ddef["Id"]["include"]

        dins = {
            "Name": {"var": Name, "cls": str},
            "Type": {"var": Type, "in": ["Tor", "Lin"]},
            "Exp": {"var": Exp, "cls": str},
            "shot": {"var": shot, "cls": int},
            "include": {"var": include, "listof": str},
        }
        dins, err, msg = cls._check_InputsGeneric(dins, tab=0)
        if err:
            raise Exception(msg)
        kwdargs.update(
            {
                "Name": Name,
                "Type": Type,
                "Exp": Exp,
                "include": include,
                "shot": shot,
            }
        )

        return kwdargs

    ###########
    # Get largs
    ###########

    @staticmethod
    def _get_largs_dStruct():
        largs = ["lStruct", "Lim"]
        return largs

    @staticmethod
    def _get_largs_dextraprop():
        largs = ["dextraprop"]
        return largs

    @staticmethod
    def _get_largs_dsino():
        largs = ["RefPt", "nP"]
        return largs

    ###########
    # Get check and format inputs
    ###########

    def _checkformat_inputs_Struct(self, struct, err=True):
        msgi = None
        c0 = issubclass(struct.__class__, Struct)
        if not c0:
            msgi = "\n\t- Not a struct subclass: {}".format(type(struct))
        else:
            c1 = struct.Id.Exp == self.Id.Exp
            c2 = struct.Id.Type == self.Id.Type
            c3 = struct.Id.Name.isidentifier()
            c4 = c3 and '_' not in struct.Id.Name
            if not (c0 and c1 and c2 and c3):
                c1 = struct.Id.Exp == self.Id.Exp
                c2 = struct.Id.Type == self.Id.Type
                c3 = struct.Id.Name.isidentifier()
                c4 = c3 and '_' not in struct.Id.Name
                msgi = "\n\t- {0} :".format(struct.Id.SaveName)
                if not c1:
                    msgi += "\n\tExp: {0}".format(struct.Id.Exp)
                if not c2:
                    msgi += "\n\tType: {0}".format(struct.Id.Type)
                if not c3:
                    msgi += "\n\tName: {0}".format(struct.Id.Name)
        if msgi is not None and err is True:
            msg = "Non-conform struct:" + msgi
            raise Exception(msg)
        return msgi

    @staticmethod
    def _errmsg_dStruct(lStruct):
        ls = ["tf.geom.{}".format(ss)
              for ss in ["PlasmaDomain", "Ves", "PFC", "CoilPF", "CoilCS"]]
        msg = ("Arg lStruct must be "
               + "a tofu.geom.Struct subclass or list of such!\n"
               + "Valid subclasses include:\n\t- "
               + "\n\t- ".join(ls)
               + "\nYou provided: {}".format(type(lStruct)))
        return msg

    def _checkformat_inputs_dStruct(self, lStruct=None, Lim=None):
        c0 = lStruct is not None
        c1 = ((isinstance(lStruct, list) or isinstance(lStruct, tuple))
              and all([issubclass(ss.__class__, Struct) for ss in lStruct]))
        c2 = issubclass(lStruct.__class__, Struct)
        if not (c0 and (c1 or c2)):
            raise Exception(self._errmsg_dStruct(lStruct))

        if c1 and isinstance(lStruct, tuple):
            lStruct = list(lStruct)
        elif c2:
            lStruct = [lStruct]

        msg = ""
        for ss in lStruct:
            msgi = self._checkformat_inputs_Struct(ss, err=False)
            if msgi is not None:
                msg += msgi
        if msg != "":
            msg = "The following objects have non-confrom Id:" + msg
            msg += "\n  => Expected values are:"
            msg += "\n      Exp: {0}".format(self.Id.Exp)
            msg += "\n      Type: {0}".format(self.Id.Type)
            msg += "\n      Name: a valid identifier, without '_'"
            msg += " (check str.isidentifier())"
            raise Exception(msg)

        if Lim is None:
            if not self.Id.Type == "Tor":
                msg = "Issue with tf.geom.Config {0}:".format(self.Id.Name)
                msg += "\n  If input Lim is None, Type should be 'Tor':"
                msg += "\n    Type = {0}".format(self.Id.Type)
                msg += "\n    Lim = {0}".format(str(Lim))
                raise Exception(msg)
            nLim = 0
        else:
            if not self.Id.Type == "Lin":
                msg = "Issue with tf.geom.Config {0}:".format(self.Id.Name)
                msg = "  If input Lim!=None, Type should be 'Lin':"
                msg += "\n    Type = {0}".format(self.Id.Type)
                msg += "\n    Lim = {0}".format(str(Lim))
                raise Exception(msg)
            Lim = np.asarray(Lim, dtype=float).ravel()
            assert Lim.size == 2 and Lim[0] < Lim[1]
            Lim = Lim.reshape((1, 2))
            nLim = 1

        return lStruct, Lim, nLim

    def _checkformat_inputs_extraval(
        self, extraval, key="", multi=True, size=None
    ):
        lsimple = [bool, float, int, np.int_, np.float64]
        C0 = type(extraval) in lsimple
        C1 = isinstance(extraval, np.ndarray)
        C2 = isinstance(extraval, dict)
        if multi:
            assert C0 or C1 or C2, str(type(extraval))
        else:
            assert C0, str(type(extraval))
        if multi and C1:
            size = self._dStruct["nObj"] if size is None else size
            C = extraval.shape == ((self._dStruct["nObj"],))
            if not C:
                msg = "The value for %s has wrong shape!" % key
                msg += "\n    Expected: ({0},)".format(self._dStruct["nObj"])
                msg += "\n    Got:      {0}".format(str(extraval.shape))
                raise Exception(msg)
            C = np.ndarray
        elif multi and C2:
            msg0 = "If an extra attribute is provided as a dict,"
            msg0 += " it should have the same structure as self.dStruct !"
            lk = sorted(self._dStruct["lCls"])
            # removing empty dict first
            extraval = {k0: v0 for k0, v0 in extraval.items() if len(v0) > 0}
            c = lk == sorted(extraval.keys())
            if not c:
                msg = "\nThe value for %s has wrong keys !" % key
                msg += "\n    expected : " + str(lk)
                msg += "\n    received : " + str(sorted(extraval.keys()))
                raise Exception(msg0 + msg)
            c = [isinstance(extraval[k], dict) for k in lk]
            if not all(c):
                msg = (
                    "\nThe value for %s shall be a dict of nested dict !" % key
                )
                msg += "\n    "
                msg += "\n    ".join(
                    [
                        "{0} : {1}".format(lk[ii], c[ii])
                        for ii in range(0, len(lk))
                    ]
                )
                raise Exception(msg0 + msg)
            c = [
                (k, sorted(v.keys()), sorted(self.dStruct["dObj"][k].keys()))
                for k, v in extraval.items()
            ]
            if not all([cc[1] == cc[2] for cc in c]):
                lc = [
                    (cc[0], str(cc[1]), str(cc[2]))
                    for cc in c
                    if cc[1] != cc[2]
                ]
                msg = "\nThe value for %s has wrong nested dict !" % key
                msg += "\n    - " + "\n    - ".join(
                    [" ".join(cc) for cc in lc]
                )
                raise Exception(msg0 + msg)
            for k in lk:
                for kk, v in extraval[k].items():
                    if not type(v) in lsimple:
                        msg = "\n    type(%s[%s][%s])" % (key, k, kk)
                        msg += " = %s" % str(type(v))
                        msg += " should be in %s" % str(lsimple)
                        raise Exception(msg)
            C = dict
        elif C0:
            C = int
        return C

    def _checkformat_inputs_dextraprop(self, dextraprop=None):
        if dextraprop is None:
            dextraprop = self._ddef["dStruct"]["dextraprop"]
        if dextraprop is None:
            dextraprop = {}
        assert isinstance(dextraprop, dict)
        dC = {}
        for k in dextraprop.keys():
            dC[k] = self._checkformat_inputs_extraval(dextraprop[k], key=k)
        return dextraprop, dC

    def _checkformat_inputs_dsino(self, RefPt=None, nP=None):
        assert type(nP) is int and nP > 0
        assert hasattr(RefPt, "__iter__")
        RefPt = np.asarray(RefPt, dtype=float).flatten()
        assert RefPt.size == 2, "RefPt must be of size=2 !"
        return RefPt

    ###########
    # Get keys of dictionnaries
    ###########

    @staticmethod
    def _get_keys_dStruct():
        lk = ["dObj", "Lim", "nLim", "nObj", "lorder", "lCls"]
        return lk

    @staticmethod
    def _get_keys_dextraprop():
        lk = ["lprop"]
        return lk

    @staticmethod
    def _get_keys_dsino():
        lk = ["RefPt", "nP"]
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
        self._dstrip["strip"] = 0

    ###########
    # set dictionaries
    ###########

    def _set_dStruct(self, lStruct=None, Lim=None):
        lStruct, Lim, nLim = self._checkformat_inputs_dStruct(
            lStruct=lStruct, Lim=Lim
        )
        self._dStruct.update({"Lim": Lim, "nLim": nLim})
        self._set_dlObj(lStruct, din=self._dStruct)

    def _set_dextraprop(self, dextraprop=None):
        dextraprop, dC = self._checkformat_inputs_dextraprop(dextraprop)
        self._dextraprop["lprop"] = sorted(list(dextraprop.keys()))

        # Init dict
        lCls = self._dStruct["lCls"]
        for pp in dextraprop.keys():
            dp = "d" + pp
            dd = dict.fromkeys(lCls, {})
            for k in lCls:
                dd[k] = dict.fromkeys(self._dStruct["dObj"][k].keys())
            self._dextraprop.update({dp: dd})

        # Populate
        for pp in dextraprop.keys():
            self._set_extraprop(pp, dextraprop[pp])

    def add_extraprop(self, key, val):
        assert type(key) is str
        d, dC = self._checkformat_inputs_dextraprop({key: val})
        self._dextraprop["lprop"] = sorted(
            set(self.dextraprop["lprop"] + [key])
        )

        # Init dict
        lCls = self._dStruct["lCls"]
        dp = "d" + key
        dd = dict.fromkeys(lCls, {})
        for k in lCls:
            dd[k] = dict.fromkeys(self._dStruct["dObj"][k].keys())
        self._dextraprop.update({dp: dd})

        # Populate
        self._set_extraprop(key, val)
        self._dynamicattr()

    def _set_extraprop(self, pp, val, k0=None, k1=None):
        assert not (k0 is None and k1 is not None)
        dp = "d" + pp
        if k0 is None and k1 is None:
            C = self._checkformat_inputs_extraval(val, pp)
            if C is int:
                for k0 in self._dStruct["dObj"].keys():
                    for k1 in self._dextraprop[dp][k0].keys():
                        self._dextraprop[dp][k0][k1] = val
            elif C is np.ndarray:
                ii = 0
                for k in self._dStruct["lorder"]:
                    k0, k1 = k.split("_")
                    self._dextraprop[dp][k0][k1] = val[ii]
                    ii += 1
            else:
                for k0 in self._dStruct["dObj"].keys():
                    if k0 in self._dextraprop[dp].keys():
                        for k1 in self._dextraprop[dp][k0].keys():
                            self._dextraprop[dp][k0][k1] = val[k0][k1]
        elif k1 is None:
            size = len(self._dextraprop[dp][k0].keys())
            C = self._checkformat_inputs_extraval(val, pp, size=size)
            assert C in [int, np.ndarray]
            if C is int:
                for k1 in self._dextraprop[dp][k0].keys():
                    self._dextraprop[dp][k0][k1] = val
            elif C is np.ndarray:
                ii = 0
                for k in self._dStruct["lorder"]:
                    kk, k1 = k.split("_")
                    if k0 == kk:
                        self._dextraprop[dp][k0][k1] = val[ii]
                        ii += 1
        else:
            C = self._checkformat_inputs_extraval(val, pp, multi=False)
            assert C is int
            self._dextraprop[dp][k0][k1] = val

    def _get_extraprop(self, pp, k0=None, k1=None):
        assert not (k0 is None and k1 is not None)
        dp = "d" + pp
        if k0 is None and k1 is None:
            k0, k1 = self._dStruct["lorder"][0].split('_')
            val = np.zeros((self._dStruct["nObj"],),
                           dtype=type(self._dextraprop[dp][k0][k1]))
            ii = 0
            for k in self._dStruct["lorder"]:
                k0, k1 = k.split("_")
                val[ii] = self._dextraprop[dp][k0][k1]
                ii += 1
        elif k1 is None:
            k1 = list(self._dStruct["dObj"][k0].keys())[0]
            val = np.zeros((len(self._dStruct["dObj"][k0].keys()),),
                           dtype=type(self._dextraprop[dp][k0][k1]))
            ii = 0
            for k in self._dStruct["lorder"]:
                k, k1 = k.split("_")
                if k0 == k:
                    val[ii] = self._dextraprop[dp][k0][k1]
                    ii += 1
        else:
            val = self._dextraprop[dp][k0][k1]
        return val

    def _set_color(self, k0, val):
        for k1 in self._dStruct["dObj"][k0].keys():
            self._dStruct["dObj"][k0][k1].set_color(val)

    def _dynamicattr(self):
        # get (key, val) pairs

        # Purge
        for k in self._ddef['dStruct']['order']:
            if hasattr(self, k):
                delattr(self, k)
                # exec("del self.{0}".format(k))

        # Set
        for k in self._dStruct["dObj"].keys():

            if len(self._dStruct["dObj"][k]) == 0:
                continue

            # Find a way to programmatically add dynamic properties to the
            # instances , like visible
            # In the meantime use a simple functions
            lset = ["set_%s" % pp for pp in self._dextraprop["lprop"]]
            lget = ["get_%s" % pp for pp in self._dextraprop["lprop"]]
            if not type(list(self._dStruct["dObj"][k].values())[0]) is str:
                for kk in self._dStruct["dObj"][k].keys():
                    for pp in self._dextraprop["lprop"]:
                        setattr(
                            self._dStruct["dObj"][k][kk],
                            "set_%s" % pp,
                            lambda val, pk=pp, k0=k, k1=kk: (
                                self._set_extraprop(pk, val, k0, k1)
                            ),
                        )
                        setattr(
                            self._dStruct["dObj"][k][kk],
                            "get_%s" % pp,
                            lambda pk=pp, k0=k, k1=kk: self._get_extraprop(
                                pk, k0, k1
                            ),
                        )
                dd = utils.Dictattr(
                    ["set_color"] + lset + lget, self._dStruct["dObj"][k]
                )
                for pp in self._dextraprop["lprop"]:
                    setattr(
                        dd,
                        "set_%s" % pp,
                        lambda val, pk=pp, k0=k: self._set_extraprop(
                            pk, val, k0
                        ),
                    )
                    setattr(
                        dd,
                        "get_%s" % pp,
                        lambda pk=pp, k0=k: self._get_extraprop(pk, k0),
                    )
                setattr(
                    dd, "set_color", lambda col, k0=k: self._set_color(k0, col)
                )
                setattr(self, k, dd)
        for pp in self._dextraprop["lprop"]:
            setattr(
                self,
                "set_%s" % pp,
                lambda val, pk=pp: self._set_extraprop(pk, val),
            )
            setattr(self, "get_%s" % pp, lambda pk=pp: self._get_extraprop(pk))

    def set_dsino(self, RefPt, nP=_def.TorNP):
        RefPt = self._checkformat_inputs_dsino(RefPt=RefPt, nP=nP)
        for k in self._dStruct["dObj"].keys():
            for kk in self._dStruct["dObj"][k].keys():
                self._dStruct["dObj"][k][kk].set_dsino(RefPt=RefPt, nP=nP)
        self._dsino = {"RefPt": RefPt, "nP": nP}

    ###########
    # strip dictionaries
    ###########

    def _strip_dStruct(self, strip=0, force=False, verb=True):
        if self._dstrip["strip"] == strip:
            return

        if self._dstrip["strip"] > strip:

            # Reload if necessary
            if self._dstrip["strip"] == 3:
                for k in self._dStruct["dObj"].keys():
                    for kk in self._dStruct["dObj"][k].keys():
                        pfe = self._dStruct["dObj"][k][kk]
                        try:
                            self._dStruct["dObj"][k][kk] = utils.load(
                                pfe, verb=verb
                            )
                        except Exception as err:
                            msg = str(err)
                            msg += "\n    k = {0}".format(str(k))
                            msg += "\n    kk = {0}".format(str(kk))
                            msg += "\n    type(pfe) = {0}".format(
                                str(type(pfe))
                            )
                            msg += "\n    self._dstrip['strip'] = {0}".format(
                                self._dstrip["strip"]
                            )
                            msg += "\n    strip = {0}".format(strip)
                            raise Exception(msg)

            for k in self._dStruct["dObj"].keys():
                for kk in self._dStruct["dObj"][k].keys():
                    self._dStruct["dObj"][k][kk].strip(strip=strip)

            lkeep = self._get_keys_dStruct()
            reset = utils.ToFuObject._test_Rebuild(self._dStruct, lkeep=lkeep)
            if reset:
                utils.ToFuObject._check_Fields4Rebuild(
                    self._dStruct, lkeep=lkeep, dname="dStruct"
                )
            self._set_dStruct(lStruct=self.lStruct, Lim=self._dStruct["Lim"])
            self._dynamicattr()

        else:
            if strip in [1, 2]:
                for k in self._dStruct["lCls"]:
                    for kk, v in self._dStruct["dObj"][k].items():
                        self._dStruct["dObj"][k][kk].strip(strip=strip)
                lkeep = self._get_keys_dStruct()

            elif strip == 3:
                for k in self._dStruct["lCls"]:
                    for kk, v in self._dStruct["dObj"][k].items():
                        path, name = v.Id.SavePath, v.Id.SaveName
                        # --- Check !
                        lf = os.listdir(path)
                        lf = [
                            ff
                            for ff in lf
                            if all([s in ff for s in [name, ".npz"]])
                        ]
                        exist = len(lf) == 1
                        # ----------
                        pathfile = os.path.join(path, name) + ".npz"
                        if not exist:
                            msg = """BEWARE:
                                You are about to delete the Struct objects
                                Only the path/name to saved objects will be
                                kept

                                But it appears that the following object has no
                                saved file where specified (obj.Id.SavePath)
                                Thus it won't be possible to retrieve it
                                (unless available in the current console:"""
                            msg += "\n    - {0}".format(pathfile)
                            if force:
                                warnings.warn(msg)
                            else:
                                raise Exception(msg)
                        self._dStruct["dObj"][k][kk] = pathfile
                self._dynamicattr()
                lkeep = self._get_keys_dStruct()
            utils.ToFuObject._strip_dict(self._dStruct, lkeep=lkeep)

    def _strip_dextraprop(self, strip=0):
        lkeep = list(self._dextraprop.keys())
        utils.ToFuObject._strip_dict(self._dextraprop, lkeep=lkeep)

    def _strip_dsino(self, lkeep=["RefPt", "nP"]):
        for k in self._dStruct["dObj"].keys():
            for kk in self._dStruct["dObj"][k].keys():
                self._dStruct["dObj"][k][kk]._strip_dsino(lkeep=lkeep)

    ###########
    # _strip and get/from dict
    ###########

    @classmethod
    def _strip_init(cls):
        cls._dstrip["allowed"] = [0, 1, 2, 3]
        nMax = max(cls._dstrip["allowed"])
        doc = """
                 1: apply strip(1) to objects in self.lStruct
                 2: apply strip(2) to objects in self.lStruct
                 3: replace objects in self.lStruct by SavePath + SaveName"""
        doc = utils.ToFuObjectBase.strip.__doc__.format(doc, nMax)
        cls.strip.__doc__ = doc

    def strip(self, strip=0, force=False, verb=True):
        # super()
        super(Config, self).strip(strip=strip, force=force, verb=verb)

    def _strip(self, strip=0, force=False, verb=True):
        self._strip_dStruct(strip=strip, force=force, verb=verb)
        # self._strip_dextraprop()
        # self._strip_dsino()

    def _to_dict(self):
        dout = {
            "dStruct": {"dict": self.dStruct, "lexcept": None},
            "dextraprop": {"dict": self._dextraprop, "lexcept": None},
            "dsino": {"dict": self.dsino, "lexcept": None},
        }
        return dout

    @classmethod
    def _checkformat_fromdict_dStruct(cls, dStruct):
        if dStruct["lorder"] is None:
            return None
        for clsn in dStruct["lorder"]:
            c, n = clsn.split("_")
            if type(dStruct["dObj"][c][n]) is dict:
                dStruct["dObj"][c][n] = eval(c).__call__(
                    fromdict=dStruct["dObj"][c][n]
                )
            lC = [
                issubclass(dStruct["dObj"][c][n].__class__, Struct),
                type(dStruct["dObj"][c][n]) is str,
            ]
            assert any(lC)

    def _from_dict(self, fd):
        self._checkformat_fromdict_dStruct(fd["dStruct"])

        self._dStruct.update(**fd["dStruct"])
        self._dextraprop.update(**fd["dextraprop"])
        self._dsino.update(**fd["dsino"])
        self._dynamicattr()

    ###########
    # SOLEDGE3X
    ###########

    @staticmethod
    def _from_SOLEDGE_extract_dict(pfe=None):
        # Check input
        c0 = (isinstance(pfe, str)
              and os.path.isfile(pfe)
              and pfe[-4:] == '.mat')
        if not c0:
            msg = ("Arg pfe must be a valid .mat file!\n"
                   + "\t- provided: {}".format(pfe))
            raise Exception(msg)
        pfe = os.path.abspath(pfe)

        # Open file
        import scipy.io as scpio
        dout = scpio.loadmat(pfe)

        # Check conformity of content
        lk = ['Nwalls', 'coord', 'type']
        lk0 = [kk for kk, vv in dout.items()
               if kk == 'walls' and isinstance(vv, np.ndarray)]
        c0 = (len(lk0) == 1
              and len(dout['walls']) == 1
              and sorted(dout['walls'][0].dtype.names) == lk
              and len(dout['walls'][0][0]) == len(lk))
        if not c0:
            msg = ("Non-conform .mat file content from SOLEDGE3X:\n"
                   + "\t- file: {}\n".format(pfe)
                   + "\t- Expected:\n"
                   + "\t\t- a unique matlab structure 'walls' with 3 fields\n"
                   + "\t\t\t- Nwalls: int\n"
                   + "\t\t\t- coord: 1xn struct\n"
                   + "\t\t\t- type: 1xn double\n"
                   + "Provided:\n"
                   + "\t- variables: {}\n".format(lk0)
                   + "\t- 1x{} struct with {} fields".format(
                       len(dout[lk0[0]]),
                       len(dout[lk0[0]][0].dtype)))
            raise Exception(msg)
        out = dout['walls'][0][0]

        # Get inside fields 'type', 'Nwalls', 'coord'
        di0 = {kk: dout['walls'][0].dtype.names.index(kk) for kk in lk}
        dout = {'type': out[di0['type']].ravel(),
                'Nwalls': out[di0['Nwalls']][0, 0]}
        out = out[di0['coord']][0]
        c0 = (sorted(out.dtype.names) == ['Rwall', 'Zwall']
              and len(out) == dout['type'].size
              and all([len(oo) == 2 for oo in out]))
        if not c0:
            msg = ("Field {} not conform:\n".format('coord')
                   + "\t- expected: 1x{} struct ".format(dout['type'].size)
                   + "with fields ('Rwall', 'Zwall')\n"
                   + "\t- provided: 1x{} struct ".format(len(out))
                   + "with fields {}".format(out.dtype.names))
            raise Exception(msg)

        dout['coord'] = [np.array([out[ii][0].ravel(), out[ii][1].ravel()])
                         for ii in range(dout['type'].size)]
        return dout

    @classmethod
    def from_SOLEDGE3X(cls, pfe=None,
                       Name=None, Exp=None):

        # Check input and extract dict from file
        dout = cls._from_SOLEDGE_extract_dict(pfe)
        npoly = len(dout['type'])

        # Prepare lStruct
        lcls = [Ves if dout['type'][ii] == 1 else PFC for ii in range(npoly)]
        lnames = ['Soledge3X{:02.0f}'.format(ii) for ii in range(npoly)]
        lS = [lcls[ii](Poly=dout['coord'][ii],
                       Type='Tor',
                       Name=lnames[ii],
                       pos=None,
                       Exp=Exp)
              for ii in range(npoly)]
        return cls(lStruct=lS, Exp=Exp, Name=Name)

    def _to_SOLEDGE3X_get_data(self,
                               type_extraprop=None,
                               matlab_version=None, matlab_platform=None):

        head = None
        # Check inputs
        if not (matlab_version is None or isinstance(matlab_version, str)):
            msg = ("Arg matlab_version must be provided as a str!\n"
                   + "\t- example: '5.0'\n"
                   + "\t- provided: {}".format(matlab_version))
            raise Exception(msg)

        # useful ? to be deprecated ?
        if matlab_platform is None:
            out = os.popen('which matlab').read()
            keypath = os.path.join('bin', 'matlab')
            if keypath in out:
                path = os.path.join(out[:out.index(keypath)], 'etc')
                lf = [ff for ff in os.listdir(path)
                      if os.path.isdir(os.path.join(path, ff))]
                if len(lf) == 1:
                    matlab_platform = lf[0].upper()
                else:
                    msg = ("Couldn't get matlab_platform from 'which matlab'\n"
                           + "  => Please provide the matlab platform\n"
                           + "     Should be in {}/../etc".format(out))
                    warnings.warn(msg)
        if not (matlab_platform is None or isinstance(matlab_platform, str)):
            msg = ("Arg matlab_platform must be provided as a str!\n"
                   + "\t- example: 'GLNXA64'\n"
                   + "\t- provided: {}".format(matlab_platform))
            raise Exception(msg)

        if matlab_version is not None and matlab_platform is not None:
            import datetime as dtm
            now = dtm.datetime.now().strftime('%a %b %d %H:%M:%S %Y')
            head = ('MATLAB {} MAT-file, '.format(matlab_version)
                    + 'Platform: {}, '.format(matlab_platform)
                    + 'Created on: {}'.format(now))

        # Build walls
        nwall = np.array([[self.nStruct]], dtype=np.int64)

        # typ (from extraprop if any, else from Ves / Struct)
        if type_extraprop is not None:
            typ = np.array([self._get_extraprop(type_extraprop)], dtype=np.int64)
        else:
            typ = np.array([[1 if ss._InOut == 'in' else -1
                             for ss in self.lStruct]], dtype=np.int64)
        # Get coord
        coord = np.array([np.array([
            (ss.Poly[0:1, :].T, ss.Poly[1:2, :].T) for ss in self.lStruct],
            dtype=[('Rwall', 'O'), ('Zwall', 'O')])],
            dtype=[('Rwall', 'O'), ('Zwall', 'O')])

        # put together
        dout = {'walls': np.array([[
            (nwall, coord, typ)]],
            dtype=[('Nwalls', 'O'), ('coord', 'O'), ('type', 'O')])}

        # Optinally set header and version
        if head is not None:
            dout['__header__'] = head.encode()
        return dout

    def to_SOLEDGE3X(self, name=None, path=None, verb=None,
                     type_extraprop=None,
                     matlab_version=None, matlab_platform=None):

        # Check inputs
        if verb is None:
            verb = True
        if name is None:
            name = self.Id.SaveName
        if not isinstance(name, str):
            msg = ("Arg name must be a str!\n"
                   + "\t- provided: {}".format(name))
            raise Exception(msg)
        if name[-4:] != '.mat':
            name = name + '.mat'

        if path is None:
            path = os.path.abspath('.')
        if not os.path.isdir(path):
            msg = ("Provided path is not a valid dir!\n"
                   + "\t- path: {}".format(path))
            raise Exception(msg)
        path = os.path.abspath(path)

        pfe = os.path.join(path, name)

        # Get data in proper shape
        dout = self._to_SOLEDGE3X_get_data(type_extraprop=type_extraprop,
                                           matlab_version=matlab_version,
                                           matlab_platform=matlab_platform)
        # save
        import scipy.io as scpio
        scpio.savemat(pfe, dout)
        if verb is True:
            print("Saved in:\n\t{}".format(pfe))

    ###########
    # Properties
    ###########

    @property
    def dStruct(self):
        return self._dStruct

    @property
    def nStruct(self):
        return self._dStruct["nObj"]

    @property
    def lStruct(self):
        """ Return the list of Struct that was used for creation

        As tofu objects or SavePath+SaveNames (according to strip status)
        """
        lStruct = []
        for k in self._dStruct["lorder"]:
            k0, k1 = k.split("_")
            lStruct.append(self._dStruct["dObj"][k0][k1])
        return lStruct

    @property
    def lStructIn(self):
        """ Return the list of StructIn contained in self.lStruct

        As tofu objects or SavePath+SaveNames (according to strip status)
        """
        lStruct = []
        for k in self._dStruct["lorder"]:
            k0, k1 = k.split("_")
            if type(self._dStruct["dObj"][k0][k1]) is str:
                if any(
                    [
                        ss in self._dStruct["dObj"][k0][k1]
                        for ss in ["Ves", "PlasmaDomain"]
                    ]
                ):
                    lStruct.append(self._dStruct["dObj"][k0][k1])
            elif issubclass(self._dStruct["dObj"][k0][k1].__class__, StructIn):
                lStruct.append(self._dStruct["dObj"][k0][k1])
        return lStruct

    @property
    def Lim(self):
        return self._dStruct["Lim"]

    @property
    def nLim(self):
        return self._dStruct["nLim"]

    @property
    def dextraprop(self):
        return self._dextraprop

    @property
    def dsino(self):
        return self._dsino

    ###########
    # public methods
    ###########

    def add_Struct(
        self,
        struct=None,
        Cls=None,
        Name=None,
        Poly=None,
        shot=None,
        Lim=None,
        Type=None,
        dextraprop=None,
    ):
        """ Add a Struct instance to the config

        An already existing Struct subclass instance can be added
        Or it will be created from the (Cls,Name,Poly,Lim) keyword args

        """
        # Check inputs
        C0a = struct is None
        C1a = all([ss is None for ss in [Cls, Name, Poly, Lim, Type]])
        if not np.sum([C0a, C1a]) == 1:
            msg = "Provide either:"
            msg += "\n    - struct: a Struct subclass instance"
            msg += "\n    - the keyword args to create one"
            msg += "\n        (Cls,Name,Poly,Lim,Type)\n"
            msg += "\n You provded:"
            msg += "\n    - struct: {0}, {1}".format(str(struct), type(struct))
            raise Exception(msg)

        # Create struct if not provided
        if C0a:
            if not (type(Cls) is str or issubclass(Cls, Struct)):
                msg = "Cls must be either:"
                msg += "\n    - a Struct subclass"
                msg += "\n    - the str Name of it (e.g.: 'PFC','CoilPF',...)"
                raise Exception(msg)
            if type(Cls) is str:
                Cls = eval("%s" % Cls)

            # Preformat Lim and Type
            if Lim is None:
                Lim = self.Lim
            if Type is None:
                Type = self.Id.Type

            # Create instance
            struct = Cls(
                Poly=Poly,
                Name=Name,
                Lim=Lim,
                Type=Type,
                shot=shot,
                Exp=self.Id.Exp,
            )

        C0b = issubclass(struct.__class__, Struct)
        assert C0b, "struct must be a Struct subclass instance !"

        # Prepare dextraprop
        dextra = self.dextraprop
        lk = sorted([k[1:] for k in dextra.keys() if k != "lprop"])
        if dextraprop is None:
            if dextra not in [None, {}]:
                msg = (
                    "The current Config instance has the following extraprop:"
                )
                msg += "\n    - " + "\n    - ".join(lk)
                msg += "\n  => Please specify a dextraprop for struct !"
                msg += "\n     (using the same keys !)"
                raise Exception(msg)
        else:
            assert isinstance(dextraprop, dict)
            assert all([k in lk for k in dextraprop.keys()])
            assert all([k in dextraprop.keys() for k in lk])
            dx = {}
            for k in lk:
                dk = "d" + k
                dx[k] = {}
                for k0 in dextra[dk].keys():
                    dx[k][k0] = {}
                    for k1 in dextra[dk][k0].keys():
                        dx[k][k0][k1] = dextra[dk][k0][k1]
                if struct.Id.Cls not in dx[k].keys():
                    dx[k][struct.Id.Cls] = {struct.Id.Name: dextraprop[k]}
                else:
                    dx[k][struct.Id.Cls][struct.Id.Name] = dextraprop[k]

        # Set self.lStruct
        lS = self.lStruct + [struct]
        self._init(lStruct=lS, Lim=self.Lim, dextraprop=dx)

    def remove_Struct(self, Cls=None, Name=None):
        # Check inputs
        assert type(Cls) is str
        assert type(Name) is str
        C0 = Cls in self._dStruct["lCls"]
        if not C0:
            msg = "The Cls must be a class existing in self.dStruct['lCls']:"
            msg += "\n    [{0}]".format(", ".join(self._dStruct["lCls"]))
            raise Exception(msg)
        C0 = Name in self._dStruct["dObj"][Cls].keys()
        if not C0:
            ln = self.dStruct["dObj"][Cls].keys()
            msg = "The Name must match an instance in"
            msg += " self.dStruct['dObj'][{0}].keys():".format(Cls)
            msg += "\n    [{0}]".format(", ".join(ln))
            raise Exception(msg)

        # Create list
        lS = self.lStruct
        if not Cls + "_" + Name in self._dStruct["lorder"]:
            msg = "The desired instance is not in self.dStruct['lorder'] !"
            lord = ", ".join(self.dStruct["lorder"])
            msg += "\n    lorder = [{0}]".format(lord)
            msg += "\n    Cls_Name = {0}".format(Cls + "_" + Name)
            raise Exception(msg)

        ind = self._dStruct["lorder"].index(Cls + "_" + Name)
        del lS[ind]
        # Important : also remove from dict ! (no reset() !)
        del self._dStruct["dObj"][Cls][Name]

        # Prepare dextraprop
        dextra = self.dextraprop
        dx = {}
        for k in dextra.keys():
            if k == "lprop":
                continue
            dx[k[1:]] = {}
            for cc in dextra[k].keys():
                dx[k[1:]][cc] = dict(dextra[k][cc])
            del dx[k[1:]][Cls][Name]

            # remove Cls if empty
            if len(dx[k[1:]][Cls]) == 0:
                del dx[k[1:]][Cls]

            # remove empty parts
            if len(dx[k[1:]]) == 0:
                del dx[k[1:]]

        self._init(lStruct=lS, Lim=self.Lim, dextraprop=dx)

    def get_color(self):
        """ Return the array of rgba colors (same order as lStruct) """
        col = np.full((self._dStruct["nObj"], 4), np.nan)
        ii = 0
        for k in self._dStruct["lorder"]:
            k0, k1 = k.split("_")
            col[ii, :] = self._dStruct["dObj"][k0][k1].get_color()
            ii += 1
        return col

    def set_colors_random(self, cmap=plt.cm.Accent):
        ii = 0
        ncol = len(cmap.colors)
        for k in self._dStruct["lorder"]:
            k0, k1 = k.split("_")
            if self._dStruct["dObj"][k0][k1]._InOut == "in":
                col = "k"
            elif "lh" in k1.lower():
                col = (1.0, 0.0, 0.0)
            elif "ic" in k1.lower():
                col = (1.0, 0.5, 0.5)
            elif "div" in k1.lower():
                col = (0.0, 1.0, 0.0)
            elif "bump" in k1.lower():
                col = (0.0, 0.0, 1.0)
            else:
                col = cmap.colors[ii % ncol]
                ii += 1
            self._dStruct["dObj"][k0][k1].set_color(col)

    def get_summary(
        self,
        sep="  ",
        line="-",
        just="l",
        table_sep=None,
        verb=True,
        return_=False,
    ):
        """ Summary description of the object content """

        # -----------------------
        # Build overview
        col0 = ["tot. Struct", "tot. occur", "tot. points"]
        noccur = np.sum([max(1, ss._dgeom["noccur"]) for ss in self.lStruct])
        npts = np.sum([ss._dgeom["nP"] for ss in self.lStruct])
        ar0 = [(self.nStruct, noccur, npts)]

        # -----------------------
        # Build detailed view
        col1 = [
            "class",
            "Name",
            "SaveName",
            "nP",
            "noccur",
            "move",
            "color",
        ] + self._dextraprop["lprop"]
        d = self._dStruct["dObj"]
        ar1 = []
        for k in self._ddef["dStruct"]["order"]:
            if k not in d.keys():
                continue
            otemp = self._dStruct["dObj"][k]
            for kk in d[k].keys():
                lu = [
                    k,
                    otemp[kk]._Id._dall["Name"],
                    otemp[kk]._Id._dall["SaveName"],
                    str(otemp[kk]._dgeom["nP"]),
                    str(otemp[kk]._dgeom["noccur"]),
                    str(otemp[kk]._dgeom["move"]),
                    ('(' + ', '.join(['{:4.2}'.format(cc)
                                      for cc in otemp[kk]._dmisc["color"]])
                     + ')'),
                ]
                for pp in self._dextraprop["lprop"]:
                    lu.append(self._dextraprop["d" + pp][k][kk])
                ar1.append(lu)

        return self._get_summary(
            [ar0, ar1],
            [col0, col1],
            sep=sep,
            line=line,
            table_sep=table_sep,
            verb=verb,
            return_=return_,
        )

    def get_reflections(self, indout, u=None, vperp=None):

        # Get global Types array
        lS = self.lStruct

        # Version only usable when indout returns npts+1 and npts+2 instead of
        # -1 and -2
        # ls = [ss._dreflect['Types'].size for ss in lS]
        # Types = np.empty((len(lS), np.max(ls)), dtype=np.int64)
        # for ii,ss in enumerate(lS):
        # Types[ii,:ls[ii]] = ss._dreflect['Types']
        # # Deduce Types
        # Types = Types[indout[0,:], indout[2,:]]

        iu = np.unique(indout[0, :])
        Types = np.empty((indout.shape[1],), dtype=np.int64)
        for ii in iu:
            ind = indout[0, :] == ii
            Types[ind] = lS[ii]._dreflect["Types"][indout[2, ind]]

        # Deduce u2
        u2 = None
        if u is not None:
            assert vperp is not None
            u2 = Struct._get_reflections_ufromTypes(u, vperp, Types)
        return Types, u2

    def _get_phithetaproj_dist(
        self, refpt=None, ntheta=None, nphi=None, theta=None, phi=None
    ):
        # Prepare repf
        if refpt is None:
            refpt = self.dsino["RefPt"]
            if refpt is None:
                msg = "Please provide refpt (R,Z)"
                raise Exception(msg)
        refpt = np.atleast_1d(np.squeeze(refpt))
        assert refpt.shape == (2,)

        # Prepare theta and phi
        if theta is None and ntheta is None:
            ntheta = _PHITHETAPROJ_NTHETA
        lc = [ntheta is None, theta is None]
        if np.sum(lc) != 1:
            msg = "Please provide either ntheta xor a theta vector !"
            raise Exception(msg)
        if theta is None:
            theta = np.linspace(-np.pi, np.pi, ntheta, endpoint=True)

        if phi is None and nphi is None:
            nphi = _PHITHETAPROJ_NPHI
        lc = [nphi is None, phi is None]
        if np.sum(lc) != 1:
            msg = "Please provide either nphi xor a phi vector !"
            raise Exception(msg)
        if phi is None:
            phi = np.linspace(-np.pi, np.pi, nphi, endpoint=True)

        # format inputs
        theta = np.atleast_1d(np.ravel(theta))
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        phi = np.atleast_1d(np.ravel(phi))
        phi = np.arctan2(np.sin(phi), np.cos(phi))
        ntheta, nphi = theta.size, phi.size

        # Get limits
        lS = self.lStruct
        dist = np.full((ntheta, nphi), np.inf)
        indStruct = np.zeros((ntheta, nphi), dtype=np.int64)
        for ii in range(0, self.nStruct):
            out = _comp._Struct_get_phithetaproj(
                refpt, lS[ii].Poly_closed, lS[ii].Lim, lS[ii].noccur
            )
            nDphi, Dphi, nDtheta, Dtheta = out

            # Get dist
            dist_theta, indphi = _comp._get_phithetaproj_dist(
                lS[ii].Poly_closed,
                refpt,
                Dtheta,
                nDtheta,
                Dphi,
                nDphi,
                theta,
                phi,
                ntheta,
                nphi,
                lS[ii].noccur,
            )
            ind = np.zeros((ntheta, nphi), dtype=bool)
            indok = ~np.isnan(dist_theta)
            ind[indok, :] = indphi[None, :]
            ind[ind] = (
                dist_theta[indok, None] < dist[indok, :][:, indphi]
            ).ravel()
            dist[ind] = (np.broadcast_to(dist_theta, (nphi, ntheta)).T)[ind]
            indStruct[ind] = ii

        dist[np.isinf(dist)] = np.nan

        return dist, indStruct

    def plot_phithetaproj_dist(self, refpt=None, ntheta=None, nphi=None,
                               theta=None, phi=None, cmap=None, invertx=None,
                               ax=None, fs=None, tit=None, wintit=None,
                               draw=None):
        dist, indStruct = self._get_phithetaproj_dist(refpt=refpt,
                                                      ntheta=ntheta, nphi=nphi,
                                                      theta=theta, phi=phi)
        return _plot.Config_phithetaproj_dist(self, refpt, dist, indStruct,
                                              cmap=cmap, ax=ax, fs=fs,
                                              tit=tit, wintit=wintit,
                                              invertx=invertx, draw=draw)

    def isInside(self, pts, In="(X,Y,Z)", log="any"):

        """ Return a 2D array of bool

        Equivalent to applying isInside to each Struct
        Check self.lStruct[0].isInside? for details

        Arg log determines how Struct with multiple Limits are treated
            - 'all' : True only if pts belong to all elements
            - 'any' : True if pts belong to any element
        """
        msg = "Arg pts must be a 1D or 2D np.ndarray !"
        assert isinstance(pts, np.ndarray) and pts.ndim in [1, 2], msg
        msg = "Arg log must be in ['any','all']"
        assert log in ["any", "all"], msg
        if pts.ndim == 1:
            msg = "Arg pts must contain the coordinates of a point !"
            assert pts.size in [2, 3], msg
            pts = pts.reshape((pts.size, 1)).astype(float)
        else:
            msg = "Arg pts must contain the coordinates of points !"
            assert pts.shape[0] in [2, 3], pts
        nP = pts.shape[1]

        ind = np.zeros((self._dStruct["nObj"], nP), dtype=bool)
        lStruct = self.lStruct
        for ii in range(0, self._dStruct["nObj"]):
            if lStruct[ii].noccur > 0:
                indi = _GG._Ves_isInside(
                    np.ascontiguousarray(pts),
                    np.ascontiguousarray(lStruct[ii].Poly),
                    ves_lims=np.ascontiguousarray(lStruct[ii].Lim),
                    nlim=lStruct[ii].noccur,
                    ves_type=lStruct[ii].Id.Type,
                    in_format=In,
                    test=True,
                )
            else:
                indi = _GG._Ves_isInside(
                    np.ascontiguousarray(pts),
                    np.ascontiguousarray(lStruct[ii].Poly),
                    ves_lims=None,
                    nlim=0,
                    ves_type=lStruct[ii].Id.Type,
                    in_format=In,
                    test=True,
                )
            if lStruct[ii].noccur > 1:
                if log == "any":
                    indi = np.any(indi, axis=0)
                else:
                    indi = np.all(indi, axis=0)
            ind[ii, :] = indi
        return ind

    # TBF
    def fdistfromwall(self, r, z, phi):
        """ Return a callable (function) for detecting trajectory collisions
        with wall

        The function is continuous wrt time and space
        It takes into account all Struct in Config, including non-axisymmetric
        ones

        It is desined for iterative root-finding algorithms and is thus called
        for a unique position

        """
        # LM: ... function NOT finished (TBF)
        # LM: ... since we are in devel this is too dangerous to keep
        # LM: ... commenting and raising warning
        # isin = [ss._InOut == "in" for ss in self.lStruct]
        # inside = self.isInside(np.r_[r, z, phi], In="(R,Z,Phi)", log="any")

        # distRZ, indStruct = self._get_phithetaproj_dist(
        #     refpt=np.r_[r, z], ntheta=ntheta, nphi=nphi, theta=theta, phi=phi
        # )
        # lSlim = [ss for ss in self.lStruct if ss.noccur > 0]
        # distPhi = r * np.min([np.min(np.abs(phi - ss.Lim)) for ss in lSlim])
        # if inside:
        #     return min(distRZ, distPhi)
        # else:
        #     return -min(distRZ, distPhi)
        warnings.warn("FUNCTION NOT DEFINED")
        return

    # Method handling reflections

    def _reflect_Types(self, indout=None, Type=None, nRays=None):
        """ Return an array indicating the Type of reflection for each LOS

        Return a (nRays,) np.ndarray of int indices, each index corresponds to:
            - 0: specular reflections
            - 1: diffusive reflections
            - 2: ccube reflections (corner cube)

        If indout is provided, the Types are computed according to the
        information stored in each corresponding Struct

        If Type is provided, the Type is forced (user-defined) for all LOS

        """
        if Type is not None:
            assert Type in ["specular", "diffusive", "ccube"]
            Types = np.full((nRays,), _DREFLECT[Type], dtype=np.int64)
        else:
            Types = self.get_reflections(indout)[0]
        return Types

    def _reflect_geom(self, u=None, vperp=None, indout=None, Type=None):
        assert u.shape == vperp.shape and u.shape[0] == 3
        if indout is not None:
            assert indout.shape == (3, u.shape[1])

        # Get Types of relection for each Ray
        Types = self._reflect_Types(indout=indout, Type=Type, nRays=u.shape[1])

        # Deduce u2
        u2 = Struct._get_reflections_ufromTypes(u, vperp, Types)
        return u2, Types

    def plot(
        self,
        lax=None,
        proj=None,
        element="P",
        dLeg=_def.TorLegd,
        indices=False,
        Lim=None,
        Nstep=None,
        draw=True,
        fs=None,
        wintit=None,
        tit=None,
        Test=True,
    ):
        assert tit in [None, False] or isinstance(tit, str)
        vis = self.get_visible()
        lStruct, lS = self.lStruct, []
        for ii in range(0, self._dStruct["nObj"]):
            if vis[ii]:
                lS.append(lStruct[ii])

        if tit is None:
            tit = self.Id.Name
        lax = _plot.Struct_plot(
            lS,
            lax=lax,
            proj=proj,
            element=element,
            Lim=Lim,
            Nstep=Nstep,
            dLeg=dLeg,
            draw=draw,
            fs=fs,
            indices=indices,
            wintit=wintit,
            tit=tit,
            Test=Test,
        )
        return lax

    def plot_sino(
        self,
        ax=None,
        dP=None,
        Ang=_def.LOSImpAng,
        AngUnit=_def.LOSImpAngUnit,
        Sketch=True,
        dLeg=_def.TorLegd,
        draw=True,
        fs=None,
        wintit=None,
        tit=None,
        Test=True,
    ):

        msg = "Set the sino params before plotting !"
        msg += "\n    => run self.set_sino(...)"
        assert self.dsino["RefPt"] is not None, msg
        assert tit in [None, False] or isinstance(tit, str)
        # Check uniformity of sinogram parameters
        for ss in self.lStruct:
            msg = "{0} {1} has different".format(ss.Id.Cls, ss.Id.Name)
            msgf = "\n    => run self.set_sino(...)"
            msg0 = msg + " sino RefPt" + msgf
            assert np.allclose(self.dsino["RefPt"], ss.dsino["RefPt"]), msg0
            msg1 = msg + " sino nP" + msgf
            assert self.dsino["nP"] == ss.dsino["nP"], msg1

        if tit is None:
            tit = self.Id.Name

        vis = self.get_visible()
        lS = self.lStruct
        lS = [lS[ii] for ii in range(0, self._dStruct["nObj"]) if vis[ii]]

        ax = _plot.Plot_Impact_PolProjPoly(
            lS,
            ax=ax,
            Ang=Ang,
            AngUnit=AngUnit,
            Sketch=Sketch,
            dP=dP,
            dLeg=dLeg,
            draw=draw,
            fs=fs,
            tit=tit,
            wintit=wintit,
            Test=Test,
        )
        return ax

    @classmethod
    def from_svg(
        cls,
        pfe,
        res=None,
        point_ref1=None,
        point_ref2=None,
        length_ref=None,
        r0=None,
        z0=None,
        scale=None,
        Exp=None,
        Name=None,
        shot=None,
        Type=None,
        SavePath=os.path.abspath("./"),
        verb=None,
        returnas=None,
    ):
        """ Build a config from a svg file (Inkscape)

        The svg shall have only:
            - closed polygons (possibly inc. Bezier curves)
            - an optional unique 2-points straight line (non-closed)
              used for auto-scaling

        If Beziers curves are included, they will be discretized according to
        resolution parameter res (absolute maximum tolerated distance between
        points)

        All closed polygons will be interpreted as:
            - a Ves instance if it has no fill color
            - a PFC instance if it has a fill color
        The names are derived from Inkscape objects id

        The coordinates are extracted from the svg
        They can be rescaled either:
            - automatically:
                scaling computed from the unique straight line
                and from the corresponding 2 points real-life coordinates
                provided by the user as 2 iterables (list, arrays or tuples)
                of len() = 2 (point_ref1 and point_ref2)
                Alternatively a single point (point_ref1) and the length_ref
                of the line can be provided
            - forcefully:
                the origin (r0, z0) and a common scaling factor (scale) are
                provided by the user

        The result Config instance must have a Name and be associated to an
        experiment (Exp).

        """
        # Check inputs
        if returnas is None:
            returnas = object
        if returnas not in [object, dict]:
            msg = (
                "Arg returnas must be either:"
                + "\t- 'object': return Config instance\n"
                + "\t- 'dict' : return a dict with polygon, cls, color"
            )
            raise Exception(msg)

        # Extract polygon from file and check
        dpath = _comp.get_paths_from_svg(
            pfe=pfe, res=res,
            point_ref1=point_ref1, point_ref2=point_ref2,
            length_ref=length_ref,
            r0=r0, z0=z0, scale=scale,
            verb=verb,
        )

        if len(dpath) == 0:
            msg = "No Struct found in {}".format(pfe)
            raise Exception(msg)

        if returnas is dict:
            return dpath

        else:
            derr = {}
            lstruct = []
            for k0, v0 in dpath.items():

                # get class
                clss = eval(v0['cls'])

                # Instanciate
                try:
                    lstruct.append(
                        clss(
                            Name=k0, Poly=v0['poly'],
                            color=v0['color'], Exp=Exp,
                        )
                    )
                except Exception as err:
                    derr[k0] = str(err)

            # Raise error if any
            if len(derr) > 0:
                lerr = [
                    '\n\t- {}: {}'.format(k0, v0) for k0, v0 in derr.items()
                ]
                msg = (
                    "\nThe following Struct could not be created:\n"
                    + '\n'.join(lerr)
                )
                warnings.warn(msg)

            SavePath = os.path.abspath(SavePath)
            return cls(
                Name=Name,
                Exp=Exp,
                shot=shot,
                Type=Type,
                lStruct=lstruct,
                SavePath=SavePath,
            )

    def save_to_imas(
        self,
        shot=None,
        run=None,
        refshot=None,
        refrun=None,
        user=None,
        database=None,
        version=None,
        occ=None,
        dryrun=False,
        verb=True,
        description_2d=None,
    ):
        import tofu.imas2tofu as _tfimas

        _tfimas._save_to_imas(
            self,
            tfversion=__version__,
            shot=shot,
            run=run,
            refshot=refshot,
            refrun=refrun,
            user=user,
            database=database,
            version=version,
            occ=occ,
            dryrun=dryrun,
            verb=verb,
            description_2d=description_2d,
        )

    def get_kwdargs_LOS_isVis(self):

        lS = self.lStruct

        # -- Getting "vessels" or IN structures -------------------------------
        lSIn = [ss for ss in lS if ss._InOut == "in"]
        if len(lSIn) == 0:
            msg = "self.config must have at least a StructIn subclass !"
            assert len(lSIn) > 0, msg
        elif len(lSIn) > 1:
            S = lSIn[np.argmin([ss.dgeom["Surf"] for ss in lSIn])]
        else:
            S = lSIn[0]

        # ... and its poly, limts, type, etc.
        VPoly = S.Poly_closed
        VVIn = S.dgeom["VIn"]
        if np.size(np.shape(S.Lim)) > 1:
            Lim = np.asarray([S.Lim[0][0], S.Lim[0][1]])
        else:
            Lim = S.Lim
        VType = self.Id.Type

        # -- Getting OUT structures -------------------------------------------
        lS = [ss for ss in lS if ss._InOut == "out"]

        if len(lS) == 0:

            lSLim, lSnLim = None, None
            num_lim_structs, num_tot_structs = 0, 0
            lSPolyx, lSPolyy = None, None
            lSVInx, lSVIny = None, None
            lsnvert = None

        else:

            # Lims
            lSLim = [ss.Lim for ss in lS]
            lSnLim = np.array([ss.noccur for ss in lS], dtype=np.int64)

            # Nb of structures and of structures inc. Lims (toroidal occurence)
            num_lim_structs = len(lS)
            num_tot_structs = int(np.sum([max(1, ss.noccur) for ss in lS]))

            # build concatenated C-contiguous arrays of x and y coordinates
            lSPolyx = np.concatenate([ss.Poly_closed[0, :] for ss in lS])
            lSPolyy = np.concatenate([ss.Poly_closed[1, :] for ss in lS])
            lSVInx = np.concatenate([ss.dgeom['VIn'][0, :] for ss in lS])
            lSVIny = np.concatenate([ss.dgeom['VIn'][1, :] for ss in lS])

            # lsnvert = cumulated number of points in the poly of each Struct
            lsnvert = np.cumsum([
                ss.Poly_closed[0].size for ss in lS],
                dtype=np.int64,
            )

        # Now setting keyword arguments:
        dkwd = dict(
            ves_poly=VPoly,
            ves_norm=VVIn,
            ves_lims=Lim,
            nstruct_tot=num_tot_structs,
            nstruct_lim=num_lim_structs,
            lstruct_polyx=lSPolyx,
            lstruct_polyy=lSPolyy,
            lstruct_lims=lSLim,
            lstruct_nlim=lSnLim,
            lstruct_normx=lSVInx,
            lstruct_normy=lSVIny,
            lnvert=lsnvert,
            ves_type=VType,
            rmin=-1,
            forbid=True,
            eps_uz=1.0e-6,
            eps_vz=1.0e-9,
            eps_a=1.0e-9,
            eps_b=1.0e-9,
            eps_plane=1.0e-9,
            test=True,
        )
        return dkwd

    def calc_solidangle_particle(
        self,
        pts=None,
        part_traj=None,
        part_radius=None,
        approx=None,
        aniso=None,
        block=None,
    ):
        """ Compute the solid angle subtended by a particle along a trajectory

        The particle has radius r, and trajectory (array of points) traj
        It is observed from pts (array of points)
        Takes into account blocking of the field of view by structural elements

        traj and pts are (3, N) and (3, M) arrays of cartesian coordinates

        approx = True => use approximation
        aniso = True => return also unit vector of emission
        block = True consider LOS collisions (with Ves, Struct...)

        if block:
            config used for LOS collisions

        Parameters
        ----------
        traj:       np.ndarray
            Array of (3, N) pts coordinates (X, Y, Z) representing the particle
            positions
        pts:        np.ndarray
            Array of (3, M) pts coordinates (X, Y, Z) representing points from
            which the particle is observed
        rad:        float / np.ndarray
            Unique of multiple values for the radius of the spherical particle
                if multiple, rad is a np.ndarray of shape (N,)
        approx:     None / bool
            Flag indicating whether to compute the solid angle using a
            1st-order series development (in which case the solid angle becomes
            proportional to the radius of the particle, see Notes_Upgrades/)
        aniso:      None / bool
            Flag indicating whether to consider anisotropic emissivity,
            meaning the routine must also compute and return the unit vector
            directing the flux from each pts to each position on the trajectory
        block:      None / bool
            Flag indicating whether to check for vignetting by structural
            elements provided by config

        Return:
        -------
        sang: np.ndarray
            (N, M) Array of floats, solid angles

        """
        return _comp_solidangles.calc_solidangle_particle(
            pts=pts,
            part_traj=part_traj,
            part_radius=part_radius,
            config=self,
            approx=approx,
            aniso=aniso,
            block=block,
        )


    def calc_solidangle_particle_integrated(
        self,
        part_traj=None,
        part_radius=None,
        approx=None,
        block=None,
        resolution=None,
        DR=None,
        DZ=None,
        DPhi=None,
        plot=None,
        vmin=None,
        vmax=None,
        scale=None,
        fs=None,
        dmargin=None,
        returnax=None,
    ):
        """ Compute the integrated solid angle map subtended by particles

        Integrates the solid angle toroidally on a volume sampling of Config

        The particle has radius r, and trajectory (array of points) traj
        It is observed from pts (array of points)
        Takes into account blocking of the field of view by structural elements

        traj and pts are (3, N) and (3, M) arrays of cartesian coordinates

        approx = True => use approximation
        block = True consider LOS collisions (with Ves, Struct...)

        if block:
            config used for LOS collisions

        Parameters
        ----------
        traj:       np.ndarray
            Array of (3, N) pts coordinates (X, Y, Z) representing the particle
            positions
        pts:        np.ndarray
            Array of (3, M) pts coordinates (X, Y, Z) representing points from
            which the particle is observed
        rad:        float / np.ndarray
            Unique of multiple values for the radius of the spherical particle
                if multiple, rad is a np.ndarray of shape (N,)
        approx:     None / bool
            Flag indicating whether to compute the solid angle using a
            1st-order series development (in which case the solid angle becomes
            proportional to the radius of the particle, see Notes_Upgrades/)
        block:      None / bool
            Flag indicating whether to check for vignetting by structural
            elements provided by config

        Return:
        -------
        sang: np.ndarray
            (N, M) Array of floats, solid angles

        """
        if plot is None:
            plot = True
        if returnax is None:
            returnax = True

        # -------------------
        # Compute
        (
            ptsRZ, sang, indices, reseff,
        ) = _comp_solidangles.calc_solidangle_particle_integ(
            part_traj=part_traj,
            part_radius=part_radius,
            config=self,
            resolution=resolution,
            DR=DR,
            DZ=DZ,
            DPhi=DPhi,
            block=block,
            approx=approx,
        )

        if plot is False:
            return ptsRZ, sang, indices, reseff

        # -------------------
        # plot
        dax = _plot.Config_plot_solidangle_map_particle(
            config=self,
            part_traj=part_traj,
            part_radius=part_radius,
            ptsRZ=ptsRZ,
            sang=sang,
            indices=indices,
            reseff=reseff,
            vmin=vmin,
            vmax=vmax,
            scale=scale,
            fs=fs,
            dmargin=dmargin,
        )
        if returnax is True:
            return ptsRZ, sang, indices, reseff, dax
        else:
            return ptsRZ, sang, indices, reseff



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
        A name string or a :class:`~tofu.pathfile.ID` to identify this
        instance,
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
    _ddef = {
        "Id": {
            "shot": 0,
            "include": [
                "Mod",
                "Cls",
                "Exp",
                "Diag",
                "Name",
                "shot",
                "version",
            ],
        },
        "dgeom": {"Type": "Tor", "Lim": [], "arrayorder": "C"},
        "dsino": {},
        "dmisc": {"color": "k"},
    }
    _dplot = {
        "cross": {
            "Elt": "P",
            "dP": {"color": "k", "lw": 2},
            "dI": {"color": "k", "ls": "--", "m": "x", "ms": 8, "mew": 2},
            "dBs": {"color": "b", "ls": "--", "m": "x", "ms": 8, "mew": 2},
            "dBv": {"color": "g", "ls": "--", "m": "x", "ms": 8, "mew": 2},
            "dVect": {"color": "r", "scale": 10},
        },
        "hor": {
            "Elt": "P",
            "dP": {"color": "k", "lw": 2},
            "dI": {"color": "k", "ls": "--"},
            "dBs": {"color": "b", "ls": "--"},
            "dBv": {"color": "g", "ls": "--"},
            "Nstep": 50,
        },
        "3d": {
            "Elt": "P",
            "dP": {
                "color": (0.8, 0.8, 0.8, 1.0),
                "rstride": 1,
                "cstride": 1,
                "linewidth": 0.0,
                "antialiased": False,
            },
            "Lim": None,
            "Nstep": 50,
        },
    }

    _dcases = {
        "A": {"type": tuple, "lk": []},
        "B": {"type": dict, "lk": ["D", "u"]},
        "C": {"type": dict, "lk": ["D", "pinhole"]},
        "D": {"type": dict, "lk": ["pinhole", "F", "nIn", "e1", "x1"]},
        "E": {"type": dict, "lk": ["pinhole", "F", "nIn", "e1", "l1", "n1"]},
        "F": {"type": dict, "lk": ["pinhole", "F", "angles", "x1"]},
        "G": {"type": dict, "lk": ["pinhole", "F", "angles", "l1", "n1"]},
    }

    _method = "optimized"

    # Does not exist beofre Python 3.6 !!!
    def __init_subclass__(cls, color="k", **kwdargs):
        # Python 2
        super(Rays, cls).__init_subclass__(**kwdargs)
        # Python 3
        # super().__init_subclass__(**kwdargs)
        cls._ddef = copy.deepcopy(Rays._ddef)
        cls._dplot = copy.deepcopy(Rays._dplot)
        cls._set_color_ddef(color)
        if cls._is2D():
            cls._dcases["D"]["lk"] += ["e2", "x2"]
            cls._dcases["E"]["lk"] += ["e2", "l2", "n2"]
            cls._dcases["F"]["lk"] += ["x2"]
            cls._dcases["G"]["lk"] += ["l2", "n2"]

    @classmethod
    def _set_color_ddef(cls, color):
        cls._ddef['dmisc']['color'] = mpl.colors.to_rgba(color)

    def __init__(self, dgeom=None, strict=None,
                 lOptics=None, Etendues=None, Surfaces=None,
                 config=None, dchans=None, dX12='geom',
                 Id=None, Name=None, Exp=None, shot=None, Diag=None,
                 sino_RefPt=None, fromdict=None, sep=None, method='optimized',
                 SavePath=os.path.abspath('./'), color=None):

        # Create a dplot at instance level
        self._dplot = copy.deepcopy(self.__class__._dplot)

        # Extra-early fix for Exp
        # Workflow to be cleaned up later ?
        if Exp is None and config is not None:
            Exp = config.Id.Exp

        kwdargs = locals()
        del kwdargs["self"]
        # super()
        super(Rays, self).__init__(**kwdargs)

    def _reset(self):
        # super()
        super(Rays, self)._reset()
        self._dgeom = dict.fromkeys(self._get_keys_dgeom())
        if self._is2D():
            self._dX12 = dict.fromkeys(self._get_keys_dX12())
        self._dOptics = dict.fromkeys(self._get_keys_dOptics())
        self._dconfig = dict.fromkeys(self._get_keys_dconfig())
        self._dsino = dict.fromkeys(self._get_keys_dsino())
        self._dchans = dict.fromkeys(self._get_keys_dchans())
        self._dmisc = dict.fromkeys(self._get_keys_dmisc())
        # self._dplot = copy.deepcopy(self.__class__._ddef['dplot'])

    @classmethod
    def _checkformat_inputs_Id(
        cls,
        Id=None,
        Name=None,
        Exp=None,
        shot=None,
        Diag=None,
        include=None,
        **kwdargs
    ):
        if Id is not None:
            assert isinstance(Id, utils.ID)
            Name, Exp, shot, Diag = Id.Name, Id.Exp, Id.shot, Id.Diag
        if shot is None:
            shot = cls._ddef["Id"]["shot"]
        if include is None:
            include = cls._ddef["Id"]["include"]

        dins = {
            "Name": {"var": Name, "cls": str},
            "Exp": {"var": Exp, "cls": str},
            "Diag": {"var": Diag, "cls": str},
            "shot": {"var": shot, "cls": int},
            "include": {"var": include, "listof": str},
        }
        dins, err, msg = cls._check_InputsGeneric(dins, tab=0)
        if err:
            raise Exception(msg)
        kwdargs.update(
            {
                "Name": Name,
                "Exp": Exp,
                "shot": shot,
                "Diag": Diag,
                "include": include,
            }
        )
        return kwdargs

    ###########
    # Get largs
    ###########

    @staticmethod
    def _get_largs_dgeom(sino=True):
        largs = ["dgeom", 'strict', "Etendues", "Surfaces"]
        if sino:
            lsino = Rays._get_largs_dsino()
            largs += ["sino_{0}".format(s) for s in lsino]
        return largs

    @staticmethod
    def _get_largs_dX12():
        largs = ["dX12"]
        return largs

    @staticmethod
    def _get_largs_dOptics():
        largs = ["lOptics"]
        return largs

    @staticmethod
    def _get_largs_dconfig():
        largs = ["config", "strict"]
        return largs

    @staticmethod
    def _get_largs_dsino():
        largs = ["RefPt"]
        return largs

    @staticmethod
    def _get_largs_dchans():
        largs = ["dchans"]
        return largs

    @staticmethod
    def _get_largs_dmisc():
        largs = ["color"]
        return largs

    ###########
    # Get check and format inputs
    ###########

    def _checkformat_inputs_dES(self, val=None):
        if val is not None:
            C0 = type(val) in [int, float, np.int64, np.float64]
            C1 = hasattr(val, "__iter__")
            assert C0 or C1
            if C0:
                val = np.asarray([val], dtype=float)
            else:
                val = np.asarray(val, dtype=float).ravel()
                assert val.size == self._dgeom["nRays"]
        return val

    def _checkformat_inputs_dgeom(self, dgeom=None):
        assert dgeom is not None
        assert isinstance(dgeom, tuple) or isinstance(dgeom, dict)
        lC = [k for k in self._dcases.keys()
              if (isinstance(dgeom, self._dcases[k]['type'])
                  and all([kk in dgeom.keys()  # noqa
                           for kk in self._dcases[k]['lk']]))]
        if not len(lC) == 1:
            lstr = [v['lk'] for v in self._dcases.values()]
            msg = "Arg dgeom must be either:\n"
            msg += "  - dict with keys:\n"
            msg += "\n    - " + "\n    - ".join(lstr)
            msg += "  - tuple of len()==2 containing (D,u)"
            raise Exception(msg)
        case = lC[0]

        def _checkformat_Du(arr, name):
            arr = np.asarray(arr, dtype=float)
            msg = f"Arg {name} must be an iterable convertible into either:"
            msg += "\n    - a 1D np.ndarray of size=3"
            msg += "\n    - a 2D np.ndarray of shape (3,N)"
            if arr.ndim not in [1, 2]:
                msg += f"\nProvided arr.shape: {arr.shape}"
                raise Exception(msg)

            if arr.ndim == 1:
                assert arr.size == 3, msg
                arr = arr.reshape((3, 1))
            else:
                assert 3 in arr.shape, msg
                if arr.shape[0] != 3:
                    arr = arr.T
            arr = np.ascontiguousarray(arr)
            return arr

        if case in ["A", "B"]:
            D = dgeom[0] if case == "A" else dgeom["D"]
            u = dgeom[1] if case == "A" else dgeom["u"]
            D = _checkformat_Du(D, "D")
            u = _checkformat_Du(u, "u")
            # Normalize u
            u = u / np.sqrt(np.sum(u ** 2, axis=0))[np.newaxis, :]
            nD, nu = D.shape[1], u.shape[1]
            C0 = nD == 1 and nu > 1
            C1 = nD > 1 and nu == 1
            C2 = nD == nu
            msg = "The number of rays is ambiguous from D and u shapes !"
            assert C0 or C1 or C2, msg
            nRays = max(nD, nu)
            dgeom = {"D": D, "u": u, "isImage": False}

        elif case == 'C':
            D = _checkformat_Du(dgeom['D'], 'D')
            dins = {'pinhole': {'var': dgeom['pinhole'], 'vectnd': 3}}
            dins, err, msg = self._check_InputsGeneric(dins)
            if err:
                raise Exception(msg)
            pinhole = dins["pinhole"]["var"]
            dgeom = {"D": D, "pinhole": pinhole, "isImage": False}
            nRays = D.shape[1]

        else:
            dins = {
                "pinhole": {"var": dgeom["pinhole"], "vectnd": 3},
                "F": {"var": dgeom["F"], "int2float": None},
            }
            if case in ["D", "E"]:
                dins["nIn"] = {"var": dgeom["nIn"], "unitvectnd": 3}
                dins["e1"] = {"var": dgeom["e1"], "unitvectnd": 3}
                if "e2" in dgeom.keys():
                    dins["e2"] = {"var": dgeom["e2"], "unitvectnd": 3}
            else:
                dins["angles"] = {"var": dgeom["angles"], "vectnd": 3}

            if case in ["D", "F"]:
                dins["x1"] = {"var": dgeom["x1"], "vectnd": None}
                if "x2":
                    dins["x2"] = {"var": dgeom["x2"], "vectnd": None}
            else:
                dins["l1"] = {"var": dgeom["l1"], "int2float": None}
                dins["n1"] = {"var": dgeom["n1"], "float2int": None}
                if "l2" in dgeom.keys():
                    dins["l2"] = {"var": dgeom["l2"], "int2float": None}
                    dins["n2"] = {"var": dgeom["n2"], "float2int": None}

            dins, err, msg = self._check_InputsGeneric(dins)
            if err:
                raise Exception(msg)
            dgeom = {"dX12": {}}
            for k in dins.keys():
                if k == "pinhole":
                    dgeom[k] = dins[k]["var"]
                else:
                    dgeom["dX12"][k] = dins[k]["var"]
            if case in ["E", "G"]:
                x1 = dgeom["dX12"]["l1"] * np.linspace(
                    -0.5, 0.5, dgeom["dX12"]["n1"], end_point=True
                )
                dgeom["dX12"]["x1"] = x1
                if self._is2D():
                    x2 = dgeom["dX12"]["l2"] * np.linspace(
                        -0.5, 0.5, dgeom["dX12"]["n2"], end_point=True
                    )
                    dgeom["dX12"]["x2"] = x2
            if self._is2D():
                nRays = dgeom["dX12"]["n1"] * dgeom["dX12"]["n2"]
                ind1, ind2, indr = self._get_ind12r_n12(
                    n1=dgeom["dX12"]["n1"], n2=dgeom["dX12"]["n2"]
                )
                dgeom["dX12"]["ind1"] = ind1
                dgeom["dX12"]["ind2"] = ind2
                dgeom["dX12"]["indr"] = indr
                dgeom["isImage"] = True
            else:
                nRays = dgeom["dX12"]["n1"]
                dgeom["isImage"] = False
        dgeom.update({"case": case, "nRays": nRays})
        return dgeom

    def _checkformat_dX12(self, dX12=None):
        lc = [
            dX12 is None,
            dX12 == "geom" or dX12 == {"from": "geom"},
            isinstance(dX12, dict),
        ]
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
            c0 = isinstance(self._dgeom["dX12"], dict)
            c1 = c0 and all([ss in self._dgeom["dX12"].keys() for ss in ls])
            c2 = c1 and all([self._dgeom["dX12"][ss] is not None for ss in ls])
            if not c2:
                msg = "dX12 is not provided as input (dX12 = None)\n"
                msg += "  => self._dgeom['dX12'] (computed) used as fallback\n"
                msg += "    - It should have non-None keys: %s\n" % str(
                    list(ls)
                )
                msg += "    - it is:\n%s" % str(self._dgeom["dX12"])
                raise Exception(msg)
            dX12 = {"from": "geom"}

        if lc[2]:
            ls = ["x1", "x2", "ind1", "ind2"]
            assert all([ss in dX12.keys() for ss in ls])
            x1 = np.asarray(dX12["x1"]).ravel()
            x2 = np.asarray(dX12["x2"]).ravel()
            n1, n2 = x1.size, x2.size
            ind1, ind2, indr = self._get_ind12r_n12(
                ind1=dX12["ind1"], ind2=dX12["ind2"], n1=n1, n2=n2
            )
            dX12 = {
                "x1": x1,
                "x2": x2,
                "n1": n1,
                "n2": n2,
                "ind1": ind1,
                "ind2": ind2,
                "indr": indr,
                "from": "self",
            }
        return dX12

    @staticmethod
    def _checkformat_dOptics(lOptics=None):
        if lOptics is None:
            lOptics = []
        assert type(lOptics) is list
        lcls = ["Apert3D", "Cryst2D"]
        nOptics = len(lOptics)
        for ii in range(0, nOptics):
            assert lOptics[ii].__class__.__name__ in lcls
        return lOptics

    @staticmethod
    def _checkformat_inputs_dconfig(config=None):
        # Check config has proper class
        if not isinstance(config, Config):
            msg = ("Arg config must be a Config instance!\n"
                   + "\t- expected: {}".format(str(Config))
                   + "\t- provided: {}".format(str(config.__class__)))
            raise Exception(msg)

        # Check all structures
        lS = config.lStruct
        lC = [hasattr(ss, "_InOut") and ss._InOut in ["in", "out"]
              for ss in lS]
        if not all(lC):
            msg = "All Struct in config must have self._InOut in ['in','out']"
            raise Exception(msg)

        # Check there is at least one struct which is a subclass of StructIn
        lSIn = [ss for ss in lS if ss._InOut == "in"]
        if len(lSIn) == 0:
            lclsnames = [
                f'\t- {ss.Id.Name}, {ss.Id.Cls}, {ss._InOut}' for ss in lS
            ]
            msg = (
                f"Config {config.Id.Name} is missing a StructIn!\n"
                + "\n".join(lclsnames)
            )
            raise Exception(msg)

        # Add 'compute' parameter if not present
        if "compute" not in config._dextraprop["lprop"]:
            config = config.copy()
            config.add_extraprop("compute", config.get_visible())
        return config

    def _checkformat_inputs_dsino(self, RefPt=None):
        assert RefPt is None or hasattr(RefPt, "__iter__")
        if RefPt is not None:
            RefPt = np.asarray(RefPt, dtype=float).flatten()
            assert RefPt.size == 2, "RefPt must be of size=2 !"
        return RefPt

    def _checkformat_inputs_dchans(self, dchans=None):
        assert dchans is None or isinstance(dchans, dict)
        if dchans is None:
            dchans = {}
        for k in dchans.keys():
            arr = np.asarray(dchans[k]).ravel()
            assert arr.size == self._dgeom["nRays"]
            dchans[k] = arr
        return dchans

    @classmethod
    def _checkformat_inputs_dmisc(cls, color=None):
        if color is None:
            color = mpl.colors.to_rgba(cls._ddef["dmisc"]["color"])
        assert mpl.colors.is_color_like(color)
        return tuple(mpl.colors.to_rgba(color))

    ###########
    # Get keys of dictionnaries
    ###########

    @staticmethod
    def _get_keys_dgeom():
        lk = [
            "D",
            "u",
            "pinhole",
            "nRays",
            "kIn",
            "kOut",
            "PkIn",
            "PkOut",
            "vperp",
            "indout",
            "indStruct",
            "kRMin",
            "PRMin",
            "RMin",
            "isImage",
            "Etendues",
            "Surfaces",
            "dX12",
            "dreflect",
            "move",
            "move_param",
            "move_kwdargs",
        ]
        return lk

    @staticmethod
    def _get_keys_dX12():
        lk = ["x1", "x2", "n1", "n2", "ind1", "ind2", "indr"]
        return lk

    @staticmethod
    def _get_keys_dOptics():
        lk = ["lorder", "lCls", "nObj", "dObj"]
        return lk

    @staticmethod
    def _get_keys_dsino():
        lk = ["RefPt", "k", "pts", "theta", "p", "phi"]
        return lk

    @staticmethod
    def _get_keys_dconfig():
        lk = ["config"]
        return lk

    @staticmethod
    def _get_keys_dchans():
        lk = []
        return lk

    @staticmethod
    def _get_keys_dmisc():
        lk = ["color"]
        return lk

    ###########
    # _init
    ###########

    def _init(
        self,
        dgeom=None,
        config=None,
        Etendues=None,
        Surfaces=None,
        sino_RefPt=None,
        dchans=None,
        method="optimized",
        **kwargs
    ):
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
        self._dstrip["strip"] = 0

    ###########
    # set dictionaries
    ###########

    def set_dconfig(self, config=None, strict=None, calcdgeom=True):
        config = self._checkformat_inputs_dconfig(config)
        self._dconfig["Config"] = config.copy()
        if calcdgeom:
            self.compute_dgeom(strict=strict)

    def _update_dgeom_from_TransRotFoc(self, val, key="x"):
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
        x1, x2 = np.unique(X12[0, :]), np.unique(X12[1, :])
        n1, n2 = x1.size, x2.size
        if n1 * n2 != X12.shape[1]:
            tol = np.linalg.norm(np.diff(X12[:, :2], axis=1)) / 100.0
            tolmag = int(np.log10(tol)) - 1
            x1 = np.unique(np.round(X12[0, :], -tolmag))
            x2 = np.unique(np.round(X12[1, :], -tolmag))
            ind1 = np.digitize(X12[0, :], 0.5 * (x1[1:] + x1[:-1]))
            ind2 = np.digitize(X12[1, :], 0.5 * (x2[1:] + x2[:-1]))
            ind1u, ind2u = np.unique(ind1), np.unique(ind2)
            x1 = np.unique([np.mean(X12[0, ind1 == ii]) for ii in ind1u])
            x2 = np.unique([np.mean(X12[1, ind2 == ii]) for ii in ind2u])
        n1, n2 = x1.size, x2.size

        if n1 * n2 != X12.shape[1]:
            msg = "The provided X12 array does not seem to correspond to"
            msg += "a n1 x n2 2D matrix, even within tolerance\n"
            msg += "  n1*n2 = %s x %s = %s\n" % (
                str(n1),
                str(n2),
                str(n1 * n2),
            )
            msg += "  X12.shape = %s" % str(X12.shape)
            raise Exception(msg)

        ind1 = np.digitize(X12[0, :], 0.5 * (x1[1:] + x1[:-1]))
        ind2 = np.digitize(X12[1, :], 0.5 * (x2[1:] + x2[:-1]))
        ind1, ind2, indr = cls._get_ind12r_n12(
            ind1=ind1, ind2=ind2, n1=n1, n2=n2
        )
        return x1, x2, n1, n2, ind1, ind2, indr

    def _complete_dX12(self, dgeom):

        # Test if unique starting point
        if dgeom["case"] in ["A", "B", "C"]:
            # Test if pinhole
            if dgeom['D'].shape[1] == 1 and dgeom['nRays'] > 1:
                dgeom['pinhole'] = dgeom['D'].ravel()
            elif dgeom['case'] in ['A', 'B']:
                u = dgeom['u'][:, 0:1]
                sca2 = np.sum(dgeom['u'][:, 1:]*u, axis=0)**2
                if np.all(sca2 < 1.0 - 1.e-9):
                    DDb = dgeom['D'][:, 1:]-dgeom['D'][:, 0:1]
                    k = np.sum(DDb*(u - np.sqrt(sca2)*dgeom['u'][:, 1:]),
                               axis=0)
                    k = k / (1.0-sca2)
                    if k[0] > 0 and np.allclose(k, k[0], atol=1.e-3,
                                                rtol=1.e-6):
                        pinhole = dgeom['D'][:, 0] + k[0]*u[:, 0]
                        dgeom['pinhole'] = pinhole

            if np.any(np.isnan(dgeom['D'])):
                msg = ("Some LOS have nan as starting point !\n"
                       + "The geometry may not be provided !")
                raise Exception(msg)

            # Test if all D are on a common plane or line
            va = dgeom["D"] - dgeom["D"][:, 0:1]

            # critetrion of unique D
            crit = np.sqrt(np.sum(va ** 2, axis=0))
            if np.sum(crit) < 1.0e-9:
                if self._is2D():
                    msg = "2D camera but dgeom cannot be obtained !\n"
                    msg += "  crit = %s\n" % str(crit)
                    msg += "  dgeom = %s" % str(dgeom)
                    raise Exception(msg)
                return dgeom

            # To avoid ||v0|| = 0
            if crit[1] > 1.0e-12:
                # Take first one by default to ensure square grid for CamLOS2D
                ind0 = 1
            else:
                ind0 = np.nanargmax(crit)
            v0 = va[:, ind0]
            v0 = v0 / np.linalg.norm(v0)
            indok = np.nonzero(crit > 1.0e-12)[0]
            van = np.full(va.shape, np.nan)
            van[:, indok] = va[:, indok] / crit[None, indok]
            vect2 = (
                (van[1, :] * v0[2] - van[2, :] * v0[1]) ** 2
                + (van[2, :] * v0[0] - van[0, :] * v0[2]) ** 2
                + (van[0, :] * v0[1] - van[1, :] * v0[0]) ** 2
            )
            # Don't forget that vect2[0] is nan
            if np.all(vect2[indok] < 1.0e-9):
                # All D are aligned
                e1 = v0
                x1 = np.sum(va * e1[:, np.newaxis], axis=0)
                if dgeom["pinhole"] is not None:
                    kref = -np.sum((dgeom["D"][:, 0] - dgeom["pinhole"]) * e1)
                    x1 = x1 - kref
                # l1 = np.nanmax(x1) - np.nanmin(x1)
                if dgeom["dX12"] is None:
                    dgeom["dX12"] = {}
                dgeom["dX12"].update({"e1": e1, "x1": x1, "n1": x1.size})

            elif self._is2D():
                ind = np.nanargmax(vect2)
                v1 = van[:, ind]
                nn = np.cross(v0, v1)
                nn = nn / np.linalg.norm(nn)
                scaabs = np.abs(np.sum(nn[:, np.newaxis] * va, axis=0))
                if np.all(scaabs < 1.0e-9):
                    # All D are in a common plane, but not aligned
                    # check nIn orientation
                    sca = np.sum(self.u * nn[:, np.newaxis], axis=0)
                    lc = [np.all(sca >= 0.0), np.all(sca <= 0.0)]

                    assert any(lc)
                    nIn = nn if lc[0] else -nn
                    e1 = v0
                    e2 = v1
                    if np.sum(np.cross(e1, nIn) * e2) < 0.0:
                        e2 = -e2
                    if np.abs(e1[2]) > np.abs(e2[2]):
                        # Try to set e2 closer to ez if possible
                        e1, e2 = -e2, e1

                    if dgeom["dX12"] is None:
                        dgeom["dX12"] = {}
                    dgeom["dX12"].update({"nIn": nIn, "e1": e1, "e2": e2})

                    # Test binning
                    if dgeom["pinhole"] is not None:
                        k1ref = -np.sum(
                            (dgeom["D"][:, 0] - dgeom["pinhole"]) * e1
                        )
                        k2ref = -np.sum(
                            (dgeom["D"][:, 0] - dgeom["pinhole"]) * e2
                        )
                    else:
                        k1ref, k2ref = 0.0, 0.0
                    x12 = np.array(
                        [
                            np.sum(va * e1[:, np.newaxis], axis=0) - k1ref,
                            np.sum(va * e2[:, np.newaxis], axis=0) - k2ref,
                        ]
                    )
                    try:
                        out_loc = self._get_x12_fromflat(x12)
                        x1, x2, n1, n2, ind1, ind2, indr = out_loc
                        dgeom["dX12"].update(
                            {
                                "x1": x1,
                                "x2": x2,
                                "n1": n1,
                                "n2": n2,
                                "ind1": ind1,
                                "ind2": ind2,
                                "indr": indr,
                            }
                        )
                        dgeom["isImage"] = True

                    except Exception as err:
                        msg = str(err)
                        msg += "\n  nIn = %s" % str(nIn)
                        msg += "\n  e1 = %s" % str(e1)
                        msg += "\n  e2 = %s" % str(e2)
                        msg += "\n  k1ref, k2ref = %s, %s" % (
                            str(k1ref),
                            str(k2ref),
                        )
                        msg += "\n  va = %s" % str(va)
                        msg += "\n  x12 = %s" % str(x12)
                        warnings.warn(msg)

        else:
            if dgeom["case"] in ["F", "G"]:
                # Get unit vectors from angles
                msg = "Not implemented yet, angles will be available for 1.4.1"
                raise Exception(msg)

            # Get D and x12 from x1, x2
            x12 = np.array(
                [
                    dgeom["dX12"]["x1"][dgeom["dX12"]["ind1"]],
                    dgeom["dX12"]["x2"][dgeom["dX12"]["ind2"]],
                ]
            )
            D = dgeom["pinhole"] - dgeom["F"] * dgeom["dX12"]["nIn"]
            D = (
                D[:, np.newaxis]
                + x12[0, :] * dgeom["dX12"]["e1"]
                + x12[1, :] * dgeom["dX12"]["e2"]
            )
            dgeom["D"] = D
        return dgeom

    def _prepare_inputs_kInOut(self, D=None, u=None, indStruct=None):

        # Prepare input: D, u
        if D is None:
            D = np.ascontiguousarray(self.D)
        else:
            D = np.ascontiguousarray(D)
        if u is None:
            u = np.ascontiguousarray(self.u)
        else:
            u = np.ascontiguousarray(u)
        assert D.shape == u.shape

        # Get reference: lS
        if indStruct is None:
            indIn, indOut = self.get_indStruct_computeInOut(unique_In=True)
            indStruct = np.r_[indIn, indOut]
        else:
            indIn = [
                ii for ii in indStruct
                if self.config.lStruct[ii]._InOut == "in"
            ]
            if len(indIn) > 1:
                ind = np.argmin([
                    self.config.lStruct[ii].dgeom['Surf']
                    for ii in indIn
                ])
                indStruct = [ii for ii in indStruct
                             if ii not in indIn or ii == ind]
                indIn = [indIn[ind]]
            indOut = [
                ii for ii in indStruct
                if self.config.lStruct[ii]._InOut == "out"
            ]

        if len(indIn) == 0:
            msg = "self.config must have at least a StructIn subclass !"
            raise Exception(msg)

        S = self.config.lStruct[indIn[0]]
        VPoly = S.Poly_closed
        VVIn = S.dgeom["VIn"]
        largs = [D, u, VPoly, VVIn]

        lS = [self.config.lStruct[ii] for ii in indOut]
        if self._method == "ref":

            Lim = S.Lim
            nLim = S.noccur
            VType = self.config.Id.Type

            lSPoly, lSVIn, lSLim, lSnLim = [], [], [], []
            for ss in lS:
                lSPoly.append(ss.Poly_closed)
                lSVIn.append(ss.dgeom["VIn"])
                lSLim.append(ss.Lim)
                lSnLim.append(ss.noccur)
            dkwd = dict(
                Lim=Lim,
                nLim=nLim,
                LSPoly=lSPoly,
                LSLim=lSLim,
                lSnLim=lSnLim,
                LSVIn=lSVIn,
                VType=VType,
                RMin=None,
                Forbid=True,
                EpsUz=1.0e-6,
                EpsVz=1.0e-9,
                EpsA=1.0e-9,
                EpsB=1.0e-9,
                EpsPlane=1.0e-9,
                Test=True,
            )

        elif self._method == "optimized":

            if np.size(np.shape(S.Lim)) > 1:
                Lim = np.asarray([S.Lim[0][0], S.Lim[0][1]])
            else:
                Lim = S.Lim
            nLim = S.noccur
            VType = self.config.Id.Type

            lSPolyx, lSVInx = [], []
            lSPolyy, lSVIny = [], []
            lSLim, lSnLim = [], []
            lsnvert = []
            num_tot_structs = 0
            num_lim_structs = 0
            for ss in lS:
                lp = ss.Poly_closed[0]
                [lSPolyx.append(item) for item in lp]
                lp = ss.Poly_closed[1]
                [lSPolyy.append(item) for item in lp]
                lp = ss.dgeom["VIn"][0]
                [lSVInx.append(item) for item in lp]
                lp = ss.dgeom["VIn"][1]
                [lSVIny.append(item) for item in lp]
                lSLim.append(ss.Lim)
                lSnLim.append(ss.noccur)
                if len(lsnvert) == 0:
                    lsnvert.append(len(ss.Poly_closed[0]))
                else:
                    lsnvert.append(
                        len(ss.Poly_closed[0]) + lsnvert[num_lim_structs - 1]
                    )
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

        return indStruct, largs, dkwd

    def _compute_kInOut(self, largs=None, dkwd=None, indStruct=None):

        # Prepare inputs
        if largs is None:
            indStruct, largs, dkwd = self._prepare_inputs_kInOut(
                indStruct=indStruct
            )
        else:
            assert dkwd is not None
            assert indStruct is not None

        if self._method == "ref":
            # call the dedicated function
            out = _GG.SLOW_LOS_Calc_PInOut_VesStruct(*largs, **dkwd)
            # Currently computes and returns too many things
            PIn, POut, kIn, kOut, VperpIn, vperp, IIn, indout = out
        elif self._method == "optimized":
            # call the dedicated function
            out = _GG.LOS_Calc_PInOut_VesStruct(*largs, **dkwd)
            # Currently computes and returns too many things
            kIn, kOut, vperp, indout = out
        else:
            pass

        # Make sure indices refer to lStruct
        indout[0, :] = indStruct[indout[0, :]]
        return kIn, kOut, vperp, indout, indStruct

    def compute_dgeom(self, extra=True, strict=None, show_debug_plot=True):
        """ Compute dictionnary of geometrical attributes (dgeom)

        Parameters
        ----------
        show_debug_plot:    bool
            In case some lines of sight have no visibility inside the tokamak,
            they will be considered invalid. tofu will issue a warning with
            their indices and if show_debug_plot is True, try to plot a 3d
            figure to help understand why these los have no visibility
        """
        # check inputs
        if strict is None:
            strict = True

        # Can only be computed if config if provided
        if self._dconfig["Config"] is None:
            msg = "Attribute dgeom cannot be computed without a config!"
            warnings.warn(msg)
            return

        # dX12
        if self._dgeom["nRays"] > 1 and strict is True:
            self._dgeom = self._complete_dX12(self._dgeom)

        # Perform computation of kIn and kOut
        kIn, kOut, vperp, indout, indStruct = self._compute_kInOut()

        # Check for LOS that have no visibility inside the plasma domain (nan)
        ind = np.isnan(kIn)
        kIn[ind] = 0.0
        ind = np.isnan(kOut) | np.isinf(kOut)
        if np.any(ind):
            msg = ("Some LOS have no visibility inside the plasma domain!\n"
                   + "Nb. of LOS concerned: {} / {}\n".format(ind.sum(),
                                                              kOut.size)
                   + "Indices of LOS ok:\n"
                   + repr((~ind).nonzero()[0])
                   + "\nIndices of LOS with no visibility:\n"
                   + repr(ind.nonzero()[0]))
            if show_debug_plot is True:
                PIn = self.D[:, ind] + kIn[None, ind] * self.u[:, ind]
                POut = self.D[:, ind] + kOut[None, ind] * self.u[:, ind]
                msg2 = ("\n\tD = {}\n".format(self.D[:, ind])
                        + "\tu = {}\n".format(self.u[:, ind])
                        + "\tPIn = {}\n".format(PIn)
                        + "\tPOut = {}".format(POut))
                warnings.warn(msg2)
                # # plot 3d debug figure
                # _plot._LOS_calc_InOutPolProj_Debug(
                    # self.config,
                    # self.D[:, ind],
                    # self.u[:, ind],
                    # PIn,
                    # POut,
                    # nptstot=kOut.size,
                    # Lim=[np.pi / 4.0, 2.0 * np.pi / 4],
                    # Nstep=50,
                # )
                # import pdb; pdb.set_trace()     # DB

            kOut[ind] = np.nan
            if strict is True:
                raise Exception(msg)
            else:
                warnings.warn(msg)

        # Handle particular cases with kIn > kOut
        ind = np.zeros(kIn.shape, dtype=bool)
        ind[~np.isnan(kOut)] = True
        ind[ind] = kIn[ind] > kOut[ind]
        kIn[ind] = 0.0

        # Update dgeom
        dd = {
            "kIn": kIn,
            "kOut": kOut,
            "vperp": vperp,
            "indout": indout,
            "indStruct": indStruct,
        }
        self._dgeom.update(dd)

        # Run extra computations
        if extra:
            self._compute_dgeom_kRMin()
            self._compute_dgeom_extra1()

    def _compute_dgeom_kRMin(self):
        # Get RMin if Type is Tor
        if self.config.Id.Type == "Tor":
            kRMin = np.atleast_1d(
                _comp.LOS_PRMin(
                    self.D, self.u, kOut=self.kOut, Eps=1.0e-12, squeeze=True
                )
            )
        else:
            kRMin = None
        self._dgeom.update({"kRMin": kRMin})

    def _compute_dgeom_extra1(self):
        if self._dgeom["kRMin"] is not None:
            PRMin = self.D + self._dgeom["kRMin"][None, :] * self.u
            RMin = np.hypot(PRMin[0, :], PRMin[1, :])
        else:
            PRMin, RMin = None, None
        PkIn = self.D + self._dgeom["kIn"][np.newaxis, :] * self.u
        PkOut = self.D + self._dgeom["kOut"][np.newaxis, :] * self.u
        dd = {"PkIn": PkIn, "PkOut": PkOut, "PRMin": PRMin, "RMin": RMin}
        self._dgeom.update(dd)

    def _compute_dgeom_extra2D(self):
        if "2d" not in self.Id.Cls.lower():
            return
        D, u = self.D, self.u
        C = np.nanmean(D, axis=1)
        CD0 = D[:, :-1] - C[:, np.newaxis]
        CD1 = D[:, 1:] - C[:, np.newaxis]
        cross = np.array(
            [
                CD1[1, 1:] * CD0[2, :-1] - CD1[2, 1:] * CD0[1, :-1],
                CD1[2, 1:] * CD0[0, :-1] - CD1[0, 1:] * CD0[2, :-1],
                CD1[0, 1:] * CD0[1, :-1] - CD1[1, 1:] * CD0[0, :-1],
            ]
        )
        crossn2 = np.sum(cross ** 2, axis=0)
        if np.all(np.abs(crossn2) < 1.0e-12):
            msg = "Is %s really a 2D camera ? (LOS aligned?)" % self.Id.Name
            warnings.warn(msg)
        cross = cross[:, np.nanargmax(crossn2)]
        cross = cross / np.linalg.norm(cross)
        nIn = cross if np.sum(cross * np.nanmean(u, axis=1)) > 0.0 else -cross

        # Find most relevant e1 (for pixels alignment), without a priori info
        D0D = D - D[:, 0][:, np.newaxis]
        dist = np.sqrt(np.sum(D0D ** 2, axis=0))
        dd = np.min(dist[1:])
        e1 = (D[:, 1] - D[:, 0]) / np.linalg.norm(D[:, 1] - D[:, 0])
        crossbis = np.sqrt(
            (D0D[1, :] * e1[2] - D0D[2, :] * e1[1]) ** 2
            + (D0D[2, :] * e1[0] - D0D[0, :] * e1[2]) ** 2
            + (D0D[0, :] * e1[1] - D0D[1, :] * e1[0]) ** 2
        )
        D0D = D0D[:, crossbis < dd / 3.0]
        sca = np.sum(D0D * e1[:, np.newaxis], axis=0)
        e1 = D0D[:, np.argmax(np.abs(sca))]
        try:
            import tofu.geom.utils as geom_utils
        except Exception:
            from . import utils as geom_utils

        nIn, e1, e2 = geom_utils.get_nIne1e2(C, nIn=nIn, e1=e1)
        if np.abs(np.abs(nIn[2]) - 1.0) > 1.0e-12:
            if np.abs(e1[2]) > np.abs(e2[2]):
                e1, e2 = e2, e1
        e2 = e2 if e2[2] > 0.0 else -e2
        self._dgeom.update({"C": C, "nIn": nIn, "e1": e1, "e2": e2})

    def set_Etendues(self, val):
        val = self._checkformat_inputs_dES(val)
        self._dgeom["Etendues"] = val

    def set_Surfaces(self, val):
        val = self._checkformat_inputs_dES(val)
        self._dgeom["Surfaces"] = val

    def _set_dgeom(
        self,
        dgeom=None,
        Etendues=None,
        Surfaces=None,
        sino_RefPt=None,
        extra=True,
        strict=None,
        sino=True,
    ):
        dgeom = self._checkformat_inputs_dgeom(dgeom=dgeom)
        self._dgeom.update(dgeom)
        self.compute_dgeom(extra=extra, strict=strict)
        self.set_Etendues(Etendues)
        self.set_Surfaces(Surfaces)
        if sino:
            self.set_dsino(sino_RefPt)

    def set_dX12(self, dX12=None):
        dX12 = self._checkformat_dX12(dX12)
        self._dX12.update(dX12)

    def _compute_dsino_extra(self):
        if self._dsino["k"] is not None:
            pts = self.D + self._dsino["k"][np.newaxis, :] * self.u
            R = np.hypot(pts[0, :], pts[1, :])
            DR = R - self._dsino["RefPt"][0]
            DZ = pts[2, :] - self._dsino["RefPt"][1]
            p = np.hypot(DR, DZ)
            theta = np.arctan2(DZ, DR)
            ind = theta < 0
            p[ind] = -p[ind]
            theta[ind] = -theta[ind]
            phipts = np.arctan2(pts[1, :], pts[0, :])
            etheta = np.array(
                [
                    np.cos(phipts) * np.cos(theta),
                    np.sin(phipts) * np.cos(theta),
                    np.sin(theta),
                ]
            )
            phi = np.arccos(np.abs(np.sum(etheta * self.u, axis=0)))
            dd = {"pts": pts, "p": p, "theta": theta, "phi": phi}
            self._dsino.update(dd)

    def set_dsino(self, RefPt=None, extra=True):
        RefPt = self._checkformat_inputs_dsino(RefPt=RefPt)
        self._dsino.update({"RefPt": RefPt})
        VType = self.config.Id.Type
        if RefPt is not None:
            self._dconfig["Config"].set_dsino(RefPt=RefPt)
            kOut = np.copy(self._dgeom["kOut"])
            kOut[np.isnan(kOut)] = np.inf
            try:
                out = _GG.LOS_sino(
                    self.D, self.u, RefPt, kOut, Mode="LOS", VType=VType
                )
                Pt, k, r, Theta, p, theta, Phi = out
                self._dsino.update({"k": k})
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
        self._dmisc["color"] = color
        self._dplot["cross"]["dP"]["color"] = color
        self._dplot["hor"]["dP"]["color"] = color
        self._dplot["3d"]["dP"]["color"] = color

    def _set_dmisc(self, color=None):
        self._set_color(color)

    ###########
    # Reflections
    ###########

    def get_reflections_as_cam(self, Type=None, Name=None, nb=None):
        """ Return a camera made of reflected LOS

        Reflected LOS can be of 3 types:
            - 'speculiar':  standard mirror-like reflection
            - 'diffusive':  random reflection
            - 'ccube':      corner-cube reflection (ray goes back its way)

        As opposed to self.add_reflections(), the reflected rays are
        return as an independent camera (CamLOS1D)

        """
        # Check inputs
        if nb is None:
            nb = 1
        nb = int(nb)
        assert nb > 0
        if Name is None:
            Name = self.Id.Name + "_Reflect%s" % str(Type)
        clas = Rays if self.__class__.__name__ == Rays else CamLOS1D

        # Run first iteration
        Types = np.full((nb, self.nRays), 0, dtype=np.int64)
        Ds = self.D + (self._dgeom["kOut"][None, :] - 1.0e-12) * self.u
        us, Types[0, :] = self.config._reflect_geom(
            u=self.u,
            vperp=self._dgeom["vperp"],
            indout=self._dgeom["indout"],
            Type=Type,
        )
        lcam = [
            clas(
                dgeom=(Ds, us),
                config=self.config,
                Exp=self.Id.Exp,
                Diag=self.Id.Diag,
                Name=Name,
                shot=self.Id.shot,
            )
        ]
        if nb == 1:
            return lcam[0], Types[0, :]

        indStruct, largs, dkwd = self._prepare_inputs_kInOut(
            D=Ds, u=us, indStruct=self._dgeom["indStruct"]
        )
        outi = self._compute_kInOut(
            largs=largs, dkwd=dkwd, indStruct=indStruct
        )
        kouts, vperps, indouts = outi[1:-1]

        # Run other iterations
        for ii in range(1, nb):
            Ds = Ds + (kouts[None, :] - 1.0e-12) * us
            us, Types[ii, :] = self.config._reflect_geom(
                u=us, vperp=vperps, indout=indouts, Type=Type
            )
            outi = self._compute_kInOut(
                largs=[Ds, us, largs[2], largs[3]],
                dkwd=dkwd,
                indStruct=indStruct,
            )
            kouts, vperps, indouts = outi[1:-1]
            lcam.append(
                clas(
                    dgeom=(Ds, us),
                    config=self.config,
                    Exp=self.Id.Exp,
                    Diag=self.Id.Diag,
                    Name=Name,
                    shot=self.Id.shot,
                )
            )
        return lcam, Types

    def add_reflections(self, Type=None, nb=None):
        """ Add relfected LOS to the camera

        Reflected LOS can be of 3 types:
            - 'speculiar':  standard mirror-like reflection
            - 'diffusive':  random reflection
            - 'ccube':      corner-cube reflection (ray goes back its way)

        As opposed to self.get_reflections_as_cam(), the reflected rays are
        stored in the camera object

        """

        # Check inputs
        if nb is None:
            nb = 1
        nb = int(nb)
        assert nb > 0

        # Prepare output
        nRays = self.nRays
        Types = np.full((nRays, nb), 0, dtype=np.int64)
        Ds = np.full((3, nRays, nb), np.nan, dtype=float)
        us = np.full((3, nRays, nb), np.nan, dtype=float)
        kouts = np.full((nRays, nb), np.nan, dtype=float)
        indouts = np.full((3, nRays, nb), 0, dtype=np.int64)
        vperps = np.full((3, nRays, nb), np.nan, dtype=float)

        # Run first iteration
        Ds[:, :, 0] = (
            self.D + (self._dgeom["kOut"][None, :] - 1.0e-12) * self.u
        )
        us[:, :, 0], Types[:, 0] = self.config._reflect_geom(
            u=self.u,
            vperp=self._dgeom["vperp"],
            indout=self._dgeom["indout"],
            Type=Type,
        )
        indStruct, largs, dkwd = self._prepare_inputs_kInOut(
            D=Ds[:, :, 0], u=us[:, :, 0], indStruct=self._dgeom["indStruct"]
        )
        outi = self._compute_kInOut(
            largs=largs, dkwd=dkwd, indStruct=indStruct
        )
        kouts[:, 0], vperps[:, :, 0], indouts[:, :, 0] = outi[1:-1]

        # Run other iterations
        for ii in range(1, nb):
            Dsi = (
                Ds[:, :, ii - 1]
                + (kouts[None, :, ii - 1] - 1.0e-12) * us[:, :, ii - 1]
            )
            usi, Types[:, ii] = self.config._reflect_geom(
                u=us[:, :, ii - 1],
                vperp=vperps[:, :, ii - 1],
                indout=indouts[:, :, ii - 1],
                Type=Type,
            )
            outi = self._compute_kInOut(
                largs=[Dsi, usi, largs[2], largs[3]],
                dkwd=dkwd,
                indStruct=indStruct,
            )
            kouts[:, ii], vperps[:, :, ii], indouts[:, :, ii] = outi[1:-1]
            Ds[:, :, ii], us[:, :, ii] = Dsi, usi

        self._dgeom["dreflect"] = {
            "nb": nb,
            "Type": Type,
            "Types": Types,
            "Ds": Ds,
            "us": us,
            "kouts": kouts,
            "indouts": indouts,
        }

    ###########
    # strip dictionaries
    ###########

    def _strip_dgeom(self, strip=0):
        if self._dstrip["strip"] == strip:
            return

        if strip < self._dstrip["strip"]:
            # Reload
            if self._dstrip["strip"] == 1:
                self._compute_dgeom_extra1()
            elif self._dstrip["strip"] >= 2 and strip == 1:
                self._compute_dgeom_kRMin()
            elif self._dstrip["strip"] >= 2 and strip == 0:
                self._compute_dgeom_kRMin()
                self._compute_dgeom_extra1()
        else:
            # strip
            if strip == 1:
                lkeep = [
                    "D",
                    "u",
                    "pinhole",
                    "nRays",
                    "kIn",
                    "kOut",
                    "vperp",
                    "indout",
                    "indStruct",
                    "kRMin",
                    "Etendues",
                    "Surfaces",
                    "isImage",
                    "dX12",
                    "dreflect",
                    "move",
                    "move_param",
                    "move_kwdargs",
                ]
                utils.ToFuObject._strip_dict(self._dgeom, lkeep=lkeep)
            elif self._dstrip["strip"] <= 1 and strip >= 2:
                lkeep = [
                    "D",
                    "u",
                    "pinhole",
                    "nRays",
                    "kIn",
                    "kOut",
                    "vperp",
                    "indout",
                    "indStruct",
                    "Etendues",
                    "Surfaces",
                    "isImage",
                    "dX12",
                    "dreflect",
                    "move",
                    "move_param",
                    "move_kwdargs",
                ]
                utils.ToFuObject._strip_dict(self._dgeom, lkeep=lkeep)

    def _strip_dconfig(self, strip=0, force=False, verb=True):
        if self._dstrip["strip"] == strip:
            return

        if strip < self._dstrip["strip"]:
            if self._dstrip["strip"] == 4:
                pfe = self._dconfig["Config"]
                try:
                    self._dconfig["Config"] = utils.load(pfe, verb=verb)
                except Exception as err:
                    msg = str(err)
                    msg += "\n    type(pfe) = {0}".format(str(type(pfe)))
                    msg += "\n    self._dstrip['strip'] = {0}".format(
                        self._dstrip["strip"]
                    )
                    msg += "\n    strip = {0}".format(strip)
                    raise Exception(msg)

            self._dconfig["Config"].strip(strip, verb=verb)
        else:
            if strip == 4:
                path, name = self.config.Id.SavePath, self.config.Id.SaveName
                # --- Check !
                lf = os.listdir(path)
                lf = [
                    ff for ff in lf if all([s in ff for s in [name, ".npz"]])
                ]
                exist = len(lf) == 1
                # ----------
                pathfile = os.path.join(path, name) + ".npz"
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
                        warnings.warn(msg)
                    else:
                        raise Exception(msg)
                self._dconfig["Config"] = pathfile

            else:
                self._dconfig["Config"].strip(strip, verb=verb)

    def _strip_dsino(self, strip=0):
        if self._dstrip["strip"] == strip:
            return

        if strip < self._dstrip["strip"]:
            if strip <= 1 and self._dsino["k"] is not None:
                self._compute_dsino_extra()
        else:
            if self._dstrip["strip"] <= 1:
                utils.ToFuObject._strip_dict(self._dsino, lkeep=["RefPt", "k"])

    def _strip_dmisc(self, lkeep=["color"]):
        utils.ToFuObject._strip_dict(self._dmisc, lkeep=lkeep)

    ###########
    # _strip and get/from dict
    ###########

    @classmethod
    def _strip_init(cls):
        cls._dstrip["allowed"] = [0, 1, 2, 3, 4]
        nMax = max(cls._dstrip["allowed"])
        doc = """
                 1: dgeom w/o pts + config.strip(1)
                 2: dgeom w/o pts + config.strip(2) + dsino empty
                 3: dgeom w/o pts + config.strip(3) + dsino empty
                 4: dgeom w/o pts + config=pathfile + dsino empty
                 """
        doc = utils.ToFuObjectBase.strip.__doc__.format(doc, nMax)
        cls.strip.__doc__ = doc

    def strip(self, strip=0, verb=True):
        # super()
        super(Rays, self).strip(strip=strip, verb=verb)

    def _strip(self, strip=0, verb=True):
        self._strip_dconfig(strip=strip, verb=verb)
        self._strip_dgeom(strip=strip)
        self._strip_dsino(strip=strip)

    def _to_dict(self):
        dout = {
            "dconfig": {"dict": self._dconfig, "lexcept": None},
            "dgeom": {"dict": self.dgeom, "lexcept": None},
            "dchans": {"dict": self.dchans, "lexcept": None},
            "dsino": {"dict": self.dsino, "lexcept": None},
        }
        if self._is2D():
            dout["dX12"] = {"dict": self._dX12, "lexcept": None}
        return dout

    @classmethod
    def _checkformat_fromdict_dconfig(cls, dconfig):
        if dconfig["Config"] is None:
            return None
        if type(dconfig["Config"]) is dict:
            dconfig["Config"] = Config(fromdict=dconfig["Config"])
        lC = [
            isinstance(dconfig["Config"], Config),
            type(dconfig["Config"]) is str,
        ]
        assert any(lC)

    def _from_dict(self, fd):
        self._checkformat_fromdict_dconfig(fd["dconfig"])

        self._dconfig.update(**fd["dconfig"])
        self._dgeom.update(**fd["dgeom"])
        self._dsino.update(**fd["dsino"])
        if "dchans" in fd.keys():
            self._dchans.update(**fd["dchans"])
        if self._is2D():
            self._dX12.update(**fd["dX12"])


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
    def lOptics(self):
        return [self._dOptics['dobj'][k0][k1]
                for (k0, k1) in map(lambda x: str.split(x, '_'),
                                    self._dOptics['lorder'])]

    @property
    def isPinhole(self):
        c0 = "pinhole" in self._dgeom.keys()
        return c0 and self._dgeom["pinhole"] is not None

    @property
    def isInPoloidalPlane(self):
        phiD = np.arctan2(self.D[1, :], self.D[0, :])
        if self.nRays > 1 and not np.allclose(phiD[0], phiD[1:]):
            return False
        phiD = phiD[0]
        ephi = np.array([-np.sin(phiD), np.cos(phiD), 0.])[:, None]
        return np.allclose(np.sum(self.u*ephi, axis=0), 0.)

    @property
    def nRays(self):
        return self._dgeom["nRays"]

    @property
    def D(self):
        if self._dgeom["D"].shape[1] < self._dgeom["nRays"]:
            D = np.tile(self._dgeom["D"], self._dgeom["nRays"])
        else:
            D = self._dgeom["D"]
        return D

    @property
    def u(self):
        if self._dgeom['u'] is not None \
          and self._dgeom['u'].shape[1] == self._dgeom['nRays']:
            u = self._dgeom['u']
        elif self.isPinhole:
            u = self._dgeom['pinhole'][:, None] - self._dgeom['D']
            u = u / np.sqrt(np.sum(u**2, axis=0))[None, :]
        elif self._dgeom['u'].shape[1] < self._dgeom['nRays']:
            u = np.tile(self._dgeom['u'], self._dgeom['nRays'])
        return u

    @property
    def pinhole(self):
        if self._dgeom["pinhole"] is None:
            msg = "This is not a pinhole camera => pinhole is None"
            warnings.warn(msg)
        return self._dgeom["pinhole"]

    @property
    def config(self):
        return self._dconfig["Config"]

    @property
    def Etendues(self):
        if self._dgeom["Etendues"] is None:
            E = None
        elif self._dgeom["Etendues"].size == self._dgeom["nRays"]:
            E = self._dgeom["Etendues"]
        elif self._dgeom["Etendues"].size == 1:
            E = np.repeat(self._dgeom["Etendues"], self._dgeom["nRays"])
        else:
            msg = "Stored Etendues is not conform !"
            raise Exception(msg)
        return E

    @property
    def Surfaces(self):
        if self._dgeom["Surfaces"] is None:
            S = None
        elif self._dgeom["Surfaces"].size == self._dgeom["nRays"]:
            S = self._dgeom["Surfaces"]
        elif self._dgeom["Surfaces"].size == 1:
            S = np.repeat(self._dgeom["Surfaces"], self._dgeom["nRays"])
        else:
            msg = "Stored Surfaces not conform !"
            raise Exception(msg)
        return S

    @property
    def kIn(self):
        return self._dgeom["kIn"]

    @property
    def kOut(self):
        return self._dgeom["kOut"]

    @property
    def kMin(self):
        if self.isPinhole:
            kMin = self._dgeom["pinhole"][:, np.newaxis] - self._dgeom["D"]
            kMin = np.sqrt(np.sum(kMin ** 2, axis=0))
        else:
            kMin = 0.0
        return kMin

    @classmethod
    def _is2D(cls):
        c0 = "2d" in cls.__name__.lower()
        return c0

    @classmethod
    def _isLOS(cls):
        c0 = "los" in cls.__name__.lower()
        return c0

    ###########
    # Movement methods
    ###########

    def _update_or_copy(self, D, u, pinhole=None,
                        return_copy=None,
                        name=None, diag=None, dchans=None):
        if return_copy is None:
            return_copy = _RETURN_COPY
        if self.isPinhole is True:
            dgeom = {'pinhole': pinhole,
                     'D': D}
        else:
            dgeom = (D, u)
        if return_copy is True:
            if name is None:
                name = self.Id.Name + 'copy'
            if diag is None:
                diag = self.Id.Diag
            if dchans is None:
                dchans = self.dchans
            return self.__class__(dgeom=dgeom,
                                  lOptics=self.lOptics,
                                  Etendues=self.Etendues,
                                  Surfaces=self.Surfaces,
                                  config=self.config,
                                  sino_RefPt=self._dsino['RefPt'],
                                  color=self._dmisc['color'],
                                  dchans=dchans,
                                  Exp=self.Id.Exp,
                                  Diag=diag,
                                  Name=name,
                                  shot=self.Id.shot,
                                  SavePath=self.Id.SavePath)
        else:
            dgeom0 = ((self.D, self.pinhole)
                      if self.isPinhole is True else (self.D, self.u))
            try:
                self._set_dgeom(dgeom=dgeom,
                                Etendues=self.Etendues,
                                Surfaces=self.Surfaces,
                                sino_RefPt=self._dsino['RefPt'],
                                extra=True,
                                sino=True)
            except Exception as err:
                # Make sure instance does not move
                self._set_dgeom(dgeom=dgeom0,
                                Etendues=self.Etendues,
                                Surfaces=self.Surfaces,
                                sino_RefPt=self._dsino['RefPt'],
                                extra=True,
                                sino=True)
                msg = (str(err)
                       + "\nAn exception occured during updating\n"
                       + "  => instance unmoved")
                raise Exception(msg)

    def _rotate_DPinholeu(self, func, **kwdargs):
        pinhole, u = None, None
        if self.isPinhole is True:
            D = np.concatenate((self.D, self._dgeom['pinhole'][:, None]),
                               axis=1)
            D = func(pts=D, **kwdargs)
            D, pinhole = D[:, :-1], D[:, -1]
        elif 'rotate' in func.__name__:
            D, u = func(pts=self.D, vect=self.u, **kwdargs)
        else:
            D = func(pts=self.D, **kwdargs)
            u = self.u
        return D, pinhole, u

    def translate_in_cross_section(self, distance=None, direction_rz=None,
                                   phi=None,
                                   return_copy=None,
                                   diag=None, name=None, dchans=None):
        """ Translate the instance in the cross-section """
        if phi is None:
            if self.isInPoloidalPlane:
                phi = np.arctan2(*self.D[1::-1, 0])
            elif self.isPinhole:
                phi = np.arctan2(*self._dgeom['pinhole'][1::-1])
            else:
                msg = ("Instance not associated to a specific poloidal plane\n"
                       + "\tPlease specify which poloidal plane (phi) to use")
                raise Exception(msg)
        D, pinhole, u = self._rotate_DPinholeu(
            self._translate_pts_poloidal_plane,
            phi=phi, direction_rz=direction_rz, distance=distance)
        return self._update_or_copy(D, u, pinhole,
                                    return_copy=return_copy,
                                    diag=diag, name=name, dchans=dchans)

    def translate_3d(self, distance=None, direction=None,
                     return_copy=None,
                     diag=None, name=None, dchans=None):
        """ Translate the instance in provided direction """
        D, pinhole, u = self._rotate_DPinholeu(
            self._translate_pts_3d,
            direction=direction, distance=distance)
        return self._update_or_copy(D, u, pinhole,
                                    return_copy=return_copy,
                                    diag=diag, name=name, dchans=dchans)

    def rotate_in_cross_section(self, angle=None, axis_rz=None,
                                phi=None,
                                return_copy=None,
                                diag=None, name=None, dchans=None):
        """ Rotate the instance in the cross-section """
        if phi is None:
            if self.isInPoloidalPlane:
                phi = np.arctan2(*self.D[1::-1, 0])
            elif self.isPinhole:
                phi = np.arctan2(*self._dgeom['pinhole'][1::-1])
            else:
                msg = ("Camera not associated to a specific poloidal plane\n"
                       + "\tPlease specify which poloidal plane (phi) to use")
                raise Exception(msg)
        D, pinhole, u = self._rotate_DPinholeu(
            self._rotate_pts_vectors_in_poloidal_plane,
            axis_rz=axis_rz, angle=angle, phi=phi)
        return self._update_or_copy(D, u, pinhole,
                                    return_copy=return_copy,
                                    diag=diag, name=name, dchans=dchans)

    def rotate_around_torusaxis(self, angle=None,
                                return_copy=None,
                                diag=None, name=None, dchans=None):
        """ Rotate the instance around the torus axis """
        if self.config is not None and self.config.Id.Type != 'Tor':
            msg = "Movement only available for Tor configurations!"
            raise Exception(msg)
        D, pinhole, u = self._rotate_DPinholeu(
            self._rotate_pts_vectors_around_torusaxis,
            angle=angle)
        return self._update_or_copy(D, u, pinhole,
                                    return_copy=return_copy,
                                    diag=diag, name=name, dchans=dchans)

    def rotate_around_3daxis(self, angle=None, axis=None,
                             return_copy=None,
                             diag=None, name=None, dchans=None):
        """ Rotate the instance around the provided 3d axis """
        D, pinhole, u = self._rotate_DPinholeu(
            self._rotate_pts_vectors_around_3daxis,
            axis=axis, angle=angle)
        return self._update_or_copy(D, u, pinhole,
                                    return_copy=return_copy,
                                    diag=diag, name=name, dchans=dchans)

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

    ###########
    # public methods
    ###########

    def get_indStruct_computeInOut(self, unique_In=None):
        """ The indices of structures with compute = True

        The indidces refer to self.config.lStruct
            - The first array corresponds to Struct of type In
            - The second array corresponds to Struct of type Out
        """
        if unique_In is None:
            unique_In = False

        compute = self.config.get_compute()
        indIn = np.array([
            ii for ii, ss in enumerate(self.config.lStruct)
            if compute[ii] and ss._InOut == "in"
        ], dtype=np.int64)
        if unique_In is True and indIn.size > 1:
            iind = np.argmin([
                self.config.lStruct[ii].dgeom['Surf'] for ii in indIn
            ])
            indIn = np.r_[indIn[iind]]

        indOut = np.array([
            ii for ii, ss in enumerate(self.config.lStruct)
            if compute[ii] and ss._InOut == "out"
        ], dtype=np.int64)
        return indIn, indOut

    def _check_indch(self, ind, out=int):
        if ind is not None:
            ind = np.asarray(ind)
            assert ind.ndim == 1
            assert ind.dtype in [np.int64, np.bool_, int]
            if ind.dtype == np.bool_:
                assert ind.size == self.nRays
                if out is int:
                    indch = ind.nonzero()[0]
                else:
                    indch = ind
            else:
                assert np.max(ind) < self.nRays
                if out is bool:
                    indch = np.zeros((self.nRays,), dtype=bool)
                    indch[ind] = True
                else:
                    indch = ind
        else:
            if out is int:
                indch = np.arange(0, self.nRays)
            elif out is bool:
                indch = np.ones((self.nRays,), dtype=bool)
        return indch

    def select(self, key=None, val=None, touch=None, log="any", out=int):
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
            Tuple indicating you want the rays that are touching some specific
            elements of self.config:
                - touch[0] : str / int or list of such
                    str : a 'Cls_Name' string indicating the element
                    int : the index of the element in self.config.lStruct
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
                - int :  a (N,) array of int indices (N=number of matching
                         rays)

        Returns
        -------
        ind :   np.ndarray
            The array of matching rays

        """
        assert out in [int, bool]
        assert log in ["any", "all", "not"]
        C = [key is None, touch is None]
        assert np.sum(C) >= 1
        if np.sum(C) == 2:
            ind = np.ones((self.nRays,), dtype=bool)
        else:
            if key is not None:
                assert type(key) is str and key in self._dchans.keys()
                ltypes = [str, int, float, np.int64, np.float64]
                C0 = type(val) in ltypes
                C1 = type(val) in [list, tuple, np.ndarray]
                assert C0 or C1
                if C0:
                    val = [val]
                else:
                    assert all([type(vv) in ltypes for vv in val])
                ind = np.vstack([self._dchans[key] == ii for ii in val])
                if log == "any":
                    ind = np.any(ind, axis=0)
                elif log == "all":
                    ind = np.all(ind, axis=0)
                else:
                    ind = ~np.any(ind, axis=0)

            elif touch is not None:
                lint = [int, np.int64]
                larr = [list, tuple, np.ndarray]
                touch = [touch] if not type(touch) is list else touch
                assert len(touch) in [1, 2, 3]

                def _check_touch(tt):
                    cS = type(tt) is str and len(tt.split("_")) == 2
                    c0 = type(tt) in lint
                    c1 = type(tt) in larr and len(tt) >= 0
                    c1 = c1 and all([type(t) in lint for t in tt])
                    return cS, c0, c1

                for ii in range(0, 3 - len(touch)):
                    touch.append([])

                ntouch = len(touch)
                assert ntouch == 3

                for ii in range(0, ntouch):
                    cS, c0, c1 = _check_touch(touch[ii])

                    if not (cS or c0 or c1):
                        msg = "Provided touch is not valid:\n" % touch
                        msg += "    - Provided: %s\n" % str(touch)
                        msg += "Please provide either:\n"
                        msg += "    - str in the form 'Cls_Name'\n"
                        msg += "    - int (index)\n"
                        msg += "    - array of int indices"
                        raise Exception(msg)

                    if cS:
                        k0, k1 = touch[ii].split("_")
                        lS = self.config.lStruct
                        ind = [
                            jj
                            for jj, ss in enumerate(lS)
                            if ss.Id.Cls == k0 and ss.Id.Name == k1
                        ]
                        assert len(ind) == 1
                        touch[ii] = [ind[0]]
                    elif c0:
                        touch[ii] = [touch[ii]]

                # Common part
                ind = np.zeros((ntouch, self.nRays), dtype=bool)
                for i in range(0, ntouch):
                    if len(touch[i]) == 0:
                        ind[i, :] = True
                    else:
                        for n in range(0, len(touch[i])):
                            ind[i, :] = np.logical_or(
                                ind[i, :],
                                self._dgeom["indout"][i, :] == touch[i][n],
                            )
                ind = np.all(ind, axis=0)
                if log == "not":
                    ind[:] = ~ind
        if out is int:
            ind = ind.nonzero()[0]
        return ind

    def get_subset(self, indch=None, Name=None):
        """ Return an instance which is a sub-set of the camera

        The subset is the same camera but with only the LOS selected by indch
        It can be assigned a new Name (str), or the same one (True)
        """
        if indch is None:
            return self
        else:

            indch = self._check_indch(indch)
            dd = self.to_dict()
            sep = [kk for kk in dd.keys()
                   if all([ss in kk for ss in ['dId', 'dall', 'Name']])][0]
            sep = sep[3]

            # Name
            assert Name in [None, True] or type(Name) is str
            if Name is True:
                pass
            elif type(Name) is str:
                dd[sep.join(['dId', 'dall', 'Name'])] = Name
            elif Name is None:
                dd[sep.join(['dId', 'dall', 'Name'])] += "-subset"

            # Resize all np.ndarrays
            for kk in dd.keys():
                vv = dd[kk]
                c0 = isinstance(vv, np.ndarray) and self.nRays in vv.shape
                if c0:
                    if vv.ndim == 1:
                        dd[kk] = vv[indch]
                    elif vv.ndim == 2 and vv.shape[1] == self.nRays:
                        dd[kk] = vv[:, indch]
                dd[sep.join(['dgeom', 'nRays'])] = (
                    dd[sep.join(['dgeom', 'D'])].shape[1])

            # Recreate from dict
            obj = self.__class__(fromdict=dd)
        return obj

    def _get_plotL(
        self,
        reflections=True,
        Lplot=None,
        proj=None,
        ind=None,
        return_pts=False,
        multi=False,
    ):
        """ Get the (R,Z) coordinates of the cross-section projections """
        # Check inputs
        if Lplot is None:
            Lplot = 'tot'
        if proj is None:
            proj = 'All'

        # Compute
        ind = self._check_indch(ind)
        if ind.size > 0:
            us = self.u[:, ind]
            kOuts = np.atleast_1d(self.kOut[ind])[:, None]
            if Lplot.lower() == "tot":
                Ds = self.D[:, ind]
            else:
                Ds = self.D[:, ind] + self.kIn[None, ind] * us
                kOuts = kOuts - np.atleast_1d(self.kIn[ind])[:, None]
            if ind.size == 1:
                Ds, us = Ds[:, None], us[:, None]
            Ds, us = Ds[:, :, None], us[:, :, None]
            # kRMin = None

            # Add reflections ?
            c0 = (
                reflections
                and self._dgeom.get("dreflect") is not None
                and self._dgeom["dreflect"].get("us") is not None
            )
            if c0:
                Dsadd = self._dgeom["dreflect"]["Ds"][:, ind, :]
                usadd = self._dgeom["dreflect"]["us"][:, ind, :]
                kOutsadd = self._dgeom["dreflect"]["kouts"][ind, :]
                if ind.size == 1:
                    Dsadd, usadd = Dsadd[:, None, :], usadd[:, None, :]
                    kOutsadd = kOutsadd[None, :]
                Ds = np.concatenate((Ds, Dsadd), axis=-1)
                us = np.concatenate((us, usadd), axis=-1)
                kOuts = np.concatenate((kOuts, kOutsadd), axis=-1)
                # if self.config.Id.Type == "Tor":
                # kRMin = _comp.LOS_PRMin(
                # Ds, us, kOut=kOuts, Eps=1.0e-12, squeeze=False
                # )

                # elif self.config.Id.Type == "Tor":
                # kRMin = self._dgeom["kRMin"][ind][:, None]

            out = _comp.LOS_CrossProj(
                self.config.Id.Type,
                Ds,
                us,
                kOuts,
                proj=proj,
                return_pts=return_pts,
                multi=multi,
            )
        else:
            out = None
        return out

    def get_sample(
        self,
        res=None,
        resMode="abs",
        DL=None,
        method="sum",
        ind=None,
        pts=False,
        compact=True,
        num_threads=_NUM_THREADS,
        Test=True,
    ):
        """ Return a linear sampling of the LOS

        The LOS is sampled into a series a points and segments lengths
        The resolution (segments length) is <= res
        The sampling can be done according to different methods
        It is possible to sample only a subset of the LOS

        Parameters
        ----------
        res:        float
            Desired resolution
        resMode:    str
            Flag indicating res should be understood as:
                - 'abs':    an absolute distance in meters
                - 'rel':    a relative distance (fraction of the LOS length)
        DL:         None / iterable
            The fraction [L1;L2] of the LOS that should be sampled, where
            L1 and L2 are distances from the starting point of the LOS (LOS.D)
            DL can be an iterable of len()==2 (identical to all los), or a
            (2,nlos) array
        method:     str
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
        ind:        None / iterable of int
            indices of the LOS to be sampled
        pts:        bool
            Flag indicating whether to return only the abscissa parameter k
            (False) or the 3D pts coordinates (True)
        compact:    bool
            Flag incating whether to retrun the sampled pts of all los in a
            single concatenated array (True) or splitted into
            a list of nlos arrays)

        Returns
        -------
        k:      np.ndarray
            if pts == False:
                A (npts,) array of the abscissa parameters
                  (i.e.: points distances from the LOS starting points)
                In order to get the 3D cartesian coordinates of pts do:
            if pts == True:
                A (3,npts) array of the sampled points 3D cartesian coordinates
        reseff: np.ndarray
            A (nlos,) array of the effective resolution (<= res input), as an
            absolute distance
        ind:    np.ndarray
            A (nlos-1,) array of integere indices (where to split k to separate
            the points of each los). e.g.: lk = np.split(k,ind)

        """
        if res is None:
            res = _RES
        ind = self._check_indch(ind)
        # preload k
        kIn = self.kIn
        kOut = self.kOut

        # Preformat DL
        if DL is None:
            DL = np.array([kIn[ind], kOut[ind]])
        elif np.asarray(DL).size == 2:
            DL = np.tile(np.asarray(DL).ravel(), (len(ind), 1)).T
        DL = np.ascontiguousarray(DL).astype(float)
        assert type(DL) is np.ndarray and DL.ndim == 2
        assert DL.shape == (2, len(ind)), "Arg DL has wrong shape !"

        # Check consistency of limits
        ii = DL[0, :] < kIn[ind]
        DL[0, ii] = kIn[ind][ii]
        ii[:] = DL[0, :] >= kOut[ind]
        DL[0, ii] = kOut[ind][ii]
        ii[:] = DL[1, :] > kOut[ind]
        DL[1, ii] = kOut[ind][ii]
        ii[:] = DL[1, :] <= kIn[ind]
        DL[1, ii] = kIn[ind][ii]

        # Preformat Ds, us
        Ds, us = self.D[:, ind], self.u[:, ind]
        if len(ind) == 1:
            Ds, us = Ds.reshape((3, 1)), us.reshape((3, 1))
        Ds, us = np.ascontiguousarray(Ds), np.ascontiguousarray(us)

        # Launch    # NB : find a way to exclude cases with DL[0,:]>=DL[1,:] !!
        # Todo : reverse in _GG : make compact default for faster computation !
        nlos = Ds.shape[1]
        k, reseff, lind = _GG.LOS_get_sample(
            nlos,
            res,
            DL,
            dmethod=resMode,
            method=method,
            num_threads=num_threads,
            Test=Test,
        )
        if pts:
            nbrep = np.r_[lind[0], np.diff(lind), k.size - lind[-1]]
            k = np.repeat(Ds, nbrep, axis=1) + k[None, :] * np.repeat(
                us, nbrep, axis=1
            )
        if not compact:
            k = np.split(k, lind, axis=-1)
        return k, reseff, lind

    def _kInOut_Isoflux_inputs(self, lPoly, lVIn=None):

        if self._method == "ref":
            D, u = np.ascontiguousarray(self.D), np.ascontiguousarray(self.u)
            Lim = self.config.Lim
            nLim = self.config.nLim
            Type = self.config.Id.Type

            largs = [D, u, lPoly[0], lVIn[0]]
            dkwd = dict(Lim=Lim, nLim=nLim, VType=Type)
        elif self._method == "optimized":
            D = np.ascontiguousarray(self.D)
            u = np.ascontiguousarray(self.u)
            if np.size(self.config.Lim) == 0 or self.config.Lim is None:
                Lim = np.array([])
            else:
                Lim = np.asarray(self.config.Lim)
                if np.size(np.shape(Lim)) > 1:
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

    def _kInOut_Isoflux_inputs_usr(self, lPoly, lVIn=None):
        c0 = type(lPoly) in [np.ndarray, list, tuple]

        # Check lPoly
        if c0 and type(lPoly) is np.ndarray:
            c0 = c0 and lPoly.ndim in [2, 3]
            if c0 and lPoly.ndim == 2:
                c0 = c0 and lPoly.shape[0] == 2
                if c0:
                    lPoly = [np.ascontiguousarray(lPoly)]
            elif c0:
                c0 = c0 and lPoly.shape[1] == 2
                if c0:
                    lPoly = np.ascontiguousarray(lPoly)
        elif c0:
            lPoly = [np.ascontiguousarray(pp) for pp in lPoly]
            c0 = all([pp.ndim == 2 and pp.shape[0] == 2 for pp in lPoly])
        if not c0:
            msg = "Arg lPoly must be either:\n"
            msg += "    - a (2,N) np.ndarray (signle polygon of N points)\n"
            msg += "    - a list of M polygons, each a (2,Ni) np.ndarray\n"
            msg += "        - where Ni is the number of pts of each polygon\n"
            msg += "    - a (M,2,N) np.ndarray where:\n"
            msg += "        - M is the number of polygons\n"
            msg += "        - N is the (common) number of points per polygon\n"
            raise Exception(msg)
        nPoly = len(lPoly)

        # Check anti-clockwise and closed
        if type(lPoly) is list:
            for ii in range(nPoly):
                # Check closed and anti-clockwise
                if not np.allclose(lPoly[ii][:, 0], lPoly[ii][:, -1]):
                    lPoly[ii] = np.concatenate(
                        (lPoly[ii], lPoly[ii][:, 0:1]), axis=-1
                    )
                try:
                    if _GG.Poly_isClockwise(lPoly[ii]):
                        lPoly[ii] = lPoly[ii][:, ::-1]
                except Exception as excp:
                    print("For structure ", ii, " : ", excp)
        else:
            # Check closed and anti-clockwise
            d = np.sum((lPoly[:, :, 0]-lPoly[:, :, -1])**2, axis=1)
            if np.allclose(d, 0.):
                pass
            elif np.all(d > 0.):
                lPoly = np.concatenate((lPoly, lPoly[:, :, 0:1]), axis=-1)
            else:
                msg = "All poly in lPoly should be closed or all non-closed!"
                raise Exception(msg)
            for ii in range(nPoly):
                try:
                    if _GG.Poly_isClockwise(lPoly[ii]):
                        lPoly[ii] = lPoly[ii][:, ::-1]
                except Exception as excp:
                    print("For structure ", ii, " : ", excp)

        # Check lVIn
        if lVIn is None:
            lVIn = []
            for pp in lPoly:
                vIn = np.diff(pp, axis=1)
                vIn = vIn/(np.sqrt(np.sum(vIn**2, axis=0))[None, :])
                vIn = np.ascontiguousarray([-vIn[1, :], vIn[0, :]])
                lVIn.append(vIn)
        else:
            c0 = type(lVIn) in [np.ndarray, list, tuple]
            if c0 and type(lVIn) is np.ndarray and lVIn.ndim == 2:
                c0 = c0 and lVIn.shape == (2, lPoly[0].shape[1]-1)
                if c0:
                    lVIn = [np.ascontiguousarray(lVIn)]
            elif c0 and type(lVIn) is np.ndarray:
                c0 = c0 and lVIn.shape == (nPoly, 2, lPoly.shape[-1]-1)
                if c0:
                    lVIn = np.ascontiguousarray(lVIn)
            elif c0:
                c0 = c0 and len(lVIn) == nPoly
                if c0:
                    c0 = c0 and all([vv.shape == (2, pp.shape[1]-1)
                                     for vv, pp in zip(lVIn, lPoly)])
                    if c0:
                        lVIn = [np.ascontiguousarray(vv) for vv in lVIn]

            # Check normalization and direction
            for ii in range(0, nPoly):
                lVIn[ii] = (lVIn[ii]
                            / np.sqrt(np.sum(lVIn[ii]**2, axis=0))[None, :])
                vect = np.diff(lPoly[ii], axis=1)
                vect = vect / np.sqrt(np.sum(vect**2, axis=0))[None, :]
                det = vect[0, :]*lVIn[ii][1, :] - vect[1, :]*lVIn[ii][0, :]
                if not np.allclose(np.abs(det), 1.):
                    msg = "Each lVIn must be perp. to each lPoly segment !"
                    raise Exception(msg)
                ind = np.abs(det+1) < 1.e-12
                lVIn[ii][:, ind] = -lVIn[ii][:, ind]

        return nPoly, lPoly, lVIn

    def calc_kInkOut_Isoflux(self, lPoly, lVIn=None, Lim=None,
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
        nPoly, lPoly, lVIn = self._kInOut_Isoflux_inputs_usr(lPoly, lVIn=lVIn)

        # Prepare output
        kIn = np.full((nPoly, self.nRays), np.nan)
        kOut = np.full((nPoly, self.nRays), np.nan)

        # Compute intersections
        assert(self._method in ['ref', 'optimized'])
        if self._method == 'ref':
            for ii in range(0, nPoly):
                largs, dkwd = self._kInOut_Isoflux_inputs([lPoly[ii]],
                                                          lVIn=[lVIn[ii]])
                out = _GG.SLOW_LOS_Calc_PInOut_VesStruct(*largs, **dkwd)
                # PIn, POut, kin, kout, VperpIn, vperp, IIn, indout = out[]
                kIn[ii, :], kOut[ii, :] = out[2], out[3]
        elif self._method == "optimized":
            for ii in range(0, nPoly):
                largs, dkwd = self._kInOut_Isoflux_inputs([lPoly[ii]],
                                                          lVIn=[lVIn[ii]])

                out = _GG.LOS_Calc_PInOut_VesStruct(*largs, **dkwd)[:2]
                kIn[ii, :], kOut[ii, :] = out
        if kInOut:
            indok = ~np.isnan(kIn)
            ind = np.zeros((nPoly, self.nRays), dtype=bool)
            kInref = np.tile(self.kIn, (nPoly, 1))
            kOutref = np.tile(self.kOut, (nPoly, 1))
            ind[indok] = (kIn[indok] < kInref[indok])
            ind[indok] = ind[indok] | (kIn[indok] > kOutref[indok])
            kIn[ind] = np.nan

            ind[:] = False
            indok[:] = ~np.isnan(kOut)
            ind[indok] = (kOut[indok] < kInref[indok]) | (
                kOut[indok] > kOutref[indok]
            )
            kOut[ind] = np.nan

        return kIn, kOut

    def calc_length_in_isoflux(self, lPoly, lVIn=None, Lim=None, kInOut=True):
        """ Return the length of each LOS inside each isoflux

        Uses self.calc_kInkOut_Isoflux() to compute the linear abscissa (k) of
        the entry points (kIn) and exit points (kOut) for each LOS

        The isofluxes must be provided as a list of polygons

        The length is returned as a (nPoly, nLOS) 2d array

        """
        kIn, kOut = self.calc_kInkOut_Isoflux(lPoly, lVIn=lVIn, Lim=Lim,
                                              kInOut=kInOut)
        return kOut-kIn

    def calc_min_geom_radius(self, axis):
        """ Return the minimum geom. radius of each LOS, from an arbitrary axis

        The axis mut be provided as a (R,Z) iterable
        Uses self.set_dsino()

        Return:
        -------
        p:      np.ndarray
            (nLOS,) array of minimal radius (or impact parameter)
        theta:  np.ndarray
            (nLOS,) array of associated theta with respect to axis
        pts:    np.ndarray
            (3,nLOS) array of (X,Y,Z) coordinates of associated points on LOS
        """
        self.set_dsino(RefPt=axis, extra=True)
        p, theta, pts = self.dsino['p'], self.dsino['theta'], self.dsino['pts']
        return p, theta, pts

    def calc_min_rho_from_Plasma2D(
        self,
        plasma=None,
        t=None,
        indt_strict=None,
        log='min',
        res=None,
        resMode='abs',
        method='sum',
        quant=None,
        pts=False,
        Test=True,
    ):
        """ Return the min/max value of scalar field quant for each LOS

        Typically used to get the minimal normalized minor radius
        But can be used for any quantity available in plasma if:
            - it is a 2d profile
            - it is a 1d profile that can be interpolated on a 2d mesh

        Currently sample each LOS with desired resolution and returns the
        absolute min/max interpolated value (and associated point)

        See self.get_sample() for details on sampling arguments:
            - res, resMode, method
        See Plasma2D.interp_pts2profile() for details on interpolation args:
            - t, quant, q2dref, q1dref

        Returns:
        --------
        val:        np.ndarray
            (nt, nLOS) array of min/max values
        pts:        np.ndarray
            (nt, nLOS, 3) array of (X,Y,Z) coordinates of associated points
            Only returned if pts = True
        t:          np.ndarray
            (nt,) array of time steps at which the interpolations were made
        """
        assert log in ['min', 'max']
        assert isinstance(pts, bool)

        # Sample LOS
        ptsi, reseff, lind = self.get_sample(
            res=res, resMode=resMode, DL=None,
            method=method, ind=None,
            pts=True, compact=True, Test=True,
        )

        # get mesh
        wbs = plasma._which_bsplines
        # keym = plasma.ddata[quant][wm]
        keybs = plasma.ddata[quant][wbs][0]
        key_reftime = plasma.ddata[quant]['ref'][0]
        key_time = plasma.get_ref_vector(ref=key_reftime, units='s')[3]
        time = plasma.ddata[key_time]['data']
        dt = [
                np.max(time[time < np.min(t)]),
                np.min(time[time > np.max(t)]) + 1e-6,
        ]

        # Interpolate values
        dout = plasma.interpolate(
            keys=quant,
            ref_key=keybs,
            x0=np.hypot(ptsi[0, ...], ptsi[1, ...]),
            x1=ptsi[2, ...],
            grid=False,
            res=res,
            mode=None,
            submesh=None,
            domain={key_time: dt},
            ref_com=None,
            details=False,
            indbs_tf=None,
            crop=None,
            deg=None,
            deriv=None,
            val_out=None,
            log_log=None,
            nan0=None,
            returnas=None,
            return_params=False,
            store=False,
            inplace=None,
            debug=None,
        )
        val = dout[quant]['data']

        # interpolate time
        indt = (time >= dt[0]) & (time <= dt[1])
        val = scpinterp.interp1d(
            time[indt],
            val,
            axis=0,
            kind='linear',
        )(t)

        # Separate val per LOS and compute min / max
        func = np.nanmin if log == 'min' else np.nanmax
        if pts:
            funcarg = np.nanargmin if log == 'min' else np.nanargmax

        if pts:
            nt = val.shape[0]
            pts = np.full((3, self.nRays, nt), np.nan)
            vals = np.full((nt, self.nRays), np.nan)
            # indt = np.arange(0, nt)
            lind = np.r_[0, lind, ptsi.shape[1]]
            for ii in range(self.nRays):
                indok = ~np.all(np.isnan(val[:, lind[ii]:lind[ii+1]]), axis=1)
                if np.any(indok):
                    vals[indok, ii] = func(
                        val[indok, lind[ii]:lind[ii+1]],
                        axis=1,
                    )
                    ind = funcarg(val[indok, lind[ii]:lind[ii+1]], axis=1)
                    pts[:, ii, indok] = ptsi[:, lind[ii]:lind[ii+1]][:, ind]
            pts = pts.T

        else:
            pts = None
            vals = np.column_stack([
                func(vv, axis=1)
                for vv in np.split(val, lind, axis=-1)
            ])
        return vals, pts, t

    def get_inspector(self, ff):
        out = inspect.signature(ff)
        pars = out.parameters.values()
        na = np.sum([(pp.kind == pp.POSITIONAL_OR_KEYWORD
                      and pp.default is pp.empty) for pp in pars])
        kw = [pp.name for pp in pars if (pp.kind == pp.POSITIONAL_OR_KEYWORD
                                         and pp.default is not pp.empty)]
        return na, kw

    def check_ff(self, ff, t=None, ani=None):
        # Initialization of function wrapper
        wrapped_ff = ff

        # Define unique error message giving all info in a concise way
        # Optionnally add error-specific line afterwards
        msg = ("User-defined emissivity function ff must:\n"
               + "\t- be a callable (function)\n"
               + "\t- take only one positional arg "
               + "and at least one keyword arg:\n"
               + "\t\t - ff(pts, t=None), where:\n"
               + "\t\t\t - pts is a (3, npts) of (x, y, z) coordinates\n"
               + "\t\t\t - t can be None / scalar / iterable of len(t) = nt\n"
               + "\t- Always return a 2d (nt, npts) np.ndarray, where:\n"
               + "\t\t - nt = len(t) if t is an iterable\n"
               + "\t\t - nt = 1 if t is None or scalar\n"
               + "\t\t - npts is the number of pts (pts.shape[1])\n\n"
               + "\t- Optionally, ff can take an extra keyword arg:\n"
               + "\t\t - ff(pts, vect=None, t=None), where:\n"
               + "\t\t\t - vect is a (3, npts) np.ndarray\n"
               + "\t\t\t - vect contains the (x, y, z) coordinates "
               + "of the units vectors of the photon emission directions"
               + "for each pts. Present only for anisotropic emissivity, "
               + "unless specifically indicated otherwise "
               + "(with ani=False in LOS_calc_signal).\n"
               + "\t\t\tDoes not affect the outpout shape (still (nt, npts))")

        # .. Checking basic definition of function ..........................
        if not hasattr(ff, '__call__'):
            msg += "\n\n  => ff must be a callable (function)!"
            raise Exception(msg)

        npos_args, kw = self.get_inspector(ff)
        if npos_args != 1:
            msg += "\n\n  => ff must take only 1 positional arg: ff(pts)!"
            raise Exception(msg)

        if 't' not in kw:
            msg += "\n\n  => ff must have kwarg 't=None' for time vector!"
            raise Exception(msg)

        # .. Checking time vector .........................................
        ltypeok = [int, float, np.int64, np.float64]
        is_t_type_valid = (type(t) in ltypeok or hasattr(t, '__iter__'))
        if not (t is None or is_t_type_valid):
            msg += "\n\n  => t must be None, scalar or iterable !"
            raise Exception(msg)
        nt = len(t) if hasattr(t, '__iter__') else 1

        # .. Test anisotropic case .......................................
        if ani is None:
            is_ani = ('vect' in kw)
        else:
            assert isinstance(ani, bool)
            is_ani = ani

        # .. Testing outputs ...............................................
        test_pts = np.array([[1, 2], [3, 4], [5, 6]])
        npts = test_pts.shape[1]
        if is_ani:
            vect = np.ones(test_pts.shape)
            try:
                out = ff(test_pts, vect=vect, t=t)
            except Exception:
                msg += "\n\n  => ff must take ff(pts, vect=vect, t=t) !"
                raise Exception(msg)
        else:
            try:
                out = ff(test_pts, t=t)
            except Exception:
                msg += "\n\n  => ff must take a ff(pts, t=t) !"
                raise Exception(msg)

        if not (isinstance(out, np.ndarray) and (out.shape == (nt, npts)
                                                 or out.shape == (npts,))):
            msg += "\n\n  => wrong output (always 2d np.ndarray) !"
            raise Exception(msg)

        if nt == 1 and out.shape == (npts,):
            def wrapped_ff(*args, **kwargs):
                res_ff = ff(*args, **kwargs)
                return np.reshape(res_ff, (1, -1))

        return is_ani, wrapped_ff

    def _calc_signal_preformat(self, ind=None, DL=None, t=None,
                               out=object, Brightness=True):
        msg = "Arg out must be in [object,np.ndarray]"
        assert out in [object, np.ndarray], msg
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
        elif np.asarray(DL).size == 2:
            DL = np.tile(np.asarray(DL).ravel()[:, np.newaxis], len(ind))
        DL = np.ascontiguousarray(DL).astype(float)
        assert type(DL) is np.ndarray and DL.ndim == 2
        assert DL.shape == (2, len(ind)), "Arg DL has wrong shape !"

        # check limits
        ii = DL[0, :] < kIn[ind]
        DL[0, ii] = kIn[ind][ii]
        ii[:] = DL[0, :] >= kOut[ind]
        DL[0, ii] = kOut[ind][ii]
        ii[:] = DL[1, :] > kOut[ind]
        DL[1, ii] = kOut[ind][ii]
        ii[:] = DL[1, :] <= kIn[ind]
        DL[1, ii] = kIn[ind][ii]

        # Preformat Ds, us and Etendue
        Ds, us = self.D[:, ind], self.u[:, ind]
        E = None
        if Brightness is False:
            E = self.Etendues
            if E.size == self.nRays:
                E = E[ind]

        # Preformat signal
        if len(ind) == 1:
            Ds, us = Ds.reshape((3, 1)), us.reshape((3, 1))
        indok = ~(
            np.any(np.isnan(DL), axis=0)
            | np.any(np.isinf(DL), axis=0)
            | ((DL[1, :] - DL[0, :]) <= 0.0)
        )

        if np.any(indok):
            Ds, us, DL = Ds[:, indok], us[:, indok], DL[:, indok]
            if indok.sum() == 1:
                Ds, us = Ds.reshape((3, 1)), us.reshape((3, 1))
                DL = DL.reshape((2, 1))
            Ds, us = np.ascontiguousarray(Ds), np.ascontiguousarray(us)
            DL = np.ascontiguousarray(DL)
        else:
            Ds, us, DL = None, None, None
        return indok, Ds, us, DL, E

    def _calc_signal_postformat(
        self,
        sig,
        Brightness=True,
        dataname=None,
        t=None,
        E=None,
        units=None,
        plot=True,
        out=object,
        fs=None,
        dmargin=None,
        wintit=None,
        invert=True,
        draw=True,
        connect=True,
    ):
        if Brightness is False:
            if dataname is None:
                dataname = r"LOS-integral x Etendue"
            if E is None or np.all(np.isnan(E)):
                msg = "Cannot use etendue, it was not set properly !"
                raise Exception(msg)
            if t is None or len(t) == 1 or E.size == 1:
                sig = sig * E
            else:
                sig = sig * E[np.newaxis, :]
            if units is None:
                units = r"origin x $m^3.sr$"
        else:
            if dataname is None:
                dataname = r"LOS-integral"
            if units is None:
                units = r"origin x m"

        if plot or out in [object, "object"]:
            kwdargs = dict(
                data=sig,
                t=t,
                lCam=self,
                Name=self.Id.Name,
                dlabels={"data": {"units": units, "name": dataname}},
                Exp=self.Id.Exp,
                Diag=self.Id.Diag,
            )
            import tofu.data as tfd

            if self._is2D():
                osig = tfd.DataCam2D(**kwdargs)
            else:
                osig = tfd.DataCam1D(**kwdargs)
            if plot:
                _ = osig.plot(
                    fs=fs,
                    dmargin=dmargin,
                    wintit=wintit,
                    invert=invert,
                    draw=draw,
                    connect=connect,
                )

        if out in [object, "object"]:
            return osig, units
        else:
            return sig, units

    def calc_signal(
        self,
        func,
        t=None,
        ani=None,
        fkwdargs={},
        Brightness=True,
        res=None,
        DL=None,
        resMode="abs",
        method="sum",
        minimize="calls",
        num_threads=16,
        reflections=True,
        coefs=None,
        coefs_reflect=None,
        ind=None,
        returnas=object,
        plot=True,
        dataname=None,
        fs=None,
        dmargin=None,
        wintit=None,
        invert=True,
        units=None,
        draw=True,
        connect=True,
        newcalc=True,
    ):
        """ Return the line-integrated emissivity

        Beware, by default, Brightness=True and it is only a line-integral !

        Indeed, to get the received power, you need an estimate of the Etendue
        (previously set using self.set_Etendues()) and use Brightness=False.

        Hence, if Brightness=True and if
        the emissivity is provided in W/m3 (resp. W/m3/sr),
        => the method returns W/m2 (resp. W/m2/sr)
        The line is sampled using :meth:`~tofu.geom.LOS.get_sample`,

        Except func, arguments common to :meth:`~tofu.geom.LOS.get_sample`

        Parameters
        ----------
        func :    callable
            The user-provided emissivity function
            Shall take at least:
                func(pts, t=None, vect=None)
            where:
                - pts : (3,N) np.ndarray, (X,Y,Z) coordinates of points
                - t   : None / (nt,) np.ndarray, time vector
                - vect: None / (3,N) np.ndarray, unit direction vectors (X,Y,Z)
            Should return at least:
                - val : (N,) np.ndarray, local emissivity values
        method : string, the integral can be computed using 3 different methods
            - 'sum':    A numpy.sum() on the local values (x segments) DEFAULT
            - 'simps':  using :meth:`scipy.integrate.simps`
            - 'romb':   using :meth:`scipy.integrate.romb`
        minimize : string, method to minimize for computation optimization
            - "calls": minimal number of calls to `func` (default)
            - "memory": slowest method, to use only if "out of memory" error
            - "hybrid": mix of before-mentioned methods.


        Returns
        -------
        sig :   np.ndarray
            The computed signal, a 1d or 2d array depending on whether a time
            vector was provided.
        units:  str
            Units of the result

        """

        # Format input

        indok, Ds, us, DL, E = self._calc_signal_preformat(
            ind=ind, DL=DL, out=returnas, Brightness=Brightness
        )

        if Ds is None:
            return None
        if res is None:
            res = _RES

        # Launch    # NB : find a way to exclude cases with DL[0,:]>=DL[1,:] !!
        # Exclude Rays not seeing the plasma
        if newcalc:
            ani, func = self.check_ff(func, t=t, ani=ani)
            s = _GG.LOS_calc_signal(
                func,
                Ds,
                us,
                res,
                DL,
                dmethod=resMode,
                method=method,
                ani=ani,
                t=t,
                fkwdargs=fkwdargs,
                minimize=minimize,
                num_threads=num_threads,
                Test=True,
            )

            c0 = (
                reflections
                and self._dgeom["dreflect"] is not None
                and self._dgeom["dreflect"].get("nb", 0) > 0
            )
            if c0:
                if coefs_reflect is None:
                    coefs_reflect = 1.0
                for ii in range(self._dgeom["dreflect"]["nb"]):
                    Dsi = np.ascontiguousarray(
                        self._dgeom["dreflect"]["Ds"][:, :, ii]
                    )
                    usi = np.ascontiguousarray(
                        self._dgeom["dreflect"]["us"][:, :, ii]
                    )
                    s += coefs_reflect * _GG.LOS_calc_signal(
                        func,
                        Dsi,
                        usi,
                        res,
                        DL,
                        dmethod=resMode,
                        method=method,
                        ani=ani,
                        t=t,
                        fkwdargs=fkwdargs,
                        minimize=minimize,
                        num_threads=num_threads,
                        Test=True,
                    )

            # Integrate
            # Creating the arrays with null everywhere..........
            if s.ndim == 2:
                sig = np.full((s.shape[0], self.nRays), np.nan)
            else:
                sig = np.full((1, self.nRays), np.nan)
            if t is None or len(t) == 1:
                sig[0, indok] = s
            else:
                sig[:, indok] = s
        else:
            # Get ptsRZ along LOS // Which to choose ???
            pts, reseff, indpts = self.get_sample(
                res,
                resMode=resMode,
                DL=DL,
                method=method,
                ind=ind,
                compact=True,
                pts=True,
            )

            if ani:
                nbrep = np.r_[
                    indpts[0], np.diff(indpts), pts.shape[1] - indpts[-1]
                ]
                vect = np.repeat(self.u, nbrep, axis=1)
            else:
                vect = None

            # Get quantity values at ptsRZ
            # This is the slowest step (~3.8 s with res=0.02
            #    and interferometer)
            val = func(pts, t=t, vect=vect)
            # Integrate
            sig = np.add.reduceat(val, np.r_[0, indpts],
                                  axis=-1)*reseff[None, :]

        # Apply user-provided coefs
        if coefs is not None:
            if hasattr(coefs, '__iter__'):
                coefs = np.atleast_1d(coefs).ravel()
                assert coefs.shape == (sig.shape[-1],)
                if sig.ndim == 2:
                    coefs = coefs[None, :]
            sig *= coefs

        # Format output
        return self._calc_signal_postformat(
            sig,
            Brightness=Brightness,
            dataname=dataname,
            t=t,
            E=E,
            units=units,
            plot=plot,
            out=returnas,
            fs=fs,
            dmargin=dmargin,
            wintit=wintit,
            invert=invert,
            draw=draw,
            connect=connect,
        )

    def calc_signal_from_Plasma2D(
        self,
        plasma2d,
        t=None,
        newcalc=True,
        quant=None,
        ref1d=None,
        ref2d=None,
        q2dR=None,
        q2dPhi=None,
        q2dZ=None,
        Type=None,
        Brightness=True,
        interp_t="nearest",
        interp_space=None,
        fill_value=None,
        res=None,
        DL=None,
        resMode="abs",
        method="sum",
        minimize="calls",
        num_threads=16,
        reflections=True,
        coefs=None,
        coefs_reflect=None,
        ind=None,
        returnas=object,
        plot=True,
        dataname=None,
        fs=None,
        dmargin=None,
        wintit=None,
        invert=True,
        units=None,
        draw=True,
        connect=True,
    ):

        # Format input
        indok, Ds, us, DL, E = self._calc_signal_preformat(
            ind=ind, out=returnas, t=t, Brightness=Brightness
        )

        if Ds is None:
            return None
        if res is None:
            res = _RES

        if newcalc:
            # Get time vector
            lc = [t is None, type(t) is str, type(t) is np.ndarray]
            assert any(lc)
            if lc[0]:
                out = plasma2d._checkformat_qr12RPZ(
                     quant=quant,
                     ref1d=ref1d,
                     ref2d=ref2d,
                     q2dR=q2dR,
                     q2dPhi=q2dPhi,
                     q2dZ=q2dZ,
                )
                t = plasma2d._get_tcom(*out[:4])[0]
            elif lc[1]:
                t = plasma2d._ddata[t]['data']
            else:
                t = np.atleast_1d(t).ravel()

            if fill_value is None:
                fill_value = 0.0

            func = plasma2d.get_finterp2d(
                quant=quant,
                ref1d=ref1d,
                ref2d=ref2d,
                q2dR=q2dR,
                q2dPhi=q2dPhi,
                q2dZ=q2dZ,
                interp_t=interp_t,
                interp_space=interp_space,
                fill_value=fill_value,
                Type=Type,
            )

            def funcbis(*args, **kwdargs):
                return func(*args, **kwdargs)[0]

            if DL is None:
                # set to [kIn,kOut]
                DL = None
            ani = quant is None
            if num_threads is None:
                num_threads = _NUM_THREADS

            if np.all(indok):
                D, u = self.D, self.u
            else:
                D = np.ascontiguousarray(self.D[:, indok])
                u = np.ascontiguousarray(self.u[:, indok])

            sig = _GG.LOS_calc_signal(
                funcbis,
                D,
                u,
                res,
                DL,
                dmethod=resMode,
                method=method,
                ani=ani,
                t=t,
                fkwdargs={},
                minimize=minimize,
                Test=True,
                num_threads=num_threads,
            )
            c0 = (
                reflections
                and self._dgeom["dreflect"] is not None
                and self._dgeom["dreflect"].get("nb", 0) > 0
            )
            if c0:
                if coefs_reflect is None:
                    coefs_reflect = 1.0
                for ii in range(self._dgeom["dreflect"]["nb"]):
                    Dsi = np.ascontiguousarray(
                        self._dgeom["dreflect"]["Ds"][:, :, ii]
                    )
                    usi = np.ascontiguousarray(
                        self._dgeom["dreflect"]["us"][:, :, ii]
                    )
                    sig += coefs_reflect * _GG.LOS_calc_signal(
                        funcbis,
                        Dsi,
                        usi,
                        res,
                        DL,
                        dmethod=resMode,
                        method=method,
                        ani=ani,
                        t=t,
                        fkwdargs={},
                        minimize=minimize,
                        num_threads=num_threads,
                        Test=True,
                    )
        else:
            # Get ptsRZ along LOS // Which to choose ???
            pts, reseff, indpts = self.get_sample(
                res,
                resMode=resMode,
                DL=DL,
                method=method,
                ind=ind,
                compact=True,
                pts=True,
            )
            if q2dR is None:
                vect = None
            else:
                nbrep = np.r_[
                    indpts[0], np.diff(indpts), pts.shape[1] - indpts[-1]
                ]
                vect = -np.repeat(self.u, nbrep, axis=1)
            if fill_value is None:
                fill_value = 0.

            # Get quantity values at ptsRZ
            # This is the slowest step (~3.8 s with res=0.02
            #    and interferometer)
            val, t = plasma2d.interp_pts2profile(
                pts=pts,
                vect=vect,
                t=t,
                quant=quant,
                ref1d=ref1d,
                ref2d=ref2d,
                q2dR=q2dR,
                q2dPhi=q2dPhi,
                q2dZ=q2dZ,
                interp_t=interp_t,
                Type=Type,
                interp_space=interp_space,
                fill_value=fill_value,
            )

            # Integrate using ufunc reduceat for speed
            # (cf. https://stackoverflow.com/questions/59079141)
            sig = np.add.reduceat(val, np.r_[0, indpts],
                                  axis=-1)*reseff[None, :]

        # Apply user-provided coefs
        if coefs is not None:
            if hasattr(coefs, '__iter__'):
                coefs = np.atleast_1d(coefs).ravel()
                assert coefs.shape == (sig.shape[-1],)
                if sig.ndim == 2:
                    coefs = coefs[None, :]
            sig *= coefs

        # Format output
        # this is the secod slowest step (~0.75 s)
        out = self._calc_signal_postformat(
            sig,
            Brightness=Brightness,
            dataname=dataname,
            t=t,
            E=E,
            units=units,
            plot=plot,
            out=returnas,
            fs=fs,
            dmargin=dmargin,
            wintit=wintit,
            invert=invert,
            draw=draw,
            connect=connect,
        )
        return out

    def plot(
        self,
        lax=None,
        proj="all",
        reflections=True,
        Lplot=_def.LOSLplot,
        element="L",
        element_config="P",
        Leg="",
        dL=None,
        dPtD=_def.LOSMd,
        dPtI=_def.LOSMd,
        dPtO=_def.LOSMd,
        dPtR=_def.LOSMd,
        dPtP=_def.LOSMd,
        dLeg=_def.TorLegd,
        multi=False,
        ind=None,
        fs=None,
        tit=None,
        wintit=None,
        draw=True,
        Test=True,
    ):
        """ Plot the Rays / LOS, in the chosen projection(s)

        Optionnally also plot associated :class:`~tofu.geom.Ves` and Struct
        The plot can also include:
            - special points
            - the unit directing vector

        Parameters
        ----------
        lax :       list / plt.Axes
            The axes for plotting (list of 2 axes if Proj='All')
            If None a new figure with new axes is created
        proj :      str
            Flag specifying the kind of projection:
                - 'Cross' : cross-section
                - 'Hor' : horizontal
                - 'All' : both cross-section and horizontal (on 2 axes)
                - '3d' : a (matplotlib) 3d plot
        projections:bool
            Flag indicating whether to plot also the reflected rays
            Assuming some reflected rays are present (self.add_reflections())
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

        return _plot.Rays_plot(
            self,
            Lax=lax,
            Proj=proj,
            reflections=reflections,
            Lplot=Lplot,
            element=element,
            element_config=element_config,
            Leg=Leg,
            dL=dL,
            dPtD=dPtD,
            dPtI=dPtI,
            dPtO=dPtO,
            dPtR=dPtR,
            dPtP=dPtP,
            dLeg=dLeg,
            multi=multi,
            ind=ind,
            fs=fs,
            tit=tit,
            wintit=wintit,
            draw=draw,
            Test=Test,
        )

    def plot_sino(
        self,
        ax=None,
        element=_def.LOSImpElt,
        Sketch=True,
        Ang=_def.LOSImpAng,
        AngUnit=_def.LOSImpAngUnit,
        Leg=None,
        dL=_def.LOSMImpd,
        dVes=_def.TorPFilld,
        dLeg=_def.TorLegd,
        ind=None,
        multi=False,
        fs=None,
        tit=None,
        wintit=None,
        draw=True,
        Test=True,
    ):
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
        if self._dsino["RefPt"] is None:
            msg = "The sinogram ref. point is not set !"
            msg += "\n  => run self.set_dsino()"
            raise Exception(msg)
        return _plot.GLOS_plot_Sino(
            self,
            Proj="Cross",
            ax=ax,
            Elt=element,
            Leg=Leg,
            Sketch=Sketch,
            Ang=Ang,
            AngUnit=AngUnit,
            dL=dL,
            dVes=dVes,
            dLeg=dLeg,
            ind=ind,
            fs=fs,
            tit=tit,
            wintit=wintit,
            draw=draw,
            Test=Test,
        )

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
        lS = self.config.lStruct
        ind = self._check_indch(ind, out=bool)
        for ii in np.r_[self.get_indStruct_computeInOut(unique_In=True)]:
            kn = "{}_{}".format(lS[ii].__class__.__name__, lS[ii].Id.Name)
            indtouch = self.select(touch=kn, out=bool)
            if np.any(indtouch):
                indok = indtouch & ind
                indout = indtouch & ~ind
                if np.any(indok) or np.any(indout):
                    if out == int:
                        indok = indok.nonzero()[0]
                        indout = indout.nonzero()[0]
                    dElt[kn] = {
                        "indok": indok,
                        "indout": indout,
                        "col": lS[ii].get_color(),
                    }
        return dElt

    def get_touch_colors(
        self,
        ind=None,
        dElt=None,
        cbck=(0.8, 0.8, 0.8),
        rgba=True,
    ):
        """ Get array of colors per LOS (color set by the touched Struct) """
        if dElt is None:
            dElt = self.get_touch_dict(ind=None, out=bool)
        else:
            assert type(dElt) is dict
            assert all(
                [type(k) is str and type(v) is dict for k, v in dElt.items()]
            )

        if rgba:
            colors = np.tile(mpl.colors.to_rgba(cbck), (self.nRays, 1)).T
            for k, v in dElt.items():
                colors[:, v["indok"]] = np.r_[mpl.colors.to_rgba(v["col"])][
                    :, None
                ]
        else:
            colors = np.tile(mpl.colors.to_rgb(cbck), (self.nRays, 1)).T
            for k, v in dElt.items():
                colors[:, v["indok"]] = np.r_[mpl.colors.to_rgb(v["col"])][
                    :, None
                ]
        return colors

    def plot_touch(
        self,
        key=None,
        quant="lengths",
        Lplot=None,
        invert=None,
        ind=None,
        Bck=True,
        fs=None,
        wintit=None,
        tit=None,
        connect=True,
        draw=True,
    ):
        """ Interactive plot of the camera and the structures it touches

        The camera LOS are plotted in poloidal and horizontal projections
        The associated Config is also plotted
        The plot shows which strutural element is touched by each LOS

        In addition, an extra quantity can be mapped to alpha (transparency)

        Parameters
        ----------
        key:        None / str
            Only relevant if self.dchans was defined
            key is then a key to sekf.dchans
        quant:      None / str
            Flag indicating which extra quantity is used to map alpha:
            - 'lengths' (default): the length of each LOS
            - 'angles' : the angle of incidence of each LOS
                         (with respect to the normal of the surface touched,
                          useful for assessing reflection probabilities)
            - 'indices': the index of each LOS
                         (useful for checking numbering)
            - 'Etendues': the etendue associated to each LOS (user-provided)
            - 'Surfaces': the surfaces associated to each LOS (user-provided)
        Lplot:      None / str
            Flag indicating whether to plot:
                - 'tot': the full length of the LOS
                - 'in': only the part that is inside the vessel
        invert:     None / bool
            Flag indicating whether to plot 2D camera images inverted (pinhole)
        ind:        None / np.ndarray
            Array of bool indices used to select only a subset of the LOS
        Bck:        None / bool
            Flag indicating whether to plot the background LOS
        fs:         None / tuple
            figure size in inches
        wintit:     None / str
            Title for the window
        tit:        None / str
            Title for the figure
        connect:    None / bool
            Flag indicating to connect interactive actuators
        draw:       None / bool
            Flag indicating whether to draw the figure
        """
        out = _plot.Rays_plot_touch(
            self,
            key=key,
            Bck=Bck,
            quant=quant,
            ind=ind,
            Lplot=Lplot,
            invert=invert,
            connect=connect,
            fs=fs,
            wintit=wintit,
            tit=tit,
            draw=draw,
        )
        return out


########################################
#       CamLOS subclasses
########################################

sig = inspect.signature(Rays)
params = sig.parameters


class CamLOS1D(Rays):
    def get_summary(
        self,
        sep="  ",
        line="-",
        just="l",
        table_sep=None,
        verb=True,
        return_=False,
    ):

        # Prepare
        kout = self._dgeom["kOut"]
        indout = self._dgeom["indout"]
        lS = self._dconfig["Config"].lStruct
        angles = np.arccos(-np.sum(self.u*self.dgeom['vperp'], axis=0))

        # ar0
        col0 = ["nb. los", "av. length", "min length", "max length",
                "nb. touch", "av. angle", "min angle", "max angle"]
        ar0 = [
            self.nRays,
            "{:.3f}".format(np.nanmean(kout)),
            "{:.3f}".format(np.nanmin(kout)),
            "{:.3f}".format(np.nanmax(kout)),
            np.unique(indout[0, :]).size,
            "{:.2f}".format(np.nanmean(angles)),
            "{:.2f}".format(np.nanmin(angles)),
            "{:.2f}".format(np.nanmax(angles)),
        ]
        if self._dgeom['move'] is not None:
            col0 += ['move', 'param']
            ar0 += [self._dgeom['move'],
                    str(round(self._dgeom['move_param'], ndigits=4))]

        # ar1
        col1 = ["los index", "length", "touch", "angle (rad)"]
        ar1 = [
            np.arange(0, self.nRays),
            np.around(kout, decimals=3).astype("U"),
            ["%s_%s" % (lS[ii].Id.Cls, lS[ii].Id.Name) for ii in indout[0, :]],
            np.around(angles, decimals=2).astype('U')
        ]

        for k, v in self._dchans.items():
            col1.append(k)
            if v.ndim == 1:
                ar1.append(v)
            else:
                ar1.append([str(vv) for vv in v])

        # call base method
        return self._get_summary(
            [ar0, ar1],
            [col0, col1],
            sep=sep,
            line=line,
            table_sep=table_sep,
            verb=verb,
            return_=return_,
        )

    def __add__(self, other):
        if not other.__class__.__name__ == self.__class__.__name__:
            msg = "Operator defined only for same-class operations !"
            raise Exception(msg)
        lc = [self.Id.Exp == other.Id.Exp, self.Id.Diag == other.Id.Diag]
        if not all(lc):
            msg = (
                "Operation only valid if objects have identical (Diag, Exp) !"
            )
            raise Exception(msg)
        if not self.config == other.config:
            msg = "Operation only valid if objects have identical config !"
            raise Exception(msg)

        Name = "%s+%s" % (self.Id.Name, other.Id.Name)
        D = np.concatenate((self.D, other.D), axis=1)
        u = np.concatenate((self.u, other.u), axis=1)

        return self.__class__(
            dgeom=(D, u),
            config=self.config,
            Name=Name,
            Diag=self.Id.Diag,
            Exp=self.Id.Exp,
        )

    def __radd__(self, other):
        return self.__add__(other)

    def save_to_imas(
        self,
        ids=None,
        shot=None,
        run=None,
        refshot=None,
        refrun=None,
        user=None,
        database=None,
        version=None,
        occ=None,
        dryrun=False,
        deep=True,
        restore_size=True,
        verb=True,
        config_description_2d=None,
        config_occ=None,
    ):
        import tofu.imas2tofu as _tfimas

        _tfimas._save_to_imas(
            self,
            tfversion=__version__,
            shot=shot,
            run=run,
            refshot=refshot,
            refrun=refrun,
            user=user,
            database=database,
            version=version,
            occ=occ,
            dryrun=dryrun,
            verb=verb,
            ids=ids,
            deep=deep,
            restore_size=restore_size,
            config_description_2d=config_description_2d,
            config_occ=config_occ,
        )


lp = [p for p in params.values() if p.name != "dX12"]
CamLOS1D.__signature__ = sig.replace(parameters=lp)


class CamLOS2D(Rays):
    def get_summary(
        self,
        sep="  ",
        line="-",
        just="l",
        table_sep=None,
        verb=True,
        return_=False,
    ):

        # Prepare
        kout = self._dgeom["kOut"]
        indout = self._dgeom["indout"]
        # lS = self._dconfig["Config"].lStruct
        angles = np.arccos(-np.sum(self.u*self.dgeom['vperp'], axis=0))

        # ar0
        col0 = ["nb. los", "av. length", "min length", "max length",
                "nb. touch", "av. angle", "min angle", "max angle"]
        ar0 = [
            self.nRays,
            "{:.3f}".format(np.nanmean(kout)),
            "{:.3f}".format(np.nanmin(kout)),
            "{:.3f}".format(np.nanmax(kout)),
            np.unique(indout[0, :]).size,
            "{:.2f}".format(np.nanmean(angles)),
            "{:.2f}".format(np.nanmin(angles)),
            "{:.2f}".format(np.nanmax(angles)),
        ]
        if self._dgeom['move'] is not None:
            col0 += ['move', 'param']
            ar0 += [self._dgeom['move'],
                    str(round(self._dgeom['move_param'], ndigits=4))]

        # call base method
        return self._get_summary(
            [ar0],
            [col0],
            sep=sep,
            line=line,
            table_sep=table_sep,
            verb=verb,
            return_=return_,
        )

    def _isImage(self):
        return self._dgeom["isImage"]

    @property
    def dX12(self):
        if self._dX12 is not None and self._dX12["from"] == "geom":
            dX12 = self._dgeom["dX12"]
        else:
            dX12 = self._dX12
        return dX12

    def get_X12plot(self, plot="imshow"):
        if plot == "imshow":
            x1, x2 = self.dX12["x1"], self.dX12["x2"]
            x1min, Dx1min = x1[0], 0.5 * (x1[1] - x1[0])
            x1max, Dx1max = x1[-1], 0.5 * (x1[-1] - x1[-2])
            x2min, Dx2min = x2[0], 0.5 * (x2[1] - x2[0])
            x2max, Dx2max = x2[-1], 0.5 * (x2[-1] - x2[-2])
            extent = (
                x1min - Dx1min,
                x1max + Dx1max,
                x2min - Dx2min,
                x2max + Dx2max,
            )
            indr = self.dX12["indr"]
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
                             self._dgeom['ddetails']['x2'].size),dtype=np.int64)
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
