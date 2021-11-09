# -*- coding: utf-8 -*-


# Built-in
import sys
import os
# import itertools as itt
import copy
import warnings
from abc import ABCMeta, abstractmethod
import inspect


# Common
import numpy as np
import scipy.interpolate as scpinterp
# import matplotlib.pyplot as plt
# from matplotlib.tri import Triangulation as mplTri


# tofu
# from tofu import __version__ as __version__
import tofu.utils as utils
from . import _DataCollection_check_inputs
from . import _comp
from . import _DataCollection_comp
from . import _DataCollection_plot
from . import _def
from . import _comp_spectrallines


__all__ = ['DataCollection']    # , 'TimeTraceCollection']


_INTERPT = 'zero'
_GROUP_0D = 'time'
_GROUP_1D = 'radius'
_GROUP_2D = 'mesh2d'


#############################################
#############################################
#       Abstract Parent class
#############################################
#############################################


class DataCollection(utils.ToFuObject):
    """ A generic class for handling data

    Provides methods for:
        - introspection
        - plateaux finding
        - visualization

    """
    __metaclass__ = ABCMeta

    # Fixed (class-wise) dictionary of default properties
    _ddef = {
        'Id': {'include': ['Mod', 'Cls', 'Name', 'version']},
        'params': {
            'ddata': {
                'source': (str, 'unknown'),
                'dim':    (str, 'unknown'),
                'quant':  (str, 'unknown'),
                'name':   (str, 'unknown'),
                'units':  (str, 'a.u.'),
            },
            'dobj': {},
         },
    }

    _forced_group = None
    if _forced_group is not None:
        _allowed_groups = [_forced_group]
    else:
        _allowed_groups = None
    # _dallowed_params = None
    _data_none = None
    _reserved_keys = None

    _show_in_summary_core = ['shape', 'ref', 'group']
    _show_in_summary = 'all'
    _max_ndim = None

    _dgroup = {}
    _dref = {}
    _dref_static = {}
    _ddata = {}
    _dobj = {}

    _group0d = _GROUP_0D
    _group1d = _GROUP_1D
    _group2d = _GROUP_2D

    def __init_subclass__(cls, **kwdargs):
        # Does not exist before Python 3.6 !!!
        # Python 2
        super(DataCollection, cls).__init_subclass__(**kwdargs)
        # Python 3
        # super().__init_subclass__(**kwdargs)
        cls._ddef = copy.deepcopy(DataCollection._ddef)
        # cls._dplot = copy.deepcopy(Struct._dplot)
        # cls._set_color_ddef(cls._color)

    def __init__(
        self,
        dgroup=None,
        dref=None,
        dref_static=None,
        ddata=None,
        dobj=None,
        Id=None,
        Name=None,
        fromdict=None,
        SavePath=None,
        include=None,
        sep=None,
    ):

        # Create a dplot at instance level
        # self._dplot = copy.deepcopy(self.__class__._dplot)
        kwdargs = locals()
        del kwdargs['self']
        super().__init__(**kwdargs)

    def _reset(self):
        # Run by the parent class __init__()
        super()._reset()
        self._dgroup = {}
        self._dref = {}
        self._dref_static = {}
        self._ddata = {}
        self._dobj = {}

    @classmethod
    def _checkformat_inputs_Id(cls, Id=None, Name=None,
                               include=None, **kwdargs):
        if Id is not None:
            assert isinstance(Id, utils.ID)
            Name = Id.Name
        # assert isinstance(Name, str), Name
        if include is None:
            include = cls._ddef['Id']['include']
        kwdargs.update({'Name': Name, 'include': include})
        return kwdargs

    ###########
    # Get check and format inputs
    ###########

    ###########
    # _init
    ###########

    def _init(
        self,
        dgroup=None,
        dref=None,
        dref_static=None,
        ddata=None,
        dobj=None,
        **kwargs,
    ):
        self.update(
            dgroup=dgroup,
            dref=dref,
            dref_static=dref_static,
            ddata=ddata,
            dobj=dobj,
        )
        self._dstrip['strip'] = 0

    ###########
    # set dictionaries
    ###########

    def update(
        self,
        dobj=None,
        ddata=None,
        dref=None,
        dref_static=None,
        dgroup=None,
    ):
        """ Can be used to set/add data/ref/group

        Will update existing attribute with new dict
        """
        # Check consistency
        self._dgroup, self._dref, self._dref_static, self._ddata, self._dobj =\
                _DataCollection_check_inputs._consistency(
                    dobj=dobj, dobj0=self._dobj,
                    ddata=ddata, ddata0=self._ddata,
                    dref=dref, dref0=self._dref,
                    dref_static=dref_static, dref_static0=self._dref_static,
                    dgroup=dgroup, dgroup0=self._dgroup,
                    allowed_groups=self._allowed_groups,
                    reserved_keys=self._reserved_keys,
                    ddefparams_data=self._ddef['params']['ddata'],
                    ddefparams_obj=self._ddef['params']['dobj'],
                    data_none=self._data_none,
                    max_ndim=self._max_ndim,
                )

    # ---------------------
    # Adding group / ref / quantity one by one
    # ---------------------

    def add_group(self, group=None):
        # Check consistency
        self.update(ddata=None, dref=None, dref_static=None, dgroup=group)

    def add_ref(self, key=None, group=None, data=None, **kwdargs):
        dref = {key: {'group': group, 'data': data, **kwdargs}}
        # Check consistency
        self.update(ddata=None, dref=dref, dref_static=None, dgroup=None)

    # TBF
    def add_ref_static(self, key=None, which=None, **kwdargs):
        dref_static = {which: {key: kwdargs}}
        # Check consistency
        self.update(
            ddata=None, dref=None, dref_static=dref_static, dgroup=None,
        )

    def add_data(self, key=None, data=None, ref=None, **kwdargs):
        ddata = {key: {'data': data, 'ref': ref, **kwdargs}}
        # Check consistency
        self.update(ddata=ddata, dref=None, dref_static=None, dgroup=None)

    def add_obj(self, which=None, key=None, **kwdargs):
        dobj = {which: {key: kwdargs}}
        # Check consistency
        self.update(dobj=dobj, dref=None, dref_static=None, dgroup=None)

    # ---------------------
    # Removing group / ref / quantities
    # ---------------------

    def remove_group(self, group=None):
        """ Remove a group (or list of groups) and all associated ref, data """
        self._dgroup, self._dref, self._dref_static, self._ddata, self._dobj =\
                _DataCollection_check_inputs._remove_group(
                    group=group,
                    dgroup0=self._dgroup, dref0=self._dref, ddata0=self._ddata,
                    dref_static0=self._dref_static,
                    dobj0=self._dobj,
                    allowed_groups=self._allowed_groups,
                    reserved_keys=self._reserved_keys,
                    ddefparams_data=self._ddef['params']['ddata'],
                    ddefparams_obj=self._ddef['params']['dobj'],
                    data_none=self._data_none,
                    max_ndim=self._max_ndim,
                )

    def remove_ref(self, key=None, propagate=None):
        """ Remove a ref (or list of refs) and all associated data """
        self._dgroup, self._dref, self._dref_static, self._ddata, self._dobj =\
                _DataCollection_check_inputs._remove_ref(
                    key=key,
                    dgroup0=self._dgroup, dref0=self._dref, ddata0=self._ddata,
                    dref_static0=self._dref_static,
                    dobj0=self._dobj,
                    propagate=propagate,
                    allowed_groups=self._allowed_groups,
                    reserved_keys=self._reserved_keys,
                    ddefparams_data=self._ddef['params']['ddata'],
                    ddefparams_obj=self._ddef['params']['dobj'],
                    data_none=self._data_none,
                    max_ndim=self._max_ndim,
                )

    def remove_ref_static(self, key=None, which=None, propagate=None):
        """ Remove a static ref (or list) or a whole category

        key os provided:
            => remove only the desired key(s)
                works only if key is not used in ddata and dobj

        which is provided:
            => treated as param, the whole category of ref_static is removed
                if propagate, the parameter is removed from ddata and dobj
        """
        _DataCollection_check_inputs._remove_ref_static(
            key=key,
            which=which,
            propagate=propagate,
            dref_static0=self._dref_static,
            ddata0=self._ddata,
            dobj0=self._dobj,
        )

    def remove_data(self, key=None, propagate=True):
        """ Remove a data (or list of data) """
        self._dgroup, self._dref, self._dref_static, self._ddata, self._dobj =\
                _DataCollection_check_inputs._remove_data(
                    key=key,
                    dgroup0=self._dgroup, dref0=self._dref, ddata0=self._ddata,
                    dref_static0=self._dref_static,
                    dobj0=self._dobj,
                    propagate=propagate,
                    allowed_groups=self._allowed_groups,
                    reserved_keys=self._reserved_keys,
                    ddefparams_data=self._ddef['params']['ddata'],
                    ddefparams_obj=self._ddef['params']['dobj'],
                    data_none=self._data_none,
                    max_ndim=self._max_ndim,
                )

    def remove_obj(self, key=None, which=None, propagate=True):
        """ Remove a data (or list of data) """
        self._dgroup, self._dref, self._dref_static, self._ddata, self._dobj =\
                _DataCollection_check_inputs._remove_obj(
                    key=key,
                    which=which,
                    dobj0=self._dobj,
                    ddata0=self._ddata,
                    dgroup0=self._dgroup,
                    dref0=self._dref,
                    dref_static0=self._dref_static,
                    allowed_groups=self._allowed_groups,
                    reserved_keys=self._reserved_keys,
                    ddefparams_data=self._ddef['params']['ddata'],
                    ddefparams_obj=self._ddef['params']['dobj'],
                    data_none=self._data_none,
                    max_ndim=self._max_ndim,
                )

    # ---------------------
    # Get / set / add / remove params
    # ---------------------

    def __check_which(self, which=None, return_dict=None):
        """ Check which in ['data'] + list(self._dobj.keys() """
        return _DataCollection_check_inputs._check_which(
            ddata=self._ddata,
            dobj=self._dobj,
            which=which,
            return_dict=return_dict,
        )

    def get_lparam(self, which=None):
        """ Return the list of params for the chosen dict ('data' or dobj[<>])
        """
        which, dd = self.__check_which(which, return_dict=True)
        if which is None:
            return

        lp = list(list(dd.values())[0].keys())
        if which == 'data':
            lp.remove('data')
        return lp

    def get_param(
        self,
        param=None,
        key=None,
        ind=None,
        returnas=None,
        which=None,
    ):
        """ Return the array of the chosen parameter (or list of parameters)

        Can be returned as:
            - dict: {param0: {key0: values0, key1: value1...}, ...}
            - np[.ndarray: {param0: np.r_[values0, value1...], ...}

        """
        which, dd = self.__check_which(which, return_dict=True)
        if which is None:
            return
        return _DataCollection_check_inputs._get_param(
            dd=dd, dd_name=which,
            param=param, key=key, ind=ind, returnas=returnas,
        )

    def set_param(
        self,
        param=None,
        value=None,
        ind=None,
        key=None,
        which=None,
    ):
        """ Set the value of a parameter

        value can be:
            - None
            - a unique value (int, float, bool, str, tuple) common to all keys
            - an iterable of vlues (array, list) => one for each key

        A subset of keys can be chosen (ind, key, fed to self.select()) to set
        only the value of some key

        """
        which, dd = self.__check_which(which, return_dict=True)
        if which is None:
            return
        _DataCollection_check_inputs._set_param(
            dd=dd, dd_name=which,
            param=param, value=value, ind=ind, key=key,
        )

    def add_param(
        self,
        param,
        value=None,
        which=None,
    ):
        """ Add a parameter, optionnally also set its value """
        which, dd = self.__check_which(which, return_dict=True)
        if which is None:
            return
        _DataCollection_check_inputs._add_param(
            dd=dd, dd_name=which,
            param=param, value=value,
        )

    def remove_param(
        self,
        param=None,
        which=None,
    ):
        """ Remove a parameter, none by default, all if param = 'all' """
        which, dd = self.__check_which(which, return_dict=True)
        if which is None:
            return
        _DataCollection_check_inputs._remove_param(
            dd=dd, dd_name=which,
            param=param,
        )

    ###########
    # strip dictionaries
    ###########

    def _strip_ddata(self, strip=0, verb=0):
        pass

    ###########
    # _strip and get/from dict
    ###########

    @classmethod
    def _strip_init(cls):
        cls._dstrip['allowed'] = [0, 1]
        nMax = max(cls._dstrip['allowed'])
        doc = """
                 1: None
                 """
        doc = utils.ToFuObjectBase.strip.__doc__.format(doc, nMax)
        cls.strip.__doc__ = doc

    def strip(self, strip=0, verb=True):
        # super()
        super(DataCollection, self).strip(strip=strip, verb=verb)

    def _strip(self, strip=0, verb=True):
        self._strip_ddata(strip=strip, verb=verb)

    def _to_dict(self):
        dout = {
            'dgroup': {'dict': self._dgroup, 'lexcept': None},
            'dref': {'dict': self._dref, 'lexcept': None},
            'dref_static': {'dict': self._dref_static, 'lexcept': None},
            'ddata': {'dict': self._ddata, 'lexcept': None},
            'dobj': {'dict': self._dobj, 'lexcept': None},
        }
        return dout

    def _from_dict(self, fd):
        for k0 in ['dgroup', 'dref', 'ddata', 'dref_static', 'dobj']:
            if fd.get(k0) is not None:
                getattr(self, '_'+k0).update(**fd[k0])
        self.update()

    ###########
    # properties
    ###########

    @property
    def dgroup(self):
        """ The dict of groups """
        return self._dgroup

    @property
    def dref(self):
        """ the dict of references """
        return self._dref

    @property
    def dref_static(self):
        """ the dict of references """
        return self._dref_static

    @property
    def ddata(self):
        """ the dict of data """
        return self._ddata

    @property
    def dobj(self):
        """ the dict of obj """
        return self._dobj

    ###########
    # General use methods
    ###########

    def to_DataFrame(self, which=None):
        which, dd = self.__check_which(which, return_dict=True)
        if which is None:
            return
        import pandas as pd
        return pd.DataFrame(dd)

    # ---------------------
    # Key selection methods
    # ---------------------

    def select(self, which=None, log=None, returnas=None, **kwdargs):
        """ Return the indices / keys of data matching criteria

        The selection is done comparing the value of all provided parameters
        The result is a boolean indices array, optionally with the keys list
        It can include:
            - log = 'all': only the data matching all criteria
            - log = 'any': the data matching any criterion

        If log = 'raw', a dict of indices arrays is returned, showing the
        details for each criterion

        """
        which, dd = self.__check_which(which, return_dict=True)
        if which is None:
            return
        return _DataCollection_check_inputs._select(
            dd=dd, dd_name=which,
            log=log, returnas=returnas, **kwdargs,
        )

    def _ind_tofrom_key(
        self,
        ind=None,
        key=None,
        group=None,
        returnas=int,
        which=None,
    ):
        """ Return ind from key or key from ind for all data """
        which, dd = self.__check_which(which, return_dict=True)
        if which is None:
            return
        return _DataCollection_check_inputs._ind_tofrom_key(
            dd=dd, dd_name=which, ind=ind, key=key,
            group=group, dgroup=self._dgroup,
            returnas=returnas,
        )

    def _get_sort_index(self, which=None, param=None):
        """ Return sorting index ofself.ddata dict """

        if param is None:
            return

        if param == 'key':
            ind = np.argsort(list(dd.keys()))
        elif isinstance(param, str):
            ind = np.argsort(
                self.get_param(param, which=which, returnas=np.ndarray)[param]
            )
        else:
            msg = "Arg param must be a valid str\n  Provided: {}".format(param)
            raise Exception(msg)
        return ind

    def sortby(self, param=None, order=None, which=None):
        """ sort the self.ddata dict by desired parameter """

        # Trivial case
        if len(self._ddata) == 0 and len(self._dobj) == 0:
            return

        # --------------
        # Check inputs

        # order
        if order is None:
            order = 'increasing'

        c0 = order in ['increasing', 'reverse']
        if not c0:
            msg = (
                """
                Arg order must be in [None, 'increasing', 'reverse']
                Provided: {}
                """.format(order)
            )
            raise Exception(msg)

        # which
        which, dd = self.__check_which(which, return_dict=True)
        if which is None:
            return

        # --------------
        # sort
        ind = self._get_sort_index(param=param, which=which)
        if ind is None:
            return
        if order == 'reverse':
            ind = ind[::-1]

        lk = list(dd.keys())
        dd = {lk[ii]: dd[lk[ii]] for ii in ind}

        if which == 'data':
            self._ddata = dd
        else:
            self._dobj[which] = dd

    # ---------------------
    # Get refs from data key
    # ---------------------

    def _get_ref_from_key(self, key=None, group=None):
        """ Get the key of the ref in chosen group """

        # Check input
        if key not in self._ddata.keys():
            msg = "Provide a valid data key!\n\t- Provided: {}".format(key)
            raise Exception(msg)

        ref = self._ddata[key]['ref']
        if len(ref) > 1:
            if group not in self._dgroup.keys():
                msg = "Provided group is not valid!\n\t{}".format(group)
                raise Exception(msg)
            ref = [rr for rr in ref if self._dref[rr]['group'] == group]
            if len(ref) != 1:
                msg = "Ambiguous ref for key {}!\n\t- {}".format(key, ref)
                raise Exception(msg)
        return ref[0]

    # ---------------------
    # Switch ref
    # ---------------------

    def switch_ref(self, new_ref=None):
        """Use the provided key as ref (if valid) """
        self._dgroup, self._dref, self._dref_static, self._ddata, self._dobj =\
                _DataCollection_check_inputs.switch_ref(
                    new_ref=new_ref,
                    ddata=self._ddata,
                    dref=self._dref,
                    dgroup=self._dgroup,
                    dobj0=self._dobj,
                    dref_static0=self._dref_static,
                    allowed_groups=self._allowed_groups,
                    reserved_keys=self._reserved_keys,
                    ddefparams_data=self._ddef['params'].get('data'),
                    data_none=self._data_none,
                    max_ndim=self._max_ndim,
                )

    # ---------------------
    # Methods for getting a subset of the collection
    # ---------------------

    # TBC
    def get_drefddata_as_input(self, key=None, ind=None, group=None):
        lk = self._ind_tofrom_key(ind=ind, key=key, group=group, returnas=str)
        lkr = [kr for kr in self._dref['lkey']
               if any([kr in self._ddata['dict'][kk]['refs'] for kk in lk])]
        dref = {kr: {'data': self._ddata['dict'][kr]['data'],
                     'group': self._dref['dict'][kr]['group']} for kr in lkr}
        lkr = dref.keys()
        ddata = {kk: self._ddata['dict'][kk] for kk in lk if kk not in lkr}
        return dref, ddata

    # TBC
    def get_subset(self, key=None, ind=None, group=None, Name=None):
        if key is None and ind is None:
            return self
        else:
            dref, ddata = self.get_drefddata_as_input(key=key, ind=ind,
                                                      group=group)
            if Name is None and self.Id.Name is not None:
                Name = self.Id.Name + '-subset'
            return self.__class__(dref=dref, ddata=ddata, Name=Name)

    # ---------------------
    # Methods for exporting plot collection (subset)
    # ---------------------

    # TBC
    def to_PlotCollection(self, key=None, ind=None, group=None, Name=None,
                          dnmax=None, lib='mpl'):
        dref, ddata = self.get_drefddata_as_input(
            key=key, ind=ind, group=group,
        )
        if Name is None and self.Id.Name is not None:
            Name = self.Id.Name + '-plot'
        import tofu.data._core_plot as _core_plot
        if lib == 'mpl':
            cls = _core_plot.DataCollectionPlot_mpl
        else:
            raise NotImplementedError
        obj = cls(dref=dref, ddata=ddata, Name=Name)
        if dnmax is not None:
            obj.set_dnmax(dnmax)
        return obj

    # ---------------------
    # Methods for showing data
    # ---------------------

    def get_summary(
        self,
        show=None,
        show_core=None,
        sep='  ',
        line='-',
        just='l',
        table_sep=None,
        verb=True,
        return_=False,
    ):
        """ Summary description of the object content """
        # # Make sure the data is accessible
        # msg = "The data is not accessible because self.strip(2) was used !"
        # assert self._dstrip['strip']<2, msg

        lcol, lar = [], []

        # -----------------------
        # Build for groups
        if len(self._dgroup) > 0:
            lcol.append(['group', 'nb. ref', 'nb. data'])
            lar.append([
                (
                    k0,
                    len(self._dgroup[k0]['lref']),
                    len(self._dgroup[k0]['ldata']),
                )
                for k0 in self._dgroup.keys()
            ])

        # -----------------------
        # Build for refs
        if len(self._dref) > 0:
            lcol.append(['ref key', 'group', 'size', 'nb. data'])
            lar.append([
                (
                    k0,
                    self._dref[k0]['group'],
                    str(self._dref[k0]['size']),
                    len(self._dref[k0]['ldata'])
                )
                for k0 in self._dref.keys()
            ])

        # -----------------------
        # Build for ddata
        if len(self._ddata) > 0:
            if show_core is None:
                show_core = self._show_in_summary_core
            if isinstance(show_core, str):
                show_core = [show_core]
            lp = self.get_lparam(which='data')
            lkcore = ['shape', 'group', 'ref']
            assert all([ss in lp + lkcore for ss in show_core])
            col2 = ['data key'] + show_core

            if show is None:
                show = self._show_in_summary
            if show == 'all':
                col2 += [pp for pp in lp if pp not in col2]
            else:
                if isinstance(show, str):
                    show = [show]
                assert all([ss in lp for ss in show])
                col2 += [pp for pp in show if pp not in col2]

            ar2 = []
            for k0 in self._ddata.keys():
                lu = [k0] + [str(self._ddata[k0].get(cc)) for cc in col2[1:]]
                ar2.append(lu)

            lcol.append(col2)
            lar.append(ar2)

        # -----------------------
        # Build for dref_static
        if len(self._dref_static) > 0:
            for k0, v0 in self._dref_static.items():
                lk = list(list(v0.values())[0].keys())
                col = [k0] + [pp for pp in lk]
                ar = [
                    tuple([k1] + [str(v1[kk]) for kk in lk])
                    for k1, v1 in v0.items()
                ]
                lcol.append(col)
                lar.append(ar)

        # -----------------------
        # Build for dobj
        if len(self._dobj) > 0:
            for k0, v0 in self._dobj.items():
                lk = self.get_lparam(which=k0)
                lk = [
                    kk for kk in lk
                    if 'func' not in kk
                    and 'class' not in kk
                ]
                lcol.append([k0] + [pp for pp in lk])
                lar.append([
                    tuple([k1] + [str(v1[kk]) for kk in lk])
                    for k1, v1 in v0.items()
                ])

        return self._get_summary(
            lar,
            lcol,
            sep=sep,
            line=line,
            table_sep=table_sep,
            verb=verb,
            return_=return_,
        )

    # -----------------
    # conversion wavelength - energy - frequency
    # ------------------

    @staticmethod
    def convert_spectral(
        data=None,
        units_in=None, units_out=None,
        returnas=None,
    ):
        """ convert wavelength / energy/ frequency

        Available units:
            wavelength: m, mm, um, nm, A
            energy:     J, eV, keV
            frequency:  Hz, kHz, MHz, GHz

        Can also just return the conversion coef if returnas='coef'
        """
        return _comp_spectrallines.convert_spectral(
            data_in=data, units_in=units_in, units_out=units_out,
            returnas=returnas,
        )

    # -----------------
    # Get common ref
    # ------------------

    def _get_common_ref_data_nearest(
        self,
        group=None,
        lkey=None,
        return_all=None,
    ):
        """ Typically used to get a common (intersection) time vector

        Returns a time vector that contains all time points from all data
        Also return a dict of indices to easily remap each time vector to tall
            such that t[ind] = tall (with nearest approximation)

        """
        return _DataCollection_comp._get_unique_ref_dind(
            dd=self._ddata, group=group,
            lkey=lkey, return_all=return_all,
        )

    def _get_pts_from_mesh(self, key=None):
        """ Get default pts from a mesh """

        # Check key is relevant
        c0 = (
            key in self._ddata.keys()
            and isinstance(self._ddata[key].get('data'), dict)
            and 'type' in self._ddata[key]['data'].keys()
        )
        if not c0:
            msg = (
                "ddata['{}'] does not exist or is not a mesh".format(key)
            )
            raise Exception(msg)

        if self.ddata[key]['data']['type'] == 'rect':
            if self.ddata[key]['data']['shapeRZ'] == ('R', 'Z'):
                R = np.repeat(self.ddata[key]['data']['R'],
                              self.ddata[key]['data']['nZ'])
                Z = np.tile(self.ddata[key]['data']['Z'],
                            self.ddata[key]['data']['nR'])
            else:
                R = np.tile(self.ddata[key]['data']['R'],
                            self.ddata[key]['data']['nZ'])
                Z = np.repeat(self.ddata[key]['data']['Z'],
                              self.ddata[key]['data']['nR'])
            pts = np.array([R, np.zeros((R.size,)), Z])
        else:
            pts = self.ddata[key]['data']['nodes']
            pts = np.array([
                pts[:, 0], np.zeros((pts.shape[0],)), pts[:, 1],
            ])
        return pts

    # ---------------------
    # Method for interpolation - inputs checks
    # ---------------------

    # Useful?
    @property
    def _get_lquant_both(self, group1d=None, group2d=None):
        """ Return list of quantities available both in 1d and 2d """
        lq1 = [
            self._ddata[vd]['quant'] for vd in self._dgroup[group1d]['ldata']
        ]
        lq2 = [
            self._ddata[vd]['quant'] for vd in self._dgroup[group2d]['ldata']
        ]
        lq = list(set(lq1).intersection(lq2))
        return lq

    def _check_qr12RPZ(
        self,
        quant=None,
        ref1d=None,
        ref2d=None,
        q2dR=None,
        q2dPhi=None,
        q2dZ=None,
        group1d=None,
        group2d=None,
    ):

        if group1d is None:
            group1d = self._group1d
        if group2d is None:
            group2d = self._group2d

        lc0 = [quant is None, ref1d is None, ref2d is None]
        lc1 = [q2dR is None, q2dPhi is None, q2dZ is None]
        if np.sum([all(lc0), all(lc1)]) != 1:
            msg = (
                "Please provide either (xor):\n"
                + "\t- a scalar field (isotropic emissivity):\n"
                + "\t\tquant : scalar quantity to interpolate\n"
                + "\t\t\tif quant is 1d, intermediate reference\n"
                + "\t\t\tfields are necessary for 2d interpolation\n"
                + "\t\tref1d : 1d reference field on which to interpolate\n"
                + "\t\tref2d : 2d reference field on which to interpolate\n"
                + "\t- a vector (R,Phi,Z) field (anisotropic emissivity):\n"
                + "\t\tq2dR :  R component of the vector field\n"
                + "\t\tq2dPhi: R component of the vector field\n"
                + "\t\tq2dZ :  Z component of the vector field\n"
                + "\t\t=> all components have the same time and mesh!\n"
            )
            raise Exception(msg)

        # Check requested quant is available in 2d or 1d
        if all(lc1):
            (
                idquant, idref1d, idref2d,
            ) = _DataCollection_check_inputs._get_possible_ref12d(
                dd=self._ddata,
                key=quant, ref1d=ref1d, ref2d=ref2d,
                group1d=group1d,
                group2d=group2d,
            )
            idq2dR, idq2dPhi, idq2dZ = None, None, None
            ani = False
        else:
            idq2dR, msg = _DataCollection_check_inputs._get_keyingroup_ddata(
                dd=self._ddata,
                key=q2dR, group=group2d, msgstr='quant', raise_=True,
            )
            idq2dPhi, msg = _DataCollection_check_inputs._get_keyingroup_ddata(
                dd=self._ddata,
                key=q2dPhi, group=group2d, msgstr='quant', raise_=True,
            )
            idq2dZ, msg = _DataCollection_check_inputs._get_keyingroup_ddata(
                dd=self._ddata,
                key=q2dZ, group=group2d, msgstr='quant', raise_=True,
            )
            idquant, idref1d, idref2d = None, None, None
            ani = True
        return idquant, idref1d, idref2d, idq2dR, idq2dPhi, idq2dZ, ani

    # ---------------------
    # Method for interpolation
    # ---------------------

    def _get_finterp(
        self,
        idquant=None, idref1d=None, idref2d=None, idmesh=None,
        idq2dR=None, idq2dPhi=None, idq2dZ=None,
        interp_t=None, interp_space=None,
        fill_value=None, ani=None, Type=None,
        group0d=None, group2d=None,
    ):

        if interp_t is None:
            interp_t = 'nearest'
        if interp_t != 'nearest':
            msg = "'nearest' is the only time-interpolation method available"
            raise NotImplementedError(msg)
        if group0d is None:
            group0d = self._group0d
        if group2d is None:
            group2d = self._group2d

        # Get idmesh
        if idmesh is None:
            if idquant is not None:
                # isotropic
                if idref1d is None:
                    lidmesh = [qq for qq in self._ddata[idquant]['ref']
                               if self._dref[qq]['group'] == group2d]
                else:
                    lidmesh = [qq for qq in self._ddata[idref2d]['ref']
                               if self._dref[qq]['group'] == group2d]
            else:
                # anisotropic
                assert idq2dR is not None
                lidmesh = [qq for qq in self._ddata[idq2dR]['ref']
                           if self._dref[qq]['group'] == group2d]
            assert len(lidmesh) == 1
            idmesh = lidmesh[0]

        # Get common time indices
        if interp_t == 'nearest':
            tall, tbinall, ntall, dind = _DataCollection_comp._get_tcom(
                idquant, idref1d, idref2d, idq2dR,
                dd=self._ddata, group=group0d,
            )

        # Get mesh
        if self._ddata[idmesh]['data']['type'] == 'rect':
            mpltri = None
            trifind = self._ddata[idmesh]['data']['trifind']
        else:
            mpltri = self._ddata[idmesh]['data']['mpltri']
            trifind = mpltri.get_trifinder()

        # # Prepare output

        # Interpolate
        # Note : Maybe consider using scipy.LinearNDInterpolator ?
        if idquant is not None:
            vquant = self._ddata[idquant]['data']
            c0 = (
                self._ddata[idmesh]['data']['type'] == 'quadtri'
                and self._ddata[idmesh]['data']['ntri'] > 1
            )
            if c0:
                vquant = np.repeat(
                    vquant,
                    self._ddata[idmesh]['data']['ntri'],
                    axis=0,
                )
            vr1 = self._ddata[idref1d]['data'] if idref1d is not None else None
            vr2 = self._ddata[idref2d]['data'] if idref2d is not None else None

            # add time dimension if none
            if vquant.ndim == 1:
                vquant = vquant[None, :]
            if vr1.ndim == 1:
                vr1 = vr1[None, :]
            if vr2.ndim == 1:
                vr2 = vr2[None, :]

        else:
            vq2dR = self._ddata[idq2dR]['data']
            vq2dPhi = self._ddata[idq2dPhi]['data']
            vq2dZ = self._ddata[idq2dZ]['data']

            # add time dimension if none
            if vq2dR.ndim == 1:
                vq2dR = vq2dR[None, :]
            if vq2dPhi.ndim == 1:
                vq2dPhi = vq2dPhi[None, :]
            if vq2dZ.ndim == 1:
                vq2dZ = vq2dZ[None, :]

        if interp_space is None:
            interp_space = self._ddata[idmesh]['data']['ftype']

        # get interpolation function
        if ani:
            # Assuming same mesh and time vector for all 3 components
            func = _comp.get_finterp_ani(
                idq2dR, idq2dPhi, idq2dZ,
                interp_t=interp_t,
                interp_space=interp_space,
                fill_value=fill_value,
                idmesh=idmesh, vq2dR=vq2dR,
                vq2dZ=vq2dZ, vq2dPhi=vq2dPhi,
                tall=tall, tbinall=tbinall, ntall=ntall,
                indtq=dind.get(idquant),
                trifind=trifind, Type=Type, mpltri=mpltri,
            )
        else:
            func = _comp.get_finterp_isotropic(
                idquant, idref1d, idref2d,
                vquant=vquant, vr1=vr1, vr2=vr2,
                interp_t=interp_t,
                interp_space=interp_space,
                fill_value=fill_value,
                idmesh=idmesh,
                tall=tall, tbinall=tbinall, ntall=ntall,
                mpltri=mpltri, trifind=trifind,
                indtq=dind.get(idquant),
                indtr1=dind.get(idref1d), indtr2=dind.get(idref2d),
            )

        return func

    def _interp_pts2d_to_quant1d(
        self,
        pts=None,
        vect=None,
        t=None,
        quant=None,
        ref1d=None,
        ref2d=None,
        q2dR=None,
        q2dPhi=None,
        q2dZ=None,
        interp_t=None,
        interp_space=None,
        fill_value=None,
        Type=None,
        group0d=None,
        group1d=None,
        group2d=None,
        return_all=None,
    ):
        """ Return the value of the desired 1d quantity at 2d points

        For the desired inputs points (pts):
            - pts are in (X, Y, Z) coordinates
            - space interpolation is linear on the 1d profiles
        At the desired input times (t):
            - using a nearest-neighbourg approach for time

        """
        # Check inputs
        if group0d is None:
            group0d = self._group0d
        if group1d is None:
            group1d = self._group1d
        if group2d is None:
            group2d = self._group2d
        # msg = "Only 'nearest' available so far for interp_t!"
        # assert interp_t == 'nearest', msg

        # Check requested quant is available in 2d or 1d
        idquant, idref1d, idref2d, idq2dR, idq2dPhi, idq2dZ, ani = \
                self._check_qr12RPZ(
                    quant=quant, ref1d=ref1d, ref2d=ref2d,
                    q2dR=q2dR, q2dPhi=q2dPhi, q2dZ=q2dZ,
                    group1d=group1d, group2d=group2d,
                )

        # Check the pts is (3,...) array of floats
        idmesh = None
        if pts is None:
            # Identify mesh to get default points
            if ani:
                idmesh = [id_ for id_ in self._ddata[idq2dR]['ref']
                          if self._dref[id_]['group'] == group2d][0]
            else:
                if idref1d is None:
                    idmesh = [id_ for id_ in self._ddata[idquant]['ref']
                              if self._dref[id_]['group'] == group2d][0]
                else:
                    idmesh = [id_ for id_ in self._ddata[idref2d]['ref']
                              if self._dref[id_]['group'] == group2d][0]

            # Derive pts
            pts = self._get_pts_from_mesh(key=idmesh)

        pts = np.atleast_2d(pts)
        if pts.shape[0] != 3:
            msg = (
                "pts must be np.ndarray of (X,Y,Z) points coordinates\n"
                + "Can be multi-dimensional, but 1st dimension is (X,Y,Z)\n"
                + "    - Expected shape : (3,...)\n"
                + "    - Provided shape : {}".format(pts.shape)
            )
            raise Exception(msg)

        # Check t
        lc = [t is None, type(t) is str, type(t) is np.ndarray]
        assert any(lc)
        if lc[1]:
            assert t in self._ddata.keys()
            t = self._ddata[t]['data']

        # Interpolation (including time broadcasting)
        # this is the second slowest step (~0.08 s)
        func = self._get_finterp(
            idquant=idquant, idref1d=idref1d, idref2d=idref2d,
            idq2dR=idq2dR, idq2dPhi=idq2dPhi, idq2dZ=idq2dZ,
            idmesh=idmesh,
            interp_t=interp_t, interp_space=interp_space,
            fill_value=fill_value, ani=ani, Type=Type,
            group0d=group0d, group2d=group2d,
        )

        # Check vect of ani
        c0 = (
            ani is True
            and (
                vect is None
                or not (
                    isinstance(vect, np.ndarray)
                    and vect.shape == pts.shape
                )
            )
        )
        if c0:
            msg = (
                "Anisotropic field interpolation needs a field of local vect\n"
                + "  => Please provide vect as (3, npts) np.ndarray!"
            )
            raise Exception(msg)

        # This is the slowest step (~1.8 s)
        val, t = func(pts, vect=vect, t=t)

        # return
        if return_all is None:
            return_all = True
        if return_all is True:
            dout = {
                't': t,
                'pts': pts,
                'ref1d': idref1d,
                'ref2d': idref2d,
                'q2dR': idq2dR,
                'q2dPhi': idq2dPhi,
                'q2dZ': idq2dZ,
                'interp_t': interp_t,
                'interp_space': interp_space,
            }
            return val, dout
        else:
            return val

    # TBC
    def _interp_one_dim(x=None, ind=None, key=None, group=None,
                        kind=None, bounds_error=None, fill_value=None):
        """ Return a dict of interpolated data

        Uses scipy.inpterp1d with args:
            - kind, bounds_error, fill_value

        The interpolated data is chosen method select() with args:
            - key, ind

        The interpolation is done against a reference vector x
            - x can be a key to an existing ref
            - x can be user-provided array
                in thay case the group should be specified
                (to properly identify the interpolation dimension)

        Returns:
        --------
        dout:       dict
            dict of interpolated data
        dfail:  dict of failed interpolations, with error messages

        """

        # Check x
        assert x is not None
        if isinstance(x) is str:
            if x not in self.lref:
                msg = "If x is a str, it must be a valid ref!\n"
                msg += "    - x: {}\n".format(x)
                msg += "    - self.lref: {}".format(self.lref)
                raise Exception(msg)
            group = self._dref[x]['group']
            x = self._ddata[x]['data']
        else:
            try:
                x = np.atleast_1d(x).ravel()
            except Exception:
                msg = (
                    "The reference with which to interpolate, x, should be:\n"
                    + "    - a key to an existing ref\n"
                    + "    - a 1d np.ndarray"
                )
                raise Exception(x)
            if group not in self.lgroup:
                msg = "Interpolation must be with respect to a group\n"
                msg += "Provided group is not in self.lgroup:\n"
                msg += "    - group: {}".format(group)
                raise Exception(msg)

        # Get keys to interpolate
        if ind is None and key in None:
            lk = self._dgroup[group]['ldata']
        else:
            lk = self._ind_tofrom_key(ind=ind, key=key, returnas=str)

        # Check provided keys are relevant, and get dim index
        dind, dfail = {}, {}
        for kk in lk:
            if kk not in self._dgroup[group]['ldata']:
                # gps = self._ddata[kk]['groups']
                # msg = "Some data not in interpolation group:\n"
                # msg += "    - self.ddata[%s]['groups'] = %s"%(kk,str(gps))
                # msg += "    - Interpolation group: %s"%group
                # raise Exception(msg)
                dfail[kk] = "Not dependent on group {}".format(group)
            else:
                dind[kk] = self._ddata[kk]['groups'].index(group)

        # Start loop for interpolation
        dout = {}
        for kk in dout.keys():
            shape = self._ddata['dict'][kk]['shape']

            if not isinstance(self._ddata[kk]['data'], np.ndarray):
                dfail[kk] = "Not a np.ndarray !"
                continue

            kr = self._ddata['dict'][kk]['refs'][dind[kk]]
            vr = self._ddata['dict'][kr]['data']
            data = self._ddata['dict'][kk]['data']
            try:
                if dind[kk] == len(shape) - 1:
                    dout[kk] = scpinterp.interp1d(vr, y,
                                                  kind=kind, axis=-1,
                                                  bounds_error=bounds_error,
                                                  fill_value=fill_value,
                                                  assume_sorted=True)(x)
                else:
                    dout[kk] = scpinterp.interp1d(vr, y,
                                                  kind=kind, axis=dind[kk],
                                                  bounds_error=bounds_error,
                                                  fill_value=fill_value,
                                                  assume_sorted=True)(x)

            except Exception as err:
                dfail[kk] = str(err)
        return dout, dfail

    # ---------------------
    # Method for fitting models in one direction
    # ---------------------

    # TBC
    def _fit_one_dim(ind=None, key=None, group=None,
                     Type=None, func=None, **kwdargs):
        """ Return the parameters of a fitted function

        The interpolated data is chosen method select() with args:
            - key, ind

        Returns:
        --------
        dout:       dict
            dict of interpolated data
        dfail:  dict of failed interpolations, with error messages

        """
        # Get keys to interpolate
        lk = self._ind_tofrom_key(ind=ind, key=key, group=group, returnas=str)

        # Start model fitting loop on data keys
        dout = {}
        for kk in lk:
            x = None
            axis = None
            dfit = _DataCollection_comp.fit(
                self._ddata['dict'][kk]['data'],
                x=x,
                axis=axis,
                func=func,
                Type=Type,
                **kwdargs,
            )
            dout[kk] = dfit

        return dout

    # ---------------------
    # Methods for plotting data
    # ---------------------

    def plot_as_matrix(
        self,
        key=None,
        ind=None,
        vmin=None,
        vmax=None,
        cmap=None,
        aspect=None,
        dax=None,
        dmargin=None,
        fs=None,
        dcolorbar=None,
        dleg=None,
    ):
        """ Plot the desired 2d data array as a matrix """
        return _DataCollection_plot.plot_as_matrix(
            coll=self,
            key=key,
            ind=ind,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            aspect=aspect,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
        )

    def _plot_timetraces(self, ntmax=1, group='time',
                         key=None, ind=None, Name=None,
                         color=None, ls=None, marker=None, ax=None,
                         axgrid=None, fs=None, dmargin=None,
                         legend=None, draw=None, connect=None, lib=None):
        plotcoll = self.to_PlotCollection(ind=ind, key=key, group=group,
                                          Name=Name, dnmax={group: ntmax})
        return _DataCollection_plot.plot_DataColl(
            plotcoll,
            color=color, ls=ls, marker=marker, ax=ax,
            axgrid=axgrid, fs=fs, dmargin=dmargin,
            draw=draw, legend=legend,
            connect=connect, lib=lib,
        )

    def _plot_axvlines(
        self,
        which=None,
        key=None,
        ind=None,
        param_x=None,
        param_txt=None,
        sortby=None,
        sortby_def=None,
        sortby_lok=None,
        ax=None,
        ymin=None,
        ymax=None,
        ls=None,
        lw=None,
        fontsize=None,
        side=None,
        dcolor=None,
        dsize=None,
        fraction=None,
        figsize=None,
        dmargin=None,
        wintit=None,
        tit=None,
    ):
        """ plot rest wavelengths as vertical lines """

        # Check inputs
        which, dd = self.__check_which(
            which=which, return_dict=True,
        )
        key = self._ind_tofrom_key(which=which, key=key, ind=ind, returnas=str)

        if sortby is None:
            sortby = sortby_def
        if sortby not in sortby_lok:
            msg = (
                """
                For plotting, sorting can be done only by:
                {}

                You provided:
                {}
                """.format(sortby_lok, sortby)
            )
            raise Exception(msg)

        return _DataCollection_plot.plot_axvline(
            din=dd,
            key=key,
            param_x='lambda0',
            param_txt='symbol',
            sortby=sortby, dsize=dsize,
            ax=ax, ymin=ymin, ymax=ymax,
            ls=ls, lw=lw, fontsize=fontsize,
            side=side, dcolor=dcolor,
            fraction=fraction,
            figsize=figsize, dmargin=dmargin,
            wintit=wintit, tit=tit,
        )

    # ---------------------
    # saving => get rid of function
    # ---------------------

    def save(self, path=None, name=None,
             strip=None, sep=None, deep=True, mode='npz',
             compressed=False, verb=True, return_pfe=False):

        # Remove function mpltri if relevant
        lk = [
            k0 for k0, v0 in self._ddata.items()
            if isinstance(v0['data'], dict)
            and 'mpltri' in v0['data'].keys()
        ]
        for k0 in lk:
            del self._ddata[k0]['data']['mpltri']
        lk = [
            k0 for k0, v0 in self._ddata.items()
            if isinstance(v0['data'], dict)
            and 'trifind' in v0['data'].keys()
        ]
        for k0 in lk:
            del self._ddata[k0]['data']['trifind']

        # call parent method
        return super().save(
            path=path, name=name,
            sep=sep, deep=deep, mode=mode,
            strip=strip, compressed=compressed,
            return_pfe=return_pfe, verb=verb
        )
