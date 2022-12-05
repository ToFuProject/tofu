# -*- coding: utf-8 -*-


# Built-in
import sys
import os
# import itertools as itt
import copy
import warnings
# from abc import ABCMeta, abstractmethod
import inspect
import copy


# Common
import numpy as np
import scipy.interpolate as scpinterp
# import matplotlib.pyplot as plt
# from matplotlib.tri import Triangulation as mplTri
import datastock as ds


# tofu
# from tofu import __version__ as __version__
import tofu.utils as utils
from . import _DataCollection_check_inputs
from . import _comp
from . import _DataCollection_comp
from . import _def
# from . import _comp_spectrallines


__all__ = ['DataCollectionBase']    # , 'TimeTraceCollection']


_INTERPT = 'zero'


#############################################
#############################################
#       Abstract Parent class
#############################################
#############################################


class DataCollection0(utils.ToFuObject):
    """ A generic class for handling data

    Provides methods for:
        - introspection
        - plateaux finding
        - visualization

    """
    # __metaclass__ = ABCMeta

    # Fixed (class-wise) dictionary of default properties
    _ddef = {
        'Id': {'include': ['Mod', 'Cls', 'Name', 'version']},
        'params': {
            'ddata': {
                'units':  (str, 'a.u.'),
                'dim':    (str, 'unknown'),
                'quant':  (str, 'unknown'),
                'name':   (str, 'unknown'),
                'source': (str, 'unknown'),
            },
            'dobj': {},
            'dstatic': {},
         },
    }

    # _dallowed_params = None
    _data_none = None
    _reserved_keys = None

    _show_in_summary_core = ['shape', 'ref']
    _show_in_summary = 'all'
    _max_ndim = None

    _dref = {}
    _dstatic = {}
    _ddata = {}
    _dobj = {}

    def __init_subclass__(cls, **kwdargs):
        # Does not exist before Python 3.6 !!!
        # Python 2
        super().__init_subclass__(**kwdargs)
        # Python 3
        # super().__init_subclass__(**kwdargs)
        cls._ddef = copy.deepcopy(cls._ddef)
        # cls._dplot = copy.deepcopy(Struct._dplot)
        # cls._set_color_ddef(cls._color)

    def __init__(
        self,
        dref=None,
        dstatic=None,
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
        self._dref = {}
        self._dstatic = {}
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
        dref=None,
        dstatic=None,
        ddata=None,
        dobj=None,
        **kwargs,
    ):
        self.update(
            dref=dref,
            dstatic=dstatic,
            ddata=ddata,
            dobj=dobj,
        )
        # self._dstrip['strip'] = 0

    ###########
    # set dictionaries
    ###########

    def update(
        self,
        dobj=None,
        ddata=None,
        dref=None,
        dstatic=None,
    ):
        """ Can be used to set/add data/ref

        Will update existing attribute with new dict
        """
        # Check consistency
        (
            self._dref, self._dstatic, self._ddata, self._dobj,
        ) = _DataCollection_check_inputs._consistency(
            dobj=dobj, dobj0=self._dobj,
            ddata=ddata, ddata0=self._ddata,
            dref=dref, dref0=self._dref,
            dstatic=dstatic, dstatic0=self._dstatic,
            reserved_keys=self._reserved_keys,
            ddefparams_data=self._ddef['params']['ddata'],
            ddefparams_obj=self._ddef['params']['dobj'],
            ddefparams_static=self._ddef['params']['dstatic'],
            data_none=self._data_none,
            max_ndim=self._max_ndim,
        )

    # ---------------------
    # Adding ref / quantity one by one
    # ---------------------

    def add_ref(self, key=None, data=None, size=None, **kwdargs):
        dref = {key: {'data': data, 'size': size, **kwdargs}}
        # Check consistency
        self.update(ddata=None, dref=dref, dstatic=None)

    def add_static(self, key=None, which=None, **kwdargs):
        dstatic = {which: {key: kwdargs}}
        # Check consistency
        self.update(ddata=None, dref=None, dstatic=dstatic)

    def add_data(self, key=None, data=None, ref=None, **kwdargs):
        ddata = {key: {'data': data, 'ref': ref, **kwdargs}}
        # Check consistency
        self.update(ddata=ddata, dref=None, dstatic=None)

    def add_obj(self, which=None, key=None, **kwdargs):
        dobj = {which: {key: kwdargs}}
        # Check consistency
        self.update(dobj=dobj, dref=None, dstatic=None)

    # ---------------------
    # Removing ref / quantities
    # ---------------------

    def remove_ref(self, key=None, propagate=None):
        """ Remove a ref (or list of refs) and all associated data """
        (
            self._dref, self._dstatic, self._ddata, self._dobj,
        ) = _DataCollection_check_inputs._remove_ref(
            key=key,
            dref0=self._dref, ddata0=self._ddata,
            dstatic0=self._dstatic,
            dobj0=self._dobj,
            propagate=propagate,
            reserved_keys=self._reserved_keys,
            ddefparams_data=self._ddef['params']['ddata'],
            ddefparams_obj=self._ddef['params']['dobj'],
            data_none=self._data_none,
            max_ndim=self._max_ndim,
        )

    def remove_static(self, key=None, which=None, propagate=None):
        """ Remove a static ref (or list) or a whole category

        key os provided:
            => remove only the desired key(s)
                works only if key is not used in ddata and dobj

        which is provided:
            => treated as param, the whole category of ref_static is removed
                if propagate, the parameter is removed from ddata and dobj
        """
        _DataCollection_check_inputs._remove_static(
            key=key,
            which=which,
            propagate=propagate,
            dstatic0=self._dstatic,
            ddata0=self._ddata,
            dobj0=self._dobj,
        )

    def remove_data(self, key=None, propagate=True):
        """ Remove a data (or list of data) """
        (
            self._dref, self._dstatic, self._ddata, self._dobj,
        ) = _DataCollection_check_inputs._remove_data(
            key=key,
            dref0=self._dref, ddata0=self._ddata,
            dstatic0=self._dstatic,
            dobj0=self._dobj,
            propagate=propagate,
            reserved_keys=self._reserved_keys,
            ddefparams_data=self._ddef['params']['ddata'],
            ddefparams_obj=self._ddef['params']['dobj'],
            data_none=self._data_none,
            max_ndim=self._max_ndim,
        )

    def remove_obj(self, key=None, which=None, propagate=True):
        """ Remove a data (or list of data) """
        (
            self._dref, self._dstatic, self._ddata, self._dobj,
        ) = _DataCollection_check_inputs._remove_obj(
            key=key,
            which=which,
            dobj0=self._dobj,
            ddata0=self._ddata,
            dref0=self._dref,
            dstatic0=self._dstatic,
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
            dref=self._dref,
            ddata=self._ddata,
            dobj=self._dobj,
            dstatic=self._dstatic,
            which=which,
            return_dict=return_dict,
        )

    def get_lparam(self, which=None):
        """ Return the list of params for the chosen dict

        which can be:
            - 'ref'
            - 'data'
            - dobj[<which>]
            - dstatic[<which>]
        """
        which, dd = self.__check_which(which, return_dict=True)
        return list(list(dd.values())[0].keys())

    def get_param(
        self,
        param=None,
        key=None,
        ind=None,
        returnas=None,
        which=None,
    ):
        """ Return the array of the chosen parameter (or list of parameters)

        which can be:
            - 'ref'
            - 'data'
            - dobj[<which>]
            - dstatic[<which>]

        param cen be a str or a list of str

        Can be returned as:
            - dict: {param0: {key0: values0, key1: value1...}, ...}
            - np.ndarray: {param0: np.r_[values0, value1...], ...}

        """
        which, dd = self.__check_which(which, return_dict=True)
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
        distribute=None,
    ):
        """ Set the value of a parameter

        which can be:
            - 'ref'
            - 'data'
            - dobj[<which>]
            - dstatic[<which>]

        value can be:
            - None
            - a unique value (int, float, bool, str, tuple) common to all keys
            - an iterable of vlues (array, list) => one for each key

        A subset of keys can be chosen (ind, key, fed to self.select()) to set
        only the value of some key

        """
        which, dd = self.__check_which(which, return_dict=True)
        _DataCollection_check_inputs._set_param(
            dd=dd, dd_name=which,
            param=param, value=value, ind=ind, key=key,
            distribute=distribute,
        )

    def add_param(
        self,
        param,
        value=None,
        which=None,
    ):
        """ Add a parameter, optionnally also set its value """
        which, dd = self.__check_which(which, return_dict=True)
        _DataCollection_check_inputs._add_param(
            dd=dd,
            dd_name=which,
            param=param,
            value=value,
        )

    def remove_param(
        self,
        param=None,
        which=None,
    ):
        """ Remove a parameter, none by default, all if param = 'all' """
        which, dd = self.__check_which(which, return_dict=True)
        _DataCollection_check_inputs._remove_param(
            dd=dd,
            dd_name=which,
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
            'dref': {'dict': self._dref, 'lexcept': None},
            'dstatic': {'dict': self._dstatic, 'lexcept': None},
            'ddata': {'dict': self._ddata, 'lexcept': None},
            'dobj': {'dict': self._dobj, 'lexcept': None},
        }
        return dout

    def _from_dict(self, fd):
        for k0 in ['dref', 'ddata', 'dstatic', 'dobj']:
            if fd.get(k0) is not None:
                getattr(self, '_'+k0).update(**fd[k0])
        self.update()

    ###########
    # properties
    ###########

    @property
    def dref(self):
        """ the dict of references """
        return self._dref

    @property
    def dstatic(self):
        """ the dict of references """
        return self._dstatic

    @property
    def ddata(self):
        """ the dict of data """
        return self._ddata

    @property
    def dobj(self):
        """ the dict of obj """
        return self._dobj

    ###########
    # set and propagate indices for refs
    ###########

    def add_indices_per_ref(self, indices=None, ref=None, distribute=None):

        lparam = self.get_lparam(which='ref')
        if 'indices' not in lparam:
            self.add_param('indices', which='ref')

        self.set_param(
            which='ref',
            param='indices',
            key=ref,
            value=np.array(indices).ravel(),
            distribute=distribute,
        )

    def propagate_indices_per_ref(
        self,
        ref=None,
        lref=None,
        ldata=None,
        param=None,
    ):
        """ Propagate the indices set for a ref to all other lref

        Index propagation is done:
            - ldata = list of len() = 1 + len(lref)
                according to arbitrary (monotonous) data for each ref
            - according to a criterion:
                - 'index': set matching indices (default)
                - param: set matching monotonous quantities depending on ref
        """
        _DataCollection_comp.propagate_indices_per_ref(
            ref=ref,
            lref=lref,
            ldata=ldata,
            dref=self._dref,
            ddata=self._ddata,
            param=param,
            lparam_data=self.get_lparam(which='data')
        )

    ###########
    # extract
    ###########

    def extract(self, keys=None):
        """ Extract some selected data and return as new instance """

        # ----------------
        # check inputs

        if keys is None:
            return
        if isinstance(keys, str):
            keys = [keys]

        keys = ds._generic_check._check_var_iter(
            keys, 'keys',
            types=list,
            allowed=self._ddata.keys(),
        )

        # -----------------------------
        # Get corresponding list of ref

        lref = set([
            k0 for k0, v0 in self._dref.items()
            if any([ss in keys for ss in v0['ldata']])
        ])

        # -------------------
        # Populate with ref

        coll = self.__class__()

        lpar = [
            pp for pp in self.get_lparam(which='ref')
            if pp not in ['ldata', 'ldata_monot', 'ind', 'data']
        ]
        for k0 in lref:
            coll.add_ref(
                key=k0,
                **copy.deepcopy({pp: self._dref[k0][pp] for pp in lpar}),
            )

        # -------------------
        # Populate with data

        lpar = [
            pp for pp in self.get_lparam(which='data')
            if pp not in ['shape', 'monot']
        ]
        for k0 in keys:
            coll.add_data(
                key=k0,
                **copy.deepcopy({pp: self._ddata[k0][pp] for pp in lpar}),
            )

        return coll

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
        return _DataCollection_check_inputs._select(
            dd=dd, dd_name=which,
            log=log, returnas=returnas, **kwdargs,
        )

    def _ind_tofrom_key(
        self,
        ind=None,
        key=None,
        returnas=int,
        which=None,
    ):
        """ Return ind from key or key from ind for all data """
        which, dd = self.__check_which(which, return_dict=True)
        return _DataCollection_check_inputs._ind_tofrom_key(
            dd=dd, dd_name=which, ind=ind, key=key,
            returnas=returnas,
        )

    def _get_sort_index(self, which=None, param=None):
        """ Return sorting index of self.ddata dict """

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

        # --------------
        # Check inputs

        # order
        order = ds._generic_check._check_var(
            order,
            'order',
            types=str,
            default='increasing',
            allowed=['increasing', 'reverse'],
        )

        # which
        which, dd = self.__check_which(which, return_dict=True)

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
        elif which == 'ref':
            self.dref = dd
        elif which in self._dobj.keys():
            self._dobj[which] = dd
        elif which in self._dstatic.keys():
            self._dstatic[which] = dd

    # ---------------------
    # Methods for getting a subset of the collection
    # ---------------------

    # TBC
    def get_drefddata_as_input(self, key=None, ind=None):
        lk = self._ind_tofrom_key(ind=ind, key=key, returnas=str)
        lkr = [kr for kr in self._dref['lkey']
               if any([kr in self._ddata['dict'][kk]['refs'] for kk in lk])]
        dref = {kr: {'data': self._ddata['dict'][kr]['data']} for kr in lkr}
        lkr = dref.keys()
        ddata = {kk: self._ddata['dict'][kk] for kk in lk if kk not in lkr}
        return dref, ddata

    # TBC
    def get_subset(self, key=None, ind=None, Name=None):
        if key is None and ind is None:
            return self
        else:
            dref, ddata = self.get_drefddata_as_input(key=key, ind=ind)
            if Name is None and self.Id.Name is not None:
                Name = self.Id.Name + '-subset'
            return self.__class__(dref=dref, ddata=ddata, Name=Name)

    # ---------------------
    # Methods for exporting plot collection (subset)
    # ---------------------

    # TBC
    """
    def to_PlotCollection(self, key=None, ind=None, Name=None,
                          dnmax=None, lib='mpl'):
        dref, ddata = self.get_drefddata_as_input(key=key, ind=ind)
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
    """

    # ---------------------
    # Methods for showing data
    # ---------------------

    def get_summary(
        self,
        show_which=None,
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

        # ------------
        # check inputs

        if show_which is None:
            show_which = ['ref', 'data', 'static', 'obj']

        lcol, lar = [], []

        # -----------------------
        # Build for dref

        if 'ref' in show_which and len(self._dref) > 0:
            lcol.append(['ref key', 'size', 'nb. data', 'nb. data monot.'])
            lar.append([
                [
                    k0,
                    str(self._dref[k0]['size']),
                    len(self._dref[k0]['ldata']),
                    len(self._dref[k0]['ldata_monot']),
                ]
                for k0 in self._dref.keys()
            ])

            lp = self.get_lparam(which='ref')
            if 'indices' in lp:
                lcol[0].append('indices')
                for ii, (k0, v0) in enumerate(self._dref.items()):
                    if self._dref[k0]['indices'] is None:
                        lar[0][ii].append(str(v0['indices']))
                    else:
                        lar[0][ii].append(str(list(v0['indices'])))

            if 'group' in lp:
                lcol[0].append('group')
                for ii, (k0, v0) in enumerate(self._dref.items()):
                    lar[0][ii].append(str(self._dref[k0]['group']))

            if 'inc' in lp:
                lcol[0].append('increment')
                for ii, (k0, v0) in enumerate(self._dref.items()):
                    lar[0][ii].append(str(self._dref[k0]['inc']))

        # -----------------------
        # Build for ddata

        if 'data' in show_which and len(self._ddata) > 0:

            if show_core is None:
                show_core = self._show_in_summary_core
            if isinstance(show_core, str):
                show_core = [show_core]

            lp = self.get_lparam(which='data')
            lkcore = ['shape', 'ref']
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
            col2 = [cc for cc in col2 if cc != 'data']

            ar2 = []
            for k0 in self._ddata.keys():
                lu = [k0] + [str(self._ddata[k0].get(cc)) for cc in col2[1:]]
                ar2.append(lu)

            lcol.append(col2)
            lar.append(ar2)

        # -----------------------
        # Build for dstatic

        anystatic = (
            len(self._dstatic) > 0
            and any([
                ss in show_which
                for ss in ['static'] + list(self._dstatic.keys())
            ])
        )
        if anystatic:
            for k0, v0 in self._dstatic.items():
                if 'static' in show_which or k0 in show_which:
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

        anyobj = (
            len(self._dobj) > 0
            and any([
                ss in show_which
                for ss in ['obj'] + list(self._dobj.keys())
            ])
        )
        if anyobj:
            for k0, v0 in self._dobj.items():
                if 'obj' in show_which or k0 in show_which:
                    lk = self.get_lparam(which=k0)
                    lk = [
                        kk for kk in lk
                        if 'func' not in kk
                        and 'class' not in kk
                        and kk not in ['handle']
                        and not (k0 == 'axes' and kk == 'bck')
                        and all([
                            not isinstance(v1[kk], dict)
                            for v1 in v0.values()
                        ])
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

    def show_all(self):
        self.get_summary(show_which=None)

    def show_data(self):
        self.get_summary(show_which=['ref', 'data'])

    def show_interactive(self):
        self.get_summary(show_which=['axes', 'mobile'])

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
        lkey=None,
        return_all=None,
    ):
        """ Typically used to get a common (intersection) time vector

        Returns a time vector that contains all time points from all data
        Also return a dict of indices to easily remap each time vector to tall
            such that t[ind] = tall (with nearest approximation)

        """
        return _DataCollection_comp._get_unique_ref_dind(
            dd=self._ddata,
            lkey=lkey,
            return_all=return_all,
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
    # Method for interpolation
    # ---------------------

    def _get_finterp(
        self,
        idquant=None, idref1d=None, idref2d=None, idmesh=None,
        idq2dR=None, idq2dPhi=None, idq2dZ=None,
        interp_t=None, interp_space=None,
        fill_value=None, ani=None, Type=None,
    ):

        if interp_t is None:
            interp_t = 'nearest'
        if interp_t != 'nearest':
            msg = "'nearest' is the only time-interpolation method available"
            raise NotImplementedError(msg)

        # Get idmesh
        if idmesh is None:
            if idquant is not None:
                # isotropic
                if idref1d is None:
                    lidmesh = [qq for qq in self._ddata[idquant]['ref']]
                else:
                    lidmesh = [qq for qq in self._ddata[idref2d]['ref']]
            else:
                # anisotropic
                assert idq2dR is not None
                lidmesh = [qq for qq in self._ddata[idq2dR]['ref']]
            assert len(lidmesh) == 1
            idmesh = lidmesh[0]

        # Get common time indices
        if interp_t == 'nearest':
            tall, tbinall, ntall, dind = _DataCollection_comp._get_tcom(
                idquant, idref1d, idref2d, idq2dR,
                dd=self._ddata,
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

    # TBC
    def _interp_one_dim(x=None, ind=None, key=None,
                        kind=None, bounds_error=None, fill_value=None):
        """ Return a dict of interpolated data

        Uses scipy.inpterp1d with args:
            - kind, bounds_error, fill_value

        The interpolated data is chosen method select() with args:
            - key, ind

        The interpolation is done against a reference vector x
            - x can be a key to an existing ref
            - x can be user-provided array
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

        # Get keys to interpolate
        lk = self._ind_tofrom_key(ind=ind, key=key, returnas=str)

        # Check provided keys are relevant, and get dim index
        dind, dfail = {}, {}
        for kk in lk:
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
    def _fit_one_dim(ind=None, key=None,
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
        lk = self._ind_tofrom_key(ind=ind, key=key, returnas=str)

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
