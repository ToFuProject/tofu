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
from . import _check_inputs
from . import _comp_new
from . import _plot_new
from . import _def
from . import _comp_spectrallines

__all__ = ['DataCollection'] # , 'TimeTraceCollection']
_INTERPT = 'zero'


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
    _ddef = {'Id': {'include': ['Mod', 'Cls', 'Name', 'version']},
             'params': {'origin': (str, 'unknown'),
                        'dim':    (str, 'unknown'),
                        'quant':  (str, 'unknown'),
                        'name':   (str, 'unknown'),
                        'units':  (str, 'a.u.')}}
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

    _dgroup = {}
    _dref = {}
    _ddata = {}

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
        ddata=None,
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
        self._ddata = {}

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

    def _init(self, dgroup=None, dref=None, ddata=None, **kwargs):
        self.update(dgroup=dgroup, dref=dref, ddata=ddata)
        self._dstrip['strip'] = 0

    ###########
    # set dictionaries
    ###########

    def update(self, ddata=None, dref=None, dgroup=None):
        """ Can be used to set/add data/ref/group

        Will update existing attribute with new dict
        """
        # Check consistency
        self._dgroup, self._dref, self._ddata = _check_inputs._consistency(
            ddata=ddata, ddata0=self._ddata,
            dref=dref, dref0=self._dref,
            dgroup=dgroup, dgroup0=self._dgroup,
            allowed_groups=self._allowed_groups,
            reserved_keys=self._reserved_keys,
            ddefparams=self._ddef['params'],
            data_none=self._data_none,
        )

    # ---------------------
    # Adding group / ref / quantity one by one
    # ---------------------

    def add_group(self, group=None):
        # Check consistency
        self.update(ddata=None, dref=None, dgroup=group)

    def add_ref(self, key=None, group=None, data=None, **kwdargs):
        dref = {key: {'group': group, 'data': data, **kwdargs}}
        # Check consistency
        self.update(ddata=None, dref=dref, dgroup=None)

    def add_data(self, key=None, data=None, ref=None, **kwdargs):
        ddata = {key: {'data': data, 'ref': ref, **kwdargs}}
        # Check consistency
        self.update(ddata=ddata, dref=None, dgroup=None)

    # ---------------------
    # Removing group / ref / quantities
    # ---------------------

    def remove_group(self, group=None):
        """ Remove a group (or list of groups) and all associated ref, data """
        self._dgroup, self._dref, self._ddata = _check_inputs._remove_group(
            group=group,
            dgroup0=self._dgroup, dref0=self._dref, ddata0=self._ddata,
        )

    def remove_ref(self, key=None, propagate=None):
        """ Remove a ref (or list of refs) and all associated data """
        self._dgroup, self._dref, self._ddata = _check_inputs._remove_ref(
            key=key,
            dgroup0=self._dgroup, dref0=self._dref, ddata0=self._ddata,
            propagate=propagate,
        )

    def remove_data(self, key=None, propagate=True):
        """ Remove a data (or list of data) """
        self._dgroup, self._dref, self._ddata = _check_inputs._remove_data(
            key=key,
            dgroup0=self._dgroup, dref0=self._dref, ddata0=self._ddata,
            propagate=propagate,
        )

    # ---------------------
    # Get / set / add / remove params
    # ---------------------

    def get_param(self, param=None, key=None, ind=None, returnas=np.ndarray):
        """ Return the array of the chosen parameter (or list of parameters)

        Can be returned as:
            - dict: {param0: {key0: values0, key1: value1...}, ...}
            - np[.ndarray: {param0: np.r_[values0, value1...], ...}

        """
        return _check_inputs._get_param(
            ddata=self._ddata, param=param,
            key=key, ind=ind, returnas=returnas,
        )

    def set_param(self, param=None, value=None, ind=None, key=None):
        """ Set the value of a parameter

        value can be:
            - None
            - a unique value (int, float, bool, str, tuple) => common to all keys
            - an iterable of vlues (array, list) => one for each key

        A subset of keys can be chosen (ind, key, fed to self.select()) to set
        only the value of some key

        """
        _check_inputs._set_param(
            ddata=self._ddata, param=param, value=value, ind=ind, key=key,
        )

    def add_param(self, param, value=None):
        """ Add a parameter, optionnally also set its value """
        _check_inputs._add_param(
            ddata=self._ddata, param=param, value=value
        )

    def remove_param(self, param=None):
        """ Remove a parameter, none by default, all if param = 'all' """
        _check_inputs._remove_param(ddata=self._ddata, param=param)

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
        dout = {'dgroup': {'dict': self._dgroup, 'lexcept': None},
                'dref': {'dict': self._dref, 'lexcept': None},
                'ddata': {'dict': self._ddata, 'lexcept': None}}
        return dout

    def _from_dict(self, fd):
        self._dgroup.update(**fd['dgroup'])
        self._dref.update(**fd['dref'])
        self._ddata.update(**fd['ddata'])
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
    def ddata(self):
        """ the dict of data """
        return self._ddata

    @property
    def lparam(self):
        return [
            k0 for k0 in list(self._ddata.values())[0].keys() if k0 != 'data'
        ]

    @property
    def dparams(self):
        """ Return a dict of params """
        dp = {
            k0: {k1: v1 for k1, v1 in v0.items() if k1 != 'data'}
            for k0, v0 in self._ddata.items()
        }
        for k0 in dp.keys():
            if isinstance(self._ddata[k0]['data'], np.ndarray):
                dp[k0]['data'] = self._ddata[k0]['data'].dtype.name
            else:
                dp[k0]['data'] = type(self._ddata[k0]['data'])
        return dp

    ###########
    # General use methods
    ###########

    def to_DataFrame(self):
        import pandas as pd
        return pd.DataFrame(self.dparams)

    # ---------------------
    # Key selection methods
    # ---------------------

    def select(self, log=None, returnas=None, **kwdargs):
        """ Return the indices / keys of data matching criteria

        The selection is done comparing the value of all provided parameters
        The result is a boolean indices array, optionally with the keys list
        It can include:
            - log = 'all': only the data matching all criteria
            - log = 'any': the data matching any criterion

        If log = 'raw', a dict of indices arrays is returned, showing the
        details for each criterion

        """
        return _check_inputs._select(
            ddata=self._ddata, log=log, returnas=returnas, **kwdargs,
        )

    def _ind_tofrom_key(self, ind=None, key=None, group=None, returnas=int):
        """ Return ind from key or key from ind for all data """
        return _check_inputs._ind_tofrom_key(
            ddata=self._ddata, ind=ind, key=key,
            group=group, dgroup=self._dgroup,
            returnas=returnas,
        )

    def get_sort_index(self, param=None):
        """ Return sorting index ofself.ddata dict """

        if param is None:
            return
        if param == 'key':
            return np.argsort(list(self._ddata.keys()))
        elif isinstance(param, str):
            return np.argsort(
                self.get_param(param, returnas=np.ndarray)[param]
            )
        else:
            msg = "Arg param must be a valid str\n  Provided: {}".format(param)
            raise Exception(msg)

    def sortby(self, param=None, order=None):
        """ sort the self.ddata dict by desired parameter """

        c0 = order in [None, 'increasing', 'reverse']
        if not c0:
            msg = (
                """
                Arg order must be in [None, 'increasing', 'reverse']
                Provided: {}
                """.format(order)
            )
            raise Exception(msg)

        lk = list(self._ddata.keys())
        ind = self.get_sort_index(param)
        if order == 'reverse':
            self._ddata = {lk[ii]: self._ddata[lk[ii]] for ii in ind[::-1]}
        else:
            self._ddata = {lk[ii]: self._ddata[lk[ii]] for ii in ind}

    # ---------------------
    # Methods for getting a subset of the collection
    # ---------------------

    def get_drefddata_as_input(self, key=None, ind=None, group=None):
        lk = self._ind_tofrom_key(ind=ind, key=key, group=group, returnas=str)
        lkr = [kr for kr in self._dref['lkey']
               if any([kr in self._ddata['dict'][kk]['refs'] for kk in lk])]
        dref = {kr: {'data': self._ddata['dict'][kr]['data'],
                     'group': self._dref['dict'][kr]['group']} for kr in lkr}
        lkr = dref.keys()
        ddata = {kk: self._ddata['dict'][kk] for kk in lk if kk not in lkr}
        return dref, ddata


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

    def to_PlotCollection(self, key=None, ind=None, group=None, Name=None,
                          dnmax=None, lib='mpl'):
        dref, ddata = self.get_drefddata_as_input(key=key, ind=ind, group=group)
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

    def get_summary(self, show=None, show_core=None,
                    sep='  ', line='-', just='l',
                    table_sep=None, verb=True, return_=False):
        """ Summary description of the object content """
        # # Make sure the data is accessible
        # msg = "The data is not accessible because self.strip(2) was used !"
        # assert self._dstrip['strip']<2, msg

        # -----------------------
        # Build for groups
        col0 = ['group name', 'nb. ref', 'nb. data']
        ar0 = [(k0,
                len(self._dgroup[k0]['lref']),
                len(self._dgroup[k0]['ldata']))
               for k0 in self._dgroup.keys()]

        # -----------------------
        # Build for refs
        col1 = ['ref key', 'group', 'size', 'nb. data']
        ar1 = [(k0,
                self._dref[k0]['group'],
                str(self._dref[k0]['size']),
                len(self._dref[k0]['ldata']))
               for k0 in self._dref.keys()]

        # -----------------------
        # Build for ddata
        col2 = ['key']
        if show_core is None:
            show_core = self._show_in_summary_core
        if isinstance(show_core, str):
            show_core = [show_core]
        lp = self.lparam
        lkcore = ['shape', 'group', 'ref']
        assert all([ss in lp + lkcore for ss in show_core])
        col2 += show_core

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
            lu = [k0] + [str(self._ddata[k0][cc]) for cc in col2[1:]]
            ar2.append(lu)

        return self._get_summary(
            [ar0, ar1, ar2], [col0, col1, col2],
            sep=sep, line=line, table_sep=table_sep,
            verb=verb, return_=return_)


    # -----------------
    # conversion wavelength - energy - frequency
    # ------------------

    @staticmethod
    def convert_spectral(
        data_in=None,
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
            data_in=data_in, units_in=units_in, units_out=units_out,
            returnas=returnas,
        )

    # ---------------------
    # Method for interpolating on ref
    # ---------------------

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
                msg += "    - x: %s\n"%str(x)
                msg += "    - self.lref: %s"%str(self.lref)
                raise Exception(msg)
            group = self._dref[x]['group']
            x = self._ddata[x]['data']
        else:
            try:
                x = np.atleast_1d(x).ravel()
            except Exception:
                msg = "The reference with which to interpolate, x, should be:\n"
                msg += "    - a key to an existing ref\n"
                msg += "    - a 1d np.ndarray"
                raise Exception(x)
            if group not in self.lgroup:
                msg = "Interpolation must be with respect to a group\n"
                msg += "Provided group is not in self.lgroup:\n"
                msg += "    - group: %s"%str(group)
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
                dfail[kk] = "Not dependent on group %s"%group
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
            dfit = _comp_new.fit(self._ddata['dict'][kk]['data'],
                                 x=x, axis=axis,
                                 func=func, Type=Type, **kwdargs)
            dout[kk] = dfit

        return dout


    # ---------------------
    # Methods for plotting data
    # ---------------------

    # To be overloaded
    @abstractmethod
    def plot(self):
        pass

    def _plot_timetraces(self, ntmax=1, group='time',
                         key=None, ind=None, Name=None,
                         color=None, ls=None, marker=None, ax=None,
                         axgrid=None, fs=None, dmargin=None,
                         legend=None, draw=None, connect=None, lib=None):
        plotcoll = self.to_PlotCollection(ind=ind, key=key, group=group,
                                          Name=Name, dnmax={group:ntmax})
        return _plot_new.plot_DataColl(plotcoll,
                                       color=color, ls=ls, marker=marker, ax=ax,
                                       axgrid=axgrid, fs=fs, dmargin=dmargin,
                                       draw=draw, legend=legend,
                                       connect=connect, lib=lib)

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
