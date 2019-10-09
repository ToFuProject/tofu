# -*- coding: utf-8 -*-

# Built-in
import sys
import os
# import itertools as itt
import copy
import warnings
from abc import ABCMeta, abstractmethod
if sys.version[0] == '3':
    import inspect
else:
    # Python 2 back-porting
    import funcsigs as inspect

# Common
import numpy as np
# import scipy.interpolate as scpinterp
# import matplotlib.pyplot as plt
# from matplotlib.tri import Triangulation as mplTri


# tofu
from tofu import __version__ as __version__
import tofu.pathfile as tfpf
import tofu.utils as utils
try:
    import tofu.data._comp as _comp
    import tofu.data._plot as _plot
    import tofu.data._def as _def
    import tofu.data._physics as _physics
except Exception:
    from . import _comp as _comp
    from . import _plot as _plot
    from . import _def as _def
    from . import _physics as _physics

__all__ = ['DataHolder', 'TimeTraceCollection']

_SAVEPATH = os.path.abspath('./')
_INTERPT = 'zero'


#############################################
#############################################
#       Abstract Parent class
#############################################
#############################################


class DataHolder(utils.ToFuObject):
    """ A generic class for handling data

    Provides methods for:
        - introspection
        - plateaux finding
        - visualization

    """
    __metaclass__ = ABCMeta

    # Fixed (class-wise) dictionary of default properties
    _ddef = {'Id': {'include': ['Mod', 'Cls', 'Name', 'version']},
             'dgroup': ['lref'],
             'dref':   ['group', 'size', 'ldata'],
             'ddata':  ['refs', 'shape', 'groups', 'data'],
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
    _dallowed_params = None

    _reserved_all = _ddef['dgroup'] + _ddef['dref'] + _ddef['ddata']
    _show_in_summary_core = ['shape', 'refs', 'groups']
    _show_in_summary = 'all'

    def __init_subclass__(cls, **kwdargs):
        # Does not exist before Python 3.6 !!!
        # Python 2
        super(DataHolder, cls).__init_subclass__(**kwdargs)
        # Python 3
        # super().__init_subclass__(**kwdargs)
        cls._ddef = copy.deepcopy(DataHolder._ddef)
        # cls._dplot = copy.deepcopy(Struct._dplot)
        # cls._set_color_ddef(cls._color)

    def __init__(self, dref=None, ddata=None,
                 Id=None, Name=None,
                 fromdict=None, SavePath=os.path.abspath('./'),
                 SavePath_Include=tfpf.defInclude):

        # To replace __init_subclass__ for Python 2
        if sys.version[0] == '2':
            self._dstrip = utils.ToFuObjectBase._dstrip.copy()
            self.__class__._strip_init()

        # Create a dplot at instance level
        # self._dplot = copy.deepcopy(self.__class__._dplot)

        kwdargs = locals()
        del kwdargs['self']
        # super()
        super(DataHolder, self).__init__(**kwdargs)

    def _reset(self):
        # Run by the parent class __init__()
        # super()
        super(DataHolder, self)._reset()
        self._dgroup = {kd[0]: kd[1] for kd in self._get_keys_dgroup()}
        self._dref = {kd[0]: kd[1] for kd in self._get_keys_dref()}
        self._ddata = {kd[0]: kd[1] for kd in self._get_keys_ddata()}

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
    # Get largs
    ###########

    @staticmethod
    def _get_largs_dref():
        largs = ['dref']
        return largs

    @staticmethod
    def _get_largs_ddata():
        largs = ['ddata']
        return largs

    ###########
    # Get check and format inputs
    ###########

    # ---------------------
    # Methods for checking and formatting inputs
    # ---------------------

    def _extract_known_params(self, key, dd, ref=False, group=None):
        # Extract relevant parameters
        dparams = {kk: vv for kk, vv in dd.items()
                   if kk not in self._reserved_all}

        if ref and group is not None and self._dallowed_params is not None:
            defpars = self._dallowed_params[group]
        else:
            defpars = self._ddef['params']

        # Add minimum default parameters if not already included
        for kk, vv in defpars.items():
            if kk not in dparams.keys():
                dparams[kk] = vv[1]
            else:
                # Check type if already included
                if not isinstance(dparams[kk], vv[0]):
                    vtyp = str(type(vv[0]))
                    msg = "A parameter for %s has the wrong type:\n"%key
                    msg += "    - Provided: type(%s) = %s\n"%(kk, vtyp)
                    msg += "    - Expected %s"%str(self._ddef['params'][kk][0])
                    raise Exception(msg)
        return dparams

    def _checkformat_dref(self, dref):
        c0 = isinstance(dref, dict)
        c0 = c0 and all([isinstance(kk, str) and isinstance(vv, dict)
                         for kk, vv in dref.items()])
        if not c0:
            msg = "Provided dref must be dict !\n"
            msg += "All its keys must be str !\n"
            msg += "All its values must be dict !"
            raise Exception(msg)

        # Two options:
        #   (A)  - {'group0':{'t0':{'data':t0, 'units':'s'}, 't1':...}}
        #   (B)  - {'t0':{'data':t0, 'units':'s', 'group':'group0'}, 't1':...}
        #   (C)  - {'t0':{'data':t0, 'units':'s'}, 't1':...}
        #   (D)  - {'t0':t0, 't1':t1, ...}

        cA = all([all([(isinstance(v1, dict) and 'group' not in v1.keys())
                       or not isinstance(v1, dict)
                       for v1 in v0.values()])
                  and 'group' not in v0.keys() for v0 in dref.values()])
        cB = all([isinstance(v0.get('group', None), str)
                  for v0 in dref.values()])
        cC = (self._forced_group is not None
              and all([not isinstance(v0, dict) for v0 in dref.values()]))
        cD = (self._forced_group is not None
              and all(['group' not in v0.keys() for v0 in dref.values()]))
        if not (cA or cB or cC or cD):
            msg = "Provided dref must formatted either as a dict with:\n\n"
            msg += "    - keys = group, values = {ref: data}:\n"
            msg += "        {'time':{'t0':{'data':t0, 'units':'s'},\n"
            msg += "                 't1':{'data':t1, 'units':'h'}},\n"
            msg += "         'dist':{'x0':{'data':x0, 'units':'m'}}}\n\n"
            msg += "    - keys = ref, values = {data, group, ...}:\n"
            msg += "        {'t0':{'data':t0, 'units':'s', 'group':'time'},\n"
            msg += "         't1':{'data':t1, 'units':'h', 'group':'time'},\n"
            msg += "         'x0':{'data':x0, 'units':'m', 'group':'dist'}\n\n"
            msg += "    If self._forced_group is not None, 2 more options:\n"
            msg += "    - keys = ref, values = {data, ...}:\n"
            msg += "        {'t0':{'data':t0, 'units':'s'},\n"
            msg += "         't1':{'data':t1, 'units':'h'},\n"
            msg += "         'x0':{'data':x0, 'units':'m'}\n"
            msg += "    - keys = ref, values = data:\n"
            msg += "        {'t0':t0,\n"
            msg += "         't1':t1,\n"
            msg += "         'x0':x0}\n"
            raise Exception(msg)

        if cA:
            # Convert to cB
            drbis = {}
            for k0, v0 in dref.items():
                for k1, v1 in v0.items():
                    if isinstance(v1, dict):
                        drbis[k1] = v1
                        drbis['group'] = k0
                    else:
                        drbis[k1] = {'data': v1, 'group': k0}
            dref = drbis

        # Check cC and cD and convert to cB
        import ipdb         # DB
        ipdb.set_trace()    # DB
        if cC:
            # Convert to cB
            for k0 in dref.keys():
                dref[k0]['group'] = self._forced_group
        elif cD:
            # Convert to cB
            for k0, v0 in dref.items():
                dref[k0] = {'data': v0, 'group': self._forced_group}


        # Check cB = normal case
        for kk, vv in dref.items():

            # Check if new group
            if vv['group'] not in self._dgroup['lkey']:
                self._dgroup['dict'][vv['group']] = {}
                self._dgroup['lkey'].append(vv['group'])

            # Check key unicity
            if kk in self._ddata['lkey']:
                msg = "key '%s' already used !\n"%kk
                msg += "  => each key must be unique !"
                raise Exception(msg)

            # Check data
            c0 = 'data' in vv.keys()
            data = vv['data']
            if not isinstance(data, np.ndarray):
                if isinstance(data, list) or isinstance(data, tuple):
                    try:
                        data = np.atleast_1d(data).ravel()
                        size = data.size
                    except Exception as err:
                        c0 = False
                else:
                    size = data.__class__.__name__
            else:
                if data.ndim != 1:
                    data = np.atleast_1d(data).ravel()
                size = data.size

            if not c0:
                msg = "dref[%s]['data'] must be array-convertible\n"%kk
                msg += "The following array conversion failed:\n"
                msg += "    - np.atleast_1d(dref[%s]['data']).ravel()"%kk
                raise Exception(msg)

            # Fill self._dref
            self._dref['dict'][kk] = {'size': size, 'group': vv['group']}
            self._dref['lkey'].append(kk)

            # Extract and check parameters
            dparams = self._extract_known_params(kk, vv, ref=True,
                                                 group=vv['group'])

            # Fill self._ddata
            self._ddata['dict'][kk] = dict(data=data, refs=(kk,),
                                           shape=(size,), **dparams)
            self._ddata['lkey'].append(kk)

    def _checkformat_ddata(self, ddata):
        c0 = isinstance(ddata, dict)
        c0 = c0 and all([isinstance(kk, str) for kk in ddata.keys()])
        if not c0:
            msg = "Provided ddata must be dict !\n"
            msg += "All its keys must be str !"
            raise Exception(msg)

        # Start check on each key
        for kk, vv in ddata.items():

            # Check key unicity
            if kk in self._ddata['lkey']:
                msg = "key '%s' already used !\n"%kk
                msg += "  => each key must be unique !"
                raise Exception(msg)

            # Check value is a dict with proper keys
            if not isinstance(vv, dict):
                vv = {'data': vv}
            if 'data' not in vv.keys():
                msg = "ddata must contain dict with at least the keys:\n"
                msg += "    - 'refs': a tuple indicating refs dependencies\n"
                msg += "    - 'data': a 1d array containing the data"
                raise Exception(msg)

            # Extract data and shape
            data = vv['data']
            if not isinstance(data, np.ndarray):
                if isinstance(data, list) or isinstance(data, tuple):
                    try:
                        data = np.asarray(data)
                        shape = data.shape
                    except Exception as err:
                        assert type(data) in [list, tuple]
                        shape = (len(data),)
                else:
                    shape = data.__class__.__name__
            else:
                data = np.atleast_1d(np.squeeze(data))
                shape = data.shape

            # Check if refs, or try to identify
            c0 = 'refs' in vv.keys() and isinstance(vv['refs'], tuple)
            if not c0:
                lr = [(rr, shape.index(vr['size']))
                      for rr, vr in self._dref['dict'].items()
                      if vr['size'] in shape]
                if len(lr) == len(shape):
                    order = np.argsort([rr[1] for rr in lr])
                    vv['refs'] = tuple(lr[order[ii]][0] for ii in range(len(lr)))
                else:
                    msg = "The refs of ddata[%s] not found automatically\n"%kk
                    msg += "  => Too many / not enough refs with good size\n"
                    msg += "    - shape  = %s\n"%str(shape)
                    msg += "    - available ref sizes:"
                    msg += "\n        "
                    msg += "\n        ".join([rr + ': ' + str(vv['size'])
                                              for rr, vv
                                              in self._dref['dict'].items()])
                    raise Exception(msg)

            # Check proper ref (existence and shape / size)
            for ii, rr in enumerate(vv['refs']):
                if rr not in self._dref['lkey']:
                    msg = "ddata[%s] depends on an unknown ref !\n"%kk
                    msg += "    - ddata[%s]['refs'] = %s\n"%(kk, rr)
                    msg += "  => %s not in self.dref !\n"%rr
                    msg += "  => self.add_ref( %s ) first !"%rr
                    raise Exception(msg)
            shaprf = tuple(self._dref['dict'][rr]['size'] for rr in vv['refs'])
            if not shape == shaprf:
                msg = "Inconsistency between data shape and ref size !\n"
                msg += "    - ddata[%s]['data'] shape: %s\n"%(kk, str(shape))
                msg += "    - sizes of refs: %s"%(str(shaprf))
                raise Exception(msg)

            # Extract params and set self._ddata
            dparams = self._extract_known_params(kk, vv)
            self._ddata['dict'][kk] = dict(data=data, refs=vv['refs'],
                                           shape=shape, **dparams)
            self._ddata['lkey'].append(kk)

    def _complement_dgrouprefdata(self):

        # --------------
        # ddata
        assert len(self._ddata['lkey']) == len(self._ddata['dict'].keys())
        for k0 in self._ddata['lkey']:
            v0 = self._ddata['dict'][k0]

            # Check all ref are in dref
            lrefout = [ii for ii in v0['refs'] if ii not in self._dref['lkey']]
            if len(lrefout) != 0:
                msg = "ddata[%s]['refs'] has keys not in dref:\n"%k0
                msg += "    - " + "\n    - ".join(lrefout)
                raise Exception(msg)

            # set group
            grps = tuple(self._dref['dict'][rr]['group'] for rr in v0['refs'])
            gout = [gg for gg in grps if gg not in self._dgroup['lkey']]
            if len(gout) > 0:
                lg = self._dgroup['lkey']
                msg = "Inconsistent grps from self.ddata[%s]['refs']:\n"%k0
                msg += "    - grps = %s\n"%str(grps)
                msg += "    - self._dgroup['lkey'] = %s\n"%str(lg)
                msg += "    - self.dgroup.keys() = %s"%str(self.dgroup.keys())
                raise Exception(msg)
            self._ddata['dict'][k0]['groups'] = grps

        # --------------
        # dref
        for k0 in self._dref['lkey']:
            ldata = [kk for kk in self._ddata['lkey']
                     if k0 in self._ddata['dict'][kk]['refs']]
            self._dref['dict'][k0]['ldata'] = ldata
            assert self._dref['dict'][k0]['group'] in self._dgroup['lkey']

        # --------------
        # dgroup
        for gg in self._dgroup['lkey']:
            vg = self._dgroup['dict'][gg]
            lref = [rr for rr in self._dref['lkey']
                    if self._dref['dict'][rr]['group'] == gg]
            ldata = [dd for dd in self._ddata['lkey']
                     if any([dd in self._dref['dict'][vref]['ldata']
                             for vref in lref])]
            # assert vg['depend'] in lidindref
            self._dgroup['dict'][gg]['lref'] = lref
            self._dgroup['dict'][gg]['ldata'] = ldata

        if self._forced_group is not None:
            if len(self.lgroup) != 1 or self.lgroup[0] != self._forced_group:
                msg = "The only allowed group is %s"%self._forced_group
                raise Exception(msg)
        if self._allowed_groups is not None:
            if any([gg not in self._allowed_groups for gg in self.lgroup]):
                msg = "Some groups are not allowed:\n"
                msg += "    - provided: %s\n"%str(self.lgroup)
                msg += "    - allowed:  %s"%str(self._allowed_groups)
                raise Exception(msg)

        # --------------
        # params
        lparam = self._ddata['lparam']
        for kk in self._ddata['lkey']:
            for pp in self._ddata['dict'][kk].keys():
                if pp not in self._reserved_all and pp not in lparam:
                    lparam.append(pp)

        for kk in self._ddata['lkey']:
            for pp in lparam:
                if pp not in self._ddata['dict'][kk].keys():
                    self._ddata['dict'][kk][pp] = None
        self._ddata['lparam'] = lparam

    ###########
    # Get keys of dictionnaries
    ###########

    @staticmethod
    def _get_keys_dgroup():
        lk = [('lkey', []), ('dict', {})]
        return lk

    @staticmethod
    def _get_keys_dref():
        lk = [('lkey', []), ('dict', {})]
        return lk

    @staticmethod
    def _get_keys_ddata():
        lk = [('lkey', []), ('dict', {}), ('lparam', [])]
        return lk

    ###########
    # _init
    ###########

    def _init(self, dref=None, ddata=None, **kwargs):
        kwdargs = dict(dref=dref, ddata=ddata, **kwargs)
        largs = self._get_largs_dref()
        kwddref = self._extract_kwdargs(kwdargs, largs)
        self._set_dref(complement=False, **kwddref)
        largs = self._get_largs_ddata()
        kwddata = self._extract_kwdargs(kwdargs, largs)
        self._set_ddata(**kwddata)
        self._dstrip['strip'] = 0

    ###########
    # set dictionaries
    ###########

    def _set_dref(self, dref, complement=True):
        self._checkformat_dref(dref)
        if complement:
            self._complement_dgrouprefdata()

    def _set_ddata(self, ddata):
        self._checkformat_ddata(ddata)
        self._complement_dgrouprefdata()

    # ---------------------
    # Methods for adding ref / quantities
    # ---------------------

    def add_ref(self, key, data=None, group=None, **kwdargs):
        """ Add a reference """
        self._set_dref({key: dict(data=data, group=group, **kwdargs)})

    def remove_ref(self, key):
        """ Remove a reference (all data depending on it are removed too) """
        assert key in self._dref['lkey']
        lkdata = self._dref['dict'][key]['ldata']
        del self._dref['dict'][key]
        self._dref['lkey'].remove(key)
        for kk in lkdata:
            if key in self._ddata['dict'][kk]['refs']:
                del self._ddata['dict'][kk]
                self._ddata['lkey'].remove(kk)
        self._complement_dgrouprefdata()

    def add_data(self, key, data=None, refs=None, **kwdargs):
        """ Add a data (all associated ref must be added first)) """
        self._set_ddata({key: dict(data=data, refs=refs, **kwdargs)})

    def remove_data(self, key, propagate=True):
        """ Remove a data

        Any associated ref reated to this data only is removed too (useless)

        """
        if key in self._dref.keys():
            self.remove_ref(key)
        else:
            assert key in self._ddata['dict'].keys()
            if propagate:
                # Check if associated ref shall be removed too
                lref = self._ddata['dict'][key]['refs']
                for kref in lref:
                    # Remove if key was the only associated data
                    if self._dref['dict'][kref]['ldata'] == [key]:
                        self.remove_ref(kref)
            del self._ddata['dict'][key]
            self._ddata['lkey'].remove(key)
        self._complement_dgrouprefdata()

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
        if sys.version[0] == '2':
            cls.strip.__func__.__doc__ = doc
        else:
            cls.strip.__doc__ = doc

    def strip(self, strip=0, verb=True):
        # super()
        super(DataHolder, self).strip(strip=strip, verb=verb)

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
        self._complement_dgrouprefdata()

    ###########
    # properties
    ###########

    @property
    def dconfig(self):
        """ The dict of configs """
        return self._dconfig

    @property
    def dgroup(self):
        """ The dict of groups """
        return self._dgroup['dict']

    @property
    def lgroup(self):
        """ The dict of groups """
        return np.array(self._dgroup['lkey'])

    @property
    def dref(self):
        """ the dict of references """
        return self._dref['dict']

    @property
    def lref(self):
        """ the dict of references """
        return np.array(self._dref['lkey'])

    @property
    def ddata(self):
        """ the dict of data """
        return self._ddata['dict']

    @property
    def ldata(self):
        """ the dict of data """
        return np.array(self._ddata['lkey'])

    @property
    def lparam(self):
        """ the dict of data """
        return np.array(self._ddata['lparam'])

    # ---------------------
    # Add / remove params
    # ---------------------

    def get_param(self, param=None, returnas=np.ndarray):
        """ Return the array of the chosen parameter values """
        # Check inputs and trivial cases
        if param is None:
            return
        assert param in self._ddata['lparam']
        assert returnas in [np.ndarray, dict, list]

        # Get output
        if returnas == dict:
            out = {kk: self._ddata['dict'][kk][param]
                   for kk in self._ddata['lkey']}
        else:
            out = [self._ddata['dict'][kk][param]
                   for kk in self._ddata['lkey']]
            if returnas == np.ndarray:
                try:
                    out = np.asarray(out)
                except Exception as err:
                    msg = "Could not convert %s to array !"
                    warnings.warn(msg)
        return out

    def set_param(self, param=None, values=None, ind=None, key=None):
        """ Set the value of a parameter

        values can be:
            - None
            - a unique value (int, float, bool, str, tuple) => common to all keys
            - an iterable of vlues (array, list) => one for each key

        A subset of keys can be chosen (ind, key, fed to self.select()) to set
        only the values of some key

        """

        # Check and format input
        if param is None:
            return
        assert param in self._ddata['lparam']

        # Update all keys with common value
        ltypes = [str, int, np.int, float, np.float, tuple]
        lc = [any([isinstance(values, tt) for tt in ltypes]),
              isinstance(values, list), isinstance(values, np.ndarray)]
        if not (values is None or any(lc)):
            msg = "Accepted types for values include:\n"
            msg += "    - None\n"
            msg += "    - %s: common to all\n"%str(ltypes)
            msg += "    - list, np.ndarray: key by key"
            raise Exception(msg)

        if values is None or lc[0]:
            key = self._ind_tofrom_key(ind=ind, key=key, returnas='key')
            for kk in key:
                self._ddata['dict'][kk][param] = values

        # Update relevant keys with corresponding values
        else:
            key = self._ind_tofrom_key(ind=ind, key=key, returnas='key')
            assert len(key) == len(values)
            for ii, kk in enumerate(key):
                self._ddata['dict'][kk][param] = values[ii]

    def add_param(self, param, values=None):
        """ Add a parameter, optionnally also set its value """
        assert isinstance(param, str)
        assert param not in self._ddata['lparam']
        self._ddata['lparam'].append(param)
        try:
            self.set_param(param=param, values=values)
        except Exception as err:
            self._ddata['lparam'].remove(param)
            raise err

    def remove_param(self, param=None):
        """ Remove a parameters """
        # Check and format input
        if param is None:
            return
        assert param in self._ddata['lparam']

        self._ddata['lparam'].remove(param)
        for kk in self._ddata['lkey']:
            del self._ddata['dict'][kk][param]

    # ---------------------
    # Read-only for internal use
    # ---------------------

    def select(self, log='all', returnas=int, **kwdargs):
        """ Return the indices / keys of data matching criteria

        The selection is done comparing the value of all provided parameters
        The result is a boolean indices array, optionally with the keys list
        It can include:
            - log = 'all': only the data matching all criteria
            - log = 'any': the data matching any criterion

        If log = 'raw', a dict of indices arrays is returned, showing the
        details for each criterion

        """

        # Format and check input
        assert returnas in [int, bool, str, 'key']
        assert log in ['all', 'any', 'raw']
        if log == 'raw':
            assert returnas == bool

        # Get list of relevant criteria
        lcritout = [ss for ss in kwdargs.keys()
                    if ss not in self._ddata['lparam']]
        if len(lcritout) > 0:
            msg = "The following criteria correspond to no parameters:\n"
            msg += "    - %s\n"%str(lcritout)
            msg += "  => only use known parameters (self.lparam):\n"
            msg += "    %s"%str(self._ddata['lparam'])
            raise Exception(msg)
        kwdargs = {kk: vv for kk, vv in kwdargs.items()
                   if vv is not None and kk in self._ddata['lparam']}
        lcrit = list(kwdargs)
        ncrit = len(kwdargs)

        # Prepare array of bool indices and populate
        ind = np.ones((ncrit, len(self._ddata['lkey'])), dtype=bool)
        for ii in range(ncrit):
            critval = kwdargs[lcrit[ii]]
            try:
                par = self.get_param(lcrit[ii], returnas=np.ndarray)
                ind[ii, :] = par == critval
            except Exception as err:
                ind[ii, :] = [self._ddata['dict'][kk][lcrit[ii]] == critval
                              for kk in self._ddata['lkey']]

        # Format output ind
        if log == 'all':
            ind = np.all(ind, axis=0)
        elif log == 'any':
            ind = np.any(ind, axis=0)
        else:
            ind = {lcrit[ii]: ind[ii, :] for ii in range(ncrit)}

        # Also return the list of keys if required
        if returnas == int:
            out = ind.nonzero()[0]
        elif returnas in [str, 'key']:
            out = self.ldata[ind.nonzero()[0]]
        else:
            out = ind
        return out

    def _ind_tofrom_key(self, ind=None, key=None, returnas=int):

        # Check / format input
        assert returnas in [int, bool, str, 'key']
        lc = [ind is not None, key is not None]
        assert np.sum(lc) <= 1

        # Initialize output
        out = np.zeros((len(self._ddata['lkey']),), dtype=bool)

        # Test
        if lc[0]:
            ind = np.atleast_1d(ind).ravel()
            assert ind.dtype == np.int or ind.dtype == np.bool
            out[ind] = True
            if returnas in [int, str, 'key']:
                out = out.nonzero()[0]
                if returnas in [str, 'key']:
                    out = [self._ddata['lkey'][ii] for ii in out]

        elif lc[1]:
            if isinstance(key, str):
                key = [key]
            if returnas in ['key', str]:
                out = key
            else:
                for kk in key:
                    out[self._ddata['lkey'].index(kk)] = True
                if returnas == int:
                    out = out.nonzero()[0]
        else:
            if returnas == bool:
                out[:] = True
            elif returnas == int:
                out = np.arange(0, len(self._ddata['lkey']))
            else:
                out = self._ddata['lkey']
        return out

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
                len(self._dgroup['dict'][k0]['lref']),
                len(self._dgroup['dict'][k0]['ldata']))
               for k0 in self._dgroup['lkey']]

        # -----------------------
        # Build for refs
        col1 = ['ref key', 'group', 'size', 'nb. data']
        ar1 = [(k0,
                self._dref['dict'][k0]['group'],
                self._dref['dict'][k0]['size'],
                len(self._dref['dict'][k0]['ldata']))
               for k0 in self._dref['lkey']]

        # -----------------------
        # Build for ddata
        col2 = ['data key']
        if show_core is None:
            show_core = self._show_in_summary_core
        if isinstance(show_core, str):
            show_core = [show_core]
        lkcore = ['shape', 'groups', 'refs']
        assert all([ss in self._ddata['lparam'] + lkcore for ss in show_core])
        col2 += show_core

        if show is None:
            show = self._show_in_summary
        if show == 'all':
            col2 += self._ddata['lparam']
        else:
            if isinstance(show, str):
                show = [show]
            assert all([ss in self._ddata['lparam'] for ss in show])
            col2 += show

        ar2 = []
        for k0 in self._ddata['lkey']:
            v0 = self._ddata['dict'][k0]
            lu = [k0] + [str(v0[cc]) for cc in col2[1:]]
            ar2.append(lu)

        return self._get_summary(
            [ar0, ar1, ar2], [col0, col1, col2],
            sep=sep, line=line, table_sep=table_sep,
            verb=verb, return_=return_)

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
    # Methods for plotting data
    # ---------------------

    # To be overloaded
    @abstractmethod
    def plot(self):
        pass




#############################################
#############################################
#       Child classes
#############################################
#############################################


class TimeTraceCollection(DataHolder):
    """ A generic class for handling multiple time traces """

    _forced_group = 'time'
    _dallowed_params = {'time':{'origin': (str, 'unknown'),
                                'dim':    (str, 'time'),
                                'quant':  (str, 't'),
                                'name':   (str, 't'),
                                'units':  (str, 's')}}


    def plot(self, ind=None, key=None,
             ax=None):
        pass
