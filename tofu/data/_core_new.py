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

__all__ = ['DataHolder']  # , 'Plasma0D']

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
    # Fixed (class-wise) dictionary of default properties
    _ddef = {'Id': {'include': ['Mod', 'Cls',
                                'Name', 'version']},
             'dgroup': ['lref'],
             'dref':   ['group', 'size', 'ldata'],
             'ddata':  ['refs', 'shape', 'groups', 'data'],
             'params': {'origin': (str, 'unknown'),
                        'dim':    (str, 'unknown'),
                        'quant':  (str, 'unknown'),
                        'name':   (str, 'unknown'),
                        'units':  (str, 'a.u.')}}
    _reserved_all = _ddef['dgroup'] + _ddef['dref'] + _ddef['ddata']
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
        assert isinstance(Name, str), Name
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

    def _extract_known_params(self, key, dd):
        # Extract relevant parameters
        dparams = {kk: vv for kk, vv in dd.items()
                   if kk not in self._reserved_all}

        # Add minimum default parameters if not already included
        for kk, vv in self._ddef['params'].items():
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

        cA = all([all([(isinstance(v1, dict) and 'group' not in v1.keys())
                       or not isinstance(v1, dict)
                       for v1 in v0.values()])
                  and 'group' not in v0.keys() for v0 in dref.values()])
        cB = all([isinstance(v0.get('group', None), str)
                  for v0 in dref.values()])
        if not (cA or cB):
            msg = "Provided dref must formatted either as a dict with:\n\n"
            msg += "    - keys = group, values = {ref: data}:\n"
            msg += "        {'g0':{'t0':{'data':t0, 'units':'s'},\n"
            msg += "                   't1':{'data':t1, 'units':'h'}},\n"
            msg += "         'g1':{'t2':{'data':t2, 'units':'min'}}}\n\n"
            msg += "    - keys = ref, values = {data, group}:\n"
            msg += "        {'t0':{'data':t0, 'units':'s', 'group':'g0'},\n"
            msg += "         't1':{'data':t1, 'units':'h', 'group':'g0'},\n"
            msg += "         't2':{'data':t2, 'units':'min', 'group':'g1'}"
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

        # Check cB
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
            dparams = self._extract_known_params(kk, vv)

            # Fill self._ddata
            self._ddata['dict'][kk] = dict(data=data, refs=(kk,),
                                           shape=(size,), **dparams)
            self._ddata['lkey'].append(kk)

    # ------------- DB (start)
    def __repr__(self):
        return self.__class__.__name__
    # ------------- DB (end)

    def _checkformat_ddata(self, ddata):
        c0 = isinstance(ddata, dict)
        c0 = c0 and all([isinstance(kk, str) for kk in ddata.keys()])
        if not c0:
            msg = "Provided ddata must be dict !\n"
            msg += "All its keys must be str !"
            raise Exception(msg)

        # Start check on each key
        for kk, vv in ddata.items():

            # Check value is a dict with proper keys
            c0 = isinstance(vv, dict)
            c0 = c0 and 'refs' in vv.keys() and isinstance(vv['refs'], tuple)
            c0 = c0 and 'data' in vv.keys()
            if not c0:
                msg = "ddata must contain dict with at least the keys:\n"
                msg += "    - 'refs': a tuple indicating refs dependencies\n"
                msg += "    - 'data': a 1d array containing the data"
                raise Exception(msg)

            # Check key unicity
            if kk in self._ddata['lkey']:
                msg = "key '%s' already used !\n"%kk
                msg += "  => each key must be unique !"
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
            self._ddata['dict'][k0]['group'] = grps

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
                    self._ddata[kk][pp] = None
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
            if self._ddata['dict'][kk]['refs'] == (key,):
                del self._ddata['dict'][kk]
                self._ddata['lkey'].remove(kk)
        self._complement_dgrouprefdata()

    def add_data(self, key, data=None, ref=None, **kwdargs):
        """ Add a data (all associated ref must be added first)) """
        self._set_ddata({key: dict(data=data, ref=ref, **kwdargs)})

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
            self._lkdata.remove(key)
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
        return self._dgroup['lkey']

    @property
    def dref(self):
        """ the dict of references """
        return self._dref['dict']

    @property
    def lref(self):
        """ the dict of references """
        return self._dref['lkey']

    @property
    def ddata(self):
        """ the dict of data """
        return self._ddata['dict']

    @property
    def ldata(self):
        """ the dict of data """
        return self._ddata['lkey']

    @property
    def lparam(self):
        """ the dict of data """
        return self._ddata['lparam']

    # ---------------------
    # Add / remove params
    # ---------------------

    # UP TO HERE

    def get_param(self, param=None, returnas=np.ndarray):

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

        # Check and format input
        if param is None:
            return
        assert param in self._ddata['lparam']

        # Update all keys with common value
        ltypes = [str, int, np.int, float, np.float, tuple]
        lc = [any([isinstance(values, tt) for tt in ltypes]),
              isinstance(values, list), isinstance(values, np.ndarray)]
        if not any(lc):
            msg = "Accepted types for values include:\n"
            msg += "    - %s: common to all\n"%str(ltypes)
            msg += "    - list, np.ndarray: key by key"
            raise Exception(msg)

        if lc0:
            key = self._ind_tofrom_key(ind=ind, key=key, out='key')
            for kk in key:
                self._ddata['dict'][kk][param] = values

        # Update relevant keys with corresponding values
        else:
            key = self._ind_tofrom_key(ind=ind, key=key, out='key')
            assert len(key) == len(values)
            for kk in range(len(key)):
                self._ddata['dict'][key[ii]][param] = values[ii]

    def add_param(self, param, values=None):
        assert isinstance(param, str)
        assert param not in self._ddata['lparam']
        self._ddata['lparam'].append(param)
        self.set_param(param=param, values=values)

    def remove_param(self, param=None):
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

    def select(self, group=None, ref=None, log='all', return_key=True,
               **kwdargs):
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
        assert log in ['all', 'any', 'raw']
        if log == 'raw':
            assert not return_key

        # Get list of relevant criteria
        lk = ['group', 'ref'] + list(kwdargs.keys())
        lcrit = [ss for ss in lk if ss is not None]
        ncrit = len(lcrit)

        # Prepare array of bool indices and populate
        ind = np.ones((ncrit, len(self._ddata['lkey'])), dtype=bool)
        for ii in range(ncrit):
            critval = eval(lcrit[ii])
            try:
                par = self.get_param(lcrit[ii], returnas=np.ndarray)
                ind[ii, :] = par == critval
            except Exception as err:
                ind[ii, :] = [self._ddata['dict'][kk][param] == critval
                              for kk in self.__lkata]

        # Format output ind
        if log == 'all':
            ind = np.all(ind, axis=0)
        elif log == 'any':
            ind = np.any(ind, axis=0)
        else:
            ind = {lcrit[ii]: ind[ii, :] for ii in range(ncrit)}

        # Also return the list of keys if required
        if return_key:
            if np.any(ind):
                out = ind, lid[ind.nonzero()[0]]
            else:
                out = ind, np.array([], dtype=int)
        else:
            out = ind
        return out

    def _ind_tofrom_key(self, ind=None, key=None, returnas=int):

        # Check / format input
        assert returnas in [int, bool, 'key']
        lc = [ind is not None, key is not None]
        assert np.sum(lc) <= 1

        # Initialize output
        out = np.zeros((len(self._ddata['lkey']),), dtype=bool)

        # Test
        if lc[0]:
            ind = np.atleast_1d(ind).ravel()
            assert ind.dtype == np.int or ind.dtype == np.bool
            out[ind] = True
            if returnas in [int, 'key']:
                out = out.nonzero()[0]
                if returnas == 'key':
                    out = [self._ddata['lkey'][ii] for ii in out]

        elif lc[1]:
            if isinstance(key, str):
                key = [key]
            if returnas == 'key':
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
               for k0, v0 in self._dref['lkey']]

        # -----------------------
        # Build for ddata
        col2 = ['data key']
        if show_core is None:
            show_core = self._show_in_summary_core
        if isinstance(show_core, str):
            show_core = [show_core]
        lkcore = ['shape', 'group', 'ref']
        assert all([ss in self._lparams + lkcore for ss in show_core])
        col2 += show_core

        if show is None:
            show = self._show_in_summary
        if show == 'all':
            col2 += self._lparams
        else:
            if isinstance(show, str):
                show = [show]
            assert all([ss in self._lparams for ss in show])
            col2 += show

        ar2 = []
        for k0 in self._lkdata:
            v0 = self._ddata[k0]
            lu = [k0] + [str(v0[cc]) for cc in col2[1:]]
            ar2.append(lu)

        return self._get_summary(
            [ar0, ar1, ar2], [col0, col1, col2],
            sep=sep, line=line, table_sep=table_sep,
            verb=verb, return_=return_)

    # ---------------------
    # Method for interpolating on ref
    # ---------------------

    def get_time_common(self, lkeys, choose=None):
        """ Return the common time vector to several quantities

        If they do not have a common time vector, a reference one is choosen
        according to criterion choose
        """
        # Check all data have time-dependency
        dout = {kk: {'t': self.get_time(kk)} for kk in lkeys}
        dtu = dict.fromkeys(set([vv['t'] for vv in dout.values()]))
        for kt in dtu.keys():
            dtu[kt] = {'ldata': [kk for kk in lkeys if dout[kk]['t'] == kt]}
        if len(dtu) == 1:
            tref = list(dtu.keys())[0]
        else:
            lt, lres = zip(*[(kt, np.mean(np.diff(self._ddata[kt]['data'])))
                             for kt in dtu.keys()])
            if choose is None:
                choose = 'min'
            if choose == 'min':
                tref = lt[np.argmin(lres)]
        return dout, dtu, tref

    @staticmethod
    def _get_time_common_arrays(dins, choose=None):
        dout = dict.fromkeys(dins.keys())
        dtu = {}
        for k, v in dins.items():
            c0 = type(k) is str
            c0 = c0 and all([ss in v.keys() for ss in ['val', 't']])
            c0 = c0 and all([type(v[ss]) is np.ndarray for ss in ['val', 't']])
            c0 = c0 and v['t'].size in v['val'].shape
            if not c0:
                msg = "dins must be a dict of the form (at least):\n"
                msg += "    dins[%s] = {'val': np.ndarray,\n"%str(k)
                msg += "                't':   np.ndarray}\n"
                msg += "Provided: %s"%str(dins)
                raise Exception(msg)

            kt, already = id(v['t']), True
            if kt not in dtu.keys():
                lisclose = [kk for kk, vv in dtu.items()
                            if (vv['val'].shape == v['t'].shape
                                and np.allclose(vv['val'], v['t']))]
                assert len(lisclose) <= 1
                if len(lisclose) == 1:
                    kt = lisclose[0]
                else:
                    already = False
                    dtu[kt] = {'val': np.atleast_1d(v['t']).ravel(),
                               'ldata': [k]}
            if already:
                dtu[kt]['ldata'].append(k)
            assert dtu[kt]['val'].size == v['val'].shape[0]
            dout[k] = {'val': v['val'], 't': kt}

        if len(dtu) == 1:
            tref = list(dtu.keys())[0]
        else:
            lt, lres = zip(*[(kt, np.mean(np.diff(dtu[kt]['val'])))
                             for kt in dtu.keys()])
            if choose is None:
                choose = 'min'
            if choose == 'min':
                tref = lt[np.argmin(lres)]
        return dout, dtu, tref

    def _interp_on_common_time(self, lkeys,
                               choose='min', interp_t=None, t=None,
                               fill_value=np.nan):
        """ Return a dict of time-interpolated data """
        dout, dtu, tref = self.get_time_common(lkeys)
        if type(t) is np.ndarray:
            tref = np.atleast_1d(t).ravel()
            tr = tref
            ltu = dtu.keys()
        else:
            if type(t) is str:
                tref = t
            tr = self._ddata[tref]['data']
            ltu = set(dtu.keys())
            if tref in dtu.keys():
                ltu = ltu.difference([tref])

        if interp_t is None:
            interp_t = _INTERPT

        # Interpolate
        for tt in ltu:
            for kk in dtu[tt]['ldata']:
                dout[kk]['val'] = scpinterp.interp1d(self._ddata[tt]['data'],
                                                     self._ddata[kk]['data'],
                                                     kind=interp_t, axis=0,
                                                     bounds_error=False,
                                                     fill_value=fill_value)(tr)

        if type(tref) is not np.ndarray and tref in dtu.keys():
            for kk in dtu[tref]['ldata']:
                dout[kk]['val'] = self._ddata[kk]['data']

        return dout, tref

    def _interp_on_common_time_arrays(self, dins,
                                      choose='min', interp_t=None, t=None,
                                      fill_value=np.nan):
        """ Return a dict of time-interpolated data """
        dout, dtu, tref = self._get_time_common_arrays(dins)
        if type(t) is np.ndarray:
            tref = np.atleast_1d(t).ravel()
            tr = tref
            ltu = dtu.keys()
        else:
            if type(t) is str:
                assert t in dout.keys()
                tref = dout[t]['t']
            tr = dtu[tref]['val']
            ltu = set(dtu.keys()).difference([tref])

        if interp_t is None:
            interp_t = _INTERPT

        # Interpolate
        for tt in ltu:
            for kk in dtu[tt]['ldata']:
                dout[kk]['val'] = scpinterp.interp1d(dtu[tt]['val'],
                                                     dout[kk]['val'],
                                                     kind=interp_t, axis=0,
                                                     bounds_error=False,
                                                     fill_value=fill_value)(tr)
        return dout, tref

    def interp_t(self, dkeys,
                 choose='min', interp_t=None, t=None,
                 fill_value=np.nan):
        # Check inputs
        assert type(dkeys) in [list, dict]
        if type(dkeys) is list:
            dkeys = {kk: {'val': kk} for kk in dkeys}
        lc = [(type(kk) is str
               and type(vv) is dict
               and type(vv.get('val', None)) in [str, np.ndarray])
              for kk, vv in dkeys.items()]
        assert all(lc), str(dkeys)

        # Separate by type
        dk0 = dict([(kk, vv) for kk, vv in dkeys.items()
                    if type(vv['val']) is str])
        dk1 = dict([(kk, vv) for kk, vv in dkeys.items()
                    if type(vv['val']) is np.ndarray])
        assert len(dkeys) == len(dk0) + len(dk1), str(dk0) + '\n' + str(dk1)

        if len(dk0) == len(dkeys):
            lk = [v['val'] for v in dk0.values()]
            dout, tref = self._interp_on_common_time(lk, choose=choose,
                                                     t=t, interp_t=interp_t,
                                                     fill_value=fill_value)
            dout = {kk: {'val': dout[vv['val']]['val'],
                         't': dout[vv['val']]['t']}
                    for kk, vv in dk0.items()}
        elif len(dk1) == len(dkeys):
            dout, tref = self._interp_on_common_time_arrays(
                dk1, choose=choose, t=t,
                interp_t=interp_t, fill_value=fill_value)

        else:
            lk = [v['val'] for v in dk0.values()]
            if type(t) is np.ndarray:
                dout, tref = self._interp_on_common_time(
                    lk, choose=choose, t=t,
                    interp_t=interp_t, fill_value=fill_value)
                dout1, _ = self._interp_on_common_time_arrays(
                    dk1, choose=choose, t=t,
                    interp_t=interp_t, fill_value=fill_value)
            else:
                dout0, dtu0, tref0 = self.get_time_common(lk,
                                                          choose=choose)
                dout1, dtu1, tref1 = self._get_time_common_arrays(
                    dk1, choose=choose)
                if type(t) is str:
                    lc = [t in dtu0.keys(), t in dout1.keys()]
                    if not any(lc):
                        msg = "if t is str, it must refer to a valid key:\n"
                        msg += "    - %s\n"%str(dtu0.keys())
                        msg += "    - %s\n"%str(dout1.keys())
                        msg += "Provided: %s"%t
                        raise Exception(msg)
                    if lc[0]:
                        t0, t1 = t, self._ddata[t]['data']
                    else:
                        t0, t1 = dtu1[dout1[t]['t']]['val'], t
                    tref = t
                else:
                    if choose is None:
                        choose = 'min'
                    if choose == 'min':
                        t0 = self._ddata[tref0]['data']
                        t1 = dtu1[tref1]['val']
                        dt0 = np.mean(np.diff(t0))
                        dt1 = np.mean(np.diff(t1))
                        if dt0 < dt1:
                            t0, t1, tref = tref0, t0, tref0
                        else:
                            t0, t1, tref = t1, tref1, tref1

                dout, tref = self._interp_on_common_time(
                    lk, choose=choose, t=t0,
                    interp_t=interp_t, fill_value=fill_value)

                dout = {kk: {'val': dout[vv['val']]['val'],
                             't': dout[vv['val']]['t']}
                        for kk, vv in dk0.items()}

                dout1, _ = self._interp_on_common_time_arrays(
                    dk1, choose=choose, t=t1,
                    interp_t=interp_t, fill_value=fill_value)

            dout.update(dout1)

        return dout, tref

    def _get_indtmult(self, idquant=None, idref1d=None, idref2d=None):

        # Get time vectors and bins
        idtq = self._ddata[idquant]['depend'][0]
        tq = self._ddata[idtq]['data']
        tbinq = 0.5*(tq[1:]+tq[:-1])
        if idref1d is not None:
            idtr1 = self._ddata[idref1d]['depend'][0]
            tr1 = self._ddata[idtr1]['data']
            tbinr1 = 0.5*(tr1[1:]+tr1[:-1])
        if idref2d is not None and idref2d != idref1d:
            idtr2 = self._ddata[idref2d]['depend'][0]
            tr2 = self._ddata[idtr2]['data']
            tbinr2 = 0.5*(tr2[1:]+tr2[:-1])

        # Get tbinall and tall
        if idref1d is None:
            tbinall = tbinq
            tall = tq
        else:
            if idref2d is None:
                tbinall = np.unique(np.r_[tbinq, tbinr1])
            else:
                tbinall = np.unique(np.r_[tbinq, tbinr1, tbinr2])
            tall = np.r_[tbinall[0] - 0.5*(tbinall[1]-tbinall[0]),
                         0.5*(tbinall[1:]+tbinall[:-1]),
                         tbinall[-1] + 0.5*(tbinall[-1]-tbinall[-2])]

        # Get indtqr1r2 (tall with respect to tq, tr1, tr2)
        indtq, indtr1, indtr2 = None, None, None
        indtq = np.digitize(tall, tbinq)
        if idref1d is None:
            assert np.all(indtq == np.arange(0, tall.size))
        if idref1d is not None:
            indtr1 = np.digitize(tall, tbinr1)
        if idref2d is not None:
            indtr2 = np.digitize(tall, tbinr2)

        ntall = tall.size
        return tall, tbinall, ntall, indtq, indtr1, indtr2

    @staticmethod
    def _get_indtu(t=None, tall=None, tbinall=None,
                   idref1d=None, idref2d=None,
                   indtr1=None, indtr2=None):
        # Get indt (t with respect to tbinall)
        indt, indtu = None, None
        if t is not None:
            indt = np.digitize(t, tbinall)
            indtu = np.unique(indt)

            # Update
            tall = tall[indtu]
            if idref1d is not None:
                assert indtr1 is not None
                indtr1 = indtr1[indtu]
            if idref2d is not None:
                assert indtr2 is not None
                indtr2 = indtr2[indtu]
        ntall = tall.size
        return tall, ntall, indt, indtu, indtr1, indtr2

    def get_tcommon(self, lq, prefer='finer'):
        """ Check if common t, else choose according to prefer

        By default, prefer the finer time resolution

        """
        if type(lq) is str:
            lq = [lq]
        t = []
        for qq in lq:
            ltr = [kk for kk in self._ddata[qq]['depend']
                   if self._dindref[kk]['group'] == 'time']
            assert len(ltr) <= 1
            if len(ltr) > 0 and ltr[0] not in t:
                t.append(ltr[0])
        assert len(t) >= 1
        if len(t) > 1:
            dt = [np.nanmean(np.diff(self._ddata[tt]['data'])) for tt in t]
            if prefer == 'finer':
                ind = np.argmin(dt)
            else:
                ind = np.argmax(dt)
        else:
            ind = 0
        return t[ind], t

    def _get_tcom(self, idquant=None, idref1d=None,
                  idref2d=None, idq2dR=None):
        if idquant is not None:
            out = self._get_indtmult(idquant=idquant,
                                     idref1d=idref1d, idref2d=idref2d)
        else:
            out = self._get_indtmult(idquant=idq2dR)
        return out

    # ---------------------
    # Methods for plotting data
    # ---------------------

    def plot(self, lquant, X=None,
             ref1d=None, ref2d=None,
             remap=False, res=0.01, interp_space=None,
             sharex=False, bck=True):
        lDat = self.get_Data(lquant, X=X, remap=remap,
                             ref1d=ref1d, ref2d=ref2d,
                             res=res, interp_space=interp_space)
        if type(lDat) is list:
            kh = lDat[0].plot_combine(lDat[1:], sharex=sharex, bck=bck)
        else:
            kh = lDat.plot(bck=bck)
        return kh

    def plot_combine(self, lquant, lData=None, X=None,
                     ref1d=None, ref2d=None,
                     remap=False, res=0.01, interp_space=None,
                     sharex=False, bck=True):
        """ plot combining several quantities from the Plasma2D itself and
        optional extra list of Data instances """
        lDat = self.get_Data(lquant, X=X, remap=remap,
                             ref1d=ref1d, ref2d=ref2d,
                             res=res, interp_space=interp_space)
        if lData is not None:
            if type(lDat) is list:
                lData = lDat[1:] + lData
            else:
                lData = lDat[1:] + [lData]
        kh = lDat[0].plot_combine(lData, sharex=sharex, bck=bck)
        return kh
