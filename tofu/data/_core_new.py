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

__all__ = ['DataHolder']#, 'Plasma0D']

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
    _ddef = {'Id':{'include':['Mod', 'Cls',
                              'Name', 'version']},
             'dgroup':['lref'],
             'dref':  ['group', 'size', 'ldata'],
             'ddata': ['refs', 'shape', 'groups', 'data'],
             'params':{'origin':(str, 'unknown'),
                       'dim':   (str, 'unknown'),
                       'quant': (str, 'unknown'),
                       'name':  (str, 'unknown'),
                       'units': (str, 'a.u.')}}
    _reserved_all = _ddef['dgroup'] + _ddef['dref'] + _ddef['ddata']
    _show_in_summary = 'all'


    # Does not exist before Python 3.6 !!!
    def __init_subclass__(cls, **kwdargs):
        # Python 2
        super(DataHolder, cls).__init_subclass__(**kwdargs)
        # Python 3
        #super().__init_subclass__(**kwdargs)
        cls._ddef = copy.deepcopy(DataHolder._ddef)
        #cls._dplot = copy.deepcopy(Struct._dplot)
        #cls._set_color_ddef(cls._color)


    def __init__(self, dref=None, ddata=None,
                 Id=None, Name=None,
                 fromdict=None, SavePath=os.path.abspath('./'),
                 SavePath_Include=tfpf.defInclude):

        # To replace __init_subclass__ for Python 2
        if sys.version[0] == '2':
            self._dstrip = utils.ToFuObjectBase._dstrip.copy()
            self.__class__._strip_init()

        # Create a dplot at instance level
        #self._dplot = copy.deepcopy(self.__class__._dplot)

        kwdargs = locals()
        del kwdargs['self']
        # super()
        super(DataHolder, self).__init__(**kwdargs)

    def _reset(self):
        # Run by the parent class __init__()
        # super()
        super(DataHolder, self)._reset()
        self._dgroup = {kd[0]:kd[1] for kd in self._get_keys_dgroup()}
        self._dref = {kd[0]:kd[1] for kd in self._get_keys_dref()}
        self._ddata = {kd[0]:kd[1] for kd in self._get_keys_ddata()}

    @classmethod
    def _checkformat_inputs_Id(cls, Id=None, Name=None,
                               include=None, **kwdargs):
        if Id is not None:
            assert isinstance(Id, utils.ID)
            Name = Id.Name
        assert isinstance(Name, str), Name
        if include is None:
            include = cls._ddef['Id']['include']
        kwdargs.update({'Name':Name, 'include':include})
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

    #---------------------
    # Methods for checking and formatting inputs
    #---------------------

    def _extract_known_params(self, key, dd):
        # Extract relevant parameters
        dparams = {kk:vv for kk, vv in dd.items()
                   if kk not in self._reserved_all}

        # Add minimum default parameters if not already included
        for kk, vv in self._ddef['params'].items():
            if kk not in dparams.keys():
                dparams[kk] = vv[1]
            else:
                # Check type if already included
                if not isinstance(dparams[kk], vv[0]):
                    msg = "A parameter for %s has the wrong type:\n"%key
                    msg += "    - Provided: type(%s) = %s\n"%(kk, str(type(vv)))
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
        cB = all([isinstance(v0.get('group', None), str) for v0 in dref.values()])
        if not (cA or cB):
            msg = "Provided dref must formatted either as:\n\n"
            msg += "    - a dict of group keys with a dict of key ref:\n"
            msg += "        {'group0':{'t0':{'data':t0, 'units':'s'},\n"
            msg += "                   't1':{'data':t1, 'units':'h'}},\n"
            msg += "         'group1':{'t2':{'data':t2, 'units':'min'}}}\n\n"
            msg += "    - a dict of key ref with a dict containing the group:\n"
            msg += "        {'t0':{'data':t0, 'units':'s', 'group':'group0'},\n"
            msg += "         't1':{'data':t1, 'units':'h', 'group':'group0'},\n"
            msg += "         't2':{'data':t2, 'units':'min', 'group':'group1'}"
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
                        drbis[k1] = {'data':v1, 'group':k0}
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
                    except:
                        c0 = False
                else:
                    size = data.__class__.__name__
            else:
                if data.ndim != 1:
                    data = np.atleast_1d(data).ravel()
                size = data.size

            if not c0:
                msg = "Each dict in dref must hold an array-convertibe 'data'\n"
                msg += "The following array conversion failed:\n"
                msg += "    - np.atleast_1d(dref[%s]['data']).ravel()"%kk
                raise Exception(msg)

            # Fill self._dref
            self._dref['dict'][kk] = {'size':size, 'group':vv['group']}
            self._dref['lkey'].append(kk)

            # Extract and check parameters
            dparams = self._extract_known_params(kk, vv)

            # Fill self._ddata
            self._ddata['dict'][kk] = {'data':data, 'refs':(kk,),
                                       'shape':(size,), **dparams}
            self._ddata['lkey'].append(kk)

    # ------------- DB (start)
    def __repr__(self):
        return self.__class__.__name__
    # ------------- DB (end)

    def _checkformat_ddata(self, ddata):
        c0 = isinstance(ddata, dict)
        c0 = c0 and  all([isinstance(kk, str) for kk in ddata.keys()])
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
                msg += "    - 'ref': a str indicating the ref(s) dependencies\n"
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
                    except:
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
            shaperef = tuple(self._dref['dict'][rr]['size'] for rr in vv['refs'])
            if not shape == shaperef:
                msg = "Inconsistency between data shape and ref size !\n"
                msg += "    - ddata[%s]['data'] shape: %s\n"%(kk, str(shape))
                msg += "    - sizes of refs: %s"%(str(shaperef))
                raise Exception(msg)

            # Extract params and set self._ddata
            dparams = self._extract_known_params(kk, vv)
            self._ddata['dict'][kk] = {'data':data, 'refs':vv['refs'],
                                       'shape':shape, **dparams}
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
            groups = tuple(self._dref['dict'][rr]['group'] for rr in v0['refs'])
            gout = [gg for gg in groups if gg not in self._dgroup['lkey']]
            if len(gout) > 0:
                lg = self._dgroup['lkey']
                msg = "Inconsistent groups from self.ddata[%s]['refs']:\n"%k0
                msg += "    - groups = %s\n"%str(groups)
                msg += "    - self._dgroup['lkey'] = %s\n"%str(lg)
                msg += "    - self.dgroup.keys() = %s"%str(self.dgroup.keys())
                raise Exception(msg)
            self._ddata['dict'][k0]['group'] = groups

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
            #assert vg['depend'] in lidindref
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
        kwdargs = {'dref':dref, 'ddata':ddata, **kwargs}
        largs = self._get_largs_dref()
        kwddref = self._extract_kwdargs(kwdargs, largs)
        self._set_dref(**kwddref, complement=False)
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

    #---------------------
    # Methods for adding ref / quantities
    #---------------------

    def add_ref(self, key, data=None, group=None, **kwdargs):
        """ Add a reference """
        self._set_dref({key:{'data':data, 'group':group, **kwdargs}})

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
        self._set_ddata({key: {'data':data, 'ref':ref, **kwdargs}})

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
        cls._dstrip['allowed'] = [0,1]
        nMax = max(cls._dstrip['allowed'])
        doc = """
                 1: None
                 """
        doc = utils.ToFuObjectBase.strip.__doc__.format(doc,nMax)
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
        dout = {'dgroup':{'dict':self._dgroup, 'lexcept':None},
                'dref':{'dict':self._dref, 'lexcept':None},
                'ddata':{'dict':self._ddata, 'lexcept':None}}
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

    #---------------------
    # Add / remove params
    #---------------------

    # UP TO HERE


    def get_param(self, param=None, returnas=np.ndarray):

        # Check inputs and trivial cases
        if param is None:
            return
        assert param in self._ddata['lparam']
        assert returnas in [np.ndarray, dict, list]

        # Get output
        if returnas == dict:
            out = {kk:self._ddata['dict'][kk][param]
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


    #---------------------
    # Read-only for internal use
    #---------------------

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
                ind[ii,:] = par == critval
            except:
                ind[ii,:] = [self._ddata['dict'][kk][param] == critval
                             for kk in self.__lkata]

        # Format output ind
        if log == 'all':
            ind = np.all(ind, axis=0)
        elif log == 'any':
            ind = np.any(ind, axis=0)
        else:
            ind = {lcrit[ii]: ind[ii,:] for ii in range(ncrit)}

        # Also return the list of keys if required
        if return_key:
            if np.any(ind):
                out = ind, lid[ind.nonzero()[0]]
            else:
                out = ind, np.array([],dtype=int)
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


    #---------------------
    # Methods for showing data
    #---------------------

    def get_summary(self, show=None, show_core=None, sep='  ', line='-', just='l',
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
               for k0,v0 in self._dref['lkey']]

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

        return self._get_summary([ar0,ar1,ar2], [col0, col1, col2],
                                  sep=sep, line=line, table_sep=table_sep,
                                  verb=verb, return_=return_)



    #---------------------
    # Method for interpolating on ref
    #---------------------


    def get_time_common(self, lkeys, choose=None):
        """ Return the common time vector to several quantities

        If they do not have a common time vector, a reference one is choosen
        according to criterion choose
        """
        # Check all data have time-dependency
        dout = {kk: {'t':self.get_time(kk)} for kk in lkeys}
        dtu = dict.fromkeys(set([vv['t'] for vv in dout.values()]))
        for kt in dtu.keys():
            dtu[kt] = {'ldata':[kk for kk in lkeys if dout[kk]['t'] == kt]}
        if len(dtu) == 1:
            tref = list(dtu.keys())[0]
        else:
            lt, lres = zip(*[(kt,np.mean(np.diff(self._ddata[kt]['data'])))
                             for kt in dtu.keys()])
            if choose is None:
                choose  = 'min'
            if choose == 'min':
                tref = lt[np.argmin(lres)]
        return dout, dtu, tref

    @staticmethod
    def _get_time_common_arrays(dins, choose=None):
        dout = dict.fromkeys(dins.keys())
        dtu = {}
        for k, v in dins.items():
            c0 = type(k) is str
            c0 = c0 and all([ss in v.keys() for ss in ['val','t']])
            c0 = c0 and all([type(v[ss]) is np.ndarray for ss in ['val','t']])
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
                                and np.allclose(vv['val'],v['t']))]
                assert len(lisclose) <= 1
                if len(lisclose) == 1:
                    kt = lisclose[0]
                else:
                    already = False
                    dtu[kt] = {'val':np.atleast_1d(v['t']).ravel(),
                               'ldata':[k]}
            if already:
                dtu[kt]['ldata'].append(k)
            assert dtu[kt]['val'].size == v['val'].shape[0]
            dout[k] = {'val':v['val'], 't':kt}

        if len(dtu) == 1:
            tref = list(dtu.keys())[0]
        else:
            lt, lres = zip(*[(kt,np.mean(np.diff(dtu[kt]['val'])))
                             for kt in dtu.keys()])
            if choose is None:
                choose  = 'min'
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
        assert type(dkeys) in [list,dict]
        if type(dkeys) is list:
            dkeys = {kk:{'val':kk} for kk in dkeys}
        lc = [(type(kk) is str
               and type(vv) is dict
               and type(vv.get('val',None)) in [str,np.ndarray])
              for kk,vv in dkeys.items()]
        assert all(lc), str(dkeys)

        # Separate by type
        dk0 = dict([(kk,vv) for kk,vv in dkeys.items()
                    if type(vv['val']) is str])
        dk1 = dict([(kk,vv) for kk,vv in dkeys.items()
                    if type(vv['val']) is np.ndarray])
        assert len(dkeys) == len(dk0) + len(dk1), str(dk0) + '\n' + str(dk1)


        if len(dk0) == len(dkeys):
            lk = [v['val'] for v in dk0.values()]
            dout, tref = self._interp_on_common_time(lk, choose=choose,
                                                     t=t, interp_t=interp_t,
                                                     fill_value=fill_value)
            dout = {kk:{'val':dout[vv['val']]['val'], 't':dout[vv['val']]['t']}
                    for kk,vv in dk0.items()}
        elif len(dk1) == len(dkeys):
            dout, tref = self._interp_on_common_time_arrays(dk1, choose=choose,
                                                            t=t, interp_t=interp_t,
                                                            fill_value=fill_value)

        else:
            lk = [v['val'] for v in dk0.values()]
            if type(t) is np.ndarray:
                dout, tref =  self._interp_on_common_time(lk, choose=choose,
                                                       t=t, interp_t=interp_t,
                                                       fill_value=fill_value)
                dout1, _   = self._interp_on_common_time_arrays(dk1, choose=choose,
                                                              t=t, interp_t=interp_t,
                                                              fill_value=fill_value)
            else:
                dout0, dtu0, tref0 = self.get_time_common(lk,
                                                          choose=choose)
                dout1, dtu1, tref1 = self._get_time_common_arrays(dk1,
                                                                  choose=choose)
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

                dout, tref =  self._interp_on_common_time(lk, choose=choose,
                                                          t=t0, interp_t=interp_t,
                                                          fill_value=fill_value)
                dout = {kk:{'val':dout[vv['val']]['val'],
                            't':dout[vv['val']]['t']}
                        for kk,vv in dk0.items()}
                dout1, _   = self._interp_on_common_time_arrays(dk1, choose=choose,
                                                                t=t1, interp_t=interp_t,
                                                                fill_value=fill_value)
            dout.update(dout1)

        return dout, tref

    #---------------------
    # Methods for computing additional plasma quantities
    #---------------------


    def _fill_dins(self, dins):
        for k in dins.keys():
            if type(dins[k]['val']) is str:
                assert dins[k]['val'] in self._ddata.keys()
            else:
                dins[k]['val'] = np.atleast_1d(dins[k]['val'])
                assert dins[k]['t'] is not None
                dins[k]['t'] = np.atleast_1d(dins[k]['t']).ravel()
                assert dins[k]['t'].size == dins[k]['val'].shape[0]
        return dins

    @staticmethod
    def _checkformat_shapes(dins):
        shape = None
        for k in dins.keys():
            dins[k]['shape'] = dins[k]['val'].shape
            if shape is None:
                shape = dins[k]['shape']
            if dins[k]['shape'] != shape:
                if dins[k]['val'].ndim > len(shape):
                    shape = dins[k]['shape']

        # Check shape consistency for broadcasting
        assert len(shape) in [1,2]
        if len(shape) == 1:
            for k in dins.keys():
                assert dins[k]['shape'][0] in [1,shape[0]]
                if dins[k]['shape'][0] < shape[0]:
                    dins[k]['val'] = np.full((shape[0],), dins[k]['val'][0])
                    dins[k]['shape'] = dins[k]['val'].shape

        elif len(shape) == 2:
            for k in dins.keys():
                if len(dins[k]['shape']) == 1:
                    if dins[k]['shape'][0] not in [1]+list(shape):
                        msg = "Non-conform shape for dins[%s]:\n"%k
                        msg += "    - Expected: (%s,...) or (1,)\n"%str(shape[0])
                        msg += "    - Provided: %s"%str(dins[k]['shape'])
                        raise Exception(msg)
                    if dins[k]['shape'][0] == 1:
                        dins[k]['val'] = dins[k]['val'][None,:]
                    elif dins[k]['shape'][0] == shape[0]:
                        dins[k]['val'] = dins[k]['val'][:,None]
                    else:
                        dins[k]['val'] = dins[k]['val'][None,:]
                else:
                    assert dins[k]['shape'] == shape
                dins[k]['shape'] = dins[k]['val'].shape
        return dins



    def compute_bremzeff(self, Te=None, ne=None, zeff=None, lamb=None,
                         tTe=None, tne=None, tzeff=None, t=None,
                         interp_t=None):
        """ Return the bremsstrahlung spectral radiance at lamb

        The plasma conditions are set by:
            - Te   (eV)
            - ne   (/m3)
            - zeff (adim.)

        The wavelength is set by the diagnostics
            - lamb (m)

        The vol. spectral emis. is returned in ph / (s.m3.sr.m)

        The computation requires an intermediate : gff(Te, zeff)
        """
        dins = {'Te':{'val':Te, 't':tTe},
                'ne':{'val':ne, 't':tne},
                'zeff':{'val':zeff, 't':tzeff}}
        lc = [vv['val'] is None for vv in dins.values()]
        if any(lc):
            msg = "All fields should be provided:\n"
            msg += "    - %s"%str(dins.keys())
            raise Exception(msg)
        dins = self._fill_dins(dins)
        dins, t = self.interp_t(dins, t=t, interp_t=interp_t)
        lamb = np.atleast_1d(lamb)
        dins['lamb'] = {'val':lamb}
        dins = self._checkformat_shapes(dins)

        val, units = _physics.compute_bremzeff(dins['Te']['val'],
                                               dins['ne']['val'],
                                               dins['zeff']['val'],
                                               dins['lamb']['val'])
        return val, t, units

    def compute_fanglev(self, BR=None, BPhi=None, BZ=None,
                        ne=None, lamb=None, t=None, interp_t=None,
                        tBR=None, tBPhi=None, tBZ=None, tne=None):
        """ Return the vector faraday angle at lamb

        The plasma conditions are set by:
            - BR    (T) , array of R component of B
            - BRPhi (T) , array of phi component of B
            - BZ    (T) , array of Z component of B
            - ne    (/m3)

        The wavelength is set by the diagnostics
            - lamb (m)

        The vector faraday angle is returned in T / m
        """
        dins = {'BR':  {'val':BR,   't':tBR},
                'BPhi':{'val':BPhi, 't':tBPhi},
                'BZ':  {'val':BZ,   't':tBZ},
                'ne':  {'val':ne,   't':tne}}
        dins = self._fill_dins(dins)
        dins, t = self.interp_t(dins, t=t, interp_t=interp_t)
        lamb = np.atleast_1d(lamb)
        dins['lamb'] = {'val':lamb}
        dins = self._checkformat_shapes(dins)

        val, units = _physics.compute_fangle(BR=dins['BR']['val'],
                                             BPhi=dins['BPhi']['val'],
                                             BZ=dins['BZ']['val'],
                                             ne=dins['ne']['val'],
                                             lamb=dins['lamb']['val'])
        return val, t, units



    #---------------------
    # Methods for interpolation
    #---------------------


    def _get_quantrefkeys(self, qq, ref1d=None, ref2d=None):

        # Get relevant lists
        kq, msg = self._get_keyingroup(qq, 'mesh', msgstr='quant', raise_=False)
        if kq is not None:
            k1d, k2d = None, None
        else:
            kq, msg = self._get_keyingroup(qq, 'radius', msgstr='quant', raise_=True)
            if ref1d is None and ref2d is None:
                msg = "quant %s needs refs (1d and 2d) for interpolation\n"%qq
                msg += "  => ref1d and ref2d cannot be both None !"
                raise Exception(msg)
            if ref1d is None:
                ref1d = ref2d
            k1d, msg = self._get_keyingroup(ref1d, 'radius',
                                            msgstr='ref1d', raise_=False)
            if k1d is None:
                msg += "\n\nInterpolation of %s:\n"%qq
                msg += "  ref could not be identified among 1d quantities\n"
                msg += "    - ref1d : %s"%ref1d
                raise Exception(msg)
            if ref2d is None:
                ref2d = ref1d
            k2d, msg = self._get_keyingroup(ref2d, 'mesh',
                                            msgstr='ref2d', raise_=False)
            if k2d is None:
                msg += "\n\nInterpolation of %s:\n"
                msg += "  ref could not be identified among 2d quantities\n"
                msg += "    - ref2d: %s"%ref2d
                raise Exception(msg)

            q1d, q2d = self._ddata[k1d]['quant'], self._ddata[k2d]['quant']
            if q1d != q2d:
                msg = "ref1d and ref2d must be of the same quantity !\n"
                msg += "    - ref1d (%s):   %s\n"%(ref1d, q1d)
                msg += "    - ref2d (%s):   %s"%(ref2d, q2d)
                raise Exception(msg)

        return kq, k1d, k2d


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
                tbinall = np.unique(np.r_[tbinq,tbinr1])
            else:
                tbinall = np.unique(np.r_[tbinq,tbinr1,tbinr2])
            tall = np.r_[tbinall[0] - 0.5*(tbinall[1]-tbinall[0]),
                         0.5*(tbinall[1:]+tbinall[:-1]),
                         tbinall[-1] + 0.5*(tbinall[-1]-tbinall[-2])]

        # Get indtqr1r2 (tall with respect to tq, tr1, tr2)
        indtq, indtr1, indtr2 = None, None, None
        indtq = np.digitize(tall, tbinq)
        if idref1d is None:
            assert np.all(indtq == np.arange(0,tall.size))
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


    def _get_finterp(self,
                     idquant=None, idref1d=None, idref2d=None,
                     idq2dR=None, idq2dPhi=None, idq2dZ=None,
                     interp_t='nearest', interp_space=None,
                     fill_value=np.nan, ani=False, Type=None):

        # Get idmesh
        if idquant is not None:
            if idref1d is None:
                lidmesh = [qq for qq in self._ddata[idquant]['depend']
                           if self._dindref[qq]['group'] == 'mesh']
            else:
                lidmesh = [qq for qq in self._ddata[idref2d]['depend']
                           if self._dindref[qq]['group'] == 'mesh']
        else:
            assert idq2dR is not None
            lidmesh = [qq for qq in self._ddata[idq2dR]['depend']
                       if self._dindref[qq]['group'] == 'mesh']
        assert len(lidmesh) == 1
        idmesh = lidmesh[0]

        # Get mesh
        mpltri = self._ddata[idmesh]['data']['mpltri']
        trifind = mpltri.get_trifinder()

        # Get common time indices
        if interp_t == 'nearest':
             out = self._get_tcom(idquant,idref1d, idref2d, idq2dR)
             tall, tbinall, ntall, indtq, indtr1, indtr2= out

        # # Prepare output

        # Interpolate
        # Note : Maybe consider using scipy.LinearNDInterpolator ?
        if idquant is not None:
            vquant = self._ddata[idquant]['data']
            if self._ddata[idmesh]['data']['ntri'] > 1:
                vquant = np.repeat(vquant,
                                   self._ddata[idmesh]['data']['ntri'], axis=0)
        else:
            vq2dR   = self._ddata[idq2dR]['data']
            vq2dPhi = self._ddata[idq2dPhi]['data']
            vq2dZ   = self._ddata[idq2dZ]['data']

        if interp_space is None:
            interp_space = self._ddata[idmesh]['data']['ftype']

        # get interpolation function
        if ani:
            # Assuming same mesh and time vector for all 3 components
            func = _comp.get_finterp_ani(self, idq2dR, idq2dPhi, idq2dZ,
                                         interp_t=interp_t,
                                         interp_space=interp_space,
                                         fill_value=fill_value,
                                         idmesh=idmesh, vq2dR=vq2dR,
                                         vq2dZ=vq2dZ, vq2dPhi=vq2dPhi,
                                         tall=tall, tbinall=tbinall,
                                         ntall=ntall,
                                         indtq=indtq, trifind=trifind,
                                         Type=Type, mpltri=mpltri)
        else:
            func = _comp.get_finterp_isotropic(self, idquant, idref1d, idref2d,
                                               interp_t=interp_t,
                                               interp_space=interp_space,
                                               fill_value=fill_value,
                                               idmesh=idmesh, vquant=vquant,
                                               tall=tall, tbinall=tbinall,
                                               ntall=ntall, mpltri=mpltri,
                                               indtq=indtq, indtr1=indtr1,
                                               indtr2=indtr2, trifind=trifind)


        return func


    def _checkformat_qr12RPZ(self, quant=None, ref1d=None, ref2d=None,
                             q2dR=None, q2dPhi=None, q2dZ=None):
        lc0 = [quant is None, ref1d is None, ref2d is None]
        lc1 = [q2dR is None, q2dPhi is None, q2dZ is None]
        if np.sum([all(lc0), all(lc1)]) != 1:
            msg = "Please provide either (xor):\n"
            msg += "    - a scalar field (isotropic emissivity):\n"
            msg += "        quant : scalar quantity to interpolate\n"
            msg += "                if quant is 1d, intermediate reference\n"
            msg += "                fields are necessary for 2d interpolation\n"
            msg += "        ref1d : 1d reference field on which to interpolate\n"
            msg += "        ref2d : 2d reference field on which to interpolate\n"
            msg += "    - a vector (R,Phi,Z) field (anisotropic emissivity):\n"
            msg += "        q2dR :  R component of the vector field\n"
            msg += "        q2dPhi: R component of the vector field\n"
            msg += "        q2dZ :  Z component of the vector field\n"
            msg += "        => all components have teh same time and mesh !\n"
            raise Exception(msg)

        # Check requested quant is available in 2d or 1d
        if all(lc1):
            idquant, idref1d, idref2d = self._get_quantrefkeys(quant, ref1d, ref2d)
            idq2dR, idq2dPhi, idq2dZ = None, None, None
            ani = False
        else:
            idq2dR, msg   = self._get_keyingroup(q2dR, 'mesh', msgstr='quant',
                                              raise_=True)
            idq2dPhi, msg = self._get_keyingroup(q2dPhi, 'mesh', msgstr='quant',
                                              raise_=True)
            idq2dZ, msg   = self._get_keyingroup(q2dZ, 'mesh', msgstr='quant',
                                              raise_=True)
            idquant, idref1d, idref2d = None, None, None
            ani = True
        return idquant, idref1d, idref2d, idq2dR, idq2dPhi, idq2dZ, ani


    def get_finterp2d(self, quant=None, ref1d=None, ref2d=None,
                      q2dR=None, q2dPhi=None, q2dZ=None,
                      interp_t=None, interp_space=None,
                      fill_value=np.nan, Type=None):
        """ Return the function interpolating (X,Y,Z) pts on a 1d/2d profile

        Can be used as input for tf.geom.CamLOS1D/2D.calc_signal()

        """
        # Check inputs
        msg = "Only 'nearest' available so far for interp_t!"
        assert interp_t == 'nearest', msg
        out = self._checkformat_qr12RPZ(quant=quant, ref1d=ref1d, ref2d=ref2d,
                                        q2dR=q2dR, q2dPhi=q2dPhi, q2dZ=q2dZ)
        idquant, idref1d, idref2d, idq2dR, idq2dPhi, idq2dZ, ani = out


        # Interpolation (including time broadcasting)
        func = self._get_finterp(idquant=idquant, idref1d=idref1d,
                                 idref2d=idref2d, idq2dR=idq2dR,
                                 idq2dPhi=idq2dPhi, idq2dZ=idq2dZ,
                                 interp_t=interp_t, interp_space=interp_space,
                                 fill_value=fill_value, ani=ani, Type=Type)
        return func


    def interp_pts2profile(self, pts=None, vect=None, t=None,
                           quant=None, ref1d=None, ref2d=None,
                           q2dR=None, q2dPhi=None, q2dZ=None,
                           interp_t=None, interp_space=None,
                           fill_value=np.nan, Type=None):
        """ Return the value of the desired profiles_1d quantity

        For the desired inputs points (pts):
            - pts are in (R,Z) coordinates
            - space interpolation is linear on the 1d profiles
        At the desired input times (t):
            - using a nearest-neighbourg approach for time

        """
        # Check inputs
        # msg = "Only 'nearest' available so far for interp_t!"
        # assert interp_t == 'nearest', msg

        # Check requested quant is available in 2d or 1d
        out = self._checkformat_qr12RPZ(quant=quant, ref1d=ref1d, ref2d=ref2d,
                                        q2dR=q2dR, q2dPhi=q2dPhi, q2dZ=q2dZ)
        idquant, idref1d, idref2d, idq2dR, idq2dPhi, idq2dZ, ani = out

        # Check the pts is (2,...) array of floats
        if pts is None:
            if ani:
                idmesh = [id_ for id_ in self._ddata[idq2dR]['depend']
                          if self._dindref[id_]['group'] == 'mesh'][0]
            else:
                if idref1d is None:
                    idmesh = [id_ for id_ in self._ddata[idquant]['depend']
                              if self._dindref[id_]['group'] == 'mesh'][0]
                else:
                    idmesh = [id_ for id_ in self._ddata[idref2d]['depend']
                              if self._dindref[id_]['group'] == 'mesh'][0]
            pts = self.dmesh[idmesh]['data']['nodes']
            pts = np.array([pts[:,0], np.zeros((pts.shape[0],)), pts[:,1]])

        pts = np.atleast_2d(pts)
        if pts.shape[0] != 3:
            msg = "pts must be np.ndarray of (X,Y,Z) points coordinates\n"
            msg += "Can be multi-dimensional, but the 1st dimension is (X,Y,Z)\n"
            msg += "    - Expected shape : (3,...)\n"
            msg += "    - Provided shape : %s"%str(pts.shape)
            raise Exception(msg)

        # Check t
        lc = [t is None, type(t) is str, type(t) is np.ndarray]
        assert any(lc)
        if lc[1]:
            assert t in self._ddata.keys()
            t = self._ddata[t]['data']

        # Interpolation (including time broadcasting)
        # this is the second slowest step (~0.08 s)
        func = self._get_finterp(idquant=idquant, idref1d=idref1d, idref2d=idref2d,
                                 idq2dR=idq2dR, idq2dPhi=idq2dPhi, idq2dZ=idq2dZ,
                                 interp_t=interp_t, interp_space=interp_space,
                                 fill_value=fill_value, ani=ani, Type=Type)

        # This is the slowest step (~1.8 s)
        val, t = func(pts, vect=vect, t=t)
        return val, t


    def calc_signal_from_Cam(self, cam, t=None,
                             quant=None, ref1d=None, ref2d=None,
                             q2dR=None, q2dPhi=None, q2dZ=None,
                             Brightness=True, interp_t=None,
                             interp_space=None, fill_value=np.nan,
                             res=0.005, DL=None, resMode='abs', method='sum',
                             ind=None, out=object, plot=True, dataname=None,
                             fs=None, dmargin=None, wintit=None, invert=True,
                             units=None, draw=True, connect=True):

        if 'Cam' not in cam.__class__.__name__:
            msg = "Arg cam must be tofu Camera instance (CamLOS1D, CamLOS2D...)"
            raise Exception(msg)

        return cam.calc_signal_from_Plasma2D(self, t=t,
                                             quant=quant, ref1d=ref1d, ref2d=ref2d,
                                             q2dR=q2dR, q2dPhi=q2dPhi,
                                             q2dZ=q2dZ,
                                             Brightness=Brightness,
                                             interp_t=interp_t,
                                             interp_space=interp_space,
                                             fill_value=fill_value, res=res,
                                             DL=DL, resMode=resMode,
                                             method=method, ind=ind, out=out,
                                             pot=plot, dataname=dataname,
                                             fs=fs, dmargin=dmargin,
                                             wintit=wintit, invert=intert,
                                             units=units, draw=draw,
                                             connect=connect)


    #---------------------
    # Methods for getting data
    #---------------------

    def get_dextra(self, dextra=None):
        lc = [dextra is None, dextra == 'all', type(dextra) is dict,
              type(dextra) is str, type(dextra) is list]
        assert any(lc)
        if dextra is None:
            dextra = {}

        if dextra == 'all':
            dextra = [k for k in self._dgroup['time']['ldata']
                      if (self._ddata[k]['lgroup'] == ['time']
                          and k not in self._dindref.keys())]

        if type(dextra) is str:
            dextra = [dextra]

        # get data
        if type(dextra) is list:
            for ii in range(0,len(dextra)):
                if type(dextra[ii]) is tuple:
                    ee, cc = dextra[ii]
                else:
                    ee, cc = dextra[ii], None
                ee, msg = self._get_keyingroup(ee, 'time', raise_=True)
                if self._ddata[ee]['lgroup'] != ['time']:
                    msg = "time-only dependent signals allowed in dextra!\n"
                    msg += "    - %s : %s"%(ee,str(self._ddata[ee]['lgroup']))
                    raise Exception(msg)
                idt = self._ddata[ee]['depend'][0]
                key = 'data' if self._ddata[ee]['data'].ndim == 1 else 'data2D'
                dd = {key: self._ddata[ee]['data'],
                      't': self._ddata[idt]['data'],
                      'label': self._ddata[ee]['name'],
                      'units': self._ddata[ee]['units']}
                if cc is not None:
                    dd['c'] = cc
                dextra[ii] = (ee, dd)
            dextra = dict(dextra)
        return dextra

    def get_Data(self, lquant, X=None, ref1d=None, ref2d=None,
                 remap=False, res=0.01, interp_space=None, dextra=None):

        try:
            import tofu.data as tfd
        except Exception:
            from .. import data as tfd

        # Check and format input
        assert type(lquant) in [str,list]
        if type(lquant) is str:
            lquant = [lquant]
        nquant = len(lquant)

        # Get X if common
        c0 = type(X) is str
        c1 = type(X) is list and (len(X) == 1 or len(X) == nquant)
        if not (c0 or c1):
            msg = "X must be specified, either as :\n"
            msg += "    - a str (name or quant)\n"
            msg += "    - a list of str\n"
            msg += "    Provided: %s"%str(X)
            raise Exception(msg)
        if c1 and len(X) == 1:
            X = X[0]

        if type(X) is str:
            idX, msg = self._get_keyingroup(X, 'radius', msgstr='X', raise_=True)

        # prepare remap pts
        if remap:
            assert self.config is not None
            refS = list(self.config.dStruct['dObj']['Ves'].values())[0]
            ptsRZ, x1, x2, extent = refS.get_sampleCross(res, mode='imshow')
            dmap = {'t':None, 'data2D':None, 'extent':extent}
            if ref is None and X in self._lquantboth:
                ref = X

        # Define Data
        dcommon = dict(Exp=self.Id.Exp, shot=self.Id.shot,
                       Diag='profiles1d', config=self.config)

        # dextra
        dextra = self.get_dextra(dextra)

        # Get output
        lout = [None for qq in lquant]
        for ii in range(0,nquant):
            qq = lquant[ii]
            if remap:
                # Check requested quant is available in 2d or 1d
                idq, idrefd1, idref2d = self._get_quantrefkeys(qq, ref1d, ref2d)
            else:
                idq, msg = self._get_keyingroup(qq, 'radius',
                                                msgstr='quant', raise_=True)
            if idq not in self._dgroup['radius']['ldata']:
                msg = "Only 1d quantities can be turned into tf.data.Data !\n"
                msg += "    - %s is not a radius-dependent quantity"%qq
                raise Exception(msg)
            idt = self._ddata[idq]['depend'][0]

            if type(X) is list:
                idX, msg = self._get_keyingroup(X[ii], 'radius',
                                                msgstr='X', raise_=True)

            dlabels = {'data':{'name': self._ddata[idq]['name'],
                               'units': self._ddata[idq]['units']},
                       'X':{'name': self._ddata[idX]['name'],
                            'units': self._ddata[idX]['units']},
                       't':{'name': self._ddata[idt]['name'],
                            'units': self._ddata[idt]['units']}}

            dextra_ = dict(dextra)
            if remap:
                dmapii = dict(dmap)
                val, tii = self.interp_pts2profile(qq, ptsRZ=ptsRZ, ref=ref,
                                                   interp_space=interp_space)
                dmapii['data2D'], dmapii['t'] = val, tii
                dextra_['map'] = dmapii
            lout[ii] = DataCam1D(Name = qq,
                                 data = self._ddata[idq]['data'],
                                 t = self._ddata[idt]['data'],
                                 X = self._ddata[idX]['data'],
                                 dextra = dextra_, dlabels=dlabels, **dcommon)
        if nquant == 1:
            lout = lout[0]
        return lout


    #---------------------
    # Methods for plotting data
    #---------------------

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
