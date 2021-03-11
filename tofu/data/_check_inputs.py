

# Standard
import itertools as itt

# Common
import numpy as np



_DRESERVED_KEYS = {
    'dgroup': ['lref', 'ldata'],
    'dref': ['ldata', 'group', 'size', 'ind'],
    'ddata': ['ref', 'group', 'shape', 'data'],
}


_DDEF_PARAMS = {
    'origin': (str, 'unknown'),
    'dim':    (str, 'unknown'),
    'quant':  (str, 'unknown'),
    'name':   (str, 'unknown'),
    'units':  (str, 'a.u.'),
}



# #############################################################################
# #############################################################################
#                           Generic
# #############################################################################


def _check_remove(key=None, dkey=None, name=None):
    c0 = isinstance(key, str) and key in dkey.keys()
    c1 = (
        isinstance(key, list)
        and all([isinstance(kk, str) and kk in dkey.keys()])
    )
    if not c0:
        msg = (
            """
            Removed {} must be a str already in self.d{}
            It can also be a list of such
            \t- provided: {}
            \t- already available: {}
            """.format(name, key, sorted(dkey.keys()))
        )
        raise Exception(msg)
    if c0:
        key = [key]
    return key


# #############################################################################
# #############################################################################
#                           Removing routines
# #############################################################################

def _remove_group(group=None, dgroup0=None, dref0=None, ddata0=None):
    """ Remove a group (or list of groups) and all associated ref, data """
    if group is None:
        return dgroup0, dref0, ddata0
    group = _check_remove(key=group, dkey=dgroup0, name='group')

    # Remove groups and orphan ref and data
    for k0 in groups:
        for k1 in dgroup0['lref']:
            del dref0[k1]
        lkdata = [k1 for k1, v1 in ddata0.items() if v1['group'] == (k0,)]
        for kk in lkdata:
            del ddata0[kk]
        del dgroup0[k0]

    # Double-check consistency
    return _consistency(
        ddata=None, ddata0=ddata0,
        dref=None, dref0=dref0,
        dgroup=None, dgroup0=dgroup0,
    )


def _remove_ref(
    key=None, dgroup0=None, dref0=None, ddata0=None, propagate=None,
):
    """ Remove a ref (or list of refs) and all associated data """
    if key is None:
        return group0, dref0, ddata0
    key = _check_inputs._check_remove(
        key=key, dkey=self._dref, name='ref',
    )

    for k0 in key:
        # Remove orphan ddata
        for k1 in dref0[k0]['ldata']:
            del ddata0[k1]

        # Remove ref from dgroup['lref']
        for k1 in dgroup0.keys():
            if k0 in dgroup0[k1]['lref']:
                dgroup0[k1]['lref'].remove(k0)
        del dref0[k0]

    # Propagate upward
    if propagate is True:
        lg = [k0 for k0 in dgroup0.keys() if len(dgroup0['lref']) == 0]
        for gg in lg:
            del dgroup0[gg]

    # Double-check consistency
    return _consistency(
        ddata=None, ddata0=ddata0,
        dref=None, dref0=dref0,
        dgroup=None, dgroup0=dgroup0,
    )


def _remove_data(
    key=None, dgroup0=None, dref0=None, ddata0=None, propagate=None,
):
    """ Remove a ref (or list of refs) and all associated data """
    if key is None:
        return group0, dref0, ddata0
    key = _check_inputs._check_remove(
        key=key, dkey=self._dref, name='data',
    )

    for k0 in key:
        # Remove key from dgroup['ldata'] and dref['ldata']
        for k1 in dgroup0.keys():
            if k0 in dgroup0[k1]['ldata']:
                dgroup0[k1]['ldata'].remove(k0)
            if k0 in dref0[k1]['ldata']:
                dref0[k1]['ldata'].remove(k0)
        del ddata0[k0]

    # Propagate upward
    if propagate is True:
        lk = [k0 for k0 in dgroup0.keys() if len(dgroup0['ldata']) == 0]
        for kk in lk:
            del dgroup0[kk]
        lk = [k0 for k0 in dref0.keys() if len(dref0['ldata']) == 0]
        for kk in lk:
            del dref0[kk]

    # Double-check consistency
    return _consistency(
        ddata=None, ddata0=ddata0,
        dref=None, dref0=dref0,
        dgroup=None, dgroup0=dgroup0,
    )

# #############################################################################
# #############################################################################
#                           dgroup
# #############################################################################


def _check_dgroup(dgroup=None, dgroup0=None, allowed_groups=None):
    """ dgroup must be
    - str: turned to list
    - list of str
    - dict of dict
    """

    # ----------------
    # Trivial case
    if dgroup in [None, {}]:
        return {}

    # ----------------
    # Check conformity
    c0 = isinstance(dgroup, str) and dgroup not in dgroup0.keys()
    c1 = (
        isinstance(goup, list)
        and all([isinstance(gg, str) and gg not in dgroup0.keys()])
    )
    c2 = (
        isinstance(dgroup, dict)
        and all([
            isinstance(k0, str)
            and k0 not in dgroup0.keys()
            and isinstance(v0, dict)
            and all([
                isinstance(k1, str)
                and k1 in _DRESERVED_KEYS['dgroup']
                and isinstance(v1, list)
                and all([isinstance(v2, str) for v2 in v1])
                for k1, v1 in v0.items()
            ])
            for k0, v0 in dgroup.items()
        ])
    )
    if not (c0 or c1 or c2):
        msg = (
            """
            Added group must be either:
            \t- str: not already in self.dgroup
            \t- list of str: each not already in self.dgroup
            \t- dict: each key not already in self.dgroup, each value a dict
            \t- allowed keys in values are:
            \t\t- {}
            You provided:
            \t- {}
            Already available in self.dgroup:
            {}
            """.format(
                sorted(_DRESERVED_KEYS['dgroup'].keys()),
                group,
                '\t- ' + '\n\t- '.join(sorted(dgroup.keys())),
            )
        )
        raise Exception(msg)

    # Check compliance with allowed groups, if any
    if allowed_groups is not None:
        if c0:
            lg = [dgroup] if dgroup not in allowed_groups else []
        elif c1:
            lg = [k0 for k0 in dgroup if k0 not in allowed_groups]
        else:
            lg = [k0 for k0 in dgroup>keys() if k0 not in allowed_groups]
        if len(lg) > 0:
            msg = (
                """
                The following group names are not allowed:
                {}

                Only the following group names are allowed:
                {}
                """.format(
                    '\t- ' + '\n\t- '.join(lg),
                    '\t- ' + '\n\t- '.join(allowed_groups),
                )
            )
            raise Exception(msg)

    # Convert if necessary
    if c0:
        dgroup = {dgroup: {'lref': [], 'ldata': []}}
    elif c1:
        dgroup = {k0: {'lref': [], 'ldata': []} for k0 in dgroup}
    else:
        dgroup = {k0: {'lref': [], 'ldata': []} for k0 in dgroup.keys()}

    return dgroup


# #############################################################################
# #############################################################################
#                           dref
# #############################################################################


class DataRefException(Exception):

    def __init__(ref=None, data=None):
        msg = (
            """
            To be a valid reference for {}, provided data must be either:
            \t- np.ndarray:  of dimension 1 with increasing values
            \t- list, tuple: convertible to the above
            \t- dict / other class: used for meshes

            You provided:
            \t- {}

            """.format(ref, data)
        )
        self.message = msg


def _check_dataref(data=None, ref=None):
    """ Check the conformity of data to be a valid reference """

    # if not array
    # => try converting or get class (dict, mesh...)
    if not isinstance(data, np.ndarray):
        if isinstance(data, list) or isinstance(data, tuple):
            try:
                data = np.array(data)
                size = data.size
            except Exception as err:
                raise DataRefException(ref=ref, data=data)
        else:
            size = data.__class__.__name__

    # if array => check unique (unique + sorted)
    if isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise DataRefException(ref=ref, data=data)

        datau = np.unique(data)
        if not (datau.size == data.size and np.allclose(datau, data)):
            raise DataRefException(ref=ref, data=data)
        size = data.size

    return data, size


def _check_dref(
    dref=None, dref0=None, dgroup0=None, ddata0=None, allowed_groups=None,
):
    """ Check and format dref

    dref can be:
        - dict

    If some groups are not already on dgroup0
        => completes dgroups0

    If some data is provided
        => returns ddata to be added

    Also think about meshes !!!
    """

    # ----------------
    # Trivial case
    if dref in [None, {}]:
        return {}, {}, {}

    # ----------------
    # Check conformity
    ngroup = len(dgroup0)
    if ngroup == 1:
        groupref = list(dgroup0.keys())[0]

    # Basis
    # lk_opt = ['ldata', 'size', 'group', 'data']
    c0 = (
        isinstance(dref, dict)
        and all([
            isinstance(k0, str)
            and k0 not in dref0.keys()
            and (
                (
                    ngroup == 1
                    and (
                        type(v0) in [np.ndarray, list, tuple]
                        or (
                            isinstance(v0, dict)
                            and all([isinstance(ss, str) for ss in v0.keys()])
                            and ('size' in v0.keys() or 'data' in v0.keys())
                        )
                    )
                )
                or (
                    ngroup > 1
                    and isinstance(v0, dict)
                    and all([isinstance(ss, str) for ss in v0.keys()])
                    and ('size' in v0.keys() or 'data' in v0.keys())
                    and 'group' in v0.keys()
                )
            )
            for k0, v0 in dref.items()
        ])
    )

    # Raise exception if non-conformity
    if not c0:
        msg = (
            """
            Arg dref must be a dict of the form:
            {
                'ref0': {'group': str, 'size': int, ...},       (A)
                'ref1': {'group': str, 'data': np.array, ...},  (B)
                'ref2': {'data': np.array, ...},                (C)
                ...
                'refn': np.array,                               (D)
            }

            Where:
                - each 'refi' is a unique str identifier
                - (A) & (B): 'group' is provided as well as 'size' of 'data'
                - (C): 'group' is not provided if len(self.dgroup) == 1
                - (D): only the data array is provided if len(self.dgroup) == 1

            Each ref shall be assigned a group:
            \t- {}

            """.format('\n\t- '.join(sorted(dgroup0.keys())))
        )
        raise Exception(msg)

    # ----------------
    # Convert and/or add group if necessary
    for k0, v0 in dref.items():
        if not isinstance(v0, dict):
            dref[k0] = {'group': groupref, 'data': v0}
        elif v0.get('group') is None:
            dref[k0]['group'] = groupref

    # Add missing groups
    lgroups = set([
        v0['group'] for v0 in dref.values()
        if 'group' in v0.keys() and v0['group'] not in dgroup0.keys()
    ])
    if len(lgroups) > 0:
        dgroup_add = _check_dgroup(
            lgroups, dgroup0=dgroup0, allowed_groups=allowed_groups,
        )

    # Add size / data if relevant
    ddata_add = {
        k0: {'data': None}
        for k0, v0 in dref.items()
        if 'data' in v0.keys() and k0 not in ddata0.keys()
    }
    for k0, v0 in dref.items():
        if 'data' in v0.keys():
            data, dref[k0]['size'] = _check_dataref(data=v0['data'], ref=k0)
            if k0 in ddata_add.keys():
                ddata_add[k0]['data'] = data
                ddata_add[k0].update({
                    k1: v1 for k1, v1 in v0.items()
                    if k1 not in ['group', 'size', 'ldata']
                })

    # get rid of extra keys
    dref = {
        k0: {k1: v0[k1] for k1 in _DRESERVED_KEYS['dref']}
        for k0 in dref.keys()
    }

    return dref, dgroup_add, ddata_add


# #############################################################################
# #############################################################################
#                           ddata
# #############################################################################


def _check_data(data=None, key=None):
    """ Check the conformity of data to be a valid reference """

    # if not array
    # => try converting or get class (dict, mesh...)
    shape = None
    if not isinstance(data, np.ndarray):
        if isinstance(data, list) or isinstance(data, tuple):
            c0 = (
                all([hasattr(oo, '__iter__') for oo in data])
                and len(set([len(oo) for oo in data])) != 1
            )
            c1 = (
                all([hasattr(oo, '__iter__') for oo in data])
                and len(set([len(oo) for oo in data])) == 1
            )
            if c0:
                data = np.array(data, dtype=object)
                shape = (data.shape[0],)
            elif c1:
                data = np.array(data)
            else:
                try:
                    data = np.array(data)
                    shape = data.shape
                except Exception as err:
                    raise DataRefException(ref=ref, data=data)
        else:
            shape = data.__class__.__name__

    # if array => check unique (unique + sorted)
    if isinstance(data, np.ndarray) and shape is None:
        shape = data.shape

    return data, shape


def _check_ddata(
    ddata=None,
    ddata0=None, dref0=None, dgroup0=None,
    reserved_keys=None,
):

    # ----------------
    # Trivial case
    if ddata in [None, {}]:
        return {}, {}, {}

    # ----------------
    # Check conformity
    nref = len(dref0)
    if nref == 1:
        refref = list(dref0.keys())[0]

    # Basis
    # lk_opt = ['ldata', 'size', 'group', 'data']
    c0 = (
        isinstance(ddata, dict)
        and all([
            isinstance(k0, str)
            and k0 not in ddata0.keys()
            and (
                (
                    nref == 1
                    and (
                        type(v0) in [np.ndarray, list, tuple]
                        or (
                            isinstance(v0, dict)
                            and all([isinstance(ss, str) for ss in v0.keys()])
                            and 'data' in v0.keys()
                            and (
                                v0.get('ref') is None
                                or isinstance(v0.get('ref'), str)
                                or isinstance(v0.get('ref'), tuple)
                            )
                        )
                    )
                )
                or (
                    nref > 1
                    and isinstance(v0, dict)
                    and all([isinstance(ss, str) for ss in v0.keys()])
                    and 'data' in v0.keys()
                    and 'ref' in v0.keys()
                    and (
                        isinstance(v0['ref'], str)
                        or isinstance(v0['ref'], tuple)
                    )
                )
            )
            for k0, v0 in ddata.items()
        ])
    )

    # Raise exception if non-conformity
    if not c0:
        msg = (
            """
            Arg ddata must be a dict of the form:
            {
                'data0': {'ref': str, 'size': int, ...},       (A)
                'data1': {'ref': ('ref0', 'ref1'), 'data': np.array, ...},  (B)
                'data2': {'data': np.array, ...},                (C)
                ...
                'datan': np.array,                               (D)
            }

            Where:
                - each 'refi' is a unique str identifier
                - (A) & (B): 'ref' is provided as well as 'size' of 'data'
                - (C): 'group' is not provided if len(self.dgroup) == 1
                - (D): only the data array is provided if len(self.dgroup) == 1

            Each data shall be assigned a ref:
            \t- {}

            """.format('\n\t- '.join(sorted(dref0.keys())))
        )
        raise Exception(msg)

    # ----------------
    # Convert and/or add ref if necessary
    for k0, v0 in ddata.items():
        if not isinstance(v0, dict):
            ddata[k0] = {'ref': (refref,), 'data': v0}
        elif v0.get('ref') is None:
            ddata[k0]['ref'] = (refref,)
        elif isinstance(v0['ref'], str):
            ddata[k0]['ref'] = (v0['ref'],)

    # Add missing refs (only in ddata)
    lref = set(itt.chain.from_iterable([
        [
            rr for rr in v0['ref']
            if rr not in dref0.keys() and rr in ddata.keys()
        ]
        for v0 in ddata.values() if 'ref' in v0.keys()
    ]
    ))
    if len(lref) > 0:
        dref_add = {rr: {'data': ddata[rr]['data']} for rr in lref}
        dref_add, dgroup_add, ddata_dadd = _check_dref(
            dref_add, dref0=dref0, dgroup0=dgroup0,
        )

    # Check data and ref vs shape
    for k0, v0 in ddata.items():
        ddata[k0]['data'], shape = _check_data(data=v0['data'], key=k0)
        if isinstance(shape, tuple):
            c0 = (
                len(v0['ref']) == len(shape)
                and tuple([dref0[rr]['size'] for rr in v0['ref']]) == shape
            )
        else:
            c0 = v0['ref'] == (k0,)
        if not c0:
            msg = (
                """
                Inconsistent shape vs ref for ddata[{0}]:
                    - ddata[{0}]['ref'] = {1}
                    - ddata[{0}]['shape'] = {2}

                If dict / object it should be its own ref!
                """.format(k0, v0['ref'], shape)
            )
            raise Exception(msg)
        ddata[k0]['shape'] = shape

    return ddata, dref_add, dgroup_add


# #############################################################################
# #############################################################################
#                           Params
# #############################################################################


def _harmonize_params(
    ddata=None, lkeys=None, reserved_keys=None, ddefparams=None,
):

    # Check inputs
    if reserved_keys is None:
        reserved_keys = _DRESERVED_KEYS['ddata']
    if ddefparams is None:
        ddefparams = _DDEF_PARAMS

    # ------------------
    # list of param keys

    # Get list of known param keys
    lparams = set(itt.chain.from_iterable([
        [k1 for k1 in v0.keys() if k1 not in reserved_keys]
        for k0, v0 in ddata.items()
    ]))

    # Add arbitrary params
    if lkeys is not None:
        if isinstance(lkeys, str):
            lkeys = [lkeys]
        c0 = (
            isinstance(lkeys, list)
            and all([isinstance(ss, str) for ss in lkeys])
        )
        if not c0:
            msg = "lkeys must be a list of str!"
            raise Exception(msg)
        lparams = set(lparams).intersection(lkeys)

    # ------------------
    # dparam
    for k0, v0 in ddefparams.items():
        for k1, v1 in ddata.items():
            if k0 not in v1.keys():
                ddata[k1][k0] = v0[1]
            else:
                # Check type if already included
                if not isinstance(ddata[k1][k0], v0[0]):
                    msg = (
                        """
                        Wrong type for parameter:
                            - type(ddata[{}][{}]) = {}
                        - Expected: {}
                        """.format(
                            k1, k0, type(ddata[k1][k0]), v0[0],
                        )
                    )
                    raise Exception(msg)

    for k0 in lparams:
        for k1, v1 in ddata.items():
            ddata[k1][k0] = ddata[k1].get(k0)
    return ddata


# #############################################################################
# #############################################################################
#                           Consistency
# #############################################################################


def _consistency(
    ddata=None, ddata0=None,
    dref=None, dref0=None,
    dgroup=None, dgroup0=None,
    allowed_groups=None,
    reserved_keys=None,
    ddefparams=None,
):

    # --------------
    # dgroup
    dgroup = _check_dgroup(
        dgroup=dgroup, dgroup0=dgroup0, allowed_groups=allowed_groups,
    )
    dgroup0.update(dgroup)

    # --------------
    # dref
    dref, dgroup_add, ddata_add = _check_dref(
        dref=dref, dref0=dref0, dgroup0=dgroup0, ddata0=ddata0,
    )
    dgroup0.update(dgroup_add)
    dref0.update(dref)

    # --------------
    # ddata
    ddata, dref_add, dgroup_add = _check_ddata(
        ddata=ddata, ddata0=ddata0, dref0=dref0, dgroup0=dgroup0,
    )
    dgroup0.update(dgroup_add)
    dref0.update(dref_add)
    ddata0.update(ddata)

    # --------------
    # params harmonization
    ddata0 = _harmonize_params(
        ddata=ddata0,
        ddefparams=ddefparams, reserved_keys=reserved_keys,
    )

    # --------------
    # Complement

    # ddata0
    for k0, v0 in ddata0.items():
        ddata0[k0]['group'] = tuple([rr['group'] for rr in v0['ref']])

    # dref0
    for k0, v0 in dref0.items():
        ldata_add = [
            k1 for k1 in ddata0.keys()
            if k0 in ddata0[k1]['ref'] and k1 not in v0['ldata']
        ]
        if len(ldata_add) > 0:
            dref0[k0]['ldata'].extend(ldata_add)

    # dgroup0
    for k0, v0 in dgroup0.items():
        lref_add = [
            k1 for k1 in dref0.keys()
            if dref0[k1]['group'] == k0 and k1 not in v0['lref']
        ]
        if len(lref_add) > 0:
            dgroup0[k0]['lref'].extend(lref_add)

        ldata_add = [
            k1 for k1 in ddata0.keys()
            if k0 in ddata0[k1]['group'] and k1 not in v0['ldata']
        ]
        if len(ldata_add) > 0:
            dgroup0[k0]['ldata'].extend(ldata_add)

    return ddata0, dref0, dgroup0

"""
    # --------------
    # params
    lparam = self._ddata['lparam']
    for kk in self._ddata['lkey']:
        for pp in self._ddata['dict'][kk].keys():
            if pp not in self._reserved_all and pp not in lparam:
                lparam.append(pp)
"""


# #############################################################################
# #############################################################################
#               Get / set / add / remove param
# #############################################################################


def _get_param(ddata=None, param=None, returnas=np.ndarray):
    """ Return the array of the chosen parameter (or list of parameters)

    Can be returned as:
        - dict: {param0: {key0: values0, key1: value1...}, ...}
        - np[.ndarray: {param0: np.r_[values0, value1...], ...}

    """
    # Trivial case
    lp = [kk for kk in ddata[list(ddata.keys())[0]].keys() if kk != 'data']
    if param is None:
        param = lp

    # Check inputs
    lc = [
        isinstance(param, str) and param in ddata.keys() and param != 'data',
        isinstance(param, list)
        and all([isinstance(pp, str) and pp in data.keys() for pp in param])
    ]
    if not any(lc):
        msg = (
            """
            Arg param must a valid param key of a list of such (except 'data')

            Valid params:
            {}

            Provided:
            {}
            """.format('\t- ' + '\n\t- '.join(lp), param)
        )
        raise Exception(msg)

    if lc[0]:
        param = [param]

    c0 = returnas in [np.ndarray, dict]
    if not c0:
        raise Exception(msg)

    # Get output
    if returnas == dict:
        out = {
            k0: {self._ddata[k1][k0] for k1 in self._ddata.keys()}
            for k0 in lp
        }
    else:
        out = {
            k0: [self._ddata[k1][k0] for k1 in self._ddatakeys()]
            for k0 in lp
        }
    return out


def _set_param(ddata=None, param=None, value=None, ind=None, key=None):
    """ Set the value of a parameter

    values can be:
        - None
        - a unique value (int, float, bool, str, tuple) => common to all keys
        - an iterable of values (array, list) => one for each key
        - a dict of values (per key)

    A subset of keys can be chosen (ind, key, fed to self.select()) to set
    only the values of some key

    """

    # Check param
    lp = [kk for kk in ddata[list(ddata.keys())[0]].keys() if kk != 'data']
    if param is None:
        return
    c0 = isinstance(param, str) and param in lp
    if not c0:
        msg = (
            """
            Provided param in not valid
            Valid param:
            {}

            Provided:
            {}
            """.format('\t- ' + '\n\t- '.join(lp), param)
        )
        raise Exception(msg)

    # Check ind / key
    key = _ind_tofrom_key(ddata=ddata, ind=ind, key=key, returnas='key')

    # Check value
    ltypes = [str, int, np.int, float, np.float, tuple]
    lc = [
        type(value) in ltypes,
        isinstance(value, list)
        and all([type(tt) in ltypes for tt in value])
        and len(value) == len(key),
        isinstance(value, np.ndarray)
        and value.shape[0] == len(key),
        isinstance(value, dict)
        and all([
            kk in ddata.keys() and type(vv) in ltypes
            for kk, vv in value.items()
        ])
    ]
    if not (value is None or any(lc)):
        msg = (
            """
            Accepted types for values include:
                - None
                - {}: common to all
                - list, np.ndarray: key by key
                - dict of {key: scalar / str}

            The length of value must match the selected keys ({})
            """.format(ltypes, len(key))
        )
        raise Exception(msg)

    # Update data
    if values is None or lc[0]:
        for kk in key:
            ddata[kk][param] = values
    elif lc[1]:
        for ii, kk in enumerate(key):
            ddata[kk][param] = values[ii]
    else:
        for kk, vv in value.items():
            ddata[kk][param] = vv


def _add_param(ddata=None, param=None, value=None):
    """ Add a parameter, optionnally also set its value """
    lp = [kk for kk in ddata[list(ddata.keys())[0]].keys() if kk != 'data']
    c0 = isinstance(param, str) and param not in lp
    if not c0:
        msg = (
            """
            param must be a str not matching any existing param

            Available param:
            {}

            Provided:
            {}
            """.format(lp, param)
        )
        raise Exception(msg)

    # Initialize and set
    for kk in ddata.keys():
        ddata[kk][param] = None
    self.set_param(ddata=ddata, param=param, value=value)


def _remove_param(ddata=None, param=None):
    """ Remove a parameter, none by default, all if param = 'all' """

    # Check inputs
    lp = [kk for kk in ddata[list(ddata.keys())[0]].keys() if kk != 'data']
    if param is None:
        return
    if param == 'all':
        param = lp

    c0 = isinstance(param, str) and param in lp
    if not c0:
        msg = ()
        raise Exception(msg)

    # Remove
    for k0 in self._ddata.keys():
        del self._ddata[k0][param]



# #############################################################################
# #############################################################################
#               Selection
# #############################################################################


def _ind_tofrom_key(
    ddata=None, dgroup=None,
    ind=None, key=None, group=None, returnas=int,
):

    # --------------------
    # Check / format input
    lc = [ind is not None, key is not None]
    if not np.sum(lc) <= 1:
        msg = ("Args ind and key cannot be prescribed simulatneously!")
        raise Exception(msg)

    if group is not None:
        if not (isinstance(group, str) and group in group.keys()):
            msg = (
                """
                Provided group must be valid key of dgroup:
                {}

                Provided:
                {}
                """.format(sorted(dgroup.keys()), group)
            )
            raise Exception(msg)

    lret = [int, bool, str, 'key']
    if returnas not in lret:
        msg = (
            """
            Possible values for returnas are:
            {}

            Provided:
            {}
            """.format(lret, returnas)
        )
        raise Exception(msg)

    # -----------------
    # Compute

    # Intialize output
    out = np.zeros((len(ddata),), dtype=bool)

    if not any(lc) and group is not None:
        key = dgroup[group]['ldata']
        lc[1] = True

    # Get output
    lk = list(ddata.keys())
    if lc[0]:

        # Check ind
        if not isinstance(ind, np.ndarray):
            ind = np.atleast_1d(ind).ravel()
        c0 = (
            ind.ndim == 1
            and (
                (ind.dtype == np.bool and ind.size == len(ddata))
                or (ind.dtype == np.int and ind.size <= len(ddata))
        ))
        if not c0:
            msg = "Arg ind must be an iterable of bool or int indices!"
            raise Exception(msg)

        # return
        out[ind] = True
        if returnas in [int, str, 'key']:
            out = out.nonzero()[0]
            if returnas in [str, 'key']:
                out = np.array(
                    [kk for ii, kk in enumerate(lk) if ii in out],
                    dtype=str
                )

    elif lc[1]:

        # Check key
        if isinstance(key, str):
            key = [key]
        c0 = (
            isinstance(key, list)
            and all([isinstance(kk, str) and kk in lk])
        )
        if not c0:
            msg = (
                """
                key must be valid key to ddata (or list of such)
                Provided: {}
                """.format(key)
            )
            raise Exception(msg)

        if returnas in ['key', str]:
            out = key
        else:
            for kk in key:
                out[lk.index(kk)] = True
            if returnas == int:
                out = out.nonzero()[0]
    else:
        if returnas == bool:
            out[:] = True
        elif returnas == int:
            out = np.arange(0, len(lk))
        else:
            out = lk
    return out


def _select(self, log='all', returnas=int, **kwdargs):
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




