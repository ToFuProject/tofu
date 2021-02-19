

# Standard
import itertools as itt



_DRESERVED_KEYS = {
    'dgroup': ['lref', 'ldata'],
    'dref': ['ldata', 'group', 'size'],
    'ddata': ['ref', 'group', 'shape', 'data'],
}


_DDEF_PARAMS = {
}


         # 'params': {'origin': (str, 'unknown'),
                    # 'dim':    (str, 'unknown'),
                    # 'quant':  (str, 'unknown'),
                    # 'name':   (str, 'unknown'),
                    # 'units':  (str, 'a.u.')}}



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
        dref=None, ddata0=dref0,
        dgroup=None, ddata0=dgroup0,
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
        dref=None, ddata0=dref0,
        dgroup=None, ddata0=dgroup0,
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
        dref=None, ddata0=dref0,
        dgroup=None, ddata0=dgroup0,
    )

# #############################################################################
# #############################################################################
#                           dgroup
# #############################################################################


def _check_dgroup(dgroup=None, dgroup0=None):
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
    c0 = isinstance(dgroup, str) and dgroup in dgroup.keys()
    c1 = (
        isinstance(goup, list)
        and all([isinstance(gg, str) and gg in dgroup.keys()])
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
            for k0, v0 not in dgroup.items()
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

    def __init__(ref=None, data=None)
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


def _check_dref(dref=None, dref0=None, dgroup0=None, ddata0=None):
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
    lgroups = sorted(set([
        v0['group'] for v0 in dref.values()
        if 'group' in v0.keys() and v0['group'] not in dgroup0.keys()
    ]))
    if len(lgroups) > 0:
        dgroup_add = _check_dgroup(lgroups, dgroup0=dgroup0)

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
    if not isinstance(data, np.ndarray):
        if isinstance(data, list) or isinstance(data, tuple):
            try:
                data = np.array(data)
                shape = data.shape
            except Exception as err:
                raise DataRefException(ref=ref, data=data)
        else:
            shape = data.__class__.__name__

    # if array => check unique (unique + sorted)
    if isinstance(data, np.ndarray):
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
    lref = sorted(set(itt.chain.from_iterable([
        [
            rr for rr in v0['ref']
            if rr not in dref0.keys() and rr in ddata.keys()
        ]
        for v0 in ddata.values() if 'ref' in v0.keys()
    ]
    )))
    if len(lref) > 0:
        dref_add = {rr: {'data': ddata[rr]['data']} for rr in lref}
        dref_add, dgroup_add, ddata_dadd = _check_dref(
            dref_add, dref0=dref0, dgroup0=dgroup0,
        )

    # Check data and ref vs shape
    for k0, v0 in ddata.items():
        ddata[k0]['data'], shape = _check_dataref(data=v0['data'], key=k0)
        if isinstance(shape, tuple):
            c0 = (
                len(v0['ref']) = len(shape)
                tuple([dref0[rr]['size'] for rr in v0['ref']]) == shape
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

    # Check for params      # DB
    lparams = _get_lparams(ddata, lkeys=None, reserved_keys=reserved_keys)
    lparams = _get_lparams(ddata0, lkeys=lparams, reserved_keys=reserved_keys)
    dparams = _get_dparams(ddata, reserved_keys=reserved_keys)

    # Get rid of extra keys
    ddata = {
        k0: {k1: v0[k1] for k1 in _DRESERVED_KEYS}
        for k0 in ddata.keys()
    }

    return ddata, dref_add, dgroup_add


# #############################################################################
# #############################################################################
#                           Consistency
# #############################################################################


def _consistency(
    ddata=None, ddata0=None,
    dref=None, dref0=None,
    dgroup=None, dgroup0=None,
):

    # --------------
    # dgroup
    dgroup = _check_dgroup(dgroup=dgroup, dgroup0=dgroup0)
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
    # Complenent

    # ddata0
    for k0, v0 in ddata0.items():
        ddata0[k0]['group'] = tuple([rr['group'] for rr in v0['ref']])

    # dref0
    for k0, v0 in dref0.items():
        lref_add = [
            k1 for k1 in ddata0.keys()
            if k0 in ddata0[k1]['ref'] and k1 not in v0['ldata']
        ]
        if len(ldata_add) > 0:
            dref0[k0]['ldata'].append(ldata_add)

    # dgroup0
    for k0, v0 in dgroup0.items():
        lref_add = [
            k1 for k1 in dref0.keys()
            if dref0[k1]['group'] == k0 and k1 not in v0['lref']
        ]
        if len(lref_add) > 0:
            dgroup0[k0]['lref'].append(lref_add)
        ldata_add = [
            k1 for k1 in ddata0.keys()
            if k0 in ddata0[k1]['group'] and k1 not in v0['ldata']
        ]
        if len(ldata_add) > 0:
            dgroup0[k0]['ldata'].append(ldata_add)

    return ddata0, dref0, dgroup0











    # --------------
    # ddata
    for k0, v0 in ddata.items():

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


# #############################################################################
# #############################################################################
#                           Params
# #############################################################################


def _get_lparams(data0i, lkeys=None, reserved_keys=None):
    if reserved_keys is None:
        reserved_keys = _DRESERVED_KEYS['ddata']

    # Get list of known param keys
    lparams = sorted(set(itt.chain.from_iterable([
        [k1 for k1 in v0.keys() if k1 not reserved_keys]
        for k0, v0 in ddata0.items()
    ])))

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
        lparam = sorted(set(lparam).intersection(lkeys))
    return lparam


def _get_dparams(ddata0, lkeys=None, reserved_keys=None):
    return {k0:
            {k1: v1.get(k0, None)}
            for k0 in _get_lparams(
                ddata0, lkeys=lkeys, reserved_keys=reserved_keys,
            )
           }


def _extract_params(
    key,
    dd,
    ref=False,
    group=None,
    reserved_params=None,
    _dallowed_params=None,
    _ddef_params=None,
):



    # Extract relevant parameters
    dparams = {
        kk: vv for kk, vv in dd.items()
        if kk not in reserved_params
    }

    if ref and group is not None and _dallowed_params is not None:
        defpars = _dallowed_params[group]
    else:
        defpars = _ddef_params

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
                msg += "    - Expected %s"%str(_ddef_params[kk][0])
                raise Exception(msg)
    return dparams
