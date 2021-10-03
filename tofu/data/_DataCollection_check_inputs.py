

# Standard
import itertools as itt
import warnings

# Common
import numpy as np
import scipy.sparse as scpsp
from matplotlib.tri import Triangulation as mplTri


_DRESERVED_KEYS = {
    'dgroup': ['lref', 'ldata'],
    'dref': ['ldata', 'group', 'size', 'ind'],
    'dref_static': [],
    'ddata': ['ref', 'group', 'shape', 'data'],
    'dobj': [],
}


_DDEF_PARAMS = {
    'ddata': {
        'source': (str, 'unknown'),
        'dim':    (str, 'unknown'),
        'quant':  (str, 'unknown'),
        'name':   (str, 'unknown'),
        'units':  (str, 'a.u.'),
    },
    'dobj': {
    },
}


_DATA_NONE = False


# #############################################################################
# #############################################################################
#                           Generic
# #############################################################################


def _check_which(ddata=None, dobj=None, which=None, return_dict=None):
    """ Check which in ['data'] + list(self._dobj.keys() """

    # --------------
    # Check inputs

    if return_dict is None:
        return_dict = True

    # Trivial case
    if len(ddata) == 0 and len(dobj) == 0:
        if return_dict is True:
            return None, None
        else:
            return

    # which ('data', or keys of dobj)
    if which is None:
        if len(dobj) == 0:
            which = 'data'
        elif len(dobj) == 1:
            which = list(dobj.keys())[0]

    c0 = which in ['data'] + list(dobj.keys())
    if not c0:
        msg = (
            "Please specify whether to sort:\n"
            + "\t- 'data': the content of self.ddata\n\t- "
            + "\n\t- ".join([
                "'{0}': the content of self.dobj['{0}']".format(k0)
                for k0 in dobj.keys()
            ])
            + "\nProvided:\n\t- {}".format(which)
        )
        raise Exception(msg)

    if return_dict is True:
        if which == 'data':
            dd = ddata
        else:
            dd = dobj[which]
        return which, dd
    else:
        return which


def _check_conflicts(dd=None, dd0=None, dd_name=None):
    """ Detect conflict with existing entries
    """
    dupdate = {}
    dconflict = {}
    for k0, v0 in dd.items():
        if k0 not in dd0.keys():
            continue
        # conflicts
        lk = set(v0.keys()).intersection(dd0[k0].keys())
        lk = [
            kk for kk in lk
            if not (
                isinstance(v0[kk], dd0[k0][kk].__class__)
                and (
                    (
                        isinstance(v0[kk], np.ndarray)
                        and np.allclose(v0[kk], dd0[k0][kk], equal_nan=True)
                    )
                    or (
                        scpsp.issparse(v0[kk])
                        and np.allclose(
                            v0[kk].data, dd0[k0][kk].data, equal_nan=True,
                        )
                    )
                    or (
                        v0[kk] == dd0[k0][kk]
                    )
                )
            )
        ]
        if len(lk) > 0:
            dconflict[k0] = lk
        # updates
        lk = [
            kk for kk in dd0[k0].keys()
            if kk not in v0.keys() and kk not in ['ldata', 'size']
        ]
        if len(lk) > 0:
            dupdate[k0] = lk

    # Conflicts => Exception
    if len(dconflict) > 0:
        msg = (
            "Conflicts with pre-existing values found in {}:\n".format(dd_name)
            + "\n".join([
                f"\t- {dd_name}['{k0}']: {v0}"
                for k0, v0 in dconflict.items()
            ])
        )
        raise Exception(msg)

    # Updates => Warning
    if len(dupdate) > 0:
        msg = (
            "\nExisting {} keys will be overwritten:\n".format(dd_name)
            + "\n".join([
                f"\t- {dd_name}['{k0}']: {v0}"
                for k0, v0 in dupdate.items()
            ])
        )
        warnings.warn(msg)


def _check_remove(key=None, dkey=None, name=None):
    c0 = isinstance(key, str) and key in dkey.keys()
    c1 = (
        isinstance(key, list)
        and all([isinstance(kk, str) and kk in dkey.keys() for kk in key])
    )
    if not (c0 or c1):
        msg = (
            """
            Removed param must be a str already in self.d{}
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

def _remove_group(
    group=None, dgroup0=None, dref0=None, ddata0=None,
    dref_static0=None,
    dobj0=None,
    allowed_groups=None,
    reserved_keys=None,
    ddefparams_data=None,
    ddefparams_obj=None,
    data_none=None,
    max_ndim=None,
):
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
        dref_static=None, dref_static0=dref_static0,
        dobj=None, dobj0=dobj0,
        dgroup=None, dgroup0=dgroup0,
        allowed_groups=allowed_groups,
        reserved_keys=reserved_keys,
        ddefparams_data=ddefparams_data,
        ddefparams_obj=ddefparams_obj,
        data_none=data_none,
        max_ndim=max_ndim,
    )


def _remove_ref(
    key=None,
    dgroup0=None, dref0=None, ddata0=None,
    dref_static0=None,
    dobj0=None,
    propagate=None,
    allowed_groups=None,
    reserved_keys=None,
    ddefparams_data=None,
    ddefparams_obj=None,
    data_none=None,
    max_ndim=None,
):
    """ Remove a ref (or list of refs) and all associated data """
    if key is None:
        return group0, dref0, ddata0
    key = _check_remove(
        key=key, dkey=dref0, name='ref',
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
        dref_static=None, dref_static0=dref_static0,
        dobj=None, dobj0=dobj0,
        dgroup=None, dgroup0=dgroup0,
        allowed_groups=allowed_groups,
        reserved_keys=reserved_keys,
        ddefparams_data=ddefparams_data,
        ddefparams_obj=ddefparams_obj,
        data_none=data_none,
        max_ndim=max_ndim,
    )


def _remove_ref_static(
    key=None,
    which=None,
    propagate=None,
    dref_static0=None,
    ddata0=None,
    dobj0=None,
):
    """ Remove a static ref (or list) or a whole category

    key os provided:
        => remove only the desired key(s)
            works only if key is not used in ddata and dobj

    which is provided:
        => treated as param, the whole category of ref_static is removed
            if propagate, the parameter is removed from ddata and dobj
    """

    lc = [
        key is not None,
        which is not None,
    ]
    if np.sum(lc) != 1:
        msg = "Please provide either key xor which!"
        raise Exception(msg)

    if key is not None:
        if isinstance(key, str):
            key = [key]

        lk0 = [
            k0 for k0, v0 in dref_static0.items()
            if all([kk in v0.keys() for kk in key])
        ]
        if len(lk0) != 1:
            msg = (
                "No / several matches for '{}' in ref_static:\n".format(key)
                + "\n".join([
                    "\t- dref_static[{}][{}]".format(k0, key) for k0 in lk0
                ])
            )
            raise Exception(msg)
        k0 = lk0[0]
        key = _check_remove(
            key=key,
            dkey=dref_static0[k0],
            name='ref_static[{}]'.format(k0),
        )

        # Make sure key is not used (condition for removing)
        for kk in key:
            lk1 = [
                k1 for k1, v1 in ddata0.items()
                if kk == v1.get(k0)
            ]
            lk2 = [
                k1 for k1, v1 in dobj0.items()
                if any([kk == v2.get(k0) for v2 in v1.values()])
            ]
            if len(lk1) > 0 or len(lk2) > 0:
                msg = (
                    "Provided ref_static key ({}) is used in:\n".format(kk)
                    + "\n".join(
                        ["\t- self.ddata['{}']".format(k1) for k1 in lk1]
                        + [
                            "\t- self.dobj['{}']['{}']".format(k2, k0)
                            for k2 in lk2
                        ]
                    )
                )
                raise Exception(msg)
            del dref_static0[k0][kk]

    elif which is not None:
        if which not in dref_static0.keys():
            msg = (
                "Provided which not in dref_static.keys():\n"
                + "\t- Available: {}\n".format(sorted(dref_static0.keys()))
                + "\t- Provided: {}".format(which)
            )
            raise Exception(msg)
        del dref_static0[which]

        # Propagate (delete as partam in ddata and dobj)
        if propagate is None:
            propagate = True

        if propagate is True:
            # ddata
            if which in list(ddata0.values())[0].keys():
                _remove_param(dd=ddata0, dd_name='ddata', param=which)

            # dobj0
            for k0 in dobj0.keys():
                if which in list(dobj0[k0].values())[0].keys():
                    _remove_param(
                        dd=dobj0[k0],
                        dd_name="ddobj['{}']".format(k0),
                        param=which,
                    )


def _remove_data(
    key=None,
    dgroup0=None, dref0=None, ddata0=None,
    dref_static0=None,
    dobj0=None,
    propagate=None,
    allowed_groups=None,
    reserved_keys=None,
    ddefparams_data=None,
    ddefparams_obj=None,
    data_none=None,
    max_ndim=None,
):
    """ Remove a ref (or list of refs) and all associated data """
    if key is None:
        return group0, dref0, ddata0
    key = _check_remove(
        key=key, dkey=ddata0, name='data',
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
        dref_static=None, dref_static0=dref_static0,
        dobj=None, dobj0=dobj0,
        dgroup=None, dgroup0=dgroup0,
        allowed_groups=allowed_groups,
        reserved_keys=reserved_keys,
        ddefparams_data=ddefparams_data,
        ddefparams_obj=ddefparams_obj,
        data_none=data_none,
        max_ndim=max_ndim,
    )


def _remove_obj(
    key=None,
    which=None,
    dobj0=None,
    ddata0=None,
    dref0=None,
    dref_static0=None,
    dgroup0=None,
    allowed_groups=None,
    reserved_keys=None,
    ddefparams_data=None,
    ddefparams_obj=None,
    data_none=None,
    max_ndim=None,
):

    # ------------
    # Check inputs

    lc = [
        key is not None,
        which is not None,
    ]
    if np.sum(lc) != 1:
        msg = "Please provide either key xor which!"
        raise Exception(msg)

    if key is not None:
        # key => delete list of obj
        if isinstance(key, str):
            key = [key]

        lk0 = [
            k0 for k0, v0 in dobj0.items()
            if all([kk in v0.keys() for kk in key])
        ]
        if len(lk0) != 1:
            msg = (
                "No / several matches for '{}' in dobj:\n".format(key)
                + "\n".join([
                    "\t- dobj[{}][{}]".format(k0, key) for k0 in lk0
                ])
            )
            raise Exception(msg)
        k0 = lk0[0]
        key = _check_remove(
            key=key,
            dkey=dobj0[k0],
            name='dobj[{}]'.format(k0),
        )
        for kk in set(key).intersection(dobj0[k0].keys()):
            del dobj0[k0][kk]

    elif which is not None:
        if which not in dobj0.keys():
            msg = (
                "Provided which is not a valid self.dobj.keys()!\n"
                + "\t- provided: {}\n".format(which)
                + "\t- available: {}\n".format(sorted(dobj0.keys()))
            )
            raise Exception(msg)

        del dobj0[which]

    return _consistency(
        ddata=None, ddata0=ddata0,
        dref=None, dref0=dref0,
        dref_static=None, dref_static0=dref_static0,
        dobj=None, dobj0=dobj0,
        dgroup=None, dgroup0=dgroup0,
        allowed_groups=allowed_groups,
        reserved_keys=reserved_keys,
        ddefparams_data=ddefparams_data,
        ddefparams_obj=ddefparams_obj,
        data_none=data_none,
        max_ndim=max_ndim,
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
        isinstance(dgroup, list)
        and all([
            isinstance(gg, str) and gg not in dgroup0.keys() for gg in dgroup
        ])
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
                sorted(_DRESERVED_KEYS['dgroup']),
                dgroup,
                '\t- ' + '\n\t- '.join(sorted(dgroup0.keys())),
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
            lg = [k0 for k0 in dgroup > keys() if k0 not in allowed_groups]
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
#                           dref_static
# #############################################################################


def _check_dref_static(
    dref_static=None, dref_static0=None,
):
    """ Check and format dref_staytic

    dref_static can be:
        - dict

    """

    # ----------------
    # Trivial case
    if dref_static in [None, {}]:
        return {}

    # ----------------
    # Check conformity

    c0 = (
        isinstance(dref_static, dict)
        and all([
            isinstance(k0, str)
            and isinstance(v0, dict)
            and all([
                isinstance(k1, str)
                and isinstance(v1, dict)
                for k1, v1 in v0.items()
            ])
            for k0, v0 in dref_static.items()
        ])
    )

    # Raise exception if non-conformity
    if not c0:
        msg = (
            """
            Arg dref_static must be a dict of the form:
            dict(
                'type0': {'k0': {...},
                          'k1': {...}},
                'type1': {'k0': {...},
                          'k1': {...}},
            )
            """
            +
            """
            Provided:
            {}
            """.format(dref_static)
        )
        raise Exception(msg)

    # raise except if conflict with existing entry
    dupdate = {}
    dconflict = {}
    for k0, v0 in dref_static.items():
        lkout = ['nb. data']
        if k0 == 'ion':
            lkout += ['ION', 'charge', 'element']
        if k0 not in dref_static0.keys():
            continue

        for k1, v1 in v0.items():
            if k1 not in dref_static0[k0].keys():
                continue
            # conflicts
            lk = set(v1.keys()).intersection(dref_static0[k0][k1].keys())
            lk = [kk for kk in lk if v1[kk] != dref_static0[k0][k1][kk]]
            if len(lk) > 0:
                dconflict[k0] = (k1, lk)
            # updates
            lk = [
                kk for kk in dref_static0[k0][k1].keys()
                if kk not in v1.keys()
                and kk not in lkout
                and 'nb. ' not in kk
            ]
            if len(lk) > 0:
                dupdate[k0] = (k1, lk)

    # Conflicts => Exception
    if len(dconflict) > 0:
        msg = (
            "The following dref_static keys are conflicting existing values:\n"
            + "\n".join([
                "\t- dref_static['{}']['{}']: {}".format(k0, v0[0], v0[1])
                for k0, v0 in dconflict.items()
            ])
        )
        raise Exception(msg)

    # Updates => Warning
    if len(dupdate) > 0:
        msg = (
            "\nThe following existing dref_static keys will be forgotten:\n"
            + "\n".join([
                "\t- dref_static['{}']['{}']: {}".format(k0, v0[0], v0[1])
                for k0, v0 in dupdate.items()
            ])
        )
        warnings.warn(msg)

    # ------------------
    # Check element / ion / charge
    _check_elementioncharge_dict(dref_static=dref_static)

    return dref_static


# #############################################################################
# #############################################################################
#                           dref
# #############################################################################


class DataRefException(Exception):

    def __init__(self, ref=None, data=None):
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


def _check_dataref(data=None, key=None):
    """ Check the conformity of data to be a valid reference """

    # if not array
    # => try converting or get class (dict, mesh...)
    group = None
    if not isinstance(data, np.ndarray):
        if isinstance(data, list) or isinstance(data, tuple):
            try:
                data = np.array(data)
                size = data.size
            except Exception as err:
                raise DataRefException(ref=key, data=data)
        else:
            try:
                data, size = _check_mesh_temp(data=data, key=key)
                if len(size) == 1:
                    size = size[0]
                group = 'mesh2d'
            except Exception as err:
                size = data.__class__.__name__

    # if array => check unique (unique + sorted)
    if isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise DataRefException(ref=key, data=data)

        datau = np.unique(data)
        if not (datau.size == data.size and np.allclose(datau, data)):
            raise DataRefException(ref=key, data=data)
        size = data.size

    return data, size, group


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
        return {}, None, None

    # ----------------
    # Check conformity
    ngroup = len(dgroup0)
    if ngroup == 1:
        groupref = list(dgroup0.keys())[0]

    # Basis
    # lk_opt = ['ldata', 'size', 'group', 'data']
    c0 = isinstance(dref, dict)
    lc = [
        k0 for k0, v0 in dref.items()
        if not (
            isinstance(k0, str)
            # and k0 not in dref0.keys()
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
                    (ngroup == 0 or ngroup > 1)
                    and isinstance(v0, dict)
                    and all([isinstance(ss, str) for ss in v0.keys()])
                    and ('size' in v0.keys() or 'data' in v0.keys())
                    and (
                        'group' in v0.keys()
                        or (
                            'data' in v0.keys()
                            and isinstance(v0['data'], dict)
                        )
                    )
                )
            )
        )
    ]

    # Raise exception if non-conformity
    if not (c0 and len(lc) == 0):
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

            Non-conform refs
            """
            + '\t- ' + '\n\t- '.join(lc)
        )
        raise Exception(msg)

    # -----------------------
    # Make sure all are dict
    for k0, v0 in dref.items():
        if not isinstance(v0, dict):
            dref[k0] = {'data': v0}

    # -----------------------
    # raise except if conflict with existing entry
    _check_conflicts(dd=dref, dd0=dref0, dd_name='dref')

    # ----------------
    # Add size / data if relevant
    ddata_add = {
        k0: {'data': None}
        for k0, v0 in dref.items()
        if 'data' in v0.keys() and k0 not in ddata0.keys()
    }
    for k0, v0 in dref.items():
        if 'data' in v0.keys():
            data, dref[k0]['size'], group = _check_dataref(
                data=v0['data'], key=k0,
            )
            if k0 in ddata_add.keys():
                ddata_add[k0]['data'] = data
                ddata_add[k0]['ref'] = (k0,)
                ddata_add[k0].update({
                    k1: v1 for k1, v1 in v0.items()
                    if k1 not in ['group', 'size', 'ldata']
                })
            if group is not None and dref.get('group') is None:
                dref[k0]['group'] = group

    # Make sure, if ngroup != 1, that NOW all refs have a group
    if ngroup != 1:
        lerr = [k0 for k0, v0 in dref.items() if v0.get('group') is None]
        if len(lerr) > 0:
            msg = "Some groups remain ambiguous!:\n{}".format(lerr)
            raise Exception(msg)

    # ----------------
    # Convert and/or add group if necessary
    for k0, v0 in dref.items():
        if v0.get('group') is None:
            dref[k0]['group'] = groupref

    # Add missing groups
    lgroups = sorted(set([
        v0['group'] for v0 in dref.values()
        if 'group' in v0.keys() and v0['group'] not in dgroup0.keys()
    ]))

    dgroup_add = None
    if len(lgroups) > 0:
        dgroup_add = _check_dgroup(
            lgroups, dgroup0=dgroup0, allowed_groups=allowed_groups,
        )

    # get rid of extra keys
    dref = {
        k0: {k1: v1 for k1, v1 in v0.items() if k1 in _DRESERVED_KEYS['dref']}
        for k0, v0 in dref.items()
    }
    return dref, dgroup_add, ddata_add


# #############################################################################
# #############################################################################
#               ddata - special case: meshes
# #############################################################################


def _get_RZ(arr, name=None, shapeRZ=None):
    if arr.ndim == 1:
        if np.any(np.diff(arr) <= 0.):
            msg = "Non-increasing {}".format(name)
            raise Exception(msg)
    else:
        lc = [np.all(np.diff(arr[0, :])) > 0.,
              np.all(np.diff(arr[:, 0])) > 0.]
        if np.sum(lc) != 1:
            msg = "Impossible to know {} dimension!".format(name)
            raise Exception(msg)
        if lc[0]:
            arr = arr[0, :]
            if shapeRZ[1] is None:
                shapeRZ[1] = name
            if shapeRZ[1] != name:
                msg = "Inconsistent shapeRZ"
                raise Exception(msg)
        else:
            arr = arr[:, 0]
            if shapeRZ[0] is None:
                shapeRZ[0] = name
            if shapeRZ[0] != name:
                msg = "Inconsistent shapeRZ"
                raise Exception(msg)
    return arr, shapeRZ


def _duplicates(arr, arru, nn, name=None, msg=None):
    msg += (
        "  Duplicate {}: {}\n".format(name, nn - arru.shape[0])
        + "\t- {}.shape: {}\n".format(name, arr.shape)
        + "\t- unique shape: {}".format(arru.shape)
    )
    return msg


def _check_trimesh_conformity(nodes, faces, key=None):
    nnodes = nodes.shape[0]
    nfaces = faces.shape[0]

    # Test for duplicates
    nodesu = np.unique(nodes, axis=0)
    facesu = np.unique(faces, axis=0)
    lc = [nodesu.shape[0] != nnodes,
          facesu.shape[0] != nfaces]
    if any(lc):
        msg = "Non-valid mesh ddata[{0}]: \n".format(key)
        if lc[0]:
            msg = _duplicates(nodes, nodesu, nnodes, name='nodes', msg=msg)
        if lc[1]:
            msg = _duplicates(faces, facesu, nfaces, name='faces', msg=msg)
        raise Exception(msg)

    # Test for unused nodes
    facesu = np.unique(facesu)
    c0 = np.all(facesu >= 0) and facesu.size == nnodes
    if not c0:
        ino = str([ii for ii in range(0, nnodes) if ii not in facesu])
        msg = "Unused nodes in ddata[{0}]:\n".format(key)
        msg += "    - unused nodes indices: {}".format(ino)
        warnings.warn(msg)

    # Check counter-clockwise orientation
    x, y = nodes[faces, 0], nodes[faces, 1]
    orient = ((y[:, 1] - y[:, 0])*(x[:, 2] - x[:, 1])
              - (y[:, 2] - y[:, 1])*(x[:, 1] - x[:, 0]))

    clock = orient > 0.
    if np.any(clock):
        msg = ("Some triangles not counter-clockwise\n"
               + "  (necessary for matplotlib.tri.Triangulation)\n"
               + "    => {}/{} triangles reshaped".format(clock.sum(), nfaces))
        warnings.warn(msg)
        faces[clock, 1], faces[clock, 2] = faces[clock, 2], faces[clock, 1]
    return faces


def _check_mesh_temp(data=None, key=None):
    # Check if provided data is mesh (as a dict)

    # ------------
    # Check basics
    lmok = ['rect', 'tri', 'quadtri']
    c0 = (
        isinstance(data, dict)
        and all([ss in data.keys() for ss in ['type']])
        and data['type'] in lmok
        and (
            (
                data['type'] == 'rect'
                and all([ss in data.keys() for ss in ['R', 'Z']])
                and isinstance(data['R'], np.ndarray)
                and isinstance(data['Z'], np.ndarray)
                and data['R'].ndim in [1, 2]
                and data['Z'].ndim in [1, 2]
            )
            or (
                data['type'] in ['tri', 'quadtri', 'quad']
                and all([ss in data.keys() for ss in ['nodes', 'faces']])
                and isinstance(data['nodes'], np.ndarray)
                and isinstance(data['faces'], np.ndarray)
                and data['nodes'].ndim == 2
                and data['faces'].ndim == 2
                and data['faces'].dtype == np.int
                and data['nodes'].shape[1] == 2
                and (
                    (
                        data['type'] in ['tri', 'quadtri']
                        and data['faces'].shape[1] == 3
                    )
                    or (
                        data['type'] == 'quad'
                        and data['faces'].shape[1] == 4
                    )
                )
                and np.max(data['faces']) <= data['nodes'].shape[0]
            )
        )
    )
    if not c0:
        msg = (
            """
            A mesh should be a dict of one of the following form:

                dict(
                 'type': 'rect',
                 'R': np.ndarray (with ndim in [1, 2]),
                 'Z': np.ndarray (with ndim in [1, 2]),
                 'shapeRZ': ('R', 'Z') or ('Z', 'R')
                )

                 dict(
                 'type': 'tri' or 'quadtri',
                 'nodes': np.ndarray of shape (N, 2),
                 'faces': np.ndarray of int of shape (N, 3)
                )

                dict(
                 'type': 'quad',
                 'nodes': np.ndarray of shape (N, 2),
                 'faces': np.ndarray of int of shape (N, 4)
                )

            Provided:
            {}
            """.format(data)
        )
        raise Exception(msg)

    # ------------
    # Check per type
    if data['type'] == 'rect':

        shapeRZ = data.get('shapeRZ', [None, None])
        if shapeRZ is None:
            shapeRZ = [None, None]
        else:
            shapeRZ = list(shapeRZ)

        R, shapeRZ = _get_RZ(data['R'], name='R', shapeRZ=shapeRZ)
        Z, shapeRZ = _get_RZ(data['Z'], name='Z', shapeRZ=shapeRZ)
        shapeRZ = tuple(shapeRZ)

        if shapeRZ not in [('R', 'Z'), ('Z', 'R')]:
            msg = "Inconsistent shapeRZ"
            raise Exception(msg)

        def trifind(
            r, z,
            Rbin=0.5*(R[1:] + R[:-1]),
            Zbin=0.5*(Z[1:] + Z[:-1]),
            nR=R.size, nZ=Z.size,
            shapeRZ=shapeRZ
        ):
            indR = np.searchsorted(Rbin, r)
            indZ = np.searchsorted(Zbin, z)
            indR[(r < R[0]) | (r > R[-1])] = -1
            indZ[(z < Z[0]) | (z > Z[-1])] = -1
            return indR, indZ
            # if shapeRZ == ('R', 'Z'):
            #     indpts = indR*nZ + indZ
            # else:
            #     indpts = indZ*nR + indR
            # indout = ((r < R[0]) | (r > R[-1])
            #           | (z < Z[0]) | (z > Z[-1]))
            # indpts[indout] = -1
            # return indpts

        data['R'] = R
        data['Z'] = Z
        data['shapeRZ'] = shapeRZ
        data['nR'] = R.size
        data['nZ'] = Z.size
        data['shape'] = (R.size, Z.size)
        data['trifind'] = trifind
        data['ftype'] = data.get('ftype', 0)

        if data['ftype'] != 0:
            msg = "Linear interpolation not handled yet !"
            raise Exception(msg)

    else:
        # Check mesh conformity for triangulation
        data['faces'] = _check_trimesh_conformity(
            nodes=data['nodes'], faces=data['faces'], key=key
        )

        data['nnodes'] = data['nodes'].shape[0]
        data['nfaces'] = data['faces'].shape[0]
        data['ftype'] = data.get('ftype', 0)

        # Convert 'quad' to 'quadtri' if relevant
        if data['type'] == 'quad':
            # Convert to tri mesh (solution for unstructured meshes)
            faces = np.empty((data['nfaces']*2, 3), dtype=int)
            faces[::2, :] = data['faces'][:, :3]
            faces[1::2, :-1] = data['faces'][:, 2:]
            faces[1::2, -1] = data['faces'][:, 0]
            data['faces'] = faces
            data['type'] = 'quadtri'
            data['ntri'] = 2

            # Re-check mesh conformity
            data['faces'] = _check_trimesh_conformity(
                nodes=data['nodes'], faces=data['faces'], key=key
            )

        # Check ntri
        if data['type'] == 'tri':
            data['ntri'] = 1
        elif 'ntri' not in data.keys():
            msg = (
                """
                For ddata[{}] of type 'quadtri', 'ntri' must be provided
                """.format(key)
            )
            raise Exception(msg)

        # Only triangular meshes so far
        if 'tri' in data['type']:
            if data.get('mpltri', None) is None:
                data['mpltri'] = mplTri(
                    data['nodes'][:, 0],
                    data['nodes'][:, 1],
                    data['faces']
                )
            if not isinstance(data['mpltri'], mplTri):
                msg = (
                    """
                    ddata[{}]['mpltri'] must be a matplotlib Triangulation
                    Provided:
                    {}
                    """.format(key, data['mpltri'])
                )
            assert data['ftype'] in [0, 1]
            if data['ftype'] == 1:
                data['shape'] = (data['nnodes'],)
            else:
                data['shape'] = (int(data['nfaces'] / data['ntri']),)

    return data, data['shape']


# #############################################################################
# #############################################################################
#               ddata - special case: roman to int (SpectralLines)
# #############################################################################


def roman2int(ss):
    """
    :type s: str
    :rtype: int

    source: https://www.tutorialspoint.com/roman-to-integer-in-python
    """
    roman = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000,
        'IV': 4,
        'IX': 9,
        'XL': 40,
        'XC': 90,
        'CD': 400,
        'CM': 900,
    }
    i = 0
    num = 0
    while i < len(ss):
        if i+1 < len(ss) and ss[i:i+2] in roman:
            num += roman[ss[i:i+2]]
            i += 2
        else:
            num += roman[ss[i]]
            i += 1
    return num


def int2roman(num):
    roman = {
        1000: "M",
        900: "CM",
        500: "D",
        400: "CD",
        100: "C",
        90: "XC",
        50: "L",
        40: "XL",
        10: "X",
        9: "IX",
        5: "V",
        4: "IV",
        1: "I",
    }

    def roman_num(num):
        for r in roman.keys():
            x, y = divmod(num, r)
            yield roman[r] * x
            num -= (r * x)
            if num <= 0:
                break

    return "".join([a for a in roman_num(num)])


# #############################################################################
# #############################################################################
#                           ddata
# #############################################################################


def _check_data(data=None, key=None, max_ndim=None):
    """ Check the conformity of data to be a valid reference """

    # if not array
    # => try converting or get class (dict, mesh...)
    shape = None
    group = None
    c0_array = (
        isinstance(data, np.ndarray)
        or scpsp.issparse(data)
    )
    if not c0_array:
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
                    raise DataRefException(ref=key, data=data)
        else:
            try:
                data, shape = _check_mesh_temp(data=data, key=key)
                group = 'mesh2d'
            except Exception as err:
                shape = data.__class__.__name__

    # if array => check unique (unique + sorted)
    if c0_array and shape is None:
        shape = data.shape

    # Check max_dim if any
    if c0_array and max_ndim is not None:
        if data.ndim > max_ndim:
            msg = (
                """
                Provided data for ddata[{}] has too many dimensions!
                - ndim:     {}
                - max_ndim: {}
                """.format(key, data.ndim, max_ndim)
            )
            raise Exception(msg)

    # Check if valid ref candidate
    if isinstance(data, np.ndarray):
        monotonous = tuple([
            bool(
                np.all(np.diff(data, axis=aa) > 0.)
                or np.all(np.diff(data, axis=aa) < 0.)
            )
            for aa in range(data.ndim)
        ])
    else:
        monotonous = (False,)
    return data, shape, group, monotonous


def _check_ddata(
    ddata=None,
    ddata0=None,
    dref0=None,
    dgroup0=None,
    reserved_keys=None,
    allowed_groups=None,
    data_none=None,
    max_ndim=None,
):

    # ----------------
    # Trivial case
    if ddata in [None, {}]:
        return {}, None, None
    if data_none is None:
        data_none = _DATA_NONE

    # ----------------
    # Check conformity
    nref = len(dref0)
    if nref == 1:
        refref = list(dref0.keys())[0]

    # Basis
    # lk_opt = ['ldata', 'size', 'group', 'data']
    c0 = isinstance(ddata, dict)
    lc = [
        k0 for k0, v0 in ddata.items()
        if not (
            isinstance(k0, str)
            # and k0 not in ddata0.keys()
            and (
                (
                    nref == 1
                    and (
                        isinstance(v0, (np.ndarray, list, tuple))
                        or scpsp.issparse(v0)
                        or (
                            isinstance(v0, dict)
                            and all([isinstance(ss, str) for ss in v0.keys()])
                            and (
                                (
                                    'data' in v0.keys()
                                    and (
                                        v0.get('ref') is None
                                        or isinstance(v0.get('ref'), str)
                                        or isinstance(v0.get('ref'), tuple)
                                        or v0.get('ref') is True
                                    )
                                )
                                or (
                                    data_none is True
                                    and v0.get('data') is None
                                )
                            )
                        )
                    )
                )
                or (
                    (nref == 0 or nref > 1)
                    and isinstance(v0, dict)
                    and all([isinstance(ss, str) for ss in v0.keys()])
                    and (
                        (
                            'data' in v0.keys()
                            and (
                                (
                                    'ref' in v0.keys()
                                    and (
                                        isinstance(v0.get('ref'), str)
                                        or isinstance(v0.get('ref'), tuple)
                                        or v0.get('ref') is True
                                    )
                                )
                                or (
                                    isinstance(v0['data'], dict)
                                    or isinstance(v0.get('ref'), str)
                                    or isinstance(v0.get('ref'), tuple)
                                    or v0.get('ref') in [None, True]
                                )
                            )
                        )
                        or (
                            data_none is True
                            and v0.get('data') is None
                        )
                    )
                )
            )
        )
    ]

    # Raise exception if non-conformity
    if not (c0 and len(lc) == 0):
        msg = (
            """
            Arg ddata must be a dict of the form:
            dict(
                'data0': {'ref': 'ref0', 'data': list, ...},       (A)
                'data1': {'ref': ('ref0', 'ref1'), 'data': np.array, ...},  (B)
                'data2': {'data': np.array, ...},                (C)
                ...
                'datan': np.array,                               (D)
            )

            Where:
                - each 'datai' is a unique str identifier
                - (A) & (B): 'data' is provided as well as 'ref'
                - (C): 'ref' is not provided if len(self.dref) == 1
                - (D): only the data array is provided if len(self.dgroup) == 1

            If ref = True, the data is itself considered a ref

            The following keys do not match the criteria:
            """
            + '\t- '+'\n\t- '.join(lc)
        )
        raise Exception(msg)

    # -----------------------
    # raise except if conflict with existing entry
    _check_conflicts(dd=ddata, dd0=ddata0, dd_name='ddata')

    # ----------------
    # Convert and/or add ref if necessary
    lref_add = None
    for k0, v0 in ddata.items():
        if not isinstance(v0, dict):
            ddata[k0] = {'ref': (refref,), 'data': v0}
        else:
            if v0.get('data') is None:
                continue
            if v0.get('ref') is None:
                if not isinstance(v0['data'], dict):
                    ddata[k0]['ref'] = (refref,)
            elif isinstance(v0['ref'], str):
                ddata[k0]['ref'] = (v0['ref'],)
            elif v0['ref'] is True:
                if k0 not in dref0.keys():
                    if lref_add is None:
                        lref_add = [k0]
                    else:
                        lref_add.append(k0)
                ddata[k0]['ref'] = (k0,)

    # Check data and ref vs shape - and optionnally add to ref if mesh2d
    for k0, v0 in ddata.items():
        if v0.get('data') is not None:
            (
                ddata[k0]['data'], ddata[k0]['shape'],
                group, ddata[k0]['monot']
            ) = _check_data(
                data=v0['data'], key=k0, max_ndim=max_ndim,
            )

            # Check if group / mesh2d
            if group is not None:
                c0 = ddata[k0].get('ref') in [None, (k0,)]
                if not c0:
                    msg = (
                        """
                        ddata[{}]['ref'] is a {}
                          => it should have ref = ({},)
                        """.format(k0, group, k0)
                    )
                    raise Exception(msg)
                ddata[k0]['ref'] = (k0,)
                c0 = (
                    (lref_add is None or k0 not in lref_add)
                    and k0 not in dref0.keys()
                )
                if c0:
                    if lref_add is None:
                        lref_add = [k0]
                    else:
                        lref_add.append(k0)

    # Add missing refs (only in ddata)
    dgroup_add = None
    dref_add = None
    lref = list(itt.chain.from_iterable([
        [
            rr for rr in v0['ref']
            if rr not in dref0.keys() and rr in ddata.keys()
        ]
        for v0 in ddata.values() if (
            'ref' in v0.keys() and v0.get('data') is not None
        )
    ]))

    if lref_add is not None:
        lref += lref_add

    if len(lref) > 0:
        lref = set(lref)
        dref_add = {rr: {'data': ddata[rr]['data']} for rr in lref}
        dref_add, dgroup_add, ddata_dadd = _check_dref(
            dref=dref_add, dref0=dref0, ddata0=ddata0, dgroup0=dgroup0,
            allowed_groups=allowed_groups,
        )

    # Check shape vs ref
    for k0, v0 in ddata.items():
        if v0.get('data') is None:
            continue
        c0 = (
            isinstance(v0['ref'], tuple)
            and all([
                ss in dref0.keys()
                or (dref_add is not None and ss in dref_add.keys())
                for ss in v0['ref']
            ])
        )
        if not c0:
            msg = (
                f"ddata['{k0}']['ref'] contains unknown ref:\n"
                f"\t- ddata['{k0}']['ref'] = {v0['ref']}\n"
                f"\t- dref0.keys() = {sorted(dref0.keys())}\n"
                + "\t- dref_add.keys() = {}".format(
                    None if dref_add is None else sorted(dref_add.keys())
                )
            )
            raise Exception(msg)
        if c0:
            if isinstance(v0['shape'], tuple):
                shaperef = [
                    dref0[rr]['size'] if rr in dref0.keys()
                    else dref_add[rr]['size']
                    for rr in v0['ref']
                ]
                c1 = (
                    len(shaperef) > 1
                    or any([isinstance(ss, tuple) for ss in shaperef])
                )
                if c1:
                    shaperef = np.r_[tuple(shaperef)].ravel()
                shaperef = tuple(shaperef)
                c0 = c0 and shaperef == v0['shape']
            else:
                c0 = v0['ref'] == (k0,)

        # Raise Exception if needed
        if not c0:
            if isinstance(v0['shape'], tuple):
                msg = (
                    """
                    Inconsistent shape vs ref for ddata[{0}]:
                        - ddata['{0}']['ref'] = {1}  ({2})
                        - ddata['{0}']['shape'] = {3}

                    If dict / object it should be its own ref!
                    """.format(k0, v0['ref'], shaperef, v0['shape'])
                )
            else:
                msg = (
                    "ddata[{0}]['ref'] != ({0},)".format(k0)
                    + "\n\t- ddata[{}]['ref'] = {}\n\n".format(k0, v0['ref'])
                    + "... or there might be an issue with:\n"
                    + "\t- type(ddata[{}]['shape']) = {} ({})".format(
                        k0, type(v0['shape']), v0['shape'],
                    )
                )
            raise Exception(msg)

    return ddata, dref_add, dgroup_add


# #############################################################################
# #############################################################################
#                           dobj
# #############################################################################


def _check_dobj(
    dobj=None, dobj0=None,
):

    # ----------------
    # Trivial case
    if dobj in [None, {}]:
        return {}

    # ----------------
    # Check conformity

    # map possible non-conformities
    if not isinstance(dobj, dict):
        msg = (
            "Arg dobj must be a dict!\n"
            "\t- Provided: {}".format(type(dobj))
        )
        raise Exception(msg)

    # Map possible non-conformities
    dc = {}
    for k0, v0 in dobj.items():
        c1 = isinstance(k0, str) and isinstance(v0, dict)
        if not c1:
            dc[k0] = "type(key) != str or type(value) != dict"
            continue

        if k0 not in dobj0.keys():
            lc2 = [k1 for k1 in v0.keys() if not isinstance(k1, str)]
            if len(lc2) > 0:
                dc[k0] = (
                    "The following keys of dobj[{}] are not str:\n".format(k0)
                    + "\n\t- "
                    + "\n\t- ".join(lc2)
                )
                continue
        else:
            lc2 = [
                k1 for k1 in v0.keys()
                if not isinstance(k1, str)
                or k1 in dobj0[k0].keys()
            ]
            if len(lc2) > 0:
                dc[k0] = (
                    "The following keys of dobj[{}] are not str:\n".format(k0)
                    + "\n\t- "
                    + "\n\t- ".join(lc2)
                    + "(or they are already in dobj0[{}]".format(k0)
                )

    # Raise Exception
    if len(dc) > 0:
        msg = (
            "The following keys of dobj are non-conform:\n"
            + "\n\n".join([
                'dobj[{}]: {}'.format(k0, v0) for k0, v0 in dc.items()
            ])
        )
        raise Exception(msg)

    return dobj


# #############################################################################
# #############################################################################
#                           Params
# #############################################################################


def _check_elementioncharge(
    ION=None, ion=None,
    element=None, charge=None,
    warn=None,
):
    """ Specific to SpectralLines """

    if warn is None:
        warn = True

    # Assess if relevant
    lc = [
        ION is not None,
        ion is not None,
        element is not None and charge is not None,
    ]
    if not any(lc):
        if warn is True:
            msg = (
                """
                To determine ION, ion, element and charge, provide either:
                - ION:  {}
                - ion:  {}
                - element and charge: {}, {}
                """.format(ION, ion, element, charge)
            )
            warnings.warn(msg)
        return None, None, None, None

    # Get element and charge from ION if any
    if lc[0] or lc[1]:
        indc = 1
        if (lc[0] and ION[1].islower()) or (lc[1] and ion[1].islower()):
            indc = 2

        # Infer element
        elementi = ION[:indc] if lc[0] else ion[:indc]
        if element is not None and element != elementi:
            msg = (
                """
                Inconsistent ION ({}) vs element ({})
                """.format(element, elementi)
            )
            raise Exception(msg)

        # Infer charge
        if lc[0]:
            chargei = roman2int(ION[indc:]) - 1
        else:
            chargei = int(ion[indc:].replace('+', ''))
        if charge is not None and charge != chargei:
            msg = (
                """
                Inconsistent ION ({}) vs charge ({})
                """.format(charge, chargei)
            )
            raise Exception(msg)
        element = elementi
        charge = chargei
        if lc[0]:
            ioni = '{}{}+'.format(element, charge)
            if lc[1] and ioni != ion:
                msg = (
                    """
                    Inconsistent ION ({}) vs ion ({})
                    """.format(ION, ion)
                )
                raise Exception(msg)
            ion = ioni

        elif lc[1]:
            IONi = '{}{}'.format(element, int2roman(charge+1))
            if lc[0] and IONi != ION:
                msg = (
                    """
                    Inconsistent ion ({}) vs ION ({})
                    """.format(ion, ION)
                )
                raise Exception(msg)
            ION = IONi

    # ion provided -> element and charge
    elif lc[2]:
        ioni = '{}{}+'.format(element, charge)
        IONi = '{}{}'.format(element, int2roman(charge+1))
        if ion is not None and ion != ioni:
            msg = (
                """
                Inconsistent (element, charge) ({}, {}) vs ion ({})
                """.format(element, charge, ion)
            )
            raise Exception(msg)
        if ION is not None and ION != IONi:
            msg = (
                """
                Inconsistent (element, charge) ({}, {}) vs ION ({})
                """.format(element, charge, ION)
            )
            raise Exception(msg)
        ion = ioni
        ION = IONi

    return ION, ion, element, charge


def _check_elementioncharge_dict(dref_static):
    """ Specific to SpectralLines """

    # Assess if relevant
    lk = [kk for kk in ['ion', 'ION'] if kk in dref_static.keys()]
    if len(lk) == 0:
        return
    kion = lk[0]
    kION = 'ION' if kion == 'ion' else 'ion'
    if kion == 'ION':
        dref_static['ion'] = {}

    lerr = []
    for k0, v0 in dref_static[kion].items():
        try:
            if kion == 'ION':
                ION, ion, element, charge = _check_elementioncharge(
                    ION=k0,
                    ion=v0.get('ion'),
                    element=v0.get('element'),
                    charge=v0.get('charge'),
                )
            else:
                ION, ion, element, charge = _check_elementioncharge(
                    ION=v0.get('ION'),
                    ion=k0,
                    element=v0.get('element'),
                    charge=v0.get('charge'),
                )

            if ION is None:
                continue
            if kion == 'ION':
                dref_static['ion'][ion] = {
                    'ION': ION,
                    'element': element,
                    'charge': charge,
                }
            else:
                dref_static['ion'][k0]['ION'] = ION
                dref_static['ion'][k0]['element'] = element
                dref_static['ion'][k0]['charge'] = charge

        except Exception as err:
            lerr.append((k0, str(err)))

    if kion == 'ION':
        del dref_static['ION']

    if len(lerr) > 0:
        lerr = ['\t- {}: {}'.format(pp[0], pp[1]) for pp in lerr]
        msg = (
            """
            The following entries have non-conform ion / ION / element / charge
            {}
            """.format('\n'.join(lerr))
        )
        raise Exception(msg)


def _harmonize_params(
    dd=None,
    dd_name=None,
    dd_name2=None,
    dref_static=None,
    lkeys=None,
    reserved_keys=None,
    ddefparams=None,
):

    # Check inputs
    if dd_name2 is None:
        dd_name2 = dd_name
    if reserved_keys is None:
        reserved_keys = _DRESERVED_KEYS[dd_name]
    if ddefparams is None:
        ddefparams = _DDEF_PARAMS[dd_name]

    # ------------------
    # list of param keys

    # Get list of known param keys
    lparams = set(itt.chain.from_iterable([
        [k1 for k1 in v0.keys() if k1 not in reserved_keys]
        for k0, v0 in dd.items()
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
        for k1, v1 in dd.items():
            if k0 not in v1.keys():
                dd[k1][k0] = v0[1]
            else:
                # Check type if already included
                if not isinstance(dd[k1][k0], v0[0]):
                    msg = (
                        """
                        Wrong type for parameter:
                            - type({}[{}][{}]) = {}
                        - Expected: {}
                        """.format(
                            dd_name2, k1, k0, type(dd[k1][k0]), v0[0],
                        )
                    )
                    raise Exception(msg)

    for k0 in lparams:
        for k1, v1 in dd.items():
            dd[k1][k0] = dd[k1].get(k0)

    # ------------------
    # Check against dref_static0
    lkpout = [
        (k0, (k1, v0[k1]))
        for k0, v0 in dd.items()
        if k1 in dref_static.keys()
        and any([v0[k1] not in dref_static[k1].keys() for k1 in lparams])
    ]
    if len(lkpout) > 0:
        lpu = sorted(set([pp[1][0] for pp in lkpout]))
        msg0 = '\n'.join([
            '\t- {}[{}]: {}'.format(pp[0], pp[1], pp[2]) for pp in lkpout
        ])
        msg1 = '\n'.join([
            '\t- dref_static[{}]: {}'.format(pp, dref_static[pp].keys())
            for pp in lpu
        ])
        msg = (
            """
            The following parameter have non-identified values in ref_static:
            {}

            Available values:
            {}
            """.format(msg0, msg1)
        )
        raise Exception(msg)

    return dd


def _update_dref_static0(dref_static0=None, ddata0=None, dobj0=None):
    """ Count nb. of matching ref_static in ddata and dobj """

    for k0, v0 in dref_static0.items():

        # ddata
        dd = {
            k2: np.sum([ddata0[k3].get(k0) == k2 for k3 in ddata0.keys()])
            for k2 in v0.keys()
            if any([ddata0[k3].get(k0) == k2 for k3 in ddata0.keys()])
        }
        if len(dd) > 0:
            ss = 'nb. data'
            for k2, v2 in v0.items():
                dref_static0[k0][k2][ss] = int(dd.get(k2, 0))

        # dobj
        for k1, v1 in dobj0.items():
            dd = {
                k2: np.sum([v1[k3].get(k0) == k2 for k3 in v1.keys()])
                for k2 in v0.keys()
                if any([v1[k3].get(k0) == k2 for k3 in v1.keys()])
            }
            if len(dd) > 0:
                ss = 'nb. {}'.format(k1)
                for k2, v2 in v0.items():
                    dref_static0[k0][k2][ss] = int(dd.get(k2, 0))


# #############################################################################
# #############################################################################
#                           Consistency
# #############################################################################


def _consistency(
    dobj=None, dobj0=None,
    ddata=None, ddata0=None,
    dref=None, dref0=None,
    dref_static=None, dref_static0=None,
    dgroup=None, dgroup0=None,
    allowed_groups=None,
    reserved_keys=None,
    ddefparams_data=None,
    ddefparams_obj=None,
    data_none=None,
    max_ndim=None,
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
        allowed_groups=allowed_groups,
    )
    if dgroup_add is not None:
        dgroup0.update(dgroup_add)
    if ddata_add is not None:
        if ddata is None:
            ddata = ddata_add
        else:
            ddata.update(ddata_add)
    dref0.update(dref)

    # --------------
    # dref_static
    dref_static = _check_dref_static(
        dref_static=dref_static, dref_static0=dref_static0,
    )
    for k0, v0 in dref_static.items():
        if k0 not in dref_static0.keys():
            dref_static0[k0] = v0
        else:
            dref_static0[k0].update(v0)

    # --------------
    # ddata
    ddata, dref_add, dgroup_add = _check_ddata(
        ddata=ddata, ddata0=ddata0,
        dref0=dref0, dgroup0=dgroup0,
        reserved_keys=reserved_keys, allowed_groups=allowed_groups,
        data_none=data_none, max_ndim=max_ndim,
    )
    if dgroup_add is not None:
        dgroup0.update(dgroup_add)
    if dref_add is not None:
        dref0.update(dref_add)
    ddata0.update(ddata)

    # -----------------
    # dobj
    dobj = _check_dobj(
        dobj=dobj, dobj0=dobj0,
    )
    for k0, v0 in dobj.items():
        if k0 not in dobj0.keys():
            dobj0[k0] = v0
        else:
            dobj0[k0].update(v0)

    # --------------
    # params harmonization - ddata
    ddata0 = _harmonize_params(
        dd=ddata0,
        dd_name='ddata',
        dref_static=dref_static0,
        ddefparams=ddefparams_data, reserved_keys=reserved_keys,
    )

    # --------------
    # params harmonization - dobj
    for k0, v0 in dobj0.items():
        dobj0[k0] = _harmonize_params(
            dd=v0,
            dd_name='dobj',
            dd_name2='dobj[{}]'.format(k0),
            dref_static=dref_static0,
            ddefparams=ddefparams_obj.get(k0),
            reserved_keys=reserved_keys,
        )

    # --------------
    # Complement

    # ddata0
    for k0, v0 in ddata0.items():
        if v0.get('data') is None:
            continue
        ddata0[k0]['group'] = tuple([dref0[rr]['group'] for rr in v0['ref']])

    # dref0
    for k0, v0 in dref0.items():
        dref0[k0]['ldata'] = sorted(set(
            k1 for k1 in ddata0.keys()
            if ddata0[k1].get('data') is not None and k0 in ddata0[k1]['ref']
        ))

    # dgroup0
    for k0, v0 in dgroup0.items():
        dgroup0[k0]['lref'] = sorted(set(
            k1 for k1, v1 in dref0.items() if v1['group'] == k0
        ))
        dgroup0[k0]['ldata'] = sorted(set(
            k1 for k1 in ddata0.keys()
            if ddata0[k1].get('data') is not None and k0 in ddata0[k1]['group']
        ))

    # dref_static0
    _update_dref_static0(dref_static0=dref_static0, ddata0=ddata0, dobj0=dobj0)

    # --------------
    # Check conventions
    for k0, v0 in ddata0.items():
        if v0.get('data') is None:
            continue
        if 'time' in v0['group'] and v0['group'].index('time') != 0:
            msg = (
                "ref 'time' must be placed at dimension 0!\n"
                + "\t- ddata['{}']['ref'] = {}\n".format(k0, v0['ref'])
                + "\t- ddata['{}']['group'] = {}".format(k0, v0['group'])
            )
            raise Exception(msg)

    return dgroup0, dref0, dref_static0, ddata0, dobj0


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
#               Switch ref
# #############################################################################


def switch_ref(
    new_ref=None,
    ddata=None,
    dref=None,
    dgroup=None,
    dobj0=None,
    dref_static0=None,
    allowed_groups=None,
    reserved_keys=None,
    ddefparams_data=None,
    data_none=None,
    max_ndim=None,
):
    """Use the provided key as ref (if valid) """

    # Check input
    c0 = (
        new_ref in ddata.keys()
        and ddata[new_ref].get('monot') == (True,)
    )
    if not c0:
        strgroup = [
            '{}: {}'.format(
                k0,
                [
                    k1 for k1 in v0['ldata']
                    if ddata[k1].get('monot') == (True,)
                ]
            )
            for k0, v0 in dgroup.items()
        ]
        msg = (
            "\nArg new_ref must be a key to a valid ref (monotonous)!\n"
            + "\t- Provided: {}\n\n".format(new_ref)
            + "Available valid ref candidates:\n"
            + "\t- {}".format('\n\t- '.join(strgroup))
        )
        raise Exception(msg)

    # Substitute in dref
    old_ref = ddata[new_ref]['ref'][0]
    dref[new_ref] = dict(dref[old_ref])
    del dref[old_ref]

    # substitute in ddata['ref']
    for k0, v0 in ddata.items():
        if v0.get('ref') is not None and old_ref in v0['ref']:
            new = tuple([rr for rr in v0['ref']])
            ddata[k0]['ref'] = tuple([
                new_ref if rr == old_ref else rr
                for rr in v0['ref']
            ])

    return _consistency(
        ddata=ddata, ddata0={},
        dref=dref, dref0={},
        dgroup=dgroup, dgroup0={},
        dobj=None, dobj0=dobj0,
        dref_static=None, dref_static0=dref_static0,
        allowed_groups=None,
        reserved_keys=None,
        ddefparams_data=ddefparams_data,
        ddefparams_obj=None,
        data_none=None,
        max_ndim=None,
    )


# #############################################################################
# #############################################################################
#               Get / set / add / remove param
# #############################################################################


def _get_param(
    dd=None, dd_name=None,
    param=None, key=None, ind=None,
    returnas=None,
):
    """ Return the array of the chosen parameter (or list of parameters)

    Can be returned as:
        - dict: {param0: {key0: values0, key1: value1...}, ...}
        - np[.ndarray: {param0: np.r_[values0, value1...], ...}

    """

    # Trivial case
    lp = [kk for kk in list(dd.values())[0].keys() if kk != 'data']
    if param is None:
        param = lp

    # Get key (which data to return param for)
    key = _ind_tofrom_key(dd=dd, key=key, ind=ind, returnas=str)

    # ---------------
    # Check inputs

    # param
    lc = [
        isinstance(param, str) and param in lp and param != 'data',
        isinstance(param, list)
        and all([isinstance(pp, str) and pp in lp for pp in param])
    ]
    if not any(lc):
        msg = (
            "Arg param must a valid param key of a list of such "
            + "(except 'data')\n\n"
            + "Valid params:\n\t- {}\n\n".format('\n\t- '.join(lp))
            + "Provided:\n\t- {}\n".format(param)
        )
        raise Exception(msg)

    if lc[0]:
        param = [param]

    # returnas
    if returnas is None:
        returnas = np.ndarray

    c0 = returnas in [np.ndarray, dict]
    if not c0:
        msg = (
            """
            Arg returnas must be in [np.ndarray, dict]
            Provided: {}
            """.format(returnas)
        )
        raise Exception(msg)

    # -------------
    # Get output

    if returnas == dict:
        out = {k0: {k1: dd[k1][k0] for k1 in key} for k0 in param}
    else:
        out = {k0: np.array([dd[k1][k0] for k1 in key]) for k0 in param}

    return out


def _set_param(
    dd=None, dd_name=None,
    param=None, value=None,
    ind=None, key=None,
):
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
    lp = [kk for kk in list(dd.values())[0].keys()]
    if dd_name == 'ddata':
        lp.remove('data')
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
    key = _ind_tofrom_key(dd=dd, ind=ind, key=key, returnas='key')

    # Check value
    ltypes = [str, int, np.integer, float, np.floating, tuple]
    lc = [
        isinstance(value, tuple(ltypes)),
        isinstance(value, list) and all([type(tt) in ltypes for tt in value])
        and len(value) == len(key),
        isinstance(value, np.ndarray) and value.shape[0] == len(key),
        isinstance(value, dict)
        and all([
            kk in dd.keys() and type(vv) in ltypes
            for kk, vv in value.items()
        ])
    ]
    if not (value is None or any(lc)):
        msg = (
            """
            Accepted types for value include:
                - None
                - {}: common to all
                - list, np.ndarray: key by key
                - dict of {key: scalar / str}

            The length of value must match the selected keys ({})
            """.format(ltypes, len(key))
        )
        raise Exception(msg)

    # Update data
    if value is None or lc[0]:
        for kk in key:
            dd[kk][param] = value
    elif lc[1] or lc[2]:
        for ii, kk in enumerate(key):
            dd[kk][param] = value[ii]
    else:
        for kk, vv in value.items():
            dd[kk][param] = vv


def _add_param(
    dd=None, dd_name=None,
    param=None, value=None,
):
    """ Add a parameter, optionnally also set its value """
    lp = [kk for kk in list(dd.values())[0].keys()]
    if dd_name == 'ddata':
        lp.remove('data')

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
    for kk in dd.keys():
        dd[kk][param] = None
    _set_param(dd=dd, param=param, value=value)


def _remove_param(dd=None, dd_name=None, param=None):
    """ Remove a parameter, none by default, all if param = 'all' """

    # Check inputs
    lp = [kk for kk in list(dd.values())[0].keys() if kk != 'data']
    if param is None:
        return
    if param == 'all':
        param = lp

    c0 = isinstance(param, str) and param in lp
    if not c0:
        msg = "Param {} is not a parameter of {}!".format(param, dd_name)
        raise Exception(msg)

    # Remove
    for k0 in dd.keys():
        del dd[k0][param]


# #############################################################################
# #############################################################################
#               Selection
# #############################################################################


def _ind_tofrom_key(
    dd=None, dd_name=None, dgroup=None,
    ind=None, key=None, group=None, returnas=int,
):

    # --------------------
    # Check / format input
    lc = [ind is not None, key is not None]
    if not np.sum(lc) <= 1:
        msg = ("Args ind and key cannot be prescribed simultaneously!")
        raise Exception(msg)

    if dd_name == 'ddata' and group is not None:
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
    out = np.zeros((len(dd),), dtype=bool)

    if not any(lc) and dd_name == 'ddata' and group is not None:
        key = dgroup[group]['ldata']
        lc[1] = True

    # Get output
    lk = list(dd.keys())
    if lc[0]:

        # Check ind
        if not isinstance(ind, np.ndarray):
            ind = np.atleast_1d(ind).ravel()
        c0 = (
            ind.ndim == 1
            and (
                (ind.dtype == np.bool and ind.size == len(dd))
                or (ind.dtype == np.int and ind.size <= len(dd))
            )
        )
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
            and all([isinstance(kk, str) and kk in lk for kk in key])
        )
        if not c0:
            msg = (
                """
                key must be valid key to {} (or list of such)
                Provided: {}
                """.format(dd_name, key)
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


def _select(dd=None, dd_name=None, log=None, returnas=None, **kwdargs):
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
    if log is None:
        log = 'all'
    if returnas is None:
        returnas = bool if log == 'raw' else int
    if log not in ['all', 'any', 'raw']:
        msg = (
            "Arg log must be:\n"
            + "\t- 'all': all criteria should match\n"
            + "\t- 'any': any criterion should match\n"
            + "\t- 'raw': return the full 2d array of boolean indices\n\n"
            + "Provided:\n\t{}".format(log)
        )
        raise Exception(msg)
    if returnas not in [int, bool, str, 'key']:
        msg = (
            "Arg returnas must be:\n"
            + "\t- bool: array of boolean indices\n"
            + "\t- int: array of int indices\n"
            + "\t- str / 'key': array of keys\n\n"
            + "Provided:\n\t{}".format(returnas)
        )
        raise Exception(msg)

    kwdargs = {k0: v0 for k0, v0 in kwdargs.items() if v0 is not None}

    # Get list of relevant criteria
    lp = [kk for kk in list(dd.values())[0].keys()]
    if dd_name == 'ddata':
        lp.remove('data')

    lcritout = [ss for ss in kwdargs.keys() if ss not in lp]
    if len(lcritout) > 0:
        msg = (
            """
            The following criteria correspond to no parameters:
                - {}
              => only use known parameters (self.dparam_{}.keys()):
                - {}
            """.format(lcritout, dd_name, '\n\t- '.join(lp))
        )
        raise Exception(msg)

    # Prepare array of bool indices and populate
    ltypes = [float, np.float_]
    lquant = [
        kk for kk in kwdargs.keys()
        if any([type(dd[k0][kk]) in ltypes for k0 in dd.keys()])
    ]

    ind = np.zeros((len(kwdargs), len(dd)), dtype=bool)
    for ii, kk in enumerate(kwdargs.keys()):
        try:
            par = _get_param(
                dd=dd, dd_name=dd_name,
                param=kk,
                returnas=np.ndarray,
            )[kk]
            if kk in lquant:
                if isinstance(kwdargs[kk], list) and len(kwdargs[kk]) == 2:
                    ind[ii, :] = (
                        (kwdargs[kk][0] <= par) & (par <= kwdargs[kk][1])
                    )
                elif isinstance(kwdargs[kk], tuple) and len(kwdargs[kk]) == 2:
                    ind[ii, :] = (
                        (kwdargs[kk][0] > par) | (par > kwdargs[kk][1])
                    )
                else:
                    ind[ii, :] = par == kwdargs[kk]
            else:
                ind[ii, :] = par == kwdargs[kk]
        except Exception as err:
            try:
                ind[ii, :] = [
                    dd[k0][kk] == kwdargs[kk] for k0 in dd.keys()
                ]
            except Exception as err:
                msg = (
                    "Could not determine whether:\n"
                    + "\t- {}['{}'] == {}".format(
                        dd_name, kk, kwdargs[kk],
                    )
                )
                raise Exception(msg)

    # Format output ind
    if log == 'raw':
        if returnas in [str, 'key']:
            ind = {
                kk: [k0 for jj, k0 in enumerate(dd.keys()) if ind[ii, jj]]
                for ii, kk in enumerate(kwdargs.keys())
            }
        if returnas == int:
            ind = {
                kk: ind[ii, :].nonzero()[0]
                for ii, kk in enumerate(kwdargs.keys())
            }
        else:
            ind = {kk: ind[ii, :] for ii, kk in enumerate(kwdargs.keys())}
    else:
        if log == 'all':
            ind = np.all(ind, axis=0)
        else:
            ind = np.any(ind, axis=0)
        if returnas == int:
            ind = ind.nonzero()[0]
        elif returnas in [str, 'key']:
            ind = np.array(
                [k0 for jj, k0 in enumerate(dd.keys()) if ind[jj]],
                dtype=str,
            )
    return ind


def _get_keyingroup_ddata(
    dd=None, dd_name='data',
    key=None, group=None, monot=None,
    msgstr=None, raise_=False,
):
    """ Return the unique data key matching key in desired group in ddata

    Here, key can be interpreted as name / source / units / quant...
    All are tested using select() and a unique match is returned
    If not unique match an error message is either returned or raised

    """

    # ------------------------
    # Trivial case: key is actually a ddata key
    if key in dd.keys():
        lg = dd[key]['group']
        if group is None or group in lg:
            return key, None
        else:
            msg = ("Required data key does not have matching group:\n"
                   + "\t- {}['{}']['group'] = {}\n".format(dd_name, key, lg)
                   + "\t- Expected group:  {}".format(group))
            if raise_:
                raise Exception(msg)

    # ------------------------
    # Non-trivial: check for a unique match on other params

    dind = _select(
        dd=dd, dd_name=dd_name,
        dim=key, quant=key, name=key, units=key, source=key,
        group=group, monot=monot,
        log='raw', returnas=bool,
    )
    ind = np.array([ind for kk, ind in dind.items() if kk != 'group'])
    if group is not None:
        ind &= dind['group'][None, :]

    # Any perfect match ?
    nind = np.sum(ind, axis=1)
    sol = (nind == 1).nonzero()[0]
    key_out, msg = None, None
    if sol.size > 0:
        if np.unique(sol).size == 1:
            indkey = ind[sol[0], :].nonzero()[0]
            key_out = list(dd.keys())[indkey]
        else:
            lstr = "[dim, quant, name, units, source]"
            msg = "Several possible matches in {} for {}".format(lstr, key)
    else:
        lstr = "[dim, quant, name, units, source]"
        msg = "No match in {} for {} in group {}".format(lstr, key, group)

    # Complement error msg and optionally raise
    if msg is not None:
        lk = ['dim', 'quant', 'name', 'units', 'source']
        dk = {
            kk: (
                dind[kk].sum(),
                sorted(set([vv[kk] for vv in dd.values()]))
            ) for kk in lk
        }
        msg += (
            "\n\nRequested {} could not be identified!\n".format(msgstr)
            + "Please provide a valid (unique) key/name/dim/quant/units:\n\n"
            + '\n'.join([
                '\t- {} ({} matches): {}'.format(kk, dk[kk][0], dk[kk][1])
                for kk in lk
            ])
            + "\nProvided:\n\t'{}'".format(key)
        )
        if raise_:
            raise Exception(msg)
    return key_out, msg


def _get_possible_ref12d(
    dd=None,
    key=None, ref1d=None, ref2d=None,
    group1d='radius',
    group2d='mesh2d',
):

    # Get relevant lists
    kq, msg = _get_keyingroup_ddata(
        dd=dd,
        key=key, group=group2d, msgstr='quant', raise_=False,
    )

    if kq is not None:
        # The desired quantity is already 2d
        k1d, k2d = None, None

    else:
        # Check if the desired quantity is 1d
        kq, msg = _get_keyingroup_ddata(
            dd=dd,
            key=key, group=group1d,
            msgstr='quant', raise_=True,
        )

        # Get dict of possible {ref1d: lref2d}
        ref = [rr for rr in dd[kq]['ref'] if dd[rr]['group'] == (group1d,)][0]
        lref1d = [
            k0 for k0, v0 in dd.items()
            if ref in v0['ref'] and v0['monot'][v0['ref'].index(ref)] is True
        ]

        # Get matching ref2d with same quant and good group
        lquant = list(set([dd[kk]['quant'] for kk in lref1d]))
        dref2d = {
            k0: [
                kk for kk in _select(
                    dd=dd, quant=dd[k0]['quant'],
                    log='all', returnas=str,
                )
                if group2d in dd[kk]['group']
                and not isinstance(dd[kk]['data'], dict)
            ]
            for k0 in lref1d
        }
        dref2d = {k0: v0 for k0, v0 in dref2d.items() if len(v0) > 0}

        if len(dref2d) == 0:
            msg = (
                "No match for (ref1d, ref2d) for ddata['{}']".format(kq)
            )
            raise Exception(msg)

        # check ref1d
        if ref1d is None:
            if ref2d is not None:
                lk = [k0 for k0, v0 in dref2d.items() if ref2d in v0]
                if len(lk) == 0:
                    msg = (
                        "\nNon-valid interpolation intermediate\n"
                        + "\t- provided:\n"
                        + "\t\t- ref1d = {}, ref2d = {}\n".format(ref1d, ref2d)
                        + "\t- valid:\n{}".format(
                            '\n'.join([
                                '\t\t- ref1d = {}  =>  ref2d in {}'.format(
                                    k0, v0
                                )
                                for k0, v0 in dref2d.items()
                            ])
                        )
                    )
                    raise Exception(msg)
                if kq in lk:
                    ref1d = kq
                else:
                    ref1d = lk[0]
            else:
                if kq in dref2d.keys():
                    ref1d = kq
                else:
                    ref1d = list(dref2d.keys())[0]
        else:
            ref1d, msg = _get_keyingroup_ddata(
                dd=dd,
                key=ref1d, group=group1d,
                msgstr='ref1d', raise_=False,
            )
        if ref1d not in dref2d.keys():
            msg = (
                "\nNon-valid interpolation intermediate\n"
                + "\t- provided:\n"
                + "\t\t- ref1d = {}, ref2d = {}\n".format(ref1d, ref2d)
                + "\t- valid:\n{}".format(
                    '\n'.join([
                        '\t\t- ref1d = {}  =>  ref2d in {}'.format(
                            k0, v0
                        )
                        for k0, v0 in dref2d.items()
                    ])
                )
            )
            raise Exception(msg)

        # check ref2d
        if ref2d is None:
            ref2d = dref2d[ref1d][0]
        else:
            ref2d, msg = _get_keyingroup_ddata(
                dd=dd,
                key=ref2d, group=group2d,
                msgstr='ref2d', raise_=False,
            )
        if ref2d not in dref2d[ref1d]:
            msg = (
                "\nNon-valid interpolation intermediate\n"
                + "\t- provided:\n"
                + "\t\t- ref1d = {}, ref2d = {}\n".format(ref1d, ref2d)
                + "\t- valid:\n{}".format(
                    '\n'.join([
                        '\t\t- ref1d = {}  =>  ref2d in {}'.format(
                            k0, v0
                        )
                        for k0, v0 in dref2d.items()
                    ])
                )
            )
            raise Exception(msg)

    return kq, ref1d, ref2d
