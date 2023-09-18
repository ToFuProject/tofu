

# Standard
import itertools as itt
import warnings


# Common
import numpy as np
import scipy.sparse as scpsp
import datastock as ds


_DRESERVED_KEYS = {
    'dref': ['ldata', 'ldata_monot', 'size', 'ind'],
    'dstatic': [],
    'ddata': ['ref', 'shape', 'data'],
    'dobj': [],
}
_LRESERVED_KEYS = list(set(itt.chain.from_iterable([
    v0 for v0 in _DRESERVED_KEYS.values()
])))


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
    'dstatic': {
    },
}


_IREF = 'iref'
_IDATA = 'data'


_DATA_NONE = False


# #############################################################################
# #############################################################################
#                           Generic
# #############################################################################


def _check_which(
    dref=None,
    ddata=None,
    dobj=None,
    dstatic=None,
    which=None,
    return_dict=None,
):
    """ Check which in ['ref', 'data'] + dobj.keys() + dstatic.keys()

    Optionally return the dict itself, by reference

    """

    # --------------
    # Check inputs

    return_dict = ds._generic_check._check_var(
        return_dict,
        'return_dict',
        types=bool,
        default=True,
    )

    lkobj = list(dobj.keys())
    lkstatic = list(dstatic.keys())
    lkok = ['ref', 'data'] + lkobj + lkstatic
    which = ds._generic_check._check_var(
        which,
        'which',
        types=str,
        allowed=lkok,
        default='data',
    )

    # -----------------
    # return right dict

    if return_dict is True:
        if which == 'ref':
            dd = dref
        elif which == 'data':
            dd = ddata
        elif which in lkobj:
            dd = dobj[which]
        else:
            dd = dstatic[which]
        return which, dd
    else:
        return which


def _check_conflicts(dd=None, dd0=None, dd_name=None, returnas=None):
    """ Detect conflict with existing entries

    Any pre-existing entry will trigger either an update or a conflict

    - conflic: same parameter, different value
    - update: new parameter
    - retro: parameter not filled (None)

    """

    # ------------
    # check inputs

    if returnas is None:
        returnas = False

    # ----------------------------
    # detect conflicts and updates

    dupdate = {}
    dconflict = {}
    for k0, v0 in dd.items():

        # k0 not in existing dict => ok
        if k0 not in dd0.keys():
            continue

        # find conflicts (same key and same parameters with different values)
        lk = set(v0.keys()).intersection(dd0[k0].keys())
        lk = [
            kk for kk in lk
            if not (
                isinstance(v0[kk], dd0[k0][kk].__class__)
                and (
                    (
                        isinstance(v0[kk], np.ndarray)
                        and v0[kk].shape == dd0[k0][kk].shape
                        and np.allclose(v0[kk], dd0[k0][kk], equal_nan=True)
                    )
                    or (
                        scpsp.issparse(v0[kk])
                        and v0[kk].shape == dd0[k0][kk].shape
                        and np.allclose(
                            v0[kk].data,
                            dd0[k0][kk].data,
                            equal_nan=True,
                        )
                    )
                    or (
                        not isinstance(v0[kk], np.ndarray)
                        and not scpsp.issparse(v0[kk])
                        and v0[kk] == dd0[k0][kk]
                    )
                    or (
                        v0[kk] == dd0[k0][kk]
                    )
                )
            )
        ]
        if len(lk) > 0:
            dconflict[k0] = lk

        # find updates (same key but new parameters)
        lkup = [
            kk for kk in v0.keys()
            if kk not in lk
            and kk not in dd0[k0].keys()
            # and kk not in ['ldata', 'size']
            and kk not in _LRESERVED_KEYS
        ]
        if len(lkup) > 0:
            dupdate[k0] = lkup

    # ---------------
    # raise or return

    if returnas is False:
        # Conflicts => Exception
        if len(dconflict) > 0:
            lstr = [
                f"\t- {dd_name}['{k0}']: {v0}"
                for k0, v0 in dconflict.items()
            ]
            msg = (
                f"Conflicts with pre-existing values found in {dd_name}:\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

        # Updates => Warning
        if len(dupdate) > 0:
            lstr = [
                f"\t- {dd_name}['{k0}']: {v0}"
                for k0, v0 in dupdate.items()
            ]
            msg = (
                f"\nExisting {dd_name} keys updated with new keys:\n"
                + "\n".join(lstr)
            )
            warnings.warn(msg)

    else:
        return dconflict, dupdate


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
            \t- provided: '{}'
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


def _get_whichorkey(key=None, which=None, din=None, dname=None):

    # -------------
    # trivial check

    lc = [
        key is not None,
        which is not None,
    ]
    if np.sum(lc) == 0:
        msg = "Please provide either key or/xor which!"
        raise Exception(msg)

    # ----------
    # which only

    if lc[1]:
        which = ds._generic_check._check_var(
            which, 'which',
            allowed=sorted(din.keys()),
        )

    # ---------------------------------------
    # key but not which => check and set which

    elif lc[0] and not lc[1]:

        if hasattr(key, '__iter__'):
            lwhich = [
                k0 for k0, v0 in din.items()
                if key in v0.keys()
            ]
        else:
            lwhich = [
                k0 for k0, v0 in din.items()
                if all([kk in v0.keys() for kk in key])
            ]

        if len(lwhich) == 1:
            which = lwhich[0]

        else:
            msg = (
                f"key {key} has no / several matches in {dname}:\n"
                f"\t- matches: {lwhich}"
            )
            raise Exception(msg)

    if key is not None:
        key = ds._generic_check._check_var_iter(
            key, 'key',
            types=list,
            allowed=sorted(din[which]),
        )

    return key, which


def _remove_ref(
    # key to remove
    key=None,
    # dict
    dref0=None,
    ddata0=None,
    dstatic0=None,
    dobj0=None,
    # parameters
    propagate=None,
    reserved_keys=None,
    ddefparams_data=None,
    ddefparams_obj=None,
    ddefparams_static=None,
    data_none=None,
    max_ndim=None,
):
    """ Remove a ref (or list of refs) and all associated data """

    # trivial case
    if key is None:
        return dref0, dstatic0, ddata0, dobj0

    # check input
    key = _check_remove(
        key=key, dkey=dref0, name='ref',
    )

    for k0 in key:
        # Remove orphan ddata
        for k1 in dref0[k0]['ldata']:
            del ddata0[k1]
        del dref0[k0]

    # Double-check consistency
    return _consistency(
        ddata=None, ddata0=ddata0,
        dref=None, dref0=dref0,
        dstatic=None, dstatic0=dstatic0,
        dobj=None, dobj0=dobj0,
        reserved_keys=reserved_keys,
        ddefparams_data=ddefparams_data,
        ddefparams_obj=ddefparams_obj,
        ddefparams_static=ddefparams_static,
        data_none=data_none,
        max_ndim=max_ndim,
    )


def _remove_static(
    # key to remove
    key=None,
    which=None,
    # dict
    dstatic0=None,
    ddata0=None,
    dobj0=None,
    # parameters
    propagate=None,
):
    """ Remove a static ref (or list) or a whole category

    key os provided:
        => remove only the desired key(s)
            works only if key is not used in ddata and dobj

    which is provided:
        => treated as param, the whole category of ref_static is removed
            if propagate, the parameter is removed from ddata and dobj
    """

    # ------------
    # check inputs

    key, which = _get_whichorkey(
        key=key,
        which=which,
        din=dstatic,
        dname='dstatic',
    )

    # --------------------------------
    # key is None => delete whole dict

    if key is None:

        # Propagate (delete as param in ddata and dobj)
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

        # remove
        del dstatic0[which]

    # -------------
    # key and which

    else:

        key = _check_remove(
            key=key,
            dkey=dstatic0[which],
            name=f'static[{k0}]',
        )

        # Make sure key is not used (condition for removing)
        for kk in key:
            lk1 = [
                k1 for k1, v1 in ddata0.items()
                if kk == v1.get(which)
            ]
            lk2 = [
                k1 for k1, v1 in dobj0.items()
                if any([kk == v2.get(which) for v2 in v1.values()])
            ]
            if len(lk1) > 0 or len(lk2) > 0:
                msg = (
                    f"Provided static key ({kk}) is used in:\n"
                    + "\n".join(
                        [f"\t- self.ddata['{k1}']" for k1 in lk1]
                        + [
                            f"\t- self.dobj['{k2}']['{which}']"
                            for k2 in lk2
                        ]
                    )
                )
                raise Exception(msg)
            del dstatic0[k0][kk]


def _remove_data(
    # key to remove
    key=None,
    # dict
    dref0=None,
    ddata0=None,
    dstatic0=None,
    dobj0=None,
    # parameters
    propagate=None,
    reserved_keys=None,
    ddefparams_data=None,
    ddefparams_obj=None,
    ddefparams_static=None,
    data_none=None,
    max_ndim=None,
):
    """ Remove a ref (or list of refs) and all associated data """

    # ------------
    # trivial case

    if key is None:
        return dref0, dstatic0, ddata0, dobj0

    # ------------
    # check inputs

    key = _check_remove(
        key=key, dkey=ddata0, name='data',
    )

    # ------
    # remove

    for k0 in key:
        # Remove key from dref['ldata']
        for k1 in dref0.keys():
            if k0 in dref0[k1]['ldata']:
                dref0[k1]['ldata'].remove(k0)
        del ddata0[k0]

    # Propagate upward to ref
    if propagate is True:
        lk = [
            k0 for k0, v0 in dref0.items()
            if len(dref0[k0].get('ldata', [])) == 0
        ]
        for kk in lk:
            del dref0[kk]

    # Double-check consistency
    return _consistency(
        ddata=None, ddata0=ddata0,
        dref=None, dref0=dref0,
        dstatic=None, dstatic0=dstatic0,
        dobj=None, dobj0=dobj0,
        reserved_keys=reserved_keys,
        ddefparams_data=ddefparams_data,
        ddefparams_obj=ddefparams_obj,
        ddefparams_static=ddefparams_static,
        data_none=data_none,
        max_ndim=max_ndim,
    )


def _remove_obj(
    # key to remove
    key=None,
    which=None,
    # dict
    dobj0=None,
    ddata0=None,
    dref0=None,
    dstatic0=None,
    # parameters
    reserved_keys=None,
    ddefparams_data=None,
    ddefparams_obj=None,
    ddefparams_static=None,
    data_none=None,
    max_ndim=None,
):

    # ------------
    # check inputs

    key, which = _get_whichorkey(
        key=key,
        which=which,
        din=dobj,
        dname='dobj',
    )

    # --------------------------------
    # key is None => delete whole dict

    if key is None:

        # remove
        del dstatic0[which]

    # ------------
    # Check inputs

    if key is not None:
        key = _check_remove(
            key=key,
            dkey=dobj0[which],
            name=f"dobj['{which}']",
        )
        for kk in set(key).intersection(dobj0[which].keys()):
            del dobj0[k0][kk]

    return _consistency(
        ddata=None, ddata0=ddata0,
        dref=None, dref0=dref0,
        dstatic=None, dstatic0=dstatic0,
        dobj=None, dobj0=dobj0,
        reserved_keys=reserved_keys,
        ddefparams_data=ddefparams_data,
        ddefparams_obj=ddefparams_obj,
        ddefparams_static=ddefparams_static,
        data_none=data_none,
        max_ndim=max_ndim,
    )


# #############################################################################
# #############################################################################
#                           dstatic
# #############################################################################


def _check_dstatic(
    dstatic=None, dstatic0=None,
):
    """ Check and format dstatic

    dstatic can be:
        - dict

    """

    # ----------------
    # Trivial case
    if dstatic in [None, {}]:
        return {}

    # ----------------
    # Check conformity

    c0 = (
        isinstance(dstatic, dict)
        and all([
            isinstance(k0, str)
            and isinstance(v0, dict)
            and all([
                isinstance(k1, str)
                and isinstance(v1, dict)
                for k1, v1 in v0.items()
            ])
            for k0, v0 in dstatic.items()
        ])
    )

    # Raise exception if non-conformity
    if not c0:
        msg = (
            "Arg dstatic must be a dict of the form:\n"
            "\t dict(\n"
            "\t     'type0': {\n"
            "\t         'k0': {...},\n"
            "\t         'k1': {...}\n"
            "\t     },\n"
            "\t     'type1': {\n"
            "\t         'k2': {...},\n"
            "\t         'k3': {...}\n"
            "\t     },\n"
            "\t )\n"
            f"\nProvided:\n{dstatic}"
        )
        raise Exception(msg)

    # ----------------------------
    # Identify conflicts / updates

    # raise except if conflict with existing entry
    dupdate = {}
    dconflict = {}
    for k0, v0 in dstatic.items():

        if k0 not in dstatic0.keys():
            continue

        dconflict0, dupdate0 = _check_conflicts(
            dd=dstatic[k0],
            dd0=dstatic0[k0],
            dd_name=f"dstatic['{k0}']",
            returnas=True,
        )

        if len(dconflict0) > 0:
            dconflict[k0] = {k1: list(v1) for k1, v1 in dconflict0.items()}

        if len(dupdate0) > 0:
            dupdate[k0] = {k1: list(v1) for k1, v1 in dupdate0.items()}

    # Conflicts => Exception
    if len(dconflict) > 0:
        msg = (
            "The following dstatic keys are conflicting existing values:\n"
            + "\n".join([
                "\t- dstatic['{}']['{}']: {}".format(k0, v0[0], v0[1])
                for k0, v0 in dconflict.items()
            ])
        )
        raise Exception(msg)

    # Updates => Warning
    if len(dupdate) > 0:
        lstr = list(itt.chain.from_iterable([
            [
                f"\t- dstatic['{k0}']['{k1}']: {v1}"
                for k1, v1 in v0.items()
            ]
            for k0, v0 in dupdate.items()
        ]))
        msg = (
            "\nThe following keys will be added to dstatic:\n"
            + "\n".join(lstr)
        )
        warnings.warn(msg)

    # ------------------
    # Check element / ion / charge

    # _check_elementioncharge_dict(dstatic=dstatic)

    return dstatic


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

    return data, size


def _check_dref(
    dref=None,
    dref0=None,
    ddata0=None,
):
    """ Check and format dref

    dref can be:
        - dict

    If some data is provided
        => returns ddata to be added

    """

    # ----------------
    # Trivial case
    if dref in [None, {}]:
        return {}, {}

    # ----------------
    # Check conformity

    # Basis
    # lk_opt = ['ldata', 'size', 'data']
    if not isinstance(dref, dict):
        msg = "Arg dref must be a dict!"
        raise Exception(msg)

    dref2 = {}
    for k0, v0 in dref.items():

        # key
        key = ds._generic_check._obj_key(d0=dref0, short=_IREF)

        # v0
        if isinstance(v0, (np.ndarray, list, tuple)):
            dref[k0] = {'data': v0}
            v0 = dref[k0]

        c0 = (
            isinstance(v0, dict)
            and (
                isinstance(v0.get('data'), (np.ndarray, list, tuple))
                or isinstance(v0.get('size'), int)
            )
        )
        if not c0:
            msg = "v0 must be a dict with either 'data' or 'size'"
            raise Exception(msg)

        dref2[key] = dict(dref[k0])

    # -----------------------
    # raise except if conflict with existing entry

    _check_conflicts(dd=dref2, dd0=dref0, dd_name='dref')

    # ----------------
    # Add size / data if relevant

    ddata_add = {
        k0: {'data': None}
        for k0, v0 in dref2.items()
        if v0.get('data') is not None
        and k0 not in ddata0.keys()
    }
    for k0, v0 in dref2.items():
        if v0.get('data') is not None:
            data, dref2[k0]['size'] = _check_dataref(
                data=v0['data'], key=k0,
            )
            if k0 in ddata_add.keys():
                ddata_add[k0]['data'] = data
                ddata_add[k0]['ref'] = (k0,)
                ddata_add[k0].update({
                    k1: v1 for k1, v1 in v0.items()
                    if k1 not in ['size', 'ldata']
                })

    # get rid of extra keys
    dref = {
        k0: {k1: v0.get(k1) for k1 in _DRESERVED_KEYS['dref']}
        for k0, v0 in dref2.items()
    }
    return dref, ddata_add


# #############################################################################
# #############################################################################
#                           ddata
# #############################################################################


def _check_data(data=None, key=None, max_ndim=None):
    """ Check the conformity of data to be a valid reference

    max_ndim allows to define a maximum number of dimensions
    lists and tuple of non-uniform len elements are converted to object arrays

    """

    # if not array
    # => try converting or get class (dict, mesh...)
    shape = None
    c0_array = (
        isinstance(data, np.ndarray)
        or scpsp.issparse(data)
    )

    # if not array => list, tuple
    if not c0_array:
        if isinstance(data, (list, tuple)):
            c0 = (
                all([hasattr(oo, '__iter__') for oo in data])
                and len(set([len(oo) for oo in data])) != 1
            )
            if c0:
                # non-uniform len of element => object array
                data = np.array(data, dtype=object)
                shape = (data.shape[0],)

            else:
                # uniform len of all elements => convert to array
                try:
                    data = np.array(data)
                    shape = data.shape
                    c0_array = True
                except Exception as err:
                    raise DataRefException(ref=key, data=data)
        else:
            msg = "Non-handled data type!"
            raise Exception(msg)

    # if array => check unique (unique + sorted)
    if shape is None:
        shape = data.shape

    # Check max_dim if any
    if c0_array and max_ndim is not None:
        if data.ndim > max_ndim:
            msg = (
                "Provided data for ddata['{key}'] has too many dimensions!\n"
                f"- ndim:     {data.ndim}\n"
                f"- max_ndim: {max_ndim}\n"
            )
            raise Exception(msg)

    # Check if valid ref candidate (monotonous = (True,))
    if c0_array:
        monotonous = tuple([
            bool(
                np.all(np.diff(data, axis=aa) > 0.)
                or np.all(np.diff(data, axis=aa) < 0.)
            )
            for aa in range(data.ndim)
        ])
    else:
        monotonous = (False,)
    return data, shape, monotonous


def _get_suitable_ref(
    shape=None,
    key=None,
    dref0=None,
    dref_add=None,
    axis=None,
):
    """  For each dimension of data.shape, identify the relevant ref index """

    if axis is None:
        axis = 0
    assert axis < len(shape)

    # list all possible ref in each dimension
    size = shape[axis]
    lref = [
        k0 for k0, v0 in dref0.items()
        if v0['size'] == size
    ]

    # perfect match
    if len(lref) == 1:
        lref = lref[0]

    # multiple matches
    elif len(lref) > 1:
        msg = (
            f"Ambiguous ref for ddata['{key}']\n"
            f"Possible matches: {lref}"
        )
        raise Exception(msg)

    # no match => create new ref
    else:
        lref0 = ds._generic_check._obj_key(d0=dref0, short=_IREF)
        lref1 = ds._generic_check._obj_key(d0=dref_add, short=_IREF)
        n0 = int(lref0[lref0[len(_IREF):]])
        n1 = int(lref1[lref1[len(_IREF):]])
        lref = lref0 if n0 > n1 else lref1

    return lref, size


def _check_data_ref(k0=None, ddata=None, dref0=None, dref_add=None):

    if dref_add is None:
        dref_add = {}

    # None => create
    if ddata[k0].get('ref') is None:
        lref = []
        for ii, ss in enumerate(ddata[k0]['shape']):
            ref, size = _get_suitable_ref(
                shape=ddata[k0]['shape'],
                key=k0,
                dref0=dref0,
                dref_add=dref_add,
                axis=ii,
            )

            if ref not in dref0.keys() and ref not in dref_add.keys():
                dref_add[ref] = {'size': size}

            lref.append(ref)

        ddata[k0]['ref'] = tuple(lref)

    # length mismatch
    elif len(ddata[k0]['shape']) != len(ddata[k0]['ref']):
        msg = (
            "Mismatching len(ref) and len(shape) for ddata['{k0}']"
        )
        raise Exception(msg)

    # length match but unknown ref or size mismatch
    else:
        for ii, ss in enumerate(ddata[k0]['shape']):
            if ddata[k0]['ref'][ii] not in dref0.keys():
                if ddata[k0]['ref'][ii] not in dref_add.keys():
                    dref_add[ddata[k0]['ref'][ii]] = {'size': ss}

            elif ss != dref0[ddata[k0]['ref'][ii]]['size']:
                msg = (
                    f"Mismatching ref size and shape for ddata['{k0}']"
                )
                raise Exception(msg)

    return dref_add


def _check_ddata(
    ddata=None,
    ddata0=None,
    dref0=None,
    reserved_keys=None,
    data_none=None,
    max_ndim=None,
):

    # ----------------
    # Trivial case

    if ddata in [None, {}]:
        return {}, {}
    if data_none is None:
        data_none = _DATA_NONE

    # ----------------
    # Check conformity

    # Basis
    # lk_opt = ['ldata', 'size', 'data']
    if not isinstance(ddata, dict):
        msg = "Arg ddata must be dict!"
        raise Exception(msg)

    ltok = (np.ndarray, list, tuple)
    lkout = [
        k0 for k0, v0 in ddata.items()
        if not (
            (k0 is None or isinstance(k0, str))
            # and k0 not in ddata0.keys()
            and (
                (isinstance(v0, ltok) or scpsp.issparse(v0))
                or (
                    isinstance(v0, dict)
                    and all([isinstance(ss, str) for ss in v0.keys()])
                    and (
                        (
                            'data' in v0.keys()
                            and (
                                isinstance(v0['data'], ltok)
                                or scpsp.issparse(v0['data'])
                            )
                            and (
                                v0.get('ref') is None
                                or isinstance(v0.get('ref'), str)
                                or (
                                    isinstance(v0.get('ref'), tuple)
                                    and all([
                                        isinstance(rr, str)
                                        for rr in v0['ref']
                                    ])
                                )
                            )
                        )
                    )
                )
            )
        )
    ]

    # Raise exception if non-conformity
    if len(lkout) != 0:
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
                - (D): only the data array is provided

            If ref = True, the data is itself considered a ref

            The following keys do not match the criteria:
            """
            + '\t- ' + '\n\t- '.join(lkout)
        )
        raise Exception(msg)

    # -----------------------
    # raise except if conflict with existing entry

    _check_conflicts(dd=ddata, dd0=ddata0, dd_name='ddata')

    # ----------------
    # Convert and/or add ref if necessary

    ddata2 = {}
    dref_add = {}
    for k0, v0 in ddata.items():

        key = ds._generic_check._obj_key(d0=ddata, short=_IDATA)

        # convert to dict if needed
        if not isinstance(v0, dict):
            ddata[k0] = {'data': v0}

        # check data itself
        data, shape, monotonous = _check_data(
            data=ddata[k0]['data'], key=k0, max_ndim=max_ndim,
        )
        ddata[k0]['data'] = data
        ddata[k0]['shape'] = shape
        ddata[k0]['monot'] = monotonous

        # find ref
        if isinstance(v0['ref'], str):
            ddata[k0]['ref'] = (v0['ref'],)

        _check_data_ref(
            k0=k0, ddata=ddata, dref0=dref0, dref_add=dref_add,
        )

        ddata2[key] = dict(ddata[k0])

    # ------------------
    # Check ref vs shape

    for k0, v0 in ddata2.items():
        c0 = (
            isinstance(v0['ref'], tuple)
            and all([
                (
                    ss in dref0.keys()
                    and dref0[ss]['size'] == ddata2[k0]['shape'][ii]
                )
                or (
                    ss in dref_add.keys()
                    and dref_add[ss]['size'] == ddata2[k0]['shape'][ii]
                )
                for ii, ss in enumerate(v0['ref'])
            ])
        )
        if not c0:
            msg = (
                f"ddata['{k0}']['ref'] contains unknown ref:\n"
                f"\t- ddata['{k0}']['ref'] = {v0['ref']}\n"
                f"\t- dref0.keys() = {sorted(dref0.keys())}\n"
                f"\t- dref_add.keys() = {sorted(dref_add.keys())}"
            )
            raise Exception(msg)

    return ddata2, dref_add


# #############################################################################
# #############################################################################
#                           dobj
# #############################################################################


def _check_dobj(
    dobj=None,
    dobj0=None,
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
            f"\t- Provided: {type(dobj)}"
        )
        raise Exception(msg)

    # Map possible non-conformities
    dc = {}
    dobj2 = {}
    for k0, v0 in dobj.items():

        # check types (str, dict)
        c1 = isinstance(k0, str) and isinstance(v0, dict)
        if not c1:
            dc[k0] = "type(key) != str or type(value) != dict"
            continue

        # check each key / value
        lc2 = [
            f'\t- {str(k1)}: type {type(v1)}'
            f', key already in dobj0: {k1 in dobj0.get(k0, {}).keys()}'
            for k1, v1 in v0.items()
            if not (
                (k1 is None or isinstance(k1, str))
                and isinstance(v1, dict)
                and k1 not in dobj0.get(k0, {}).keys()
            )
        ]
        if len(lc2) > 0:
            dc[k0] = (
                f"The following keys of dobj['{k0}'] are not valid:\n"
                + "\n".join(lc2)
            )
            continue

        # set None to default keys if any None
        dobj2[k0] = {}
        for k1 in v0.keys():
            key = ds._generic_check._obj_key(
                d0=dobj0.get(k0, {}), short=k0[:4],
            )
            dobj2[k0][key] = dict(dobj[k0][k1])

    # Raise Exception
    if len(dc) > 0:
        msg = (
            "The following keys of dobj are non-conform:\n"
            + "\n\n".join([f"dobj['{k0}']: {v0}" for k0, v0 in dc.items()])
        )
        raise Exception(msg)

    return dobj2


# #############################################################################
# #############################################################################
#                           Params
# #############################################################################


def _harmonize_params(
    dd=None,
    dd_name=None,
    dd_name2=None,
    dstatic=None,
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
    # Check against dstatic0
    lkpout = [
        (k0, (k1, v0[k1]))
        for k0, v0 in dd.items()
        if k1 in dstatic.keys()
        and any([v0[k1] not in dstatic[k1].keys() for k1 in lparams])
    ]
    if len(lkpout) > 0:
        lpu = sorted(set([pp[1][0] for pp in lkpout]))
        msg0 = '\n'.join([
            '\t- {}[{}]: {}'.format(pp[0], pp[1], pp[2]) for pp in lkpout
        ])
        msg1 = '\n'.join([
            '\t- dstatic[{}]: {}'.format(pp, dstatic[pp].keys())
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


def _update_dstatic0(dstatic0=None, ddata0=None, dobj0=None):
    """ Count nb. of matching ref_static in ddata and dobj """

    for k0, v0 in dstatic0.items():

        # ddata
        dd = {
            k2: np.sum([ddata0[k3].get(k0) == k2 for k3 in ddata0.keys()])
            for k2 in v0.keys()
            if any([ddata0[k3].get(k0) == k2 for k3 in ddata0.keys()])
        }
        if len(dd) > 0:
            ss = 'nb. data'
            for k2, v2 in v0.items():
                dstatic0[k0][k2][ss] = int(dd.get(k2, 0))

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
                    dstatic0[k0][k2][ss] = int(dd.get(k2, 0))


# #############################################################################
# #############################################################################
#                           Consistency
# #############################################################################


def _consistency(
    dobj=None, dobj0=None,
    ddata=None, ddata0=None,
    dref=None, dref0=None,
    dstatic=None, dstatic0=None,
    reserved_keys=None,
    ddefparams_data=None,
    ddefparams_obj=None,
    ddefparams_static=None,
    data_none=None,
    max_ndim=None,
):

    # --------------
    # dref
    dref, ddata_add = _check_dref(
        dref=dref, dref0=dref0, ddata0=ddata0,
    )
    if ddata_add is not None:
        if ddata is None:
            ddata = ddata_add
        else:
            ddata.update(ddata_add)
    dref0.update(dref)

    # --------------
    # dstatic
    dstatic = _check_dstatic(
        dstatic=dstatic, dstatic0=dstatic0,
    )
    for k0, v0 in dstatic.items():
        if k0 not in dstatic0.keys():
            dstatic0[k0] = v0
        else:
            dstatic0[k0].update(v0)

    # --------------
    # ddata
    ddata, dref_add = _check_ddata(
        ddata=ddata, ddata0=ddata0,
        dref0=dref0,
        reserved_keys=reserved_keys,
        data_none=data_none,
        max_ndim=max_ndim,
    )
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
        dstatic=dstatic0,
        ddefparams=ddefparams_data,
        reserved_keys=reserved_keys,
    )

    # --------------
    # params harmonization - dstatic
    for k0, v0 in dstatic0.items():
        dstatic0[k0] = _harmonize_params(
            dd=v0,
            dd_name='dstatic',
            dd_name2=f'dstatic[{k0}]',
            dstatic=dstatic0,
            ddefparams=ddefparams_static.get(k0),
            reserved_keys=reserved_keys,
        )

    # --------------
    # params harmonization - dobj
    for k0, v0 in dobj0.items():
        dobj0[k0] = _harmonize_params(
            dd=v0,
            dd_name='dobj',
            dd_name2='dobj[{}]'.format(k0),
            dstatic=dstatic0,
            ddefparams=ddefparams_obj.get(k0),
            reserved_keys=reserved_keys,
        )

    # --------------
    # Complement

    # ddata0
    for k0, v0 in ddata0.items():
        if v0.get('data') is None:
            continue

    # dref0
    for k0, v0 in dref0.items():
        dref0[k0]['ldata'] = sorted(set(
            k1 for k1 in ddata0.keys()
            if ddata0[k1].get('data') is not None and k0 in ddata0[k1]['ref']
        ))
        dref0[k0]['ldata_monot'] = [
            k1 for k1 in dref0[k0]['ldata']
            if ddata0[k1]['monot'] == (True,)
        ]

    # dstatic0
    _update_dstatic0(dstatic0=dstatic0, ddata0=ddata0, dobj0=dobj0)

    # --------------
    # Check conventions

    for k0, v0 in ddata0.items():
        if v0.get('data') is None:
            continue

    return dref0, dstatic0, ddata0, dobj0


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
    dobj0=None,
    dstatic0=None,
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
        msg = (
            "\nArg new_ref must be a key to a valid ref (monotonous)!\n"
            + "\t- Provided: {}\n\n".format(new_ref)
            + "Available valid ref candidates:\n"
            + "\t- {}".format('\n\t- '.join(list(dref.keys())))
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
        dobj=None, dobj0=dobj0,
        dstatic=None, dstatic0=dstatic0,
        reserved_keys=None,
        ddefparams_data=ddefparams_data,
        ddefparams_obj=None,
        ddefparams_static=None,
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
        - np.ndarray: {param0: np.r_[values0, value1...], ...}

    """

    # ---------------
    # Check inputs

    # Get key (which data to return param for)
    key = _ind_tofrom_key(dd=dd, key=key, ind=ind, returnas=str)

    # param
    lp = [kk for kk in list(dd.values())[0].keys() if kk != 'data']
    param = ds._generic_check._check_var_iter(
        param,
        'param',
        types=list,
        types_iter=str,
        allowed=lp,
    )

    # returnas
    returnas = ds._generic_check._check_var(
        returnas,
        'returnas',
        allowed=[np.ndarray, dict],
        default=np.ndarray,
    )

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
    distribute=None,
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

    # ---------------
    # Check inputs

    # Get key (which data to return param for)
    key = _ind_tofrom_key(dd=dd, key=key, ind=ind, returnas=str)

    # param
    lp = [kk for kk in list(dd.values())[0].keys() if kk != 'data']
    param = ds._generic_check._check_var(
        param,
        'param',
        types=str,
        allowed=lp,
    )

    # distribute
    if isinstance(value, (list, tuple, np.ndarray)):
        defdist = len(key) == len(value)
    else:
        defdist = False
    distribute = ds._generic_check._check_var(
        distribute,
        'distribute',
        types=bool,
        default=defdist,
    )

    # ---------------
    # Set value

    # Check value - TBC: allow list
    ltypes = [str, int, np.integer, float, np.floating, tuple]
    lc = [
        isinstance(value, tuple(ltypes)),
        isinstance(value, (list, tuple, np.ndarray))
        and distribute is False,
        isinstance(value, (list, tuple, np.ndarray))
        and np.array(value).shape[0] == len(key)
        and distribute is True,
        isinstance(value, dict)
        and all([
            kk in dd.keys()
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
                - dict of scalar / str

            The length of value must match the selected keys ({})
            """.format(ltypes, len(key))
        )
        raise Exception(msg)

    # Update data
    if value is None or lc[0] or lc[1]:
        for kk in key:
            dd[kk][param] = value
    elif lc[2]:
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

    # ---------------
    # Check inputs

    # Get key (which data to return param for)
    # key = _ind_tofrom_key(dd=dd, key=key, ind=ind, returnas=str)

    # param
    lp = [kk for kk in list(dd.values())[0].keys() if kk != 'data']
    param = ds._generic_check._check_var(
        param,
        'param',
        types=str,
        excluded=lp,
    )

    # Initialize and set
    for kk in dd.keys():
        dd[kk][param] = None
    _set_param(dd=dd, param=param, value=value)


def _remove_param(dd=None, dd_name=None, param=None):
    """ Remove a parameter, none by default, all if param = 'all' """

    # Check inputs
    lp = [kk for kk in list(dd.values())[0].keys() if kk != 'data']
    if param == 'all':
        param = lp
    if isinstance(param, str):
        param = [param]
    param = ds._generic_check._check_var_iter(
        param,
        'param',
        types=list,
        types_iter=str,
        allowed=lp,
    )

    # Remove
    if param is not None:
        for pp in param:
            for k0 in dd.keys():
                del dd[k0][pp]


def _ind_tofrom_key(
    dd=None, dd_name=None,
    ind=None, key=None, returnas=int,
):
    """

    From key return ind and vice-versa

    """

    # --------------------
    # Check / format input

    lc = [ind is not None, key is not None]
    if not np.sum(lc) <= 1:
        msg = ("Args ind and key cannot be prescribed simultaneously!")
        raise Exception(msg)

    returnas = ds._generic_check._check_var(
        returnas,
        'returnas',
        allowed=[int, bool, str, 'key'],
        default='key',
    )

    # -----------------
    # Compute

    # Intialize output
    out = np.zeros((len(dd),), dtype=bool)

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
                or (
                    ind.dtype == int
                    and np.all(np.isfinite(ind))
                    and np.max(ind) <= len(dd)
                )
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
                    dtype=str,
                )

    elif lc[1]:

        # Check key
        if isinstance(key, str):
            key = [key]
        key = ds._generic_check._check_var_iter(
            key,
            'key',
            types_iter=str,
            allowed=lk,
        )

        # return
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
