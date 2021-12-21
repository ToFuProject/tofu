

# Standard
import itertools as itt
import warnings

# Common
import numpy as np
import scipy.sparse as scpsp
from matplotlib.tri import Triangulation as mplTri


from . import _generic_check


_DRESERVED_KEYS = {
    'dref': ['ldata', 'size', 'ind'],
    'dstatic': [],
    'ddata': ['ref', 'shape', 'data'],
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


def _check_which(
    dref=None,
    ddata=None,
    dobj=None,
    dstatic=None,
    which=None,
    return_dict=None,
):
    """ Check which in ['data'] + list(self._dobj.keys() """

    # --------------
    # Check inputs

    return_dict = _generic_check._check_var(
        return_dict,
        'return_dict',
        types=bool,
        default=True,
    )

    lkobj = list(dobj.keys())
    lkstatic = list(dstatic.keys())
    lkok = ['ref', 'data'] + lkobj + lkstatic
    which = _generic_check._check_var(
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
        elif which == 'ddata':
            dd = ddata
        elif which in lkobj:
            dd = dobj[which]
        else:
            dd = dstatic[which]
        return which, dd
    else:
        return which


def _check_conflicts(dd=None, dd0=None, dd_name=None):
    """ Detect conflict with existing entries

    Any pre-existing entry will trigger either an update or a conflict
    """

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
                            v0[kk].data, dd0[k0][kk].data, equal_nan=True,
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
            and kk not in ['ldata', 'size']
        ]
        if len(lkup) > 0:
            dupdate[k0] = lk

    # Conflicts => Exception
    if len(dconflict) > 0:
        lstr = [f"\t- {dd_name}['{k0}']: {v0}" for k0, v0 in dconflict.items()]
        msg = (
            f"Conflicts with pre-existing values found in {dd_name}:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # Updates => Warning
    if len(dupdate) > 0:
        lstr = [f"\t- {dd_name}['{k0}']: {v0}" for k0, v0 in dupdate.items()]
        msg = (
            f"\nExisting {dd_name} keys will be overwritten:\n"
            + "\n".join(lstr)
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


def _remove_ref(
    key=None,
    dref0=None, ddata0=None,
    dstatic0=None,
    dobj0=None,
    propagate=None,
    reserved_keys=None,
    ddefparams_data=None,
    ddefparams_obj=None,
    data_none=None,
    max_ndim=None,
):
    """ Remove a ref (or list of refs) and all associated data """
    if key is None:
        return dref0, ddata0
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
        data_none=data_none,
        max_ndim=max_ndim,
    )


def _remove_ref_static(
    key=None,
    which=None,
    propagate=None,
    dstatic0=None,
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
            k0 for k0, v0 in dstatic0.items()
            if all([kk in v0.keys() for kk in key])
        ]
        if len(lk0) != 1:
            msg = (
                "No / several matches for '{}' in ref_static:\n".format(key)
                + "\n".join([
                    "\t- dstatic[{}][{}]".format(k0, key) for k0 in lk0
                ])
            )
            raise Exception(msg)
        k0 = lk0[0]
        key = _check_remove(
            key=key,
            dkey=dstatic0[k0],
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
            del dstatic0[k0][kk]

    elif which is not None:
        if which not in dstatic0.keys():
            msg = (
                "Provided which not in dstatic.keys():\n"
                + "\t- Available: {}\n".format(sorted(dstatic0.keys()))
                + "\t- Provided: {}".format(which)
            )
            raise Exception(msg)
        del dstatic0[which]

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
    dref0=None, ddata0=None,
    dstatic0=None,
    dobj0=None,
    propagate=None,
    reserved_keys=None,
    ddefparams_data=None,
    ddefparams_obj=None,
    data_none=None,
    max_ndim=None,
):
    """ Remove a ref (or list of refs) and all associated data """
    if key is None:
        return dref0, ddata0
    key = _check_remove(
        key=key, dkey=ddata0, name='data',
    )

    for k0 in key:
        # Remove key from dref['ldata']
        for k1 in dref0.keys():
            if k0 in dref0[k1]['ldata']:
                dref0[k1]['ldata'].remove(k0)
        del ddata0[k0]

    # Propagate upward
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
        data_none=data_none,
        max_ndim=max_ndim,
    )


def _remove_obj(
    key=None,
    which=None,
    dobj0=None,
    ddata0=None,
    dref0=None,
    dstatic0=None,
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
        dstatic=None, dstatic0=dstatic0,
        dobj=None, dobj0=dobj0,
        reserved_keys=reserved_keys,
        ddefparams_data=ddefparams_data,
        ddefparams_obj=ddefparams_obj,
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
    """ Check and format dref_staytic

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
            """
            Arg dstatic must be a dict of the form:
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
            """.format(dstatic)
        )
        raise Exception(msg)

    # raise except if conflict with existing entry
    dupdate = {}
    dconflict = {}
    for k0, v0 in dstatic.items():
        lkout = ['nb. data']
        if k0 == 'ion':
            lkout += ['ION', 'charge', 'element']
        if k0 not in dstatic0.keys():
            continue

        for k1, v1 in v0.items():
            if k1 not in dstatic0[k0].keys():
                continue
            # conflicts
            lk = set(v1.keys()).intersection(dstatic0[k0][k1].keys())
            lk = [kk for kk in lk if v1[kk] != dstatic0[k0][k1][kk]]
            if len(lk) > 0:
                dconflict[k0] = (k1, lk)
            # updates
            lk = [
                kk for kk in dstatic0[k0][k1].keys()
                if kk not in v1.keys()
                and kk not in lkout
                and 'nb. ' not in kk
            ]
            if len(lk) > 0:
                dupdate[k0] = (k1, lk)

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
        msg = (
            "\nThe following existing dstatic keys will be forgotten:\n"
            + "\n".join([
                "\t- dstatic['{}']['{}']: {}".format(k0, v0[0], v0[1])
                for k0, v0 in dupdate.items()
            ])
        )
        warnings.warn(msg)

    # ------------------
    # Check element / ion / charge
    _check_elementioncharge_dict(dstatic=dstatic)

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
    dref=None, dref0=None, ddata0=None,
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
        return {}, None, None

    # ----------------
    # Check conformity

    # Basis
    # lk_opt = ['ldata', 'size', 'data']
    c0 = isinstance(dref, dict)
    if not isinstance(dref, dict):
        msg = "Arg dref must be a dict!"
        raise Exception(msg)

    keyroot = 'iref'
    for k0, v0 in dref.items():

        # key
        nmax = _generic_check._name_key(
            dd=None, dd_name=None, keyroot=keyroot,
        )[1]
        key = f'{keyroot}{nmax:02.0f}'

        key = _generic_checks._check_var(
            k0,
            'k0',
            types=str,
            default=key,
        )

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
            data, dref[k0]['size'] = _check_dataref(
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
        k0: {k1: v1 for k1, v1 in v0.items() if k1 in _DRESERVED_KEYS['dref']}
        for k0, v0 in dref.items()
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
                except Exception as err:
                    raise DataRefException(ref=key, data=data)

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
    return data, shape, monotonous


def _get_suitable_ref(shape=None, key=None, dref=None):

    lref = [
        [
            k0 for k0, v0 in dref.items()
            if v0['size'] == shape[ii]
        ]
        for ii in range(len(shape))
    ]

    dnew = {}
    for ii, rr in enumerate(lref):
        if len(rr) == 1:
            lref[ii] == rr[0]
        elif len(rr) > 1:
            msg = (
                f"Ambiguous ref for ddata['{key}']\n"
                f"Possible matches: {lref}"
            )
            raise Exception(msg)
        else:
            keyroot = 'iref'
            nmax = _generic_check._name_key(
                dd=None, dd_name=None, keyroot=keyroot,
            )[1]
            lref[ii] = f'{keyroot}{nmax:02.0f}'
            dnew[lref[ii]] = {'size': shape[ii]}
    return lref, dnew


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
        return {}, None, None
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
            isinstance(k0, str)
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
                                or isinstance(v0.get('ref'), tuple)
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
            + '\t- '+'\n\t- '.join(lkout)
        )
        raise Exception(msg)

    # -----------------------
    # raise except if conflict with existing entry

    _check_conflicts(dd=ddata, dd0=ddata0, dd_name='ddata')

    # ----------------
    # Convert and/or add ref if necessary

    dref_add = {}
    for k0, v0 in ddata.items():

        if not isinstance(v0, dict):
            lref, dnew = _get_suitable_ref(shape=, key=k0, dref=dref0)
            ddata[k0] = {'ref': lref, 'data': v0}
            dref_add.update(dnew)

        else:
            if v0.get('data') is None:
                continue

            if v0.get('ref') is None:
                lref, dnew = _get_suitable_ref(shape=, key=k0, dref=dref0)
                ddata[k0['ref']] = lref
                dref_add.update(dnew)

            elif isinstance(v0['ref'], str):
                ddata[k0]['ref'] = (v0['ref'],)

    # Check data and ref vs shape - and optionnally add to ref if mesh2d
    for k0, v0 in ddata.items():
        if v0.get('data') is not None:
            (
                ddata[k0]['data'], ddata[k0]['shape'], ddata[k0]['monot']
            ) = _check_data(
                data=v0['data'], key=k0, max_ndim=max_ndim,
            )

            # Check if mesh2d
            c0 = ddata[k0].get('ref') in [None, (k0,)]
            if not c0:
                msg = (
                    f"ddata[{k0}]['ref'] should have ref = ({k0},)"
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
        dref_add, ddata_dadd = _check_dref(
            dref=dref_add, dref0=dref0, ddata0=ddata0,
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
                    + "\n\t- ddata['{}']['ref'] = {}\n\n".format(k0, v0['ref'])
                    + "... or there might be an issue with:\n"
                    + "\t- type(ddata['{}']['shape']) = {} ({})".format(
                        k0, type(v0['shape']), v0['shape'],
                    )
                )
            raise Exception(msg)

    return ddata, dref_add


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


def _check_elementioncharge_dict(dstatic):
    """ Specific to SpectralLines """

    # Assess if relevant
    lk = [kk for kk in ['ion', 'ION'] if kk in dstatic.keys()]
    if len(lk) == 0:
        return
    kion = lk[0]
    kION = 'ION' if kion == 'ion' else 'ion'
    if kion == 'ION':
        dstatic['ion'] = {}

    lerr = []
    for k0, v0 in dstatic[kion].items():
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
                dstatic['ion'][ion] = {
                    'ION': ION,
                    'element': element,
                    'charge': charge,
                }
            else:
                dstatic['ion'][k0]['ION'] = ION
                dstatic['ion'][k0]['element'] = element
                dstatic['ion'][k0]['charge'] = charge

        except Exception as err:
            lerr.append((k0, str(err)))

    if kion == 'ION':
        del dstatic['ION']

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
        ddefparams=ddefparams_data, reserved_keys=reserved_keys,
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
    param = _generic_check._check_var_iter(
        param,
        'param',
        types=list,
        types_iter=str,
        allowed=lp,
    )

    # returnas
    returnas = _generic_check._check_var(
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
    param = _generic_check._check_var_iter(
        param,
        'param',
        types=list,
        types_iter=str,
        allowed=lp,
    )

    # ---------------
    # Set value

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

    # ---------------
    # Check inputs

    # Get key (which data to return param for)
    key = _ind_tofrom_key(dd=dd, key=key, ind=ind, returnas=str)

    # param
    lp = [kk for kk in list(dd.values())[0].keys() if kk != 'data']
    param = _generic_check._check_var(
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
    param = _generic_check._check_var_iter(
        param,
        'param',
        types=list,
        types_iter=str,
        allowed=lp,
    )

    # Remove
    if param is not None:
        for k0 in dd.keys():
            del dd[k0][param]


# #############################################################################
# #############################################################################
#               Selection
# #############################################################################


def _ind_tofrom_key(
    dd=None, dd_name=None,
    ind=None, key=None, returnas=int,
):

    # --------------------
    # Check / format input

    lc = [ind is not None, key is not None]
    if not np.sum(lc) <= 1:
        msg = ("Args ind and key cannot be prescribed simultaneously!")
        raise Exception(msg)

    returnas = _check_generic._check_var(
        returnas,
        'returnas',
        types=str,
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
                    ind.dtype == np.int
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
        key = _generic_check._check_var_iter(
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

    # -----------
    # check input

    # log
    log = _generic_var._check_var(
        log,
        'log',
        types=str,
        default='all',
        allowed=['all', 'any', 'raw'],
    )

    # returnas
    # 'raw' => return the full 2d array of boolean indices
    returnas = _generic_var._check_var(
        returnas,
        'returnas',
        default=bool if log == 'raw' else int,
        allowed=[int, bool, str, 'key'],
    )

    kwdargs = {k0: v0 for k0, v0 in kwdargs.items() if v0 is not None}

    # Get list of relevant criteria
    lp = [kk for kk in list(dd.values())[0].keys() if kk != 'data']
    lk = list(kwdargs.keys())
    lk = _generic_var._check_var_iter(
        lk,
        'lk',
        types_iter=str,
        default=lp,
        allowed=lp,
    )

    # --------------------
    # Get raw bool indices

    # Get list of accessible param
    ltypes = [float, np.float_]
    lquant = [
        kk for kk in kwdargs.keys()
        if any([type(dd[k0][kk]) in ltypes for k0 in dd.keys()])
    ]

    # Prepare array of bool indices and populate
    ind = np.zeros((len(kwdargs), len(dd)), dtype=bool)
    for ii, kk in enumerate(kwdargs.keys()):
        try:
            par = _get_param(
                dd=dd, dd_name=dd_name,
                param=kk,
                returnas=np.ndarray,
            )[kk]
            if kk in lquant:
                # list => in interval
                if isinstance(kwdargs[kk], list) and len(kwdargs[kk]) == 2:
                    ind[ii, :] = (
                        (kwdargs[kk][0] <= par) & (par <= kwdargs[kk][1])
                    )

                # tuple => out of interval
                elif isinstance(kwdargs[kk], tuple) and len(kwdargs[kk]) == 2:
                    ind[ii, :] = (
                        (kwdargs[kk][0] > par) | (par > kwdargs[kk][1])
                    )

                # float / int => equal
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

    # -----------------
    # Format output ind

    # return raw 2d array of bool indices
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
        # return all or any
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
    key=None, monot=None,
    msgstr=None, raise_=False,
):
    """ Return the unique data key matching key

    Here, key can be interpreted as name / source / units / quant...
    All are tested using select() and a unique match is returned
    If not unique match an error message is either returned or raised

    """

    # ------------------------
    # Trivial case: key is actually a ddata key

    if key in dd.keys():
        return key, None

    # ------------------------
    # Non-trivial: check for a unique match on other params

    dind = _select(
        dd=dd, dd_name=dd_name,
        dim=key, quant=key, name=key, units=key, source=key,
        monot=monot,
        log='raw',
        returnas=bool,
    )
    ind = np.array([ind for kk, ind in dind.items()])

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
        msg = "No match in {} for {}".format(lstr, key)

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
