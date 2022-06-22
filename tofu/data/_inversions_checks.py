# -*- coding: utf-8 -*-


# Built-in
import copy
import warnings


# Common
import numpy as np
import scipy.sparse as scpsp


# specific
from . import _generic_check


# optional
try:
    from .. import tomotok2tofu
except Exception as err:
    tomotok2tofu = False


_DALGO = {
    'algo0': {
        'name': 'algo0',
        'source': 'tofu',
        'family': 'Phillips-Tikhonov',
        'reg_operator': 'any linear',
        'reg_param': 'augTikho',
        'decomposition': '',
        'positive': False,
        'sparse': True,
        'isotropic': True,
        'func': 'inv_linear_augTikho_sparse',
    },
    'algo1': {
        'name': 'algo1',
        'source': 'tofu',
        'family': 'Phillips-Tikhonov',
        'reg_operator': 'any linear',
        'reg_param': 'augTikho',
        'decomposition': '',
        'positive': False,
        'sparse': False,
        'isotropic': True,
        'func': 'inv_linear_augTikho_dense',
    },
    'algo2': {
        'name': 'algo2',
        'source': 'tofu',
        'family': 'Phillips-Tikhonov',
        'reg_operator': 'any linear',
        'reg_param': 'augTikho',
        'decomposition': 'cholesky',
        'positive': False,
        'sparse': False,
        'isotropic': True,
        'func': 'inv_linear_augTikho_chol_dense',
    },
    'algo3': {
        'name': 'algo3',
        'source': 'tofu',
        'family': 'Phillips-Tikhonov',
        'reg_operator': 'any linear',
        'reg_param': 'augTikho',
        'decomposition': 'cholesky',
        'positive': False,
        'sparse': True,
        'isotropic': True,
        'func': 'inv_linear_augTikho_chol_sparse',
    },
    'algo4': {
        'name': 'algo4',
        'source': 'tofu',
        'family': 'Phillips-Tikhonov',
        'reg_operator': 'any linear',
        'reg_param': 'augTikho',
        'decomposition': '',
        'positive': True,
        'sparse': False,
        'isotropic': True,
        'func': 'inv_linear_augTikho_pos_dense',
    },
    'algo5': {
        'name': 'algo5',
        'source': 'tofu',
        'family': 'Phillips-Tikhonov',
        'reg_operator': 'any linear',
        'reg_param': 'DisPrinc',
        'decomposition': '',
        'positive': False,
        'sparse': True,
        'isotropic': True,
        'func': 'inv_linear_DisPrinc_sparse',
    },
    'algo6': {
        'name': 'algo6',
        'source': 'tofu',
        'family': 'Non-regularized',
        'reg_operator': None,
        'reg_param': None,
        'decomposition': '',
        'positive': False,
        'sparse': True,
        'isotropic': True,
        'func': 'inv_linear_leastsquares_bounds',
    },
    'algo7': {
        'name': 'algo7',
        'source': 'tofu',
        'family': 'Non-regularized',
        'reg_operator': None,
        'reg_param': None,
        'decomposition': '',
        'positive': True,
        'sparse': True,
        'isotropic': True,
        'func': 'inv_linear_leastsquares_bounds',
    },
}
_LREGPARAM_ALGO = [
    'augTikho',
    'DisPrinc',
]


# #############################################################################
# #############################################################################
#                           info
# #############################################################################


def get_available_inversions_algo(
    # for filtering
    source=None,
    family=None,
    reg_operator=None,
    reg_param=None,
    decomposition=None,
    positive=None,
    sparse=None,
    isotropic=None,
    dalgo=None,
    # parameters
    returnas=None,
    verb=None,
):

    # --------------
    # check inputs

    returnas = _generic_check._check_var(
        returnas,
        'returnas',
        default=False,
        allowed=[False, dict, list, str]
    )

    verb = _generic_check._check_var(
        verb,
        'verb',
        default=returnas is False,
        types=bool,
    )

    # ------------------------------------
    # filter according to criteria, if any

    dalgo = match_algo(
        # filtering
        source=source,
        family=family,
        reg_operator=reg_operator,
        reg_param=reg_param,
        decomposition=decomposition,
        positive=positive,
        sparse=sparse,
        isotropic=isotropic,
    )

    # ------------
    # print or str

    if verb is True or returnas is str:

        head = ['key'] + [
            'source', 'family', 'reg_operator', 'reg_param', 'decomposition',
            'positive', 'sparse',
        ]
        sep = ['-'*len(kk) for kk in head]
        lstr = [head, sep] + [
            [k0] + [v0[k1] for k1 in head[1:]]
            for k0, v0 in dalgo.items()
        ]

        nmax = np.max(np.char.str_len(np.char.array(lstr)), axis=0)
        lstr = [
            '  '.join([str(ss).ljust(nmax[ii]) for ii, ss in enumerate(line)])
            for line in lstr
        ]
        msg = "\n".join(lstr)

        if verb:
            print(msg)

    # -------
    # return

    if returnas is dict:
        return dalgo
    elif returnas is list:
        return sorted(dalgo.keys())
    elif returnas is str:
        return msg


def match_algo(
    source=None,
    family=None,
    reg_operator=None,
    reg_param=None,
    decomposition=None,
    positive=None,
    sparse=None,
    isotropic=None,
):

    # ------------
    # check inputs

    dargs = {
        k0: v0 for k0, v0 in locals().items()
        if v0 is not None
    }

    # --------------
    # Get tofu algo

    dalgo = _DALGO

    # -----------------
    # Get tomotok algo

    if tomotok2tofu is not False:
        dalgo.update(tomotok2tofu.get_dalgo())

    # ------------
    # find matches

    if len(dargs) > 0:
        lmatch = [
            k0 for k0, v0 in dalgo.items()
            if all([v0[k1] == v1 for k1, v1 in dargs.items()])
        ]
        if len(lmatch) == 0:
            lstr = [f'\t- {k0}: {v0}' for k0, v0 in dargs.items()]
            msg = (
                "No / several algorithms matching the following criteria:\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)
        dalgo = {k0: v0 for k0, v0 in dalgo.items() if k0 in lmatch}

    return copy.deepcopy(dalgo)


# #############################################################################
# #############################################################################
#                           main
# #############################################################################


def _compute_check(
    # input data
    coll=None,
    key=None,
    key_matrix=None,
    key_data=None,
    key_sigma=None,
    sigma=None,
    # constraints
    dconstraints=None,
    # regularity operator
    solver=None,
    operator=None,
    geometry=None,
    # choice of algo
    algo=None,
    # misc
    conv_crit=None,
    chain=None,
    verb=None,
    store=None,
    # algo and solver-specific options
    kwdargs=None,
    method=None,
    options=None,
):

    # ----
    # keys

    # key_matrix
    lk = list(coll.dobj.get('matrix', {}).keys())
    if key_matrix is None and len(lk):
        key_matrix = lk[0]
    key_matrix = _generic_check._check_var(
        key_matrix, 'key_matrix',
        types=str,
        allowed=lk,
    )
    keybs = coll.dobj['matrix'][key_matrix]['bsplines']
    deg = coll.dobj['bsplines'][keybs]['deg']
    keym = coll.dobj['bsplines'][keybs]['mesh']
    mtype = coll.dobj[coll._which_mesh][keym]['type']
    matrix = coll.ddata[coll.dobj['matrix'][key_matrix]['data']]['data']
    nchan, nbs = matrix.shape[-2:]
    m3d = matrix.ndim == 3
    crop = coll.dobj['matrix'][key_matrix]['crop']

    if np.any(~np.isfinite(matrix)):
        msg = "Geometry matrix should not contain NaNs or infs!"
        raise Exception(msg)

    # key_data
    lk = [
        kk for kk, vv in coll.ddata.items()
        if vv['data'].ndim in [1, 2]
        and vv['data'].shape[-1] == nchan
    ]
    key_data = _generic_check._check_var(
        key_data, 'key_data',
        types=str,
        allowed=lk,
    )
    data = coll.ddata[key_data]['data']
    if data.ndim == 1:
        data = data[None, :]

    # key_sigma
    if key_sigma is not None:
        lk = [
            kk for kk, vv in coll.ddata.items()
            if vv['data'].shape == coll.ddata[key_data]['data'].shape
            or vv['data'].shape == (nchan,)
        ]
        key_sigma = _generic_check._check_var(
            key_sigma, 'key_sigma',
            types=str,
            allowed=lk,
        )
        sigma = coll.ddata[key_sigma]['data']

    elif sigma is None:
        sigma = 0.05
    elif not np.isscalar(sigma):
        msg = "Provide key_sigma xor sigma (as scalar only)!"
        raise Exception(msg)

    if np.isscalar(sigma):
        key_sigma = sigma
        sigma = np.full((1, nchan), sigma*np.nanmean(np.abs(data)))

    if sigma.ndim == 1:
        sigma = sigma[None, :]

    # key_inv
    if coll.dobj.get('inversions') is None:
        ninv = 0
    else:
        ninv = [
            int(kk[3:]) for kk in coll.dobj['inversions']
            if kk.startswith('inv')
            and len(kk) > 3
            and kk[3:].isnumeric()
        ]
        if len(ninv) > 0:
            ninv = np.max(ninv) + 1
        else:
            ninv = 0

    if key is None:
        keyinv = f'inv{ninv}'
    else:
        c0 = (
            isinstance(key, str)
            and (ninv == 0 or key not in coll.dobj['inversions'].keys())
        )
        if not c0:
            msg = (
                "Arg key (used for inversion produced) already exists!\n"
                f"\t- Provided: {key}\n"
                f"\t- Existing: {coll.dobj.get('inversions', {}).keys()}"

            )
            raise Exception(msg)
        keyinv = key

    # -----------
    # constraints

    dconstraints = _check_constraints(
        coll=coll,
        keym=keym,
        mtype=mtype,
        deg=deg,
        dconst=dconstraints,
    )

    # --------------------------------------------
    # Time synchronisation between matrix and data

    # list of keys with potential time-dependence
    lk = [key_data, key_matrix]
    if dconstraints is not None:
        if isinstance(dconstraints.get('rmax'), str):
            lk.append(dconstraints['rmax'])
        if isinstance(dconstraints.get('rmin'), str):
            lk.append(dconstraints['rmin'])

    # check if common / different time dependence
    hastime, hasvect, t, dind = coll.get_time_common(keys=lk)

    # update all accordingly
    if hastime and hasvect:
        lt = [v0['key_vector'] for v0 in dind.values()]

        if len(lt) == 1:
            t = lt[0]

        else:
            # consistency check
            assert m3d

            # matrix side
            matrix = matrix[dind[key_matrix]['ind'], ...]

            # data side
            data = data[dind[key_data]['ind'], :]
            if sigma.shape[0] > 1:
                sigma = sigma[dind[key_data]['ind'], :]

    if m3d:
        assert matrix.shape[0] == data.shape[0]

    # ------------------
    # constraints update

    dconstraints = _update_constraints(
        coll=coll,
        keybs=keybs,
        dconst=dconstraints,
        dind=dind,
        nt=None,
    )

    # --------------
    # inversion refs

    refbs = coll.dobj['bsplines'][keybs]['ref-bs']
    if hastime and hasvect:
        if len(lt) == 1:
            reft = lt[0]
        elif m3d:
            reft = f'{keyinv}-nt'
        refinv = (reft, keybs)
    else:
        refinv = keybs

    notime = (refinv == keybs)

    # --------------------------------------------
    # valid indices of data / sigma

    indok = np.isfinite(data) & np.isfinite(sigma)
    if not np.all(indok):

        # remove channels
        iok = np.any(indok, axis=0)
        if np.any(~iok):

            if not np.any(iok):
                msg = "No valid data (all non-finite)"
                raise Exception(msg)

            msg = (
                "Removed the following channels (all times invalid):\n"
                f"{(~iok).nonzero()[0]}"
            )
            warnings.warn(msg)
            data = data[:, iok]
            sigma = sigma[:, iok]
            matrix = matrix[:, iok, :] if m3d else matrix[iok, :]
            indok = indok[:, iok]

        # remove time steps
        iok = np.any(indok, axis=1)
        if np.any(~iok):
            msg = (
                "Removed the following time steps (all channels invalid):\n"
                f"{(~iok).nonzero()[0]}"
            )
            warnings.warn(msg)
            data = data[iok, :]
            if sigma.shape[0] == iok.size:
                sigma = sigma[iok, :]
            if m3d:
                matrix = matrix[iok, :, :]
            if isinstance(t, np.ndarray):
                t = t[iok]
            indok = indok[iok, :]

            nt, nchan = data.shape

    if np.all(indok):
        indok = None

    # -----------
    # constraints

    err = False
    if isinstance(dconstraints, str):
        dconstraints = None
    elif isinstance(dconstraints, np.ndarray):
        dconstraints = {'coefs': dconstraints}

    if isinstance(dconstraints, dict):
        if dconstraints.get('coefs') is None:
            dconstraints['coefs'] = np.eye(nbs, dtype=float)
        if dconstraints.get('offset') is None:
            dconstraints['offset'] = np.zeros((nbs,), dtype=float)

        c0 = (
            isinstance(dconstraints.get('coefs'), np.ndarray)
            and dconstraints['coefs'].ndim == 2
            and dconstraints['coefs'].shape[1] == nbs
        )
        if not c0:
            err = True
        else:
            new = dconstraints['coefs'].shape[0]
            c0 = (
                isinstance(dconstraints.get('offset'), np.ndarray)
                and dconstraints['offset'].shape == (new,)
            )
            if not c0:
                err = True

    elif dconstraints is not None:
        err = True

    # raise err if relevant
    if err is True:
        msg = (
            "Arg dconstraints must be either:\n"
            "\t- str: valid key of pre-defined constraint\n"
            f"\t- np.ndarray: (nbs, new) constraint matrix\n"
            "\t- dict: {\n"
            "        'coefs':  (nbs, new) np.ndarray,\n"
            "        'offset': (nbs,)     np.ndarray,\n"
            "}\n"
            "\nUsed to define a new vector of unknowns X, from x, such that:\n"
            "    x = coefs * X + offset\n"
            "Then:\n"
            "    Ax = B\n"
            "<=> A(coefs*X + offset) = B\n"
            "<=> (A*coefs)X = (B - A*offset)\n"
            "\nProvided:\n{dconstraints}"
        )
        raise Exception(msg)

    # --------------
    # choice of algo

    # regularization necessary ?
    if nbs >= nchan:
        lok = [
            k0 for k0, v0 in _DALGO.items()
            if v0.get('family') != 'Non-regularized'
        ]
        defalgo = lok[0]
    elif nbs < nchan:
        lok = list(_DALGO.keys())
        defalgo = (
            [
                k0 for k0, v0 in _DALGO.items()
                if v0.get('family') == 'Non-regularized'
            ]
            + [
                k0 for k0, v0 in _DALGO.items()
                if v0.get('family') != 'Non-regularized'
            ]
        )[0]

    algo = _generic_check._check_var(
        algo, 'algo',
        default=defalgo,
        types=str,
        allowed=lok,
    )

    # define dalgo
    dalgo = _DALGO[algo]
    dalgo['name'] = algo

    # check vs deg
    deg = coll.dobj['bsplines'][keybs]['deg']
    if dalgo['source'] == 'tomotok' and dalgo['reg_operator'] == 'MinFisher':
        if deg != 0:
            msg = (
                "MinFisher regularization from tomotok requires deg = 0\n"
                f"\t- deg: {deg}"
            )
            raise Exception(msg)

    # --------------------
    # algo vs dconstraints

    if dalgo['family'] != 'Non-regularized' and dconstraints is not None:
        msg = (
            "Constraints for regularized algorithms not implemented yet!\n"
            f"\t- algo:         {dalgo['name']}\n"
            f"\t- dconstraints: {dconstraints}\n"
        )
        raise NotImplementedError(msg)

    # -------------------
    # regularity operator

    # get operator
    if dalgo['family'] != 'Non-regularized':
        opmat, operator, geometry, dim, ref, crop = coll.add_bsplines_operator(
            key=keybs,
            operator=operator,
            geometry=geometry,
            returnas=True,
            store=False,
            crop=crop,
        )

        if isinstance(opmat, tuple):
            assert all([op.shape == (nbs, nbs) for op in opmat])
        elif opmat.ndim == 1:
            msg = "Inversion algorithm requires a quadratic operator!"
            raise Exception(msg)
        else:
            assert opmat.shape == (nbs,) or opmat.shape == (nbs, nbs)
            opmat = (opmat,)

        if not scpsp.issparse(opmat[0]):
            assert all([np.all(np.isfinite(op)) for op in opmat])
    else:
        opmat = None
        operator = None
        geometry = None
        dim = None
        ref = None

    assert data.shape[1] == nchan
    nt = data.shape[0]

    # -------------------
    # consistent sparsity

    # sparse
    if m3d:
        dalgo['sparse'] = False

    if dalgo['sparse'] is True:
        if not scpsp.issparse(matrix):
            matrix = scpsp.csc_matrix(matrix)
        if opmat is not None and not scpsp.issparse(opmat[0]):
            opmat = [scpsp.csc_matrix(pp) for pp in opmat]
    elif dalgo['sparse'] is False:
        if scpsp.issparse(matrix):
            matrix = matrix.toarray()
        if opmat is not None and scpsp.issparse(opmat[0]):
            opmat = [scpsp.csc_matrix(pp).toarray() for pp in opmat]

    # -----------------------
    # miscellaneous parameter

    # conv_crit
    conv_crit = _generic_check._check_var(
        conv_crit, 'conv_crit',
        default=1e-4,
        types=float,
    )

    # chain
    chain = _generic_check._check_var(
        chain, 'chain',
        default=True,
        types=bool,
    )

    # verb
    verb = _generic_check._check_var(
        verb, 'verb',
        default=True,
        types=(bool, int),
        allowed=[False, 0, True, 1, 2],
    )
    if verb is False:
        verb = 0
    if verb is True:
        verb = 1

    # store
    store = _generic_check._check_var(
        store, 'store',
        default=True,
        types=bool,
    )
    if key_data is None:
        store = False

    # solver
    solver = _generic_check._check_var(
        solver, 'solver',
        default='spsolve',
        types=str,
        allowed=['spsolve'],
    )

    # ----------------------------------------
    # algo-specific kwdargs and solver options

    # kwdargs, method, options
    kwdargs, method, options = _algo_check(
        dalgo,
        kwdargs=kwdargs,
        options=options,
        nchan=nchan,
        nbs=nbs,
        conv_crit=conv_crit,
    )

    return (
        key_matrix, key_data, key_sigma, keybs, keym,
        data, sigma, matrix, t,
        m3d, indok,
        dconstraints,
        opmat, operator, geometry,
        dalgo,
        conv_crit, crop, chain, kwdargs, method, options,
        solver, verb, store,
        keyinv, refinv, reft, notime,
    )


# #############################################################################
# #############################################################################
#                  dconstraints
# #############################################################################


def _check_rminmax(
    coll=None,
    dconst=None,
    rm=None,
    mtype=None,
):

    # if exists
    if dconst.get(rm) is None:
        return

    # check against mesh type
    if mtype != 'polar':
        msg = f"constraint '{rm}' cannot be used with mesh type {mtype}"
        warnings.warn(msg)
        return

    # check format
    err = False
    # should be scalar or valid data key
    if np.isscalar(dconst[rm]):
        pass
    elif isinstance(dconst[rm], str):
        lok = [
            k0 for k0, v0 in coll.ddata.keys()
            if len(v0['ref']) == 1
        ]
        if dconst[rm] not in lok:
            err = True
    else:
        err = True

    # raise err if relevant
    if err:
        msg = (
            f"dconstraints['{rm}'] must be either:\n"
            "\t- a scalar\n"
            "\t- a valid data key with a unique ref\n"
            "Provided: {dconstraints['{rm}']}"
        )
        raise Exception(msg)


def _check_deriv(
    coll=None,
    keym=None,
    mtype=None,
    dconst=None,
    deriv=None,
    deg=None,
):

    # check mesh type
    if mtype != 'polar':
        msg = f"constraint '{rm}' cannot be used with mesh type {mtype}"
        warnings.warn(msg)
        return

    # check format
    err = False
    if isinstance(dconst[deriv], dict):

        # 'rad' and 'val' should be 1d finite arrays of same shape, sorted
        lk = ['rad', 'val']
        if all([k0 in dconst[deriv].keys() for k0 in lk]):
            for k0 in lk:
                dconst[deriv][k0] = np.atleast_1d(dconst[deriv][k0]).ravel()

            if dconst[deriv]['rad'].shape == dconst[deriv]['val'].shape:

                # keep only finite values
                iok = (
                    np.isfinite(dconst[deriv]['rad'])
                    & np.isfinite(dconst[deriv]['val'])
                )
                if not np.all(iok):
                    msg = (
                        "The following constraint rad are non-finite:\n"
                        f"{dconst[deriv]['rad'][~iok]}\n"
                        "  => excluded"
                    )
                    warnings.warn

                dconst[deriv]['rad'] = dconst[deriv]['rad'][iok]
                dconst[deriv]['rad'] = dconst[deriv]['rad'][iok]

                # keep only values in mesh interval
                krad = coll.dobj[coll._which_mesh][keym]['knots'][0]
                rad = coll.ddata[krad]['data']
                iok = (
                    (dconst[deriv]['rad'] >= rad[0])
                    & (dconst[deriv]['rad'] <= rad[-1])
                )
                if not np.all(iok):
                    msg = (
                        "The following constraint rad are out of range:\n"
                        f"{dconst[deriv]['rad'][~iok]}\n"
                        "  => excluded"
                    )
                    warnings.warn

                dconst[deriv]['rad'] = dconst[deriv]['rad'][iok]
                dconst[deriv]['rad'] = dconst[deriv]['rad'][iok]

                # sort vs radius
                inds = np.argsort(dconst[deriv]['rad'])
                dconst[deriv]['rad'] = dconst[deriv]['rad'][inds]
                dconst[deriv]['val'] = dconst[deriv]['val'][inds]

            else:
                err = True
        else:
            err = True
    else:
        err = True

    # raise err
    if err:
        msg = (
            f"dconstraints['{deriv}'] must be dict of 2 same shape 1d arrays\n"
            "-\t 'rad': the radius position at which the constraints are\n"
            "-\t 'val': the values of the derivative at these positions\n"
            f"Provided:\n{dconst}"
        )
        raise Exception(msg)

    # check against deg
    if int(deriv[-1]) > deg:
        msg = (
            f"A constraint on {deriv} can be used for bsplines of degree {deg}"
        )
        raise Exception(msg)

    return dconst


def _check_constraints(
    coll=None,
    keym=None,
    mtype=None,
    deg=None,
    dconst=None,
):

    # ----------------
    # check conformity

    if dconst is None:
        return
    elif not isinstance(dconst, dict):
        msg = f"Arg dconstraints must be a dict!\nProvided: {dconstraints}"
        raise Exception(msg)
    elif mtype != 'polar':
        msg = "Arg dconstraints cannot be used with non-polar mesh!"
        warnings.warn(msg)
        return

    # ----------
    # rmin, rmax

    _check_rminmax(coll=coll, dconst=dconst, rm='rmin', mtype=mtype)
    _check_rminmax(coll=coll, dconst=dconst, rm='rmax', mtype=mtype)

    # -----------
    # derivatives

    for deriv in set(['deriv0', 'deriv1']).intersection(dconst.keys()):
        dconst = _check_deriv(
            coll=coll,
            keym=keym,
            mtype=mtype,
            dconst=dconst,
            deriv=deriv,
            deg=deg,
        )

    return dconst


def _constraints_conflict(dcon=None):
    """  Check whether some constrants are conflicting

    Based on studying which variables are involved by each constraint

    ex:
        offset-only equations sharing any variable are in conflict

        Different equations sharing the exact same variables are in conflict
    """

    lconst = list(dcon.keys())

    if hastime:
        ibs = np.array([
            dcon[k0]['indbs']
            if dcon[k0]['indbs'].ndim == 2
            else np.repeat(dcon[k0]['indbs'][None, :], nt, axis=0)
            for k0 in lconst
        ])
    else:
        ibs = np.array([dcon[k0]['indbs'][None, :] for k0 in lconst])
        nt = 1

    # index of offset-only constraints
    ioffset = np.array([dcon[k0].get('coefs') is None for k0 in lconst])

    # check pure offset
    if np.any(ioffset):
        ibs_offset = ibs[ioffset, :, :]
        lo = np.sum(ibs_offset, axis=0) > 1
        if np.any(lo):
            msg = (
                "Multiple pure offset constraints on bsplines:\n"
                f"{lo.nonzero()[-1]}"
            )
            raise Exception(msg)

    # if np.unique(ibs, axis=0).shape[]
    if np.unique(ibs, axis=0).shape[0] < ibs.shape[0]:
        msg = "There seem to be several mutually exclusive equations!"
        raise Exception(msg)


def _update_constraints(
    coll=None,
    keybs=None,
    dconst=None,
    dind=None,
    nt=None,
):

    if dconst is None:
        return

    # ---------------------------------------------
    # initialize bool index of constrained bsplines

    clas = coll.dobj['bsplines'][keybs]['class']
    dcon = {}

    # ----------
    # rmin, rmax

    for rm in ['rmin', 'rmax']:

        if dconst.get(rm) is None:
            continue

        # make it a 1d vector with good shape
        if isinstance(dconst.get(rm), str):
            rmax = dconst[rm]
            if dind is not None and rm in dind.keys():
                dconst[rm] = coll.ddata[rm]['data'][dind[rm]['ind']]
            else:
                dconst[rm] = coll.ddata[rm]['data']

        else:
            dconst[rm] = np.full((nt,), dconst[rm])

        # compute indbs
        indbs, offset = clas.get_constraints_out_rlim(rlim=dconst[rm], rm=rm)

        dcon[rm] = {
            'indbs': indbs,
            'offset': offset,
        }

    # ------
    # deriv

    for deriv in ['deriv0', 'deriv1']:
        if dconst.get(deriv) is None:
            continue

        # get coefs
        indbs, coefs, offset = clas.get_constraints_deriv(
            deriv=deriv,
            rad=dconst[deriv]['rad'],
            val=dconst[deriv]['val'],
        )

        # assemble
        if not np.any(indbs):
            msg = f"Constraint {deriv} not used (no match)"
            warnings.warn(msg)
            continue

        for ii in range(dconst[deriv]['rad'].size):
            dcon[f'{deriv}-{ii}'] = {
                'indbs': indbs[ii, :],
                'coefs': coefs[ii, :],
                'offset': offset[ii, :],
            }

    l2d = [k0 for k0, v0 in dcon.items() if v0['indbs'].ndim == 2]
    hastime = len(l2d) > 0
    if hastime:
        lnt = list(set([dcon[k0]['indbs'].shape[0] for k0 in l2d]))
        assert len(lnt) == 1
        nt = lnt[0]

    # ---------------
    # check conflicts

    lconst = list(dcon.keys())
    doffset = {
        k0: ii for ii, k0 enumerate(lconst)
        if dcon[k0].get('coefs') is None
    }
    dcoefs = {
        k0: ii for ii, k0 enumerate(lconst)
        if dcon[k0].get('coefs') is not None
    }
    _constraints_conflict(
        dcon=dcon,
        lconst=lconst,
        doffset=doffset,
        dcoefs=dcoefs,
    )

    # --------------------------
    # select bsplines to be removed as results of equations

    if np.any(ioffset):
        iout = np.array([
            np.any(ibs_offset[:, ii, :], axis=0)
            for ii in range(nt)
        ])
    else:
        iout = np.zeros((nt, ibs.shape[-1]), dtype=bool)

    ibs_coefs = ibs[~ioffset, :, :]
    if np.any(~ioffset):
        for ii in range(nt):
            icom = iout[ii:ii+1, :] & ibs_coefs[:, ii, :]

            icontra = np.all(icom, axis=0)
            if np.any(icontra):
                msg = (
                    "Contradictory constraints on bsplines:\n"
                    f"{icontra.nonzero()[0]}"
                )
                raise Exception(msg)

            if np.all(np.any(icom, axis=0)):
                # all equations already have an identified variable from offset
                pass
            else:
                icom_no = ~np.any(icom, axis=1)
                if icom_no.sum() == 1:
                    iout[ii, ibs_coefs[icom_no, ii, :].nonzero()[0][0]] = True
                else:
                    msg = "Multiple equations not handled yet"
                    raise Exception(msg)

    # --------------------
    # build coefs / offset

    nbs = ibs.shape[-1]
    coefs, offset = [], []
    for ii in range(nt):
        nbsi = nbs - iout[ii, :].sum()
        ci = np.zeros((nbs, nbsi), dtype=float)
        oi = np.zeros((nbs,), dtype=float)

        # offset only
        if np.any(ioffset):
            for jj, k0 in enumerate(ioffset.sum()):
                k0 = lconst[ioffset.nonzero()[0][jj]]
                ind = ibs_offset[jj, ii, :]
                oi[ind] = dcon[k0]['offset'][ind]

        # coefs + offset
        if np.any(~ioffset):
            import pdb; pdb.set_trace()     # DB
            ibsi = None
            ci[ibsi] = None

    # -------------
    # format output

    if nt == 1:
        indbs = indbs[0]
        coefs = coefs[0]
        offset = offset[0]
        hastime = False
    else:
        hastime = True

    dcon = {
        'hastime': hastime,
        'indbs': None,
        'coefs': None,
        'offset': None,
    }

    return dconstraints, dcon


# #############################################################################
# #############################################################################
#                  kwdargs / options for each algo
# #############################################################################


def _algo_check(
    dalgo,
    kwdargs=None,
    method=None,
    options=None,
    nchan=None,
    nbs=None,
    conv_crit=None,
):

    # ------------------------
    # generic kwdargs

    # kwdargs
    if kwdargs is None:
        kwdargs = {}

    # generic kwdargs
    if kwdargs.get('maxiter') is None:
        kwdargs['maxiter'] = 100

    if kwdargs.get('tol') is None:
        kwdargs['tol'] = 1.e-6

    # ------------------------
    # algo specific kwdargs

    # kwdargs specific to aug. tikhonov
    if dalgo['reg_param'] == 'augTikho':
        a0 = kwdargs.get('a0', 10)
        a1 = kwdargs.get('a1', 2)

        # to have [x]=1
        kwdargs['b0'] = np.math.factorial(a0)**(1 / (a0 + 1))
        kwdargs['b1'] = np.math.factorial(a1)**(1 / (a1 + 1))
        kwdargs['a0'] = a0
        kwdargs['a1'] = a1

        # Exponent for rescaling of a0bis
        # typically in [1/3 ; 1/2], but real limits are 0 < d < 1 (or 2 ?)
        if kwdargs.get('d') is None:
            kwdargs['d'] = 0.95

        if kwdargs.get('conv_reg') is None:
            kwdargs['conv_reg'] = True

        if kwdargs.get('nbs_fixed') is None:
            kwdargs['nbs_fixed'] = True

        if kwdargs['nbs_fixed']:
            kwdargs['a0bis'] = kwdargs['a0'] - 1. + 1200./2.
        else:
            kwdargs['a0bis'] = kwdargs['a0'] - 1. + nbs/2.
        kwdargs['a1bis'] = kwdargs['a1'] - 1. + nchan/2.

    # kwdargs specific to discrepancy principle
    elif dalgo['reg_param'] == 'DisPrinc':
        if kwdargs.get('chi2n_obj') is None:
            kwdargs['chi2n_obj'] = 1.
        if kwdargs.get('chi2n_tol') is None:
            kwdargs['chi2n_tol'] = 0.05

    # ------------------------
    # low-level solver options

    if dalgo['positive'] is True:

        if options is None:
            options = {}

        if method is None:
            if dalgo['name'] == 'algo7':
                method = 'trf'
            else:
                method = 'L-BFGS-B'

        if method == 'L-BFGS-B':
            if options.get('ftol') is None:
                options['ftol'] = conv_crit/100.
            if options.get('disp') is None:
                options['disp'] = False
        elif dalgo['name'] != 'algo7':
            raise NotImplementedError

    return kwdargs, method, options


# ##################################################################
# ##################################################################
#               _DALGO at import time
# ##################################################################


_DALGO = get_available_inversions_algo(returnas=dict, verb=False)
