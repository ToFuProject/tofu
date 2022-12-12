# -*- coding: utf-8 -*-


# Built-in
import copy
import warnings


# Common
import numpy as np
import scipy.sparse as scpsp
import datastock as ds


# optional
try:
    from .. import tomotok2tofu
except Exception as err:
    tomotok2tofu = False


_SIGMA = 0.05


_DALGO0 = {
    'algo0': {
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

    returnas = ds._generic_check._check_var(
        returnas,
        'returnas',
        default=False,
        allowed=[False, dict, list, str]
    )

    verb = ds._generic_check._check_var(
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

    # complemet with name
    for k0 in dalgo.keys():
        dalgo[k0]['name'] = k0

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

    dalgo = _DALGO0

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


# ##################################################################
# ##################################################################
#                           main
# ##################################################################


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
    **kwargs,
):

    # ----
    # key

    # key_matrix
    lk = list(coll.dobj.get('geom matrix', {}).keys())
    key_matrix = ds._generic_check._check_var(
        key_matrix, 'key_matrix',
        types=str,
        allowed=lk,
    )

    key_diag = coll.dobj['geom matrix'][key_matrix]['diagnostic']
    key_cam = coll.dobj['geom matrix'][key_matrix]['camera']

    keybs = coll.dobj['geom matrix'][key_matrix]['bsplines']
    deg = coll.dobj['bsplines'][keybs]['deg']
    keym = coll.dobj['bsplines'][keybs]['mesh']
    mtype = coll.dobj[coll._which_mesh][keym]['type']

    # matrix itself
    matrix, ref, dind = coll.get_geometry_matrix_concatenated(key_matrix)
    lkmat = coll.dobj['geom matrix'][key_matrix]['data']
    units_gmat = coll.ddata[lkmat[0]]['units']
    nchan, nbs = matrix.shape[-2:]
    m3d = matrix.ndim == 3
    crop = coll.dobj['geom matrix'][key_matrix]['crop']

    if np.any(~np.isfinite(matrix)):
        msg = "Geometry matrix should not contain NaNs or infs!"
        raise Exception(msg)

    # datai
    ddata = _check_data(
        coll=coll,
        key_diag=key_diag,
        key_data=key_data,
    )

    # sigma
    dsigma = _check_sigma(
        coll=coll,
        key_diag=key_diag,
        key_sigma=key_sigma,
        sigma=sigma,
        ddata=ddata,
        nchan=nchan,
    )

    # key_inv
    key = ds._generic_check._obj_key(
        d0=coll.dobj.get('inversions', {}),
        short='inv',
        key=key,
    )

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
    lk = ddata['keys'] + coll.dobj['geom matrix'][key_matrix]['data']
    if dconstraints is not None:
        if isinstance(dconstraints.get('rmax', {}).get('val'), str):
            lk.append(dconstraints['rmax']['val'])
        if isinstance(dconstraints.get('rmin', {}).get('val'), str):
            lk.append(dconstraints['rmin']['val'])

    # check if common / different time dependence
    hastime, reft, keyt, t, dind = coll.get_time_common(keys=lk)
    if reft is None:
        reft = f'{key}-nt'

    # update all accordingly
    if hastime and dind is not None:
        # matrix side
        if dind.get(key_matrix, {}).get('ind') is not None:
            matrix = matrix[dind[key_matrix]['ind'], ...]

        # data side
        c0 = any([
            k0 in dind.keys() for k0 in ddata['keys']
            if dind[k0].get('ind') is not None
        ])
        if c0:
            assert all([k0 in dind.keys() for k0 in ddata['keys']])
            lind = [dind[k0]['ind'] for k0 in ddata['keys']]
            assert all([iii.size == lind[0].size for iii in lind[1:]])

            ind0 = lind[0]
            assert np.allclose(lind, ind0[None, :])

            ddata['data'] = ddata['data'][ind0, :]
            if dsigma['data'].shape[0] > 1:
                dsigma['data'] = dsigma['data'][ind0, :]

    if m3d:
        assert matrix.shape[0] == ddata['data'].shape[0]

    # ------------------
    # constraints update

    dconstraints, dcon, iconflict = _update_constraints(
        coll=coll,
        keybs=keybs,
        dconst=dconstraints,
        dind=dind,
        nbs=nbs,
    )

    # --------------
    # inversion refs

    refbs = coll.dobj['bsplines'][keybs]['ref-bs']
    if hastime:
        refinv = (reft, keybs)
    else:
        refinv = keybs

    notime = (refinv == keybs)

    # --------------
    # choice of algo

    if algo is None:
        msg = (
            "\nPlease provide an algorithm, to be chosen from:\n"
            + get_available_inversions_algo(verb=False, returnas=str)
        )
        raise Exception(msg)

    # regularization necessary ?
    if nbs >= nchan:
        lok = [
            k0 for k0, v0 in _DALGO.items()
            if v0.get('family') != 'Non-regularized'
        ]
    else:
        lok = list(_DALGO.keys())

    algo = ds._generic_check._check_var(
        algo, 'algo',
        types=str,
        allowed=lok,
    )

    # define dalgo
    dalgo = _DALGO[algo]

    # check vs deg
    deg = coll.dobj['bsplines'][keybs]['deg']
    if dalgo['source'] == 'tomotok' and dalgo['reg_operator'] == 'MinFisher':
        if deg != 0:
            msg = (
                "MinFisher regularization from tomotok requires deg = 0\n"
                f"\t- deg: {deg}"
            )
            raise Exception(msg)

    # regul
    regul = dalgo['family'] != 'Non-regularized'

    # --------------------------------------------
    # valid chan / time indices of data / sigma (+ constraints)

    indok = np.isfinite(ddata['data']) & np.isfinite(dsigma['data'])
    if not np.all(indok):

        # remove channels
        iok = np.any(indok, axis=0)
        if np.any(~iok):

            if not np.any(iok):
                msg = "No valid data (all non-finite)"
                raise Exception(msg)

            # raise warning
            msg = (
                "Removed the following channels (all times invalid):\n"
                f"{(~iok).nonzero()[0]}"
            )
            warnings.warn(msg)

            # update
            ddata['data'] = ddata['data'][:, iok]
            dsigma['data'] = dsigma['data'][:, iok]
            matrix = matrix[:, iok, :] if m3d else matrix[iok, :]
            indok = indok[:, iok]

        # remove time steps
        (
            indok, ddata['data'], dsigma['data'],
            matrix, t, dcon, iokt,
        ) = _check_time_steps(
            indok=indok,
            data=ddata['data'],
            sigma=dsigma['data'],
            matrix=matrix,
            t=t,
            m3d=m3d,
            dcon=dcon,
            regul=regul,
            iconflict=iconflict,
        )

        # get new nt, nchan
        nt, nchan = ddata['data'].shape
    else:
        iokt = np.ones((ddata['data'].shape[0],), dtype=bool)

    if np.all(indok):
        indok = None

    # --------------------
    # algo vs dconstraints

    if regul and dconstraints is not None:
        msg = (
            "Constraints for regularized algorithms not implemented yet!\n"
            f"\t- algo:         {dalgo['name']}\n"
            f"\t- dconstraints: {dconstraints}\n"
        )
        raise NotImplementedError(msg)

    # -------------------
    # regularity operator

    # get operator
    if regul:
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

    assert ddata['data'].shape[1] == nchan
    nt = ddata['data'].shape[0]

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
    conv_crit = ds._generic_check._check_var(
        conv_crit, 'conv_crit',
        default=1e-4,
        types=float,
    )

    # chain
    chain = ds._generic_check._check_var(
        chain, 'chain',
        default=True,
        types=bool,
    )

    # verb
    verb = ds._generic_check._check_var(
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
    store = ds._generic_check._check_var(
        store, 'store',
        default=True,
        types=bool,
    )
    if key_data is None:
        store = False

    # solver
    solver = ds._generic_check._check_var(
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
        key_matrix,
        key_diag, key_data, key_sigma,
        keybs, keym, mtype,
        ddata, dsigma, matrix, units_gmat,
        keyt, t, reft, notime,
        m3d, indok, iokt,
        dconstraints,
        opmat, operator, geometry,
        dalgo, dconstraints, dcon,
        conv_crit, crop, chain, kwdargs, method, options,
        solver, verb, store,
        key, refinv, regul,
    )


# #############
#  input data
# #############


def _check_data(
    coll=None,
    key_diag=None,
    key_data=None,
):

    # load ddata from key_data
    ddata = coll.get_diagnostic_data_concatenated(
        key=key_diag,
        key_data=key_data,
        flat=True,
    )

    # make sure one time step is present
    if ddata['data'].ndim == 1:
        ddata['data'] = ddata['data'][None, :]

    assert ddata['data'].ndim == 2, "Wrong dimensions for inversion input!"

    return ddata


def _check_sigma(
    coll=None,
    key_diag=None,
    key_sigma=None,
    sigma=None,
    ddata=None,
    nchan=None,
):

    if key_sigma is None:

        if sigma is None:
            sigma = _SIGMA

        if np.isscalar(sigma):
            mean = np.repeat(
                np.nanmean(np.abs(ddata['data']), axis=1)[:, None],
                nchan,
                axis=1,
            )
            sigma = sigma * mean

        else:
            msg = "Provide key_sigma xor sigma (as scalar only)!"
            raise Exception(msg)

        dsigma = {
            'data': sigma,
            'units': ddata['units'],
            'ref': None,
            'axis': None,
            'flat': None,
            'dind': None,
        }

    else:
        dsigma = _check_data(
            coll=coll,
            key=key_diag,
            key_data=key_sigma,
            flat=True,
        )

        # make sure one time step is present
        if dsigma['data'].ndim == 1:
            dsigma['data'] = dsigma['data'][None, :]

        assert dsigma['data'].ndim == 2, "Wrong dimensions for inversion input!"

    return dsigma


# ##################################################################
# ##################################################################
#                  removing time steps
# ##################################################################


def _remove_time_steps(
    iok=None,
    indok=None,
    data=None,
    sigma=None,
    matrix=None,
    m3d=None,
    dcon=None,
):
    # update
    data = data[iok, :]
    if sigma.shape[0] == iok.size:
        sigma = sigma[iok, :]
    if m3d:
        matrix = matrix[iok, :, :]
    indok = indok[iok, :]

    # update constraints too if relevant
    if dcon is not None and dcon['hastime']:
        dcon['indbs_free'] = dcon['indbs_free'][iok, :]
        dcon['indbs'] = [
            vv for ii, vv in enumerate(dcon['indbs']) if iok[ii]
        ]
        dcon['offset'] = dcon['offset'][iok, :]
        dcon['coefs'] = [
            vv for ii, vv in enumerate(dcon['coefs']) if iok[ii]
        ]

    return indok, data, sigma, matrix, dcon


def _check_time_steps(
    indok=None,
    data=None,
    sigma=None,
    matrix=None,
    t=None,
    m3d=None,
    dcon=None,
    regul=None,
    iconflict=None,
):

    # ---------------------
    # check data validity

    # remove time steps
    iok = np.any(indok, axis=1)
    if iconflict is not None:
        iok &= (~iconflict)

    iokt = iok

    if np.any(~iok):

        # raise warning
        msg = (
            "Removed the following time steps (all channels invalid):\n"
            f"{(~iok).nonzero()[0]}"
        )
        warnings.warn(msg)

        # apply
        indok, data, sigma, matrix, dcon = _remove_time_steps(
            iok=iok,
            indok=indok,
            data=data,
            sigma=sigma,
            matrix=matrix,
            m3d=m3d,
            dcon=dcon,
        )

    # --------------------------------------------------
    # check underdetermintion for non-regularized algos

    if not regul:

        # remove time steps with nbs > nchan or bs not contrained
        nt = data.shape[0]
        nbs = matrix.shape[-1]
        if dcon is None:
            ifree = np.ones((nt, nbs), dtype=bool)
        else:
            if dcon['hastime']:
                ifree = dcon['indbs_free']
            else:
                ifree = np.repeat(dcon['indbs_free'], nt, axis=0)

        if m3d:
            litout = np.array([
                np.any(np.all(
                    matrix[it, indok[it, :], :][:, ifree[it, :]] == 0,
                    axis=0,
                ))
                for it in range(nt)
            ])
        else:
            litout = np.array([
                np.any(np.all(
                    matrix[indok[it, :], :][:, ifree[it, :]] == 0,
                    axis=0,
                ))
                for it in range(nt)
            ])

        # raise warning
        if np.any(litout):
            msg = (
                "The following time steps are under-determined:\n"
                f"{litout.nonzero()[0]}\n"
                "  => They are removed (incompatible with non-regularized)"
            )
            warnings.warn(msg)

            # apply
            iokt[iokt] = ~litout
            indok, data, sigma, matrix, dcon = _remove_time_steps(
                iok=~litout,
                indok=indok,
                data=data,
                sigma=sigma,
                matrix=matrix,
                m3d=m3d,
                dcon=dcon,
            )

    return indok, data, sigma, matrix, t, dcon, iokt




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
    lok = [k0 for k0, v0 in coll.ddata.items() if len(v0['ref']) == 1]
    if isinstance(dconst[rm], str) and dconst[rm] in lok:
        dconst[rm] = {'val': dconst[rm]}
    elif np.isscalar(dconst[rm]):
        dconst[rm] = {'val': dconst[rm]}

    # dict
    err = True
    if isinstance(dconst[rm], dict):
        c0 = (
            dconst[rm].get('val') is not None
            and np.isscalar(dconst[rm]['val'])
            or (
                isinstance(dconst[rm]['val'], str)
                and dconst[rm]['val'] in lok
            )
        )
        if c0:
            err = False

    # raise err if relevant
    if err:
        msg = (
            f"dconstraints['{rm}'] must be either:\n"
            "\t- a scalar\n"
            "\t- a valid data key with a unique ref\n"
            f"Provided: {dconst[rm]}"
        )
        raise Exception(msg)

    # lim
    lok = ['inner', 'outer']
    if rm == 'rmax':
        lok.append('allout')
        dok = 'allout'
    else:
        lok.append('allin')
        dok = 'allin'
    dconst[rm]['lim'] = ds._generic_check._check_var(
        dconst[rm].get('lim'), "dconstraints['{rm}']['lim']",
        default=dok,
        types=str,
        allowed=lok,
    )


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

    # copy to avoid modifying reference
    dconst = copy.deepcopy(dconst)

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


def _constraints_conflict_iout(
    dcon=None,
    loffset=None,
    lcoefs=None,
    nt=None,
):
    """  Check whether some constrants are conflicting

    Based on studying which variables are involved by each constraint

    ex:
        offset-only equations sharing any variable are in conflict

        Different equations sharing the exact same variables are in conflict
    """

    # -----------------
    # check pure offset

    # aggregated ibs for checking global consistency
    if len(loffset) > 0:
        ibs_offset = np.array([dcon[k0]['indbs'] for k0 in loffset])

        # check offset consistency
        lo = np.sum(ibs_offset, axis=0) > 1
        if np.any(lo):
            msg = (
                "Multiple pure offset constraints on bsplines:\n"
                f"{lo.nonzero()[-1]}"
            )
            raise Exception(msg)
        nt, nbs = ibs_offset.shape[1:]

    # aggregate coefs
    if len(lcoefs) > 0:
        ibs_coefs = np.array([dcon[k0]['indbs'] for k0 in lcoefs])

        # check redundancy of coefs equations
        if np.unique(ibs_coefs, axis=0).shape[0] < ibs_coefs.shape[0]:
            msg = "There seem to be several redundant equations!"
            raise Exception(msg)

        # assuming all equations are strictly independent from each other
        if np.any(np.sum(ibs_coefs, axis=0) > 1):
            msg = (
                "Current version only manages strictly independent equations!"
            )
            raise NotImplementedError(msg)

        if len(loffset) > 0:
            assert (nt, nbs) == ibs_coefs.shape[1:], ibs_coefs.shape
        else:
            nt, nbs = ibs_coefs.shape[1:]

    # -----------------
    # check coefs vs offset

    if len(loffset) > 0 and len(lcoefs) > 0:
        # (nt, nbs) array with True for offset
        iout_offset = np.any(ibs_offset, axis=0)
        for ii, k0 in enumerate(lcoefs):
            lt = np.all(iout_offset[ibs_coefs[ii, ...]], axis=-1).nonzero()[0]
            if lt.size > 0:
                msg = (
                    f"Redundancy between equation {k0} and offsets!\n"
                    f"For the following time steps: {lt}"
                )
                raise Exception(msg)

        iconflict = np.any(
            np.any(ibs_coefs & iout_offset[None, ...], axis=0),
            axis=-1,
        )
        # if np.any(iconflict):
            # msg = "Auto-solve mutually dependent equations not implemented yet!"
            # raise NotImplementedError(msg)
    elif len(loffset) > 0:
        iout_offset = np.any(ibs_offset, axis=0)
        iconflict = np.zeros((nt,), dtype=bool)
    elif len(lcoefs) > 0:
        iout_offset = np.zeros((nt, nbs), dtype=bool)
        iconflict = np.zeros((nt,), dtype=bool)
    else:
        msg = "No detected constraints => should have stopped earlier!"
        raise Exception(msg)

    # ------------------
    # get really free equations

    # from here all equations are independent from offsets
    iout_coefs = np.zeros((nt, nbs), dtype=bool)
    iin_coefs = ~iout_offset

    for ii, k0 in enumerate(lcoefs):
        dcon[k0]['iout'] = np.zeros((nt,), dtype=int)
        dcon[k0]['iin'] = [
            np.zeros((ibs_coefs[ii, it, :].sum() - 1,), dtype=int)
            for it in range(nt)
        ]

        for it in range(ibs_coefs.shape[1]):
            # get first variable
            indout = ibs_coefs[ii, it, :].nonzero()[0][0]
            indin = ibs_coefs[ii, it, :].nonzero()[0][1:]

            iout_coefs[it, indout] = True
            iin_coefs[it, indout] = False

            dcon[k0]['iout'][it] = indout
            dcon[k0]['iin'][it][:] = indin

    # final consistency check
    lterr = np.any(
        (iout_offset & iout_coefs & iin_coefs)[~iconflict, :],
        axis=1,
    )
    if np.any(lterr):
        msg = (
            "Inconsistency for the following time steps:\n"
            f"{lterr.nonzero()[0]}"
        )
        raise Exception(msg)

    return iout_offset, iout_coefs, iin_coefs, iconflict


def _update_constraints(
    coll=None,
    keybs=None,
    dconst=None,
    dind=None,
    nbs=None,
):

    if dconst is None:
        return None, None, None

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
        if isinstance(dconst[rm]['val'], str):
            rmstr = dconst[rm]['val']
            if dind is not None and rmstr in dind.keys():
                dconst[rm]['val'] = coll.ddata[rmstr]['data'][dind[rmstr]['ind']]
            else:
                dconst[rm]['val'] = coll.ddata[rmstr]['data']

        else:
            dconst[rm]['val'] = dconst[rm]['val']


        # compute indbs
        indbs, offset = clas.get_constraints_out_rlim(
            rm=rm,
            rlim=dconst[rm]['val'],
            lim=dconst[rm]['lim'],
        )

        dcon[rm] = {
            'indbs': indbs,
            'offset': offset,
        }

    # ------
    # deriv

    for deriv in ['deriv0', 'deriv1']:
        if dconst.get(deriv) is None:
            continue

        # get indbs, coefs and offset as 3 (nconstraints, nbs) arrays
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
    else:
        nt = 1

    # harmonize shapes to (nt, nbs)
    if hastime:
        for k0, v0 in dcon.items():
            if v0['indbs'].ndim == 1:
                dcon[k0]['indbs'] = np.repeat(v0['indbs'][None, :], nt, axis=0)
            else:
                assert dcon[k0]['indbs'].shape[0] == nt
    else:
        for k0, v0 in dcon.items():
            dcon[k0]['indbs'] = v0['indbs'][None, :]

    # ---------------
    # check conflicts

    loffset = [k0 for k0 in dcon.keys() if dcon[k0].get('coefs') is None]
    lcoefs = [k0 for k0 in dcon.keys() if dcon[k0].get('coefs') is not None]
    iout_offset, iout_coefs, iin_coefs, iconflict = _constraints_conflict_iout(
        dcon=dcon,
        loffset=loffset,
        lcoefs=lcoefs,
        nt=nt,
    )

    # --------------------
    # build coefs / offset

    nbsnew = iin_coefs.sum(axis=1)
    indbs = [iin_coefs[ii, :].nonzero()[0] for ii in range(nt)]
    coefs = [np.zeros((nbs, nbsnew[ii]), dtype=float) for ii in range(nt)]
    offset = np.zeros((nt, nbs), dtype=float)

    for k0 in loffset:
        ind = dcon[k0]['indbs']
        offset[ind] = np.repeat(dcon[k0]['offset'][None, :], nt, axis=0)[ind]

    for k0 in lcoefs:
        for it in range(nt):
            iout = dcon[k0]['iout'][it]
            iin = dcon[k0]['iin'][it]
            ci = -dcon[k0]['coefs'][iin] / dcon[k0]['coefs'][iout]
            oi = dcon[k0]['offset'][iout] / dcon[k0]['coefs'][iout]
            offset[it, iout] = oi
            ibsin = np.array([(indbs[it] == i0).nonzero()[0][0] for i0 in iin])
            coefs[it][iout, ibsin] = ci

    for it in range(nt):
        coefs[it][tuple(indbs[it]), tuple(np.arange(indbs[it].size))] = 1.

    # -------------
    # format output

    if nt == 1:
        hastime = False
    else:
        hastime = True

    dcon = {
        'hastime': hastime,
        'indbs': indbs,
        'coefs': coefs,
        'offset': offset,
        'indbs_free': iin_coefs,        # bool indices of free variables
    }

    return dconst, dcon, iconflict


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
