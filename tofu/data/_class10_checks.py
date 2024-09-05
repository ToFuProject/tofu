# -*- coding: utf-8 -*-


# Built-in
import copy
import warnings
import math


# Common
import numpy as np
import scipy.sparse as scpsp
import datastock as ds


from . import _class10_algos as _algos
from . import _class10_refs as _refs


_SIGMA = 0.05
_LREGPARAM_ALGO = [
    'augTikho',
    'DisPrinc',
]


# ################################################################
# ################################################################
#                           main
# ################################################################


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
    maxiter_outer=None,
    # misc
    conv_crit=None,
    chain=None,
    verb=None,
    store=None,
    # algo and solver-specific options
    kwdargs=None,
    method=None,
    options=None,
    # ref vector specifiers
    dref_vector=None,
    ref_vector_strategy=None,
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
    nd = coll.dobj[coll._which_mesh][keym]['nd']
    mtype = coll.dobj[coll._which_mesh][keym]['type']

    # matrix itself
    matrix, ref, dind = coll.get_geometry_matrix_concatenated(
        key_matrix,
        key_cam=key_cam,
    )
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
        key_cam=key_cam,
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
        nd=nd,
        mtype=mtype,
        deg=deg,
        dconst=dconstraints,
    )

    # --------------------------------------------
    # Time synchronisation between matrix and data

    # check if common / different time dependence
    hastime, reft, keyt, t, dindt = _refs._get_ref_vector_common(
        coll=coll,
        ddata=ddata,
        key_matrix=key_matrix,
        dconstraints=dconstraints,
        dref_vector=dref_vector,
        strategy=ref_vector_strategy,
    )

    if reft is None:
        reft = f'{key}_nt'

    # update all accordingly
    if hastime and dindt is not None:
        # matrix side
        lkmat = coll.dobj['geom matrix'][key_matrix]['data']
        c0 = any([dindt.get(k0, {}).get('ind') is not None for k0 in lkmat])
        if c0:
            matrix = matrix[dindt[lkmat[0]]['ind'], ...]

        # data side
        c0 = any([
            k0 in dindt.keys() for k0 in ddata['keys']
            if dindt[k0].get('ind') is not None
        ])
        if c0:
            assert all([k0 in dindt.keys() for k0 in ddata['keys']])
            lind = [dindt[k0]['ind'] for k0 in ddata['keys']]
            assert all([iii.size == lind[0].size for iii in lind[1:]])

            ind0 = lind[0]
            assert np.allclose(lind, ind0[None, :])

            ddata['data'] = ddata['data'][ind0, :]
            if dsigma['data'].shape[0] > 1:
                dsigma['data'] = dsigma['data'][ind0, :]

    if m3d:
        if matrix.ndim != 3 or matrix.shape[0] != ddata['data'].shape[0]:
            msg = (
                "Inconsistent interpretation of matrix and data shapes:\n"
                f"\t- m3d: {m3d}\n"
                f"\t- matrix.shape: {matrix.shape}\n"
                f"\t- ddata['data'].shape: {ddata['data'].shape}\n"
            )
            raise Exception(msg)

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

    # refbs = coll.dobj['bsplines'][keybs]['ref_bs']
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
            + _algos.get_available_inversions_algo(verb=False, returnas=str)
        )
        raise Exception(msg)

    # regularization necessary ?
    if nbs >= nchan:
        lok = [
            k0 for k0, v0 in _algos._DALGO.items()
            if v0.get('family') != 'Non-regularized'
        ]
    else:
        lok = list(_algos._DALGO.keys())

    algo = ds._generic_check._check_var(
        algo, 'algo',
        types=str,
        allowed=lok,
    )

    # define dalgo
    dalgo = _algos._DALGO[algo]

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

    indok = (
        np.isfinite(ddata['data'])
        & np.isfinite(dsigma['data'])
    )

    # add matrix = 0 to indok
    if m3d is True:
        indok &= (np.sum(matrix, axis=-1) > 0)
    else:
        indok &= (np.sum(matrix, axis=-1) > 0)[None, ...]

    # if relevant, permanently remove some channels
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

    # -------------------
    # regularity operator

    # get operator
    if regul:

        if operator is None or 'N2' not in operator:
            msg = (
                "Quadratic operator needed for inversions!"
                f"Provided: {operator}"
            )
            raise Exception(msg)

        dopmat, dpar = coll.add_bsplines_operator(
            key=keybs,
            operator=operator,
            geometry=geometry,
            returnas=True,
            return_param=True,
            store=False,
            crop=crop,
        )

        lk = dpar['keys']
        operator = dpar['operator']
        geometry = dpar['geometry']
        crop = dpar['crop']

        # safety check 0
        lfail = [
            k0 for k0 in lk
            if not (
                dopmat[k0]['data'].shape == (nbs, nbs)
                and (
                    scpsp.issparse(dopmat[k0]['data'])
                    or np.all(np.isfinite(dopmat[k0]['data']))
                )
            )
        ]
        if len(lfail) > 0:
            lstr = [f'\t- {k0}' for k0 in lfail]
            msg = (
                "Wrong operator shape or non-finite values!\n"
                f"\t- operator: {operator}\n"
                f"\t- operator shape: {dopmat[lk[0]]['data'].shape}\n"
                f"\t- expected: {(nbs, nbs)}\n\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

    else:
        dopmat = None
        operator = None
        geometry = None

    assert ddata['data'].shape[1] == nchan
    # nt = ddata['data'].shape[0]

    # -------------------
    # consistent sparsity

    # sparse
    if m3d:
        dalgo['sparse'] = False

    if dalgo['sparse'] is True:
        if not scpsp.issparse(matrix):
            matrix = scpsp.csc_matrix(matrix)
        if dopmat is not None:
            for k0, v0 in dopmat.items():
                if not scpsp.issparse(v0['data']):
                    dopmat[k0]['data'] = scpsp.csc_matrix(v0['data'])
    elif dalgo['sparse'] is False:
        if scpsp.issparse(matrix):
            matrix = matrix.toarray()
        if dopmat is not None:
            for k0, v0 in dopmat.items():
                if scpsp.issparse(v0['data']):
                    dopmat[k0]['data'] = v0['data'].toarray()

    # -----------------------
    # miscellaneous parameter

    # conv_crit
    conv_crit = ds._generic_check._check_var(
        conv_crit, 'conv_crit',
        default=1e-4,
        types=float,
    )

    # maxiter_outer
    maxiter_outer = ds._generic_check._check_var(
        maxiter_outer, 'maxiter_outer',
        default=1000,
        types=(int, float),
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
        allowed=[False, 0, True, 1, 2, 3],
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
        # scaling parameters for the hyperparameters of augTikhonov
        sigma_rel=np.nanmean(dsigma['data']) / np.nanmean(ddata['data']),
    )

    return (
        key_matrix,
        key_diag, key_data, key_sigma,
        keybs, keym, nd, mtype,
        ddata, dsigma, matrix, units_gmat,
        keyt, t, reft, notime,
        m3d, indok, iokt,
        dconstraints,
        dopmat, operator, geometry,
        dalgo, dconstraints, dcon,
        conv_crit, maxiter_outer,
        crop, chain, kwdargs, method, options,
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
    key_cam=None,
):

    # load ddata from key_data
    ddata = coll.get_diagnostic_data_concatenated(
        key=key_diag,
        key_data=key_data,
        key_cam=key_cam,
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
    key_cam=None,
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
            key_cam=key_cam,
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
    nd=None,
    mtype=None,
):

    # if exists
    if dconst.get(rm) is None:
        return

    # check against mesh type
    if nd != '1d':
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
    nd=None,
    mtype=None,
    dconst=None,
    deriv=None,
    deg=None,
):

    # check mesh type
    if nd != '1d':
        msg = f"constraint '{deriv}' cannot be used with mesh non-1d"
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
                    warnings.warn(msg)

                dconst[deriv]['rad'] = dconst[deriv]['rad'][iok]
                dconst[deriv]['val'] = dconst[deriv]['val'][iok]

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
    nd=None,
    mtype=None,
    deg=None,
    dconst=None,
):

    # ----------------
    # check conformity

    if dconst is None:
        return
    elif not isinstance(dconst, dict):
        msg = f"Arg dconstraints must be a dict!\nProvided: {dconst}"
        raise Exception(msg)
    elif nd != '1d':
        msg = "Arg dconstraints cannot be used with non-1d meshes!"
        warnings.warn(msg)
        return

    # copy to avoid modifying reference
    dconst = copy.deepcopy(dconst)

    # ----------
    # rmin, rmax

    _check_rminmax(coll=coll, dconst=dconst, rm='rmin', mtype=mtype, nd=nd)
    _check_rminmax(coll=coll, dconst=dconst, rm='rmax', mtype=mtype, nd=nd)

    # -----------
    # derivatives

    for deriv in set(['deriv0', 'deriv1']).intersection(dconst.keys()):
        dconst = _check_deriv(
            coll=coll,
            keym=keym,
            nd=nd,
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

            # safety check
            if indin.size == 0:
                msg = (
                    "Eq from constraint seems to address a single bspline!"
                )
                raise NotImplementedError(msg)

            iout_coefs[it, indout] = True
            iin_coefs[it, indout] = False

            dcon[k0]['iout'][it] = indout
            dcon[k0]['iin'][it][:] = indin

    # -----------------------
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
            x0=dconst[deriv]['rad'],
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
    sigma_rel=None,
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

        # Exponent for rescaling of a0bis
        # typically in [1/3 ; 1/2], but real limits are 0 < d < 1 (or 2 ?)
        if kwdargs.get('d') is None:
            kwdargs['d'] = 0.4 # 0.95

        # determination of a0 is an important parameter
        # the result is sensitive to the order of magnitude of a0 (<1 or >1)
        # change a0 is there is strong over or under-smoothing

        # mu = lamb / tau

        # # def a0
        a0 = 0.1*sigma_rel**(-kwdargs['d'])
        # typically set b0 to reg
        b0 = 1

        # (a0, b0) are the gamma distribution parameters for lamb
        if kwdargs.get('a0') is None:
            kwdargs['a0'] = a0  # np.nanmean(dsigma['data']))  # 10 ?

        # to have [x]=1
        if kwdargs.get('b0') is None:
            kwdargs['b0'] = b0  # math.factorial(a0)**(1 / (a0 + 1))

        # (a1, b1) are the gamma distribution parameters for tau
        if kwdargs.get('a1') is None:
            kwdargs['a1'] = 1

        # to have [x]=1
        if kwdargs.get('b1') is None:
            kwdargs['b1'] =(
                math.factorial(kwdargs['a1'])**(1 / (kwdargs['a1'] + 1))
            )

        if kwdargs.get('conv_reg') is None:
            kwdargs['conv_reg'] = True

        if kwdargs.get('nbs_fixed') is None:
            kwdargs['nbs_fixed'] = False    # True

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
                method = 'TNC'     # more robust ?

        # solver-specific options
        elif method == 'TNC':
            if options.get('ftol') is None:
                options['ftol'] = conv_crit/100.
            if options.get('disp') is None:
                options['disp'] = False

        elif method == 'L-BFGS-B':
            if options.get('ftol') is None:
                options['ftol'] = conv_crit/100.
            if options.get('disp') is None:
                options['disp'] = False

        elif dalgo['name'] != 'algo7':
            raise NotImplementedError

    return kwdargs, method, options