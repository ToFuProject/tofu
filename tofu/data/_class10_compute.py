# -*- coding: utf-8 -*-


# Built-in
import time


# Common
import numpy as np
import scipy.sparse as scpsp


import datastock as ds


# tofu
from . import _class8_compute_signal
from . import _class10_checks as _checks
from . import _class10_algos as _algos
from . import _class10_refs as _refs
tomotok2tofu = _algos.tomotok2tofu


# ################################################################
# ################################################################
#                           main
# ################################################################


def compute_inversions(
    # resources
    coll=None,
    # inversion name
    key=None,
    # input data
    key_matrix=None,
    key_data=None,
    key_sigma=None,
    sigma=None,
    # constraints
    dconstraints=None,
    # regularity operator
    operator=None,
    geometry=None,
    dop_coefs=None,
    # choice of algo
    algo=None,
    maxiter_outer=None,
    # misc
    solver=None,
    conv_crit=None,
    chain=None,
    verb=None,
    store=None,
    # algo and solver-specific options
    kwdargs=None,
    method=None,
    options=None,
    # ref vector specifier
    dref_vector=None,
    ref_vector_strategy=None,
):

    # -------------
    # check inputs

    (
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
        keyinv, refinv, regul,
    ) = _checks._compute_check(**locals())

    data = ddata['data']
    sigma = dsigma['data']

    nt, nchan = data.shape
    nbs = matrix.shape[-1]

    # -------------
    # get func

    if isinstance(dalgo['func'], str):
        if dalgo['source'] == 'tofu':
            dalgo['func'] = getattr(_algos, dalgo['func'])
        elif dalgo['source'] == 'tomotok':
            dalgo['func'] = getattr(tomotok2tofu, dalgo['func'])

    # -------------
    # prepare data

    if verb >= 1:
        # t0 = time.process_time()
        t0 = time.perf_counter()
        print("Preparing data... ", end='', flush=True)

    # indt (later)

    # normalization
    data_n = (data / sigma)
    mu0 = 1.

    # Define Regularization operator
    R = _get_operator(
        dalgo=dalgo,
        operator=operator,
        dopmat=dopmat,
        dop_coefs=dop_coefs,
    )

    # normalize geometry matrix to avoid having 1e-15 * 1e16
    mmm = matrix
    if hasattr(matrix, 'toarray'):
        mmm =matrix.toarray()

    matnorm = np.mean(np.mean(mmm, axis=-1, where=mmm>0), axis=-1)
    matrix_norm = matrix / matnorm[:, None, None] if m3d else matrix / matnorm

    # prepare computation intermediates
    precond = None
    Tyn = np.full((nbs,), np.nan)
    mat0 = matrix_norm[0, ...] if m3d else matrix_norm
    if dalgo['sparse'] is True:
        Tn = scpsp.diags(1./np.nanmean(sigma, axis=0)).dot(mat0)
        TTn = Tn.T.dot(Tn)
        if solver != 'spsolve':
            # preconditioner to approx inv(TTn + reg*Rn)
            # should improves convergence, but actually slower...
            # precond = scpsp.linalg.inv(TTn + mu0*R)
            pass

    else:
        Tn = mat0 / np.nanmean(sigma, axis=0)[:, None]
        TTn = Tn.T.dot(Tn)

    # prepare output arrays
    sol = np.full((nt, nbs), np.nan)
    mu = np.full((nt,), np.nan)
    chi2n = np.full((nt,), np.nan)
    regularity = np.full((nt,), np.nan)
    niter = np.zeros((nt,), dtype=int)
    spec = [None for ii in range(nt)]

    # -------------
    # initial guess

    if indok is None:
        sol0 = np.full((nbs,), np.mean(data[0, :] / np.sum(mat0, axis=1)))
    else:
        sol0 = np.full(
            (nbs,),
            np.mean(
                data[0, indok[0, :]] / np.sum(mat0[indok[0, :], :], axis=1),
            ),
        )

    if verb >= 1:
        # t1 = time.process_time()
        t1 = time.perf_counter()
        print(f"{t1-t0} s", end='\n', flush=True)
        print(f"Geom matrix normalized to {np.mean(matnorm)} (m3d {m3d})")
        print("Setting inital guess... ", end='', flush=True)

    # -------------
    # compute

    if verb >= 1:
        # t2 = time.process_time()
        t2 = time.perf_counter()
        print(f"{t2-t1} s", end='\n', flush=True)
        print("Starting time loop...", end='\n', flush=True)

    if dalgo['source'] == 'tofu':
        out = _compute_inv_loop(
            sol0=sol0,
            mu0=mu0,
            matrix=matrix_norm,
            Tn=Tn,              # normalized geometry matrix (T) by sigma
            TTn=TTn,            # normalized tTT
            Tyn=Tyn,            # normalized
            R=R,                # Regularity operator
            precond=precond,
            data_n=data_n,      # normalized data
            sigma=sigma,
            indok=indok,
            # parameters
            m3d=m3d,
            dalgo=dalgo,
            isotropic=dalgo['isotropic'],
            conv_crit=conv_crit,
            sparse=dalgo['sparse'],
            positive=dalgo['positive'],
            chain=chain,
            verb=verb,
            kwdargs=kwdargs,
            method=method,
            options=options,
            dcon=dcon,
            regul=regul,
            maxiter_outer=maxiter_outer,
            # output
            sol=sol,
            mu=mu,
            chi2n=chi2n,
            regularity=regularity,
            niter=niter,
            spec=spec,
        )

    elif dalgo['source'] == 'tomotok':
        out = _compute_inv_loop_tomotok(
            sol0=sol0,
            mu0=mu0,
            matrix=matrix_norm,
            Tn=Tn,              # normalized geometry matrix (T)
            TTn=TTn,            # normalized tTT
            Tyn=Tyn,            # normalized
            R=R,                # Regularity operator
            precond=precond,
            data_n=data_n,      # normalized data
            sigma=sigma,
            indok=indok,
            # parameters
            m3d=m3d,
            dalgo=dalgo,
            isotropic=dalgo['isotropic'],
            conv_crit=conv_crit,
            sparse=dalgo['sparse'],
            positive=dalgo['positive'],
            chain=chain,
            verb=verb,
            kwdargs=kwdargs,
            method=method,
            options=options,
            dcon=dcon,
            regul=regul,
            # output
            sol=sol,
            mu=mu,
            chi2n=chi2n,
            regularity=regularity,
            niter=niter,
            spec=spec,
        )

    if verb >= 1:
        # t3 = time.process_time()
        t3 = time.perf_counter()
        print(f"{t3-t2} s", end='\n', flush=True)
        print("Post-formatting results...", end='\n', flush=True)

    # ---------------------------------------------
    # estimate relative regularity for polar mesh of not regul

    if (not regul) and nd == '1d':
        clas = coll.dobj['bsplines'][keybs]['class']

        # if clas.knotsa is None:
            # # estimate 1d squared gradient
            # kr = coll.dobj[coll._which_mesh][keym]['knots'][0]
            # rr = coll.ddata[kr]['data']
            # regularity = np.nansum(
                # clas(
                    # radius=np.linspace(rr[0], rr[-1], rr.size*10),
                    # coefs=sol,
                    # radius_vs_time=False,
                    # deriv=1,
                # )**2,
                # axis=1,
            # )

    # -------------
    # format output

    # de-normalize by matnorm
    if m3d:
        sol = sol / matnorm[:, None]
    else:
        sol = sol / matnorm

    # reshape solution
    if crop is True:
        shapebs = coll.dobj['bsplines'][keybs]['shape']
        shapesol = tuple(np.r_[nt, shapebs])
        sol_full = np.zeros(shapesol, dtype=float)
        cropbs = coll.ddata[coll.dobj['bsplines'][keybs]['crop']]['data']
        cropbsflat = cropbs.ravel(order='F')
        iR = np.tile(np.arange(0, shapebs[0]), shapebs[1])[cropbsflat]
        iZ = np.repeat(np.arange(0, shapebs[1]), shapebs[0])[cropbsflat]
        sol_full[:, iR, iZ] = sol
    else:
        sol_full = sol

    # -------------
    # store

    if store is True:
        units = ddata['units'] / units_gmat
        key_data = ddata['keys']

        _store(**locals())

    else:
        return sol_full, mu, chi2n, regularity, niter, spec, t


# ##################################################################
# ##################################################################
#                   get operator
# ##################################################################


def _get_operator(
    dalgo=None,
    operator=None,
    dop_coefs=None,
    dopmat=None,
):
    """ Return time-independent operator """

    # -------
    # Trivial

    if dalgo['family'] == 'Non-regularized':
        return None

    # -------
    # check

    if dop_coefs is None:
        dop_coefs = {k0: 1. for k0 in dopmat.keys()}
    c0 = (
        isinstance(dop_coefs, dict)
        and all([
            isinstance(dop_coefs.get(k0), (float, int))
            for k0 in dopmat.keys()
        ])
    )
    if not c0:
        msg = (
            "Arg dop_coefs must be a dict of scalar coefficients\n"
            "The key are the operator components they are applied to\n"
            "\t- Expected keys: {sorted(dopmat.keys())}\n"
            "\t- Provided: {dop_coefs}\n"
        )
        raise Exception(msg)

    # -----------
    # non-trivial

    # if dalgo['source'] == 'tomotok' and dalgo['reg_operator'] == 'MinFisher':
    #     R = opmat

    if operator == 'D0N2':
        R = dopmat['tMM']['data']

    elif operator == 'D1N2':
        R = 0
        for k0 in dopmat:
            R += dop_coefs[k0] * dopmat[k0]['data']

    elif operator == 'D2N2':
        R = 0
        for k0 in dopmat:
            R += dop_coefs[k0] * dopmat[k0]['data']

    else:
        msg = 'unknown operator!'
        raise Exception(msg)

    return R


# ##################################################################
# ##################################################################
#                   store
# ##################################################################


def _store(
    coll=None,
    sol_full=None,
    notime=None,
    iokt=None,
    chi2n=None,
    mu=None,
    regularity=None,
    niter=None,
    t=None,
    keyinv=None,
    refinv=None,
    reft=None,
    keyt=None,
    key_diag=None,
    key_data=None,
    key_sigma=None,
    key_matrix=None,
    operator=None,
    geometry=None,
    dalgo=None,
    solver=None,
    chain=None,
    conv_crit=None,
    units=None,
    # ref vector specifier
    dref_vector=None,
    ref_vector_strategy=None,
    **kwdargs,
):

    # ------------
    # check input

    if dref_vector is None:
        dref_vector = {}

    # ---------------------------
    # reshape if unique time step

    if notime:
        if not sol_full.shape[0] == 1:
            msg = (
                "Inconsistency:\n"
                "notime = True but sol_full.shape[0] > 1\n"
                f"\t- key_data: '{key_data}'\n"
                f"\t- key_matrix: '{key_matrix}'\n"
                f"\t- sol_full.shape: {sol_full.shape}\n'"
            )
            raise Exception(msg)
        sol_full = sol_full[0, ...]
    else:
        # restore full size
        sol_full, chi2n, mu, regularity, niter = _restore_fullt(
            iokt=iokt,
            sol_full=sol_full,
            chi2n=chi2n,
            mu=mu,
            regularity=regularity,
            niter=niter,
        )
        nt = t.size

    # --------------------
    # Build dict

    # ddata
    ddata = {
        keyinv: {
            'data': sol_full,
            'ref': refinv,
            'units': units,
        },
    }

    dref = None
    if notime is False:
        if keyt is None:
            dref = {
                reft: {'size': nt},
            }
            ddata.update({
                f'{keyinv}_t': {
                    'data': t,
                    'ref': reft,
                    'dim': 'time',
                },
            })

        ddata.update({
            f'{keyinv}_chi2n': {
                'data': chi2n,
                'ref': reft,
            },
            f'{keyinv}_mu': {
                'data': mu,
                'ref': reft,
            },
            f'{keyinv}_reg': {
                'data': regularity,
                'ref': reft,
            },
            f'{keyinv}_niter': {
                'data': niter,
                'ref': reft,
            },
        })

    # add synthetic data
    kretro = f'{keyinv}_retro'

    # add inversion
    dobj = {
        'inversions': {
            keyinv: {
                'retrofit': kretro,
                'data_in': key_data,
                'sigma_in': key_sigma,
                'matrix': key_matrix,
                'sol': keyinv,
                'operator': operator,
                'geometry': geometry,
                'isotropic': dalgo['isotropic'],
                'algo': dalgo['name'],
                'solver': solver,
                'chain': chain,
                'positive': dalgo['positive'],
                'conv_crit': conv_crit,
            },
        },
    }

    # adjust for time
    if notime is True:
        dobj['inversions'][keyinv].update({
            'chi2n': chi2n,
            'mu': mu,
            'reg': regularity,
            'niter': niter,
        })

    # update instance
    coll.update(dobj=dobj, dref=dref, ddata=ddata)

    # add synthetic data
    keyt = coll.get_ref_vector(key=keyinv, ref=reft, **dref_vector)[3]
    data_synth = coll.add_retrofit_data(
        key=kretro,
        key_diag=key_diag,
        key_matrix=key_matrix,
        key_profile2d=keyinv,
        t=keyt,
        # ref vector specifier
        dref_vector=dref_vector,
        ref_vector_strategy=ref_vector_strategy,
        store=True,
    )


def _restore_fullt(
    iokt=None,
    sol_full=None,
    chi2n=None,
    mu=None,
    regularity=None,
    niter=None,
):

    # sol_full
    shape = tuple(np.r_[iokt.size, sol_full.shape[1:]])
    sol_fulli = np.full(shape, np.nan)
    sol_fulli[iokt, :] = sol_full

    # 1d
    chi2ni = np.full((iokt.size,), np.nan)
    mui = np.full((iokt.size,), np.nan)
    regularityi = np.full((iokt.size,), np.nan)
    niteri = np.full((iokt.size,), np.nan)

    chi2ni[iokt] = chi2n
    mui[iokt] = mu
    regularityi[iokt] = regularity
    niteri[iokt] = niter

    return sol_fulli, chi2ni, mui, regularityi, niteri


# ################################################################
# ################################################################
#                   _compute time loop
# ################################################################


def _compute_inv_loop(
    # inputs
    dalgo=None,
    sol0=None,
    mu0=None,
    matrix=None,
    Tn=None,        # normalized geometry matrix (T)
    TTn=None,       # normalized tTT
    Tyn=None,       # normalized tTy
    R=None,
    precond=None,
    data_n=None,
    sigma=None,
    indok=None,
    # parameters
    m3d=None,
    conv_crit=None,
    isotropic=None,
    sparse=None,
    positive=None,
    chain=None,
    verb=None,
    kwdargs=None,
    method=None,
    options=None,
    dcon=None,
    regul=None,
    maxiter_outer=None,
    # output
    sol=None,
    mu=None,
    chi2n=None,
    regularity=None,
    niter=None,
    spec=None,
):

    # -----------------------------------
    # Getting initial solution - step 1/2

    nt, nchan = data_n.shape
    nbs = Tn.shape[1]

    if verb >= 2:
        form = "nchan * chi2n   +   mu *  R           "
        verb2head = f"\n\t\titer    {form} \t\t\t  tau  \t      conv"
    else:
        verb2head = None

    # -----------------------------------
    # Options for quadratic solvers only

    bounds = None
    func_val, func_jac, func_hess = None, None, None
    if regul and positive is True:

        bounds = tuple([(0., None) for ii in range(0, sol0.size)])

        def func_val(x, mu=mu0, Tn=Tn, yn=data_n[0, :], TTn=None, Tyn=None, R=R):
            return np.sum((Tn.dot(x) - yn)**2) + mu*x.dot(R.dot(x))

        def func_jac(x, mu=mu0, Tn=None, yn=None, TTn=TTn, Tyn=Tyn, R=R):
            return 2.*(TTn + mu*R).dot(x) - 2.*Tyn

        def func_hess(x, mu=mu0, Tn=None, yn=None, TTn=TTn, Tyn=Tyn, R=R):
            return 2.*(TTn + mu*R)

    elif not regul:

        if positive is True:
            bounds = (
                np.zeros((nbs,), dtype=float),
                np.full((nbs,), np.inf),
            )
        else:
            bounds = (-np.inf, np.inf)

    # ---------
    # time loop

    # Beware of element-wise operations vs matrix operations !!!!
    nbsi = nbs
    bi = bounds
    Ri = R
    indbsi = np.ones((nbsi,), dtype=bool)
    for ii in range(0, nt):

        if verb >= 1:
            msg = f"\ttime step {ii+1} / {nt} "
            print(msg, end='', flush=True)

        # update terms
        Tni, TTni, Tyni, yni, nchani = _update_TTyn(
            sparse=sparse,
            data_n=data_n,
            sigma=sigma,
            matrix=matrix,
            Tn=Tn,
            TTn=TTn,
            Tyn=Tyn,
            indok=indok,
            ii=ii,
            m3d=m3d,
            regul=regul,
        )

        if dcon is not None:
            ic = 0 if ii == 0 or not dcon['hastime'] else ii
            nbsi, indbsi, Tni, TTni, Tyni, Ri, yni, bi = _update_ttyn_constraints(
                sparse=sparse,
                Tni=Tni,
                TTni=TTni,
                Tyni=Tyni,
                yni=yni,
                Ri=R,
                mu=mu0,
                bounds=bounds,
                ii=ic,
                dcon=dcon,
                regul=regul,
            )

        # solving
        (
            sol[ii, indbsi], mu[ii], chi2n[ii], regularity[ii],
            niter[ii], spec[ii],
        ) = dalgo['func'](
            Tn=Tni,
            TTn=TTni,
            Tyn=Tyni,
            R=Ri,
            yn=yni,
            # initial guess
            sol0=sol0[indbsi],
            mu0=mu0,
            # problem size
            nchan=nchani,
            nbs=nbsi,
            # parameters
            conv_crit=conv_crit,
            precond=precond,
            verb=verb,
            verb2head=verb2head,
            maxiter_outer=maxiter_outer,
            # quad-only
            func_val=func_val,
            func_jac=func_jac,
            func_hess=func_hess,
            bounds=bi,
            method=method,
            options=options,
            **kwdargs,
        )

        if dcon is not None:
            if dcon['coefs'] is not None:
                sol[ii, :] = (
                    dcon['coefs'][ic].dot(sol[ii, indbsi])
                    + dcon['offset'][ic, :]
                )
            else:
                sol[ii, :] = sol[ii, :] + dcon['offset'][ic, :]

        # post
        if chain:
            sol0[:] = sol[ii, :]
        mu0 = mu[ii]

        if verb == 1:
            msg = f"   chi2n = {chi2n[ii]:.3e}    reg = {regularity[ii]:.3e}    niter = {niter[ii]}"
            print(msg, end='\n', flush=True)


# ##################################################################
# ##################################################################
#                   _compute time loop - TOMOTOK
# ##################################################################


def _compute_inv_loop_tomotok(
    # inputs
    dalgo=None,
    sol0=None,
    mu0=None,
    matrix=None,
    Tn=None,        # normalized geometry matrix (T)
    TTn=None,       # normalized tTT
    Tyn=None,       # normalized tTy
    R=None,
    precond=None,
    data_n=None,
    sigma=None,
    indok=None,
    # parameters
    m3d=None,
    conv_crit=None,
    isotropic=None,
    sparse=None,
    positive=None,
    chain=None,
    verb=None,
    kwdargs=None,
    method=None,
    options=None,
    dcon=None,
    regul=None,
    # output
    sol=None,
    mu=None,
    chi2n=None,
    regularity=None,
    niter=None,
    spec=None,
):

    # -----------------------------------
    # Getting initial solution - step 1/2

    nt, nchan = data_n.shape
    if not isinstance(R, np.ndarray):
        nbs = R[0].shape[0]
        R = (R,)
    else:
        nbs = R.shape[0]

    if dcon is not None:
        msg = "constraints not handled by tomotok algorithms"
        raise Exception(msg)

    if verb >= 2:
        form = "nchan * chi2n   +   mu *  R           "
        verb2head = f"\n\t\titer    {form} \t\t\t  tau  \t      conv"
    else:
        verb2head = None

    # ---------
    # time loop

    # Beware of element-wise operations vs matrix operations !!!!
    for ii in range(0, nt):

        if verb >= 1:
            msg = f"\ttime step {ii+1} / {nt} "
            print(msg, end='\n', flush=True)

        # update intermediates if multiple sigmas
        if sigma.shape[0] > 1:
            _update_TTyn(
                sparse=sparse,
                sigma=sigma,
                mati=mi,
                Tn=Tn,
                TTn=TTn,
                ii=ii,
            )

        Tyn[:] = Tn.T.dot(data_n[ii, :])

        # update terms
        Tni, TTni, Tyni, yni, nchani = _update_TTyn(
            sparse=sparse,
            data_n=data_n,
            sigma=sigma,
            matrix=matrix,
            Tn=Tn,
            TTn=TTn,
            Tyn=Tyn,
            indok=indok,
            ii=ii,
            m3d=m3d,
            regul=regul,
        )

        # solving
        (
            sol[ii, :], chi2n[ii], regularity[ii],
        ) = dalgo['func'](
            sig_norm=yni,
            gmat_norm=Tni,
            deriv=R,
            method=None,
            num=None,
            # additional
            nchan=nchan,
            **kwdargs,
        )

        # post
        if chain:
            sol0[:] = sol[ii, :]
        # mu0 = mu[ii]

        if verb == 1:
            msg = f"   chi2n = {chi2n[ii]:.3e}    niter = {niter[ii]}"
            print(msg, end='\n', flush=True)


# ##################################################################
# ##################################################################
#                      utility
# ##################################################################


def _update_TTyn(
    sparse=None,
    data_n=None,
    sigma=None,
    matrix=None,
    Tn=None,
    TTn=None,
    Tyn=None,
    indok=None,
    ii=None,
    m3d=None,
    regul=None,
):
    # update matrix
    if indok is None:
        if m3d:
            mi = matrix[ii, :, :]
        else:
            mi = matrix
    else:
        if m3d:
            mi = matrix[ii, indok[ii, :], :]
        else:
            mi = matrix[indok[ii, :], :]

    # update sig
    if sigma.shape[0] == 1:
        if indok is None:
            sig = sigma[0, :]
        else:
            sig = sigma[0, indok[ii, :]]
    else:
        if indok is None:
            sig = sigma[ii, :]
        else:
            sig = sigma[ii, indok[ii, :]]

    # update data_n
    if indok is None:
        yn = data_n[ii, :]
    else:
        yn = data_n[ii, indok[ii, :]]

    # intermediates
    if indok is None:
        if sparse:
            Tn.data = scpsp.diags(1./sig).dot(mi).data
            TTn.data = Tn.T.dot(Tn).data
        else:
            Tn[...] = mi / sig[:, None]
            TTn[...] = Tn.T.dot(Tn)

    else:
        if sparse:
            Tn = scpsp.diags(1./sig).dot(mi)
            TTn = Tn.T.dot(Tn)
        else:
            Tn = mi / sig[:, None]
            TTn = Tn.T.dot(Tn)

    # Tyn (for reguarized algorithms)
    if regul:
        if indok is None:
            Tyn[:] = Tn.T.dot(yn)
        else:
            Tyn = Tn.T.dot(yn)

    return Tn, TTn, Tyn, yn, yn.size


def _update_ttyn_constraints(
    sparse=None,
    Tni=None,
    TTni=None,
    Tyni=None,
    yni=None,
    Ri=None,
    mu=None,
    bounds=None,
    ii=None,
    dcon=None,
    regul=None,
):

    # pre-assign
    A = dcon['coefs'][ii]
    At = A.T
    O = dcon['offset'][ii, :]

    # yni
    yni = yni - Tni.dot(O)

    # Tni
    Tni = Tni.dot(A)

    # regul => Tyni, TTni
    if regul:

        # Tyni
        Tyni = At.dot( Tyni -TTni.dot(O) - mu * Ri.dot(O) )

        # TTni
        TTni = At.dot( TTni.dot(A) )

        # Ri
        Ri = At.dot( Ri.dot(A) )

    # indbsi
    indbsi = dcon['indbs'][ii]

    # nbsi
    nbsi = Tni.shape[1]

    # bounds
    if np.isscalar(bounds[0]):
        pass
    elif isinstance(bounds[0], np.ndarray):
        bounds = (bounds[0][indbsi], bounds[1][indbsi])
    else:
        bounds = tuple([bounds[ii] for ii in indbsi])

    return nbsi, indbsi, Tni, TTni, Tyni, Ri, yni, bounds


# ##################################################################
# ##################################################################
#               retrofit
# ##################################################################


def compute_retrofit_data(
    # resources
    coll=None,
    # inputs
    key=None,
    key_diag=None,
    key_matrix=None,
    key_profile2d=None,
    t=None,
    # ref_vector_specifier
    dref_vector=None,
    ref_vector_strategy=None,
    # parameters
    store=None,
    returnas=None,
):

    # ------------
    # check inputs
    # --------------

    (
        key, key_diag, key_cam, keybs, keym, mtype,
        key_matrix, key_profile2d,
        is2d, matrix, dindmat,
        hastime, t, keyt, reft, ref,
        nt, nchan, nbs,
        ist_mat, ist_prof, dind,
        store, returnas,
    ) = _compute_retrofit_data_check(
        # resources
        coll=coll,
        # inputs
        key=key,
        key_diag=key_diag,
        key_matrix=key_matrix,
        key_profile2d=key_profile2d,
        t=t,
        # ref_vector_specifier
        dref_vector=dref_vector,
        ref_vector_strategy=ref_vector_strategy,
        # parameters
        store=store,
        returnas=returnas,
    )

    # --------
    # prepare
    # --------------

    kmat = coll.dobj['geom matrix'][key_matrix]['data']
    gunits = coll.ddata[kmat[0]]['units']

    coefs = coll.ddata[key_profile2d]['data']

    # coefs
    if mtype == 'rect':
        indbs_tf = coll.select_bsplines(
            key=keybs,
            returnas='ind',
        )
        if hastime and ist_prof:
            coefs = coefs[:, indbs_tf[0], indbs_tf[1]]
        else:
            coefs = coefs[indbs_tf[0], indbs_tf[1]]

    # --------
    # compute
    # ---------

    # time-dependent
    if hastime:

        # retro = np.full((nt, nchan, nbs), np.nan)

        # get time indices
        if ist_mat:
            if dind.get(kmat[0], {}).get('ind') is not None:
                imat = dind[kmat[0]]['ind']
            else:
                imat = np.arange(nt)

        if ist_prof:
            if dind.get(key_profile2d, {}).get('ind') is not None:
                iprof = dind[key_profile2d]['ind']
            else:
                iprof = np.arange(nt)

        # compute matrix product
        if ist_mat and ist_prof:
            retro = np.array([
                matrix[imat[ii], :, :].dot(coefs[iprof[ii], :])
                for ii in range(nt)
            ])
        elif ist_mat:
            retro = np.array([
                matrix[imat[ii], :, :].dot(coefs)
                for ii in range(nt)
            ])
        elif ist_prof:
            retro = np.array([
                matrix.dot(coefs[iprof[ii], :])
                for ii in range(nt)
            ])
    else:
        retro = matrix.dot(coefs)

    # --------------
    # format
    # --------------

    i0 = 0
    dout = {}
    for ii, k0 in enumerate(key_cam):
        npix = coll.dobj['camera'][k0]['dgeom']['pix_nb']
        ind = i0 + np.arange(0, npix)

        # extract relevant part
        retroi = retro[:, ind] if hastime else retro[ind]
        refi = list(ref)
        axis = refi.index(None)

        # reshape if 2d
        if is2d:
            sh = list(retroi.shape)
            sh[axis] = coll.dobj['camera'][k0]['dgeom']['shape']
            sh = tuple(np.r_[sh[:axis], sh[axis], sh[axis+1:]].astype(int))
            retroi = retroi.reshape(sh)

        # ref
        refi[axis] = coll.dobj['camera'][k0]['dgeom']['ref']
        refi = tuple(np.r_[refi[:axis], refi[axis], refi[axis+1:]])

        # dict
        dout[k0] = {
            'key': f'{key}_{k0}',
            'data': retroi,
            'ref': refi,
            'units': coll.ddata[key_profile2d]['units'] * gunits,
        }
        i0 += npix

    # --------------
    # store
    # --------------

    if store:
        _class8_compute_signal._store(
            coll=coll,
            key=key,
            key_diag=key_diag,
            dout=dout,
            key_matrix=key_matrix,
        )

    # -------------
    # return
    # --------------

    if returnas is dict:
        return dout


# ###################
#   checking
# ###################


def _compute_retrofit_data_check(
    # resources
    coll=None,
    # inputs
    key=None,
    key_diag=None,
    key_matrix=None,
    key_profile2d=None,
    t=None,
    # ref vector specifier
    dref_vector=None,
    ref_vector_strategy=None,
    # parameters
    store=None,
    returnas=None,
):

    # ----------
    # keys

    # key_diag
    lok = list(coll.dobj.get('diagnostic', {}).keys())
    key_diag = ds._generic_check._check_var(
        key_diag, 'key_diag',
        types=str,
        allowed=lok,
    )
    is2d = coll.dobj['diagnostic'][key_diag]['is2d']

    # key
    key = ds._generic_check._obj_key(
        coll.dobj.get('synth sig', {}),
        short='synth',
        key=key,
        ndigits=2,
    )

    # key_matrix
    lok = coll.dobj.get('geom matrix', {}).keys()
    key_matrix = ds._generic_check._check_var(
        key_matrix, 'key_matrix',
        types=str,
        allowed=lok,
    )

    key_cam = coll.dobj['geom matrix'][key_matrix]['camera']
    keybs = coll.dobj['geom matrix'][key_matrix]['bsplines']
    keym = coll.dobj['bsplines'][keybs]['mesh']
    mtype = coll.dobj[coll._which_mesh][keym]['type']

    matrix, ref, dindmat = coll.get_geometry_matrix_concatenated(key=key_matrix)
    nchan, nbs = matrix.shape[-2:]
    # refbs = ref[-1]

    # key_pofile2d
    lok = [
        k0 for k0, v0 in coll.ddata.items()
        if v0['bsplines'] is not None
        and keybs in v0['bsplines']
    ]
    key_profile2d = ds._generic_check._check_var(
        key_profile2d, 'key_profile2d',
        types=str,
        allowed=lok,
    )

    # time management
    lkmat = coll.dobj['geom matrix'][key_matrix]['data']

    hastime, reft, keyt, t_out, dind = _refs._get_ref_vector_common(
        coll=coll,
        key_matrix=key_matrix,
        key_profile2d=key_profile2d,
        strategy=ref_vector_strategy,
        dref_vector=dref_vector,
    )

    if hastime and reft is not None and dref_vector.get('ref') is None:
        dref_vector['ref'] = reft

    if hastime and t_out is not None and reft is None:
        reft = f'{key}_nt'
        keyt = f'{key}_t'

    ist_mat = coll.get_ref_vector(
        key=lkmat[0],
        **{
            k0: v0 for k0,v0 in dref_vector.items()
            if k0 != 'ref'
            or (k0 == 'ref' and v0 in coll.ddata[lkmat[0]]['ref'])
        },
        # **dref_vector,
    )[0]
    ist_prof = coll.get_ref_vector(
        key=key_profile2d,
        **{
            k0: v0 for k0,v0 in dref_vector.items()
            if k0 != 'ref'
            or (k0 == 'ref' and v0 in coll.ddata[lkmat[0]]['ref'])
        },
        # **dref_vector,
    )[0]

    # reft, keyt and refs
    if hastime and t_out is not None:
        nt = t_out.size
        ref = (reft, None)
    else:
        nt = 0
        reft = None
        keyt = None
        ref = (None,)

    # safety check
    if hastime and (not ist_mat) and (not ist_prof):
        msg = (
            "Common (time) vector between matrix and profile2d "
            "not identified:\n"
            f"\t- matrix: '{lkmat[0]}'\n"
            f"\t- profile2d: '{key_profile2d}'\n"
            "Specify common ref vector to use with dref_vector={}\n\n"
            + str(coll.show('data', verb=False, returnas=str))
        )
        raise Exception(msg)

    # store
    store = ds._generic_check._check_var(
        store, 'store',
        types=bool,
        default=True,
    )

    # returnas
    returnas = ds._generic_check._check_var(
        returnas, 'returnas',
        default=False if store else dict,
        allowed=[False, dict],
    )

    return (
        key, key_diag, key_cam, keybs, keym, mtype,
        key_matrix, key_profile2d,
        is2d, matrix, dindmat,
        hastime, t_out, keyt, reft, ref,
        nt, nchan, nbs,
        ist_mat, ist_prof, dind,
        store, returnas,
    )
