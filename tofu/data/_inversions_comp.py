# -*- coding: utf-8 -*-


# Built-in
import time


# Common
import numpy as np
import scipy.sparse as scpsp


# tofu
from . import _generic_check
from . import _inversions_checks
from . import _inversions_algos
tomotok2tofu = _inversions_checks.tomotok2tofu


__all__ = ['get_available_inversions_algo']


# #############################################################################
# #############################################################################
#                           main
# #############################################################################


def compute_inversions(
    # input data
    coll=None,
    key_matrix=None,
    key_data=None,
    key_sigma=None,
    data=None,
    sigma=None,
    # regularity operator
    operator=None,
    geometry=None,
    # choice of algo
    algo=None,
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
):

    # -------------
    # check inputs

    (
        key_matrix, key_data, key_sigma, keybs, keym,
        data, sigma, matrix, opmat, operator, geometry,
        dalgo,
        conv_crit, crop, chain, kwdargs, method, options,
        solver, verb, store,
    ) = _inversions_checks._compute_check(
        # input data
        coll=coll,
        key_matrix=key_matrix,
        key_data=key_data,
        key_sigma=key_sigma,
        data=data,
        sigma=sigma,
        # choice of algo
        algo=algo,
        # regularity operator
        solver=solver,
        operator=operator,
        geometry=geometry,
        # misc
        conv_crit=conv_crit,
        chain=chain,
        verb=verb,
        store=store,
        # algo and solver-specific options
        kwdargs=kwdargs,
        method=method,
        options=options,
    )

    nt, nchan = data.shape
    nbs = matrix.shape[1]

    # -------------
    # get func

    if isinstance(dalgo['func'], str):
        if dalgo['source'] == 'tofu':
            dalgo['func'] = getattr(_inversions_algos, dalgo['func'])
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
    if dalgo['source'] == 'tomotok' and dalgo['reg_operator'] == 'MinFisher':
        R = opmat
    elif operator == 'D0N2':
        R = opmat[0]
    elif operator == 'D1N2':
        R = opmat[0] + opmat[1]
    elif operator == 'D2N2':
        R = opmat[0] + opmat[1]
    else:
        msg = 'unknown operator!'
        raise Exception(msg)

    # prepare computation intermediates
    precond = None
    Tyn = np.full((nbs,), np.nan)
    if dalgo['sparse'] is True:
        Tn = scpsp.diags(1./np.nanmean(sigma, axis=0)).dot(matrix)
        TTn = Tn.T.dot(Tn)
        if solver != 'spsolve':
            # preconditioner to approx inv(TTn + reg*Rn)
            # should improves convergence, but actually slower...
            # precond = scpsp.linalg.inv(TTn + mu0*R)
            pass

    else:
        Tn = matrix / np.nanmean(sigma, axis=0)[:, None]
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

    sol0 = np.full((nbs,), np.nanmean(data[0, :]) / matrix.mean())

    if verb >= 1:
        # t1 = time.process_time()
        t1 = time.perf_counter()
        print(f"{t1-t0} s", end='\n', flush=True)
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
            dalgo=dalgo,
            # func=func,
            sol0=sol0,
            mu0=mu0,
            matrix=matrix,
            Tn=Tn,
            TTn=TTn,
            Tyn=Tyn,
            R=R,
            precond=precond,
            data_n=data_n,
            sigma=sigma,
            isotropic=dalgo['isotropic'],
            conv_crit=conv_crit,
            sparse=dalgo['sparse'],
            positive=dalgo['positive'],
            chain=chain,
            verb=verb,
            sol=sol,
            mu=mu,
            chi2n=chi2n,
            regularity=regularity,
            niter=niter,
            spec=spec,
            kwdargs=kwdargs,
            method=method,
            options=options,
        )

    elif dalgo['source'] == 'tomotok':
        out = _compute_inv_loop_tomotok(
            dalgo=dalgo,
            # func=func,
            sol0=sol0,
            mu0=mu0,
            matrix=matrix,
            Tn=Tn,
            TTn=TTn,
            Tyn=Tyn,
            R=R,
            precond=precond,
            data_n=data_n,
            sigma=sigma,
            isotropic=dalgo['isotropic'],
            conv_crit=conv_crit,
            sparse=dalgo['sparse'],
            positive=dalgo['positive'],
            chain=chain,
            verb=verb,
            sol=sol,
            mu=mu,
            chi2n=chi2n,
            regularity=regularity,
            niter=niter,
            spec=spec,
            kwdargs=kwdargs,
            method=method,
            options=options,
        )

    if verb >= 1:
        # t3 = time.process_time()
        t3 = time.perf_counter()
        print(f"{t3-t2} s", end='\n', flush=True)
        print("Post-formatting results...", end='\n', flush=True)

    # -------------
    # format output

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

    # store
    if store is True:

        # key
        if coll.dobj.get('inversions') is None:
            ninv = 0
        else:
            ninv = np.max([int(kk[3:]) for kk in coll.dobj['inversions']]) + 1
        keyinv = f'inv{ninv}'

        # ref
        refmat = coll.ddata[key_matrix]['ref']
        refdata = coll.ddata[key_data]['ref']

        if refdata == (refmat[0],):
            refinv = coll.dobj['bsplines'][keybs]['ref']
            assert sol_full.shape[0] == 1
            sol_full = sol_full[0, ...]
            notime = True

        elif refdata[0] not in refmat:
            refinv = tuple(
                [refdata[0]] + list(coll.dobj['bsplines'][keybs]['ref'])
            )
            notime = False

        else:
            msg = (
                "Unreckognized shape of sol_full vs refinv!\n"
                f"\t- sol_full.shape: {sol_full.shape}\n"
                f"\t- inv['ref']:    {refinv}\n"
                f"\t- matrix['ref']: {refmat}\n"
                f"\t- data['ref']:   {refdata}\n"
            )
            raise Exception(msg)

        # dict
        ddata = {
            keyinv: {
                'data': sol_full,
                'ref': refinv,
            },
        }
        if notime is False:
            ddata.update({
                f'{keyinv}-chi2n': {
                    'data': chi2n,
                    'ref': refinv[0],
                },
                f'{keyinv}-mu': {
                    'data': mu,
                    'ref': refinv[0],
                },
                f'{keyinv}-reg': {
                    'data': regularity,
                    'ref': refinv[0],
                },
                f'{keyinv}-niter': {
                    'data': niter,
                    'ref': refinv[0],
                },
            })

        if key_sigma is None:
            key_sigma = f'{key_data}-sigma'
            if notime:
                ref_sigma = coll.ddata[key_data]['ref']
                sigma = sigma[0, :]
            else:
                if sigma.shape == data.shape:
                    ref_sigma = coll.ddata[key_data]['ref']
                else:
                    ref_sigma = coll.ddata[key_data]['ref'][1:]
                    sigma = sigma[0, :]

            ddata.update({
                key_sigma: {
                    'data': sigma,
                    'ref': ref_sigma,
                    'units': coll.ddata[key_data].get('units'),
                    'dim': coll.ddata[key_data].get('dim'),
                },
            })

        dobj = {
            'inversions': {
                keyinv: {
                    'data_in': key_data,
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
        if notime is True:
            dobj['inversions'][keyinv].update({
                'chi2n': chi2n,
                'mu': mu,
                'reg': regularity,
                'niter': niter,
            })

        coll.update(dobj=dobj, ddata=ddata)

    else:
        return sol_full, mu, chi2n, regularity, niter, spec


# #############################################################################
# #############################################################################
#                   _compute time loop
# #############################################################################


def _compute_inv_loop(
    dalgo=None,
    # func=None,
    sol0=None,
    mu0=None,
    matrix=None,
    Tn=None,
    TTn=None,
    Tyn=None,
    R=None,
    precond=None,
    data_n=None,
    sigma=None,
    conv_crit=None,
    isotropic=None,
    sparse=None,
    positive=None,
    chain=None,
    verb=None,
    sol=None,
    mu=None,
    chi2n=None,
    regularity=None,
    niter=None,
    spec=None,
    kwdargs=None,
    method=None,
    options=None,
):

    # -----------------------------------
    # Getting initial solution - step 1/2

    nt, nchan = data_n.shape
    nbs = R.shape[0]

    if verb >= 2:
        form = "nchan * chi2n   +   mu *  R           "
        verb2head = f"\n\t\titer    {form} \t\t\t  tau  \t      conv"
    else:
        verb2head = None

    # -----------------------------------
    # Options for quadratic solvers only

    if positive is True:

        bounds = tuple([(0., None) for ii in range(0, sol0.size)])

        def func_val(x, mu=mu0, Tn=Tn, yn=data_n[0, :], TTn=None, Tyn=None):
            return np.sum((Tn.dot(x) - yn)**2) + mu*x.dot(R.dot(x))

        def func_jac(x, mu=mu0, Tn=None, yn=None, TTn=TTn, Tyn=Tyn):
            return 2.*(TTn + mu*R).dot(x) - 2.*Tyn

        def func_hess(x, mu=mu0, Tn=None, yn=None, TTn=TTn, Tyn=Tyn):
            return 2.*(TTn + mu*R)

    else:
        bounds = None
        func_val, func_jac, func_hess = None, None, None

    # ---------
    # time loop

    # Beware of element-wise operations vs matrix operations !!!!
    for ii in range(0, nt):

        if verb >= 1:
            msg = f"\ttime step {ii+1} / {nt} "
            print(msg, end='', flush=True)

        # update intermediates if multiple sigmas
        if sigma.shape[0] > 1:
            _update_TTyn(
                sparse=sparse,
                sigma=sigma,
                matrix=matrix,
                Tn=Tn,
                TTn=TTn,
                ii=ii,
            )

        Tyn[:] = Tn.T.dot(data_n[ii, :])

        # solving
        (
            sol[ii, :], mu[ii], chi2n[ii], regularity[ii],
            niter[ii], spec[ii],
        ) = dalgo['func'](
            Tn=Tn,
            TTn=TTn,
            Tyn=Tyn,
            R=R,
            yn=data_n[ii, :],
            sol0=sol0,
            nchan=nchan,
            nbs=nbs,
            mu0=mu0,
            conv_crit=conv_crit,
            precond=precond,
            verb=verb,
            verb2head=verb2head,
            # quad-only
            func_val=func_val,
            func_jac=func_jac,
            func_hess=func_hess,
            bounds=bounds,
            method=method,
            options=options,
            **kwdargs,
        )

        # post
        if chain:
            sol0[:] = sol[ii, :]
        mu0 = mu[ii]

        if verb == 1:
            msg = f"   chi2n = {chi2n[ii]:.3e}    niter = {niter[ii]}"
            print(msg, end='\n', flush=True)


# #############################################################################
# #############################################################################
#                   _compute time loop - TOMOTOK
# #############################################################################


def _compute_inv_loop_tomotok(
    dalgo=None,
    # func=None,
    sol0=None,
    mu0=None,
    matrix=None,
    Tn=None,
    TTn=None,
    Tyn=None,
    R=None,
    precond=None,
    data_n=None,
    sigma=None,
    conv_crit=None,
    isotropic=None,
    sparse=None,
    positive=None,
    chain=None,
    verb=None,
    sol=None,
    mu=None,
    chi2n=None,
    regularity=None,
    niter=None,
    spec=None,
    kwdargs=None,
    method=None,
    options=None,
):

    # -----------------------------------
    # Getting initial solution - step 1/2

    nt, nchan = data_n.shape
    if not isinstance(R, np.ndarray):
        nbs = R[0].shape[0]
        R = (R,)
    else:
        nbs = R.shape[0]

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
                matrix=matrix,
                Tn=Tn,
                TTn=TTn,
                ii=ii,
            )

        Tyn[:] = Tn.T.dot(data_n[ii, :])

        # solving
        (
            sol[ii, :], chi2n[ii], regularity[ii],
        ) = dalgo['func'](
            sig_norm=data_n[ii, :],
            gmat_norm=Tn,
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
        mu0 = mu[ii]

        if verb == 1:
            msg = f"   chi2n = {chi2n[ii]:.3e}    niter = {niter[ii]}"
            print(msg, end='\n', flush=True)


# #############################################################################
# #############################################################################
#                      utility
# #############################################################################


def _update_TTyn(
    sparse=None,
    sigma=None,
    matrix=None,
    Tn=None,
    TTn=None,
    ii=None,
):
    # intermediates
    if sparse:
        Tn.data = scpsp.diags(1./sigma[ii, :]).dot(matrix).data
        TTn.data = Tn.T.dot(Tn).data
    else:
        Tn[...] = matrix / sigma[ii, :][:, None]
        TTn[...] = Tn.T.dot(Tn)
