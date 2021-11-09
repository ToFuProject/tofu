# -*- coding: utf-8 -*-


# Built-in
import time
import warnings


# Common
import numpy as np
import scipy.linalg as scplin
import scipy.sparse as scpsp
import scipy.optimize as scpop
import matplotlib.pyplot as plt
try:
    import sksparse as sksp
except Exception as err:
    sksp = False
    msg = "Consider installing scikit-sparse for faster innversions"
    warnings.warn(msg)

try:
    import scikits.umfpack
except Exception as err:
    skumf = False
    msg = "Consider installing scikit-umfpack for faster innversions"
    warnings.warn(msg)

# tofu
from . import _inversions_checks


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
    # choice of algo
    isotropic=None,
    sparse=None,
    positive=None,
    cholesky=None,
    regparam_algo=None,
    algo=None,
    # regularity operator
    operator=None,
    geometry=None,
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
        isotropic, sparse, positive, cholesky, regparam_algo, algo,
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
        isotropic=isotropic,
        sparse=sparse,
        positive=positive,
        cholesky=cholesky,
        regparam_algo=regparam_algo,
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
    func = eval(algo)

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
    if operator == 'D0N2':
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
    if sparse is True:
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

    out = _compute_inv_loop(
        algo=algo,
        func=func,
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
        isotropic=isotropic,
        conv_crit=conv_crit,
        sparse=sparse,
        positive=positive,
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
                    'isotropic': isotropic,
                    'algo': algo,
                    'solver': solver,
                    'chain': chain,
                    'positive': positive,
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
    algo=None,
    func=None,
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
        ) = func(
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


# #############################################################################
# #############################################################################
#                      Basic routines - augmented tikhonov
# #############################################################################


def inv_linear_augTikho_dense(
    Tn=None,
    TTn=None,
    Tyn=None,
    R=None,
    yn=None,
    sol0=None,
    nchan=None,
    nbs=None,
    mu0=None,
    conv_crit=None,
    a0bis=None,
    b0=None,
    a1bis=None,
    b1=None,
    d=None,
    conv_reg=True,
    verb=None,
    verb2head=None,
    **kwdargs,
):
    """
    Linear algorithm for Phillips-Tikhonov regularisation
    Called "Augmented Tikhonov", dense matrix version
    """

    conv = 0.           # convergence variable
    niter = 0           # number of iterations
    mu1 = 0.            # regularisation param

    # verb
    if verb >= 2:
        chi2n = np.sum((Tn.dot(sol0) - yn)**2) / nchan
        reg = sol0.dot(R.dot(sol0))
        temp = f"{nchan} * {chi2n:.3e} + {mu0:.3e} * {reg:.3e}"
        print(
            f"{verb2head}\n\t\t\t {temp} = {nchan*chi2n + mu0*reg:.3e}",
            end='\n',
        )

    # loop
    # Continue until convergence criterion, and at least 2 iterations
    while niter < 2 or conv > conv_crit:

        # call solver
        sol = scplin.solve(
            TTn + mu0*R, Tyn,
            assume_a='pos',      # faster than 'sym'
            overwrite_a=True,    # no significant gain
            overwrite_b=False,   # True is faster, but a copy of Tyn is needed
            check_finite=False,  # small speed gain compared to True
            transposed=False,
        )  # 3

        # compute residu, regularity...
        res2 = np.sum((Tn.dot(sol)-yn)**2)  # residu**2
        reg = sol.dot(R.dot(sol))           # regularity term

        # update lamb, tau
        lamb = a0bis/(0.5*reg + b0)           # Update reg. param. estimate
        tau = a1bis/(0.5*res2 + b1)           # Update noise coef. estimate
        mu1 = (lamb/tau) * (2*a1bis/res2)**d  # rescale mu with noise estimate

        # Compute convergence variable
        if conv_reg:
            conv = np.abs(mu1 - mu0) / mu1
        else:
            sol2 = sol**2
            sol2max = np.max(sol2)
            sol2[sol2 < 0.001*sol2max] = 0.001*sol2max
            conv = np.sqrt(np.sum((sol - sol0)**2 / sol2) / nbs)

        # verb
        if verb >= 2:
            temp1 = f"{nchan} * {res2/nchan:.3e} + {mu1:.3e} * {reg:.3e}"
            temp2 = f"{res2 + mu1*reg:.3e}"
            temp = f"{temp1} = {temp2}"
            print(f"\t\t{niter} \t {temp}   {tau:.3e}   {conv:.3e}")

        # update sol0, mu0 for next iteration
        sol0[:] = sol[:]
        mu0 = mu1
        niter += 1

    return sol, mu1, res2/nchan, reg, niter, [tau, lamb]


def inv_linear_augTikho_sparse(
    Tn=None,
    TTn=None,
    Tyn=None,
    R=None,
    yn=None,
    sol0=None,
    nchan=None,
    nbs=None,
    mu0=None,
    conv_crit=None,
    a0bis=None,
    b0=None,
    a1bis=None,
    b1=None,
    d=None,
    conv_reg=True,
    verb=None,
    verb2head=None,
    maxiter=None,
    tol=None,
    precond=None,       # test
    **kwdargs,
):
    """
    Linear algorithm for Phillips-Tikhonov regularisation
    Called "Augmented Tikhonov", sparese matrix version
    see InvLin_AugTikho_V1.__doc__ for details
    """

    conv = 0.           # convergence variable
    niter = 0           # number of iterations
    mu1 = 0.            # regularisation param

    # verb
    if verb >= 2:
        chi2n = np.sum((Tn.dot(sol0) - yn)**2) / nchan
        reg = sol0.dot(R.dot(sol0))
        temp = f"{nchan} * {chi2n:.3e} + {mu0:.3e} * {reg:.3e}"
        print(
            f"{verb2head}\n\t\t\t {temp} = {nchan*chi2n + mu0*reg:.3e}",
            end='\n',
        )

    # loop
    # Continue until convergence criterion, and at least 2 iterations
    while niter < 2 or conv > conv_crit:

        # sol = scpsp.linalg.spsolve(
        #    TTn + mu0*R, Tyn,
        #    permc_spec=None,
        #    use_umfpack=True,
        # )

        # seems faster
        sol, itconv = scpsp.linalg.cg(
            TTn + mu0*R, Tyn,
            x0=sol0,
            tol=tol,
            maxiter=maxiter,
            M=precond,
        )

        res2 = np.sum((Tn.dot(sol)-yn)**2)    # residu**2
        reg = sol.dot(R.dot(sol))             # regularity term

        lamb = a0bis/(0.5*reg + b0)           # Update reg. param. estimate
        tau = a1bis/(0.5*res2 + b1)           # Update noise coef. estimate
        mu1 = (lamb/tau) * (2*a1bis/res2)**d  # rescale mu with noise estimate

        # Compute convergence variable
        if conv_reg:
            conv = np.abs(mu1 - mu0) / mu1
        else:
            sol2 = sol**2
            sol2max = np.max(sol2)
            sol2[sol2 < 0.001*sol2max] = 0.001*sol2max
            conv = np.sqrt(np.sum((sol - sol0)**2 / sol2) / nbs)

        # verb
        if verb >= 2:
            temp1 = f"{nchan} * {res2/nchan:.3e} + {mu1:.3e} * {reg:.3e}"
            temp2 = f"{res2 + mu1*reg:.3e}"
            temp = f"{temp1} = {temp2}"
            print(
                f"\t\t{niter} \t {temp}   {tau:.3e}   {conv:.3e}"
            )

        sol0[:] = sol[:]            # Update reference solution
        niter += 1                  # Update number of iterations
        mu0 = mu1
    return sol, mu1, res2/nchan, reg, niter, [tau, lamb]


def inv_linear_augTikho_chol_dense(
    Tn=None,
    TTn=None,
    Tyn=None,
    R=None,
    yn=None,
    sol0=None,
    nchan=None,
    nbs=None,
    mu0=None,
    conv_crit=None,
    a0bis=None,
    b0=None,
    a1bis=None,
    b1=None,
    d=None,
    conv_reg=True,
    verb=None,
    verb2head=None,
    **kwdargs,
):
    """
    """

    conv = 0.           # convergence variable
    niter = 0           # number of iterations
    mu1 = 0.            # regularisation param

    # verb
    if verb >= 2:
        chi2n = np.sum((Tn.dot(sol0) - yn)**2) / nchan
        reg = sol0.dot(R.dot(sol0))
        temp = f"{nchan} * {chi2n:.3e} + {mu0:.3e} * {reg:.3e}"
        print(
            f"{verb2head}\n\t\t\t {temp} = {nchan*chi2n + mu0*reg:.3e}",
            end='\n',
        )

    # loop
    # Continue until convergence criterion, and at least 2 iterations
    while niter < 2 or conv > conv_crit:
        try:
            # choleski decomposition requires det(TT + mu0*LL) != 0
            # (chol(A).T * chol(A) = A
            chol = scplin.cholesky(
                TTn + mu0*R,
                lower=False,
                check_finite=False,
                overwrite_a=False,
            )
            # Use np.linalg.lstsq for double-solving the equation
            sol = scplin.cho_solve(
                (chol, False), Tyn,
                overwrite_b=None,
                check_finite=True,
            )
        except Exception as err:
            # call solver
            sol = scplin.solve(
                TTn + mu0*R, Tyn,
                assume_a='sym',         # chol failed => not 'pos'
                overwrite_a=True,       # no significant gain
                overwrite_b=False,      # True faster, but a copy of Tyn needed
                check_finite=False,     # small speed gain compared to True
                transposed=False,
            )  # 3

        # compute residu, regularity...
        res2 = np.sum((Tn.dot(sol)-yn)**2)  # residu**2
        reg = sol.dot(R.dot(sol))           # regularity term

        # update lamb, tau
        lamb = a0bis/(0.5*reg + b0)           # Update reg. param. estimate
        tau = a1bis/(0.5*res2 + b1)           # Update noise coef. estimate
        mu1 = (lamb/tau) * (2*a1bis/res2)**d  # mu rescale with noise estimate

        # Compute convergence variable
        if conv_reg:
            conv = np.abs(mu1 - mu0) / mu1
        else:
            sol2 = sol**2
            sol2max = np.max(sol2)
            sol2[sol2 < 0.001*sol2max] = 0.001*sol2max
            conv = np.sqrt(np.sum((sol - sol0)**2 / sol2) / nbs)

        # verb
        if verb >= 2:
            temp1 = f"{nchan} * {res2/nchan:.3e} + {mu1:.3e} * {reg:.3e}"
            temp2 = f"{res2 + mu1*reg:.3e}"
            temp = f"{temp1} = {temp2}"
            print(f"\t\t{niter} \t {temp}   {tau:.3e}   {conv:.3e}")

        # update sol0, mu0 for next iteration
        sol0[:] = sol[:]
        mu0 = mu1
        niter += 1

    return sol, mu1, res2/nchan, reg, niter, [tau, lamb]


def inv_linear_augTikho_chol_sparse(
    Tn=None,
    TTn=None,
    Tyn=None,
    R=None,
    yn=None,
    sol0=None,
    nchan=None,
    nbs=None,
    mu0=None,
    conv_crit=None,
    a0bis=None,
    b0=None,
    a1bis=None,
    b1=None,
    d=None,
    conv_reg=True,
    verb=None,
    verb2head=None,
    **kwdargs,
):
    """
    Linear algorithm for Phillips-Tikhonov regularisation
    Called "Augmented Tikhonov"

    Augmented in the sense that bayesian statistics are combined
        with standard Tikhonov regularisation
    Determines both noise (common multiplicative coefficient) and
        regularisation parameter automatically
    We assume here that all arrays are scaled (noise, conditioning...)
    Sparse matrixes are also prefered to speed-up the computation

    In this method:
      tau is an approximation of the inverse of the noise coefficient
      lamb is an approximation of the regularisation parameter

    N.B.: The noise and reg. param. have probability densities of the form:
        f(x) = x^(a-1) * exp(-bx)
    This function's maximum is in x = (a-1)/b, so a = b+1 gives a maximum at 1.
        (a0, b0) for the reg. param.
        (a1, b1) for the noise estimate

    Ref:
      [1] Jin B., Zou J., Inverse Problems, vol.25, nb.2, 025001, 2009
      [2] http://www.math.uni-bremen.de/zetem/cms/media.php/250/nov14talk_jin%20bangti.pdf
      [3] Kazufumi Ito, Bangti Jin, Jun Zou,
        "A New Choice Rule for Regularization Parameters in Tikhonov
        Regularization", Research report, University of Hong Kong, 2008
    """

    conv = 0.           # convergence variable
    niter = 0           # number of iterations
    mu1 = 0.            # regularisation param

    # verb
    if verb >= 2:
        chi2n = np.sum((Tn.dot(sol0) - yn)**2) / nchan
        reg = sol0.dot(R.dot(sol0))
        temp = f"{nchan} * {chi2n:.3e} + {mu0:.3e} * {reg:.3e}"
        print(
            f"{verb2head}\n\t\t\t {temp} = {nchan*chi2n + mu0*reg:.3e}",
            end='\n',
        )

    # loop
    # Continue until convergence criterion, and at least 2 iterations
    factor = None
    while niter < 2 or conv > conv_crit:
        try:
            # choleski decomposition requires det(TT + mu0*LL) != 0
            # A = (chol(A).T * chol(A)
            # optimal if matrix is csc
            if sksp is False:
                factor = scpsp.linalg.factorized(TTn + mu0*R)
                sol = factor(Tyn)
            else:
                if factor is None:
                    factor = sksp.cholmod.cholesky(
                        TTn + mu0*R,
                        beta=0,
                        mode='auto',
                        ordering_method='default',
                        use_long=False,
                    )
                else:
                    # re-use same factor
                    factor.cholesky_inplace(TTn + mu0*R, beta=0)
                sol = factor.solve_A(Tyn)
        except Exception as err:
            # call solver
            sol = scpsp.linalg.spsolve(
                TTn + mu0*R, Tyn,
                permc_spec=None,
                use_umfpack=True,
            )

        # compute residu, regularity...
        res2 = np.sum((Tn.dot(sol)-yn)**2)  # residu**2
        reg = sol.dot(R.dot(sol))           # regularity term

        # update lamb, tau
        lamb = a0bis/(0.5*reg + b0)             # Update reg. param. estimate
        tau = a1bis/(0.5*res2 + b1)             # Update noise coef. estimate
        mu1 = (lamb/tau) * (2*a1bis/res2)**d    # Update reg. param. rescaling

        # Compute convergence variable
        if conv_reg:
            conv = np.abs(mu1 - mu0) / mu1
        else:
            sol2 = sol**2
            sol2max = np.max(sol2)
            sol2[sol2 < 0.001*sol2max] = 0.001*sol2max
            conv = np.sqrt(np.sum((sol - sol0)**2 / sol2) / nbs)

        # verb
        if verb >= 2:
            temp1 = f"{nchan} * {res2/nchan:.3e} + {mu1:.3e} * {reg:.3e}"
            temp2 = f"{res2 + mu1*reg:.3e}"
            temp = f"{temp1} = {temp2}"
            print(f"\t\t{niter} \t {temp}   {tau:.3e}   {conv:.3e}")

        # update sol0, mu0 for next iteration
        sol0[:] = sol[:]
        mu0 = mu1
        niter += 1

    return sol, mu1, res2/nchan, reg, niter, [tau, lamb]


def inv_linear_augTikho_pos_dense(
    Tn=None,
    TTn=None,
    Tyn=None,
    R=None,
    yn=None,
    sol0=None,
    nchan=None,
    nbs=None,
    mu0=None,
    conv_crit=None,
    a0bis=None,
    b0=None,
    a1bis=None,
    b1=None,
    d=None,
    conv_reg=True,
    verb=None,
    verb2head=None,
    # specific
    method=None,
    options=None,
    bounds=None,
    func_val=None,
    func_jac=None,
    func_hess=None,
    **kwdargs,
):
    """
    Quadratic algorithm for Phillips-Tikhonov regularisation
    Alternative to the linear version with positivity constraint
    see TFI.InvLin_AugTikho_V1.__doc__ for details
    """

    conv = 0.           # convergence variable
    niter = 0           # number of iterations
    mu1 = 0.            # regularisation param

    # verb
    if verb >= 2:
        chi2n = np.sum((Tn.dot(sol0) - yn)**2) / nchan
        reg = sol0.dot(R.dot(sol0))
        temp = f"{nchan} * {chi2n:.3e} + {mu0:.3e} * {reg:.3e}"
        print(
            f"{verb2head}\n\t\t\t {temp} = {nchan*chi2n + mu0*reg:.3e}",
            end='\n',
        )

    while niter < 2 or conv > conv_crit:
        # quadratic method for positivity constraint
        sol = scpop.minimize(
            func_val, sol0,
            args=(mu0, Tn, yn, TTn, Tyn),
            jac=func_jac,
            hess=func_hess,
            method=method,
            bounds=bounds,
            options=options,
        ).x

        # compute residu, regularity...
        res2 = np.sum((Tn.dot(sol)-yn)**2)  # residu**2
        reg = sol.dot(R.dot(sol))           # regularity term

        # update lamb, tau
        lamb = a0bis/(0.5*reg + b0)             # Update reg. param. estimate
        tau = a1bis/(0.5*res2 + b1)             # Update noise coef. estimate
        mu1 = (lamb/tau) * (2*a1bis/res2)**d    # Update reg. param. rescaling

        # Compute convergence variable
        if conv_reg:
            conv = np.abs(mu1 - mu0) / mu1
        else:
            sol2 = sol**2
            sol2max = np.max(sol2)
            sol2[sol2 < 0.001*sol2max] = 0.001*sol2max
            conv = np.sqrt(np.sum((sol - sol0)**2 / sol2) / nbs)

        # verb
        if verb >= 2:
            temp1 = f"{nchan} * {res2/nchan:.3e} + {mu1:.3e} * {reg:.3e}"
            temp2 = f"{res2 + mu1*reg:.3e}"
            temp = f"{temp1} = {temp2}"
            print(f"\t\t{niter} \t {temp}   {tau:.3e}   {conv:.3e}")

        # update sol0, mu0 for next iteration
        sol0[:] = sol[:]
        mu0 = mu1
        niter += 1

    return sol, mu1, res2/nchan, reg, niter, [tau, lamb]


# #############################################################################
# #############################################################################
#               Basic routines - discrepancy principle
# #############################################################################


def inv_linear_DisPrinc_sparse(
    Tn=None,
    TTn=None,
    Tyn=None,
    R=None,
    yn=None,
    sol0=None,
    nchan=None,
    mu0=None,
    precond=None,
    verb=None,
    verb2head=None,
    # specific
    chi2n_tol=None,
    chi2n_obj=None,
    maxiter=None,
    tol=None,
    **kwdargs,
):
    """
    Discrepancy principle: find mu such that chi2n = 1 +/- tol
    """

    niter = 0
    lchi2n = np.array([np.sum((Tn.dot(sol0) - yn)**2) / nchan])
    lmu = np.array([mu0])
    chi2n_obj_log = np.log(chi2n_obj)

    # verb
    if verb >= 2:
        reg = sol0.dot(R.dot(sol0))
        temp = f"{nchan} * {lchi2n[0]:.3e} + {mu0:.3e} * {reg:.3e}"
        print(
            f"{verb2head}\n\t\t\t {temp} = {nchan*lchi2n[0] + mu0*reg:.3e}",
            end='\n',
        )

    while niter == 0 or np.abs(lchi2n[-1] - chi2n_obj) > chi2n_tol:
        sol, itconv = scpsp.linalg.cg(
            TTn + lmu[-1]*R, Tyn,
            x0=sol0,
            tol=tol,
            maxiter=maxiter,
            M=precond,
        )

        lchi2n = np.append(lchi2n, np.sum((Tn.dot(sol) - yn)**2) / nchan)

        if niter == 0:
            if lchi2n[-1] >= chi2n_obj + chi2n_tol:
                lmu = np.append(lmu, lmu[-1] / 50.)
            elif lchi2n[-1] <= chi2n_obj - chi2n_tol:
                lmu = np.append(lmu, lmu[-1] * 50.)
            else:
                lmu = np.append(lmu, lmu[-1])
        elif niter == 1 or (
            np.all(lchi2n >= chi2n_obj + chi2n_tol)
            or np.all(lchi2n <= chi2n_obj - chi2n_tol)
        ):
            if lchi2n[-1] >= chi2n_obj + chi2n_tol:
                lmu = np.append(lmu, lmu[-1] / 50.)
            else:
                lmu = np.append(lmu, lmu[-1] * 50.)
        else:
            if lmu[-2] == lmu[-1]:
                # if the algo is stuck => break to avoid infinite loop
                ind = np.argmin(lchi2n[1:] - chi2n_obj)
                lmu[-1] = lmu[ind]
                lchi2n[-1] = lchi2n[ind]
                sol, itconv = scpsp.linalg.cg(
                    TTn + lmu[-1]*R, Tyn,
                    x0=sol0,
                    tol=tol,
                    maxiter=maxiter,
                    M=precond,
                )
                break
            else:
                indsort = np.argsort(lchi2n[1:])
                lmu = np.append(lmu, np.exp(np.interp(
                    chi2n_obj_log,
                    np.log(lchi2n[1:])[indsort],
                    np.log(lmu)[indsort]
                )))

        # verb
        if verb >= 2:
            reg = sol.dot(R.dot(sol))
            res2 = np.sum((Tn.dot(sol)-yn)**2)
            temp1 = f"{nchan} * {lchi2n[-1]:.3e} + {lmu[-1]:.3e} * {reg:.3e}"
            temp2 = f"{res2 + lmu[-1]*reg:.3e}"
            temp = f"{temp1} = {temp2}"
            print(f"\t\t{niter} \t {temp}")

        sol0[:] = sol
        niter += 1

    reg = sol.dot(R.dot(sol))           # regularity term
    return sol, lmu[-1], lchi2n[-1], reg, niter, None
