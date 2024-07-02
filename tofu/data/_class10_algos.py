# -*- coding: utf-8 -*-


import copy
import warnings


import numpy as np
import scipy.linalg as scplin
import scipy.optimize as scpop
import scipy.sparse as scpsp
import datastock as ds


dfail = {}
try:
    import sksparse as sksp
except Exception as err:
    sksp = False
    dfail['sksparse'] = "For cholesk factorizations"

try:
    import scikits.umfpack as skumf
except Exception as err:
    skumf = False
    dfail['umfpack'] = "For faster sparse matrices"

if len(dfail) > 0:
    lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
    msg = (
        "Consider installing the following for faster inversions:\n"
        + "\n".join(lstr)
    )
    warnings.warn(msg)


# optional
try:
    from .. import tomotok2tofu
except Exception as err:
    tomotok2tofu = False


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


# ################################################################
# ################################################################
#                      Basic routines - augmented tikhonov
# ################################################################


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
    maxiter_outer=None,
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
    while niter <= 2 or (conv > conv_crit and niter < maxiter_outer):

        # call solver
        sol = scplin.solve(
            TTn + mu0*R, Tyn,
            assume_a='pos',      # 'pos' faster than 'sym'
            overwrite_a=True,    # no significant gain
            overwrite_b=False,   # True is faster, but a copy of Tyn is needed
            check_finite=False,  # small speed gain compared to True
            transposed=False,
        )  # 3

        # call augmented Tikhonov update of mu
        mu1, conv, res2, reg, tau, lamb = _augTikho_update(
            Tn, sol, yn, R,
            a0bis, b0, a1bis, b1,
            d, mu0, conv_reg, nbs, sol0,
            # verb
            verb=verb,
            nchan=nchan,
            niter=niter,
        )

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
    maxiter_outer=None,
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

    # import matplotlib.pyplot as plt # DB
    # import datastock as ds
    # print(R.shape, Tn.shape)
    # print()
    # ds.plot_as_array(R)
    # ds.plot_as_array(Tn)
    # ds.plot_as_array(yn)
    # ds.plot_as_array(sol0)
    # ds.plot_as_array(Tn.dot(sol0))

    # raise Exception()

    # loop
    # Continue until convergence criterion, and at least 2 iterations
    while niter <= 2 or (conv > conv_crit and niter < maxiter_outer):

        # sol = scpsp.linalg.spsolve(
        #    TTn + mu0*R, Tyn,
        #    permc_spec=None,
        #    use_umfpack=True,
        # )

        # seems faster
        sol, itconv = scpsp.linalg.cg(
            TTn + mu0*R, Tyn,
            x0=sol0,
            atol=tol,
            maxiter=maxiter,
            M=precond,
        )

        # call augmented Tikhonov update of mu
        mu1, conv, res2, reg, tau, lamb = _augTikho_update(
            Tn, sol, yn, R,
            a0bis, b0, a1bis, b1,
            d, mu0, conv_reg, nbs, sol0,
            # verb
            verb=verb,
            nchan=nchan,
            niter=niter,
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
    maxiter_outer=None,
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
    while niter <= 2 or (conv > conv_crit and niter < maxiter_outer):
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

        # call augmented Tikhonov update of mu
        mu1, conv, res2, reg, tau, lamb = _augTikho_update(
            Tn, sol, yn, R,
            a0bis, b0, a1bis, b1,
            d, mu0, conv_reg, nbs, sol0,
            # verb
            verb=verb,
            nchan=nchan,
            niter=niter,
        )

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
    maxiter_outer=None,
    **kwdargs,
):

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
    while niter <= 2 or (conv > conv_crit and niter < maxiter_outer):
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

        # call augmented Tikhonov update of mu
        mu1, conv, res2, reg, tau, lamb = _augTikho_update(
            Tn, sol, yn, R,
            a0bis, b0, a1bis, b1,
            d, mu0, conv_reg, nbs, sol0,
            # verb
            verb=verb,
            nchan=nchan,
            niter=niter,
        )

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
    maxiter_outer=None,
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

    while niter <= 2 or (conv > conv_crit and niter < maxiter_outer):
        # quadratic method for positivity constraint
        sol = scpop.minimize(
            func_val, sol0,
            args=(mu0, Tn, yn, TTn, Tyn, R),
            jac=func_jac,
            hess=func_hess,
            method=method,
            bounds=bounds,
            options=options,
        ).x

        # call augmented Tikhonov update of mu
        mu1, conv, res2, reg, tau, lamb = _augTikho_update(
            Tn, sol, yn, R,
            a0bis, b0, a1bis, b1,
            d, mu0, conv_reg, nbs, sol0,
            # verb
            verb=verb,
            nchan=nchan,
            niter=niter,
        )

        # update sol0, mu0 for next iteration
        sol0[:] = sol[:]
        mu0 = mu1
        niter += 1

    return sol, mu1, res2/nchan, reg, niter, [tau, lamb]


def _augTikho_update(
    Tn, sol, yn, R,
    a0bis, b0, a1bis, b1,
    d, mu0, conv_reg, nbs, sol0,
    # verb
    verb=None,
    nchan=None,
    niter=None,
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

    res2 = np.sum(np.power(Tn.dot(sol)-yn, 2))    # residu**2
    reg = sol.dot(R.dot(sol))             # regularity term

    lamb = a0bis/(0.5*reg + b0)           # Update reg. param. estimate
    tau = a1bis/(0.5*res2 + b1)           # Update noise coef. estimate

    # original formula
    mu1 = (lamb/tau) * (2*a1bis/res2)**d  # rescale mu with noise estimate
    # mu1 = (lamb/tau) * (2*a1bis/max(res2, 1e-3))**d  # rescale mu with noise estimate

    if verb >= 3:
        msg = (
            f"\n\t\tconv_reg, d = {conv_reg}, {d}\n"
            f"\t\ta0bis, b0 = {a0bis}, {b0}\n"
            f"\t\ta1bis, b1 = {a1bis}, {b1}\n"
            f"\t\tres2, reg = {res2}, {reg}\n"
            f"\t\tlamb = a0bis/(0.5*reg + b0) = {lamb}\n"
            f"\t\ttau = a1bis/(0.5*res2 + b1) = {tau}\n"
            "\t\tmu = (lamb/tau) * (2*a1bis/res2)**d"
            f" = \t{lamb/tau} * {(2*a1bis/res2)**d} = {mu1}\n"
        )
        print(msg)

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

    return mu1, conv, res2, reg, tau, lamb


# ################################################################
# ################################################################
#               Basic routines - discrepancy principle
# ################################################################


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
    maxiter_outer=None,
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

    while niter < 2 or (np.abs(lchi2n[-1] - chi2n_obj) > chi2n_tol and niter < maxiter_outer):

        sol, itconv = scpsp.linalg.cg(
            TTn + lmu[-1]*R, Tyn,
            x0=sol0,
            atol=tol,
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
                    atol=tol,
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
            print(f"\t\t{niter} \t {temp}   \t   {np.abs(lchi2n[-1] - chi2n_obj):.3e}")

        sol0[:] = sol
        niter += 1

    reg = sol.dot(R.dot(sol))           # regularity term
    return sol, lmu[-1], lchi2n[-1], reg, niter, None


# #############################################################################
# #############################################################################
#               Non-regularized - least squares
# #############################################################################


def inv_linear_leastsquares_bounds(
    Tn=None,
    TTn=None,
    Tyn=None,
    R=None,
    yn=None,
    sol0=None,
    nchan=None,
    mu0=None,       # useless
    precond=None,
    verb=None,
    verb2head=None,
    # specific
    A=None,
    b=None,
    maxiter=None,
    tol=None,
    lsmr_tol=None,
    lsq_solver=None,
    method=None,
    bounds=None,
    dconstraints=None,
    **kwdargs,
):
    """
    Discrepancy principle: find mu such that chi2n = 1 +/- tol
    """

    # ---------
    # check inputs

    verbscp = verb
    if method is None:
        method = 'trf'
    if lsq_solver is None:
        if scpsp.issparse(Tn):
            lsq_solver = 'lsmr'
        else:
            lsq_solver = 'exact'

    # ------------
    # optimization

    # verb
    if verb >= 2:
        chi2n0 = np.sum((Tn.dot(sol0) - yn)**2) / nchan
        print(
            f"{verb2head}\n\t\t\t {nchan} * {chi2n0:.3e} = {nchan*chi2n0:.3e}",
            end='\n',
        )

    # optimize
    res = scpop.lsq_linear(
        Tn,
        yn,
        bounds=bounds,
        method=method,
        tol=tol,
        lsq_solver=lsq_solver,
        lsmr_tol=lsmr_tol,
        max_iter=maxiter,
        verbose=verb,
    )

    # --------
    # return

    chi2n = np.sum((Tn.dot(res.x) - yn)**2) / nchan

    # verb
    if verb >= 2:
        temp = f"{nchan} * {chi2n:.3e} = {nchan * chi2n:.3e}"
        print(f"\t\t{res.nit} \t {temp}")

    if dconstraints is None:
        return res.x, None, chi2n, None, res.nit, None
    else:
        return (
            dconstraints['coefs'].dot(res.x) + dconstraints['offset']
        )


# ##################################################################
# ##################################################################
#               _DALGO at import time
# ##################################################################


_DALGO = get_available_inversions_algo(returnas=dict, verb=False)