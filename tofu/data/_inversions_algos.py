

import warnings


import numpy as np
import scipy.linalg as scplin
import scipy.optimize as scpop
import scipy.sparse as scpsp


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
