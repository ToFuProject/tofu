# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as nip
import scipy.sparse as scpsp


# tofu
from . import _inversions_checks


# #############################################################################
# #############################################################################
#                           main
# #############################################################################


def compute_inversions(
    coll=None,
    key=None,
    data=None,
    sigma=None,
    operator=None,
    geometry=None,
    isotropic=None,
    method=None,
    sparse=None,
):

    # -------------
    # check inputs

    (
        keymat, keybs, keym, data, sigma,
        isotropic, sparse, matrix, crop,
    ) = _inversions_checks._compute_check(
        coll=coll,
        key=key,
        data=data,
        sigma=sigma,
        isotropic=isotropic,
        sparse=sparse,
    )

    # -------------
    # prepare data

    opmat, operator, geometry, dim, ref, crop = mesh.add_bsplines_operator(
        operator=operator,
        geometry=geometry,
        returnas=True,
        store=False,
        crop=crop,
    )

    # -------------
    # compute

    out = _compute_inv(
        matrix=matrix,
        key=key,
        keybs=keybs,
        keym=keym,
        data=data,
        sigma=sigma,
        isotropic=isotropic,
        sparse=sparse,
    )

    # -------------
    # format output


    return


# #############################################################################
# #############################################################################
#                           _compute
# #############################################################################


def _compute_inv(
    matrix=None,
    key=None,
    keybs=None,
    keym=None,
    data=None,
    sigma=None,
    isotropic=None,
    sparse=None,
):

    # LL = opmat
    # mm = indicator of type of operator
    # i


    t = t.flatten()
    if sigma is None:
        sigma = 0.5*np.mean(np.mean(data))*np.ones((TMat.shape[0],))
    NdimSig = sigma.ndim

    # indt
    if Dt is None:
        indt = np.ones((t.size,),dtype=bool)
    else:
        if type(Dt) is float:
            indt = np.nanargmin(np.abs(t-Dt))
        elif len(Dt)==2:
            indt = (t>=Dt[0]) & (t<=Dt[1])
    t = t[indt]

    # data
    data = data[indt,:]
    if data.ndim==1:
        data = data.reshape((1,data.size))
    if NdimSig==2:
        sigma = sigma[indt,:]
    #SigMean = np.mean(np.mean(sigma))

    # shape
    NMes, Nbf = TMat.shape
    Nt = t.size

    # method
    if SolMethod == 'InvLin_AugTikho_V1' and Pos:
        SolMethod = 'InvLinQuad_AugTikho_V1'

    # kwdargs
    if KWARGS is None:
        if SolMethod in 'InvLin_AugTikho_V1':
            KWARGS = {'a0':TFD.AugTikho_a0, 'b0':TFD.AugTikho_b0, 'a1':TFD.AugTikho_a1, 'b1':TFD.AugTikho_b1, 'd':TFD.AugTikho_d, 'ConvReg':True, 'FixedNb':True}
        elif SolMethod == 'InvLin_DisPrinc_V1':
            KWARGS = {'chi2Tol':TFD.chi2Tol, 'chi2Obj':TFD.chi2Obj}
        elif SolMethod == 'InvLinQuad_AugTikho_V1':
            KWARGS = {'a0':TFD.AugTikho_a0, 'b0':TFD.AugTikho_b0, 'a1':TFD.AugTikho_a1, 'b1':TFD.AugTikho_b1, 'd':TFD.AugTikho_d, 'ConvReg':True, 'FixedNb':True}
        elif SolMethod == 'InvQuad_AugTikho_V1':
            KWARGS = {'a0':TFD.AugTikho_a0, 'b0':TFD.AugTikho_b0, 'a1':TFD.AugTikho_a1, 'b1':TFD.AugTikho_b1, 'd':TFD.AugTikho_d, 'ConvReg':True, 'FixedNb':True}

    if VerbNb is None:
        VerbNb = np.ceil(Nt/20.)

    # -----------------------------------
    # Getting initial solution - step 1/2

    if scpsp.issparse(TMat):
        tmat = TMat.toarray()
    else:
        tmat = TMat.copy()
    if NdimSig==2:
        ssigm = np.mean(sigma,axis=0)
    else:
        ssigm = sigma.copy()
    if data.ndim==1:
        ddat = data.copy()
    else:
        ddat = np.copy(data[0,:])

    sol0 = Default_sol(BF2, tmat, ddat, sigma0=ssigm, N=3)

    # Using discrepancy principle on averaged signal
    LL, mm = BF2.get_IntOp(Deriv=Deriv, Mode=IntMode)

    if mm==2:
        LL = LL[0]+LL[1]
    b = ddat/ssigm                      # DB
    Tm = np.diag(1./ssigm).dot(tmat)    # DB
    Tb = Tm.T.dot(b)                    # DB
    TT = Tm.T.dot(Tm)                   # DB
    sol0, mu0 = InvLin_DisPrinc_V1_Sparse(NMes, Nbf, SpFc[SpType](Tm), SpFc[SpType](TT), SpFc[SpType](LL), Tb, b, sol0, ConvCrit=ConvCrit, mu0=mu0, chi2Tol=0.2, chi2Obj=1.2, M=None, Verb=True)[0:2]    # DB

    # Launching inversions with chosen method - step 2/2 of initial solution included below
    Sol, Mu, Chi2N, R, Nit, Spec = np.nan*np.ones((Nt,Nbf)), np.nan*np.ones((Nt,)), np.nan*np.ones((Nt,)), np.nan*np.ones((Nt,)), np.nan*np.ones((Nt,)), [[] for ii in range(0,Nt)]
    NormL = sol0.dot(LL.dot(sol0))



    if sparse:

        func = globals()[SolMethod+'_Sparse']

        if not SpIs[SpType](TMat):
            TMat = SpFc[SpType](TMat)

        if not SpIs[SpType](LL):
            LL = SpFc[SpType](LL)

        b = ddat/ssigm
        Tm = np.diag(1./ssigm).dot(tmat)
        Tb = Tm.T.dot(b)
        TT = SpFc[SpType](Tm.T.dot(Tm))

        sol0, mu0, chi2N0, R0, Nit0, spec0 = func(
            NMes, Nbf, SpFc[SpType](Tm), TT, LL, Tb, b, sol0,
            ConvCrit=ConvCrit, mu0=mu0, M=None, Verb=False, **KWARGS,
        )
        M = scpsplin.inv((TT+mu0*LL/sol0.dot(LL.dot(sol0))).tocsc())

        # Beware of element-wise operations vs matrix operations !!!!
        if mm == 0:
            msg = (
                "Inversion routines require a quadratic operator!"
            )
            raise Exception(msg)

        if NdimSig == 2:
            b = data/sigma
            for ii in range(0,Nt):
                if ii==0 or (ii+1)%VerbNb==0:
                    print "    Running inversion for time step ", ii+1,' / ', Nt

                Tm = np.diag(1./sigma[ii,:]).dot(tmat)
                Tb = np.concatenate(
                    tuple([
                        (Tm.T.dot(b[ii,:])).reshape((1,Nbf)) for ii in range(0,Nt)
                    ]),
                    axis=0,
                )
                TT = SpFc[SpType](Tm.T.dot(Tm))
                Sol[ii,:], Mu[ii], Chi2N[ii], R[ii], Nit[ii], Spec[ii] = func(
                    NMes, Nbf, SpFc[SpType](Tm), TT, LL, Tb[ii,:], b[ii,:],
                    sol0, ConvCrit=ConvCrit, mu0=mu0, NormL=NormL,
                    M=M, Verb=Verb, **KWARGS,
                )
                sol0[:] = Sol[ii,:]
                mu0 = Mu[ii]
                #print '    Nit or Ntry : ', Spec[ii][0]
        else:
            b = data.dot(np.diag(1./sigma))
            Tm = np.diag(1./sigma).dot(tmat)
            Tb = np.concatenate(
                tuple([
                    (Tm.T.dot(b[ii,:])).reshape((1,Nbf))
                    for ii in range(0,Nt)
                ]),
                axis=0,
            )
            TT = SpFc[SpType](Tm.T.dot(Tm))
            Tm = SpFc[SpType](Tm)
            for ii in range(0,Nt):
                if ii==0 or (ii+1)%VerbNb==0:
                    msg = (
                        f"\tRunning inversion for time step {ii+1} / {Nt}, "
                        f"{KWARGS['ConvReg']}, {KWARGS['FixedNb']}"
                    )
                    print(msg)
                Sol[ii,:], Mu[ii], Chi2N[ii], R[ii], Nit[ii], Spec[ii] = func(NMes, Nbf, Tm, TT, LL, Tb[ii,:], b[ii,:], sol0, ConvCrit=ConvCrit, mu0=mu0, NormL=NormL, M=M, Verb=Verb, **KWARGS)
                sol0[:] = Sol[ii,:]
                mu0 = Mu[ii]
                #print '    Nit or Ntry : ', Spec[ii][0]

    else:
        pass

    return


# #############################################################################
# #############################################################################
#                       functions
# #############################################################################
i

def InvLin_AugTikho_V1(
    NMes, Nbf, Tm, TT, LL, Tb, b, sol0,
    mu0=TFD.mu0,
    ConvCrit=TFD.ConvCrit,
    a0=TFD.AugTikho_a0,
    b0=TFD.AugTikho_b0,
    a1=TFD.AugTikho_a1,
    b1=TFD.AugTikho_b1,
    d=TFD.AugTikho_d,
    NormL=None,
    Verb=False,
    ConvReg=True,
    FixedNb=True,
):
    """ Linear algorithm for Phillips-Tikhonov regularisation, called "Augmented Tikhonov"
    augmented in the sense that bayesian statistics are combined with standard Tikhonov regularisation
    Determines both noise (common multiplicative coefficient) and regularisation paremeter automatically
    We assume here that all arrays have previously been scaled (noise, conditioning...)
    Sparse matrixes are also prefered to speed-up the computation
    In this method:
      tau is an approximation of the inverse of the noise coefficient
      lamb is an approximation of the regularisation parameter
    N.B.: The noise and reg. param. have probability densities of the form : f(x) = x^(a-1) * exp(-bx)
    This function maximum is in x = (a-1)/b, so a = b+1 gives a maximum at 1.
    (a0,b0) for the reg. param. and (a1,b1) for the noise estimate
    Ref:
      [1] Jin B., Zou J., Inverse Problems, vol.25, nb.2, 025001, 2009
      [2] http://www.math.uni-bremen.de/zetem/cms/media.php/250/nov14talk_jin%20bangti.pdf
      [3] Kazufumi Ito, Bangti Jin, Jun Zou, "A New Choice Rule for Regularization Parameters in Tikhonov Regularization", Research report, University of Hong Kong, 2008
    """

    a0bis = a0-1.+Nbf/2. if not FixedNb else a0-1.+1200./2.
    a1bis = a1-1.+NMes/2.

    Conv = 0.                                                   # Initialise convergence variable
    Nit = 0                                                     # Initialise number of iterations
    chi2N, R = [], []                                           # Initialise residu list and regularisation term list
    lmu = [mu0]                                                 # Initialise regularisation parameter
    if NormL is None:
        NormL = sol0.dot(LL.dot(sol0))
    LL = LL / NormL                                               # Scale the regularity operator
    if Verb:
        print "        Init. guess : N*Chi2N + mu*RN = ", NMes,'*',np.sum((Tm.dot(sol0)-b)**2)/NMes,'+',lmu[-1],'*',sol0.dot(LL.dot(sol0)), ' = ', np.sum((Tm.dot(sol0)-b)**2)+lmu[-1]*sol0.dot(LL.dot(sol0))
        while  Nit<2 or Conv>ConvCrit:                              # Continue until convergence criterion is fulfilled, and do at least 2 iterations
            if np.linalg.det(TT + lmu[-1]*LL)==0.:
                sol = np.linalg.lstsq(TT + lmu[-1]*LL,Tb)[0]
            else:
                Chol = np.linalg.cholesky(TT + lmu[-1]*LL)                   # Use Choleski factorisation (Chol(A).T * Chol(A) = A) for faster computation
                sol = np.linalg.lstsq(Chol,np.linalg.lstsq(Chol.T,Tb)[0])[0]  # Use np.linalg.lstsq for double-solving the equation

            Res = np.sum((Tm.dot(sol)-b)**2)                        # Compute residu**2
            chi2N.append(Res/NMes)                                  # Record normalised residu history
            R.append(sol.dot(LL.dot(sol)))                          # Compute and record regularity term

            lamb = a0bis/(0.5*R[Nit]+b0)                            # Update reg. param. estimate
            tau = a1bis/(0.5*Res+b1)                                # Update noise coef. estimate
            lmu.append((lamb/tau) * (2*a1bis/Res)**d)               # Update regularisation parameter taking into account rescaling with noise estimate
            if ConvReg:
                Conv = np.abs(lmu[-1]-lmu[-2])/lmu[-1]
            else:
                Conv = np.sqrt(np.sum((sol-sol0)**2/np.max([sol**2,0.001*np.max(sol**2)*np.ones((Nbf,))],axis=0))/Nbf)        # Compute convergence variable
            print "        Nit = ",str(Nit),"   N*Chi2N + mu*R = ", NMes,'*',chi2N[Nit],'+',lmu[-1],'*',R[Nit], ' = ', Res+lmu[-1]*R[Nit], '    tau = ',tau, '    Conv = ',Conv
            sol0[:] = sol[:]                                           # Update reference solution
            Nit += 1                                                # Update number of iterations
    else:
        while  Nit<2 or Conv>ConvCrit:                              # Continue until convergence criterion is fulfilled, and do at least 2 iterations
            if np.linalg.det(TT + lmu[-1]*LL)==0.:
                sol = np.linalg.lstsq(TT + lmu[-1]*LL,Tb)[0]
            else:
                Chol = np.linalg.cholesky(TT + lmu[-1]*LL)                   # Use Choleski factorisation (Chol(A).T * Chol(A) = A) for faster computation
                sol = np.linalg.lstsq(Chol,np.linalg.lstsq(Chol.T,Tb)[0])[0]  # Use np.linalg.lstsq for double-solving the equation

            Res = np.sum((Tm.dot(sol)-b)**2)                        # Compute residu**2
            chi2N.append(Res/NMes)                                  # Record normalised residu history
            R.append(sol.dot(LL.dot(sol)))                          # Compute and record regularity term

            lamb = a0bis/(0.5*R[Nit]+b0)                            # Update reg. param. estimate
            tau = a1bis/(0.5*Res+b1)                                # Update noise coef. estimate
            lmu.append((lamb/tau) * (2*a1bis/Res)**d)               # Update regularisation parameter taking into account rescaling with noise estimate
            if ConvReg:
                Conv = np.abs(lmu[-1]-lmu[-2])/lmu[-1]
            else:
                Conv = np.sqrt(np.sum((sol-sol0)**2/np.max([sol**2,0.001*np.max(sol**2)*np.ones((Nbf,))],axis=0))/Nbf)        # Compute convergence variable
            sol0[:] = sol[:]                                           # Update reference solution
            Nit += 1                                                # Update number of iterations

    return sol, lmu[-1], chi2N[-1], R[-1], Nit, [tau, lamb]


def InvLin_AugTikho_V1_Sparse(
    NMes, Nbf, Tm, TT, LL, Tb, b, sol0,
    mu0=TFD.mu0,
    ConvCrit=TFD.ConvCrit,
    a0=TFD.AugTikho_a0,
    b0=TFD.AugTikho_b0,
    a1=TFD.AugTikho_a1,
    b1=TFD.AugTikho_b1,
    d=TFD.AugTikho_d,
    atol=TFD.AugTkLsmrAtol,
    btol=TFD.AugTkLsmrBtol,
    conlim=TFD.AugTkLsmrConlim,
    maxiter=TFD.AugTkLsmrMaxiter,
    NormL=None,
    M=None,
    Verb=False,
    ConvReg=True,
    FixedNb=True,
):       # Install scikit !
    """ Linear algorithm for Phillips-Tikhonov regularisation, called "Augmented Tikhonov", sparese matrix version
    see TFI.InvLin_AugTikho_V1.__doc__ for details
    """

    a0bis = a0-1.+Nbf/2. if not FixedNb else a0-1.+1200./2.
    a1bis = a1-1.+NMes/2.

    Conv = 0.                                                   # Initialise convergence variable
    Nit = 0                                                     # Initialise number of iterations
    chi2N, R = [], []                                           # Initialise residu list and regularisation term list
    lmu = [mu0]                                                 # Initialise regularisation parame
    if NormL is None:
        NormL = sol0.dot(LL.dot(sol0))
    LL = LL / NormL                                               # Scale the regularity operator
    if M is None:
        M = scpsplin.inv((TT + lmu[-1]*LL).tocsc())
    if Verb:
        print "        Initial guess : N*Chi2N + mu*RN = ", NMes,'*',np.sum((Tm.dot(sol0)-b)**2)/NMes,'+',lmu[-1],'*',sol0.dot(LL.dot(sol0)), ' = ', np.sum((Tm.dot(sol0)-b)**2)+lmu[-1]*sol0.dot(LL.dot(sol0))
        while  Nit<2 or Conv>ConvCrit:                              # Continue until convergence criterion is fulfilled, and do at least 2 iterations
            #sol, itconv = scpsplin.minres(TT + mu*LL, Tb, x0=sol0, shift=0.0, tol=1e-10, maxiter=None, M=M, show=False, check=False)   # 2
            sol, itconv = scpsplin.cg(TT + lmu[-1]*LL, Tb, x0=sol0, tol=1e-08, maxiter=None, M=M)    # 1
            #sol = scpsplin.spsolve(TT + lmu[-1]*LL, Tb, permc_spec=None, use_umfpack=False) # 3
            #sol, isstop, itn, normr, normar, norma, conda, normx = scpsplin.lsmr(TT + lmu[-1]*LL, Tb, atol=atol, btol=btol, conlim=conlim, maxiter=maxiter, show=False) # Install scikits and use CHOLMOD fast cholesky factorization !!!

            Res = np.sum((Tm.dot(sol)-b)**2)                        # Compute residu**2
            chi2N.append(Res/NMes)                                  # Record normalised residu history
            R.append(sol.dot(LL.dot(sol)))                          # Compute and record regularity term

            lamb = a0bis/(0.5*R[Nit]+b0)                            # Update reg. param. estimate
            tau = a1bis/(0.5*Res+b1)                                # Update noise coef. estimate
            lmu.append((lamb/tau) * (2*a1bis/Res)**d)               # Update regularisation parameter taking into account rescaling with noise estimate
            if ConvReg:
                Conv = np.abs(lmu[-1]-lmu[-2])/lmu[-1]
            else:
                Conv = np.sqrt(np.sum((sol-sol0)**2/np.max([sol**2,0.001*np.max(sol**2)*np.ones((Nbf,))],axis=0))/Nbf)        # Compute convergence variable
            print "        Nit = ",str(Nit),"  N*Chi2N + mu*R = ", NMes,'*',chi2N[Nit],'+',lmu[-1],'*',R[Nit], ' = ', Res+lmu[-1]*R[Nit], '  tau = ',tau, '  Conv = ',Conv
            #print '                     isstop,itn=',isstop,itn, '  normRArAX=',normr, normar, norma,normx, 'condA=',conda
            sol0[:] = sol[:]                                           # Update reference solution
            Nit += 1                                                # Update number of iterations
    else:
        while  Nit<2 or Conv>ConvCrit:                              # Continue until convergence criterion is fulfilled, and do at least 2 iterations
            #sol, itconv = scpsplin.minres(TT + lmu[-1]*LL, Tb, x0=sol0, shift=0.0, tol=1e-10, maxiter=None, M=M, show=False, check=False)   # 2
            sol, itconv = scpsplin.cg(TT + lmu[-1]*LL, Tb, x0=sol0, tol=1e-08, maxiter=None, M=M)    # 1
            #sol = scpsplin.spsolve(TT + lmu[-1]*LL, Tb, permc_spec=None, use_umfpack=False) # 3
            #sol, isstop, itn, normr, normar, norma, conda, normx = scpsplin.lsmr(TT + lmu[-1]*LL, Tb, atol=atol, btol=btol, conlim=conlim, maxiter=maxiter, show=False)

            Res = np.sum((Tm.dot(sol)-b)**2)                        # Compute residu**2
            chi2N.append(Res/NMes)                                  # Record normalised residu history
            R.append(sol.dot(LL.dot(sol)))                          # Compute and record regularity term

            lamb = a0bis/(0.5*R[Nit]+b0)                            # Update reg. param. estimate
            tau = a1bis/(0.5*Res+b1)                                # Update noise coef. estimate
            lmu.append((lamb/tau) * (2*a1bis/Res)**d)               # Update regularisation parameter taking into account rescaling with noise estimate
            if ConvReg:
                Conv = np.abs(lmu[-1]-lmu[-2])/lmu[-1]
            else:
                Conv = np.sqrt(np.sum((sol-sol0)**2/np.max([sol**2,0.001*np.max(sol**2)*np.ones((Nbf,))],axis=0))/Nbf)        # Compute convergence variable
            sol0[:] = sol[:]                                        # Update reference solution
            Nit += 1

    return sol, lmu[-1], chi2N[-1], R[-1], Nit, [tau, lamb]


def InvLinQuad_AugTikho_V1_Sparse(NMes, Nbf, Tm, TT, LL, Tb, b, sol0, mu0=TFD.mu0, ConvCrit=TFD.ConvCrit, a0=TFD.AugTikho_a0, b0=TFD.AugTikho_b0, a1=TFD.AugTikho_a1, b1=TFD.AugTikho_b1, d=TFD.AugTikho_d, atol=TFD.AugTkLsmrAtol, btol=TFD.AugTkLsmrBtol, conlim=TFD.AugTkLsmrConlim, maxiter=TFD.AugTkLsmrMaxiter, NormL=None, M=None, Verb=False, ConvReg=True, FixedNb=True):       # Install scikit !
    """ Quadratic algorithm for Phillips-Tikhonov regularisation, alternative to the linear version with positivity constraint
    see TFI.InvLin_AugTikho_V1.__doc__ for details
    """

    a0bis = a0-1.+Nbf/2. if not FixedNb else a0-1.+1200./2.
    a1bis = a1-1.+NMes/2.

    Conv = 0.                                                   # Initialise convergence variable
    Nit = 0                                                     # Initialise number of iterations
    chi2N, R = [], []                                           # Initialise residu list and regularisation term list
    lmu = [mu0]                                                 # Initialise regularisation parame
    if NormL is None:
        NormL = sol0.dot(LL.dot(sol0))
    LL = LL/NormL                                               # Scale the regularity operator
    if M is None:
        M = scpsplin.inv((TT + lmu[-1]*LL).tocsc())

    Bds = tuple([(-0.2,None) for ii in range(0,sol0.size)])
    F = lambda X, mu=lmu[-1]: np.sum((Tm.dot(X)-b)**2) + mu*X.dot(LL.dot(X))
    F_g = lambda X, mu=lmu[-1]: 2.*(TT + mu*LL).dot(X) - 2.*Tb
    F_h = lambda X, mu=lmu[-1]: 2.*(TT + mu*LL)

    Method = 'L-BFGS-B'
    if Method == 'L-BFGS-B':
        options={'ftol':ConvCrit/100., 'disp':False}
    elif Method == 'TNC':
        options={'ftol':ConvCrit/100., 'disp':False, 'minfev':1, 'rescale':1., 'maxCGit':0, 'offset':0.}  # Not working....
    elif Method == 'SLSQP':
        options={'ftol':ConvCrit/100., 'disp':False}      # Very slow... (maybe not updated ?)

    if Verb:
        print "        Initial guess : N*Chi2N + mu*RN = ", NMes,'*',np.sum((Tm.dot(sol0)-b)**2)/NMes,'+',lmu[-1],'*',sol0.dot(LL.dot(sol0)), ' = ', np.sum((Tm.dot(sol0)-b)**2)+lmu[-1]*sol0.dot(LL.dot(sol0))
    while  Nit<2 or Conv>ConvCrit:                              # Continue until convergence criterion is fulfilled, and do at least 2 iterations
        Out = scpop.minimize(F, sol0, args=([lmu[-1]]), jac=F_g, hess=F_h, method=Method, bounds=Bds, options=options)       # Minimisation de la fonction (method quadratique)
        sol = Out.x
        Res = np.sum((Tm.dot(sol)-b)**2)                        # Compute residu**2
        chi2N.append(Res/NMes)                                  # Record normalised residu history
        R.append(sol.dot(LL.dot(sol)))                          # Compute and record regularity term

        lamb = a0bis/(0.5*R[Nit]+b0)                            # Update reg. param. estimate
        tau = a1bis/(0.5*Res+b1)                                # Update noise coef. estimate
        lmu.append((lamb/tau) * (2*a1bis/Res)**d)               # Update regularisation parameter taking into account rescaling with noise estimate
        if ConvReg:
            Conv = np.abs(lmu[-1]-lmu[-2])/lmu[-1]
        else:
            Conv = np.sqrt(np.sum((sol-sol0)**2/np.max([sol**2,0.001*np.max(sol**2)*np.ones((Nbf,))],axis=0))/Nbf)        # Compute convergence variable
        if Verb:
            print "        Nit = ",str(Nit),"  N*Chi2N + mu*R = ", NMes,'*',chi2N[Nit],'+',lmu[-1],'*',R[Nit], ' = ', Res+lmu[-1]*R[Nit], '  tau = ',tau, '  Conv = ',Conv
        sol0[:] = sol[:]                                           # Update reference solution
        Nit += 1                                                # Update number of iterations

    return sol, lmu[-1], chi2N[-1], R[-1], Nit, [tau, lamb]



def estimate_jacob_scalar_Spec(ff, x0, ratio=0.05):
    nV = x0.size
    jac = np.zeros((nV,))
    dx = np.zeros((nV,))
    for ii in range(0,nV):
        dx[ii] = ratio*x0[ii]
        jac[ii] = (ff(x0+dx)[-1]-ff(x0)[-1])/dx[ii]
        dx[ii] = 0.
    return jac



def InvQuad_AugTikho_V1(NMes, Nbf, Tm, TT, LL, Tb, b, sol0, mu0=TFD.mu0, ConvCrit=TFD.ConvCrit, a0=TFD.AugTikho_a0, b0=TFD.AugTikho_b0, a1=TFD.AugTikho_a1, b1=TFD.AugTikho_b1, d=TFD.AugTikho_d, NormL=None, Verb=False, ConvReg=True, ratiojac=0.05, Method='Newton-CG', FixedNb=True):
    """ Non-linear (quadratic) algorithm for Phillips-Tikhonov regularisation, inspired from "Augmented Tikhonov"
    see TFI.InvLin_AugTikho_V1.__doc__ for details
    """

    a0bis = a0-1.+Nbf/2. if not FixedNb else a0-1.+1200./2.
    a1bis = a1-1.+NMes/2.

    Conv = 0.                                                   # Initialise convergence variable
    Nit = 0                                                     # Initialise number of iterations
    chi2N, R = [], []                                           # Initialise residu list and regularisation term list
    mu = mu0                                                    # Initialise regularisation parameter
    if NormL is None:
        NormL = sol0.dot(LL.dot(sol0))
    LL = LL/NormL                                               # Scale the regularity operator
    if Verb:
        print "        Init. guess : N*Chi2N + mu*RN = ", NMes,'*',np.sum((Tm.dot(sol0)-b)**2)/NMes,'+',mu,'*',sol0.dot(LL.dot(sol0)), ' = ', np.sum((Tm.dot(sol0)-b)**2)+mu*sol0.dot(LL.dot(sol0))
        while  Nit<2 or Conv>ConvCrit:                              # Continue until convergence criterion is fulfilled, and do at least 2 iterations
            F = lambda X, mu=mu: np.sum((Tm.dot(X)-b)**2) + mu*X.dot(LL.dot(X))
            Out = scpop.minimize(F, sol0, tol=None, method=Method, options={'maxiter':100, 'disp':True})       # Minimisation de la fonction (method quadratique)
            sol = Out.x
            Res = np.sum((Tm.dot(sol)-b)**2)                        # Compute residu**2
            chi2N.append(Res/NMes)                                  # Record normalised residu history
            R.append(sol.dot(LL.dot(sol)))                          # Compute and record regularity term

            lamb = a0bis/(0.5*R[Nit]+b0)                            # Update reg. param. estimate
            tau = a1bis/(0.5*Res+b1)                                # Update noise coef. estimate
            mu = (lamb/tau) * (2*a1bis/Res)**d                      # Update regularisation parameter taking into account rescaling with noise estimate
            if ConvReg:
                Conv = np.abs(lmu[-1]-lmu[-2])/lmu[-1]
            else:
                Conv = np.sqrt(np.sum((sol-sol0)**2/np.max([sol**2,0.001*np.max(sol**2)*np.ones((Nbf,))],axis=0))/Nbf)        # Compute convergence variable
            print "        Nit = ",str(Nit),"   N*Chi2N + mu*R = ", NMes,'*',chi2N[Nit],'+',mu,'*',R[Nit], ' = ', Res+mu*R[Nit], '    tau = ',tau, '    Conv = ',Conv
            sol0[:] = sol[:]                                        # Update reference solution
            Nit += 1                                                # Update number of iterations
    else:
        xx = np.zeros((Nbf,))
        F = lambda X : np.sum((Tm.dot(X[:-1])-b)**2) + X[-1]*X[:-1].dot(LL.dot(X[:-1]))
        Out = scpop.minimize(F, np.append(sol0,mu), tol=None, options={'maxiter':100, 'disp':False})
        sol = Out.x[:-1]
        Res = np.sum((Tm.dot(sol)-b)**2)
        chi2N.append(np.sum((Tm.dot(sol)-b)**2)/NMes)
        R.append(sol.dot(LL.dot(sol)))
        lamb = a0bis/(0.5*R[-1]+b0)                            # Update reg. param. estimate
        tau = a1bis/(0.5*Res+b1)                                # Update noise coef. estimate
        mu = (lamb/tau) * (2*a1bis/Res)**d
        print "    Done :", chi2N

    return sol, mu, chi2N[-1], R[-1], Nit, [tau, lamb]



def InvLin_DisPrinc_V1_Sparse(NMes, Nbf, Tm, TT, LL, Tb, b, sol0, mu0=TFD.mu0, ConvCrit=TFD.ConvCrit, chi2Tol=TFD.chi2Tol, chi2Obj=TFD.chi2Obj, M=None, Verb=False, NormL=None):
    """ Discrepancy principle """

    Ntry = 0
    LogTol = np.log(chi2Obj)

    chi2N = []                                                  # Initialise residu list and regularisation term list
    lmu = [mu0]                                                 # Initialise regularisation parame
    if NormL is None:
        NormL = sol0.dot(LL.dot(sol0))
    LL = LL/NormL                                               # Scale the regularity operator
    if M is None:
        M = scpsplin.inv((TT + lmu[-1]*LL).tocsc())
    #try:
    if Verb:
        print "        Initial guess : N*Chi2N + mu*RN = ", NMes,'*',np.sum((Tm.dot(sol0)-b)**2)/NMes,'+',lmu[-1],'*',sol0.dot(LL.dot(sol0)), ' = ', np.sum((Tm.dot(sol0)-b)**2)+lmu[-1]*sol0.dot(LL.dot(sol0))
        while Ntry==0 or np.abs(chi2N[-1]-chi2Obj)>chi2Tol:
            sol, itconv = scpsplin.cg(TT + lmu[-1]*LL, Tb, x0=sol0, tol=1e-08, maxiter=None, M=M)    # 1
            Ntry += 1
            chi2N.append(np.sum((Tm.dot(sol)-b)**2)/NMes)
            if Ntry==1:
                if chi2N[-1]>=chi2Obj+chi2Tol:
                    lmu.append(lmu[-1]/50.)
                elif chi2N[-1]<=chi2Obj-chi2Tol:
                    lmu.append(lmu[-1]*50.)
                else:
                    lmu.append(lmu[-1])
            elif all([chi>=chi2Obj+chi2Tol for chi in chi2N]) or all([chi<=chi2Obj-chi2Tol for chi in chi2N]):
                if chi2N[-1]>=chi2Obj+chi2Tol:
                    lmu.append(lmu[-1]/50.)
                else:
                    lmu.append(lmu[-1]*50.)
            else:
                indsort = np.argsort(chi2N)
                lmu.append(np.exp(np.interp(LogTol, np.log(np.asarray(chi2N)[indsort]), np.log(np.asarray(lmu)[indsort]))))
            print "        InvLin_DisPrinc_V1 : try ",Ntry-1, "mu = ",lmu[-2],"chi2N = ",chi2N[-1], "New mu = ", lmu[-1]
    else:
        while Ntry==0 or np.abs(chi2N[-1]-chi2Obj)>chi2Tol:
            sol, itconv = scpsplin.cg(TT + lmu[-1]*LL, Tb, x0=sol0, tol=1e-08, maxiter=None, M=M)
            Ntry += 1
            chi2N.append(np.sum((Tm.dot(sol)-b)**2)/NMes)
            if Ntry==1:
                if chi2N[-1]>=chi2Obj+chi2Tol:
                    lmu.append(lmu[-1]/50.)
                elif chi2N[-1]<=chi2Obj-chi2Tol:
                    lmu.append(lmu[-1]*50.)
                else:
                    lmu.append(lmu[-1])
            elif all([chi>=chi2Obj+chi2Tol for chi in chi2N]) or all([chi<=chi2Obj-chi2Tol for chi in chi2N]):
                if chi2N[-1]>=chi2Obj+chi2Tol:
                    lmu.append(mu[-1]/50.)
                else:
                    lmu.append(mu[-1]*50.)
            else:
                indsort = np.argsort(chi2N)
                #mu.append(np.exp(np.log(mu[-1]) - np.log(chi2N[-1])*(np.log(mu[-1])-np.log(mu[-2]))/(np.log(chi2N[-1])-np.log(chi2N[-2])))
                lmu.append(np.exp(np.interp(LogTol, np.log(np.asarray(chi2N)[indsort]),np.log(np.asarray(lmu[:-1])[indsort]))))
    """
    except:
        print np.asarray(lmu[:-1]), np.asarray(chi2N), np.arange(0,len(chi2N))
        plt.figure()
        plt.scatter(np.asarray(lmu[:-1]),np.asarray(chi2N), c=np.arange(0,len(chi2N)),edgecolors='none')
        plt.gca().set_xscale('log'), plt.gca().set_yscale('log')
        plt.gca().set_xlim(r"$\log_10(\mu)$"), plt.gca().set_ylim(r"$\log_10(\chi^2_N)$")
        plt.figure()
        plt.plot(np.asarray(LL),'b-')
        plt.figure()
        plt.plot(TT,'ko'), plt.plot(TT+lmu[-1]*LL,'r--')
        plt.show()
    """

    return sol, lmu[-1], chi2N[-1], sol.dot(LL.dot(sol)), Ntry, []






