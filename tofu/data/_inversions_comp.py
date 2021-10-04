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
    sparse=None,
):

    # -------------
    # check inputs

    (
        keymat, keybs, keym, data, sigma,
        isotropic, sparse, matrix,
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
