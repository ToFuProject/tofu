
# Common
import numpy as np
import scipy.optimize as scpopt
import scipy.sparse as scpsparse
from scipy.interpolate import BSpline

# Temporary for debugging
import matplotlib.pyplot as plt


_JAC = 'dense'


###########################################################
###########################################################
#
#           Main function for fit1d
#
###########################################################
###########################################################


def multigausfit1d_from_dlines_funccostjac(
    lamb,
    indx=None,
    dinput=None,
    dind=None,
    scales=None,
    jac=None,
):

    if jac is None:
        jac = _JAC

    ibckax = dind['bck_amp']['x']
    ibckrx = dind['bck_rate']['x']
    nbck = 1    # ibckax.size + ibckrx.size
    iax = dind['amp']['x']
    iwx = dind['width']['x']
    ishx = dind['shift']['x']
    idratiox, idshx = None, None
    if dinput['double'] is not False:
        c0 = dinput['double'] is True
        if c0 or dinput['double'].get('dratio') is None:
            idratiox = dind['dratio']['x']
        if c0 or dinput['double'].get('dshift') is None:
            idshx = dind['dshift']['x']

    ial = dind['amp']['lines']
    iwl = dind['width']['lines']
    ishl = dind['shift']['lines']

    iaj = dind['amp']['jac']
    iwj = dind['width']['jac']
    ishj = dind['shift']['jac']

    coefsal = dinput['amp']['coefs']
    coefswl = dinput['width']['coefs']
    coefssl = dinput['shift']['coefs']

    offsetal = dinput['amp']['offset']
    offsetwl = dinput['width']['offset']
    offsetsl = dinput['shift']['offset']

    lambrel = lamb - np.nanmin(lamb)
    lambnorm = lamb[..., None]/dinput['lines'][None, ...]

    xscale = np.full((dind['sizex'],), np.nan)
    if indx is None:
        indx = np.ones((dind['sizex'],), dtype=bool)

    # # bsplines-specific
    # lambnormcost = lamb[indok].ravel()[:, None] / dinput['lines'][None, :]
    # libs = np.array([((phicost>=km[ii]) & (phicost<=km[ii+kpb-1]))
        # for ii in range(nbs)])
    # lbs = [BSpline.basis_element(
        # km[ii:ii+kpb],
        # extrapolate=False)(phicost[libs[ii]])[:, None]
        # for ii in range(nbs)]
    # BS = BSpline(km, np.ones(ial.shape, dtype=float), dinput['deg'],
        # extrapolate=False, axis=0)

    # lcond = [np.any(np.isnan(bs)) for bs in lbs]
    # if np.any(lcond):
        # msg = ("Some nan have been detected in the jacobian!\n"
        # + "\t- lbs[{}]".format(lcond.index(True)))
        # raise Exception(msg)

    # func_details returns result in same shape as input
    def func_detail(
        x,
        xscale=xscale,
        indx=indx,
        lambrel=lambrel,
        lambnorm=lambnorm,
        ibckax=ibckax,
        ibckrx=ibckrx,
        ial=ial,
        iwl=iwl,
        ishl=ishl,
        idratiox=idratiox,
        idshx=idshx,
        nlines=dinput['nlines'],
        nbck=nbck,
        coefsal=coefsal[None, :],
        coefswl=coefswl[None, :],
        coefssl=coefssl[None, :],
        offsetal=offsetal[None, :],
        offsetwl=offsetwl[None, :],
        offsetsl=offsetsl[None, :],
        double=dinput['double'],
        scales=None,
        indok=None,
        const=None,
    ):
        if indok is None:
            indok = np.ones(lamb.shape, dtype=bool)
        shape = tuple(np.r_[indok.sum(), nbck+nlines])
        y = np.full(shape, np.nan)
        xscale[indx] = x*scales[indx]
        xscale[~indx] = const

        # Prepare
        amp = xscale[ial]*coefsal + offsetal
        wi2 = xscale[iwl]*coefswl + offsetwl
        shift = xscale[ishl]*coefssl + offsetsl
        exp = np.exp(-(lambnorm[indok, :] - (1 + shift))**2 / (2*wi2))

        if double is not False:
            # coefssl are line-specific, they do not affect dshift
            if double is True:
                dratio = xscale[idratiox]
                dshift = shift + xscale[idshx]  # *coefssl
            else:
                dratio = double.get('dratio', xscale[idratiox])
                dshift = shift + double.get('dshift', xscale[idshx])
            expd = np.exp(-(lambnorm[indok, :] - (1 + dshift))**2 / (2*wi2))

        # compute y
        y[:, :nbck] = (
            xscale[ibckax]
            * np.exp(xscale[ibckrx]*lambrel[indok])
        )[:, None]
        y[:, nbck:] = amp * exp
        if double is not False:
            y[:, nbck:] += amp * dratio * expd
        return y

    # cost and jacob return flattened results (for least_squares())
    def cost(
        x,
        xscale=xscale,
        indx=indx,
        lambrel=lambrel,
        lambnorm=lambnorm,
        ibckax=ibckax,
        ibckrx=ibckrx,
        ial=ial,
        iwl=iwl,
        ishl=ishl,
        idratiox=idratiox,
        idshx=idshx,
        scales=scales,
        coefsal=coefsal[None, :],
        coefswl=coefswl[None, :],
        coefssl=coefssl[None, :],
        offsetal=offsetal[None, :],
        offsetwl=offsetwl[None, :],
        offsetsl=offsetsl[None, :],
        double=dinput['double'],
        indok=None,
        const=None,
        data=0.,
    ):
        if indok is None:
            indok = np.ones(lamb.shape, dtype=bool)

        # xscale = x*scales
        xscale[indx] = x*scales[indx]
        xscale[~indx] = const

        # make sure iwl is 2D to get all lines at once
        amp = xscale[ial] * coefsal + offsetal
        inv_2wi2 = 1./(2.*(xscale[iwl] * coefswl + offsetwl))
        shift = xscale[ishl] * coefssl + offsetsl
        y = (
            np.nansum(
                amp * np.exp(-(lambnorm[indok, :]-(1 + shift))**2 * inv_2wi2),
                axis=1,
            )
            + xscale[ibckax]*np.exp(xscale[ibckrx]*lambrel[indok])
        )
        if double is not False:
            if double is True:
                dratio = xscale[idratiox]
                # scales[ishl] or scales[idshx] ? coefssl ? +> no
                dshift = shift + xscale[idshx]
            else:
                dratio = double.get('dratio', xscale[idratiox])
                dshift = shift + double.get('dshift', xscale[idshx])

            y += np.nansum((amp * dratio
                            * np.exp(-(lambnorm[indok, :] - (1 + dshift))**2
                                     * inv_2wi2)), axis=1)

        if isinstance(data, np.ndarray):
            return y - data[indok]
        else:
            return y - data

    # Prepare jac
    if jac in ['dense', 'sparse']:
        # Define a callable jac returning (nlamb, sizex) matrix of partial
        # derivatives of np.sum(func_details(scaled), axis=0)

        # Pre-instanciate jacobian
        jac0 = np.zeros((lamb.size, dind['sizex']), dtype=float)
        # jac0 = np.zeros((lamb.size, indx.sum()), dtype=float)
        indxn = indx.nonzero()[0]

        def jacob(
            x,
            xscale=xscale,
            indx=indx,
            indxn=indxn,
            lambnorm=lambnorm,
            ibckax=ibckax,
            ibckrx=ibckrx,
            iax=iax, iaj=iaj, ial=ial,
            iwx=iwx, iwj=iwj, iwl=iwl,
            ishx=ishx, ishj=ishj, ishl=ishl,
            idratiox=idratiox, idshx=idshx,
            coefsal=coefsal[None, :],
            coefswl=coefswl[None, :],
            coefssl=coefssl[None, :],
            offsetal=offsetal[None, :],
            offsetwl=offsetwl[None, :],
            offsetsl=offsetsl[None, :],
            double=dinput['double'],
            scales=None, indok=None, data=None,
            jac0=jac0,
            const=None,
        ):
            """ Basic docstr """
            if indok is None:
                indok = np.ones(lamb.shape, dtype=bool)
            xscale[indx] = x*scales[indx]
            xscale[~indx] = const

            # Intermediates
            amp = xscale[ial] * coefsal + offsetal
            wi2 = xscale[iwl] * coefswl + offsetwl
            inv_wi2 = 1./wi2
            shift = xscale[ishl] * coefssl + offsetsl
            beta = (lambnorm[indok, :] - (1 + shift)) * inv_wi2 / 2.
            alpha = -beta**2 * (2*wi2)
            exp = np.exp(alpha)

            # Background
            jac0[indok, ibckax[0]] = (
                scales[ibckax[0]]
                * np.exp(xscale[ibckrx]*lambrel[indok])
            )
            jac0[indok, ibckrx[0]] = (
                xscale[ibckax[0]] * scales[ibckrx[0]] * lambrel[indok]
                * np.exp(xscale[ibckrx]*lambrel[indok])
            )

            # amp (shape: nphi/lamb, namp[jj])
            # for jj in range(len(iaj)):
            for jj, aa in enumerate(iaj):
                jac0[indok, iax[jj]] = np.sum(
                    exp[:, aa] * coefsal[:, aa],
                    axis=1) * scales[iax[jj]]

            # width2
            for jj in range(len(iwj)):
                jac0[indok, iwx[jj]] = np.sum(
                    (-alpha[:, iwj[jj]] * amp[:, iwj[jj]]
                     * exp[:, iwj[jj]] * coefswl[:, iwj[jj]]
                     * inv_wi2[:, iwj[jj]]), axis=1) * scales[iwx[jj]]

            # shift
            for jj in range(len(ishj)):
                jac0[indok, ishx[jj]] = np.sum(
                    (amp[:, ishj[jj]] * 2. * beta[:, ishj[jj]])
                    * exp[:, ishj[jj]] * coefssl[:, ishj[jj]],
                    axis=1) * scales[ishx[jj]]

            # double
            if double is not False:

                if double is True:
                    dratio = xscale[idratiox]
                    # coefssl are line-specific, they do not affect dshift
                    dshift = shift + xscale[idshx]
                else:
                    dratio = double.get('dratio', xscale[idratiox])
                    dshift = shift + double.get('dshift', xscale[idshx])
                # ampd not defined to save memory => *dratio instead
                # ampd = amp*dratio
                betad = (lambnorm[indok, :] - (1 + dshift)) * inv_wi2 / 2.
                alphad = -betad**2 * (2*wi2)
                expd = np.exp(alphad)

                # amp
                for jj in range(len(iaj)):
                    jac0[indok, iax[jj]] += dratio*np.sum(
                        expd[:, iaj[jj]] * coefsal[:, iaj[jj]],
                        axis=1) * scales[iax[jj]]

                # width2
                for jj in range(len(iwj)):
                    jac0[indok, iwx[jj]] += np.sum(
                        (
                            -alphad[:, iwj[jj]] * amp[:, iwj[jj]]
                            * expd[:, iwj[jj]] * coefswl[:, iwj[jj]]
                            * inv_wi2[:, iwj[jj]]
                        ),
                        axis=1,
                    ) * scales[iwx[jj]] * dratio

                # shift
                for jj in range(len(ishj)):
                    jac0[indok, ishx[jj]] += np.sum(
                        (amp[:, ishj[jj]] * 2.*betad[:, ishj[jj]]
                         * expd[:, ishj[jj]] * coefssl[:, ishj[jj]]),
                        axis=1) * scales[ishx[jj]] * dratio

                # dratio
                if double is True or double.get('dratio') is None:
                    jac0[indok, idratiox] = (
                        scales[idratiox] * np.sum(amp * expd, axis=1)
                    )

                # dshift
                if double is True or double.get('dshift') is None:
                    jac0[indok, idshx] = dratio * np.sum(
                        amp * 2.*betad*scales[idshx] * expd, axis=1)

            return jac0[indok, :][:, indx]

    elif jac == 'sparse':
        msg = "Sparse jacobian is pointless for 1d spectrum fitting"
        raise Exception(msg)

    if jac not in ['dense', 'sparse', 'LinearSparseOperator']:
        if jac not in ['2-point', '3-point']:
            msg = ("jac should be in "
                   + "['dense', 'sparse', 'LinearsparseOp', "
                   + "'2-point', '3-point']\n"
                   + "\t- provided: {}".format(jac))
            raise Exception(msg)
        jacob = jac

    return func_detail, cost, jacob

###########################################################
###########################################################
#
#           Main function for fit2d
#
###########################################################
###########################################################


def multigausfit2d_from_dlines_funccostjac(lamb, phi,
                                           dinput=None,
                                           binning=None,
                                           dind=None,
                                           scales=None,
                                           indok=None,
                                           jac=None):

    if jac is None:
        jac = _JAC

    ibckx = dind['bck']['x']
    iax = dind['amp']['x']
    iwx = dind['width']['x']
    ishx = dind['shift']['x']
    idratiox, idshx = None, None
    if dinput['double'] is not False:
        c0 = dinput['double'] is True
        if c0 or dinput['double'].get('dratio') is None:
            idratiox = dind['dratio']['x']
        if c0 or dinput['double'].get('dshift') is None:
            idshx = dind['dshift']['x']

    ial = dind['amp']['lines']
    iwl = dind['width']['lines']
    ishl = dind['shift']['lines']

    iaj = dind['amp']['jac']
    iwj = dind['width']['jac']
    ishj = dind['shift']['jac']

    coefsal = dinput['amp']['coefs']
    coefswl = dinput['width']['coefs']
    coefssl = dinput['shift']['coefs']

    offsetal = dinput['amp']['offset']
    offsetwl = dinput['width']['offset']
    offsetsl = dinput['shift']['offset']

    # If no binning => no fast indexing on lamb / phi (e.g.: for sparse)
    km = dinput['knots_mult']
    kpb = dinput['nknotsperbs']
    nbs = dinput['nbs']

    # bsplines-specific
    phicost = phi[indok].ravel()
    lambnormcost = lamb[indok].ravel()[:, None] / dinput['lines'][None, :]
    libs = np.array([((phicost >= km[ii]) & (phicost <= km[ii+kpb-1]))
                     for ii in range(nbs)])
    lbs = [BSpline.basis_element(
        km[ii:ii+kpb],
        extrapolate=False)(phicost[libs[ii]])[:, None]
           for ii in range(nbs)]
    BS = BSpline(km, np.ones(ial.shape, dtype=float), dinput['deg'],
                 extrapolate=False, axis=0)

    lcond = [np.any(np.isnan(bs)) for bs in lbs]
    if np.any(lcond):
        msg = ("Some nan have been detected in the jacobian!\n"
               + "\t- lbs[{}]".format(lcond.index(True)))
        raise Exception(msg)

    # func_details returns result in same shape as input
    def func_detail(x, phi=phi,
                    lambnorm=lamb[..., None]/dinput['lines'][None, ...],
                    ibckx=ibckx,
                    ial=ial,
                    iwl=iwl,
                    ishl=ishl,
                    idratiox=idratiox,
                    idshx=idshx,
                    nlines=dinput['nlines'],
                    km=km, kpb=kpb, nbs=nbs,
                    deg=dinput['deg'],
                    BS=BS,
                    nbck=int(ibckx.size/dinput['nbs']),
                    coefsal=coefsal[None, :],
                    coefswl=coefswl[None, :],
                    coefssl=coefssl[None, :],
                    double=dinput['double'],
                    scales=None,
                    indok=None,
                    indbs=None):
        if indok is None:
            indok = np.ones(phi.shape, dtype=bool)
        shape = tuple(np.r_[phi.shape, nbck+nlines, nbs])
        y = np.full(shape, np.nan)
        xscale = x*scales
        # make sure iwl is 2D to get all lines at once
        BS.c = xscale[iwl] * coefswl
        wi2 = BS(phi)
        BS.c = xscale[ishl] * coefssl
        shift = BS(phi)
        exp = np.exp(-(lambnorm - (1 + shift))**2 / (2*wi2))

        if double is not False:
            # coefssl are line-specific, they do not affect dshift
            if double is True:
                dratio = xscale[idratiox]
                dshift = shift + xscale[idshx]  # *coefssl
            else:
                dratio = double.get('dratio', xscale[idratiox])
                dshift = shift + double.get('dshift', xscale[idshx])
            expd = np.exp(-(lambnorm - (1 + dshift))**2 / (2*wi2))

        # Loop on individual bsplines for amp
        for ii in range(nbs):
            bs = BSpline.basis_element(km[ii:ii+kpb],
                                       extrapolate=False)(phi)

            indbs = ~np.isnan(bs)
            bs = bs[indbs]
            y[indbs, 0, ii] = xscale[ibckx[ii]]*bs
            for jj in range(nlines):
                amp = bs * xscale[ial[ii, jj]] * coefsal[0, jj]
                y[indbs, nbck+jj, ii] = amp * exp[indbs, jj]
                if double is not False:
                    y[indbs, nbck+jj, ii] += (amp * dratio * expd[indbs, jj])
        return y

    # cost and jacob return flattened results (for least_squares())
    def cost(x, phi=phicost,
             lambnorm=lambnormcost,
             ibckx=ibckx,
             ial=ial,
             iwl=iwl,
             ishl=ishl,
             idratiox=idratiox,
             idshx=idshx,
             km=km, kpb=kpb,
             deg=dinput['deg'],
             scales=scales,
             coefsal=coefsal[None, :],
             coefswl=coefswl[None, :],
             coefssl=coefssl[None, :],
             double=dinput['double'],
             indok_var=None, ind_bs=None, data=0.):

        xscale = x*scales

        # Background
        y = BSpline(km, xscale[ibckx], deg,
                    extrapolate=False, axis=0)(phi)

        # make sure iwl is 2D to get all lines at once
        amp = BSpline(km, xscale[ial] * coefsal, deg,
                      extrapolate=False, axis=0)(phi)
        wi2 = BSpline(km, xscale[iwl] * coefswl, deg,
                      extrapolate=False, axis=0)(phi)
        csh = xscale[ishl] * coefssl
        y += np.nansum((amp
                        * np.exp(-(lambnormcost
                                   - (1 + BSpline(km, csh, deg,
                                                  extrapolate=False,
                                                  axis=0)(phi)))**2
                                 / (2*wi2))),
                       axis=1)
        import pdb; pdb.set_trace()    # DB
        if double is not False:
            if double is True:
                dratio = xscale[idratiox]
                # scales[ishl] or scales[idshx] ? coefssl ? +> no
                csh = csh + xscale[idshx]
            else:
                dratio = double.get('dratio', xscale[idratiox])
                csh = csh + double.get('dshift', xscale[idshx])

            y += np.nansum((amp * dratio
                            * np.exp(-(lambnormcost
                                       - (1 + BSpline(
                                           km, csh, deg,
                                           extrapolate=False,
                                           axis=0)(phi)))**2
                                     / (2*wi2))),
                           axis=1)
        return y - data

    # Prepare jac
    if jac in ['dense', 'sparse']:
        # Define a callable jac returning (nlamb, sizex) matrix of partial
        # derivatives of np.sum(func_details(scaled), axis=0)

        # Pre-instanciate jacobian
        jac0 = np.zeros((phicost.size, dind['sizex']), dtype=float)

        def jacob(x, lambnorm=lambnormcost, phi=phicost,
                  ibckx=ibckx,
                  iax=iax, iaj=iaj, ial=ial,
                  iwx=iwx, iwj=iwj, iwl=iwl,
                  ishx=ishx, ishj=ishj, ishl=ishl,
                  idratiox=idratiox, idshx=idshx,
                  km=km, kpb=kpb, nbs=nbs,
                  coefsal=coefsal[None, :],
                  coefswl=coefswl[None, :],
                  coefssl=coefssl[None, :],
                  double=dinput['double'],
                  scales=None, indok_var=None, data=None,
                  jac0=jac0, lbs=lbs, libs=libs, BS=BS):
            """ Basic docstr """
            xscale = x*scales

            # Intermediates

            # Loop on bs
            for ii in range(nbs):
                bs = lbs[ii]

                # Intermediates
                BS.c = xscale[ial] * coefsal
                amp = BS(phi[libs[ii]])
                BS.c = xscale[iwl] * coefswl
                wi2 = BS(phi[libs[ii]])
                BS.c = xscale[ishl] * coefssl
                shift = BS(phi[libs[ii]])
                beta = (lambnorm[libs[ii], :] - (1 + shift)) / (2*wi2)
                alpha = -beta**2 * (2*wi2)
                # exp = np.exp(alpha)
                bsexp = bs * np.exp(alpha)

                # Background
                jac0[libs[ii], ibckx[ii]] = bs[:, 0] * scales[ibckx[ii]]

                # amp (shape: nphi/lamb, namp[jj])
                for jj in range(len(iaj)):
                    ix = iax[ii, jj]
                    # jac0[libs[ii], ix] = np.sum(
                    # (bs * exp[:, iaj[jj]] * scales[ix]
                    # * coefsal[0:1, iaj[jj]]),
                    # axis=1)
                    jac0[libs[ii], ix] = np.sum(
                        bsexp[:, iaj[jj]] * coefsal[:, iaj[jj]],
                        axis=1) * scales[ix]

                # width2
                for jj in range(len(iwj)):
                    ix = iwx[ii, jj]
                    # jac0[libs[ii], ix] = np.sum(
                    # (amp[:, iwj[jj]]
                    # * (-alpha[:, iwj[jj]]
                    # * bs * exp[:, iwj[jj]] * scales[ix]
                    # * coefswl[0:1, iwj[jj]]
                    # / wi2[:, iwj[jj]])),
                    # axis=1)
                    jac0[libs[ii], ix] = np.sum(
                        (
                            -alpha[:, iwj[jj]] * amp[:, iwj[jj]]
                            * bsexp[:, iwj[jj]] * coefswl[:, iwj[jj]]
                            / wi2[:, iwj[jj]]
                        ),
                        axis=1,
                    ) * scales[ix]

                # shift
                for jj in range(len(ishj)):
                    ix = ishx[ii, jj]
                    # jac0[libs[ii], ix] = np.sum(
                    # (amp[:, ishj[jj]]
                    # * 2.*beta[:, ishj[jj]]
                    # * bs * exp[:, ishj[jj]] * scales[ix]
                    # * coefssl[0:1, ishj[jj]]),
                    # axis=1)
                    jac0[libs[ii], ix] = np.sum(
                        (
                            amp[:, ishj[jj]] * 2. * beta[:, ishj[jj]]
                            * bsexp[:, ishj[jj]] * coefssl[:, ishj[jj]]
                        ),
                        axis=1,
                    ) * scales[ix]

                # double
                if double is False:
                    continue

                if double is True:
                    dratio = xscale[idratiox]
                    # coefssl are line-specific, they do not affect dshift
                    dshift = shift + xscale[idshx]
                else:
                    dratio = double.get('dratio', xscale[idratiox])
                    dshift = shift + double.get('dshift', xscale[idshx])

                # ampd = amp*dratio
                betad = (lambnorm[libs[ii], :] - (1 + dshift)) / (2*wi2)
                alphad = -betad**2 * (2*wi2)
                expd = np.exp(alphad)
                bsexpd = bs * expd

                # amp
                for jj in range(len(iaj)):
                    ix = iax[ii, jj]
                    # jac0[libs[ii], ix] += dratio*np.sum(
                    # (bs * scales[ix] * coefsal[0:1, iaj[jj]]
                    # * expd[:, iaj[jj]]),
                    # axis=1)
                    jac0[libs[ii], ix] += dratio*np.sum(
                        bsexpd[:, iaj[jj]] * coefsal[:, iaj[jj]],
                        axis=1) * scales[ix]

                # width2
                for jj in range(len(iwj)):
                    ix = iwx[ii, jj]
                    # jac0[libs[ii], ix] += np.sum(
                    # (ampd[:, iwj[jj]]
                    # * (-alphad[:, iwj[jj]]
                    # * bs * scales[ix] * coefswl[0:1, iwj[jj]]
                    # / wi2[:, iwj[jj]])
                    # * expd[:, iwj[jj]]),
                    # axis=1)
                    jac0[libs[ii], ix] += np.sum(
                        (-alphad[:, iwj[jj]] * amp[:, iwj[jj]]
                         * bsexpd[:, iwj[jj]] * coefswl[:, iwj[jj]]
                         / wi2[:, iwj[jj]]), axis=1) * scales[ix] * dratio

                # shift
                for jj in range(len(ishj)):
                    ix = ishx[ii, jj]
                    # jac0[libs[ii], ix] += np.sum(
                    # (ampd[:, ishj[jj]]
                    # * 2.*betad[:, ishj[jj]]
                    # * bs * scales[ix] * coefssl[0:1, ishj[jj]]
                    # * expd[:, ishj[jj]]),
                    # axis=1)
                    jac0[libs[ii], ix] += np.sum(
                        (amp[:, ishj[jj]] * 2.*betad[:, ishj[jj]]
                         * bsexpd[:, ishj[jj]] * coefssl[:, ishj[jj]]),
                        axis=1) * scales[ix] * dratio

                # dratio
                if double is True or double.get('dratio') is None:
                    jac0[libs[ii], idratiox] = (scales[idratiox]
                                                * np.sum(amp * expd, axis=1))

                # dshift
                if double is True or double.get('dshift') is None:
                    jac0[libs[ii], idshx] = dratio * np.sum(
                        amp * 2.*betad*scales[idshx] * expd, axis=1)
            return jac0

    elif jac == 'sparse':

        msg = ""
        raise Exception(msg)

        # Define a callable jac returning (nlamb, sizex) matrix of partial
        # derivatives of np.sum(func_details(scaled), axis=0)

        # Pre-instanciate jacobian
        # libsT = libs.T
        # indflat = np.flatnonzero(libsT)
        # row_ind, col_ind = libsT.nonzero()
        # libsi = [indflat[col_ind==ii] for ii in range(nbs)]

        # jac0 = scpsparse.csr_matrix((libsT[libsT].astype(float),
        # (row_ind, col_ind)),
        # shape=libsT.shape)

        # def jacob(x, lamb=lambcost, phi=phicost,
        # ibckx=ibckx,
        # iax=iax, iaj=iaj, ial=ial,
        # iwx=iwx, iwj=iwj, iwl=iwl,
        # ishx=ishx, ishj=ishj, ishl=ishl,
        # idratiox=idratiox, idshx=idshx,
        # lines=dinput['lines'][None, :],
        # km=km, kpb=kpb, nbs=nbs,
        # deg=dinput['deg'],
        # coefsal=coefsal[None, :],
        # coefswl=coefswl[None, :],
        # coefssl=coefssl[None, :],
        # double=dinput['double'],
        # scales=None, indok_var=None, data=None,
        # jac0=jac0, lbs=lbs, libsi=libsi):
        # """ Basic docstr """
        # xscale = x*scales

        # # Intermediates

    # # Loop on bs
    # for ii in range(nbs):
        # bs = lbs[ii]

        # # Intermediates
        # amp = BSpline(km, xscale[ial] * coefsal, deg,
        # extrapolate=False, axis=0)(phi[libs[ii]])
        # wi2 = BSpline(km, xscale[iwl] * coefswl, deg,
        # extrapolate=False, axis=0)(phi[libs[ii]])
        # shift = BSpline(km, xscale[ishl] * coefssl, deg,
        # extrapolate=False, axis=0)(phi[libs[ii]])
        # beta = (lamb[libs[ii], :]/lines - (1 + shift)) / (2*wi2)
        # alpha = -beta**2 * (2*wi2)
        # exp = np.exp(alpha)

        # # Background
        # import ipdb; ipdb.set_trace()       # DB
        # jac0.data[libsi[ibckx[ii]]] = bs[:, 0] * scales[ibckx[ii]]
        # #jac0[libs[ii], ibckx[ii]] = bs[:, 0] * scales[ibckx[ii]]

        # # amp
        # for jj in range(len(iaj)):
        # ix = iax[ii, jj]
        # jac0.data[libs[ii], ix] = np.sum(
        # (bs * exp[:, iaj[jj]] * scales[ix]
        # * coefsal[0:1, iaj[jj]]),
        # axis=1)

        # # width2
        # for jj in range(len(iwj)):
        # ix = iwx[ii, jj]
        # jac0[libs[ii], ix] = np.sum(
        # (amp[:, iwj[jj]]
        # * (-alpha[:, iwj[jj]]
        # * bs * exp[:, iwj[jj]] * scales[ix]
        # * coefswl[0:1, iwj[jj]]
        # / wi2[:, iwj[jj]])),
        # axis=1)

        # # shift
        # for jj in range(len(ishj)):
        # ix = ishx[ii, jj]
        # jac0[libs[ii], ix] = np.sum(
        # (amp[:, ishj[jj]]
        # * 2.*beta[:, ishj[jj]]
        # * bs * exp[:, ishj[jj]] * scales[ix]
        # * coefssl[0:1, ishj[jj]]),
        # axis=1)

        # # double
        # if double is False:
        # continue

        # if double is True:
        # dratio = xscale[idratiox]
        # # coefssl are line-specific, they do not affect dshift
        # shiftd = shift + xscale[idshx]
        # else:
        # if double.get('dratio') is None:
        # dratio = xscale[idratiox]
        # else:
        # dratio = double.get('dratio')
        # if double.get('dshift') is None:
        # shiftd = shift + xscale[idshx]
        # else:
        # shiftd = shift + double.get('dshift')

        # ampd = amp*dratio
        # betad = (lamb[libs[ii]]/lines - (1 + shiftd)) / (2*wi2)
        # alphad = -betad**2 * (2*wi2)
        # expd = np.exp(alphad)

        # # amp
        # for jj in range(len(iaj)):
        # ix = iax[ii, jj]
        # jac0[libs[ii], ix] += dratio*np.sum(
        # (bs * scales[ix] * coefsal[0:1, iaj[jj]]
        # * expd[:, iaj[jj]]),
        # axis=1)

        # # width2
        # for jj in range(len(iwj)):
        # ix = iwx[ii, jj]
        # jac0[libs[ii], ix] += np.sum(
        # (ampd[:, iwj[jj]]
        # * (-alphad[:, iwj[jj]]
        # * bs * scales[ix] * coefswl[0:1, iwj[jj]]
        # / wi2[:, iwj[jj]])
        # * expd[:, iwj[jj]]),
        # axis=1)

        # # shift
        # for jj in range(len(ishj)):
        # ix = ishx[ii, jj]
        # jac0[libs[ii], ix] += np.sum(
        # (ampd[:, ishj[jj]]
        # * 2.*betad[:, ishj[jj]]
        # * bs * scales[ix] * coefssl[0:1, ishj[jj]]
        # * expd[:, ishj[jj]]),
        # axis=1)

        # # dratio
        # if double is True or double.get('dratio') is None:
        # jac0[libs[ii], idratiox] = (scales[idratiox]
        # *np.sum(amp*expd, axis=1))

        # # dshift
        # if double is True or double.get('dshift') is None:
        # jac0[libs[ii], idshx] = np.sum(
        # ampd * 2.*betad*scales[idshx] * expd, axis=1)
        # return jac0[indok, :]

    if jac not in ['dense', 'sparse', 'LinearSparseOperator']:
        if jac not in ['2-point', '3-point']:
            msg = ("jac should be in "
                   + "['dense', 'sparse', 'LinearsparseOp', "
                   + "'2-point', '3-point']\n"
                   + "\t- provided: {}".format(jac))
            raise Exception(msg)
        jacob = jac

    return func_detail, cost, jacob
