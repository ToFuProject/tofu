
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
    lamb=None,
    indx=None,
    dinput=None,
    dind=None,
    jac=None,
):

    if jac is None:
        jac = _JAC

    ibckax = dind['bck_amp']['x'][0, :]
    ibckrx = dind['bck_rate']['x'][0, :]
    nbck = 1    # ibckax.size + ibckrx.size
    iax = dind['amp']['x'][0, :]
    iwx = dind['width']['x'][0, :]
    ishx = dind['shift']['x'][0, :]
    idratiox, idshx = None, None
    if dinput['double'] is not False:
        c0 = dinput['double'] is True
        if c0 or dinput['double'].get('dratio') is None:
            idratiox = dind['dratio']['x'][0, :]
        if c0 or dinput['double'].get('dshift') is None:
            idshx = dind['dshift']['x'][0, :]

    ial = dind['amp']['lines'][0, :]
    iwl = dind['width']['lines'][0, :]
    ishl = dind['shift']['lines'][0, :]

    iaj = dind['amp']['jac']
    iwj = dind['width']['jac']
    ishj = dind['shift']['jac']

    coefsal = dinput['amp']['coefs']
    coefswl = dinput['width']['coefs']
    coefssl = dinput['shift']['coefs']

    offsetal = dinput['amp']['offset']
    offsetwl = dinput['width']['offset']
    offsetsl = dinput['shift']['offset']

    lambrel = lamb - dinput['lambmin_bck']
    lambnorm = lamb[..., None]/dinput['lines'][None, ...]

    xscale = np.full((dind['sizex'],), np.nan)
    if indx is None:
        indx = np.ones((dind['sizex'],), dtype=bool)

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
        scales=None,
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
            indok = np.ones(lambrel.shape, dtype=bool)

        # xscale = x*scales   !!! scales ??? !!! TBC
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


def multigausfit2d_from_dlines_funccostjac(
    phi_flat=None,
    dinput=None,
    dind=None,
    jac=None,
):

    return_costjac = phi_flat is not None

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

    coefsal = dinput['amp']['coefs'][None, :]
    coefswl = dinput['width']['coefs'][None, :]
    coefssl = dinput['shift']['coefs'][None, :]

    offsetal = dinput['amp']['offset'][None, :]
    offsetwl = dinput['width']['offset'][None, :]
    offsetsl = dinput['shift']['offset'][None, :]

    xscale = np.full((dind['sizex'],), np.nan)

    km = dinput['knots_mult']
    kpb = dinput['nknotsperbs']
    nbs = dinput['nbs']

    BS = BSpline(
        km,
        np.ones(ial.shape, dtype=float),
        dinput['deg'],
        extrapolate=False,
        axis=0,
    )

    nlines = dinput['nlines']
    deg = dinput['deg']
    double = dinput['double']
    lambmin_bck = dinput['lambmin_bck']

    # Pre-set kwdargs
    dkwdargs = dict(
        ibckax=ibckax,
        ibckrx=ibckrx,
        ial=ial,
        iwl=iwl,
        ishl=ishl,
        idratiox=idratiox,
        idshx=idshx,
        nlines=nlines,
        nbck=nbck,
        km=km,
        kpb=kpb,
        nbs=nbs,
        deg=deg,
        BS=BS,
        coefsal=coefsal[None, :],
        coefswl=coefswl[None, :],
        coefssl=coefssl[None, :],
        offsetal=offsetal[None, :],
        offsetwl=offsetwl[None, :],
        offsetsl=offsetsl[None, :],
        double=double,
        lambmin_bck=lambmin_bck,
    )

    # ----------------------------
    # details and sum for end-user

    # func_details returns result in same shape as input
    def func_detail(
        x,
        xscale=xscale,
        phi=None,
        lamb=None,
        scales=None,
        ind_bs=None,
        const=None,
        indx=None,
        **dkwdargs,
    ):
        # normalize lamb
        lambrel = lamb - lambmin_bck
        shape = tuple(np.r_[[1 for ii in range(lamb.ndim)], -1])
        lambn = lamb[..., None] / dinput['lines'].reshape(shape)

        # shape = tuple(np.r_[indok.sum(), nbck+nlines, nbs])
        shape = tuple(np.r_[phi.shape, nbck + nlines, nbs])
        y = np.full(shape, np.nan)

        if indx is None:
            xscale = x*scales
        else:
            xscale[indx] = x*scales[indx]
            xscale[~indx] = const

        # bck rate
        BS.c = xscale[ibckrx]
        bckr = BS(phi)

        # make sure iwl is 2D to get all lines at once
        BS.c = xscale[iwl] * coefswl + offsetwl
        wi2 = BS(phi)
        BS.c = xscale[ishl] * coefssl + offsetsl
        shift = BS(phi)
        exp = np.exp(-(lambn - (1 + shift))**2 / (2*wi2))

        if double is not False:
            # coefssl are line-specific, they do not affect dshift
            if double is True:
                dratio = xscale[idratiox[:, 0]]
                dshift = shift + xscale[idshx[:, 0]]  # *coefssl
            else:
                dratio = double.get('dratio', xscale[idratiox[:, 0]])
                dshift = shift + double.get('dshift', xscale[idshx[:, 0]])
            expd = np.exp(-(lambn - (1 + dshift))**2 / (2*wi2))

        # Loop on individual bsplines for amp
        for ii in range(nbs):

            if ind_bs is not None and ii not in ind_bs:
                continue

            bs = BSpline.basis_element(
                km[ii:ii + kpb],
                extrapolate=False,
            )(phi)

            indbs = np.isfinite(bs)
            if not np.any(indbs):
                continue
            bs = bs[indbs]

            # bck
            y[indbs, 0, ii] = (
                xscale[ibckax[ii, 0]]*bs
                * np.exp(bckr[indbs, 0]*lambrel[indbs])
            )

            # lines
            for jj in range(nlines):
                amp = (
                    bs * xscale[ial[ii, jj]] * coefsal[:, jj] + offsetal[:, jj]
                )
                y[indbs, nbck+jj, ii] = amp * exp[indbs, jj]
                if double is not False:
                    y[indbs, nbck+jj, ii] += (amp * dratio * expd[indbs, jj])

        return y

    # cost and jacob return flattened results (for least_squares())
    def func_sum(
        x,
        xscale=xscale,
        phi=None,
        lamb=None,
        scales=None,
        ind_bs=None,
        const=None,
        indx=None,
        **dkwdargs,
    ):

        # normalize lamb
        lambrel = lamb - lambmin_bck
        shape = tuple(np.r_[[1 for ii in range(lamb.ndim)], -1])
        lambn = lamb[..., None] / dinput['lines'].reshape(shape)

        # xscale = x*scales
        if indx is None:
            xscale = x*scales
        else:
            xscale[indx] = x*scales[indx]
            xscale[~indx] = const

        # Background
        BS.c = xscale[ibckax][:, 0]
        bcka = BS(phi)
        BS.c = xscale[ibckrx][:, 0]
        y = bcka * np.exp(BS(phi)*lambrel)

        # make sure iwl is 2D to get all lines at once
        BS.c = xscale[ial] * coefsal + offsetal
        amp = BS(phi)
        BS.c = xscale[iwl] * coefswl + offsetwl
        wi2 = BS(phi)
        BS.c = xscale[ishl] * coefssl + offsetsl
        csh = BS(phi)

        y += np.nansum(
            amp * np.exp(-(lambn - (1 + csh))**2 / (2*wi2)),
            axis=-1,
        )

        if double is not False:
            if double is True:
                dratio = xscale[idratiox[:, 0]]
                # scales[ishl] or scales[idshx] ? coefssl ? +> no
                dcsh = csh + xscale[idshx[:, 0]]
            else:
                dratio = double.get('dratio', xscale[idratiox[:, 0]])
                dcsh = csh + double.get('dshift', xscale[idshx[:, 0]])

            expd = np.exp(-(lambn - (1 + dcsh))**2 / (2*wi2))
            y += np.nansum(amp * dratio * expd, axis=-1)
        return y

    # -------------------------------
    # cost and jacob for optimization

    func_cost, func_jacob = None, None
    if return_costjac:

        def func_cost(
            x,
            xscale=xscale,
            scales=None,
            indok_flat=None,
            ind_bs=None,
            const=None,
            indx=None,
            data_flat=None,
            phi_flat=None,
            lambrel_flat=None,
            lambn_flat=None,
            jac0=None,
            libs=None,
            **dkwdargs,
        ):

            assert data_flat.shape == lambrel_flat.shape
            assert indok_flat is not None

            if indx is None:
                xscale = x*scales
            else:
                xscale[indx] = x*scales[indx]
                xscale[~indx] = const

            # Background
            BS.c = xscale[ibckax][:, 0]
            bcka = BS(phi_flat)
            BS.c = xscale[ibckrx][:, 0]
            y = bcka * np.exp(BS(phi_flat)*lambrel_flat)

            # make sure iwl is 2D to get all lines at once
            BS.c = xscale[ial] * coefsal + offsetal
            amp = BS(phi_flat)
            BS.c = xscale[iwl] * coefswl + offsetwl
            wi2 = BS(phi_flat)
            BS.c = xscale[ishl] * coefssl + offsetsl
            csh = BS(phi_flat)

            y += np.nansum(
                amp * np.exp(-(lambn_flat - (1 + csh))**2 / (2*wi2)),
                axis=-1,
            )

            if double is not False:
                if double is True:
                    dratio = xscale[idratiox[:, 0]]
                    # scales[ishl] or scales[idshx] ? coefssl ? +> no
                    dcsh = csh + xscale[idshx[:, 0]]
                else:
                    dratio = double.get('dratio', xscale[idratiox[:, 0]])
                    dcsh = csh + double.get('dshift', xscale[idshx[:, 0]])

                expd = np.exp(-(lambn_flat - (1 + dcsh))**2 / (2*wi2))
                y += np.nansum(amp * dratio * expd, axis=-1)

            return y - data_flat

        def func_jacob(
            x,
            xscale=xscale,
            scales=None,
            indok_flat=None,
            ind_bs=None,
            const=None,
            indx=None,
            data_flat=0.,
            phi_flat=None,
            lambrel_flat=None,
            lambn_flat=None,
            jac0=None,
            libs=None,
            **dkwdargs,
        ):
            """ Basic docstr """
            if indx is None:
                xscale = x*scales
            else:
                xscale[indx] = x*scales[indx]
                xscale[~indx] = const

            # Intermediates

            # Loop on bs
            for ii in range(nbs):

                # phi interval
                ibs = libs[ii]

                # bspline
                bs = BSpline.basis_element(
                    km[ii:ii+kpb],
                    extrapolate=False,
                )(phi_flat[ibs])[:, None]

                # check bspline
                if np.any(~np.isfinite(bs)):
                    msg = "Non-finite values in bs (affecting jacobian)!"
                    raise Exception(msg)

                # Intermediates - background
                BS.c = xscale[ibckax[:, 0]]
                bcka = BS(phi_flat[ibs])
                BS.c[...] = xscale[ibckrx[:, 0]]
                bckr = BS(phi_flat[ibs])
                expbck = np.exp(bckr*lambrel_flat[ibs])

                # Intermediates - amp, wi2, shift
                BS.c = xscale[ial] * coefsal + offsetal
                amp = BS(phi_flat[ibs])
                BS.c[...] = xscale[iwl] * coefswl + offsetwl
                wi2 = BS(phi_flat[ibs])
                BS.c[...] = xscale[ishl] * coefssl + offsetsl
                shift = BS(phi_flat[ibs])

                beta = (lambn_flat[ibs, :] - (1 + shift)) / (2*wi2)
                alpha = -beta**2 * (2*wi2)
                # exp = np.exp(alpha)
                bsexp = bs * np.exp(alpha)

                # Background amplitude
                jac0[ibs, ibckax[ii, 0]] = (
                    bs[:, 0] * scales[ibckax[ii, 0]] * expbck
                )

                # Background rate
                jac0[ibs, ibckrx[ii, 0]] = (
                    bs[:, 0] * scales[ibckrx[ii, 0]]
                    * lambrel_flat[ibs] * bcka * expbck
                )

                # amp (shape: nphi/lamb, namp[jj])
                for jj in range(len(iaj)):
                    ix = iax[ii, jj]
                    jac0[ibs, ix] = np.sum(
                        bsexp[:, iaj[jj]] * coefsal[:, iaj[jj]],
                        axis=1,
                    ) * scales[ix]

                # width2
                for jj in range(len(iwj)):
                    ix = iwx[ii, jj]
                    jac0[ibs, ix] = np.sum(
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
                    jac0[ibs, ix] = np.sum(
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
                    dratio = xscale[idratiox[:, 0]]
                    # coefssl are line-specific, they do not affect dshift
                    dshift = shift + xscale[idshx[:, 0]]
                else:
                    dratio = double.get('dratio', xscale[idratiox[:, 0]])
                    dshift = shift + double.get('dshift', xscale[idshx[:, 0]])

                # ampd = amp*dratio
                betad = (lambn_flat[ibs, :] - (1 + dshift)) / (2*wi2)
                alphad = -betad**2 * (2*wi2)
                expd = np.exp(alphad)
                bsexpd = bs * expd

                # amp
                for jj in range(len(iaj)):
                    ix = iax[ii, jj]
                    jac0[ibs, ix] += dratio*np.sum(
                        bsexpd[:, iaj[jj]] * coefsal[:, iaj[jj]],
                        axis=1,
                    ) * scales[ix]

                # width2
                for jj in range(len(iwj)):
                    ix = iwx[ii, jj]
                    jac0[ibs, ix] += np.sum(
                        (-alphad[:, iwj[jj]] * amp[:, iwj[jj]]
                         * bsexpd[:, iwj[jj]] * coefswl[:, iwj[jj]]
                         / wi2[:, iwj[jj]]),
                        axis=1,
                    ) * scales[ix] * dratio

                # shift
                for jj in range(len(ishj)):
                    ix = ishx[ii, jj]
                    jac0[ibs, ix] += np.sum(
                        (amp[:, ishj[jj]] * 2.*betad[:, ishj[jj]]
                         * bsexpd[:, ishj[jj]] * coefssl[:, ishj[jj]]),
                        axis=1,
                    ) * scales[ix] * dratio

                # dratio
                if double is True or double.get('dratio') is None:
                    jac0[ibs, idratiox[:, 0]] = (
                        scales[idratiox[:, 0]] * np.sum(amp * expd, axis=1)
                    )

                # dshift
                if double is True or double.get('dshift') is None:
                    jac0[ibs, idshx[:, 0]] = dratio * np.sum(
                        amp * 2.*betad*scales[idshx] * expd,
                        axis=1,
                    )
            if indx is None:
                return jac0
            else:
                return jac0[:, indx]

    return func_detail, func_sum, func_cost, func_jacob
