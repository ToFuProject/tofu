# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


__all__ = ['spectral_range_2d']


# #################################################################
# #################################################################
#               Main
# #################################################################


def spectral_range_2d(
    # crystal
    lamb0=None,
    bragg0=None,
    # geometry basis
    ap=None,
    ex=None,
    ey=None,
    beta_max=None,
    # geometry
    xx=None,
    length=None,
    rcurve=None,
    dist=None,
    # camera
    dcam=None,
    # options
    npts=None,
    # plotting
    plot=None,
    dax=None,
    # saving
    save=None,
    pfe_fig=None,
    pfe_npz=None,
):

    # --------
    # check

    din, npts, plot, save, pfe_fig, pfe_npz = _check(**locals())

    # --------------
    # compute

    crystx, crysty, endx, endy, lamb = _compute(
        npts=npts,
        beta_max=beta_max,
        dcam=dcam,
        **din,
    )

    # -------------
    # format output

    dout = dict(din)
    dout.update({
        'crystx': crystx,
        'crysty': crysty,
        'endx': endx,
        'endy': endy,
        'lamb': lamb,
        'lamb_min': np.nanmin(lamb, axis=0),
        'lamb_max': np.nanmax(lamb, axis=0),
        'Dlamb': np.nanmax(lamb, axis=0) - np.nanmin(lamb, axis=0),
    })

    # ---------
    # plot

    if plot is True:
        dax = _plot(
            dax=dax,
            dcam=dcam,
            pfe_fig=pfe_fig,
            **dout,
        )

    # ----------
    # save

    if save is True:
        np.savez(pfe_npz, **dout)

    # ---------
    # return 
    
    if plot is True:
        return dout, dax
    else:
        return dout


# #################################################################
# #################################################################
#               Check
# #################################################################


def _check(
    # crystal
    lamb0=None,
    bragg0=None,
    # geometry basis
    ap=None,
    ex=None,
    ey=None,
    # geometry
    xx=None,
    length=None,
    rcurve=None,
    dist=None,
    # options
    npts=None,
    # plotting
    plot=None,
    ax=None,
    # saving
    save=None,
    pfe_fig=None,
    pfe_npz=None,
    # unused
    **kwdargs,
):

    # --------------
    # geometry basis

    basis_def = {'ap': np.r_[0, 0], 'ex': np.r_[1, 0], 'ey': np.r_[0, 1]}
    din_basis = {'ap': ap, 'ex': ex, 'ey': ey}
    for k0, v0 in din_basis.items():

        if v0 is None:
            din_basis[k0] = basis_def[k0]

        din_basis[k0] = np.atleast_1d(din_basis[k0]).ravel().astype(float)
        if din_basis[k0].size != 2:
            msg = f"Arg '{k0}' must be a 2d array!\nProvided: {din_basis[k0]}"
            raise Exception(msg)

    # normalize ex
    din_basis['ex'] = din_basis['ex'] / np.linalg.norm(din_basis['ex'])

    # perpendicular + normalize ey
    sca = np.sum(din_basis['ex']*din_basis['ey'])
    din_basis['ey'] = din_basis['ey'] - sca * din_basis['ex']
    din_basis['ey'] = din_basis['ey'] / np.linalg.norm(din_basis['ey'])

    # -----------------
    # initialize dict

    din = {
        'lamb0': lamb0,
        'bragg0': bragg0,
        # geometry
        'xx': xx,
        'length': length,
        'rcurve': rcurve,
        'dist': dist,
    }

    # -------------
    # get size

    # make all arrays
    for k0, v0 in din.items():
        din[k0] = np.atleast_1d(v0).ravel().astype(float)

    # sizes
    lsizes = list(set([v0.size for v0 in din.values()]))
    if len(lsizes) == 1:
        pass
    elif 1 in lsizes and len(lsizes) == 2:
        size = [ss for ss in lsizes if ss != 1][0]
        for k0, v0 in din.items():
            if v0.size == 1:
                din[k0] = np.full((size,), v0[0])
    else:
        lstr = [f"\t- '{k0}': {v0.size}" for k0, v0 in din.items()]
        msg = (
            "All args must be either scalar or 1d arrays of the same size:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # -------
    # values

    for k0, v0 in din.items():
        if k0 != 'rcurve':
            c0 = np.all(np.isfinite(v0)) and np.all(v0 >= 0.)

            if not c0:
                msg = (
                    f"Arg '{k0}' must be finite and positive\n"
                    "Provided: {v0}"
                )
                raise Exception(msg)

    # ------------
    # add basis
    din.update(din_basis)

    # ---------
    # npts

    if npts is None:
        npts = 101

    # ---------
    # plot

    # plot
    plot = ds._generic_check._check_var(
        plot, 'plot',
        types=bool,
        default=True,
    )

    # ---------
    # save

    # save
    save = ds._generic_check._check_var(
        save, 'save',
        types=bool,
        default=False,
    )

    return din, npts, plot, save, pfe_fig, pfe_npz


# #################################################################
# #################################################################
#               Compute
# #################################################################


def _compute(
    # crystal
    lamb0=None,
    bragg0=None,
    # geometry basis
    ap=None,
    ex=None,
    ey=None,
    beta_max=None,
    # geometry
    xx=None,
    length=None,
    rcurve=None,
    dist=None,
    # options
    npts=None,
    # camera
    dcam=None,
):

    # ------------
    # initialize

    size = lamb0.size
    cx = np.full((size,), np.nan)
    cy = np.full((size,), np.nan)

    crystx = np.full((npts, size), np.nan)
    crysty = np.full((npts, size), np.nan)
    vnx = np.full((npts, size), np.nan)
    vny = np.full((npts, size), np.nan)

    # ----------------
    # compute geometry

    # 2d
    d2 = lamb0 / np.sin(bragg0)

    # summit of crystal
    sx = ap[0] + xx * ex[0]
    sy = ap[1] + xx * ex[1]

    # indices of curved crystals
    indc = np.isfinite(rcurve)

    # center of curvature
    cx[indc] = sx[indc] - rcurve[indc] * np.sin(bragg0[indc])
    cy[indc] = sy[indc] + rcurve[indc] * np.cos(bragg0[indc])

    # half angular opening of crystal
    dalpha = 0.5*length[indc] / rcurve[indc]
    theta = (
        np.pi/2. - bragg0[None, indc]
        + dalpha * np.linspace(-1, 1, npts)[:, None]
    )

    # crystal plotting - curved
    crystx[:, indc] = cx[None, indc] + rcurve[indc][None, :] * np.cos(theta)
    crysty[:, indc] = cy[None, indc] - rcurve[indc][None, :] * np.sin(theta)

    # crystal plotting - straight
    ll = 0.5 * length[None, ~indc] * np.linspace(-1, 1, npts)[:, None]
    crystx[:, ~indc] = sx[None, ~indc] + ll*np.cos(bragg0)[None, ~indc]
    crysty[:, ~indc] = sy[None, ~indc] + ll*np.sin(bragg0)[None, ~indc]

    # ----------------
    # compute rays

    # vectors of incident rays
    vix = crystx - ap[0]
    viy = crysty - ap[1]
    vin = np.sqrt(vix**2 + viy**2)
    vix = vix / vin
    viy = viy / vin

    # local normal vectors
    vnx[:, indc] = -np.sin(np.pi/2 - theta)
    vny[:, indc] = np.cos(np.pi/2. - theta)
    vnx[:, ~indc] = -np.sin(bragg0[None, ~indc])
    vny[:, ~indc] = np.cos(bragg0[None, ~indc])

    # reflected vectors
    sca = vix*vnx + viy*vny
    vrx = vix - 2.*sca*vnx
    vry = viy - 2.*sca*vny

    # end of rays at dist
    endx = crystx + dist * vrx
    endy = crysty + dist * vry

    # ----------------------
    # compute spectral range

    # get local bragg angle - top and bottom
    bragg = np.pi/2. - np.arccos(sca)

    # lamb
    lamb = d2 * np.sin(bragg)

    # beta_max
    if beta_max is not None:
        beta = np.arctan2(crysty - ap[1], crystx - ap[0])
        ind = np.abs(beta) > beta_max
        endx[ind] = np.nan
        endy[ind] = np.nan
        lamb[ind] = np.nan

    # lamb min, max
    lambm = np.nanmin(lamb, axis=0)
    lambM = np.nanmax(lamb, axis=0)

    # Dlamb
    Dlamb = np.nanmax(lamb, axis=0) - np.nanmin(lamb, axis=0)

    # -----------------
    # impacts on camera

    if dcam is not None:
        ninx, niny = dcam['nin'][:2]
        sca_up = (
            (dcam['cent'][0] - crystx) * ninx
            + (dcam['cent'][1] - crysty) * niny
        )
        sca_bot = vrx*ninx + vry*niny

        kk = sca_up / sca_bot
        ptsx = crystx + kk * vrx
        ptsy = crysty + kk * vry

        e0x = -dcam['nin'][1]
        e0y = dcam['nin'][0]
        dcam['x0'] = (
            (ptsx - dcam['cent'][0]) * e0x
            + (ptsy - dcam['cent'][1]) * e0y
        )

    return crystx, crysty, endx, endy, lamb


# #################################################################
# #################################################################
#               Plot
# #################################################################


def _plot(
    # crystal
    lamb0=None,
    bragg0=None,
    # geometry
    xx=None,
    length=None,
    rcurve=None,
    dist=None,
    # computed
    ap=None,
    crystx=None,
    crysty=None,
    endx=None,
    endy=None,
    lamb_min=None,
    lamb_max=None,
    Dlamb=None,
    x0=None,
    # camera
    dcam=None,
    # plotting
    dax=None,
    # saving
    pfe_fig=None,
    # unused
    **kwdargs,
):

    # ----------
    # prepare

    npts, size = crystx.shape

    # envelop
    iok = np.isfinite(endx)
    i0 = tuple([iok[:, ii].nonzero()[0][0] for ii in range(size)])
    i1 = tuple([iok[:, ii].nonzero()[0][-1] for ii in range(size)])
    nind = tuple(range(size))

    # envelop
    envx = np.array([
        endx[i1, nind], crystx[i1, nind],
        np.full((size,), ap[0]),
        crystx[i0, nind], endx[i0, nind],
    ])
    envy = np.array([
        endy[i1, nind], crysty[i1, nind],
        np.full((size,), ap[1]),
        crysty[i0, nind], endy[i0, nind],
    ])

    # central rays
    ind = int((npts-1)/2)
    raycx = np.array([np.full((size,), ap[0]), crystx[ind, :], endx[ind, :]])
    raycy = np.array([np.full((size,), ap[1]), crysty[ind, :], endy[ind, :]])

    # dcam
    if dcam is not None:
        ninx, niny = dcam['nin'][:2]
        e0x, e0y = -niny, ninx
        clen = dcam['length']
        camx = dcam['cent'][0] + 0.5*clen*np.r_[-1, 1] * e0x
        camy = dcam['cent'][1] + 0.5*clen*np.r_[-1, 1] * e0y

    # --------------
    # prepare figure

    if dax is None:
        dax = _get_axes()

    # -----------
    # plot

    color = None
    for ii in range(size):

        # ---
        # hor

        kax = 'hor'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            # crystals
            ll, = ax.plot(
                crystx[:, ii],
                crysty[:, ii],
                ls='-',
                lw=2,
                marker='None',
            )
            color = ll.get_color()

            # central rays
            ax.plot(
                raycx[:, ii],
                raycy[:, ii],
                ls='--',
                lw=1,
                marker='None',
                c=color,
            )

            # edge rays
            ax.plot(
                envx[:, ii],
                envy[:, ii],
                ls='-',
                lw=1,
                marker='None',
                c=color,
            )

        kax = 'cam'
        if dcam is not None and dax.get(kax) is not None:
            ax = dax[kax]['handle']

            # images
            ax.plot(
                dcam['x0'][:, ii],
                np.full((npts,), ii+1),
                ls='None',
                marker='.',
                color=color,
                ms=6,
            )

            # lamb min, max
            ax.text(
                np.nanmin(dcam['x0'][:, ii]),
                ii + 1 - 0.1,
                f'{lamb_min[ii]*1e10:2.3} AA',
                color=color,
                size=8,
                horizontalalignment='center',
                verticalalignment='top',
            )

            ax.text(
                np.nanmax(dcam['x0'][:, ii]),
                ii + 1 - 0.1,
                f'{lamb_max[ii]*1e10:2.3} AA',
                color=color,
                size=8,
                horizontalalignment='center',
                verticalalignment='top',
            )

    # ------------
    # camera

    if dcam is not None:

        kax = 'hor'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            ax.plot(
                camx,
                camy,
                ls='-',
                lw=2.,
                marker='None',
                c='k',
            )

        kax = 'cam'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            ax.axvline(-0.5*dcam['length'], c='k', ls='-', lw=1.)
            ax.axvline(0.5*dcam['length'], c='k', ls='-', lw=1.)

            ax.set_ylim(0, size + 1)

    # ----------
    # saving

    if pfe_fig is not None:
        dax['hor']['handle'].figure.savefig(pfe_fig, format='png', dpi=200)

    return ax


def _get_axes():

    # --------------
    # prepare figure

    dmargin = {
        'left': 0.1, 'right': 0.98,
        'bottom': 0.1, 'top': 0.90,
        'hspace': 0.2, 'wspace': 0.2,
    }

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('2d ray-tracing model')
    gs = gridspec.GridSpec(ncols=3, nrows=2, **dmargin)

    # ----------
    # make axes

    # ax0 - hor
    ax0 = fig.add_subplot(
        gs[:, :-1],
        aspect='equal',
        adjustable='datalim',
    )

    ax0.set_xlabel("x (m)", size=12)
    ax0.set_ylabel("y (m)", size=12)
    ax0.set_title("2d ray tracing", size=12, fontweight='bold')

    # ax1 - cam
    ax1 = fig.add_subplot(
        gs[0, -1],
        aspect='auto',
    )

    ax1.set_xlabel("x0 (m)", size=12)
    ax1.set_title("Image on camera", size=12, fontweight='bold')

    # ------------
    # dict

    dax = {
        'hor': {'handle': ax0},
        'cam': {'handle': ax1},
    }

    return dax
