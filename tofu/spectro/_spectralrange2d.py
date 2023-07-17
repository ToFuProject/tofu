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
    
    """
    
    lamb0: target wavelength
    bragg0: target bragg angle
    rcurve: radii of curvature
    
    ap: point source position
    xx: distance between point source and crystals
    dist: lenght of rays after reflexion
    beta_max: maximum angular opening from point source (optionnal)
    npts: nb of rays from point source to crystals
    length: crystal length

    """

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

    ilamb_min = np.nanargmin(lamb, axis=0)
    ilamb_max = np.nanargmax(lamb, axis=0)
    
    lamb_min = np.array([lamb[imin, ii] for ii, imin in enumerate(ilamb_min)])
    lamb_max = np.array([lamb[imax, ii] for ii, imax in enumerate(ilamb_max)])

    dout = dict(din)
    dout.update({
        'beta_max': beta_max,
        'crystx': crystx,
        'crysty': crysty,
        'endx': endx,
        'endy': endy,
        'lamb': lamb,
        'ilamb_min': ilamb_min,
        'ilamb_max': ilamb_max,
        'lamb_min': lamb_min,
        'lamb_max': lamb_max,
        'Dlamb': lamb_max - lamb_min,
    })
    
    if dcam is not None:
        dout['dcam'] = dcam

    # ---------
    # plot

    if plot is True:
        dax = _plot(
            dax=dax,
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

        din_basis[k0] = np.atleast_1d(din_basis[k0]).ravel().astype(float)[:2]

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
                    f"Provided: {v0}"
                )
                raise Exception(msg)

    # ------------
    # add basis
    
    din.update(din_basis)

    # ---------
    # npts

    if npts is None:
        npts = 101
    npts = int(npts)
    if npts % 2 == 0:
        npts += 1

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
    ecx = np.sin(bragg0[indc]) * ex[0] - np.cos(bragg0[indc]) * ey[0]
    ecy = np.sin(bragg0[indc]) * ex[1] - np.cos(bragg0[indc]) * ey[1]
    ecx_p = -ecy
    ecy_p = ecx
    
    cx = sx[indc] - rcurve[indc] * ecx
    cy = sy[indc] - rcurve[indc] * ecy

    # half angular opening of crystal
    dalpha = 0.5*length[indc] / rcurve[indc]
    theta = dalpha * np.linspace(-1, 1, npts)[:, None]

    # crystal plotting - curved
    ethetax = np.cos(theta) * ecx[None, :] + np.sin(theta) * ecx_p[None, :]
    ethetay = np.cos(theta) * ecy[None, :] + np.sin(theta) * ecy_p[None, :]
    
    crystx[:, indc] = cx[None, :] + rcurve[indc][None, :] * ethetax
    crysty[:, indc] = cy[None, :] + rcurve[indc][None, :] * ethetay

    # crystal plotting - straight
    estraightx = np.cos(bragg0)[~indc] * ex[0] + np.sin(bragg0)[~indc] * ey[0]
    estraighty = np.cos(bragg0)[~indc] * ex[1] + np.sin(bragg0)[~indc] * ey[1]
    
    ll = 0.5 * length[None, ~indc] * np.linspace(-1, 1, npts)[:, None]
    crystx[:, ~indc] = sx[None, ~indc] + ll*estraightx[None, :]
    crysty[:, ~indc] = sy[None, ~indc] + ll*estraighty[None, :]
    
    # ----------------
    # compute rays

    # vectors of incident rays
    vix = crystx - ap[0]
    viy = crysty - ap[1]
    vin = np.sqrt(vix**2 + viy**2)
    vix = vix / vin
    viy = viy / vin

    # local normal vectors
    vnx[:, indc] = -ethetax
    vny[:, indc] = -ethetay
    vnx[:, ~indc] = -estraighty
    vny[:, ~indc] = estraightx

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
    bragg = np.arccos(sca) - np.pi/2.

    # lamb
    lamb = d2 * np.sin(bragg)

    # beta_max
    if beta_max is not None:
        dvx, dvy = crystx - ap[0], crysty - ap[1]
        beta = np.arctan2(dvx*ey[0] + dvy*ey[1], dvx*ex[0] + dvy*ex[1])
        ind = np.abs(beta) > beta_max
        endx[ind] = np.nan
        endy[ind] = np.nan
        lamb[ind] = np.nan

    # -----------------
    # impacts on camera

    if dcam is not None:
        ninx, niny = dcam['nin'][:2]
        ninn = np.sqrt(ninx**2 + niny**2)
        ninx, niny = ninx/ninn, niny/ninn
        
        ninx_r = ninx * ex[0] + niny * ey[0]
        niny_r = ninx * ex[1] + niny * ey[1]
        
        camx = ap[0] + dcam['cent'][0] * ex[0] + dcam['cent'][1] * ey[0]
        camy = ap[1] + dcam['cent'][0] * ex[1] + dcam['cent'][1] * ey[1]
        
        sca_up = (camx - crystx) * ninx_r + (camy - crysty) * niny_r
        sca_bot = vrx*ninx_r + vry*niny_r

        kk = sca_up / sca_bot
        ptsx = crystx + kk * vrx
        ptsy = crysty + kk * vry

        e0x = -niny_r
        e0y = ninx_r
        x0 = (ptsx - camx) * e0x + (ptsy - camy) * e0y
        
        if beta_max is not None:
            x0[ind] = np.nan
            
        dcam['x0'] = x0
        dcam['cent_r'] = np.r_[camx, camy]
        dcam['nin_r'] = np.r_[ninx_r, niny_r]

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
    beta_max=None,
    # computed
    ap=None,
    crystx=None,
    crysty=None,
    endx=None,
    endy=None,
    ilamb_min=None,
    ilamb_max=None,
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
        ninx, niny = dcam['nin_r'][:2]
        ninn = np.sqrt(ninx**2 + niny**2)
        ninx, niny = ninx/ninn, niny/ninn
        e0x, e0y = -niny, ninx
        e0n = np.sqrt(e0x**2 + e0y**2)
        e0x, e0y = e0x/e0n, e0y/e0n
        clen = dcam['length']
        camx = dcam['cent_r'][0] + 0.5*clen*np.r_[-1, 1] * e0x
        camy = dcam['cent_r'][1] + 0.5*clen*np.r_[-1, 1] * e0y

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
                label=(
                    f"r = {rcurve[ii]} m\t"
                    + r"$\lambda_0$" + f" = {lamb0[ii]*1e10:5.3f} AA\t"
                    + r"$\beta_0$" + f" = {bragg0[ii]*180/np.pi:5.2f} deg"
                ),
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
                dcam['x0'][ilamb_min[ii], ii],
                ii + 1 - 0.1,
                f'{lamb_min[ii]*1e10:2.3} AA',
                color=color,
                size=8,
                horizontalalignment='center',
                verticalalignment='top',
            )

            ax.text(
                dcam['x0'][ilamb_max[ii], ii],
                ii + 1 - 0.1,
                f'{lamb_max[ii]*1e10:2.3} AA',
                color=color,
                size=8,
                horizontalalignment='center',
                verticalalignment='top',
            )

    # ---------------
    # plot input data

    kax = 'hor'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']
        ax.legend(fontsize=12)
        
    if beta_max is None:
        beta_str = 'None'
    else:
        beta_str = f'{beta_max*180/np.pi:5.3} deg'
    
    msg = (
        f"beta_max = {beta_str}\n"
    )

    ax.text(
        0.8,
        0.4,
        msg,
        color='k',
        size=10,
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.figure.transFigure,
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
