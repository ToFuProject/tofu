# -*- coding: utf-8 -*-


import numpy as np


from ._spectralrange2d_check import main as _check
from ._spectralrange2d_plot import main as _plot


# #################################################################
# #################################################################
#               Main
# #################################################################


def spectral_range_2d(
    # optics
    dap=None,
    dcrystals=None,
    dcam=None,
    # matching
    dmatch=None,
    # geometry basis
    beta_max=None,
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
    varrad: for variable-radii spiral

    """

    # -------------
    # check inputs
    # -------------

    (
        dap, dcrystals, dcam, dmatch,
        plot, save, pfe_fig, pfe_npz,
    ) = _check(**locals())

    # --------------
    # prepare
    # --------------

    for k0, v0 in dcrystals.items():
        dcrystals[k0]['d2'] = v0['lamb0'] / np.sin(v0['bragg0'])
        csummit = dapi['cent'] + dcrysti['dist_from_ap'] * dapi['ex']

    # --------------
    # compute
    # --------------

    _compute(
        dap=dap,
        dcrystals=dcrystals,
        dcam=dcam,
        dmatch=dmatch,
    )


    crystx, crysty, endx, endy, lamb = _compute(
        npts=npts,
        beta_max=beta_max,
        dcam=dcam,
        key_crystals=key_crystals,
        **din,
    )

    # -------------
    # format output

    ilamb_min = np.full((lamb.shape[1],), -1)
    ilamb_max = np.full((lamb.shape[1],), -1)
    iok = np.any(np.isfinite(lamb), axis=0)
    ilamb_min[iok] = np.nanargmin(lamb[:, iok], axis=0)
    ilamb_max[iok] = np.nanargmax(lamb[:, iok], axis=0)

    lamb_min = np.array([
        lamb[imin, ii] if imin >= 0 else np.nan
        for ii, imin in enumerate(ilamb_min)
    ])
    lamb_max = np.array([
        lamb[imax, ii] if imax >= 0 else np.nan
        for ii, imax in enumerate(ilamb_max)
    ])

    dout = dict(din)
    dout.update({
        'key_crystals': key_crystals,
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
#               Compute
# #################################################################


def _compute(
    dap=None,
    dcrystals=None,
    dcam=None,
    dmatch=None,
):

    # ------------
    # loop on match
    # ------------

    for i0, (k0, v0) in enumerate(dmatch.items()):

        kap = v0['keys']['aperture']
        kcryst = v0['keys']['crystal']
        kcam = v0['keys']['cam']

        dapi = dap[kap]
        dcrysti = dcrystals[kcryst]
        dcami = dam[kcam]

        # -----------------
        # summit of crystal

        csummit = dapi['cent'] + dcrysti['dist_from_ap'] * dapi['ex']



        # -----------------
        # sort by crystal type

        # flat crystals
        if np.isinf(dcrysti['rcurve']):
            _compute_flat()

        # variable radii crystals
        elif np.isfinite(dcrysti['varrad_b']):
            _compute_varrad()

        # curved crystals
        else:
            _compute_curved()




    # ------------
    # initialize

    size = lamb0.size

    crystx = np.full((npts, size), np.nan)
    crysty = np.full((npts, size), np.nan)
    vnx = np.full((npts, size), np.nan)
    vny = np.full((npts, size), np.nan)

    # ----------------
    # compute geometry
    # ----------------

    # ------------------------
    # indices of crystal types

    # variable-radii sinusoidal spiral
    indb = np.isfinite(varrad_b)

    # indices of curved crystals
    indc = np.isfinite(rcurve) & (~indb)

    # flat
    indf = (~indc) & (~indb)

    # safety check
    if not np.all(np.sum([indc, indb, indf], axis=0) == 1):
        msg = (
            "Some undetermined 2d crystal shapes:\n"
            f"\t- indc = {indc}\n"
            f"\t- indb = {indb}\n"
            f"\t- indf = {indf}\n"
        )
        raise Exception(msg)


    # ----------------
    # compute rays
    # ----------------

    # vectors of incident rays
    vix = crystx - ap[0]
    viy = crysty - ap[1]
    vin = np.sqrt(vix**2 + viy**2)
    vix = vix / vin
    viy = viy / vin

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
    # -----------------

    if dcam is not None:
        ninx, niny = dcam['nin']
        ninn = np.sqrt(ninx**2 + niny**2)
        ninx, niny = ninx/ninn, niny/ninn

        # cam center
        if dcam['abs'] is True:

            camx_r = dcam['cent'][0]
            camy_r = dcam['cent'][1]
            camx = np.sum((dcam['cent'] - ap) * ex)
            camy = np.sum((dcam['cent'] - ap) * ey)

            ninx_r = ninx
            niny_r = niny
            ninx = ninx_r * ex[0] + niny_r * ex[1]
            niny = ninx_r * ey[0] + niny_r * ey[1]

        else:
            camx = dcam['cent'][0]
            camy = dcam['cent'][1]
            camx_r = ap[0] + camx * ex[0] + camy * ey[0]
            camy_r = ap[1] + camx * ex[1] + camy * ey[1]

            ninx_r = ninx * ex[0] + niny * ey[0]
            niny_r = ninx * ex[1] + niny * ey[1]

        # rays x0
        sca_up = (camx_r - crystx) * ninx_r + (camy_r - crysty) * niny_r
        sca_bot = vrx*ninx_r + vry*niny_r

        kk = sca_up / sca_bot
        ptsx = crystx + kk * vrx
        ptsy = crysty + kk * vry

        e0x = -niny_r
        e0y = ninx_r
        x0 = (ptsx - camx_r) * e0x + (ptsy - camy_r) * e0y

        if beta_max is not None:
            x0[ind] = np.nan

        dcam['x0'] = x0
        dcam['cent'] = np.r_[camx, camy]
        dcam['cent_r'] = np.r_[camx_r, camy_r]
        dcam['nin'] = np.r_[ninx, niny]
        dcam['nin_r'] = np.r_[ninx_r, niny_r]
        dcam['abs'] = False

    return crystx, crysty, endx, endy, lamb


# #################################################################
# #################################################################
#               Compute by crystal type
# #################################################################


def _compute_flat():

    # crystal plotting - straight
    estraightx = np.cos(bragg0)[indf] * ex[0] + np.sin(bragg0)[indf] * ey[0]
    estraighty = np.cos(bragg0)[indf] * ex[1] + np.sin(bragg0)[indf] * ey[1]

    ll = 0.5 * length[None, indf] * np.linspace(-1, 1, npts)[:, None]
    crystx[:, indf] = sx[None, indf] + ll*estraightx[None, :]
    crysty[:, indf] = sy[None, indf] + ll*estraighty[None, :]

    # local normal vectors
    vnx[:, indf] = -estraighty
    vny[:, indf] = estraightx

    return


def _compute_curved():

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

    # local normal vectors
    vnx[:, indc] = -ethetax
    vny[:, indc] = -ethetay
    return


def _compute_varrad():

    # main parameters
    gam0 = bragg0[indb]
    r0 = rcurve[indb]
    ix = ~np.isfinite(r0)
    r0[ix] = xx[indb][ix]
    b = varrad_b[indb]

    # local radius of curvature at center
    # rc0 = r0 / (b * np.sin(gam0))

    # dOMx = r / (b-1) * (cos(phi) / tan(gam) - sin(phi))
    # dOMy = r / (b-1) * (sin(phi) / tan(gam) + cos(phi))
    # dL = r/(b-1) * 1 / sin(gam)
    # dL ~ r0/(b-1) * 1/sin(gam0) * Dgam

    # half angular opening of crystal (approximative)
    # dgam = 0.5*length / rc0
    dgam = 1.1 * length[indb] * np.sin(gam0) * (b-1) / r0 / 2

    # gam
    gam = gam0[None, :] + dgam[None, :] * np.linspace(-1, 1, npts)[:, None]

    # r
    r = r0[None, :] * (np.sin(gam) / np.sin(gam0)[None, :])**(1/(b[None, :]-1))

    # phi
    phi = (gam - gam0[None, :]) / (b[None, :]-1)

    # pts on cryst
    crystx[:, indb] = (
        ap[0]
        + (xx[indb] - r0) * ex[0]
        + r * (np.cos(phi) * ex[0] + np.sin(phi) * ey[0])
    )
    crysty[:, indb] = (
        ap[1]
        + (xx[indb] - r0) * ex[1]
        + r * (np.cos(phi) * ex[1] + np.sin(phi) * ey[1])
    )

    # derivative
    c0 = r / (b[None, :] - 1)
    c1 = np.cos(gam) / np.sin(gam)
    dOMxx = c0 * (c1 * np.cos(phi) - np.sin(phi))
    dOMyy = c0 * (c1 * np.sin(phi) + np.cos(phi))
    dOMx = dOMxx * ex[0] + dOMyy * ey[0]
    dOMy = dOMxx * ex[1] + dOMyy * ey[1]

    # local normal vectors
    vnx[:, indb] = dOMy / np.sqrt(dOMx**2 + dOMy**2)
    vny[:, indb] = -dOMx / np.sqrt(dOMx**2 + dOMy**2)
    return


# #################################################################
# #################################################################
#               Compute - old
# #################################################################


# DEPRECATED
def _compute_old(
    key_crystals=None,
    # crystal
    lamb0=None,
    bragg0=None,
    # geometry basis
    beta_max=None,
    # geometry
    xx=None,
    length=None,
    rcurve=None,
    varrad_b=None,
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
    # ----------------

    # 2d
    d2 = lamb0 / np.sin(bragg0)

    # summit of crystal
    sx = ap[0] + xx * ex[0]
    sy = ap[1] + xx * ex[1]

    # ------------------------
    # indices of crystal types

    # variable-radii sinusoidal spiral
    indb = np.isfinite(varrad_b)

    # indices of curved crystals
    indc = np.isfinite(rcurve) & (~indb)

    # flat
    indf = (~indc) & (~indb)

    # safety check
    if not np.all(np.sum([indc, indb, indf], axis=0) == 1):
        msg = (
            "Some undetermined 2d crystal shapes:\n"
            f"\t- indc = {indc}\n"
            f"\t- indb = {indb}\n"
            f"\t- indf = {indf}\n"
        )
        raise Exception(msg)

    # ---------------------
    # curved crystals

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

    # local normal vectors
    vnx[:, indc] = -ethetax
    vny[:, indc] = -ethetay

    # -----------------------
    # flat crystals

    # crystal plotting - straight
    estraightx = np.cos(bragg0)[indf] * ex[0] + np.sin(bragg0)[indf] * ey[0]
    estraighty = np.cos(bragg0)[indf] * ex[1] + np.sin(bragg0)[indf] * ey[1]

    ll = 0.5 * length[None, indf] * np.linspace(-1, 1, npts)[:, None]
    crystx[:, indf] = sx[None, indf] + ll*estraightx[None, :]
    crysty[:, indf] = sy[None, indf] + ll*estraighty[None, :]

    # local normal vectors
    vnx[:, indf] = -estraighty
    vny[:, indf] = estraightx

    # -----------------------
    # variable radii crystals

    # main parameters
    gam0 = bragg0[indb]
    r0 = rcurve[indb]
    ix = ~np.isfinite(r0)
    r0[ix] = xx[indb][ix]
    b = varrad_b[indb]

    # local radius of curvature at center
    # rc0 = r0 / (b * np.sin(gam0))

    # dOMx = r / (b-1) * (cos(phi) / tan(gam) - sin(phi))
    # dOMy = r / (b-1) * (sin(phi) / tan(gam) + cos(phi))
    # dL = r/(b-1) * 1 / sin(gam)
    # dL ~ r0/(b-1) * 1/sin(gam0) * Dgam

    # half angular opening of crystal (approximative)
    # dgam = 0.5*length / rc0
    dgam = 1.1 * length[indb] * np.sin(gam0) * (b-1) / r0 / 2

    # gam
    gam = gam0[None, :] + dgam[None, :] * np.linspace(-1, 1, npts)[:, None]

    # r
    r = r0[None, :] * (np.sin(gam) / np.sin(gam0)[None, :])**(1/(b[None, :]-1))

    # phi
    phi = (gam - gam0[None, :]) / (b[None, :]-1)

    # pts on cryst
    crystx[:, indb] = (
        ap[0]
        + (xx[indb] - r0) * ex[0]
        + r * (np.cos(phi) * ex[0] + np.sin(phi) * ey[0])
    )
    crysty[:, indb] = (
        ap[1]
        + (xx[indb] - r0) * ex[1]
        + r * (np.cos(phi) * ex[1] + np.sin(phi) * ey[1])
    )

    # derivative
    c0 = r / (b[None, :] - 1)
    c1 = np.cos(gam) / np.sin(gam)
    dOMxx = c0 * (c1 * np.cos(phi) - np.sin(phi))
    dOMyy = c0 * (c1 * np.sin(phi) + np.cos(phi))
    dOMx = dOMxx * ex[0] + dOMyy * ey[0]
    dOMy = dOMxx * ex[1] + dOMyy * ey[1]

    # local normal vectors
    vnx[:, indb] = dOMy / np.sqrt(dOMx**2 + dOMy**2)
    vny[:, indb] = -dOMx / np.sqrt(dOMx**2 + dOMy**2)

    # ----------------
    # compute rays
    # ----------------

    # vectors of incident rays
    vix = crystx - ap[0]
    viy = crysty - ap[1]
    vin = np.sqrt(vix**2 + viy**2)
    vix = vix / vin
    viy = viy / vin

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
    # -----------------

    if dcam is not None:
        ninx, niny = dcam['nin']
        ninn = np.sqrt(ninx**2 + niny**2)
        ninx, niny = ninx/ninn, niny/ninn

        # cam center
        if dcam['abs'] is True:

            camx_r = dcam['cent'][0]
            camy_r = dcam['cent'][1]
            camx = np.sum((dcam['cent'] - ap) * ex)
            camy = np.sum((dcam['cent'] - ap) * ey)

            ninx_r = ninx
            niny_r = niny
            ninx = ninx_r * ex[0] + niny_r * ex[1]
            niny = ninx_r * ey[0] + niny_r * ey[1]

        else:
            camx = dcam['cent'][0]
            camy = dcam['cent'][1]
            camx_r = ap[0] + camx * ex[0] + camy * ey[0]
            camy_r = ap[1] + camx * ex[1] + camy * ey[1]

            ninx_r = ninx * ex[0] + niny * ey[0]
            niny_r = ninx * ex[1] + niny * ey[1]

        # rays x0
        sca_up = (camx_r - crystx) * ninx_r + (camy_r - crysty) * niny_r
        sca_bot = vrx*ninx_r + vry*niny_r

        kk = sca_up / sca_bot
        ptsx = crystx + kk * vrx
        ptsy = crysty + kk * vry

        e0x = -niny_r
        e0y = ninx_r
        x0 = (ptsx - camx_r) * e0x + (ptsy - camy_r) * e0y

        if beta_max is not None:
            x0[ind] = np.nan

        dcam['x0'] = x0
        dcam['cent'] = np.r_[camx, camy]
        dcam['cent_r'] = np.r_[camx_r, camy_r]
        dcam['nin'] = np.r_[ninx, niny]
        dcam['nin_r'] = np.r_[ninx_r, niny_r]
        dcam['abs'] = False

    return crystx, crysty, endx, endy, lamb
