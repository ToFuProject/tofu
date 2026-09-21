import numpy as np


# #################################################################
# #################################################################
#               Compute
# #################################################################


def main(
    # aperture
    ap0=None,
    ap1=None,
    ex0=None,
    ex1=None,
    ey0=None,
    ey1=None,
    semi_angle_max=None,
    # crystal
    dist_from_ap=None,
    lamb0=None,
    bragg0=None,
    rcurve=None,
    length=None,
    varrad_b=None,
    lamb0_min=None,
    lamb0_max=None,
    # camera
    cam_c0=None,
    cam_c1=None,
    cam_nin0=None,
    cam_nin1=None,
    # options
    npts=None,
):

    # ------------
    # crystal's summit
    # ------------

    csummit0 = ap0 + dist_from_ap * ex0
    csummit1 = ap1 + dist_from_ap * ex1

    # -----------------
    # sort by crystal type
    # -----------------

    # variable-radii sinusoidal spiral
    ispiral = np.isfinite(varrad_b)

    # indices of curved crystals
    icurve = np.isfinite(rcurve) & (~ispiral)

    # flat
    iflat = (~icurve) & (~ispiral)

    # safety check
    if not np.all(np.sum([icurve, iflat, ispiral], axis=0) == 1):
        msg = (
            "Some undetermined 2d crystal shapes:\n"
            f"\t- iflat   = {iflat}\n"
            f"\t- icurve  = {icurve}\n"
            f"\t- ispiral = {ispiral}\n"
        )
        raise Exception(msg)

    # ----------------
    # initialize
    # ----------------

    shape = (npts,) + ex0.shape
    cryst0 = np.full(shape, np.nan)
    cryst1 = np.full(shape, np.nan)
    vn0 = np.full(shape, np.nan)
    vn1 = np.full(shape, np.nan)
    lamb = np.full(shape, np.nan)

    # ----------------
    # sample rays on crytals
    # ----------------

    kpts = np.linspace(-1, 1, npts)
    lif = [
        (iflat, _compute_flat),
        (icurve, _compute_curve),
        (ispiral, _compute_spiral),
    ]

    for ind, func in lif:
        if np.any(ind):
            (
                cryst0[:, ind], cryst1[:, ind],
                vn0[:, ind], vn1[:, ind],
            ) = func(
                csummit0=csummit0[ind],
                csummit1=csummit1[ind],
                bragg0=bragg0[ind],
                ex0=ex0[ind],
                ex1=ex1[ind],
                ey0=ey0[ind],
                ey1=ey1[ind],
                length=length[ind],
                kpts=kpts[ind],
                # curved
                rcurve=rcurve[ind],
                # spiral
                ap0=ap0[ind],
                ap1=ap1[ind],
                dist_from_ap=dist_from_ap[ind],
                varrad_b=varrad_b[ind],
                # lamb_min, lamb_max
                lamb0_min=lamb0_min[ind],
                lamb0_max=lamb0_max[ind],
            )

    # ----------------
    # compute rays
    # ----------------

    # vectors of incident rays
    vi0 = cryst0 - ap0
    vi1 = cryst1 - ap1
    vin = np.sqrt(vi0**2 + vi1**2)
    vi0 = vi0 / vin
    vi1 = vi1 / vin

    # reflected vectors
    sca = vi0*vn0 + vi1*vn1
    vr0 = vi0 - 2.*sca*vn0
    vr1 = vi1 - 2.*sca*vn1

    # ----------------------
    # compute spectral range
    # ----------------------

    # get local bragg angle - top and bottom
    bragg = np.arccos(sca) - np.pi/2.

    # lamb
    d2 = lamb0 / np.sin(bragg0)
    lamb = d2 * np.sin(bragg)

    # ----------------
    # intersection with camera plane
    # ----------------

    kk = (
        (cam_c0 - cryst0) * cam_nin0
        + (cam_c1 - cryst1) * cam_nin1
    ) / (vr0 * cam_nin0 + vr1 * cam_nin1)

    # end of rays at camera
    end0 = cryst0 + kk * vr0
    end1 = cryst1 + kk * vr1

    # -----------
    # coordinates on cameras
    # -----------

    # get lateral cam unit vector
    cam_e00 = -cam_nin1
    cam_e01 = cam_nin0
    ineg = (cam_e00 * ex0 + cam_e01 * ex1) < 0.
    cam_e00[ineg] = -cam_e00[ineg]
    cam_e01[ineg] = -cam_e01[ineg]

    cam_coord = (end0 - cam_c0) * cam_e00 + (end1 - cam_c1) * cam_e01

    # -----------
    # semi_angle_max
    # -----------

    iout = np.isfinite(semi_angle_max)
    if np.any(iout):
        dv0 = (cryst0 - ap0)[iout]
        dv1 = (cryst1 - ap1)[iout]
        semi_angle = np.arctan2(
            dv0*ey0[iout] + dv1*ey1[iout],
            dv0*ex0[iout] + dv1*ex1[iout],
        )
        ind = np.abs(semi_angle) <= semi_angle_max[iout]
        iout[ind] = False
        end0[iout] = np.nan
        end1[iout] = np.nan
        lamb[iout] = np.nan

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

    return {
        'cryst0': cryst0,
        'cryst1': cryst1,
        'end0': end0,
        'end1': end1,
        'lamb': lamb,
        'cam_coord': cam_coord,
    }


# #################################################################
# #################################################################
#               Compute by crystal type
# #################################################################


def _compute_flat(
    csummit0=None,
    csummit1=None,
    bragg0=None,
    ex0=None,
    ex1=None,
    ey0=None,
    ey1=None,
    length=None,
    kpts=None,
    # lamb_min, max
    lamb0_min=None,
    lamb0_max=None,
    # unused
    **kwdargs,
):

    # --------------------
    # prepare
    # --------------------

    # crystal plotting - straight
    estraight0 = np.cos(bragg0) * ex0 + np.sin(bragg0) * ey0
    estraight1 = np.cos(bragg0) * ex1 + np.sin(bragg0) * ey1

    # sample length of crystal
    sli = (slice(None),) + (None,) * ex0.ndim
    ll = 0.5 * length[None, ...] * kpts[sli]

    # pts on crystal surface
    cryst0 = csummit0[None, ...] + ll * estraight0[None, ...]
    cryst1 = csummit1[None, ...] + ll * estraight1[None, ...]

    # local normal vectors
    vn0 = -estraight1
    vn1 = estraight0

    # --------------------
    # lamb0_min, lamb0_max
    # --------------------

    iok = np.isfinite(lamb0_min)
    if np.any(lamb0_min):
        k_l0min = None

    return cryst0, cryst1, vn0, vn1


def _compute_curve(
    csummit0=None,
    csummit1=None,
    bragg0=None,
    ex0=None,
    ex1=None,
    ey0=None,
    ey1=None,
    rcurve=None,
    length=None,
    kpts=None,
    # unused
    **kwdargs,
):

    # center of curvature
    ec0 = np.sin(bragg0) * ex0 - np.cos(bragg0) * ey0
    ec1 = np.sin(bragg0) * ex1 - np.cos(bragg0) * ey1
    ec0_p = -ec1
    ec1_p = ec0

    # crystal center of curvature
    cc0 = csummit0 - rcurve * ec0
    cc1 = csummit1 - rcurve * ec1

    # half angular opening of crystal
    dalpha = 0.5 * length / rcurve
    sli = (slice(None),) + (None,) * ex0.ndim
    theta = dalpha * kpts[sli]

    # crystal plotting - curved
    etheta0 = np.cos(theta) * ec0[None, :] + np.sin(theta) * ec0_p[None, :]
    etheta1 = np.cos(theta) * ec1[None, :] + np.sin(theta) * ec1_p[None, :]

    cryst0 = cc0[None, :] + rcurve[None, :] * etheta0
    cryst1 = cc1[None, :] + rcurve[None, :] * etheta1

    # local normal vectors
    vn0 = -etheta0
    vn1 = -etheta1

    return cryst0, cryst1, vn0, vn1


def _compute_spiral(
    csummit0=None,
    csummit1=None,
    bragg0=None,
    ex0=None,
    ex1=None,
    ey0=None,
    ey1=None,
    rcurve=None,
    length=None,
    kpts=None,
    dist_from_ap=None,
    varrad_b=None,
    ap0=None,
    ap1=None,
    # unused
    **kwdargs,
):

    # main parameters
    r0 = rcurve
    ix = ~np.isfinite(r0)
    r0[ix] = dist_from_ap[ix]

    # local radius of curvature at center
    # rc0 = r0 / (b * np.sin(bragg0))

    # dOMx = r / (b-1) * (cos(phi) / tan(gam) - sin(phi))
    # dOMy = r / (b-1) * (sin(phi) / tan(gam) + cos(phi))
    # dL = r/(b-1) * 1 / sin(gam)
    # dL ~ r0/(b-1) * 1/sin(bragg0) * Dgam

    # half angular opening of crystal (approximative)
    # dgam = 0.5*length / rc0
    dgam = 1.1 * length * np.sin(bragg0) * (b-1) / r0 / 2

    # gam
    sli = (slice(None),) + (None,) * ex0.ndim
    gam = bragg0[None, ...] + dgam[None, ...] * kpts[sli]

    # rr
    rr = (
        r0[None, ...]
        * (np.sin(gam) / np.sin(bragg0)[None, ...])**(
            1 / (varrad_b[None, ...] - 1)
        )
    )

    # phi
    phi = (gam - bragg0[None, ...]) / (varrad_b[None, ...] - 1)

    # pts on cryst
    cryst0 = (
        ap0
        + (dist_from_ap - r0) * ex0
        + rr * (np.cos(phi) * ex0 + np.sin(phi) * ey0)
    )
    cryst1 = (
        ap1
        + (dist_from_ap - r0) * ex1
        + rr * (np.cos(phi) * ex1 + np.sin(phi) * ey1)
    )

    # derivative
    c0 = rr / (varrad_b[None, ...] - 1.)
    c1 = np.cos(gam) / np.sin(gam)
    dOMxx = c0 * (c1 * np.cos(phi) - np.sin(phi))
    dOMyy = c0 * (c1 * np.sin(phi) + np.cos(phi))
    dOMx = dOMxx * ex0 + dOMyy * ey0
    dOMy = dOMxx * ex1 + dOMyy * ey1

    # local normal vectors
    vn0 = dOMy / np.sqrt(dOMx**2 + dOMy**2)
    vn1 = -dOMx / np.sqrt(dOMx**2 + dOMy**2)

    return cryst0, cryst1, vn0, vn1


# #################################################################
# #################################################################
#               Compute - old
# #################################################################


# DEPRECATED
def _compute_old(
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
