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

    # d2
    d2 = lamb0 / np.sin(bragg0)

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
    dout = {
        'cryst0': np.full(shape, np.nan),
        'cryst1': np.full(shape, np.nan),
        'vn0': np.full(shape, np.nan),
        'vn1': np.full(shape, np.nan),
        'end0': np.full(shape, np.nan),
        'end1': np.full(shape, np.nan),
        'lamb': np.full(shape, np.nan),
        'cam_coord': np.full(shape, np.nan),
        'dmask': {
            'semi_angle_max': np.ones(shape, dtype=bool),
            'crystal': np.ones(shape, dtype=bool),
            'camera': np.ones(shape, dtype=bool),
            'lamb0': np.zeros(shape, dtype=bool),
        }
    }

    # ----------------
    # sample rays on crytals
    # ----------------

    # sample + 2 extra points
    kpts = np.linspace(-1., 1., npts-2)
    dk = kpts[1] - kpts[0]
    kpts = np.r_[kpts[0] - dk, kpts, kpts[-1] + dk]

    # crystal types
    lif = [
        (iflat, _compute_flat),
        (icurve, _compute_curve),
        (ispiral, _compute_spiral),
    ]

    for ind, func in lif:
        if np.any(ind):
            (
                dout['cryst0'][:, ind],
                dout['cryst1'][:, ind],
                dout['vn0'][:, ind],
                dout['vn1'][:, ind],
                dout['mask']['crystal'][:, ind],
            ) = func(
                csummit0=csummit0[ind],
                csummit1=csummit1[ind],
                bragg0=bragg0[ind],
                ex0=ex0[ind],
                ex1=ex1[ind],
                ey0=ey0[ind],
                ey1=ey1[ind],
                length=length[ind],
                kpts=kpts,
                # curved
                rcurve=rcurve[ind],
                # spiral
                ap0=ap0[ind],
                ap1=ap1[ind],
                dist_from_ap=dist_from_ap[ind],
                varrad_b=varrad_b[ind],
                # lamb_min, lamb_max
                d2=d2[ind],
                dist_from_ap=dist_from_ap[ind],
                lamb0_min=lamb0_min[ind],
                lamb0_max=lamb0_max[ind],
                npts=npts,
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
    dout['lamb'] = d2 * np.sin(bragg)

    # ---------------------
    # lamb0_min, lamb0_max
    # ---------------------

    dlamb = np.mean(np.diff(lamb, axis=0), axis=0)
    import pdb; pdb.set_trace()     # DB

    ilamb0 = np.abs(lamb - lamb0[None, ...]) < 0.1 * dlamb[None, ...]
    assert ilamb0.sum() == 1

    ilamb0_min = np.abs(lamb - lamb0_min[None, ...]) < 0.1 * dlamb[None, ...]
    assert ilamb0_min.sum() == 1

    ilamb0_max = np.abs(lamb - lamb0_max[None, ...]) < 0.1 * dlamb[None, ...]
    assert ilamb0_max.sum() == 1
    import pdb; pdb.set_trace()     # DB

    dout['dmask']['lamb0'] = ilamb0 | ilamb0_min | ilamb0_max

    # ----------------
    # intersection with camera plane
    # ----------------

    (
        dout['end0'],
        dout['end1'],
        dout['cam_coord'],
        dout['dmask']['camera'],
    ) = _camera_plane(
        ex0=ex0,
        ex1=ex1,
        cryst0=cryst0,
        cryst1=cryst1,
        vr0=vr0,
        vr1=vr1,
        cam_c0=cam_c0,
        cam_c1=cam_c1,
        cam_nin0=cam_nin0,
        cam_nin1=cam_nin1,
    )

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
        dout['dmask']['semi_angle_max'] = ~iout

    return dout


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
    d2=None,
    dist_from_ap=None,
    lamb0_min=None,
    lamb0_max=None,
    npts=None,
    # unused
    **kwdargs,
):

    # --------------------
    # lamb_min, max
    # --------------------

    imin = np.isfinite(lamb0_min)
    k_lambmin = np.full(lamb0_min.shape, np.nan)
    bragg0_min = np.arcsin(lamb0_min[imin] / d2[imin])
    k_lambmin[imin] = (
        dist_from_ap[imin]
        * np.sin(bragg0[imin] - bragg0_min) / np.sin(bragg0_min)
    )

    imax = np.isfinite(lamb0_max)
    k_lambmax = np.full(lamb0_max.shape, np.nan)
    bragg0_max = np.arcsin(lamb0_max[imax] / d2[imax])
    k_lambmax[imax] = (
        dist_from_ap[imax]
        * np.sin(bragg0[imax] - bragg0_max) / np.sin(bragg0_max)
    )

    kmin = np.nanmin([-0.5 * length, k_lambmin, k_lambmax], axis=0)
    kmax = np.nanmax([0.5 * length, k_lambmin, k_lambmax], axis=0)
    kk = np.linspace(kmin, kmax, npts, axis=0)

    mask_cryst = np.abs(kk) <= 0.5 * length

    # --------------------
    # prepare
    # --------------------

    # crystal plotting - straight
    estraight0 = np.cos(bragg0) * ex0 + np.sin(bragg0) * ey0
    estraight1 = np.cos(bragg0) * ex1 + np.sin(bragg0) * ey1

    # pts on crystal surface
    cryst0 = csummit0[None, ...] + kk * estraight0[None, ...]
    cryst1 = csummit1[None, ...] + kk * estraight1[None, ...]

    # local normal vectors
    vn0 = -estraight1
    vn1 = estraight0

    return cryst0, cryst1, vn0, vn1, mask_cryst


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
#               camera plane
# #################################################################


def _camera_plane(
    ex0=None,
    ex1=None,
    cryst0=None,
    cryst1=None,
    vr0=None,
    vr1=None,
    cam_c0=None,
    cam_c1=None,
    cam_nin0=None,
    cam_nin1=None,
    cam_length=None,
):

    # -----------
    # end points
    # -----------

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

    # ----------
    # mask
    # ----------

    mask_cam = np.abs(cam_coord) < cam_length * 0.5

    return end0, end1, cam_coord, mask_cam
