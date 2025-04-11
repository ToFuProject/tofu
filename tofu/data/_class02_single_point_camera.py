

import numpy as np
import scipy.spatial as scpsp
from matplotlib.path import Path
import datastock as ds


# ########################################################
# ########################################################
#              DEFAULTS
# ########################################################


_DANGLES = {
    'angle0': {
        'nb': 101,
        'lim': np.r_[-1, 1] * 90,
        'units': 'deg',
    },
    'angle1': {
        'nb': 101,
        'lim': np.r_[-1, 1] * 90,
        'units': 'deg',
    },
}


# ########################################################
# ########################################################
#              Main
# ########################################################


def main(
    coll=None,
    key=None,
    # from rays
    key_rays=None,
    segment=None,
    # from user-defined
    cent=None,
    nin=None,
    e0=None,
    e1=None,
    # angles
    angle0=None,
    angle1=None,
    config=None,
    strict=None,
    # optional naming
    key_angle0=None,
    key_angle1=None,
    ref_angle0=None,
    ref_angle1=None,
    units_angles=None,
):

    # -------------
    # check inputs
    # -------------

    key, cent, nin, e0, e1, dangles, strict = _check(
        coll=coll,
        key=key,
        # from rays
        key_rays=key_rays,
        segment=segment,
        # user-defined
        cent=cent,
        nin=nin,
        e0=e0,
        e1=e1,
        # angles
        angle0=angle0,
        angle1=angle1,
        strict=strict,
        # optional naming
        key_angle0=key_angle0,
        key_angle1=key_angle1,
        ref_angle0=ref_angle0,
        ref_angle1=ref_angle1,
        units_angles=units_angles,
    )

    # -------------
    # prepare
    # -------------

    # -----------
    # angles ref

    for k0, v0 in dangles.items():
        if v0['ref'] not in coll.dref.keys():
            coll.add_ref(v0['ref'], size=v0['data'].size)

    # ------------
    # angles data

    for k0, v0 in dangles.items():
        if v0['key'] not in coll.ddata.keys():
            coll.add_data(**v0)

    # -------------
    # unit vectors
    # -------------

    # angles, full shape
    angle0f = dangles['angle0']['data'][:, None]
    if dangles['angle0']['units'] == 'deg':
        angle0f = angle0f * np.pi/180.

    angle1f = dangles['angle1']['data'][None, :]
    if dangles['angle1']['units'] == 'deg':
        angle1f = angle1f * np.pi/180.

    # unit vectors
    vx = (
        np.cos(angle1f)
        * (np.cos(angle0f) * (nin[0]) + np.sin(angle0f) * e0[0])
        + np.sin(angle1f) * e1[0]
    )
    vy = (
        np.cos(angle1f)
        * (np.cos(angle0f) * (nin[1]) + np.sin(angle0f) * e0[1])
        + np.sin(angle1f) * e1[1]
    )
    vz = (
        np.cos(angle1f)
        * (np.cos(angle0f) * (nin[2]) + np.sin(angle0f) * e0[2])
        + np.sin(angle1f) * e1[2]
    )

    # -------------
    # compute
    # -------------

    coll.add_rays(
        key,
        start_x=np.full(vx.shape, cent[0]),
        start_y=np.full(vx.shape, cent[1]),
        start_z=np.full(vx.shape, cent[2]),
        # vectors
        vect_x=vx,
        vect_y=vy,
        vect_z=vz,
        # ref
        ref=(dangles['angle0']['ref'], dangles['angle1']['ref']),
        # specific to ptcam
        angles=(dangles['angle0']['key'], dangles['angle1']['key']),
        nin=nin,
        e0=e0,
        e1=e1,
        # config
        config=config,
        strict=strict,
    )

    return


# ########################################################
# ########################################################
#              Check
# ########################################################


def _check(
    coll=None,
    key=None,
    # from rays
    key_rays=None,
    segment=None,
    # from user-defined
    cent=None,
    nin=None,
    e0=None,
    e1=None,
    angle0=None,
    angle1=None,
    strict=None,
    # optional naming
    key_angle0=None,
    key_angle1=None,
    ref_angle0=None,
    ref_angle1=None,
    units_angles=None,
):

    # --------------
    # key
    # --------------

    wrays = coll._which_rays
    lout = list(coll.dobj.get(wrays, {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        default='ptcam',
        excluded=lout,
    )

    # ---------------------
    # rays vs user-defined
    # ---------------------

    lc = [
        key_rays is not None,
        cent is not None,
    ]
    if np.sum(lc) != 1:
        msg = (
            "Please provide either key_rays xor cent!\n"
            f"\t- key_rays: {key_rays}\n"
            f"\t- cent: {cent}\n"
        )
        raise Exception(msg)

    # --------------
    # key_rays
    # --------------

    if key_rays is not None:

        if isinstance(key_rays, str):
            key_rays = [key_rays]

        # key_rays
        lok = list(coll.dobj.get(wrays, {}).keys())
        key_rays = ds._generic_check._check_var_iter(
            key_rays, 'key_rays',
            types=list,
            types_iter=str,
            allowed=lok,
        )

        # segment
        nseg = [coll.dobj[wrays][kk]['shape'][0] for kk in key_rays]
        segment = ds._generic_check._check_var(
            segment, 'segment',
            types=int,
            default=0,
            allowed=list(range(np.min(nseg))),
        )

        # derive cent
        cent, nini = _rays_intersection(
            coll,
            key_rays=key_rays,
            segment=segment,
        )

        # nin
        if nin is None:
            nin = nini

    # --------------
    # cent
    # --------------

    else:

        cent = ds._generic_check._check_flat1darray(
            cent, 'cent',
            dtype=float,
            size=3,
        )

    # --------------
    # unit vectors
    # --------------

    nin, e0, e1 = ds._generic_check._check_vectbasis(
        e0=nin,
        e1=e0,
        e2=e1,
        dim=3,
        direct=False,
    )

    # --------------
    # angle0, angle1 angles
    # --------------

    dangles = {
        'angle0': _check_angles(
            coll=coll,
            key=key,
            ang=angle0,
            ang_name='angle0',
            ang_key=key_angle0,
            ang_ref=ref_angle0,
        ),
        'angle1': _check_angles(
            coll=coll,
            key=key,
            ang=angle1,
            ang_name='angle1',
            ang_key=key_angle1,
            ang_ref=ref_angle1,
        ),
    }

    # --------------
    # options
    # --------------

    # strict
    strict = ds._generic_check._check_var(
        strict, 'strict',
        types=bool,
        default=False,
    )

    return key, cent, nin, e0, e1, dangles, strict


def _check_angles(
    coll=None,
    key=None,
    ang=None,
    ang_name=None,
    ang_key=None,
    ang_ref=None,
    ang_units=None,
):

    # --------------
    # existing key
    # --------------

    if isinstance(ang, str):
        lok = [
            k0 for k0, v0 in coll.ddata.items()
            if v0['monot'] == (True,)
        ]
        ang = ds._generic_check._check_var(
            ang, ang_name,
            types=str,
            allowed=lok,
        )

        ang_key = ang
        ang_data = coll.ddata[ang_key]['data']
        ang_ref = coll.ddata[ang_key]['ref'][0]
        ang_units = coll.ddata[ang_key]['units']

    # --------------
    # int, float, array
    # --------------

    else:

        if ang is None:
            ang = _DANGLES[ang_name]['nb']

        if isinstance(ang, (float, int)):
            ang = np.linspace(
                _DANGLES[ang_name]['lim'][0],
                _DANGLES[ang_name]['lim'][1],
                int(ang),
            )
            ang_units = _DANGLES[ang_name]['units']

        # --------------
        # flat unique array

        ang_data = ds._generic_check._check_flat1darray(
            ang, ang_name,
            dtype=float,
            unique=True,
        )

        # -------
        # keys

        lout = list(coll.ddata.keys())
        ang_key = ds._generic_check._check_var(
            ang_key, ang_name,
            types=str,
            default=f"{key}_{ang_name}",
            excluded=lout,
        )

        # -------
        # ref

        lout = list(coll.dref.keys())
        ang_ref = ds._generic_check._check_var(
            ang_ref, f"{ang_name}",
            types=str,
            default=f"{key}_n{ang_name}",
            excluded=lout,
        )

        # -------
        # units

        ang_units = ds._generic_check._check_var(
            ang_units, "units_angles",
            types=str,
            default="deg",
            allowed=['deg', 'rad'],
        )

    return {
        'key': ang_key,
        'data': ang_data,
        'ref': ang_ref,
        'units': ang_units,
    }


# ##################################################
# ##################################################
#          Get rays intersection
# ##################################################


def _rays_intersection(
    coll,
    key_rays=None,
    segment=None,
):

    # ----------------
    # prepare
    # ----------------

    wrays = coll._which_rays
    nrays = len(key_rays)
    nrays_tot = np.array([
        np.prod(coll.dobj[wrays][kray]['shape'][1:]) for kray in key_rays
    ])

    cx = np.full((nrays,), np.nan)
    cy = np.full((nrays,), np.nan)
    cz = np.full((nrays,), np.nan)

    ux = np.full((nrays,), np.nan)
    uy = np.full((nrays,), np.nan)
    uz = np.full((nrays,), np.nan)

    # ----------------
    # loop on key_rays
    # ----------------

    for ir, kray in enumerate(key_rays):

        # starting points
        ptsx, ptsy, ptsz = coll.get_rays_pts(
            kray,
            segment=segment,
        )
        px = ptsx[0, ...]
        py = ptsy[0, ...]
        pz = ptsz[0, ...]

        # unit vectors
        vx, vy, vz = coll.get_rays_vect(
            kray,
            segment=segment,
        )

        # intersection
        shape = px.shape + px.shape
        interx = np.full(shape, np.nan)
        intery = np.full(shape, np.nan)
        interz = np.full(shape, np.nan)
        for i0, ind0 in enumerate(np.ndindex(px.shape)):
            for i1, ind1 in enumerate(np.ndindex(px.shape)):

                if i1 <= i0:
                    continue

                sli = ind0 + ind1
                interx[sli], intery[sli], interz[sli] = _intersect_line2line(
                    px[ind0], py[ind0], pz[ind0],
                    px[ind1], py[ind1], pz[ind1],
                    vx[ind0], vy[ind0], vz[ind0],
                    vx[ind1], vy[ind1], vz[ind1],
                )

        cx[ir] = np.nanmean(interx)
        cy[ir] = np.nanmean(intery)
        cz[ir] = np.nanmean(interz)

        ux[ir] = np.nanmean(vx)
        uy[ir] = np.nanmean(vy)
        uz[ir] = np.nanmean(vz)

    # --------------
    # take average
    # --------------

    ntot = np.sum(nrays_tot)
    cent = np.r_[
        np.sum(cx * nrays_tot) / ntot,
        np.sum(cy * nrays_tot) / ntot,
        np.sum(cz * nrays_tot) / ntot,
    ]
    nin = np.r_[
        np.sum(ux * nrays_tot) / ntot,
        np.sum(uy * nrays_tot) / ntot,
        np.sum(uz * nrays_tot) / ntot,
    ]

    return cent, nin


def _intersect_line2line(
    px0, py0, pz0,
    px1, py1, pz1,
    vx0, vy0, vz0,
    vx1, vy1, vz1,
):

    # A0M = k0u0
    # A1N = k1u1
    # MN = -A0M + A0A1 + A1N
    # MN = -k0u0 + A0A1 + k1u1
    # MN.u0 = 0 = -k0 + A0A1.u0 + k1(u0.u1)
    # MN.u1 = 0 = -k0(u0.u1) + A0A1.u1 + k1
    #
    # k0 = A0A1.u0 + k1(u0.u1)
    # 0 = k1(1 - (u0.u1)^2) - (A0A1.u0)(u0.u1) + A0A1.u1
    # k1 = ((A0A1.u0)(u0.u1) - A0A1.u1) / (1 - (u0.u1)^2)

    A0A1_x = px1 - px0
    A0A1_y = py1 - py0
    A0A1_z = pz1 - pz0

    A0A1_u0 = A0A1_x * vx0 + A0A1_y * vy0 + A0A1_z * vz0
    A0A1_u1 = A0A1_x * vx1 + A0A1_y * vy1 + A0A1_z * vz1
    u0_u1 = vx0 * vx1 + vy0 * vy1 + vz0 * vz1

    k1 = ((A0A1_u0) * (u0_u1) - A0A1_u1) / (1. - (u0_u1)**2)
    k0 = A0A1_u0 + k1 * u0_u1

    # halfway
    interx = 0.5 * (px0 + k0 * vx0 + px1 + k1 * vx1)
    intery = 0.5 * (py0 + k0 * vy0 + py1 + k1 * vy1)
    interz = 0.5 * (pz0 + k0 * vz0 + pz1 + k1 * vz1)

    return interx, intery, interz


# ########################################################
# ########################################################
#              Get rays angles
# ########################################################


def _get_rays_angles(
    coll=None,
    key_single_pt_cam=None,
    # rays to get angles of
    key_rays=None,
    segment=None,
    # max tolerance
    tol_radius=None,
    # optional indices and convex
    return_indices=None,
    convex_axis=None,
    # verb
    verb=None,
):

    # -------------
    # check inputs
    # -------------

    (
        key_single_pt_cam, key_rays,
        segment, tol_radius,
        return_indices, convex_axis,
        verb,
    ) = _check_rays_angles(
        coll=coll,
        key_single_pt_cam=key_single_pt_cam,
        key_rays=key_rays,
        segment=segment,
        tol_radius=tol_radius,
        return_indices=return_indices,
        convex_axis=convex_axis,
        verb=verb,
    )

    # -------------
    # prepare
    # -------------

    wrays = coll._which_rays
    ref = coll.dobj[wrays][key_rays]['ref'][1:]
    shape = coll.dobj[wrays][key_rays]['shape'][1:]
    kang0, kang1 = coll.dobj[wrays][key_single_pt_cam]['angles']
    units = coll.ddata[kang0]['units']

    # unit vectors
    nin = coll.dobj[wrays][key_single_pt_cam]['nin']
    e0 = coll.dobj[wrays][key_single_pt_cam]['e0']
    e1 = coll.dobj[wrays][key_single_pt_cam]['e1']

    ptsx, ptsy, ptsz = coll.get_rays_pts(key_single_pt_cam)
    cent = np.r_[ptsx[0, 0, 0], ptsy[0, 0, 0], ptsz[0, 0, 0]]

    # -------------
    # get impact factor vs tol_radius
    # -------------

    ptsx, ptsy, ptsz = coll.get_rays_pts(key_rays, segment=segment)
    vx, vy, vz = coll.get_rays_vect(key_rays, segment=segment, norm=True)

    p0c_x = (ptsx[0, ...] - cent[0])
    p0c_y = (ptsy[0, ...] - cent[1])
    p0c_z = (ptsz[0, ...] - cent[2])

    crossx = p0c_y * vz - p0c_z * vy
    crossy = p0c_z * vx - p0c_x * vz
    crossz = p0c_x * vy - p0c_y * vx

    impact = np.sqrt(crossx**2 + crossy**2 + crossz**2)

    # verb
    if verb is True:
        msg = (
            "Angles computation for rays, seen from single_point_camera2d:\n"
            f"\t- single_point_camera2d: {key_single_pt_cam}\n"
            f"\t- key_rays: {key_rays}\n"
            f"\t- max impact: {np.max(impact)} m\n"
        )
        print(msg)

    # -------------
    # compute
    # -------------

    # intialize
    ang0 = np.full(shape, np.nan)
    ang1 = np.full(shape, np.nan)

    # vect from cent
    vx = ptsx[1, ...] - cent[0]
    vy = ptsy[1, ...] - cent[1]
    vz = ptsz[1, ...] - cent[2]
    vnorm = np.sqrt(vx**2 + vy**2 + vz**2)
    vx = vx / vnorm
    vy = vy / vnorm
    vz = vz / vnorm

    # derive angles
    sin1 = vx * e1[0] + vy * e1[1] + vz * e1[2]
    cos1_sin0 = vx * e0[0] + vy * e0[1] + vz * e0[2]
    cos1_cos0 = vx * nin[0] + vy * nin[1] + vz * nin[2]
    cos12 = cos1_sin0**2 + cos1_cos0**2
    cos1 = np.sqrt(cos12)
    ang1 = np.arctan2(sin1, cos1)
    ang0 = np.arctan2(cos1_sin0 / cos1, cos1_cos0 / cos1)

    # adjust units
    if str(units) == 'deg':
        ang0 = ang0 * 180/np.pi
        ang1 = ang1 * 180/np.pi

    # -------------
    # indices
    # -------------

    if return_indices is True:

        bin0 = coll.ddata[kang0]['data']
        dang0 = bin0[1] - bin0[0]
        bin0 = 0.5*(bin0[1:] + bin0[:-1])
        bin0 = np.r_[bin0[0]-dang0, bin0]

        bin1 = coll.ddata[kang0]['data']
        dang1 = bin1[1] - bin1[0]
        bin1 = 0.5*(bin1[1:] + bin1[:-1])
        bin1 = np.r_[bin1[0]-dang1, bin1]

        ind0 = np.searchsorted(bin0, ang0) - 1
        ind1 = np.searchsorted(bin1, ang1) - 1

        # adjust for edges
        ind0[ind0 < 0] = 0
        ind1[ind1 < 0] = 0

        # -------
        # convex_axis

        if convex_axis is not False:

            # shape_angles
            shape_angles = coll.dobj[wrays][key_single_pt_cam]['shape'][1:]

            # shape
            shape_c = tuple([
                ss for ii, ss in enumerate(shape)
                if ii not in convex_axis
            ])

            # unique call
            if len(shape_c) == 0:
                sli = tuple([slice(None) for ii in shape])
                dhull = {
                    (0,): _convexhull(
                        shape_angles=shape_angles,
                        ind0=ind0,
                        ind1=ind1,
                        sli=sli,
                    )
                }

            else:
                # initialize
                dhull = {}
                sli = np.array([
                    slice(None) if ii in convex_axis else 0
                    for ii in range(len(shape))
                ])
                isli = np.array([
                    ii for ii in range(len(shape))
                    if ii not in convex_axis
                ])

                i0 = np.arange(shape_angles[0])
                i1 = np.arange(shape_angles[1])
                i01_full = np.array([
                    np.repeat(i0[:, None], i1.size, axis=1).ravel(),
                    np.repeat(i1[None, :], i0.size, axis=0).ravel(),
                ]).T

                # loop
                for ii, ind in enumerate(np.ndindex(shape_c)):

                    # update slice
                    sli[isli] = ind
                    slii = tuple(sli)

                    # get convex_axis hull
                    dhull[ind] = _convexhull(
                        shape_angles=shape_angles,
                        ind0=ind0,
                        ind1=ind1,
                        sli=slii,
                        i01_full=i01_full,
                    )

    # -------------
    # output
    # -------------

    dout = {
        'angle0': {
            'key': f'{key_rays}_{key_single_pt_cam}_ang0',
            'data': ang0,
            'units': units,
            'ref': ref,
            'dim': 'angle',
        },
        'angle1': {
            'key': f'{key_rays}_{key_single_pt_cam}_ang1',
            'data': ang1,
            'units': units,
            'ref': ref,
            'dim': 'angle',
        },
        'impact': {
            'key': f'{key_rays}_{key_single_pt_cam}_impact',
            'data': impact,
            'ref': ref,
            'units': 'm',
            'dim': 'distance',
        },
        'indices': {},
    }

    if return_indices:
        dout['ind0'] = {
            'key': f'{key_rays}_{key_single_pt_cam}_ind0',
            'data': ind0,
            'ref': ref,
        }
        dout['ind1'] = {
            'key': f'{key_rays}_{key_single_pt_cam}_ind1',
            'data': ind1,
            'ref': ref,
        }

        if convex_axis is not False:
            dout['hull'] = dhull

    return dout


# ########################################################
# ########################################################
#              check inputs rays angles
# ########################################################


def _check_rays_angles(
    coll=None,
    key_single_pt_cam=None,
    # rays to get angles of
    key_rays=None,
    segment=None,
    # max tolerance
    tol_radius=None,
    # optional indices
    return_indices=None,
    convex_axis=None,
    # verb
    verb=None,
):

    # -------------
    # keys
    # -------------

    wrays = coll._which_rays

    # key_rays
    lok = list(coll.dobj.get(wrays, {}).keys())
    key_rays = ds._generic_check._check_var(
        key_rays, 'key_rays',
        types=str,
        allowed=lok,
    )

    # key_single_pt_cam
    lok = [
        k0 for k0 in lok
        if isinstance(coll.dobj[wrays][k0].get('angles'), tuple)
        and len(coll.dobj[wrays][k0]['angles']) == 2
        and all([
            ss in coll.ddata.keys()
            for ss in coll.dobj[wrays][k0].get('angles')
        ])
        and isinstance(coll.dobj[wrays][k0].get('nin'), np.ndarray)
        and isinstance(coll.dobj[wrays][k0].get('e0'), np.ndarray)
        and isinstance(coll.dobj[wrays][k0].get('e1'), np.ndarray)
    ]
    key_single_pt_cam = ds._generic_check._check_var(
        key_single_pt_cam, 'key_single_pt_cam',
        types=str,
        allowed=lok,
    )

    # -----------------
    # segment
    # -----------------

    segment = int(ds._generic_check._check_var(
        segment, 'segment',
        types=(int, float),
        default=0,
    ))

    # -----------------
    # tol_radius
    # -----------------

    if tol_radius is not None:
        tol_radius = float(ds._generic_check._check_var(
            tol_radius, 'tol_radius',
            types=(int, float),
            sign='>0',
        ))

    # -----------------
    # return_indices
    # -----------------

    return_indices = ds._generic_check._check_var(
        return_indices, 'return_indices',
        types=bool,
        default=convex_axis not in [None, False],
    )

    # --------------
    # convex_axis
    # --------------

    if np.isscalar(convex_axis) and not isinstance(convex_axis, bool):
        convex_axis = (int(convex_axis),)

    convex_axis = ds._generic_check._check_var(
        convex_axis, 'convex_axis',
        types=(tuple, bool),
        default=False,
    )

    # case by case
    ndim = len(coll.dobj[wrays][key_rays]['shape'][1:])
    if convex_axis is True:
        convex_axis = tuple(range(ndim))

    elif convex_axis is not False:
        convex_axis = ds._generic_check._check_var_iter(
            convex_axis, 'convex_axis',
            types=tuple,
            types_iter=int,
            allowed=list(range(ndim)) + list(range(-1, -ndim-1, -1)),
        )

        convex_axis = np.array(convex_axis)
        ineg = convex_axis < 0
        convex_axis[ineg] = ndim + convex_axis[ineg]
        convex_axis = tuple(sorted(convex_axis))

    # -----------------
    # verb
    # -----------------

    verb = ds._generic_check._check_var(
        verb, 'verb',
        types=bool,
        default=True,
    )

    return (
        key_single_pt_cam, key_rays,
        segment, tol_radius,
        return_indices, convex_axis,
        verb,
    )


# ####################################################
# ####################################################
#            get ConvexHull
# ####################################################


def _convexhull(
    shape_angles=None,
    ind0=None,
    ind1=None,
    sli=None,
    i01_full=None,
):
    # ---------------
    # initialize
    # ---------------

    ind = np.zeros(shape_angles, dtype=bool)

    # ---------------
    # get convex hull
    # ---------------

    ui0 = np.unique(ind0[sli])
    ui1 = np.unique(ind1[sli])
    if ui0.size == 1:
        npts = ui1[-1] - ui1[0] + 1
        slii = (
            np.full((npts,), ui0[0]),
            np.arange(ui1[0], min(ui1[-1]+1, ind.shape[1])),
        )
        ind[slii] = True

    elif ui1.size == 1:
        npts = ui0[-1] - ui0[0] + 1
        slii = (
            np.arange(ui0[0], min(ui0[-1]+1, ind.shape[0])),
            np.full((npts,), ui1[0]),
        )
        ind[slii] = True

    # ---------------
    # get convex hull
    # ---------------

    else:
        i0_flat = ind0[sli].ravel()
        i1_flat = ind1[sli].ravel()
        hull = scpsp.ConvexHull(
            np.array([i0_flat, i1_flat]).T
        )

        # get indices of angles in hull
        path = Path(
            np.array([i0_flat[hull.vertices], i1_flat[hull.vertices]]).T
        )
        ind = path.contains_points(i01_full).reshape(shape_angles)

    return ind
