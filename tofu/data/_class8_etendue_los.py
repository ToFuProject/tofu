# -*- coding: utf-8 -*-


import warnings


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


import Polygon as plg
import datastock as ds


from ..geom import _comp_solidangles


__all__ = ['compute_etendue_los']


# ##################################################################
# ##################################################################
#                       Main
# ##################################################################


def compute_etendue_los(
    coll=None,
    key=None,
    # parameters
    analytical=None,
    numerical=None,
    res=None,
    margin_par=None,
    margin_perp=None,
    # options
    add_points=None,
    # bool
    convex=None,
    check=None,
    verb=None,
    plot=None,
    store=None,
):

    # ------------
    # check inputs

    (
        key,
        spectro,
        is2d,
        doptics,
        dcompute,
        analytical,
        numerical,
        res,
        margin_par,
        margin_perp,
        check,
        verb,
        plot,
        store,
    ) = _diag_compute_etendue_check(
        coll=coll,
        key=key,
        analytical=analytical,
        numerical=numerical,
        res=res,
        margin_par=margin_par,
        margin_perp=margin_perp,
        check=check,
        verb=verb,
        plot=plot,
        store=store,
    )

    if verb is True:
        msg = f"\nComputing etendue / los for diag '{key}':"
        print(msg)

    # prepare optics
    for key_cam, v0 in dcompute.items():

        # ------------------------
        # get equivalent apertures for all pixels

        (
            x0, x1, kref, iok,
            px, py, pz,
            cx, cy, cz,
            centsx, centsy, centsz,
            ap_area, plane_nin,
            spectro,
        ) = coll.get_diagnostic_equivalent_aperture(
            key=key,
            key_cam=key_cam,
            # inital contour
            add_points=add_points,
            # options
            convex=convex,
            harmonize=True,
            reshape=False,
            # plot
            plot=False,
            verb=verb,
            store=False,
            return_for_etendue=True,
        )

        # ------------------------------------------
        # get distance, area, solid_angle, los, dlos

        (
            det_area, distances,
            los_x, los_y, los_z,
            dlos_x, dlos_y, dlos_z,
            cos_los_det, cos_los_ap, solid_angles, res,
        ) = _loop_on_pix(
            coll=coll,
            ldet=v0['ldet'],
            spectro=spectro,
            # optics
            x0=x0,
            x1=x1,
            px=px,
            py=py,
            pz=pz,
            iok=iok,
            cx=cx,
            cy=cy,
            cz=cz,
            centsx=centsx,
            centsy=centsy,
            centsz=centsz,
            plane_nin=plane_nin,
            ap_area=ap_area,
        )

        # --------------------
        # compute analytically

        nd = len(v0['ldet'])
        if analytical is True:
            etend0 = np.full(tuple(np.r_[3, nd]), np.nan)

            # 0th order
            etend0[0, :] = ap_area * det_area / distances**2

            # 1st order
            etend0[1, :] = (
                cos_los_ap * ap_area
                * cos_los_det * det_area / distances**2
            )

            # 2nd order
            etend0[2, :] = cos_los_ap * ap_area * solid_angles

        else:
            etend0 = None

        # --------------------
        # compute numerically

        if numerical is True:

            etend1 = _compute_etendue_numerical(
                ldeti=v0['ldet'],
                aperture=aperture,
                pix_ap=pix_ap,
                res=res,
                los_x=los_x,
                los_y=los_y,
                los_z=los_z,
                margin_par=margin_par,
                margin_perp=margin_perp,
                check=check,
                verb=verb,
            )

        else:
            etend1 = None

        # --------------------
        # optional plotting
    
        if plot is True:
            dax = _plot_etendues(
                etend0=etend0,
                etend1=etend1,
                res=res,
            )
    
        # --------
        # reshape
    
        # etend0
        if etend0 is not None and is2d:
            etend0 = etend0.reshape(tuple(np.r_[3, v0['shape0']]))
    
        # etend1
        if etend1 is not None and is2d:
            etend1 = etend1.reshape(tuple(np.r_[res.size, v0['shape0']]))
    
        # los
        if los_x.shape != v0['shape0']:
            los_x = los_x.reshape(v0['shape0'])
            los_y = los_y.reshape(v0['shape0'])
            los_z = los_z.reshape(v0['shape0'])

        # --------------------
        # return dict
    
        dcompute[key_cam].update({
            'analytical': etend0,
            'numerical': etend1,
            'res': res,
            'kref': kref,
            'los_x': los_x,
            'los_y': los_y,
            'los_z': los_z,
            'dlos_x': dlos_x,
            'dlos_y': dlos_y,
            'dlos_z': dlos_z,
            'iok': iok,
            'is2d': is2d,
            'cx': cx,
            'cy': cy,
            'cz': cz,
        })

    # ----------
    # store

    if store is not False:

        for key_cam, v0 in dcompute.items():
        
            # ref
            ref = coll.dobj['camera'][key_cam]['dgeom']['ref']
    
            # data
            etendue = v0[store][-1, ...]
    
            if store == 'analytical':
                etend_type = store
            else:
                etend_type = v0['res'][-1]
    
            # keys
            ketendue = f'{key}_{key_cam}_etend'
            ddata = {
                ketendue: {
                    'data': etendue,
                    'ref': ref,
                    'dim': 'etendue',
                    'quant': 'etendue',
                    'name': 'etendue',
                    'units': 'm2.sr',
                },
            }
            
            coll.update(ddata=ddata)
    
            coll._dobj['diagnostic'][key]['doptics'][key_cam]['etendue'] = ketendue
            coll._dobj['diagnostic'][key]['doptics'][key_cam]['etend_type'] = etend_type

    return dcompute, store


# ##################################################################
# ##################################################################
#                       Check
# ##################################################################


def _diag_compute_etendue_check(
    coll=None,
    key=None,
    analytical=None,
    numerical=None,
    res=None,
    margin_par=None,
    margin_perp=None,
    check=None,
    verb=None,
    plot=None,
    store=None,
):

    # --------
    # key

    lok = [
        k0 for k0, v0 in coll.dobj.get('diagnostic', {}).items()
        if any([len(v1['optics']) > 0 for v1 in v0['doptics'].values()])
    ]
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # spectro, is2d
    spectro = coll.dobj['diagnostic'][key]['spectro']
    is2d = coll.dobj['diagnostic'][key]['is2d']

    # doptics
    doptics = coll.dobj['diagnostic'][key]['doptics']
    dcompute = {
        k0: {'compute': len(v0['optics']) > 0}
        for k0, v0 in doptics.items()
    }

    # -------------------------------------------------
    # ldeti: list of individual camera dict (per pixel)

    for k0, v0 in doptics.items():

        dgeom = coll.dobj['camera'][k0]['dgeom']
        cx, cy, cz = coll.get_camera_cents_xyz(key=k0)
        dvect = coll.get_camera_unit_vectors(key=k0)
        outline = dgeom['outline']
        out0 = coll.ddata[outline[0]]['data']
        out1 = coll.ddata[outline[1]]['data']
        is2d = dgeom['type'] == '2d'
        par = dgeom['parallel']
        dcompute[k0]['shape0'] = cx.shape
    
        if is2d:
            cx = cx.ravel()
            cy = cy.ravel()
            cz = cz.ravel()
        nd = cx.size
    
        dcompute[k0]['ldet'] = [
            {
                'cents_x': cx[ii],
                'cents_y': cy[ii],
                'cents_z': cz[ii],
                'outline_x0': out0,
                'outline_x1': out1,
                'nin_x': dvect['nin_x'] if par else dvect['nin_x'][ii],
                'nin_y': dvect['nin_y'] if par else dvect['nin_y'][ii],
                'nin_z': dvect['nin_z'] if par else dvect['nin_z'][ii],
                'e0_x': dvect['e0_x'] if par else dvect['e0_x'][ii],
                'e0_y': dvect['e0_y'] if par else dvect['e0_y'][ii],
                'e0_z': dvect['e0_z'] if par else dvect['e0_z'][ii],
                'e1_x': dvect['e1_x'] if par else dvect['e1_x'][ii],
                'e1_y': dvect['e1_y'] if par else dvect['e1_y'][ii],
                'e1_z': dvect['e1_z'] if par else dvect['e1_z'][ii],
            }
            for ii in range(nd)
        ]

    # -----------
    # analytical

    analytical = ds._generic_check._check_var(
        analytical, 'analytical',
        types=bool,
        default=True,
    )

    # -----------
    # numerical

    numerical = ds._generic_check._check_var(
        numerical, 'numerical',
        types=bool,
        default=False,
    )

    # -----------
    # res

    if res is not None:
        res = np.atleast_1d(res).ravel()

    # -----------
    # margin_par

    margin_par = ds._generic_check._check_var(
        margin_par, 'margin_par',
        types=float,
        default=0.05,
    )

    # -----------
    # margin_perp

    margin_perp = ds._generic_check._check_var(
        margin_perp, 'margin_perp',
        types=float,
        default=0.05,
    )

    # -----------
    # check

    check = ds._generic_check._check_var(
        check, 'check',
        types=bool,
        default=True,
    )

    # -----------
    # verb

    verb = ds._generic_check._check_var(
        verb, 'verb',
        types=bool,
        default=True,
    )

    # -----------
    # plot

    if plot is None:
        plot = True
    if not isinstance(plot, bool):
        msg = "Arg plot must be a bool"
        raise Exception(msg)

    # -----------
    # store

    lok = [False]
    if analytical is True:
        lok.append('analytical')
    if numerical is True:
        lok.append('numerical')
    store = ds._generic_check._check_var(
        store, 'store',
        default=lok[-1],
        allowed=lok,
    )

    return (
        key,
        spectro,
        is2d,
        doptics,
        dcompute,
        analytical,
        numerical,
        res,
        margin_par,
        margin_perp,
        check,
        verb,
        plot,
        store,
    )


# ##################################################################
# ##################################################################
#                    Loop on camera pixels
# ##################################################################


def _loop_on_pix(
    coll=None,
    # detectors
    ldet=None,
    spectro=None,
    # equivalent aperture
    x0=None,
    x1=None,
    px=None,
    py=None,
    pz=None,
    iok=None,
    cx=None,
    cy=None,
    cz=None,
    centsx=None,
    centsy=None,
    centsz=None,
    plane_nin=None,
    ap_area=None,
    # extra
    res=None,
):

    # prepare data
    nd = len(ldet)
    los_x = np.full((nd,), np.nan)
    los_y = np.full((nd,), np.nan)
    los_z = np.full((nd,), np.nan)
    solid_angles = np.zeros((nd,), dtype=float)
    cos_los_det = np.full((nd,), np.nan)
    distances = np.full((nd,), np.nan)
    mindiff = np.full((nd,), np.nan)

    # -------------------------
    # compute area, solid angle, los

    nd = x0.shape[0]
    for ii in range(nd):

        if not iok[ii]:
            continue

        # ------------
        # solid angles

        solid_angles[ii] = _comp_solidangles.calc_solidangle_apertures(
            # observation points
            pts_x=centsx[ii],
            pts_y=centsy[ii],
            pts_z=centsz[ii],
            # polygons
            apertures=None,
            detectors=ldet[ii],
            # possible obstacles
            config=None,
            # parameters
            visibility=False,
            return_vector=False,
        )[0, 0]

    # -------------
    # normalize los

    los_x = centsx - cx
    los_y = centsy - cy
    los_z = centsz - cz

    distances = np.sqrt(los_x**2 + los_y**2 + los_z**2)

    los_x = los_x / distances
    los_y = los_y / distances
    los_z = los_z / distances

    if spectro:
        dlos_x = px - cx[:, None]
        dlos_y = py - cy[:, None]
        dlos_z = pz - cz[:, None]
        ddist = np.sqrt(dlos_x**2 + dlos_y**2 + dlos_z**2)
        dlos_x = dlos_x / ddist
        dlos_y = dlos_y / ddist
        dlos_z = dlos_z / ddist
    else:
        dlos_x = None
        dlos_y = None
        dlos_z = None

    # ------
    # angles

    for ii in range(nd):
        cos_los_det[ii] = (
            los_x[ii] * ldet[ii]['nin_x']
            + los_y[ii] * ldet[ii]['nin_y']
            + los_z[ii] * ldet[ii]['nin_z']
        )

    # abs() because for spectro nin is the other way around
    cos_los_ap = np.abs(
        los_x * plane_nin[0]
        + los_y * plane_nin[1]
        + los_z * plane_nin[2]
    )

    # -----------
    # surfaces

    # det
    if ldet[0].get('pix_area') is None:
        det_area = plg.Polygon(np.array([
            ldet[0]['outline_x0'],
            ldet[0]['outline_x1'],
        ]).T).area()
    else:
        det_area = ldet[0]['pix_area']

    # -------------------------------------
    # det outline discretization resolution

    if res is None:

        res = min(np.sqrt(det_area), np.nanmin(mindiff))
        if np.any(ap_area > 0.):
            res = min(res, np.sqrt(np.min(ap_area[ap_area > 0.])))

        res = res * np.r_[1., 0.5, 0.1]

    iok = np.isfinite(res)
    iok[iok] = res[iok] > 0
    if not np.any(iok):
        res = np.r_[0.001]
    else:
        res = res[iok]

    return (
        det_area, distances,
        los_x, los_y, los_z,
        dlos_x, dlos_y, dlos_z,
        cos_los_det, cos_los_ap, solid_angles, res,
    )


# ##################################################################
# ##################################################################
#                   op_post interpolation
# ##################################################################


def _get_lpoly_post(coll=None, lop_post_cls=None, lop_post=None):

    if len(lop_post) == 0:
        return

    lpoly_post = []
    for cc, oo in zip(lop_post_cls, lop_post):
        dgeom = coll.dobj[cc][oo]['dgeom']
        if dgeom['type'] == 'planar':
            p0, p1 = dgeom['outline']
            lpoly_post.append((
                coll.ddata[p0]['data'],
                coll.ddata[p1]['data'],
            ))
        else:
            px, py, pz = dgeom['poly']
            lpoly_post.append((
                coll.ddata[px]['data'],
                coll.ddata[py]['data'],
                coll.ddata[pz]['data'],
            ))
    return lpoly_post


# ##################################################################
# ##################################################################
#           Numerical etendue estimation routine
# ##################################################################


def _compute_etendue_numerical(
    ldeti=None,
    aperture=None,
    pix_ap=None,
    res=None,
    margin_par=None,
    margin_perp=None,
    los_x=None,
    los_y=None,
    los_z=None,
    check=None,
    verb=None,
):

    # shape0 = det['cents_x'].shape
    nd = len(ldeti)

    ap_ind = np.cumsum([v0['poly_x'].size for v0 in aperture.values()][:-1])

    ap_tot_px = np.concatenate(tuple(
        [v0['poly_x'] for v0 in aperture.values()]
    ))
    ap_tot_py = np.concatenate(tuple(
        [v0['poly_y'] for v0 in aperture.values()]
    ))
    ap_tot_pz = np.concatenate(tuple(
        [v0['poly_z'] for v0 in aperture.values()]
    ))

    # ------------------------------
    # Get plane perpendicular to los

    etendue = np.full((res.size, nd), np.nan)
    for ii in range(nd):

        if verb is True:
            msg = f"Numerical etendue for det {ii+1} / {nd}"
            print(msg)

        if np.isnan(los_x[ii]):
            continue

        # get det corners to aperture corners vectors
        out_c_x0 = np.r_[0, ldeti[ii]['outline_x0']]
        out_c_x1 = np.r_[0, ldeti[ii]['outline_x1']]

        # det poly 3d
        det_Px = (
            ldeti[ii]['cents_x']
            + ldeti[ii]['outline_x0']*ldeti[ii]['e0_x']
            + ldeti[ii]['outline_x1']*ldeti[ii]['e1_x']
        )
        det_Py = (
            ldeti[ii]['cents_y']
            + ldeti[ii]['outline_x0']*ldeti[ii]['e0_y']
            + ldeti[ii]['outline_x1']*ldeti[ii]['e1_y']
        )
        det_Pz = (
            ldeti[ii]['cents_z']
            + ldeti[ii]['outline_x0']*ldeti[ii]['e0_z']
            + ldeti[ii]['outline_x1']*ldeti[ii]['e1_z']
        )

        # det to ap vectors
        PA_x = ap_tot_px[:, None] - det_Px[None, :]
        PA_y = ap_tot_py[:, None] - det_Py[None, :]
        PA_z = ap_tot_pz[:, None] - det_Pz[None, :]

        sca1 = PA_x * los_x[ii] + PA_y * los_y[ii] + PA_z * los_z[ii]
        # get length along los
        k_los = (1. + margin_par) * np.max(sca1)

        # get center of plane perpendicular to los
        c_los_x = ldeti[ii]['cents_x'] + k_los * los_x[ii]
        c_los_y = ldeti[ii]['cents_y'] + k_los * los_y[ii]
        c_los_z = ldeti[ii]['cents_z'] + k_los * los_z[ii]

        # get projections of corners on plane perp. to los
        sca0 = (
            (c_los_x - det_Px[None, :]) * los_x[ii]
            + (c_los_y - det_Py[None, :]) * los_y[ii]
            + (c_los_z - det_Pz[None, :]) * los_z[ii]
        )
        k_plane = sca0 / sca1

        # get LOS-specific unit vectors

        e0_xi = (
            los_y[ii] * ldeti[ii]['e1_z'] - los_z[ii] * ldeti[ii]['e1_y']
        )
        e0_yi = (
            los_z[ii] * ldeti[ii]['e1_x'] - los_x[ii] * ldeti[ii]['e1_z']
        )
        e0_zi = (
            los_x[ii] * ldeti[ii]['e1_y'] - los_y[ii] * ldeti[ii]['e1_x']
        )

        e0_normi = np.sqrt(e0_xi**2 + e0_yi**2 + e0_zi**2)
        e0_xi = e0_xi / e0_normi
        e0_yi = e0_yi / e0_normi
        e0_zi = e0_zi / e0_normi

        e1_xi = los_y[ii] * e0_zi - los_z[ii] * e0_yi
        e1_yi = los_z[ii] * e0_xi - los_x[ii] * e0_zi
        e1_zi = los_x[ii] * e0_yi - los_y[ii] * e0_xi

        # get projections on det_e0 and det_e1 in plane

        x0 = np.split(
            ((det_Px[None, :] + k_plane * PA_x) - c_los_x)*e0_xi
            + ((det_Py[None, :] + k_plane * PA_y) - c_los_y)*e0_yi
            + ((det_Pz[None, :] + k_plane * PA_z) - c_los_z)*e0_zi,
            ap_ind,
            axis=0,
        )
        x1 = np.split(
            ((det_Px[None, :] + k_plane * PA_x) - c_los_x)*e1_xi
            + ((det_Py[None, :] + k_plane * PA_y) - c_los_y)*e1_yi
            + ((det_Pz[None, :] + k_plane * PA_z) - c_los_z)*e1_zi,
            ap_ind,
            axis=0,
        )

        x0_min = np.max([np.min(x0s) for x0s in x0])
        x0_max = np.min([np.max(x0s) for x0s in x0])
        x1_min = np.max([np.min(x1s) for x1s in x1])
        x1_max = np.min([np.max(x1s) for x1s in x1])

        w0 = x0_max - x0_min
        w1 = x1_max - x1_min

        min_res = min(2*margin_perp*w0, 2*margin_perp*w1)
        too_large = res >= min_res
        if np.any(too_large):
            msg = (
                f"Minimum etendue resolution for det {ii} / {nd}: {min_res}\n"
                "The following res values may lead to errors:\n"
                f"\t- res values = {res}\n"
                f"\t- too large  = {too_large}\n"
            )
            warnings.warn(msg)

        # -------------------
        # Discretize aperture

        for jj in range(res.size):

            coef = 1. + 2.*margin_perp
            n0 = int(np.ceil(coef*w0 / res[jj]))
            n1 = int(np.ceil(coef*w1 / res[jj]))

            d0 = coef*w0 / n0
            d1 = coef*w1 / n1

            ds = d0 * d1

            pts_0 = np.linspace(
                x0_min - margin_perp*w0,
                x0_max + margin_perp*w0,
                n0 + 1,
            )
            pts_1 = np.linspace(
                x1_min - margin_perp*w1,
                x1_max + margin_perp*w1,
                n1 + 1,
            )
            pts_0 = 0.5 * (pts_0[1:] + pts_0[:-1])
            pts_1 = 0.5 * (pts_1[1:] + pts_1[:-1])

            # debug
            # n0, n1 = 2, 2
            # pts_0 = np.r_[pts_0[0], pts_0[0]]
            # pts_1 = np.r_[0, 0]

            pts_x = (
                c_los_x + pts_0[:, None] * e0_xi + pts_1[None, :] * e1_xi
            ).ravel()
            pts_y = (
                c_los_y + pts_0[:, None] * e0_yi + pts_1[None, :] * e1_yi
            ).ravel()
            pts_z = (
                c_los_z + pts_0[:, None] * e0_zi + pts_1[None, :] * e1_zi
            ).ravel()

            if verb is True:
                msg = (
                    f"\tres = {res[jj]} ({jj+1} / {res.size})"
                    f"    nb. of points = {pts_x.size}"
                )
                print(msg)

            # ----------------------------------
            # compute solid angle for each pixel

            if check is True:
                solid_angle = _comp_solidangles.calc_solidangle_apertures(
                    # observation points
                    pts_x=pts_x,
                    pts_y=pts_y,
                    pts_z=pts_z,
                    # polygons
                    apertures=aperture,
                    detectors=ldeti[ii],
                    # possible obstacles
                    config=None,
                    # parameters
                    visibility=False,
                    return_vector=False,
                    return_flat_pts=True,
                    return_flat_det=True,
                )

                sar = solid_angle.reshape((n0, n1))
                c0 = (
                    ((pts_0[0] < x0_min) == np.all(sar[0, :] == 0))
                    and ((pts_0[-1] > x0_max) == np.all(sar[-1, :] == 0))
                    and ((pts_1[0] < x1_min) == np.all(sar[:, 0] == 0))
                    and ((pts_1[-1] > x1_max) == np.all(sar[:, -1] == 0))
                )
                if not c0 and not too_large[jj]:
                    # debug
                    plt.figure()
                    plt.imshow(
                        sar.T,
                        extent=(
                            x0_min - margin_perp*w0, x0_max + margin_perp*w0,
                            x1_min - margin_perp*w1, x1_max + margin_perp*w1,
                        ),
                        interpolation='nearest',
                        origin='lower',
                        aspect='equal',
                    )

                    lc = ['r', 'm', 'c', 'y']
                    for ss in range(len(x0)):
                        iss = np.r_[np.arange(0, x0[ss].shape[0]), 0]
                        plt.plot(
                            x0[ss][iss, :],
                            x1[ss][iss, :],
                            c=lc[ss % len(lc)],
                            marker='o',
                            ls='-',
                        )

                    plt.plot(
                        pts_0, np.mean(pts_1)*np.ones((n0,)),
                        c='k', marker='.', ls='None',
                    )
                    plt.plot(
                        np.mean(pts_0)*np.ones((n1,)), pts_1,
                        c='k', marker='.', ls='None',
                    )
                    plt.gca().set_xlabel('x0')
                    plt.gca().set_xlabel('x1')
                    # import pdb; pdb.set_trace()
                    msg = "Something is wrong with solid_angle or sampling"
                    raise Exception(msg)
                else:
                    etendue[jj, ii] = np.sum(solid_angle) * ds

            else:
                etendue[jj, ii] = _comp_solidangles.calc_solidangle_apertures(
                    # observation points
                    pts_x=pts_x,
                    pts_y=pts_y,
                    pts_z=pts_z,
                    # polygons
                    apertures=aperture,
                    detectors=ldeti[ii],
                    # possible obstacles
                    config=None,
                    # parameters
                    summed=True,
                    visibility=False,
                    return_vector=False,
                    return_flat_pts=True,
                    return_flat_det=True,
                ) * ds

    return etendue


# ##################################################################
# ##################################################################
#                   Plotting routine
# ##################################################################


def _plot_etendues(
    etend0=None,
    etend1=None,
    res=None,
):

    # -------------
    # prepare data

    nmax = 0
    if etend0 is not None:
        if etend0.ndim > 2:
            etend0 = etend0.reshape((etend0.shape[0], -1))
        nmax = max(nmax, etend0.shape[0])
    if etend1 is not None:
        if etend1.ndim > 2:
            etend1 = etend1.reshape((etend1.shape[0], -1))
        nmax = max(nmax, etend1.shape[0])

    x0 = None
    if etend0 is not None:
        x0 = [
            f'order {ii}' if ii < 3 else '' for ii in range(nmax)
        ]
    if etend1 is not None:
        x1 = [f'{res[ii]}' if ii < res.size-1 else '' for ii in range(nmax)]
        if x0 is None:
            x0 = x1
        else:
            x0 = [f'{x0[ii]}\n{x1[ii]}' for ii in range(nmax)]

    # -------------
    # prepare axes

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    ax.set_ylabel('Etendue ' + r'($m^2.sr$)', size=12, fontweight='bold')
    ax.set_xlabel('order of approximation', size=12, fontweight='bold')

    ax.set_xticks(range(0, nmax))
    ax.set_xticklabels(x0)

    # -------------
    # plot

    if etend0 is not None:
        lines = ax.plot(
            etend0,
            ls='-',
            marker='o',
            ms=6,
        )
        lcol = [ll.get_color() for ll in lines]
    else:
        lcol = [None for ii in range(etend1.shape[1])]

    if etend1 is not None:
        for ii in range(etend1.shape[1]):
            ax.plot(
                etend1[:, ii],
                ls='--',
                marker='*',
                ms=6,
                color=lcol[ii],
            )

    # -------------
    # legend

    handles = [
        mlines.Line2D(
            [], [],
            c='k', marker='o', ls='-', ms=6,
            label='analytical',
        ),
        mlines.Line2D(
            [], [],
            c='k', marker='*', ls='--', ms=6,
            label='numerical',
        ),
    ]
    ax.legend(handles=handles)

    return ax
