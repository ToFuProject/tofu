# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


import datastock as ds


from ..geom._comp_solidangles import calc_solidangle_apertures


# ##################################################################
# ##################################################################
#             solid angles from any points
# ##################################################################


def compute_solid_angles(
    coll=None,
    key=None,
    key_cam=None,
    # pts
    ptsx=None,
    ptsy=None,
    ptsz=None,
    # options
    config=None,
    visibility=None,
    # return
    return_vect=None,
    return_alpha=None,
):
    # ---------
    # check

    (
        key, key_cam, spectro,
        ptsx, ptsy, ptsz, shape0_pts,
        return_vect, return_alpha,
    ) = _compute_solid_angles_check(
        coll=coll,
        key=key,
        key_cam=key_cam,
        # pts
        ptsx=ptsx,
        ptsy=ptsy,
        ptsz=ptsz,
        # return
        return_vect=return_vect,
        return_alpha=return_alpha,
    )

    # -----------
    # prepare

    if spectro:
        raise NotImplementedError()

    else:

        dout = _compute_solid_angles_regular(
            coll=coll,
            key=key,
            key_cam=key_cam,
            # pts
            ptsx=ptsx,
            ptsy=ptsy,
            ptsz=ptsz,
            shape0_pts=shape0_pts,
            # options
            config=config,
            visibility=visibility,
            # return
            return_vect=return_vect,
        )

    return dout


def _compute_solid_angles_check(
    coll=None,
    key=None,
    key_cam=None,
    # pts
    ptsx=None,
    ptsy=None,
    ptsz=None,
    # options
    config=None,
    visibility=None,
    # return
    return_vect=None,
    return_alpha=None,
):
    # ---------
    # check

    # key_cam
    key, key_cam = coll.get_diagnostic_cam(key=key, key_cam=key_cam)
    spectro = coll.dobj['diagnostic'][key]['spectro']

    # pts
    ptsx = np.atleast_1d(ptsx)
    ptsy = np.atleast_1d(ptsy)
    ptsz = np.atleast_1d(ptsz)

    if not (ptsx.shape == ptsy.shape == ptsz.shape):
        msg = (
            "Args ptsx, ptsy, ptsz must be 3 np.ndarray of the same shape!"
        )
        raise Exception(msg)

    shape0_pts = ptsx.shape
    if ptsx.ndim > 1:
        ptsx = ptsx.ravel()
        ptsy = ptsy.ravel()
        ptsz = ptsz.ravel()

    # return_vect
    return_vect = ds._generic_check._check_var(
        return_vect, 'return_vect',
        types=bool,
        default=False,
    )

    # return_alpha
    return_alpha = ds._generic_check._check_var(
        return_alpha, 'return_alpha',
        types=bool,
        default=False,
    )

    return (
        key, key_cam, spectro,
        ptsx, ptsy, ptsz, shape0_pts,
        return_vect, return_alpha,
    )


def _compute_solid_angles_regular(
    coll=None,
    key=None,
    key_cam=None,
    # pts
    ptsx=None,
    ptsy=None,
    ptsz=None,
    shape0_pts=None,
    # options
    config=None,
    visibility=None,
    # return
    return_vect=None,
):

    doptics = coll.dobj['diagnostic'][key]['doptics']
    dout = {k0: {} for k0 in key_cam}

    for k0 in key_cam:

        # prepare apertures
        dap = {}
        for op, opc in zip(doptics[k0]['optics'], doptics[k0]['cls']):
            dg = coll.dobj[opc][op]['dgeom']
            if dg['type'] == '3d':
                px, py, pz = dg['poly_x'], dg['poly_y'], dg['poly_z']
            else:
                cc = dg['cent']
                out0, out1 = dg['outline']
                out0, out1 = coll.ddata[out0]['data'], coll.ddata[out1]['data']
                px = cc[0] + out0*dg['e0'][0] + out1*dg['e1'][0]
                py = cc[1] + out0*dg['e0'][1] + out1*dg['e1'][1]
                pz = cc[2] + out0*dg['e0'][2] + out1*dg['e1'][2]

            dap[op] = {
                'nin': dg['nin'],
                'poly_x': px,
                'poly_y': py,
                'poly_z': pz,
            }

        # prepare camera
        dg = coll.dobj['camera'][k0]['dgeom']
        ddet = {}

        # cents
        cx, cy, cz = coll.get_camera_cents_xyz(k0)
        npts = cx.size
        sh = cx.shape
        ddet['cents_x'] = cx
        ddet['cents_y'] = cy
        ddet['cents_z'] = cz

        # vectors
        ddet.update(coll.get_camera_unit_vectors(k0))
        for k1 in ['nin', 'e0', 'e1']:
            for ii, ss in enumerate(['x', 'y', 'z']):
                kk = f'{k1}_{ss}'
                if np.isscalar(ddet[kk]):
                    ddet[kk] = np.full(sh, ddet[kk])

        out0, out1 = dg['outline']
        out0, out1 = coll.ddata[out0]['data'], coll.ddata[out1]['data']
        ddet['outline_x0'] = out0
        ddet['outline_x1'] = out1

        # compute
        out = calc_solidangle_apertures(
            # observation points
            pts_x=ptsx,
            pts_y=ptsy,
            pts_z=ptsz,
            # polygons
            apertures=dap,
            detectors=ddet,
            # possible obstacles
            config=config,
            # parameters
            summed=False,
            visibility=visibility,
            return_vector=return_vect,
            return_flat_pts=True,
            return_flat_det=None,
        )

        # store
        if return_vect is True:
            dout[k0]['solid_angle'] = out[0]
            dout[k0]['vectx'] = out[1]
            dout[k0]['vecty'] = out[2]
            dout[k0]['vectz'] = out[3]
        else:
            dout[k0]['solid_angle'] = out

        # reshape
        if shape0_pts != ptsx.shape:
            shape = tuple(np.r_[dout[k0]['solid_angle'].shape, shape0_pts])
            for k1, v1 in dout[k0].items():
                dout[k0][k1] = v1.reshape(shape)

    return dout


# ##################################################################
# ##################################################################
#             interpolated along los
# ##################################################################


def _interpolated_along_los(
    coll=None,
    key=None,
    key_cam=None,
    key_data_x=None,
    key_data_y=None,
    # sampling
    res=None,
    mode=None,
    segment=None,
    radius_max=None,
    # plotting
    vmin=None,
    vmax=None,
    plot=None,
    dcolor=None,
    dax=None,
    ):

    # ------------
    # check inputs

    # key_cam
    key, key_cam = coll.get_diagnostic_cam(key=key, key_cam=key_cam)

    # key_data
    lok_coords = [
        'x', 'y', 'z', 'R', 'phi', 'ang_vs_ephi',
        'k', 'l', 'ltot', 'itot',
    ]
    lok_2d = [
        k0 for k0, v0 in coll.ddata.items()
        if v0.get('bsplines') is not None
    ]

    key_data_x = ds._generic_check._check_var(
        key_data_x, 'key_data_x',
        types=str,
        default='k',
        allowed=lok_coords + lok_2d,
    )

    key_data_y = ds._generic_check._check_var(
        key_data_y, 'key_data_y',
        types=str,
        default='k',
        allowed=lok_coords + lok_2d,
    )

    # segment
    segment = ds._generic_check._check_var(
        segment, 'segment',
        types=int,
        default=-1,
    )

    # plot
    plot = ds._generic_check._check_var(
        plot, 'plot',
        types=bool,
        default=True,
    )

    # dcolor
    if not isinstance(dcolor, dict):
        if dcolor is None:
            lc = ['k', 'r', 'g', 'b', 'm', 'c']
        elif mcolors.is_color_like(dcolor):
            lc = [dcolor]

        dcolor = {
            kk: lc[ii%len(lc)]
            for ii, kk in enumerate(key_cam)
            }

    # --------------
    # prepare output

    ncam = len(key_cam)

    xx = [None for ii in range(ncam)]
    yy = [None for ii in range(ncam)]

    # ---------------
    # loop on cameras

    if key_data_x in lok_coords and key_data_y in lok_coords:

        for ii, kk in enumerate(key_cam):

            klos = coll.dobj['diagnostic'][key]['doptics'][kk]['los']
            if klos is None:
                continue

            xx[ii], yy[ii] = coll.sample_rays(
                key=klos,
                res=res,
                mode=mode,
                segment=segment,
                radius_max=radius_max,
                concatenate=True,
                return_coords=[key_data_x, key_data_y],
                )

            if key_data_x in ['x', 'y', 'z', 'R', 'l', 'ltot']:
                xlab = f"{key_data_x} (m)"
            else:
                xlab = key_data_x

            if key_data_y in ['x', 'y', 'z', 'R', 'l', 'ltot']:
                ylab = f"{key_data_y} (m)"
            else:
                ylab = key_data_y

    elif key_data_x in lok_coords or key_data_y in lok_coords:

        if key_data_x in lok_coords:
            cll = key_data_x
            c2d = key_data_y
            if key_data_x in ['x', 'y', 'z', 'R', 'l', 'ltot']:
                xlab = f"{key_data_x} (m)"
            else:
                xlab = key_data_x
            ylab = f"{key_data_y} ({coll.ddata[key_data_y]['units']})"
        else:
            cll = key_data_y
            c2d = key_data_x
            if key_data_y in ['x', 'y', 'z', 'R', 'l', 'ltot']:
                ylab = f"{key_data_y} (m)"
            else:
                ylab = key_data_y
            xlab = f"{key_data_x} ({coll.ddata[key_data_x]['units']})"

        for ii, kk in enumerate(key_cam):

            klos = coll.dobj['diagnostic'][key]['doptics'][kk]['los']
            if klos is None:
                continue

            pts_x, pts_y, pts_z, pts_ll = coll.sample_rays(
                key=klos,
                res=res,
                mode=mode,
                segment=segment,
                radius_max=radius_max,
                concatenate=True,
                return_coords=['x', 'y', 'z', cll],
                )

            Ri = np.hypot(pts_x, pts_y)

            q2d, _ = coll.interpolate_profile2d(
                key=c2d,
                R=Ri,
                Z=pts_z,
                grid=False,
                crop=True,
                nan0=True,
                val_out=True,
                imshow=False,
                return_params=None,
                store=False,
                inplace=False,
            )

            isok = ~(np.isnan(q2d) & (~np.isnan(Ri)))
            if key_data_x in lok_coords:
                xx[ii] = pts_ll[isok]
                yy[ii] = q2d[isok]
            else:
                xx[ii] = q2d[isok]
                yy[ii] = pts_ll[isok]

    else:
        for ii, kk in enumerate(key_cam):

            klos = coll.dobj['diagnostic'][key]['doptics'][kk]['los']
            if klos is None:
                continue

            pts_x, pts_y, pts_z = coll.sample_rays(
                key=klos,
                res=res,
                mode=mode,
                segment=segment,
                radius_max=radius_max,
                concatenate=True,
                return_coords=['x', 'y', 'z'],
                )

            Ri = np.hypot(pts_x, pts_y)

            q2dx, _ = coll.interpolate_profile2d(
                key=key_data_x,
                R=Ri,
                Z=pts_z,
                grid=False,
                crop=True,
                nan0=True,
                val_out=True,
                imshow=False,
                return_params=None,
                store=False,
                inplace=False,
            )

            q2dy, _ = coll.interpolate_profile2d(
                key=key_data_y,
                R=Ri,
                Z=pts_z,
                grid=False,
                crop=True,
                nan0=True,
                val_out=True,
                imshow=False,
                return_params=None,
                store=False,
                inplace=False,
            )

            isok = ~((np.isnan(q2dx) | np.isnan(q2dy)) & (~np.isnan(Ri)))
            xx[ii] = q2dx[isok]
            yy[ii] = q2dy[isok]

            xlab = f"{key_data_x} ({coll.ddata[key_data_x]['units']})"
            ylab = f"{key_data_y} ({coll.ddata[key_data_y]['units']})"

    # ------------
    # plot

    if plot is True:
        if dax is None:

            fig = plt.figure()

            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

            tit = f"{key} LOS"
            ax.set_title(tit, size=12, fontweight='bold')
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)

            dax = {'main': ax}

        # main
        kax = 'main'
        if dax.get(kax) is not None:
            ax = dax[kax]

            for ii, kk in enumerate(key_cam):
                ax.plot(
                    xx[ii],
                    yy[ii],
                    c=dcolor[kk],
                    marker='.',
                    ls='-',
                    ms=8,
                    label=kk,
                )

            ax.legend()
            
            if vmin is not None:
                ax.set_ylim(bottom=vmin)
            if vmax is not None:
                ax.set_ylim(top=vmax)

        return xx, yy, dax
    else:
        return xx, yy
