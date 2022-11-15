# -*- coding: utf-8 -*-


import copy
import itertools as itt

import numpy as np
import scipy.interpolate as scpinterp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


import datastock as ds


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
    res=None,
    mode=None,
    segment=None,
    radius_max=None,
    plot=None,
    dcolor=None,
    dax=None,
    ):

    # ------------
    # check inputs

    # key_cam
    key, key_cam = coll.get_diagnostic_cam(key=key, key_cam=key_cam)

    # key_data
    lok_coords = ['x', 'y', 'z', 'R', 'phi', 'k', 'l', 'ltot', 'itot']
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
                nan_out=True,
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
                nan_out=True,
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
                nan_out=True,
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

            tit = f"{key} LOS\nminor radius vs major radius"
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

        return xx, yy, dax
    else:
        return xx, yy
