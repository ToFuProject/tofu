# -*- coding: utf-8 -*-


import numpy as np
import scipy.interpolate as scpinterp
import datastock as ds


# ###############################################################
# ###############################################################
#                   sample
# ###############################################################


def _sample(
    coll=None,
    key=None,
    res=None,
    mode=None,
):

    # ------------
    # check inputs

    # res
    res = ds._generic_check._check_var(
        res, 'res',
        types=float,
        default=0.25,
        sign='> 0',
    )

    # mode
    mode = ds._generic_check._check_var(
        mode, 'mode',
        types=str,
        default='rel',
        allowed=['rel', 'abs'],
    )

    # --------
    # compute

    pts_x, pts_y, pts_z = coll.get_rays_pts(key=key)

    if mode == 'rel':

        # make sure npts allow to describe all integer indices
        i0 = np.arange(0, pts_x.shape[0])
        npts = i0[-1] * int(np.ceil(1./res))

        N = int(np.ceil((npts - 1) / (i0[-1] - i0[0])))
        npts = N * (i0[-1] - i0[0]) + 1
        i1 = np.linspace(i0[0], i0[-1], npts)

        # interpolate
        pts_x = scpinterp.interp1d(
            i0,
            pts_x,
            kind='linear',
            axis=0,
        )(i1)
        pts_y = scpinterp.interp1d(
            i0,
            pts_y,
            kind='linear',
            axis=0,
        )(i1)
        pts_z = scpinterp.interp1d(
            i0,
            pts_z,
            kind='linear',
            axis=0,
        )(i1)

    else:
        raise NotImplementedError()

    return pts_x, pts_y, pts_z
