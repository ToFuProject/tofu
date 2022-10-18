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
    segment=None,
    concatenate=None,
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

    # concatenate
    concatenate = ds._generic_check._check_var(
        concatenate, 'concatenate',
        types=bool,
        default=False,
    )
    
    if mode == 'abs':
        concatenate = False

    # segment
    if segment is not None:
        segment = np.atleast_1d(segment).astype(int).ravel()

    # -----------
    # compute
    # -----------

    # ------------------------
    # get pooints of interest
    
    pts_x, pts_y, pts_z = coll.get_rays_pts(key=key)
    npts = pts_x.shape[0]
    
    if segment is not None:
        segment[segment < 0] = npts - 1 + segment[segment < 0]
        
        iseg = np.r_[segment, segment[-1] + 1]
        pts_x = pts_x[iseg, :]
        pts_y = pts_y[iseg, :]
        pts_z = pts_z[iseg, :]
        
        npts = pts_x.shape[0]

   # -------------------------
   # prepare sampling indices

    # rel
    if mode == 'rel':

        # make sure npts allow to describe all integer indices
        i0 = np.arange(0, npts)
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


    # abs
    else:
        norm = np.sqrt(
            np.diff(pts_x, axis=0)**2
            + np.diff(pts_y, axis=0)**2
            + np.diff(pts_z, axis=0)**2
            )
        
        nn = np.ceil(norm / res).astype(int)
        i0 = np.arange(0, npts)
        
        lpx, lpy, lpz = [], [], []
        for ii in range(pts_x.shape[1]):
            
            i1 = np.concatenate(tuple(
                [
                    np.linspace(i0[jj], i0[jj+1], nn[jj, ii] + 1)[:-1]
                    for jj in range(npts - 1)
                ]
                + [[i0[-1]]]
                ))
        
            # interpolate
            lpx.append(scpinterp.interp1d(
                i0,
                pts_x[:, ii],
                kind='linear',
                axis=0,
            )(i1))
            
            lpy.append(scpinterp.interp1d(
                i0,
                pts_y[:, ii],
                kind='linear',
                axis=0,
            )(i1))
            
            lpz.append(scpinterp.interp1d(
                i0,
                pts_z[:, ii],
                kind='linear',
                axis=0,
            )(i1))
        
        pts_x, pts_y, pts_z = lpx, lpy, lpz

    # -------------------------------------
    # optional concatenation (for plotting)
    
    if concatenate is True:
        if mode == 'rel':
            shape = tuple(np.r_[np.r_[1], pts_x.shape[1:]])
            nan = np.full(shape, np.nan)
            pts_x = np.concatenate((pts_x, nan), axis=0).T.ravel()
            pts_y = np.concatenate((pts_y, nan), axis=0).T.ravel()
            pts_z = np.concatenate((pts_z, nan), axis=0).T.ravel()
        else:
            pts_x = np.concatenate([np.append(pp, np.nan) for pp in pts_x])
            pts_y = np.concatenate([np.append(pp, np.nan) for pp in pts_y])
            pts_z = np.concatenate([np.append(pp, np.nan) for pp in pts_z])

    return pts_x, pts_y, pts_z
