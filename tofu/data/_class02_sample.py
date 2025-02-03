# -*- coding: utf-8 -*-


import itertools as itt


import numpy as np
import scipy.interpolate as scpinterp
import datastock as ds


# ###############################################################
# ###############################################################
#                   main
# ###############################################################


def main(
    coll=None,
    key=None,
    key_cam=None,
    res=None,
    mode=None,
    segment=None,
    ind_ch=None,
    radius_max=None,
    concatenate=None,
    return_coords=None,
):

    # ------------
    # check inputs

    (
        res, mode, concatenate,
        segment, radius_max,
        return_coords, out_xyz, out_k, out_l,
    ) = _sample_check(
        res=res,
        mode=mode,
        concatenate=concatenate,
        segment=segment,
        radius_max=radius_max,
        return_coords=return_coords,
    )

    # -----------
    # prepare
    # -----------

    # ------------------------
    # get points of interest

    # get rays points (nseg+1, nx0, nx1, ...)
    pts_x, pts_y, pts_z = coll.get_rays_pts(key=key, key_cam=key_cam)

    # extract sizes
    npts = pts_x.shape[0]
    i0 = np.arange(0, npts)

    # ---------------------------
    # optional segment selection

    if segment is not None:
        segment[segment < 0] = npts - 1 + segment[segment < 0]
        iseg = np.r_[segment, segment[-1] + 1]

    # ---------------------------------------
    # optional ind_flat (selection of pixels)

    if ind_ch is not None:

        # check ind
        ind_ch = _check_ind_channels(ind_ch, shape=pts_x.shape[1:], key=key)

        sli = tuple([slice(None)] + list(ind_ch))
        pts_x = pts_x[sli]
        pts_y = pts_y[sli]
        pts_z = pts_z[sli]

    # -------------------------------
    # get original length per segment

    length_orig = np.sqrt(
        np.diff(pts_x, axis=0)**2
        + np.diff(pts_y, axis=0)**2
        + np.diff(pts_z, axis=0)**2
    )

    # total length from starting point
    zeros = np.zeros(tuple(np.r_[1, length_orig.shape[1:]]), dtype=float)
    length0 = np.cumsum(np.concatenate((zeros, length_orig), axis=0), axis=0)
    length1 = np.concatenate((length_orig, length_orig[-1:, ...]), axis=0)

    # ---------------------------
    # changes to pts

    if radius_max is not None:
        pts_x, pts_y, pts_z, i0 = coll.get_rays_intersect_radius(
            key=key,
            key_cam=key_cam,
            segment=segment,
            axis_radius=radius_max,
            lim_to_segments=True,
            return_pts=True,
            return_itot=True,
        )[3:]

        # ind_flat
        if ind_ch is not None:
            sli = tuple([slice(None)] + list(ind_ch))
            pts_x = pts_x[sli]
            pts_y = pts_y[sli]
            pts_z = pts_z[sli]
            i0 = i0[sli]

        length_rad = np.sqrt(
            np.diff(pts_x, axis=0)**2
            + np.diff(pts_y, axis=0)**2
            + np.diff(pts_z, axis=0)**2
        )

        if segment is not None:
            length0 = length0[iseg, ...]
            length1 = length1[iseg, ...]

    else:
        if segment is not None:
            i0 = i0[iseg]
            pts_x = pts_x[i0, ...]
            pts_y = pts_y[i0, ...]
            pts_z = pts_z[i0, ...]
            length0 = length0[iseg, ...]
            length1 = length1[iseg, ...]

        length_rad = length0[1:, ...]

    # -----------
    # trivial
    # -----------

    if pts_x.size == 0 or not np.any(np.isfinite(pts_x)):
        return [None]*len(return_coords)

    # --------------------
    # compute sampling
    # ---------------------

    # ------------------
    # relative sampling
    # => each segment / pixel has ame number of points

    # rel
    if mode == 'rel':

        # make sure npts allow to describe all integer indices
        nptsi = (i0[-1] - i0[0]) * int(np.ceil(1./res))
        N = int(np.ceil((nptsi - 1) / (i0[-1] - i0[0])))
        nptsi = N * (i0[-1] - i0[0]) + 1
        itot = np.linspace(i0[0], i0[-1], nptsi)

        # interpolate
        if out_xyz:
            pts_x = scpinterp.interp1d(
                i0,
                pts_x,
                kind='linear',
                axis=0,
            )(itot)
            pts_y = scpinterp.interp1d(
                i0,
                pts_y,
                kind='linear',
                axis=0,
            )(itot)
            pts_z = scpinterp.interp1d(
                i0,
                pts_z,
                kind='linear',
                axis=0,
            )(itot)

        # get length
        if out_l:
            i1 = np.floor(itot).astype(int)
            i1[i1 == length1.shape[0] - 1] -= 1
            length = length1[i1, ...]
            lengthtot = length0[i1, ...]

    # --------------------
    # absolute sampling
    # different nb of pts per segment / pixel

    else:

        # get indices of valid pts
        iok = np.isfinite(pts_x)

        # get shape of arrays based on max nb of points ()
        nn = np.ceil(length_rad / res).astype(int)
        nmax = np.nansum(nn, axis=0) + 1
        shape = tuple(np.r_[np.nanmax(nmax), length_rad.shape[1:]])

        # initialize arrays
        itot = np.full(shape, np.nan)

        if out_xyz:
            lpx = np.full(shape, np.nan)
            lpy = np.full(shape, np.nan)
            lpz = np.full(shape, np.nan)

        if out_l:
            llen = np.full(shape, np.nan)
            lentot = np.full(shape, np.nan)

        # loop in pixels
        for ind in itt.product(*[range(ss) for ss in pts_x.shape[1:]]):

            sli0 = tuple([slice(None)] + list(ind))
            ioki = iok[sli0]
            iokin = ioki[:-1] * ioki[1:]

            # trivial case
            if not np.any(iokin):
                continue

            # get sli2 and i0i
            sli2 = tuple([ioki] + list(ind))
            if radius_max is None:
                i0i = i0[ioki]
            else:
                i0i = i0[sli2]
            nni = nn[tuple([iokin] + list(ind))]

            # itoti
            itoti = []
            anyok = False
            for jj in range(i0i.size - 1):
                if np.all(np.isfinite(i0i[jj:jj+2])):
                    itoti.append(
                        np.linspace(
                            i0i[jj],
                            i0i[jj+1],
                            nni[jj] + 1,
                        )[:-1]
                    )
                    anyok = True

            if anyok is False:
                continue

            if np.isfinite(i0i[-1]):
                itoti.append([i0i[-1]])

            itoti = np.concatenate(tuple(itoti))

            # safety check
            ni = itoti.size
            if nmax[ind] != ni:
                msg = (
                    "Mismatch between:\n"
                    f"\t- nmax[{ind}] = {nmax[ind]}\n"
                    f"\t- itoti.size = {itoti.size}\n"
                )
                raise Exception(msg)

            slin = tuple([np.arange(ni)] + list(ind))
            itot[slin] = itoti

            # interpolate
            if out_xyz:
                lpx[slin] = scpinterp.interp1d(
                    i0i,
                    pts_x[sli2],
                    kind='linear',
                    axis=0,
                )(itoti)

                lpy[slin] = scpinterp.interp1d(
                    i0i,
                    pts_y[sli2],
                    kind='linear',
                    axis=0,
                )(itoti)

                lpz[slin] = scpinterp.interp1d(
                    i0i,
                    pts_z[sli2],
                    kind='linear',
                    axis=0,
                )(itoti)

            if out_l:
                i1 = np.floor(itoti).astype(int)
                i1[i1 == ioki.sum()] -= 1

                llen[slin] = length1[sli2][i1]
                lentot[slin] = length0[sli2][i1]

        if out_xyz:
            pts_x, pts_y, pts_z = lpx, lpy, lpz
        if out_l:
            length = llen
            lengthtot = lentot

    # -------------
    # adjust npts
    # -------------

    iok = np.any(np.isfinite(itot), axis=tuple(range(1, itot.ndim)))
    if not np.all(iok):
        sli = (iok,) + tuple([slice(None) for ii in itot.shape[1:]])

        itot = itot[sli]
        if out_xyz:
            pts_x = pts_x[sli]
            pts_y = pts_y[sli]
            pts_z = pts_z[sli]
        if out_l:
            length = length[sli]
            lengthtot = lengthtot[sli]

    # -------------------------------------
    # optional concatenation (for plotting)
    # -------------------------------------

    if concatenate is True:

        # if mode == 'rel':
        shape_nan = tuple(np.r_[np.r_[1], pts_x.shape[1:]])
        nan = np.full(shape_nan, np.nan)

        if out_k:
            itot2 = np.full(pts_x.shape, np.nan)
            for ii in range(pts_x.shape[0]):
                itot2[ii, ...] = itot[ii]
            itot = remove_consecutive_nans(
                np.concatenate((itot2, nan), axis=0).T.ravel()
            )

        if out_xyz:
            pts_x = remove_consecutive_nans(
                np.concatenate((pts_x, nan), axis=0).T.ravel()
            )
            pts_y = remove_consecutive_nans(
                np.concatenate((pts_y, nan), axis=0).T.ravel()
            )
            pts_z = remove_consecutive_nans(
                np.concatenate((pts_z, nan), axis=0).T.ravel()
            )

        if out_l:
            length = remove_consecutive_nans(
                np.concatenate((length, nan), axis=0).T.ravel()
            )
            lengthtot = remove_consecutive_nans(
                np.concatenate((lengthtot, nan), axis=0).T.ravel()
            )

    # -------------
    # adjust kk
    # -------------

    if out_k:
        kk = itot - np.floor(itot)
        kk[itot == np.nanmax(itot)] = 1.

    # -------------
    # return
    # -------------

    lout = []
    for cc in return_coords:
        if cc == 'x':
            lout.append(pts_x)

        elif cc == 'y':
            lout.append(pts_y)

        elif cc == 'z':
            lout.append(pts_z)

        elif cc == 'R':
            lout.append(np.hypot(pts_x, pts_y))

        elif cc == 'phi':
            lout.append(np.arctan2(pts_y, pts_x))

        elif cc == 'ang_vs_ephi':
            phi = np.arctan2(pts_y, pts_x)
            ux = np.diff(pts_x, axis=0)
            uy = np.diff(pts_y, axis=0)
            ux = np.concatenate((ux[0:1, ...], ux), axis=0)
            uy = np.concatenate((uy[0:1, ...], uy), axis=0)
            vn = np.sqrt(ux**2 + uy**2)
            ux = ux / vn
            uy = uy / vn
            lout.append(np.arccos(-np.sin(phi)*ux + np.cos(phi)*uy))

        elif cc == 'itot':
            lout.append(itot)

        elif cc == 'k':
            lout.append(kk)

        elif cc == 'l':
            lout.append(kk*length)

        elif cc == 'ltot':
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.subplot(1,4,1)
            # plt.plot(kk)
            # plt.subplot(1,4,2)
            # plt.plot(length)
            # plt.subplot(1,4,3)
            # plt.plot(lengthtot)
            # plt.subplot(1,4,4)
            # plt.plot(kk*length + lengthtot)
            # import pdb; pdb.set_trace()     # DB

            lout.append(kk*length + lengthtot)

    return lout


# ###############################################################
#                   check inputs
# ###############################################################


def _sample_check(
    res=None,
    mode=None,
    segment=None,
    radius_max=None,
    concatenate=None,
    return_coords=None,
):

    # -------------
    # booleans
    # -------------

    # ---------
    # res

    res = ds._generic_check._check_var(
        res, 'res',
        types=float,
        default=0.1,
        sign='> 0',
    )

    # ---------
    # mode

    mode = ds._generic_check._check_var(
        mode, 'mode',
        types=str,
        default='rel',
        allowed=['rel', 'abs'],
    )

    # ------------
    # concatenate

    concatenate = ds._generic_check._check_var(
        concatenate, 'concatenate',
        types=bool,
        default=False,
    )

    # ----------------
    # segment indices
    # ----------------

    if segment is not None:
        segment = np.atleast_1d(segment).astype(int).ravel()

    # --------------------
    # tangency_radius_max

    if radius_max is not None:
        radius_max = ds._generic_check._check_var(
            radius_max, 'radius_max',
            types=(float, int),
            sign='> 0',
        )

        if mode == 'rel':
            msg = "radius_max can only be used with mode='abs'!"
            raise Exception(msg)

    # -------------
    # return_coords
    # -------------

    if isinstance(return_coords, str):
        return_coords = [return_coords]

    lok_k = ['k', 'l', 'ltot', 'itot']
    lok_xyz = ['x', 'y', 'z', 'R', 'phi', 'ang_vs_ephi']
    return_coords = ds._generic_check._check_var_iter(
        return_coords, 'return_coords',
        types=list,
        types_iter=str,
        default=['x', 'y', 'z'],
        allowed=lok_k + lok_xyz,
    )

    out_xyz = any([ss in return_coords for ss in lok_xyz])
    out_k = any([ss in return_coords for ss in lok_k])
    out_l = any(['l' in return_coords, 'ltot' in return_coords])

    return (
        res, mode, concatenate,
        segment, radius_max,
        return_coords, out_xyz, out_k, out_l
    )


# ###############################################################
#                   check indices of channels
# ###############################################################


def _check_ind_channels(
    ind_ch=None,
    shape=None,
    key=None,
):
    """ Make sure ind_ch is a tuple (one dim per dim of shape)

    Make sure it is made of slice or np.ndarray
    To preserve the dimensions of the array even if ind_ch is a scalar

    Parameters
    ----------
    ind_ch : tuple of np.ndarrays, like from np.nonzero() or np.unravel_index()
        indices of the desired channels
    shape : TYPE, optional
        DESCRIPTION. The default is None.
    key : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    ind_ch : TYPE
        DESCRIPTION.

    """

    # ------------------------
    # special retro-compatible case
    # ------------------------

    if np.isscalar(ind_ch) and len(shape) > 1:
        ind_ch = np.unravel_index(ind_ch, shape)

    # ------------------------
    # basic conformity checks
    # ------------------------

    # single index
    if len(shape) == 1 and not isinstance(ind_ch, tuple):
        ind_ch = (ind_ch,)

    # check is tuple
    c0 = (
        isinstance(ind_ch, tuple)
        and len(ind_ch) == len(shape)
    )
    if not c0:
        _err_ind(ind_ch, shape, key)

    # check all parts are np.ndarrays
    ind_ch = tuple([
        ii if isinstance(ii, (slice, np.ndarray)) else np.r_[ii]
        for ii in ind_ch
    ])

    # ------------------------
    # test
    # ------------------------

    try:
        aa = np.empty(shape)[ind_ch]
        del aa
    except Exception as err:
        _err_ind(ind_ch, shape, key, err=err)

    return ind_ch


def _err_ind(ind_ch, shape, key, err=None):
    msg = (
        f"Arg ind_ch must be compatible with shape of rays '{key}':\n"
        f"\t- rays shape: {shape}\n"
        f"\t- ind_ch: {ind_ch}\n"
    )
    if err is not None:
        msg += "\n\n{err}"

    raise Exception(msg)


# ###############################################################
#                   remove consecutive nans
# ###############################################################


def remove_consecutive_nans(arr):

    # -----------------
    # preliminary check
    # -----------------

    assert arr.ndim == 1, f"arr must be 1d: {arr.shape}"

    # -------------------
    # get indices on nans
    # -------------------

    ind = np.nonzero(np.isnan(arr))[0]
    cons = np.r_[False, np.diff(ind) == 1]

    return np.delete(arr, ind[cons])