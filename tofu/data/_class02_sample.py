# -*- coding: utf-8 -*-


import itertools as itt


import numpy as np
import scipy.interpolate as scpinterp
import datastock as ds


# ###############################################################
# ###############################################################
#                   main
# ###############################################################


def _sample(
    coll=None,
    key=None,
    key_cam=None,
    res=None,
    mode=None,
    segment=None,
    ind_flat=None,
    radius_max=None,
    concatenate=None,
    return_coords=None,
):

    # ------------
    # check inputs

    (
        res, mode, concatenate,
        segment, ind_flat, radius_max,
        return_coords, out_xyz, out_k, out_l,
    ) = _sample_check(
        res=res,
        mode=mode,
        concatenate=concatenate,
        segment=segment,
        ind_flat=ind_flat,
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
    npix = np.prod(pts_x.shape[1:])
    i0 = np.arange(0, npts)

    # ---------------------------
    # optional segment selection

    if segment is not None:
        segment[segment < 0] = npts - 1 + segment[segment < 0]
        iseg = np.r_[segment, segment[-1] + 1]

    # ---------------------------------------
    # optional ind_flat (selection of pixels)

    if ind_flat is not None:
        ind_flat[ind_flat < 0] = npix - 1 + ind_flat[ind_flat < 0]

        pts_x = np.reshape(pts_x, (npts, -1))[:, ind_flat]
        pts_y = np.reshape(pts_y, (npts, -1))[:, ind_flat]
        pts_z = np.reshape(pts_z, (npts, -1))[:, ind_flat]

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
        if ind_flat is not None:
            pts_x = np.reshape(pts_x, (pts_x.shape[0], -1))[:, ind_flat]
            pts_y = np.reshape(pts_y, (pts_x.shape[0], -1))[:, ind_flat]
            pts_z = np.reshape(pts_z, (pts_x.shape[0], -1))[:, ind_flat]
            i0 = np.reshape(i0, (pts_x.shape[0], -1))[:, ind_flat]

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
        # lpx, lpy, lpz, itot, llen, lentot = [], [], [], [], [], []
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
                # itot.append(nan)
                # if out_xyz:
                #     lpx.append(nan)
                #     lpy.append(nan)
                #     lpz.append(nan)
                # if 'l' in return_coords or 'ltot' in return_coords:
                #     llen.append(nan)
                #     lentot.append(nan)
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

            # itoti = np.concatenate(tuple(itoti))
            # itot.append(itoti)

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
                # i1[i1 == length0.shape[0] - 1] -= 1
                i1[i1 == ioki.sum()] -= 1

                llen[slin] = length1[sli2][i1]
                lentot[slin] = length0[sli2][i1]

                # llen.append(length1[sli2][i1])
                # lentot.append(length0[sli2][i1])

        if out_xyz:
            pts_x, pts_y, pts_z = lpx, lpy, lpz
        if out_l:
            length = llen
            lengthtot = lentot

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
            itot = np.concatenate((itot2, nan), axis=0).T.ravel()
        if out_xyz:
            pts_x = np.concatenate((pts_x, nan), axis=0).T.ravel()
            pts_y = np.concatenate((pts_y, nan), axis=0).T.ravel()
            pts_z = np.concatenate((pts_z, nan), axis=0).T.ravel()
        if out_l:
            length = np.concatenate((length, nan), axis=0).T.ravel()
            lengthtot = np.concatenate((lengthtot, nan), axis=0).T.ravel()

        # else:
        #     if out_k:
        #         itot = np.concatenate(
        #             tuple([np.append(pp, np.nan) for pp in itot])
        #             )
        #     if out_xyz:
        #         pts_x = np.concatenate(
        #             tuple([np.append(pp, np.nan) for pp in pts_x])
        #             )
        #         pts_y = np.concatenate(
        #             tuple([np.append(pp, np.nan) for pp in pts_y])
        #             )
        #         pts_z = np.concatenate(
        #             tuple([np.append(pp, np.nan) for pp in pts_z])
        #             )
        #     if 'l' in return_coords or 'ltot' in return_coords:
        #         length = np.concatenate(
        #             tuple([np.append(pp, np.nan) for pp in length])
        #             )
        #         lengthtot = np.concatenate(
        #             tuple([np.append(pp, np.nan) for pp in lengthtot])
        #             )

    # -------------
    # adjust npts
    # -------------

    iok = np.any(np.isfinite(itot), axis=0)
    if not np.all(iok):
        sli= tuple([itot] + [slice(None) for ss in itot.shape[1:]])
        itot = itot[sli]
        if out_xyz:
            pts_x = pts_x[sli]
            pts_y = pts_y[sli]
            pts_z = pts_z[sli]
        if out_l:
            length = length[sli]
            lengthtot = lengthtot[sli]

    # -------------
    # adjust kk
    # -------------

    if out_k:
        # if concatenate is True or mode == 'rel':
        kk = itot - np.floor(itot)
        kk[itot == np.nanmax(itot)] = 1.
        # else:
        #     kk = [ii - np.floor(ii) for ii in itot]
        #     for ij in range(len(kk)):
        #         kk[ij][itot[ij] == np.nanmax(itot[ij])] = 1.

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
            if concatenate is True or mode == 'rel':
                lout.append(np.hypot(pts_x, pts_y))
            else:
                lout.append([
                    np.hypot(px, py)
                    for px, py in zip(pts_x, pts_y)
                ])

        elif cc == 'phi':
            if concatenate is True or mode == 'rel':
                lout.append(np.arctan2(pts_y, pts_x))
            else:
                lout.append([
                    np.arctan2(py, px)
                    for px, py in zip(pts_x, pts_y)
                ])

        elif cc == 'ang_vs_ephi':
            if concatenate is True or mode == 'rel':
                phi = np.arctan2(pts_y, pts_x)
                ux = np.diff(pts_x, axis=0)
                uy = np.diff(pts_y, axis=0)
                ux = np.concatenate((ux[0:1, ...], ux), axis=0)
                uy = np.concatenate((uy[0:1, ...], uy), axis=0)
                vn = np.sqrt(ux**2 + uy**2)
                ux = ux / vn
                uy = uy / vn
                lout.append(np.arccos(-np.sin(phi)*ux + np.cos(phi)*uy))
            else:
                lout.append([])
                for px, py in zip(pts_x, pts_y):
                    phi = np.arctan2(py, px)
                    ux = np.diff(px, axis=0)
                    uy = np.diff(py, axis=0)
                    ux = np.concatenate((ux[0:1, ...], ux), axis=0)
                    uy = np.concatenate((uy[0:1, ...], uy), axis=0)
                    vn = np.sqrt(ux**2 + uy**2)
                    ux = ux / vn
                    uy = uy / vn
                    lout[-1].append(np.arccos(-np.sin(phi)*ux + np.cos(phi)*uy))

        elif cc == 'itot':
            lout.append(itot)

        elif cc == 'k':
            lout.append(kk)

        elif cc == 'l':
            if concatenate is True or mode == 'rel':
                lout.append(kk*length)
            else:
                lout.append([
                    kki * ll
                    for kki, ll in zip(kk, length)
                ])

        elif cc == 'ltot':
            # if concatenate is True or mode == 'rel':
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
            # else:
            #     lout.append([
            #         kki * ll1 + ll0
            #         for kki, ll1, ll0 in zip(kk, length, lengthtot)
            #     ])

    return lout


# ###############################################################
#                   check inputs
# ###############################################################


def _sample_check(
    res=None,
    mode=None,
    segment=None,
    ind_flat=None,
    radius_max=None,
    concatenate=None,
    return_coords=None,
):

    # res
    res = ds._generic_check._check_var(
        res, 'res',
        types=float,
        default=0.1,
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

    # segment
    if segment is not None:
        segment = np.atleast_1d(segment).astype(int).ravel()

    # ind_flat
    if ind_flat is not None:
        ind_flat = np.atleast_1d(ind_flat).astype(int).ravel()

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

    # return_coords
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
        segment, ind_flat, radius_max,
        return_coords, out_xyz, out_k, out_l
    )