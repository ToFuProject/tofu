# -*- coding: utf-8 -*-


import itertools as itt


import numpy as np
import scipy.interpolate as scpinterp
import datastock as ds


from . import _class2_check as _check


# ###############################################################
# ###############################################################
#                   sample
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
        return_coords, out_xyz, out_k,
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

    pts_x, pts_y, pts_z = coll.get_rays_pts(key=key, key_cam=key_cam)
    npts = pts_x.shape[0]
    npix = np.prod(pts_x.shape[1:])
    i0 = np.arange(0, npts)

    # segment
    if segment is not None:
        segment[segment < 0] = npts - 1 + segment[segment < 0]
        iseg = np.r_[segment, segment[-1] + 1]

    # ind_flat
    if ind_flat is not None:
        ind_flat[ind_flat < 0] = npix - 1 + ind_flat[ind_flat < 0]

        pts_x = np.reshape(pts_x, (npts, -1))[:, ind_flat]
        pts_y = np.reshape(pts_y, (npts, -1))[:, ind_flat]
        pts_z = np.reshape(pts_z, (npts, -1))[:, ind_flat]

    # length
    length_orig = np.sqrt(
        np.diff(pts_x, axis=0)**2
        + np.diff(pts_y, axis=0)**2
        + np.diff(pts_z, axis=0)**2
    )
    zeros = np.zeros(tuple(np.r_[1, length_orig.shape[1:]]), dtype=float)
    length0 = np.cumsum(np.concatenate((zeros, length_orig), axis=0), axis=0)
    length1 = np.concatenate((length_orig, length_orig[-1:, ...]), axis=0)

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

    # -----------
    # compute
    # -----------

    # -------------------------
    # prepare sampling indices

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

        if 'l' in return_coords or 'ltot' in return_coords:
            i1 = np.floor(itot).astype(int)
            i1[i1 == length1.shape[0] - 1] -= 1
            length = length1[i1, ...]
            lengthtot = length0[i1, ...]

    # abs => for pts.ndim >= 3 (2d cameras and above), flattened list
    else:

        iok = np.isfinite(pts_x)
        nn = np.ceil(length_rad / res).astype(int)

        nan = np.r_[np.nan, np.nan]
        lpx, lpy, lpz, itot, llen, lentot = [], [], [], [], [], []
        for ind in itt.product(*[range(ss) for ss in pts_x.shape[1:]]):

            sli = tuple([slice(None)] + list(ind))
            ioki = iok[sli]
            iokin = ioki[:-1] * ioki[1:]

            if not np.any(iokin):
                itot.append(nan)
                if out_xyz:
                    lpx.append(nan)
                    lpy.append(nan)
                    lpz.append(nan)
                if 'l' in return_coords or 'ltot' in return_coords:
                    llen.append(nan)
                    lentot.append(nan)
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
            for jj in range(i0i.size - 1):
                if np.all(np.isfinite(i0i[jj:jj+2])):
                    itoti.append(
                        np.linspace(
                            i0i[jj],
                            i0i[jj+1],
                            nni[jj] + 1,
                        )[:-1]
                    )

            if np.isfinite(i0i[-1]):
                itoti.append([i0i[-1]])

            itoti = np.concatenate(tuple(itoti))
            itot.append(itoti)

            # interpolate
            if out_xyz:
                lpx.append(scpinterp.interp1d(
                    i0i,
                    pts_x[sli2],
                    kind='linear',
                    axis=0,
                )(itoti))

                lpy.append(scpinterp.interp1d(
                    i0i,
                    pts_y[sli2],
                    kind='linear',
                    axis=0,
                )(itoti))

                lpz.append(scpinterp.interp1d(
                    i0i,
                    pts_z[sli2],
                    kind='linear',
                    axis=0,
                )(itoti))

            if 'l' in return_coords or 'ltot' in return_coords:
                i1 = np.floor(itoti).astype(int)
                # i1[i1 == length0.shape[0] - 1] -= 1
                i1[i1 == ioki.sum()] -= 1
                llen.append(length1[sli2][i1])
                lentot.append(length0[sli2][i1])

        if out_xyz:
            pts_x, pts_y, pts_z = lpx, lpy, lpz
        if 'l' in return_coords or 'ltot' in return_coords:
            length = llen
            lengthtot = lentot

    # -------------------------------------
    # optional concatenation (for plotting)

    if concatenate is True:
        if mode == 'rel':
            shape = tuple(np.r_[np.r_[1], pts_x.shape[1:]])
            nan = np.full(shape, np.nan)
            if out_k:
                itot2 = np.full(pts_x.shape, np.nan)
                for ii in range(pts_x.shape[0]):
                    itot2[ii, ...] = itot[ii]
                itot = np.concatenate((itot2, nan), axis=0).T.ravel()
            if out_xyz:
                pts_x = np.concatenate((pts_x, nan), axis=0).T.ravel()
                pts_y = np.concatenate((pts_y, nan), axis=0).T.ravel()
                pts_z = np.concatenate((pts_z, nan), axis=0).T.ravel()
            if 'l' in return_coords or 'ltot' in return_coords:
                length = np.concatenate((length, nan), axis=0).T.ravel()
                lengthtot = np.concatenate((lengthtot, nan), axis=0).T.ravel()


        else:
            if out_k:
                itot = np.concatenate(
                    tuple([np.append(pp, np.nan) for pp in itot])
                    )
            if out_xyz:
                pts_x = np.concatenate(
                    tuple([np.append(pp, np.nan) for pp in pts_x])
                    )
                pts_y = np.concatenate(
                    tuple([np.append(pp, np.nan) for pp in pts_y])
                    )
                pts_z = np.concatenate(
                    tuple([np.append(pp, np.nan) for pp in pts_z])
                    )
            if 'l' in return_coords or 'ltot' in return_coords:
                length = np.concatenate(
                    tuple([np.append(pp, np.nan) for pp in length])
                    )
                lengthtot = np.concatenate(
                    tuple([np.append(pp, np.nan) for pp in lengthtot])
                    )

    # -------------
    # return

    if out_k:
        if concatenate is True or mode == 'rel':
            kk = itot - np.floor(itot)
            kk[itot == np.nanmax(itot)] = 1.
        else:
            kk = [ii - np.floor(ii) for ii in itot]

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
            if concatenate is True or mode == 'rel':
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
            else:
                lout.append([
                    kki * ll1 + ll0
                    for kki, ll1, ll0 in zip(kk, length, lengthtot)
                ])

    return lout


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

    return (
        res, mode, concatenate,
        segment, ind_flat, radius_max,
        return_coords, out_xyz, out_k,
    )


# ###############################################################
# ###############################################################
#                  tangency radius
# ###############################################################


def _tangency_radius_check(
    coll=None,
    key=None,
    key_cam=None,
    quantity=None,
    segment=None,
    lim_to_segments=None,
    axis_pt=None,
    axis_vect=None,
):

    # key
    key = _check._check_key(coll=coll, key=key, key_cam=key_cam)

    # quantity
    quantity = ds._generic_check._check_var(
        quantity, 'quantity',
        types=str,
        default='tangency radius',
        allowed=['alpha', 'tangency radius', 'length'],
    )

    # lim_to_segments
    lim_to_segments = ds._generic_check._check_var(
        lim_to_segments, 'lim_to_segments',
        types=bool,
        default=False,
    )

    # segment
    if segment is not None:
        segment = np.atleast_1d(segment).astype(int).ravel()

    # --------
    # tangency radius-specific

    if quantity == 'tangency radius':
        # axis_pt
        if axis_pt is None:
            axis_pt = [0., 0., 0.]

        axis_pt = ds._generic_check._check_flat1darray(
            axis_pt, 'axis_pt',
            dtype=float,
            size=3,
            can_be_None=False,
        )

        # axis_vect
        if axis_vect is None:
            axis_vect = [0., 0., 1.]

        axis_vect = ds._generic_check._check_flat1darray(
            axis_vect, 'axis_vect',
            dtype=float,
            size=3,
            norm=True,
            can_be_None=False,
        )

    return key, quantity, segment, lim_to_segments, axis_pt, axis_vect


def _tangency_radius_prepare(
    coll=None,
    key=None,
    quantity=None,
    segment=None,
    axis_pt=None,
    axis_vect=None,
):

    pts_x, pts_y, pts_z = coll.get_rays_pts(key=key)
    i0 = np.arange(0, pts_x.shape[0])

    # select segment
    if segment is not None:
        npts = pts_x.shape[0]
        segment[segment < 0] = npts - 1 + segment[segment < 0]

        iseg = np.r_[segment, segment[-1] + 1]
        i0 = i0[iseg]
        pts_x = pts_x[iseg, :]
        pts_y = pts_y[iseg, :]
        pts_z = pts_z[iseg, :]


    # define vectors
    ABx = np.diff(pts_x, axis=0)
    ABy = np.diff(pts_y, axis=0)
    ABz = np.diff(pts_z, axis=0)

    if quantity == 'tangency radius':
        AOx = axis_pt[0] - pts_x[:-1, ...]
        AOy = axis_pt[1] - pts_y[:-1, ...]
        AOz = axis_pt[2] - pts_z[:-1, ...]

        # --------
        # get kk

        # Eq:
        # k^2 |AB x v|^2 + 2k((AB.v)(AO.v) - AB.AO) + |AO x v|^2 = dist^2
        #
        # minimum:
        # k |AB x v|^2 + (AB.v)(AO.v) - AB.AO = 0

        ABvn2 = (
            (ABy*axis_vect[2] - ABz*axis_vect[1])**2
            + (ABz*axis_vect[0] - ABx*axis_vect[2])**2
            + (ABx*axis_vect[1] - ABy*axis_vect[0])**2
            )
        AOvn2 = (
            (AOy*axis_vect[2] - AOz*axis_vect[1])**2
            + (AOz*axis_vect[0] - AOx*axis_vect[2])**2
            + (AOx*axis_vect[1] - AOy*axis_vect[0])**2
            )

        ABv = ABx*axis_vect[0] + ABy*axis_vect[1] + ABz*axis_vect[2]
        AOv = AOx*axis_vect[0] + AOy*axis_vect[1] + AOz*axis_vect[2]
        ABAO = ABx*AOx + ABy*AOy + ABz*AOz
        B = ABv*AOv - ABAO

    else:
        AOx, AOy, AOz = None, None, None
        ABvn2, AOvn2, B = None, None, None
        i0 = None

    return (
        pts_x, pts_y, pts_z,
        ABx, ABy, ABz,
        AOx, AOy, AOz,
        ABvn2, AOvn2, B,
        i0,
    )


def _tangency_radius(
    coll=None,
    key=None,
    key_cam=None,
    quantity=None,
    # limits
    segment=None,
    lim_to_segments=None,
    # tangency radius
    axis_pt=None,
    axis_vect=None,
):

    # --------------
    # check inputs
    # --------------

    (
        key, quantity, segment, lim_to_segments, axis_pt, axis_vect,
    ) = _tangency_radius_check(
        coll=coll,
        key=key,
        key_cam=key_cam,
        quantity=quantity,
        segment=segment,
        lim_to_segments=lim_to_segments,
        axis_pt=axis_pt,
        axis_vect=axis_vect,
    )

    # -----------
    # compute
    # -----------

    # --------
    # prepare

    (
     pts_x, pts_y, pts_z,
     ABx, ABy, ABz,
     AOx, AOy, AOz,
     ABvn2, AOvn2, B,
     i0,
     ) = _tangency_radius_prepare(
         coll=coll,
         key=key,
         quantity=quantity,
         segment=segment,
         axis_pt=axis_pt,
         axis_vect=axis_vect,
    )

    # --------
    # radius

    if quantity == 'tangency radius':
        kk = np.zeros(ABx.shape, dtype=float)
        iok = ABvn2 > 0.
        kk[iok] = -B[iok] / ABvn2[iok]

        # ------------------------
        # lim_to_segments

        if lim_to_segments is True:
            i0 = kk < 0.
            i1 = kk > 1.

            kk[i0] = 0.
            kk[i1] = 1.

        out = np.sqrt(kk**2 * ABvn2 + 2*kk*B + AOvn2)

    elif quantity == 'length':
        out = np.sqrt(ABx**2 + ABy**2 + ABz**2)
        kk = None

    else:
        kdata = coll.dobj['rays'][key]['alpha']
        out = coll.ddata[kdata]['data']
        kk = None

    # -------
    # ref

    if segment is None:
        ref = coll.dobj['rays'][key]['ref']
    elif segment.size == 1:
        ref = coll.dobj['rays'][key]['ref'][1:]
        out = out[0]
        if quantity == 'tangency radius':
            kk = kk[0]
    else:
        ref = None

    # -------
    # return

    return out, kk, ref


def intersect_radius(
    coll=None,
    key=None,
    key_cam=None,
    axis_pt=None,
    axis_vect=None,
    axis_radius=None,
    segment=None,
    lim_to_segments=None,
    return_pts=None,
    return_itot=None,
):

    # --------------
    # check inputs
    # --------------

    (
     key, quantity, segment, lim_to_segments, axis_pt, axis_vect,
     ) = _tangency_radius_check(
         coll=coll,
         key=key,
         key_cam=key_cam,
         quantity='tangency radius',
         axis_pt=axis_pt,
         axis_vect=axis_vect,
         segment=segment,
         lim_to_segments=lim_to_segments,
    )

    # axis_radius
    axis_radius = ds._generic_check._check_var(
        axis_radius, 'axis_radius',
        types=(float, int),
        sign='> 0.',
    )

    # return_pts
    return_pts = ds._generic_check._check_var(
        return_pts, 'return_pts',
        types=bool,
        default=False,
    )

    if return_pts is True and lim_to_segments is False:
        msg = (
            "return_pts requires lim_to_segments = True"
            )
        raise Exception(msg)

    # return_itot
    return_itot = ds._generic_check._check_var(
        return_itot, 'return_itot',
        types=bool,
        default=False,
    )

    # -----------
    # compute
    # -----------

    # --------
    # prepare

    (
     pts_x, pts_y, pts_z,
     ABx, ABy, ABz,
     AOx, AOy, AOz,
     ABvn2, AOvn2, B,
     i0,
     ) = _tangency_radius_prepare(
         coll=coll,
         key=key,
         quantity=quantity,
         segment=segment,
         axis_pt=axis_pt,
         axis_vect=axis_vect,
         )

    # pre-select according to rad_min
    kmin = np.zeros(ABx.shape, dtype=float)

    iok = ABvn2 > 0.
    kmin[iok] = -B[iok] / ABvn2[iok]
    rad_min2 = kmin**2 * ABvn2 + 2*kmin*B + AOvn2
    rad_min2[np.abs(rad_min2) < 1e-9] = 0.
    rad_min = np.sqrt(rad_min2)

    # prepare solutions
    iin = (rad_min < axis_radius) & iok

    # there can be up to 2 solutions
    k0 = np.full(ABx.shape, np.nan)
    k1 = np.full(ABx.shape, np.nan)

    if np.any(iin):
        delta = B[iin]**2 - ABvn2[iin]*(AOvn2[iin] - axis_radius**2)

        k0[iin] = (-B[iin] - np.sqrt(delta)) / ABvn2[iin]
        k1[iin] = (-B[iin] + np.sqrt(delta)) / ABvn2[iin]

    # ----------------
    # lim_to_segments

    if lim_to_segments is True:

        ind = np.copy(iin)
        ind[ind] = (k0[ind] < 0) & (k1[ind] > 0)
        k0[ind] = 0.

        ind = np.copy(iin)
        ind[ind] = (k0[ind] < 1) & (k1[ind] > 1)
        k1[ind] = 1.

        ind = np.copy(iin)
        ind[ind] = (k0[ind] >= 0.) & (k0[ind] <= 1.)
        k0[~ind] = np.nan

        ind = np.copy(iin)
        ind[ind] = (k1[ind] >= 0.) & (k1[ind] <= 1.)
        k1[~ind] = np.nan

    # ----------------------
    # additional derivations

    iok = np.isfinite(k0) & np.isfinite(k1)

    if return_pts is True:

        iok2 = np.copy(iok)
        iok2[iok] = k1[iok] < 1.

        false = np.zeros(tuple(np.r_[1, iok.shape[1:]]), dtype=bool)
        iok0 = np.concatenate((iok, false), axis=0)
        iok02 = np.concatenate((iok2, false), axis=0)
        iok12 = np.concatenate((false, iok2), axis=0)

        # Make sure there a single continued sequence per ray
        # build index and check continuity

        px = np.full(pts_x.shape, np.nan)
        py = np.full(pts_x.shape, np.nan)
        pz = np.full(pts_x.shape, np.nan)

        px[iok0] = pts_x[iok0] + k0[iok] * ABx[iok]
        py[iok0] = pts_y[iok0] + k0[iok] * ABy[iok]
        pz[iok0] = pts_z[iok0] + k0[iok] * ABz[iok]

        px[iok12] = pts_x[iok02] + k1[iok2] * ABx[iok2]
        py[iok12] = pts_y[iok02] + k1[iok2] * ABy[iok2]
        pz[iok12] = pts_z[iok02] + k1[iok2] * ABz[iok2]
        px[-1, ...] = pts_x[-2, ...] + k1[-1, ...] * ABx[-1, ...]
        py[-1, ...] = pts_y[-2, ...] + k1[-1, ...] * ABy[-1, ...]
        pz[-1, ...] = pts_z[-2, ...] + k1[-1, ...] * ABz[-1, ...]

    if return_itot is True:

        i02 = np.full(pts_x.shape, np.nan)
        for ii in range(pts_x.shape[0]):
            i02[ii, ...] = i0[ii]

        itot = np.full(pts_x.shape, np.nan)

        itot[iok0] = i02[iok0] + k0[iok]
        itot[iok12] = i02[iok02] + k1[iok2]
        itot[-1, ...] = i02[-2, ...] + k1[-1, ...]

    # ------
    # return

    if return_pts is True and return_itot is True:
        return k0, k1, iok, px, py, pz, itot
    elif return_pts is True:
        return k0, k1, iok, px, py, pz
    elif return_itot is True:
        return k0, k1, iok, itot
    else:
        return k0, k1, iok
