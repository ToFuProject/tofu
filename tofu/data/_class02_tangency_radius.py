# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 18:23:48 2024

@author: dvezinet
"""


import numpy as np
import datastock as ds


from . import _class2_check as _check


# ###############################################################
# ###############################################################
#                  tangency radius
# ###############################################################


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

    if quantity == 'tangency_radius':
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


# ###############################################################
#              tangency radius - check
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
        default='tangency_radius',
        allowed=['alpha', 'tangency_radius', 'length'],
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

    if quantity == 'tangency_radius':
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


# ###############################################################
#            tangency radius - prepare data
# ###############################################################


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

    if quantity == 'tangency_radius':
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


# ###############################################################
# ###############################################################
#                  intersect radius
# ###############################################################


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
         quantity='tangency_radius',
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
