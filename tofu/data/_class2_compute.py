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
    res=None,
    mode=None,
    segment=None,
    radius_max=None,
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

    # segment
    if segment is not None:
        segment = np.atleast_1d(segment).astype(int).ravel()

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

    # -----------
    # compute
    # -----------

    # ------------------------
    # get points of interest
    
    # changes to pts
    if radius_max is not None:
        pts_x, pts_y, pts_z = coll.get_rays_intersect_radius(
            key=key,
            segment=segment,
            axis_radius=radius_max,
            lim_to_segments=True,
            return_pts=True,
            )[3:]
        
    else:
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


    # abs => for pts.ndim >= 3 (2d cameras and above), flattened list
    else:
        
        iok = np.isfinite(pts_x)
        
        norm = np.sqrt(
            np.diff(pts_x, axis=0)**2
            + np.diff(pts_y, axis=0)**2
            + np.diff(pts_z, axis=0)**2
            )
        
        nn = np.ceil(norm / res).astype(int)
        i0 = np.arange(0, npts)
        
        lpx, lpy, lpz = [], [], []  
        for ind in itt.product(*[range(ss) for ss in pts_x.shape[1:]]):
            
            sli = tuple([slice(None)] + list(ind))
            if not np.any(iok[sli]):
                continue
            
            i0i = i0[iok[sli]]
            
            i1 = np.concatenate(tuple(
                [
                    np.linspace(
                        i0i[jj], i0i[jj+1], nn[tuple(np.r_[i0i[jj], ind])] + 1,
                        )[:-1]
                    for jj in range(i0i.size - 1)
                ]
                + [[i0i[-1]]]
                ))
        
            # interpolate
            lpx.append(scpinterp.interp1d(
                i0i,
                pts_x[sli],
                kind='linear',
                axis=0,
            )(i1))
            
            lpy.append(scpinterp.interp1d(
                i0i,
                pts_y[sli],
                kind='linear',
                axis=0,
            )(i1))
            
            lpz.append(scpinterp.interp1d(
                i0i,
                pts_z[sli],
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
            pts_x = np.concatenate(
                tuple([np.append(pp, np.nan) for pp in pts_x])
                )
            pts_y = np.concatenate(
                tuple([np.append(pp, np.nan) for pp in pts_y])
                )
            pts_z = np.concatenate(
                tuple([np.append(pp, np.nan) for pp in pts_z])
                )

    return pts_x, pts_y, pts_z


# ###############################################################
# ###############################################################
#                  tangency radius
# ###############################################################


def _tangency_radius_check(
    coll=None,
    key=None,
    axis_pt=None,
    axis_vect=None,
    segment=None,
    lim_to_segments=None,
    ):
    
    # key
    key = _check._check_key(coll=coll, key=key)
    
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
    
    # lim_to_segments
    lim_to_segments = ds._generic_check._check_var(
        lim_to_segments, 'lim_to_segments',
        types=bool,
        default=False,
    )
    
    # segment
    if segment is not None:
        segment = np.atleast_1d(segment).astype(int).ravel()

    return key, axis_pt, axis_vect, segment, lim_to_segments


def _tangency_radius_prepare(
        coll=None,
        key=None,
        segment=None,
        axis_pt=None,
        axis_vect=None,
        ):
    
    pts_x, pts_y, pts_z = coll.get_rays_pts(key=key)
    
    # select segment
    if segment is not None:
        npts = pts_x.shape[0]
        segment[segment < 0] = npts - 1 + segment[segment < 0]
        
        iseg = np.r_[segment, segment[-1] + 1]
        pts_x = pts_x[iseg, :]
        pts_y = pts_y[iseg, :]
        pts_z = pts_z[iseg, :]
    
    # define vectors
    ABx = np.diff(pts_x, axis=0)
    ABy = np.diff(pts_y, axis=0)
    ABz = np.diff(pts_z, axis=0)
    
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

    return (
        pts_x, pts_y, pts_z,
        ABx, ABy, ABz,
        AOx, AOy, AOz,
        ABvn2, AOvn2, B,
        )
        

def _tangency_radius(
    coll=None,
    key=None,
    axis_pt=None,
    axis_vect=None,
    segment=None,
    lim_to_segments=None,
    ):

    # --------------
    # check inputs
    # --------------
    
    (
     key, axis_pt, axis_vect, segment, lim_to_segments,
     ) = _tangency_radius_check(
         coll=coll,
         key=key,
         axis_pt=axis_pt,
         axis_vect=axis_vect,
         segment=segment,
         lim_to_segments=lim_to_segments,
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
     ) = _tangency_radius_prepare(
         coll=coll,
         key=key,
         segment=segment,
         axis_pt=axis_pt,
         axis_vect=axis_vect,
         )
    
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
        
    # --------
    # radius
        
    radius = np.sqrt(kk**2 * ABvn2 + 2*kk*B + AOvn2)
    
    # -------
    # ref
    
    if segment is None:
        ref = coll.dobj['rays'][key]['ref']
    elif segment.size == 1:
        ref = coll.dobj['rays'][key]['ref'][1:]
        kk = kk[0]
        radius = radius[0]
    else:
        ref = None
    
    return radius, kk, ref


def intersect_radius(
    coll=None,
    key=None,
    axis_pt=None,
    axis_vect=None,
    axis_radius=None,
    segment=None,
    lim_to_segments=None,
    return_pts=None,
    ):
    
    # --------------
    # check inputs
    # --------------
    
    (
     key, axis_pt, axis_vect, segment, lim_to_segments,
     ) = _tangency_radius_check(
         coll=coll,
         key=key,
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
     ) = _tangency_radius_prepare(
         coll=coll,
         key=key,
         segment=segment,
         axis_pt=axis_pt,
         axis_vect=axis_vect,
         )
    
    # pre-select according to rad_min
    kmin = np.zeros(ABx.shape, dtype=float)
    
    iok = ABvn2 > 0.
    kmin[iok] = -B[iok] / ABvn2[iok]
    rad_min = np.sqrt(kmin**2 * ABvn2 + 2*kmin*B + AOvn2)
    
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
    
    # ------
    # return
    
    iok = np.isfinite(k0) & np.isfinite(k1)
    iok2 = np.copy(iok)
    iok2[iok] = k1[iok] < 1.
    
    false = np.zeros(tuple(np.r_[1, iok.shape[1:]]), dtype=bool)
    iok0 = np.concatenate((iok, false), axis=0)
    iok1 = np.concatenate((false, iok), axis=0)
    iok02 = np.concatenate((iok2, false), axis=0)
    iok12 = np.concatenate((false, iok2), axis=0)

    if return_pts is True:
        shape = tuple(np.r_[iok.shape[0], np.ones((k0.ndim-1,), dtype=int)])
        
        # Make sure there a single continued sequence per ray 
        # build index and check continuity
        # ind = np.arange(0, k0.shape[0]).reshape(shape)
        # ind = np.zeros(k0.shape) + ind

        # ii = np.full(pts_x.shape, np.nan)
        # ii[iok0] = k0[iok] + ind[iok]
        # ii[iok12] = k1[iok2] + ind[iok2]
        # ii[-1, ...] = k1[-1, ...] + ind[-1, ...]

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
        
        return k0, k1, iok, px, py, pz
    
    else:
        return k0, k1, iok