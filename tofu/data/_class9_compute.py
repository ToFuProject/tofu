# -*- coding: utf-8 -*-


# Built-in
import copy


# Common
import numpy as np
import scipy.integrate as scpinteg
import astropy.units as asunits


import datastock as ds


# #############################################################################
# #############################################################################
#                           Matrix - compute
# #############################################################################


def compute(
    coll=None,
    key=None,
    key_bsplines=None,
    key_diag=None,
    key_cam=None,
    # sampling
    res=None,
    mode=None,
    method=None,
    crop=None,
    # options
    brightness=None,
    # output
    store=None,
    verb=None,
):
    """ Compute the geometry matrix using:
            - a Plasma2DRect instance with a key to a bspline set
            - a cam instance with a resolution
    """

    # -----------
    # check input
    # -----------

    (
        key,
        key_bsplines, key_mesh, key_mesh0, mtype,
        key_diag, key_cam,
        method, res, mode, crop,
        brightness,
        store, verb,
    ) = _compute_check(
        coll=coll,
        key=key,
        key_bsplines=key_bsplines,
        key_diag=key_diag,
        key_cam=key_cam,
        # sampling
        method=method,
        res=res,
        mode=mode,
        crop=crop,
        # options
        brightness=brightness,
        # output
        store=store,
        verb=verb,
    )

    # -----------
    # prepare
    # -----------

    key_kR = coll.dobj['mesh'][key_mesh0]['knots'][0]
    radius_max = np.max(coll.ddata[key_kR]['data'])

    shapebs = coll.dobj['bsplines'][key_bsplines]['shape']

    # prepare indices
    indbs = coll.select_ind(
        key=key_bsplines,
        returnas=bool,
        crop=crop,
    )

    # prepare matrix
    is3d = False
    if mtype == 'polar':
        radius2d = coll.dobj[coll._which_mesh][key_mesh]['radius2d']
        r2d_reft = coll.get_time(key=radius2d)[2]
        if r2d_reft is not None:
            r2d_nt = coll.dref[r2d_reft]['size']
            if r2d_nt > 1:
                shapemat = tuple(np.r_[r2d_nt, None, indbs.sum()])
                is3d = True

    if not is3d:
        shapemat = tuple(np.r_[None, indbs.sum()])

    if verb is True:
        msg = f"Geom matrix for diag '{key_diag}':"
        print(msg)

    # -----------
    # compute
    # -----------

    if method == 'los':
        dout, units, axis = _compute_los(
            coll=coll,
            key_bsplines=key_bsplines,
            key_diag=key_diag,
            key_cam=key_cam,
            # sampling
            indbs=indbs,
            res=res,
            mode=mode,
            radius_max=radius_max,
            # groupby=groupby,
            is3d=is3d,
            # other
            shapemat=shapemat,
            brightness=brightness,
            verb=verb,
        )

    else:
        raise NotImplementedError()

    # ---------------
    # store / return
    # ---------------

    if store:
        _store(
            coll=coll,
            key=key,
            key_bsplines=key_bsplines,
            key_diag=key_diag,
            key_cam=key_cam,
            method=method,
            res=res,
            crop=crop,
            dout=dout,
            units=units,
            axis=axis,
        )

    else:
        return mat


# ###################
#   checking
# ###################


def _compute_check(
    coll=None,
    key=None,
    key_bsplines=None,
    key_diag=None,
    key_cam=None,
    # sampling
    method=None,
    res=None,
    mode=None,
    crop=None,
    # options
    brightness=None,
    # output
    store=None,
    verb=None,
):

    # key
    key = ds._generic_check._obj_key(
        d0=coll.dobj.get('geom matrix', {}),
        short='gmat',
        key=key,
    )

    # key_bsplines
    lk = list(coll.dobj.get('bsplines', {}).keys())
    key_bsplines = ds._generic_check._check_var(
        key_bsplines, 'key_bsplines',
        types=str,
        allowed=lk,
    )

    # key_mesh0
    key_mesh = coll.dobj['bsplines'][key_bsplines]['mesh']
    mtype = coll.dobj['mesh'][key_mesh]['type']
    if mtype == 'polar':
        key_mesh0 = coll.dobj['mesh'][key_mesh]['submesh']
    else:
        key_mesh0 = key_mesh

    # key_diag, key_cam
    key_diag, key_cam = coll.get_diagnostic_cam(key=key_diag, key_cam=key_cam)

    # method
    method = ds._generic_check._check_var(
        method, 'method',
        default='los',
        types=str,
        allowed=['los'],
    )

    # res
    res = ds._generic_check._check_var(
        res, 'res',
        default=0.01,
        types=float,
        sign='> 0.',
    )

    # mode
    mode = ds._generic_check._check_var(
        mode, 'mode',
        default='abs',
        types=str,
        allowed=['abs', 'rel'],
    )

    # crop
    crop = ds._generic_check._check_var(
        crop, 'crop',
        default=True,
        types=bool,
    )
    crop = (
        crop
        and coll.dobj['bsplines'][key_bsplines]['crop'] not in [None, False]
    )

    # brightness
    brightness = ds._generic_check._check_var(
        brightness, 'brightness',
        types=bool,
        default=False,
    )

    # store
    store = ds._generic_check._check_var(
        store, 'store',
        default=True,
        types=bool,
    )

    # verb
    if verb is None:
        verb = True
    if not isinstance(verb, bool):
        msg = (
            f"Arg verb must be a bool!\n"
            f"\t- provided: {verb}"
        )
        raise Exception(msg)

    return (
        key,
        key_bsplines, key_mesh, key_mesh0, mtype,
        key_diag, key_cam,
        method, res, mode, crop,
        brightness,
        store, verb,
    )


# ###################
#   compute_los                   
# ###################


def _compute_los(
    coll=None,
    key_bsplines=None,
    key_diag=None,
    key_cam=None,
    # sampling
    indbs=None,
    res=None,
    mode=None,
    key_integrand=None,
    radius_max=None,
    is3d=None,
    # other
    shapemat=None,
    brightness=None,
    verb=None,
):

    # ----------------
    # loop on cameras

    dout = {}
    doptics = coll.dobj['diagnostic'][key_diag]['doptics']
    for k0 in key_cam:

        npix = coll.dobj['camera'][k0]['dgeom']['pix_nb']
        key_los = doptics[k0]['los']

        sh = tuple([npix if ss is None else ss for ss in shapemat])
        mat = np.zeros(sh, dtype=float)

        # -----------------------
        # loop on group of pixels (to limit memory footprint)

        for ii in range(npix):

            # verb
            if verb is True:
                msg = f"\t camera '{k0}': pixel {ii + 1} / {npix}"
                end = '\n' if ii == npix - 1 else '\r'
                print(msg, flush=True, end=end)

            # sample los
            R, Z, length = coll.sample_rays(
                key=key_los,
                res=res,
                mode=mode,
                segment=None,
                ind_flat=ii,
                radius_max=radius_max,
                concatenate=False,
                return_coords=['R', 'z', 'ltot'],
            )

            # -------------
            # interpolate

            datai, units, refi = coll.interpolate_profile2d(
                key=key_bsplines,
                R=R[0],
                Z=Z[0],
                grid=False,
                azone=None,
                indbs=indbs,
                details=True,
                reshape=False,
                crop=None,
                nan0=True,
                val_out=np.nan,
                return_params=False,
                store=False,
            )

            axis = refi.index(None)
            iok = np.isfinite(datai)

            if not np.any(iok):
                continue

            datai[~iok] = 0.

            # ------------
            # integrate

            assert datai.ndim in [2, 3], datai.shape

            # integrate
            if is3d:
                mat[:, ii, :] = scpinteg.simpson(
                    datai,
                    x=length[0],
                    axis=axis,
                )
            elif datai.ndim == 3 and datai.shape[0] == 1:
                mat[ii, :] = scpinteg.simpson(
                    datai[0, ...],
                    x=length[0],
                    axis=axis,
                )
                # mat[ii, :] = np.nansum(mati[0, ...], axis=0) * reseff[ii]
            else:
                mat[ii, :] = scpinteg.simpson(
                    datai,
                    x=length[0],
                    axis=axis,
                )

        # --------------
        # post-treatment

        # brightness
        if brightness is False:
            ketend = doptics[k0]['etendue']
            etend = coll.ddata[ketend]['data']
            sh_etend = [-1 if aa == axis else 1 for aa in range(len(refi))]
            mat *= etend.reshape(sh_etend)

        # set ref
        refi = list(refi)
        refi[axis] = coll.dobj['camera'][k0]['dgeom']['ref_flat']
        refi = tuple(np.r_[refi[:axis], refi[axis], refi[axis+1:]])

        # fill dout
        dout[k0] = {
            'data': mat,
            'ref': refi,
        }

    # -----
    # units

    units = asunits.m
    if brightness is False:
        units = units * coll.ddata[ketend]['units']

    return dout, units, axis


# ###################
#   compute_vos                   
# ###################


def _compute_vos(
    coll=None,
    is2d=None,
    key_diag=None,
    key_cam=None,
    res=None,
    mode=None,
    key_integrand=None,
    radius_max=None,
    groupby=None,
    val_init=None,
    brightness=None,
):



    return None, None


# ###################
#   storing                   
# ###################


def _store(
    coll=None,
    key=None,
    key_bsplines=None,
    key_diag=None,
    key_cam=None,
    method=None,
    res=None,
    crop=None,
    dout=None,
    units=None,
    axis=None,
):

    # add data
    ddata = {}
    for k0, v0 in dout.items():
        ki = f"{key}_{k0}"
        ddata[ki] = {
            'data': v0['data'],
            'ref': v0['ref'],
            'units': units,
        }

    # shapes
    shapes = [v0['data'].shape for v0 in dout.values()]
    assert all([len(ss) == len(shapes[0]) for ss in shapes[1:]])
    shapes = np.array(shapes)
    assert np.allclose(shapes[1:, :axis], shapes[0:1, :axis])
    assert np.allclose(shapes[1:, axis+1:], shapes[0:1, axis+1:])

    # add matrix obj
    dobj = {
        'geom matrix': {
            key: {
                'data': [f"{key}_{k0}" for k0 in key_cam],
                'bsplines': key_bsplines,
                'diagnostic': key_diag,
                'camera': key_cam,
                'method': method,
                'res': res,
                'crop': crop,
                'shape': tuple(shapes[0, :]),
                'axis_chan': axis,
            },
        },
    }

    coll.update(ddata=ddata, dobj=dobj)


# ##################################################################
# ##################################################################
#               retrofit                   
# ##################################################################


def _concatenate(
    coll=None,
    key=None,
):

    # ------------
    # check inputs

    lok = list(coll.dobj.get('geom matrix', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # -----------
    # concatenate

    key_data = coll.dobj['geom matrix'][key]['data']
    key_cam = coll.dobj['geom matrix'][key]['camera']
    axis = coll.dobj['geom matrix'][key]['axis_chan']

    ref = list(coll.ddata[key_data[0]]['ref'])
    ref[axis] = None

    ind = 0
    dind = {}
    ldata = []
    for ii, k0 in enumerate(key_cam):
        datai = coll.ddata[key_data[ii]]['data']
        dind[k0] = ind + np.arange(0, datai.shape[axis])
        ldata.append(datai)
        ind += datai.shape[axis]

    data = np.concatenate(ldata, axis=axis)

    return data, ref, dind
