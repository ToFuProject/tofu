# -*- coding: utf-8 -*-


# Built-in
import copy
import warnings


# Common
import numpy as np
import scipy.integrate as scpinteg
import astropy.units as asunits


import datastock as ds


from . import _generic_plot


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
    dvos=None,
    # common ref
    ref_com=None,
    ref_vector_strategy=None,
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
        key_bs, key_m, ndim,
        subkey, key_bs0, key_m0,
        key_diag, key_cam,
        radius_max, method, res, mode, crop,
        dvos,
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
        dvos=dvos,
        # options
        brightness=brightness,
        # output
        store=store,
        verb=verb,
    )

    # -----------
    # prepare
    # -----------

    # prepare indices
    indbs = coll.select_ind(
        key=key_bs,
        returnas=bool,
        crop=crop,
    )

    # prepare slicing
    shape_mat, sli_mat, axis_pix, axis_bs, axis_other = _prepare(
        coll=coll,
        indbs=indbs,
        key_bs0=key_bs0,
        subkey=subkey,
    )

    if verb is True:
        msg = f"Geom matrix for diag '{key_diag}' and bs '{key_bs}':"
        print(msg)

    # -----------
    # compute
    # -----------

    if method == 'los':
        dout, axis = _compute_los(
            coll=coll,
            key=key,
            key_bs=key_bs,
            key_diag=key_diag,
            key_cam=key_cam,
            # sampling
            indbs=indbs,
            res=res,
            mode=mode,
            radius_max=radius_max,
            # common ref
            ref_com=ref_com,
            ref_vector_strategy=ref_vector_strategy,
            # groupby=groupby,
            shape_mat=shape_mat,
            sli_mat=sli_mat,
            axis_pix=axis_pix,
            # other
            brightness=brightness,
            verb=verb,
        )

    else:
        dout, axis = _compute_vos(
            coll=coll,
            key=key,
            key_bs=key_bs,
            key_diag=key_diag,
            key_cam=key_cam,
            # dvos
            dvos=dvos,
            # sampling
            indbs=indbs,
            res=res,
            mode=mode,
            radius_max=radius_max,
            # common ref
            ref_com=ref_com,
            ref_vector_strategy=ref_vector_strategy,
            # groupby=groupby,
            shape_mat=shape_mat,
            sli_mat=sli_mat,
            axis_pix=axis_pix,
            # other
            brightness=brightness,
            verb=verb,
        )

    # ---------------
    # check
    # ---------------

    if axis is None:
        _no_interaction(
            coll=coll,
            key=key,
            key_bs=key_bs,
            key_diag=key_diag,
            key_cam=key_cam,
        )
        store = False
        import pdb; pdb.set_trace() # DB

    # ---------------
    # store / return
    # ---------------

    if store:
        _store(
            coll=coll,
            key=key,
            key_bs=key_bs,
            key_diag=key_diag,
            key_cam=key_cam,
            method=method,
            res=res,
            crop=crop,
            dout=dout,
            axis_chan=axis,
            axis_bs=axis_bs,
            axis_other=axis_other,
        )

    else:
        return dout


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
    dvos=None,
    # options
    brightness=None,
    # output
    store=None,
    verb=None,
):

    # --------------
    # keys

    wm = coll._which_mesh
    wbs = coll._which_bsplines

    # key
    key = ds._generic_check._obj_key(
        d0=coll.dobj.get('geom matrix', {}),
        short='gmat',
        key=key,
    )

    # key_diag, key_cam
    key_diag, key_cam = coll.get_diagnostic_cam(
        key=key_diag,
        key_cam=key_cam,
    )

    spectro = coll.dobj['diagnostic'][key_diag]['spectro']
    if spectro:
        msg = (
            "Geometry matrix can only be computed for non-spectro diags"
        )
        raise Exception(msg)

    # key_bs
    lk = list(coll.dobj.get(wbs, {}).keys())
    key_bs = ds._generic_check._check_var(
        key_bsplines, 'key_bsplines',
        types=str,
        allowed=lk,
    )

    # key_m
    key_m = coll.dobj[wbs][key_bs][wm]
    submesh = coll.dobj[wm][key_m]['submesh']
    if submesh is not None:
        key_m0 = submesh
        key_bs0 = coll.dobj[wm][key_m]['subbs']
        subkey = coll.dobj[wm][key_m]['subkey'][0]
    else:
        key_m0, key_bs0, subkey = None, None, None

    # -------------------
    # dimensions and axis

    if submesh is None:
        key_kR = coll.dobj[wm][key_m]['knots'][0]
        ndim = len(coll.dobj[wbs][key_bs]['shape'])
    else:
        key_kR = coll.dobj[wm][key_m0]['knots'][0]
        ndim = len(coll.ddata[subkey]['shape'])

    radius_max = np.max(coll.ddata[key_kR]['data'])

    # --------------
    # parameters

    # method
    method = ds._generic_check._check_var(
        method, 'method',
        default='los',
        types=str,
        allowed=['los', 'vos'],
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
        and coll.dobj[wbs][key_bs]['crop'] not in [None, False]
    )

    # dvos
    if method == 'vos':
        dvos = coll.check_diagnostic_dvos(
            key=key_diag,
            key_cam=key_cam,
            dvos=dvos,
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
        key_bs, key_m, ndim,
        subkey, key_bs0, key_m0,
        key_diag, key_cam,
        radius_max, method, res, mode, crop,
        dvos,
        brightness,
        store, verb,
    )


# ###################
#   prepare
# ###################


def _prepare(
    coll=None,
    indbs=None,
    key_bs0=None,
    subkey=None,
):

    # shapes
    nbs = indbs.sum()
    nchan = None        # depends on cam

    # cases
    wm = coll._which_mesh
    wbs = coll._which_bsplines
    if subkey is None:
        shape_mat = (nchan, nbs)

        sli_mat = [None, slice(None)]
        axis_pix = 0
        axis_other = None

    else:

        refbs = coll.dobj[wbs][key_bs0]['ref']
        ref = coll.ddata[subkey]['ref']
        sh = list(coll.ddata[subkey]['shape'])
        axis_pix = ref.index(refbs[0])

        if len(refbs) == 1:  # tri
            sh = sh[:axis_pix+1] + [None] + sh[axis_pix+1:]
        axis_bs = axis_pix + 1

        axis_other = [ii for ii, rr in enumerate(ref) if rr not in refbs]
        if len(axis_other) == 1:
            axis_other = axis_other[0]
            sh[axis_pix] = nchan
            sh[axis_bs] = nbs
            shape_mat = tuple(sh)

            sli_mat = [None, None, None]
            sli_mat[axis_bs] = slice(None)
            sli_mat[axis_other] = slice(None)
        else:
            shape_mat = (nchan, nbs)

            sli_mat = [None, slice(None)]
            axis_pix = 0
            axis_other = None

    axis_bs = axis_pix + 1

    return shape_mat, sli_mat, axis_pix, axis_bs, axis_other


# ###################
#   compute_los
# ###################


def _compute_los(
    coll=None,
    key=None,
    key_bs=None,
    key_diag=None,
    key_cam=None,
    # sampling
    indbs=None,
    res=None,
    mode=None,
    key_integrand=None,
    radius_max=None,
    is3d=None,
    # common ref
    ref_com=None,
    ref_vector_strategy=None,
    # slicing
    shape_mat=None,
    sli_mat=None,
    axis_pix=None,
    # parameters
    brightness=None,
    verb=None,
):

    # -----
    # units

    units = asunits.m
    units_coefs = asunits.Unit()

    # ----------------
    # loop on cameras

    dout = {}
    doptics = coll.dobj['diagnostic'][key_diag]['doptics']
    for k0 in key_cam:

        npix = coll.dobj['camera'][k0]['dgeom']['pix_nb']
        key_los = doptics[k0]['los']
        key_mat = f'{key}_{k0}'

        sh = tuple([npix if ss is None else ss for ss in shape_mat])
        mat = np.zeros(sh, dtype=float)

        # -----------------------
        # loop on group of pixels (to limit memory footprint)

        anyok = False
        for ii in range(npix):

            # verb
            if verb is True:
                msg = (
                    f"\t- '{key_mat}' for cam '{k0}': pixel {ii + 1} / {npix}"
                    f"\t{(mat > 0).sum()} / {mat.size}\t\t"
                )
                end = '\n' if ii == npix - 1 else '\r'
                print(msg, flush=True, end=end)

            # sample los
            out_sample = coll.sample_rays(
                key=key_los,
                res=res,
                mode=mode,
                segment=None,
                ind_flat=ii,
                radius_max=radius_max,
                concatenate=False,
                return_coords=['R', 'z', 'ltot'],
            )

            if out_sample is None:
                continue

            R, Z, length = out_sample

            # -------------
            # interpolate

            # datai, units, refi = coll.interpolate(
            douti = coll.interpolate(
                keys=None,
                ref_key=key_bs,
                # interpolation pts
                x0=R[0],
                x1=Z[0],
                submesh=True,
                grid=False,
                # common ref
                ref_com=ref_com,
                ref_vector_strategy=ref_vector_strategy,
                # bsplines-specific
                # azone=None,
                indbs_tf=indbs,
                details=True,
                crop=None,
                nan0=True,
                val_out=np.nan,
                return_params=False,
                store=False,
            )[f'{key_bs}_details']

            datai, refi = douti['data'], douti['ref']
            axis = refi.index(None)
            iok = np.isfinite(datai)

            if not np.any(iok):
                continue

            datai[~iok] = 0.

            # ------------
            # integrate

            # check and update slice
            assert datai.ndim in [2, 3], datai.shape
            sli_mat[axis_pix] = ii

            # integrate
            mat[tuple(sli_mat)] = scpinteg.simpson(
                datai,
                x=length[0],
                axis=axis,
            )

            anyok = True

        # --------------
        # post-treatment

        if anyok:
            # brightness
            if brightness is False:
                ketend = doptics[k0]['etendue']
                units_coefs = coll.ddata[ketend]['units']
                etend = coll.ddata[ketend]['data']
                sh_etend = [-1 if aa == axis else 1 for aa in range(len(refi))]
                mat *= etend.reshape(sh_etend)

            # set ref
            refi = list(refi)
            refi[axis] = coll.dobj['camera'][k0]['dgeom']['ref_flat']
            refi = tuple(np.r_[refi[:axis], refi[axis], refi[axis+1:]])

        else:
            refi = None
            axis = None

        # fill dout
        dout[key_mat] = {
            'data': mat,
            'ref': refi,
            'units': units * units_coefs,
        }

    return dout, axis


# ###################
#   compute_vos
# ###################


def _compute_vos(
    coll=None,
    key=None,
    key_bs=None,
    key_diag=None,
    key_cam=None,
    # dvos
    dvos=None,
    # sampling
    indbs=None,
    res=None,
    mode=None,
    key_integrand=None,
    radius_max=None,
    is3d=None,
    # common ref
    ref_com=None,
    ref_vector_strategy=None,
    # slicing
    shape_mat=None,
    sli_mat=None,
    axis_pix=None,
    # parameters
    brightness=None,
    verb=None,
):

    # -----
    # units

    units = asunits.Unit(dvos[key_cam[0]]['sang']['units'])
    units_coefs = asunits.Unit()

    # -------------
    # mesh sampling

    wbs = coll._which_bsplines
    key_mesh = coll.dobj[wbs][key_bs]['mesh']

    # res
    lres = set([tuple(v0['res_RZ']) for v0 in dvos.values()])
    if len(lres) > 1:
        msg = "All cameras do not have the same mesh sampling resolution"
        raise Exception(msg)

    res = list(list(lres)[0])

    # mesh sampling
    dsamp = coll.get_sample_mesh(
        key=key_mesh,
        res=res,
        mode='abs',
        grid=False,
        in_mesh=True,
        # non-used
        x0=None,
        x1=None,
        Dx0=None,
        Dx1=None,
        imshow=False,
        store=False,
        kx0=None,
        kx1=None,
    )

    x0u = dsamp['x0']['data']
    x1u = dsamp['x1']['data']

    # ----------------
    # loop on cameras

    dout = {}
    doptics = coll.dobj['diagnostic'][key_diag]['doptics']
    for k0 in key_cam:

        npix = coll.dobj['camera'][k0]['dgeom']['pix_nb']
        key_mat = f'{key}_{k0}'

        # -------------
        # slicing

        is2d = coll.dobj['camera'][k0]['dgeom']['nd'] == '2d'
        if is2d:
            n0, n1 = coll.dobj['camera'][k0]['dgeom']['shape']
            sli = lambda ii: (ii // n1, ii % n1, slice(None))
        else:
            sli = lambda ii: (ii, slice(None))

        # shape, key
        sh = tuple([npix if ss is None else ss for ss in shape_mat])
        mat = np.zeros(sh, dtype=float)

        # ---------------------------------------------------
        # loop on group of pixels (to limit memory footprint)

        anyok = False
        for ii in range(npix):

            # verb
            if verb is True:
                msg = (
                    f"\t- '{key_mat}' for cam '{k0}': pixel {ii + 1} / {npix}"
                    f"\t{(mat > 0).sum()} / {mat.size}\t\t"
                )
                end = '\n' if ii == npix - 1 else '\r'
                print(msg, flush=True, end=end)

            # sample los
            indok = np.isfinite(dvos[k0]['sang']['data'][sli(ii)])
            if not np.any(indok):
                continue

            # indices + dv
            indr = dvos[k0]['indr'][sli(ii)][indok]
            indz = dvos[k0]['indz'][sli(ii)][indok]

            # -------------
            # interpolate

            # datai, units, refi = coll.interpolate(
            douti = coll.interpolate(
                keys=None,
                ref_key=key_bs,
                x0=x0u[indr],
                x1=x1u[indz],
                submesh=True,
                grid=False,
                # common ref
                ref_com=ref_com,
                ref_vector_strategy=ref_vector_strategy,
                # bsplines-specific
                # azone=None,
                indbs_tf=indbs,
                details=True,
                crop=None,
                nan0=True,
                val_out=np.nan,
                return_params=False,
                store=False,
            )[f'{key_bs}_details']

            datai, refi = douti['data'], douti['ref']
            axis = refi.index(None)
            iok = np.isfinite(datai)

            if not np.any(iok):
                continue

            datai[~iok] = 0.

            # ------------
            # integrate

            # check and update slice
            assert datai.ndim in [2, 3], datai.shape
            sli_mat[axis_pix] = ii

            # integrate
            mat[tuple(sli_mat)] = np.sum(
                datai * dvos[k0]['sang']['data'][sli(ii)][indok][:, None],
                axis=axis,
            )

            anyok = True

        # --------------
        # post-treatment

        if anyok:
            # brightness
            if brightness is True:
                ketend = doptics[k0]['etendue']
                units_coefs = coll.ddata[ketend]['units']
                etend = coll.ddata[ketend]['data']
                sh_etend = [-1 if aa == axis else 1 for aa in range(len(refi))]
                mat /= etend.reshape(sh_etend)

            # set ref
            refi = list(refi)
            refi[axis] = coll.dobj['camera'][k0]['dgeom']['ref_flat']
            refi = tuple(np.r_[refi[:axis], refi[axis], refi[axis+1:]])

        else:
            refi = None
            axis = None

        # fill dout
        dout[key_mat] = {
            'data': mat,
            'ref': refi,
            'units': units / units_coefs,
        }

    return dout, axis


# ###################
#   storing
# ###################


def _store(
    coll=None,
    key=None,
    key_bs=None,
    key_diag=None,
    key_cam=None,
    method=None,
    res=None,
    crop=None,
    dout=None,
    axis_chan=None,
    axis_bs=None,
    axis_other=None,
):

    # shapes
    shapes = [v0['data'].shape for v0 in dout.values()]
    assert all([len(ss) == len(shapes[0]) for ss in shapes[1:]])
    shapes = np.array(shapes)
    assert np.allclose(shapes[1:, :axis_chan], shapes[0:1, :axis_chan])
    assert np.allclose(shapes[1:, axis_bs:], shapes[0:1, axis_bs:])

    # add matrix obj
    dobj = {
        'geom matrix': {
            key: {
                'data': list(dout.keys()),
                'bsplines': key_bs,
                'diagnostic': key_diag,
                'camera': key_cam,
                'method': method,
                'res': res,
                'crop': crop,
                'shape': tuple(shapes[0, :]),
                'axis_chan': axis_chan,
                'axis_bs': axis_bs,
                'axis_other': axis_other,
            },
        },
    }

    coll.update(ddata=dout, dobj=dobj)


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


# ###################
#   no interaction
# ###################


def _no_interaction(
    coll=None,
    key=None,
    key_bs=None,
    key_diag=None,
    key_cam=None,
):

    # ----------
    # plot

    wm = coll._which_mesh
    wbs = coll._which_bsplines
    keym = coll.dobj[wbs][key_bs][wm]
    submesh = coll.dobj[wm][keym]['submesh']

    is2d = coll.dobj['diagnostic'][key_diag]['is2d']

    # prepare dax
    dax0 = _generic_plot.get_dax_diag(
        proj=['cross', 'hor', '3d', 'camera'],
        dmargin=None,
        fs=None,
        wintit=None,
        tit='debug',
        is2d=is2d,
        key_cam=key_cam,
    )

    # mesh
    if submesh is None:
        dax = coll.plot_mesh(
            key=keym,
            dax={'cross': dax0['cross']},
            crop=True,
        )

    else:
        dax = coll.plot_mesh(
            key=submesh,
            dax={'cross': dax0['cross']},
            crop=True,
        )

        dax = coll.plot_mesh(keym)

    # cam
    dax = coll.plot_diagnostic(
        key=key_diag,
        key_cam=key_cam,
        elements='o',
        dax=dax0,
    )

    # -----
    # msg

    msg = (
        "No interaction detected between:\n"
        f"\t- camera: '{key_cam}'\n"
        f"\t- bsplines: '{key_bs}'\n"
        f"\t- submesh: '{submesh}'\n"
    )
    warnings.warn(msg)