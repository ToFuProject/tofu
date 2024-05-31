# -*- coding: utf-8 -*-


# Built-in
import warnings


# Common
import numpy as np
import datastock as ds


from . import _generic_plot
from . import _class09_compute_broadband as _compute_broadband


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
        spectro,
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
        msg = f"\nGeom matrix for diag '{key_diag}' and bs '{key_bs}':"
        print(msg)

    # -----------
    # compute
    # -----------

    if spectro is True:
        raise NotImplementedError("Spectro geom matrix not implemented!")

    else:

        if method == 'los':
            dout, axis = _compute_broadband._compute_los(
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
            dout, axis = _compute_broadband._compute_vos(
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
        raise NotImplementedError(msg)

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
        key_diag, dvos, isstore = coll.check_diagnostic_dvos(
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
        spectro,
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
    key_cam=None,
):

    # ------------
    # check inputs

    lok = list(coll.dobj.get('geom matrix', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    key_cam0 = coll.dobj['geom matrix'][key]['camera']
    key_cam = ds._generic_check._check_var_iter(
        key_cam, 'key_cam',
        default=key_cam0,
        types=(list, tuple),
        types_iter=str,
        allowed=key_cam0,
    )

    # -----------
    # concatenate

    key_data = coll.dobj['geom matrix'][key]['data']
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
        _ = coll.plot_mesh(
            key=keym,
            dax={'cross': dax0['cross']},
            crop=True,
        )

    else:
        _ = coll.plot_mesh(
            key=submesh,
            dax={'cross': dax0['cross']},
            crop=True,
        )

        _ = coll.plot_mesh(keym)

    # cam
    _ = coll.plot_diagnostic(
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