# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:39:45 2024

@author: dvezinet
"""


import itertools as itt


import numpy as np
import datastock as ds


# ##################################################################
# ##################################################################
#                   main
# ##################################################################


def main(
    coll=None,
    key=None,
    key_data=None,
    key_cam=None,
    flat=None,
):

    # ------------
    # check inputs

    key, key_data, key_cam, key_cam_equi, is2d, stack, ref, flat = _check(
        coll=coll,
        key=key,
        key_data=key_data,
        key_cam=key_cam,
        flat=flat,
    )

    # ------------
    # prepare

    if is2d:
        ax0 = ref.index(None)
        ax1 = len(ref) - 1 - ref[::-1].index(None)
        if flat:
            ldata = []
            for k0 in key_data:
                sh = list(coll.ddata[k0]['data'].shape)
                size = sh[ax0] * sh[ax1]
                sh = tuple(np.r_[sh[:ax0], size, sh[ax1+1:]].astype(int))
                ldata.append(coll.ddata[k0]['data'].reshape(sh))
            axis = ax0
        else:
            axis = ax0 if stack == 'horizontal' else ax1
            ldata = [coll.ddata[k0]['data'] for k0 in key_data]
            lsize = [dd.shape[axis] for dd in ldata]
            if len(set(lsize)) != 1:
                msg = (
                    "Data for diag '{key}' cannot be stacked {stack}:\n"
                    f"\t- shapes: {[dd.shape for dd in ldata]}\n"
                    f"\t- axis: {axis}"
                )
                raise Exception(msg)

    else:
        ldata = [coll.ddata[k0]['data'] for k0 in key_data]
        axis = ref.index(None)

    # ------------
    # concatenate

    data = np.concatenate(tuple(ldata), axis=axis)
    units = coll.ddata[key_data[0]]['units']

    # dind
    i0 = 0
    dind = {}
    for ii, k0 in enumerate(key_data):
        npix = ldata[ii].shape[axis]
        ind = i0 + np.arange(0, npix)
        dind[k0] = ind
        i0 += npix

    return {
        'data': data,
        'keys': key_data,
        'keys_cam': key_cam,
        'keys_cam_equi': key_cam_equi,
        'units': units,
        'ref': ref,
        'axis': axis,
        'flat': flat,
        'dind': dind,
    }


# ##################################################################
# ##################################################################
#                   check
# ##################################################################


def _check(
    coll=None,
    key=None,
    key_data=None,
    key_cam=None,
    flat=None,
):

    # ---------------
    # key and key_cam
    # ---------------

    key, key_cam = coll.get_diagnostic_cam(key=key, key_cam=key_cam)
    # spectro = coll.dobj['diagnostic'][key]['spectro']
    is2d = coll.dobj['diagnostic'][key]['is2d']
    stack = coll.dobj['diagnostic'][key]['stack']

    # ---------------------------
    # get list of equivalent camera series
    # ---------------------------

    wdiag = 'diagnostic'
    wcam = 'camera'
    ldiag_equi = [
        kdiag
        for kdiag, vdiag in coll.dobj.get(wdiag, {}).items()
        if kdiag != key
        and len(vdiag['camera']) == len(coll.dobj[wdiag][key]['camera'])
        and all([
            coll.dobj[wcam][vdiag['camera'][ii]]['dgeom']['shape']
            == coll.dobj[wcam][kcam]['dgeom']['shape']
            for ii, kcam in enumerate(coll.dobj[wdiag][key]['camera'])
        ])
    ]

    # ---------------------------
    # get list of equivalent camera series
    # ---------------------------

    dcam_equi = {
        kdiag: [
            coll.dobj[wdiag][kdiag]['camera'][ii]
            for ii, kcam in enumerate(coll.dobj[wdiag][key]['camera'])
            if kcam in key_cam
        ]
        for kdiag in ldiag_equi
    }

    # ------------
    # key_data
    # ------------

    # ----------
    # key_data

    if isinstance(key_data, str):

        if key_data in coll.ddata.keys():
            key_data = [key_data]

        else:
            lok = coll.dobj['diagnostic'][key]['signal']
            lok_equi = list(itt.chain.from_iterable([
                coll.dobj['diagnostic'][kdiag]['signal']
                for kdiag in ldiag_equi
            ]))

            if lok is None:
                lok = []
            if lok_equi is None:
                lok_equi = []

            key_data = ds._generic_check._check_var(
                key_data, 'key_data',
                types=str,
                allowed=lok + lok_equi,
            )

            key_data = coll.dobj['synth sig'][key_data]['data']

    # ------------------------
    # basic consistency check
    # ------------------------

    c0 = (
        isinstance(key_data, list)
        and all([
            isinstance(kk, str) and kk in coll.ddata.keys()
            for kk in key_data
        ])
    )
    if not c0:
        msg = (
            "Arg key_data must be a list of valid data keys!\n"
            f"Provided: {key_data}"
        )
        raise Exception(msg)

    # -----------------------
    # check cameras validity
    # -----------------------

    lcam = [coll.ddata[k0]['camera'] for k0 in key_data]

    lc = (
        [all([kcam in key_cam for kcam in lcam])]
        + [
            all([kcam in dcam_equi[kdiag] for kcam in lcam])
            for kdiag in ldiag_equi
        ]
    )

    if not any(lc):
        lstr = (
            [f"\t- {key_cam}"]
            + [f"\t- {dcam_equi[kdiag]}" for kdiag in ldiag_equi]
        )
        lstr2 = [f"\t- '{k0}': '{lcam[ii]}'" for ii, k0 in enumerate(key_data)]
        msg = (
            "All data provided must be associated to one "
            "of the following cameras series:\n"
            + "\n".join(lstr)
            + "But the provided data has:\n"
            + "\n".join(lstr2)
        )
        raise Exception(msg)

    # ---------------------
    # check cameras unicity
    # ---------------------

    # check unicity of cameras
    if len(set(lcam)) > len(lcam):
        msg = (
            f"Non-unique camera references for diag '{key}'\n:"
            f"\t- key_data = {key_data}\n"
            f"\t- lcam     = {lcam}\n"
        )
        raise Exception(msg)

    # ---------------------
    # is_equi
    # ---------------------

    if lc[0]:

        key_cam = [k0 for k0 in key_cam if k0 in lcam]
        key_cam_equi = None
        key_cam_ref = [coll.dobj[wcam][kk]['dgeom']['ref'] for kk in key_cam]

    else:
        kdiag = ldiag_equi[lc.index(True) - 1]
        key_cam_equi = [k0 for k0 in dcam_equi[kdiag] if k0 in lcam]
        key_cam = [
            k0 for ii, k0 in enumerate(key_cam)
            if dcam_equi[kdiag][ii] in lcam
        ]

        key_cam_ref = [coll.dobj[wcam][kk]['dgeom']['ref'] for kk in key_cam_equi]

    # ------------
    # re-order data
    # -------------

    if key_cam_equi is None:
        key_data = [key_data[lcam.index(k0)] for k0 in key_cam]
    else:
        key_data = [key_data[lcam.index(k0)] for k0 in key_cam_equi]

    # check uniformity of ref dimensions
    lref = [coll.ddata[k0]['ref'] for k0 in key_data]
    if any([len(ref) != len(lref[0]) for ref in lref[1:]]):
        msg = (
            f"Non uniform refs length for data in diag '{key}':\n"
            f"\t- key_data = {key_data}\n"
            f"\t- lref = {lref}\n"
        )
        raise Exception(msg)

    # -------------
    # ref and axis
    # -------------

    laxcam = [
        [ref.index(rr) for rr in key_cam_ref[ii]]
        for ii, ref in enumerate(lref)
    ]

    # check all 2d or all 1d (future: flatten if needed)
    if any([len(ax) != len(laxcam[0]) for ax in laxcam[1:]]):
        msg = (
            f"Non-uniform camera concatenation axis for diag '{key}':\n"
            f"\t- key_data: {key_data}\n"
            f"\t- laxcam:   {laxcam}"
        )
        raise Exception(msg)

    # concatenation axis uniformity
    laxcam = np.array(laxcam)
    if not np.allclose(laxcam, laxcam[0:1, :]):
        msg = (
            f"Non-uniform axis for concatenation for diag '{key}':\n"
            f"\t- laxcam: {laxcam}"
        )
        raise Exception(msg)
    axcam = laxcam[0]

    # consistency vs is2d
    if len(axcam) != 1 + is2d:
        msg = (
            f"ref not consistent with is2d for diag '{key}':\n"
            f"\t- is2d: {is2d}\n"
            f"\t- axcam: {axcam}\n"
        )
        raise Exception(msg)

    # get unique ref and set concatenation axis to None
    ref = list(lref[0])
    for ii in axcam:
        ref[ii] = None

    # --------
    # flat
    # --------

    flat = ds._generic_check._check_var(
        flat, 'flat',
        types=bool,
        default=is2d,
    )

    return key, key_data, key_cam, key_cam_equi, is2d, stack, ref, flat