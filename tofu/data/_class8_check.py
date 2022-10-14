# -*- coding: utf-8 -*-


import warnings
import itertools as itt


import numpy as np
import matplotlib.colors as mcolors
import datastock as ds


# #############################################################################
# #############################################################################
#                           Diagnostics
# #############################################################################


def _diagnostics_check(
    coll=None,
    key=None,
    optics=None,
):

    # ----
    # key

    key = ds._generic_check._obj_key(
        d0=coll.dobj.get('diagnostic', {}), short='diag', key=key,
    )

    # ------
    # optics

    if isinstance(optics, str):
        optics = (optics,)

    lcam = list(coll.dobj.get('camera', {}).keys())
    lap = list(coll.dobj.get('aperture', {}).keys())
    lfilt = list(coll.dobj.get('filter', {}).keys())
    lcryst = list(coll.dobj.get('crystal', {}).keys())
    lgrat = list(coll.dobj.get('grating', {}).keys())
    optics = ds._generic_check._check_var_iter(
        optics, 'optics',
        types_iter=str,
        types=(tuple, list),
        allowed=lcam + lap + lfilt + lcryst + lgrat,
    )
    if isinstance(optics, list):
        optics = tuple(optics)

    # check starts with camera
    if optics[0] not in lcam:
        msg = f"Arg optics must start with a camera!\nProvided: {optics}"
        raise Exception(msg)

    if len(optics) > 1 and any([oo in lcam for oo in optics[1:]]):
        msg = f"Arg optics can only have one camera!\nProvided: {optics}"
        raise Exception(msg)

    # -----------------
    # type of camera

    is2d = coll.dobj['camera'][optics[0]]['dgeom']['type'] == '2d'

    # -------------------------------------------
    # check all optics are on good side of camera

    cam = optics[0]
    dgeom_cam = coll.dobj['camera'][cam]['dgeom']
    last_ref = cam
    last_ref_cls = 'camera'
    for oo in optics[1:]:

        if oo in lap:
            cls = 'aperture'
        elif oo in lfilt:
            cls = 'filter'
        elif oo in lcryst:
            cls = 'crystal'
        else:
            cls = 'grating'

        dgeom = coll.dobj[cls][oo]['dgeom']

        px, py, pz = coll.get_optics_poly(key=oo)

        dgeom_lastref = coll.dobj[last_ref_cls][last_ref]['dgeom']

        if (last_ref == cam and is2d) or last_ref != cam:
            cent = dgeom_lastref['cent']
            nin = dgeom_lastref['nin']

            iout = (
                (px - cent[0])*nin[0]
                + (py - cent[1])*nin[1]
                + (pz - cent[2])*nin[2]
            ) <= 0
            if np.any(iout):
                msg = (
                    f"The following points of aperture '{oo}' are on the wrong"
                    f"side of lastref '{cam}':\n"
                    f"{iout.nonzero()[0]}"
                )
                raise Exception(msg)

        else:
            assert last_ref == cam and not is2d

            cx, cy, cz = dgeom_cam['cents']
            cx = coll.ddata[cx]['data'][None, :]
            cy = coll.ddata[cy]['data'][None, :]
            cz = coll.ddata[cz]['data'][None, :]

            if dgeom_cam['parallel']:
                ninx, niny, ninz = dgeom_cam['nin']
            else:
                ninx, niny, ninz = dgeom_cam['nin']
                ninx = coll.ddata[ninx]['data'][None, :]
                niny = coll.ddata[niny]['data'][None, :]
                ninz = coll.ddata[ninz]['data'][None, :]

            iout = (
                (px[:, None] - cx)*ninx
                + (py[:, None] - cy)*niny
                + (pz[:, None] - cz)*ninz
            ) <= 0
            if np.any(iout):
                msg = (
                    f"The following points of {cls} '{oo}' are on the wrong"
                    f"side of camera '{cam}':\n"
                    f"{np.unique(iout.nonzero()[0])}"
                )
                warnings.warn(msg)

        # update last_ref ?
        if cls in ['crystal', 'grating']:
            last_ref = oo
            last_ref_cls = cls

    # -----------------
    # compute los

    compute = len(optics) > 1

    return key, optics, is2d, compute


def _diagnostics(
    coll=None,
    key=None,
    optics=None,
    **kwdargs,
):

    # ------------
    # check inputs

    key, optics, is2d, compute = _diagnostics_check(
        coll=coll,
        key=key,
        optics=optics,
    )

    # ----------
    # is spectro

    spectro = any([
        k0 in coll.dobj.get('crystal', {}).keys()
        or k0 in coll.dobj.get('grating', {}).keys()
        for k0 in optics
    ])

    # --------
    # dobj

    dobj = {
        'diagnostic': {
            key: {
                'optics': optics,
                'spectro': spectro,
                'etendue': None,
                'etend_type': None,
                'los': None,
                'vos': None,
                'amin': None,
                'amax': None,
            },
        },
    }

    # -----------
    # kwdargs

    if len(kwdargs) > 0:
        for k0, v0 in kwdargs.items():
            if not isinstance(k0, str):
                continue
            elif k0 in dobj['diagnostic'][key].keys():
                continue
            else:
                dobj['diagnostic'][key][k0] = v0

    return None, None, dobj


# ##################################################################
# ##################################################################
#                           get ref
# ##################################################################


def get_ref(coll=None, key=None):

    # ---------
    # check key

    lok = list(coll.dobj.get('diagnostic', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # -----------
    # return

    cam = coll.dobj['diagnostic'][key]['optics'][0]
    return coll.dobj['camera'][cam]['dgeom']['ref']


# ##################################################################
# ##################################################################
#                       get optics
# ##################################################################


def _get_optics(coll=None, key=None, optics=None):

    # ---------
    # check key

    lcls = ['camera', 'aperture', 'filter', 'crystal', 'grating']
    if optics is None:
        lok = list(coll.dobj.get('diagnostic', {}).keys())
        key = ds._generic_check._check_var(
            key, 'key',
            types=str,
            allowed=lok,
        )
        optics = coll.dobj['diagnostic'][key]['optics']

    else:
        if isinstance(optics, str):
            optics = [optics]
        lok = itt.chain.from_iterable([
            list(coll.dobj.get(cc, {}).keys())
            for cc in lcls
        ])
        optics = ds._generic_check._check_var_iter(
            optics, 'optics',
            types=list,
            types_iter=str,
            allowed=lok,
        )

    # -----------
    # return

    optics_cls = []
    for ii, oo in enumerate(optics):

        lc = [cc for cc in lcls if oo in coll.dobj.get(cc, {}).keys()]
        if len(lc) == 1:
            optics_cls.append(lc[0])

        else:
            msg = f"Diagnostic {key}:"
            if len(lc) == 0:
                msg = f"{msg} no matching class for optics {oo}"
            else:
                msg = f"{msg} multiple matching classes for optics {oo}: {lc}"
            raise Exception(msg)

    return optics, optics_cls


# ##################################################################
# ##################################################################
#                           set color
# ##################################################################


def _set_optics_color(
    coll=None,
    key=None,
    color=None,
):

    # ------------
    # check inputs

    # key
    lk = ['aperture', 'filter', 'crystal', 'grating', 'camera', 'diagnostic']
    dk = {
        k0: list(coll.dobj.get(k0, {}).keys())
        for k0 in lk
    }
    lok = itt.chain.from_iterable([vv for vv in dk.values()])

    if isinstance(key, str):
        key = [key]

    key = ds._generic_check._check_var_iter(
        key, 'key',
        types=(list, tuple),
        types_iter=str,
        allowed=lok,
    )

    # color
    if color is None:
        color = 'k'

    if not mcolors.is_color_like(color):
        msg = (
            f"Arg color for '{key}' must be a matplotlib color!\n"
            f"Provided: {color}\n"
        )
        raise Exception(msg)

    color = mcolors.to_rgba(color)

    # ------------
    # set color

    for k0 in key:
        cls = [k1 for k1, v1 in dk.items() if k0 in v1][0]
        if cls == 'diagnostic':
            for k2 in coll._dobj[cls][k0]['optics']:
                cls2 = [k1 for k1, v1 in dk.items() if k2 in v1][0]
                coll._dobj[cls2][k2]['dmisc']['color'] = color
        else:
            coll._dobj[cls][k0]['dmisc']['color'] = color

    return
