# -*- coding: utf-8 -*-


import numpy as np
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
    lcryst = list(coll.dobj.get('crystal', {}).keys())
    lgrat = list(coll.dobj.get('grating', {}).keys())
    optics = ds._generic_check._check_var_iter(
        optics, 'optics',
        types_iter=str,
        types=(tuple, list),
        allowed=lcam + lap + lcryst + lgrat,
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

    is2d = coll.dobj['camera'][optics[0]]['type'] == '2d'

    # -------------------------------------------
    # check all optics are on good side of camera

    cam = optics[0]
    last_ref = cam
    for oo in optics[1:]:

        if oo in lap:
            cls = 'aperture'
        elif oo in lcryst:
            cls = 'crystal'
        else:
            cls = 'grating'

        px, py, pz = coll.dobj[cls][oo]['dgeom']['poly']
        px = coll.ddata[px]['data']
        py = coll.ddata[py]['data']
        pz = coll.ddata[pz]['data']

        if last_ref == cam and is2d:
            cent = coll.dobj['camera'][cam]['cent']
            nin = coll.dobj['camera'][cam]['nin']

            iout = (
                (px - cent[0])*nin[0]
                + (py - cent[1])*nin[1]
                + (pz - cent[2])*nin[2]
            ) <= 0
            if np.any(iout):
                msg = (
                    f"The following points of aperture '{oo}' are on the wrong"
                    f"side of camera '{cam}':\n"
                    f"{iout.nonzero()[0]}"
                )
                raise Exception(msg)

        elif last_ref == cam:
            cx, cy, cz = coll.dobj['camera'][cam]['cents']
            cx = coll.ddata[cx]['data'][None, :]
            cy = coll.ddata[cy]['data'][None, :]
            cz = coll.ddata[cz]['data'][None, :]

            if coll.dobj['camera'][cam]['parallel']:
                ninx, niny, ninz = coll.dobj['camera'][cam]['nin']
            else:
                ninx, niny, ninz = coll.dobj['camera'][cam]['nin']
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
                raise Exception(msg)

        else:
            cent = coll.dobj[last_ref_cls][last_ref]['dgeom']['cent']
            nin = coll.dobj[last_ref_cls][last_ref]['dgeom']['nin']

            iout = (
                (px - cent[0])*nin[0]
                + (py - cent[1])*nin[1]
                + (pz - cent[2])*nin[2]
            ) <= 0
            if np.any(iout):
                msg = (
                    f"The following points of optics '{oo}' are on the wrong"
                    f"side of {last_ref_cls} '{last_ref}':\n"
                    f"{iout.nonzero()[0]}"
                )
                raise Exception(msg)

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
