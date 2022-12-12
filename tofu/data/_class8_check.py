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
    doptics=None,
    stack=None,
):

    # ----
    # key

    key = ds._generic_check._obj_key(
        d0=coll.dobj.get('diagnostic', {}), short='diag', key=key,
    )

    # ------
    # doptics

    # preliminary checks
    if isinstance(doptics, str):
        doptics = {doptics: []}
    if isinstance(doptics, (list, tuple)):
        doptics = {doptics[0]: list(doptics[1:])}

    err = False
    if not isinstance(doptics, dict):
        err = True

    c0 = any([
        not isinstance(k0, str) or not isinstance(v0, (str, list))
        for k0, v0 in doptics.items()
        ])
    if c0:
        err = True

    # detailed checks
    dkout = None
    if err is False:

        for k0, v0 in doptics.items():
            if isinstance(v0, str):
                doptics[k0] = [v0]

        lcam = list(coll.dobj.get('camera', {}).keys())
        lap = list(coll.dobj.get('aperture', {}).keys())
        lfilt = list(coll.dobj.get('filter', {}).keys())
        lcryst = list(coll.dobj.get('crystal', {}).keys())
        lgrat = list(coll.dobj.get('grating', {}).keys())

        lop = lap + lfilt + lcryst + lgrat

        dkout = {
            k0: [k1 for k1 in v0 if k1 not in lop]
            for k0, v0 in doptics.items()
            if k0 not in lcam
            or any([k1 not in lop for k1 in v0])
            }
        if len(dkout) > 0:
            err = True

    if err:
        msg = (
            f"diag '{key}': arg doptics must be a dict with:\n"
            "\t- keys: key to existing camera\n"
            "\t- values: list of existing optics (aperture, filter, crystal)\n"
            )
        if dkout is not None and len(dkout) > 0:
            lstr = [f"\t- {k0}: {v0}" for k0, v0 in dkout.items()]
            msg += "Wrong key / value pairs:\n" + "\n".join(lstr)
        else:
            msg += f"\nProvided:\n{doptics}"
        raise Exception(msg)

    # -----------------
    # types of camera

    lcam = list(doptics.keys())
    types = [coll.dobj['camera'][k0]['dgeom']['type'] for k0 in lcam]

    if len(set(types)) > 1:
        msg = (
            f"diag '{key}': all cameras must be of the same type (1d or 2d)!\n"
            f"\t- cameras: {lcam}\n"
            f"\t- types: {types}"
            )
        raise Exception(msg)

    is2d = types[0] == '2d'

    # -------------------------------------------------
    # check all optics are on good side of each camera

    for cam in lcam:

        dgeom_cam = coll.dobj['camera'][cam]['dgeom']
        last_ref = cam
        last_ref_cls = 'camera'
        for oo in doptics[cam]:

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
                        f"diag '{key}':\n"
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
    # is spectro

    dspectro = {
        k0 : any([
            k1 in coll.dobj.get('crystal', {}).keys()
            or k1 in coll.dobj.get('grating', {}).keys()
            for k1 in v0
            ])
        for k0, v0 in doptics.items()
    }

    lc = [
        all([v0 for v0 in dspectro.values()]),
        all([not v0 for v0 in dspectro.values()]),
    ]
    if not any(lc):
        msg = (
            f"diag '{key}' must be either all spectro or all non-spectro!\n"
            + "\n".join([f"\t- {k0}: {v0}" for k0, v0 in dspectro.items()])
            )
        raise Exception(msg)

    spectro = lc[0] is True

    # -----------------
    # rearrange doptics

    doptics2 = {}
    for k0, v0 in doptics.items():
        doptics2[k0] = {
            'camera': k0,
            'los': None,
            'vos': None,
            'etendue': None,
            'etend_type': None,
            'amin': None,
            'amax': None,
            }

        doptics2[k0]['optics'], doptics2[k0]['cls'] = _get_optics_cls(
            coll=coll,
            optics=v0,
        )

    # -----------
    # ispectro

    if spectro:
        for k0, v0 in doptics2.items():
            doptics2[k0]['ispectro'] = [
                ii for ii, cc in enumerate(v0['cls'])
                if cc in ['grating', 'crystal']
            ]

    # -----------------
    # stack

    stack = ds._generic_check._check_var(
        stack, 'stack',
        types=str,
        default='horizontal',
        allowed=['horizontal', 'vertical'],
        )

    return key, lcam, doptics2, is2d, spectro,stack


def _diagnostics(
    coll=None,
    key=None,
    doptics=None,
    stack=None,
    **kwdargs,
):

    # ------------
    # check inputs

    key, lcam, doptics, is2d, spectro, stack = _diagnostics_check(
        coll=coll,
        key=key,
        doptics=doptics,
        stack=stack,
    )

    # --------
    # dobj

    dobj = {
        'diagnostic': {
            key: {
                'camera': lcam,
                'doptics': doptics,
                'ncam': len(doptics),
                # 'npix tot.': np.sum(),
                'is2d': is2d,
                'spectro': spectro,
                'stack': stack,
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


def _get_default_cam(coll=None, key=None, key_cam=None):

    # ----------
    # key

    lok = list(coll.dobj.get('diagnostic', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )
    is2d = coll.dobj['diagnostic'][key]['is2d']
    # spectro = coll.dobj['diagnostic'][key]['spectro']

    # -----------------------
    # key_cam (only 1 if is2d)

    lok = list(coll.dobj['diagnostic'][key]['doptics'].keys())
    if is2d:
        # 2d: can only select one camera at a time
        key_cam_def = [coll.dobj['diagnostic'][key]['camera'][0]]
    else:
        key_cam_def = None

    if isinstance(key_cam, str):
        key_cam = [key_cam]

    key_cam = ds._generic_check._check_var_iter(
        key_cam, 'key_cam',
        types=list,
        types_iter=str,
        allowed=lok,
        default=key_cam_def,
    )

    return key, key_cam


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


def _get_optics_cls(coll=None, optics=None):

    # ---------
    # check key

    lcls = ['camera', 'aperture', 'filter', 'crystal', 'grating']

    if isinstance(optics, str):
        optics = [optics]

    lok = list(itt.chain.from_iterable([
        list(coll.dobj.get(cc, {}).keys())
        for cc in lcls
    ]))
    optics = ds._generic_check._check_var_iter(
        optics, 'optics',
        types=list,
        types_iter=str,
        allowed=lok,
    )

    # -----------
    # return

    optics_cls = []
    derr = {}
    for ii, oo in enumerate(optics):
        lc = [cc for cc in lcls if oo in coll.dobj.get(cc, {}).keys()]
        if len(lc) == 1:
            optics_cls.append(lc[0])
        else:
            derr[oo] = lc

    # --------
    # error

    if len(derr) > 0:
        msg = (
            f"The following have no / several classes:\n"
            + "\n".join([f"\t- {k0}: {v0}" for k0, v0 in derr.items()])
        )
        raise Exception(msg)

    return optics, optics_cls


# def _get_diagnostic_doptics(coll=None, key=None):

#     # ---------
#     # check key

#     lok = list(coll.dobj.get('diagnostic', {}).keys())
#     key = ds._generic_check._check_var(
#         key, 'key',
#         types=str,
#         allowed=lok,
#     )

#     # ------------------
#     # optics and classes

#     doptics = {}
#     for k0, v0 in coll.dobj['diagnostic'][key]['doptics']:
#         doptics[k0]['camera'] = k0
#         doptics[k0]['optics'], doptics[k0]['cls'] = _get_optics_cls(
#             coll=coll,
#             optics=v0,
#         )

#     # -----------
#     # ispectro

#     if coll.dobj['diagnostic'][key]['spectro']:
#         for k0, v0 in doptics.items():
#             doptics[k0]['ispectro'] = [
#                 ii for ii, cc in enumerate(v0['cls'])
#                 if cc in ['grating', 'crystal']
#             ]

#     return doptics


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


# ##################################################################
# ##################################################################
#                           remove
# ##################################################################

def _remove(
    coll=None,
    key=None,
    key_cam=None,
):

    # ------------
    # check inputs

    lok = list(coll.dobj.get('diagnostic', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )
    doptics = coll.dobj['diagnostic'][key]['doptics']

    if key_cam is None:
        key_cam = coll.dobj['diagnostic'][key]['camera']

    else:
        if isinstance(key_cam, str):
            key_cam = [key_cam]

        key_cam = ds._generic_check._check_var_iter(
            key_cam, 'key_cam',
            types=list,
            types_iter=str,
            allowed=coll.dobj['diagnostic'][key]['camera'],
        )

    # ------------
    # list data

    lkd = ['etendue']
    ld = []
    for k0 in lkd:
        for k1 in key_cam:
            if doptics[k1][k0] is not None:
                ld.append(doptics[k1][k0])

    # -----------
    # remove data

    if len(ld) > 0:
        coll.remove_data(ld, propagate=True)

    # ----------
    # remove los

    for k1 in key_cam:
        if doptics[k1]['los'] is not None:
            coll.remove_rays(key=doptics[k1]['los'])

    # -----------
    # remove diag

    if len(key_cam) == len(doptics):
        del coll._dobj['diagnostic'][key]
