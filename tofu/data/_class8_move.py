# -*- coding: utf-8 -*-


import copy
import itertools as itt

import numpy as np
import scipy.interpolate as scpinterp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


import datastock as ds


from . import _class7_compute


# ##################################################################
# ##################################################################
#             MOVE
# ##################################################################


def move_to(
    coll=None,
    key=None,
    key_cam=None,
    optics=None,
    # location
    x=None,
    y=None,
    R=None,
    z=None,
    phi=None,
    theta=None,
    dphi=None,
    tilt=None,
):

    # ------------
    # check inputs

    # trivial case
    nochange = all([ss is None for ss in [x, y, z, R, phi, theta, dphi, tilt]])

    # key, key_cam
    key, key_cam = coll.get_diagnostic_cam(key=key, key_cam=key_cam)
    is2d = coll.dobj['diagnostic'][key]['is2d']

    if len(key_cam) != 1:
        msg = "move_diagnostic_to() can only be used on one camera"
        raise Exception(msg)
    key_cam = key_cam[0]
    doptics = coll.dobj['diagnostic'][key]['doptics'][key_cam]

    # optics
    lok = [key_cam] + doptics['optics']
    op_ref = ds._generic_check._check_var(
        optics, 'optics',
        types=str,
        default=doptics['optics'][0],
        allowed=lok,
    )

    # get optics, op_cls
    optics = doptics['optics']
    op_cls = doptics['cls']

    # ----------------------------------
    # use the chosen optics center as reference

    if op_ref == key_cam:
        if is2d:
            cls_ref = 'camera'
            cc = coll.dobj[cls_ref][op_ref]['dgeom']['cent']
            nin = coll.dobj[cls_ref][op_ref]['dgeom']['nin']
            e0 = coll.dobj[cls_ref][op_ref]['dgeom']['e0']
            e1 = coll.dobj[cls_ref][op_ref]['dgeom']['e1']

        else:
            raise NotImplementedError()

    else:
        cls_ref = op_cls[optics.index(op_ref)]
        cc = coll.dobj[cls_ref][op_ref]['dgeom']['cent']
        nin = coll.dobj[cls_ref][op_ref]['dgeom']['nin']
        e0 = coll.dobj[cls_ref][op_ref]['dgeom']['e0']
        e1 = coll.dobj[cls_ref][op_ref]['dgeom']['e1']

    # ----------------------------------
    # get initial local coordinates in this frame

    dinit = _get_initial_parameters(
        cc=cc,
        nin=nin,
        e0=e0,
        e1=e1,
    )

    # ----------------------------------
    # get all local coordinates in this frame

    dcoords = {}

    # camera
    if is2d:
        dcoords[key_cam] = _extract_coords(
            dg=coll.dobj['camera'][key_cam]['dgeom'],
            cc=cc,
            nin=nin,
            e0=e0,
            e1=e1,
        )
    else:
        dcoords[key_cam] = _extract_coords_cam1d(
            coll=coll,
            key_cam=key_cam,
            cc=cc,
            nin=nin,
            e0=e0,
            e1=e1,
        )

    # optics
    for op, opc in zip(optics, op_cls):
        dcoords[op] = _extract_coords(
            dg=coll.dobj[opc][op]['dgeom'],
            cc=cc,
            nin=nin,
            e0=e0,
            e1=e1,
        )

    # ----------------------------------
    # get new default values

    cc_new, nin_new, e0_new, e1_new = _class7_compute._pinhole_position(
        # center
        x=x,
        y=y,
        R=R,
        z=z,
        phi=phi,
        # angles
        theta=theta,
        dphi=dphi,
        tilt=tilt,
        # default
        ddef=dinit,
    )

    # ----------------------------------
    # Update all coordinates

    # camera
    if is2d:
        reset_coords(
            coll=coll,
            op=key_cam,
            opc='camera',
            dcoords=dcoords,
            cc_new=cc_new,
            nin_new=nin_new,
            e0_new=e0_new,
            e1_new=e1_new,
        )
    else:
        reset_coords_cam1d(
            coll=coll,
            op=key_cam,
            opc='camera',
            dcoords=dcoords,
            cc_new=cc_new,
            nin_new=nin_new,
            e0_new=e0_new,
            e1_new=e1_new,
        )

    # optics
    for op, opc in zip(optics, op_cls):
        reset_coords(
            coll=coll,
            op=op,
            opc=opc,
            dcoords=dcoords,
            cc_new=cc_new,
            nin_new=nin_new,
            e0_new=e0_new,
            e1_new=e1_new,
        )

    return



def _get_initial_parameters(
    cc=None,
    nin=None,
    e0=None,
    e1=None,
):

    # cordinates
    x, y, z = cc
    R = np.hypot(x, y)

    # angles
    phi = np.arctan2(y, x)

    # unit vectors
    eR = np.r_[np.cos(phi), np.sin(phi), 0.]
    ephi = np.r_[-np.sin(phi), np.cos(phi), 0.]

    # orientation angles: dphi
    dphi = np.pi/2. - np.arccos(np.sum(nin * ephi))

    # orientation angles: theta
    ni = nin - np.sum(nin*ephi)*ephi
    ni = ni / np.linalg.norm(ni)
    theta = np.arctan2(ni[2], np.sum(ni * eR))

    # orientation: tilt
    er = np.cos(theta) * eR + np.sin(theta) * np.r_[0, 0, 1]
    etheta = -np.sin(theta) * eR + np.cos(theta) * np.r_[0, 0, 1]
    e0bis = -np.cos(dphi) * ephi + np.sin(dphi) * er

    tilt = np.arctan2(np.sum(e0*etheta), np.sum(e0*e0bis))

    return {
        'x': x,
        'y': y,
        'z': z,
        'R': R,
        'phi': phi,
        'dphi': dphi,
        'theta': theta,
        'tilt': tilt,
    }


def get_new_frame(
    key_cam=None,
    dinit=None,
    x=None,
    y=None,
    z=None,
    phi=None,
    dphi=None,
    theta=None,
    tilt=None,
    # safety check
    nochange=None,
    cc=None,
    nin=None,
    e0=None,
    e1=None,
):

    # orientation
    eR = np.r_[np.cos(phi), np.sin(phi), 0.]
    ephi = np.r_[-np.sin(phi), np.cos(phi), 0.]
    er = np.cos(theta) * eR + np.sin(theta) * np.r_[0, 0, 1]
    etheta = -np.sin(theta) * eR + np.cos(theta) * np.r_[0, 0, 1]
    e0bis = -np.cos(dphi) * ephi + np.sin(dphi) * er

    # translation
    cc_new = np.r_[x, y, z]

    # new unit vectors
    nin_new = np.cos(dphi) * er + np.sin(dphi) * ephi
    e0_new = np.cos(tilt) * e0bis + np.sin(tilt) * etheta
    e1_new = np.cross(nin_new, e0_new)

    # safety check
    nin_new, e0_new, e1_new = ds._generic_check._check_vectbasis(
        e0=nin_new,
        e1=e0_new,
        e2=e1_new,
        dim=3,
        tol=1e-12,
    )

    # safety check
    if nochange:
        dout = {}
        for ss in ['cc', 'nin', 'e0', 'e1']:
            if not np.allclose(eval(ss), eval(f'{ss}_new')):
                dout[ss] = (eval(ss), eval(f'{ss}_new'))

        if len(dout) > 0:
            lstr = [f"\t- '{k0}': {v0[0]} vs {v0[1]}" for k0, v0 in dout.items()]
            msg = (
                f"Immobile diagnostic camera '{key_cam}' has moved:\n"
                + "\n".join(lstr)
                + f"\n\ndinit = {dinit}"
            )
            raise Exception(msg)

    return cc_new, nin_new, e0_new, e1_new


def _extract_coords(
    dg=None,
    cc=None,
    nin=None,
    e0=None,
    e1=None,
    ):

    return {
        'c_n01': np.r_[
            np.sum((dg['cent'] - cc) * nin),
            np.sum((dg['cent'] - cc) * e0),
            np.sum((dg['cent'] - cc) * e1),
        ],
        'n_n01': np.r_[
            np.sum(dg['nin'] * nin),
            np.sum(dg['nin'] * e0),
            np.sum(dg['nin'] * e1),
        ],
        'e0_n01': np.r_[
            np.sum(dg['e0'] * nin),
            np.sum(dg['e0'] * e0),
            np.sum(dg['e0'] * e1),
        ],
        'e1_n01': np.r_[
            np.sum(dg['e1'] * nin),
            np.sum(dg['e1'] * e0),
            np.sum(dg['e1'] * e1),
        ],
    }


def _extract_coords_cam1d(
    coll=None,
    key_cam=None,
    cc=None,
    nin=None,
    e0=None,
    e1=None,
    ):

    dout = {}
    kc = coll.dobj['camera'][key_cam]['dgeom']['cents']
    parallel = coll.dobj['camera'][key_cam]['dgeom']['parallel']

    # cents 
    shape = tuple(np.r_[3, coll.ddata[kc[0]]['data'].shape])
    dout['cents'] = np.zeros(shape)
    for ss, ii in [('x', 0), ('y', 1), ('z', 2)]:
        dout['cents'] += np.array([
            (coll.ddata[kc[ii]]['data'] - cc[ii]) * nin[ii],
            (coll.ddata[kc[ii]]['data'] - cc[ii]) * e0[ii],
            (coll.ddata[kc[ii]]['data'] - cc[ii]) * e1[ii],
        ])

    # unit vectors
    if parallel:
        for kk in ['nin', 'e0', 'e1']:
            dout[f'{kk}_n01'] = np.array([
                np.sum(coll.dobj['camera'][key_cam]['dgeom'][kk] * nin),
                np.sum(coll.dobj['camera'][key_cam]['dgeom'][kk] * e0),
                np.sum(coll.dobj['camera'][key_cam]['dgeom'][kk] * e1),
            ])
    else:
        for kk in ['nin', 'e0', 'e1']:
            dout[kk] = np.zeros(shape)
            kv = coll.dobj['camera'][key_cam]['dgeom'][kk]
            for ss, ii in [('x', 0), ('y', 1), ('z', 2)]:
                dout[kk] += np.array([
                    coll.ddata[kv[ii]]['data'] * nin[ii],
                    coll.ddata[kv[ii]]['data'] * e0[ii],
                    coll.ddata[kv[ii]]['data'] * e1[ii],
                ])
    return dout


def reset_coords(
    coll=None,
    op=None,
    opc=None,
    dcoords=None,
    cc_new=None,
    nin_new=None,
    e0_new=None,
    e1_new=None,
    ):

    if coll._dobj[opc][op]['dgeom']['type'] == '3d':
        raise NotImplementedError()

    # translate
    coll._dobj[opc][op]['dgeom']['cent'] = (
        cc_new
        + dcoords[op]['c_n01'][0] * nin_new
        + dcoords[op]['c_n01'][1] * e0_new
        + dcoords[op]['c_n01'][2] * e1_new
    )

    # rotate
    nin = (
        dcoords[op]['n_n01'][0] * nin_new
        + dcoords[op]['n_n01'][1] * e0_new
        + dcoords[op]['n_n01'][2] * e1_new
    )
    e0 = (
        dcoords[op]['e0_n01'][0] * nin_new
        + dcoords[op]['e0_n01'][1] * e0_new
        + dcoords[op]['e0_n01'][2] * e1_new
    )
    e1 = (
        dcoords[op]['e1_n01'][0] * nin_new
        + dcoords[op]['e1_n01'][1] * e0_new
        + dcoords[op]['e1_n01'][2] * e1_new
    )

    # --------------
    # safety check

    nin, e0, e1 = ds._generic_check._check_vectbasis(
        e0=nin,
        e1=e0,
        e2=e1,
        dim=3,
        tol=1e-12,
    )

    # store
    coll._dobj[opc][op]['dgeom']['nin'] = nin
    coll._dobj[opc][op]['dgeom']['e0'] = e0
    coll._dobj[opc][op]['dgeom']['e1'] = e1


def reset_coords_cam1d(
    coll=None,
    op=None,
    opc=None,
    dcoords=None,
    cc_new=None,
    nin_new=None,
    e0_new=None,
    e1_new=None,
):

    kc = coll.dobj[opc][op]['dgeom']['cents']
    parallel = coll.dobj[opc][op]['dgeom']['parallel']

    # cents 
    for ss, ii in [('x', 0), ('y', 1), ('z', 2)]:
        coll._ddata[kc[ii]]['data'] = (
            cc_new[ii]
            + dcoords[op]['cents'][0] * nin_new[ii]
            + dcoords[op]['cents'][1] * e0_new[ii]
            + dcoords[op]['cents'][2] * e1_new[ii]
        )

    # rotate
    if parallel:
        nin = (
            dcoords[op]['nin_n01'][0] * nin_new
            + dcoords[op]['nin_n01'][1] * e0_new
            + dcoords[op]['nin_n01'][2] * e1_new
        )
        e0 = (
            dcoords[op]['e0_n01'][0] * nin_new
            + dcoords[op]['e0_n01'][1] * e0_new
            + dcoords[op]['e0_n01'][2] * e1_new
        )
        e1 = (
            dcoords[op]['e1_n01'][0] * nin_new
            + dcoords[op]['e1_n01'][1] * e0_new
            + dcoords[op]['e1_n01'][2] * e1_new
        )

        # safety check
        nin, e0, e1 = ds._generic_check._check_vectbasis(
            e0=nin,
            e1=e0,
            e2=e1,
            dim=3,
            tol=1e-12,
        )

        coll._dobj[opc][op]['dgeom']['nin'] = nin
        coll._dobj[opc][op]['dgeom']['e0'] = e0
        coll._dobj[opc][op]['dgeom']['e1'] = e1

    else:
        for kk in ['nin', 'e0', 'e1']:
            kv = coll.dobj[opc][op]['dgeom'][kk]
            for ss, ii in [('x', 0), ('y', 1), ('z', 2)]:
                coll.ddata[kv[ii]]['data'] = (
                    dcoords[op][kk][0] * nin_new[ii]
                    + dcoords[op][kk][1] * e0_new[ii]
                    + dcoords[op][kk][2] * e1_new[ii]
                )
