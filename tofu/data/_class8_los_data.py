# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 18:12:43 2020

@author: Didou09
"""


import numpy as np
import datastock as ds


from ..geom._comp_solidangles import calc_solidangle_apertures


# ##################################################################
# ##################################################################
#             solid angles from any points
# ##################################################################


def compute_solid_angles(
    coll=None,
    key=None,
    key_cam=None,
    # pts
    ptsx=None,
    ptsy=None,
    ptsz=None,
    # options
    config=None,
    visibility=None,
    # return
    return_vect=None,
    return_alpha=None,
):
    # ---------
    # check

    (
        key, key_cam, spectro,
        ptsx, ptsy, ptsz, shape0_pts,
        return_vect, return_alpha,
    ) = _compute_solid_angles_check(
        coll=coll,
        key=key,
        key_cam=key_cam,
        # pts
        ptsx=ptsx,
        ptsy=ptsy,
        ptsz=ptsz,
        # return
        return_vect=return_vect,
        return_alpha=return_alpha,
    )

    # -----------
    # prepare

    if spectro:
        raise NotImplementedError()

    else:

        dout = _compute_solid_angles_broadband(
            coll=coll,
            key=key,
            key_cam=key_cam,
            # pts
            ptsx=ptsx,
            ptsy=ptsy,
            ptsz=ptsz,
            shape0_pts=shape0_pts,
            # options
            config=config,
            visibility=visibility,
            # return
            return_vect=return_vect,
        )

    return dout


# ##################################################################
#             check inputs
# ##################################################################


def _compute_solid_angles_check(
    coll=None,
    key=None,
    key_cam=None,
    # pts
    ptsx=None,
    ptsy=None,
    ptsz=None,
    # options
    config=None,
    visibility=None,
    # return
    return_vect=None,
    return_alpha=None,
):
    # ---------
    # check

    # key_cam
    key, key_cam = coll.get_diagnostic_cam(key=key, key_cam=key_cam)
    spectro = coll.dobj['diagnostic'][key]['spectro']

    # pts
    ptsx = np.atleast_1d(ptsx)
    ptsy = np.atleast_1d(ptsy)
    ptsz = np.atleast_1d(ptsz)

    if not (ptsx.shape == ptsy.shape == ptsz.shape):
        msg = (
            "Args ptsx, ptsy, ptsz must be 3 np.ndarray of the same shape!"
        )
        raise Exception(msg)

    shape0_pts = ptsx.shape
    if ptsx.ndim > 1:
        ptsx = ptsx.ravel()
        ptsy = ptsy.ravel()
        ptsz = ptsz.ravel()

    # return_vect
    return_vect = ds._generic_check._check_var(
        return_vect, 'return_vect',
        types=bool,
        default=False,
    )

    # return_alpha
    return_alpha = ds._generic_check._check_var(
        return_alpha, 'return_alpha',
        types=bool,
        default=False,
    )

    return (
        key, key_cam, spectro,
        ptsx, ptsy, ptsz, shape0_pts,
        return_vect, return_alpha,
    )


# ##################################################################
#             regular
# ##################################################################


def _compute_solid_angles_broadband(
    coll=None,
    key=None,
    key_cam=None,
    # pts
    ptsx=None,
    ptsy=None,
    ptsz=None,
    shape0_pts=None,
    # options
    config=None,
    visibility=None,
    # return
    return_vect=None,
):

    doptics = coll.dobj['diagnostic'][key]['doptics']
    dout = {k0: {} for k0 in key_cam}

    for k0 in key_cam:

        # prepare apertures
        dap = {}
        for op, opc in zip(doptics[k0]['optics'], doptics[k0]['cls']):
            dg = coll.dobj[opc][op]['dgeom']
            if dg['type'] == '3d':
                px, py, pz = dg['poly_x'], dg['poly_y'], dg['poly_z']
            else:
                cc = dg['cent']
                out0, out1 = dg['outline']
                out0, out1 = coll.ddata[out0]['data'], coll.ddata[out1]['data']
                px = cc[0] + out0*dg['e0'][0] + out1*dg['e1'][0]
                py = cc[1] + out0*dg['e0'][1] + out1*dg['e1'][1]
                pz = cc[2] + out0*dg['e0'][2] + out1*dg['e1'][2]

            dap[op] = {
                'nin': dg['nin'],
                'poly_x': px,
                'poly_y': py,
                'poly_z': pz,
            }

        # prepare camera
        dg = coll.dobj['camera'][k0]['dgeom']
        ddet = {}

        # cents
        cx, cy, cz = coll.get_camera_cents_xyz(k0)
        sh = cx.shape
        ddet['cents_x'] = cx
        ddet['cents_y'] = cy
        ddet['cents_z'] = cz

        # vectors
        ddet.update(coll.get_camera_unit_vectors(k0))
        for k1 in ['nin', 'e0', 'e1']:
            for ii, ss in enumerate(['x', 'y', 'z']):
                kk = f'{k1}_{ss}'
                if np.isscalar(ddet[kk]):
                    ddet[kk] = np.full(sh, ddet[kk])

        out0, out1 = dg['outline']
        out0, out1 = coll.ddata[out0]['data'], coll.ddata[out1]['data']
        ddet['outline_x0'] = out0
        ddet['outline_x1'] = out1

        # compute
        out = calc_solidangle_apertures(
            # observation points
            pts_x=ptsx,
            pts_y=ptsy,
            pts_z=ptsz,
            # polygons
            apertures=dap,
            detectors=ddet,
            # possible obstacles
            config=config,
            # parameters
            summed=False,
            visibility=visibility,
            return_vector=return_vect,
            return_flat_pts=True,
            return_flat_det=None,
        )

        # store
        if return_vect is True:
            dout[k0]['solid_angle'] = out[0]
            dout[k0]['vectx'] = out[1]
            dout[k0]['vecty'] = out[2]
            dout[k0]['vectz'] = out[3]
        else:
            dout[k0]['solid_angle'] = out

        # reshape
        if shape0_pts != ptsx.shape:
            shape = tuple(np.r_[dout[k0]['solid_angle'].shape[0], shape0_pts])
            for k1, v1 in dout[k0].items():
                dout[k0][k1] = v1.reshape(shape)

    return dout