# -*- coding: utf-8 -*-


import numpy as np
import scipy.constants as scpct
import datastock as ds


from . import _utils_surface3d
from ..geom._comp_solidangles import _check_polygon_2d, _check_polygon_3d


# #############################################################################
# #############################################################################
#                       Generic for 3d surfaces
# #############################################################################


def _add_surface3d(
    coll=None,
    key=None,
    which=None,
    which_short=None,
    # 2d outline
    outline_x0=None,
    outline_x1=None,
    cent=None,
    # 3d outline
    poly_x=None,
    poly_y=None,
    poly_z=None,
    # normal vector
    nin=None,
    e0=None,
    e1=None,
    # extenthalf
    extenthalf=None,
    # curvature
    curve_r=None,
    curve_npts=None,
):

    # ------------
    # check inputs

    # key
    key = ds._generic_check._obj_key(
        d0=coll.dobj.get(which, {}), short=which_short, key=key,
    )

    # geometry
    (
        cent,
        outline_x0, outline_x1,
        poly_x, poly_y, poly_z,
        nin, e0, e1,
        extenthalf, area, curve_r, gtype,
    ) = _utils_surface3d._surface3d(
        key=key,
        # 2d outline
        outline_x0=outline_x0,
        outline_x1=outline_x1,
        cent=cent,
        # 3d outline
        poly_x=poly_x,
        poly_y=poly_y,
        poly_z=poly_z,
        # normal vector at cent
        nin=nin,
        e0=e0,
        e1=e1,
        # extenthalf
        extenthalf=extenthalf,
        # curvature
        curve_r=curve_r,
        curve_npts=curve_npts,
    )

    # ----------
    # create dict

    # keys
    knpts = f'{key}-npts'
    kpx = f'{key}-x'
    kpy = f'{key}-y'
    kpz = f'{key}-z'
    if gtype == 'planar':
        kp0 = f'{key}-x0'
        kp1 = f'{key}-x1'
        outline = (kp0, kp1)
    else:
        outline = None

    # refs
    npts = poly_x.size

    dref = {
        knpts: {'size': npts},
    }

    # data
    ddata = {
        kpx: {
            'data': poly_x,
            'ref': knpts,
            'dim': 'distance',
            'name': 'x',
            'quant': 'x',
            'units': 'm',
        },
        kpy: {
            'data': poly_y,
            'ref': knpts,
            'dim': 'distance',
            'name': 'y',
            'quant': 'y',
            'units': 'm',
        },
        kpz: {
            'data': poly_z,
            'ref': knpts,
            'dim': 'distance',
            'name': 'z',
            'quant': 'z',
            'units': 'm',
        },
    }
    if gtype == 'planar':
        ddata.update({
            kp0: {
                'data': outline_x0,
                'ref': knpts,
                'dim': 'distance',
                'name': 'x0',
                'quant': 'x0',
                'units': 'm',
            },
            kp1: {
                'data': outline_x1,
                'ref': knpts,
                'dim': 'distance',
                'name': 'x1',
                'quant': 'x1',
                'units': 'm',
            },
        })

    # dobj
    dobj = {
        which: {
            key: {
                'dgeom': {
                    'type': gtype,
                    'curve_r': curve_r,
                    'outline': outline,
                    'extenthalf': extenthalf,
                    'poly': (kpx, kpy, kpz),
                    'area': area,
                    'cent': cent,
                    'nin': nin,
                    'e0': e0,
                    'e1': e1,
                },
            },
        },
    }

    return dref, ddata, dobj


# #############################################################################
# #############################################################################
#                       Utilities
# #############################################################################


def _return_as_dict(
    coll=None,
    which=None,
    key=None,
):
    """ Return camera or apertres as dict (input for low-level routines) """

    if which == 'camera':

        dout = coll.get_camera_unit_vectors(key=key)
        cam = coll.dobj['camera'][key]
        cx, cy, cz = coll.get_camera_cents_xyz(key=key)

        dout.update({
            'outline_x0': coll.ddata[cam['outline'][0]]['data'],
            'outline_x1': coll.ddata[cam['outline'][1]]['data'],
            'cents_x': cx,
            'cents_y': cy,
            'cents_z': cz,
            'pix area': cam['pix area'],
            'parallel': cam['parallel'],
        })

    elif which == 'aperture':

        if isinstance(key, str):
            key = [key]
        key = ds._generic_check._check_var_iter(
            key, 'key',
            types=(list, tuple),
            types_iter=str,
            allowed=list(coll.dobj.get('aperture', {}).keys()),
        )

        dout = {}
        for k0 in key:
            ap = coll.dobj['aperture'][k0]['dgeom']
            dout[k0] = {
                'cent': ap['cent'],
                'poly_x': coll.ddata[ap['poly'][0]]['data'],
                'poly_y': coll.ddata[ap['poly'][1]]['data'],
                'poly_z': coll.ddata[ap['poly'][2]]['data'],
                'nin': ap['nin'],
                'e0': ap.get('e0'),
                'e1': ap.get('e1'),
            }

    else:
        msg = f"Un-handled category '{which}'"
        raise Exception(msg)

    return dout
