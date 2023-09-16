# -*- coding: utf-8 -*-


import numpy as np
import scipy.constants as scpct
import matplotlib.colors as mcolors
import datastock as ds


from . import _utils_surface3d
from ..geom._comp_solidangles import _check_polygon_2d, _check_polygon_3d


# ################################################################
# ################################################################
#                       Generic for 3d surfaces
# ################################################################


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
    make_planar=None,
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
        make_planar=make_planar,
    )

    # ----------
    # create dict

    # keys
    knpts = f'{key}-npts'
    if gtype != '3d':
        kp0 = f'{key}-x0'
        kp1 = f'{key}-x1'
        outline = (kp0, kp1)
        poly = None
        npts = outline_x0.size
    else:
        kpx = f'{key}-x'
        kpy = f'{key}-y'
        kpz = f'{key}-z'
        poly = (kpx, kpy, kpz)
        outline = None
        npts = poly_x.size

    # refs
    dref = {
        knpts: {'size': npts},
    }

    # data
    if gtype != '3d':
        ddata = {
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
        }
    else:
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

    # dobj
    dobj = {
        which: {
            key: {
                'dgeom': {
                    'type': gtype,
                    'curve_r': curve_r,
                    'outline': outline,
                    'extenthalf': extenthalf,
                    'poly': poly,
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
#                       Generic dmisc for 3d surfaces
# #############################################################################


def _dmisc(key=None, color=None):

    # --------
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

    return {'color': color}


# #############################################################################
# #############################################################################
#                       Utilities
# #############################################################################


def _return_as_dict(
    coll=None,
    key=None,
):
    """ Return as dict the following objects

    - camera
    - aperture
    - filter (treated as aperture)

    useful input for low-level routines

    """

    if isinstance(key, str):
        key = [key]

    lcam = list(coll.dobj.get('camera', {}).keys())
    lap = list(coll.dobj.get('aperture', {}).keys())
    lfilt = list(coll.dobj.get('filter', {}).keys())
    key = ds._generic_check._check_var_iter(
        key, 'key',
        types=(list, tuple),
        types_iter=str,
        allowed=lcam + lap + lfilt,
    )

    dout = {}
    for k0 in key:

        if k0 in lcam:
            cls = 'camera'
        elif k0 in lap:
            cls = 'aperture'
        elif k0 in lfilt:
            cls = 'filter'

        # compute
        if cls == 'camera':

            dout[k0] = coll.get_camera_unit_vectors(key=k0)
            dgeom = coll.dobj['camera'][k0]['dgeom']
            cx, cy, cz = coll.get_camera_cents_xyz(key=k0)

            dout[k0].update({
                'outline_x0': coll.ddata[dgeom['outline'][0]]['data'],
                'outline_x1': coll.ddata[dgeom['outline'][1]]['data'],
                'cents_x': cx,
                'cents_y': cy,
                'cents_z': cz,
                'pix_area': dgeom['pix_area'],
                'parallel': dgeom['parallel'],
            })

        elif cls in ['aperture', 'filter']:

            ap = coll.dobj[cls][k0]['dgeom']
            dout[k0] = {
                'cent': ap['cent'],
                'poly_x': coll.ddata[ap['poly'][0]]['data'],
                'poly_y': coll.ddata[ap['poly'][1]]['data'],
                'poly_z': coll.ddata[ap['poly'][2]]['data'],
                'nin': ap['nin'],
                'e0': ap.get('e0'),
                'e1': ap.get('e1'),
            }

    return dout
