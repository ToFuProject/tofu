# -*- coding: utf-8 -*-


import warnings


import numpy as np


import datastock as ds



# ##############################################################
# ##############################################################
#           Local to global coordinates
# ##############################################################


def _get_x01toxyz(
    coll=None,
    key=None,
):

    # ---------
    # key

    lcryst = list(coll.dobj.get('crystal', {}).keys())
    lgrat = list(coll.dobj.get('grating', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lcryst + lgrat,
    )

    cls = 'crystal' if key in lcryst else 'grating'
    dgeom = coll.dobj[cls][key]['dgeom']

    # -------------------
    #     Planar
    # -------------------

    if dgeom['type'] == 'planar':

        def x01toxyz(
            x0=None,
            x1=None,
            # surface
            cent=dgeom['cent'],
            e0=dgeom['e0'],
            e1=dgeom['e1'],
        ):
            """ Coordinate transform """

            return (
                cent[0] + x0*e0[0] + x1*e1[0],
                cent[1] + x0*e0[1] + x1*e1[1],
                cent[2] + x0*e0[2] + x1*e1[2],
            )

    # ----------------
    #   Cylindrical
    # ----------------

    elif dgeom['type'] == 'cylindrical':

        iplan = np.isinf(dgeom['curve_r']).nonzero()[0][0]
        eax = ['e0', 'e1'][iplan]
        erot = ['e0', 'e1'][1-iplan]

        def x01toxyz(
            theta=None,
            xx=None,
            # surface
            O=dgeom['cent'] + dgeom['nin'] * dgeom['curve_r'][1 - iplan],
            rc=dgeom['curve_r'][1 - iplan],
            eax=dgeom[eax],
            erot=dgeom[erot],
            # local coordinates
            nin=dgeom['nin'],
        ):
            """ Coordinate transform """

            nox = np.cos(theta)*(-nin[0]) + np.sin(theta)*erot[0]
            noy = np.cos(theta)*(-nin[1]) + np.sin(theta)*erot[1]
            noz = np.cos(theta)*(-nin[2]) + np.sin(theta)*erot[2]

            return (
                O[0] + xx*eax[0] + rc*nox,
                O[1] + xx*eax[1] + rc*noy,
                O[2] + xx*eax[2] + rc*noz,
            )

    # ----------------
    #   Spherical
    # ----------------

    elif dgeom['type'] == 'spherical':

        def x01toxyz(
            dtheta=None,
            phi=None,
            # surface
            O=dgeom['cent'] + dgeom['curve_r'][0]*dgeom['nin'],
            rc=dgeom['curve_r'][0],
            # local coordinates
            nin=dgeom['nin'],
            e0=dgeom['e0'],
            e1=dgeom['e1'],
        ):

            ephix = np.cos(phi)*(-nin[0]) + np.sin(phi)*e0[0]
            ephiy = np.cos(phi)*(-nin[1]) + np.sin(phi)*e0[1]
            ephiz = np.cos(phi)*(-nin[2]) + np.sin(phi)*e0[2]

            nox = np.cos(dtheta)*ephix + np.sin(theta)*e1[0]
            noy = np.cos(dtheta)*ephiy + np.sin(theta)*e1[1]
            noz = np.cos(dtheta)*ephiz + np.sin(theta)*e1[2]

            return (
                O[0] + rc * nox,
                O[1] + rc * noy,
                O[2] + rc * noz,
            )

    # ----------------
    #   Toroidal
    # ----------------

    elif dgeom['type'] == 'toroidal':

        raise NotImplementedError()

    return pts2pt


# ##############################################################
# ##############################################################
#           Global to local coordinates
# ##############################################################


