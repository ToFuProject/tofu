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
    asplane=None,
):

    # ---------
    # key

    key, cls = coll.get_optics_cls(optics=key)
    key, cls = key[0], cls[0]
    dgeom = coll.dobj[cls][key]['dgeom']

    # asplane
    asplane = ds._generic_check._check_var(
        asplane, 'asplane',
        types=bool,
        default=False,
    )

    # -------------------
    #     Planar
    # -------------------

    if dgeom['type'] == 'planar' or asplane is True:

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

        rc = dgeom['curve_r'][1 - iplan]
        rcs = np.sign(rc)
        rca = np.abs(rc)

        def x01toxyz(
            x0=None,
            x1=None,
            # surface
            O=dgeom['cent'] + dgeom['nin'] * rc,
            rcs=rcs,
            rca=rca,
            eax=dgeom[eax],
            erot=dgeom[erot],
            # local coordinates
            nin=dgeom['nin'],
            iplan=iplan,
        ):
            """ Coordinate transform """

            if iplan == 0:
                xx, theta = x0, x1
            else:
                xx, theta = x1, x0

            nox = np.cos(theta)*(-rcs*nin[0]) + np.sin(theta)*erot[0]
            noy = np.cos(theta)*(-rcs*nin[1]) + np.sin(theta)*erot[1]
            noz = np.cos(theta)*(-rcs*nin[2]) + np.sin(theta)*erot[2]

            return (
                O[0] + xx*eax[0] + rca*nox,
                O[1] + xx*eax[1] + rca*noy,
                O[2] + xx*eax[2] + rca*noz,
            )

    # ----------------
    #   Spherical
    # ----------------

    elif dgeom['type'] == 'spherical':

        rc = dgeom['curve_r'][0]
        rcs = np.sign(rc)
        rca = np.abs(rc)

        def x01toxyz(
            x0=None,
            x1=None,
            # surface
            O=dgeom['cent'] + dgeom['nin'] * rc,
            rcs=rcs,
            rca=rca,
            # local coordinates
            nin=dgeom['nin'],
            e0=dgeom['e0'],
            e1=dgeom['e1'],
        ):

            dtheta, phi = x1, x0

            ephix = np.cos(phi)*(-rcs*nin[0]) + np.sin(phi)*e0[0]
            ephiy = np.cos(phi)*(-rcs*nin[1]) + np.sin(phi)*e0[1]
            ephiz = np.cos(phi)*(-rcs*nin[2]) + np.sin(phi)*e0[2]

            nox = np.cos(dtheta)*ephix + np.sin(dtheta)*e1[0]
            noy = np.cos(dtheta)*ephiy + np.sin(dtheta)*e1[1]
            noz = np.cos(dtheta)*ephiz + np.sin(dtheta)*e1[2]

            return (
                O[0] + rca * nox,
                O[1] + rca * noy,
                O[2] + rca * noz,
            )

    # ----------------
    #   Toroidal
    # ----------------

    elif dgeom['type'] == 'toroidal':

        raise NotImplementedError()

    return x01toxyz
