# -*- coding: utf-8 -*-


import warnings


import numpy as np


import datastock as ds



__all__ = ['compute_los_angles']


# ##################################################################
# ##################################################################
#                       Main
# ##################################################################


def compute_los_angles(
    coll=None,
    key=None,
    # los
    los_x=None,
    los_y=None,
    los_z=None,
    dlos_x=None,
    dlos_y=None,
    dlos_z=None,
    # for storing los
    config=None,
    length=None,
    reflections_nb=None,
    reflections_type=None,
    **kwdargs,
):

    # ------------
    # check inputs

    # key
    lok = list(coll.dobj.get('diagnostic', {}))
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # klos
    klos = f'{key}-los'

    # ref
    optics = coll.dobj['diagnostic'][key]['optics']
    ref = coll.dobj['camera'][optics[0]]['dgeom']['ref']

    # ------------
    # add los

    cx, cy, cz = coll.get_camera_cents_xyz(key=optics[0])

    coll.add_rays(
        key=klos,
        start_x=cx,
        start_y=cy,
        start_z=cz,
        vect_x=los_x,
        vect_y=los_y,
        vect_z=los_z,
        ref=ref,
        diag=key,
        config=config,
        length=length,
        reflections_nb=reflections_nb,
        reflections_type=reflections_type,
    )

    coll.set_param(
        which='diagnostic',
        key=key,
        param='los',
        value=klos,
    )

    # ------------
    # for spectro => estimate angle variations

    if dlos_x is not None:

        angmin = np.full(los_x.shape, np.nan)
        angmax = np.full(los_x.shape, np.nan)

        angles = reflect_ptsvect(
            pts_x=start_x,
            pts_y=start_y,
            pts_z=start_z,
            vect_x=dlos_x,
            vect_y=dlos_y,
            vect_z=dlos_z,
            strict=True,
        )[6]

        angmin = np.min(angles, axis=0)
        angmax = np.min(angles, axis=0)

        # ddata
        kamin = None
        kamax = None
        ddata = {
            kamin: {
                'data': angmin,
                'ref': ref,
                'dim': 'angle',
                'quant': 'angle',
                'name': 'alpha',
                'units': 'rad',
            },
            kamax: {
                'data': angmax,
                'ref': ref,
                'dim': 'angle',
                'quant': 'angle',
                'name': 'alpha',
                'units': 'rad',
            },
        }

        coll.set_param(
            which='diagnostic',
            key=key,
            param='alphamin',
            value=kamin,
        )
        coll.set_param(
            which='diagnostic',
            key=key,
            param='alphamax',
            value=kamax,
        )
