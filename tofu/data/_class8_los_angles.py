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
    kref=None,
    is2d=None,
    # los
    los_x=None,
    los_y=None,
    los_z=None,
    dlos_x=None,
    dlos_y=None,
    dlos_z=None,
    iok=None,
    cx=None,
    cy=None,
    cz=None,
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

    cx2, cy2, cz2 = coll.get_camera_cents_xyz(key=optics[0])

    coll.add_rays(
        key=klos,
        start_x=cx2,
        start_y=cy2,
        start_z=cz2,
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

        angmin = np.full(cx.size, np.nan)
        angmax = np.full(cx.size, np.nan)

        ptsvect = coll.get_optics_reflect_ptsvect(key=kref)

        for ii in range(cx.size):
            if not iok[ii]:
                continue
            angles = ptsvect(
                pts_x=cx[ii],
                pts_y=cy[ii],
                pts_z=cz[ii],
                vect_x=dlos_x[ii, ...],
                vect_y=dlos_y[ii, ...],
                vect_z=dlos_z[ii, ...],
                strict=True,
                return_x01=False,
            )[6]

            angmin[ii] = np.nanmin(angles)
            angmax[ii] = np.nanmax(angles)

        if is2d:
            angmin = angmin.reshape(los_x.shape)
            angmax = angmax.reshape(los_x.shape)

        # ddata
        kamin = f'{key}_amin'
        kamax = f'{key}_amax'
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
        coll.update(ddata=ddata)

        coll.set_param(
            which='diagnostic',
            key=key,
            param='amin',
            value=kamin,
        )
        coll.set_param(
            which='diagnostic',
            key=key,
            param='amax',
            value=kamax,
        )
