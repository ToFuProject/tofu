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
    dcompute=None,
    # for storing los
    config=None,
    length=None,
    reflections_nb=None,
    reflections_type=None,
    key_nseg=None,
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
    is2d = coll.dobj['diagnostic'][key]['is2d']

    for key_cam, v0 in dcompute.items():
        
        if v0['los_x'] is None or not np.any(np.isfinite(v0['los_x'])):
            continue
        
        # klos
        klos = f'{key}_{key_cam}_los'
    
        # ref
        ref = coll.dobj['camera'][key_cam]['dgeom']['ref']
    
        # ------------
        # add los
    
        cx2, cy2, cz2 = coll.get_camera_cents_xyz(key=key_cam)
    
        coll.add_rays(
            key=klos,
            start_x=cx2,
            start_y=cy2,
            start_z=cz2,
            vect_x=v0['los_x'],
            vect_y=v0['los_y'],
            vect_z=v0['los_z'],
            ref=ref,
            diag=key,
            key_cam=key_cam,
            config=config,
            length=length,
            reflections_nb=reflections_nb,
            reflections_type=reflections_type,
            key_nseg=key_nseg,
        )
    
        coll._dobj['diagnostic'][key]['doptics'][key_cam]['los'] = klos

        # ------------
        # for spectro => estimate angle variations

        if v0.get('dlos_x') is not None:

            angmin = np.full(v0['cx'].size, np.nan)
            angmax = np.full(v0['cx'].size, np.nan)
    
            ptsvect = coll.get_optics_reflect_ptsvect(key=v0['kref'])
    
            for ii in range(v0['cx'].size):
                
                if not v0['iok'][ii]:
                    continue
                
                angles = ptsvect(
                    pts_x=v0['cx'][ii],
                    pts_y=v0['cy'][ii],
                    pts_z=v0['cz'][ii],
                    vect_x=v0['dlos_x'][ii, ...],
                    vect_y=v0['dlos_y'][ii, ...],
                    vect_z=v0['dlos_z'][ii, ...],
                    strict=True,
                    return_x01=False,
                )[6]
    
                angmin[ii] = np.nanmin(angles)
                angmax[ii] = np.nanmax(angles)
    
            if is2d:
                angmin = angmin.reshape(v0['shape0'])
                angmax = angmax.reshape(v0['shape0'])
    
            # ddata
            kamin = f'{key}_{key_cam}_amin'
            kamax = f'{key}_{key_cam}_amax'
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
    
            coll._dobj['diagnostic'][key]['doptics'][key_cam]['amin'] = kamin
            coll._dobj['diagnostic'][key]['doptics'][key_cam]['amax'] = kamax