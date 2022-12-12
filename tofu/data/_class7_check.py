# -*- coding: utf-8 -*-


import numpy as np
import scipy.constants as scpct
import datastock as ds


from . import _utils_surface3d
from . import _class4_check

from ..geom._comp_solidangles import _check_polygon_2d


_DMAT_KEYS = {
    'name': {'types': str},
    'symbol': {'types': str},
    'thickness': {'types': float, 'sign': '> 0.'},
    'energy': {'dtype': float, 'sign': '> 0.'},
    'qeff': {'dtype': float, 'sign': ['>= 0.', '<= 1.']},
}



# #############################################################################
# #############################################################################
#                   Camera 1d
# #############################################################################


def _camera_1d_check(
    coll=None,
    key=None,
    # outline
    outline_x0=None,
    outline_x1=None,
    # centers of all pixels
    cents_x=None,
    cents_y=None,
    cents_z=None,
    # inwards normal vectors
    nin_x=None,
    nin_y=None,
    nin_z=None,
    # orthonormal direct base
    e0_x=None,
    e0_y=None,
    e0_z=None,
    e1_x=None,
    e1_y=None,
    e1_z=None,
):

    # ----
    # key

    key = ds._generic_check._obj_key(
        d0=coll.dobj.get('camera', {}), short='cam', key=key,
    )

    # ---------
    # outline

    outline_x0, outline_x1, area = _check_polygon_2d(
        poly_x=outline_x0,
        poly_y=outline_x1,
        poly_name=f'{key}-outline',
        can_be_None=False,
        closed=False,
        counter_clockwise=True,
        return_area=True,
    )

    # -----------
    # cents

    cents_x = np.atleast_1d(cents_x).ravel().astype(float)
    cents_y = np.atleast_1d(cents_y).ravel().astype(float)
    cents_z = np.atleast_1d(cents_z).ravel().astype(float)

    # shapes
    if not (cents_x.shape == cents_y.shape == cents_z.shape):
        lstr = [
            ('cents_x', cents_x.shape),
            ('cents_y', cents_y.shape),
            ('cents_z', cents_z.shape),
        ]
        lstr = [f"\t- {kk}.shape: {vv}" for kk, vv in lstr]
        msg = (
            "Args cents_x, cents_y, cents_z must have the same shape!\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    iout = ~(
        np.isfinite(cents_x) & np.isfinite(cents_y) & np.isfinite(cents_z)
    )
    if np.any(iout):
        msg = (
            f"Non-finite cents detected for cam1d '{key}':\n"
            f"\t- indices: {iout.nonzero()[0]}\n"
            f"\t- cents_x: {cents_x[iout]}\n"
            f"\t- cents_y: {cents_y[iout]}\n"
            f"\t- cents_z: {cents_z[iout]}\n"
        )
        raise Exception(msg)

    # total nb of pixels
    npix = cents_x.size

    # make sure all cents are different
    dist = np.full((npix,), np.nan)
    for ii in range(npix):
        dist[:] = (
            (cents_x - cents_x[ii])**2
            + (cents_y - cents_y[ii])**2
            + (cents_z - cents_z[ii])**2
        )
        dist[ii] = 10
        if np.any(dist < 1.e-15):
            msg = (
                "Identical cents detected:\n"
                f"\t- ref: {ii}\n"
                f"\t- identicals: {(dist < 1.e-15).nonzero()[0]}\n"
            )
            raise Exception(msg)

    # -----------
    # unit vectors

    lv = [
        ['nin_x', nin_x], ['nin_y', nin_y], ['nin_z', nin_z],
        ['e0_x', e0_x], ['e0_y', e0_y], ['e0_z', e0_z],
        ['e1_x', e1_x], ['e1_y', e1_y], ['e1_z', e1_z],
    ]

    # check they are all provided
    lNone = [vv[0] for vv in lv if vv[1] is None]
    if len(lNone) > 0:
        msg = (
            f"All unit vectors must be provided for camera '{key}'!\n"
            f"The following are not provided: {lNone}"
        )
        raise Exception(msg)

    # particular case: scalar because common to all
    c0 = all([np.isscalar(vv[1]) for vv in lv])
    if c0:
        parallel = True
        nin, e0, e1 = ds._generic_check._check_vectbasis(
            e0=np.r_[nin_x, nin_y, nin_z],
            e1=np.r_[e0_x, e0_y, e0_z],
            e2=np.r_[e1_x, e1_y, e1_z],
            dim=3,
        )

    else:

        parallel = False

        # force into numpy array
        for vv in lv:
            vv[1] = np.atleast_1d(vv[1]).ravel().astype(float)

        # check shapes
        dshape = {vv[0]: vv[1].shape for vv in lv if vv[1].shape != (npix,)}
        if len(set(dshape.values())) > 1:
            lstr = [f"\t- {k0}: {v0}" for k0, v0 in dshape.items()]
            msg = (
                f"All unit vector componant must have shape ({npix},)!\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

        # force normalization
        norm = np.sqrt((lv[0][1]**2 + lv[1][1]**2 + lv[2][1]**2))
        nin_x = lv[0][1] / norm
        nin_y = lv[1][1] / norm
        nin_z = lv[2][1] / norm

        norm = np.sqrt((lv[3][1]**2 + lv[4][1]**2 + lv[5][1]**2))
        e0_x = lv[3][1] / norm
        e0_y = lv[4][1] / norm
        e0_z = lv[5][1] / norm

        norm = np.sqrt((lv[6][1]**2 + lv[7][1]**2 + lv[8][1]**2))
        e1_x = lv[6][1] / norm
        e1_y = lv[7][1] / norm
        e1_z = lv[8][1] / norm

        # check perpendicularity
        sca = (nin_x*e0_x + nin_y*e0_y + nin_z*e0_z)
        if np.any(np.abs(sca) > 1e-14):
            msg = "Non-perpendicular nin vs e0:\n{(sca > 1.e-14).nonzero()[0]}"
            raise Exception(msg)

        sca = (nin_x*e1_x + nin_y*e1_y + nin_z*e1_z)
        if np.any(np.abs(sca) > 1e-14):
            msg = "Non-perpendicular nin vs e1:\n{(sca > 1.e-14).nonzero()[0]}"
            raise Exception(msg)

        sca = (e0_x*e1_x + e0_y*e1_y + e0_z*e1_z)
        if np.any(np.abs(sca) > 1e-14):
            msg = "Non-perpendicular e0 vs e1:\n{(sca > 1.e-14).nonzero()[0]}"
            raise Exception(msg)

        # check right-handedness
        sca = (
            e1_x * (nin_y * e0_z - nin_z * e0_y)
            + e1_y * (nin_z * e0_x - nin_x * e0_z)
            + e1_z * (nin_x * e0_y - nin_y * e0_x)
        )
        if np.any(sca <= 0.):
            msg = (
                "The following unit vectors do not seem right-handed:\n"
                f"{(sca <= 0.).nonzero()[0]}"
            )
            raise Exception(msg)

        nin = (nin_x, nin_y, nin_z)
        e0 = (e0_x, e0_y, e0_z)
        e1 = (e1_x, e1_y, e1_z)

    return (
        key,
        outline_x0, outline_x1,
        area,
        cents_x,
        cents_y,
        cents_z,
        npix,
        parallel,
        nin, e0, e1,
    )


def _camera_1d(
    coll=None,
    key=None,
    # common 2d outline
    outline_x0=None,
    outline_x1=None,
    # centers of all pixels
    cents_x=None,
    cents_y=None,
    cents_z=None,
    # inwards normal vectors
    nin_x=None,
    nin_y=None,
    nin_z=None,
    # orthonormal direct base
    e0_x=None,
    e0_y=None,
    e0_z=None,
    e1_x=None,
    e1_y=None,
    e1_z=None,
):

    # ------------
    # check inputs

    (
        key,
        outline_x0, outline_x1,
        area,
        cents_x,
        cents_y,
        cents_z,
        npix,
        parallel,
        nin, e0, e1,
    ) = _camera_1d_check(
        coll=coll,
        key=key,
        # outline
        outline_x0=outline_x0,
        outline_x1=outline_x1,
        # centers of all pixels
        cents_x=cents_x,
        cents_y=cents_y,
        cents_z=cents_z,
        # inwards normal vectors
        nin_x=nin_x,
        nin_y=nin_y,
        nin_z=nin_z,
        # orthonormal direct base
        e0_x=e0_x,
        e0_y=e0_y,
        e0_z=e0_z,
        e1_x=e1_x,
        e1_y=e1_y,
        e1_z=e1_z,
    )

    # ----------
    # dref

    npts = outline_x0.size
    knpts = f'{key}-npts'
    knpix = f'{key}-npix'
    dref = {
        knpts: {'size': npts},
        knpix: {'size': npix},
    }

    # -------------
    # ddata

    kcx = f'{key}-cx'
    kcy = f'{key}-cy'
    kcz = f'{key}-cz'
    kout0 = f'{key}_outx0'
    kout1 = f'{key}_outx1'

    ddata = {
        kout0: {
            'data': outline_x0,
            'ref': knpts,
            'dim': 'distance',
            'quant': 'x0',
            'name': 'x0',
            'units': 'm',
        },
        kout1: {
            'data': outline_x1,
            'ref': knpts,
            'dim': 'distance',
            'quant': 'x1',
            'name': 'x1',
            'units': 'm',
        },
        kcx: {
            'data': cents_x,
            'ref': knpix,
            'dim': 'distance',
            'quant': 'x',
            'name': 'x',
            'units': 'm',
        },
        kcy: {
            'data': cents_y,
            'ref': knpix,
            'dim': 'distance',
            'quant': 'y',
            'name': 'y',
            'units': 'm',
        },
        kcz: {
            'data': cents_z,
            'ref': knpix,
            'dim': 'distance',
            'quant': 'z',
            'name': 'z',
            'units': 'm',
        },
    }

    # -----
    # dobj

    if parallel:
        o_nin = nin
        o_e0 = e0
        o_e1 = e1

    else:
        kinx = f'{key}-nin_x'
        kiny = f'{key}-nin_y'
        kinz = f'{key}-nin_z'
        ke0x = f'{key}-e0_x'
        ke0y = f'{key}-e0_y'
        ke0z = f'{key}-e0_z'
        ke1x = f'{key}-e1_x'
        ke1y = f'{key}-e1_y'
        ke1z = f'{key}-e1_z'

        o_nin = (kinx, kiny, kinz)
        o_e0 = (ke0x, ke0y, ke0z)
        o_e1 = (ke1x, ke1y, ke1z)

    # dobj
    dobj = {
        'camera': {
            key: {
                'dgeom': {
                    'type': '1d',
                    'parallel': parallel,
                    'shape': (npix,),
                    'ref': (knpix,),
                    'ref_flat': (knpix,),
                    'pix_area': area,
                    'pix_nb': npix,
                    'outline': (kout0, kout1),
                    'cent': None,
                    'cents': (kcx, kcy, kcz),
                    'nin': o_nin,
                    'e0': o_e0,
                    'e1': o_e1,
                },
            },
        },
    }

    # ------------------------
    # parallel vs non-parallel

    if not parallel:
        ddata.update({
            kinx: {
                'data': nin[0],
                'ref': knpix,
                'dim': 'distance',
                'quant': 'x',
                'name': 'x',
                'units': 'm',
            },
            kiny: {
                'data': nin[1],
                'ref': knpix,
                'dim': 'distance',
                'quant': 'y',
                'name': 'y',
                'units': 'm',
            },
            kinz: {
                'data': nin[2],
                'ref': knpix,
                'dim': 'distance',
                'quant': 'z',
                'name': 'z',
                'units': 'm',
            },
            ke0x: {
                'data': e0[0],
                'ref': knpix,
                'dim': 'distance',
                'quant': 'x',
                'name': 'x',
                'units': 'm',
            },
            ke0y: {
                'data': e0[1],
                'ref': knpix,
                'dim': 'distance',
                'quant': 'y',
                'name': 'y',
                'units': 'm',
            },
            ke0z: {
                'data': e0[2],
                'ref': knpix,
                'dim': 'distance',
                'quant': 'z',
                'name': 'z',
                'units': 'm',
            },
            ke1x: {
                'data': e1[0],
                'ref': knpix,
                'dim': 'distance',
                'quant': 'x',
                'name': 'x',
                'units': 'm',
            },
            ke1y: {
                'data': e1[1],
                'ref': knpix,
                'dim': 'distance',
                'quant': 'y',
                'name': 'y',
                'units': 'm',
            },
            ke1z: {
                'data': e1[2],
                'ref': knpix,
                'dim': 'distance',
                'quant': 'z',
                'name': 'z',
                'units': 'm',
            },
        })

    return dref, ddata, dobj


# ##################################################################
# ##################################################################
#                           Camera 2d
# ##################################################################


def _camera_2d_check(
    coll=None,
    key=None,
    # outline
    outline_x0=None,
    outline_x1=None,
    # centers of all pixels
    cent=None,
    cents_x0=None,
    cents_x1=None,
    # inwards normal vectors
    nin=None,
    e0=None,
    e1=None,
):

    # ----
    # key

    key = ds._generic_check._obj_key(
        d0=coll.dobj.get('camera', {}), short='cam', key=key,
    )

    # ---------
    # outline

    outline_x0, outline_x1, area = _check_polygon_2d(
        poly_x=outline_x0,
        poly_y=outline_x1,
        poly_name=f'{key}-outline',
        can_be_None=False,
        closed=False,
        counter_clockwise=True,
        return_area=True,
    )

    # -----------
    # cent

    cent = np.atleast_1d(cent).ravel().astype(float)
    if cent.shape != (3,) or np.any(~np.isfinite(cent)):
        msg = f"Arg cent non valid shape {cent.shape} vs (3,) or non-finite!"
        raise Exception(msg)

    # -----------
    # cents

    cents_x0 = np.atleast_1d(cents_x0).ravel().astype(float)
    cents_x1 = np.atleast_1d(cents_x1).ravel().astype(float)

    # finite
    iout = ~np.isfinite(cents_x0)
    if np.any(iout):
        msg = "Non-finite cents_x0 detected:\n{iout.nonzero()[0]}"
        raise Exception(msg)

    iout = ~np.isfinite(cents_x1)
    if np.any(iout):
        msg = "Non-finite cents_x1 detected:\n{iout.nonzero()[0]}"
        raise Exception(msg)

    # total nb of pixels
    npix0 = cents_x0.size
    npix1 = cents_x1.size

    # make sure all cents are different
    if np.unique(cents_x0).size != cents_x0.size:
        msg = "Double values found in cents_x0!"
        raise Exception(msg)

    if np.unique(cents_x1).size != cents_x1.size:
        msg = "Double values found in cents_x1!"
        raise Exception(msg)

    # -----------
    # unit vectors

    lv = [('nin', nin), ('e0', e0), ('e1', e1)]

    # check they are all provided
    lNone = [vv[0] for vv in lv if vv[1] is None]
    if len(lNone) > 0:
        msg = (
            f"All unit vectors must be provided for camera '{key}'!\n"
            f"The following are not provided: {lNone}"
        )
        raise Exception(msg)

    # particular case: scalar because common to all
    nin, e0, e1 = ds._generic_check._check_vectbasis(
        e0=nin,
        e1=e0,
        e2=e1,
        dim=3,
    )

    return (
        key,
        outline_x0, outline_x1,
        area,
        cent,
        cents_x0, cents_x1,
        npix0, npix1,
        nin, e0, e1,
    )


def _camera_2d(
    coll=None,
    key=None,
    # common 2d outline
    outline_x0=None,
    outline_x1=None,
    # centers of all pixels
    cent=None,
    cents_x0=None,
    cents_x1=None,
    # inwards normal vectors
    nin=None,
    e0=None,
    e1=None,
):

    # ------------
    # check inputs

    (
        key,
        outline_x0, outline_x1,
        area,
        cent,
        cents_x0, cents_x1,
        npix0, npix1,
        nin, e0, e1,
    ) = _camera_2d_check(
        coll=coll,
        key=key,
        # outline
        outline_x0=outline_x0,
        outline_x1=outline_x1,
        # centers of all pixels
        cent=cent,
        cents_x0=cents_x0,
        cents_x1=cents_x1,
        # inwards normal vectors
        nin=nin,
        e0=e0,
        e1=e1,
    )

    # ----------
    # dref

    npts = outline_x0.size
    knpts = f'{key}-npts'
    knpix0 = f'{key}-npix0'
    knpix1 = f'{key}-npix1'
    knpix = f'{key}-npix'
    dref = {
        knpts: {'size': npts},
        knpix0: {'size': npix0},
        knpix1: {'size': npix1},
        knpix: {'size': npix0*npix1},
    }

    # -------------
    # ddata

    kc0 = f'{key}-c0'
    kc1 = f'{key}-c1'
    kout0 = f'{key}_outx0'
    kout1 = f'{key}_outx1'

    ddata = {
        kout0: {
            'data': outline_x0,
            'ref': knpts,
            'dim': 'distance',
            'quant': 'x0',
            'name': 'x0',
            'units': 'm',
        },
        kout1: {
            'data': outline_x1,
            'ref': knpts,
            'dim': 'distance',
            'quant': 'x1',
            'name': 'x1',
            'units': 'm',
        },
        kc0: {
            'data': cents_x0,
            'ref': knpix0,
            'dim': 'distance',
            'quant': 'x0',
            'name': 'x0',
            'units': 'm',
        },
        kc1: {
            'data': cents_x1,
            'ref': knpix1,
            'dim': 'distance',
            'quant': 'x1',
            'name': 'x1',
            'units': 'm',
        },
    }

    # -----
    # dobj

    dobj = {
        'camera': {
            key: {
                'dgeom': {
                    'type': '2d',
                    'parallel': True,
                    'shape': (npix0, npix1),
                    'ref': (knpix0, knpix1),
                    'ref_flat': (knpix,),
                    'pix_area': area,
                    'pix_nb': npix0 * npix1,
                    'outline': (kout0, kout1),
                    'cent': cent,
                    'cents': (kc0, kc1),
                    'nin': nin,
                    'e0': e0,
                    'e1': e1,
                },
            },
        },
    }

    return dref, ddata, dobj


# ##################################################################
# ##################################################################
#                       Camera dmat
# ##################################################################


def _dmat(
    key=None,
    dmat=None,
):
    """ Check qeff is provided as a 1d vector vs lamb or energy

    lamb is assumed to be in m and converted to energy
    energy is assumed to be in eV

    """

    # ------------
    # trivial case

    if dmat is None:
        return None, None, None

    # ------
    # Check

    # Check dict type and content (each key is a valid string)
    dmat = ds._generic_check._check_dict_valid_keys(
        var=dmat,
        varname='dmat',
        has_all_keys=False,
        has_only_keys=True,
        keys_can_be_None=True,
        dkeys=_DMAT_KEYS,
    )

    # -----------------------------------
    # check energy / qeff values

    dref, ddata = None, None
    if all([dmat.get(kk) is not None for kk in ['energy', 'qeff']]):
        
        dmat['energy'], dmat['qeff'] = _class4_check._dmat_energy_trans(
            energ=dmat['energy'],
            trans=dmat['qeff'],
        )

        # ----------
        # dref
    
        kne = f'{key}-nE'
        ne = dmat['energy'].size
        dref = {
            kne: {'size': ne},
        }
    
        # ----------
        # ddata
    
        kE = f'{key}-E'
        kqeff = f'{key}-qeff'
    
        ddata = {
            kE: {
                'data': dmat['energy'],
                'ref': kne,
                'dim': 'energy',
                'quant': 'energy',
                'name': 'E',
                'units': 'eV',
            },
            kqeff: {
                'data': dmat['qeff'],
                'ref': kne,
                'dim': None,
                'quant': 'quantum eff.',
                'name': '',
                'units': '',
            },
        }
    
        # -----------
        # dmat
    
        dmat['energy'] = kE
        dmat['qeff'] = kqeff

    return dref, ddata, dmat


# ##################################################################
# ##################################################################
#                           Utilities
# ##################################################################


def get_camera_unitvectors(
    coll=None,
    key=None,
):

    # ---------
    # check key

    lok = list(coll.dobj.get('camera', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # ---------------------------
    # get unit vector components

    dgeom = coll.dobj['camera'][key]['dgeom']
    if dgeom['parallel']:
        dout = {
            'nin_x': dgeom['nin'][0],
            'nin_y': dgeom['nin'][1],
            'nin_z': dgeom['nin'][2],
            'e0_x':  dgeom['e0'][0],
            'e0_y':  dgeom['e0'][1],
            'e0_z':  dgeom['e0'][2],
            'e1_x':  dgeom['e1'][0],
            'e1_y':  dgeom['e1'][1],
            'e1_z':  dgeom['e1'][2],
        }
    else:
        dout = {
            'nin_x': coll.ddata[dgeom['nin'][0]]['data'],
            'nin_y': coll.ddata[dgeom['nin'][1]]['data'],
            'nin_z': coll.ddata[dgeom['nin'][2]]['data'],
            'e0_x':  coll.ddata[dgeom['e0'][0]]['data'],
            'e0_y':  coll.ddata[dgeom['e0'][1]]['data'],
            'e0_z':  coll.ddata[dgeom['e0'][2]]['data'],
            'e1_x':  coll.ddata[dgeom['e1'][0]]['data'],
            'e1_y':  coll.ddata[dgeom['e1'][1]]['data'],
            'e1_z':  coll.ddata[dgeom['e1'][2]]['data'],
        }

    return dout


def get_camera_cents_xyz(coll=None, key=None):

    # ---------
    # check key

    lok = list(coll.dobj.get('camera', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # ---------------------------
    # get unit vector components

    dgeom = coll.dobj['camera'][key]['dgeom']
    if dgeom['type'] == '1d':
        cx = coll.ddata[dgeom['cents'][0]]['data']
        cy = coll.ddata[dgeom['cents'][1]]['data']
        cz = coll.ddata[dgeom['cents'][2]]['data']
    else:
        c0 = coll.ddata[dgeom['cents'][0]]['data']
        c1 = coll.ddata[dgeom['cents'][1]]['data']

        cx = (
            dgeom['cent'][0]
            + np.repeat(c0[:, None], c1.size, axis=1) * dgeom['e0'][0]
            + np.repeat(c1[None, :], c0.size, axis=0) * dgeom['e1'][0]
        )
        cy = (
            dgeom['cent'][1]
            + np.repeat(c0[:, None], c1.size, axis=1) * dgeom['e0'][1]
            + np.repeat(c1[None, :], c0.size, axis=0) * dgeom['e1'][1]
        )
        cz = (
            dgeom['cent'][2]
            + np.repeat(c0[:, None], c1.size, axis=1) * dgeom['e0'][2]
            + np.repeat(c1[None, :], c0.size, axis=0) * dgeom['e1'][2]
        )

    return cx, cy, cz
