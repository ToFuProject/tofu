# -*- coding: utf-8 -*-


import numpy as np
import scipy.constants as scpct
import datastock as ds


from . import _utils_surface3d
from ..geom._comp_solidangles import _check_polygon_2d, _check_polygon_3d


# #############################################################################
# #############################################################################
#                           Generic
#                   (To be moved to datastock)
# #############################################################################


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
    # curvature
    curve_r=None,
    curve_npts=None,
):

    # ------------
    # check inputs

    # key
    key = _obj_key(coll=coll, which=which, short=which_short, key=key)

    # geometry
    (
        cent,
        outline_x0, outline_x1,
        poly_x, poly_y, poly_z,
        nin, e0, e1,
        area, curve_r, gtype,
    ) = _utils_surface3d._surface3d(
        # 2d outline
        outline_x0=outline_x0,
        outline_x1=outline_x1,
        cent=cent,
        # 3d outline
        poly_x=poly_x,
        poly_y=poly_y,
        poly_z=poly_z,
        # extenthalf
        extenthalf=extenthalf,
        # normal vector at cent
        nin=nin,
        e0=e0,
        e1=e1,
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
    if planar:
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
#                           Camera 1d
# #############################################################################


def _camera_qeff(
    key=None,
    lamb=None,
    energy=None,
    qeff=None,
):
    """ Check qeff is provided as a 1d vector vs lamb or energy

    lamb is assumed to be in m and converted to energy
    energy is assumed to be in eV

    """

    # trivial case
    if qeff is None or (energy is None and lamb is None):
        return None, None

    # non-trivial
    if energy is not None and lamb is not None:
        msg = "Please provide either energy vector xor lamb vector!"
        raise Exception(msg)

    if lamb is not None:
        lamb = np.atleast_1d(lamb).ravel().astype(float)
        energy = (scpct.h * scpct.c / lamb) / scpct.e

    energy = np.atleast_1d(energy).ravel().astype(float)
    qeff = np.atleast_1d(qeff).ravel().astype(float)

    # basic checks
    if energy.shape != qeff.shape:
        msg = "Args energy (or lamb) and qeff must have the same shape!"
        raise Exception(msg)

    iout = (~np.isfinite(energy)) | (energy <= 0.)
    if np.any(iout):
        msg = "Arg energy/lamb must contain only finite positive values!"
        raise Exception(msg)

    iok = np.isfinite(qeff)
    iout = (qeff[iok] < 0.) | (qeff[iok] > 1.)
    if np.any(iout):
        msg = "Arg qeff must contains values in [0, 1] only"
        raise Exception(msg)

    return energy, qeff


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
    # quantum efficiency
    lamb=None,
    energy=None,
    qeff=None,
):

    # ----
    # key

    key = _obj_key(coll=coll, which='camera', short='cam', key=key)

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
            "Non-finite cents detected:\n{iout.nonzero()[0]}"
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
            e0=nin,
            e1=e0,
            e2=e1,
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

    # ------------------
    # quantum efficiency

    energy, qeff = _camera_qeff(
        key=key,
        lamb=lamb,
        energy=energy,
        qeff=qeff,
    )

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
        energy, qeff,
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
    # quantum efficiency
    lamb=None,
    energy=None,
    qeff=None,
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
        energy, qeff,
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
        # quantum efficiency
        lamb=lamb,
        energy=energy,
        qeff=qeff,
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

    if qeff is not None:
        kenergy = f'{key}-energy'
        kqeff = f'{key}-qeff'
        nenergy = energy.size
        dref[kenergy] = {'size': nenergy}
    else:
        kenergy = None
        kqeff = None

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

    if qeff is not None:
        ddata[kenergy] = {
            'data': energy,
            'ref': kenergy,
            'dim': 'energy',
            'quant': 'energy',
            'name': 'energy',
            'units': 'eV',
        }
        ddata[kqeff] = {
            'data': qeff,
            'ref': kenergy,
            'dim': '',
            'quant': '',
            'name': 'quantum efficiency',
            'units': '',
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
                'type': '1d',
                'parallel': parallel,
                'shape': (npix,),
                'ref': (knpix,),
                'pix area': area,
                'pix nb': npix,
                'outline': (kout0, kout1),
                'cent': None,
                'cents': (kcx, kcy, kcz),
                'nin': o_nin,
                'e0': o_e0,
                'e1': o_e1,
                'qeff_energy': kenergy,
                'qeff': kqeff,
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


# #############################################################################
# #############################################################################
#                           Camera 2d
# #############################################################################


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
    # quantum efficiency
    lamb=None,
    energy=None,
    qeff=None,
):

    # ----
    # key

    key = _obj_key(coll=coll, which='camera', short='cam', key=key)

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

    # ------------------
    # quantum efficiency

    energy, qeff = _camera_qeff(
        key=key,
        lamb=lamb,
        energy=energy,
        qeff=qeff,
    )

    return (
        key,
        outline_x0, outline_x1,
        area,
        cent,
        cents_x0, cents_x1,
        npix0, npix1,
        nin, e0, e1,
        energy, qeff,
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
    # quantum efficiency
    lamb=None,
    energy=None,
    qeff=None,
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
        energy, qeff,
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
        # quantum efficiency
        lamb=lamb,
        energy=energy,
        qeff=qeff,
    )

    # ----------
    # dref

    npts = outline_x0.size
    knpts = f'{key}-npts'
    knpix0 = f'{key}-npix0'
    knpix1 = f'{key}-npix1'
    dref = {
        knpts: {'size': npts},
        knpix0: {'size': npix0},
        knpix1: {'size': npix1},
    }

    if qeff is not None:
        kenergy = f'{key}-energy'
        kqeff = f'{key}-qeff'
        nenergy = energy.size
        dref[kenergy] = {'size': nenergy}
    else:
        kenergy = None
        kqeff = None

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

    if qeff is not None:
        ddata[kenergy] = {
            'data': energy,
            'ref': kenergy,
            'dim': 'energy',
            'quant': 'energy',
            'name': 'energy',
            'units': 'eV',
        }
        ddata[kqeff] = {
            'data': qeff,
            'ref': kenergy,
            'dim': '',
            'quant': '',
            'name': 'quantum efficiency',
            'units': '',
        }

    # -----
    # dobj

    dobj = {
        'camera': {
            key: {
                'type': '2d',
                'parallel': True,
                'shape': (npix0, npix1),
                'ref': (knpix0, knpix1),
                'pix area': area,
                'pix nb': npix0 * npix1,
                'outline': (kout0, kout1),
                'cent': cent,
                'cents': (kc0, kc1),
                'nin': nin,
                'e0': e0,
                'e1': e1,
                'qeff_energy': kenergy,
                'qeff': kqeff,
            },
        },
    }

    return dref, ddata, dobj


# #############################################################################
# #############################################################################
#                           Utilities
# #############################################################################


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

    if coll.dobj['camera'][key]['parallel']:
        dout = {
            'nin_x': coll.dobj['camera'][key]['nin'][0],
            'nin_y': coll.dobj['camera'][key]['nin'][1],
            'nin_z': coll.dobj['camera'][key]['nin'][2],
            'e0_x': coll.dobj['camera'][key]['e0'][0],
            'e0_y': coll.dobj['camera'][key]['e0'][1],
            'e0_z': coll.dobj['camera'][key]['e0'][2],
            'e1_x': coll.dobj['camera'][key]['e1'][0],
            'e1_y': coll.dobj['camera'][key]['e1'][1],
            'e1_z': coll.dobj['camera'][key]['e1'][2],
        }
    else:
        cam = coll.dobj['camera'][key]
        dout = {
            'nin_x': coll.ddata[cam['nin'][0]]['data'],
            'nin_y': coll.ddata[cam['nin'][1]]['data'],
            'nin_z': coll.ddata[cam['nin'][2]]['data'],
            'e0_x': coll.ddata[cam['e0'][0]]['data'],
            'e0_y': coll.ddata[cam['e0'][1]]['data'],
            'e0_z': coll.ddata[cam['e0'][2]]['data'],
            'e1_x': coll.ddata[cam['e1'][0]]['data'],
            'e1_y': coll.ddata[cam['e1'][1]]['data'],
            'e1_z': coll.ddata[cam['e1'][2]]['data'],
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

    cam = coll.dobj['camera'][key]
    if cam['type'] == '1d':
        cx = coll.ddata[cam['cents'][0]]['data']
        cy = coll.ddata[cam['cents'][1]]['data']
        cz = coll.ddata[cam['cents'][2]]['data']
    else:
        c0 = coll.ddata[cam['cents'][0]]['data']
        c1 = coll.ddata[cam['cents'][1]]['data']

        cx = (
            cam['cent'][0]
            + np.repeat(c0[:, None], c1.size, axis=1) * cam['e0'][0]
            + np.repeat(c1[None, :], c0.size, axis=0) * cam['e1'][0]
        )
        cy = (
            cam['cent'][1]
            + np.repeat(c0[:, None], c1.size, axis=1) * cam['e0'][1]
            + np.repeat(c1[None, :], c0.size, axis=0) * cam['e1'][1]
        )
        cz = (
            cam['cent'][2]
            + np.repeat(c0[:, None], c1.size, axis=1) * cam['e0'][2]
            + np.repeat(c1[None, :], c0.size, axis=0) * cam['e1'][2]
        )

    return cx, cy, cz


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
            ap = coll.dobj['aperture'][k0]
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
