# -*- coding: utf-8 -*-


# Built-in
import copy


# Common
import numpy as np
import datastock as ds


_DPINHOLE = {
    # center
    'x': None,
    'y': None,
    'R': 3.,
    'z': 0.,
    'phi': 0.,
    # angles
    'theta': np.pi,
    'dphi': 0.,
    'tilt': 0,
}


# ##########################################################################
# ##########################################################################
#                   Add pinhole camera
# ##########################################################################


def add_camera_pinhole(
    coll=None,
    key=None,
    key_pinhole=None,
    key_diag=None,
    cam_type=None,
    # position
    x=None,
    y=None,
    R=None,
    z=None,
    phi=None,
    # orientation
    theta=None,
    dphi=None,
    tilt=None,
    # camera
    focal=None,
    pix_nb=None,
    pix_size=None,
    pix_spacing=None,
    # pinhole
    pinhole_radius=None,
    pinhole_size=None,
    # reflections
    reflections_nb=None,
    reflections_type=None,
    # diagnostic
    compute=None,
    config=None,
    length=None,
    # dmat
    dmat=None,
):

    # --------------
    # check inputs

    key, key_pinhole, key_diag, newdiag = _check_camera_pinhole(
        coll=coll,
        key=key,
        key_pinhole=key_pinhole,
        key_diag=key_diag,
    )

    # ------------------------
    # compute pinhole position

    pc, nin, e0, e1 = _pinhole_position(
        # center
        x=x,
        y=y,
        R=R,
        z=z,
        phi=phi,
        # angles
        theta=theta,
        dphi=dphi,
        tilt=tilt,
    )

    # ------------------------
    # compute pinhole contour

    pin_out0, pin_out1 = _pinhole_contour(
        pinhole_radius=pinhole_radius,
        pinhole_size=pinhole_size,
    )

    dgeom_pin = {
        'cent': pc,
        'nin': nin,
        'e0': e0,
        'e1': e1,
        'outline_x0': pin_out0,
        'outline_x1': pin_out1,
    }

    # ------------------------
    # compute camera position

    dgeom_cam, cam_type = _camera_position(
        dgeom_pin=dgeom_pin,
        cam_type=cam_type,
        focal=focal,
        pix_nb=pix_nb,
        pix_size=pix_size,
        pix_spacing=pix_spacing,
    )

    # -------
    # store

    _add_camera_pinhole_store(
        coll=coll,
        key=key,
        key_pinhole=key_pinhole,
        key_diag=key_diag,
        cam_type=cam_type,
        dgeom_pin=dgeom_pin,
        dgeom_cam=dgeom_cam,
        newdiag=newdiag,
        # reflections
        reflections_nb=reflections_nb,
        reflections_type=reflections_type,
        compute=compute,
        config=config,
        length=length,
        # dmat
        dmat=dmat,
    )


def _check_camera_pinhole(
    coll=None,
    key=None,
    key_pinhole=None,
    key_diag=None,
):
    # key
    lout = list(coll.dobj.get('camera', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        excluded=lout,
    )

    # key_pinhole
    if key_pinhole is None:
        key_pinhole = f'{key}_pin'

    lout = list(coll.dobj.get('aperture', {}).keys())
    key_pinhole = ds._generic_check._check_var(
        key_pinhole, 'key_pinhole',
        types=str,
        excluded=lout,
    )

    # key_diag
    if key_diag is not None:
        lok = list(coll.dobj.get('diagnostic', {}).keys())
        key_diag = ds._generic_check._check_var(
            key_diag, 'key_diag',
            types=str,
        )

        newdiag = key_diag not in lok
    else:
        newdiag = True

    return key, key_pinhole, key_diag, newdiag


def _pinhole_position(
    # center
    x=None,
    y=None,
    R=None,
    z=None,
    phi=None,
    # angles
    theta=None,
    dphi=None,
    tilt=None,
    # default
    ddef=None,
):

    # -------------
    # check inputs

    # ddef
    if ddef is None:
        ddef = _DPINHOLE

    # (x, y) vs (R, phi)
    lc = [
        x is not None or y is not None,
        R is not None or phi is not None,
    ]
    if np.sum(lc) > 1:
        msg = "Please provide (x, y) xor (R, phi) !"
        raise Exception(msg)

    # (x, y) vs (R, phi)
    if lc[0]:
        if x is None:
            x = ddef['x']
        if y is None:
            y = ddef['y']
        R = np.hypot(x, y)
        phi = np.arctan2(y, x)
    else:
        if R is None:
            R = ddef['R']
        if phi is None:
            phi = ddef['phi']
        x = R * np.cos(phi)
        y = R * np.sin(phi)

    # z
    if z is None:
        z = ddef['z']

    # dphi
    if dphi is None:
        dphi = ddef['dphi']

    # theta
    if theta is None:
        theta = ddef['theta']

    # tilt
    if tilt is None:
        tilt = ddef['tilt']

    # -------------
    # compute

    # unit vectors of reference
    eR = np.r_[np.cos(phi), np.sin(phi), 0.]
    ephi = np.r_[-np.sin(phi), np.cos(phi), 0.]
    er = np.cos(theta) * eR + np.sin(theta) * np.r_[0, 0, 1]
    etheta = -np.sin(theta) * eR + np.cos(theta) * np.r_[0, 0, 1]
    e0bis = -np.cos(dphi) * ephi + np.sin(dphi) * er

    # translation
    pc = np.r_[x, y, z]

    # unit vectors after rotation
    nin = np.cos(dphi) * er + np.sin(dphi) * ephi
    e0 = np.cos(tilt) * e0bis + np.sin(tilt) * etheta
    e1 = np.cross(nin, e0)

    # safety check
    nin, e0, e1 = ds._generic_check._check_vectbasis(
        e0=nin,
        e1=e0,
        e2=e1,
        dim=3,
        tol=1e-12,
    )

    return pc, nin, e0, e1


def _pinhole_contour(
    pinhole_radius=None,
    pinhole_size=None,
):

    # ------
    # check

    if pinhole_radius is not None and pinhole_size is not None:
        msg = (
            "Please provide pinhole_radius xor pinhole_size!\n"
            "\t- pinhole_radius: created a circular pinhole\n"
            "\t- pinhole_size: created a rectangular pinhole\n"
        )
        raise Exception(msg)

    # ----------
    # compute

    # circular
    if pinhole_radius is not None:

        # check
        pinhole_radius = ds._generic_check._check_var(
            pinhole_radius, 'pinhole_radius',
            types=(int, float),
            sign='> 0.',
        )

        # compute
        theta = np.pi * np.linspace(-1, 1, 50)[:-1]
        out0 = pinhole_radius * np.cos(theta)
        out1 = pinhole_radius * np.sin(theta)

    else:
        # check
        if np.isscalar(pinhole_size):
            pinhole_size = [pinhole_size, pinhole_size]

        pinhole_size = ds._generic_check._check_flat1darray(
            pinhole_size, 'pinhole_size',
            dtype=float,
            size=2,
            sign='> 0.',
            norm=False,
            can_be_None=False,
        )

        # compute
        out0 = pinhole_size[0] * np.r_[-1, 1, 1, -1]
        out1 = pinhole_size[1] * np.r_[-1, -1, 1, 1]

    return out0, out1


def _camera_position(
    dgeom_pin=None,
    cam_type=None,
    focal=None,
    pix_nb=None,
    pix_size=None,
    pix_spacing=None,
):

    # -------------
    # check

    # cam_type
    cam_type = ds._generic_check._check_var(
        cam_type, 'cam_type',
        types=str,
        default='1d',
        allowed=['1d', '2d'],
    )

    # cam_type
    focal = ds._generic_check._check_var(
        focal,'focal',
        types=(int, float),
        default=0.1,
        sign='>0.',
    )

    if cam_type == '1d':

        # pix_nb
        pix_nb = ds._generic_check._check_var(
            pix_nb, 'pix_nb',
            types=int,
            sign='> 0',
        )

        # pix_spacing
        pix_spacing = ds._generic_check._check_var(
            pix_spacing, 'pix_spacing',
            types=(float, int),
            default=0,
            sign='>= 0',
        )

    else:

        # pix_nb
        if np.isscalar(pix_nb):
            pix_nb = pix_nb * np.r_[1, 1]

        pix_nb = ds._generic_check._check_flat1darray(
            pix_nb, 'pix_nb',
            size=2,
            dtype=int,
            sign='> 0',
        )

        # pix_spacing
        if pix_spacing is None:
            pix_spacing = 0.
        if np.isscalar(pix_spacing):
            pix_spacing = pix_spacing * np.r_[1, 1]

        pix_spacing = ds._generic_check._check_flat1darray(
            pix_spacing, 'pix_spacing',
            size=2,
            dtype=float,
            sign='>= 0',
        )

    # pix_size
    if np.isscalar(pix_size):
        pix_size = pix_size * np.r_[1, 1]

    pix_size = ds._generic_check._check_flat1darray(
        pix_size, 'pix_size',
        size=2,
        dtype=float,
        sign='> 0',
    )

    # ----------
    # compute

    dgeom_cam = copy.deepcopy(dgeom_pin)
    cent = dgeom_pin['cent'] - focal * dgeom_pin['nin']

    if cam_type == '1d':

        # vectors
        for k0 in ['nin', 'e0', 'e1']:
            for ii, ss in enumerate(['x', 'y', 'z']):
                dgeom_cam[f'{k0}_{ss}'] = dgeom_cam[k0][ii]
            del dgeom_cam[k0]

        # cents
        dd = (pix_size[0] + pix_spacing) * np.arange(0, pix_nb)
        dd = dd - np.mean(dd)
        dgeom_cam['cents_x'] = cent[0] + dd * dgeom_pin['e0'][0]
        dgeom_cam['cents_y'] = cent[1] + dd * dgeom_pin['e0'][1]
        dgeom_cam['cents_z'] = cent[2] + dd * dgeom_pin['e0'][2]
        del dgeom_cam['cent']

    else:
        for ii in [0, 1]:
            dd = (pix_size[ii] + pix_spacing[ii]) * np.arange(0, pix_nb[ii])
            dgeom_cam[f'cents_x{ii}'] = dd - np.mean(dd)

        dgeom_cam['cent'] = cent

    # -------------
    # pixel outline

    dgeom_cam['outline_x0'] = 0.5 * pix_size[0] * np.r_[-1, 1, 1, -1]
    dgeom_cam['outline_x1'] = 0.5 * pix_size[1] * np.r_[-1, -1, 1, 1]

    return dgeom_cam, cam_type


def _add_camera_pinhole_store(
    coll=None,
    key=None,
    key_pinhole=None,
    key_diag=None,
    cam_type=None,
    dgeom_pin=None,
    dgeom_cam=None,
    newdiag=None,
    # reflections
    reflections_nb=None,
    reflections_type=None,
    compute=None,
    config=None,
    length=None,
    # dmat
    dmat=None,
):

    # compute
    compute = ds._generic_check._check_var(
        compute, 'compute',
        types=bool,
        default=True,
    )

    # pinhole aperture
    coll.add_aperture(
        key=key_pinhole,
        **dgeom_pin,
    )

    # camera
    if cam_type == '1d':
        coll.add_camera_1d(
            key=key,
            dgeom=dgeom_cam,
            dmat=dmat,
        )
    else:
        coll.add_camera_2d(
            key=key,
            dgeom=dgeom_cam,
            dmat=dmat,
        )

    # doptics
    doptics = {key: [key_pinhole]}

    # diagnostic
    if newdiag is False:

        # delete and recreate diagnostic
        dop = {
            k0: v0['optics']
            for k0, v0 in coll._dobj['diagnostic'][key_diag]['doptics'].items()
        }
        dop.update(doptics)
        doptics = dop
        coll.remove_diagnostic(key=key_diag)

    # create
    coll.add_diagnostic(
        key=key_diag,
        doptics=doptics,
        reflections_nb=reflections_nb,
        reflections_type=reflections_type,
        compute=compute,
        config=config,
        length=length,
    )
