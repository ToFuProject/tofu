

import warnings


import numpy as np
import datastock as ds


from ..spectro import _rockingcurve


# ###############################################################
# ###############################################################
#                   Rocking curve
# ###############################################################


def rocking_curve(coll=None, key=None):

    # ------------
    # check inputs

    lok = list(coll.dobj.get('crystal', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # --------
    # compute

    raise NotImplementedError
    _rockingcurve.compute_rockingcurve(
    )

    # -------
    # store

    return


# ###############################################################
# ###############################################################
#                   get bragg lambda
# ###############################################################


def _bragglamb(
    coll=None,
    key=None,
    lamb=None,
    bragg=None,
    norder=None,
    rocking_curve=None,
):
    """ Return bragg angle

    If bragg provided, return lamb
    If lamb provided, return bragg

    If lamb is provided return corresponding:
        - bragg angle (simple bragg's law)
        - bragg angle + reflectivity, interpolated on rocking_curve

    """

    # ------------
    # check inputs

    # key
    lok = list(coll.dobj.get('crystal', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # norder
    norder = ds._generic_check._check_var(
        norder, 'norder',
        types=int,
        default=1,
    )

    # rocking_curve
    dmat = coll.dobj['crystal'][key]['dmat']
    lok = [False]
    if dmat is None or dmat.get('drock', {}).get('delta_bragg') is None:
        defrock = False
        lok = (False,)
    else:
        defrock = False
        lok = (False, True)

    rocking_curve = ds._generic_check._check_var(
        rocking_curve, 'rocking_curve',
        types=bool,
        default=defrock,
        allowed=lok,
    )

    # -------------------
    # no input => default

    if bragg is None and lamb is None:
        if dmat is None or dmat.get('target') is None:
            msg = (
                f"Crystal '{key}' has no target lamb!\n"
                f"dmat:\n{dmat}"
            )
            raise Exception(msg)
        else:
            lamb = np.r_[dmat['target']['lamb']]

    if bragg is not None:
        bragg = np.atleast_1d(bragg).astype(float)
    if lamb is not None:
        lamb = np.atleast_1d(lamb).astype(float)

    dist = coll.dobj['crystal'][key]['dmat']['d_hkl']
    if rocking_curve is True:
        delta_bragg = dmat['drock']['delta_bragg']

    # -------------
    # bragg vs lamb

    if bragg is not None and lamb is None:

        if rocking_curve is True:
            lamb = 2. * dist * np.sin(bragg - delta_bragg) / norder

        else:
            lamb = 2. * dist * np.sin(bragg) / norder

    elif lamb is not None and bragg is None:

        if rocking_curve is True:
            bragg = np.arcsin(norder * lamb / (2.*dist)) + delta_bragg

        else:
            bragg = np.arcsin(norder * lamb / (2.*dist))

    else:
        msg = "Interpolate on rocking curve"
        raise NotImplementedError(msg)

    return bragg, lamb


# ###############################################################
# ###############################################################
#                   Ideal configurations
# ###############################################################


def _ideal_configuration_check(
    coll=None,
    key=None,
    configuration=None,
    defocus=None,
    strict_aperture=None,
    # parameters
    cam_on_e0=None,
    # johann-specific
    cam_tangential=None,
    # pinhole-specific
    cam_dimensions=None,
    focal_distance=None,
    # store
    store=None,
    key_cam=None,
    key_aperture=None,
    aperture_dimensions=None,
    pinhole_radius=None,
    cam_pixels_nb=None,
    # returnas
    returnas=None,
):
    # --------------
    # geometry

    # key
    lok = list(coll.dobj.get('crystal', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # gtype
    gtype = coll.dobj['crystal'][key]['dgeom']['type']

    # configuration
    if gtype == 'planar':
        conf = ['pinhole']
    elif gtype == 'cylindrical':
        conf = ['pinhole', 'von hamos', 'johann']
    elif gtype == 'spherical':
        conf = ['pinhole', 'johann']
    elif gtype == 'toroidal':
        conf = []

    if isinstance(configuration, str):
        configuration = configuration.lower()

    configuration = ds._generic_check._check_var(
        configuration, 'configuration',
        types=str,
        allowed=conf,
    )

    # defocus
    defocus = float(ds._generic_check._check_var(
        defocus, 'defocus',
        types=(float, int),
        default=0.,
    ))

    # strict_aperture
    strict_aperture = ds._generic_check._check_var(
        strict_aperture, 'strict_aperture',
        types=bool,
        default=True,
    )

    # cam_on_e0
    cam_on_e0 = ds._generic_check._check_var(
        cam_on_e0, 'cam_on_e0',
        types=bool,
        default=True,
    )

    # cam_tangential
    cam_tangential = ds._generic_check._check_var(
        cam_tangential, 'cam_tangential',
        types=bool,
        default=True,
    )

    # cam_dimensions
    cam_dimensions = ds._generic_check._check_flat1darray(
        cam_dimensions, 'cam_dimensions',
        dtype=float,
        size=[1, 2],
        sign='> 0.',
    )

    if cam_dimensions.size == 1:
        cam_dimensions = cam_dimensions * np.r_[1., 1.]

    # --------------
    # special cases

    # focal_distance
    if configuration == 'pinhole' and gtype == 'planar':
        focal_distance = ds._generic_check._check_var(
            focal_distance, 'focal_distance',
            types=(float, int),
            sign='> 0.',
        )
        focal_distance = float(focal_distance)

    # --------------
    # store-specific

    # store
    store = ds._generic_check._check_var(
        store, 'store',
        types=bool,
        default=False,
    )

    if store is True:

        # key_cam
        lout = list(coll.dobj.get('camera', {}).keys())
        key_cam = ds._generic_check._check_var(
            key_cam, 'key_cam',
            types=str,
            default=f'{key}_cam',
            excluded=lout,
        )

        # key_aperture
        if configuration != 'johann':

            if configuration == 'pinhole':

                lc = [pinhole_radius is None, aperture_dimensions is None]
                if np.sum(lc) != 1:
                    msg = "Provide pinhole_radius xor aperture_dimensions!"
                    raise Exception(msg)

                if pinhole_radius is not None:
                    ap = 'pinhole'
                else:
                    ap = 'slit'
            else:
                ap = 'slit'

            lout = list(coll.dobj.get('aperture', {}).keys())
            key_aperture = ds._generic_check._check_var(
                key_aperture, 'key_aperture',
                types=str,
                default=f'{key}_{ap}',
                excluded=None,
            )

            key_aperture_in = key_aperture in lout

        else:
            key_aperture = None
            key_aperture_in = False

        # cam_pixels_nb
        cam_pixels_nb = ds._generic_check._check_flat1darray(
            cam_pixels_nb, 'cam_pixels_nb',
            dtype=int,
            size=[1, 2],
            sign='> 0.',
        )

        if cam_pixels_nb.size == 1:
            cam_pixels_nb = cam_pixels_nb * np.r_[1, 1]

        # aperture_dimensions
        if key_aperture_in is False:
            if configuration == 'pinhole':
                if pinhole_radius is not None:
                    pinhole_radius = ds._generic_check._check_var(
                        pinhole_radius, 'pinhole_radius',
                        types=float,
                        sign='> 0.',
                    )
                elif aperture_dimensions is not None:
                    aperture_dimensions = ds._generic_check._check_flat1darray(
                        aperture_dimensions, 'aperture_dimensions',
                        dtype=float,
                        size=[1, 2],
                        sign='> 0.',
                    )

            elif configuration == 'von hamos':
                aperture_dimensions = ds._generic_check._check_flat1darray(
                    aperture_dimensions, 'aperture_dimensions',
                    dtype=float,
                    size=[1, 2],
                    sign='> 0.',
                )

            if aperture_dimensions is not None:
                if len(aperture_dimensions) == 1:
                    aperture_dimensions = aperture_dimensions * np.r_[1., 1.]

    # returnas
    returnas = ds._generic_check._check_var(
        returnas, 'returnas',
        default=False if store else dict,
        allowed=[False, dict, list],
    )

    if store is False and returnas is list:
        msg = "returnas = list only of store = True"
        raise Exception(msg)

    return (
        key, gtype, configuration,
        defocus, strict_aperture,
        cam_on_e0, cam_tangential,
        store, key_cam, key_aperture, key_aperture_in,
        cam_pixels_nb, aperture_dimensions,
        focal_distance, pinhole_radius,
        returnas,
    )


def _ideal_configuration(
    coll=None,
    key=None,
    configuration=None,
    lamb=None,
    bragg=None,
    norder=None,
    defocus=None,
    strict_aperture=None,
    # parameters
    cam_on_e0=None,
    # johann-specific
    cam_tangential=None,
    # pinhole-specific
    cam_dimensions=None,
    cam_distance=None,
    focal_distance=None,
    # store
    store=None,
    key_cam=None,
    key_aperture=None,
    aperture_dimensions=None,
    pinhole_radius=None,
    cam_pixels_nb=None,
    # returnas
    returnas=None,
):

    # --------------
    # check inputs

    (
        key, gtype, configuration,
        defocus, strict_aperture,
        cam_on_e0, cam_tangential,
        store, key_cam, key_aperture, key_aperture_in,
        cam_pixels_nb, aperture_dimensions,
        focal_distance, pinhole_radius,
        returnas,
    ) = _ideal_configuration_check(
        coll=coll,
        key=key,
        configuration=configuration,
        defocus=defocus,
        strict_aperture=strict_aperture,
        # parameters
        cam_on_e0=cam_on_e0,
        # johann-specific
        cam_tangential=cam_tangential,
        # pinhole-specific
        cam_dimensions=cam_dimensions,
        focal_distance=focal_distance,
        # store
        store=store,
        key_cam=key_cam,
        key_aperture=key_aperture,
        aperture_dimensions=aperture_dimensions,
        pinhole_radius=pinhole_radius,
        cam_pixels_nb=cam_pixels_nb,
        # returnas
        returnas=returnas,
    )

    # bragg / lamb
    bragg = coll.get_crystal_bragglamb(
        key=key,
        lamb=lamb,
        bragg=bragg,
        norder=norder,
    )[0]
    if bragg.size != 1:
        msg = (
            "Please only provide a single lamb or bragg value!\n"
            f"Provided: {bragg}"
        )
        raise Exception(msg)

    # --------------
    # prepare data

    dgeom = coll.dobj['crystal'][key]['dgeom']
    extenthalf = dgeom['extenthalf']
    curve_r = dgeom['curve_r']
    cent = dgeom['cent']
    nin = dgeom['nin']
    e0 = dgeom['e0']
    e1 = dgeom['e1']

    # radius of curvature
    rc = None
    if gtype == 'spherical':
        rc = curve_r[0]
    elif gtype == 'cylindrical':
        icurv = (~np.isinf(curve_r)).nonzero()[0][0]
        rc = curve_r[icurv]

    # unit vectors
    vect_cam = np.cos(bragg) * e0 + np.sin(bragg) * nin
    vect_los = -np.cos(bragg) * e0 + np.sin(bragg) * nin
    if cam_on_e0 is False:
        vect_cam, vect_los = vect_los, vect_cam

    # ----------------------
    # compute configuration
    # ----------------------

    # --------
    # johann

    if configuration == 'johann':

        if rc < 0:
            msg = (
                f"crystal {key} is convex: Johann not possible!\n"
                f" \t- curve_r = {curve_r}"
            )
            raise Exception(msg)

        med = rc * np.sin(bragg)
        sag = -med / np.cos(2.*bragg)

        meridional = cent + med * vect_los
        sagittal = cent + sag * vect_los

        cam_cent = cent + (med + defocus) * vect_cam

        if cam_tangential is True:
            cam_nin = (cent + nin*rc/2. - cam_cent)
            cam_nin = cam_nin / np.linalg.norm(cam_nin)
        else:
            cam_nin = -vect_los

        dout = {
            'meridional': {
                'cent': meridional,
                'dist': med,
            },
            'sagittal': {
                'cent': sagittal,
                'dist': sag,
            },
        }

    # --------
    # von hamos

    elif configuration == 'von hamos':

        if rc < 0:
            msg = (
                f"crystal {key} is convex: von hamos not possible!\n"
                f" \t- curve_r = {curve_r}"
            )
            raise Exception(msg)

        dist_pin = rc / np.sin(bragg)
        pin_cent = cent + dist_pin * vect_los
        pin_nin = vect_los

        cam_cent = cent + (dist_pin + defocus) * vect_cam
        if cam_tangential is True:
            cam_nin = -nin
        else:
            cam_nin = -vect_cam

        dout = {
            'aperture': {
                'cent': pin_cent,
                'nin': pin_nin,
            },
        }

    # --------
    # pinhole

    elif configuration == 'pinhole':

        # pinhole
        if focal_distance is None:
            if gtype == 'cylindrical':
                focus = np.abs(rc) / np.sin(bragg)
            elif gtype == 'spherical':
                focus = np.abs(rc) * np.sin(bragg)
            else:
                msg = "Please provide focal_distance!"
                raise Exception(msg)

        else:
            focus = focal_distance

        pin_cent = cent + focus * vect_los
        pin_nin = vect_los

        # camera
        if cam_distance is None:
            cam_height = cam_dimensions[1]

            if gtype == 'planar' or (gtype == 'cylindrical' and icurv == 0):
                cryst_height = 2. * extenthalf[1]

                if cam_height <= cryst_height:
                    msg = (
                        f"Height for ideal camera of '{key}' too small:\n"
                        f"\t- crystal height (flat): {cryst_height}\n"
                        f"\t- camera height: {cam_height}\n"
                    )
                    raise Exception(msg)

                cam_dist = focal_distance * (cam_height / cryst_height - 1.)

            else:
                if gtype == 'cylindrical':
                    cryst_height = 2. * extenthalf[icurv] * np.abs(rc)

                elif gtype == 'spherical':
                    cryst_height = 2. * extenthalf[1] * np.abs(rc)

                if cam_height >= cryst_height:
                    msg = (
                        f"Height for ideal camera of '{key}' too large:\n"
                        f"\t- crystal height (gtype): {cryst_height}\n"
                        f"\t- camera height: {cam_height}\n"
                    )
                    raise Exception(msg)

                cam_dist = np.abs(rc) * (1. - cam_height / cryst_height)

        else:
            cam_dist = cam_distance

        cam_nin = -vect_cam
        cam_cent = cent + (cam_dist + defocus) * vect_cam

        dout = {
            'aperture': {
                'cent': pin_cent,
                'nin': pin_nin,
            },
        }

    # ----------------------------------
    # complete with missing unit vectors

    if 'aperture' in dout.keys():

        # check against existing aperture is any
        if key_aperture_in is True:

            dd = coll.dobj['aperture'][key_aperture]['dgeom']
            temp_dist = np.linalg.norm(dd['cent'] - pin_cent)
            temp_ang = np.arctan2(
                np.linalg.norm(np.cross(dd['nin'], pin_nin)),
                np.sum(dd['nin'] * pin_nin),
            )

            if not (temp_dist < 1e-6 and np.abs(temp_ang) < 0.01*np.pi/180.):
                # dist = np.linalg.norm(dd['cent'] - pin_cent)
                msg = (
                    f"Ideal configuration '{configuration}' for crystal '{key}':\n"
                    f"Predefined aperture {key_aperture} does not seem fit:\n"
                    f"\t- cent: {temp_dist} m\n"
                    f"\t\t- {key_aperture}: {dd['cent']}\n"
                    f"\t\t- ideal: {pin_cent}\n"
                    f"\t- nin: {temp_ang} deg.\n"
                    f"\t\t- {key_aperture}: {dd['nin']}\n"
                    f"\t\t- ideal: {pin_nin}\n"
                )
                if strict_aperture is True:
                    raise Exception(msg)
                else:
                    warnings.warn(msg)
            del dout['aperture']

        # new aperture
        else:
            ap_e0 = np.cross(e1, dout['aperture']['nin'])
            ap_e0 = ap_e0 / np.linalg.norm(ap_e0)
            ap_e1 = np.cross(dout['aperture']['nin'], ap_e0)
            dout['aperture']['e0'] = ap_e0
            dout['aperture']['e1'] = ap_e1

    cam_e0 = np.cross(e1, cam_nin)
    cam_e0 = cam_e0 / np.linalg.norm(cam_e0)
    cam_e1 = np.cross(cam_nin, cam_e0)

    # ---------
    # return

    dout.update({
        'camera': {
            'cent': cam_cent,
            'nin': cam_nin,
            'e0': cam_e0,
            'e1': cam_e1,
        },
    })

    # ---------
    # store

    if store is True:

        dout = _ideal_configuration_store(
            coll=coll,
            configuration=configuration,
            dout=dout,
            key_cam=key_cam,
            key_aperture=key_aperture,
            cam_dimensions=cam_dimensions,
            cam_pixels_nb=cam_pixels_nb,
            aperture_dimensions=aperture_dimensions,
            pinhole_radius=pinhole_radius,
        )

    # ---------
    # return

    if returnas is dict:
        return dout

    elif returnas is list:
        loptics = [key_cam, key]
        if key_aperture is not None:
            loptics.append(key_aperture)
        return loptics


def _ideal_configuration_store(
    coll=None,
    configuration=None,
    dout=None,
    # store
    key_cam=None,
    key_aperture=None,
    cam_dimensions=None,
    cam_pixels_nb=None,
    aperture_dimensions=None,
    pinhole_radius=None,
):

    # -------
    # camera

    # pixels dimensions, outline and cents
    dim0, dim1 = cam_dimensions
    nx0, nx1 = cam_pixels_nb

    dx0 = dim0 / nx0
    dx1 = dim1 / nx1

    outline_x0 = 0.5 * dx0 * np.r_[-1., 1., 1., -1.]
    outline_x1 = 0.5 * dx1 * np.r_[-1., -1., 1., 1.]

    cents_x0 = 0.5 * dim0 * np.linspace(-1., 1., nx0 + 1)
    cents_x1 = 0.5 * dim1 * np.linspace(-1., 1., nx1 + 1)
    cents_x0 = 0.5 * (cents_x0[:-1] + cents_x0[1:])
    cents_x1 = 0.5 * (cents_x1[:-1] + cents_x1[1:])

    # complement camera
    dout['camera'].update({
        'outline_x0': outline_x0,
        'outline_x1': outline_x1,
        'cents_x0': cents_x0,
        'cents_x1': cents_x1,
    })

    # add camera
    coll.add_camera_2d(
        key=key_cam,
        dgeom=dout['camera'],
    )

    # -------
    # aperture

    if 'aperture' in dout.keys():

        if pinhole_radius is not None:
            theta = np.pi * np.linspace(-1, 1, 50)[:-1]
            outline_x0 = pinhole_radius * np.cos(theta)
            outline_x1 = pinhole_radius * np.sin(theta)
        else:
            # outline
            lx0, lx1 = aperture_dimensions
            outline_x0 = 0.5 * lx0 * np.r_[-1., 1., 1., -1.]
            outline_x1 = 0.5 * lx1 * np.r_[-1., -1., 1., 1.]

        # complement dict
        dout['aperture'].update({
            'outline_x0': outline_x0,
            'outline_x1': outline_x1,
        })

        # add aperture
        coll.add_aperture(
            key=key_aperture,
            **dout['aperture'],
        )

    return dout