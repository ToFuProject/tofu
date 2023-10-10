# Built-in
# import warnings

# Common
import numpy as np
# import scipy.interpolate as scpinterp
# import scipy.integrate as scpintg
# from inspect import signature as insp
import matplotlib.pyplot as plt
import datastock as ds


import datetime as dtm


# local
from . import _GG


_APPROX = True
_ANISO = False
_BLOCK = True
_LTYPES = [int, float, np.int_, np.float_]


__all__ = [
    'calc_solidangle_particle',
    'calc_solidangle_apertures',
]


###############################################################################
###############################################################################
#                       Check inputs - Particle
###############################################################################


def _check_calc_solidangle_particle(
    traj=None,
    pts=None,
    rad=None,
    config=None,
    approx=None,
    aniso=None,
    block=None,
):

    # Check booleans
    if approx is None:
        approx = _APPROX
    if aniso is None:
        aniso = _ANISO

    lbool = [('approx', approx), ('aniso', aniso)]
    for kk, vv in lbool:
        if not isinstance(vv, bool):
            msg = ("Arg {} must be a bool\n".format(kk)
                   + "\t- provided: {}".format(vv))
            raise Exception(msg)

    # Check config
    c0 = [config is None, config.__class__.__name__ == "Config"]
    if not any(c0):
        msg = ("Arg config must be either Noen or a tf.geom.Config instance!\n"
               + "\t- provided: {}".format(config))
        raise Exception(msg)

    # Double-check block vs config
    if block is None:
        if config is None:
            block = False
        else:
            block = _BLOCK
    if not isinstance(block, bool):
        msg = ("Arg {} must be a bool\n".format('block')
               + "\t- provided: {}".format(block))
        raise Exception(msg)
    if config is None and block is True:
        msg = ("Arg block cannot be True of config is not provided!")
        raise Exception(msg)

    # arrays
    try:
        traj = np.ascontiguousarray(traj, dtype=float)
        rad = np.r_[rad].astype(float).ravel()

        # Check pts, traj and r are array of good shape
        c0 = traj.ndim in [1, 2] and 3 in traj.shape
        if pts is not False:
            pts = np.ascontiguousarray(pts, dtype=float)
            c0 = c0 and pts.ndim in [1, 2] and 3 in pts.shape
        assert c0
        if traj.ndim == 1:
            traj = traj.reshape((3, 1))
        if traj.shape[0] != 3:
            traj = traj.T
        traj = np.ascontiguousarray(traj)
        if pts is not False:
            if pts.ndim == 1:
                pts = pts.reshape((3, 1))
            if pts.shape[0] != 3:
                pts = pts.T
            pts = np.ascontiguousarray(pts)
    except Exception:
        msg = (
            "Args traj and pts must be convertible to np.ndarrays of shape"
            + "\n\t- traj: (N,), (3, N) or (N, 3)"
            + "\n\t- pts: (M,), (3, M) or (M, 3)"
            + "\n\n   You provided:\n"
            + "\n\t- traj: {}".format(traj)
            + "\n\t- pts: {}".format(pts)
            + "\n\t- rad: {}".format(rad)
        )
        raise Exception(msg)

    # check rad vs traj
    ntraj = traj.shape[1]
    nrad = rad.size

    nmax = max(nrad, ntraj)
    if not (nrad in [1, nmax] and ntraj in [1, nmax]):
        msg = ("rad must be an array with shape (1,) or (N,)\n"
               + "  provided: {}".format(rad))
        raise Exception(msg)
    if nrad < nmax:
        rad = np.full((nmax,), rad[0])
    if ntraj < nmax:
        traj = np.repeat(traj, nmax, axis=1)
    return traj, pts, rad, config, approx, aniso, block


###############################################################################
###############################################################################
#                       Solid Angle particle
###############################################################################


def calc_solidangle_particle(
    pts=None,
    part_traj=None,
    part_radius=None,
    config=None,
    approx=None,
    aniso=None,
    block=None,
):
    """ Compute the solid angle subtended by a particle along a trajectory

    The particle has radius r, and trajectory (array of points) traj
    It is observed from pts (array of points)

    traj and pts are (3, N) and (3, M) arrays of cartesian coordinates

    approx = True => use approximation
    aniso = True => return also unit vector of emission
    block = True consider LOS collisions (with Ves, Struct...)

    if block:
        config used for LOS collisions

    Parameters
    ----------
    pts:            np.ndarray
        Array of (3, M) pts coordinates (X, Y, Z) representing the points from
        which the particle is observed
    part_traj:      np.ndarray
        Array of (3, N) pts coordinates (X, Y, Z) representing the particle
        positions
    part_radius:    float / np.ndarray
        Unique of multiple values for the radius of the spherical particle
            if multiple, rad is a np.ndarray of shape (N,)
    config:         None / tf.geom.Config
        if block = True, solid angles are non-zero only if the field of view is
        not blocked bya structural element in teh chamber
    approx:         None / bool
        Flag indicating whether to compute the solid angle using an 1st-order
        series development (in whichcase the solid angle becomes proportional
        to the radius of the particle, see Notes_Upgrades/)
    aniso:          None / bool
        Flag indicating whether to consider anisotropic emissivity, meaning the
        routine must also compute and return the unit vector directing the flux
        from each pts to each position on the trajectory of the particle
    block:          None / bool
        Flag indicating whether to check for vignetting by structural elements
        provided by config

    Return:
    -------
    sang:           np.ndarray
        (N, M) Array of floats, solid angles

    """
    ################
    # Prepare inputs
    (
        part_traj, pts, part_radius, config,
        approx, aniso, block
    ) = _check_calc_solidangle_particle(
        traj=part_traj,
        pts=pts,
        rad=part_radius,
        config=config,
        approx=approx,
        aniso=aniso,
        block=block,
    )

    ################
    # Main computation

    # traj2pts vector, with length (3d array (3, N, M))
    vect = - pts[:, :, None] + part_traj[:, None, :]
    len_v = np.ascontiguousarray(np.sqrt(np.sum(vect**2, axis=0)))

    # If aniso or block, normalize
    if aniso or block:
        vect = vect / len_v[None, :, :]

    # Solid angle
    r_d = part_radius[None, :] / len_v
    where_zero = len_v <= part_radius[None, :]
    r_d[where_zero] = 0.  # temporary value
    if approx:
        sang = np.pi * (r_d**2 + r_d**4 / 4. + r_d**6 / 8. + r_d**8 * 5 / 64)
    else:
        sang = 2.*np.pi * (1 - np.sqrt(1. - r_d ** 2))

    # when particle in mesh point, distance len_v = 0 thus sang neglected
    sang[where_zero] = 0.

    # block
    if block:
        kwdargs = config.get_kwdargs_LOS_isVis()
        indvis = _GG.LOS_areVis_PtsFromPts_VesStruct(
            pts, part_traj, dist=len_v, **kwdargs
        )
        iout = indvis == 0
        sang[iout] = 0.
        vect[:, iout] = np.nan

    ################
    # Return
    if aniso:
        return sang, vect

    return sang


def calc_solidangle_particle_integ(
    part_traj=None,
    part_radius=None,
    config=None,
    approx=True,
    block=True,
    resolution=None,
    DR=None,
    DZ=None,
    DPhi=None,
):

    # step0: if block : generate kwdargs from config

    # step 1: sample cross-section

    # step 2: loop on R of  pts of cross-section (parallelize ?)
    # => fix nb. of phi for the rest of the loop

    # loop of Z

    # step 3: loop phi
    # Check visibility (if block = True) for each phi (LOS collision)
    # If visible => compute solid angle
    # integrate (sum * res) on each phi the solid angle

    # Return sang as (N,nR,nZ) array

    # ----------------
    # check resolution

    if resolution is None:
        resolution = 0.1
    if type(resolution) in _LTYPES:
        resolution = [resolution, resolution, resolution]
    c0 = (
        isinstance(resolution, list)
        and all([type(ss) in _LTYPES for ss in resolution])
    )
    if not c0:
        msg = (
            "Arg resolution must be a list of 3 floats [r, z, rphi]\n"
            "Each representing the spatial sampling step in a direction\n"
            "If a single float is provided, the same is used for all"
        )
        raise Exception(msg)
    resolution = [float(rr) for rr in resolution]

    # ------------------
    # Check DR, DZ, DPhi
    dD = {'DR': DR, 'DZ': DZ, 'DPhi': DPhi}
    dfail = {}
    for k0, v0 in dD.items():
        c0 = (
            v0 is None
            or (
                isinstance(v0, list)
                and len(v0) == 2
                and all([v1 is None or type(v1) in _LTYPES for v1 in v0])
            )
        )
        if not c0:
            dfail[k0] = str(v0)
    if len(dfail) > 0:
        lstr = [f'\t- {k0}: {v0}' for k0, v0 in dfail.items()]
        msg = (
            "The following arguments are invalid:\n"
            "Expected None or a list of len(2) of None or floats!\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # ------------------
    # check other inputs

    (
        part_traj, _, part_radius, config,
        approx, _, block
    ) = _check_calc_solidangle_particle(
        traj=part_traj,
        pts=False,
        rad=part_radius,
        config=config,
        approx=approx,
        aniso=False,
        block=block,
    )

    # ------------------
    # Define the volume to be sampled: smallest vessel

    # Get kwdargs for LOS blocking
    kwdargs = config.get_kwdargs_LOS_isVis()

    # derive limits for sampling
    limits_r = np.r_[
        np.min(kwdargs['ves_poly'][0, :]),
        np.max(kwdargs['ves_poly'][0, :]),
    ]
    limits_z = np.r_[
        np.min(kwdargs['ves_poly'][1, :]),
        np.max(kwdargs['ves_poly'][1, :]),
    ]

    return _GG.compute_solid_angle_map(
        part_traj, part_radius,
        resolution[0], resolution[1], resolution[2],
        limits_r, limits_z,
        DR=DR, DZ=DZ,
        DPhi=DPhi,
        block=block,
        approx=approx,
        limit_vpoly=kwdargs['ves_poly'],
        **kwdargs,
    )


###############################################################################
###############################################################################
#           Triangulation
###############################################################################


def triangulate_polygon_2d(
    poly_x=None,
    poly_y=None,
):

    # ----------
    # check

    if not (isinstance(poly_x, np.ndarray) and isinstance(poly_x, np.ndarray)):
        msg = "both poly_x and poly_y must be nd.ndarray"
        raise Exception(msg)

    if not (poly_x.shape == poly_y.shape and poly_x.ndim == 1):
        msg = "poly_x and poly_y must be 1d arrays of the same shape"
        raise Exception(msg)

    poly_x = poly_x.astype(float)
    poly_y = poly_y.astype(float)

    # ----------
    # un-close

    if poly_x[-1] == poly_x[0] and poly_y[-1] == poly_y[0]:
        poly_x = poly_x[:-1]
        poly_y = poly_y[:-1]

    # ------------------
    # couter-clockwise

    clock = np.sum((poly_x[1:] - poly_x[:-1]) * (poly_y[:1] + poly_y[:-1]))
    if clock > 0:
        poly_x = poly_x[::-1]
        poly_y = poly_y[::-1]

    # ----------------
    # triangulate

    tri = _GG.triangulate_by_earclipping_2d(np.array([poly_x, poly_y]))

    # -------
    # format

    if clock > 0:
        tri = poly_x.size - 1 - tri

    return tri


###############################################################################
###############################################################################
#           Check inputs - arbitrary points, multiple apertures
###############################################################################


def _check_pts(pts=None, pts_name=None):

    if not isinstance(pts, np.ndarray):
        try:
            pts = np.atleast_1d(pts)
        except Exception as err:
            msg = (
                f"Arg {pts_name} must be convertible to a np.ndarray float\n"
                "Provided: {pts}"
            )
            raise Exception(msg)
    return pts.astype(float)


def _check_polygon_2d_area_ccw(
    poly_x0=None,
    poly_x1=None,
):
    """ Assumes non-closed polygon """

    assert not (poly_x0[0] == poly_x0[-1] and poly_x1[0] == poly_x1[-1])
    i0 = np.arange(0, poly_x0.size)
    i1 = np.r_[np.arange(1, poly_x0.size), 0]
    return 0.5*np.sum(
        (poly_x0[i1] + poly_x0[i0]) * (poly_x1[i1] - poly_x1[i0])
    )


def _check_polygon_3d_counter_clockwise(
    poly_x=None,
    poly_y=None,
    poly_z=None,
    normal=None,
):
    """ Assumes non-closed polygon """

    cent = np.r_[np.mean(poly_x), np.mean(poly_y), np.mean(poly_z)]
    normal = normal / np.linalg.norm(normal)
    if np.abs(normal[2]) < 0.99:
        e0 = np.r_[-normal[1], normal[0], 0.]
    else:
        e0 = np.r_[-normal[2], 0., normal[0]]
    e0 = e0 / np.linalg.norm(e0)
    e1 = np.cross(normal, e0)

    # project in 2d
    px0 = (
        (poly_x - cent[0]) * e0[0]
        + (poly_y - cent[1]) * e0[1]
        + (poly_z - cent[2]) * e0[2]
    )

    px1 = (
        (poly_x - cent[0]) * e1[0]
        + (poly_y - cent[1]) * e1[1]
        + (poly_z - cent[2]) * e1[2]
    )

    # counter-clockwise?
    return _check_polygon_2d_area_ccw(
        poly_x0=px0,
        poly_x1=px1,
    ) > 0


def _check_polygon_2d(
    poly_x=None,
    poly_y=None,
    poly_name=None,
    can_be_None=None,
    closed=None,
    counter_clockwise=None,
    return_area=None,
):

    # -------------
    # check inputs

    # closed
    closed = ds._generic_check._check_var(
        closed, 'closed',
        types=bool,
        default=True,
    )

    # counter_clockwise
    counter_clockwise = ds._generic_check._check_var(
        counter_clockwise, 'counter_clockwise',
        types=bool,
        default=True,
    )

    # can_be_None
    can_be_None = ds._generic_check._check_var(
        can_be_None, 'can_be_None',
        types=bool,
        default=False,
    )

    # return_area
    return_area = ds._generic_check._check_var(
        return_area, 'return_area',
        types=bool,
        default=False,
    )

    # -------------
    # Trivial case

    if can_be_None and poly_x is None:
        return None, None

    # ------------------------
    # check each is a 1d array

    if isinstance(poly_x, (list, tuple)):
        poly_x = np.atleast_1d(poly_x)
        poly_y = np.atleast_1d(poly_y)

    c0 = (
        all([
            isinstance(pp, np.ndarray) for pp in [poly_x, poly_y]
        ])
        and poly_x.ndim == poly_y.ndim == 1
        and poly_x.shape == poly_y.shape
    )
    if not c0:
        msg = (
            f"Arg {poly_name} must be 2 (npts,) arrays, where\n"
            "\t- poly_x, poly_y are (X, Y) coordinates\n"
            "\t- npts is the number of vertices\n"
            f"Provided:\n{poly_x, poly_y}"
        )
        raise Exception(msg)

    # ------------------------
    # make sure not closed for ccw test

    pt0 = np.r_[poly_x[0], poly_y[0]]
    pt1 = np.r_[poly_x[-1], poly_y[-1]]
    if np.allclose(pt0, pt1):
        poly_x = poly_x[:-1]
        poly_y = poly_y[:-1]

    # ------------------------
    # make sure float

    poly_x = poly_x.astype(float)
    poly_y = poly_y.astype(float)

    # ------------------------
    # check counter-clockwise ass seen from normal vector

    area_ccw = _check_polygon_2d_area_ccw(
        poly_x0=poly_x,
        poly_x1=poly_y,
    )
    if counter_clockwise != (area_ccw > 0):
        poly_x = np.ascontiguousarray(poly_x[::-1])
        poly_y = np.ascontiguousarray(poly_y[::-1])

    # ------------------------
    # check if closed vs close

    if closed:
        poly_x = np.r_[poly_x, poly_x[0]]
        poly_y = np.r_[poly_y, poly_y[0]]

    # --------------
    # return

    if return_area:
        return poly_x, poly_y, np.abs(area_ccw)
    else:
        return poly_x, poly_y


def _check_polygon_3d(
    poly_x=None,
    poly_y=None,
    poly_z=None,
    poly_name=None,
    can_be_None=None,
    closed=None,
    counter_clockwise=None,
    normal=None,
):

    # -------------
    # check inputs

    # closed
    closed = ds._generic_check._check_var(
        closed, 'closed',
        types=bool,
        default=True,
    )

    # counter_clockwise
    counter_clockwise = ds._generic_check._check_var(
        counter_clockwise, 'counter_clockwise',
        types=bool,
        default=True,
    )

    # can_be_None
    can_be_None = ds._generic_check._check_var(
        can_be_None, 'can_be_None',
        types=bool,
        default=False,
    )

    # -------------
    # Trivial case

    if can_be_None and poly_x is None:
        return None, None, None

    # ------------------------
    # check each is a 1d array

    if isinstance(poly_x, (list, tuple)):
        poly_x = np.atleast_1d(poly_x)
        poly_y = np.atleast_1d(poly_y)
        poly_z = np.atleast_1d(poly_z)

    c0 = (
        all([
            isinstance(pp, np.ndarray) for pp in [poly_x, poly_y, poly_z]
        ])
        and poly_x.ndim == poly_y.ndim == poly_z.ndim == 1
        and poly_x.shape == poly_y.shape == poly_z.shape
    )
    if not c0:
        msg = (
            f"Arg {poly_name} must be 3 (npts,) arrays, where\n"
            "\t- poly_x, poly_y, poly_z are (X, Y, Z) coordinates\n"
            "\t- npts is the number of vertices\n"
            f"Provided:\n{poly_x, poly_y, poly_z}"
        )
        raise Exception(msg)

    # ------------------------
    # make sure not closed for ccw test

    pt0 = np.r_[poly_x[0], poly_y[0], poly_z[0]]
    pt1 = np.r_[poly_x[-1], poly_y[-1], poly_z[-1]]
    if np.allclose(pt0, pt1):
        poly_x = poly_x[:-1]
        poly_y = poly_y[:-1]
        poly_z = poly_z[:-1]

    # ------------------------
    # make sure float

    poly_x = poly_x.astype(float)
    poly_y = poly_y.astype(float)
    poly_z = poly_z.astype(float)

    # ------------------------
    # check counter-clockwise ass seen from normal vector

    ccw = _check_polygon_3d_counter_clockwise(
        poly_x=poly_x,
        poly_y=poly_y,
        poly_z=poly_z,
        normal=normal,
    )
    if counter_clockwise != ccw:
        poly_x = poly_x[::-1]
        poly_y = poly_y[::-1]
        poly_z = poly_z[::-1]

    # ------------------------
    # check if closed vs close

    if closed:
        poly_x = np.r_[poly_x, poly_x[0]]
        poly_y = np.r_[poly_y, poly_y[0]]
        poly_z = np.r_[poly_z, poly_z[0]]

    return poly_x, poly_y, poly_z


def _check_unit_vectors(det=None):
    # check normalization
    dnorm = {
        k0: np.sqrt(det[f'{k0}_x']**2 + det[f'{k0}_y']**2 + det[f'{k0}_z']**2)
        for k0 in ['e0', 'e1', 'nin']
    }
    if not np.allclose(list(dnorm.values()), 1):
        lstr = [f"\t- {k0} = {v0}" for k0, v0 in dnorm.items()]
        msg = (
            "All unit vectors ['e0', 'e1', 'nin'] must be normalized!\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # check perpendicularity
    dsca = {
        f'{v0}.{v1}': (
            det[f'{v0}_x']*det[f'{v1}_x']
            + det[f'{v0}_y']*det[f'{v1}_y']
            + det[f'{v0}_z']*det[f'{v1}_z']
        )
        for v0, v1 in [('e0', 'e1'), ('e0', 'nin'), ('e1', 'nin')]
    }
    if not np.allclose(list(dsca.values()), 0):
        lstr = [f"\t- {k0} = {v0}" for k0, v0 in dsca.items()]
        msg = (
            "All unit vectors ['e0', 'e1', 'nin'] must be perpendicular!\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)


def _check_det_dict(detectors=None):

    lk_out = ['outline_x0', 'outline_x1']
    lk_shape = [
        'cents_x', 'cents_y', 'cents_z',
        'e0_x', 'e0_y', 'e0_z',
        'e1_x', 'e1_y', 'e1_z',
        'nin_x', 'nin_y', 'nin_z',
    ]

    # make sure array + float
    for k0 in set(lk_shape).intersection(detectors.keys()):
        detectors[k0] = np.atleast_1d(detectors[k0]).astype(float)

    # check shapes
    c0 = (
        all([
            isinstance(detectors.get(ss), np.ndarray)
            for ss in lk_out + lk_shape
        ])
        and all([
            detectors[ss].shape == detectors['outline_x0'].shape
            for ss in lk_out
        ])
        and all([
            detectors[ss].shape == detectors['cents_x'].shape
            for ss in lk_shape
        ])
    )
    if not c0:
        lmis = [kk for kk in lk_out + lk_shape if kk not in detectors.keys()]
        msg = (
            "Arg detectors, if a dict, must contain the following keys:\n"
            + f"\t- missing keys: {lmis}\n"
            "And the following keys should be array of identical shape:\n"
            + "\n".join([f"\t- {kk}" for kk in lk_out])
            + "\nAnd the following keys should be array of identical shape:\n"
            + "\n".join([f"\t- {kk}" for kk in lk_shape])
            + f"\nProvided:\n{detectors}"
        )
        raise Exception(msg)

    # check outline
    detectors['outline_x0'], detectors['outline_x1'] = _check_polygon_2d(
        poly_x=detectors['outline_x0'],
        poly_y=detectors['outline_x1'],
        poly_name='det',
        can_be_None=False,
        closed=False,
        counter_clockwise=True,
    )

    # check unit vectors (normalize + perpendicular)
    _check_unit_vectors(detectors)

    # copy dict and flatten (if not 1d)
    # if detectors['cents_x'].ndim > 1:
        # detectors = dict(detectors)
        # lk = [k0 for k0 in detectors.keys() if 'outline' not in k0]
        # for k0 in lk:
            # detectors[k0] = detectors[k0].ravel()

    return detectors


def _check_ap_dict(apertures=None):

    lk0 = ['poly_x', 'poly_y', 'poly_z']
    lk1 = ['nin']

    err = False
    if not isinstance(apertures, dict):
        err = True
    else:
        if all([k0 in apertures.keys() for k0 in lk0 + lk1]):
            apertures = {'ap0': apertures}

        lkout = [
            k0 for k0, v0 in apertures.items()
            if not (
                all([k1 in v0.keys() for k1 in lk0 + lk1])
                and all([
                    isinstance(v0[k1], np.ndarray)
                    and v0['poly_x'].shape == v0[k1].shape
                    for k1 in lk0
                ])
                and all([v0[k1].shape == (3,) for k1 in lk1])
                and np.sqrt(np.sum([v0[k1]**2 for k1 in lk1])) == 1.
            )
        ]
        if len(lkout) > 1:
            err = True

    if err:
        msg = (
            "Arg apertures must be a dict with keys:\n"
            "\t- 'nin': normal vector oriented towards the plasma\n"
            "\t- 'poly_x', 'poly_y', 'poly_z': 3d (x, y, z) polygon\n"
            "        must be counter-clockwise from 'nin'"
        )
        raise Exception(msg)

    # make sure float
    for k0, v0 in apertures.items():
        for k1 in lk0 + lk1:
            apertures[k0][k1] = np.atleast_1d(v0[k1]).ravel().astype(float)

    # check polygons
    for k0, v0 in apertures.items():
        (
            apertures[k0]['poly_x'],
            apertures[k0]['poly_y'],
            apertures[k0]['poly_z'],
        ) = _check_polygon_3d(
            poly_x=v0['poly_x'],
            poly_y=v0['poly_y'],
            poly_z=v0['poly_z'],
            poly_name=k0,
            can_be_None=False,
            closed=False,
            counter_clockwise=True,
            normal=v0['nin'],
        )

    return apertures


def _calc_solidangle_apertures_check(
    # observation points
    pts_x=None,
    pts_y=None,
    pts_z=None,
    # polygons
    apertures=None,
    detectors=None,
    # possible obstacles
    config=None,
    # parameters
    summed=None,
    visibility=None,
    return_vector=None,
    # output formatting
    return_flat_pts=None,
    return_flat_det=None,
    # options
    timing=None,
):

    # ---------------
    # pts coordinates

    pts_x = _check_pts(pts=pts_x)
    pts_y = _check_pts(pts=pts_y)
    pts_z = _check_pts(pts=pts_z)

    if not (pts_x.shape == pts_y.shape == pts_z.shape):
        msg = (
            "Arg pts_x, pts_y and pts_z must share the same shape!\n"
            f"\t- pts_x.shape = {pts_x.shape}\n"
            f"\t- pts_y.shape = {pts_y.shape}\n"
            f"\t- pts_z.shape = {ptszz.shape}\n"
        )
        raise Exception(msg)

    mask = np.isfinite(pts_x) & np.isfinite(pts_y) & np.isfinite(pts_z)
    if np.all(mask):
        mask = None

    # ---------
    # detectors

    detectors = _check_det_dict(detectors)

    # ---------
    # apertures

    if apertures is not None:
        apertures = _check_ap_dict(apertures)

    # ----------
    # summed

    summed = ds._generic_check._check_var(
        summed, 'summed',
        types=bool,
        default=False,
    )

    # ----------
    # visibility

    visibility = ds._generic_check._check_var(
        visibility, 'visibility',
        types=bool,
        default=False,
    )

    # -------------
    # return_vector

    return_vector = ds._generic_check._check_var(
        return_vector, 'return_vector',
        types=bool,
        default=False,
    )

    # -------------
    # timing

    timing = ds._generic_check._check_var(
        timing, 'timing',
        types=bool,
        default=False,
    )

    # -------------
    # compatibility

    if summed is True:
        if (visibility or return_vector or apertures is None):
            msg = (
                "Arg summed = True only usable with (so far):\n"
                "\t- visibility = False\n"
                "\t- return_vector = False\n"
                "\t- aperture != None"
            )
            raise NotImplementedError(msg)

    # ------
    # config

    if visibility:
        if not config.__class__.__name__ == 'Config':
            msg = (
                "If visibility, config must be provided (Config instance)\n"
                f"Provided: {config}"
            )
            raise Exception(msg)

    # ----------
    # check aperture vs visibility

    if apertures is None and (visibility is True or return_vector is True):
        msg = (
            "No apertures provided!\n"
            "=> visibility must be False\n"
            "=> return_vector must be False\n"
        )
        raise Exception(msg)

    return (
        pts_x, pts_y, pts_z, mask,
        apertures, detectors,
        summed, visibility,
        return_vector, timing,
    )


###############################################################################
###############################################################################
#           Prepare data - arbitrary points, multiple apertures
###############################################################################


def _calc_solidangle_apertures_prepare(
    # observation points
    pts_x=None,
    pts_y=None,
    pts_z=None,
    mask=None,
    # polygons
    apertures=None,
    detectors=None,
    # possible obstacles
    config=None,
):

    # -----------------------------
    # pts as 1d C-contiguous arrays

    ndim0 = pts_x.ndim
    shape0 = pts_x.shape

    if mask is not None:
        pts_x = pts_x[mask]
        pts_y = pts_y[mask]
        pts_z = pts_z[mask]

    if ndim0 > 1:
        pts_x = pts_x.ravel()
        pts_y = pts_y.ravel()
        pts_z = pts_z.ravel()

    # ---------
    # apertures

    if apertures is None:
        ap_ind, ap_x, ap_y, ap_z = None, None, None, None
        ap_nin_x, ap_nin_y, ap_nin_z = None, None, None
    else:
        lka = list(apertures.keys())
        ap_ind = np.r_[
            0, np.cumsum([apertures[k0]['poly_x'].size for k0 in lka])
        ]
        ap_x = np.concatenate([apertures[k0]['poly_x'] for k0 in lka])
        ap_y = np.concatenate([apertures[k0]['poly_y'] for k0 in lka])
        ap_z = np.concatenate([apertures[k0]['poly_z'] for k0 in lka])
        ap_nin_x = np.array([apertures[k0]['nin'][0] for k0 in lka])
        ap_nin_y = np.array([apertures[k0]['nin'][1] for k0 in lka])
        ap_nin_z = np.array([apertures[k0]['nin'][2] for k0 in lka])

    # ---------
    # detectors

    det_shape0 = detectors['cents_x'].shape

    det_outline_x0 = detectors['outline_x0']
    det_outline_x1 = detectors['outline_x1']
    det_cents_x = detectors['cents_x'].ravel()
    det_cents_y = detectors['cents_y'].ravel()
    det_cents_z = detectors['cents_z'].ravel()
    det_nin_x = detectors['nin_x'].ravel()
    det_nin_y = detectors['nin_y'].ravel()
    det_nin_z = detectors['nin_z'].ravel()
    det_e0_x = detectors['e0_x'].ravel()
    det_e0_y = detectors['e0_y'].ravel()
    det_e0_z = detectors['e0_z'].ravel()
    det_e1_x = detectors['e1_x'].ravel()
    det_e1_y = detectors['e1_y'].ravel()
    det_e1_z = detectors['e1_z'].ravel()

    return (
        ndim0, shape0, mask,
        pts_x, pts_y, pts_z,
        ap_ind, ap_x, ap_y, ap_z,
        ap_nin_x, ap_nin_y, ap_nin_z,
        det_shape0, det_outline_x0, det_outline_x1,
        det_cents_x, det_cents_y, det_cents_z,
        det_nin_x, det_nin_y, det_nin_z,
        det_e0_x, det_e0_y, det_e0_z,
        det_e1_x, det_e1_y, det_e1_z,
    )


def _visibility_unit_vectors(
    # points
    pts_x=None,
    pts_y=None,
    pts_z=None,
    # det
    det_cents_x=None,
    det_cents_y=None,
    det_cents_z=None,
    det_nin_x=None,
    det_nin_y=None,
    det_nin_z=None,
    # results
    solid_angle=None,
    unit_vector_x=None,
    unit_vector_y=None,
    unit_vector_z=None,
    **kwdargs,
):
    """

    """

    # compute pts on det surfaces
    # re-use pre-existing code (TBF)
    for ii in range(pts_x.size):

        iok = solid_angle[:, ii] > 0
        if not np.any(iok):
            continue

        # Compute point P such that:
        #   MP = kk * unit_vect
        #   CP.norm = 0
        #   => kk = (MC.norm) / (unit_vect.norm)

        # un = unit_vect.norm
        un = (
            unit_vector_x[iok, ii]*det_nin_x[iok]
            + unit_vector_y[iok, ii]*det_nin_y[iok]
            + unit_vector_z[iok, ii]*det_nin_z[iok]
        )

        # MC = point to centers
        MCx = det_cents_x[iok] - pts_x[ii]
        MCy = det_cents_y[iok] - pts_y[ii]
        MCz = det_cents_z[iok] - pts_z[ii]

        # MCn = MC.norm
        MCn = (
            MCx*det_nin_x[iok]
            + MCy*det_nin_y[iok]
            + MCz*det_nin_z[iok]
        )

        # kk = (MC.norm) / (unit_vect.norm)
        kk = MCn / un

        # P = M + kk * unit_vect
        Px = pts_x[ii] + kk * unit_vector_x[iok, ii]
        Py = pts_y[ii] + kk * unit_vector_y[iok, ii]
        Pz = pts_z[ii] + kk * unit_vector_z[iok, ii]

        # Estimate visibility
        vis = _GG.LOS_isVis_PtFromPts_VesStruct(
            Px, Py, Pz,
            np.array([[pts_x[ii]], [pts_y[ii]], [pts_z[ii]]]),
            dist=kk,
            **kwdargs,
        )

        # Set non-visible to 0 / nan
        iokn = iok.nonzero()[0]
        iout = iokn[vis == 0]
        solid_angle[iout, ii] = 0.
        unit_vector_x[iout, ii] = np.nan
        unit_vector_y[iout, ii] = np.nan
        unit_vector_z[iout, ii] = np.nan


###############################################################################
###############################################################################
#                           Main entry
#       arbitrary points, multiple detectors, multiple apertures
###############################################################################


def calc_solidangle_apertures(
    # observation points
    pts_x=None,
    pts_y=None,
    pts_z=None,
    # polygons
    apertures=None,
    detectors=None,
    # possible obstacles
    config=None,
    # parameters
    summed=None,
    visibility=None,
    return_vector=None,
    return_flat_pts=None,
    return_flat_det=None,
    timing=None,
):
    """ Return the solid angle subtended by na apertures and nd detectors

    Uses non-closed polygons

    See the following issue for details on the implementation:
        https://github.com/ToFuProject/tofu/issues/653

    The observation points are:
        - defined in (X, Y, Z) coordinates using arrays pts_x, pts_y, pts_z
        - They should have the same shape (shape0)

    The apertures are defined as a list of closed 3d polygons:
        - defined in (X, Y, Z) coordinates

    The detectors are defined another list of closed 3d polygon:
        - defined in (X, Y, Z) coordinates

    Alternatively, detectors can also be provided as:
        - being planar and sharing the same 2d outline
        - using a dict with keys:
            - 'outline_x0': 1d array of (ncorners,) coordinates
            - 'outline_x1': 1d array of (ncorners,) coordinates
            - 'centers_x': detector's center position x as (nd,) array
            - 'centers_y': detector's center position y as (nd,) array
            - 'centers_z': detector's center position z as (nd,) array
            - 'nin_x': normal unit vector x as (nd,) array
            - 'nin_y': normal unit vector y as (nd,) array
            - 'nin_z': normal unit vector z as (nd,) array
            - 'e1_x': x0 unit vector x as (nd,) array
            - 'e1_y': x0 unit vector y as (nd,) array
            - 'e1_z': x0 unit vector z as (nd,) array
            - 'e2_x': x1 unit vector x as (nd,) array
            - 'e2_y': x1 unit vector y as (nd,) array
            - 'e2_z': x1 unit vector z as (nd,) array

    Config is needed if visibility = True to check for obstacles (ray-tracing)
    It is a tofu Config class

    Return
    ----------
    solid_angle:        np.ndarray of shape (nd, shape0)
        The solid angles
            computed for each point / det pair
            considering all apertures
    unit_vector_x:      np.ndarray of shape (nd, shape0)  (optional)
        The local unit vectors x coordinates
    unit_vector_y:      np.ndarray of shape (nd, shape0) (optional)
        The local unit vectors y coordinates
    unit_vector_z:      np.ndarray of shape (nd, shape0)  (optional)
        The local unit vectors z coordinates

    """

    # --------------------------------------
    # check inputs (robust vs user mistakes)

    if timing:
        t0 = dtm.datetime.now()     # DB

    (
        # observation points
        pts_x,
        pts_y,
        pts_z,
        mask,
        # polygons
        apertures,
        detectors,
        # parameters
        summed,
        visibility,
        return_vector,
        timing,
    ) = _calc_solidangle_apertures_check(
        # observation points
        pts_x=pts_x,
        pts_y=pts_y,
        pts_z=pts_z,
        # polygons
        apertures=apertures,
        detectors=detectors,
        # possible obstacles
        config=config,
        # parameters
        summed=summed,
        visibility=visibility,
        return_vector=return_vector,
        # output formatting
        return_flat_pts=return_flat_pts,
        return_flat_det=return_flat_det,
        # options
        timing=timing,
    )

    # ----------------
    # pre-format input

    (
        ndim0, shape0, mask,
        pts_x, pts_y, pts_z,
        ap_ind, ap_x, ap_y, ap_z,
        ap_nin_x, ap_nin_y, ap_nin_z,
        det_shape0, det_outline_x0, det_outline_x1,
        det_cents_x, det_cents_y, det_cents_z,
        det_nin_x, det_nin_y, det_nin_z,
        det_e0_x, det_e0_y, det_e0_z,
        det_e1_x, det_e1_y, det_e1_z,
    ) = _calc_solidangle_apertures_prepare(
        # observation points
        pts_x=pts_x,
        pts_y=pts_y,
        pts_z=pts_z,
        mask=mask,
        # polygons
        apertures=apertures,
        detectors=detectors,
    )

    nd = det_cents_x.size

    # Get kwdargs for LOS blocking
    if config is not None:
        kwdargs = {
            k0: v0 for k0, v0 in config.get_kwdargs_LOS_isVis().items()
            if 'eps' not in k0
            and k0 not in ['ves_type', 'test', 'forbid', 'k']
        }

    if timing:
        t1 = dtm.datetime.now()     # DB
        dt1 = (t1 - t0).total_seconds()

    # ------------------------------------------------
    # compute (call appropriate version for each case)

    if apertures is None:
        # call fastest / simplest version without apertures
        # (no computation / storing of unit vector)

        solid_angle = _GG.compute_solid_angle_noapertures(
            # pts as 1d arrays
            pts_x=pts_x,
            pts_y=pts_y,
            pts_z=pts_z,
            # detector polygons as 1d arrays
            det_outline_x0=det_outline_x0,
            det_outline_x1=det_outline_x1,
            det_cents_x=det_cents_x,
            det_cents_y=det_cents_y,
            det_cents_z=det_cents_z,
            det_norm_x=det_nin_x,
            det_norm_y=det_nin_y,
            det_norm_z=det_nin_z,
            det_e0_x=det_e0_x,
            det_e0_y=det_e0_y,
            det_e0_z=det_e0_z,
            det_e1_x=det_e1_x,
            det_e1_y=det_e1_y,
            det_e1_z=det_e1_z,
        )

    elif return_vector:
        # call most complete version
        # (computation + storing of unit vector)
        (
            solid_angle,
            unit_vector_x, unit_vector_y, unit_vector_z,
        ) = _GG.compute_solid_angle_apertures_unitvectors(
            # pts as 1d arrays
            pts_x=pts_x,
            pts_y=pts_y,
            pts_z=pts_z,
            # detector polygons as 1d arrays
            det_outline_x0=det_outline_x0,
            det_outline_x1=det_outline_x1,
            det_cents_x=det_cents_x,
            det_cents_y=det_cents_y,
            det_cents_z=det_cents_z,
            det_norm_x=det_nin_x,
            det_norm_y=det_nin_y,
            det_norm_z=det_nin_z,
            det_e0_x=det_e0_x,
            det_e0_y=det_e0_y,
            det_e0_z=det_e0_z,
            det_e1_x=det_e1_x,
            det_e1_y=det_e1_y,
            det_e1_z=det_e1_z,
            # apertures
            ap_ind=ap_ind,
            ap_x=ap_x,
            ap_y=ap_y,
            ap_z=ap_z,
            ap_norm_x=ap_nin_x,
            ap_norm_y=ap_nin_y,
            ap_norm_z=ap_nin_z,
        )

        if visibility:
            _visibility_unit_vectors(
                # points
                pts_x=pts_x,
                pts_y=pts_y,
                pts_z=pts_z,
                # det
                det_cents_x=det_cents_x,
                det_cents_y=det_cents_y,
                det_cents_z=det_cents_z,
                det_nin_x=det_nin_x,
                det_nin_y=det_nin_y,
                det_nin_z=det_nin_z,
                # results
                solid_angle=solid_angle,
                unit_vector_x=unit_vector_x,
                unit_vector_y=unit_vector_y,
                unit_vector_z=unit_vector_z,
                **kwdargs,
            )

    elif visibility:
        # call intermediate version
        # (computation for visibility but no storing of unit vector)
        solid_angle = _GG.compute_solid_angle_apertures_visibility(
            # pts as 1d arrays
            pts_x=pts_x,
            pts_y=pts_y,
            pts_z=pts_z,
            # detector polygons as 1d arrays
            det_outline_x0=det_outline_x0,
            det_outline_x1=det_outline_x1,
            det_cents_x=det_cents_x,
            det_cents_y=det_cents_y,
            det_cents_z=det_cents_z,
            det_norm_x=det_nin_x,
            det_norm_y=det_nin_y,
            det_norm_z=det_nin_z,
            det_e0_x=det_e0_x,
            det_e0_y=det_e0_y,
            det_e0_z=det_e0_z,
            det_e1_x=det_e1_x,
            det_e1_y=det_e1_y,
            det_e1_z=det_e1_z,
            # apertures
            ap_ind=ap_ind,
            ap_x=ap_x,
            ap_y=ap_y,
            ap_z=ap_z,
            ap_norm_x=ap_nin_x,
            ap_norm_y=ap_nin_y,
            ap_norm_z=ap_nin_z,
            # possible obstacles
            **kwdargs,
        )

    else:
        # call fastest / simplest version
        # (no computation / storing of unit vector)

        if summed is True:
            solid_angle = _GG.compute_solid_angle_apertures_light_summed(
                # pts as 1d arrays
                pts_x=pts_x,
                pts_y=pts_y,
                pts_z=pts_z,
                # detector polygons as 1d arrays
                det_outline_x0=det_outline_x0,
                det_outline_x1=det_outline_x1,
                det_cents_x=det_cents_x,
                det_cents_y=det_cents_y,
                det_cents_z=det_cents_z,
                det_norm_x=det_nin_x,
                det_norm_y=det_nin_y,
                det_norm_z=det_nin_z,
                det_e0_x=det_e0_x,
                det_e0_y=det_e0_y,
                det_e0_z=det_e0_z,
                det_e1_x=det_e1_x,
                det_e1_y=det_e1_y,
                det_e1_z=det_e1_z,
                # apertures
                ap_ind=ap_ind,
                ap_x=ap_x,
                ap_y=ap_y,
                ap_z=ap_z,
                ap_norm_x=ap_nin_x,
                ap_norm_y=ap_nin_y,
                ap_norm_z=ap_nin_z,
            )

        else:
            solid_angle = _GG.compute_solid_angle_apertures_light(
                # pts as 1d arrays
                pts_x=pts_x,
                pts_y=pts_y,
                pts_z=pts_z,
                # detector polygons as 1d arrays
                det_outline_x0=det_outline_x0,
                det_outline_x1=det_outline_x1,
                det_cents_x=det_cents_x,
                det_cents_y=det_cents_y,
                det_cents_z=det_cents_z,
                det_norm_x=det_nin_x,
                det_norm_y=det_nin_y,
                det_norm_z=det_nin_z,
                det_e0_x=det_e0_x,
                det_e0_y=det_e0_y,
                det_e0_z=det_e0_z,
                det_e1_x=det_e1_x,
                det_e1_y=det_e1_y,
                det_e1_z=det_e1_z,
                # apertures
                ap_ind=ap_ind,
                ap_x=ap_x,
                ap_y=ap_y,
                ap_z=ap_z,
                ap_norm_x=ap_nin_x,
                ap_norm_y=ap_nin_y,
                ap_norm_z=ap_nin_z,
            )

    if timing:
        t2 = dtm.datetime.now()     # DB
        dt2 = (t2 - t1).total_seconds()

    # -------------
    # format output

    # solid_angle = 0 => nan for unit vectors
    if return_vector:
        i0 = solid_angle == 0
        unit_vector_x[i0] = np.nan
        unit_vector_y[i0] = np.nan
        unit_vector_z[i0] = np.nan

    # reshape if necessary
    if summed is False:
        shape = tuple(np.r_[det_shape0, shape0])
        if mask is None:
            if ndim0 > 1:
                solid_angle = np.reshape(solid_angle, shape)
                if return_vector:
                    unit_vector_x = np.reshape(unit_vector_x, shape)
                    unit_vector_y = np.reshape(unit_vector_y, shape)
                    unit_vector_z = np.reshape(unit_vector_z, shape)
        else:
            sa = np.zeros(shape, dtype=float)
            sa[:, mask] = solid_angle
            if return_vector:
                ux = np.fill(shape0, np.nan)
                uy = np.fill(shape0, np.nan)
                uz = np.fill(shape0, np.nan)
                ux[:, mask] = unit_vector_x
                uy[:, mask] = unit_vector_y
                uz[:, mask] = unit_vector_z

            # replace
            solid_angle = sa
            if return_vector:
                unit_vector_x, unit_vector_y, unit_vector_z = ux, uy, uz

    if timing:
        t3 = dtm.datetime.now()     # DB
        dt3 = (t3 - t2).total_seconds()

    # ------
    # return

    if return_vector:
        if timing:
            return (
                solid_angle, unit_vector_x, unit_vector_y, unit_vector_z,
                dt1, dt2, dt3,
            )
        else:
            return solid_angle, unit_vector_x, unit_vector_y, unit_vector_z

    else:
        if timing:
            return solid_angle, dt1, dt2, dt3
        else:
            return solid_angle