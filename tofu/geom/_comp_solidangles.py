# Built-in
# import warnings

# Common
import numpy as np
# import scipy.interpolate as scpinterp
# import scipy.integrate as scpintg
# from inspect import signature as insp

# local
from . import _GG


_APPROX = True
_ANISO = False
_BLOCK = True
_LTYPES = [int, float, np.int_, np.float_]


###############################################################################
###############################################################################
#                       Check inputs
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
