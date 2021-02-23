

# Built-in
import warnings

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
        pts = np.ascontiguousarray(pts, dtype=float)
        rad = np.r_[rad].astype(float).ravel()

        # Check pts, traj and r are array of good shape
        c0 = (traj.ndim in [1, 2]
              and pts.ndim in [1, 2]
              and 3 in traj.shape
              and 3 in pts.shape)
        assert c0
        if traj.ndim == 1:
            traj = traj.reshape((3, 1))
        if traj.shape[0] != 3:
            traj = traj.T
        if pts.ndim == 1:
            pts = pts.reshape((3, 1))
        if pts.shape[0] != 3:
            pts = pts.T
    except Exception as err:
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
    traj=None,
    rad=None,
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
    traj:       np.ndarray
        Array of (3, N) pts coordinates (X, Y, Z) representing the particle
        positions
    pts:        np.ndarray
        Array of (3, M) pts coordinates (X, Y, Z) representing the points from
        which the particle is observed
    rad:        float / np.ndarray
        Unique of multiple values for the radius of the spherical particle
            if multiple, rad is a np.ndarray of shape (N,)
    config:     None / tf.geom.Config
        if block = True, solid angles are non-zero only if the field of view is
        not blocked bya structural element in teh chamber
    approx:     None / bool
        Flag indicating whether to compute the solid angle using an 1st-order
        series development (in whichcase the solid angle becomes proportional
        to the radius of the particle, see Notes_Upgrades/)
    aniso:      None / bool
        Flag indicating whether to consider anisotropic emissivity, meaning the
        routine must also compute and return the unit vector directing the flux
        from each pts to each position on the trajectory of the particle
    block:      None / bool
        Flag indicating whether to check for vignetting by structural elements
        provided by config

    Return:
    -------
    sang: np.ndarray
        (N, M) Array of floats, solid angles

    """
    ################
    # Prepare inputs
    (
        traj, pts, rad, config,
        approx, aniso, block
    ) = _check_calc_solidangle_particle(
        traj=traj,
        pts=pts,
        rad=rad,
        config=config,
        approx=approx,
        aniso=aniso,
        block=block,
    )

    ################
    # Main computation

    # traj2pts vector, with length (3d array (3, N, M))
    vect = traj[:, :, None] - pts[:, None, :]
    len_v = np.ascontiguousarray(np.sqrt(np.sum(vect**2, axis=0)))

    # If aniso or block, normalize
    if aniso or block:
        vect = vect / len_v[None, :, :]

    # Solid angle
    if approx:
        sang = np.pi * rad[:, None]**2 / len_v**2
    else:
        sang = 2.*np.pi * (1 - np.sqrt(1. - rad[:, None]**2 / len_v**2))

    # block
    if block:
        kwdargs = config.get_kwdargs_LOS_isVis()
        # Issue 471: k=len_v is a 2d array, _GG only takes 1d array...
        indvis = _GG.LOS_areVis_PtsFromPts_VesStruct(
            traj, pts, dist=len_v, **kwdargs
        )
        # Because indvis is an array of int (cf. issue 471)
        iout = indvis == 0
        sang[iout] = 0.
        vect[:, iout] = np.nan

    ################
    # Return
    if aniso:
        return sang, vect
    else:
        return sang


def calc_solidangle_particle_integ(
    traj, r=1.0, config=None, approx=True, block=True, res=0.01
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
    return
