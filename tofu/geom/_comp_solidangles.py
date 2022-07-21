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


def _check_list_3dpolygons(
    lpoly=None,
    lpoly_name=None,
    closed=None,
    can_be_None=None,
    detectors_normal=None,
):

    # trivial case
    if can_be_None and lpoly is None:
        return

    # check inpiut
    if closed is None:
        closed = True

    # check lpoly is a list of (3, npi) arrays
    c0 = (
        isinstance(lpoly, list)
        and all([
            isinstance(pp, np.ndarray)
            and pp.ndim == 2
            and pp.shape[0] == 3
            for pp in lpoly
        ])
    )
    if not c0:
        msg = (
            f"Arg {lpoly_name} must be a list of (3, npi) arrays, where\n"
            "\t- dimension 0 corresponds to (X, Y, Z) coordinates\n"
            "\t- dimension 1 corresponds to the npi corners of polygon i\n"
            f"Provided:\n{lpoly}"
        )
        raise Exception(msg)

    # check if they are closed vs close
    for ii, pp in enumerate(lpoly):
        if not np.allclose(pp[:, 0], pp[:, -1]) and closed:
            lpoly[ii] = np.append(pp, pp[:, 0:1], axis=1)
        elif np.allclose(pp[:, 0], pp[:, -1]) and not closed:
            lpoly[ii] = pp[:, :-1]

        # make sure float
        lpoly[ii] = lpoly[ii].astype(float)

    # check counter-clockwise ass seen from normal vector
    for ii, pp in enumerate(lpoly):
        pass

    return lpoly


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


def _check_convert_det_dict(detectors=None):

    lk_out = ['outline_x0', 'outline_x1']
    lk_shape = [
        'cents_x', 'cents_y', 'cents_z',
        'e0_x', 'e0_y', 'e0_z',
        'e1_x', 'e1_y', 'e1_z',
        'nin_x', 'nin_y', 'nin_z',
    ]
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

    # check unit vectors (normalize + perpendicular)
    _check_unit_vectors(detectors)

    # copy dict and flatten (if not 1d)
    if detectors['cents_x'].ndim > 1:
        detectors = dict(detectors)
        lk = [k0 for k0 in detectors.keys() if 'outline' not in k0]
        for k0 in lk:
            detectors[k0] = detectors[k0].ravel()

    # turn into list of detectors arrays
    det = np.array([
        detectors['cents_x'][:, None]
        + detectors['e0_x'][:, None] * detectors['outline_x0'][None, :]
        + detectors['e1_x'][:, None] * detectors['outline_x1'][None, :],
        detectors['cents_y'][:, None]
        + detectors['e0_y'][:, None] * detectors['outline_x0'][None, :]
        + detectors['e1_y'][:, None] * detectors['outline_x1'][None, :],
        detectors['cents_z'][:, None]
        + detectors['e0_z'][:, None] * detectors['outline_x0'][None, :]
        + detectors['e1_z'][:, None] * detectors['outline_x1'][None, :],
    ]).swapaxes(0, 1)
    det = [dd for dd in det]

    # build detectors_normal
    det_norm = np.array([
        detectors['nin_x'], detectors['nin_y'], detectors['nin_z'],
    ])

    return det, det_norm


def _calc_solidangle_apertures_check(
    # observation points
    pts_x=None,
    pts_y=None,
    pts_z=None,
    # polygons
    apertures=None,
    detectors=None,
    detectors_normal=None,
    # possible obstacles
    config=None,
    # parameters
    visibility=None,
    return_vector=None,
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

    if isinstance(detectors, dict):
        # convert from dict to list of arrays
        detectors, detectors_normal = _check_convert_det_dict(detectors)

    detectors = _check_list_3dpolygons(
        lpoly=detectors,
        lpoly_name='detectors',
        closed=False,
        can_be_None=False,
        detectors_normal=detectors_normal,
    )

    # ---------
    # apertures

    apertures = _check_list_3dpolygons(
        lpoly=apertures,
        lpoly_name='apertures',
        closed=False,
        can_be_None=True,
        detectors_normal=detectors_normal,
    )

    # ----------------
    # detectors_normal

    c0 = (
        isinstance(detectors_normal, np.ndarray)
        and detectors_normal.ndim == 2
        and detectors_normal.shape == (3, len(detectors))
    )
    if not c0:
        msg = (
            "Arg detectors_normal must be a (3, nd) array, where\n"
            "\t- dimension 0 corresponds to (X, Y, Z) coordinates\n"
            "\t- nd = number of detectors (nd = len(detectors))\n"
            f"Provided:\n{detectors_normal}"
        )
        raise Exception(msg)

    # normalize
    detectors_normal /= np.sqrt(np.sum(detectors_normal**2, axis=0))

    # ----------
    # visibility

    if visibility is None:
        visibility = True
    if not isinstance(visibility, bool):
        msg = f"Arg visibility must be a bool!\nProvided: {visibility}"
        raise Exception(msg)

    # -------------
    # return_vector

    if return_vector is None:
        return_vector = False
    if not isinstance(return_vector, bool):
        msg = f"Arg return_vector must be a bool!\nProvided: {return_vector}"
        raise Exception(msg)

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
        apertures, detectors, detectors_normal,
        visibility, return_vector,
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
    detectors_normal=None,
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
        ap_split, ap_ind, ap_x, ap_y, ap_z = None, None, None, None, None
    else:
        ap_split = np.array([aa.shape[1] for aa in apertures])
        ap_ind = np.array([0] + [aa.shape[1] for aa in apertures[:-1]])
        ap_x = np.concatenate([aa[0, :] for aa in apertures])
        ap_y = np.concatenate([aa[1, :] for aa in apertures])
        ap_z = np.concatenate([aa[2, :] for aa in apertures])

    # ---------
    # detectors

    det_split = np.array([dd.shape[1] for dd in detectors])
    det_ind = np.r_[0, np.cumsum([dd.shape[1] for dd in detectors])]
    det_x = np.concatenate([dd[0, :] for dd in detectors])
    det_y = np.concatenate([dd[1, :] for dd in detectors])
    det_z = np.concatenate([dd[2, :] for dd in detectors])

    det_norm_x = detectors_normal[0, :]
    det_norm_y = detectors_normal[1, :]
    det_norm_z = detectors_normal[2, :]

    return (
        ndim0, shape0, mask,
        pts_x, pts_y, pts_z,
        ap_split, ap_ind, ap_x, ap_y, ap_z,
        det_split, det_ind, det_x, det_y, det_z,
        det_norm_x, det_norm_y, det_norm_z,
    )


def _visibility_unit_vectors(
    # points
    pts_x=None,
    pts_y=None,
    pts_z=None,
    # det
    det_split=None,
    det_x=None,
    det_y=None,
    det_z=None,
    det_norm_x=None,
    det_norm_y=None,
    det_norm_z=None,
    # results
    solid_angle=None,
    unit_vector_x=None,
    unit_vector_y=None,
    unit_vector_z=None,
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
            unit_vector_x[iok, ii]*det_norm_x[iok]
            + unit_vector_y[iok, ii]*det_norm_y[iok]
            + unit_vector_z[iok, ii]*det_norm_z[iok]
        )

        # detector centers
        dcent_x = np.array([
            np.mean(dd[:-1])
            for ii, dd in det_x.split(det_split)
            if iok[ii]
        ])
        dcent_y = np.array([
            np.mean(dd[:-1])
            for ii, dd in det_y.split(det_split)
            if iok[ii]
        ])
        dcent_z = np.array([
            np.mean(dd[:-1])
            for ii, dd in det_z.split(det_split)
            if iok[ii]
        ])

        # MC = point to centers
        MCx = dcent_x - pts_x[ii]
        MCy = dcent_y - pts_y[ii]
        MCz = dcent_z - pts_z[ii]

        # MCn = MC.norm
        MCn = (
            MCx*det_norm_x[iok]
            + MCy*det_norm_y[iok]
            + MCz*det_norm_z[iok]
        )

        # kk = (MC.norm) / (unit_vect.norm)
        kk = MCn / un

        # P = M + kk * unit_vect
        Px = pts_x[ii] + kk * unit_vector_x[iok, ii]
        Py = pts_y[ii] + kk * unit_vector_y[iok, ii]
        Pz = pts_z[ii] + kk * unit_vector_z[iok, ii]

        # Estimate visibility
        vis = _GG.LOS_areVis_PtsFromPts_VesStruct(
            np.array([pts_x[ii], pts_y[ii], pts_z[ii]]),
            np.array([Mx, My, Mz]),
            dist=kk[ii, iok],
            **kwdargs,
        )

        # Set non-visible to 0 / nan
        iout = vis == 0
        solid_angle[ii, iout] = 0.
        unit_vector_x[ii, iout] = np.nan
        unit_vector_y[ii, iout] = np.nan
        unit_vector_z[ii, iout] = np.nan


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
    detectors_normal=None,
    # possible obstacles
    config=None,
    # parameters
    visibility=None,
    return_vector=None,
):
    """ Return the solid angle subtended by na apertures and nd detectors

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

    (
        # observation points
        pts_x,
        pts_y,
        pts_z,
        mask,
        # polygons
        apertures,
        detectors,
        detectors_normal,
        # parameters
        visibility,
        return_vector,
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
        visibility=visibility,
        return_vector=return_vector,
    )

    # ----------------
    # pre-format input

    (
        ndim0, shape0, mask,
        pts_x, pts_y, pts_z,
        ap_split, ap_ind, ap_x, ap_y, ap_z,
        det_split, det_ind, det_x, det_y, det_z,
        det_norm_x, det_norm_y, det_norm_z,
    ) = _calc_solidangle_apertures_prepare(
        # observation points
        pts_x=pts_x,
        pts_y=pts_y,
        pts_z=pts_z,
        mask=mask,
        # polygons
        apertures=apertures,
        detectors=detectors,
        detectors_normal=detectors_normal,
    )
    nd = len(detectors)

    # Get kwdargs for LOS blocking
    if config is not None:
        kwdargs = config.get_kwdargs_LOS_isVis()

    # ------------------------------------------------
    # compute (call appropriate version for each case)


    if apertures is None:
        # call fastest / simplest version without apertures
        # (no computation / storing of unit vector)

        import pdb; pdb.set_trace()     # DB

        # get 2d det for triangulation
        det_x_2d, det_y_2d, det_z_2d = _get_2d(
        )

        solid_angle = _GG.compute_solid_angle_noapertures(
            # pts as 1d arrays
            pts_x=pts_x,
            pts_y=pts_y,
            pts_z=pts_z,
            # detector polygons as 1d arrays
            det_ind=det_ind,
            det_x=det_x,
            det_y=det_y,
            det_z=det_z,
            det_norm_x=det_norm_x,
            det_norm_y=det_norm_y,
            det_norm_z=det_norm_z,
            # for triangulation (assumes counter_clockwise)
            det_x=det_x,
            det_y=det_y,
            det_z=det_z,
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
            det_ind=det_ind,
            det_x=det_x,
            det_y=det_y,
            det_z=det_z,
            det_norm_x=det_norm_x,
            det_norm_y=det_norm_y,
            det_norm_z=det_norm_z,
            # aperture polygons as 1d arrays
            ap_ind=ap_ind,
            ap_x=ap_x,
            ap_y=ap_y,
            ap_z=ap_z,
        )

        if visibility:
            _visibility_unit_vectors(
                # points
                pts_x=pts_x,
                pts_y=pts_y,
                pts_z=pts_z,
                # det
                det_split=det_split,
                det_x=det_x,
                det_y=det_y,
                det_z=det_z,
                det_norm_x=det_norm_x,
                det_norm_y=det_norm_y,
                det_norm_z=det_norm_z,
                # results
                solid_angle=solid_angle,
                unit_vector_x=unit_vector_x,
                unit_vector_y=unit_vector_y,
                unit_vector_z=unit_vector_z,
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
            det_ind=det_ind,
            det_x=det_x,
            det_y=det_y,
            det_z=det_z,
            # aperture polygons as 1d arrays
            ap_ind=ap_ind,
            ap_x=ap_x,
            ap_y=ap_y,
            ap_z=ap_z,
            # possible obstacles
            **kwdargs,
        )

    else:
        # call fastest / simplest version
        # (no computation / storing of unit vector)
        solid_angle = _GG.compute_solid_angle_apertures_light(
            # pts as 1d arrays
            pts_x=pts_x,
            pts_y=pts_y,
            pts_z=pts_z,
            # detector polygons as 1d arrays
            det_ind=det_ind,
            det_x=det_x,
            det_y=det_y,
            det_z=det_z,
            # aperture polygons as 1d arrays
            ap_ind=ap_ind,
            ap_x=ap_x,
            ap_y=ap_y,
            ap_z=ap_z,
        )

    # -------------
    # format output

    if return_vector:
        i0 = solid_angle == 0
        unit_vector_x[i0] = np.nan
        unit_vector_y[i0] = np.nan
        unit_vector_z[i0] = np.nan

    shape = tuple(np.r_[nd, shape0])
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
        unit_vector_x, unit_vector_y, unit_vector_z = ux, uy, uz

    # ------
    # return

    if return_vector:
        return solid_angle, unit_vector_x, unit_vector_y, unit_vector_z
    else:
        return solid_angle
