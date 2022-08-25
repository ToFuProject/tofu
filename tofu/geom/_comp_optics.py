

import warnings

import numpy as np
import scipy.interpolate as scpinterp
import scipy.stats as scpstats
import matplotlib.pyplot as plt


_LTYPES = [int, float, np.int_, np.float_]
_MISCUT = True


# ###############################################
#           utility
# ###############################################


def _check_bool(var, vardef=None, varname=None):
    if var is None:
        var = vardef
    if not isinstance(var, bool):
        msg = (
            "Arg {} must be a bool\n".format(varname)
            + "  You provided: {}".format(type(var))
        )
        raise Exception(msg)
    return var


def _are_broadcastable(**kwdargs):

    # Check if broadcastable
    lv = list(kwdargs.values())
    c0 = (
        all([isinstance(vv, np.ndarray) for vv in lv])
        and all([
            all([
                (m == n) or (m == 1) or (n == 1)
                for m, n in zip(vv.shape[::-1], lv[0].shape[::-1])
            ])
            for vv in lv
        ])
    )

    # raise Exception if strict
    if not c0:
        msg = (
            "All args must be broadcastable with each other!\n"
            + "You provided:\n"
            + "\n".join([
                '\t- {}.shape = {}'.format(k0, v0.shape)
                for k0, v0 in kwdargs.items()
            ])
        )
        raise Exception(msg)


# ###############################################
# ###############################################
#           CrystalBragg
# ###############################################
# ###############################################


def _checkformat_xixj(xi, xj):

    if xi is None or xj is None:
        msg = (
            "Arg xi and xj must be provided!\n"
            "Provided:\n"
            f"\t- xi: {xi}\n"
            f"\t- xj: {xj}\n"
        )
        raise Exception(msg)

    xi = np.atleast_1d(xi)
    xj = np.atleast_1d(xj)

    if xi.shape == xj.shape:
        return xi, xj, (xi, xj)
    else:
        return xi, xj, np.meshgrid(
            xi, xj,
            copy=True, sparse=False, indexing='ij',
        )


# ###############################################
#           sampling
# ###############################################


def _check_dthetapsi(
    dtheta=None, psi=None,
    extenthalf_psi=None, extenthalf_dtheta=None,
    ntheta=None, npsi=None,
    include_summit=None,
):
    """ Return formatted dtheta and psi

    They are returned with the same shape (at least 1d arrays)

    They can be:
        - 'envelop': if psi of dtheta = 'envelop', psi and dtheta are computed
            to describe the contour of the crystal as 2 (nenvelop,) arrays
            The nvelop is computed from (npsi, ntheta)
        - np.ndarrays or scalar: the routine just converts to np.ndarrays
            (using np.atleast_1d()) and checks they have the same shape

    """

    # Check inputs
    if dtheta is None:
        dtheta = 0.
    if psi is None:
        psi = 0.

    # if envelop => get all points around cryst + summit
    if any([isinstance(vv, str) and vv == 'envelop' for vv in [dtheta, psi]]):
        psi, dtheta = CrystBragg_sample_outline_sphrect(
            extenthalf_psi, extenthalf_dtheta,
            npsi=npsi, ntheta=ntheta,
            include_summit=include_summit,
        )

    c0 = all([
        type(vv) in _LTYPES
        or isinstance(vv, np.ndarray)
        for vv in [dtheta, psi]
    ])

    dtheta = np.atleast_1d(dtheta)
    psi = np.atleast_1d(psi)
    if psi.shape != dtheta.shape:
        msg = (
            "dtheta and psi should have the same shape\n"
            + "\t- dtheta.shape = {}\n".format(dtheta.shape)
            + "\t- psi.shape = {}".format(psi.shape)
        )
        raise Exception(msg)
    return dtheta, psi


def CrystBragg_sample_outline_sphrect(
    extent_psi, extent_dtheta,
    npsi=None, ntheta=None,
    include_summit=None,
):
    """ Return psi, dtheta describing the envelop of a crystal

    They are computed from
        - extent_psi, extent_dtheta: np.ndarrays for size 2
        - npsi, ntheta: integers

    They are returned with the same shape:
        (nenvelop,) arrays describing the contour of the crystal

    Optionally, the crystal summit can be appended at the end

    """

    # check inputs
    if include_summit is None:
        include_summit = True
    if ntheta is None:
        ntheta = 5
    if npsi is None:
        npsi = 3

    # compute
    psi = extent_psi*np.linspace(-1, 1., npsi)
    dtheta = extent_dtheta*np.linspace(-1, 1., ntheta)
    psimin = np.full((ntheta,), psi[0])
    psimax = np.full((ntheta,), psi[-1])
    dthetamin = np.full((npsi,), dtheta[0])
    dthetamax = np.full((npsi,), dtheta[-1])
    psi = np.concatenate((psi, psimax, psi[::-1], psimin))
    dtheta = np.concatenate((dthetamin, dtheta, dthetamax, dtheta[::-1]))
    if include_summit is True:
        psi = np.r_[psi, 0.]
        dtheta = np.r_[dtheta, 0.]
    return psi, dtheta

def CrystBragg_sample_outline_cylrect(
    extent_de1, extent_dtheta,
    ndtheta=None, nde1=None,
    include_summit=None,
):
    """ Return dtheta, de1 describing the envelop of a cylindrical crystal

    They are computed from
        - extent_dtheta, extent_de1: np.ndarrays for size 2
        - ndtheta, nde1a: integers

    They are returned with the same shape:
        (nenvelop,) arrays describing the contour of the crystal

    Optionally, the crystal summit can be appended at the end

    """

    # check inputs
    if include_summit is None:
        include_summit = True
    if ndtheta is None:
        ndtheta = 10
    if nde1 is None:
        nde1 = 9

    # compute
    de1 = extent_de1*np.linspace(-1, 1., nde1)
    dtheta = extent_dtheta*np.linspace(-1, 1., ndtheta)
    de1min = np.full((ndtheta,), de1[0])
    de1max = np.full((ndtheta,), de1[-1])
    dthetamin = np.full((nde1,), dtheta[0])
    dthetamax = np.full((nde1,), dtheta[-1])
    de1 = np.concatenate((de1, de1max, de1[::-1], de1min))
    dtheta = np.concatenate((dthetamin, dtheta, dthetamax, dtheta[::-1]))
    if include_summit is True:
        de1 = np.r_[de1, 0.]
        dtheta = np.r_[dtheta, 0.]
    cry_dpts = {}
    cry_dpts['de1'] = de1.reshape(de1.size,1,1)
    cry_dpts['dtheta'] = dtheta.reshape(dtheta.size,1,1)
    return cry_dpts

def CrystBragg_sample_outline_flatrect(
    extent_de1, extent_de2,
    nde1=None, nde2=None,
    include_summit=None,
):
    """ Return de1, de2 describing the envelop of a flat crystal

    They are computed from
        - extent_de1, extent_de2: np.ndarrays for size 2
        - nde1, nde2: integers

    They are returned with the same shape:
        (nenvelop,) arrays describing the contour of the crystal

    Optionally, the crystal summit can be appended at the end

    """

    # check inputs
    if include_summit is None:
        include_summit = True
    if nde1 is None:
        nde1 = 5
    if nde2 is None:
        nde2 = 3

    # compute
    de1 = extent_de1*np.linspace(-1, 1., nde1)
    de2 = extent_de2*np.linspace(-1, 1., nde2)
    de1min = np.full((nde2,), de1[0])
    de1max = np.full((nde2,), de1[-1])
    de2min = np.full((nde1,), de2[0])
    de2max = np.full((nde1,), de2[-1])
    de1 = np.concatenate((de1, de1max, de1[::-1], de1min))
    de2 = np.concatenate((de2min, de2, de2max, de2[::-1]))
    if include_summit is True:
        de1 = np.r_[de1, 0.]
        de2 = np.r_[de2, 0.]
    cry_dpts = {}
    cry_dpts['de1'] = de1.reshape(de1.size,1,1)
    cry_dpts['de2'] = de2.reshape(de2.size,1,1)
    cry_dpts['phi'] = np.zeros(cry_dpts['de2'].shape)
    return cry_dpts


def CrystBragg_get_noute1e2_from_psitheta(
    nout, e1, e2,
    psi=None, dtheta=None,
    e1e2=None, sameshape=None,
    extenthalf_psi=None,
    extenthalf_dtheta=None,
    ntheta=None, npsi=None,
    include_summit=None,
):
    """ Return local unit vectors at chosen points on the crystal surface

    The points are defined by (psi, dtheta), which have to be the same shape
    and can be:
        - arbitrary: must be the same shape, can have up to 4 dimensions
        - defined from the envelop using (nspi, ntheta) fed to
            CrystBragg_sample_outline_sphrect()
            In this case psi and theta are 1d

    Local unit vectors (vout, ve1, ve2) at each point are defined the global
    unit vectors (taken at the crystal summit) passed as input (nout, e1, e2)

    Return
    ------
    vout:   np.ndarray
        (X, Y, Z) coordinates of nout normal outwards local vectors
    ve1:    np.ndarray
        (X, Y, Z) coordinates of e1 local tangential vectors
        Only returned if e1e2 is True
    ve2:    np.ndarray
        (X, Y, Z) coordinates of e2 local tangential vectors
        Only returned if e1e2 is True

    In all cases, the shape of the unit vectors is (3, psi.shape)

    """

    # check inputs
    if e1e2 is None:
        e1e2 = True
    if sameshape is None:
        sameshape = psi.shape == nout.shape[1:]
    if sameshape:
        assert psi.shape == nout.shape[1:]

    dtheta, psi = _check_dthetapsi(
        dtheta=dtheta, psi=psi,
        extenthalf_psi=extenthalf_psi,
        extenthalf_dtheta=extenthalf_dtheta,
        ntheta=ntheta, npsi=npsi,
        include_summit=include_summit,
    )

    # Prepare
    if sameshape is False:
        assert psi.ndim in [1, 2, 3, 4]
        if psi.ndim == 1:
            nout = nout[:, None]
            e1, e2 = e1[:, None], e2[:, None]
        elif psi.ndim == 2:
            nout = nout[:, None, None]
            e1, e2 = e1[:, None, None], e2[:, None, None]
        elif psi.ndim == 3:
            nout = nout[:, None, None, None]
            e1, e2 = e1[:, None, None, None], e2[:, None, None, None]
        else:
            nout = nout[:, None, None, None, None]
            e1 = e1[:, None, None, None, None]
            e2 = e2[:, None, None, None, None]

    # Not necessary for broadcasting (last dims first)
    theta = dtheta  # + np.pi/2.
    # psi = psi[None, ...]

    # Compute
    # vout = (
    # (np.cos(psi)*nout + np.sin(psi)*e1)*np.sin(theta) + np.cos(theta)*e2
    # )
    vout = (
         (np.cos(psi)*nout + np.sin(psi)*e1)*np.cos(theta) + np.sin(theta)*e2
         )
    if e1e2:
        ve1 = -np.sin(psi)*nout + np.cos(psi)*e1
        ve2 = np.array([vout[1, ...]*ve1[2, ...] - vout[2, ...]*ve1[1, ...],
                        vout[2, ...]*ve1[0, ...] - vout[0, ...]*ve1[2, ...],
                        vout[0, ...]*ve1[1, ...] - vout[1, ...]*ve1[0, ...]])
        return vout, ve1, ve2
    else:
        return vout

def CrystalBragg_get_noute1e2_from_dthetade1(
    nout,e1,e2,
    cry_dpts=None,
    e1e2=True
):
    '''
    #### CYLINDRICAL CRYSTAL

    CrystalBragg_get_noute1e2_from_dthetade1 is a function that calculates
    the crystal local basis vector for points sampled to have a valid Bragg reflection

    INPUTS:
        nout -      vector, dim (3,1), summit outward normal axis
        e1 -        vector, dim (3,1), summit horizontal axis (along cylinder)
        e2 -        vector, dim (3,1), summit vertical axis
        cry_dpts -  dictionary, contains variables defining displacement on a cylinder
                    from the summit; dtheta -> dim (nlamb, npts, ndtheta), rotational
                    displacement; de1 -> dim (nlamb, npts, ndtheta), translational displacement
        e1e2 -      ## OPTIONAL, flag, if user wants the e1,e2 local vector

    OUTPUTS:
        vout -      matrix, dim (3,nlamb,npts,ndtheta), local outward normal axis
        ve1 -       matrix, dim (3,nlamb,npts,ndtheta), local horizontal axis (along cylinder)
        ve2 -       matrix, dim (3,nlamb,npts,ndtheta), local vertical axis
    '''

    # check inputs
    if e1e2 is None:
        e1e2 = True

    # Initializes matrices to store crystal local unit vectors
    vout = np.full(np.append(3, cry_dpts['dtheta'].shape), np.nan) # dim (3, nlamb,npts, ndtheta), horizontal points on the crystal surface
    ve1 = np.full(np.append(3, cry_dpts['dtheta'].shape), np.nan) # dim (3, nlamb,npts, ndtheta), horizontal points on the crystal surface
    ve2 = np.full(np.append(3, cry_dpts['dtheta'].shape), np.nan) # dim (3,nlamb,npts, ndtheta), horizontal points on the crystal surface

    # Projection matrix to crystal summit basis
    nout = nout.reshape(1,3) # dim (1,3)
    e1 = e1.reshape(1,3) # dim (1,3)
    e2 = e2.reshape(1,3) # dim (1,3)
    RR = np.concatenate((nout, e2, e1), axis = 0) # dim (3,3)

    # Projects summit normal vector onto the crystal summit basis (not the origin)
    nout_summit = np.matmul(RR, np.transpose(nout)) # dum (3,1)
    e1_summit = np.matmul(RR, np.transpose(e1)) # dum (3,1)
    e2_summit = np.matmul(RR, np.transpose(e2)) # dum (3,1)

    # Loop over Bragg angles
    for ii in np.arange(cry_dpts['dtheta'].shape[0]):
        # Loop over pts
        for jj in np.arange(cry_dpts['dtheta'].shape[1]):
            # Loop over vertical locations
            for kk in np.arange(cry_dpts['dtheta'].shape[2]):
                # Cosine and sine for this rotation
                cc, ss = np.cos(cry_dpts['dtheta'][ii,jj,kk]), np.sin(cry_dpts['dtheta'][ii,jj,kk]) # scalars

                # Rotation matrix from crystal summit to local position
                Rot = np.array([[cc,-ss,0], [ss,cc,0], [0,0,1]]).reshape(3,3) # dim (3,3)

                # Rotates crystal summit normal vector to the local normal vector
                nout_local = np.matmul(Rot, nout_summit) # dim (3,1)
                e1_local = np.matmul(Rot, e1_summit) # dim (3,1)
                e2_local = np.matmul(Rot, e2_summit) # dim (3,1)

                # Projects local crystal normal vector onto origin base
                nout_mod = np.matmul(np.linalg.inv(RR), nout_local) # dim (3,1)
                e1_mod = np.matmul(np.linalg.inv(RR), e1_local) # dim (3,1)
                e2_mod = np.matmul(np.linalg.inv(RR), e2_local) # dim (3,1)

                # Stores the local basis results
                vout[:,ii,jj,kk] = nout_mod.reshape(3)
                ve1[:,ii,jj,kk] = e1_mod.reshape(3)
                ve2[:,ii,jj,kk] = e2_mod.reshape(3)

    # Returns local basis vectors
    if e1e2:
        return vout, ve1, ve2
    else:
        return vout


def CrystalBragg_get_noute1e2_from_de1de2(
    nout,e1,e2,
    cry_dpts=None,
    e1e2=True
):
    '''
    #### FLAT CRYSTAL

    CrystalBragg_get_noute1e2_from_de1de2 is a function that calculates
    the crystal local basis vector for points sampled to have a valid Bragg reflection

    INPUTS:
        nout -      vector, dim (3,1), summit outward normal axis
        e1 -        vector, dim (3,1), summit horizontal axis (along cylinder)
        e2 -        vector, dim (3,1), summit vertical axis
        cry_dpts -  dictionary, contains variables defining displacement on a cylinder
                    from the summit; dtheta -> dim (nlamb, npts, ndtheta), rotational
                    displacement; de1 -> dim (nlamb, npts, ndtheta), translational displacement
        e1e2 -      ## OPTIONAL, flag, if user wants the e1,e2 local vector

    OUTPUTS:
        vout -      matrix, dim (3,nlamb,npts,ndtheta), local outward normal axis
        ve1 -       matrix, dim (3,nlamb,npts,ndtheta), local horizontal axis (along cylinder)
        ve2 -       matrix, dim (3,nlamb,npts,ndtheta), local vertical axis
    '''
    # check inputs
    if e1e2 is None:
        e1e2 = True

    # Initializes matrices to store crystal local unit vectors
    vout = np.full(np.append(3, cry_dpts['de1'].shape), np.nan) # dim (3, nlamb,npts, ndtheta), horizontal points on the crystal surface
    ve1 = np.full(np.append(3, cry_dpts['de1'].shape), np.nan) # dim (3, nlamb,npts, ndtheta), horizontal points on the crystal surface
    ve2 = np.full(np.append(3, cry_dpts['de1'].shape), np.nan) # dim (3,nlamb,npts, ndtheta), horizontal points on the crystal surface

    # Loop over Bragg angles
    for ii in np.arange(cry_dpts['de1'].shape[0]):
        # Loop over pts
        for jj in np.arange(cry_dpts['de1'].shape[1]):
            # Loop over vertical locations
            for kk in np.arange(cry_dpts['de1'].shape[2]):
                # Stores the local basis results
                vout[:,ii,jj,kk] = nout
                ve1[:,ii,jj,kk] = e1
                ve2[:,ii,jj,kk] = e2

    # Returns local basis vectors
    if e1e2:
        return vout, ve1, ve2
    else:
        return vout


def CrystBragg_sample_outline_plot_sphrect(
    center, nout, e1, e2,
    rcurve, extenthalf, res=None,
):
    """ Get the set of points in (x, y, z) coordinates sampling the spherical crystal
    outline
    """

    # check inputs
    if res is None:
        res = np.min(extenthalf)/5.

    # compute
    npsi = 2*int(np.ceil(extenthalf[0] / res)) + 1
    ntheta = 2*int(np.ceil(extenthalf[1] / res)) + 1

    psi, dtheta = CrystBragg_sample_outline_sphrect(
        extenthalf[0], extenthalf[1],
        npsi=npsi, ntheta=ntheta,
        include_summit=False,
    )

    vout = CrystBragg_get_noute1e2_from_psitheta(
        nout, e1, e2, psi, dtheta,
        e1e2=False, sameshape=False,
    )
    return center[:, None] + rcurve*vout


def CrystBragg_sample_outline_plot_cylrect(
    center, nout, e1, e2,
    rcurve, extenthalf, res=None,
):
    """ Get the set of points in (x, y, z) coordinates sampling the cylindrical crystal
    outline
    """

    # check inputs
    if res is None:
        res = np.min(extenthalf)/5.

    # Resolution to calculate the outline
    nde1 = 2*int(np.ceil(extenthalf[0] / res)) + 1
    ndtheta = 2*int(np.ceil(extenthalf[1] / res)) + 1

    # Calculates the displacement variables for the outline
    cry_dpts = CrystBragg_sample_outline_cylrect(
        extenthalf[0], extenthalf[1],
        ndtheta=ndtheta, nde1=nde1,
        include_summit=False,
    )

    # Calculates the crystal local basis vectors at the outline
    vout, ve1, ve2 = CrystalBragg_get_noute1e2_from_dthetade1(
        nout,e1,e2,
        cry_dpts=cry_dpts,
        e1e2=True
    )

    # Returns (X,Y,Z) coordinates of the outline
    return center[:,None] + rcurve*vout[:,:,0,0] + cry_dpts['de1'][:,0,0]*ve1[:,:,0,0]


def CrystBragg_sample_outline_plot_flatrect(
    summit, nout, e1, e2,
    extenthalf, res=None,
):
    """ Get the set of points in (x, y, z) coordinates sampling the flat crystal
    outline
    """

    # check inputs
    if res is None:
        res = np.min(extenthalf)/5.

    # Resolution to calculate the outline
    nde1 = 2*int(np.ceil(extenthalf[0] / res)) + 1
    nde2 = 2*int(np.ceil(extenthalf[1] / res)) + 1

    # Calculates the displacement variables for the outline
    cry_dpts = CrystBragg_sample_outline_flatrect(
        extenthalf[0], extenthalf[1],
        nde1=nde1, nde2=nde2,
        include_summit=False,
    )

    # Calculates the crystal local basis vectors at the outline
    vout, ve1, ve2 = CrystalBragg_get_noute1e2_from_de1de2(
        nout,e1,e2,
        cry_dpts=cry_dpts,
        e1e2=True
    )

    # Returns (X,Y,Z) coordinates of the outline
    return summit[:,None] + cry_dpts['de1'][:,0,0]*ve1[:,:,0,0] + cry_dpts['de2'][:,0,0]*ve2[:,:,0,0]


# ###############################################
#           lamb <=> bragg
# ###############################################

def get_bragg_from_lamb(lamb, d, n=None):
    """ n*lamb = 2d*sin(bragg)
    The angle bragg is defined as the angle of incidence of the emissed photon
    vector and the crystal mesh, and not the crystal dioptre.
    For record, both are parallel and coplanar when is defined parallelism
    into the crystal.
    """
    if n is None:
        n = 1
    bragg = np.full(lamb.shape, np.nan)
    sin = n*lamb/(2.*d)
    indok = np.abs(sin) <= 1.
    bragg[indok] = np.arcsin(sin[indok])
    return bragg


def get_lamb_from_bragg(bragg, d, n=None):
    """ n*lamb = 2d*sin(bragg)
    The angle bragg is defined as the angle of incidence of the emissed photon
    vector and the crystal mesh, and not the crystal dioptre.
    For record, both are parallel and coplanar when is defined parallelism
    into the crystal.
    """
    if n is None:
        n = 1
    return 2*d*np.sin(bragg) / n


# ###############################################
#           vectors <=> angles
# ###############################################


def get_vectors_from_angles(alpha, beta, nout, e1, e2):
    """Return new unit vectors according to alpha and beta entries from user
    caused by the miscut assumed on the crystal.
    """

    e1_bis = (
        np.cos(alpha)*(np.cos(beta)*e1 + np.sin(beta)*e2) - np.sin(alpha)*nout
    )

    e2_bis = np.cos(beta)*e2-np.sin(beta)*e1

    nout_bis = (
        np.cos(alpha)*nout + np.sin(alpha)*(np.cos(beta)*e1 + np.sin(beta)*e2)
    )

    nin_bis = -nout_bis

    return nin_bis, nout_bis, e1_bis, e2_bis


# ###############################################
#           Approximate solution
# ###############################################


def get_rowland_dist_from_bragg(bragg=None, rcurve=None):
    return rcurve*np.sin(bragg)


def get_approx_detector_rel(rcurve, bragg,
                            braggref=None, xiref=None,
                            bragg01=None, dist01=None,
                            tangent_to_rowland=None):
    """ Return the approximative detector position on the Rowland circle
    relatively to the Bragg crystal.
    Possibility to define tangential position of the detector to the Rowland
    circle or not.
    On WEST, the maximum miscut between two halves can be up to few
    arcmin so here, doesn't need to define the precise location of the detector
    The bragg angle is provided and naturally defined as the angle between the
    emissed photon vector and the crystal mesh.
    So, if miscut approuved, bragg is relative
    to the vector basis dmat(nout,e1,e2).
    The position of the detector, relatively to the crystal, will be so in
    another Rowland circle with its center shifted from the original one.
    """

    if tangent_to_rowland is None:
        tangent_to_rowland = True

    # distance crystal - det_center
    # bragg between incident vector and mesh
    det_dist = rcurve*np.sin(bragg)

    # det_nout and det_e1 in (nout, e1, e2) (det_e2 = e2)
    n_crystdet_rel = np.r_[-np.sin(bragg), np.cos(bragg), 0.]
    if tangent_to_rowland is True:
        bragg2 = 2.*bragg
        det_nout_rel = np.r_[-np.cos(bragg2), -np.sin(bragg2), 0.]
        det_ei_rel = np.r_[np.sin(bragg2), -np.cos(bragg2), 0.]
    else:
        det_nout_rel = -n_crystdet_rel
        det_ei_rel = np.r_[np.cos(bragg), np.sin(bragg), 0]

    # update with bragg01 and dist01
    if bragg01 is not None:
        ang = np.diff(np.sort(bragg01))
        # h = l1 tan(theta1) = l2 tan(theta2)
        # l = l2 (tan(theta1) + tan(theta2)) / tan(theta1)
        # l = l2 / cos(theta2)
        # l = l tan(theta1) / (cos(theta2) * (tan(theta1) + tan(theta2)))
        theta2 = bragg if tangent_to_rowland is True else np.pi/2
        theta1 = np.abs(bragg-bragg01[0])
        tan1 = np.tan(theta1)
        d0 = det_dist * tan1 / (np.cos(theta2) * (tan1+np.tan(theta2)))
        theta1 = np.abs(bragg-bragg01[1])
        tan1 = np.tan(theta1)
        d1 = det_dist * tan1 / (np.cos(theta2) * (tan1+np.tan(theta2)))
        if np.prod(np.sign(bragg01-bragg)) >= 0:
            d01 = np.abs(d0 - d1)
        else:
            d01 = d0 + d1
        det_dist = det_dist * dist01 / d01

    return det_dist, n_crystdet_rel, det_nout_rel, det_ei_rel


def get_det_abs_from_rel(det_dist, n_crystdet_rel, det_nout_rel, det_ei_rel,
                         summit, nout, e1, e2,
                         ddist=None, di=None, dj=None,
                         dtheta=None, dpsi=None, tilt=None):
    """ Return the absolute detector position, according to tokamak's frame,
    on the Rowland circle from its relative position to the Bragg crystal.
    If miscut approuved, bragg is relative to the vector basis
    dmat(nout,e1,e2).
    The position of the detector, relatively to the crystal, will be so in
    another Rowland circle with its center shifted from the original one.
    """

    # Reference on detector
    det_nout = (det_nout_rel[0]*nout
                + det_nout_rel[1]*e1 + det_nout_rel[2]*e2)
    det_ei = (det_ei_rel[0]*nout
                + det_ei_rel[1]*e1 + det_ei_rel[2]*e2)
    det_ej = np.cross(det_nout, det_ei)

    # Apply translation of center (ddist, di, dj)
    if ddist is None:
        ddist = 0.
    if di is None:
        di = 0.
    if dj is None:
        dj = 0.
    det_dist += ddist

    n_crystdet = (n_crystdet_rel[0]*nout
                  + n_crystdet_rel[1]*e1 + n_crystdet_rel[2]*e2)
    det_cent = summit + det_dist*n_crystdet + di*det_ei + dj*det_ej

    # Apply angles on unit vectors with respect to themselves
    if dtheta is None:
        dtheta = 0.
    if dpsi is None:
        dpsi = 0.
    if tilt is None:
        tilt = 0.

    # dtheta and dpsi
    det_nout2 = (
        (np.cos(dpsi)*det_nout + np.sin(dpsi)*det_ei)*np.cos(dtheta)
        + np.sin(dtheta)*det_ej
    )
    det_ei2 = (np.cos(dpsi)*det_ei - np.sin(dpsi)*det_nout)
    det_ej2 = np.cross(det_nout2, det_ei2)

    # tilt
    det_ei3 = np.cos(tilt)*det_ei2 + np.sin(tilt)*det_ej2
    det_ej3 = np.cross(det_nout2, det_ei3)

    return det_cent, det_nout2, det_ei3, det_ej3


# ###############################################
#           Sagital / meridional focus
# ###############################################


def calc_meridional_sagittal_focus(
    rcurve=None,
    bragg=None,
    alpha=None,
    miscut=None,
    verb=None,
):

    # Check input
    if rcurve is None or bragg is None:
        msg = (
            "Args rcurve and bragg must be provided!"
        )
        raise Exception(msg)

    verb = _check_bool(verb, vardef=True, varname='verb')
    miscut = _check_bool(
        miscut,
        vardef=_MISCUT,
        varname='miscut',
    )
    # Compute
    s_merid_ref = rcurve*np.sin(bragg)
    s_sagit_ref = -s_merid_ref/np.cos(2.*bragg)

    s_merid_unp = rcurve*(np.sin(bragg) + np.cos(bragg)*np.sin(alpha))
    s_sagit_unp = -s_merid_unp/(1-2.*np.sin(bragg+alpha)**2.)

    # verb
    if verb is True:
        mr = round(s_merid_ref, ndigits=3)
        sr = round(s_sagit_ref, ndigits=3)
        msg = (
            "Assuming a perfect crystal, from it are:\n"
            f"\t-the meridonal focus at {mr} m.\n"
            f"\t-the sagittal focus at {sr} m.\n"
        )

        if miscut is True:
            delta_merid = abs(s_merid_unp - s_merid_ref)
            delta_sagit = abs(s_sagit_unp - s_sagit_ref)
            mnp = round(s_merid_unp, ndigits=3)
            snp = round(s_sagit_unp, ndigits=3)
            mca = round(delta_merid, ndigits=3)
            sca = round(delta_sagit, ndigits=3)
            mcr = round(100. * delta_merid / s_merid_ref, ndigits=3)
            scr = round(100. * delta_sagit / s_sagit_ref, ndigits=3)
            msg += (
                f"\nConsidering a miscut angle (alpha = {alpha} rad):\n"
                f"\t-the meridonal focus at {mnp}m (delta = {mca}m / {mcr}%)\n"
                f"\t-the sagittal focus at {snp}m (delta = {sca}m / {scr}%)"
            )
        print(msg)

    return s_merid_ref, s_sagit_ref, s_merid_unp, s_sagit_unp


# ###############################################
#           Coordinates transforms
# ###############################################


def _checkformat_pts(pts=None):
    pts = np.atleast_1d(pts)
    if pts.ndim == 1:
        pts = pts.reshape((3, 1))
    if pts.shape[0] != 3 or pts.ndim < 2:
        msg = "pts must be a (3, ...) array of (X, Y, Z) coordinates!"
        raise Exception(msg)
    return pts


def checkformat_vectang(Z, nn, frame_cent, frame_ang):
    # Check / format inputs
    nn = np.atleast_1d(nn).ravel()
    assert nn.size == 3
    nn = nn / np.linalg.norm(nn)
    Z = float(Z)

    frame_cent = np.atleast_1d(frame_cent).ravel()
    assert frame_cent.size == 2
    frame_ang = float(frame_ang)

    return Z, nn, frame_cent, frame_ang


def get_e1e2_detectorplane(nn, nIn):
    e1 = np.cross(nn, nIn)
    e1n = np.linalg.norm(e1)
    if e1n < 1.e-10:
        e1 = np.array([nIn[2], -nIn[1], 0.])
        e1n = np.linalg.norm(e1)
    e1 = e1 / e1n
    e2 = np.cross(nn, e1)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2


# To be made cleaner vs option 0/ 1 => grid = True, False
def calc_xixj_from_braggphi(
    det_cent=None,
    det_nout=None, det_ei=None, det_ej=None,
    det_outline=None,
    summit=None, nout=None, e1=None, e2=None,
    bragg=None, phi=None,
    option=None, strict=None,
):
    """ Several options for shapes

    de_cent, det_nout, det_ei and det_ej are always of shape (3,)

    option:
        0:
            (summit, e1, e2).shape = (3,)
            (bragg, phi).shape = (nbragg,)
            => (xi, xj).shape = (nbragg,)
        1:
            (summit, e1, e2).shape = (3, nlamb, npts, nbragg)
            (bragg, phi).shape = (nlamb, npts, nbragg)
            => (xi, xj).shape = (nlamb, npts, nbragg)
    """
    # check inputs
    if strict is None:
        strict = True

    # Check option
    gdet = [det_cent, det_nout, det_ei, det_ej]
    g0 = [summit, nout, e1, e2]
    g1 = [bragg, phi]

    # check nbroadcastable
    _are_broadcastable(bragg=bragg, phi=phi)
    assert all([gg.shape == (3,) for gg in gdet]), "gdet no broadcast!"
    assert all([gg.shape == g0[0].shape for gg in g0]), "g0 no broadcast!"
    lc = [
        g0[0].size == 3 and g1[0].ndim == 1,
        g0[0].ndim in [4, 5] and g0[0].shape[0] == 3
        and phi.shape == g0[0].shape[1:],
    ]
    if np.sum(lc) == 0:
        lstr = [
            '\t- {}: {}'.format(kk, vv.shape)
            for kk, vv in [
                ('summit', summit), ('nout', nout), ('e1', e1), ('e2', e2),
                ('bragg', bragg), ('phi', phi),
            ]
        ]
        msg = (
            "Please provide either:\n"
            + "\t- option 0:\n"
            + "\t\t- (summit, nout, e1, e2).shape[0] = 3\n"
            + "\t\t- (bragg, phi).ndim = 1\n"
            + "\t- option 1:\n"
            + "\t\t- (summit, nout, e1, e2).ndim in [4, 5]\n"
            + "\t\t- (bragg, phi).shape[0] = 3\n\n"
            + "You provided:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)
    elif all(lc):
        msg = ("Multiple options!")
        raise Exception(msg)

    if option is None:
        option = lc.index(True)
    assert (lc[0] and option == 0) or (lc[1] and option == 1)

    if option == 0:
        summit = summit.ravel()
        nout, e1, e2 = nout.ravel(), e1.ravel(), e2.ravel()
        det_cent = det_cent[:, None]
        det_nout = det_nout[:, None]
        det_ei, det_ej = det_ei[:, None], det_ej[:, None]
        summit, nout = summit[:, None], nout[:, None],
        e1, e2 = e1[:, None], e2[:, None]
    else:
        det_cent = det_cent[:, None, None, None]
        det_nout = det_nout[:, None, None, None]
        det_ei = det_ei[:, None, None, None]
        det_ej = det_ej[:, None, None, None]
        if g0[0].ndim == 5:
            det_cent = det_cent[..., None]
            det_nout = det_nout[..., None]
            det_ei = det_ei[..., None]
            det_ej = det_ej[..., None]

    # Not necessary for broadcasting (last dims first)
    # bragg = bragg[None, ...]
    # phi = phi[None, ...]

    # Compute
    vect = (
        -np.sin(bragg)*nout
        + np.cos(bragg)*(np.cos(phi)*e1 + np.sin(phi)*e2)
    )
    k = np.sum(
        (det_cent-summit)*det_nout, axis=0
        ) / np.sum(vect*det_nout, axis=0)
    pts = summit + k[None, ...]*vect
    xi = np.sum((pts - det_cent)*det_ei, axis=0)
    xj = np.sum((pts - det_cent)*det_ej, axis=0)

    # Optional: eliminate points outside the det outline
    if det_outline is not None and strict is True:
        ind = (
            (xi < np.min(det_outline[0, :]))
            | (xi > np.max(det_outline[0, :]))
            | (xj < np.min(det_outline[1, :]))
            | (xj > np.max(det_outline[1, :]))
        )
        xi[ind] = np.nan
        xj[ind] = np.nan
    return xi, xj, strict

def calc_xixj_from_bragge2(
    det_cent=None,
    det_nout=None, det_ei=None, det_ej=None,
    det_outline=None,
    summit=None, nout=None, e1=None, e2=None,
    bragg=None,
    option=None, strict=None,
):
    '''
    ##### FLAT or CYLINDRICAL CRYSTAL

    calc_xixj_from_bragge2 is a function to calculate the points on a detector
    surface from Bragg reflection given local points and basis vectors on the 
    crystal surface when this reflrection can be parameterized as a np.pi/2
    rotation in the local vertical (e2) basis

    INPUTS:
        det_cent -      vector, dim (3,), detector summit coordinate
        det_nout -      vector, dim (3,), detector outward vector
        det_ei -        vector, dim (3,), detector horizontal vector
        det_ej -        vector, dim (3,), detector vertical vector
        det_outline -   OPTIONAL, matrix, dim (2,5), locations of detector corners
        summit -        matrix, dim (3, nlamb, npts, ndtheta), local points on crystal surface with vvalid Bragg reflections
        nout -          matrix, dim (3, nlamb, npts, ndtheta), outward vector on crystal surface
        e1 -            matrix, dim (3, nlamb, npts, ndtheta), horizontal vector on crystal surface
        e2 -            matrix, dim (3, nlamb, npts, ndtheta), vertical vector on crystal surface
        bragg -         array, dim (nlamb,), Bragg angles
        option -        OPTIONAL,
        strict -        OPTIONAL,
    
    OUTPUTS:
        xi -        matrix, dim (nlamb, npts, ndtheta), horizontal displacement from detector summit
        xj -        matrix, dim (nlamb, npts, ndtheta), vertical displacement from detector summit

    '''
    # Initializes output matrices
    xi = np.full(summit.shape[1:], np.nan) # dim (nlmab, npts, ndtheta)
    xj = np.full(summit.shape[1:], np.nan) # dim (nlamb, npts, ndtheta)

    # Loop over nlamb
    for ii in np.arange(summit.shape[1]):
        # Calculates rotation matrix for Bragg reflection about the crystal local vertical axis (e2)
        ang_rot = np.pi - 2*bragg[ii]
        c_rot, s_rot = np.cos(-ang_rot), np.sin(-ang_rot)
        Rot = np.array([[c_rot,0,-s_rot], [0,1,0], [s_rot,0,c_rot]]).reshape(3,3) # dim (3,3)

        # Calculates sine and cosine of Bragg angle
        cc, ss = np.cos(bragg[ii]), np.sin(bragg[ii]) 

        # Calculates the vector from a point on the crystal (d), towards a point with a valid Bragg reflrection (p), in the crystal local basis
        v_dp_loc = np.r_[-ss, 0, -cc] # dim (3,), basis (nout, e2, e1)

        # Loop over npts
        for jj in np.arange(summit.shape[2]):
            # Loop over ndtheta
            for kk in np.arange(summit.shape[3]):
                # Obtains the local basis vector
                nout_loc = nout[:,ii,jj,kk].reshape(1,3) # dim (1,3)
                e2_loc = e2[:,ii,jj,kk].reshape(1,3) # dim (1,3)
                e1_loc = e1[:,ii,jj,kk].reshape(1,3) # dim (1,3)

                # Projection matrix from the origin basis to the crystal local basis
                RR = np.concatenate((nout_loc, e2_loc, e1_loc), axis = 0) # dim (3,3)

                # Bragg reflection towards a point on the detector (x)
                v_dx_loc = np.matmul(Rot, v_dp_loc) # dim (3,), basis (nout, e2, e1)
                # Obviously, np.r_[-s, 0, c]

                # Projects reflection vector onto origin base
                v_dx = np.matmul(np.linalg.inv(RR), v_dx_loc) # dim (3,)
                v_dx = v_dx/np.linalg.norm(v_dx) # ensures normalized

                # Calculates the distance from the crystal local position to the detector plane
                dx = np.sum(
                    (det_cent-summit[:,ii,jj,kk])*det_nout, axis=0
                    ) / np.sum(v_dx*det_nout, axis=0)


                # Calculates the reflection position on the detector plane
                det_pt = summit[:,ii,jj,kk] + dx*v_dx

                # Calculates the horizontal, vertical displacement, in the detector basis, from the center center
                xi[ii,jj,kk] = np.sum((det_pt - det_cent)*det_ei, axis=0)
                xj[ii,jj,kk] = np.sum((det_pt - det_cent)*det_ej, axis=0)

    # Optional: eliminate points outside the det outline
    if det_outline is not None and strict is True:
        ind = (
            (xi < np.min(det_outline[0, :]))
            | (xi > np.max(det_outline[0, :]))
            | (xj < np.min(det_outline[1, :]))
            | (xj > np.max(det_outline[1, :]))
        )
        xi[ind] = np.nan
        xj[ind] = np.nan
    # Returns points on the detector
    return xi, xj, strict


def _calc_braggphi_from_pts_summits(
    pts=None,
    summits=None,
    vin=None, ve1=None, ve2=None,
):
    # check inputs
    _are_broadcastable(pts=pts, summits=summits, vin=vin, ve1=ve1, ve2=ve2)

    if pts.ndim != summits.ndim:
        msg = "Differents dimensions!"
        raise Exception(msg)

    # compute
    vect = pts - summits
    vect = vect / np.sqrt(np.sum(vect**2, axis=0))[None, ...]
    bragg = np.arcsin(np.sum(vect*vin, axis=0))
    if np.any(bragg < 0.):
        msg = (
            "There seems to be negative bragg angles!\n"
            + "  => double-check inputs!"
        )
        raise Exception(msg)
    phi = np.arctan2(np.sum(vect*ve2, axis=0), np.sum(vect*ve1, axis=0))
    return bragg, phi


def calc_braggphi_from_xixjpts(
    pts=None,
    xi=None, xj=None, det=None,
    summit=None, nin=None, e1=None, e2=None,
    grid=None,
):
    """ Return bragg phi for pts or (xj, xi) seen from (summit, nin, e1, e2)

    Either provide:
        pts => (3, npts)
        xi, xj => pts with shape (3, nxi, nxj)

    summit, nin, e1, e2 must have the same shape (3, nsumm)

    bragg.shape = (nsum, )

    if grid is True:
        all pts evaluated for all summ/nin
        return  (nsumm, npts) or (nsum, nxi, nxj) arrays
    else:
        each pts has a unique corresponding summ/nin (except possibly ndtheta)
        return (npts,) or (nxi, nxj) arrays
            or (npts, ndtheta) or (nxi, nxj, ndtheta) arrays
    """

    # --------------
    # Check format
    if grid is None:
        grid = True

    # Either pts or (xi, xj)
    lc = [pts is not None, all([xx is not None for xx in [xi, xj]])]
    if np.sum(lc) != 1:
        msg = "Provide either pts xor (xi, xj)!"
        raise Exception(msg)

    if lc[0]:
        # pts
        pts = _checkformat_pts(pts)

    elif lc[1]:

        # xi, xj => compute pts using det
        c0 = (
            isinstance(det, dict)
            and all([
                ss in det.keys() and len(det[ss]) == 3
                for ss in ['cent', 'ei', 'ej']
            ])
        )
        if not c0:
            msg = (
                "Arg det must be provided as a dict if xi, xj are provided!\n"
                + "Provided: {}".format(det)
            )
            raise Exception(msg)

        # pts from xi, xj
        xi, xj, (xii, xjj) = _checkformat_xixj(xi, xj)
        if xii.shape != xjj.shape:
            msg = "xi and xj must have the same shape!"
            raise Exception(msg)
        assert xii.ndim in [1, 2]

        if xii.ndim == 1:
            pts = (det['cent'][:, None]
                   + xii[None, :]*det['ei'][:, None]
                   + xjj[None, :]*det['ej'][:, None])
        else:
            pts = (det['cent'][:, None, None]
                   + xii[None, ...]*det['ei'][:, None, None]
                   + xjj[None, ...]*det['ej'][:, None, None])

    c0 = summit.shape == nin.shape == e1.shape == e2.shape
    if not c0:
        msg = "(summit, nin, e1, e2) must all have the same shape"
        raise Exception(msg)

    c0 = (
        (
            grid is True
            and pts.ndim in [1, 2, 3]
            and summit.ndim in [1, 2, 3, 4, 5]
        )
        or (
            grid is False
            and pts.ndim == summit.ndim
        )
    )
    if not c0:
        msg = (
            "Args pts and summit/nin/e1/e2 must be such that:\n"
            + "\t- grid = True:\n"
            + "\t\tpts.ndim in [1, 2, 3] and pts.shape[0] == 3\n"
            + "\t\tsummit.ndim in [1, 2, 3, 4, 5]\n"
            + "\t- grid = False:\n"
            + "\t\tpts can be directly broadcasted to summit (and same dim)\n"
            + "  You provided:\n"
            + "\t- grid: {}\n".format(grid)
            + "\t- pts.shape = {}\n".format(pts.shape)
            + "\t- summit.shape = {}".format(summit.shape)
        )
        raise Exception(msg)

    # --------------
    # Prepare
    # This part should be re-checked for all combinations!
    if grid is False:
        return _calc_braggphi_from_pts_summits(
            pts=pts,
            summits=summit,
            vin=nin, ve1=e1, ve2=e2,
        )

    else:
        # Typical dim
        # (3, npts, nlamb, ndtheta, 2)
        # (3, nxi, nxj, nlamb, ndtheta, 2)
        ptsdim = pts.ndim
        sumdim = summit.ndim
        if ptsdim == 2:
            if sumdim == 1:
                summit = summit[:, None]
                nin = nin[:, None]
                e1, e2 = e1[:, None], e2[:, None]
            else:
                summit = summit[:, None, ...]
                nin = nin[:, None, ...]
                e1, e2 = e1[:, None, ...], e2[:, None, ...]
        else:
            if sumdim == 1:
                summit = summit[:, None, None]
                nin = nin[:, None, None]
                e1, e2 = e1[:, None, None], e2[:, None, None]
            else:
                summit = summit[:, None, None, ...]
                nin = nin[:, None, None, ...]
                e1, e2 = e1[:, None, None, ...], e2[:, None, None, ...]
        if sumdim in [1, 2]:
            pts = pts[..., None]
        elif sumdim == 3:
            pts = pts[..., None, None]
        elif sumdim == 4:
            pts = pts[..., None, None, None]

        # --------------
        # Compute
        # Everything has shape (3, nxi0, nxi1, npts0, npts1) => sum on axis=0
        # or is broadcastable
        return _calc_braggphi_from_pts_summits(
            pts=pts,
            summits=summit,
            vin=nin, ve1=e1, ve2=e2,
        )


# ###############################################
#           lamb available from pts
# ###############################################


def _get_lamb_avail_from_pts_phidtheta_xixj(
    cryst=None,
    lamb=None,
    n=None,
    ndtheta=None,
    pts=None,
    miscut=None,
    return_phidtheta=None,
    return_xixj=None,
    strict=None,
    det=None,
):
    """

    Inputs
    ------
        pts = (3, npts) array
        lamb = (npts, nlamb) array

    Return
    ------
        lamb    (npts, nlamb)
        xi      (npts, nlamb, ndtheta, 2)   There can be 2 solutions
        xj      (npts, nlamb, ndtheta, 2)
        phi     (npts, nlamb, ndtheta, 2)
        dtheta  (npts, nlamb, ndtheta, 2)

    """

    keepon = return_phidtheta or return_xixj or strict

    if keepon:
        # Continue to get phi, dtheta...
        bragg = cryst._checkformat_bragglamb(lamb=lamb, n=n)

        # Compute dtheta / psi for each pts and lamb
        npts, nlamb = lamb.shape
        dtheta = np.full((npts, nlamb, ndtheta, 2), np.nan)
        psi = np.full((npts, nlamb, ndtheta, 2), np.nan)
        phi = np.full((npts, nlamb, ndtheta, 2), np.nan)

        if ndtheta is not None:
            ndpts = {}
            ndpts['ndtheta'] = ndtheta
        else:
            ndpts = None

        for ii in range(nlamb):
            #(
            #    dtheta[:, ii, :, :], psi[:, ii, :, :],
            #    phi[:, ii, :, :],
            #) = cryst._calc_dpts_from_lambpts(
            #    pts=pts, bragg=bragg[:, ii], lamb=None,
            #    n=n, ndpts=ndpts,#ndtheta=ndtheta,
            #    use_non_parallelism=use_non_parallelism,
            #    grid=False,
            #)[:3]
            cry_dpts, _ = cryst._calc_dpts_from_lambpts(
                pts=pts, bragg=bragg[:, ii], lamb=None,
                n=n,
                ndpts=ndpts,#ndtheta=ndtheta,
                miscut=miscut,
                grid=False,
            )

            dtheta[:, ii, :, :] = cry_dpts['dtheta']
            psi[:, ii, :, :] = cry_dpts['psi']
            phi[:, ii, :, :] = cry_dpts['phi']

        if return_xixj is True or strict is True:
            cry_dpts['dtheta'] = dtheta
            cry_dpts['psi'] = psi
            cry_dpts['phi'] = phi
            xi, xj, strict = cryst.calc_xixj_from_braggphi(
                #phi=phi + np.pi,    # from plasma to det
                bragg=bragg[..., None, None],
                n=n,
                #dtheta=dtheta,
                #psi=psi,
                cry_dpts=cry_dpts,
                det=det,
                data=None,
                miscut=miscut,
                strict=strict,
                return_strict=True,
                plot=False,
                dax=None,
            )
            if strict is True and np.any(np.isnan(xi)):
                indnan = np.isnan(xi)
                lamb[np.all(np.all(indnan, axis=-1), axis=-1)] = np.nan
                if return_phidtheta:
                    phi[indnan] = np.nan
                    dtheta[indnan] = np.nan
                    psi[indnan] = np.nan

    # -----------
    # return

    if return_phidtheta and return_xixj:
        return lamb, phi, dtheta, psi, xi, xj
    elif return_phidtheta:
        return lamb, phi, dtheta, psi
    elif return_xixj:
        return lamb, xi, xj
    else:
        return lamb


# ###############################################
#           2D spectra to 1D
# ###############################################


def get_lambphifit(lamb, phi, nxi, nxj):
    lambD = np.nanmax(lamb)-np.nanmin(lamb)
    lambfit = np.nanmin(lamb) + lambD*np.linspace(0, 1, nxi)
    phiD = np.nanmax(phi) - np.nanmin(phi)
    phifit = np.nanmin(phi) + phiD*np.linspace(0, 1, nxj)
    return lambfit, phifit


def _calc_spect1d_from_data2d(ldata, lamb, phi,
                              nlambfit=None, nphifit=None,
                              spect1d=None, mask=None,
                              vertsum1d=None):
    # Check / format inputs
    if spect1d is None:
        spect1d = 'mean'
    if isinstance(ldata, np.ndarray):
        ldata = [ldata]
    lc = [isinstance(spect1d, tuple) and len(spect1d) == 2,
          (isinstance(spect1d, list)
           and all([isinstance(ss, tuple) and len(ss) == 2
                    for ss in spect1d])),
          spect1d in ['mean', 'cent']]
    if lc[0]:
        spect1d = [spect1d]
    elif lc[1]:
        pass
    elif lc[2]:
        if spect1d == 'cent':
            spect1d = [(0., 0.2)]
            nspect = 1
    else:
        msg = ("spect1d must be either:\n"
               + "\t- 'mean': the avearge spectrum\n"
               + "\t- 'cent': the central spectrum +/- 20%\n"
               + "\t- (target, tol); a tuple of 2 floats:\n"
               + "\t\ttarget: the central value of the window in [-1,1]\n"
               + "\t\ttol:    the window tolerance (width) in [0,1]\n"
               + "\t- list of (target, tol)")
        raise Exception(msg)

    if not isinstance(nlambfit, int) or not isinstance(nphifit, int):
        msg = ("nlambfit and nphifit must be int!\n"
               + "\t- nlambfit provided: {}\n".format(nlambfit)
               + "\t- nphifit provided : {}\n".format(nphifit))
        raise Exception(msg)

    if vertsum1d is None:
        vertsum1d = True

    # Compute lambfit / phifit and spectrum1d
    if mask is not None:
        for ii in range(len(ldata)):
            ldata[ii][~mask] = np.nan
    lambfit, phifit = get_lambphifit(lamb, phi, nlambfit, nphifit)
    lambfitbins = 0.5*(lambfit[1:] + lambfit[:-1])
    ind = np.digitize(lamb, lambfitbins)

    # Get phi window
    if spect1d == 'mean':
        phiminmax = np.r_[phifit.min(), phifit.max()][None, :]
        spect1d_out = [np.array([np.nanmean(dd[ind == jj])
                                 for jj in np.unique(ind)])[None, :]
                       for dd in ldata]
    else:
        nspect = len(spect1d)
        dphi = np.nanmax(phifit) - np.nanmin(phifit)
        spect1d_out = [np.full((nspect, lambfit.size), np.nan)
                       for dd in ldata]
        phiminmax = np.full((nspect, 2), np.nan)
        for ii in range(nspect):
            phicent = np.nanmean(phifit) + spect1d[ii][0]*dphi/2.
            indphi = np.abs(phi - phicent) < spect1d[ii][1]*dphi
            for jj in np.unique(ind):
                indj = indphi & (ind == jj)
                if np.any(indj):
                    for ij in range(len(ldata)):
                        spect1d_out[ij][ii, jj] = np.nanmean(ldata[ij][indj])
            phiminmax[ii, :] = (np.nanmin(phi[indphi]),
                                np.nanmax(phi[indphi]))

    if vertsum1d is True:
        phifitbins = 0.5*(phifit[1:] + phifit[:-1])
        ind = np.digitize(phi, phifitbins)
        vertsum1d = [np.array([np.nanmean(dd[ind == ii])
                               for ii in np.unique(ind)])
                     for dd in ldata]
    if len(ldata) == 1:
        spect1d_out = spect1d_out[0]
        if vertsum1d is not False:
            vertsum1d = vertsum1d[0]
    return spect1d_out, lambfit, phifit, vertsum1d, phiminmax


# ###############################################
#           From plasma pts
# ###############################################


def calc_dthetapsiphi_from_lambpts(
    pts,
    bragg,
    summit=None, rcurve=None,
    nout=None, e1=None, e2=None,
    extenthalf=None,
    ndtheta=None,
    grid=None,
):
    """ 
    ###### SPHERICAL CRYSTAL
    
    Return (dtheta, psi) of pts on spherical crystal where bragg diffraction happens

    For given pts and lamb/bragg

    For each pts/lamb, there may be up to 2 arcs on the crystal
    Only returns valid solution (inside extenthalf), with nan elsewhere

    psi and dtheta returned as (nlamb, npts, 2, ndtheta) arrays

    Here nout, e1, e2 are at the unique crystal summit!

    """

    # Check input
    if ndtheta is None:
        ndtheta = 10

    npts = pts.shape[1]
    nlamb = bragg.size
    if grid is None:
        grid = True
    if grid is False:
        if nlamb != npts:
            msg = "If grid = False, lamb.shape should be (pts.shape[1],)"
            raise Exception(msg)

    # Prepare output
    if grid is True:
        scaPCem = np.full((nlamb, npts, 2), np.nan)
        dtheta = np.full((nlamb, npts, ndtheta, 2), np.nan)
        psi = np.full((nlamb, npts, ndtheta, 2), np.nan)
        num = np.full((nlamb, npts, ndtheta, 2), np.nan)
        angextra = np.full((nlamb, npts, ndtheta, 2), np.nan)
        dtheta_u = np.full((nlamb, npts, ndtheta), np.nan)
        psi_u = np.full((nlamb, npts, ndtheta), np.nan)
        sol1 = np.full((nlamb, npts), np.nan)
        sol2 = np.full((nlamb, npts), np.nan)
    else:
        scaPCem = np.full((npts, 2), np.nan)
        dtheta = np.full((npts, ndtheta, 2), np.nan)
        psi = np.full((npts, ndtheta, 2), np.nan)
        num = np.full((npts, ndtheta, 2), np.nan)
        angextra = np.full((npts, ndtheta, 2), np.nan)
        dtheta_u = np.full((npts, ndtheta), np.nan)
        psi_u = np.full((npts, ndtheta), np.nan)
        sol1 = np.full((npts,), np.nan)
        sol2 = np.full((npts,), np.nan)

    # Get to scalar product scaPCem
    # Already ok for miscut (via nout)
    center = summit - rcurve*nout
    PC = center[:, None] - pts
    PCnorm2 = np.sum(PC**2, axis=0)
    cos2 = np.cos(bragg)**2
    # PM.CM = Rsca + R**2  (ok)
    # PMCM = PMnR*sin       (ok)
    # PMn2 = PCn2*sin2 + 2Rsin2*sca + R2sin2
    #
    # sca**2 + 2Rcos2*sca + R2cos2 - PCnsin2 = 0
    if grid is True:
        deltaon4 = np.sin(bragg)[:, None]**2*(
            PCnorm2[None, :] - rcurve**2*cos2[:, None]
        )
    else:
        deltaon4 = np.sin(bragg)**2 * (PCnorm2 - rcurve**2*cos2)

    # Get two relevant solutions
    ind = deltaon4 >= 0.
    if grid is True:
        cos2 = np.repeat(cos2[:, None], npts, axis=1)[ind]
        PCnorm = np.tile(np.sqrt(PCnorm2), (nlamb, 1))[ind]
    else:
        cos2 = cos2[ind]
        PCnorm = np.sqrt(PCnorm2)[ind]
    sol1 = -rcurve*cos2 - np.sqrt(deltaon4[ind])
    sol2 = -rcurve*cos2 + np.sqrt(deltaon4[ind])

    # Only keep solution going outward sphere
    # scaPMem = scaPCem + rcurve >= 0
    ind1 = (sol1 >= -rcurve)
    ind2 = (sol2 >= -rcurve)

    sol1 = sol1[ind1]
    sol2 = sol2[ind2]
    if grid is True:
        indn = ind.nonzero()
        ind1 = [indn[0][ind1], indn[1][ind1]]
        ind2 = [indn[0][ind2], indn[1][ind2]]
        scaPCem[ind1[0], ind1[1], 0] = sol1
        scaPCem[ind2[0], ind2[1], 1] = sol2
    else:
        indn = ind.nonzero()[0]
        scaPCem[indn[ind1], 0] = sol1
        scaPCem[indn[ind2], 1] = sol2
    ind = ~np.isnan(scaPCem)

    # Get equation on PCem
    # CM = rcurve * (sin(dtheta)e2 + cos(dtheta)(cos(psi)nout + sin(psi)e1))
    # PC.eM = scaPCem, thus introducing Z = PC.e2, Y = PC.e1, X = PC.nout
    # Xcos(dtheta)cos(psi) + Ycos(dtheta)sin(psi) + Zsin(dtheta) = scaPCem
    # dtheta is specified, psi is deduced
    X = np.sum(PC*nout[:, None], axis=0)
    Y = np.sum(PC*e1[:, None], axis=0)
    Z = np.sum(PC*e2[:, None], axis=0)

    if grid is True:
        scaPCem = np.repeat(scaPCem[:, :, None, :], ndtheta, axis=2)
    else:
        scaPCem = np.repeat(scaPCem[:, None, :], ndtheta, axis=1)

    # broadcast and specify dtheta
    ind = ~np.isnan(scaPCem)
    if grid is True:
        XYnorm = np.repeat(
            np.repeat(
                np.repeat(
                    np.sqrt(X**2 + Y**2)[None, :], nlamb, axis=0,
                )[..., None],
                ndtheta,
                axis=-1,
            )[..., None],
            2,
            axis=-1,
        )[ind]
        Z = np.repeat(
            np.repeat(
                np.repeat(Z[None, :], nlamb, axis=0)[..., None],
                ndtheta,
                axis=-1,
            )[..., None],
            2,
            axis=-1,
        )[ind]
        # Define angextra to get
        # sin(psi + angextra) = (scaPCem - Z*sin(theta)) / (XYnorm*cos(theta))
        angextra[ind] = np.repeat(
            np.repeat(
                np.repeat(np.arctan2(X, Y)[None, :], nlamb, axis=0)[..., None],
                ndtheta, axis=-1)[..., None],
            2, axis=-1)[ind]
        dtheta[ind] = np.repeat(
            np.repeat(
                np.repeat(
                    extenthalf[1]*np.linspace(-1, 1, ndtheta)[:, None],
                    2, axis=1)[None, ...],
                npts, axis=0)[None, ...],
            nlamb, axis=0)[ind]
    else:
        XYnorm = np.repeat(
            np.repeat(np.sqrt(X**2+Y**2)[:, None], ndtheta, axis=1)[..., None],
            2,
            axis=-1,
        )[ind]
        Z = np.repeat(
            np.repeat(Z[:, None], ndtheta, axis=1)[..., None],
            2,
            axis=-1,
        )[ind]
        angextra[ind] = np.repeat(
            np.repeat(np.arctan2(X, Y)[:, None], ndtheta, axis=-1)[..., None],
            2,
            axis=-1,
        )[ind]
        dtheta[ind] = np.repeat(
            np.repeat(
                extenthalf[1]*np.linspace(-1, 1, ndtheta)[:, None],
                2,
                axis=1,
            )[None, ...],
            npts,
            axis=0,
        )[ind]

    num[ind] = (
        (scaPCem[ind] - Z*np.sin(dtheta[ind])) / (XYnorm*np.cos(dtheta[ind]))
    )
    ind[ind] = np.abs(num[ind]) <= 1.
    # minus psi ref ?
    # value of psi + angextra with cos > 0
    psiang_pos = np.arcsin(num[ind])
    psi1 = np.full(psi.shape, np.nan)
    psi2 = np.full(psi.shape, np.nan)
    psi1[ind] = (np.arctan2(num[ind], np.cos(psiang_pos)) - angextra[ind])
    psi2[ind] = (np.arctan2(num[ind], -np.cos(psiang_pos)) - angextra[ind])

    # Make sure only one of the 2 psi is correct
    ind1 = np.copy(ind)
    ind2 = np.copy(ind)
    ind1[ind] = np.abs(psi1[ind]) <= extenthalf[0]
    ind2[ind] = np.abs(psi2[ind]) <= extenthalf[0]
    assert not np.any(ind1 & ind2), "Multiple solutions for psi!"
    ind = ind1 | ind2

    # Finally store into psi
    psi[ind1] = psi1[ind1]
    psi[ind2] = psi2[ind2]
    psi[~ind] = np.nan
    dtheta[~ind] = np.nan
    if np.any(np.sum(ind, axis=-1) == 2):
        msg = (
            "\nDouble solutions found for {} / {} points!".format(
                np.sum(np.sum(ind, axis=-1) == 2),
                np.prod(ind.shape[:-1]),
            )
        )
        warnings.warn(msg)

    return dtheta, psi, ind, grid

def calc_dthetade1_from_lambpts(
    pts,
    bragg,
    summit=None, rcurve=None,
    nout=None, e1=None, e2=None,
    extenthalf=None,
    ndtheta=None,
    grid=None,
    ):

    '''
    ##### CYLINDRICAL CRYSTAL

    Return (dtheta, de1) of pts on cylindrical crystal, relative to the summit, 
    where bragg diffraction happens for given pts and lamb/bragg

    de1 is the horizontal direction (e1), dtheta is the vertical direction (e2)

    For each pts/lamb, there is 1 sol'n for a valid triangle at the Bragg angle
    Only returns valid solution (inside extenthalf), with nan elsewhere

    de1 and dtheta returned as (nlamb, npts, ndtheta) arrays

    Here nout, e1, e2 are at the unique crystal summit!

    INPUTS:
        pts -       vector, dim (3, npts), Cartesian coordinate of point source
        bragg -     vector, dim (1, nlamb), wavelengths to map onto crystal
        summit -    vector, dim (3,1), mid-point of crystal
        rcurve -    scalar, radius of curvature
        nout -      vector, dim (1,3), normal vector out of crystal summit
        e1 -        vector, dim (1,3), horizontal vector of crystal summit
        e2 -        vector, dim (1,3), vertical vector of crystal summit
        extenthalf- array, dim (2,), half-extent from crystal summit in the (horizontal, vertical) direction
        ndtheta -   OPTIONAL, scalar, number of vertical points on the crystal surface to uniformly scan
        grid -      ############

    OUTPUTS:
        dtheta -    matrix, dim (nlamb, npts, ndtheta), e2-angle on crystal surface scanned over
        de1 -       matrix, dim (nlamb, npts, ndtheta), unique e1-location on crystal surface where Bragg diffraction happens
        dphi -      matrix, dim (nlamb, npts, ndz), angle made between the local horizontal axis and ray
        ind -       OPTIONAL, matrix, dim (nlamb, npts), indices where Bragg reflection happened outside crystal extent
        grid -      ###########

    '''

    # Check input for number of vertical points to scan
    if ndtheta is None:
        # Defaults is 10
        ndtheta = 10

    # Number of points to scan over
    npts = pts.shape[1]

    # Number of wavelength to scan over
    nlamb = bragg.size

    ########
    if grid is None:
        grid = True
    if grid is False:
        if nlamb != npts:
            msg = "If grid = False, lamb.shape should be (pts.shape[1],)"
            raise Exception(msg)

    # Prepare output
    if grid is True:
        de1 = np.full((nlamb,npts, ndtheta), np.nan) # dim (nlamb,npts, ndtheta), horizontal points on the crystal surface
        dtheta = np.full((nlamb,npts, ndtheta), np.nan) # dim (nlamb,npts, ndtheta), vertical points on the crystal surface
        dphi = np.full((nlamb,npts, ndtheta), np.nan) # dim (nlamb,npts, ndtheta), angle made between the local horizontal axis and ray


    else:
        de1 = np.full((npts, ndtheta), np.nan) # dim (npts, ndtheta), horizontal points on the crystal surface
        dtheta = np.full((npts, ndtheta), np.nan) # dim (npts,ndtheta), vertical points on the crystal surface
        dphi = np.full((npts, ndtheta), np.nan) # dim (nlamb,npts, ndz), angle made between the local horizontal axis and ray

    # Makes sure the shape is correct for vectors
    nout = nout.reshape(1,3) # dim (1,3)
    e1 = e1.reshape(1,3) # dim (1,3)
    e2 = e2.reshape(1,3) # dim (1,3)
    summit = summit.reshape(3,1) # dim (3,1)
    center = summit - rcurve*np.transpose(nout) # dim (3,1)

    # Note that the cylinder axis is parameterized as center + de1*\hat{e1}
    # detheta is an angle measured between the e2 and nout directions (in plane)

    # Calculates inverse tangent of bragg angle
    tb = np.tan(bragg.reshape(1,nlamb)) # dim (1,nlamb)

    # Vertical extent to scan over
    dtheta_1 = extenthalf[1]*np.linspace(-1,1,ndtheta) # dim (ndtheta,)

    # Calculates the distance from each point to the summit
    PC = np.transpose(pts-center) # dim (npts, 3)

    # Function for the angle between the source ray and crystal summit
    import scipy.optimize
    def func(x, *args):
        PC = args[0] # dim (3,), need floats
        nout = args[1][0] # dim (3,)
        e1 = args[2][0] # dim (3,)
        e2 = args[3][0] # dim (3,)
        rcurve = args[4] # dim (1,)
        tb = args[5] # dim (1,)
        return np.dot(PC, np.transpose(1/(tb*np.cos(x))*nout-e2*(np.tan(x)+1/np.tan(x)))) - rcurve*1/(tb*np.cos(x))

    # Loop over Bragg angles
    for ii in np.arange(nlamb):
        # Loop over points
        for jj in np.arange(npts):
            # Loop over vertical positions
            for kk in np.arange(ndtheta):
                # Cosine and sine for this rotation
                cc, ss = np.cos(dtheta_1[kk]), np.sin(dtheta_1[kk]) # scalars

                # Calculates the local basis vectors
                e1_local = e1 # dim (1,3)
                e2_local = -1*nout*ss + e2*cc # dim (1,3)
                nout_local = nout*cc + e2*ss # dim (1,3)

                # Angles between ray and summit basis
                phi_sgn = np.sign(np.dot(-1*PC[jj,:], np.transpose(e2_local)))
                if phi_sgn[0]==0:
                    phi_sgn[0]=1
                phi = scipy.optimize.fsolve(func, phi_sgn[0]*0.1, args=(PC[jj,:],nout_local, e1_local, e2_local, rcurve, tb[ii])) # dim(1,), [rad]

                # Calculates a modified vector from the crystal basis
                vect = nout/(np.cos(phi)*tb[ii]) - e1_local - np.tan(phi)*e2_local

                # Solves for the valid Bragg reflection along the crystal e1 axis
                d1 = np.dot(PC[jj,:], np.transpose(vect)) - rcurve/(np.cos(phi)*tb[ii])  # dim (1,)

                # Other equation for d1 that has divide by zero problems
                #vect0 = (1/np.tan(phi))*e2_local - e1_local # dim (1,3)
                #d10 = np.dot(PC[jj,:], np.transpose(vect))

                #ddd = center - d1*np.transpose(e1_local) + rcurve*np.transpose(nout_local)
                #v_dp = pts[:,jj]- np.transpose(ddd)
                #err1 = np.dot(v_dp, np.transpose(e2_local))/np.dot(v_dp, np.transpose(e1_local))
                #err12 = np.tan(phi)
                #q = np.cos(phi)*e1_local + np.sin(phi)*e2_local
                #err2 = np.dot(v_dp, np.transpose(nout_local))/np.dot(v_dp, np.transpose(q))
                #err22 = tb[ii]

                # Finds indices where reflection is not on the cystal's horizontal axis
                ind = np.where(np.abs(d1)>=extenthalf[0])
                d1[ind] = np.nan

                # Matrix to handle vertical displacement
                mat = np.ones(d1.shape)
                mat[ind] = np.nan

                # Populates de1 at each vertical location
                de1[ii,jj,kk] = d1

                # Populates de2 at each vertical location
                dtheta[ii,jj,kk] = dtheta_1[kk]*mat

                # Populates the dphi at each vertical location
                dphi[ii,jj,kk] = phi*mat

    # Returns variables defining location of Bragg reflection
    return dtheta, de1, dphi, ind, grid



def calc_de1de2_from_lambpts(
    pts,
    bragg,
    summit=None,
    nout=None, e1=None, e2=None,
    extenthalf=None,
    ndz=None,
    grid=None,
    ):

    '''
    ##### FLAT CRYSTAL

    Return (de1, de2, dphi) of pts on flat crystal, relative to the summit, 
    where bragg diffraction happens for given pts and lamb/bragg

    de1 is the horizontal direction (e1), de2 is the vertical direction (e2)

    For each pts/lamb, there is 1 sol'n for a valid triangle at the Bragg angle
    Only returns valid solution (inside extenthalf), with nan elsewhere

    de1, de2, and dphi returned as (npts, nlamb, ndz) arrays

    Here nout, e1, e2 are at the unique crystal summit!

    INPUTS:
        pts -       vector, dim (3, npts), Cartesian coordinate of point source
        bragg -     vector, dim (nlamb,1), wavelengths to map onto crystal
        summit -    vector, dim (3,1), mid-point of crystal
        nout -      vector, dim (3,1), normal vector out of crystal summit
        e1 -        vector, dim (3,1), horizontal vector of crystal summit
        e2 -        vector, dim (3,1), vertical vector of crystal summit
        extenthalf- array, dim (2,), half-extent from crystal summit in the (horizontal, vertical) direction
        ndz -       OPTIONAL, scalar, number of vertical points on the crystal surface to uniformly scan
        grid -      ############

    OUTPUTS:
        de1 -       matrix, dim (nlamb, npts, ndz), unique e1-location on crystal surface where Bragg diffraction happens
        de2 -       matrix, dim (nlamb, npts, ndz), e2-location on crystal surface scanned over
        dphi -      matrix, dim (nlamb, npts, ndz), angle made between the local horizontal axis and ray
        ind -       OPTIONAL, matrix, dim (nlamb, npts), indices where Bragg reflection happened outside crystal extent
        grid -      ###########

    '''

    # Check input for number of vertical points to scan
    if ndz is None:
        # Defaults is 10
        ndz = 10

    # Number of points to scan over
    npts = pts.shape[1]

    # Number of wavelength to scan over
    nlamb = bragg.size

    ########
    if grid is None:
        grid = True
    if grid is False:
        if nlamb != npts:
            msg = "If grid = False, lamb.shape should be (pts.shape[1],)"
            raise Exception(msg)

    # Prepare output
    if grid is True:
        de1 = np.full((nlamb,npts, ndz), np.nan) # dim (nlamb,npts, ndz), horizontal points on the crystal surface
        de2 = np.full((nlamb,npts, ndz), np.nan) # dim (nlamb,npts, ndz), vertical points on the crystal surface
        dphi = np.full((nlamb,npts, ndz), np.nan) # dim (nlamb,npts, ndz), angle made between the local horizontal axis and ray

    else:
        de1 = np.full((npts, ndz), np.nan) # dim (npts, ndz), horizontal points on the crystal surface
        de2 = np.full((npts, ndz), np.nan) # dim (npts,ndz), vertical points on the crystal surface
        dphi = np.full((npts, ndz), np.nan) # dim (nlamb,npts, ndz), angle made between the local horizontal axis and ray

    # Makes sure the shape is correct for vectors
    nout = nout.reshape(1,3) # dim (1,3)
    e1 = e1.reshape(1,3) # dim (1,3)
    e2 = e2.reshape(1,3) # dim (1,3)
    summit = summit.reshape(3,1) # dim (3,1)

    # Calculates inverse tan of the Bragg angle
    tb = 1/np.tan(bragg.reshape(1,nlamb)) # dim (1,nlamb)

    # Calculates the distance from each point to the summit
    PS = np.transpose(pts-summit) # dim (npts, 3)

    # Function for the angle between the source ray and crystal summit
    import scipy.optimize
    def func(x, *args):
        PS = args[0] # dim (3,), need floats
        nout = args[1][0] # dim (3,)
        e1 = args[2][0] # dim (3,)
        e2 = args[3][0] # dim (3,)
        d2 = args[4] # dim (1,)
        tb = args[5] # dim (1,)
        return np.dot(PS, np.transpose(1/(tb*np.cos(x))*nout-e2*(np.tan(x)+1/np.tan(x)))) + d2*(np.tan(x)+1/np.tan(x))

    # Loop over Bragg angles
    for ii in np.arange(nlamb):
        # Loop over number of points
        for jj in np.arange(npts):
            # Calculates the d2 bounds
            d2_pi2 = -1*np.dot(PS[jj,:], np.transpose(1/tb[ii]*nout-e2))
            d2_npi2 = np.dot(PS[jj,:], np.transpose(1/tb[ii]*nout+e2))
            d2_top = np.min([extenthalf[1], d2_pi2[0]])
            d2_bot = np.max([-extenthalf[1], d2_npi2[0]])

            # Array of vertical locations to scan over
            d2 = np.linspace(d2_bot,d2_top,ndz) # [m], dim (ndz,)

            # Dummy check to avoid problems with zeros
            ind_d2 = np.where(np.abs(d2- np.dot(e2, pts[:,jj])[0]) < 1e-3)
            d2[ind_d2] = d2[ind_d2]*1.01

            # Loop over the scanned vertical distance
            for kk in np.arange(ndz):
                # Angles between ray and summit basis
                phi_sgn = np.sign(np.dot(-1*PS[jj,:], np.transpose(e2))+ d2[kk])
                phi = scipy.optimize.fsolve(func, phi_sgn[0]*0.1, args=(PS[jj,:],nout, e1,e2, d2[kk], tb[ii])) # dim(1,), [rad]

                # Calculates a modified vector from the crystal basis
                vect = (1/np.tan(phi))*e2 - e1 # dim (3,1)

                # Solves for the valid Bragg reflection along the crystal e1 axis
                d1 = np.dot(PS[jj,:], np.transpose(vect)) - d2[kk]*1/np.tan(phi) # dim (1,)

                # Finds indices where reflection is not on the cystal's horizontal axis
                ind = np.where(np.abs(d1)>=1.01*extenthalf[0])

                # Checks to make sure reflection is on the crystal's horizontal axis
                d1[ind] = np.nan

                # Mask to handle vertical displacement
                mat = np.ones(d1.shape)
                mat[ind] = np.nan

                # Populates de1 at each vertical location
                de1[ii,jj,kk] = d1

                # Populates de2 at each vertical location
                de2[ii,jj,kk] = d2[kk]*mat

                # Populates the dphi at each vertical location
                dphi[ii,jj,kk] = phi*mat

    # Returns variables defining location of Bragg reflection
    return de1, de2, dphi, ind, grid
