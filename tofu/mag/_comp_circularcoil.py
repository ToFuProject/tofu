
import scipy.constants as scpct
import numpy as np


###############################################################################
###############################################################################
#               Magnetic field from 3d cicular spires
#                       Check inputs
###############################################################################


def _check_inputs_spire(rad=None, cent=None, axis=None):
    """ Check conformity of inputs for defining 3d circles """

    # Basic check (should be np.ndarray)
    try:
        rad = np.ateast_1d(rad).ravel()
        cent = np.atleast_2d(cent)
        axis= np.atleast_2d(axis)
        if 3 not in cent.shape or 3 not in axis.shape:
            raise Exception
    except Exception as err:
        msg = ("Arg rad, cent and axis must be convertible to respectively:\n"
               + "\t- a 1d np.ndarray\n"
               + "\t- a 2d np.ndarray with 3 in shape\n"
               + "\t- a 2d np.ndarray with 3 in shape\n"
               + "Provided:\n"
               + "\t- rad: {}".format(rad)
               + "\t- cent: {}\n".format(cent)
               + "\t- axis: {}\n".format(axis)
              )
        raise Exception(msg)

    # Transpose if necessary
    if cent.shape[0] != 3:
        cent = cent.T
    if axis.shape[0] != 3:
        axis = axis.T

    # Check all are either unique or the good shape
    nspire = max(rad.size, cent.shape[1], axis.shape[1])
    c0 = (rad.size in [1, nspire]
          and cent.shape[1] in [1, nspire]
          and axis.shape[1] in [1, spire])
    if not c0:
        msg = ()
        raise Exception(msg)

    # Complement if necessary
    if rad.size < nspire:
        rad = np.repeat(rad, nspire)
    if cent.shape[1] < nspire:
        cent = np.repeat(cent, nspire, axis=1)
    if axis.shape[1] < nspire:
        axis = np.repeat(axis, nspire, axis=1)

    # Make sure axis is normalized
    axis = axis / np.sqrt(np.sum(axis**2, axis=0))[None, :]

    return rad, cent, axis


def _check_inputs_spire_2d(rad=None, cent_Z=None):
    """ Check conformity of inputs for defining axisymmetric circles """

    # Basic check (should be np.ndarray)
    try:
        rad = np.ateast_1d(rad).ravel()
        cent_Z = np.atleast_1d(cent_Z).ravel()
    except Exception as err:
        msg = ("Arg rad, cent_Z must be convertible to respectively:\n"
               + "\t- a 1d np.ndarray\n"
               + "\t- a 1d np.ndarray\n"
               + "Provided:\n"
               + "\t- rad: {}".format(rad)
               + "\t- cent_Z: {}\n".format(cent_Z)
              )
        raise Exception(msg)

    # Check all are either unique or the good shape
    nspire = max(rad.size, cent_Z.size)
    c0 = (rad.size in [1, nspire]
          and cent_Z.size in [1, nspire])
    if not c0:
        msg = ('Arg rad and cent_Z must be of size 1 or nspire:\n'
               + '\t- rad.size: {}\n'.format(rad.size)
               + '\t- cent_Z.size: {}'.format(cent_Z.size))
        raise Exception(msg)

    # Complement if necessary
    if rad.size < nspire:
        rad = np.repeat(rad, nspire)
    if cent_Z.size < nspire:
        cent_Z = np.repeat(cent_Z, nspire)

    return rad, cent_Z


def _check_inputs_nn(nn=None):
    """ Check nn is an integer >= 1 """

    # Set default if not provided
    if nn is None:
        nn = 1

    # Check
    try:
        nn = int(nn)
        if nn < 1:
            raise Exception
    except Exception as err:
        msg = ("Arg nn must be an integer >= 1\n"
               + "Provided: {}".format(nn))
        raise Exception(msg)
    return nn


def _check_inputs_constraint(rad=None, nn=None, constraint=None):
    """ Check the constraint to be used for spire discretization """
    if constraint is None:
        constraint = 'B0'
    dok = {'mag': 'same magnetic field at spire center',
           'perimeter': 'same perimeter',
           'area': 'same area'}
    if constraint not in dok.keys():
        ls = ['\t- {}: {}'.format(k0, v0) for k0, v0 in dok.items()]
        msg = ("Arg constraint must be either:\n"
               + "\n".join(ls)
               + "\n  You provided: {}".format(constraint))
        raise Exception(msg)
    return constraint


def _check_inputs_pts(pts=None):
    """ Check pts contains the 3D coordinates of points """

    try:
        pts = np.atleast_2d(pts)
        if 3 not in pts.shape:
            raise Exception
    except Exception as err:
        msg = ("Arg pts must be 2d (3, N) np.ndarray\n"
               + "Provided: {}".format(pts))
        raise Exception(msg)

    if pts.shape[0] != 3:
        pts = pts.T
    return pts


def _check_inputs_ptsRZ(ptsRZ=None):
    """ Check pts contains the 2D (R, Z) coordinates of points """

    try:
        ptsRZ = np.atleast_2d(ptsRZ)
        if 2 not in ptsRZ.shape:
            raise Exception
    except Exception as err:
        msg = ("Arg ptsRZ must be 2d (2, N) np.ndarray\n"
               + "Provided: {}".format(ptsRZ))
        raise Exception(msg)

    if ptsRZ.shape[0] != 2:
        ptsRZ = ptsRZ.T
    return ptsRZ


def _check_inputs_I(I=None, nspire=None):

    if I is None:
        I = 1.

    try:
        I = np.atleats_1d(I).ravel()
        if I.size != 1 and I.size != nspire:
            raise Exception
    except Exception as err:
        msg = ""
        raise Exception(msg)

    if I.size < nspire:
        I = np.repeat(I, spire)

    return I


###############################################################################
###############################################################################
#               Magnetic field from 3d cicular spires
#                       Check inputs
###############################################################################


def _get_hLcossin(rad=None, nn=None, constraint=None):
    """ Get parameters of discretized spires

    h = base height of the polygons (one for each spire) 
    Lhalf = half-length of the polygons outer side (one for each spire)
    cos, sin = cosinus and sinus of the half-circle (of set for all)
    """

    # Check inputs
    constraint = _check_inputs_constraint(constraint)
    nn = _check_inputs_nn(nn=nn)

    # Get cos, sin
    # dimensions: [discret]
    dtheta = np.pi/(2.*nn)
    theta = dtheta/2. + np.arange(0, 2*nn)*dtheta
    cos = np.cos(theta)
    sin = np.sin(theta)

    # Base angle (unique)
    ang = np.pi/(4*nn)
    tan = np.tan(ang)

    # Get h from constraint (one for each spire)
    # dimension: [spire]
    if constraint == 'perimeter':
        h = rad * ang / tan
    elif constraint == 'area':
        h = rad * np.sqrt(ang/tan)
    elif constraint == 'mag':
        h = rad * (tan/ang) / np.sqrt(tan**2 + 1)

    # Get Lhalf from h
    # dimension: [spire]
    Lhalf = h*np.tan(ang)

    return h, Lhalf, cos, sin


def get_B_3d(rad=None, cent=None, axis=None,
             pts=None, nn=None, constraint=None, I=None,
             returnas=None):
    """ Return the 3d field generated at pts from a set of circular coils

    pts are defined by their 3d cartesian coordinates
    coils are defined by their:
        - centers: 3d cartesian coordinates
        - unit vector axis: 3d cartesian coordinates
        - radii: positive
    Each coil can have a different center / axis / radius
    When a single value is provided, it is assumed identical for all coils

    Computation is done by discretizing circular coils as 4nn-sided polygons
    Where nn is set by the user (>=1)
    Disctreziation is done following a constraint:
        - 'perimeter':  same perimeter as the circle
        - 'area':       same area as the circle
        - 'mag':        same magnetic field on its center
    In practice 'area' or 'mag' are the most relevant

    The total magnetic field is returned as a np.ndarray of 3 dimensions:
        [(X, Y, Z) coordinates, npts, ncoils]

    If returnas is set to 'sum', instead of returning the detail for each coil,
    only the total field is returned with dimensions:
        [(X, Y, Z) coordinates, npts]

    """

    # Check inputs
    rad, cent, axis = _check_inputs_spire(rad=rad, cent=cent, axis=axis)
    pts = _check_inputs_pts(pts=pts)

    # Check I and broadcast for future uses
    # dimensions for h, Lhalf: [spire] => [coords, pts, spires, discret]
    I = _check_inputs_I(I=I, nspire=rad.size)[None, None, :, None]

    # Get local coordinates and unit vectors for spire / pts
    # Broadcasted for future use => [coords, pts, spires, discret]
    # dimensions for e1: [coords, pts, spire]
    # dimensions for r, z: [pts, spire]
    axis = axis[:, None, :, None]
    CM = (pts[:, :, None, None] - cent[:, None, :, None])
    z = np.sum(CM*axis, axis=0, keepdims=True)
    e1 = CM - z*axis
    e1 = e1 / np.sqrt(np.sum(e1**2, axis=0, keepdims=True))
    r = np.sum(CM*e1, axis=0, keepdims=True)

    # Get computation intermediate: h, Lhalf, cos, sin, and broadcast 
    # dimensions for h, Lhalf: [spire]
    # dimensions for cos, sin: [discret]
    h, Lhalf, cos, sin = _get_hLcossin(rad=rad, nn=nn, constraint=constraint)
    h, Lhalf = h[None, None, :, None], Lhalf[None, None, :, None]
    cos, sin = cos[None, None, None, :], sin[None, None, None, :]

    # Get vectors for segment pairs (one per pair per spire per pts)
    # dimensions: [coords, pts, spires, discret]
    vect = (z*cos*e1 + (h - r*cos)*axis)

    # Get 2 symmetric terms in parenthesis in Bi
    # dimensions: [pts, spires, discret]
    alphai = Lhalf**2 + h**2 + r**2 + z**2 - 2.*h*r*cos
    rsin = r*sin
    L2rsin = 2.*Lhalf*rsin
    terms_sum = ((Lhalf + rsin) / np.sqrt(alphai + L2rsin)
                 + (Lhalf - rsin) / np.sqrt(alphai - L2rsin))
    const_inv_rMi2 = scpct.mu_0/(2.*np.pi) / (z**2 + (h-r*cos)**2)

    # Get field for segment pairs and sum to get field per pts and spire
    # dimensions of B: [coords, pts, spires]
    B = np.sum(I * const_inv_rMi2 * terms_sum * vect, axis=-1)

    # Sum on discretization and optionally on spires
    if returnas == 'sum':
        return np.sum(B, axis=-1)
    else:
        return B


def get_B_2d_RZ(rad=None, cent_Z=None,
                pts_RZ=None, nn=None, constraint=None,
                returnas=None):
    """ Simplified version of get_B_3d() in axisymmetric configuration

    Here, all coils are supposed to be centered on the (O, z) axis
    Only their radii and height can be varied
    pts are only passed via their (R, Z) coordinates due to axisymmetry

    Due to axisymmetry, the magnetic field is returned using its cylindrical
    coordinates (R, Z):
        [(R, Z) coordinates, npts, ncoils]
    Unless returnas = 'sum' in which case it is summed on all coils:
        [(R, Z) coordinates, npts]

    """
    # Check inputs
    rad, cent_Z = _check_inputs_spire_2d(rad=rad, cent_Z=cent_Z)
    pts_RZ = _check_inputs_ptsRZ(pts_RZ=pts_RZ)

    # Check I and broadcast for future uses
    # dimensions for h, Lhalf: [spire] => [coords, pts, spires, discret]
    I = _check_inputs_I(I=I, nspire=rad.size)[None, None, :, None]

    # Get local coordinates and unit vectors for spire / pts
    # Broadcasted for future use => [coords, pts, spires, discret]
    # dimensions for e1: [coords, pts, spire]
    # dimensions for r, z: [pts, spire]
    z = ptsRZ[1:2, :, None, None] - cent_Z[None, None, :, None]
    r = ptsRZ[0:1, :, None, None]

    # Get computation intermediate: h, Lhalf, cos, sin, and broadcast 
    # dimensions for h, Lhalf: [spire]
    # dimensions for cos, sin: [discret]
    h, Lhalf, cos, sin = _get_hLcossin(rad=rad, nn=nn, constraint=constraint)
    h, Lhalf = h[None, None, :, None], Lhalf[None, None, :, None]
    cos, sin = cos[None, None, None, :], sin[None, None, None, :]

    # Get vectors for segment pairs (one per pair per spire per pts)
    # dimensions: [coords, pts, spires, discret]
    vect = (z*cos*np.r_[1, 0][:, None, None, None]
            + (h - r*cos)*np.r_[0, 1][:, None, None, None])

    # Get 2 symmetric terms in parenthesis in Bi
    # dimensions: [pts, spires, discret]
    alphai = Lhalf**2 + h**2 + r**2 + z**2 - 2.*h*r*cos
    rsin = r*sin
    L2rsin = 2.*Lhalf*rsin
    terms_sum = ((Lhalf + rsin) / np.sqrt(alphai + L2rsin)
                 + (Lhalf - rsin) / np.sqrt(alphai - L2rsin))
    const_inv_rMi2 = scpct.mu_0/(2.*np.pi) / (z**2 + (h-r*cos)**2)

    # Get field for segment pairs and sum to get field per pts and spire
    # dimensions of B: [coords, pts, spires]
    B = np.sum(I * const_inv_rMi2 * terms_sum * vect, axis=-1)

    # Sum on discretization and optionally on spires
    if returnas == 'sum':
        return np.sum(B, axis=-1)
    else:
        return B


###############################################################################
###############################################################################
#               3D circular coils - DEPRECATED
###############################################################################


def discretize_circles_check(cent, rad, axis=None,
                             nn=None, pts=None):
    """ Check conformity of inputs for discretize_circles()  """
    c0 = isinstance(cent, np.ndarray) and cent.ndim == 2 and 3 in cent.shape
    if not c0:
        cent = np.atleast_2d(cent)
        if 3 not in cent.shape:
            msg = ""
            raise Exception(msg)
    if cent.shape[0] != 3:
        cent = cent.T

    if axis is None:
        pass

    if pts is None:
        pts = (0., 0., 0.)

    return cent, rad, axis, nn, pts

def discretize_circles(cent, rad,
                       axis=None, nn=None, pts=None,
                       constraint=None):
    """ Discretize circles into 4n-sided polygon with symmetry plane at pts """

    # Check inputs
    cent, rad, axis, nn, pts = discretize_circles_check(cent, rad, axis=axis,
                                                        nn=nn, pts=pts)

    ncirc = cent.shape[1]
    npts = pts.shape[1]

    # Get heights
    pin = np.pi/(4*nn)
    tanpin = np.tan(pin)
    lconst = ['radius', 'perimeter', 'area', 'field']
    if constraint == lconst[0]:
        C = 1.
    elif constraint == lconst[1]:
        C = pin / tanpin
    elif constraint == lconst[2]:
        C = np.sqrt(pin / tanpin)
    elif constraint == lconst[3]:
        C = (tanpin/pin) / np.sqrt(1 + tanpin**2)
    else:
        msg = ("constraint not recognized:\n"
               + "\t- available: {}\n".format(lconst)
               + "\t- provided: {}".format(constraint))
        raise Exception(msg)
    h = rad * C

    # Get segments 
    phi_ref = None
    dphi = np.pi/4./nn
    phi = dphi_half + np.linspace(0. )
    CM = pts - cent
    CM = CM - np.sum(CM*axis)*axis
    e1 = CM / np.sqrt(np.sum(CM**2, axis=0))
    e2 = np.array([axis[1, ...]*e1[2,...] - axis[2, ...]*e1[1,...],
                   axis[2, ...]*e1[0,...] - axis[0, ...]*e1[2,...],
                   axis[0, ...]*e1[1,...] - axis[1, ...]*e1[0,...]])
    seg_cent = cent + h*(np.cos(phi)*e1 + np.sin(phi)*e2)
    seg_vect = None

    # lM, rM, dA

    # Bs

    return seg_cent, seg_vect


def field_from_spire():
    """ Get magnetic field from arbitrary 3D circular coil """
    return


###############################################################################
###############################################################################
#               circular poloidal field coils
###############################################################################


def field_from_poloidal_coil(ptsR, ptsZ,
                             coils_Z=None, coils_R=None, coils_I=None):


    return

