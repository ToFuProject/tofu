
import numpy as np



###############################################################################
###############################################################################
#               3D circular coils
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
:

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

