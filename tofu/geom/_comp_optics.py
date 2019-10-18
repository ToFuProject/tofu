

import numpy as np




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
    e1 = np.cross(nIn, nn)
    e1n = np.linalg.norm(e1)
    if e1n < 1.e-10:
        e1 = np.array([nIn[2], -nIn[1], 0.])
    else:
        e1 = e1 / e1n
    e2 = np.cross(nn, e1)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2

def get_bragg_from_lamb(lamb, d, n=1):
    """ n*lamb = 2d*sin(bragg) """
    lamb = np.atleast_1d(lamb).ravel()
    nord = np.atleast_1d(n).ravel()

    bragg= np.full((lamb.size, nord.size), np.nan)
    sin = nord[None, :]*lamb[:, None]/(2.*d)
    indok = np.abs(sin) <= 1.
    bragg[indok] = np.arcsin(sin[indok])
    return bragg

def get_lamb_from_bragg(bragg, d, n=1):
    """ n*lamb = 2d*sin(bragg) """
    bragg = np.atleast_1d(bragg).ravel()
    nord = np.atleast_1d(n).ravel()

    lamb = 2*d*np.sin(bragg[:, None]) / nord[None, :]
    return lamb

def calc_xixj_from_braggangle(Z, nIn,
                              frame_cent, frame_ang,
                              nn, e1, e2,
                              bragg, angle):
        # Deduce key angles
        costheta = np.cos(np.pi/2 - bragg)
        sintheta = np.sin(np.pi/2 - bragg)
        cospsi = np.sum(nIn*nn)
        sinpsi = np.sum(np.cross(nIn, nn)*e1)

        # Deduce ellipse parameters
        cos2sin2 = costheta**2 - sinpsi**2
        x2C = Z * sinpsi * sintheta**2 / cos2sin2
        a = Z * sintheta * cospsi / np.sqrt(cos2sin2)
        b = Z * sintheta * cospsi * costheta / cos2sin2

        # ang_param with respect to axis => epsilon with respect to center
        x1 = None
        x2 = None
        PMnorm = np.sqrt(x1**2 + (x2-x2C)**2)
        acose = PMnorm[None, :]*np.cos(angle[:, None])
        bsinePx2C = PMnorm[None, :]*np.sin(angle[:, None])

        # Deduce xi, xj
        rot = np.array([np.cos(frame_ang), np.sin(frame_ang)])
        rot2 = np.array([-np.sin(frame_ang), np.cos(frame_ang)])
        ellipse_trans = np.array([acose - frame_cent[0],
                                  bsinePx2C - frame_cent[1]])
        xi = np.sum(ellipse_trans*rot[:, None,None], axis=0)
        xj = np.sum(ellipse_trans*rot2[:, None,None], axis=0)

        return xi, xj

def calc_braggangle_from_xixj(xi, xj, Z, nn, frame_cent, frame_ang,
                              nIn, e1, e2):

        # We have e1, e2 => compute x1, x2
        x1 = (frame_cent[0]
              + xi[:,None]*np.cos(frame_ang)
              - xj[None,:]*np.sin(frame_ang))
        x2 = (frame_cent[1]
              + xi[:,None]*np.sin(frame_ang)
              + xj[None,:]*np.cos(frame_ang))

        # Deduce OM
        sca = Z + x1*np.sum(e1*nIn) + x2*np.sum(e2*nIn)
        norm = np.sqrt((x1*e1[0] + x2*e2[0])**2
                       + (x1*e1[1] + x2*e2[1])**2
                       + (Z + x1*e1[2] + x2*e2[2])**2)
        costheta = sca/norm
        bragg = np.pi/2 - np.arccos(costheta)

        # Get angle with respect to axis ! (not center)
        ang = np.arctan2(x2, x1)

        # costheta = np.cos(bragg)
        # sintheta = np.sin(bragg)
        # cospsi = np.sum(nIn*nn)
        # sinpsi = np.sum(np.cross(nIn, nn)*e1)
        # cos2sin2 = costheta**2 - sinpsi**2
        # x2C = Z * sinpsi * sintheta**2 / cos2sin2
        # ang = np.arctan2(x2-x2C, x1)

        return bragg, ang
