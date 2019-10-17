

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
    lamb = np.atleast_1d(lamb).ravel()
    nord = np.atleast_1d(n).ravel()

    theta = np.full((lamb.size, nord.size), np.nan)
    sin = nord[None, :]*lamb[:, None]/(2.*d)
    indok = np.abs(sin) <= 1.
    theta[indok] = np.arcsin(sin[indok])
    return theta

def calc_xixj_from_braggangle(Z, nIn,
                              frame_cent, frame_ang,
                              nn, e1, e2,
                              ang_bragg, ang_param):
        # Deduce key angles
        costheta = np.cos(ang_bragg)
        sintheta = np.sin(ang_bragg)
        cospsi = np.sum(nIn*nn)
        sinpsi = np.sum(np.cross(nIn, nn)*e1)

        # Deduce ellipse parameters
        cos2sin2 = costheta**2 - sinpsi**2
        x2C = Z * sinpsi * sintheta**2 / cos2sin2
        a = Z * sintheta * cospsi / np.sqrt(cos2sin2)
        b = Z * sintheta * cospsi * costheta / cos2sin2

        # Deduce xi, xj
        rot = np.array([np.cos(frame_ang), np.sin(frame_ang)])
        rot2 = np.array([-np.sin(frame_ang), np.cos(frame_ang)])
        ellipse_trans = np.array([a[None, :]*np.cos(ang_param[:, None])
                                  - frame_cent[0],
                                  b[None, :]*np.sin(ang_param[:, None])
                                  - frame_cent[1] + x2C[None, :]])
        xi = np.sum(ellipse_trans*rot[:, None,None], axis=0)
        xj = np.sum(ellipse_trans*rot2[:, None,None], axis=0)

        return xi, xj

def calc_braggangle_from_xixj(xi, xj, Z, nn, frame_cent, frame_ang,
                              nIn, e1, e2):
        sinpsi = np.sum(np.cross(nIn, nn)*e1)
        sinpsi2 = sinpsi**2
        xij2 = xi[:,None]**2 + xj[None,:]**2
        xjsz = xj[None,:]*sinpsi + Z

        A = xij2 + xjsz**2 - ((xj*sinpsi)**2)[None,:]
        B = xij2*sinpsi2 + xjsz**2 + Z*sinpsi2*(2.*xj[None,:]*sinpsi + Z)
        C = xjsz*sinpsi2

        Delta = B**2 - 4*A*C
        indok = Delta >= 0
        costheta = np.full(A.shape, np.nan)

        s1 = (B[indok] + np.sqrt(Delta[indok])) / (2*A[indok])
        s2 = (B[indok] - np.sqrt(Delta[indok])) / (2*A[indok])

        ind = (s1 >= 0) & (s1 <= 1)
        iok0, iok1 = indok.nonzero()
        import ipdb         # DB
        ipdb.set_trace()    # DB

        costheta[iok0[ind], iok1[ind]] = np.sqrt(s1[ind])
        ind[:] = (s2 >= 0) & (s2 <= 1)
        costheta[iok0[ind], iok1[ind]] = np.sqrt(s2[ind])
        return np.arccos(costheta)
