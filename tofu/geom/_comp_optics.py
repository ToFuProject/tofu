
import numpy as np

# ###############################################
# ###############################################
#           CrystalBragg
# ###############################################
# ###############################################

# ###############################################
#           Coordinates transforms
# ###############################################

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

        # Get radius from axis => x1, x2 => xi, xj
        asin2bcos2 = (b[None,:]**2*np.cos(angle[:,None])**2
                      + a[None,:]**2*np.sin(angle[:,None])**2)
        l = ((a[None,:]**2*x2C[None,:]*np.sin(angle[:,None])
              + a[None,:]*b[None,:]*np.sqrt(asin2bcos2 -
                                            x2C[None,:]**2*np.cos(angle[:,None])**2))
             / asin2bcos2)

        x1_frame = l*np.cos(angle[:,None]) - frame_cent[0]
        x2_frame = l*np.sin(angle[:,None]) - frame_cent[1]

        xi = x1_frame*np.cos(frame_ang) + x2_frame*np.sin(frame_ang)
        xj = -x1_frame*np.sin(frame_ang) + x2_frame*np.cos(frame_ang)
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
        norm = np.sqrt((Z*nIn[0] + x1*e1[0] + x2*e2[0])**2
                       + (Z*nIn[1] + x1*e1[1] + x2*e2[1])**2
                       + (Z*nIn[2] + x1*e1[2] + x2*e2[2])**2)
        bragg = np.pi/2 - np.arccos(sca/norm)

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


# ###############################################
#           Spectral fit 2d
# ###############################################

def get_2dspectralfit_func(lambrest,
                           bsamp=None, bsshift=None, bswidth=None,
                           deg=None, knots=None):

    lambrest = np.atleast_1d(lambrest).ravel()
    nlamb = lambrest.size
    knots = np.atleast_1d(knots).ravel()
    nknots = knots.size
    nbsplines = nknots - 1 + deg
    assert bsamp.shape == bsshift.shape == bswidth.shape == (nlamb, nbsplines)

    # Get 3 sets of bsplines for each lamb
    lbsamp = [scpinterp.Bspline(knots, bsamp[ii,:], deg,
                               extrapolate=False, axis=0)
             for ii in range(nlamb)]
    lbsshift = [scpinterp.Bspline(knots, bsshift[ii,:], deg,
                                  extrapolate=False, axis=0)
                for ii in range(nlamb)]
    lbswidth = [scpinterp.Bspline(knots, bswidth[ii,:], deg,
                                  extrapolate=False, axis=0)
                for ii in range(nlamb)]

    # Define function
    def func(lamb, angle, lambrest=lambrest,
             lbsamp=lbsamp, lbsshift=lbsshift, lbswidth=lbswidth):
        nlamb = lambrest.size
        ldata = np.array([lbsamps[ii](angle)[None,:]
                          *np.exp(-(lamb[:,None]
                                    - (lambrest[ii]
                                       + lbsshift[ii](angle)[None,:]))**2
                                  /lbswidth[ii](angle)[None,:]**2)
                          for ii in range(nlamb)])
        return np.sum(ldata, axis=0)
    return func
