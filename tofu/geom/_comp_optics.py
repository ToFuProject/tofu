
import numpy as np
import scipy.interpolate as scpinterp

# ###############################################
# ###############################################
#           CrystalBragg
# ###############################################
# ###############################################

# ###############################################
#           sampling
# ###############################################

def CrystBragg_sample_outline_sphrect(dpsi, dtheta, npsi=None, ntheta=None):
    psi = dpsi*np.linspace(-1, 1., npsi)
    theta = np.pi/2. + dtheta*np.linspace(-1, 1., ntheta)
    psimin = np.full((ntheta,), psi[0])
    psimax = np.full((ntheta,), psi[-1])
    thetamin = np.full((npsi,), theta[0])
    thetamax = np.full((npsi,), theta[-1])
    psi = np.concatenate((psi, psimax,
                          psi[::-1], psimin))
    theta = np.concatenate((thetamin, theta,
                            thetamax, theta[::-1]))
    return psi, theta

def CrystBragg_get_noute1e2_from_psitheta(nout, e1, e2, psi, theta,
                                          e1e2=True):
    vout = ((np.cos(psi)[None, :]*nout[:, None]
             + np.sin(psi)[None, :]*e1[:, None])*np.sin(theta)[None, :]
            + np.cos(theta)[None, :]*e2[:, None])
    if e1e2:
        ve1 = (-np.sin(psi)[None, :]*nout[:, None] + np.cos(psi)[None, :]*e1[:, None])
        ve2 = np.array([vout[1, :]*ve1[2, :] - vout[2, :]*ve1[1, :],
                        vout[2, :]*ve1[0, :] - vout[0, :]*ve1[2, :],
                        vout[0, :]*ve1[1, :] - vout[1, :]*ve1[0, :]])
        return vout, ve1, ve2
    else:
        return vout

def CrystBragg_sample_outline_plot_sphrect(center, nout, e1, e2,
                                           rcurve, extenthalf, res=None):
    dpsi, dtheta = extenthalf
    if res is None:
        res = np.min(extenthalf)/5.
    npsi = 2*int(np.ceil(dpsi / res)) + 1
    ntheta = 2*int(np.ceil(dtheta / res)) + 1
    psi, theta = CrystBragg_sample_outline_sphrect(dpsi, dtheta,
                                                   npsi=npsi, ntheta=ntheta)
    vout = CrystBragg_get_noute1e2_from_psitheta(nout, e1, e2, psi, theta,
                                                 e1e2=False)
    return center[:, None] + rcurve*vout

def CrystBragg_sample_outline_Rays(center, nout, e1, e2,
                                   rcurve, extenthalf,
                                   bragg, phi):
    dpsi, dtheta = extenthalf
    psi, theta = CrystBragg_sample_outline_sphrect(dpsi, dtheta,
                                                   npsi=3, ntheta=3)
    psi = np.append(psi, [0])
    theta = np.append(theta, [np.pi/2.])
    npts = psi.size

    # add repetitions for rays
    nrays = phi.size
    psi = np.repeat(psi, nrays)
    theta = np.repeat(theta, nrays)

    # add tiling for pts
    bragg = np.tile(bragg, npts)
    phi = np.tile(phi, npts)

    # Compute local vectors
    vout, ve1, ve2 = CrystBragg_get_noute1e2_from_psitheta(nout, e1, e2,
                                                           psi, theta)
    # Deduce D, u
    D = center[:, None] + rcurve*vout
    u = (-np.sin(bragg)*vect
         + np.cos(bragg)*(np.cos(phi)*ve1 + np.sin(phi)*ve2))
    return D, u


# ###############################################
#           lamb <=> bragg
# ###############################################

def get_bragg_from_lamb(lamb, d, n=None):
    """ n*lamb = 2d*sin(bragg) """
    if n is None:
        n = 1
    bragg= np.full(lamb.shape, np.nan)
    sin = n*lamb/(2.*d)
    indok = np.abs(sin) <= 1.
    bragg[indok] = np.arcsin(sin[indok])
    return bragg

def get_lamb_from_bragg(bragg, d, n=None):
    """ n*lamb = 2d*sin(bragg) """
    if n is None:
        n = 1
    return 2*d*np.sin(bragg) / n


# ###############################################
#           Approximate solution
# ###############################################

def get_approx_detector_rel(rcurve, bragg, tangent_to_rowland=None):

    if tangent_to_rowland is None:
        tangent_to_rowland = True

    # distance crystal - det_center
    det_dist = rcurve*np.sin(bragg)

    # det_nout and det_e1 in (nout, e1, e2) (det_e2 = e2)
    n_crystdet_rel = np.r_[-np.sin(bragg), np.cos(bragg), 0.]
    if tangent_to_rowland:
        bragg2 = 2.*bragg
        det_nout_rel = np.r_[-np.cos(bragg2), -np.sin(bragg2), 0.]
        det_ei_rel = np.r_[np.sin(bragg2), -np.cos(bragg2), 0.]
    else:
        det_nout_rel = -n_crystdet_rel
        det_ei_rel = np.r_[np.cos(bragg), np.sin(bragg), 0]
    return det_dist, n_crystdet_rel, det_nout_rel, det_ei_rel


def get_det_abs_from_rel(det_dist, n_crystdet_rel, det_nout_rel, det_ei_rel,
                         summit, nout, e1, e2,
                         ddist=None, di=None, dj=None,
                         dtheta=None, dpsi=None, tilt=None):
    # Reference
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
    det_nout2 = ((np.cos(dpsi)*det_nout
                 + np.sin(dpsi)*det_ei)*np.cos(dtheta)
                 + np.sin(dtheta)*det_ej)
    det_ei2 = (np.cos(dpsi)*det_ei - np.sin(dpsi)*det_nout)
    det_ej2 = np.cross(det_nout2, det_ei2)

    # tilt
    det_ei3 = np.cos(tilt)*det_ei2 + np.sin(tilt)*det_ej2
    det_ej3 = np.cross(det_nout2, det_ei3)

    return det_cent, det_nout2, det_ei3, det_ej3


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


def calc_xixj_from_braggphi(summit, det_cent, det_nout, det_ei, det_ej,
                            nout, e1, e2, bragg, phi):
    sp = (det_cent - summit)
    vect = (-np.sin(bragg)[None, :]*nout[:, None]
            + np.cos(bragg)[None, :]*(np.cos(phi)[None, :]*e1[:, None]
                                      + np.sin(phi)[None, :]*e2[:, None]))
    k = np.sum(sp*det_nout) / np.sum(vect*det_nout[:, None], axis=0)
    pts = summit[:, None] + k[None, :]*vect

    xi = np.sum((pts - det_cent[:, None])*det_ei[:, None], axis=0)
    xj = np.sum((pts - det_cent[:, None])*det_ej[:, None], axis=0)
    return xi, xj


def calc_braggphi_from_xixjpts(det_cent, det_ei, det_ej,
                               summit, nin, e1, e2,
                               xi=None, xj=None, pts=None):

    if pts is None:
        xi = xi[None, ...]
        xj = xj[None, ...]
        if xi.ndim == 1:
            summit = summit[:, None]
            det_cent = det_cent[:, None]
            det_ei, det_ej = det_ei[:, None], det_ej[:, None]
            nin, e1, e2 = nin[:, None], e1[:, None], e2[:, None]
        else:
            summit = summit[:, None, None]
            det_cent = det_cent[:, None, None]
            det_ei, det_ej = det_ei[:, None, None], det_ej[:, None, None]
            nin, e1, e2 = (nin[:, None, None],
                           e1[:, None, None], e2[:, None, None])
        pts = det_cent + xi*det_ei + xj*det_ej
    else:
        assert pts.ndim == 2
        pts = pts[:, :, None]
        summit = summit[:, None, None]
        nin, e1, e2 = nin[:, None, None], e1[:, None, None], e2[:, None, None]

    vect = pts - summit
    vect = vect / np.sqrt(np.sum(vect**2, axis=0))[None, ...]
    bragg = np.arcsin(np.sum(vect*nin, axis=0))

    phi = np.arctan2(np.sum(vect*e2, axis=0), np.sum(vect*e1, axis=0))
    return bragg, phi


def get_lambphifit(lamb, phi, nxi, nxj):
    lambD = lamb.max()-lamb.min()
    lambfit = lamb.min() +lambD*np.linspace(0, 1, nxi)
    phiD = phi.max() - phi.min()
    phifit = phi.min() + phiD*np.linspace(0, 1, nxj)
    return lambfit, phifit


# ###############################################
#           From plasma pts
# ###############################################

def calc_psidthetaphi_from_pts_lamb(pts, center, rcurve,
                                    bragg, nlamb, npts,
                                    nout, e1, e2, extenthalf, ntheta=None):

    if ntheta is None:
        ntheta = 100

    scaPCem = np.full((nlamb, npts), np.nan)
    dtheta = np.full((nlamb, npts, ntheta), np.nan)
    psi = np.full((nlamb, npts, ntheta), np.nan)

    # Get to scalar product
    PC = center[:, None] - pts
    PCnorm2 = np.sum(PC**2, axis=0)
    cos2 = np.cos(bragg)**2
    deltaon4 = (rcurve**2*cos2[:, None]**2
                - (rcurve**2*cos2[:, None]
                   - PCnorm2[None, :]*np.sin(bragg)[:, None]**2))

    # Get two relevant solutions
    ind = deltaon4 >= 0.
    cos2 = np.repeat(cos2[:, None], npts, axis=1)[ind]
    PCnorm = np.tile(np.sqrt(PCnorm2), (nlamb, 1))[ind]
    sol1 = -rcurve*cos2 - np.sqrt(deltaon4[ind])
    sol2 = -rcurve*cos2 + np.sqrt(deltaon4[ind])
    # em is a unit vector and ...
    ind1 = (np.abs(sol1) <= PCnorm) & (sol1 >= -rcurve)
    ind2 = (np.abs(sol2) <= PCnorm) & (sol2 >= -rcurve)
    assert not np.any(ind1 & ind2)
    sol1 = sol1[ind1]
    sol2 = sol2[ind2]
    indn = ind.nonzero()
    ind1 = [indn[0][ind1], indn[1][ind1]]
    ind2 = [indn[0][ind2], indn[1][ind2]]
    scaPCem[ind1[0], ind1[1]] = sol1
    scaPCem[ind2[0], ind2[1]] = sol2
    ind = ~np.isnan(scaPCem)

    # Get equation on PCem
    X = np.sum(PC*nout[:, None], axis=0)
    Y = np.sum(PC*e1[:, None], axis=0)
    Z = np.sum(PC*e2[:, None], axis=0)

    scaPCem = np.repeat(scaPCem[..., None], ntheta, axis=-1)
    ind = ~np.isnan(scaPCem)
    XYnorm = np.repeat(np.repeat(np.sqrt(X**2 + Y**2)[None, :],
                                 nlamb, axis=0)[..., None],
                       ntheta, axis=-1)[ind]
    Z = np.repeat(np.repeat(Z[None, :], nlamb, axis=0)[..., None],
                  ntheta, axis=-1)[ind]
    angextra = np.repeat(
        np.repeat(np.arctan2(Y, X)[None, :], nlamb, axis=0)[..., None],
        ntheta, axis=-1)[ind]
    dtheta[ind] = np.repeat(
        np.repeat(extenthalf[1]*np.linspace(-1, 1, ntheta)[None, :],
                  npts, axis=0)[None, ...],
        nlamb, axis=0)[ind]

    psi[ind] = (np.arccos(
        (scaPCem[ind] - Z*np.sin(dtheta[ind]))/(XYnorm*np.cos(dtheta[ind])))
                + angextra)
    psi[ind] = np.arctan2(np.sin(psi[ind]), np.cos(psi[ind]))
    indnan = (~ind) | (np.abs(psi) > extenthalf[0])
    psi[indnan] = np.nan
    dtheta[indnan] = np.nan
    return dtheta, psi, indnan
