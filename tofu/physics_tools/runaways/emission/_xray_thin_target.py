

import os
import warnings


import numpy as np
import scipy.constants as scpct
import scipy.special as scpsp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


# ####################################################
# ####################################################
#           DEFAULT
# ####################################################


_PATH_HERE = os.path.dirname(__file__)


# ####################################################
# ####################################################
#        Maxwellian-averaged
# ####################################################


def get_bremss_maxwell(
    # parameters
    kTe_eV=None,
    ne_m3=None,
    Zeff=None,
    # expressed on
    E_ph_eV=None,
    lambda_m=None,
):

    # -------------
    # check inputs
    # -------------

    kTe_eV, ne_m3, Zeff, E_ph_eV = _check_maxwell(
        # parameters
        kTe_eV=kTe_eV,
        ne_m3=ne_m3,
        Zeff=Zeff,
        # expressed on
        E_ph_eV=E_ph_eV,
        lambda_m=lambda_m,
    )

    # -------------
    # Prepare
    # -------------

    # e, C
    e = scpct.e

    # h, eV.s
    h_eVs = scpct.h / e

    # hbar, eV.s
    hbar_eVs = h_eVs / (2.*np.pi)

    # fine structure
    alpha = scpct.alpha

    # mec2, eV
    mec2_eV = scpct.m_e * scpct.c**2 / e

    # mec eV.s/m
    mec_eVsm = mec2_eV / scpct.c

    # first Borh radius, m
    a0 = hbar_eVs / (mec_eVsm * alpha)

    # ER, eV
    ER_eV = hbar_eVs * scpct.c * alpha / (2 * a0)

    # gaunt
    gaunt = 1.

    # -------------
    # compute
    # -------------

    # Taken from
    # [1] Vezinet, PhD thesis, 2013, p. 51, eq. 41

    # coef0, m^3
    coef0_m3 = 2**5 * np.sqrt(np.pi) * (alpha * a0)**3 / (3*np.sqrt(3))

    # kk, ph.m3/s/sr
    kk = (
        coef0_m3
        * (ER_eV / h_eVs)
        * np.sqrt(ER_eV / kTe_eV)
        * np.exp(-E_ph_eV / kTe_eV)
        * gaunt
    )

    # emiss, ph/m3/s
    emiss = ne_m3**2 * Zeff * kk

    # -------------
    # outputs
    # -------------

    dout = {
        'emiss': {
            'data': emiss,
            'units': 'ph/m3/s/sr',
        },
    }

    return dout


def _check_maxwell(
    # parameters
    kTe_eV=None,
    ne_m3=None,
    Zeff=None,
    # expressed on
    E_ph_eV=None,
    lambda_m=None,
):

    # -------------
    # E vs lambda
    # -------------

    if E_ph_eV is None:

        if lambda_m is None:
            msg = 'Please provide either E_ph_eV xor lambda_m!'
            raise Exception(msg)

        # energy eV
        E_ph_eV = scpct.h * scpct.c / lambda_m / scpct.e

    # -------------
    # broadcast
    # -------------

    dparams, shape = ds._generic_check._check_all_broadcastable(
        # parameters
        kTe_eV=kTe_eV,
        ne_m3=ne_m3,
        Zeff=Zeff,
        # expressed on
        E_ph_eV=E_ph_eV,
    )

    lk = ['kTe_eV', 'ne_m3', 'Zeff', 'E_ph_eV']
    return [dparams[k0] for k0 in lk]


# ####################################################
# ####################################################
#  Integrate cross-section over electron distribution
# ####################################################


# ####################################################
# ####################################################
#   Integrate cross-section over electron direction
# ####################################################


# ####################################################
# ####################################################
#        Differential cross-section
# ####################################################


def get_dcross_ei(
    # inputs
    Z=None,
    E_e0_eV=None,
    E_e1_eV=None,
    # directions
    theta_ph=None,
    theta_e=None,
    dphi=None,
    # hypergeometric parameter
    ninf=None,
    source=None,
    # version
    version=None,
    # plot
    plot=None,
    # debug
    debug=None,
):
    """ Return a differential cross-section for thin-target bremsstrahlung

    First based on [1]
    Now uses the BHE or BE formulas (version):
        - 'BE': Elwert-Haug [2]
            . most general and accurate
            . Uses Sommerfield-Maue eigenfunctions
            . eq. (30) in [2]
        - 'BHE': Bethe-Heitler-Elwert [3]
            . Faster computation
            . uses Born approximation

    Valid for small atomic numbers, deviates at high Z, see [4]

    Uses:
        [1] Y. Peysson, "Rayonnement electromagnetique des plasmas"
        [2] G. Elwert and E. Haug, Phys. Rev., 183, p.90, 1969
            doi: 10.1103/PhysRev.183.90.
        [3]
        [4] Starek et al., Physics Letters A, 39, p. 151, 1972
            doi: 10.1016/0375-9601(72)91059-6.

    Inputs:
        E_e0_eV = kinetic energy of incident electron in eV
        E_e1_eV = kinetic energy of scattered electron in eV
        theta_e = (spherical) theta angle of scattered e vs incident e
        theta_ph = (spherical) theta angle of photon vs incident e
        phi_e = (spherical) phi angle of scattered e vs incident e
        phi_ph = (spherical) theta angle of photon vs incident e
        (all angles in rad)

    Limitations:
        - 'EH' implementation currently stalled because:
            scipy.special.hyp2f1(a, b, c, z)
            does not handle complex input (a, b) as required by the formula
            => ticket https://github.com/scipy/scipy/issues/23450

    """

    # -------------
    # check input
    # -------------

    (
        Z,
        E_e0_J, E_e1_J,
        theta_e, theta_ph, dphi,
        shape,
        version,
        plot,
        debug,
    ) = _check_cross(
        # inputs
        Z=Z,
        E_e0_eV=E_e0_eV,
        E_e1_eV=E_e1_eV,
        # directions
        theta_ph=theta_ph,
        theta_e=theta_e,
        dphi=dphi,
        # version
        version=version,
        # plot
        plot=plot,
        # debug
        debug=debug,
    )

    # -------------
    # prepare
    # -------------

    ddata = {
        # energies
        'E_e0': {
            'data': E_e0_J,
            'units': 'J',
        },
        'E_e1': {
            'data': E_e1_J,
            'units': 'J',
        },
        # angles
        'theta_e': {
            'data': theta_e,
            'units': 'rad',
        },
        'theta_ph': {
            'data': theta_ph,
            'units': 'rad',
        },
        'dphi': {
            'data': dphi,
            'units': 'rad',
        },
        # cross-section
        'cross': {
            'data': np.full(shape, 0.),
            'units': 'm2/(sr2.J)',
        },
    }

    # -------------
    # compute
    # -------------

    if version == 'BHE':
        ddata['cross']['data'][...], dcrit = _cross_BetheHeitlerElwert(
            Z=Z,
            E_e0_J=E_e0_J,
            E_e1_J=E_e1_J,
            # directions
            theta_ph=theta_ph,
            theta_e=theta_e,
            dphi=dphi,
        )

    else:
        ddata['cross']['data'][...], dcrit = _cross_ElwertHaug(
            Z=Z,
            E_e0_J=E_e0_J,
            E_e1_J=E_e1_J,
            # directions
            theta_ph=theta_ph,
            theta_e=theta_e,
            dphi=dphi,
            # hypergeometric parameter
            ninf=ninf,
            source=source,
            # debug
            debug=debug,
        )

    # -------------
    # plot
    # -------------

    if plot is True:
        pass

    # -------------
    # format output
    # -------------

    return ddata


# ####################################################
# ####################################################
#        check cross-section inputs
# ####################################################


def _check_cross(
    # inputs
    Z=None,
    E_e0_eV=None,
    E_e1_eV=None,
    # directions
    theta_ph=None,
    theta_e=None,
    dphi=None,
    # version
    version=None,
    # plot
    plot=None,
    # debug
    debug=None,
):

    # -------------
    # Z
    # ------------

    Z = ds._generic_check._check_var(
        Z, 'Z',
        types=int,
        sign='>0',
        default=1,
    )

    # ------------
    # E_e0_eV (kinetic) => E_e0_J (total)
    # ------------

    mc2 = scpct.m_e*scpct.c**2
    if E_e0_eV is None:
        E_e0_eV = np.r_[10e3, 50e3, 100e3, 500e3, 1e6]

    E_e0_J = np.atleast_1d(E_e0_eV) * scpct.e + mc2

    # ------------
    # E_e1_eV (kinetic) => E_e1_J (total)
    # ------------

    if E_e1_eV is None:
        E_e1_eV = np.linspace(1e3, 1e6, 200)

    E_e1_J = np.atleast_1d(E_e1_eV) * scpct.e + mc2

    # -----------------------
    # theta_e, theta_ph, dphi
    # -----------------------

    # theta_e
    if theta_e is None:
        theta_e = 10 * np.pi / 180

    theta_e = np.atleast_1d(theta_e)

    # theta_ph
    if theta_ph is None:
        theta_ph = np.linspace(0, 180, 181) * np.pi / 180

    theta_ph = np.atleast_1d(theta_ph)

    # dphi
    if dphi is None:
        dphi = 0 * np.pi / 180

    dphi = np.atleast_1d(dphi)

    # -------------
    # Broadcastable
    # -------------

    dout, shape = ds._generic_check._check_all_broadcastable(
        return_full_arrays=False,
        E_e0_J=E_e0_J,
        E_e1_J=E_e1_J,
        # directions
        theta_ph=theta_ph,
        theta_e=theta_e,
        dphi=dphi,
    )

    # safety check
    ineg = E_e1_J > E_e0_J
    if np.any(ineg):
        msg = (
            "Bremsstrahlung cross-section inputs:\n"
            "scattered electron energy can't be more than incident electron\n"
            f"\t- E_e0_eV = {E_e0_eV} eV\n"
            f"\t- E_e1_eV = {E_e1_eV} eV\n"
        )
        raise Exception(msg)

    # ------------
    # version
    # ------------

    version = ds._generic_check._check_var(
        version, 'version',
        types=str,
        allowed=['BHE', 'EH'],
        default='EH',
    )

    # ------------
    # plot
    # ------------

    plot = ds._generic_check._check_var(
        plot, 'plot',
        types=bool,
        default=True,
    )

    # ------------
    # debug
    # ------------

    if debug is None:
        debug = False

    if debug is not False:
        debug = ds._generic_check._check_var(
            debug, 'debug',
            types=str,
            default='vs_theta_ph',
            allowed=['vs_theta_ph'],
        )

    return (
        Z,
        E_e0_J, E_e1_J,
        theta_e, theta_ph, dphi,
        shape,
        version, plot, debug,
    )


# ####################################################
# ####################################################
#        Cross-section - Bethe-Heitler-Elwert
# ####################################################


def _cross_BetheHeitlerElwert(
    Z=None,
    E_e0_J=None,
    E_e1_J=None,
    # directions
    theta_e=None,
    theta_ph=None,
    dphi=None,
):
    """

    Valid for small atomic numbers, deviates at high Z

    E_e0_J = (total) energy of incident electron (initial)
    E_e1_J = (total) energy of electron after collision (final)
    E_ph_J = energy of emitted photon

    p0 = momentum of incident electron (initial)
    p1 = momentum of electron after collision (final)

    eta0 =

    All from
        [1] Y. Peysson, "Rayonnement electromagnetique des plasmas"
            (pages 19-20)
        [2] W. Nakel, “The elementary process of bremsstrahlung”.
            (eq. (2))

    """

    # -------------
    # constants
    # -------------

    # 2pi
    pi2 = 2. * np.pi

    # alpha
    alpha = scpct.alpha

    # electron rest energy
    E0_J = scpct.c**2 * scpct.m_e

    # -------------
    # Energy-derived
    # -------------

    # E_ph_J
    # E_ph_J = E_e0_J - E_e1_J

    # momentum (kg m / s) x c (m / s) => J
    p0c_J = np.sqrt(E_e0_J**2 - E0_J**2)
    p1c_J = np.sqrt(E_e1_J**2 - E0_J**2)

    # -------------
    # eta
    # -------------

    eta0 = Z * alpha * E_e0_J / (p0c_J)
    eta1 = Z * alpha * E_e1_J / (p1c_J)

    # -------------
    # Elwert correction factor for E_ph close to E_e0
    # -------------

    F_Elwert = (
        (eta1 / eta0)
        * ((1. - np.exp(-pi2*eta0)) / (1. - np.exp(-pi2*eta1)))
    )

    # --------------------------------
    # Bethe-Heitler Born approximation
    # --------------------------------

    cross_BH = None

    # --------------------------------
    # combine into final cross-section
    # --------------------------------

    cross_ei_BH = Z**2 * F_Elwert * cross_BH

    # ------------------
    # validity criterion
    # ------------------

    dcrit = {
        'data': pi2 * eta0,
        'lim': 1,
        'comment': r'$2 \pi \eta_0 < 1$'
    }

    return cross_ei_BH, dcrit


# ####################################################
# ####################################################
#        Cross-section - Elwert-Haug
# ####################################################


def _cross_ElwertHaug(
    Z=None,
    E_e0_J=None,
    E_e1_J=None,
    # directions
    theta_ph=None,
    theta_e=None,
    dphi=None,
    # hypergeometric parameter
    ninf=None,
    source=None,
    # debug
    debug=None,
):
    """
    More accurate than Bethe-Heitler-Elwert
    Uses Sommerfield-Maue eigenfunctions

    Implements general form eq (30) in [1]

    [1] G. Elwert and E. Haug, Phys. Rev., 183, p.90, 1969
        10.1103/PhysRev.183.90.

    Valid for small atomic numbers, deviates at high Z, see:
    [2] Starek et al., Physics Letters A, 39, p. 151, 1972
        doi: 10.1016/0375-9601(72)91059-6.

    E_e0_J = (total) energy of incident electron (initial)
    E_e1_J = (total) energy of electron after collision (final)
    E_ph_J = energy of emitted photon (aka k)

    Scalar product and directiosn are expressed in spherical coordinates
    where ez = direction of incident electron (p0)

    k = cos(theta) ez + sin(theta) * (cos(phi)ex + sin(phi)ey)

    """

    # -------------
    # constants
    # -------------

    # 2pi
    pi2 = 2. * np.pi

    # alpha
    alpha = scpct.alpha

    # mc (kg.m/s = J.s/m)
    mc = scpct.c * scpct.m_e

    # electron rest energy J
    E0_J = scpct.c * mc

    # r0 = classical electron radius (m)
    r0 = scpct.e**2 / E0_J / (4.*np.pi*scpct.epsilon_0)

    # --------------
    # Quasi-constants
    # --------------

    # a (adim)
    aa = alpha * Z

    # -------------
    # Energy-derived
    # -------------

    # eps = total eletron energy / mc2 (adim)
    eps0 = E_e0_J / E0_J
    eps1 = E_e1_J / E0_J

    # pc: J
    p0c_J = np.sqrt(E_e0_J**2 - E0_J**2)
    p1c_J = np.sqrt(E_e1_J**2 - E0_J**2)

    # p (adim)
    p0 = p0c_J / E0_J
    p1 = p1c_J / E0_J

    # squared
    p02 = p0**2
    p12 = p1**2

    # a (adim)
    a0 = aa * eps0 / p0
    a1 = aa * eps1 / p1

    # k = photon energy / mc2 (adim)
    kk = eps0 - eps1
    k2 = kk**2

    # kappa (adim)
    kappa = eps0 / p0 + eps1 / p1

    # rho (adim)
    rho = 1/p0 + 1/p1

    # -------------
    # angles internediates
    # -------------

    coste = np.cos(theta_e)
    sinte = np.sin(theta_e)
    costp = np.cos(theta_ph)
    sintp = np.sin(theta_ph)
    cosdphi = np.cos(dphi)

    cossindphi = costp*coste + sintp*sinte*cosdphi

    # -------------
    # Vectors / scalar product
    # -------------

    # scalar
    sca_kp0 = kk * p0 * costp
    sca_p01 = p1 * p0 * coste
    sca_kp1 = kk * p1 * cossindphi

    # vect{q} = vect{p0 - p1 - k}
    q2 = (
        p0**2 + p1**2 + k2
        - 2.*p0*p1*coste
        - 2.*p0*kk*costp
        + 2.*p1*kk*cossindphi
    )

    # -------------
    # eta and products
    # -------------

    # eta0 = p0 * (
    #     -sintp*np.sin(phi_ph),   # x
    #     sintp*np.cos(phi_ph),    # y
    #     0,                       # z
    # )

    # eta1 = p1 * (
    #     sinte*sinpe*costp - coste*sintp*sinpp,              # x
    #     coste*sintp*cospp - sinte*cospe*costp,              # y
    #     sinte*cospe*sintp*sinpp - sinte*sinpe*sintp*cospp,  # z
    # )

    # ---------
    # norm2

    eta02 = p02 * sintp**2
    # eta12 = p12 * (
    #     (sinte*costp)**2
    #     + (coste*sintp)**2
    #     + (sinte*sintp)**2 * ((cospe*sinpp)**2 + (sinpe*cospp)**2)
    #     - 2*sinte*sinpe*costp*coste*sintp*sinpp
    #     - 2*coste*sintp*cospp*sinte*cospe*costp
    #     - 2*(sinte*sintp)**2*cospe*sinpe*sinpp*cospp
    # )
    # = p12 * (
    #     (sinte*costp)**2
    #     + (coste*sintp)**2
    #     - 2*sinte*coste*costp*sintp*cosdphi
    #     + (sinte*sintp)**2 * sin(phi_p - phi_e)**2
    # )
    eta12 = p12 * (
        (sinte*costp)**2
        + (coste*sintp)**2
        - 2*(coste*costp)*(sinte*sintp*cosdphi)
        + (sinte*sintp)**2 * (1 - cosdphi**2)
    )

    # --------
    # scalar

    # sca_eta01 = p0 * p1 * (
    #     - sintp*sinpp * (sinte*sinpe*costp - coste*sintp*sinpp)
    #     + sintp*cospp * (coste*sintp*cospp - sinte*cospe*costp)
    # )
    # = p0 * p1 * sintp * (
    #     + sinpp * (coste*sintp*sinpp - sinte*sinpe*costp)
    #     + cospp * (coste*sintp*cospp - sinte*cospe*costp)
    # )
    # = p0 * p1 * sintp * (
    #     coste*sintp*(sinpp**2 + cospp**2)
    #     - sinte*costp*(sinpe*sinpp + cospe*cospp)
    # )
    sca_eta01 = p0 * p1 * sintp * (coste*sintp - sinte*costp*cosdphi)

    # ---------------
    # Intermediates 1
    # ---------------

    # D
    D0 = 2.*(eps0*kk - sca_kp0)
    D1 = 2.*(eps1*kk - sca_kp1)
    D0D1 = D0 * D1

    # mu
    mu = (p0 + p1)**2 - k2

    # hypergeometric variable
    xx = 1. - mu*q2 / D0D1

    # safety check
    assert np.all(kk < eps0)
    assert np.all(D0D1 > 0.)
    assert np.all(mu > 0.)
    assert np.all(q2 > 0.)
    assert np.all(xx < 1.)

    # hypergeometric functions
    # V = scpsp.hyp2f1(1j*a0, 1j*a1, 1., x)
    # W = scpsp.hyp2f1(1. + 1j*a0, 1. + 1j*a1, 2., x)
    V = _hyp2F1(
        aa=1j*a0,
        bb=1j*a1,
        cc=np.ones(a0.shape, dtype=float),
        zz=xx,
        ninf=ninf,
        source=source,
    )
    W = _hyp2F1(
        aa=1.+1j*a0,
        bb=1.+1j*a1,
        cc=2.*np.ones(a0.shape, dtype=float),
        zz=xx,
        ninf=ninf,
        source=source,
    )

    # ---------------
    # Intermediates 2
    # ---------------

    A0 = (V - 1j*a0*(1-xx)*W) / (D0*q2)
    A1 = (V - 1j*a1*(1-xx)*W) / (D1*q2)

    B = 1j * aa * W / D0D1

    # ---------------
    # squared modulus

    modA02 = np.abs(A0)**2
    modA12 = np.abs(A1)**2
    modB2 = np.abs(B)**2

    # --------------------------------
    # terms combination
    # --------------------------------

    # adim
    term0 = (pi2*a0) / (np.exp(pi2*a0) - 1.)
    term1 = (pi2*a1) / (1. - np.exp(-pi2*a1))
    term2 = (r0/np.pi)**2   # m2
    term3 = alpha*Z**2
    term4 = p1/p0

    # adim
    ReA0A1 = np.real(np.conjugate(A0)*A1)
    ReA0B = np.real(np.conjugate(A0)*B)
    ReA1B = np.real(np.conjugate(A1)*B)

    # adim
    E0 = (
        (4*eps1**2 - q2) * eta02
        + ((eta12 + 1)*2*k2/D1 + eta02 - sca_eta01) * D0
    )
    E1 = (
        (4*eps0**2 - q2) * eta12
        + ((eta02 + 1)*2*k2/D0 + eta12 - sca_eta01) * D1
    )
    E2 = (
        4.*(eps0*eps1 - q2)*sca_eta01
        + 0.5*D0*(sca_eta01 - eta12)
        + 0.5*D1*(eta02 - sca_eta01)
        + 2.*k2*(sca_eta01 + 1.)
    )

    # adim
    F0 = (
        kk * rho * (eta02 - sca_eta01)
        + kappa * (sca_kp0 * (sca_p01 - sca_kp1 + p12) + 2.*k2)
        - (kappa*p0*p1 - 2.*kk/p0) * (sca_kp0 + sca_kp1 - k2)
    )
    F1 = (
        kk * rho * (eta12 - sca_eta01)
        + kappa * (sca_kp1 * (sca_p01 - sca_kp0 + p02) + 2.*k2)
        - (kappa*p0*p1 - 2.*kk/p1) * (sca_kp0 + sca_kp1 - k2)
    )
    F2 = mu * (
        k2
        - (sca_kp0 * sca_kp1) / (p0*p1)
        + (p02 - p12) / (p0*p1)**2
        * (p02 - p12 + sca_kp0 + sca_kp1)
    ) - 2.*k2*rho**2

    # --------------------------------
    # combine into final cross-section
    # --------------------------------

    # 3-differential cross-section - m2 / sr^2
    d3cross_ei_EH = (
        (term0 * term1 * term2 * term3 * term4 * kk)
        * (
            E0 * modA02 + E1 * modA12
            - 2. * E2 * ReA0A1
            - 2. * F0 * ReA0B
            - 2. * F1 * ReA1B
            + F2 * modB2
        )
    )

    # ------------------
    # validity criterion
    # ------------------

    dcrit = {
        'data': aa,
        'lim': 1,
        'comment': r'$\alpha Z < 1$'
    }

    # -------------
    # debug
    # -------------

    if debug is not False:
        if debug == 'vs_theta_ph':
            _debug_EH_vs_theta_ph(**locals())

    return d3cross_ei_EH, dcrit


# ##################################
#        Debug - EH
# ##################################


def _debug_EH_vs_theta_ph(
    # inputs
    E_e0_J=None,
    E_e1_J=None,
    theta_ph=None,
    theta_e=None,
    dphi=None,
    # terms
    term0=None,
    term1=None,
    term2=None,
    term3=None,
    term4=None,
    kk=None,
    # composite
    E0=None,
    modA02=None,
    E1=None,
    modA12=None,
    E2=None,
    ReA0A1=None,
    F0=None,
    ReA0B=None,
    F1=None,
    ReA1B=None,
    F2=None,
    modB2=None,
    # scalar
    sca_kp0=None,
    sca_p01=None,
    sca_kp1=None,
    q2=None,
    eta02=None,
    eta12=None,
    sca_eta01=None,
    xx=None,
    # complex
    V=None,
    W=None,
    A0=None,
    A1=None,
    B=None,
    # unused
    **kwdargs,
):

    # ------------
    # prepare data
    # ------------

    dterms = {
        'E0': E0,
        'E1': E1,
        'E2': E2,
        'F0': F0,
        'F1': F1,
        'F2': F2,
        'modA02': modA02,
        'modA12': modA12,
        'modB2': modB2,
        'ReA0A1': ReA0A1,
        'ReA0B': ReA0B,
        'ReA1B': ReA1B,
    }

    dcomp = {
        'V': V,
        'W': W,
        'A0': A0,
        'A1': A1,
        'B': B,
    }

    dsca = {
        'sca_kp0': sca_kp0,
        'sca_p01': sca_p01,
        'sca_kp1': sca_kp1,
        'q2': q2,
        'eta02': eta02,
        'eta12': eta12,
        'sca_eta01': sca_eta01,
        'xx': xx,
    }

    # --------------
    # prepare axes
    # --------------

    fontsize = 14
    tit = "Debugging of cross-section terms from Elwert-Haug 1969"

    dmargin = {
        'left': 0.08, 'right': 0.95,
        'bottom': 0.08, 'top': 0.85,
        'wspace': 0.2, 'hspace': 0.2,
    }

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(tit, size=fontsize+2, fontweight='bold')

    gs = gridspec.GridSpec(ncols=2, nrows=2, **dmargin)
    dax = {}

    # --------------
    # prepare axes
    # --------------

    # --------------
    # ax - main terms

    ax = fig.add_subplot(gs[0, 0], aspect='auto')
    ax.set_xlabel(
        r"$\theta_{ph}$ (photon emission angle, deg)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        'Main terms in cross-section',
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['main'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # ax - main terms

    ax = fig.add_subplot(gs[0, 1], aspect='auto', sharex=ax)
    ax.set_xlabel(
        r"$\theta_{ph}$ (photon emission angle, deg)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        'scalar products',
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['sca'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # ax - complex real

    ax = fig.add_subplot(gs[1, 0], aspect='auto', sharex=ax)
    ax.set_xlabel(
        r"$\theta_{ph}$ (photon emission angle, deg)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        'Real',
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['real'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # ax - complex imag

    ax = fig.add_subplot(gs[1, 1], aspect='auto', sharex=ax)
    ax.set_xlabel(
        r"$\theta_{ph}$ (photon emission angle, deg)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        'Imag',
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['imag'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # plot main terms
    # --------------

    kax = 'main'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # literature data
        for kvar, vvar in dterms.items():
            if theta_ph.size == vvar.size:
                ax.plot(
                    theta_ph.ravel()*180/np.pi,
                    vvar.ravel(),
                    label=kvar,
                )
            else:
                msg = f"\t- {kvar}.shape = {vvar.shape}"
                print(msg)

        ax.legend()

    # --------------
    # plot scalar products
    # --------------

    kax = 'sca'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # literature data
        for kvar, vvar in dsca.items():
            if theta_ph.size == vvar.size:
                ax.plot(
                    theta_ph.ravel()*180/np.pi,
                    vvar.ravel(),
                    label=kvar,
                )
            else:
                msg = f"\t- {kvar}.shape = {vvar.shape}"
                print(msg)

        ax.legend()
        ax.axhline(-1, c='k', ls='--')
        ax.axhline(0, c='k', ls='--')

    # --------------
    # plot real
    # --------------

    kax = 'real'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # literature data
        for kvar, vvar in dcomp.items():
            if theta_ph.size == vvar.size:
                ax.plot(
                    theta_ph.ravel()*180/np.pi,
                    np.real(vvar).ravel(),
                    label=kvar,
                )
            else:
                msg = f"\t- {kvar}.shape = {vvar.shape}"
                print(msg)

        ax.legend()

    # --------------
    # plot imag
    # --------------

    kax = 'imag'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # literature data
        for kvar, vvar in dcomp.items():
            if theta_ph.size == vvar.size:
                ax.plot(
                    theta_ph.ravel()*180/np.pi,
                    np.imag(vvar).ravel(),
                    label=kvar,
                )

        ax.legend()
    return


# ####################################################
# ####################################################
#        Homemade hypergeometric
# ####################################################


def _hyp2F1(
    aa=None,
    bb=None,
    cc=None,
    zz=None,
    ninf=None,
    source=None,
):
    """ Hypergeometric function 2F1 with complex arguments

    Home-made, replacement for:
        https://github.com/scipy/scipy/issues/23450

    Inspired from:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp2f1.html

    And:
        https://www.johndcook.com/blog/2024/04/16/hypergeometric-large-negative-z/

    """

    # ----------
    # Safety check
    # ----------

    if np.any(zz > 1):
        msg = (
            "This homemade implementation of Hyp2F1 is particularly unstable"
            " for |zz| > 1\n"
            f"Found {(np.abs(zz) > 1).sum()} / {zz.size} pts with |zz| > 1\n"
        )
        raise Exception(msg)

    # ----------
    # inputs
    # ----------

    lok = ['mpmath', 'z/(z-1)', '1/z']
    source = ds._generic_check._check_var(
        source, 'source',
        types=str,
        allowed=lok,
        default=lok[0],
    )

    # try import
    if source == 'mpmath':
        try:
            import mpmath
        except Exception:
            source = lok[1]
            msg = (
                "_hyp2F1(source='{lok[0]}') requires mpmath to be installed\n"
                "See https://pypi.org/project/mpmath/\n"
                f"Setting to source = '{source}'\n"
            )
            warnings.warn(msg)

    # ----------
    # broadcast
    # ----------

    aa, bb, cc, zz = np.broadcast_arrays(aa, bb, cc, zz)

    # ----------
    # Number of terms
    # ----------

    if ninf is None:
        ninf = 50

    nn = np.arange(0, ninf)[None, :]

    # ----------------
    # indices + initialize
    # ----------------

    ismall = np.abs(zz) < 1.
    ilarge = ~ismall
    out = np.full(zz.shape, np.nan, dtype=complex)

    # ----------------
    # source = mpmath
    # ----------------

    if source == 'mpmath':

        for ind in np.ndindex(zz.shape):
            out[ind] = mpmath.hyp2f1(aa[ind], bb[ind], cc[ind], zz[ind])

    # ----------------
    # source = 1/z or z/(z-1)
    # ----------------

    else:

        # ----------------
        # |z| < 1
        # ----------------

        if np.any(ismall):

            sli = (ismall, None)

            # ----------
            # Pochammer

            poch_a = scpsp.gamma(aa[sli] + nn) / scpsp.gamma(aa[sli])
            poch_b = scpsp.gamma(bb[sli] + nn) / scpsp.gamma(bb[sli])
            poch_c = scpsp.gamma(cc[sli] + nn) / scpsp.gamma(cc[sli])

            # ----------
            # fact

            fact = zz[sli]**nn / scpsp.factorial(nn)

            # ----------
            # sum

            # Hotfix: pcoh_a * pch_b can give negative values...
            tot = poch_a * poch_b * fact / poch_c
            iok = np.isfinite(tot)

            out[ismall] = np.sum(tot, axis=-1, where=iok)

        # ----------------
        # |z| > 1
        # ----------------

        if np.any(ilarge):

            # apply:
            # https://www.johndcook.com/blog/2024/04/16/hypergeometric-large-negative-z/

            # reduce
            an = aa[ilarge]
            bn = bb[ilarge]
            cn = cc[ilarge]
            zn = zz[ilarge]

            if source == '1/z':

                coef0 = (
                    scpsp.gamma(cn) * scpsp.gamma(bn-an)
                    / (scpsp.gamma(bn) * scpsp.gamma(cn-an))
                )
                coef1 = (
                    scpsp.gamma(cn) * scpsp.gamma(an-bn)
                    / (scpsp.gamma(an) * scpsp.gamma(cn-bn))
                )

                out[ilarge] = (
                    coef0 * (-zn)**(-an) * _hyp2F1(an, 1-cn+an, 1-bn+an, 1./zn)
                    + coef1 * (-zn)**(-bn) * _hyp2F1(bn, 1-cn+bn, 1-an+bn, 1./zn)
                )

            else:
                out[ilarge] = (
                    (1. - zn)**(-an)
                    * _hyp2F1(an, cn-bn, cn, zn/(zn-1.))
                )

    return out


# ####################################################
# ####################################################
#        Cross-section - plotting
# ####################################################


def plot_xray_thin_ddcross_ei_vs_Literature(
    ninf=None,
    source=None,
):
    """ Compare computed cross-sections vs literature values from Elwert-Haug

    Triply differential cross-section
    Reproduces figures 2, 5, 6 and 7

    [1] G. Elwert and E. Haug, Phys. Rev., 183, p.90, 1969
    [2] W. Nakel, Physics Reports, 243, p. 317—353, 1994


    """

    # --------------
    # Load data
    # --------------

    # isolines
    pfe_isolines = os.path.join(
        _PATH_HERE,
        'RE_HXR_CrossSection_ThinTarget_Isolines_ElwertHaug_fig2.csv',
    )
    out_isolines = np.loadtxt(pfe_isolines, delimiter=',')

    # ph_dist
    pfe_ph_dist = os.path.join(
        _PATH_HERE,
        'RE_HXR_CrossSection_ThinTarget_PhotonDist_ElwertHaug_fig5.csv',
    )
    out_ph_dist = np.loadtxt(pfe_ph_dist, delimiter=',')

    # ph_dist_nakel
    pfe_ph_dist_nakel = os.path.join(
        _PATH_HERE,
        'RE_HXR_CrossSection_ThinTarget_PhotonDist_Nakel_fig5.csv',
    )
    out_ph_dist_nakel = np.loadtxt(pfe_ph_dist_nakel, delimiter=',')

    # --------------
    # Compute
    # --------------

    # --------------
    # isolines data

    te0 = np.linspace(-np.pi/2, np.pi/2, 91)[None, :]
    te1 = np.linspace(-np.pi/2, np.pi/2, 92)[:, None]

    ddata_iso = get_dcross_ei(
        # inputs
        Z=13,
        E_e0_eV=180e3,
        E_e1_eV=90e3,
        # directions
        theta_ph=np.abs(te0),
        theta_e=np.abs(te1),
        dphi=(te0*te1 < 0)*np.pi,
        # hypergeometric parameter
        ninf=ninf,
        source=source,
        # version
        version=None,
    )

    # ------------
    # photon distribution

    tph = np.linspace(-np.pi/2, np.pi/2, 92)

    ddata_ph_dist = get_dcross_ei(
        # inputs
        Z=79,
        E_e0_eV=300e3,
        E_e1_eV=170e3,
        # directions
        theta_ph=np.abs(tph),
        theta_e=0.,
        dphi=(tph < 0.)*np.pi,
        # hypergeometric parameter
        ninf=ninf,
        source=source,
        # version
        version=None,
        # debug
        debug='vs_theta_ph',
    )

    # ------------
    # photon distribution - nakel

    tph_nakel = np.linspace(-80, 60, 141)*np.pi/180.

    ddata_ph_dist_nakel = get_dcross_ei(
        # inputs
        Z=47,
        E_e0_eV=180e3,
        E_e1_eV=100e3,
        # directions
        theta_ph=np.abs(tph_nakel),
        theta_e=30.*np.pi/180.,
        dphi=(tph_nakel < 0.)*np.pi,
        # hypergeometric parameter
        ninf=ninf,
        source=source,
        # version
        version=None,
        # debug
        debug='vs_theta_ph',
    )

    # --------------
    # prepare axes
    # --------------

    fontsize = 14
    tit = (
        "Comparison vs literature:\n"
        "[1] G. Elwert and E. Haug, Phys. Rev., 183, p.90, 1969\n"
        "[2] W. Nakel, Physics Reports, 243, p. 317—353, 1994\n"
    )

    dmargin = {
        'left': 0.08, 'right': 0.95,
        'bottom': 0.06, 'top': 0.85,
        'wspace': 0.2, 'hspace': 0.30,
    }

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(tit, size=fontsize+2, fontweight='bold')

    gs = gridspec.GridSpec(ncols=2, nrows=2, **dmargin)
    dax = {}

    # --------------
    # prepare axes
    # --------------

    # --------------
    # ax - isolines

    ax = fig.add_subplot(gs[0, 0], aspect='equal', adjustable='datalim')
    ax.set_xlabel(
        r"$\theta_{ph}$ (photon emission angle, deg)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$\theta_e$ (e scattering angle, deg)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        "[1] Fig 2. Isolines of the differential cross-section\n"
        + r"$Z = 13$ (Al), $E_{e0} = 180 keV$, $E_{e1} = 90 keV$"
        + "\n1 barn = 1e-28 m2",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['isolines'] = {'handle': ax, 'type': 'isolines'}

    # ------------
    # ax - ph_dist

    ax = fig.add_subplot(gs[0, 1], aspect='auto', projection='polar')
    ax.set_theta_zero_location("N")
    ax.set_title(
        "[1] Fig 5. Photon angular distribution\n"
        + r"$Z = 79$ (Au), $E_{e0} = 300 keV$, "
        + r"$E_{e1} = 170 keV$, $\theta_e = 0$"
        + "\n1 barn = 1e-28 m2",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['ph_dist'] = {'handle': ax, 'type': 'ph_dist'}

    # ------------
    # ax - ph_dist_nakel

    ax = fig.add_subplot(gs[1, 0], aspect='auto')
    ax.set_title(
        "[2] Fig 5. Photon angular distribution\n"
        + r"$Z = 47$ (Ag), $E_{e0} = 180 keV$, "
        + r"$E_{e1} = 100 keV$, $\theta_e = 30$ deg"
        + "\n1 barn = 1e-28 m2",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_xlabel(
        r"$\theta_{ph}$ (photon emission angle, deg)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$\frac{d^3 \sigma}{d\Omega_e d\Omega_{ph} dk}$"
        + "   [b/(sr.sr.MeV)]",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['ph_dist_nakel'] = {'handle': ax, 'type': 'ph_dist'}

    # --------------
    # plot isolines
    # --------------

    kax = 'isolines'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # literature data
        ax.plot(
            np.r_[out_isolines[:, 0], np.nan, -out_isolines[:, 0]],
            np.r_[out_isolines[:, 1], np.nan, -out_isolines[:, 1]],
            c='k',
            ls='--',
        )

        # computed data
        im = ax.contour(
            te0.ravel() * 180/np.pi,
            te1.ravel() * 180/np.pi,
            ddata_iso['cross']['data']*1e28,
            cmap=None,
            levels=8,
            colors='b',
            linestyles='-',
            # levels=[0.1, 0.5, 1, 2, 3, 4, 5, 6, 7],
        )

        # add labels
        ax.clabel(im, im.levels, inline=True, fmt='%r', fontsize=10)

        # add refs
        ax.axvline(0, c='k', ls='--')
        ax.axhline(0, c='k', ls='--')
        ax.set_xlim(-90, 90)

    # ------------------------
    # plot photon distribution
    # ------------------------

    kax = 'ph_dist'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # literature data
        ax.plot(
            out_ph_dist[:, 1] * np.pi/180,
            out_ph_dist[:, 0],
            c='k',
            ls='-',
        )

        # computed data
        im = ax.plot(
            tph,
            ddata_ph_dist['cross']['data']*1e28,
            c='b',
            ls='-',
        )

        # limits
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
        # ax.set_rmax(2)
        # ax.set_rmin(0)  # Change the radial axis to only go from 1 to 2
        # ax.set_rticks([1, 2])  # Fewer radial ticks
        ax.set_rorigin(0)
        ax.set_rlabel_position(0)

    # ------------------------
    # plot photon distribution - nakel
    # ------------------------

    kax = 'ph_dist_nakel'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # literature data
        ax.plot(
            out_ph_dist_nakel[:, 0],
            out_ph_dist_nakel[:, 1],
            c='k',
            ls='-',
        )

        # computed data
        im = ax.plot(
            tph_nakel*180/np.pi,
            ddata_ph_dist_nakel['cross']['data']*1e28,
            c='b',
            ls='-',
        )

        # add
        ax.axvline(0, c='k', ls='-')
        ax.axvline(0, c='k', ls='--')

        # limits
        ax.set_xlim(-80, 60)
        ax.set_ylim(0, 60)

    return dax, ddata_iso, ddata_ph_dist, ddata_ph_dist_nakel
