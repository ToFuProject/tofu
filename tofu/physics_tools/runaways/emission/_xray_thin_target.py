

import numpy as np
import scipy.constants as scpct
import scipy.special as scpsp
import datastock as ds


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

    kTe_eV, ne_m3, Zeff, E_ph_eV = _check(
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


def _check(
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
    E_e0_J=None,
    E_e1_J=None,
    # directions
    theta_ph=None,
    theta_e=None,
    phi_ph=None,
    phi_e=None,
    # version
    version=None,
    # plot
    plot=None,
):
    """ Return a differential cross-section for thin-target bremsstrahlung

    Uses the BHE or BE formulas [1]:
        - BHE: Bethe-Heitler-Elwert [2]
        - BE: Elwert-Haug [3]

    Uses:
        [1] Y. Peysson, "Rayonnement electromagnetique des plasmas"
        [2]
        [3]

    """

    # -------------
    # check input
    # -------------

    (
        Z,
        E_e0_J, E_e1_J,
        theta_e, phi_e,
        theta_ph, phi_ph,
        shape,
        version, plot,
    ) = _check_cross(
        # inputs
        Z=Z,
        E_e0_J=E_e0_J,
        E_e1_J=E_e1_J,
        # directions
        theta_ph=None,
        theta_e=None,
        phi_ph=None,
        phi_e=None,
        # version
        version=version,
        # plot
        plot=plot,
    )

    # -------------
    # prepare
    # -------------

    dout = {
        'E_e0': {
            'data': E_e0_J,
            'units': 'J',
        },
        'E_e1': {
            'data': E_e1_J,
            'units': 'J',
        },
        'cross': {
            'data': np.full(shape, 0.),
            'units': 'J',
        },
    }

    # -------------
    # compute
    # -------------

    if version == 'BHE':
        cross[...], dcrit = _cross_BetheHeitlerElwert(
            Z=Z,
            E_e0_J=E_e0_J,
            E_ph_J=E_ph_J,
        )

    else:
        cross[...], dcrit = _cross_ElwertHaug(
            Z=Z,
            E_e0_J=E_e0_J,
            E_ph_J=E_ph_J,
            # directions
        )

    # -------------
    # plot
    # -------------

    if plot is True:
        _plot_cross(
            dout=dout,
        )

    # -------------
    # format output
    # -------------

    return dout


# ####################################################
# ####################################################
#        check cross-section inputs
# ####################################################


def _check_cross(
    # inputs
    Z=None,
    E_e0_J=None,
    E_e1_J=None,
    # directions
    theta_ph=None,
    theta_e=None,
    phi_ph=None,
    phi_e=None,
    # version
    version=None,
    # plot
    plot=None,
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
    # E_e0_J
    # ------------

    if E_e0_J is None:
        E_e0_J = np.r_[10e3, 50e3, 100e3, 500e3, 1e6] * scpct.e

    E_e0_J = np.atleast_1d(E_e0_J)

    # ------------
    # E_e1_J
    # ------------

    if E_e1_J is None:
        E_e1_J = np.linspace(1e3, 1e6, 200) * scpct.e

    E_e1_J = np.atleast_1d(E_e1_J)

    # ------------
    # theta_e, phi_e
    # ------------

    # theta
    if theta_e is None:
        theta_e = 10 * np.pi / 180

    theta_e = np.atleast_1d(theta_e)

    # phi
    if phi_e is None:
        phi_e = np.r_[0, 10, 45, 90, 135, 180] * np.pi / 180

    phi_e = np.atleast_1d(phi_e)

    # ------------
    # theta_ph, phi_ph
    # ------------

    # theta
    if theta_ph is None:
        theta_ph = np.linspace(0, 180, 181) * np.pi / 180

    theta_ph = np.atleast_1d(theta_ph)

    # phi
    if phi_ph is None:
        phi_ph =  * np.pi / 180

    phi_ph = np.atleast_1d(phi_ph)

    # -------------
    # Broadcastable
    # -------------

    dout = ds._generic_check._check_all_broadcastable(
        return_full_arrays=False,
        E_e0_J=E_e0_J,
        E_e1_J=E_e1_J,
        # directions
        theta_ph=theta_ph,
        theta_e=theta_e,
        phi_ph=phi_ph,
        phi_e=phi_e,
    )

    shape = np.broadcast_shapes(
        E_e0_J.shape,
        E_e1_J.shape,
        # directions
        theta_ph=theta_ph,
        theta_e=theta_e,
        phi_ph=phi_ph,
        phi_e=phi_e,
    )

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

    return (
        dout,
        shape,
        version, plot,
    )


# ####################################################
# ####################################################
#        Cross-section - Bethe-Heitler-Elwert
# ####################################################


def _cross_BetheHeitlerElwert(
    Z=None,
    E_e0_J=None,
    E_e1_J=None,
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
    E_ph_J = E_e0_J - E_e1_J

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
    phi_ph=None,
    phi_e=None,
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
    a = alpha * Z

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

    # a (adim)
    a0 = a * eps0 / p0
    a1 = a * eps1 / p1

    # k = photon energy / mc2 (adim)
    k = eps0 - eps1

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
    cosdphi = np.cos(phi_e-phi_ph)

    # -------------
    # Vectors / scalar product
    # -------------

    # scalar
    sca_kp0 = k * E_e0_J * costp
    sca_p1p0 = E_e0_J * E_e1_J * coste
    sca_kp1 = k * E_e1_J * (costp*coste + sintp*sinte*cosdphi)

    # vect{q} = vect{p0 - p1 - k}
    q2 = (
        (E_e0_J - E_e1_J*coste - k*costp)**2
        # + (E_e1_J*sinte*cos(phi_e) + k*sintp*cos(phi_ph))**2
        # + (E_e1_J*sinte*sin(phi_e) + k*sintp*sin(phi_ph))**2
        + (E_e1_J*sinte)**2
        + (k*sintp)**2
        + 2*E_e1_J*k*sinte*sintp*cosdphi
    )

    # -------------
    # eta and products
    # -------------

    eta0 = p0 * (
        -sintp*np.sin(phi_ph),   # x
        sintp*np.cos(phi_ph),  # y
        0,    # z
    )

    eta1 = p1 * (
        sinte*sin(phi_e)*costp - coste*sintp*sin(phi_ph), # x
        coste*sintp*cos(phi_p) - sinte*cos(phi_e)*costp, # y
        sinte*cos(phi_e)*sintp*sin(phi_p) - sinte*sin(phi_e)*sintp*cos(phi_p),
    )

    # norm2
    eta02 = p0**2 * sintp**2
    eta12 = p1**2 * (
        (sinte*costp)**2
        + (coste*sintp)**2
        + (sinte*sintp)**2 * ((cospe*sinpp)**2 + (sinpe*cospp)**2)
        # + (sinte*sintp)**2 * ((cospe*sinpp)**2 + (1-cospe**2)*(1-sinpp**2))
        # + (sinte*sintp)**2 * ((cospe*sinpp)**2 + 1 - cospe**2 - sinpp**2 + (cospe*sinpp)**2)
        - 2*sinte*coste*costp*sintp*sin(phi_e)*sin(phi_p)
        - 2*sinte*coste*costp*sintp*cos(phi_e)*cos(phi_p)
        - 2*sinte**2*sintp**2*cos(phi_e)*sin(phi_e)*cos(phi_p)*sin(phi_p)
    )
    eta12 = (
        (sinte*costp)**2
        + (coste*sintp)**2
        + (sinte*sintp)**2 * ((cospe*sinpp)**2 + (1-cospe**2)*(1-sinpp**2))
        - 2*sinte*coste*costp*sintp*sin(phi_e)*sin(phi_p)
        - 2*sinte*coste*costp*sintp*cos(phi_e)*cos(phi_p)
        - 2*sinte**2*sintp**2*cos(phi_e)*sin(phi_e)*cos(phi_p)*sin(phi_p)
    )

    # -------------
    # Intermediates 1
    # -------------

    # D
    D0 = 2.*(eps0*k - sca_kp0)
    D1 = 2.*(eps1*k - sca_kp1)
    D0D1 = D0 * D1

    # mu
    mu = (p0 + p1)**2 - k**2

    # hypergeometric variable
    x = 1. - mu*q2 / D0D1

    # hypergeometric functions
    V = scpsp.hyp2F1(1j*a0, 1j*a1, 1., x)
    W = scpsp.hyp2F1(1. + 1j*a0, 1. + 1j*a1, 2., x)

    # -------------
    # Intermediates 2
    # -------------

    A0 = (V - 1j*a0*(1-x)*W) / (D0*q2)
    A1 = (V - 1j*a1*(1-x)*W) / (D1*q2)

    B = 1j * a * W / D0D1

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
    term1 = (pi2*a1) / (np.exp(pi2*a1) - 1.)
    term2 = (r0/np.pi)**2   # m2
    term3 = alpha*Z**2
    term4 = p1/p0

    # adim
    ReA0A1 = np.real(np.conjugate(A0)*A1)
    ReA0B = np.real(np.conjugate(A0)*B)
    ReA1B = np.real(np.conjugate(A1)*B)

    # adim
    E0 = (
        (4*eps1**2 - q2) * eta0**2
        + ((eta1**2 + 1)*2*k**2/D1 + eta0**2 - sca_eta01) * D0
    )
    E1 = (
        (4*eps0**2 - q2) * eta1**2
        + ((eta0**2 + 1)*2*k**2/D0 + eta1**2 - sca_eta01) * D1
    )
    E2 = None

    # adim
    F1 = None
    F2 = None
    F3 = None

    # --------------------------------
    # combine into final cross-section
    # --------------------------------

    # m2
    d3cross_ei_EH = (
        (term0 * term1 * term2 * term3 * term4 * k)
        * (
            E0*modA02 + E1*modA12
            - 2.*E2*ReA0A1
            - 2.*F1*ReA0B
            - 2.*F2*ReA1B
            + F3*modB2
        )
    )

    # ------------------
    # validity criterion
    # ------------------

    dcrit = {
        'data': a,
        'lim': 1,
        'comment': r'$\alpha Z < 1$'
    }

    return d3cross_ei_EH, dcrit
