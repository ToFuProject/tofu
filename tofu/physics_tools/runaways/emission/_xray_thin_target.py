

import numpy as np
import scipy.constants as scpct
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
#        Cross-section
# ####################################################


def get_dcross_ei():
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

    # -------------
    # prepare
    # -------------

    # -------------
    # compute
    # -------------

    # -------------
    # format output
    # -------------

    return dout


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

    # c (m/s)
    c = scpct.c

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

    return


# ####################################################
# ####################################################
#        Cross-section - Elwert-Haug
# ####################################################


def _cross_ElwertHaug(
    Z=None,
    E_e0_J=None,
    E_e1_J=None,
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

    """

    # -------------
    # constants
    # -------------

    # c (m/s)
    c = scpct.c

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

    # -------------
    # Intermediates 0
    # -------------

    sca_kp0 = None
    sca_kp1 = None

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
    x = 1. - mu*q**2 / D0D1

    # hypergeometric functions
    V = None
    W = None

    # -------------
    # Intermediates 2
    # -------------

    A1 = None
    A2 = None

    B = i * a * W / D0D1

    # -------------
    # Energy-derived
    # -------------

    modB2 = None

    # --------------------------------
    # terms combination
    # --------------------------------

    # adim
    term0 = (pi2*a0) / (np.exp(pi2*a0) - 1.)
    term1 = (pi2*a1) / (np.exp(pi2*a1) - 1.)
    term2 = (r0/np.pi)**2   # m2
    term3 = alpha*Z**2
    term4 = p1/p0

    # --------------------------------
    # combine into final cross-section
    # --------------------------------

    # m2
    d3cross_ei_EH = (
        (term0 * term1 * term2 * term3 * term4 * k)
        * (
            E1*A1**2 + E2*A2**2
            - 2.*E3*ReA0A1
            - 2.*F1*ReA0B
            - 2.*F2*ReA1B
            + F3*B**2
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

    return
