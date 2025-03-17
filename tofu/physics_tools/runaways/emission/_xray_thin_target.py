

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
