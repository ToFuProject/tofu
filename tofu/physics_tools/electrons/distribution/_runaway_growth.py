

import numpy as np
import scipy.constants as scpct
import datastock as ds


# ##############################################################
# ##############################################################
#                 DEFAULTS
# ##############################################################


# see:
# https://docs.plasmapy.org/en/stable/notebooks/formulary/coulomb.html
_SIGMAP = 1.
_LNG = 20.


# ##############################################################
# ##############################################################
#           Critical and Dreicer electric fields
# ##############################################################


def get_RE_critical_dreicer_electric_fields(
    ne_m3=None,
    kTe_eV=None,
    lnG=None,
):

    # -------------
    # check input
    # -------------

    ne_m3, kTe_eV, lnG = _check_critical_dreicer(
        ne_m3=ne_m3,
        kTe_eV=kTe_eV,
        lnG=lnG,
    )

    # -------------
    # prepare
    # -------------

    # vacuum permittivity in C/(V.m), scalar
    eps0 = scpct.epsilon_0

    # custom computation intermediates C^2/(V^2.m^2), scalar
    pie02 = np.pi * eps0**2

    # electron charge (C), scalar
    e = scpct.e

    # electron rest energy (J = C.V), scalar
    mec2_CV = scpct.m_e * scpct.c**2

    # -------------
    # compute
    # -------------

    # critical electric field (V/m)
    Ec_Vm = ne_m3 * e**3 * lnG / (4 * pie02 * mec2_CV)

    # Dreicer electric field
    if kTe_eV is not None:
        Ed_Vm = Ec_Vm * (mec2_CV / e) / kTe_eV
    else:
        Ed_Vm = None

    # -------------
    # format output
    # -------------

    dout = {
        'E_C': {
            'data': Ec_Vm,
            'units': 'V/m',
        },
    }

    if Ed_Vm is not None:
        dout['E_D'] = {
            'data': Ed_Vm,
            'units': 'V/m',
        }

    return dout


def _check_critical_dreicer(
    ne_m3=None,
    kTe_eV=None,
    lnG=None,
):

    # -----------------
    # preliminary: lnG
    # -----------------

    if lnG is None:
        lnG = _LNG

    # -----------------
    # broadcastable
    # -----------------

    dparams, shape = ds._generic_check._check_all_broadcastable(
        ne_m3=ne_m3,
        kTe_eV=kTe_eV,
        lnG=lnG,
    )

    return [dparams[kk] for kk in ['ne_m3', 'kTe_eV', 'lnG']]


# ##############################################################
# ##############################################################
#            Primary & secondary growth source terms
# ##############################################################


def get_RE_growth_source_terms(
    ne_m3=None,
    lnG=None,
    Epar_Vm=None,
    kTe_eV=None,
    Zeff=None,
):
    """ Return the source terms in the RE dynamic equation

    S_primary: dreicer growth  (1/m3/s)

    S_secondary: avalanche growth (1/s)

    """

    # -------------
    # check inputs
    # -------------

    ne_m3, lnG, Epar_Vm, kTe_eV, Zeff = _check_growth(
        ne_m3=ne_m3,
        lnG=lnG,
        Epar_Vm=Epar_Vm,
        kTe_eV=kTe_eV,
        Zeff=Zeff,
    )

    # -------------
    # prepare
    # -------------

    # vacuum permittivity in C/(V.m), scalar
    eps0 = scpct.epsilon_0

    # charge C
    e = scpct.e

    # mec2 (J = CV)
    mec2_CV = scpct.m_e * scpct.c**2

    # mec C.V.s/m
    mec = mec2_CV / scpct.c

    # me2c3 J**2 / (m/s) = C^2 V^2 s / m
    me2c3 = mec2_CV**2 / scpct.c

    # Dreicer electric field - shape
    dEcEd = get_RE_critical_dreicer_electric_fields(
        ne_m3=ne_m3,
        kTe_eV=kTe_eV,
        lnG=lnG,
    )

    Ec_Vm = dEcEd['E_C']['data']
    Ed_Vm = dEcEd['E_D']['data']

    # -------------
    # pre-compute
    # -------------

    # term1 (m^3/s)
    term1 = e**4 * lnG / (4 * np.pi * eps0**2 * me2c3)

    # term2 - unitless (convert kTe_eV => J)
    term2 = (mec2_CV / (2. * kTe_eV * e))**1.5

    # term3 - unitless
    term3 = (Ed_Vm / Epar_Vm)**(3*(1. + Zeff) / 16.)

    # exp - unitless
    exp = np.exp(
        -Ed_Vm / (4.*Epar_Vm) - np.sqrt((1. + Zeff) * Ed_Vm / Epar_Vm)
    )

    # sqrt - unitless
    sqrt = np.sqrt(np.pi / (3 * (5 + Zeff)))

    # -------------
    # Compute
    # -------------

    # 1/m^3/s
    S_primary = ne_m3**2 * term1 * term2 * term3 * exp

    # 1/s   (C / C.V.s/m * V.m)
    S_secondary = sqrt * (e / mec) * (Epar_Vm - Ec_Vm) / lnG

    # -------------
    # format output
    # -------------

    dout = {
        'S_primary': {
            'data': S_primary,
            'units': '1/m3/s',
        },
        'S_secondary': {
            'data': S_secondary,
            'units': '1/s',
        },
    }

    return dout


def _check_growth(
    ne_m3=None,
    lnG=None,
    Epar_Vm=None,
    kTe_eV=None,
    Zeff=None,
):

    # -----------------
    # preliminary: lnG
    # -----------------

    if lnG is None:
        lnG = _LNG

    # -----------------------
    # all broadcastable
    # -----------------------

    dparams, shape = ds._generic_check._check_all_broadcastable(
        return_full_arrays=False,
        **locals(),
    )
    lk = ['ne_m3', 'lnG', 'Epar_Vm', 'kTe_eV', 'Zeff']
    lout = [dparams[k0] for k0 in lk]

    return lout
