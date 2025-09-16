

import numpy as np
import scipy.constants as scpct
import datastock as ds


# ############################################
# ############################################
#        main
# ############################################


def get_maxwellian(
    # plasma paremeters
    Te_eV=None,
    ne_m3=None,
    jp_Am2=None,
    # coordinate: momentum
    p=None,
    p_perp=None,
    p_par=None,
    # coordinate: velocity
    v_ms=None,
    v_perp_ms=None,
    v_par_ms=None,
    # coordinate: energy
    E_eV=None,
    E_perp_eV=None,
    E_par_eV=None,
    # return as
    returnas=None,
    key=None,
):
    """ Return a thermal single or double maxwellian disctribution
    Expressed as an interpolation vs desired coordinates and units

    If 1d, it is assumed to p = p_par and p_perp = 0

    jp_Am2 is the local plasma current used to shift the Maxwellian center

    Makes use of formulas from [1]

    [1] D. Moseev and M. Salewski, Physics of Plasmas, 26, p. 020901, 2019
        doi: 10.1063/1.5085429.
    """

    # ---------------
    # check inputs
    # ---------------

    dinputs = _check(**locals())

    # -------------------
    # 1d Maxwellian vs p_perp and p_par
    # -------------------

    ddata = _get_maxwellian_2d()

    return ddata


# ############################################
# ############################################
#        Check
# ############################################


def _check(
    Te_eV=None,
    nd=None,
    vs_units=None,
    norm=None,
):

    # ---------------
    # Te_eV
    # ---------------

    Te_eV = float(ds._generic_check._check_var(
        Te_eV, 'Te_eV',
        types=(float, int),
        sign='>0',
        default=1e3,
    ))

    # ---------------
    # Te_eV
    # ---------------

    Te_eV = float(ds._generic_check._check_var(
        Te_eV, 'Te_eV',
        types=(float, int),
        sign='>0',
        default=1e3,
    ))

    return (
        Te_eV,
        nd,
        vs_units,
        norm,
    )


# ############################################
# ############################################
#        Maxwellian - 1d
# ############################################


def _get_maxwellian_2d_vs_p(
    Te_eV=None,
    ne_m3=None,
    jp_Am2=None,
    # norm
    norm=None,
):

    # ---------------
    # Prepare
    # ---------------

    me = scpct.m_e
    mc = me * scpct.c

    # kbTe_J
    kbTe_J = Te_eV * scpct.e

    # electron rest energy J
    # E0_J = scpct.c * mc

    # ---------------
    # Maxwell - non-relativistic
    # ---------------

    # mkT
    # mk2T = 2.*me*kbTe_J

    # ---------------
    # Norm vs scale
    # ---------------

    if norm is False:
        dist = None

    # ---------------
    # Format ouput
    # ---------------

    ddata = {
        'dist': {
            'key': None,
            'data': dist,
            # 'units': units,
            # 'ref': ref,
        },
    }

    return ddata


# #####################################################
# #####################################################
#           Elementary Maxwellians
# #####################################################


def f3d_cart_vperp_vpar_norm(v_par, v_perp, vt_par, vt_perp, v0_par):
    return (
        np.exp(-(v_par - v0_par)**2/vt_par**2 - v_perp**2/vt_perp**2)
        / (np.pi**1.5 * vt_par * vt_perp**2)
    )


def f3d_cyl_vperp_vpar_norm(v_par, v_perp, vt_par, vt_perp, v0_par):
    return v_perp * f3d_cart_vperp_vpar_norm(
        v_par, v_perp, vt_par, vt_perp, v0_par,
    )


def f2d_cart_vperp_vpar_norm(v_par, v_perp, vt_par, vt_perp, v0_par):
    return 2 * np.pi * v_perp * f3d_cart_vperp_vpar_norm(
        v_par, v_perp, vt_par, vt_perp, v0_par,
    )


def f2d_E_pitch_norm(E, pitch, kbT_par, kbT_perp, v0_par):
    me = scpct.m_e
    return (
        np.sqrt(E / (np.pi * kbT_par * kbT_perp**2))
        * np.exp(
            - (pitch * np.sqrt(E) - np.sqrt(me/2.) * v0_par)**2 / kbT_par**2
            - (1 - pitch**2)*E / kbT_perp**2
        )
    )
