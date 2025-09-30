

import numpy as np
import scipy.scontants as scpct
import astropy.units as asunits
import datastock as ds


# ##############################################
# ##############################################
#         DEFAULT
# ##############################################


# ##############################################
# ##############################################
#         Main
# ##############################################


def get_isotropic_bremsstrahlung(
    E_ph_eV=None,
    Te_eV=None,
    ne_m3=None,
    nZ_m3=None,
    Z=None,
):

    # ------------
    # check inputs
    # ------------

    dinputs = _check(
        E_ph_eV=E_ph_eV,
        Te_eV=Te_eV,
        ne_m3=ne_m3,
        nZ_m3=nZ_m3,
        Z=Z,
    )

    # ------------
    # compute gauntff
    # ------------

    gauntff = _get_gauntff(
        Te_eV=Te_eV,
    )

    # ------------
    # compute spectrum
    # ------------

    emiss_norm, units0 = _spectrum_norm(
        Te_eV=Te_eV,
        E_ph_eV=E_ph_eV,
        Z=Z,
    )
    emiss = ne_m3 * nZ_m3 * emiss_norm * gauntff / E_ph_eV

    # units
    units = units0 * asunits.Unit('1/(m^3.m^3.eV)')

    # ------------
    # format
    # ------------

    ddata = {
        'emiss': {
            'key': None,
            'data': emiss,
            'units': units,
        },
    }

    # ------------
    # add_inputs
    # ------------

    ddata.update(**dinputs)

    return


# ##############################################
# ##############################################
#         check
# ##############################################


def _check(
    E_ph_eV=None,
    Te_eV=None,
    ne_m3=None,
    nZ_m3=None,
    Z=None,
):

    # ----------------
    # build dinputs
    # ----------------

    dinputs = {}
    for k0, v0 in locals().items():
        if k0 == 'dinputs':
            continue
        if k0 == 'Z':
            key = k0
            units = None
        else:
            ls = k0.split('_')
            units = ls[-1]
            key = '_'.join(ls[:-1])

        dinputs[k0] = {
            'key': key,
            'data': v0,
            'units': units,
        }

    # ----------------
    # format
    # ----------------

    for k0, v0 in dinputs.items():
        dinputs[k0]['data'] = np.atleast_1d(v0['data'])

    # check broadcastable
    _, shapef = ds._generic_check._check_all_broadcastable(
        return_full_arrays=False,
        **{k0: v0['data'] for k0. v0 in dinputs.items()}
    )

    return dinputs


# ##############################################
# ##############################################
#         gauntff
# ##############################################


def _get_gauntff(
    Te_eV=None,
):

    # ---------------
    # prepare
    # ---------------

    # Euler-Mascheronni constant
    # C = 0.5772156649

    # ---------------
    # compute
    # ---------------

    # gauntff = (
    # np.sqrt(3./np.pi)
    # * (
    # np.log()
    # - 5.*C/2.
    # )
    # )

    gauntff = 1.1

    return gauntff


# ##############################################
# ##############################################
#         spectrum
# ##############################################


def _spectrum_norm(
    Te_eV=None,
    E_eV=None,
    Z=None,
):
    """
    Eq (1) in [1]

    [1] W. Halverson, Plasma Phys., 14, pp. 601–604, 1972
        doi: 10.1088/0032-1028/14/6/004.

    """

    # ------------
    # prepare
    # ------------

    h = scpct.h
    e = scpct.e
    c = scpct.c
    me = scpct.m_e

    # ------------
    # compute
    # ------------

    emiss_norm = (
        Z**2
        * (8./(3*np.sqrt(3)))
        * np.sqrt(2*np.pi/Te_eV)
        * e**6 / (h*c**3*me*np.sqrt(me))
        * np.exp(-E_eV / Te_eV)
    )

    # units
    units = asunits.Unit('')

    return emiss_norm, units
