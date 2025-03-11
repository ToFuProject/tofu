
import os


import numpy as np


from .. import _utils


# ##############################################################
# ##############################################################
#               Bremsstrahlung anisotropy factor
# ##############################################################


_PATH_HERE = os.path.dirname(__file__)


# ##############################################################
# ##############################################################
#               Bremsstrahlung anisotropy factor
# ##############################################################


def anisotropy(
    gamma=None,
    costheta=None,
):
    """ Return the anisotropic factor (unitless)

    Depends on:
        - gamma: thelorentz factor of the Runaway electron
        - costheta: angle of observation relative to electron direction

    ref:
    [1] Pandya et al., Physica Scripta 93, no. 11 (November 1, 2018): 115601

    """

    # -----------
    # prepare
    # -----------

    # gamma => beta
    beta = _utils.convert_momentum_velocity_energy(
        gamma=gamma,
    )['beta']['data']

    # -----------
    # compute
    # -----------

    # anisotropy of cross-section
    anis = (
        (3/8) * (1 + ((costheta - beta) / (1 - beta * costheta))**2)
        / (gamma**2 * (1 - beta * costheta)**2)
    )

    return anis


# ##############################################################
# ##############################################################
#            Differencial Bremsstrahlung cross-section
# ##############################################################


def ddcross_ei(
    E_re_eV=None,
    E_ph_eV=None,
    costheta=None,
    anisotropy=None,
    # options
    return_intermediates=None,
):
    """ Return the energy-dependent HXR generation cros-section

    Considers electron-ion bremsstrahlung

    To be multiplied by the anisotropy factor
        => see get_anisotropy_factor()

    Sources:
        [1] Nocente et al., Nuclear Fusion 57, no. 7 (July 1, 2017): 076016.
        [2] Salvat et al., Nuclear Instruments and Methods in Physics Research
            Section B: Beam Interactions with Materials and Atoms 63,
            no. 3 (February 1992): 255–69
    """

    # --------------------
    # Load tabulated data
    # --------------------

    # load screening radius
    fname = "RE_HXR_CrossSection_ScreeningRadius_Salvat.csv"
    pfe = os.path.join(_PATH_HERE, fname)
    Z_R, RZ3a0 = np.loadtxt(pfe, delimiter=',').T

    fname = "RE_HXR_ElectronElectron_Salvat.csv"
    pfe = os.path.join(_PATH_HERE, fname)
    Z_eta, eta_inf = np.loadtxt(pfe, delimiter=',').T

    # -----------------------------------
    # cross-section (without anisotropy)
    # ----------------------------------

    # -------------------
    # prepare constants

    # hbar (J.s => eV.s)
    hbar_eVs = scpct.hbar / scpct.e

    # rest energy of electron (J => eV)
    mc2_eV = scpct.m_e * scpct.c**2 / scpct.e

    # mc (eV.s/m)
    mc_eVsm = mc2 / scpct.c

    # fine structure constant (adim.)
    alpha = scpct.alpha

    # a0 = bohr radius (m)
    a0 = 5.291772e-11

    # screening radius (should be tabulate from fig. 4), m
    # R = 0.81 * a0 / Z**(1/3)
    Rz3a0 = scpinterp.interp1d(
        np.round(Z_R),
        RZ3a0,
        kind='linear',
    )(Z)
    R = Rz3a0 * a0 / Z**(1/3)
    print(f"Screening radius for Z = {Z}:\n\t- RZ^(1/3)/a0 = {Rz3a0}\n\t- R = {R}")

    # high-energy coulomb correction (adim.)
    aa = (alpha * Z)**2
    fc = aa * np.sum([1./(nn * (nn**2 + aa)) for nn in range(1, 101)])

    # log(R * mc/hbar)   adim
    logRmchb = np.log(R * mc / hbar)

    # ----------------------
    # prepare (nEe,) vectors

    # should tabulate eta_inf vs Z from graph (cf. fig. 5), adim.
    # eta_inf = 1.158
    eta_inf = scpinterp.interp1d(
        np.round(Z_eta),
        eta_inf,
        kind='linear',
    )(Z)
    print(f"eta_inf for Z = {Z}:\n\t- eta_inf = {eta_inf}")
    eta = (E_e / mc2)**0.8 / ((E_e/mc2)**0.8 + 2.43) * eta_inf

    # correction term at low energies adim
    F2 = (
        (2.04 + 9.09*alpha*Z)
        * (mc2**2 / (E_e * (E_e + mc2))) ** (1.26 - 0.93*alpha*Z)
    )

    # useful for q0 = minimum momentum transfer (eV.s/m)
    gamma = (E_e + mc2) / mc2

    # --------------------------
    # prepare (nEe, nEph) arrays

    # reduced energy of the photon (adim.)
    eps = E_ph[None, :] / (E_e[:, None] + mc2)
    eps[eps > 1] = np.nan
    epsd = (E_e - 5*mc2) / (E_e + mc2)

    # turn off fc wen eps > epsd
    theta = eps < epsd

    # q0 = minimum momentum transfer (eV.s/m)
    q0 = (mc / (2*gamma[:, None])) * eps / (1 - eps)

    # adim
    bb = R * q0 / hbar

    # Phi1
    Phi1 = 2 - 2*np.log(1 + bb**2) - 4*bb*np.arctan(1/bb) + 4*logRmchb

    # Phi2
    term2 = 2*bb**2 * (4 - 4*bb*np.arctan(1/bb) - 3*np.log(1 + 1/bb**2))
    if adjust is True:
        bb_too_large = bb > 1000
        term2[bb_too_large] = -10/3.

    Phi2 = (
        (4/3) - 2*np.log(1 + bb**2)
        + term2
        + 4*logRmchb
    )

    # adim
    f0 = 4*logRmchb + F2 - 4*fc*theta
    f1 = Phi1 - 4*logRmchb
    f2 = 0.5*(3*Phi1 - Phi2) - 4*logRmchb

    phi1 = f1 + f0
    phi2 = (4/3) * (1 - eps) * (f2 + f0)

    # cross-section in m2
    dcross_E_deps = (
        a0**2 * alpha**5 * Z * (Z + eta) * (phi1 * eps + phi2 / eps)
    )

    # change of variable to find derivative vs Eph (m2/eV)
    # deps / dEp = 1/(E_re + mc2)
    dcross_E = dcross_E_deps / (E_e[:, None] + mc2)

    # -------------
    # format output
    # -------------

    dout = {
        'ddcross_ei': {
            'data': None,
            'units': '?',
        },
    }

    # -----------------
    # intermediates
    # -----------------

    if return_intermediates is True:
        pass

    return dout
