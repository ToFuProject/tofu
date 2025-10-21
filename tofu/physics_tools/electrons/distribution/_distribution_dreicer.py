

import numpy as np
import scipy.constants as scpct
import scipy.special as scpsp
import astropy.units as asunits


from .. import _convert


# #####################################################
# #####################################################
#           Elementary functions
# #####################################################


def f2d_ppar_pperp(
    p_par_norm=None,
    p_perp_norm=None,
    Cs=None,
    Etild=None,
    Zeff=None,
    # unused
    **kwdargs,
):
    """ See [1], eq. (7-8)

    [1] S. P. Pandya et al., Phys. Scr., 93, p. 115601, 2018
        doi: 10.1088/1402-4896/aaded0.
    """

    shape = np.broadcast_shapes(
        p_par_norm.shape,
        p_perp_norm.shape,
        Etild.shape,
    )
    iok = np.broadcast_to(p_par_norm > 0, shape)
    p_par_norm = np.broadcast_to(p_par_norm, shape)

    # pper2par
    pperp2par = np.zeros(shape, dtype=float)
    pperp2par[iok] = (
        np.broadcast_to(p_perp_norm**2, shape)[iok]
        / p_par_norm[iok]
    )

    # Hypergeometric confluent Kummer function
    term1 = 1 - Cs / (Etild + 1)
    term2 = ((Etild + 1) / (2.*(1. + Zeff))) * pperp2par
    F1 = np.zeros(shape, dtype=float)
    F1[iok] = scpsp.hyp1f1(np.broadcast_to(term1, shape)[iok], 1, term2[iok])

    # ppar_exp_inv
    ppar_exp_inv = np.zeros(shape, dtype=float)
    power = np.broadcast_to((Cs - 2.) / (Etild - 1.), shape)[iok]
    ppar_exp_inv[iok] = 1. / (p_par_norm[iok]**power)

    # exponential
    exponential = np.exp(-((Etild + 1) / (2 * (1 + Zeff))) * pperp2par)

    # distribution
    dist = np.zeros(shape, dtype=float)
    iok = np.isfinite(F1)
    dist[iok] = ppar_exp_inv[iok] * exponential[iok] * F1[iok]

    # units
    units = asunits.Unit('')

    return dist, units


def f2d_momentum_pitch(
    pnorm=None,
    pitch=None,
    # params
    E_hat=None,
    Zeff=None,
    # unused
    **kwdargs,
):
    """ See [1]
    [1] https://soft2.readthedocs.io/en/latest/scripts/DistributionFunction/ConnorHastie.html#module-distribution-connor
    """
    B = (E_hat + 1) / (Zeff + 1)

    shape = np.broadcast_shapes(
        pnorm.shape,
        pitch.shape,
        E_hat.shape,
    )
    iok = np.broadcast_to((pitch > 0.) & (pnorm > 0.), shape)
    dist = np.zeros(shape, dtype=float)
    dist[iok] = (
        np.exp(-0.5*B * (1 - pitch**2) * pnorm / np.abs(pitch))[iok]
        / np.broadcast_to(pnorm * pitch, shape)[iok]
    )
    units = asunits.Unit('')
    return dist, units


def f2d_momentum_theta(
    pnorm=None,
    theta=None,
    # params
    E_hat=None,
    Zeff=None,
    # unused
    **kwdargs,
):
    dist0, units0 = f2d_momentum_pitch(
        pnorm=pnorm,
        pitch=np.cos(theta),
        # params
        E_hat=E_hat,
        Zeff=Zeff,
    )

    dist = np.sin(theta) * dist0
    units = units0 * asunits.Unit('1/rad')

    return dist, units


def f2d_E_theta(
    E_eV=None,
    theta=None,
    # params
    E_hat=None,
    Zeff=None,
    # unused
    **kwdargs,
):

    # -----------------------
    # get momentum normalized

    pnorm = _convert.convert_momentum_velocity_energy(
        energy_kinetic_eV=E_eV,
    )['momentum_normalized']['data']

    # ---------
    # get dist0

    dist0, units0 = f2d_momentum_theta(
        pnorm=pnorm,
        theta=theta,
        # params
        E_hat=E_hat,
        Zeff=Zeff,
    )

    # -------------
    # jacobian
    # dp = gam / sqrt(gam^2 - 1)  dgam
    # dgam = dE / mc2

    gamma = _convert.convert_momentum_velocity_energy(
        energy_kinetic_eV=E_eV,
    )['gamma']['data']
    mc2_eV = scpct.m_e * scpct.c**2 / scpct.e

    jac = gamma / np.sqrt(gamma**2 - 1) / mc2_eV

    dist = dist0 * jac
    units = units0 * asunits.Unit('1/eV')

    return dist, units


def f3d_E_theta(
    E_eV=None,
    theta=None,
    # params
    E_hat=None,
    Zeff=None,
    # unused
    **kwdargs,
):

    # ---------
    # get dist0

    dist0, units0 = f2d_E_theta(
        E_eV=E_eV,
        theta=theta,
        # params
        E_hat=E_hat,
        Zeff=Zeff,
    )

    # ---------
    # adjust

    dist = dist0 / (2.*np.pi)
    units = units0 * asunits.Unit('1/rad')

    return dist, units


# #####################################################
# #####################################################
#           Dict of functions
# #####################################################


_DFUNC = {
    'f2d_E_theta_dreicer': {
        'func': f2d_E_theta,
        'latex': (
            r"$dn_e = \int_{E_{min}}^{E_{max}} \int_0^{\pi}$"
            r"$f^{2D}_{E, \theta}(E, \theta) dEd\theta$"
            + "\n" +
            r"\begin{eqnarray*}"
            r"\end{eqnarray*}"
        ),
    },
}
