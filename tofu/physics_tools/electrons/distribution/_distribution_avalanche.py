

import numpy as np
import scipy.constants as scpct
import astropy.units as asunits


from .. import _convert


# #####################################################
# #####################################################
#           Dict of functions
# #####################################################


def f2d_ppar_pperp(
    p_par_norm=None,
    p_perp_norm=None,
    p_max_norm=None,
    p_crit=None,
    Cz=None,
    lnG=None,
    E_hat=None,
    sigmap=None,
    # unused
    **kwdargs,
):
    """ See [1], eq. (6)

    [1] S. P. Pandya et al., Phys. Scr., 93, p. 115601, 2018
        doi: 10.1088/1402-4896/aaded0.
    """

    shape = np.broadcast_shapes(
        p_par_norm.shape,
        p_perp_norm.shape,
        E_hat.shape,
    )
    p_par_norm = np.broadcast_to(p_par_norm, shape)
    iok = p_par_norm > 0.

    # fermi decay factor, adim
    fermi = np.broadcast_to(
        1. / (np.exp((p_par_norm - p_max_norm) / sigmap) + 1.),
        shape,
    )

    # ratio2
    pperp2par = np.zeros(shape, dtype=float)
    pperp2par[iok] = (
        np.broadcast_to(p_perp_norm, shape)[iok]**2
        / p_par_norm[iok]
    )

    # distribution, adim
    dist = np.zeros(shape, dtype=float)
    exp = np.zeros(shape, dtype=float)
    exp[iok] = np.exp(-p_par_norm / (Cz*lnG) - 0.5*E_hat*pperp2par)[iok]
    dist[iok] = (
        np.broadcast_to(E_hat / (2.*np.pi*Cz*lnG), shape)[iok]
        * (1. / p_par_norm[iok])     # Not in formula, but necessary
        * exp[iok]
        * fermi[iok]
    )

    # critical momentum
    iout = np.sqrt(p_par_norm**2 + p_perp_norm**2) < p_crit
    dist[iout] = 0.

    units = asunits.Unit('')

    return dist, units


def f2d_momentum_theta(
    pnorm=None,
    theta=None,
    p_max_norm=None,
    p_crit=None,
    Cz=None,
    lnG=None,
    E_hat=None,
    sigmap=None,
    # unused
    **kwdargs,
):
    """ Based on f2d_ppar_pperp + jacobian
    """

    dist0, units0 = f2d_ppar_pperp(
        p_par_norm=pnorm * np.cos(theta),
        p_perp_norm=pnorm * np.sin(theta),
        p_max_norm=p_max_norm,
        p_crit=p_crit,
        Cz=Cz,
        lnG=lnG,
        E_hat=E_hat,
        sigmap=sigmap,
    )

    # jacobian
    jac = pnorm

    # dist
    dist = jac * dist0
    units = units0 * asunits.Unit('1/rad')

    return dist, units


def f2d_momentum_pitch(
    pnorm=None,
    pitch=None,
    Cz=None,
    lnG=None,
    E_hat=None,
    Zeff=None,
    # unused
    **kwdargs,
):
    """ Based on [1] eq (2.17)
    [1] O. Embréus et al., J. Plasma Phys., 84, p. 905840506, 2018
        doi: 10.1017/S0022377818001010.

    !!! Reversed convention in paper: electrons accelerated towards -1 !!!

    """
    gamma = _convert.convert_momentum_velocity_energy(
        momentum_normalized=pnorm,
    )['gamma']['data']
    gam0 = lnG * np.sqrt(Zeff + 5.)

    mec_kgms = scpct.m_e * scpct.c
    pp_kgms = pnorm * mec_kgms

    Ap = gamma * (E_hat + 1) / (Zeff + 1)

    # reverse sign of pitch
    pitch = -pitch

    dist = (
        (Ap/(2.*np.pi*mec_kgms*pp_kgms**2*gam0))
        * np.exp(-gamma / gam0 - Ap*(1 + pitch))
        / (1. - np.exp(-2.*Ap))
    )

    units = asunits.Unit('s^3/(kg^3.m^3)')

    return dist, units


def f2d_E_theta(
    E_eV=None,
    theta=None,
    p_max_norm=None,
    p_crit=None,
    Cz=None,
    lnG=None,
    E_hat=None,
    sigmap=None,
    # unused
    **kwdargs,
):
    """ Based on f2d_ppar_pperp + jacobian
    """

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
        p_max_norm=p_max_norm,
        p_crit=p_crit,
        Cz=Cz,
        lnG=lnG,
        E_hat=E_hat,
        sigmap=sigmap,
    )

    # -------------
    # jacobian
    # dp = gam / sqrt(gam^2 - 1)  dgam
    # dgam = dE / mc2

    gamma = _convert.convert_momentum_velocity_energy(
        energy_kinetic_eV=E_eV,
    )['gamma']['data']
    mc2_eV = scpct.m_e * scpct.c**2 / scpct.e

    # jacobian
    jac = gamma / np.sqrt(gamma**2 - 1) / mc2_eV

    # dist
    dist = dist0 * jac
    units = units0 * asunits.Unit('1/eV')

    return dist, units


def f3d_E_theta(
    E_eV=None,
    theta=None,
    p_max_norm=None,
    p_crit=None,
    Cz=None,
    lnG=None,
    E_hat=None,
    sigmap=None,
    # unused
    **kwdargs,
):
    """ Based on f2d_E_theta / 2pi
    """

    # ---------
    # get dist0

    dist0, units0 = f2d_E_theta(
        E_eV=E_eV,
        theta=theta,
        p_max_norm=p_max_norm,
        p_crit=p_crit,
        Cz=Cz,
        lnG=lnG,
        E_hat=E_hat,
        sigmap=sigmap,
    )

    # ---------
    # adjust

    dist = dist0 / (2.*np.pi)
    units = units0 * asunits.Unit('1/rad')

    return dist, units
