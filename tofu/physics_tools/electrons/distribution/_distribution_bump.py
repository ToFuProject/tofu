

import numpy as np
import scipy.constants as scpct
import astropy.units as asunits


from .. import _convert


# #####################################################
# #####################################################
#           Dict of functions
# #####################################################


def f2d_momentum_pitch(
    pnorm=None,
    pitch=None,
    # params
    step=None,
    pnorm0=None,
    pmin=None,
    pnormW=None,
    pitch_width=None,
    E_hat=None,
    Zeff=None,
    # unused
    **kwdargs,
):
    """ Ad-hoc bump on tail

    """
    # B = (E_hat + 1) / (Zeff + 1)
    # B = 1.

    shape = np.broadcast_shapes(
        pnorm.shape,
        pitch.shape,
        E_hat.shape,
    )
    pnorm = np.broadcast_to(pnorm, shape)
    iok = np.broadcast_to((pitch > 0.) & (pnorm > 0.), shape)

    dist = np.zeros(shape, dtype=float)

    dreicer_like = pnorm[iok]

    drop = np.exp(-(pnorm - pnorm0)**2/(2*pnormW)**2)[iok]
    ione = (pnorm <= pnorm0)[iok]
    drop[ione] = 1.
    ilow = (pnorm <= pmin)[iok]
    drop[ilow] = 0.

    pitch_term = np.broadcast_to(
        np.exp(-(1 - pitch**2)**2 / pitch_width**2),
        shape,
    )[iok]

    dist[iok] = drop * (dreicer_like + 0) * pitch_term
    units = asunits.Unit('')

    return dist, units


def f2d_momentum_theta(
    pnorm=None,
    theta=None,
    # params
    step=None,
    pnorm0=None,
    pmin=None,
    pnormW=None,
    pitch_width=None,
    E_hat=None,
    Zeff=None,
    # unused
    **kwdargs,
):
    dist0, units0 = f2d_momentum_pitch(
        pnorm=pnorm,
        pitch=np.cos(theta),
        # params
        step=step,
        pnorm0=pnorm0,
        pmin=pmin,
        pnormW=pnormW,
        pitch_width=pitch_width,
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
    step=None,
    pnorm0=None,
    pmin=None,
    pnormW=None,
    pitch_width=None,
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
        step=step,
        pnorm0=pnorm0,
        pmin=pmin,
        pnormW=pnormW,
        pitch_width=pitch_width,
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
    step=None,
    pnorm0=None,
    pmin=None,
    pnormW=None,
    pitch_width=None,
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
        step=step,
        pnorm0=pnorm0,
        pmin=pmin,
        pnormW=pnormW,
        pitch_width=pitch_width,
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
    'f2d_E_theta_bump': {
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
