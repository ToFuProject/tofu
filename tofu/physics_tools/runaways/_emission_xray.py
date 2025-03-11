

from . import _utils


# ##############################################################
# ##############################################################
#               Bremsstrahlung anisotropy factor
# ##############################################################


def get_anisotropy_factor(
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


def get_ddcross_brems_ei(
    E_re_eV=None,
    E_ph_eV=None,
):

    # -------------
    # format output
    # -------------

    dout = {
        'ddcross_ei': {
            'data': None,
            'units': '?',
        },
    }

    return dout
