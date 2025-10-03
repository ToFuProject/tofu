

import numpy as np
import scipy.constants as scpct


# ##############################################################
# ##############################################################
#           Conversions momentum - velocity - energy
# ##############################################################


def convert_momentum_velocity_energy(
    energy_kinetic_eV=None,
    velocity_ms=None,
    momentum_normalized=None,
    gamma=None,
    beta=None,
):
    """ Convert any input to all outputs

    returns a dict with, for each ouput key 'data' and 'units'

    - momentum_normalized: total relativistic momentum / mec2
    - energy_kinetic_eV: kinetic energy in eV
    - gamma: Lorentz factor
    - beta = v / c
    - velocity_ms : velocity in m/s

    see:
    [1] Pandya et al., Physica Scripta 93, no. 11 (November 1, 2018): 115601
    [2] https://en.wikipedia.org/wiki/Energy%E2%80%93momentum_relation

    """

    # ---------------
    # dict in - check
    # ---------------

    din0 = locals()
    din = {k0: v0 for k0, v0 in din0.items() if v0 is not None}

    if len(din) != 1:
        lstr = [f"\t- {k0}" for k0 in din.keys()]
        msg = (
            "Please provide only one input of the following:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    key = list(din.keys())[0]
    val = np.atleast_1d(din[key])

    # -----------------
    # convert to gamma
    # -----------------

    if 'gamma' not in din.keys():
        gamma = _to_gamma(key=key, val=val)

    else:
        gamma = val

    # -----------------
    # initialize dout
    # -----------------

    dout = {
        'gamma': {
            'data': gamma,
            'units': None,
        },
    }

    # -----------------
    # convert from gamma
    # -----------------

    lk = [k0 for k0 in din0.keys() if k0 != 'gamma']
    for k0 in lk:
        dout[k0] = _from_gamma(key=k0, gamma=gamma)

    return dout


def _to_gamma(key, val):

    if key == 'beta':
        gamma = np.sqrt(1. / (1. - val**2))

    elif key == 'velocity_ms':
        gamma = np.sqrt(1. / (1. - (val/scpct.c)**2))

    elif key == 'momentum_normalized':
        gamma = np.sqrt(val**2 + 1)

    elif key == 'energy_kinetic_eV':
        mc2_eV = scpct.m_e * scpct.c**2 / scpct.e
        gamma = (val + mc2_eV) / mc2_eV

    else:
        msg = f"key {key} not implemented in _to_gamma()!"
        raise Exception(msg)

    return gamma


def _from_gamma(key, gamma):

    if key == 'beta':
        out = np.sqrt(gamma**2 - 1) / gamma
        units = None

    elif key == 'velocity_ms':
        out = scpct.c * np.sqrt(gamma**2 - 1) / gamma
        units = 'm/s'

    elif key == 'momentum_normalized':
        out = np.sqrt(gamma**2 - 1)
        units = None

    elif key == 'energy_kinetic_eV':
        mc2_eV = scpct.m_e * scpct.c**2 / scpct.e
        out = mc2_eV * (gamma - 1)
        units = 'eV'

    else:
        msg = f"key {key} not implemented in _from_gamma()!"
        raise Exception(msg)

    return {'data': out, 'units': units}
