

import numpy as np
import scipy.constants as scpct
import datastock as ds


# ##############################################################
# ##############################################################
#              Maxwellian
# ##############################################################


def get_maxwellian(
    kTe_eV=None,
    mass=None,
    # output
    energy_eV=None,
    velocity_ms=None,
):
    """ Return a Maxwell-Boltzmann distribution

    Calculated to be:
        - analytically normalized
        - against the variable provided

    See:
        https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution

    """

    # --------------
    # check input
    # --------------

    kTe_eV, mass, val, key = _check_maxwell(
        kTe_eV=kTe_eV,
        mass=mass,
        # output
        energy_eV=energy_eV,
        velocity_ms=velocity_ms,
    )

    # --------------
    # Compute
    # --------------

    if key == 'velocity_ms':

        dist = (
            (2./np.pi)
            * (mass / kTe_eV)**1.5
            * velocity_ms**2
            * np.exp(-mass*val**2 / (2 * kTe_eV))
        )

    elif key == 'energy_eV':

        dist = (
            2 * np.sqrt(energy_eV / np.pi)
            * (1./kTe_eV)**1.5
            * np.exp(-energy_eV / kTe_eV)
        )

    else:
        raise NotImplementedError()

    # --------------
    # format output
    # --------------

    dout = {
        'maxwell': {
            'vs': key,
            'data': dist,
            'units': None,
        },
    }

    return dout


def _check_maxwell(
    kTe_eV=None,
    mass=None,
    # output
    energy_eV=None,
    velocity_ms=None,
):

    # ------------------
    # energy vs velocity
    # ------------------

    dunique = {
        'energy_eV': energy_eV,
        'velocity_ms': velocity_ms,
    }

    lok = [k0 for k0, v0 in dunique.items() if v0 is not None]
    if len(lok) != 1:
        lstr = [f"\t- {k0}: v0" for k0, v0 in dunique.items()]
        msg = (
            "Please provide only one of the following:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    key = lok[0]
    val = dunique[key]

    # ------------
    # mass
    # ------------

    if key == 'velocity_ms':
        if mass is None:
            mass = scpct.m_e
    else:
        mass = None

    # ---------------
    # broadcastable
    # ---------------

    dout = {
        'kTe_eV': kTe_eV,
        key: val,
    }
    if key == 'velocity_ms':
        dout['mass'] = mass

    dout, shape = ds._generic_check._check_all_broadcastable(
        **dout,
    )

    lout = ['kTe_eV', key, 'mass']
    lout = [dout[k0] for k0 in lout if k0 in dout.keys()]
    if key == 'energy_eV':
        lout.append(None)
    lout.append(key)

    return lout
