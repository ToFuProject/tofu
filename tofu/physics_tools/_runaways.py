

import numpy as np
import scipy.constants as scpct


__all__ = [
    'convert_momentum_velocity_energy',
    'normalized_momentum_distribution',
    'anisotropy_factor',
]


# ########################################################################
# ########################################################################
#                 Conversions momentum - velocity - energy
# ########################################################################


def convert_momentum_velocity_energy(
    energy_kinetic_eV=None,
    velocity_ms=None,
    momentum_kinetic_normalized=None,
    gamma=None,
    beta=None,
):
    """ Convert any input to all outputs

    returns a dict with, for each ouput key 'data' and 'units'

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

    elif key == 'momentum_kinetic_normalized':
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

    elif key == 'momentum_kinetic_normalized':
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


# ########################################################################
# ########################################################################
#                 Normalized Momentum Distribution
# ########################################################################


def normalized_momentum_distribution(
    pp=None,
    # parameters
    ne=None,
    Zeff=None,
    Epar=None,
    Emax=None,
    # options
    plot=None,
):
    """ Return the normalized RE momentum distribution

    Depends on:
        - pp: normalized kinetic momentum (variable)

    Parameters:
        - ne: background electron density (1/m3)
        - Zeff: effectove charge
        - Epar: parallel electric field (V/m)
        - Emax: maximum kinetic energy (eV)

    Distribution is analytically nornalized

    ref:
    [1] Pandya et al., Physica Scripta 93, no. 11 (November 1, 2018): 115601

    Here we assume a pitch angle of 0:
        - p_perp = 0
        - p_par = E
    """

    # -------------
    # check inputs
    # -------------

    pp, ne, Zeff, Epar, Emax = _check_dist(
        pp=pp,
        ne=ne,
        Zeff=Zeff,
        Epar=Epar,
        Emax=Emax,
    )

    # -------------
    # prepare
    # -------------

    # vacuum ermittivity in C/(V.m), scalar
    eps0 = scpct.epsilon_0

    # rest energy of electron in eV, scalar
    mec2 = scpct.m_e * scpct.c**2 / scpct.e
    mec = mec2 / scpct.c    # eV.s/m

    # get relativistic factors, [1, 1, nE]
    gamma = convert_momentum_velocity_energy(
        momentum_normalized=pp,
    )['gamma']['data']

    # get momentum from total energy eV.s/m
    pp = ve * scpct.m_e * gamma / scpct.e                  # [1, 1, nE]
    pmax = np.sqrt((Emax + mec2)**2 - mec2**2) / scpct.c   # [nt, 1, 1]

    # normalize by mec
    pp = pp / mec        # [1, 1, nE]
    pmax = pmax / mec    # [nt, 1, 1]

    # for dp
    beta_edges = ve_edges / scpct.c                         # [1, 1, nE]
    gam_edges = 1 / np.sqrt(1 - beta_edges**2)              # [1, 1, nE]
    pp_edges = ve_edges * scpct.m_e * gam_edges / scpct.e   # [1, 1, nE]
    dp = np.diff(pp_edges) / mec                            # [1, 1, nE-1]

    # -------------
    # compute
    # -------------

    # Coulomb logarithm
    # see: https://docs.plasmapy.org/en/stable/notebooks/formulary/coulomb.html
    lnG = 20

    # critical electric field (V/m) (mec in J)    # [nt, nr, 1]
    Ec = ne * scpct.e**3 * lnG / (4*np.pi * eps0**2 * (mec2 * scpct.e))

    # rnormalized electric field, adim
    Etild = Epar / Ec                             # [nt, nr, 1]
    Ehat = (Etild - 1) / (1 + Zeff)               # [nt, nr, 1]

    # adim
    Cz = np.sqrt(3 * (Zeff + 5) / np.pi)          # [nt, 1, 1]

    # critical momentum, adim
    pc = 1. / np.sqrt(Etild - 1.)                 # [nt, nr, 1]

    # --------------
    # total distribution
    # --------------

    if np.any(Etild > 5):

        # fermi decay width, eV
        sigmap = 1.

        # fermi decay factor, adim
        fermi = 1. / (np.exp((pp - pmax) / sigmap) + 1.)

        # distribution, adim
        re_dist = (
            (Ehat / (2*np.pi*Cz*lnG))
            * (1/pp)
            * np.exp(- pp / (Cz * lnG))
            * fermi
        )

    else:

        # Cs
        Cs = Etild - ((1 + Zeff)/4) * (Etild - 2) * \
            np.sqrt(Etild / (Etild - 1))

        if not (2 < Cs < 1 + Etild):
            msg = (
                "No valid formulation available\n"
                f"\t- Epar = {Epar}\n"
                f"\t- Ec = {Ec}\n"
                f"\t- Etild = {Etild}\n"
                f"\t- Cs = {Cs}\n"
                f"\t- ne = {ne}\n"
                f"\t- Zeff = {Zeff}\n"
            )
            raise Exception(msg)

        # distribution
        # F1 = None
        # re_dist = (
        #     (1 / (E**((Cs - 2.) / (Etild - 1))))
        #     * np.exp( - (Etild + 1) / (2 * (1 + Zeff)) * (0 / E))
        #     * F1
        # )
        # confluent hypergeometric (kummer) function

        raise NotImplementedError("confluent hypergeometric (kummer) function")

    # make sure sure remove all below Ec
    re_dist[pp < pc] = 0.

    # -------------
    # plot
    # -------------

    if plot is True:
        pass

    return re_dist


def _check_dist(
    pp=None,
    ne=None,
    Zeff=None,
    Epar=None,
    Emax=None,
):

    dparams = {k0: v0 for k0, v0 in locals().items() if k0 != 'pp'}

    # -----------------------
    # pp = normalized momentum
    # -----------------------

    try:
        pp = np.atleast_1d(pp)
    except Exception as err:
        msg = (
            "Arg pp (normalized kinetic momentum) must be a np.ndarray!\n"
            f"Provided:\n{pp}\n"
        )
        raise Exception(msg)

    shape_pp = pp.shape

    # -----------------------
    # all others
    # -----------------------

    dparams = ds._generic_check._uniformize_params_shapes(**dparams)

    return [pp] + [dparams[k0] for k0 in ['ne', 'zeff', 'Epar', 'Emax']]


# ########################################################################
# ########################################################################
#               Bremsstrahlung anisotropy factor
# ########################################################################


def anisotropy_factor(
    gamma=None,
    costheta=None,
):
    """ Return the anisotropic factor

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
    beta = convert_momentum_velocity_energy(
        gamma=gamma,
    )['beta']['data']

    # -----------
    # compute
    # -----------

    # anisotropy of cross-section
    anis = (
        (3/8) * (1 + ((costheta - beta) / (1 - beta*costheta))**2)
        / (gamma**2 * (1 - beta*costheta)**2)
    )

    return anis


# ########################################################################
# ########################################################################
#            Clean up
# ########################################################################


# del np, scpct
