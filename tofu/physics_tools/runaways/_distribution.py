

import numpy as __np
import scipy.constants as __scpct
import scipy.special as __scpsp
import datastock as __ds


__all__ = [
    'convert_momentum_velocity_energy',
    'get_critical_dreicer_electric_fields',
    'normalized_momentum_distribution',
    'anisotropy_factor',
    'get_growth_source_terms',
    'get_ddcross_brems_ei',
]


# ########################################################################
# ########################################################################
#                 DEFAULTS
# ########################################################################


# see:
# https://docs.plasmapy.org/en/stable/notebooks/formulary/coulomb.html
_SIGMAP = 1.
_LNG = 20.


# ########################################################################
# ########################################################################
#                 Conversions momentum - velocity - energy
# ########################################################################


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
    val = __np.atleast_1d(din[key])

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
        gamma = __np.sqrt(1. / (1. - val**2))

    elif key == 'velocity_ms':
        gamma = __np.sqrt(1. / (1. - (val/__scpct.c)**2))

    elif key == 'momentum_normalized':
        gamma = __np.sqrt(val**2 + 1)

    elif key == 'energy_kinetic_eV':
        mc2_eV = __scpct.m_e * __scpct.c**2 / __scpct.e
        gamma = (val + mc2_eV) / mc2_eV

    else:
        msg = f"key {key} not implemented in _to_gamma()!"
        raise Exception(msg)

    return gamma


def _from_gamma(key, gamma):

    if key == 'beta':
        out = __np.sqrt(gamma**2 - 1) / gamma
        units = None

    elif key == 'velocity_ms':
        out = __scpct.c * __np.sqrt(gamma**2 - 1) / gamma
        units = 'm/s'

    elif key == 'momentum_normalized':
        out = __np.sqrt(gamma**2 - 1)
        units = None

    elif key == 'energy_kinetic_eV':
        mc2_eV = __scpct.m_e * __scpct.c**2 / __scpct.e
        out = mc2_eV * (gamma - 1)
        units = 'eV'

    else:
        msg = f"key {key} not implemented in _from_gamma()!"
        raise Exception(msg)

    return {'data': out, 'units': units}


# ########################################################################
# ########################################################################
#                 Critical and Dreicer electric fields
# ########################################################################


def get_critical_dreicer_electric_fields(
    ne_m3=None,
    kTe_eV=None,
    lnG=None,
):

    # -------------
    # check input
    # -------------

    ne_m3, kTe_eV, lnG = _check_critical_dreicer(
        ne_m3=ne_m3,
        kTe_eV=kTe_eV,
        lnG=lnG,
    )

    # -------------
    # prepare
    # -------------

    # vacuum permittivity in C/(V.m), scalar
    eps0 = __scpct.epsilon_0

    # custom computation intermediates, scalars
    pie02 = __np.pi * eps0**2

    # electron charge (C), scalar
    e = __scpct.e

    # electron rest energy (eV), scalar
    mec2_eV = __scpct.m_e * __scpct.c**2 / e

    # rest energy of electron in eV, scalar
    # mec_eVsm = mec2_eV / scpct.c    # eV.s/m

    # -------------
    # compute
    # -------------

    # critical electric field (V/m)
    Ec_Vm = ne_m3 * e**3 * lnG / (4 * pie02 * mec2_eV)

    # Dreicer electric field
    if kTe_eV is not None:
        Ed_Vm = Ec_Vm * mec2_eV / kTe_eV
    else:
        Ed_Vm = None

    # -------------
    # format output
    # -------------

    dout = {
        'E_C': {
            'data': Ec_Vm,
            'units': 'V/m',
        },
    }

    if Ed_Vm is not None:
        dout['E_D'] = {
            'data': Ed_Vm,
            'units': 'V/m',
        }

    return dout


def _check_critical_dreicer(
    ne_m3=None,
    kTe_eV=None,
    lnG=None,
):

    # -----------------
    # preliminary: lnG
    # -----------------

    if lnG is None:
        lnG = _LNG

    # -----------------
    # broadcastable
    # -----------------

    dparams, shape = __ds._generic_check._check_all_broadcastable(
        ne_m3=ne_m3,
        kTe_eV=kTe_eV,
        lnG=lnG,
    )

    return [dparams[kk] for kk in ['ne_m3', 'kTe_eV', 'lnG']]


# ########################################################################
# ########################################################################
#                 Normalized Momentum Distribution
# ########################################################################


def normalized_momentum_distribution(
    momentum_normalized=None,
    # parameters
    ne_m3=None,
    Zeff=None,
    electric_field_par_Vm=None,
    energy_kinetic_max_eV=None,
    # optional
    lnG=None,
    sigmap=None,
    # options
    plot=None,
):
    """ Return the normalized RE momentum distribution, interpolated at pp

    Depends on:
        - pp: normalized kinetic momentum (variable)
              Assumed to ba a flat np.ndarray of shape (npp,)

    Parameters:
        - ne:_m3 background electron density (1/m3)
        - Zeff: effectove charge
        - Epar_Vm: parallel electric field (V/m)
        - Emax_eV: maximum kinetic energy (eV)
    All assumed to be broadcastable against each other

    Return a distribution of shape = (npp,) + shape of parameters

    Distribution is analytically nornalized

    ref:
    [1] Pandya et al., Physica Scripta 93, no. 11 (November 1, 2018): 115601

    Here we assume a pitch angle of 0:
        - p_perp = 0
        - p_par = pp
    """

    # -------------
    # check inputs
    # -------------

    pp, ne_m3, Zeff, Epar_Vm, Emax_eV, lnG, shape = _check_dist(
        pp=momentum_normalized,
        ne_m3=ne_m3,
        Zeff=Zeff,
        Epar_Vm=electric_field_par_Vm,
        Emax_eV=energy_kinetic_max_eV,
        sigmap=sigmap,
    )

    # pp.shape = (npp,)
    # all others = shape

    # -------------
    # prepare
    # -------------

    # get momentum max from total energy eV.s/m - shape
    pmax = convert_momentum_velocity_energy(
        energy_kinetic_eV=Emax_eV,
    )['momentum_normalized']['data']

    # Critical electric field - shape
    Ec_Vm = get_critical_dreicer_electric_fields(
        ne_m3=ne_m3,
        kTe_eV=None,
        lnG=lnG,
    )['E_C']['data']

    # -------------
    # Intermediates
    # -------------

    # normalized electric field, adim
    Etild = Epar_Vm / Ec_Vm                             # [nt, nr, 1]
    Ehat = (Etild - 1) / (1 + Zeff)               # [nt, nr, 1]

    # adim
    Cz = __np.sqrt(3 * (Zeff + 5) / __np.pi)          # [nt, 1, 1]

    # critical momentum, adim
    pc = 1. / __np.sqrt(Etild - 1.)                 # [nt, nr, 1]

    # Cs
    Cs = (
        Etild
        - ((1 + Zeff)/4) * (Etild - 2) * __np.sqrt(Etild / (Etild - 1))
    )

    # -------------------
    # total distribution
    # ------------------

    kwdargs = {
        'sigmap': sigmap,
        'pp': pp,
        'pmax': pmax,
        'Ehat': Ehat,
        'Cz': Cz,
        'Cs': Cs,
        'lnG': lnG,
    }

    # -----------
    # Initialize

    if shape is None:

        if Etild > 5:
            re_dist = _re_dist_avalanche(**kwdargs)
        elif (2 < Cs < 1 + Etild):
            re_dist = _re_dist_dreicer(**kwdargs)
        else:
            re_dist = __np.nan

    else:

        # initilize
        re_dist = __np.full(shape, __np.nan)

        # avalanche
        ind_av = Etild > 5.
        if __np.any(ind_av):
            kwdargsi = {k0: v0[ind_av] for k0, v0 in kwdargs.items()}
            re_dist[ind_av] = _re_dist_avalanche(**kwdargsi)

        # Dreicer
        ind_dr = (2 < Cs < 1 + Etild)
        if __np.any(ind_dr):
            kwdargsi = {k0: v0[ind_dr] for k0, v0 in kwdargs.items()}
            re_dist[ind_dr] = _re_dist_dreicer(**kwdargsi)

    # --------------------------------
    # Dreicer (primary) dominated

    # make sure sure remove all below Ec
    re_dist[pp < pc] = 0.

    return re_dist


def _check_dist(
    pp=None,
    ne_m3=None,
    Zeff=None,
    Epar_Vm=None,
    Emax_eV=None,
    sigmap=None,
    lnG=None,
):

    # -----------------------
    # sigmap
    # -----------------------

    # Fermi decay width, dimensionless
    # [1] Pandya et al., Physica Scripta 93, no. 11 (November 1, 2018): 115601

    if sigmap is None:
        sigmap = _SIGMAP

    # -----------------
    # preliminary: lnG
    # -----------------

    if lnG is None:
        lnG = _LNG

    # -----------------------
    # all broadcastable
    # -----------------------

    dparams, shape = __ds._generic_check._check_all_broadcastable(
        return_full_arrays=True,
        **locals(),
    )
    lk = ['pp', 'ne_m3', 'zeff', 'Epar_Vm', 'Emax_eV', 'sigmap', 'lnG']
    lout = [dparams[k0] for k0 in lk]

    return lout, shape


def _re_dist_avalanche(
    sigmap=None,
    pp=None,
    pmax=None,
    Ehat=None,
    Cz=None,
    lnG=None,
    # unused
    **kwdargs,
):

    # fermi decay factor, adim
    fermi = 1. / (__np.exp((pp - pmax) / sigmap) + 1.)

    # distribution, adim
    re_dist = (
        (Ehat / (2*__np.pi*Cz*lnG))
        * (1/pp)
        * __np.exp(- pp / (Cz * lnG))
        * fermi
    )

    return re_dist


def _re_dist_dreicer(
    pp=None,
    Etild=None,
    Zeff=None,
    Cs=None,
    # unused
    **kwdargs,
):
    """ Distribution when primary RE generation is dominant
    see eq (7) in:
        Pandya et al. 2018

    """

    # assumption
    p_perp = 0.
    p_par = pp

    # pper2par
    pperp2par = p_perp**2 / p_par

    # Hypergeometric confluent Kummer function
    term1 = 1 - Cs / (Etild + 1)
    term2 = ((Etild + 1) / (2.*(1. + Zeff))) * pperp2par
    F1 = __scpsp.hyp1f1(term1, 1, term2)

    # ppar_exp_inv
    ppar_exp_inv = 1./(p_par**((Cs - 2.) / (Etild - 1.)))

    # exponential
    exponential = __np.exp(-((Etild + 1) / (2 * (1 + Zeff))) * pperp2par)

    # distribution
    re_dist = ppar_exp_inv * exponential * F1

    return re_dist


# ########################################################################
# ########################################################################
#               Bremsstrahlung anisotropy factor
# ########################################################################


def anisotropy_factor(
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
    beta = convert_momentum_velocity_energy(
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


# ########################################################################
# ########################################################################
#            Primary & secondary growth source terms
# ########################################################################


def get_growth_source_terms(
    ne_m3=None,
    lnG=None,
    Epar_Vm=None,
    kTe_eV=None,
    Zeff=None,
):
    """ Return the source terms in the RE dynamic equation

    S_primary: dreicer growth  (1/m3/s)

    S_secondary: avalanche growth (1/s)

    """

    # -------------
    # check inputs
    # -------------

    ne_m3, lnG, Epar_Vm, kTe_eV, Zeff = _check_growth(
        ne_m3=ne_m3,
        lnG=lnG,
        Epar_Vm=Epar_Vm,
        kTe_eV=kTe_eV,
        Zeff=Zeff,
    )

    # -------------
    # prepare
    # -------------

    # vacuum permittivity in C/(V.m), scalar
    eps0 = __scpct.epsilon_0

    # charge
    e = __scpct.e

    # mec2
    mec2_eV = __scpct.m_e * __scpct.c**2 / e

    # mec eV.s/m
    mec = mec2_eV / __scpct.c

    # me2c3 eV**2 / (m/s) = C^2 V^2 s / m
    me2c3 = mec2_eV**2 / __scpct.c

    # Dreicer electric field - shape
    dEcEd = get_critical_dreicer_electric_fields(
        ne_m3=ne_m3,
        kTe_eV=kTe_eV,
        lnG=lnG,
    )

    Ec_Vm = dEcEd['E_C']['data']
    Ed_Vm = dEcEd['E_D']['data']

    # -------------
    # pre-compute
    # -------------

    # term1 (m^3/s)
    term1 = e**4 * lnG / (4 * __np.pi * eps0**2 * me2c3)

    # term2 - unitless
    term2 = (mec2_eV / (2.*kTe_eV))**1.5

    # term3 - unitless
    term3 = (Ed_Vm / Epar_Vm)**(3*(1. + Zeff) / 16.)

    # exp - unitless
    exp = __np.exp(
        -Ed_Vm / (4.*Epar_Vm) - __np.sqrt((1. + Zeff) * Ed_Vm / Epar_Vm)
    )

    # sqrt
    sqrt = __np.sqrt(__np.pi / (3 * (5 + Zeff)))

    # -------------
    # Compute
    # -------------

    S_primary = ne_m3**2 * term1 * term2 * term3 * exp

    S_secondary = sqrt * (e / mec) * (Epar_Vm - Ec_Vm) / lnG

    # -------------
    # format output
    # -------------

    dout = {
        'S_primary': {
            'data': S_primary,
            'units': '1/m3/s',
        },
        'S_secondary': {
            'data': S_secondary,
            'units': '1/s',
        },
    }

    return dout


def _check_growth(
    ne_m3=None,
    lnG=None,
    Epar_Vm=None,
    kTe_eV=None,
    Zeff=None,
):

    # -----------------
    # preliminary: lnG
    # -----------------

    if lnG is None:
        lnG = _LNG

    # -----------------------
    # all broadcastable
    # -----------------------

    dparams, shape = __ds._generic_check._check_all_broadcastable(
        return_full_arrays=False,
        **locals(),
    )
    lk = ['ne_m3', 'lnG', 'Epar_Vm', 'kTe_eV', 'Zeff']
    lout = [dparams[k0] for k0 in lk]

    return lout


# ########################################################################
# ########################################################################
#            Differencial Bremsstrahlung cross-section
# ########################################################################


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
