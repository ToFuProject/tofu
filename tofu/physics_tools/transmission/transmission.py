# -*- coding: utf-8 -*-


import os


import numpy as np
import scipy.interpolate as scpinterp
import datastock as ds


# #################
# #################
# DEFAULTS
# #################


# PATH
_PATH_HERE = os.path.dirname(__file__)
_PATH_INPUTS = os.path.join(_PATH_HERE, 'inputs_filter')


# MASS DENSITIE, g/cm3
_DMASS_DENS = {
    'LaBr3': 5.06,
}


# AVAILABLE MATERIALS
_LMAT_LENGTH = [
    ss for ss in os.listdir(_PATH_INPUTS)
    if ss.startswith('AttenuationLength_')
    and ss.endswith('.txt')
    and ss.count('_') == 1
]
_LMAT_LENGTH = [ss.split('_')[1].replace('.txt', '') for ss in _LMAT_LENGTH]
_LMAT_MASS = [
    ss for ss in os.listdir(_PATH_INPUTS)
    if ss.endswith('_MassAttenuationCoef.csv')
    and ss.count('_') == 1
]
_LMAT_MASS = [ss.split('_')[0] for ss in _LMAT_MASS]
_LMAT_MASS = [ss for ss in _LMAT_MASS if ss in _DMASS_DENS.keys()]
_LOK_MAT = _LMAT_LENGTH + _LMAT_MASS


# #################################################################
# #################################################################
#               main
# #################################################################


def main(
    dthick=None,
    E=None,
    # plotting
    plot=None,
):
    """ Return dict of linear transmission

    dthick is a dict of:
        - 'key': {
            'mat': str (refers to available),
            'thick': thickness (scalar or array, in meters),
        }

    Several different keys can use the same material if needs be

    If plot=True, 'key' will be used for labelling

    E = array of energy values in eV

    """

    # ---------------
    # check inputs
    # ---------------

    dthick, E, plot = _check(
        dthick=dthick,
        E=E,
        plot=plot,
    )

    # -----------------
    # initialize output
    # -----------------

    dout = {
        'E': {
            'data': E,
            'units': 'eV',
        },
        'keys': {},
        'total': {},
    }

    # ---------------
    # loop on keys
    # ---------------

    transtot = None
    for k0, v0 in dthick.items():

        # get transmission function from type
        if v0['type'] == 'length':
            func = _get_transmission_from_length
        else:
            func = _get_transmission_from_mass

        # compute
        trans = func(
            mat=v0['mat'],
            E=E,
            thick=v0['thick'],
        )

        # store
        dout['keys'][k0] = {
            'trans': trans,
            'thick': v0['thick'],
        }

        # transtot
        if transtot is None:
            transtot = trans
        else:
            transtot = transtot * trans

    # -------------------
    # total transmission
    # -------------------

    dout['total'] = transtot

    # --------------------
    # plot
    # --------------------

    if plot is True:
        pass
        # fig = plt.figure()

        # ax = fig.add_subplot()

        # for kloc in  dfilter[boxi].keys():
        # ax.semilogy(
        # E,
        # dfilter[boxi][kloc]['total'],
        # ls='-',
        # lw=1,
        # label=kloc,
        # )
        # ax.legend()

    return dout


# ###############################################################################
# ###############################################################################
#                Check inputs
# ###############################################################################


def _check(
    dthick=None,
    E=None,
    plot=None,
):

    # -----------------
    # dthick
    # -----------------

    # -----------------
    # first check

    c0 = (
        isinstance(dthick, dict)
        and all([
            isinstance(k0, str)
            and isinstance(v0, dict)
            for k0, v0 in dthick.items()
        ])
    )
    if not c0:
        _error_dthick(dthick)

    # -----------------
    # second check

    dfail = {}
    for k0, v0 in dthick.items():

        # material
        mat = v0.get('mat')
        if mat not in _LOK_MAT:
            dfail[k0] = f"'mat' {mat} not available"
            continue

        # type
        if mat in _LMAT_LENGTH:
            dthick[k0]['type'] = 'length'
        else:
            dthick[k0]['type'] = 'mass'

        # thickness
        thick = v0.get('thick')
        try:
            thick = np.atleast_1d(thick).astype(float)
            assert np.all(np.isfinite(thick))
            assert np.all(thick > 0.)
            dthick[k0]['thick'] = thick
        except Exception as err:
            dfail[k0] = (
                "'thick' must be convertible to a strictly"
                " positive finite array of floats"
            )
            raise err

    # ----------------------
    # Raise Exception if any

    if len(dfail) > 0:
        _error_dthick(dthick, dfail=dfail)

    # -----------------
    # E
    # -----------------

    # ---------
    # E itself

    try:
        E = np.asarray(E).astype(float)
        assert np.all(np.isfinite(E))
        assert np.all(E > 0)
    except Exception as err:
        msg = (
            "Arg E must be convertible to a np.ndarray "
            "of strictly positive float\n"
            f"Provided:\n{E}\n"
        )
        raise Exception(msg) from err

    # ------------------
    # Broadcastability
    # -------------------

    dthick, shapef = ds._generic_check._check_all_broadcastable(
        E=E,
        **dthick,
    )

    E = dthick['E']
    dthick = {k0: v0 for k0, v0 in dthick.items() if k0 != 'E'}

    # -----------------
    # plot
    # -----------------

    plot = ds._generic_check._check_var(
        plot, 'plot',
        types=bool,
        default=True,
    )

    return dthick, E, plot


# #################
# Exception
# #################


def _error_dthick(dthick, dfail=None):

    msg = (
        "Arg dthick must be a dict with a series of:\n"
        "\t- 'key0': {'mat': str, 'thick': float or array},\n"
        "\t- 'key1': {'mat': str, 'thick': float or array},\n"
        "\t- ...\n"
        "\t- 'key2N': {'mat': str, 'thick': float or array},\n"
        "Where 'mat' must refer to an available material, in:\n"
        f"{_LOK_MAT}\n\n"
    )

    if dfail is None:
        msg += (
            "Provided:\n"
            f"{dthick}"
        )

    else:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg += (
            "\n" + "\n".join(lstr)
        )

    raise Exception(msg)


# ###############################################################################
# ###############################################################################
#               get transmission
# ###############################################################################


def _get_transmission_from_length(mat=None, E=None, thick=None):

    # load absorption length
    name = f"AttenuationLength_{mat}.txt"
    pfe = os.path.join(_PATH_INPUTS, name)
    # length in microns
    Att_E, Att_L = np.loadtxt(pfe).T
    # length microns => m
    Att_L = Att_L * 1e-6

    # compute transmision
    length = 10**(scpinterp.interp1d(
        np.log10(Att_E),
        np.log10(Att_L),
        kind='linear',
        bounds_error=False,
        fill_value=(np.log10(Att_L[0]), np.log10(Att_L[-1])),
    )(np.log10(E)))

    return np.exp(-thick / length)


def _get_transmission_from_mass(mat=None, E=None, thick=None):

    """

    absorp = 1 - np.exp(
        - 0.02 / scpinterp.interp1d(
            E_abs * 1e6,
            att_length,
            kind='linear',
            bounds_error=False,
            fill_value=(att_length[0], att_length[-1]),
        )(E)
    )
    """

    # crystal absorption (MeV, cm2/g)
    name = f"{mat}_MassAttenuationCoef.csv"
    pfe = os.path.join(_PATH_INPUTS, name)
    E_abs, coef_mass = np.loadtxt(pfe, delimiter=',').T

    # intermediates
    mass_density = _DMASS_DENS[mat]
    # length, m
    att_length = 1e-2 / (coef_mass * mass_density)

    # interpolat log log
    length = 10**(scpinterp.interp1d(
        np.log10(E_abs) + 6.,  # MeV => eV
        np.log10(att_length),
        kind='linear',
        bounds_error=False,
        fill_value=(np.log10(att_length[0]), np.log10(att_length[-1])),
    )(np.log10(E)))

    trans = np.exp(-thick / length)

    return trans
