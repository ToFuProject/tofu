

import numpy as np
import scipy.special as scpcpe
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ########################################################
# ########################################################
#                DEFAULT
# ########################################################


_DMAT = {
    'Au': {
        'kappa': {
            'data': 315,
            'units': 'W/m/K',
            'name': 'heat conductivity',
        },
        'rho': {
            'data': 19.3e3,
            'units': 'kg/m3',
            'name': 'mass density',
        },
        'c': {
            'data': 126,
            'units': 'J/kg/K',
            'name': 'specific heat capacity',
        },
        'Tfus': {
            'data': 1064,
            'units': 'C',
            'name': 'fusion temperature',
        },
    },
    'Si3N4': {
        'kappa': {
            'data': 20,
            'units': 'W/m/K',
            'name': 'heat conductivity',
        },
        'rho': {
            'data': 3.17e3,
            'units': 'kg/m3',
            'name': 'mass density',
        },
        'c': {
            'data': 700,
            'units': 'J/kg/K',
            'name': 'specific heat capacity',
        },
        'Tfus': {
            'data': 1900,
            'units': 'C',
            'name': 'fusion temperature',
        },
    },
}


for k0, v0 in _DMAT.items():
    _DMAT[k0]['gamma'] = {
        'data': v0['kappa'] / (v0['c'] * v0['rho']),
        'units': 'W/m2',
        'name': 'thermal diffusivity',
    }


_DLAYERS = {
    'absorber': {
        'mat': 'Au',
        'thick': 0.1e-3,
    },
    'support': {
        'mat': 'Si3N4',
        'thick': 0.6e-3,
    },
    'order': ['absorber', 'support'],
}


# ########################################################
# ########################################################
#                Main
# ########################################################


def main(
    xx=None,
    t=None,
    # conditions
    T0=None,
    q0=None,
    # assumption
    assume=None,
    # layers
    dlayers=_DLAYERS,
):

    # --------------
    # prepare data
    # --------------

    # distance
    if xx is None:
        thick = np.sum([v0['thick'] for v0 in dlayers.values()])
        xx = np.linspace(0, thick, 100)

    # time
    if t is None:
        t = np.linspace(0, 2e-3, 5)

    # --------------
    # get func sol
    # --------------

    # func
    func = _get_sol(assume)

    # apply

    # --------------
    # compute
    # --------------

    dout = None

    # --------------
    # prepare
    # --------------

    dax = _get_dax()

    # --------------
    # plot
    # --------------

    return dout, dax


# ########################################################
# ########################################################
#                solutions
# ########################################################


def _get_sol(
    assume=None,
):

    if assume == 'constant_flux':
        def func(x, t, kappa, gamma, T0, q0):
            term_exp = 2*np.sqrt(gamma*t/np.pi) * np.exp(-x**2/(4.*gamma*t))
            term_erf = x*(1 - scpspe.erf(x / (2*np.sqrt(gamma*t))))
            return T0 + (q0/kappa) * (term_exp - term_erf)

    else:
        raise NotImplementedError()

    return func

# ########################################################
# ########################################################
#                dax
# ########################################################


def _get_dax():

    # ---------------
    # prepare
    # ---------------

    tit = (
        "Distribution of observation combinations\n"
        f"res_RZ = {resRZ_mm} mm, res_phi = {resphi_mm} mm"
    )

    dmargin = {
        'left': 0.15, 'right': 0.97,
        'bottom': 0.10, 'top': 0.90,
        'wspace': 0.25, 'hspace': 0.20,
    }

    # ---------------
    # create figure
    # ---------------

    dax = {}
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, **dmargin)

    # ---------------
    # create axes
    # ---------------

    # temp
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlabel('X (m)', size=fontsize, fontweight='bold')
    ax.set_ylabel('T (C)', size=fontsize, fontweight='bold')
    dax['temp'] = {'handle': ax}



    return
