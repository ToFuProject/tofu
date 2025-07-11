

import numpy as np
import scipy.special as scpspe
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


from ._def import _DMAT


# ########################################################
# ########################################################
#                Default
# ########################################################


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
    # material, thickness
    mat=None,
    thick_m=None,
    # space / time
    x_m=None,
    t_s=None,
    # conditions
    T0_C=None,
    q0_Wm2=None,
    # spevific to finite slab
    lamb=None,
    nterms=None,
    # assumption
    assume=None,
    # plotting
    plot=None,
    dax=None,
    color=None,
    ls=None,
):
    """ Compute the 1d temperature profile in a medium at chosen times

    Considers a uniform medium of chosen material 'mat' at initial uniform T
    Evolved the 1d temperature profile assuming boundary conditions

    Return
    ------
    dout: dict
        dictionary of output results
    dax: dict
        dictionary of axes for plotting

    """

    # --------------
    # check inputs
    # --------------

    din = _check(**locals())

    # --------------
    # get func sol
    # --------------

    # func
    func = _DSOL[din['assume']]['func']

    # kwdards
    lmat = ['gamma', 'kappa']
    lcond = ['T0_C', 'q0_Wm2']

    # apply
    T_C = func(
        x_m=din['x_m'][:, None],
        t_s=din['t_s'][None, :],
        # specific to finite
        H_m=din['thick_m'],
        lamb=din['lamb'],
        nterms=din['nterms'],
        # generic
        **{kk: vv['data'] for kk, vv in din.items() if kk in lmat + lcond}
    )

    # --------------
    # compute
    # --------------

    dout = dict(din)
    dout.update({
        'T_C': {
            'data': T_C,
            'units': 'C',
            'ref': ('nx', 'nt'),
        },
    })

    # --------------
    # plot
    # --------------

    if din['plot'] is True:

        # prepare
        if dax is None:
            dax = _get_dax(dout)

        # plot
        _plot(
            dax=dax,
            dout=dout,
            # options
            color=color,
            ls=ls,
        )

    return dout, dax


# ########################################################
# ########################################################
#                check
# ########################################################


def _check(**kwdargs):

    din = dict(kwdargs)

    # ------------
    # material
    # ------------

    din['mat'] = ds._generic_check._check_var(
        din['mat'], 'mat',
        types=str,
        allowed=list(_DMAT.keys()),
    )

    # ------------
    # material properties
    # ------------

    for k0, v0 in _DMAT[din['mat']].items():
        din[k0] = v0

    # ------------
    # thickness
    # ------------

    din['thick_m'] = float(ds._generic_check._check_var(
        din['thick_m'], 'thick_m',
        types=(float, int),
        sign='>0',
    ))

    # ------------
    # distance
    # ------------

    # distance
    if din.get('x_m') is None:
        din['x_m'] = np.linspace(0, din['thick_m'], 100)

    din['x_m'] = ds._generic_check._check_flat1darray(
        din['x_m'], 'x_m',
        dtype=float,
        unique=True,
    )

    # ------------
    # time
    # ------------

    # time
    msg = (
        "Arg 't_s' must be a 1d time vector in seconds at which the "
        "temperature profile will be plotted!\n"
        "e.g.: t = np.linspace(0, 2e-3, 5)\n"
    )

    din['t_s'] = ds._generic_check._check_flat1darray(
        din['t_s'], 't_s',
        dtype=float,
        unique=True,
        extra_msg=msg,
    )

    # ------------
    # assume
    # ------------

    lstr = []
    for k0, v0 in _DSOL.items():
        lstri = "\n".join([f"\t\t- {k1}" for k1 in v0['comments']])
        lstr.append(f"\t- {k0}:\n{lstri}")
    msg = (
        "Arg 'assume' must be one of:\n"
        + "\n".join(lstr)
        + f"\nProvided: {din.get('assume')}"
    )

    din['assume'] = ds._generic_check._check_var(
        din['assume'], 'assume',
        types=str,
        allowed=list(_DSOL.keys()),
        extra_msg=msg,
    )

    # ------------
    # conditions
    # ------------

    # T0
    msg = (
        "Arg 'T0_C' must be a scalar\n"
        "It is the initial uniform temperature of the medium\n"
        "e.g.: T0_C = 300\n"
    )

    T0_C = float(ds._generic_check._check_var(
        din['T0_C'], 'T0_C',
        types=(float, int),
        sign='>=0',
        extra_msg=msg,
    ))
    din['T0_C'] = {
        'data': T0_C,
        'units': 'C',
        'name': "initial uniform temperature",
    }

    # q0
    msg = (
        "Arg 'q0_Wm2' must be a positive scalar\n"
        "It is the power flux at x = 0\n"
        "e.g.: q0_Wm2 = 1e6\n"
    )

    q0_Wm2 = float(ds._generic_check._check_var(
        din['q0_Wm2'], 'q0_Wm2',
        types=(float, int),
        sign='>=0',
        extra_msg=msg,
    ))
    din['q0_Wm2'] = {
        'data': q0_Wm2,
        'units': 'W/m2',
        'name': "constant heat flux at x = 0",
    }

    # ------------
    # specific to finite
    # ------------

    # lamb
    din['lamb'] = float(ds._generic_check._check_var(
        din['lamb'], 'lamb',
        types=(float, int),
        sign='>=0',
        default=1.,
    ))

    # nterms
    din['nterms'] = int(ds._generic_check._check_var(
        din['nterms'], 'nterms',
        types=(float, int),
        sign='>=0',
        default=10,
    ))

    # ------------
    # plotting
    # ------------

    # plot
    din['plot'] = ds._generic_check._check_var(
        din['plot'], 'plot',
        types=bool,
        default=True,
    )

    return din


# ########################################################
# ########################################################
#                solutions
# ########################################################


def _func_0flux_infinite(
    x_m=None,
    t_s=None,
    # material props
    kappa=None,
    gamma=None,
    # conditions
    T0_C=None,
    q0_Wm2=None,
    # unused
    **kwdargs,
):
    """ Analytical solution to non-steady heat conduction equation
    Assuming semi-infinite slab in cartesian coordinates
    Initial condition: T = T0 everywhere
    Boundary: constant heat flux q0 at x = 0

    [1] “1­D Thermal Diffusion Equation and Solutions,” MIT, 3.185, 2003.
        https://ocw.mit.edu/courses/3-185-transport-phenomena-in-materials-engineering-fall-2003/927ddb6b3dfc423de5570a0070614129_handout_htrans.pdf

    """
    term_exp = 2*np.sqrt(gamma*t_s/np.pi) * np.exp(-x_m**2/(4.*gamma*t_s))
    term_erf = x_m*(1 - scpspe.erf(x_m / (2*np.sqrt(gamma*t_s))))
    return T0_C + (q0_Wm2/kappa) * (term_exp - term_erf)


def _func_0flux_1T(
    x_m=None,
    t_s=None,
    # material props
    kappa=None,
    gamma=None,
    # conditions
    T0_C=None,
    q0_Wm2=None,
    # slab
    H_m=None,
    lamb=None,
    # nterms
    nterms=10,
    # unused
    **kwdargs,
):
    """ Analytical solution to non-steady heat conduction equation
    Assuming finite slab of thickness H, in cartesian coordinates
    Initial condition: T = T0 everywhere
    Boundary:
        - constant heat flux q0 at x = 0
        - T = T0 at x = H

    [1] S. M. Pineda et al., Journal of Heat Transfer, 133, 7, 2011
        doi: 10.1115/1.4003544.

    """

    # xi (normalized x)
    xi = x_m[None, ...] / H_m

    # Fourier factor - noramilzed time
    F = gamma * t_s[None, ...] / H_m**2

    # m
    m = np.arange(1, nterms + 1)
    shape = (m.size,) + tuple([1]*x_m.ndim)
    m = m.reshape(shape)

    # betam
    betam = (2*m - 1) * np.pi / (2.*lamb)

    # each term
    terms = np.cos(betam * xi) * (1. - np.exp(-betam**2 * F)) / betam**2

    # normalized temp
    theta = (2./lamb) * np.sum(terms, axis=0)

    return T0_C + theta * H_m * q0_Wm2 / kappa


def _func_0flux_1flux(
    x_m=None,
    t_s=None,
    # material props
    kappa=None,
    gamma=None,
    # conditions
    T0_C=None,
    q0_Wm2=None,
    # slab
    H_m=None,
    lamb=None,
    # nterms
    nterms=10,
    # unused
    **kwdargs,
):
    """ Analytical solution to non-steady heat conduction equation
    Assuming finite slab of thickness H, in cartesian coordinates
    Initial condition: T = T0 everywhere
    Boundary:
        - constant heat flux q0 at x = 0
        - T = T0 at x = H

    [1] S. M. Pineda et al., Journal of Heat Transfer, 133, 7, 2011
        doi: 10.1115/1.4003544.

    """

    # xi (normalized x)
    xi = x_m[None, ...] / H_m

    # Fourier factor - noramilzed time
    F = gamma * t_s[None, ...] / H_m**2

    # m
    m = np.arange(2, nterms+1)
    shape = (m.size,) + tuple([1]*x_m.ndim)
    m = m.reshape(shape)

    # betam
    betam = (m-1) * np.pi / lamb

    # factor
    fact = 2./lamb

    # each term
    terms = (
        fact * np.cos(betam * xi) * (1. - np.exp(-betam**2 * F)) / betam**2
    )

    # normalized temp
    theta = F[0, ...]/lamb + np.sum(terms, axis=0)

    return T0_C + theta * H_m * q0_Wm2 / kappa


# ########################################################
# ########################################################
#                dax
# ########################################################


def _get_dax(din):

    # ---------------
    # prepare
    # ---------------

    fontsize = 14

    tit = (
        "Heat transport in a 1d solid medium\n"
        f"Assumes: {din['assume']}"
    )

    dmargin = {
        'left': 0.08, 'right': 0.97,
        'bottom': 0.08, 'top': 0.90,
        'wspace': 0.20, 'hspace': 0.30,
    }

    # ---------------
    # create figure
    # ---------------

    dax = {}
    fig = plt.figure(figsize=(13, 9))
    fig.suptitle(tit, size=fontsize, fontweight='bold')
    gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, **dmargin)

    # ---------------
    # create axes
    # ---------------

    # T_C
    ax = fig.add_subplot(gs[:-1, :])
    ax.set_xlabel('X (mm)', size=fontsize, fontweight='bold')
    ax.set_ylabel('T (C)', size=fontsize, fontweight='bold')
    ax.set_title('T profiles in slab', size=fontsize, fontweight='bold')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    dax['T_C'] = {'handle': ax}

    # T0_C
    ax = fig.add_subplot(gs[-1, -1])
    ax.set_xlabel('t (ms)', size=fontsize, fontweight='bold')
    ax.set_ylabel('T(x=0) (C)', size=fontsize, fontweight='bold')
    ax.set_title('Surface T vs time', size=fontsize, fontweight='bold')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    dax['T_x0_C'] = {'handle': ax}

    # text
    ax = fig.add_subplot(gs[-1, 0], frameon=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks([])
    plt.yticks([])
    dax['text'] = {'handle': ax}

    return dax


# ########################################################
# ########################################################
#                Plot
# ########################################################


def _plot(
    dax=None,
    dout=None,
    color=None,
    ls=None,
):

    fontsize = 12

    # ---------------
    #    prepare
    # ---------------

    # colors
    if color is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    else:
        colors = [color]
    ncolors = len(colors)

    # ls
    if ls is None:
        ls = '-'

    # ---------------
    #    plot vs X
    # ---------------

    kax = 'T_C'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for it in range(dout['t_s'].size):
            lab = f"t = {round(dout['t_s'][it]*1e3, ndigits=1)} ms"

            l0, = ax.plot(
                dout['x_m']*1e3,
                dout['T_C']['data'][:, it],
                label=lab,
                ls=ls,
                color=colors[it % ncolors],
            )

            ax.text(
                dout['x_m'][0]*1e3,
                dout['T_C']['data'][0, it],
                lab,
                color=l0.get_color(),
                fontsize=fontsize,
                horizontalalignment='left',
                verticalalignment='bottom',
            )

        # T0_C
        ax.axhline(
            dout['T0_C']['data'],
            ls='--',
            c='k',
        )

        # Tfus
        ax.axhline(
            _DMAT[dout['mat']]['Tfus']['data'],
            ls='--',
            c='k' if color is None else color,
        )

        ax.text(
            dout['x_m'][0]*1e3,
            dout['T_C']['data'][0, it],
            lab,
            color=l0.get_color(),
            fontsize=fontsize,
            horizontalalignment='left',
            verticalalignment='bottom',
        )

        # thickness
        ax.axvline(
            dout['thick_m']*1e3,
            ls='--',
            c='k',
        )

        ax.set_ylim(bottom=0)

    # ---------------
    #    plot T0 vs t
    # ---------------

    kax = 'T_x0_C'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        l0, = ax.plot(
            dout['t_s']*1e3,
            dout['T_C']['data'][0, :],
            ls=ls,
            color=colors[it % ncolors],
        )

        # T0_C
        ax.axhline(
            dout['T0_C']['data'],
            ls='--',
            c='k',
        )

        # Tfus
        ax.axhline(
            _DMAT[dout['mat']]['Tfus']['data'],
            ls='--',
            c='k' if color is None else color,
        )

        ax.set_ylim(bottom=0)

    # ---------------
    #    plot text
    # ---------------

    kax = 'text'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        lk = ['kappa', 'c', 'rho', 'gamma']
        lstr = []
        for kk in lk:
            if kk == 'gamma':
                val = f"{dout[kk]['data']:3.2e}"
            else:
                val = round(dout[kk]['data'])
            lstr.append(f"{kk} = {val} {dout[kk]['units']}")

        q0str = f"{round(dout['q0_Wm2']['data']*1e-6)} MW/m2"
        msg = (
            f"Initial T0 = {dout['T0_C']['data']} {dout['T0_C']['units']}\n"
            f"Power flux q0 = {q0str}\n\n"
            f"Material: {dout['mat']}\n"
            + "\n".join(lstr)
        )

        ax.text(
            0.,
            1,
            msg,
            size=16,
            color=color,
            fontweight='bold',
            transform=ax.transAxes,
            horizontalalignment='left',
            verticalalignment='top',
        )

    return


# ########################################################
# ########################################################
#                _DSOL
# ########################################################


_DSOL = {
    '0flux_infinite': {
        'func': _func_0flux_infinite,
        'comments': [
            'semi-infinite slab from x>=0',
            'uniform T0 (C) at t = 0',
            'constant heat flux q0 (W/m2) at x = 0',
        ],
    },
    '0flux_1T': {
        'func': _func_0flux_1T,
        'comments': [
            'finite slab from x in [0, H]',
            'uniform T0 (C) at t = 0',
            'T(x=H) = T0 at all t',
            'constant heat flux q0 (W/m2) at x = 0',
        ],
    },
    '0flux_1flux': {
        'func': _func_0flux_1flux,
        'comments': [
            'finite slab from x in [0, H]',
            'uniform T0 (C) at t = 0',
            'T(x=H) = T0 at all t',
            'constant heat flux q0 (W/m2) at x = 0',
        ],
    },
}
