

import numpy as np
# import scipy.stats as scpstats
import scipy.integrate as scpinteg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.units as asunits
import datastock as ds


from . import _convert
from . import _distribution_check
from . import _distribution


try:
    plt.rcParams['text.usetex'] = True
except Exception:
    pass


# ############################################
# ############################################
#        Default
# ############################################


_DPLASMA = {
    'Te_eV': {
        'def': 1e3,
        'units': 'eV',
    },
    'ne_m3': {
        'def': 1e19,
        'units': '1/m3',
    },
    'jp_Am2': {
        'def': 1e6,
        'units': 'A/m2',
    },
    'jp_fraction_re': {
        'def': np.r_[0., 0.1, 0.9],
        'units': None,
    },
    'Zeff': {
        'def': 1.,
        'units': None,
    },
    'Ekin_max_eV': {
        'def': 10e6,
        'units': 'eV',
    },
    'Efield_par_Vm': {
        'def': 1.,
        'units': 'V/m',
    },
    'lnG': {
        'def': 20.,
        'units': None,
    },
    'sigmap': {
        'def': 1.,
        'units': None,
    },
}


# DCOORDS
_EMAX_EV = 10e6
_DCOORDS = {
    'E_eV': np.logspace(2, np.log10(_EMAX_EV), 61),
    'ntheta': 31,
    'nperp': 41,
}


# ############################################
# ############################################
#        Maxwellian - 2d
# ############################################


def main(
    # -----------
    # plasma paremeters
    Te_eV=None,
    ne_m3=None,
    jp_Am2=None,
    jp_fraction_re=None,
    # RE-specific
    Zeff=None,
    Ekin_max_eV=None,
    Efield_par_Vm=None,
    lnG=None,
    sigmap=None,
    # -----------
    # coordinates
    E_eV=None,
    ntheta=None,
    nperp=None,
    # plotting
    dax=None,
    fontsize=None,
    dmargin=None,
):

    # ----------------
    # check inputs
    # ----------------

    (
        dplasma,
        dprop,
        dcoords,
    ) = _check(
        **locals(),
    )

    # ----------------
    # Compute
    # ----------------

    # f2D_E_theta
    ddist_E_theta = _distribution.main(
        # coordinate: momentum
        E_eV=dcoords['E_eV'],
        theta=dcoords['theta'],
        # return as
        returnas=dict,
        # version
        dist=('maxwell', 'RE'),
        version='f2d_E_theta',
        # plasma
        **{kk: vv['data'] for kk, vv in dplasma.items()},
    )

    # f2D_vpar_vperp
    ddist_ppar_pperp = _distribution.main(
        # coordinate: momentum
        p_par_norm=dcoords['p_par_norm'],
        p_perp_norm=dcoords['p_perp_norm'],
        # return as
        returnas=dict,
        # version
        dist=('maxwell', 'RE'),
        version='f2d_ppar_pperp',
        # plasma
        **{kk: vv['data'] for kk, vv in dplasma.items()},
    )

    # ----------------
    # Derive 1d
    # ----------------

    # E
    units = ddist_E_theta['dist']['RE']['dist']['units'] * asunits.Unit('rad')
    ddist_E = {
        kdist: {
            'data': scpinteg.trapezoid(
                ddist_E_theta['dist'][kdist]['dist']['data'],
                x=ddist_E_theta['coords']['x1']['data'],
                axis=-1,
            ),
            'units': units,
        }
        for kdist in ddist_E_theta['dist'].keys()
    }

    # v
    # v_edge = 0.5*(v_perp_ms[1:] + v_perp_ms[:-1])
    # dv = v_perp_ms[1] - v_perp_ms[0]
    # v_edge = np.r_[v_edge[0] - dv, v_edge, v_edge[-1] + dv]
    # shape = dout_E['dist']['data'].shape[:-2] + (v_edge.size-1,)
    # dist_1d_v = np.full(shape, np.nan)
    # for ii, ind in enumerate(np.ndindex(shape[:-1])):
    # sli0 = ind + (slice(None),)
    # sli1 = ind + (slice(None), slice(None))
    # dist_1d_v[sli0] = scpstats.binned_statistic(
    # np.sqrt(v_par_ms[:, None]**2 + v_perp_ms[None, :]**2).ravel(),
    # dout_v['dist']['data'][sli1].ravel(),
    # statistic='sum',
    # bins=v_edge,
    # ).statistic

    # ----------------
    # plot
    # ----------------

    dax = _plot(
        ddist_E_theta=ddist_E_theta,
        ddist_ppar_pperp=ddist_ppar_pperp,
        # 1d
        ddist_E=ddist_E,
        # dist_1d_v=dist_1d_v,
        # v_edge=v_edge,
        # props
        dprop=dprop,
        # plotting
        dax=dax,
        fontsize=fontsize,
        dmargin=dmargin,
    )

    return dax, ddist_E_theta, ddist_ppar_pperp


# #####################################################
# #####################################################
#               check
# #####################################################


def _check(
    **kwdargs,
):

    # -----------------
    # plasma parameters
    # -----------------

    dplasma = _distribution_check._plasma(
        ddef=_DPLASMA,
        **kwdargs,
    )
    shape_plasma = dplasma['Te_eV']['data'].shape

    # -----------------
    # Properties
    # -----------------

    dprop = {}
    lc = ['b', 'r', 'g', 'c', 'm']
    for ii, ind in enumerate(np.ndindex(shape_plasma)):
        dprop[ind] = {
            'color': lc[ii % len(lc)]
        }

    # -----------------
    # E_eV, theta
    # -----------------

    # E_eV
    if kwdargs['E_eV'] is None:
        kwdargs['E_eV'] = _DCOORDS['E_eV']
    E_eV = ds._generic_check._check_flat1darray(
        kwdargs['E_eV'], 'E_eV',
        dtype=float,
        unique=True,
        sign='>=0',
    )

    # theta
    ntheta = int(ds._generic_check._check_var(
        kwdargs['ntheta'], 'ntheta',
        types=(int, float),
        default=_DCOORDS['ntheta'],
        sign='>0',
    ))
    if ntheta % 2 == 0:
        ntheta += 1
    theta = np.linspace(0, np.pi, ntheta)

    # -----------------
    # v_par, v_perp
    # -----------------

    pmax_norm = _convert.convert_momentum_velocity_energy(
        energy_kinetic_eV=E_eV.max(),
    )['momentum_normalized']['data']

    # npar, nperp
    nperp = int(ds._generic_check._check_var(
        kwdargs['nperp'], 'nperp',
        types=(int, float),
        default=_DCOORDS['nperp'],
        sign='>0',
    ))

    p_par_norm = np.linspace(-1, 1, 2*nperp+1) * pmax_norm
    p_perp_norm = np.linspace(0, 1, nperp) * pmax_norm

    dcoords = {
        'E_eV': E_eV,
        'theta': theta,
        'p_par_norm': p_par_norm,
        'p_perp_norm': p_perp_norm,
    }

    return (
        dplasma,
        dprop,
        dcoords,
    )


# #####################################################
# #####################################################
#               plot
# #####################################################


def _plot(
    ddist_E_theta=None,
    ddist_ppar_pperp=None,
    # 1d
    v_edge=None,
    ddist_E=None,
    ddist_p=None,
    # props
    dprop=None,
    # plotting
    dax=None,
    fontsize=None,
    dmargin=None,
):

    # ----------------
    # prepare
    # ----------------

    shape_plasma = ddist_E_theta['dist']['maxwell']['dist']['data'].shape[:-2]

    # ----------------
    # dax
    # ----------------

    if dax is None:
        dax = _get_dax(
            fontsize=fontsize,
            dmargin=dmargin,
            # units_E_pitch=dout_E['dist']['units'],
            # units_E=dout_E_1d['dist']['units'],
            # units_v=dout_v['dist']['units'],
        )

    dax = ds._generic_check._check_dax(dax)

    # ----------------
    # plot vs E, theta
    # ----------------

    kax = '(E, theta)'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # data
        maxwell = ddist_E_theta['dist']['maxwell']['dist']['data']
        RE = ddist_E_theta['dist']['RE']['dist']['data']

        # vmax, vmin
        vmax = np.max(maxwell + RE)
        vmaxRE = np.max(RE[RE > maxwell])
        vmin = np.min(RE[RE > 0.])
        levels = np.unique(np.r_[
            np.logspace(np.log10(vmin), np.log10(vmaxRE), 6),
            np.logspace(np.log10(vmaxRE), np.log10(vmax), 6),
        ])

        for ii, ind in enumerate(np.ndindex(shape_plasma)):
            sli = ind + (slice(None), slice(None))
            val = maxwell[sli] + RE[sli]
            # levels = _get_levels(val, 20)

            im = ax.contour(
                ddist_E_theta['coords']['x0']['data']*1e-3,
                ddist_E_theta['coords']['x1']['data']*180/np.pi,
                val.T,
                levels=levels,
                colors=dprop[ind]['color'],
            )

            plt.clabel(im, fmt='%1.2e', fontsize=fontsize)

        ax.set_ylim(0, 180)
        ax.set_xlim(left=0.)

    # ----------------
    # plot vs E
    # ----------------

    kax = 'E1d'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for ii, ind in enumerate(np.ndindex(shape_plasma)):
            sli = ind + (slice(None),)
            maxwell = ddist_E['maxwell']['data'][sli]
            re = ddist_E['RE']['data'][sli]

            # maxwell
            ax.semilogy(
                ddist_E_theta['coords']['x0']['data']*1e-3,
                maxwell,
                ls='-',
                lw=1,
                color=dprop[ind]['color'],
                label="Maxwell",
            )

            # RE
            ax.semilogy(
                ddist_E_theta['coords']['x0']['data']*1e-3,
                re,
                ls='--',
                lw=1,
                color=dprop[ind]['color'],
                label="RE",
            )

            # total
            ax.semilogy(
                ddist_E_theta['coords']['x0']['data']*1e-3,
                maxwell + re,
                ls='-',
                lw=2,
                color=dprop[ind]['color'],
            )

        ax.set_xlim(left=0.)

    # ----------------
    # plot vs velocities - 2D
    # ----------------

    kax = '(p_par, p_perp)'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # data
        maxwell = ddist_ppar_pperp['dist']['maxwell']['dist']['data']
        RE = ddist_ppar_pperp['dist']['RE']['dist']['data']

        # vmax, vmin
        vmax = np.max(maxwell + RE)
        vmaxRE = np.max(RE[RE > maxwell])
        vmin = np.min(RE[RE > 0.])
        levels = np.unique(np.r_[
            np.logspace(np.log10(vmin), np.log10(vmaxRE), 6),
            np.logspace(np.log10(vmaxRE), np.log10(vmax), 6),
        ])

        import pdb; pdb.set_trace()     # DB

        for ii, ind in enumerate(np.ndindex(shape_plasma)):
            sli = ind + (slice(None), slice(None))
            val = maxwell[sli] + RE[sli]
            # levels = _get_levels(val, 20)

            im = ax.contour(
                ddist_ppar_pperp['coords']['x0']['data'],
                ddist_ppar_pperp['coords']['x1']['data'],
                val.T,
                levels=levels,
                colors=dprop[ind]['color'],
            )

            plt.clabel(im, fmt='%1.2e', fontsize=fontsize)

        ax.set_ylim(bottom=0.)

    # ----------------
    # plot vs v 1D
    # ----------------

    kax = 'p1d'
    if dax.get(kax) is not None and False:
        ax = dax[kax]['handle']

        shape = dout_v['dist']['data'].shape[:-2]
        for ii, ind in enumerate(np.ndindex(shape)):
            sli = ind + (slice(None),)
            ax.stairs(
                dist_1d_v[sli],
                edges=v_edge*1e-3,
                fill=False,
                baseline=0,
                color=dprop[ind]['color'],
            )

    return dax


# #####################################################
# #####################################################
#               levels
# #####################################################


def _get_levels(val, nn=10):

    vmax = np.max(val)
    vmin = np.min(val[val > 0.])
    vmax_log10 = np.log10(vmax)
    vmin_log10 = np.log10(vmin)

    nn = int(np.ceil((vmax_log10 - vmin_log10) / nn))
    levels = np.arange(np.floor(vmin_log10), np.ceil(vmax_log10)+1, nn)
    levels = 10**(levels)
    levels = np.unique(np.r_[levels, 0.99*vmax])

    return levels


# #####################################################
# #####################################################
#               dax
# #####################################################


def _get_dax(
    fontsize=None,
    dmargin=None,
    # units
    units_E_pitch=None,
    units_E=None,
    units_v=None,
):
    # --------------
    # check inputs
    # --------------

    # fontsize
    fontsize = ds._generic_check._check_var(
        fontsize, 'fontsize',
        types=(int, float),
        default=12,
        sign='>0',
    )

    # --------------
    # prepare data
    # --------------

    # str_fE2d = (
        # _distribution._DFUNC['f2d_E_pitch']['latex']
        # + f'\n{units_E_pitch}'
    # )
    # str_fE1d = (
        # _distribution._DFUNC['f1d_E']['latex']
        # + f'\n{units_E}'
    # )
    # str_fv2d = _distribution._DFUNC['f2d_cart_vpar_vperp']['latex']
    # str_fv3d = (
        # _distribution._DFUNC['f3d_cart_vpar_vperp']['latex']
        # + f'\n{units_v}'
    # )

    # --------------
    # prepare axes
    # --------------

    tit = (
        "Maxwellian distribution\n"
        "[1] D. Moseev and M. Salewski, Physics of Plasmas, 26, p.020901, 2019"
    )

    if dmargin is None:
        dmargin = {
            'left': 0.08, 'right': 0.95,
            'bottom': 0.06, 'top': 0.83,
            'wspace': 0.2, 'hspace': 0.50,
        }

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(tit, size=fontsize+2, fontweight='bold')

    gs = gridspec.GridSpec(ncols=2, nrows=2, **dmargin)
    dax = {}

    # --------------
    # prepare axes
    # --------------

    # --------------
    # (E, pitch) - map

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(
        "E (keV)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        "theta (deg)",
        size=fontsize,
        fontweight='bold',
    )
    # ax.set_title(
        # str_fE2d,
        # size=fontsize,
        # fontweight='bold',
    # )
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # store
    dax['(E, theta)'] = {'handle': ax, 'type': 'Ep'}

    # --------------
    # (v_par, v_perp) - map

    ax = fig.add_subplot(gs[0, 1], aspect='equal', adjustable='datalim')
    ax.set_xlabel(
        r"$p_{//}$ (adim.)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$p_{\perp}$ (adim.)",
        size=fontsize,
        fontweight='bold',
    )
    # ax.set_title(
        # str_fv2d,
        # size=fontsize,
        # fontweight='bold',
    # )
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # store
    dax['(p_par, p_perp)'] = {'handle': ax, 'type': 'vv2d'}

    # --------------
    # E1d

    ax = fig.add_subplot(
        gs[1, 0],
        xscale='linear',
        yscale='log',
        sharex=dax['(E, theta)']['handle'],
    )
    ax.set_xlabel(
        "E (keV)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        "sum",
        size=fontsize,
        fontweight='bold',
    )
    # ax.set_title(
        # str_fE1d,
        # size=fontsize,
        # fontweight='bold',
    # )
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # store
    dax['E1d'] = {'handle': ax, 'type': 'Ep'}

    # --------------
    # v1d

    ax = fig.add_subplot(gs[1, 1], xscale='log', yscale='log')
    ax.set_xlabel(
        "p (adim",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        "sum",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        '',   # str_fE,
        size=fontsize,
        fontweight='bold',
    )
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # store
    dax['p1d'] = {'handle': ax, 'type': 'Ep'}

    return dax
