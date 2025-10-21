

import numpy as np
# import scipy.stats as scpstats
import scipy.integrate as scpinteg
import scipy.stats as scpstats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import astropy.units as asunits
import datastock as ds


from .. import _convert
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
        'def': np.r_[1, 1, 5, 5]*1e3,
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
        'def': np.r_[0.1, 0.9, 0.1, 0.9],
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
_EMAX_EV = 20e6
_DCOORDS = {
    'E_eV': np.logspace(1, np.log10(_EMAX_EV), 201),
    'ntheta': 41,
    'nperp': 201,
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
    ddist_E_num = {
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

    # pnorm
    pnorm = np.sqrt(
        ddist_ppar_pperp['coords']['x0']['data'][:, None]**2
        + ddist_ppar_pperp['coords']['x1']['data'][None, :]**2
    )
    pnmin = np.nanmin(pnorm[pnorm > 0.])
    pnmax = np.nanmax(pnorm)
    pbins = np.logspace(
        np.log10(pnmin),
        np.log10(pnmax),
        int(np.min(pnorm.shape) - 1),
    )
    pbins = np.r_[0., pbins]
    shape_plasma = ddist_E_theta['dist']['maxwell']['dist']['data'].shape[:-2]
    ddist_pnorm_num = {
        kdist: {
            'data': scpstats.binned_statistic(
                pnorm.ravel(),
                vdist['dist']['data'].reshape(shape_plasma + (-1,)),
                statistic='sum',
                bins=pbins,
            ).statistic,
            'units': None,
        }
        for kdist, vdist in ddist_ppar_pperp['dist'].items()
    }

    # ----------------
    # plot
    # ----------------

    dax = _plot(
        ddist_E_theta=ddist_E_theta,
        ddist_ppar_pperp=ddist_ppar_pperp,
        # 1d
        ddist_E_num=ddist_E_num,
        ddist_pnorm_num=ddist_pnorm_num,
        pbins=pbins,
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

    pmin_norm = _convert.convert_momentum_velocity_energy(
        energy_kinetic_eV=E_eV.min(),
    )['momentum_normalized']['data'][0]

    pmax_norm = _convert.convert_momentum_velocity_energy(
        energy_kinetic_eV=E_eV.max(),
    )['momentum_normalized']['data'][0]

    # npar, nperp
    nperp = int(ds._generic_check._check_var(
        kwdargs['nperp'], 'nperp',
        types=(int, float),
        default=_DCOORDS['nperp'],
        sign='>0',
    ))

    p_perp_norm = np.logspace(np.log10(pmin_norm), np.log10(pmax_norm), nperp)
    p_par_norm = np.r_[-p_perp_norm[::-1], 0, p_perp_norm]
    p_perp_norm = np.r_[0., p_perp_norm]

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
    ddist_E_num=None,
    ddist_pnorm_num=None,
    pbins=None,
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
            vmin, max(vmin, vmaxRE/10.),
            np.logspace(np.log10(max(vmin, vmaxRE/10.)), np.log10(vmaxRE), 6),
            np.logspace(np.log10(vmaxRE), np.log10(vmax), 4),
        ])

        for ii, ind in enumerate(np.ndindex(shape_plasma)):
            sli = ind + (slice(None), slice(None))
            val = maxwell[sli] + RE[sli]

            ax.contour(
                ddist_E_theta['coords']['x0']['data']*1e-3,
                ddist_E_theta['coords']['x1']['data']*180/np.pi,
                val.T,
                levels=levels,
                colors=dprop[ind]['color'],
            )

        ax.set_ylim(0, 180)
        ax.set_xlim(left=0.)

    # ----------------
    # plot vs E
    # ----------------

    kax = 'E1d'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        lh = []
        for ii, ind in enumerate(np.ndindex(shape_plasma)):
            sli = ind + (slice(None),)
            maxwell_num = ddist_E_num['maxwell']['data'][sli]
            re_num = ddist_E_num['RE']['data'][sli]
            color = dprop[ind]['color']

            # maxwell
            ax.semilogy(
                ddist_E_theta['coords']['x0']['data']*1e-3,
                maxwell_num,
                ls='-',
                lw=1,
                color=color,
                label="Maxwell_num",
            )

            # RE
            ax.semilogy(
                ddist_E_theta['coords']['x0']['data']*1e-3,
                re_num,
                ls='--',
                lw=1,
                color=color,
                label="RE_num",
            )

            # total
            ax.semilogy(
                ddist_E_theta['coords']['x0']['data']*1e-3,
                maxwell_num + re_num,
                ls='-',
                lw=2,
                color=color,
            )

            # label
            nei = ddist_E_theta['plasma']['ne_m3']['data'][ind]
            jpi = ddist_E_theta['plasma']['jp_Am2']['data'][ind]
            Tei = ddist_E_theta['plasma']['Te_eV']['data'][ind]
            jp_fraci = ddist_E_theta['plasma']['jp_fraction_re']['data'][ind]
            lab = (
                f"ne = {nei:1.0e} /m3  jp = {jpi*1e-6:1.0f} MA/m2"
                f"Te = {Tei*1e-3:1.0e} keV  jp_frac = {jp_fraci:1.1f}"
            )

            lh.append(mlines.Line2D([], [], c=color, ls='-', label=lab))

        # legend & lims
        ax.legend(handles=lh, loc='upper right', fontsize=12)
        ax.set_xlim(left=0.)
        ax.set_ylim(
            f"integral ({ddist_E_num['maxwell']['units']})",
            fontisize=fontsize,
            fontweight='bold',
        )

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
            vmin, max(vmin, vmaxRE/10.),
            np.logspace(np.log10(max(vmin, vmaxRE/10.)), np.log10(vmaxRE), 6),
            np.logspace(np.log10(vmaxRE), np.log10(vmax), 4),
        ])

        for ii, ind in enumerate(np.ndindex(shape_plasma)):
            sli = ind + (slice(None), slice(None))
            val = maxwell[sli] + RE[sli]

            # plot
            color = dprop[ind]['color']
            ax.contour(
                ddist_ppar_pperp['coords']['x0']['data'],
                ddist_ppar_pperp['coords']['x1']['data'],
                val.T,
                levels=levels,
                colors=color,
            )

        # legend & lims
        ax.set_ylim(bottom=0.)

    # ----------------
    # plot vs v 1D
    # ----------------

    kax = 'p1d'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        lh = []
        for ii, ind in enumerate(np.ndindex(shape_plasma)):
            sli = ind + (slice(None),)
            maxwell_num = ddist_pnorm_num['maxwell']['data'][sli]
            re_num = ddist_pnorm_num['RE']['data'][sli]
            color = dprop[ind]['color']

            # maxwell
            ax.stairs(
                maxwell_num,
                edges=pbins,
                orientation='vertical',
                baseline=0.,
                fill=False,
                ls='-',
                lw=1,
                color=color,
                label="Maxwell_num",
            )

            # RE
            ax.stairs(
                re_num,
                edges=pbins,
                orientation='vertical',
                baseline=0.,
                fill=False,
                ls='--',
                lw=1,
                color=color,
                label="RE_num",
            )

            # total
            ax.stairs(
                maxwell_num + re_num,
                edges=pbins,
                orientation='vertical',
                baseline=0.,
                fill=False,
                ls='-',
                lw=2,
                color=color,
            )

            # label
            nei = ddist_E_theta['plasma']['ne_m3']['data'][ind]
            jpi = ddist_E_theta['plasma']['jp_Am2']['data'][ind]
            Tei = ddist_E_theta['plasma']['Te_eV']['data'][ind]
            jp_fraci = ddist_E_theta['plasma']['jp_fraction_re']['data'][ind]
            lab = (
                f"ne = {nei:1.0e} /m3  jp = {jpi*1e-6:1.0f} MA/m2"
                f"Te = {Tei*1e-3:1.0e} keV  jp_frac = {jp_fraci:1.1f}"
            )

            lh.append(mlines.Line2D([], [], c=color, ls='-', label=lab))

        # legend & lims
        ax.legend(handles=lh, loc='upper right', fontsize=12)
        ax.set_xlim(left=0.)
        ax.set_ylim(
            f"integral ({ddist_pnorm_num['maxwell']['units']})",
            fontisize=fontsize,
            fontweight='bold',
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
