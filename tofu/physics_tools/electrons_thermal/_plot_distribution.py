

import numpy as np
import scipy.constants as scpct
import scipy.stats as scpstats
import scipy.integrate as scpinteg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


from . import _distribution


plt.rcParams['text.usetex'] = True


# ############################################
# ############################################
#        Default
# ############################################


_E_MAX_EV = 1e6
_E_EV = np.logspace(0, np.log10(_E_MAX_EV), 1000)
_PITCH = np.linspace(-1, 1, 51)

_V_MAX_MS = np.sqrt(2.*_E_MAX_EV * scpct.e / scpct.m_e)
_V_PAR_MS = np.linspace(-1, 1, 101) * _V_MAX_MS
_V_PERP_MS = np.linspace(1e-9, 1, 51) * _V_MAX_MS


# ############################################
# ############################################
#        Maxwellian - 2d
# ############################################


def plot_maxwellian(
    # plasma paremeters
    Te_eV=None,
    ne_m3=None,
    jp_Am2=None,
    # coordinates
    E_eV=None,
    pitch=None,
    v_par_ms=None,
    v_perp_ms=None,
    # plotting
    dax=None,
    fontsize=None,
    dmargin=None,
):

    # ----------------
    # check inputs
    # ----------------

    (
        Te_eV, ne_m3, jp_Am2,
        E_eV, pitch,
        v_par_ms, v_perp_ms,
        dprop,
    ) = _check(
        # plasma paremeters
        Te_eV=Te_eV,
        ne_m3=ne_m3,
        jp_Am2=jp_Am2,
        # coordinates
        E_eV=E_eV,
        pitch=pitch,
        v_par_ms=v_par_ms,
        v_perp_ms=v_perp_ms,
    )

    # ----------------
    # Compute
    # ----------------

    # f2D_vpar_vperp
    dout_v = _distribution.get_maxwellian(
        # plasma paremeters
        Te_eV=Te_eV,
        ne_m3=ne_m3,
        jp_Am2=jp_Am2,
        # coordinate: momentum
        v_perp_ms=v_perp_ms,
        v_par_ms=v_par_ms,
        # return as
        returnas=dict,
        # version
        version='f2d_cart_vpar_vperp',
    )

    # f2D_E_pitch
    dout_E = _distribution.get_maxwellian(
        # plasma paremeters
        Te_eV=Te_eV,
        ne_m3=ne_m3,
        jp_Am2=jp_Am2,
        # coordinate: energy
        E_eV=E_eV,
        pitch=pitch,
        # return as
        returnas=dict,
        # version
        version='f2d_E_pitch',
    )

    # f1D_E
    dout_E_1d = _distribution.get_maxwellian(
        # plasma paremeters
        Te_eV=Te_eV,
        ne_m3=ne_m3,
        jp_Am2=jp_Am2,
        # coordinate: energy
        E_eV=E_eV,
        # return as
        returnas=dict,
        version='f1d_E',
    )

    # ----------------
    # Derive 1d
    # ----------------

    # E
    dist_1d_E = scpinteg.trapezoid(
        dout_E['dist']['data'],
        x=pitch,
        axis=-1,
    )

    # v
    v_edge = 0.5*(v_perp_ms[1:] + v_perp_ms[:-1])
    dv = v_perp_ms[1] - v_perp_ms[0]
    v_edge = np.r_[v_edge[0] - dv, v_edge, v_edge[-1] + dv]
    shape = dout_E['dist']['data'].shape[:-2] + (v_edge.size-1,)
    dist_1d_v = np.full(shape, np.nan)
    for ii, ind in enumerate(np.ndindex(shape[:-1])):
        sli0 = ind + (slice(None),)
        sli1 = ind + (slice(None), slice(None))
        dist_1d_v[sli0] = scpstats.binned_statistic(
            np.sqrt(v_par_ms[:, None]**2 + v_perp_ms[None, :]**2).ravel(),
            dout_v['dist']['data'][sli1].ravel(),
            statistic='sum',
            bins=v_edge,
        ).statistic

    # ----------------
    # plot
    # ----------------

    dax = _plot(
        # plasma
        Te_eV=Te_eV,
        # coords
        E_eV=E_eV,
        pitch=pitch,
        v_par_ms=v_par_ms,
        v_perp_ms=v_perp_ms,
        # distribution
        dout_E=dout_E,
        dout_E_1d=dout_E_1d,
        dout_v=dout_v,
        # 1d
        v_edge=v_edge,
        dist_1d_E=dist_1d_E,
        dist_1d_v=dist_1d_v,
        # props
        dprop=dprop,
        # plotting
        dax=dax,
        fontsize=fontsize,
        dmargin=dmargin,
    )

    return dax


# #####################################################
# #####################################################
#               check
# #####################################################


def _check(
    # plasma paremeters
    Te_eV=None,
    ne_m3=None,
    jp_Am2=None,
    # coordinates
    E_eV=None,
    pitch=None,
    v_par_ms=None,
    v_perp_ms=None,
):

    # -----------------
    # plasma parameters
    # -----------------

    if Te_eV is None:
        Te_eV = 1e3
    Te_eV = np.atleast_1d(Te_eV).ravel()

    if ne_m3 is None:
        ne_m3 = 1e19
    ne_m3 = np.atleast_1d(ne_m3).ravel()

    if jp_Am2 is None:
        jp_Am2 = 0.
    jp_Am2 = np.atleast_1d(jp_Am2).ravel()

    Te_eV, ne_m3, jp_Am2 = np.broadcast_arrays(Te_eV, ne_m3, jp_Am2)

    dprop = {}
    lc = ['b', 'r', 'g', 'c', 'm']
    for ii, ind in enumerate(np.ndindex(Te_eV.shape)):
        dprop[ind] = {
            'color': lc[ii % len(lc)]
        }

    # -----------------
    # E_eV, pitch
    # -----------------

    # E_eV
    if E_eV is None:
        E_eV = _E_EV
    E_eV = ds._generic_check._check_flat1darray(
        E_eV, 'E_eV',
        dtype=float,
        sign='>=0',
    )

    # pitch
    if pitch is None:
        pitch = _PITCH
    pitch = ds._generic_check._check_flat1darray(
        pitch, 'pitch',
        dtype=float,
        sign=['>=-1', '<=1'],
    )

    # -----------------
    # v_par, v_perp
    # -----------------

    # v_par_ms
    if v_par_ms is None:
        v_par_ms = _V_PAR_MS
    v_par_ms = ds._generic_check._check_flat1darray(
        v_par_ms, 'v_par_ms',
        dtype=float,
    )

    # pitch
    if v_perp_ms is None:
        v_perp_ms = _V_PERP_MS
    v_perp_ms = ds._generic_check._check_flat1darray(
        v_perp_ms, 'v_perp_ms',
        dtype=float,
        sign='>=0',
    )

    return (
        Te_eV, ne_m3, jp_Am2,
        E_eV, pitch,
        v_par_ms, v_perp_ms,
        dprop,
    )


# #####################################################
# #####################################################
#               plot
# #####################################################


def _plot(
    # plasma
    Te_eV=None,
    # coords
    E_eV=None,
    pitch=None,
    v_par_ms=None,
    v_perp_ms=None,
    # distribution
    dout_E=None,
    dout_E_1d=None,
    dout_v=None,
    # 1d
    v_edge=None,
    dist_1d_E=None,
    dist_1d_v=None,
    # props
    dprop=None,
    # plotting
    dax=None,
    fontsize=None,
    dmargin=None,
):

    # ----------------
    # dax
    # ----------------

    if dax is None:
        dax = _get_dax(
            fontsize=fontsize,
            dmargin=dmargin,
        )

    dax = ds._generic_check._check_dax(dax)

    # ----------------
    # plot vs E, pitch
    # ----------------

    kax = '(E, pitch) - map'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        shape = dout_E['dist']['data'].shape[:-2]
        for ii, ind in enumerate(np.ndindex(shape)):
            sli = ind + (slice(None), slice(None))
            val = dout_E['dist']['data'][sli]
            levels = _get_levels(val)

            im = ax.contour(
                E_eV*1e-3,
                pitch,
                val.T,
                levels=levels,
                colors=dprop[ind]['color'],
            )

            plt.clabel(im, fmt='%1.2e', fontsize=fontsize)

        ax.set_ylim(-1, 1)

    # ----------------
    # plot vs velocities - 2D
    # ----------------

    kax = '(v_par, v_perp) - map'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        shape = dout_v['dist']['data'].shape[:-2]
        for ii, ind in enumerate(np.ndindex(shape)):
            sli = ind + (slice(None), slice(None))
            val = dout_v['dist']['data'][sli]
            levels = _get_levels(val)

            im = ax.contour(
                v_par_ms,
                v_perp_ms,
                val.T,
                levels=levels,
                colors=dprop[ind]['color'],
            )

            plt.clabel(im, fmt='%1.2e', fontsize=fontsize)

        ax.axvline(0, c='k', ls='--')
        ax.set_ylim(bottom=0)

    # ----------------
    # plot vs velocities - 3D
    # ----------------

    kax = '(v_par, v_perp)3D - map'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        shape = dout_v['dist']['data'].shape[:-2]
        for ii, ind in enumerate(np.ndindex(shape)):
            sli = ind + (slice(None), slice(None))
            im = ax.contour(
                v_par_ms,
                v_perp_ms,
                np.log10(dout_v['dist']['data'][sli]).T,
                levels=10,  # dout_v['dist']['levels'][ind],
                colors=dprop[ind]['color'],
            )

            plt.clabel(im)

        ax.axvline(0, c='k', ls='--')
        ax.set_ylim(bottom=0)

    # ----------------
    # plot vs E 1D
    # ----------------

    kax = 'E1d'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        shape = dout_E['dist']['data'].shape[:-2]
        for ii, ind in enumerate(np.ndindex(shape)):
            sli = ind + (slice(None),)
            ax.plot(
                E_eV*1e-3,
                dout_E_1d['dist']['data'][sli],
                color=dprop[ind]['color'],
                ls='-',
            )

            if Te_eV.size > 1:
                ax.axvline(Te_eV[ind]*1e-3, c=dprop[ind]['color'], ls='--')
        if Te_eV.size == 1:
            ax.axvline(Te_eV*1e-3, c='k', ls='--')

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    # ----------------
    # plot vs v 1D
    # ----------------

    kax = 'v1d'
    if dax.get(kax) is not None:
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


def _get_levels(val):

    vmax = np.max(val)
    vmin = np.min(val[val > 0.])
    vmax_log10 = np.log10(vmax)
    vmin_log10 = np.log10(vmin)

    nn = int(np.ceil((vmax_log10 - vmin_log10) / 10))
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
):
    # --------------
    # check inputs
    # --------------

    # fontsize
    fontsize = ds._generic_check._check_var(
        fontsize, 'fontsize',
        types=(int, float),
        default=14,
        sign='>0',
    )

    # --------------
    # prepare data
    # --------------

    str_fE2d = _distribution._DFUNC['f2d_E_pitch']['latex']
    str_fE1d = _distribution._DFUNC['f1d_E']['latex']
    str_fv2d = _distribution._DFUNC['f2d_cart_vpar_vperp']['latex']
    str_fv3d = _distribution._DFUNC['f3d_cart_vpar_vperp']['latex']

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
            'wspace': 0.2, 'hspace': 0.40,
        }

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(tit, size=fontsize+2, fontweight='bold')

    gs = gridspec.GridSpec(ncols=3, nrows=2, **dmargin)
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
        "pitch",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        str_fE2d,
        size=fontsize,
        fontweight='bold',
    )
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # store
    dax['(E, pitch) - map'] = {'handle': ax, 'type': 'Ep'}

    # --------------
    # (v_par, v_perp) - map

    ax = fig.add_subplot(gs[0, 1], aspect='equal', adjustable='datalim')
    ax.set_xlabel(
        r"$v_{//}$ (m/s)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$v_{\perp}$ (m/s)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        str_fv2d,
        size=fontsize,
        fontweight='bold',
    )
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # store
    dax['(v_par, v_perp) - map'] = {'handle': ax, 'type': 'vv2d'}

    # --------------
    # (v_par, v_perp)3D - map

    ax = fig.add_subplot(gs[0, 2], aspect='equal', adjustable='datalim')
    ax.set_xlabel(
        r"$v_{//}$ (m/s)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$v_{\perp}$ (m/s)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        str_fv3d,
        size=fontsize,
        fontweight='bold',
    )
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # store
    dax['(v_par, v_perp)3D - map'] = {'handle': ax, 'type': 'vv3d'}

    # --------------
    # E1d

    ax = fig.add_subplot(gs[1, 0], xscale='linear', yscale='linear')
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
    ax.set_title(
        str_fE1d,
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
        "v (m/s)",
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
    dax['v1d'] = {'handle': ax, 'type': 'Ep'}

    return dax
