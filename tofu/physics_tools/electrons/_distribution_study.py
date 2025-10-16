

import numpy as np
import scipy.integrate as scpinteg
import astropy.units as asunits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import datastock as ds


from . import _distribution
from . import _distribution_check


# ###############################################3
# ###############################################3
#               DEFAULT
# ###############################################3


_DPLASMA = {
    'Te_eV': {
        'def': np.linspace(0.5, 15, 59)[:, None, None, None] * 1e3,
        'units': 'eV',
    },
    'ne_m3': {
        'def': np.r_[1e19, 1e20][None, None, :, None],
        'units': '1/m^3',
    },
    'jp_Am2': {
        'def': np.r_[1e6, 10e6][None, None, None, :],
        'units': 'A/m^2',
    },
    'jp_fraction_re': {
        'def': np.linspace(0.01, 0.99, 51)[None, :, None, None],
        'units': None,
    },
}


_EMAX_EV = 20e6
_DCOORDS = {
    'E_eV': np.logspace(1, np.log10(_EMAX_EV), 201),
    'ntheta': 41,
}


_LEVELS_E_EV = np.r_[40, 150]*1e3


# ###############################################3
# ###############################################3
#               main
# ###############################################3


def study_RE_vs_Maxwellian_distribution(
    # plasma
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
    # coords
    E_eV=None,
    ntheta=None,
    # levels
    levels_E_eV=None,
    colors=None,
    # plotting
    dax=None,
    fontsize=None,
    dmargin=None,
):
    """

    Return dax, ddist

    """

    # --------------
    # check inputs
    # --------------

    dplasma, dcoords, levels_E_eV = _check(**locals())

    # --------------
    # compute
    # --------------

    # f2D_E_theta
    ddist = _distribution.main(
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

    # ------------------
    # integrate
    # ------------------

    # E
    units = ddist['dist']['RE']['dist']['units'] * asunits.Unit('rad')
    ddist_E_num = {
        kdist: {
            'data': scpinteg.trapezoid(
                ddist['dist'][kdist]['dist']['data'],
                x=ddist['coords']['x1']['data'],
                axis=-1,
            ),
            'units': units,
        }
        for kdist in ddist['dist'].keys()
    }

    # ------------------
    # extract threshold
    # ------------------

    ddata = _get_threshold(
        ddist=ddist,
        ddist_E_num=ddist_E_num,
    )

    # --------------
    # plot
    # --------------

    dax = _plot(
        ddist=ddist,
        ddist_E_num=ddist_E_num,
        ddata=ddata,
        levels_E_eV=levels_E_eV,
        colors=colors,
        dax=dax,
        fontsize=fontsize,
        dmargin=dmargin,
    )

    return dax, ddist


# ###############################################3
# ###############################################3
#               check
# ###############################################3


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

    # -----------------
    # levels_E_eV
    # -----------------

    if kwdargs['levels_E_eV'] is None:
        kwdargs['levels_E_eV'] = _LEVELS_E_EV

    levels_E_eV = kwdargs['levels_E_eV']
    if np.isscalar(levels_E_eV):
        if isinstance(levels_E_eV, int):
            levels_E_eV = ds._generic_check._check_var(
                levels_E_eV, 'levels_E_eV',
                types=int,
                sign='>0',
            )
        else:
            levels_E_eV = np.r_[levels_E_eV]

    if not isinstance(levels_E_eV, int):
        levels_E_eV = ds._generic_check._check_flat1darray(
            levels_E_eV, 'levels_E_eV',
            unique=True,
            sign='>0',
        )

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

    dcoords = {
        'E_eV': E_eV,
        'theta': theta,
    }

    return (
        dplasma,
        dcoords,
        levels_E_eV,
    )


# ###############################################3
# ###############################################3
#           Threshold
# ###############################################3


def _get_threshold(
    ddist=None,
    ddist_E_num=None,
):

    # -----------
    # prepare
    # ----------

    E_eV = ddist['coords']['x0']['data']
    RE = ddist_E_num['RE']['data']
    maxwell = ddist_E_num['maxwell']['data']

    shape = RE.shape
    iok = np.isfinite(RE) & np.isfinite(maxwell)
    iok2 = np.any(iok, axis=-1)
    iok2[iok2] = (
        np.any(RE[iok2, :] > maxwell[iok2, :], axis=-1)
        & np.any(RE[iok2, :] < maxwell[iok2, :], axis=-1)
    )

    # -----------
    # compute
    # ----------

    sli = (None,)*len(shape[:-1]) + (slice(None),)
    ind = np.arange(E_eV.size)[sli]
    ind = np.copy(np.broadcast_to(ind, shape))
    iout = (RE < maxwell)
    ind[iout] = ddist['coords']['x0']['data'].size + 1
    imin = np.argmin(ind, axis=-1)

    E_min = E_eV[imin]
    E_min[~iok2] = np.nan

    # -----------
    # format
    # -----------

    ddata = {
        'E_min': {
            'data': E_min,
            'units': ddist['coords']['x0']['units'],
        },
    }

    return ddata


# ###############################################3
# ###############################################3
#               plot
# ###############################################3


def _plot(
    ddist=None,
    ddist_E_num=None,
    ddata=None,
    levels_E_eV=None,
    colors=None,
    # plotting
    dax=None,
    fontsize=None,
    dmargin=None,
):

    # ----------------
    # prepare
    # ----------------

    E_min = ddata['E_min']['data']
    shape_plasma = E_min.shape[2:]
    Te_eV = ddist['plasma']['Te_eV']['data'][:, 0, 0, 0]
    jp_fraction_re = ddist['plasma']['jp_fraction_re']['data'][0, :, 0, 0]
    ne_m3 = ddist['plasma']['ne_m3']['data'][0, 0, :, 0]
    jp_Am2 = ddist['plasma']['jp_Am2']['data'][0, 0, 0, :]

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
    # plot vs E, theta
    # ----------------

    kax = 'main'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        lh = []
        lc = ['r', 'g', 'b', 'm', 'y', 'c']
        for ii, ind in enumerate(np.ndindex(shape_plasma)):
            sli = (slice(None),)*2 + ind

            # label
            nei = ne_m3[ind[0]]
            jpi = jp_Am2[ind[1]]
            lab = f"ne = {nei:1.0e} /m3  jp = {jpi*1e-6:1.0f} MA/m2"

            # plot
            if colors is None:
                color = lc[ii % len(lc)]
            else:
                color = colors
            im = ax.contourf(
                Te_eV*1e-3,
                jp_fraction_re,
                1e-3*E_min[sli].T,
                levels=levels_E_eV*1e-3,
                colors=color,
            )

            lh.append(mlines.Line2D([], [], c=color, ls='-', label=lab))
            plt.clabel(im, inline=True, fontsize=12)

        # legend
        ax.legend(handles=lh, loc='upper right', fontsize=12)

        # lim
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)

    return dax


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
        default=12,
        sign='>0',
    )

    # --------------
    # prepare data
    # --------------

    # --------------
    # prepare axes
    # --------------

    tit = ""

    if dmargin is None:
        dmargin = {
            'left': 0.08, 'right': 0.95,
            'bottom': 0.06, 'top': 0.83,
            'wspace': 0.2, 'hspace': 0.50,
        }

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(tit, size=fontsize+2, fontweight='bold')

    gs = gridspec.GridSpec(ncols=1, nrows=1, **dmargin)
    dax = {}

    # --------------
    # prepare axes
    # --------------

    # --------------
    # (E, pitch) - map

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(
        "Te (keV)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        "jp_fraction_re (adim.)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        "Electron energy above which RE dominate",
        size=fontsize,
        fontweight='bold',
    )
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # store
    dax['main'] = {'handle': ax, 'type': 'Ep'}

    return dax
