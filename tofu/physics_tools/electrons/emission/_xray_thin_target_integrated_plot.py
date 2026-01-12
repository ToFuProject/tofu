

import copy
from typing import Optional   # Any, Dict


import numpy as np
import scipy.integrate as scpinteg
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import datastock as ds


from . import _xray_thin_target_integrated as _mod
from ._xray_thin_target_integrated_cases import _DCASES_PRE


TupleDict = tuple[dict]


# ####################################################
# ####################################################
#           DEFAULT
# ####################################################


# Energy vectors
_E_E0_EV = np.logspace(3, 6, 51)
_E_PH_EV = np.linspace(1, 100, 25) * 1e3
_THETA_PH = np.linspace(0, np.pi, 41)
_VERSION = 'BHE'


# DSCALES
_DSCALES = {
    'E_ph': 'log',
    'E_e0': 'log',
    'theta': 'linear',
}


# ANISOTROPY CASES FORMATTING
_DCASES_FORMAT = {
    'E_e0_eV': {'type': (int, float), 'val': 1e3},
    'E_ph_eV': {'type': (int, float), 'val': 10e3},
    'color': {'type': (str, tuple), 'val': 'k'},
    'marker': {'type': str, 'val': '*'},
    'ms': {'type': (int, float), 'val': 18},
    'ls': {'type': str, 'val': '-'},
}


# ANISOTROPY CASES DEFAULT
_DCASES_CASE = 'standard'


# ####################################################
# ####################################################
#        plot anisotropy
# ####################################################


def plot_xray_thin_d2cross_ei_anisotropy(
    # optional input d2cross file
    d2cross: Optional[str | dict] = None,
    # target ion charge
    Z: Optional[int] = None,
    # Energy
    E_e0_eV=None,
    E_ph_eV=None,
    theta_ph=None,
    # hypergeometric parameter
    ninf: Optional[int] = None,
    source: Optional[str] = None,
    # output customization
    per_energy_unit: Optional[str] = None,
    # version
    version: Optional[str] = None,
    # selected cases
    dcases: Optional[dict[int, dict]] = None,
    # verb
    verb: Optional[bool] = None,
    # plot
    dax: Optional[dict] = None,
    fontsize: Optional[int] = None,
    E_e0_scale: Optional[str] = None,
    E_ph_scale: Optional[str] = None,
    theta_scale: Optional[str] = None,
    dplot_forbidden=None,
    dplot_peaking=None,
    dplot_thetamax=None,
    dplot_mean=None,
) -> TupleDict:
    """ Compute and plot a (E_e0, E_ph) countour map of the d2cross section

    Where d2cross is the fully differentiated cross-section (d3cross),
    integrated over one of the two the emission angle (dphi)

    Actually 3 overlayed contour plots with:
        - integral of of the cross-section (over photon emission angle)
        - angle of max cross-section
        - peaking of the cross-section (std vs angle)

    Can overlay a few selected cases and plot them vs angle of emission
    In normalized-linear and log scales

    """

    # ---------------
    # check inputs
    # ---------------

    (
        E_e0_eV, E_ph_eV, theta_ph,
        version,
        dscales,
        verb,
        fontsize,
        dplot_forbidden, dplot_peaking, dplot_thetamax, dplot_mean,
    ) = _check_anisotropy(**locals())

    # ---------------
    # prepare data
    # ---------------

    d2cross = _mod.get_xray_thin_d2cross_ei_integrated_thetae_dphi(
        d2cross=d2cross,
        # inputs
        Z=Z,
        E_e0_eV=E_e0_eV[None, :, None],
        E_ph_eV=E_ph_eV[None, None, :],
        theta_ph=theta_ph[:, None, None],
        # output customization
        per_energy_unit=per_energy_unit,
        # version
        version=version,
        # hypergeometric
        ninf=ninf,
        source=source,
        # verb
        verb=verb,
    )

    # -------------------
    # update from d2cross
    # -------------------

    # if d2cross was provided
    theta_ph = d2cross['theta_ph']['data'].ravel()
    E_ph_eV = d2cross['E_ph']['data'].ravel()
    E_e0_eV = d2cross['E_e0']['data'].ravel()

    # dcases
    dcases = _check_dcases(
        dcases=dcases,
        E_e0_eV=E_e0_eV,
        E_ph_eV=E_ph_eV,
    )

    # --------------
    # prepare axes
    # --------------

    if dax is None:
        dax = _dax(
            Z=Z,
            version=version,
            fontsize=fontsize,
            dscales=dscales,
        )

    dax = ds._generic_check._check_dax(dax)

    # ---------------
    # plot - map
    # ---------------

    kax = 'map'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for iv, (kk, vv) in enumerate(d2cross['cross'].items()):

            # compute integral and peaking
            mean, peaking = _get_peaking(
                vv['data'],
                theta_ph*180/np.pi,
                axis=0,
            )
            mean_log10 = np.full(mean.shape, np.nan)
            iok = np.isfinite(mean)
            iok[iok] = mean[iok] > 0.
            mean_log10[iok] = np.log10(mean[iok])
            mean_units = vv['units']

            # integral
            if dplot_mean is not False:
                im0 = ax.contour(
                    E_e0_eV * 1e-3,
                    E_ph_eV * 1e-3,
                    mean_log10.T,
                    levels=dplot_mean['levels'],
                    colors=dplot_mean['colors'],
                )

                # clabels
                ax.clabel(
                    im0,
                    inline=1,
                    fontsize=12,
                    # fmt=lambda val: f"{10**val:3.1f}",
                )

            # peaking
            if dplot_peaking is not False:
                im0 = ax.contour(
                    E_e0_eV * 1e-3,
                    E_ph_eV * 1e-3,
                    peaking.T,
                    levels=dplot_peaking['levels'],
                    colors=dplot_peaking['colors'],
                )

                # clabels
                ax.clabel(
                    im0,
                    inline=1,
                    fontsize=12,
                    # fmt=lambda val: f"{10**val:3.1f}",
                )

            # where peaked
            if dplot_thetamax is not False:
                imax = np.argmax(vv['data'], axis=0)
                yy = theta_ph[imax].T*180/np.pi
                im1 = ax.contour(
                    E_e0_eV * 1e-3,
                    E_ph_eV * 1e-3,
                    yy,
                    levels=dplot_thetamax['levels'],
                    colors=dplot_thetamax['colors'],
                )

                # clabels
                ax.clabel(
                    im1,
                    inline=1,
                    fontsize=12,
                    fmt=lambda val: f"{val:3.0f} deg",
                )

        # forbidden
        ymax = np.max(E_ph_eV)

        if dplot_forbidden is not False:
            xE = E_e0_eV[E_e0_eV <= E_ph_eV[-1]]
            xx = np.r_[xE, 0., 0]
            yy = np.r_[xE, ymax, 0]
            patch = mpatches.Polygon(
                1e-3*np.array([xx, yy]).T,
                hatch=dplot_forbidden['hatch'],
                facecolor=dplot_forbidden['facecolor'],
                edgecolor=dplot_forbidden['edgecolor'],
            )
            ax.add_patch(patch)

        # legend
        lh = []
        if dplot_mean is not False:
            lh.append(mlines.Line2D(
                [], [],
                c=dplot_mean['colors'],
                label=f'log10(<mean>) (log10({mean_units}))',
            ))
        if dplot_peaking is not False:
            lh.append(mlines.Line2D(
                [], [],
                c=dplot_peaking['colors'],
                label='peaking (1/std)',
            ))
        if dplot_thetamax is not False:
            lh.append(mlines.Line2D(
                [], [],
                c=dplot_thetamax['colors'],
                label='theta_max (deg)',
            ))
        if len(lh) > 0:
            ax.legend(handles=lh, loc='upper left')

        # add cases
        for ic, (kcase, vcase) in enumerate(dcases.items()):

            ee0 = E_e0_eV[vcase['ie']]
            eph = E_ph_eV[vcase['iph']]
            ax.plot(
                [ee0*1e-3],
                [eph*1e-3],
                marker=vcase['marker'],
                c=vcase['color'],
                ms=vcase['ms'],
            )

        # limits
        if dscales['E_ph'] == 'linear':
            ax.set_ylim(0, ymax*1e-3)

    # ---------------
    # plot - cases
    # ---------------

    for ic, (kcase, vcase) in enumerate(dcases.items()):

        lab = vcase['lab']
        for kv, vv in d2cross['cross'].items():
            labi = lab + f" - {kv}"
            yy = vv['data'][:, vcase['ie'], vcase['iph']]
            if np.any(yy > 0):

                # theta_norm
                kax = 'theta_norm'
                if dax.get(kax) is not None:
                    ax = dax[kax]['handle']

                    ax.plot(
                        theta_ph * 180/np.pi,
                        yy / np.max(yy),
                        c=vcase['color'],
                        ls=vcase['ls'],
                        label=labi,
                    )

                # theta_abs
                kax = 'theta_abs'
                if dax.get(kax) is not None:
                    ax = dax[kax]['handle']

                    l0, = ax.semilogy(
                        theta_ph * 180/np.pi,
                        yy*1e28,
                        c=vcase['color'],
                        ls=vcase['ls'],
                        label=labi,
                    )

    # normalized
    kax = 'theta_norm'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']
        ax.legend(prop={'size': 12})
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 180)

    # normalized
    kax = 'theta_abs'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']
        ax.legend(prop={'size': 12})
        units = str(vv['units'])
        units.replace('m2', 'barn')
        ax.set_ylabel(
            r"$\frac{d^2\sigma_{ei}}{dkd\Omega_{ph}}$" + f"  ({units})",
            size=fontsize,
            fontweight='bold',
        )
        ax.grid(True)

    return dax, d2cross


# #############################################
# #############################################
#        check
# #############################################


def _check_anisotropy(
    E_e0_eV=None,
    E_ph_eV=None,
    theta_ph=None,
    version=None,
    # verb
    verb=None,
    # scales
    E_e0_scale=None,
    E_ph_scale=None,
    theta_scale=None,
    # plotting
    fontsize=None,
    dplot_forbidden=None,
    dplot_peaking=None,
    dplot_thetamax=None,
    dplot_mean=None,
    # unused
    **kwdargs,
):

    # E_e0_eV
    if E_e0_eV is None:
        E_e0_eV = _E_E0_EV

    E_e0_eV = ds._generic_check._check_flat1darray(
        E_e0_eV, 'E_e0_eV',
        dtype=float,
        sign='>0',
        unique=True,
    )

    # E_ph_eV
    if E_ph_eV is None:
        E_ph_eV = _E_PH_EV

    E_ph_eV = ds._generic_check._check_flat1darray(
        E_ph_eV, 'E_ph_eV',
        dtype=float,
        sign='>0',
        unique=True,
    )

    # theta_ph
    if theta_ph is None:
        theta_ph = _THETA_PH

    # version
    if version is None:
        version = _VERSION

    # -----------
    # verb
    # -----------

    verb = ds._generic_check._check_var(
        verb, 'verb',
        types=(bool, int),
        default=False,
    )

    # ------------
    # plotting
    # ------------

    # ----------
    # fontsize

    if fontsize is None:
        fontsize = 14

    # --------------
    # plot dicts

    # dplot_forbidden
    ddef = {'edgecolor': 'k', 'facecolor': 'None', 'hatch': '\\'}
    dplot_forbidden = _check_anisotropy_dplot(
        dplot_forbidden,
        'dplot_forbidden',
        ddef,
    )

    # dplot_peaking
    ddef = {'colors': 'k', 'levels': 20}
    dplot_peaking = _check_anisotropy_dplot(
        dplot_peaking,
        'dplot_peaking',
        ddef,
    )

    # dplot_thetamax
    ddef = {'colors': 'b', 'levels': np.r_[0.1, 30, 50, 90]}
    dplot_thetamax = _check_anisotropy_dplot(
        dplot_thetamax,
        'dplot_thetamax',
        ddef,
    )

    # dplot_mean
    ddef = {'colors': 'g', 'levels': 20}
    dplot_mean = _check_anisotropy_dplot(
        dplot_mean,
        'dplot_mean',
        ddef,
    )

    # ------------
    # scales
    # ------------

    dscales = {
        'E_ph': E_ph_scale,
        'E_e0': E_e0_scale,
        'theta': theta_scale,
    }

    for kk, vv in dscales.items():
        dscales[kk] = ds._generic_check._check_var(
            vv, f'{kk}_scale',
            types=str,
            allowed=['log', 'linear'],
            default=_DSCALES[kk],
        )

    return (
        E_e0_eV, E_ph_eV, theta_ph,
        version,
        dscales,
        verb,
        fontsize,
        dplot_forbidden, dplot_peaking, dplot_thetamax, dplot_mean,
    )


def _check_dcases(
    dcases=None,
    E_e0_eV=None,
    E_ph_eV=None,
):

    # --------------
    # dcases default
    # --------------

    ddef = copy.deepcopy(_DCASES_FORMAT)
    if dcases in [None, True]:
        dcases = _DCASES_CASE

    # --------------
    # dcases from predefined
    # --------------

    if isinstance(dcases, str):
        lok = sorted(_DCASES_PRE.keys())
        if dcases not in lok:
            lstr = [f"\t- {kk}" for kk in lok]
            msg = (
                "Arg 'dcases' must be either:\n"
                "\t- dict of cases\n"
                "\t- a key to a predefined dict of cases\n"
                "Available predefined dcases:\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)
        dcases = copy.deepcopy(_DCASES_PRE[dcases])

    # --------------
    # generic check
    # --------------

    if dcases is not False:
        for k0, v0 in dcases.items():
            dcases[k0] = _check_anisotropy_dplot(
                v0,
                f'dcases[{k0}]',
                ddef,
            )

            # update with indices
            ie = np.argmin(np.abs(E_e0_eV - dcases[k0]['E_e0_eV']))
            iph = np.argmin(np.abs(E_ph_eV - dcases[k0]['E_ph_eV']))
            dcases[k0].update({'ie': ie, 'iph': iph})

            # update with label
            ee0 = E_e0_eV[ie]
            eph = E_ph_eV[iph]
            dcases[k0]['lab'] = (
                r"$E_{e0} / E_{ph}$ = "
                + f"{ee0*1e-3:3.0f} / {eph*1e-3:3.0f} keV = "
                + f"{round(ee0 / eph, ndigits=1)}"
            )
    else:
        dcases = {}

    return dcases


def _check_anisotropy_dplot(din, dname, ddef):

    # -------------
    # default
    # -------------

    if din in [None, True]:
        din = {}

    # -------------
    # format
    # -------------

    if din is not False:
        c0 = (
            isinstance(din, dict)
            and all([
                kk in ddef.keys()
                and (
                    isinstance(ddef[kk], dict)
                    and ddef[kk].get('type') is not None
                    and isinstance(din[kk], ddef[kk]['type'])
                )
                for kk in din.keys()
            ])
        )
        if not c0:
            lstr = [f"\t- '{k0}': {v0['type']}" for k0, v0 in ddef.items()]
            msg = (
                f"Arg '{dname}' must be either False or a dict with:\n"
                + "\n".join(lstr)
                + f"\nProvided:\n{din}\n"
            )
            raise Exception(msg)

    # -------------
    # fill
    # -------------

    if din is not False:
        for k0, v0 in ddef.items():
            if isinstance(v0, dict):
                vv = v0['val']
            else:
                vv = v0
            din[k0] = din.get(k0, vv)

    return din


# #############################################
# #############################################
#        Peaking
# #############################################


def _get_peaking(data, x, axis=None):

    # ----------
    # normalize as dist
    # ----------

    integ = scpinteg.trapezoid(data, x=x, axis=axis)
    shape_integ = list(data.shape)
    shape_integ[axis] = 1

    data_n = np.full(data.shape, np.nan)
    iok = np.isfinite(integ)
    iok[iok] = integ[iok] > 0
    iokn = iok.nonzero()
    sli0 = list(iokn)
    sli1 = list(iokn)
    sli0.insert(axis, None)
    sli1.insert(axis, slice(None))
    data_n[tuple(sli1)] = data[tuple(sli1)] / integ[tuple(sli0)]

    # ----------
    # get average
    # ----------

    shape_x = tuple([-1 if ii == axis else 1 for ii in range(data.ndim)])
    xf = x.reshape(shape_x)
    x_avf = scpinteg.trapezoid(
        data_n * xf,
        x=x,
        axis=axis,
    ).reshape(shape_integ)
    std = np.sqrt(scpinteg.simpson(data_n * (xf - x_avf)**2, x=x, axis=axis))

    return integ/180, 1/std


# #############################################
# #############################################
#        Axes for anisotropy
# #############################################


def _dax(
    Z=None,
    version=None,
    fontsize=None,
    dax=None,
    dscales=None,
):

    tit = (
        "Thin-target Bremsstrahlung cross-section anisotropy"
    )

    dmargin = {
        'left': 0.06, 'right': 0.95,
        'bottom': 0.06, 'top': 0.90,
        'wspace': 0.20, 'hspace': 0.20,
    }

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(tit, size=fontsize+2, fontweight='bold')

    gs = gridspec.GridSpec(ncols=2, nrows=2, **dmargin)
    dax = {}

    # --------------
    # prepare axes
    # --------------

    # --------------
    # ax - isolines

    ax = fig.add_subplot(
        gs[:, 0],
        xscale=dscales['E_e0'],
        yscale=dscales['E_ph'],
    )
    ax.set_xlabel(
        r"$E_{e,0}$ (keV)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$E_{ph}$ (keV)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        r"$d^2\sigma(E_{e0}, E_{ph}, \theta_{ph}, Z)$"
        + f"\n Z = {Z}, version = {version}",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['map'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # ax - theta_norm

    ax = fig.add_subplot(
        gs[0, 1],
        xscale=dscales['theta'],
    )
    ax.set_xlabel(
        r"$\theta_{ph}$ (deg)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        "normalized cross-section (adim.)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        "Normalized cross-section vs photon emission angle",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['theta_norm'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # ax - theta_abs

    ax = fig.add_subplot(
        gs[1, 1],
        sharex=dax['theta_norm']['handle'],
    )
    ax.set_xlabel(
        r"$\theta_{ph}$ (deg)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        "Absolute cross-section vs photon emission angle",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['theta_abs'] = {'handle': ax, 'type': 'isolines'}

    return dax
