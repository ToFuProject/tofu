

import copy


import numpy as np
import scipy.integrate as scpinteg
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import datastock as ds


from . import _xray_thin_target_integrated


# ####################################################
# ####################################################
#           DEFAULT
# ####################################################


# ANISOTROPY CASES
_DCASES = {
    0: {
        'E_e0_eV': 20e3,
        'E_ph_eV': 10e3,
        'color': 'r',
        'marker': '*',
        'ms': 14,
    },
    1: {
        'E_e0_eV': 100e3,
        'E_ph_eV': 50e3,
        'color': 'c',
        'marker': '*',
        'ms': 14,
    },
    2: {
        'E_e0_eV': 100e3,
        'E_ph_eV': 10e3,
        'color': 'm',
        'marker': '*',
        'ms': 14,
    },
    3: {
        'E_e0_eV': 1000e3,
        'E_ph_eV': 10e3,
        'color': (0.8, 0.8, 0),
        'marker': '*',
        'ms': 14,
    },
    4: {
        'E_e0_eV': 10000e3,
        'E_ph_eV': 10e3,
        'color': (0., 0.8, 0.8),
        'marker': '*',
        'ms': 14,
    },
    5: {
        'E_e0_eV': 1000e3,
        'E_ph_eV': 50e3,
        'color': (0.8, 0., 0.8),
        'marker': '*',
        'ms': 14,
    },
}
# ####################################################
# ####################################################
#        plot anisotropy
# ####################################################


def plot_xray_thin_d2cross_ei_anisotropy(
    # compute
    Z=None,
    E_e0_eV=None,
    E_ph_eV=None,
    theta_ph=None,
    per_energy_units=None,
    version=None,
    # hypergeometrc
    ninf=None,
    source=None,
    # selected cases
    dcases=None,
    # plot
    dax=None,
    fontsize=None,
    dplot_forbidden=None,
    dplot_peaking=None,
    dplot_thetamax=None,
    dplot_integ=None,
):

    # ---------------
    # check inputs
    # ---------------

    (
        E_e0_eV, E_ph_eV, theta_ph,
        dcases,
        fontsize,
        dplot_forbidden, dplot_peaking, dplot_thetamax, dplot_integ,
    ) = _check_anisotropy(
        E_e0_eV=E_e0_eV,
        E_ph_eV=E_ph_eV,
        theta_ph=theta_ph,
        version=version,
        # selected cases
        dcases=dcases,
        # plotting
        fontsize=fontsize,
        dplot_forbidden=dplot_forbidden,
        dplot_peaking=dplot_peaking,
        dplot_thetamax=dplot_thetamax,
        dplot_integ=dplot_integ,
    )

    # ---------------
    # prepare data
    # ---------------

    mod = _xray_thin_target_integrated
    d2cross = mod.get_xray_thin_d2cross_ei_integrated_thetae_dphi(
        # inputs
        Z=Z,
        E_e0_eV=E_e0_eV[None, :, None],
        E_ph_eV=E_ph_eV[None, None, :],
        theta_ph=theta_ph[:, None, None],
        # output customization
        per_energy_unit=per_energy_units,
        # version
        version=version,
        # hypergeometric
        ninf=ninf,
        source=source,
        # verb
        verb=False,
    )

    # --------------
    # prepare axes
    # --------------

    if dax is None:
        dax = _get_axes_anisotropy(
            Z=Z,
            version=version,
            fontsize=fontsize,
        )

    # ---------------
    # plot - map
    # ---------------

    kax = 'map'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for iv, (kk, vv) in enumerate(d2cross['cross'].items()):

            # compute integral and peaking
            integ, peaking = _get_peaking(
                vv['data'],
                theta_ph*180/np.pi,
                axis=0,
            )

            # integral
            if dplot_integ is not False:
                im0 = ax.contour(
                    E_e0_eV * 1e-3,
                    E_ph_eV * 1e-3,
                    np.log10(integ).T,
                    levels=dplot_integ['levels'],
                    colors=dplot_integ['colors'],
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
        lh = [
            mlines.Line2D(
                [], [],
                c=dplot_integ['colors'],
                label='log10(integral)',
            ),
            mlines.Line2D(
                [], [],
                c=dplot_peaking['colors'],
                label='peaking (1/std)',
            ),
            mlines.Line2D(
                [], [],
                c=dplot_thetamax['colors'],
                label='theta_max (deg)',
            ),
        ]
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

                # normalized
                kax = 'norm'
                if dax.get(kax) is not None:
                    ax = dax[kax]['handle']

                    ax.plot(
                        theta_ph * 180/np.pi,
                        yy / np.max(yy),
                        c=vcase['color'],
                        label=labi,
                    )

                # abs
                kax = 'log'
                if dax.get(kax) is not None:
                    ax = dax[kax]['handle']

                    l0, = ax.semilogy(
                        theta_ph * 180/np.pi,
                        yy*1e28,
                        c=vcase['color'],
                        label=labi,
                    )

    # normalized
    kax = 'norm'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']
        ax.legend(prop={'size': 12})
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 180)

    # normalized
    kax = 'abs'
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
    # selected cases
    dcases=None,
    # plotting
    fontsize=None,
    dplot_forbidden=None,
    dplot_peaking=None,
    dplot_thetamax=None,
    dplot_integ=None,
):

    # E_e0_eV
    if E_e0_eV is None:
        E_e0_eV = np.logspace(3, 6, 51)

    E_e0_eV = ds._generic_check._check_flat1darray(
        E_e0_eV, 'E_e0_eV',
        dtype=float,
        sign='>0',
    )

    # E_ph_eV
    if E_ph_eV is None:
        E_ph_eV = np.linspace(1, 100, 25) * 1e3

    E_ph_eV = ds._generic_check._check_flat1darray(
        E_ph_eV, 'E_ph_eV',
        dtype=float,
        sign='>0',
    )

    # theta_ph
    if theta_ph is None:
        theta_ph = np.linspace(0, np.pi, 41)

    # version
    if version is None:
        version = 'BHE'

    # ------------
    # dcases
    # ------------

    ddef = copy.deepcopy(_DCASES)
    if dcases in [None, True]:
        dcases = ddef

    if dcases is not False:
        for k0, v0 in dcases.items():
            dcases[k0] = _check_anisotropy_dplot(
                v0,
                f'dcases[{k0}]',
                ddef[0],
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

    # dplot_integ
    ddef = {'colors': 'g', 'levels': 20}
    dplot_integ = _check_anisotropy_dplot(
        dplot_integ,
        'dplot_integ',
        ddef,
    )

    return (
        E_e0_eV, E_ph_eV, theta_ph,
        dcases,
        fontsize,
        dplot_forbidden, dplot_peaking, dplot_thetamax, dplot_integ,
    )


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
            and all([kk in ddef.keys() for kk in din.keys()])
        )
        if not c0:
            lstr = [f"\t- ''{k0}': {v0}" for k0, v0 in ddef.items()]
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
            din[k0] = din.get(k0, v0)

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
    shape_integ = tuple([
        1 if ii == axis else ss
        for ii, ss in enumerate(data.shape)
    ])
    data_n = data / integ.reshape(shape_integ)

    # ----------
    # get average
    # ----------

    shape_x = tuple([-1 if ii == axis else 1 for ii in range(data.ndim)])
    xf = x.reshape(shape_x)
    x_avf = scpinteg.simpson(data_n * xf, x=x, axis=axis).reshape(shape_integ)
    std = np.sqrt(scpinteg.simpson(data_n * (xf - x_avf)**2, x=x, axis=axis))

    return integ, 1/std


# #############################################
# #############################################
#        Axes for anisotropy
# #############################################


def _get_axes_anisotropy(
    Z=None,
    version=None,
    fontsize=None,
):

    tit = (
        "Anisotropy"
    )

    dmargin = {
        'left': 0.08, 'right': 0.95,
        'bottom': 0.06, 'top': 0.85,
        'wspace': 0.2, 'hspace': 0.40,
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

    ax = fig.add_subplot(gs[:, 0], xscale='log')
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
    # ax - norm

    ax = fig.add_subplot(gs[0, 1])
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
    dax['norm'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # ax - log

    ax = fig.add_subplot(gs[1, 1], sharex=dax['norm']['handle'])
    ax.set_xlabel(
        r"$\theta_{ph}$ (deg)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$\frac{d^2\sigma_{ei}}{dkd\Omega_{ph}}$  ()",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        "Absolute cross-section vs photon emission angle",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['log'] = {'handle': ax, 'type': 'isolines'}

    return dax
