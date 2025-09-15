

import os


import numpy as np
import scipy.integrate as scpinteg
import astropy.units as asunits
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import datastock as ds


from . import _xray_thin_target


# ####################################################
# ####################################################
#           DEFAULT
# ####################################################


_PATH_HERE = os.path.dirname(__file__)


_E_E0_EV = 45e3
_E_PH_EV = 40e3
_THETA_PH = np.linspace(0, np.pi, 31)

# Integration
_NTHETAE = 31
_NDPHI = 51


# ####################################################
# ####################################################
#        main
# ####################################################


def get_xray_thin_d2cross_ei_integrated_thetae_dphi(
    # inputs
    Z=None,
    E_e0_eV=None,
    E_ph_eV=None,
    theta_ph=None,
    # hypergeometric parameter
    ninf=None,
    source=None,
    # integration parameters
    nthetae=None,
    ndphi=None,
    # output customization
    per_energy_unit=None,
    # version
    version=None,
    # verb
    verb=None,
):

    # ------------
    # inputs
    # ------------

    (
        E_e0_eV, E_ph_eV, theta_ph,
        nthetae, ndphi,
        shape, shape_theta_e, shape_dphi,
        verb,
    ) = _check(
        # inputs
        E_e0_eV=E_e0_eV,
        E_ph_eV=E_ph_eV,
        theta_ph=theta_ph,
        # integration parameters
        nthetae=nthetae,
        ndphi=ndphi,
        # verb
        verb=verb,
    )

    # ------------------
    # Derive angles
    # ------------------

    # E_e1_eV
    E_e1_eV = E_e0_eV - E_ph_eV

    # angles
    theta_e = np.pi * np.linspace(0, 1, nthetae)
    dphi = np.pi * np.linspace(-1, 1, ndphi)
    theta_ef = theta_e.reshape(shape_theta_e)
    dphif = dphi.reshape(shape_dphi)

    # derived
    sinte = np.sin(theta_ef)

    # ------------------
    # get d3cross
    # ------------------

    if verb is True:
        msg = "Computing d3cross..."
        print(msg)

    d3cross = _xray_thin_target.get_xray_thin_d3cross_ei(
        # inputs
        Z=Z,
        E_e0_eV=E_e0_eV[..., None, None],
        E_e1_eV=E_e1_eV[..., None, None],
        # directions
        theta_ph=theta_ph[..., None, None],
        theta_e=theta_ef,
        dphi=dphif,
        # hypergeometric parameter
        ninf=ninf,
        source=source,
        # output customization
        per_energy_unit=per_energy_unit,
        # version
        version=version,
        # debug
        debug=False,
    )

    # ------------------
    # prepare output
    # ------------------

    d2cross = {
        # energies
        'E_e0': {
            'data': E_e0_eV,
            'units': 'eV',
        },
        'E_ph': {
            'data': E_ph_eV,
            'units': 'eV',
        },
        # angles
        'theta_ph': {
            'data': theta_ph,
            'units': 'rad',
        },
        'theta_e': {
            'data': theta_e,
            'units': 'rad',
        },
        'dphi': {
            'data': dphi,
            'units': 'rad',
        },
        # cross-section
        'cross': {
            vv: {
                'data': np.full(shape, 0.),
                'units': asunits.Unit(vcross['units']) * asunits.Unit('sr'),
            }
            for vv, vcross in d3cross['cross'].items()
        },
    }

    # ------------------
    # integrate
    # ------------------

    if verb is True:
        msg = "Integrating..."
        print(msg)

    for vv, vcross in d3cross['cross'].items():
        d2cross['cross'][vv]['data'][...] = scpinteg.simpson(
            scpinteg.simpson(
                vcross['data'] * sinte,
                x=theta_e,
                axis=-1,
            ),
            x=dphi,
            axis=-1,
        )

    return d2cross


# ####################################################
# ####################################################
#        check
# ####################################################


def _check(
    # inputs
    E_e0_eV=None,
    E_ph_eV=None,
    theta_ph=None,
    # integration parameters
    nthetae=None,
    ndphi=None,
    # verb
    verb=None,
):

    # -----------
    # arrays
    # -----------

    # --------
    # E_e0_eV

    if E_e0_eV is None:
        E_e0_eV = _E_E0_EV
    E_e0_eV = np.atleast_1d(E_e0_eV)

    # --------
    # E_ph_eV

    if E_ph_eV is None:
        E_ph_eV = _E_PH_EV
    E_ph_eV = np.atleast_1d(E_ph_eV)

    # -------
    # theta_e

    if theta_ph is None:
        theta_ph = _THETA_PH
    theta_ph = np.atleast_1d(theta_ph)

    # -------------
    # Broadcastable

    dout, shape = ds._generic_check._check_all_broadcastable(
        return_full_arrays=False,
        E_e0_eV=E_e0_eV,
        E_ph_eV=E_ph_eV,
        # directions
        theta_ph=theta_ph,
    )

    # -----------
    # shapes
    # -----------

    shape = np.broadcast_shapes(E_e0_eV.shape, E_ph_eV.shape, theta_ph.shape)
    shape_theta_e = (1,) * (len(shape)+1) + (-1,)
    shape_dphi = (1,) * len(shape) + (-1, 1)

    # -----------
    # integers
    # -----------

    # nthetae
    nthetae = ds._generic_check._check_var(
        nthetae, 'nthetae',
        types=int,
        sign='>0',
        default=_NTHETAE,
    )

    # ndphi
    ndphi = ds._generic_check._check_var(
        ndphi, 'ndphi',
        types=int,
        sign='>0',
        default=_NDPHI,
    )

    # -----------
    # verb
    # -----------

    verb = ds._generic_check._check_var(
        verb, 'verb',
        types=bool,
        default=True,
    )

    return (
        E_e0_eV, E_ph_eV, theta_ph,
        nthetae, ndphi,
        shape, shape_theta_e, shape_dphi,
        verb,
    )


# ####################################################
# ####################################################
#        plot vs litterature
# ####################################################


def plot_xray_thin_d2cross_ei_vs_literature():
    """ Plot electron-angle-integrated cross section vs

    [1] G. Elwert and E. Haug, Phys. Rev., 183, pp. 90–105, 1969
        doi: 10.1103/PhysRev.183.90.

    """

    # --------------
    # Load literature data
    # --------------

    # isolines
    pfe_fig12 = os.path.join(
        _PATH_HERE,
        'RE_HXR_CrossSection_ThinTarget_PhotonAngle_ElwertHaug_fig12.csv',
    )
    out_fig12 = np.loadtxt(pfe_fig12, delimiter=',')

    # --------------------
    # prepare data
    # --------------------

    msg = "\nComputing data for fig12 (1/3):"
    print(msg)

    theta_ph = np.linspace(0, 1, 31)*np.pi

    # -----------
    # fig 12

    msg = "\t- For Z = 8... (1/2)"
    print(msg)

    d2cross_fig12_Z8 = get_xray_thin_d2cross_ei_integrated_thetae_dphi(
        # inputs
        Z=8,
        E_e0_eV=45e3,
        E_ph_eV=40e3,
        theta_ph=theta_ph,
        # output customization
        per_energy_unit=None,
        # version
        version=['EH', 'BH'],
        # verb
        verb=False,
    )

    msg = "\t- For Z = 13... (1/2)"
    print(msg)

    d2cross_fig12_Z13 = get_xray_thin_d2cross_ei_integrated_thetae_dphi(
        # inputs
        Z=13,
        E_e0_eV=45e3,
        E_ph_eV=40e3,
        theta_ph=theta_ph,
        # output customization
        per_energy_unit=None,
        # version
        version=['EH', 'BH'],
        # verb
        verb=False,
    )

    # --------------
    # prepare axes
    # --------------

    fontsize = 14
    tit = (
        "[1] G. Elwert and E. Haug, Phys. Rev., 183, p.90, 1969\n"
    )

    dmargin = {
        'left': 0.08, 'right': 0.95,
        'bottom': 0.06, 'top': 0.85,
        'wspace': 0.2, 'hspace': 0.40,
    }

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(tit, size=fontsize+2, fontweight='bold')

    gs = gridspec.GridSpec(ncols=2, nrows=1, **dmargin)
    dax = {}

    # --------------
    # prepare axes
    # --------------

    # --------------
    # ax - isolines

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(
        r"$\theta_{ph}$ (photon emission angle, deg)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$\frac{k}{Z^2}\frac{d^2\sigma}{dkd\Omega_{ph}}$ (mb/sr)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        "[1] Fig 12. Integrated cross-section (vs theta_e and phi)\n"
        "Comparisation between experimental values and models\n"
        + r"$Z = O$ (O) and $Z = 13$ (Al), "
        + r"$E_{e0} = 45 keV$, $E_{e1} = 5 keV$"
        + "\nTarget was " + r"$Al_2O_3$",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['fig12'] = {'handle': ax, 'type': 'isolines'}

    # ------------
    # ax - ph_dist

    # ---------------
    # plot fig 12
    # ---------------

    kax = 'fig12'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # literature data
        inan = np.r_[0, np.any(np.isnan(out_fig12), axis=1).nonzero()[0], -1]
        dls = {
            0: {'ls': '--', 'lab': 'Born approx'},
            1: {'ls': '-.', 'lab': 'Z = 8, EH'},
            2: {'ls': '-', 'lab': 'Z = 13, EH'},
            3: {'ls': '-', 'lab': 'Z = 13, Non-rel.'},
            4: {'ls': 'None', 'lab': 'exp.'},
        }
        for ii, ia in enumerate(inan[:-1]):
            ax.plot(
                out_fig12[inan[ii]:inan[ii+1], 0],
                out_fig12[inan[ii]:inan[ii+1], 1],
                c='k',
                ls=dls[ii]['ls'],
                marker='o' if ii == 4 else 'None',
                ms=10,
                label=dls[ii]['lab'],
            )

        # -------------
        # computed data

        # Z = 13
        Z = 13
        for k0, v0 in d2cross_fig12_Z13['cross'].items():
            ax.plot(
                theta_ph * 180/np.pi,
                v0['data']*1e28*1e3 * 40e3 / Z**2,
                ls='-',
                lw=3 if v0 == 'EH' else 1.5,
                alpha=0.5,
                label=f'computed - {k0} Z = {Z}',
            )

        # Z = 8
        Z = 8
        for k0, v0 in d2cross_fig12_Z8['cross'].items():
            ax.plot(
                theta_ph * 180/np.pi,
                v0['data']*1e28*1e3 * 40e3 / Z**2,
                ls='-',
                lw=3 if v0 == 'EH' else 1.5,
                alpha=0.5,
                label=f'computed - {k0} Z = {Z}',
            )

        ax.set_xlim(0, 180)
        ax.set_ylim(0, 8)

        # add legend
        ax.legend()

    # ------------------------
    # plot photon distribution
    # ------------------------

    return dax, d2cross_fig12_Z13, d2cross_fig12_Z8


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

    d2cross = get_xray_thin_d2cross_ei_integrated_thetae_dphi(
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
        units = str(vv['units']).replace('m2', 'barn')
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

    ddef = {
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

    integ = scpinteg.simpson(data, x=x, axis=axis)
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
