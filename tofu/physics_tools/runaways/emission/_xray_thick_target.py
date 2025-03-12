
import os


import numpy as np
import scipy.constants as scpct
import scipy.interpolate as scpinterp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


from .. import _utils


# ##############################################################
# ##############################################################
#                        DEFAULTS
# ##############################################################


_PATH_HERE = os.path.dirname(__file__)


# ##############################################################
# ##############################################################
#               Bremsstrahlung anisotropy factor
# ##############################################################


def anisotropy(
    gamma=None,
    costheta=None,
):
    """ Return the anisotropic factor (unitless)

    Depends on:
        - gamma: thelorentz factor of the Runaway electron
        - costheta: angle of observation relative to electron direction

    ref:
    [1] Pandya et al., Physica Scripta 93, no. 11 (November 1, 2018): 115601

    """

    # -----------
    # check inputs
    # -----------

    gamma, costheta = _check_anisotropy(
        gamma=gamma,
        costheta=costheta,
    )

    # -----------
    # prepare
    # -----------

    # gamma => beta
    beta = _utils.convert_momentum_velocity_energy(
        gamma=gamma,
    )['beta']['data']

    # -----------
    # compute
    # -----------

    # anisotropy of cross-section
    anis = (
        (3/8) * (1 + ((costheta - beta) / (1 - beta * costheta))**2)
        / (gamma**2 * (1 - beta * costheta)**2)
    )

    return anis


def _check_anisotropy(
    gamma=None,
    costheta=None,
):

    dout, shape = ds._generic_check._check_all_broadcastable(
        gamma=gamma,
        costheta=costheta,
    )

    return dout['gamma'], dout['costheta']


# ##############################################################
# ##############################################################
#            Differencial Bremsstrahlung cross-section
# ##############################################################


def dcross_ei(
    E_re_eV=None,
    E_ph_eV=None,
    atomic_nb=None,
    adjust=None,
    # optional anistropy
    costheta=None,
    # options
    return_intermediates=None,
):
    """ Return the energy-dependent HXR generation cros-section

    Considers electron-ion bremsstrahlung

    To be multiplied by the anisotropy factor
        => see get_anisotropy_factor()

    Sources:
        [1] Nocente et al., Nuclear Fusion 57, no. 7 (July 1, 2017): 076016.
        [2] Salvat et al., Nuclear Instruments and Methods in Physics Research
            Section B: Beam Interactions with Materials and Atoms 63,
            no. 3 (February 1992): 255–69
    """

    # --------------------
    # check inputs
    # --------------------

    (
        E_re_eV, E_ph_eV,
        atomic_nb, adjust,
        costheta,
        return_intermediates,
    ) = _check_dcross_ei(
        E_re_eV=E_re_eV,
        E_ph_eV=E_ph_eV,
        atomic_nb=atomic_nb,
        adjust=adjust,
        # optional anistropy
        costheta=costheta,
        # options
        return_intermediates=return_intermediates,
    )

    # --------------------
    # Load tabulated data
    # --------------------

    # load screening radius
    fname = "RE_HXR_CrossSection_ScreeningRadius_Salvat.csv"
    pfe = os.path.join(_PATH_HERE, fname)
    Z_R, RZ3a0 = np.loadtxt(pfe, delimiter=',').T

    fname = "RE_HXR_ElectronElectron_Salvat.csv"
    pfe = os.path.join(_PATH_HERE, fname)
    Z_eta, eta_inf = np.loadtxt(pfe, delimiter=',').T

    # -----------------------------------
    # cross-section (without anisotropy)
    # ----------------------------------

    # -------------------
    # prepare constants

    # hbar (J.s => eV.s)
    hbar_eVs = scpct.hbar / scpct.e

    # rest energy of electron (J => eV)
    mc2_eV = scpct.m_e * scpct.c**2 / scpct.e

    # mc (eV.s/m)
    mc_eVsm = mc2_eV / scpct.c

    # fine structure constant (adim.)
    alpha = scpct.alpha

    # a0 = bohr radius (m)
    a0 = 5.291772e-11

    # screening radius (should be tabulate from fig. 4), m
    # R = 0.81 * a0 / Z**(1/3)
    Rz3a0 = scpinterp.interp1d(
        np.round(Z_R),
        RZ3a0,
        kind='linear',
    )(atomic_nb)
    R = Rz3a0 * a0 / atomic_nb**(1/3)

    # high-energy coulomb correction (adim.)
    aa = (alpha * atomic_nb)**2
    fc = aa * np.sum([1./(nn * (nn**2 + aa)) for nn in range(1, 101)])

    # log(R * mc/hbar)   adim
    logRmchb = np.log(R * mc_eVsm / hbar_eVs)

    # ----------------------
    # prepare (nEe,) vectors

    # should tabulate eta_inf vs Z from graph (cf. fig. 5), adim.
    # eta_inf = 1.158
    eta_inf = scpinterp.interp1d(
        np.round(Z_eta),
        eta_inf,
        kind='linear',
    )(atomic_nb)

    # eta
    eta = (E_re_eV / mc2_eV)**0.8 / ((E_re_eV / mc2_eV)**0.8 + 2.43) * eta_inf

    # correction term at low energies adim
    F2 = (
        (2.04 + 9.09 * alpha * atomic_nb)
        * (mc2_eV**2 / (E_re_eV * (E_re_eV + mc2_eV)))
        ** (1.26 - 0.93 * alpha * atomic_nb)
    )

    # useful for q0 = minimum momentum transfer (eV.s/m)
    gamma = _utils.convert_momentum_velocity_energy(
        energy_kinetic_eV=E_re_eV,
    )['gamma']['data']

    # --------------------------
    # prepare (nEe, nEph) arrays

    # reduced energy of the photon (adim.)
    eps = E_ph_eV / (E_re_eV + mc2_eV)
    eps[eps > 1] = np.nan
    epsd = (E_re_eV - 5 * mc2_eV) / (E_re_eV + mc2_eV)

    # turn off fc when eps > epsd
    theta = eps < epsd

    # q0 = minimum momentum transfer (eV.s/m)
    q0 = (mc_eVsm / (2 * gamma)) * eps / (1 - eps)

    # adim
    bb = R * q0 / hbar_eVs

    # Phi1
    Phi1 = 2 - 2*np.log(1 + bb**2) - 4*bb*np.arctan(1/bb) + 4*logRmchb

    # Phi2
    term2 = 2*bb**2 * (4 - 4*bb*np.arctan(1/bb) - 3*np.log(1 + 1/bb**2))

    # adjustment
    if adjust is True:
        bb_too_large = bb > 1000
        term2[bb_too_large] = -10/3.

    Phi2 = (
        (4/3) - 2*np.log(1 + bb**2)
        + term2
        + 4*logRmchb
    )

    # adim
    f0 = 4*logRmchb + F2 - 4*fc*theta
    f1 = Phi1 - 4*logRmchb
    f2 = 0.5*(3*Phi1 - Phi2) - 4*logRmchb

    phi1 = f1 + f0
    phi2 = (4/3) * (1 - eps) * (f2 + f0)

    # cross-section in m2
    dcross_Ere_deps = (
        a0**2 * alpha**5 * atomic_nb * (atomic_nb + eta)
        * (phi1 * eps + phi2 / eps)
    )

    # change of variable to find derivative vs Eph (m2/eV)
    # deps / dEp = 1/(E_re + mc2)
    # m2 / eV
    dcross_Ere = dcross_Ere_deps / (E_re_eV + mc2_eV)

    # -------------
    # Optional anisotropy
    # -------------

    if costheta is not None:
        anis = anisotropy(
            costheta=costheta,
            gamma=gamma,
        )

        dcross_Ere = dcross_Ere * anis

    # -------------
    # format output
    # -------------

    dout = {
        'dcross_ei_Ere': {
            'data': dcross_Ere,
            'units': 'm2/eV',
        },
    }

    # -----------------
    # intermediates
    # -----------------

    if return_intermediates is True:

        dout.update({
            'RZ13a0': {
                'data': Rz3a0,
                'units': '',
            },
            'eta_inf': {
                'data': eta_inf,
                'units': '',
            },
        })

    return dout


def _check_dcross_ei(
    E_re_eV=None,
    E_ph_eV=None,
    atomic_nb=None,
    adjust=None,
    # optional anistropy
    costheta=None,
    # options
    return_intermediates=None,
):

    # -----------------
    # options
    # -----------------

    # adjust
    adjust = ds._generic_check._check_var(
        adjust, 'adjust',
        types=bool,
        default=True,
    )

    # return_intermediates
    return_intermediates = ds._generic_check._check_var(
        return_intermediates, 'return_intermediates',
        types=bool,
        default=False,
    )

    # -----------------
    # broadcastable
    # -----------------

    dout = {
        'E_re_eV': E_re_eV,
        'E_ph_eV': E_ph_eV,
        'atomic_nb': atomic_nb,
    }
    if costheta is not None:
        dout['costheta'] = costheta

    dout, shape = ds._generic_check._check_all_broadcastable(
        **dout,
    )

    lk = ['E_re_eV', 'E_ph_eV', 'atomic_nb']
    E_re_eV, E_ph_eV, atomic_nb = [dout[k0] for k0 in lk]
    if costheta is not None:
        costheta = dout['costheta']

    return (
        E_re_eV, E_ph_eV,
        atomic_nb, adjust,
        costheta,
        return_intermediates,
    )


# ##############################################################
# ##############################################################
#            plot Differencial Bremsstrahlung cross-section
# ##############################################################


def plot_dcross_ei(
    atomic_nb=None,
    Eplot_e=None,
    Eplot_ph=None,
    ang=None,
):

    # --------------
    # Check inputs
    # --------------

    # atomic number
    if atomic_nb is None:
        atomic_nb = 13

    # runaway
    if Eplot_e is None:
        Eplot_e = np.r_[1e6, 10e6, 15e6]
    E_e = Eplot_e

    # photon
    if Eplot_ph is None:
        Eplot_ph = np.r_[500e3, 5e6, 10e6]
    E_ph = np.linspace(Eplot_ph.min()*0.9, Eplot_ph.max()*1.1, 100)

    # angle
    if ang is None:
        ang = np.pi * np.linspace(0, 1, 50)

    # --------------
    # compute data
    # --------------

    # cross-section
    dcross = dcross_ei(
        E_re_eV=E_e[:, None, None],
        E_ph_eV=E_ph[None, :, None],
        atomic_nb=atomic_nb,
        adjust=True,
        # optional anistropy
        costheta=np.cos(ang[None, None, :]),
        # options
        return_intermediates=True,
    )['dcross_ei_Ere']['data']

    # --------------
    # prepare data
    # --------------

    # prepare
    indE_e = np.array([np.argmin(np.abs(E_e - ee)) for ee in Eplot_e])
    indE_ph = np.array([np.argmin(np.abs(E_ph - ee)) for ee in Eplot_ph])

    # str
    E_re_str = [
        f"{round(Eplot_e[ii]*1e-6, ndigits=1)} MeV"
        for ii in range(Eplot_e.size)
    ]
    E_ph_str = [
        f"{round(Eplot_ph[ii]*1e-6, ndigits=1)} MeV"
        for ii in range(Eplot_ph.size)
    ]

    # vmin, vmax
    vmin = np.min(np.log10(dcross[dcross > 0]))
    vmax = np.max(np.log10(dcross[dcross > 0]))
    dcross_plot = np.full(dcross.shape[1:], np.nan)

    # --------------
    # prepare figure
    # --------------

    tit = (
        "cross section vs E_e, E_ph, angle\n"
        f"Z = {atomic_nb}"
    )

    dmargin = {
        'left': 0.05, 'right': 0.95,
        'bottom': 0.05, 'top': 0.95,
        'wspace': 0.2, 'hspace': 0.2,
    }

    # --------------
    # figure
    # --------------

    # prepare figure
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(tit, size=12, fontweight='bold')
    gs = gridspec.GridSpec(ncols=3, nrows=indE_e.size, **dmargin)

    # --------------
    # plot
    # --------------

    axref = None
    axref1 = None
    dcol = {}
    for ii in range(indE_e.size):

        # ax2
        ax2 = fig.add_subplot(gs[ii, 0])
        ax2.set_xlabel(
            r'$log_{10}(E_{ph})$ (eV)',
            size=12,
            fontweight='bold',
        )
        ax2.set_ylabel('angle (rad)', size=12, fontweight='bold')

        # ax0 - anisotropy
        ax0 = fig.add_subplot(
            gs[ii, 1],
            aspect='equal',
            # adjustable='datalim',
            sharex=axref,
            sharey=axref,
        )
        ax0.set_xlabel(r'$\log_{10}(\sigma)$', size=12, fontweight='bold')
        ax0.set_ylabel(r'$\log_{10}(\sigma)$', size=12, fontweight='bold')
        ax0.set_title(f'E_re = {E_re_str[ii]}', size=12, fontweight='bold')

        # ax1
        ax1 = fig.add_subplot(gs[ii, 2], sharey=axref1)
        ax1.set_xlabel('E_ph (eV)', size=12, fontweight='bold')
        ax1.set_ylabel('cross-section (m2/eV)', size=12, fontweight='bold')

        # contour
        dcross_plot[...] = np.nan
        iok = np.isfinite(dcross[indE_e[ii], :, :])
        iok[iok] = dcross[indE_e[ii], iok] > 0
        dcross_plot[iok] = np.log10(dcross[indE_e[ii], iok])
        im = ax2.contourf(
            np.log10(E_ph),
            ang,
            dcross_plot.T,
            levels=15,
            vmin=vmin,
            vmax=vmax,
        )

        # contour
        for jj, iEph in enumerate(indE_ph):
            ll, = ax0.plot(
                (dcross_plot[iEph, :] - vmin) * np.cos(ang),
                (dcross_plot[iEph, :] - vmin) * np.sin(ang),
                '.-',
                label=f"E_ph = {E_ph_str[jj]}",
                c=dcol.get(jj),
            )
            if ii == 0:
                dcol[jj] = ll.get_color()

            ax2.axvline(np.log10(E_ph[iEph]), c=dcol[jj], ls='--', lw=1)

        ax0.legend()

        # slice
        ax1.loglog(
            E_ph,
            dcross_plot[:, 0],
            '.-',
            c='k',
            label=f"{ang[0]*180/np.pi:.1f} deg",
        )
        ax1.loglog(
            E_ph,
            dcross_plot[:, -1],
            '.--',
            c='k',
            label=f"{ang[-1]*180/np.pi:.1f} deg",
        )
        ax1.legend()

        plt.colorbar(im, ax=ax2, label=r'$\log_{10}(\sigma)$')

        ax0.axhline(0, c='k', lw=1, ls='--')
        ax0.axvline(0, c='k', lw=1, ls='--')
        ax0.set_ylim(bottom=0)

        if ii == 0:
            axref = ax0
            axref1 = ax1

    return {'map': ax2, 'anisotropy': ax0, 'curve': ax1}


# #################################################################
# #################################################################
#                Plot vs SALVAT
# #################################################################


def plot_dcross_vs_Salvat(
):

    # --------------
    # input data
    # --------------

    E_re = np.r_[1, 10, 100]*1e6
    E_ph = np.linspace(0.01*np.min(E_re), np.max(E_re), 10000)

    E_re_str = [f"{round(ee*1e-6)} MeV" for ee in E_re]

    # --------------
    # prepare figure
    # --------------

    # cross-section
    dcross = dcross_ei(
        E_re_eV=E_re[None, :],
        E_ph_eV=E_ph[:, None],
        atomic_nb=13,
        adjust=True,
        # optional anistropy
        costheta=None,
        # options
        return_intermediates=True,
    )['dcross_ei_Ere']['data']

    # beta
    beta = _utils.convert_momentum_velocity_energy(
        energy_kinetic_eV=E_re,
    )['beta']['data']

    # --------------
    # prepare figure
    # --------------

    ERE = np.r_[1., 10., 100.]*1e6
    RSB = [
        0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 1.,
    ]
    ySB = np.array([
        [14.03, 10.17, 7.878, 6.296, 5.092, 4.111, 3.287, 2.564, 1.913, 1.312,
         1.001, 0.87, 0.735, 0.665],
        [16.45, 12.68, 10.80, 9.34, 8.077, 7.057, 6.165, 5.309, 4.388, 3.105,
         2.111, 1.588, 0.945, 0.52],
        [16.36, 13.24, 11.98, 10.94, 10.10, 9.454, 8.967, 8.585, 8.173, 7.262,
         6.022, 5.004, 2.964, 0.496],
    ])

    # --------------
    # prepare figure
    # --------------

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        "Comparing to literature for Al (Z=13)\n"
        "Salvat et al. 1992",
        size=14,
        fontweight='bold',
    )

    ylab = r'$\left(\frac{\beta}{Z}\right)^2 E_{ph} \frac{d\sigma}{dE_{ph}}$'

    # --------------
    # axes
    # --------------

    dax = {}
    ax0 = None
    for ii, ee in enumerate(E_re):
        ax = fig.add_axes(
            [0.08 + ii*(0.27 + 0.05), 0.08, 0.27, 0.75],
            sharex=ax0,
            sharey=ax0,
        )

        ax.set_xlabel(r"$\frac{E_{ph}}{E_{re}}$", size=12, fontweight='bold')
        ax.set_title(
            r"$E_{re}$" + f" = {E_re_str[ii]}",
            fontweight='bold',
            size=12,
        )

        if ii == 0:
            ax0 = ax
            ax.set_ylabel(
                ylab + ' (mb)',
                size=12,
                fontweight='bold',
            )

        dax[ii] = ax

    # --------------
    # plot
    # --------------

    for ii, ee in enumerate(E_re):
        ax = dax[ii]

        indE = np.argmin(np.abs(E_re - ee))
        indES = np.argmin(np.abs(ERE - ee))

        ax.plot(
            E_ph / E_re[indE],
            (beta[ii] / 13**2) * E_ph * dcross[:, indE] * 1e31,
            '.-',
            label=f'calc (E_re = {E_re_str[ii]})',
        )

        ax.plot(
            RSB,
            ySB[indES, :],
            'ok',
            markerfacecolor='None',
            label=f'ref (E_re = {ee*1e-6} MeV)',
        )
        ax.legend()

    ax0.set_xlim(0, 1)

    return dax
