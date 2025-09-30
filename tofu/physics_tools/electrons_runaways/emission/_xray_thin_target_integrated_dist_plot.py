

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import datastock as ds


from . import _xray_thin_target_integrated_dist


# ############################################
# ############################################
#             Default
# ############################################


_DPLASMA_ANISOTROPY_MAP = {
    'Te_eV': np.r_[1, 5, 10, 20]*1e3,
    'ne_m3': np.r_[1, 3, 5, 10, 30, 50, 100]*1e19,
    'jp_Am2': np.r_[0, 1, 3, 5, 8, 10]*1e6,
}


_DPLASMA_ANGULAR_PROFILES = {
    'Te_eV': np.r_[1, 1, 10, 1]*1e3,
    'ne_m3': np.r_[1e19, 1e20, 1e19, 1e19],
    'jp_Am2': np.r_[1e6, 1e6, 1e6, 10e6],
    're_fraction_Ip': 0.,
}


_THETA_PH_VSB = np.linspace(0, np.pi, 7)    # 7
# _THETA_E0_VSB_NPTS = 15                     # 19
_THETA_E0_VSB_NPTS = 9                       # 19
_PHI_E0_VSB_NPTS = _THETA_E0_VSB_NPTS*2 + 1
# _E_PH_EV = np.r_[5., 10., 15., 20., 30., 50.] * 1e3
_E_PH_EV = np.r_[5., 15., 30.] * 1e3
# _E_E0_EV = np.logspace(-2, 4, 61)*1e3      # 31
_E_E0_EV = np.logspace(-2, 4, 31)*1e3      # 31


# ############################################
# ############################################
#             Main
# ############################################


def plot_xray_thin_integ_dist(
    # ----------------
    # electron distribution
    Te_eV=None,
    ne_m3=None,
    jp_Am2=None,
    re_fraction_Ip=None,
    # ----------------
    # cross-section
    E_ph_eV=None,
    E_e0_eV=None,
    theta_e0_vsB_npts=None,
    phi_e0_vsB_npts=None,
    theta_ph_vsB=None,
    # inputs
    Z=None,
    # hypergeometric parameter
    ninf=None,
    source=None,
    # integration parameters
    nthetae=None,
    ndphi=None,
    # output customization
    version_cross=None,
    # plots
    plot_angular_spectra=None,
    plot_anisotropy_map=None,
    # verb
    verb=None,
):
    """ plot the Maxwellian-integrated Bremsstrahlung

    fig.1: vs the standard isotropic formula (quantotative validation)

    fig. 2: to show the anisotropy vs B at various plasma parameters and
    energies

    """

    # -------------
    # check inputs
    # -------------

    (
        dplasma,
        E_ph_eV, E_e0_eV,
        theta_ph_vsB,
        theta_e0_vsB_npts, phi_e0_vsB_npts,
        version_cross,
        plot_angular_spectra,
        plot_anisotropy_map,
        verb,
    ) = _check(
        # electron distribution
        Te_eV=Te_eV,
        ne_m3=ne_m3,
        jp_Am2=jp_Am2,
        re_fraction_Ip=re_fraction_Ip,
        # cross-section
        E_ph_eV=E_ph_eV,
        E_e0_eV=E_e0_eV,
        theta_e0_vsB_npts=theta_e0_vsB_npts,
        phi_e0_vsB_npts=phi_e0_vsB_npts,
        theta_ph_vsB=theta_ph_vsB,
        version_cross=version_cross,
        # plots
        plot_angular_spectra=plot_angular_spectra,
        plot_anisotropy_map=plot_anisotropy_map,
        verb=verb,
    )

    # -------------
    # compute
    # -------------

    (
        demiss, ddist,
    ) = _xray_thin_target_integrated_dist.get_xray_thin_integ_dist(
        # ----------------
        # cross-section
        E_ph_eV=E_ph_eV,
        E_e0_eV=E_e0_eV,
        theta_e0_vsB_npts=theta_e0_vsB_npts,
        phi_e0_vsB_npts=phi_e0_vsB_npts,
        theta_ph_vsB=theta_ph_vsB,
        # inputs
        Z=Z,
        # hypergeometric parameter
        ninf=ninf,
        source=source,
        # integration parameters
        nthetae=nthetae,
        ndphi=ndphi,
        # output customization
        version_cross=version_cross,
        # verb
        verb=verb-1,
        # ----------------
        # electron distribution
        **dplasma,
    )

    # -------------
    # plots
    # -------------

    if plot_angular_spectra is True:
        dax = _plot_angular_spectra(
            **locals(),
        )

    if plot_anisotropy_map is True:
        dax = _plot_anisotropy_map(
            **locals(),
        )
        pass

    return dax, demiss, ddist, dplasma


# ############################################
# ############################################
#             Check
# ############################################


def _check(
    # ----------------
    # electron distribution
    Te_eV=None,
    ne_m3=None,
    jp_Am2=None,
    re_fraction_Ip=None,
    # ----------------
    # cross-section
    E_ph_eV=None,
    E_e0_eV=None,
    theta_e0_vsB_npts=None,
    phi_e0_vsB_npts=None,
    theta_ph_vsB=None,
    # version
    version_cross=None,
    # plots
    plot_angular_spectra=None,
    plot_anisotropy_map=None,
    verb=None,
):

    # --------------------
    # plot_angular_spectra
    # --------------------

    plot_angular_spectra = ds._generic_check._check_var(
        plot_angular_spectra, 'plot_angular_spectra',
        types=bool,
        default=True,
    )

    # --------------------
    # plot_anisotropy_map
    # --------------------

    plot_anisotropy_map = ds._generic_check._check_var(
        plot_anisotropy_map, 'plot_anisotropy_map',
        types=bool,
        default=True,
    )

    # -------------------------------------
    # Te_eV, ne_m3, jp_Am2, re_fraction_Ip
    # -------------------------------------

    if plot_anisotropy_map is False:
        dplasma = _dplasma_asis()
    else:
        dplasma = _dplasma_map()

    # ----------
    # E_ph_eV
    # ----------

    if E_ph_eV is None:
        E_ph_eV = _E_PH_EV

    E_ph_eV = ds._generic_check._check_flat1darray(
        E_ph_eV, 'E_ph_eV',
        dtype=float,
        sign='>=0',
    )

    # ----------
    # E_e0_eV
    # ----------

    if E_e0_eV is None:
        E_e0_eV = _E_E0_EV

    E_e0_eV = ds._generic_check._check_flat1darray(
        E_e0_eV, 'E_e0_eV',
        dtype=float,
        sign='>=0',
    )

    # ------------
    # theta_ph_vsB
    # ------------

    if theta_ph_vsB is None:
        theta_ph_vsB = _THETA_PH_VSB

    theta_ph_vsB = ds._generic_check._check_flat1darray(
        theta_ph_vsB, 'theta_ph_vsB',
        dtype=float,
    )

    # ------------
    # theta_e0_vsB
    # ------------

    theta_e0_vsB_npts = int(ds._generic_check._check_var(
        theta_e0_vsB_npts, 'theta_e0_vsB_npts',
        types=(int, float),
        sign='>=3',
        default=_THETA_E0_VSB_NPTS,
    ))

    # --------------------
    # phi_e0_vsB
    # --------------------

    phi_e0_vsB_npts = int(ds._generic_check._check_var(
        phi_e0_vsB_npts, 'phi_e0_vsB_npts',
        types=(int, float),
        sign='>=5',
        default=_PHI_E0_VSB_NPTS,
    ))

    # --------------------
    # version_cross
    # --------------------

    lok = ['BHE', 'EH']
    version_cross = ds._generic_check._check_var(
        version_cross, 'version_cross',
        types=str,
        allowed=lok,
        default=lok[0],
    )

    # --------------------
    # verb
    # --------------------

    lok = [False, True, 0, 1, 2, 3]
    verb = int(ds._generic_check._check_var(
        verb, 'verb',
        types=(int, bool),
        default=lok[-1],
        allowed=lok,
    ))

    return (
        dplasma,
        E_ph_eV, E_e0_eV,
        theta_ph_vsB,
        theta_e0_vsB_npts, phi_e0_vsB_npts,
        version_cross,
        plot_angular_spectra,
        plot_anisotropy_map,
        verb,
    )


def _dplasma_asis(
    Te_eV=None,
    ne_m3=None,
    jp_Am2=None,
):

    # -----------------
    # initialize
    # -----------------

    dplasma = {
        'Te_eV': Te_eV,
        'ne_m3': ne_m3,
        'jp_Am2': jp_Am2,
        # 're_fraction_Ip': re_fraction_Ip,
    }

    # -----------------
    # set default + array
    # -----------------

    # default + np.ndarray
    size = 1
    for k0, v0 in dplasma.items():
        if v0 is None:
            v0 = _DPLASMA_ANGULAR_PROFILES[k0]
        v0 = np.atleast_1d(v0).ravel()
        dplasma[k0] = v0
        size = max(size, v0.size)

    # -----------------
    # broadcastable
    # -----------------

    # shape consistency
    dout = {
        k0: v0.size for k0, v0 in dplasma.items()
        if v0.size not in [1, size]
    }
    if len(dout) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dout.items()]
        msg = (
            "All plasma parameter args must be either scalar "
            "or flat arrays of same size!\n"
            f"\t- max detected size: {size}\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # format to shape
    for k0, v0 in dplasma.items():
        dplasma[k0] = np.broadcast_to(v0, (size,))

    # --------------
    # add isotropic
    # --------------

    for k0, v0 in dplasma.items():
        if k0 == 'jp_Am2':
            v00 = 0.
        else:
            v00 = v0[0]
        dplasma[k0] = np.r_[v00, v0]

    return dplasma


def _dplasma_map(
    Te_eV=None,
    ne_m3=None,
    jp_Am2=None,
):

    # -----------------
    # initialize
    # -----------------

    dplasma = {
        'Te_eV': Te_eV,
        'ne_m3': ne_m3,
        'jp_Am2': jp_Am2,
        # 're_fraction_Ip': re_fraction_Ip,
    }

    # -----------------
    # set default + array
    # -----------------

    # default + np.ndarray
    for k0, v0 in dplasma.items():
        if v0 is None:
            v0 = _DPLASMA_ANISOTROPY_MAP[k0]
        v0 = np.atleast_1d(v0).ravel()
        dplasma[k0] = v0

    # ---------------------
    # broadcast
    # ---------------------

    dplasma = {
        'Te_eV': dplasma['Te_eV'][:, None, None],
        'ne_m3': dplasma['ne_m3'][None, :, None],
        'jp_Am2': dplasma['jp_Am2'][None, None, :],
    }

    return dplasma


# ############################################
# ############################################
#           Plot angular spectra
# ############################################


def _plot_angular_spectra(
    E_ph_eV=None,
    theta_ph_vsB=None,
    ddist=None,
    demiss=None,
    # plotting
    dax=None,
    dparam=None,
    dmargin=None,
    fs=None,
    fontsize=None,
    version_cross=None,
    # unused
    **kwdargs,
):

    # ----------------
    # inputs
    # ----------------

    dparam = _check_plot_angular_spectra(
        E_ph_eV=E_ph_eV,
        dparam=dparam,
    )

    # ----------------
    # prepare dax
    # ----------------

    if dax is None:
        dax = _get_dax_angular_spectra(
            ddist=ddist,
            demiss=demiss,
            dmargin=dmargin,
            fs=fs,
            fontsize=fontsize,
            version_cross=version_cross,
        )

    dax = ds._generic_check._check_dax(dax, main='isotropic')

    # ----------------
    # plot
    # ----------------

    # -----------
    # shapes

    vmax0 = 0
    for ii, ind in enumerate(np.ndindex(ddist['Te_eV']['data'].shape)):

        # kax
        kax0 = _get_kax(ind, ddist)
        kax = f"{kax0} - shape"

        # get ax
        if dax.get(kax) is None:
            continue
        ax = dax[kax]['handle']

        # plot - shape
        for iE, ee in enumerate(E_ph_eV):
            sli = ind[:-2] + (iE, slice(None))
            vmax = np.max(demiss['emiss']['data'][sli])
            ax.plot(
                theta_ph_vsB*180/np.pi,
                demiss['emiss']['data'][sli] / vmax,
                **dparam[iE],
            )
            vmax0 = max(vmax0, vmax)

        # lim
        if ii == 0:
            ax.set_xlim(0, 180)
            ax.set_ylim(0, 1)

        # legend
        ax.legend(title=r"$E_{ph}$ (keV)", fontsize=fontsize)

    # -----------
    # abs

    for ii, ind in enumerate(np.ndindex(ddist['Te_eV']['data'].shape)):

        # kax
        kax0 = _get_kax(ind, ddist)
        kax = f"{kax0} - abs"

        # get ax
        if dax.get(kax) is None:
            continue
        ax = dax[kax]['handle']

        # plot - abs
        for iE, ee in enumerate(E_ph_eV):
            sli = ind[:-2] + (iE, slice(None))
            ax.semilogy(
                theta_ph_vsB*180/np.pi,
                demiss['emiss']['data'][sli],
                **dparam[iE],
            )

        # lim
        if ii == 0:
            ax.set_ylim(0, vmax0)

    # -----------
    # spect

    for ii, ind in enumerate(np.ndindex(ddist['Te_eV']['data'].shape)):

        # kax
        kax0 = _get_kax(ind, ddist)
        kax = f"{kax0} - spect"

        # get ax
        if dax.get(kax) is None:
            continue
        ax = dax[kax]['handle']

        # plot - spect
        for it, tt in enumerate(theta_ph_vsB):
            sli = ind[:-2] + (slice(None), it)
            ax.semilogy(
                E_ph_eV*1e-3,
                demiss['emiss']['data'][sli],
                ls='-',
                label=f"{tt*180/np.pi:3.0f}",
            )

        ax.legend(title=r"$\theta$ (deg)", fontsize=fontsize)

    return dax


# ############################################
# ############################################
#             _check_plot
# ############################################


def _check_plot_angular_spectra(
    E_ph_eV=None,
    dparam=None,
):

    # -------------
    # dparam
    # -------------

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    dparam_def = {}
    for ii, ee in enumerate(E_ph_eV):
        dparam_def[ii] = {
            'c': colors[ii % len(colors)],
            'ls': '-',
            'lw': 1,
            'marker': 'None',
            'label': f'{ee*1e-3:3.0f}',
        }

    if dparam is None:
        dparam = dparam_def

    lok = np.arange(E_ph_eV.size)
    c0 = (
        isinstance(dparam, dict)
        and all([
            isinstance(k0, int)
            and k0 in lok
            and isinstance(v0, dict)
            for k0, v0 in dparam.items()
        ])
    )
    if not c0:
        msg = (
            "Arg dparam must be a dict with keys in range(0, E_ph_eV.size) "
            "and values must be dict with:\n"
            "\t- 'c': color-like\n"
            "\t- 'ls': ls-like\n"
            "\t- 'lw': int/float\n"
            "\t- 'marker': marker-like\n"
            f"Provided:\n{dparam}\n"
        )
        raise Exception(msg)

    # Fill with default values
    for k0, v0 in dparam.items():
        for k1, v1 in dparam_def[k0].items():
            dparam[k0][k1] = dparam[k0].get(k1, v1)

    return dparam


# ############################################
# ############################################
#             _get_dax_angular_spectra
# ############################################


def _get_dax_angular_spectra(
    ddist=None,
    demiss=None,
    dmargin=None,
    fs=None,
    fontsize=None,
    version_cross=None,
):
    # ---------------
    # check inputs
    # --------------

    # fs
    if fs is None:
        fs = (17, 10)

    fs = tuple(ds._generic_check._check_flat1darray(
        fs, 'fs',
        dtype=float,
        sign='>0',
        size=2,
    ))

    # fontsize
    fontsize = ds._generic_check._check_var(
        fontsize, 'fontsize',
        types=(int, float),
        default=12,
        sign='>0',
    )

    # dmargin
    if dmargin is None:
        dmargin = {
            'left': 0.06, 'right': 0.98,
            'bottom': 0.06, 'top': 0.90,
            'wspace': 0.20, 'hspace': 0.20,
        }

    # ---------------
    # prepare data
    # ---------------

    xlab = r"$\theta_B$ (deg)"
    ylab = r"$\epsilon$" + f"({demiss['emiss']['units']})"

    # ---------------
    # prepare figure
    # ---------------

    tit = (
        f"{version_cross} Bremsstrahlung cross-section integrated over "
        "electron distribution"
    )

    fig = plt.figure(figsize=fs)
    fig.suptitle(tit, size=fontsize+2, fontweight='bold')

    gs = gridspec.GridSpec(
        ncols=ddist['Te_eV']['data'].size,
        nrows=3,
        **dmargin,
    )
    dax = {}

    # ---------------
    # prepare axes
    # --------------

    ax0n, ax0a, ax0s = None, None, None
    for ii, ind in enumerate(np.ndindex(ddist['Te_eV']['data'].shape)):

        # kax
        kax0 = _get_kax(ind, ddist)

        # ---------------
        # create - shape

        ax = fig.add_subplot(gs[0, ii], sharex=ax0n, sharey=ax0n)
        ax.set_xlabel(
            xlab,
            fontweight='bold',
            size=fontsize,
        )
        ax.set_title(
            kax0,
            fontweight='bold',
            size=fontsize,
        )
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

        # ax0
        if ii == 0:
            ax.set_ylabel(
                "Normalized (a.u.)",
                fontweight='bold',
                size=fontsize,
            )
            ax0n = ax

        # store
        dax[f"{kax0} - shape"] = {'handle': ax}

        # ---------------
        # create - abs

        ax = fig.add_subplot(gs[1, ii], sharex=ax0n, sharey=ax0a)
        ax.set_xlabel(
            xlab,
            fontweight='bold',
            size=fontsize,
        )
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

        # ax0
        if ii == 0:
            ax.set_ylabel(
                ylab,
                fontweight='bold',
                size=fontsize,
            )
            ax0a = ax

        # store
        dax[f"{kax0} - abs"] = {'handle': ax}

        # ---------------
        # create - spect

        ax = fig.add_subplot(gs[2, ii], sharex=ax0s, sharey=ax0a)
        ax.set_xlabel(
            "E_ph (keV)",
            fontweight='bold',
            size=fontsize,
        )
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

        # ax0
        if ii == 0:
            ax.set_ylabel(
                ylab,
                fontweight='bold',
                size=fontsize,
            )
            ax0s = ax

        # store
        dax[f"{kax0} - spect"] = {'handle': ax}

    return dax


def _get_kax(ind, ddist):
    Te = ddist['Te_eV']['data'][ind]
    ne = ddist['ne_m3']['data'][ind]
    jp = ddist['jp_Am2']['data'][ind]
    integ = 100 * (ddist['dist_integ']['data'][ind[:-2]] / ne - 1.)
    vdvt = ddist['v0_par_ms']['data'][ind] / ddist['vt_ms']['data'][ind]
    kax = (
        f"Te = {Te*1e-3} keV, "
        f"ne = {ne:1.1e} /m3, "
        f"jp = {jp*1e-6} MA/m2\n"
        f"integral = {integ:3.1f} % error\n"
        + r"$v_0 / v_T = \frac{j}{en_e}\frac{m_e}{\sqrt{2k_BT_e}}$ = "
        + f"{vdvt:3.3f}"
    )
    if np.sum(ind) == 0:
        kax = f'isotropic\n{kax}'
    return kax


# ############################################
# ############################################
#           Plot anisotropy map
# ############################################


def _plot_anisotropy_map(
    E_ph_eV=None,
    theta_ph_vsB=None,
    ddist=None,
    demiss=None,
    # plotting
    dax=None,
    dparam=None,
    dmargin=None,
    fs=None,
    fontsize=None,
    version_cross=None,
    # unused
    **kwdargs,
):

    # ----------------
    # inputs
    # ----------------

    # ----------------
    # prepare data
    # ----------------

    # shape_plasma
    stream = np.squeeze(ddist['v0_par_ms']['data'] / ddist['vt_ms']['data'])

    # shape_plasma + (nE_ph,)
    anis = demiss['anis']['data']
    theta_peak = demiss['theta_peak']['data']

    Te = ddist['Te_eV']['data'].ravel()

    # ----------------
    # prepare dax
    # ----------------

    if dax is None:
        dax = _get_dax_anisotropy_map(
            demiss=demiss,
            dmargin=dmargin,
            fs=fs,
            fontsize=fontsize,
            version_cross=version_cross,
        )

    dax = ds._generic_check._check_dax(dax)

    # ----------------
    # plot
    # ----------------

    kax = 'map'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        lcolor = ['r', 'g', 'b', 'm', 'y', 'c']
        lls = ['-', '--', ':', '-.']
        for iE, ee in enumerate(E_ph_eV):

            color = lcolor[iE % len(lcolor)]
            for iT, tt in enumerate(Te):

                slip = (iT, slice(None), slice(None))
                sli = slip + (iE,)
                if iT == 0:
                    lab = f'{ee*1e-3:3.0f}'
                else:
                    lab = None
                ls = lls[iT % len(lls)]

                # indices
                iok = stream[slip].ravel() > 1e-3
                i0 = iok & (theta_peak[sli].ravel() < 5*np.pi/180)
                i1 = iok & (theta_peak[sli].ravel() > 5*np.pi/180)

                # anisotropy vs streaming parameter
                inds = np.argsort(stream[slip].ravel()[i0])
                l0, = ax.semilogx(
                    stream[slip].ravel()[i0][inds],
                    anis[sli].ravel()[i0][inds],
                    marker='.',
                    ms=12,
                    ls=ls,
                    c=color,
                    label=lab,
                )

                # paeked at > 5 degrees
                inds = np.argsort(stream[slip].ravel()[i1])
                ax.semilogx(
                    stream[slip].ravel()[i1][inds],
                    anis[sli].ravel()[i1][inds],
                    marker='s',
                    markerfacecolor='None',
                    ms=10,
                    ls=ls,
                    c=color,
                )

        # lims
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        # legend - E_ph
        leg = ax.legend(title=r"$E_{ph}$ (keV)", fontsize=fontsize, loc=2)
        ax.add_artist(leg)

        # legend - Te
        lh = [
            mlines.Line2D(
                [], [],
                c='k',
                ls=lls[iT % len(lls)],
                label=f"{tt*1e-3:3.0f}",
            )
            for iT, tt in enumerate(Te)
        ]
        ax.legend(
            handles=lh,
            title=r"$T_{e}$ (keV)",
            fontsize=fontsize,
            loc=6,
        )

    return


# ############################################
# ############################################
#       _get_dax_anisotropy_map
# ############################################


def _get_dax_anisotropy_map(
    demiss=None,
    dmargin=None,
    fs=None,
    fontsize=None,
    version_cross=None,
):
    # ---------------
    # check inputs
    # --------------

    # fs
    if fs is None:
        fs = (12, 8)

    fs = tuple(ds._generic_check._check_flat1darray(
        fs, 'fs',
        dtype=float,
        sign='>0',
        size=2,
    ))

    # fontsize
    fontsize = ds._generic_check._check_var(
        fontsize, 'fontsize',
        types=(int, float),
        default=12,
        sign='>0',
    )

    # dmargin
    if dmargin is None:
        dmargin = {
            'left': 0.06, 'right': 0.98,
            'bottom': 0.10, 'top': 0.90,
            'wspace': 0.20, 'hspace': 0.20,
        }

    # ---------------
    # prepare data
    # ---------------

    xlab = (
        r"$\xi_{Th} = \frac{v_d}{v_{Th}} $"
        r"$= \frac{j_{Th}}{en_e}\sqrt{\frac{m_e}{T_e[J]}}$"
    )
    ylab = r"$\epsilon_{max} / \epsilon_{min}$"

    # ---------------
    # prepare figure
    # ---------------

    tit = (
        f"{version_cross} Bremsstrahlung cross-section integrated over "
        "electron distribution\nAnisotropy dependency"
    )

    fig = plt.figure(figsize=fs)
    fig.suptitle(tit, size=fontsize+2, fontweight='bold')

    gs = gridspec.GridSpec(
        ncols=1,
        nrows=1,
        **dmargin,
    )
    dax = {}

    # ---------------
    # prepare axes
    # --------------

    # ---------------
    # create - map

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(
        xlab,
        fontweight='bold',
        size=fontsize,
    )
    ax.set_ylabel(
        ylab,
        fontweight='bold',
        size=fontsize,
    )
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # store
    dax["map"] = {'handle': ax}

    return dax
