

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
    'jp_fraction_re': 0.1,
}


_DPLASMA_ANGULAR_PROFILES = {
    'Te_eV': np.r_[1, 1, 10, 1]*1e3,
    'ne_m3': np.r_[1e19, 1e20, 1e19, 1e19],
    'jp_Am2': np.r_[1e6, 1e6, 1e6, 10e6],
    'jp_fraction_re': 0.,
}


_THETA_PH_VSB = np.linspace(0, np.pi, 11)    # 7
_THETA_E0_VSB_NPTS = 17                      # 19
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
    jp_fraction_re=None,
    # RE-specific
    Zeff=None,
    Ekin_max_eV=None,
    Efield_par_Vm=None,
    lnG=None,
    sigmap=None,
    Te_eV_re=None,
    ne_m3_re=None,
    dominant=None,
    # ----------------
    # cross-section
    E_ph_eV=None,
    E_e0_eV=None,
    E_e0_eV_npts=None,
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
    version_cross=None,
    # save / load
    pfe_d2cross_phi=None,
    # -----------------
    # optional responsivity
    dresponsivity=None,
    # -----------------
    # plots
    plot_angular_spectra=None,
    plot_anisotropy_map=None,
    plot_E_ph=None,
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
        dplasma, indplot,
        E_ph_eV, E_e0_eV,
        theta_ph_vsB,
        theta_e0_vsB_npts,
        version_cross,
        plot_angular_spectra,
        plot_anisotropy_map,
        verb,
    ) = _check(
        # electron distribution
        Te_eV=Te_eV,
        ne_m3=ne_m3,
        jp_Am2=jp_Am2,
        jp_fraction_re=jp_fraction_re,
        # cross-section
        E_ph_eV=E_ph_eV,
        E_e0_eV=E_e0_eV,
        theta_e0_vsB_npts=theta_e0_vsB_npts,
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
        demiss, ddist, d2cross_phi,
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
        pfe_d2cross_phi=pfe_d2cross_phi,
        # verb
        verb=verb-1,
        # optional responsivity
        dresponsivity=dresponsivity,
        # ----------------
        # electron distribution
        **dplasma,
    )

    # --------------
    # extract
    # --------------

    theta_ph_vsB = d2cross_phi['theta_ph_vsB']
    E_ph_eV = d2cross_phi['E_ph_eV']

    # -------------
    # plots
    # -------------

    dax0 = None
    if plot_angular_spectra is True:
        dax0 = _plot_angular_spectra(
            **locals(),
        )

    dax1 = None
    if plot_anisotropy_map is True:
        dax1 = _plot_anisotropy_map(
            **locals(),
        )

    return dax0, dax1, demiss, ddist, dplasma


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
    jp_fraction_re=None,
    # ----------------
    # cross-section
    E_ph_eV=None,
    E_e0_eV=None,
    theta_e0_vsB_npts=None,
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

    dplasma, indplot = _dplasma_asis(
        Te_eV=Te_eV,
        ne_m3=ne_m3,
        jp_Am2=jp_Am2,
        jp_fraction_re=jp_fraction_re,
    )
    if plot_anisotropy_map is True:
        dplasma, indplot = _dplasma_map(
            Te_eV=Te_eV,
            ne_m3=ne_m3,
            jp_Am2=jp_Am2,
            jp_fraction_re=jp_fraction_re,
            dplasma0=dplasma,
        )

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
        dplasma, indplot,
        E_ph_eV, E_e0_eV,
        theta_ph_vsB,
        theta_e0_vsB_npts,
        version_cross,
        plot_angular_spectra,
        plot_anisotropy_map,
        verb,
    )


def _dplasma_asis(
    Te_eV=None,
    ne_m3=None,
    jp_Am2=None,
    jp_fraction_re=None,
):

    # -----------------
    # initialize
    # -----------------

    dplasma = {
        'Te_eV': Te_eV,
        'ne_m3': ne_m3,
        'jp_Am2': jp_Am2,
        'jp_fraction_re': jp_fraction_re,
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

    # --------------
    # indplot
    # --------------

    indplot = np.ones(dplasma[k0].shape, dtype=bool)

    return dplasma, indplot


def _dplasma_map(
    Te_eV=None,
    ne_m3=None,
    jp_Am2=None,
    jp_fraction_re=None,
    dplasma0=None,
):

    # -----------------
    # initialize
    # -----------------

    dplasma = {
        'Te_eV': Te_eV,
        'ne_m3': ne_m3,
        'jp_Am2': jp_Am2,
        'jp_fraction_re': jp_fraction_re,
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
        'Te_eV': dplasma['Te_eV'][:, None, None, None],
        'ne_m3': dplasma['ne_m3'][None, :, None, None],
        'jp_Am2': dplasma['jp_Am2'][None, None, :, None],
        'jp_fraction_re': dplasma['jp_fraction_re'][None, None, None, :],
    }

    # --------------
    # indplot
    # --------------

    shapef = np.broadcast_shapes(*[vv.shape for vv in dplasma.values()])
    indplot = np.zeros(shapef, dtype=bool)

    for ind in np.ndindex(dplasma0['Te_eV'].shape):
        iTe = np.argmin(np.abs(dplasma['Te_eV'] - dplasma0['Te_eV'][ind]))
        ine = np.argmin(np.abs(dplasma['ne_m3'] - dplasma0['ne_m3'][ind]))
        ijp = np.argmin(np.abs(dplasma['jp_Am2'] - dplasma0['jp_Am2'][ind]))
        ijf = np.argmin(np.abs(
            dplasma['jp_fraction_re'] - dplasma0['jp_fraction_re'][ind]
        ))
        indplot[iTe, ine, ijp, ijf] = True

    return dplasma, indplot


# ############################################
# ############################################
#           Plot angular spectra
# ############################################


def _plot_angular_spectra(
    E_ph_eV=None,
    theta_ph_vsB=None,
    ddist=None,
    demiss=None,
    indplot=None,
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
            indplot=indplot,
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
    lkp = ['Te_eV', 'ne_m3', 'jp_Am2', 'jp_fraction_re']
    shapef = np.broadcast_shapes(
        *[ddist['plasma'][kk]['data'].shape for kk in lkp]
    )
    kdist = 'maxwell'
    for ii, ind in enumerate(np.ndindex(shapef)):

        if not indplot[ind]:
            continue

        # kax
        kax0 = _get_kax(ind, ddist)
        kax = f"{kax0} - shape"

        # get ax
        if dax.get(kax) is None:
            continue
        ax = dax[kax]['handle']

        # plot - shape
        for iE, ee in enumerate(E_ph_eV):
            sli = ind + (iE, slice(None))
            vmax = np.max(demiss['emiss'][kdist]['emiss']['data'][sli])
            ax.plot(
                theta_ph_vsB*180/np.pi,
                demiss['emiss'][kdist]['emiss']['data'][sli] / vmax,
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

    for ii, ind in enumerate(np.ndindex(shapef)):

        if not indplot[ind]:
            continue

        # kax
        kax0 = _get_kax(ind, ddist)
        kax = f"{kax0} - abs"

        # get ax
        if dax.get(kax) is None:
            continue
        ax = dax[kax]['handle']

        # plot - abs
        for iE, ee in enumerate(E_ph_eV):
            sli = ind + (iE, slice(None))
            ax.semilogy(
                theta_ph_vsB*180/np.pi,
                demiss['emiss'][kdist]['emiss']['data'][sli],
                **dparam[iE],
            )

        # lim
        if ii == 0:
            ax.set_ylim(0, vmax0)

    # -----------
    # spect

    for ii, ind in enumerate(np.ndindex(shapef)):

        if not indplot[ind]:
            continue

        # kax
        kax0 = _get_kax(ind, ddist)
        kax = f"{kax0} - spect"

        # get ax
        if dax.get(kax) is None:
            continue
        ax = dax[kax]['handle']

        # plot - spect
        for it, tt in enumerate(theta_ph_vsB):
            sli = ind + (slice(None), it)
            ax.semilogy(
                E_ph_eV*1e-3,
                demiss['emiss'][kdist]['emiss']['data'][sli],
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
    indplot=None,
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

    # shapef
    shapef = np.broadcast_shapes(*[
        v0['data'].shape
        for k0, v0 in ddist['plasma'].items()
        if k0 in ['Te_eV', 'ne_m3', 'jp_Am2', 'jp_fraction_re']
    ])

    # ---------------
    # prepare data
    # ---------------

    kdist = 'maxwell'
    xlab = r"$\theta_B$ (deg)"
    ylab = r"$\epsilon$" + f"({demiss['emiss'][kdist]['emiss']['units']})"

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
        ncols=indplot.sum(),
        nrows=3,
        **dmargin,
    )
    dax = {}

    # ---------------
    # prepare axes
    # --------------

    ax0n, ax0a, ax0s = None, None, None
    i0 = 0
    for ii, ind in enumerate(np.ndindex(shapef)):

        if not indplot[ind]:
            continue

        # kax
        kax0 = _get_kax(ind, ddist)

        # ---------------
        # create - shape

        ax = fig.add_subplot(gs[0, i0], sharex=ax0n, sharey=ax0n)
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

        ax = fig.add_subplot(gs[1, i0], sharex=ax0n, sharey=ax0a)
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

        ax = fig.add_subplot(gs[2, i0], sharex=ax0s, sharey=ax0a)
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

        i0 += 1

    return dax


def _get_kax(ind, ddist, kdist='maxwell'):

    if len(ind) == 4:
        indTe = (ind[0], 0, 0, 0)
        indne = (0, ind[1], 0, 0)
        indjp = (0, 0, ind[2], 0)
        indjf = (0, 0, 0, ind[3])
        indv0 = (0, ind[1], ind[2], 0, 0, 0)
        indvt = indTe + (0, 0)
        ind_int = ind
    else:
        indTe, indne, indjp, indjf = ind, ind, ind, ind
        indv0, indvt, ind_int = ind, ind, ind[0]

    Te = ddist['plasma']['Te_eV']['data'][indTe]
    ne = ddist['plasma']['ne_m3']['data'][indne]
    jp = ddist['plasma']['jp_Am2']['data'][indjp]
    jf = ddist['plasma']['jp_Am2']['data'][indjf]
    integ = 100 * (ddist['dist'][kdist]['integ_ne']['data'][ind_int] / ne - 1.)
    vdvt = (
        ddist['dist'][kdist]['v0_par_ms']['data'][indv0]
        / ddist['dist'][kdist]['vt_ms']['data'][indvt]
    )
    kax = (
        f"Te = {Te*1e-3} keV, "
        f"ne = {ne:1.1e} /m3, "
        f"jp = {jp*1e-6} MA/m2\n"
        f"jf = {jf}\n"
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
    plot_E_ph=None,
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

    kdist = 'maxwell'

    # shape_plasma
    stream = (
        ddist['dist'][kdist]['v0_par_ms']['data']
        / ddist['dist'][kdist]['vt_ms']['data']
    )

    # shape_plasma + (nE_ph,)
    kdist = 'maxwell'
    anis = demiss['emiss'][kdist]['anis']['data']
    theta_peak = demiss['emiss'][kdist]['theta_peak']['data']

    Te = np.unique(ddist['plasma']['Te_eV']['data'])

    # E_ph_eV
    if plot_E_ph is None:
        plot_E_ph = E_ph_eV

    # deco
    lcolor = ['r', 'g', 'b', 'm', 'y', 'c']
    lls = ['-', '--', ':', '-.']

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
    # plot curves
    # ----------------

    kax = 'curves'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for iE, ee in enumerate(E_ph_eV):

            color = lcolor[iE % len(lcolor)]
            for iT, tt in enumerate(Te):

                slip = (iT, slice(None), slice(None), slice(None))
                slipf = slip + (0, 0)
                sli = slip + (iE,)
                if iT == 0:
                    lab = f'{ee*1e-3:3.0f}'
                else:
                    lab = None
                ls = lls[iT % len(lls)]

                # indices
                streamf = stream[slipf].ravel()
                anisf = anis[sli].ravel()
                iok = streamf > 1e-3
                i0 = iok & (theta_peak[sli].ravel() < 5*np.pi/180)
                i1 = iok & (theta_peak[sli].ravel() > 5*np.pi/180)

                # anisotropy vs streaming parameter
                inds = np.argsort(streamf[i0])
                l0, = ax.semilogx(
                    streamf[i0][inds],
                    anisf[i0][inds],
                    marker='.',
                    ms=12,
                    ls=ls,
                    c=color,
                    label=lab,
                )

                # peaked at > 5 degrees
                inds = np.argsort(streamf[i1])
                ax.semilogx(
                    streamf[i1][inds],
                    anisf[i1][inds],
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

    # ----------------
    # plot anisotropy map
    # ----------------

    kax = 'map_anisotropy'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for iE, ee in enumerate(E_ph_eV):

            color = lcolor[iE % len(lcolor)]



    return dax


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
        ncols=3,
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
    dax["curves"] = {'handle': ax}

    # ---------------
    # create - map - anisotropy

    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlabel(
        xlab,
        fontweight='bold',
        size=fontsize,
    )
    ax.set_ylabel(
        "Te (keV)",
        fontweight='bold',
        size=fontsize,
    )
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # store
    dax["map_anisotropy"] = {'handle': ax}

    # ---------------
    # create - map - amplitude

    ax = fig.add_subplot(
        gs[0, 1],
        sharex=dax["map_anisotropy"]['handle'],
        sharey=dax["map_anisotropy"]['handle'],
    )
    ax.set_xlabel(
        xlab,
        fontweight='bold',
        size=fontsize,
    )
    ax.set_ylabel(
        'Te (keV)',
        fontweight='bold',
        size=fontsize,
    )
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # store
    dax["map_amplitude"] = {'handle': ax}

    return dax
