

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


from . import _xray_thin_target_integrated_dist


# ############################################
# ############################################
#             Default
# ############################################


# _DPLASMA = {
    # 'Te_eV': np.r_[1, 1, 5, 1]*1e3,
    # 'ne_m3': np.r_[1e19, 1e20, 1e19, 1e19],
    # 'jp_Am2': np.r_[1e6, 1e6, 1e6, 10e6],
    # 're_fraction_Ip': 0.,
# }
_DPLASMA = {
    'Te_eV': np.r_[1., 5.]*1e3,
    'ne_m3': np.r_[1e19, 1e19],
    'jp_Am2': np.r_[1e6, 1e6],
    're_fraction_Ip': 0.,
}


_THETA_PH_VSB = np.linspace(0, np.pi, 3)    # 7
_THETA_E0_VSB_NPTS = 13                     # 19
_PHI_E0_VSB_NPTS = 21                       # 29
_E_PH_EV = np.r_[5., 15., 30.] * 1e3
_E_E0_EV = np.linspace(_E_PH_EV.min()*1e-3 + 1e-9, 500, 31)*1e3      # 31


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
    # plot
    # -------------

    dax = _plot(
        **locals(),
    )

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
    verb=None,
):

    # -------------------------------------
    # Te_eV, ne_m3, jp_Am2, re_fraction_Ip
    # -------------------------------------

    dplasma = {
        'Te_eV': Te_eV,
        'ne_m3': ne_m3,
        'jp_Am2': jp_Am2,
        're_fraction_Ip': re_fraction_Ip,
    }

    # default + np.ndarray
    size = 1
    for k0, v0 in dplasma.items():
        if v0 is None:
            v0 = _DPLASMA[k0]
        v0 = np.atleast_1d(v0).ravel()
        dplasma[k0] = v0
        size = max(size, v0.size)

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
        verb,
    )


# ############################################
# ############################################
#             Check
# ############################################


def _plot(
    E_ph_eV=None,
    theta_ph_vsB=None,
    dplasma=None,
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

    dparam = _check_plot(
        E_ph_eV=E_ph_eV,
        dparam=dparam,
    )

    # ----------------
    # prepare dax
    # ----------------

    if dax is None:
        dax = _get_dax(
            dplasma=dplasma,
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
    for ii, ind in enumerate(np.ndindex(dplasma['Te_eV'].shape)):

        # kax
        kax0 = _get_kax(ii, dplasma)
        kax = f"{kax0} - shape"

        # get ax
        if dax.get(kax) is None:
            continue
        ax = dax[kax]['handle']

        # plot - shape
        for iE, ee in enumerate(E_ph_eV):
            sli = ind + (iE, slice(None))
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
        ax.legend()

    # -----------
    # abs

    for ii, ind in enumerate(np.ndindex(dplasma['Te_eV'].shape)):

        # kax
        kax0 = _get_kax(ii, dplasma)
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
                demiss['emiss']['data'][sli],
                **dparam[iE],
            )

        # lim
        if ii == 0:
            ax.set_ylim(0, vmax0)

        # legend
        ax.legend()

    return dax


# ############################################
# ############################################
#             _check_plot
# ############################################


def _check_plot(
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
            'label': f'E_ph = {ee*1e-3:3.1f} keV',
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
#             _get_dax
# ############################################


def _get_dax(
    dplasma=None,
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
        fs = (16, 10)

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
        f"{version_cross} Bremsstrhlung cross-section integrated over "
        "electron distribution"
    )

    fig = plt.figure(figsize=fs)
    fig.suptitle(tit, size=fontsize+2, fontweight='bold')

    gs = gridspec.GridSpec(ncols=len(dplasma['Te_eV']), nrows=2, **dmargin)
    dax = {}

    # ---------------
    # prepare axes
    # --------------

    ax0s, ax0a = None, None
    for ii in range(len(dplasma['Te_eV'])):

        # kax
        kax0 = _get_kax(ii, dplasma)

        # ---------------
        # create - shape

        ax = fig.add_subplot(gs[0, ii], sharex=ax0s, sharey=ax0s)
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

        # ax0
        if ii == 0:
            ax.set_ylabel(
                ylab,
                fontweight='bold',
                size=fontsize,
            )
            ax0s = ax

        # store
        dax[f"{kax0} - shape"] = {'handle': ax}

        # ---------------
        # create - abs

        ax = fig.add_subplot(gs[1, ii], sharex=ax0s, sharey=ax0a)
        ax.set_xlabel(
            xlab,
            fontweight='bold',
            size=fontsize,
        )

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

    return dax


def _get_kax(ii, dplasma):
    Te = dplasma['Te_eV'][ii]
    ne = dplasma['ne_m3'][ii]
    jp = dplasma['jp_Am2'][ii]
    kax = (
        f"Te = {Te*1e-3} keV, "
        f"ne = {ne:1.1e} /m3, "
        f"jp = {jp*1e-6} MA/m2"
    )
    if ii == 0:
        kax = f'isotropic\n{kax}'
    return kax
