

import numpy as np
import scipy.constants as scpct
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


from ._distribution import get_maxwellian


plt.rcParams['text.usetex'] = True


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
    E_max_eV=None,
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
        E_max_eV,
        dprop,
    ) = _check(
        # plasma paremeters
        Te_eV=Te_eV,
        ne_m3=ne_m3,
        jp_Am2=jp_Am2,
        # coordinates
        E_max_eV=E_max_eV,
    )

    # ----------------
    # Prepare data
    # ----------------

    # E_eV, pitch
    E_eV = np.linspace(0, E_max_eV, 1000)
    pitch = np.linspace(-1, 1, 100)

    # Velocity
    me = scpct.m_e
    v_max_ms = np.sqrt(2.*E_max_eV * scpct.e / me)
    v_par_ms = v_max_ms * np.linspace(-1, 1, 1000)
    v_perp_ms = v_max_ms * np.linspace(0, 1, 100)

    # ----------------
    # Compute
    # ----------------

    dout_v = get_maxwellian(
        # plasma paremeters
        Te_eV=Te_eV,
        ne_m3=ne_m3,
        jp_Am2=jp_Am2,
        # coordinate: momentum
        v_perp_ms=v_perp_ms,
        v_par_ms=v_par_ms,
        # return as
        returnas=dict,
    )

    dout_E = get_maxwellian(
        # plasma paremeters
        Te_eV=Te_eV,
        ne_m3=ne_m3,
        jp_Am2=jp_Am2,
        # coordinate: energy
        E_eV=E_eV,
        pitch=pitch,
        # return as
        returnas=dict,
    )

    # ----------------
    # plot
    # ----------------

    dax = _plot(
        E_eV=E_eV,
        pitch=pitch,
        v_par_ms=v_par_ms,
        v_perp_ms=v_perp_ms,
        # parameters
        E_max_eV=E_max_eV,
        v_max_ms=v_max_ms,
        # distribution
        dout_E=dout_E,
        dout_v=dout_v,
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
    E_max_eV=None,
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
    # E_max_eV
    # -----------------

    E_max_eV = ds._generic_check._check_var(
        E_max_eV, 'E_max_eV',
        types=(float, int),
        sign='>0',
        default=50e3,
    )

    return (
        Te_eV, ne_m3, jp_Am2,
        E_max_eV,
        dprop,
    )


# #####################################################
# #####################################################
#               plot
# #####################################################


def _plot(
    E_eV=None,
    pitch=None,
    v_par_ms=None,
    v_perp_ms=None,
    # parameters
    E_max_eV=None,
    v_max_ms=None,
    # distribution
    dout_E=None,
    dout_v=None,
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
            im = ax.contour(
                E_eV*1e-3,
                pitch,
                np.log10(dout_E['dist']['data'][sli]).T,
                levels=10,  # dout_E['dist']['levels'][ind],
                colors=dprop[ind]['color'],
            )

            plt.clabel(im, fmt=lambda vv: f"{10**vv:3.2e}", fontsize=fontsize)

        ax.set_xlim(0, E_max_eV*1e-3)
        ax.set_ylim(-1, 1)

    # ----------------
    # plot vs velocities
    # ----------------

    kax = '(v_par, v_perp) - map'
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

        ax.set_xlim(-v_max_ms, v_max_ms)
        ax.set_ylim(0, v_max_ms)

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
        default=14,
        sign='>0',
    )

    # --------------
    # prepare data
    # --------------

    str_fE = (
        r"\begin{eqnarray*}"
        r"dn_e = f^{2D}_{E, p}(E, p) dEdp\\"
        r"n_e \sqrt{\frac{E}{\pi T^2_{\perp}T_{//}}}"
        r"\exp\left("
        r"-\frac{\left(p\sqrt{E} - \sqrt{m_e/2}v_{d//}\right)^2}{T_{//}}"
        r"- \frac{(1-p^2)E}{T_{\perp}}"
        r"\right)"
        r"\end{eqnarray*}"
    )

    str_fv = (
        r"\begin{eqnarray*}[c]"
        r"dn_e = f^{2D}_{v_{//}, v_{\perp}}(v_{//}, v_{\perp}) dv_{//}dv_{\perp}\\"
        r"\frac{2n_e v_{\perp}}{\sqrt{\pi} v_{T//} v^2_{T\perp}}"
        r"\exp\left("
        r"-\frac{\left(v_{//} - v_{d//}\right)^2}{v^2_{T//}}"
        r"-\frac{v^2_{\perp}}{v^2_{T\perp}}"
        r"\right)"
        r"\end{eqnarray*}"
    )

    str_fv3D = (
        r"\begin{eqnarray*}[c]"
        r"dn_e = f^{2D}_{v_{//}, v_{\perp}}(v_{//}, v_{\perp}) dv_{//}dv_{\perp}\\"
        r"\frac{2n_e v_{\perp}}{\sqrt{\pi} v_{T//} v^2_{T\perp}}"
        r"\exp\left("
        r"-\frac{\left(v_{//} - v_{d//}\right)^2}{v^2_{T//}}"
        r"-\frac{v^2_{\perp}}{v^2_{T\perp}}"
        r"\right)"
        r"\end{eqnarray*}"
    )

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
            'bottom': 0.06, 'top': 0.80,
            'wspace': 0.2, 'hspace': 0.30,
        }

    fig = plt.figure(figsize=(15, 12))
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
        str_fE,
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['(E, pitch) - map'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # (v_par, v_perp) - map

    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlabel(
        "v_par (m/s)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        "v_perp (m/s)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        str_fv,
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['(v_par, v_perp) - map'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # (v_par, v_perp)3D - map

    ax = fig.add_subplot(gs[0, 2])
    ax.set_xlabel(
        "v_par (m/s)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        "v_perp (m/s)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        str_fv3D,
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['(v_par, v_perp)3D - map'] = {'handle': ax, 'type': 'isolines'}

    return dax
