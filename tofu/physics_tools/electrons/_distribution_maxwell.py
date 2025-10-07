

import numpy as np
import scipy.constants as scpct
import astropy.units as asunits


# #####################################################
# #####################################################
#           Main
# #####################################################


def main(
    # coordinates
    dcoords=None,
    version=None,
    # plasma
    dplasma=None,
    # unused
    **kwdargs,
):

    # --------------
    # prepare
    # --------------

    # electron mass
    me = scpct.m_e

    # kbTe_J
    kbT_J = dplasma['Te_eV']['data'] * scpct.e

    # v0_par from current  (m/s)
    v0_par_ms = (
        dplasma['jp_Am2']['data']
        / (scpct.e * dplasma['ne_m3']['data'])
    )
    vt_ms = np.sqrt(2. * kbT_J / me)

    # --------------
    # format output
    # --------------

    dist, units = eval(version)(
        vt_par_ms=vt_ms,
        vt_perp_ms=vt_ms,
        v0_par_ms=v0_par_ms,
        kbT_par_J=kbT_J,
        kbT_perp_J=kbT_J,
        **dcoords,
    )

    # --------------
    # format output
    # --------------

    dout = {
        'dist': {
            'data': dist,
            'units': units,
        },
        'v0_par_ms': {
            'data': v0_par_ms,
            'units': 'm/s',
        },
        'vt_ms': {
            'data': vt_ms,
            'units': 'm/s',
        },
        'kbT_J': {
            'data': kbT_J,
            'units': 'J',
        },
    }

    return dout


# #####################################################
# #####################################################
#           Elementary Maxwellians
# #####################################################


def f3d_cart_vpar_vperp(
    v_par_ms=None,
    v_perp_ms=None,
    vt_par_ms=None,
    vt_perp_ms=None,
    v0_par_ms=None,
    # unused
    **kwdargs,
):
    term0 = 1. / (np.pi**1.5 * vt_par_ms * vt_perp_ms**2)
    term_par = (v_par_ms - v0_par_ms)**2 / vt_par_ms**2
    term_perp = v_perp_ms**2 / vt_perp_ms**2

    dist = term0 * np.exp(- term_par - term_perp)
    units = asunits.Unit('s^3/m^3')
    return dist, units


def f3d_cyl_vpar_vperp(
    v_par_ms=None,
    v_perp_ms=None,
    vt_par_ms=None,
    vt_perp_ms=None,
    v0_par_ms=None,
    # unused
    **kwdargs,
):
    dist0, units0 = f3d_cart_vpar_vperp(
        v_par_ms,
        v_perp_ms,
        vt_par_ms,
        vt_perp_ms,
        v0_par_ms,
    )
    dist = v_perp_ms * dist0
    units = units0 * asunits.Unit('m/s')
    return dist, units


def f2d_cart_vpar_vperp(
    v_par_ms=None,
    v_perp_ms=None,
    vt_par_ms=None,
    vt_perp_ms=None,
    v0_par_ms=None,
    # unused
    **kwdargs,
):
    dist0, units0 = f3d_cart_vpar_vperp(
        v_par_ms,
        v_perp_ms,
        vt_par_ms,
        vt_perp_ms,
        v0_par_ms,
    )
    dist = 2. * np.pi * v_perp_ms * dist0
    units = units0 * asunits.Unit('m/s')
    return dist, units


def f2d_ppar_pperp(
    p_par_norm=None,
    p_perp_norm=None,
    vt_par_ms=None,
    vt_perp_ms=None,
    v0_par_ms=None,
    # unused
    **kwdargs,
):
    """ Integral not unit => problem somewhere !"""

    dist0, units0 = f2d_cart_vpar_vperp(
        v_par_ms=p_par_norm * scpct.c,
        v_perp_ms=p_perp_norm * scpct.c,
        vt_par_ms=vt_par_ms,
        vt_perp_ms=vt_perp_ms,
        v0_par_ms=v0_par_ms,
    )

    dist = dist0 * scpct.c**2
    units = units0 * asunits.Unit('m^2/s^2')

    return dist, units


def f2d_E_pitch(
    E_eV=None,
    pitch=None,
    kbT_par_J=None,
    kbT_perp_J=None,
    v0_par_ms=None,
    # unused
    **kwdargs,
):
    me_kg = scpct.m_e
    qq = scpct.e
    E_J = E_eV * qq

    term0 = np.sqrt(E_J / (np.pi * kbT_par_J * kbT_perp_J**2))
    term_par = (pitch * np.sqrt(E_J) - np.sqrt(me_kg/2.) * v0_par_ms)**2
    term_perp = (1 - pitch**2) * E_J

    dist = qq * term0 * np.exp(-term_par / kbT_par_J - term_perp / kbT_perp_J)
    units = asunits.Unit('1/eV')

    return dist, units


def f3d_E_theta(
    E_eV=None,
    theta=None,
    kbT_par_J=None,
    kbT_perp_J=None,
    v0_par_ms=None,
    # unused
    **kwdargs,
):

    dist0, units0 = f2d_E_pitch(
        E_eV=E_eV,
        pitch=np.cos(theta),
        kbT_par_J=kbT_par_J,
        kbT_perp_J=kbT_perp_J,
        v0_par_ms=v0_par_ms,
    )

    dist = np.sin(theta) * dist0 / (2.*np.pi)
    units = units0 * asunits.Unit('1/rad^2')

    return dist, units


def f2d_E_theta(
    E_eV=None,
    theta=None,
    kbT_par_J=None,
    kbT_perp_J=None,
    v0_par_ms=None,
    # unused
    **kwdargs,
):

    dist0, units0 = f2d_E_pitch(
        E_eV=E_eV,
        pitch=np.cos(theta),
        kbT_par_J=kbT_par_J,
        kbT_perp_J=kbT_perp_J,
        v0_par_ms=v0_par_ms,
    )

    dist = np.sin(theta) * dist0
    units = units0 * asunits.Unit('1/rad')

    return dist, units


def f1d_E(
    E_eV=None,
    kbT_par_J=None,
    kbT_perp_J=None,
    v0_par_ms=None,
    # unused
    **kwdargs,
):

    # ------------
    # safety check
    if not np.allclose(kbT_par_J, kbT_perp_J):
        msg = "f1d_E assumes kbT_par_J == kbT_perp_J"
        raise Exception(msg)

    me_kg = scpct.m_e
    mev2 = 0.5*me_kg*v0_par_ms**2
    qq = scpct.e
    E_J = E_eV * qq

    iok = v0_par_ms[..., 0] > 0.
    shapef = np.broadcast_shapes(kbT_par_J.shape, v0_par_ms.shape, E_J.shape)
    dist = np.full(shapef, np.nan)

    if np.any(iok):
        denom = (2. * np.pi * kbT_par_J[iok, :] * me_kg)
        term0 = 1. / (v0_par_ms[iok, :] * np.sqrt(denom))
        term_p = (np.sqrt(E_J) + np.sqrt(mev2[iok, :]))**2 / kbT_par_J[iok, :]
        term_m = (np.sqrt(E_J) - np.sqrt(mev2[iok, :]))**2 / kbT_par_J[iok, :]

        dist[iok, :] = qq * term0 * (np.exp(-term_m) - np.exp(-term_p))

    if np.any(~iok):
        i0 = ~iok
        dist[i0, :] = 2. * f2d_E_pitch(
            E_eV=E_eV,
            pitch=0.,
            kbT_par_J=kbT_par_J[i0, :],
            kbT_perp_J=kbT_perp_J[i0, :],
            v0_par_ms=0.,
        )[0]

    units = asunits.Unit('1/eV')

    return dist, units


# #####################################################
# #####################################################
#           Dict of functions
# #####################################################


_DFUNC = {
    'f3d_cart_vpar_vperp': {
        'func': f3d_cart_vpar_vperp,
        'latex': (
            r"$dn_e = \int_0^\infty \int_{-\infty}^\infty$"
            r"$f^{2D}_{v_{//}, v_{\perp}}(v_{//}, v_{\perp})$"
            r"$dv_{//}dv_{\perp}$"
            + "\n" +
            r"\begin{eqnarray*}"
            r"\frac{n_e}{\pi^{3/2} v_{T//} v^2_{T\perp}}"
            r"\exp\left("
            r"-\frac{\left(v_{//} - v_{d//}\right)^2}{v^2_{T//}}"
            r"-\frac{v^2_{\perp}}{v^2_{T\perp}}"
            r"\right)"
            r"\end{eqnarray*}"
        ),
    },
    'f2d_cart_vpar_vperp': {
        'func': f2d_cart_vpar_vperp,
        'latex': (
            r"$dn_e = \int_0^\infty \int_{-\infty}^\infty$"
            r"$f^{2D}_{v_{//}, v_{\perp}}(v_{//}, v_{\perp})$"
            r"$dv_{//}dv_{\perp}$"
            + "\n" +
            r"\begin{eqnarray*}"
            r"\frac{2n_e v_{\perp}}{\sqrt{\pi} v_{T//} v^2_{T\perp}}"
            r"\exp\left("
            r"-\frac{\left(v_{//} - v_{d//}\right)^2}{v^2_{T//}}"
            r"-\frac{v^2_{\perp}}{v^2_{T\perp}}"
            r"\right)"
            r"\end{eqnarray*}"
        ),
    },
    'f3d_cyl_vpar_vperp': {
        'func': f3d_cyl_vpar_vperp,
        'latex': (
        ),
    },
    'f2d_E_pitch': {
        'func': f2d_E_pitch,
        'latex': (
            r"$dn_e = \int_0^{\infty} \int_{-1}^1$"
            r"$f^{2D}_{E, p}(E, p) dEdp$"
            + "\n" +
            r"\begin{eqnarray*}"
            r"n_e \sqrt{\frac{E}{\pi T^2_{\perp}T_{//}}}"
            r"\exp\left("
            r"-\frac{\left(p\sqrt{E} - \sqrt{m_e/2}v_{d//}\right)^2}{T_{//}}"
            r"- \frac{(1-p^2)E}{T_{\perp}}"
            r"\right)"
            r"\end{eqnarray*}"
        ),
    },
    'f3d_E_theta': {
        'func': f3d_E_theta,
        'latex': (
            r"$dn_e = \int_0^\infty \int_0\pi \int_0^{2\pi}$"
            r"$f^{3D}_{E, \theta}(E, \theta) dEd\thetad\phi$"
            + "\n" +
            r"\begin{eqnarray*}"
            r"\frac{n_e}{2\pi}"
            r"\sin{\theta}\sqrt{\frac{E}{\pi T^2_{\perp}T_{//}}}"
            r"\exp\left("
            r"-\frac{\left(p\sqrt{E} - \sqrt{m_e/2}v_{d//}\right)^2}{T_{//}}"
            r"- \frac{(1-p^2)E}{T_{\perp}}"
            r"\right)"
            r"\end{eqnarray*}"
        ),
    },
    'f2d_E_theta': {
        'func': f2d_E_theta,
        'latex': (
            r"$dn_e = \int_0^\infty \int_0\pi$"
            r"$f^{2D}_{E, \theta}(E, \theta) dEd\theta$"
            + "\n" +
            r"\begin{eqnarray*}"
            r"n_e \sin{\theta}\sqrt{\frac{E}{\pi T^2_{\perp}T_{//}}}"
            r"\exp\left("
            r"-\frac{\left(p\sqrt{E} - \sqrt{m_e/2}v_{d//}\right)^2}{T_{//}}"
            r"- \frac{(1-p^2)E}{T_{\perp}}"
            r"\right)"
            r"\end{eqnarray*}"
        ),
    },
    'f1d_E': {
        'func': f1d_E,
        'latex': (
            "Assumes " + r"$T_{\perp} = T_{//} = T$"
            + "\n" +
            r"$dn_e = \int_0^\infty f^{1D}_{E}(E) dE$"
            + "\n" +
            r"\begin{eqnarray*}"
            r"\frac{n_e}{v_{d//}\sqrt{\pi T 2m_e}}"
            r"\left("
            r"  \exp\left("
            r"    \frac{\left(\sqrt{E} - \sqrt{m_e v_{d//}^2/2}\right)^2}{T}"
            r"  \right)"
            r"  - \exp\left("
            r"    \frac{\left(\sqrt{E} + \sqrt{m_e v_{d//}^2/2}\right)^2}{T}"
            r"  \right)"
            r"\right)"
            r"\end{eqnarray*}"
        ),
    },
}
