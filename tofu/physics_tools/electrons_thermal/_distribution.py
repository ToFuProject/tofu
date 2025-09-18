

import warnings


import numpy as np
import scipy.constants as scpct
import astropy.units as asunits
import datastock as ds
import tofu as tf


# ############################################
# ############################################
#        main
# ############################################


def get_maxwellian(
    # plasma paremeters
    Te_eV=None,
    ne_m3=None,
    jp_Am2=None,
    # coordinate: momentum
    v_perp_ms=None,
    v_par_ms=None,
    # coordinate: energy
    E_eV=None,
    pitch=None,
    # version
    version=None,
    # return as
    returnas=None,
    key=None,
):
    """ Return a thermal single or double maxwellian disctribution
    Expressed as an interpolation vs desired coordinates and units

    If 1d, it is assumed to p = p_par and p_perp = 0

    jp_Am2 is the local plasma current used to shift the Maxwellian center

    Makes use of formulas from [1]

    [1] D. Moseev and M. Salewski, Physics of Plasmas, 26, p. 020901, 2019
        doi: 10.1063/1.5085429.
    """

    # ---------------
    # check inputs
    # ---------------

    dinputs, dcoord, ref, version = _check(**locals())

    # -------------------
    # 1d Maxwellian vs p_perp and p_par
    # -------------------

    ddata = _get_maxwellian_2d(
        Te_eV=dinputs['Te_eV'],
        ne_m3=dinputs['ne_m3'],
        jp_Am2=dinputs['jp_Am2'],
        # coord
        dcoord=dcoord,
        ref=ref,
        # version
        version=version,
    )

    return ddata


# ############################################
# ############################################
#        Check
# ############################################


def _check(
    # plasma paremeters
    Te_eV=None,
    ne_m3=None,
    jp_Am2=None,
    # coordinate: momentum
    v_perp_ms=None,
    v_par_ms=None,
    # coordinate: energy
    E_eV=None,
    pitch=None,
    # version
    version=None,
    # return as
    returnas=None,
    key=None,
):

    # ---------------
    # returnas / key
    # ---------------

    if returnas is None:
        returnas = dict

    if returnas is dict:
        key = None
        coll = None
    elif isinstance(returnas, tf.data.Collection):
        coll = returnas
        lok = list(coll.ddata.keys())
        key = ds._generic_check._check_var(
            key, 'key',
            types=str,
            excluded=lok,
            extra_msg="Pick a name not already used!\n"
        )

    else:
        msg = (
            "Arg 'returnas' must be either:\n"
            "\t- dict: return outup as a dict\n"
            "\t- coll: a tf.data.Collection instance to add to / draw from\n"
            f"Provided:\n{returnas}\n"
        )
        raise Exception(msg)

    # ---------------
    # Te_eV, ne_m3, jp_Am2
    # ---------------

    dinputs = {
        'Te_eV': Te_eV,
        'ne_m3': ne_m3,
        'jp_Am2': jp_Am2,
    }

    # ------------
    # safety check

    if returnas is dict:
        lout = [
            kk for kk, vv in dinputs.items()
            if not isinstance(vv, (int, float, np.ndarray))
        ]
        if len(lout) > 0:
            lstr = [f"\t- {kk}: {dinputs[kk]}" for kk in lout]
            msg = (
                "If returnas = dict, plasma parameters must be arrays!\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

    else:
        lout = [
            kk for kk, vv in dinputs.items()
            if not (isinstance(vv, str) and vv in lok)
        ]
        if len(lout) > 0:
            lstr = [f"\t- {kk}: {dinputs[kk]}" for kk in lout]
            msg = (
                "If returnas = coll, "
                "plasma parameters must be valid data keys!\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

    # ------------
    # extract data

    if coll is not None:
        dref = _extract(dinputs, dinputs.keys(), coll)

    # ---------------
    # check data

    for k0, v0 in dinputs.items():
        dinputs[k0] = np.atleast_1d(v0)

        iok = np.isfinite(dinputs[k0])
        if 'j' not in k0:
            iok[iok] = dinputs[k0][iok] >= 0.

        if np.any(~iok):
            msg = (
                "Arg 'k0' seem to have non-finite (or negative) values!"
            )
            raise Exception(msg)

    # -------------------
    # check broadcastable
    # -------------------

    dinputs, shapef = ds._generic_check._check_all_broadcastable(
        return_full_arrays=None,
        **dinputs,
    )

    if shapef is None:
        shapef = (1,)

    # ref_inputs
    if coll is not None:
        refu = set(list(dref.values()))
        if len(refu) == 1:
            ref_inputs = list(refu)[0]
        else:
            lref = list(refu)
            nref = [len(ref) for ref in lref]
            ref_max = lref[np.argmax(nref)]
            c0 = all([
                all([rr in ref_max for rr in ref])
                for ref in lref
            ])
            if c0:
                ref_inputs = ref_max
            else:
                raise NotImplementedError()

    # ---------------
    # Coordinates
    # ---------------

    lc = [
        v_par_ms is not None and v_perp_ms is not None,
        E_eV is not None and pitch is not None,
        E_eV is not None and pitch is None,
    ]
    if np.sum(lc) != 1:
        lstr = [
            (v_par_ms, 'v_par_ms'), (v_perp_ms, 'v_perp_ms'),
            (E_eV, 'E_J'), (pitch, 'pitch'),
        ]
        lstr = [f'\t- {ss[1]}' for ss in lstr if ss[0] is not None]
        msg = (
            "For distribution coordinates, please provide either (xor):\n"
            "\t- velocities: v_par_ms and v_perp_ms\n"
            "\t- Energy-pitch: E_eV and pitch\n"
            "\t- Energy alone: E_eV\n"
            "Provided:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # ----------
    # dcoords

    if lc[0]:
        dcoords = {
            'name': '(v_par_ms, v_perp_ms)',
            'v_par_ms': v_par_ms,
            'v_perp_ms': v_perp_ms,
        }

    elif lc[1]:
        dcoords = {
            'name': '(E_eV, pitch)',
            'E_eV': E_eV,
            'pitch': pitch,
        }

    elif lc[2]:
        dcoords = {
            'name': '(E_eV,)',
            'E_eV': E_eV,
        }

    # -------------
    # extract

    lname = dcoords['name'][1:-1].split(',')
    lname = [nn.strip(' ') for nn in lname if len(nn) > 0]
    if coll is not None:
        lk = [kk for kk in dcoords.keys() if kk != 'name']
        dref = _extract(dcoords, lk, coll)
        ref_coords = tuple([dref[nn] for nn in lname])

    # -------------------
    # check broadcastable
    # -------------------

    ref = None
    if coll is not None:
        ref = ref_inputs + ref_coords

    try:
        _ = np.broadcast_arrays(
            dinputs['Te_eV'],
            dinputs['ne_m3'],
            dinputs['jp_Am2'],
            *[dcoords[nn] for nn in lname],
        )
    except Exception:
        for k0, v0 in dinputs.items():
            axis = tuple(len(shapef) + np.arange(len(lname)))
            dinputs[k0] = np.expand_dims(v0, axis)

        for ii, nn in enumerate(lname):
            axis = tuple(np.arange(len(shapef)))
            if len(lname) > 1:
                axis += (-1 - ii,)
            dcoords[nn] = np.expand_dims(dcoords[nn], axis)

    # double check
    _ = np.broadcast_arrays(
        dinputs['Te_eV'],
        dinputs['ne_m3'],
        dinputs['jp_Am2'],
        *[dcoords[nn] for nn in lname],
    )

    # ----------------
    # version
    # ----------------

    if dcoords.get('E_eV') is not None:
        lok = ['f1d_E']
        if dcoords.get('pitch') is not None:
            lok.append('f2d_E_pitch')
    else:
        lok = [
            'f3d_cart_vpar_vperp',
            'f2d_cyl_vpar_vperp',
            'f2d_cart_vpar_vperp',
        ]

    vdef = lok[-1]
    version = ds._generic_check._check_var(
        version, 'version',
        types=str,
        allowed=lok,
        default=vdef,
    )

    return dinputs, dcoords, ref, version


def _extract(din, lk, coll):

    dref = {}
    dout = {}
    for k0 in lk:
        units0 = asunits.Unit(k0.split('_')[-1])
        units1 = asunits.Unit(coll.ddata[din[k0]]['units'])
        if units0 != units1:
            dout[k0] = f"expected {units0}, got {units1}"

        dref[k0] = coll.ddata[din[k0]]['ref']
        din[k0] = coll.ddata[din[k0]]['data']

    if len(dout) > 0.:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dout.items()]
        msg = (
            "The following keys do not seem have the proper units:\n"
            + "\n".join(lstr)
        )
        warnings.warn(msg)

    return dref


# ############################################
# ############################################
#        Maxwellian - 2d
# ############################################


def _get_maxwellian_2d(
    Te_eV=None,
    ne_m3=None,
    jp_Am2=None,
    # coord
    dcoord=None,
    ref=None,
    # version
    version=None,
):

    # ---------------
    # Prepare
    # ---------------

    # electron mass
    me = scpct.m_e

    # v0_par from current  (m/s)
    v0_par_ms = np.abs(jp_Am2) / (scpct.e * ne_m3)

    # kbTe_J
    kbT_J = Te_eV * scpct.e
    vt_ms = np.sqrt(2. * kbT_J / me)

    # ---------------
    # Maxwell - non-relativistic
    # ---------------

    dist, units = _DFUNC[version]['func'](
        # coords
        v_par_ms=dcoord.get('v_par_ms'),
        v_perp_ms=dcoord.get('v_perp_ms'),
        E_eV=dcoord.get('E_eV'),
        pitch=dcoord.get('pitch'),
        # parameters
        vt_par_ms=vt_ms,
        vt_perp_ms=vt_ms,
        kbT_par_J=kbT_J,
        kbT_perp_J=kbT_J,
        v0_par_ms=v0_par_ms,
    )

    # ---------------
    # scale
    # ---------------

    dist = dist * ne_m3

    # ---------------
    # Format ouput
    # ---------------

    ddata = {
        'dist': {
            'key': None,
            'data': dist,
            'units': units,
            'ref': ref,
            'dim': 'PDF',
            'coords': dcoord['name'],
        },
    }

    return ddata


# #####################################################
# #####################################################
#           Elementary Maxwellians
# #####################################################


def f3d_cart_vpar_vperp_norm(
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
    units = asunits.Unit('s^3/m^6')
    return dist, units


def f3d_cyl_vpar_vperp_norm(
    v_par_ms=None,
    v_perp_ms=None,
    vt_par_ms=None,
    vt_perp_ms=None,
    v0_par_ms=None,
    # unused
    **kwdargs,
):
    dist0, units0 = f3d_cart_vpar_vperp_norm(
        v_par_ms,
        v_perp_ms,
        vt_par_ms,
        vt_perp_ms,
        v0_par_ms,
    )
    dist = v_perp_ms * dist0
    units = units0 * asunits.Unit('m/s')
    return dist, units


def f2d_cart_vpar_vperp_norm(
    v_par_ms=None,
    v_perp_ms=None,
    vt_par_ms=None,
    vt_perp_ms=None,
    v0_par_ms=None,
    # unused
    **kwdargs,
):
    dist0, units0 = f3d_cart_vpar_vperp_norm(
        v_par_ms,
        v_perp_ms,
        vt_par_ms,
        vt_perp_ms,
        v0_par_ms,
    )
    dist = 2. * np.pi * v_perp_ms * dist0
    units = units0 * asunits.Unit('m/s')
    return dist, units


def f2d_E_pitch_norm(
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

    dist = qq * term0 * np.exp(- term_par / kbT_par_J - term_perp / kbT_perp_J)
    units = asunits.Unit('1/eV')

    return dist, units


def f1d_E_norm(
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
        msg = "f1d_E_norm assumes kbT_par_J == kbT_perp_J"
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
        term_p = ((np.sqrt(E_J) + np.sqrt(mev2[iok, :]))**2 / kbT_par_J[iok, :])
        term_m = ((np.sqrt(E_J) - np.sqrt(mev2[iok, :]))**2 / kbT_par_J[iok, :])

        dist[iok, :] = qq * term0 * (np.exp(-term_m) - np.exp(-term_p))

    if np.any(~iok):
        i0 = ~iok
        dist[i0, :] = 2. * f2d_E_pitch_norm(
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
        'func': f3d_cart_vpar_vperp_norm,
        'latex': (
            r"$dn_e = $"
            r"$f^{2D}_{v_{//}, v_{\perp}}(v_{//}, v_{\perp}) dv_{//}dv_{\perp}$"
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
        'func': f2d_cart_vpar_vperp_norm,
        'latex': (
            r"$dn_e = $"
            r"$f^{2D}_{v_{//}, v_{\perp}}(v_{//}, v_{\perp}) dv_{//}dv_{\perp}$"
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
        'func': f3d_cyl_vpar_vperp_norm,
        'latex': (
        ),
    },
    'f2d_E_pitch': {
        'func': f2d_E_pitch_norm,
        'latex': (
            r"$dn_e = f^{2D}_{E, p}(E, p) dEdp$"
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
    'f1d_E': {
        'func': f1d_E_norm,
        'latex': (
            "Assumes " + r"$T_{\perp} = T_{//} = T$"
            + "\n" +
            r"$dn_e = f^{1D}_{E}(E) dE$"
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
