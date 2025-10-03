

import warnings


import numpy as np
import scipy.constants as scpct
import scipy.integrate as scpinteg
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
    # coordinate: velocity
    v_perp_ms=None,
    v_par_ms=None,
    # coordinate: energy
    E_eV=None,
    pitch=None,
    theta=None,
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
        Te_eV=dinputs['Te_eV']['data'],
        ne_m3=dinputs['ne_m3']['data'],
        jp_Am2=dinputs['jp_Am2']['data'],
        # coord
        dcoord=dcoord,
        ref=ref,
        # version
        version=version,
    )

    # -------------
    # add inputs & coords
    # -------------

    ddata.update(**dinputs)
    ddata.update(**dcoord)

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
    # for RE
    Zeff=None,
    electric_field_par_Vm=None,
    energy_kinetic_max_eV=None,
    # coordinate: momentum
    v_perp_ms=None,
    v_par_ms=None,
    # coordinate: energy
    E_eV=None,
    pitch=None,
    theta=None,
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
        'Te_eV': {
            'data': Te_eV,
            'units': 'eV',
        },
        'ne_m3': {
            'data': ne_m3,
            'units': '1/m^3',
        },
        'jp_Am2': {
            'data': jp_Am2,
            'units': 'A/m^2',
        },
        'Zeff': {
            'data': Zeff,
            'units': None,
        },
        'electric_field_par_Vm': {
            'data': electric_field_par_Vm,
            'units': 'V/m',
        },
        'energy_kinetic_max_eV': {
            'data': energy_kinetic_max_eV,
            'units': 'eV',
        },
    }

    # ------------
    # consistency check

    lc = [
        Zeff is None,
        electric_field_par_Vm is None,
        energy_kinetic_max_eV is None,
    ]
    if np.sum(lc) not in [0, 3]:
        lstr = ['Zeff', 'electric_field_par_Vm', 'energy_kinetic_max_eV']
        lstr = [f"\t- {kk}" for kk in lstr]
        msg = (
            "Args shall be either all None or all provided:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    dinputs = {kk: vv for kk, vv in dinputs.items() if vv['data'] is not None}

    # ------------
    # safety check

    if returnas is dict:
        lout = [
            kk for kk, vv in dinputs.items()
            if not isinstance(vv['data'], (int, float, np.ndarray))
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
        _extract(dinputs, coll)

    # ---------------
    # check data

    for k0, v0 in dinputs.items():
        dinputs[k0]['data'] = np.atleast_1d(v0['data'])

        iok = np.isfinite(dinputs[k0]['data'])
        if 'j' not in k0:
            iok[iok] = dinputs[k0]['data'][iok] >= 0.

        if np.any(~iok):
            msg = (
                "Arg 'k0' seem to have non-finite (or negative) values!"
            )
            raise Exception(msg)

    # -------------------
    # check broadcastable
    # -------------------

    _, shapef = ds._generic_check._check_all_broadcastable(
        return_full_arrays=None,
        **{kk: vv['data'] for kk, vv in dinputs.items()},
    )

    if shapef is None:
        shapef = (1,)

    # ref_inputs
    if coll is not None:
        refu = set([v0['ref'] for v0 in dinputs.values()])
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
        E_eV is not None and pitch is not None and theta is None,
        E_eV is not None and theta is not None and pitch is None,
        E_eV is not None and pitch is None and theta is None,
    ]
    if np.sum(lc) != 1:
        lstr = [
            (v_par_ms, 'v_par_ms'), (v_perp_ms, 'v_perp_ms'),
            (E_eV, 'E_J'), (pitch, 'pitch'), (theta, 'theta')
        ]
        lstr = [f'\t- {ss[1]}' for ss in lstr if ss[0] is not None]
        msg = (
            "For distribution coordinates, please provide either (xor):\n"
            "\t- velocities: v_par_ms and v_perp_ms\n"
            "\t- Energy-pitch: E_eV and pitch\n"
            "\t- Energy-theta: E_eV and theta\n"
            "\t- Energy alone: E_eV\n"
            "Provided:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # ----------
    # dcoords

    if lc[0]:
        dcoords = {
            'x0': {
                'key': 'v_par_ms',
                'data': v_par_ms,
                'units': 'm/s',
                'ref': (None,)
            },
            'x1': {
                'key': 'v_perp_ms',
                'data': v_perp_ms,
                'units': 'm/s',
                'ref': (None,)
            },
        }

    elif lc[1]:
        dcoords = {
            'x0': {
                'key': 'E_eV',
                'data': E_eV,
                'units': 'eV',
                'ref': (None,)
            },
            'x1': {
                'key': 'pitch',
                'data': pitch,
                'units': None,
                'ref': (None,)
            },
        }

    elif lc[2]:
        dcoords = {
            'x0': {
                'key': 'E_eV',
                'data': E_eV,
                'units': 'eV',
                'ref': (None,)
            },
            'x1': {
                'key': 'theta',
                'data': theta,
                'units': 'rad',
                'ref': (None,)
            },
        }

    elif lc[3]:
        dcoords = {
            'x0': {
                'key': 'E_eV',
                'data': E_eV,
                'units': 'eV',
                'ref': (None,)
            },
        }

    # -------------
    # extract

    if coll is not None:
        _extract(dcoords, coll)

    lcoords = sorted(dcoords.keys())
    ref_coords = tuple([dcoords[k0]['ref'] for k0 in lcoords])
    shapef_coords = tuple([dcoords[k0]['data'].size for k0 in lcoords])

    # -------------------
    # check coords 1d

    dout = {
        v0['key']: v0['data'].shape
        for k0, v0 in dcoords.items()
        if v0['data'].ndim != 1
    }
    if len(dout) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dout.items()]
        msg = (
            "Args must be flat 1d arrays!\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # -------------------
    # make all broadcastable
    # -------------------

    ref = None
    if coll is not None:
        ref = ref_inputs + ref_coords

    for k0, v0 in dinputs.items():
        axis = tuple(len(shapef) + np.arange(len(shapef_coords)))
        dinputs[k0]['data'] = np.expand_dims(v0['data'], axis)

    for ii, nn in enumerate(lcoords):
        axis = tuple(np.arange(len(shapef)))
        if len(lcoords) > 1:
            axis_add = np.delete(np.arange(len(lcoords)) + 1, ii)
            axis += tuple(axis[-1] + axis_add)
        dcoords[nn]['data'] = np.expand_dims(dcoords[nn]['data'], axis)

    # double check
    _ = np.broadcast_arrays(
        dinputs['Te_eV']['data'],
        dinputs['ne_m3']['data'],
        dinputs['jp_Am2']['data'],
        *[dcoords[nn]['data'] for nn in lcoords],
    )

    # ----------------
    # version
    # ----------------

    if dcoords['x0']['key'] == 'E_eV':
        lok = ['f1d_E']
        if dcoords.get('x1', {}).get('key') == 'pitch':
            lok.append('f2d_E_pitch')
        if dcoords.get('x1', {}).get('key') == 'theta':
            lok += ['f2d_E_theta', 'f3d_E_theta']
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


def _extract(din, coll):

    dout = {}
    for k0, v0 in din.items():
        if isinstance(v0['data'], str):
            units0 = asunits.Unit(v0['units'])
            units1 = asunits.Unit(coll.ddata[din[k0]['data']]['units'])
            if units0 != units1:
                dout[v0.get('key', k0)] = f"expected {units0}, got {units1}"

            din[k0]['data'] = coll.ddata[din[k0]['data']]['data']

    if len(dout) > 0.:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dout.items()]
        msg = (
            "The following keys do not seem have the proper units:\n"
            + "\n".join(lstr)
        )
        warnings.warn(msg)

    return


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
        v_par_ms=dcoord['x0']['data'],
        v_perp_ms=dcoord.get('x1', {}).get('data'),
        E_eV=dcoord['x0']['data'],
        pitch=dcoord.get('x1', {}).get('data'),
        theta=dcoord.get('x1', {}).get('data'),
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

    # add density
    dist = dist * ne_m3
    units = units * asunits.Unit('1/m3')

    # ---------------
    # integrate to compare vs ne
    # ---------------

    integ, units_integ, ref_integ = _integrate(
        dist=dist,
        units=units,
        dcoord=dcoord,
        ref=ref,
        version=version,
    )

    # ---------------
    # Format ouput
    # ---------------

    ddata = {
        'dist': {
            'key': None,
            'type': 'maxwell',
            'data': dist,
            'units': units,
            'ref': ref,
            'dim': 'PDF',
        },
        'dist_integ': {
            'key': None,
            'type': 'maxwell',
            'data': integ,
            'units': units_integ,
            'ref': ref_integ,
            'dim': None,
        },
        'v0_par_ms': {
            'key': None,
            'data': v0_par_ms,
            'units': 'm/s',
            'ref': None,
            'dim': 'velocity',
        },
        'vt_ms': {
            'key': None,
            'data': vt_ms,
            'units': 'm/s',
            'ref': None,
            'dim': 'velocity',
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
    units = asunits.Unit('s^3/m^3')
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

    dist = qq * term0 * np.exp(-term_par / kbT_par_J - term_perp / kbT_perp_J)
    units = asunits.Unit('1/eV')

    return dist, units


def f3d_E_theta_norm(
    E_eV=None,
    theta=None,
    kbT_par_J=None,
    kbT_perp_J=None,
    v0_par_ms=None,
    # unused
    **kwdargs,
):

    dist0, units0 = f2d_E_pitch_norm(
        E_eV=E_eV,
        pitch=np.cos(theta),
        kbT_par_J=kbT_par_J,
        kbT_perp_J=kbT_perp_J,
        v0_par_ms=v0_par_ms,
    )

    dist = np.sin(theta) * dist0 / (2.*np.pi)
    units = units0 * asunits.Unit('1/rad^2')

    return dist, units


def f2d_E_theta_norm(
    E_eV=None,
    theta=None,
    kbT_par_J=None,
    kbT_perp_J=None,
    v0_par_ms=None,
    # unused
    **kwdargs,
):

    dist0, units0 = f2d_E_pitch_norm(
        E_eV=E_eV,
        pitch=np.cos(theta),
        kbT_par_J=kbT_par_J,
        kbT_perp_J=kbT_perp_J,
        v0_par_ms=v0_par_ms,
    )

    dist = np.sin(theta) * dist0
    units = units0 * asunits.Unit('1/rad')

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
        term_p = (np.sqrt(E_J) + np.sqrt(mev2[iok, :]))**2 / kbT_par_J[iok, :]
        term_m = (np.sqrt(E_J) - np.sqrt(mev2[iok, :]))**2 / kbT_par_J[iok, :]

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
#           Integrate numerically
# #####################################################


def _integrate(
    dist=None,
    units=None,
    dcoord=None,
    ref=None,
    version=None,
):

    # ---------
    # integrate
    # ---------

    # integrate over x1
    if dcoord.get('x1') is None:
        integ = dist
        x0 = dcoord['x0']['data']
    else:
        integ = scpinteg.trapezoid(
            dist,
            x=dcoord['x1']['data'],
            axis=-1,
        )
        x0 = dcoord['x0']['data'][..., 0]

    # integrate over x0
    integ = scpinteg.trapezoid(
        integ,
        x=x0,
        axis=-1,
    )

    # adjust if needed
    if version == 'f3d_E_theta':
        integ = integ * (2.*np.pi)

    # ---------
    # ref
    # ---------

    if ref is None:
        ref_integ = None
    else:
        ref_integ = ref[:-2]

    # ---------
    # units
    # ---------

    units_integ = units
    for k0, v0 in dcoord.items():
        if v0['units'] not in ['', None]:
            units_integ = units_integ * asunits.Unit(v0['units'])

    # adjust of needed
    if version == 'f3d_E_theta':
        units_integ = units_integ * asunits.Unit('rad')

    return integ, units_integ, ref_integ


# #####################################################
# #####################################################
#           Dict of functions
# #####################################################


_DFUNC = {
    'f3d_cart_vpar_vperp': {
        'func': f3d_cart_vpar_vperp_norm,
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
        'func': f2d_cart_vpar_vperp_norm,
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
        'func': f3d_cyl_vpar_vperp_norm,
        'latex': (
        ),
    },
    'f2d_E_pitch': {
        'func': f2d_E_pitch_norm,
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
        'func': f3d_E_theta_norm,
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
        'func': f2d_E_theta_norm,
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
        'func': f1d_E_norm,
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
