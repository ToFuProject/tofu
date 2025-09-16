

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

    dinputs, dcoord, ref = _check(**locals())

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

    else:
        dcoords = {
            'name': '(E_eV, pitch)',
            'E_eV': E_eV,
            'pitch': pitch,
        }

    # -------------
    # extract

    lname = dcoords['name'].strip('(').strip(')').split(', ')
    if coll is not None:
        lk = [kk for kk in dcoords.keys() if kk != 'name']
        dref = _extract(dcoords, lk, coll)
        ref_coords = dref[lname[0]] + dref[lname[1]]

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
            dcoords[lname[0]],
            dcoords[lname[1]],
        )
    except Exception:
        for k0, v0 in dinputs.items():
            dinputs[k0] = v0[..., None, None]

        for ii, nn in enumerate(lname):
            sh = (1,)*len(shapef) + ((-1, 1) if ii == 0 else (1, -1))
            dcoords[nn] = dcoords[nn].reshape(sh)

    # double check
    _ = np.broadcast_arrays(
        dinputs['Te_eV'],
        dinputs['ne_m3'],
        dinputs['jp_Am2'],
        dcoords[lname[0]],
        dcoords[lname[1]],
    )

    return dinputs, dcoords, ref


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

    if dcoord['name'] == '(v_par_ms, v_perp_ms)':
        dist, units = f2d_cart_vperp_vpar_norm(
            v_par_ms=dcoord['v_par_ms'],
            v_perp_ms=dcoord['v_perp_ms'],
            vt_par_ms=vt_ms,
            vt_perp_ms=vt_ms,
            v0_par_ms=v0_par_ms,
        )

    else:
        dist, units = f2d_E_pitch_norm(
            E_eV=dcoord['E_eV'],
            pitch=dcoord['pitch'],
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


def f3d_cart_vperp_vpar_norm(
    v_par_ms=None,
    v_perp_ms=None,
    vt_par_ms=None,
    vt_perp_ms=None,
    v0_par_ms=None,
):
    term0 = 1. / (np.pi**1.5 * vt_par_ms * vt_perp_ms**2)
    term_par = (v_par_ms - v0_par_ms)**2 / vt_par_ms**2
    term_perp = v_perp_ms**2 / vt_perp_ms**2

    dist = term0 * np.exp(- term_par - term_perp)
    units = asunits.Unit('s^3/m^6')
    return dist, units


def f3d_cyl_vperp_vpar_norm(
    v_par_ms=None,
    v_perp_ms=None,
    vt_par_ms=None,
    vt_perp_ms=None,
    v0_par_ms=None,
):
    dist0, units0 = f3d_cart_vperp_vpar_norm(
        v_par_ms,
        v_perp_ms,
        vt_par_ms,
        vt_perp_ms,
        v0_par_ms,
    )
    dist = v_perp_ms * dist0
    units = units0 * asunits.Unit('m/s')
    return dist, units


def f2d_cart_vperp_vpar_norm(
    v_par_ms=None,
    v_perp_ms=None,
    vt_par_ms=None,
    vt_perp_ms=None,
    v0_par_ms=None,
):
    dist0, units0 = f3d_cart_vperp_vpar_norm(
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
