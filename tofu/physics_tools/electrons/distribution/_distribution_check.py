

import numpy as np
import astropy.units as asunits
import datastock as ds
import tofu as tf


from . import _distribution_maxwell as _maxwell
from . import _distribution_re as _re


# #######################################################
# #######################################################
#                   DEFAULTS
# #######################################################


_DPLASMA = {
    'Te_eV': {
        'def': 1e3,
        'units': 'eV',
    },
    'ne_m3': {
        'def': 1e19,
        'units': '1/m3',
    },
    'jp_Am2': {
        'def': 1e6,
        'units': 'A/m2',
    },
    # RE
    'jp_fraction_re': {
        'def': 0.,
        'units': 'A/m2',
    },
    'Te_eV_re': {
        'def': 0.,
        'units': 'eV',
    },
    'ne_m3_re': {
        'def': 0.,
        'units': '1/m3',
    },
    'Zeff': {
        'def': 1.,
        'units': None,
    },
    'Ekin_max_eV': {
        'def': 10e6,
        'units': 'eV',
    },
    'Efield_par_Vm': {
        'def': 0.1,
        'units': 'V/m',
    },
    'lnG': {
        'def': 20.,
        'units': '',
    },
    'sigmap': {
        'def': 0.1,
        'units': '',
    },
}


_DCOORDS = {
    'v_par_ms': {'units': 'm/s'},
    'v_perp_ms': {'units': 'm/s'},
    'p_par_norm': {'units': ''},
    'p_perp_norm': {'units': ''},
    'E_eV': {'units': 'eV'},
    'pitch': {'units': ''},
    'theta': {'units': 'rad'},
}


_DFUNC = {
    ('v_par_ms', 'v_perp_ms'): [
        'f3d_cart_vpar_vperp',
        'f2d_cart_vpar_vperp',
        'f2d_cyl_vpar_vperp',
    ],
    ('p_par_norm', 'p_perp_norm'): ['f2d_ppar_pperp'],
    ('E_eV',): ['f1d_E'],
    ('E_eV', 'pitch'): ['f2d_E_pitch'],
    ('E_eV', 'theta'): ['f2d_E_theta', 'f3d_E_theta'],
}


# #######################################################
# #######################################################
#                   Main
# #######################################################


def main(
    **kwdargs,
):

    # ---------------
    # dist
    # ---------------

    dist = _dist(**kwdargs)

    # ---------------
    # returnas / key
    # ---------------

    returnas, coll = _returnas(**kwdargs)

    # -------------------------
    # plasma parameters
    # -------------------------

    dplasma = _plasma(
        ddef=_DPLASMA,
        **kwdargs,
    )

    # adjust
    if dist == ('maxwell',):
        dplasma['jp_fraction_re']['data'][...] = 0.

    # -------------------------
    # coordinates & versions
    # -------------------------

    dcoords = _coords(**kwdargs)

    # -------------------------
    # versions and func
    # -------------------------

    dcoords, dfunc = _dfunc(
        dcoords=dcoords,
        version=kwdargs['version'],
        dist=dist,
    )

    # -------------------------
    # verb
    # -------------------------

    lok = [False, True, 0, 1, 2]
    verb = int(ds._generic_check._check_var(
        kwdargs['verb'], 'verb',
        types=(bool, int),
        default=lok[-1],
        allowed=lok,
    ))

    return dist, dplasma, dcoords, dfunc, coll, verb


# #######################################################
# #######################################################
#                   dist
# #######################################################


def _dist(
    dist=None,
    # unused
    **kwdargs,
):

    # -----------
    # str => tuple
    # -----------

    if isinstance(dist, str):
        dist = (dist,)

    # -----------
    # check allowed + set default
    # -----------

    lok = ['maxwell', 'RE']
    dist = tuple(ds._generic_check._check_var_iter(
        dist, 'dist',
        types=(tuple, list),
        default=lok,
        allowed=lok,
    ))

    # ------------
    # checks
    # ------------

    if 'maxwell' not in dist:
        msg = "Arg 'dist' must include 'maxwell'!"
        raise Exception(msg)

    return dist


# #######################################################
# #######################################################
#                   Returnas
# #######################################################


def _returnas(
    returnas=None,
    # unused
    **kwdargs,
):

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

    return returnas, coll


# #######################################################
# #######################################################
#               Plasma
# #######################################################


def _plasma(
    ddef=None,
    **kwdargs,
):

    # -------------------
    # prelim check
    # -------------------

    # initialize
    lk = list(_DPLASMA.keys())
    dinputs = {kk: kwdargs[kk] for kk in lk}

    # coll
    coll = kwdargs.get('coll')

    # -------------------
    # loop
    # -------------------

    dout = _extract(dinputs, coll, ddef, _DPLASMA)

    # Exception
    if len(dout) > 0.:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dout.items()]
        msg = (
            "The following plasma parameters are not properly set:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # -------------------
    # broadcast
    # -------------------

    dbroad, _ = ds._generic_check._check_all_broadcastable(
        return_full_arrays=True,
        **{kk: vv['data'] for kk, vv in dinputs.items()},
    )

    # update dinputs
    for k0, v0 in dbroad.items():
        dinputs[k0]['data'] = v0

    return dinputs


# #######################################################
# #######################################################
#               Generic extract routine
# #######################################################


def _extract(din, coll, ddef, ddef0):

    dout = {}
    for k0 in din.keys():
        # units
        units = ddef.get(k0, ddef0[k0]).get('units')

        # check vs None
        if din.get(k0) is None:
            data = np.asarray(ddef.get(k0, ddef0[k0]).get('def'))
        else:
            data = din[k0]

        # if str => coll.ddata
        if isinstance(data, str):

            if coll is None:
                dout[k0] = "for using str, provide returnas=coll"
                continue

            if data not in coll.ddata.keys():
                dout[k0] = f"not available in coll.ddata: {data}"
                continue

            units0 = coll.ddata[data]['units']
            if asunits.Unit(units0) != units:
                dout[k0] = f"wrong units: {units0} vs {units}"
                continue

            # ref = coll.ddata[data]['ref']
            data = np.copy(coll.ddata[data]['data'])

        else:
            # ref = None
            data = np.atleast_1d(data)

        # set subdict
        din[k0] = {
            'data': data,
            'units': units,
            # 'ref': ref,   # not relevant due to later broadcasting
        }

    return dout


# #######################################################
# #######################################################
#               Coords
# #######################################################


def _coords(
    **kwdargs,
):

    # --------------
    # preliminary
    # --------------

    # initialize
    lk = list(_DCOORDS.keys())
    dcoords = {kk: kwdargs[kk] for kk in lk if kwdargs[kk] is not None}

    # coll
    coll = kwdargs.get('coll')

    # -------------------
    # loop
    # -------------------

    dout = _extract(dcoords, coll, _DCOORDS, _DCOORDS)

    # Exception
    if len(dout) > 0.:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dout.items()]
        msg = (
            "The following plasma parameters are not properly set:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # --------------
    # check 1d
    # --------------

    dout = {}
    for k0, v0 in dcoords.items():
        if v0['data'].ndim != 1:
            dout[k0] = f"coordinate '{k0}' not 1d: {v0['data'].shape}"

    # Exception
    if len(dout) > 0.:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dout.items()]
        msg = (
            "The following coordinates should be 1d arrays!:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    return dcoords


# #######################################################
# #######################################################
#               dfunc
# #######################################################


def _dfunc(
    dist=None,
    dcoords=None,
    version=None,
):

    # --------------
    # check a pair exist
    # --------------

    pair = tuple(sorted(dcoords.keys()))
    if pair not in _DFUNC.keys():
        lstr0 = [f"\t- {k0}: {v0}" for k0, v0 in _DFUNC.items()]
        lstr1 = [f"\t- {k0}" for k0 in pair]
        msg = (
            "Please provide 1 or 2 coordinates max!\n"
            "Possible pairs and matching func:\n"
            + "\n".join(lstr0)
            + "\nProvided:\n"
            + "\n".join(lstr1)
        )
        raise Exception(msg)

    # --------------
    # remap to x0, x1
    # --------------

    dnew = {}
    for ii, kk in enumerate(pair):
        dnew[f"x{ii}"] = {
            'key': kk,
            'data': dcoords[kk]['data'],
            'units': dcoords[kk]['units'],
        }
    dcoords = dnew

    # --------------
    # version
    # --------------

    version = ds._generic_check._check_var(
        version, 'version',
        types=str,
        allowed=_DFUNC[pair],
        default=_DFUNC[pair][-1],
    )

    # --------------
    # dfunc
    # --------------

    dfunc = {}
    for kdist in dist:

        # choose module
        if kdist == 'maxwell':
            mod = _maxwell
        else:
            mod = _re

        func = getattr(mod, 'main')

        # store
        dfunc[kdist] = {
            'version': version,
            'func': func,
        }

    return dcoords, dfunc
