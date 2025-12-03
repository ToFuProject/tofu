

import numpy as np


from .. import _convert
from . import _runaway_growth
from . import _distribution_maxwell as _maxwell
from . import _distribution_dreicer as _dreicer
from . import _distribution_avalanche as _avalanche


# ########################################################
# ########################################################
#                DEFAULT
# ########################################################


_DMOD = {
    'maxwell': _maxwell,
    'dreicer': _dreicer,
    'avalanche': _avalanche,
}


_DOMINANT = {
    'dreicer': 0,
    'avalanche': 1,
    'maxwell': 2,
}


# ########################################################
# ########################################################
#                Main
# ########################################################


def main(
    # coordinates
    dcoords=None,
    version=None,
    # plasma
    dplasma=None,
    # assmume dominant
    dominant=None,
    # unused
    **kwdargs,
):

    # -----------
    # initialize
    # -----------

    ncoords = len(dcoords)
    shape_plasma = dplasma['Te_eV']['data'].shape[:-ncoords]
    shape_coords = np.broadcast_shapes(*[v0.shape for v0 in dcoords.values()])
    shape_coords = shape_coords[-ncoords:]
    shape = shape_plasma + shape_coords
    re_dist = np.zeros(shape, dtype=float)

    # slices
    sli_coords = (slice(None),)*len(dcoords)
    sliok = (slice(None),)*len(shape_plasma) + (0,)*len(dcoords)

    # -------------
    # prepare
    # -------------

    # get momentum max from total energy eV.s/m - shape
    pmax = _convert.convert_momentum_velocity_energy(
        energy_kinetic_eV=dplasma['Ekin_max_eV']['data'],
    )['momentum_normalized']['data']

    # Critical electric field - shape
    Ec_Vm = _runaway_growth.get_RE_critical_dreicer_electric_fields(
        ne_m3=dplasma['ne_m3']['data'],
        kTe_eV=None,
        lnG=dplasma['lnG']['data'],
    )['E_C']['data']

    # -------------
    # Intermediates
    # -------------

    # normalized electric field, adim
    Etild = dplasma['Efield_par_Vm']['data'] / Ec_Vm

    shapeE = Etild.shape
    p_crit = np.full(shapeE, np.nan)
    E_hat = np.full(shapeE, np.nan)
    Cz = np.full(shapeE, np.nan)
    Cs = np.full(shapeE, np.nan)

    # ---------------------------
    # get dominant distribution
    # ---------------------------

    dominant, dind = _get_dominant(
        Etild=Etild,
        E_hat=E_hat,
        Cz=Cz,
        Cs=Cs,
        p_crit=p_crit,
        dplasma=dplasma,
        # dominant
        dominant=dominant,
    )

    # --------------------
    # loop on dominant
    # --------------------

    dunits = {}
    ncoords = len(dcoords)
    for vv, ind in dind.items():

        iok = dind[vv]['ind']
        sli0 = (iok[sliok],) + sli_coords
        sli1 = sli0

        # -------------------
        # kwdargs to func
        if dominant['meaning'][vv] == 'maxwell':

            kwdargsi = {
                'Te_eV': {'data': dplasma['Te_eV_re']['data'][sli0]},
                'ne_m3': {'data': dplasma['ne_m3_re']['data'][sli0]},
                'jp_Am2': {'data': dplasma['jp_Am2']['data'][sli0]},
            }

            dout = _maxwell.main(
                dcoords=dcoords,
                dplasma=kwdargsi,
                version=version,
            )

            # store
            re_dist[sli1] = dout['dist']['data']
            dunits[dominant['meaning'][vv]] = dout['dist']['units']

        else:

            # kwdargs to func
            kwdargsi = {
                'sigmap': dplasma['sigmap']['data'][sli0],
                'p_max_norm': pmax[sli0],
                'Etild': Etild[sli0],
                'Zeff': dplasma['Zeff']['data'][sli0],
                'E_hat': E_hat[sli0],
                'Cz': Cz[sli0],
                'Cs': Cs[sli0],
                'lnG': dplasma['lnG']['data'][sli0],
                'p_crit': p_crit[sli0],
            }

            # update with coords
            kwdargsi.update(**dcoords)

            # compute
            re_dist[sli1], dunits[dominant['meaning'][vv]] = getattr(
                _DMOD[dominant['meaning'][vv]],
                version,
            )(**kwdargsi)

        # -------------------
        # threshold on p_crit

        if dominant['meaning'][vv] != 'maxwell':
            pnorm = np.broadcast_to(_get_pnorm(dcoords), shape)
            iokp = np.copy(np.broadcast_to(iok, shape))
            iokp[iokp] = pnorm[iokp] < np.broadcast_to(p_crit, shape)[iokp]
            re_dist[iokp] = 0.

    # ----------------------
    # sanity check on units
    # ----------------------

    lunits = list(set([uu for uu in dunits.values()]))
    if len(lunits) != 1:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dunits.items()]
        msg = (
            "Different units:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # units
    units = lunits[0]

    # --------------
    # format output
    # --------------

    dout = {
        'dist': {
            'data': re_dist,
            'units': units,
        },
        'Cs': {
            'data': Cs,
            'units': None,
        },
        'Cz': {
            'data': Cz,
            'units': None,
        },
        'Ec': {
            'data': Ec_Vm,
            'units': 'V/m',
        },
        'Etild': {
            'data': Etild,
            'units': None,
        },
        'E_hat': {
            'data': E_hat,
            'units': None,
        },
        'p_crit': {
            'data': p_crit,
            'units': None,
        },
        'p_max': {
            'data': pmax,
            'units': None,
        },
        'dominant': {
            'data': dominant,
            'units': None,
            'meaning': {0: 'avalanche', 1: 'dreicer'}
        },
    }

    return dout


# ##############################################
# ##############################################
#               _get_pp
# ##############################################


def _get_pnorm(dcoords):

    if dcoords.get('E_eV') is not None:

        pnorm = _convert.convert_momentum_velocity_energy(
            energy_kinetic_eV=dcoords['E_eV'],
        )['momentum_normalized']['data']

    elif dcoords.get('p_par_norm') is not None:

        pnorm = np.sqrt(
            dcoords['p_par_norm']**2
            + dcoords['p_perp_norm']**2
        )

    else:
        raise NotImplementedError(sorted(dcoords.keys))

    return pnorm


# ##############################################
# ##############################################
#               dominant
# ##############################################


def _get_dominant(
    Etild=None,
    E_hat=None,
    Cz=None,
    Cs=None,
    p_crit=None,
    dplasma=None,
    # dominant
    dominant=None,
):

    # -----------------
    # dominant_exp
    # -----------------

    shape = Etild.shape
    dominant_exp = np.full(shape, np.nan)

    iok = Etild > 1.
    if np.any(iok):

        # E_hat
        E_hat[iok] = (Etild[iok] - 1) / (1 + dplasma['Zeff']['data'][iok])

        # adim
        Cz[iok] = np.sqrt(3 * (dplasma['Zeff']['data'][iok] + 5) / np.pi)

        # critical momentum, adim
        p_crit[iok] = 1. / np.sqrt(Etild[iok] - 1.)

        # Cs
        Cs[iok] = (
            Etild[iok]
            - (
                ((1 + dplasma['Zeff']['data'][iok])/4)
                * (Etild[iok] - 2)
                * np.sqrt(Etild[iok] / (Etild[iok] - 1))
            )
        )

        # ------------------
        # Compute

        # Dreicer-dominated
        iok_dreicer = np.copy(iok)
        iok_dreicer[iok] = (2 < Cs[iok]) & (Cs[iok] < 1 + Etild[iok])
        dominant_exp[iok_dreicer] = 0

        # avalanche-dominated
        iok_avalanche = np.copy(iok)
        iok_avalanche[iok] = (~iok_dreicer[iok]) & (Etild[iok] > 5.)
        dominant_exp[iok_avalanche] = 1

    else:
        iok_dreicer = iok
        iok_avalanche = iok

    # maxwell-dominated
    iok_maxwell = iok & (~iok_dreicer) & (~iok_avalanche)
    dominant_exp[iok_maxwell] = 2

    # -----------------
    # check dominant
    # -----------------

    if dominant is None:
        dominant = -np.ones(shape, dtype=float)

    lv = sorted(_DOMINANT.values())
    lc = [
        isinstance(dominant, str) and dominant in _DOMINANT.keys(),
        isinstance(dominant, int) and dominant in lv,
        isinstance(dominant, np.ndarray)
        and dominant.shape == shape
        and np.all(np.any([dominant == vv for vv in lv + [-1]], axis=0)),
    ]

    if lc[0]:
        dominant = np.full(shape, _DOMINANT[dominant], dtype=float)

    elif lc[1]:
        dominant = np.full(shape, dominant, dtype=float)

    elif lc[2]:
        pass

    else:
        lstr = [
            f"\t- {k0} or {v0}: {k0}-dominated"
            for k0, v0 in _DOMINANT.items()
        ]
        msg = (
            "Arg dominant must specify, for each plasma point, "
            "the dominant RE distribution:\n"
            + "\n".join(lstr)
            + "Alternatively, can be provided as a np.ndarray of:\n"
            f"\t- shape: {shape}\n"
            f"\t- values in {lv}\n"
            "Value = -1 => whatever distribution should dominante\n"
        )
        raise Exception(msg)

    # -----------------------
    # set with exprimental where not specified
    # -----------------------

    iexp = dominant < 0
    dominant[iexp] = dominant_exp[iexp]

    # ------------------------
    # adjust for computability
    # ------------------------

    iout = (
        (dominant < 0)
        | ((dominant == 0) & (~iok))
        | ((dominant == 1) & (~iok))
    )
    dominant[iout] = np.nan

    dominant = {
        'ind': dominant,
        'meaning': {vv: kk for kk, vv in _DOMINANT.items()}
    }

    # -----------------
    # dind
    # -----------------

    iok = np.isfinite(dominant['ind'])
    iok[iok] = dominant['ind'][iok] >= 0.
    lv = sorted(np.unique(dominant['ind'][iok]))

    dind = {
        vv: {
            'ind': dominant['ind'] == vv,
        }
        for vv in lv
    }
    return dominant, dind
