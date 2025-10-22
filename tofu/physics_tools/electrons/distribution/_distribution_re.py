

import numpy as np


from .. import _convert
from . import _runaway_growth
from . import _distribution_dreicer as _dreicer
from . import _distribution_avalanche as _avalanche


# ########################################################
# ########################################################
#                DEFAULT
# ########################################################


_DMOD = {
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
    # check
    # -----------

    # -----------
    # initialize
    # -----------

    ncoords = len(dcoords)
    shape_plasma = dplasma['Te_eV']['data'].shape[:-ncoords]
    shape_coords = np.broadcast_shapes(*[v0.shape for v0 in dcoords.values()])
    shape_coords = shape_coords[-ncoords:]
    shape = shape_plasma + shape_coords
    re_dist = np.zeros(shape, dtype=float)

    sli_coords = (slice(None),)*len(dcoords)
    sli0 = (slice(None),)*len(shape_plasma) + (0,)*len(dcoords)

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

    p_crit = np.full(Etild.shape, np.nan)
    E_hat = np.full(Etild.shape, np.nan)
    Cz = np.full(Etild.shape, np.nan)
    Cs = np.full(Etild.shape, np.nan)

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

    for vv, ind in dind.items():

        iok = dind[vv]['ind']
        sli0 = (iok,) + (None,)*len(dcoords)
        sli1 = (iok,) + (slice(None),)*len(dcoords)

        # -------------------
        # kwdargs to func

        kwdargs = {
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

        # -------------------
        # kwdargs to func

        kwdargsi.update(**dcoords)
        iok0 = np.copy(iok)
        iok0[iok0] = ioki
        sli = (iok0[sli0],) + sli_coords

        # -------------------
        # kwdargs to func

        dom = [k0 for k0, v0 in _DOMINANT.items() if v0 == vv][0]
        re_dist[sli1], units0 = getattr(_DMOD[dom], version)(**kwdargsi)


    # ---------------------------
    # intermediate check on Etild
    # ---------------------------

    iok = Etild > 1.
    if np.any(iok):

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

        # -------------------
        # kwdargs to func

        kwdargs = {
            'sigmap': dplasma['sigmap']['data'][iok],
            'p_max_norm': pmax[iok],
            'Etild': Etild[iok],
            'Zeff': dplasma['Zeff']['data'][iok],
            'E_hat': E_hat[iok],
            'Cz': Cz[iok],
            'Cs': Cs[iok],
            'lnG': dplasma['lnG']['data'][iok],
            'p_crit': p_crit[iok],
        }

        # ------------------
        # Compute

        # avalanche-dominated
        ioki = Etild[iok] > 5.
        units0 = None
        if np.any(ioki):
            sli = (ioki,) + (None,)*len(dcoords)
            kwdargsi = {k0: v0[sli] for k0, v0 in kwdargs.items()}
            kwdargsi.update(**dcoords)
            iok0 = np.copy(iok)
            iok0[iok0] = ioki
            sli = (iok0[sli0],) + sli_coords
            re_dist[sli], units0 = getattr(_avalanche, version)(**kwdargsi)
            dominant[sli] = 0.

        # Dreicer-dominated
        ioki = (2 < Cs[iok]) & (Cs[iok] < 1 + Etild[iok])
        units1 = None
        if np.any(ioki):
            sli = (ioki,) + (None,)*len(dcoords)
            kwdargsi = {k0: v0[sli] for k0, v0 in kwdargs.items()}
            kwdargsi.update(**dcoords)
            iok0 = np.copy(iok)
            iok0[iok0] = ioki
            sli = (iok0[sli0],) + sli_coords
            re_dist[sli], units1 = getattr(_dreicer, version)(**kwdargsi)

            dominant[sli] = 1.

        # sanity check
        if units0 is not None and units1 is not None:
            if units0 != units1:
                msg = (
                    "Different units for avalanche vs Dreicer!\n"
                    f"\t- avalanche: {units0}\n"
                    f"\t- dreicer:   {units0}\n"
                )
                raise Exception(msg)

        units = units0 if units1 is None else units1

        # -------------------
        # threshold on p_crit

        pnorm = np.broadcast_to(_get_pnorm(dcoords), re_dist.shape)
        iok = np.copy(np.broadcast_to(iok, re_dist.shape))
        iok[iok] = pnorm[iok] < np.broadcast_to(p_crit, re_dist.shape)[iok]
        re_dist[iok] = 0.

    else:
        kwdargsi = {
            'sigmap': np.r_[1.],
            'p_max_norm': np.r_[10],
            'Etild': np.r_[1.5],
            'Zeff': np.r_[1],
            'E_hat': np.r_[2.],
            'Cz': np.r_[2.5],
            'Cs': np.r_[2.],
            'lnG': np.r_[20.],
            'p_crit': np.r_[0.1],
        }
        kwdargsi.update(**dcoords)
        units = getattr(_avalanche, version)(**kwdargsi)[1]

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
        dominant = np.full(shape, _DOMINANT[dominant])

    elif lc[1]:
        dominant = np.full(shape, dominant)

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

    # -----------------
    # dind
    # -----------------

    ipos = dominant >= 0
    lv = sorted(np.unique(dominant[ipos]))

    dind = {
        vv: {
            'ind': dominant == vv,
        }
        for vv in lv
    }

    return dominant, dind
