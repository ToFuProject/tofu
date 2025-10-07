

import numpy as np


from . import _convert
from . import _runaway_growth
from . import _distribution_dreicer as _dreicer
from . import _distribution_avalanche as _avalanche


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
    }

    return dout
