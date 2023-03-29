# -*- coding: utf-8 -*-


import copy


import numpy as np
import scipy.constants as scpct
import scipy.interpolate as scpinterp
import datastock as ds


from . import _generic_check
from ..spectro import _rockingcurve_def
from ..spectro import _rockingcurve


_DMAT_KEYS = {
    'material': {'types': str},
    'name': {'types': str},
    'symbol': {'types': str},
    'd_hkl': {'types': float, 'sign': '> 0.'},
    'target': {'types': dict},
    'miller': {'dtype': int, 'size': 3, 'sign': '>= 0'},
    'alpha': {'types': float, 'default': 0., 'sign': '>= 0'},
    'beta': {'types': float, 'default': 0.},
    'mesh': {'types': dict},
    'miscut': {'types': bool, 'default': True},
}


# ################################################################
# ################################################################
#                           Crystal
# ################################################################


def _dmat(
    coll=None,
    key=None,
    dmat=None,
    alpha=None,
    beta=None,
    dgeom=None,
):

    # ---------------------
    # Easy cases
    # ---------------------

    # not mat
    if dmat is None:
        return dmat

    # known crystal
    ready_to_compute = False
    if isinstance(dmat, str):
        if dmat not in _rockingcurve_def._DCRYST.keys():
            msg = (
                f"Arg dmat points to an unknown crystal: '{dmat}'"
            )
            raise Exception(msg)
        dmat = _rockingcurve_def._DCRYST[dmat]
        ready_to_compute = True

    # ---------------------
    # check dict integrity
    # ---------------------

    # Check dict typeand content (each key is a valid string)
    dmat = ds._generic_check._check_dict_valid_keys(
        var=dmat,
        varname='dmat',
        has_all_keys=False,
        has_only_keys=False,
        keys_can_be_None=True,
        dkeys=_DMAT_KEYS,
    )

    dmat['ready_to_compute'] = ready_to_compute

    # -------------------------------
    # check parallelism
    # -------------------------------

    _check_parallelism(
        alpha=alpha,
        beta=beta,
        dmat=dmat,
        dgeom=dgeom,
    )

    # -------------------------------
    # check rocking curve
    # -------------------------------

    dref = None
    if ready_to_compute:
        drock = _rockingcurve.compute_rockingcurve(
            # Type of crystal
            crystal=dmat['name'],
            din=None,
            # Wavelength
            lamb=dmat['target']['lamb'],
            # Lattice modifications
            miscut=dmat['miscut'],
            nn=None,
            alpha_limits=None,
            therm_exp=True,
            temp_limits=None,
            # Plot
            plot_therm_exp=False,
            plot_asf=False,
            plot_power_ratio=False,
            plot_asymmetry=False,
            plot_cmaps=False,
            # Returning dictionnary
            returnas=dict,
        )

        lk0 = [
            'material',
            'name',
            'symbol',
            'miller',
            'd_hkl',
            'target',
            'alpha',
            'beta',
            'miscut',
            'nin',
            'e0',
            'e1',
        ]
        dmat = {
            k0: copy.deepcopy(drock[k0])
            for k0 in lk0
        }
        dmat['mesh'] = {'type': str(drock['mesh']['type'])}

        # -------------------------------------------
        # interpolate at desired alpha vs T and angle

        dref, ddata, drock2 = _extract_rocking_curve(
            key=key,
            drock=drock,
        )

        dmat['drock'] = drock2


    elif isinstance(dmat, dict):

        if 'drock' in dmat.keys():
            dref, ddata = _extract_rocking_curve_from_array(dmat)

    else:
        msg = f"Unknown dmat:\n{dmat}"
        raise Exception(msg)

    # ---------------
    # add dref, ddata

    if dref is not None:
        coll.update(dref=dref, ddata=ddata)

    return dmat



# ################################################################
# ################################################################
#                   Utilities
# ################################################################


def _extract_rocking_curve(key=None, drock=None):

    # 'Power ratio', (2, 1, 41, 201)
    # shape = (polar, nT, nalpha, ny = nangles)

    # ----------------
    # extract key data

    _, nT, nc, na = drock['Power ratio'].shape
    braggref = drock['Bragg angle of reference (rad)']
    ang_rel = drock['Glancing angles'] - braggref[None, :, None, None]
    amin, amax = np.nanmin(ang_rel), np.nanmax(ang_rel)
    angles = np.linspace(amin, amax, na)
    temp = drock['Temperature ref'] + drock['Temperature changes (°C)']

    # ---------------
    # interpolate

    # ind_alpha
    ind_alpha = 0

    power_ratio = np.full((2, na, nT), np.nan)
    for ii in range(2):
        for jj in range(nT):
            power_ratio[ii, :, jj] = scpinterp.interp1d(
                ang_rel[ii, jj, ind_alpha, :],
                drock['Power ratio'][ii, jj, ind_alpha, :],
                kind='linear',
                axis=0,
                bounds_error=False,
                fill_value=0,
            )(angles)

    # ------------
    # fill dict

    # dref
    knang = f'{key}_rc_angn'
    kntemp = f'{key}_rc_tempn'

    dref = {
        knang: {'size': na},
        kntemp: {'size': nT},
    }

    # ddata
    kang = f'{key}_rc_ang'
    ktemp = f'{key}_rc_temp'
    krc = f'{key}_rc'

    ddata = {
        kang: {
            'data': angles,
            'ref': knang,
            'units': 'rad',
            'dim': 'angle',
        },
        ktemp: {
            'data': temp,
            'ref': kntemp,
            'units': 'C',
            'dim': 'temperature',
        },
        krc: {
            'data': np.mean(power_ratio, axis=0),
            'ref': (knang, kntemp),
            'units': '',
            'dim': 'ratio',
        },
    }

    # drock2
    drock2 = {
        'temperature': ktemp,
        'angle_rel': kang,
        'power_ratio': krc,
    }

    return dref, ddata, drock2

# ################################################################
# ################################################################
#                           Parallelism
# ################################################################


def _check_parallelism(
    alpha=None,
    beta=None,
    dmat=None,
    dgeom=None,
):

    # ------------
    # check inputs

    # alpha
    alpha = float(ds._generic_check._check_var(
        alpha, 'alpha',
        types=(float, int),
        default=0.,
        sign='>=0',
    )) % (np.pi/2.)

    # beta
    beta = float(ds._generic_check._check_var(
        beta, 'beta',
        types=(float, int),
        default=0.,
        sign='>=0',
    )) % (2.*np.pi)

    # ------------
    # set in dict

    # alpha
    if dmat.get('alpha') is None:
        dmat['alpha'] = alpha
    dmat['alpha'] = np.abs(np.arctan2(
        np.sin(dmat['alpha']),
        np.cos(dmat['alpha']),
    ))

    # beta
    if dmat.get('beta') is None:
        dmat['beta'] = beta
    dmat['beta'] = np.arctan2(
        np.sin(dmat['beta']),
        np.cos(dmat['beta']),
    )

    # ---------------------------------
    # vector basis with non-paralellism

    # nin
    nin = (
        np.cos(dmat['alpha'])*(dgeom['nin'])
        + np.sin(dmat['alpha']) * (
            np.cos(dmat['beta'])*dgeom['e0']
            + np.sin(dmat['beta'])*dgeom['e1']
        )
    )

    # e0
    e0 = (
        - np.sin(dmat['alpha'])*(dgeom['nin'])
        + np.cos(dmat['alpha']) * (
            np.cos(dmat['beta'])*dgeom['e0']
            + np.sin(dmat['beta'])*dgeom['e1']
        )
    )

    # e1 + check
    nin, e0, e1 = ds._generic_check._check_vectbasis(
        e0=nin,
        e1=e0,
        e2=None,
        dim=3,
    )

    # store
    dmat['nin'] = nin
    dmat['e0'] = e0
    dmat['e1'] = e1

    return
