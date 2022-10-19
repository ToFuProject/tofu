# -*- coding: utf-8 -*-


import numpy as np
import scipy.constants as scpct
import datastock as ds


from . import _generic_check
from ..spectro import _rockingcurve_def


_DMAT_KEYS = {
    'name': {'types': str},
    'symbol': {'types': str},
    'd_hkl': {'types': float, 'sign': '> 0.'},
    'target_ion': {'types': str},
    'target_lamb': {'types': float, 'sign': '> 0.'},
    'atoms': {'types': (list, np.ndarray), 'types_iter': str},
    'atoms_Z': {'dtype': int, 'size': 2, 'sign': '> 0'},
    'atoms_nb': {'dtype': int, 'size': 2, 'sign': '> 0'},
    'miller': {'dtype': int, 'size': 3, 'sign': '>= 0'},
    'alpha': {'types': float, 'default': 0., 'sign': '>= 0'},
    'beta': {'types': float, 'default': 0.},
    'mesh': {'types': dict},
    'phases': {'types': dict},
    'inter_atomic': {'types': dict},
    'thermal_expansion': {'types': dict},
    'atomic_scattering': {'types': dict},
    'sin_theta_lambda': {'types': dict},
}


# #############################################################################
# #############################################################################
#                           Crystal
# #############################################################################


def _dmat(
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
    # check each value independently
    # -------------------------------

    # target ion

    # alpha
    if alpha is not None:
        dmat['alpha'] = alpha
    if dmat['alpha'] is not None:
        dmat['alpha'] = np.abs(np.arctan(
            np.sin(dmat['alpha']),
            np.cos(dmat['alpha']),
        ))

    # beta
    if beta is not None:
        dmat['beta'] = beta
    if dmat['beta'] is not None:
        dmat['beta'] = np.arctan2(np.sin(dmat['beta']), np.cos(dmat['beta']))

    # vector basis with non-paralellism
    if all([dmat[k0] is not None for k0 in ['alpha', 'beta']]):
        nin = (
            np.cos(dmat['alpha'])*(dgeom['nin'])
            + np.sin(dmat['alpha']) * (
                np.cos(dmat['beta'])*dgeom['e0']
                + np.sin(dmat['beta'])*dgeom['e1']
            )
        )
        e0 = (
            - np.sin(dmat['alpha'])*(dgeom['nin'])
            + np.cos(dmat['alpha']) * (
                np.cos(dmat['beta'])*dgeom['e0']
                + np.sin(dmat['beta'])*dgeom['e1']
            )
        )
        nin, e0, e1 = ds._generic_check._check_vectbasis(
            e0=nin,
            e1=e0,
            e2=None,
            ndim=3,
        )

        dmat['nin'] = nin
        dmat['e0'] = e0
        dmat['e1'] = e1

    # -------------------------------
    # check sub-dict
    # -------------------------------

    ldict = [k0 for k0, v0 in _DMAT_KEYS.items() if v0.get('types') is dict]
    for k0 in ldict:
        pass
    # TODO: implement further checks of sub-dict ?

    return dmat
