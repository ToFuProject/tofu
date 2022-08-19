# -*- coding: utf-8 -*-


import numpy as np
import scipy.constants as scpct
import datastock as ds


from . import _generic_check
from . import _def_crystals


_DMAT_KEYS = {
    'name': {'types': str},
    'symbol': {'types': str},
    'target_ion': {'types': str},
    'target_lamb': {'types': float},
    'atoms': {'types': (list, np.ndarray), 'types_iter': str},
    'atoms_Z': {'types': (list, np.ndarray)},
    'atoms_nb': {'types': (list, np.ndarray)},
    'miller': {'types': (list, np.ndarray)},
    'alpha': {'types': float, 'default': 0.},
    'beta': {'types': float, 'default': 0.},
    'dmesh': {'types': dict},
    'dphases': {'types': dict},
    'dinter_atomic': {'types': dict},
    'dthermal_exp': {'types': dict},
    'datom_scatter': {'types': dict},
    'dsin(theta)/lambda': {'types': dict},
}


# #############################################################################
# #############################################################################
#                           Crystal
# #############################################################################


def _dmat(
    dmat=None,
):

    # ---------------------
    # Easy cases
    # ---------------------

    # not mat
    if dmat is None:
        return dobj

    # known crystal
    if isinstance(dmat, str):
        if dmat not in _def_crystals._DCRYST.keys():
            msg = (
                f"Arg dmat points to an unknown crystal: '{dmat}'"
            )
            raise Exception(msg)
        dmat = _def_crystals._DCRYST[dmat]

    # ---------------------
    # check dict integrity
    # ---------------------

    # Check dict typeand content (each key is a valid string)
    ds._check_dict_valid_keys(
        var=dmat,
        varname='dmat',
        dkey_types=dkey_types,
    )

    # -------------------------------
    # check each value independently
    # -------------------------------

    # target ion

    # target lamb
    if dmat['target_lamb'] is not None:
        assert dmat['target_lamb'] > 0., dmat['target_lamb']

    # atoms
    if dmat['atoms'] is not None:
        dmat['atoms'] = np.atleast_1d(dmat['atoms']).ravel()

    # atoms_Z
    if dmat['atoms_Z'] is not None:
        dmat['atoms_Z'] = np.atleast_1d(dmat['atoms_Z']).ravel().astype(int)
        assert dmat['atoms_Z'].size == dmat['atoms'].size

    # atoms_nb
    if dmat['atoms_nb'] is not None:
        dmat['atoms_nb'] = np.atleast_1d(dmat['atoms_nb']).ravel().astype(int)
        assert dmat['atoms_nb'].size == dmat['atoms'].size

    # miller
    if dmat['miller'] is not None:
        dmat['miller'] = np.atleast_1d(dmat['miller']).ravel().astype(int)
        assert dmat['miller'].size == 3

    # alpha
    if dmat['alpha'] is not None:
        dmat['alpha'] = np.abs(np.arctan(
            np.sin(dmat['alpha']),
            np.cos(dmat['alpha']),
        ))

    # beta
    if dmat['beta'] is not None:
        dmat['beta'] = np.arctan2(np.sin(dmat['beta']), np.cos(dmat['beta']))

    # -------------------------------
    # check sub-dict
    # -------------------------------

    dmat['angles'] = _check_flat1darray_size(
        var=dmat.get('angles'), varname='angles', size=3
    )

    # inter-reticular distance
    if dmat['d'] is None:
        msg = "Arg dmat['d'] must be convertible to a float."
        raise Exception(msg)
    dmat['d'] = float(dmat['d'])

    # density
    if dmat['density'] is None:
        msg = "Arg dmat['density'] must be convertible to a float."
        raise Exception(msg)
    dmat['density'] = float(dmat['density'])

    # formula
    if isinstance(dmat['formula'], str) is False:
        msg = (
            """
            Var {} must be a valid string.

            Provided
                - type: {}
            """.format('formula', type(dmat['formula']))
        )
        raise Exception(msg)

    # miller indices
    if dmat.get('cut') is not None:
        dmat['cut'] = np.atleast_1d(dmat['cut']).ravel().astype(int)
        if dmat['cut'].size == 3:
            pass
        elif dmat['cut'].size == 4:
            pass
        else:
            msg = (
                "Var 'cut' should be convertible to a 1d array of size 3 or 4"
                f"\nProvided: {dmat.get('cut')}"
            )
            raise Exception(msg)

    # Check orthonormal direct basis
    _check_dict_unitvector(dd=dmat, dd_name='dmat')

    # Check all additionnal angles to define the new basis
    for k0 in ['alpha', 'beta']:
        dmat[k0] = _check_flat1darray_size(
            var=dmat.get(k0), varname=k0, size=1, norm=False)

    # -------------------------------------------------------------
    # Add missing vectors and parameters according to the new basis
    # -------------------------------------------------------------

    if all([dgeom[kk] is not None for kk in ['nout', 'e1', 'e2']]):

        # dict of value, comment, default value and type of
        #  alpha and beta angles in dmat
        dpar = {
            'alpha': {
                # 'alpha': alpha,
                'com': 'non-parallelism amplitude',
                'default': 0.,
                'type': float,
            },
            'beta': {
                # 'beta': beta,
                'com': 'non-parallelism orientation',
                'default': 0.,
                'type': float,
            },
        }

        # setting to default value if any is None
        lparNone = [aa for aa in dpar.keys() if dmat.get(aa) is None]

        # if any is None, assigning default value and send a warning message
        if len(lparNone) > 0:
            msg = "The following parameters were set to their default values:"
            for aa in lparNone:
                dmat[aa] = dpar[aa]['default']
                msg += "\n\t - {} = {} ({})".format(
                    aa, dpar[aa]['default'], dpar[aa]['com'],
                )
            warnings.warn(msg)

        # check conformity of type of angles
        lparWrong = []
        for aa in dpar.keys():
            try:
                dmat[aa] = float(dmat[aa])
            except Exception as err:
                lparWrong.append(aa)

        if len(lparWrong) > 0:
            msg = "The following parameters must be convertible to:"
            for aa in lparWrong:
                msg += "\n\t - {} = {} ({})".format(
                    aa, dpar[aa]['type'], type(dmat[aa]),
                )
            raise Exception(msg)

        # Check value of alpha
        if dmat['alpha'] < 0 or dmat['alpha'] > np.pi/2:
            msg = (
                "Arg dmat['alpha'] must be an angle (radians) in [0; pi/2]"
                + "\nProvided:\n\t{}".format(dmat['alpha'])
            )
            raise Exception(msg)

        if dmat['beta'] < -np.pi or dmat['beta'] > np.pi:
            msg = (
                "Arg dmat['beta'] must be an angle (radians) in [-pi; pi]"
                + "\nProvided:\n\t{}".format(dmat['beta'])
            )
            raise Exception(msg)

        # dict of value, comment, default value and type of unit vectors in
        #  dmat, computting default value of unit vectors, corresponding to
        #  dgeom values if angles are 0.
        dvec = {
            'e1': {
                # 'e1': e1,
                'com': 'unit vector (non-parallelism)',
                'default': (
                    np.cos(dmat['alpha'])*(
                        np.cos(dmat['beta'])*dgeom['e1']
                        + np.sin(dmat['beta'])*dgeom['e2']
                    )
                    - np.sin(dmat['alpha'])*dgeom['nout']
                ),
                'type': float,
            },
            'e2': {
                # 'e2': e2,
                'com': 'unit vector (non-parallelism)',
                'default': (
                    np.cos(dmat['beta'])*dgeom['e2']
                    - np.sin(dmat['beta'])*dgeom['e1']
                ),
                'type': float,
            },
            'nout': {
                # 'nout': nout,
                'com': 'outward unit vector (normal to non-parallel mesh)',
                'default': (
                    np.cos(dmat['alpha'])*dgeom['nout']
                    + np.sin(dmat['alpha']) * (
                        np.cos(dmat['beta'])*dgeom['e1']
                        + np.sin(dmat['beta'])*dgeom['e2']
                    )
                ),
                'type': float,
            }
        }

        # setting to default value if any is None
        lvecNone = [bb for bb in dvec.keys() if dmat.get(bb) is None]

        # if any is None, assigning default value
        if len(lvecNone) > 0:
            for bb in lvecNone:
                dmat[bb] = dvec[bb]['default']

        # check conformity of type of unit vectors
        lvecWrong = []
        for bb in dvec.keys():
            try:
                dmat[bb] = np.atleast_1d(dmat[bb]).ravel().astype(float)
                dmat[bb] = dmat[bb] / np.linalg.norm(dmat[bb])
                assert np.allclose(dmat[bb], dvec[bb]['default'])
            except Exception as err:
                lvecWrong.append(bb)

        if len(lvecWrong) > 0:
            msg = "The following parameters must be convertible to:"
            for bb in lvecWrong:
                msg += "\n\t - {} = {} ({})".format(
                    bb, dvec[bb]['type'], type(dmat[bb]),
                )
            msg += "\nAnd (nout , e1, e2) must be an orthonormal direct basis"
            raise Exception(msg)

        # Computation of unit vector nin
        dmat['nin'] = -dmat['nout']

    return dmat
