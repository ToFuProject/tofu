# -*- coding: utf-8 -*-


import numpy as np
import scipy.constants as scpct
import datastock as ds


_DMAT_KEYS = {
    'name': {'types': str},
    'symbol': {'types': str},
    'thickness': {'types': float, 'sign': '> 0.'},
    'energy': {'dtype': float, 'sign': '> 0.'},
    'transmission': {'dtype': float, 'sign': '>= 0.'},
}


# #############################################################################
# #############################################################################
#                       dmat for filter
# #############################################################################


def _dmat(
    key=None,
    dmat=None,
):

    # ---------------------
    # Easy cases
    # ---------------------

    # not mat
    if dmat is None:
        return None, None, None

    # ---------------------
    # check dict integrity
    # ---------------------

    # Check dict type and content (each key is a valid string)
    dmat = ds._generic_check._check_dict_valid_keys(
        var=dmat,
        varname='dmat',
        has_all_keys=True,
        has_only_keys=False,
        keys_can_be_None=False,
        dkeys=_DMAT_KEYS,
    )

    # -----------------------------------
    # check energy / transmission values
    # -----------------------------------

    energ = dmat['energy']
    trans = dmat['transmission']
    if energ.size != trans.size:
        msg = (
            f"The following should be 1d arrays of the same size:\n"
            f"\t- dmat['energy'].shape = {dmat['energy'].shape}\n"
            f"\t- dmat['transmission'].shape = {dmat['transmission'].shape}\n"
        )
        raise Exception(msg)

    # make sure all values are finite
    iok = np.isfinite(energ) & np.isfinite(trans)
    energ = energ[iok]
    trans = trans[iok]

    # make sure energy os sorted and unique
    energ, inds = np.unique(energ, return_index=True)
    trans = trans[inds]

    # make sure trans <= 1.
    if np.any(trans > 1.):
        msg = "Arg dmat['transmission'] should be < 1!\nProvided: {trans}"
        raise Exception(msg)

    # remove trailing 0 and 1 in trans
    nE = energ.size
    ind = np.ones((nE,), dtype=bool)

    i0 = (trans == 0.).nonzero()[0]
    i0 = np.array([ii for ii in range(i0[-1]) if ii in i0 and ii + 1 in i0])

    i1 = (trans == 1.).nonzero()[0]
    i1 = np.array([
        ii for ii in range(trans.size-1, i1[0]-1, -1)
        if ii in i1 and ii - 1 in i1
    ])

    ind[i0] = False
    ind[i1] = False

    energ = energ[ind]
    trans = trans[ind]

    # ----------
    # dref

    kne = f'{key}-nE'
    ne = energ.size
    dref = {
        kne: {'size': ne},
    }

    # ----------
    # ddata

    kE = f'{key}-E'
    ktrans = f'{key}-trans'

    ddata = {
        kE: {
            'data': energ,
            'ref': kne,
            'dim': 'energy',
            'quant': 'energy',
            'name': 'E',
            'units': 'eV',
        },
        ktrans: {
            'data': trans,
            'ref': kne,
            'dim': None,
            'quant': 'trans. coef.',
            'name': '',
            'units': '',
        },
    }

    # -----------
    # dmat

    dmat['energy'] = kE
    dmat['transmission'] = ktrans

    return dref, ddata, dmat
