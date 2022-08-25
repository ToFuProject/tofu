# -*- coding: utf-8 -*-


import numpy as np
import scipy.constants as scpct
import datastock as ds


_DMAT_KEYS = {
    'name': {'types': str},
    'symbol': {'types': str},
    'thickness': {'types': float, 'sign': '> 0.'},
    'energy': {'dtype': float, 'sign': '> 0.'},
    'transmission': {'dtype': float, 'sign': ['>= 0.', '<= 1.']},
}


# #############################################################################
# #############################################################################
#                       dmat for filter
# #############################################################################


def _trim_1d(y=None, val=None, trim=None):

    # check input
    if trim is None:
        trim = 'fb'

    if 'f' not in trim and 'b' not in trim:
        return

    # compute
    iv = (y == val).nonzero()[0]

    ind = None
    if iv.size > 0:

        if 'f' in trim:
            i0 = np.array([
                ii for ii in range(iv[-1])
                if ii in iv and ii + 1 in iv
            ])
            if i0.size > 0:
                ind = i0

        if 'b' in trim:
            i1 = np.sort([
                ii for ii in range(y.size - 1, iv[0] - 1, -1)
                if ii in iv and ii - 1 in iv
            ])

            if i1.size > 0:
                if ind is None:
                    ind = i1
                else:
                    ind = np.unique(np.r_[ind, i1])

    return ind


def _dmat_energy_trans(energ=None, trans=None):

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

    # remove trailing 0 and 1 in trans
    ind0 = _trim_1d(y=trans, val=0, trim='fb')
    ind1 = _trim_1d(y=trans, val=1, trim='fb')

    ind = None
    if ind0 is not None:
        ind = ind0
    if ind1 is not None:
        if ind0 is None:
            ind = ind1
        else:
            ind = np.r_[ind0, ind1]

    if ind is not None:
        energ = np.delete(energ, np.r_[ind0, ind1])
        trans = np.delete(trans, np.r_[ind0, ind1])

    return energ, trans


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

    dmat['energy'], dmat['transmission'] = _dmat_energy_trans(
        energ=dmat['energy'],
        trans=dmat['transmission'],
    )

    # ----------
    # dref

    kne = f'{key}-nE'
    ne = dmat['energy'].size
    dref = {
        kne: {'size': ne},
    }

    # ----------
    # ddata

    kE = f'{key}-E'
    ktrans = f'{key}-trans'

    ddata = {
        kE: {
            'data': dmat['energy'],
            'ref': kne,
            'dim': 'energy',
            'quant': 'energy',
            'name': 'E',
            'units': 'eV',
        },
        ktrans: {
            'data': dmat['transmission'],
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
