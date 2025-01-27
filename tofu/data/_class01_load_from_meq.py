# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:24:53 2025

@author: dvezinet
"""



# Built-in
import os
import warnings


# Common
import numpy as np


__all_ = ['load_meq']


# ########################################################
# ########################################################
#               Units
# ########################################################


_DUNITS = {
    # ------------
    # str / scalar
    'Brx': {
        'key': 'BR',
        'units': 'T',
        'ref': ('neq', 'mRZ'),
        'transpose': True,
    },
    'Bzx': {
        'key': 'BZ',
        'units': 'T',
        'ref': ('neq', 'mRZ'),
        'transpose': True,
    },
    'Btx': {
        'key': 'Bphi',
        'units': 'T',
        'ref': ('neq', 'mRZ'),
        'transpose': True,
    },
    # Poloidal flux
    'Fx': {
        'key': 'psi',
        'units': 'Wb',
        'ref': ('neq', 'mRZ'),
        'transpose': True,
    },
    'FA': {
        'key': 'psi_axis',
        'units': 'Wb',
        'ref': 'neq',
    },
    'FB': {
        'key': 'psi_sep',
        'units': 'Wb',
        'ref': 'neq',
    },
    # Magnetic axis
    'rA': {
        'key': 'magaxR',
        'units': 'm',
        'ref': 'neq',
    },
    'zA': {
        'key': 'magaxZ',
        'units': 'm',
        'ref': 'neq',
    },
    # Separatrix
    'rS': {
        'key': 'sepR',
        'units': 'm',
        'ref': ('neq', 'nsep'),
    },
    'zS': {
        'key': 'sepZ',
        'units': 'm',
        'ref': ('neq', 'nsep'),
    },
    # Others
    'Ip': {
        'key': 'Ip',
        'units': 'A',
        'ref': 'neq',
    },
    # 'aminor': {
    #     'key': 'r',
    #     'units': 'm',
    #     'ref': ('neq', 'mrhotn'),
    # },
    'q95': {
        'key': 'q95',
        'units': None,
        'ref': 'neq',
    },
    'qA': {
        'key': 'qax',
        'units': None,
        'ref': 'neq',
    },
    'qmin': {
        'key': 'qmin',
        'units': None,
        'ref': 'neq',
    },
    'Vp': {
        'key': 'Vp',
        'units': 'm3',
        'ref': 'neq',
    },
    # 'rhotornorm': {
    #     'key': 'rhotn',
    #     'units': None,
    #     'ref': ('neq', 'mrhotn'),
    # },
    'shot': {
        'key': 'shot',
        'units': None,
        'ref': 'neq',
    },
    't': {
        'key': 't',
        'units': 's',
        'ref': 'neq',
    },
}


for k0, v0 in _DUNITS.items():
    _DUNITS[k0]['key'] = _DUNITS[k0].get('key', k0)
    if isinstance(v0['ref'], str):
        _DUNITS[k0]['ref'] = (v0['ref'],)


# ########################################################
# ########################################################
#               check shapes
# ########################################################


def get_load_pfe():

    # -----------------
    # check dependency
    # -----------------

    try:
        import scipy.io as scpio
    except Exception as err:
        msg = (
            "loading an mat file requires an optional dependency:\n"
            "\t- file trying to load: {pfe}\n"
            "\t- required dependency: scipy.io"
        )
        err.args = (msg,)
        raise err

    # -----------------
    # define load_pfe
    # -----------------

    def func(pfe):

        dout = scpio.loadmat(pfe)
        dout = {
            k0: v0
            for k0, v0 in dout.items()
            if (
                (not k0.startswith('__'))
                and (
                    (isinstance(v0, np.ndarray) and v0.size > 0)
                    or not isinstance(v0, np.ndarray)
                )
            )
        }

        return dout

    return func


# ########################################################
# ########################################################
#               extract grid
# ########################################################


def _extract_grid(dout, kmesh):

    # -------------------
    # preliminary checks
    # -------------------


    # -------------------
    # get 2d mesh
    # -------------------

    # extract nb of knots
    nZ, nR = dout['Brx'].shape

    # extract R
    R = np.linspace(1.24, 2.45, nR)

    # extract Z
    Z = np.linspace(-1.6, 1.6, nZ)

    # -------------------
    # get 1d mesh
    # -------------------

    # rhotn =  dout['rhotornorm']

    # -------------------
    # package
    # -------------------

    dmesh = {
        'mRZ': {
            'key': kmesh['mRZ'],
            'knots0': R,
            'knots1': Z,
            'units': ['m', 'm'],
            'deg': 1,
        },
        # 'mrhotn': {
        #     'key': kmesh['mrhotn'],
        #     'knots': rhotn,
        #     'units': None,
        #     'deg': 1,
        # },
    }

    return dmesh