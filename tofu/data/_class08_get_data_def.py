# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:45:21 2024

@author: dvezinet
"""

import copy


# ##################################################################
# ##################################################################
#                   Default dict
# ##################################################################


_DAVAIL = {

    'standard - stored': {
        'conditions': [],
        'fields': {
            'etendue': {
                'doc': 'etendue per pixel',
            },
            'amin': {
                'doc': '',
            },
            'amax': {
                'doc': '',
            },
        },
    },

    'standard - computed': {
        'conditions': [],
        'fields': {
            'length': {
                'doc': 'length of the LOS',
            },
            'tangency radius': {
                'doc': '',
            },
            'alpha': {
                'doc': '',
            },
            'alpha_pixel': {
                'doc': '',
            },
            '': {
                'doc': '',
            },
        },
    },

    'broadband - vos': {
        'conditions': [],
        'fields': {
            'vos_cross_rz': {
                'doc': '',
            },
            'vos_sang_integ': {
                'doc': '',
            },
            'vos_vect_cross': {
                'doc': '',
            },
        },
    },

    'spectro - vos': {
        'conditions': [],
        'fields': {
            'vos_lamb': {
                'doc': '',
            },
            'vos_dlamb': {
                'doc': '',
            },
            'vos_ph_integ': {
                'doc': '',
            },
        },
    },

    'spectro - lamb': {
        'conditions': [],
        'fields': {
            'lamb': {
                'doc': 'reference wavelength, from LOS, per pixel',
            },
            'lambmin': {
                'doc': 'minimal wavelgenth, from LOS, per pixel',
            },
            'lambmax': {
                'doc': 'minimal wavelgenth, from LOS, per pixel',
            },
            'dlamb': {
                'doc': 'wavelength span, from LOS, per pixel',
            },
            'res': {
                'doc': ', from LOS, per pixel',
            },
        },
    },

    '': {
        'conditions': [],
        'fields': {
            '': {
                'doc': '',
            },
        },
    },
}


# ##################################################################
# ##################################################################
#                   function
# ##################################################################


def get_davail(key=None, spectro=None, is_vos=None, is_3d=None, coll=None):

    # --------------
    # initialize
    # --------------

    davail = copy.deepcopy(_DAVAIL)

    lcomp += llamb

    # -------------------------
    # update for specific diag
    # -------------------------

    lok = []
    if key is not None:

        # --------------------
        # reduce to applicable

        for k0, v0 in davail.items():

            c0 = all([])

            if c0:
                lok.append(k0)

        # update
        davail = {k0: davail[k0] for k0 in lok}

        # ----------------
        # add raw data

        if len(key_cam) == 1:
            lraw = [
                k0 for k0, v0 in coll.ddata.items()
                if v0['ref'] == coll.dobj['camera'][key_cam[0]]['dgeom']['ref']
            ]
        else:
            lraw = []

        # ------------------
        # add synthetic data

        lsynth = coll.dobj['diagnostic'][key]['signal']

        if lsynth is None:
            lsynth = []

    return davail