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

    'standard - LOS': {
        'conditions': None,
        'fields': {
            'etendue': {
                'doc': 'etendue per pixel, from LOS',
            },
            # 'amin': {
            #     'doc': '',
            # },
            # 'amax': {
            #     'doc': '',
            # },
            'length': {
                'doc': 'length of the LOS',
            },
            'tangency_radius': {
                'doc': 'minimum major radius of LOS',
            },
            'alpha': {
                'doc': 'incidence angle of LOS on PFC (end point)',
            },
            'alpha_pixel': {
                'doc': 'incidence angle of LOS on pixel (start point)',
            },
            '': {
                'doc': '',
            },
        },
    },

    'broadband - vos': {
        'conditions': {'spectro': False, 'is_vos': True},
        'fields': {
            'vos_cross_rz': {
                'doc': '',
            },
            'vos_cross_vect': {
                'doc': '',
            },
            'vos_sang_integ': {
                'doc': '',
            },
        },
    },

    'spectro - vos': {
        'conditions': {'spectro': True, 'is_vos': True},
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
        'conditions': {'spectro': True},
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
        'conditions': None,
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


def get_davail(
    coll=None,
    key=None,
    key_cam=None,
    # conditions
    **kwdargs, # spectro, is_vos, is_3d
):

    # --------------
    # initialize
    # --------------

    davail = copy.deepcopy(_DAVAIL)

    # lcomp += llamb

    # -------------------------
    # update for specific diag
    # -------------------------

    lok = []
    if key is not None:

        # --------------------
        # reduce to applicable

        for k0, v0 in davail.items():

            c0 = (
                v0['conditions'] is None
                or all([
                    kwdargs.get(cc) is vv
                    for cc, vv in v0['conditions'].items()
                ])
            )

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