# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:36:24 2024

@author: dvezinet
"""


import numpy as np


#############################################
#############################################
#       DEFAULTS
#############################################


_LORDER = [
    'algo', 'chain', 'conv_crit',
    'data_in',
    'geometry',
    'isotropic', ' matrix',
    'operator',  'positive',
    'retrofit', 'sigma_in', 'sol', 'solver',
]


#############################################
#############################################
#       Show
#############################################


def _show(coll=None, which=None, lcol=None, lar=None, show=None):

    # ---------------------------
    # column names
    # ---------------------------

    lcol.append([which] + _LORDER)

    # ---------------------------
    # data
    # ---------------------------

    lkey = [
        k1 for k1 in coll._dobj.get(which, {}).keys()
        if show is None or k1 in show
    ]

    lar0 = []
    for k0 in lkey:

        # initialize with key
        arr = [k0]

        # add nb of func of each type
        dinv = coll.dobj[which][k0]

        # loop
        for k1 in _LORDER:

            # data_in
            if k1 in ['data_in']:
                if dinv.get(k1) is None:
                    nn = ''
                elif len(dinv[k1]) <= 3:
                    nn = str(dinv[k1])
                else:
                    nn = f'[{dinv[k1][0]}, ..., {dinv[k1][-1]}]'

            # los
            else:
                nn = str(dinv.get(k1))

            arr.append(nn)

        lar0.append(arr)

    lar.append(lar0)

    return lcol, lar


#############################################
#############################################
#       Show single diag
#############################################


def _show_details(coll=None, key=None, lcol=None, lar=None, show=None):

    # ---------------------------
    # get basics
    # ---------------------------

    winv = coll._which_inversion
    ldata_in = coll.dobj[winv][key]['data_in']

    wgmat = coll._which_gmat
    key_matrix = coll.dobj[winv][key]['matrix']
    key_cam = coll.dobj[wgmat][key_matrix]['camera']
    wcam = coll._which_cam

    sigma = coll.dobj[winv][key]['sigma_in']

    # cam-specific fit chi2n
    # TODO: revize inv storing to store normalized err per channel

    # ---------------------------
    # column names
    # ---------------------------

    lcol.append([
        'camera',
        'shape',
        'data_in',
        'shape',
        'sol',
        'retrofit',
        '< delta / sigma >',
    ])

    # ---------------------------
    # data
    # ---------------------------

    lar0 = []
    for ii, kdata in enumerate(ldata_in):

        # camera
        kcam = key_cam[ii]
        arr = [kcam, str(coll.dobj[wcam][kcam]['dgeom']['shape'])]

        # data_in
        arr += [kdata, str(coll.ddata[kdata]['data'].shape)]

        # sol
        arr.append(coll.dobj[winv][key]['sol'])

        # retrofit
        kretro = f"{coll.dobj[winv][key]['retrofit']}_{key_cam[ii]}"
        arr.append(kretro)

        # delta / sigma
        data = coll.ddata[kdata]['data']
        sig = coll.ddata[kretro]['data']
        delta = sig - data
        if sigma is None:
            sigma = 1.
        elif isinstance(sigma, str):
            sigma = coll.ddata[sigma]['data']
        elif np.isscalar(sigma):
            pass
        nn = f"{np.nanmean(delta / sigma): 1.3e}"
        arr.append(nn)

        # aggregate
        lar0.append(arr)

    lar.append(lar0)

    return lcol, lar
