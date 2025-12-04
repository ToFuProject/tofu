# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:36:24 2024

@author: dvezinet
"""


#############################################
#############################################
#       DEFAULTS
#############################################


_LORDER = [
    'camera', 'data', 'diag',
    'geom_matrix', 'integrand',
    'method', 'res',
]


#############################################
#############################################
#       Show
#############################################


def _show(coll=None, which=None, lcol=None, lar=None, show=None):

    # ---------------------------
    # column names
    # ---------------------------

    wsynth = coll._which_synth_sig
    lcol.append([which] + _LORDER)

    # ---------------------------
    # list of keys
    # ---------------------------

    lkey = [
        k1 for k1 in coll._dobj.get(which, {}).keys()
        if show is None or k1 in show
    ]

    # ---------------------------
    # loop on keys
    # ---------------------------

    lar0 = []
    for k0 in lkey:

        # initialize with key
        arr = [k0]

        # dsynth
        dsynth = coll.dobj[wsynth][k0]

        # loop
        for k1 in _LORDER:

            # cameras, data
            if k1 in ['camera', 'data']:
                if dsynth.get(k1) is None:
                    nn = ''
                elif len(dsynth[k1]) <= 3:
                    nn = str(dsynth[k1])
                else:
                    nn = f'[{dsynth[k1][0]}, ..., {dsynth[k1][-1]}]'

            # los
            else:
                nn = str(dsynth[k1])

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

    wcam = coll._which_cam
    wsynth = coll._which_synth_sig
    dsynth = coll.dobj[wsynth][key]

    # ---------------------------
    # column names
    # ---------------------------

    lcol.append([wcam, 'shape', 'sig', 'shape'] + _LORDER[2:])

    # ---------------------------
    # data
    # ---------------------------

    lar0 = []
    for ii, kcam in enumerate(dsynth[wcam]):

        # camera
        arr = [kcam, str(coll.dobj[wcam][kcam]['dgeom']['shape'])]

        # data
        kdata = dsynth['data'][ii]
        arr += [kdata, str(coll.ddata[kdata]['data'].shape)]

        for k1 in _LORDER[2:]:
            nn = str(dsynth[k1])
            arr.append(nn)

        # aggregate
        lar0.append(arr)

    lar.append(lar0)

    return lcol, lar
