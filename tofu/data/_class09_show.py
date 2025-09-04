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
    'bsplines', 'diagnostic', 'camera',
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

    wbs = coll._which_bsplines
    wcam = coll._which_cam
    wdiag = coll._which_diagnostic
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

        # din
        din = coll.dobj[which][k0]

        # initialize with key
        arr = [k0]

        # loop
        for k1 in _LORDER:

            # parameters
            if k1 in [wbs, wdiag, 'method', 'res']:
                nn = str(din.get(k1))

            # cameras, signal
            elif k1 in [wcam]:
                lcam = din[wcam]
                if len(lcam) <= 5:
                    nn = str(lcam)
                else:
                    nn = f'[{lcam[0]}, {lcam[1]}, ..., {lcam[-2]}, {lcam[-1]}]'

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

    wgmat = coll._which_gmat
    wcam = coll._which_cam
    wbs = coll._which_bsplines
    lcam = coll.dobj[wgmat][key][wcam]

    kbs = coll.dobj[wgmat][key][wbs]
    method = coll.dobj[wgmat][key]['method']
    res = str(coll.dobj[wgmat][key]['res'])

    # ---------------------------
    # column names
    # ---------------------------

    lcol.append([wcam, wbs, 'shape', 'method', 'res', 'data'])

    # ---------------------------
    # data
    # ---------------------------

    lar0 = []
    for icam, kcam in enumerate(lcam):

        # initialize with key, type
        arr = [kcam]

        # bsplines
        arr.append(kbs)

        # shape
        kdata = coll.dobj[wgmat][key]['data'][icam]
        arr.append(str(coll.ddata[kdata]['shape']))

        # method
        arr.append(method)

        # res
        arr.append(res)

        # data
        arr.append(kdata)

        # aggregate
        lar0.append(arr)

    lar.append(lar0)

    return lcol, lar
