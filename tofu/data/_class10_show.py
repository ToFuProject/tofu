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

    wcam = coll._which_cam
    wdiag = coll._which_diagnostic
    lcam = coll.dobj[wdiag][key][wcam]
    doptics = coll.dobj[wdiag][key]['doptics']
    dproj = coll.check_diagnostic_vos_proj()
    lproj = ['cross', 'hor', '3d']

    # ---------------------------
    # column names
    # ---------------------------

    lcol.append([
        wcam, '2d', 'pinhole', 'optics',
        'los', 'vos_proj', 'vos_resRZPhi',
    ])

    # ---------------------------
    # data
    # ---------------------------

    lar0 = []
    for kcam in lcam:

        din = doptics[kcam]

        # initialize with key, type
        arr = [kcam]

        # is2d
        arr.append(str(coll.dobj[wcam][kcam]['dgeom']['nd'] == '2d'))

        # pinhole
        arr.append(str(din['pinhole']))

        # optics
        if len(din['optics']) > 5:
            nn = (
                f"[{din['optics'][0]}, {din['optics'][1]}, "
                "..., "
                f"{din['optics'][-2]}, {din['optics'][-1]}]"
            )
        else:
            nn = str(din['optics'])
        arr.append(nn)

        # los
        if din.get('los') is None:
            nn = 'False'
        else:
            nn = din['los']
        arr.append(nn)

        # vos_proj
        nn = ', '.join([pp for pp in lproj if kcam in dproj[pp]])
        arr.append(nn)

        # vos_res
        if doptics[kcam].get('dvos', {}).get('keym') is None:
            nn = ''
        else:
            nn = (
                doptics[kcam]['dvos']['res_RZ']
                + [doptics[kcam]['dvos']['res_phi']]
            )
            nn = str(tuple(nn))
        arr.append(nn)

        # aggregate
        lar0.append(arr)

    lar.append(lar0)

    return lcol, lar
