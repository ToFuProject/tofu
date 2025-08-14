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
    'is2d', 'spectro', 'PHA',
    'camera', 'signal',
    'los', 'vos',
    'nb geom_matrix',
]


#############################################
#############################################
#       Show
#############################################


def _show(coll=None, which=None, lcol=None, lar=None, show=None):

    # ---------------------------
    # column names
    # ---------------------------

    wcam = coll._which_cam
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
        ddiag = coll.dobj[which][k0]
        lcam = ddiag[wcam]

        # loop
        for k1 in _LORDER:

            # parameters
            if k1 in ['is2d', 'spectro', 'PHA', 'nb geom_matrix']:
                nn = str(ddiag.get(k1))

            # cameras, signal
            elif k1 in ['camera', 'signal']:
                if ddiag.get(k1) is None:
                    nn = ''
                elif len(ddiag[k1]) <= 3:
                    nn = str(ddiag[k1])
                else:
                    nn = f'[{ddiag[k1][0]}, ..., {ddiag[k1][-1]}]'

            # los
            elif k1 == 'los':
                nlos = len([
                    kcam for kcam in lcam
                    if ddiag['doptics'][kcam].get('los') is not None
                ])
                nn = f"{nlos} / {len(lcam)}"

            # vos
            elif k1 == 'vos':
                dproj = coll.check_diagnostic_vos_proj(k0)
                lproj = []
                partial = 0
                for kproj in ['3d', 'cross', 'hor']:
                    if all([kcam in dproj[kproj] for kcam in lcam]):
                        lproj.append(kproj)
                    elif any([kcam in dproj[kproj] for kcam in lcam]):
                        partial = True

                if len(lproj) > 0:
                    nn = ', '.join(lproj)
                elif partial is True:
                    nn = 'partial'
                else:
                    nn = 'False'

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

    # ---------------------------
    # column names
    # ---------------------------

    lcol.append([wcam, '2d', 'pinhole', 'optics', 'los', 'vos'])

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

        # vos
        nn = ''
        arr.append(nn)

        # aggregate
        lar0.append(arr)

    lar.append(lar0)

    return lcol, lar
