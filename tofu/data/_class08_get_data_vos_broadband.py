# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:33:59 2024

@author: dvezinet
"""


import numpy as np


# ##################################################################
# ##################################################################
#                   main
# ##################################################################


def main(
    coll=None,
    key=None,
    key_cam=None,
    data=None,
):

    # ---------------------
    # initialize
    # ---------------------

    static = True
    ddata, dref = {}, {}
    doptics = coll.dobj['diagnostic'][key]['doptics']

    # ---------------------
    # loop on cameras
    # ---------------------

    for cc in key_cam:

        # --------------
        # safety check

        ref = coll.dobj['camera'][cc]['dgeom']['ref']
        dvos = doptics[cc].get('dvos')
        if dvos is None:
            msg = (
                f"Data '{data}' cannot be retrived for diag '{key}' "
                "cam '{cc}' because no dvos computed"
            )
            raise Exception(msg)

        # ----------------
        # vos_pix_sang_sum

        if data == 'vos_pix_sang_integ':

            kdata = dvos['sang_cross']
            ddata[cc] = np.nansum(coll.ddata[kdata]['data'], axis=-1)
            dref[cc] = ref
            units = coll.ddata[kdata]['units']

        # --------------
        # average observation angle per RZ point per pixel in cross section

        elif data == 'vos_cross_ang':

            raise NotImplementedError()

            # solid angle map
            ksang = 'sang_cross'
            ref = coll.ddata[ksang]['ref']
            iok = np.isfinite(coll.ddata[ksang]['data'])

            # pixel centers
            # cr =
            # cz =

            # vect_cross
            ang = np.full(iok.shape, np.nan)
            sli = [None for ss in iok.shape]
            sli[-1] = slice(None)
            sli = tuple(sli)

            # ang[iok] = np.arctan2(
            #    R[sli] - cz[..., None],
            #    Z[sli] - cr[..., None],
            # )

            ddata[cc] = ang
            dref[cc] = ref
            units = coll.ddata[kdata]['units']

    return ddata, dref, units, static