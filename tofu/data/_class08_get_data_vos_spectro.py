# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:40:00 2024

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

    static = True
    ddata, dref = {}, {}
    doptics = coll.dobj['diagnostic'][key]['doptics']

    for cc in key_cam:

        # safety check
        ref = coll.dobj['camera'][cc]['dgeom']['ref']
        dvos = doptics[cc].get('dvos')
        if dvos is None:
            msg = (
                f"Data '{data}' cannot be retrived for diag '{key}' "
                "cam '{cc}' because no dvos computed"
            )
            raise Exception(msg)

        # vos-derived wavelength quantities
        if data in ['vos_lamb', 'vos_dlamb', 'vos_ph_integ']:

            kph = dvos['ph']
            ph = coll.ddata[kph]['data']
            ph_tot = np.sum(ph, axis=(-1, -2))

            if data == 'vos_ph_integ':
                out = ph_tot
                kout = kph
            else:
                kout = dvos['lamb']
                re_lamb = [1 for rr in ref] + [1, -1]
                lamb = coll.ddata[kout]['data'].reshape(re_lamb)

                i0 = ph == 0
                if data == 'vos_lamb':
                    out = np.sum(ph * lamb, axis=(-1, -2)) / ph_tot
                else:
                    for ii, i1 in enumerate(re_lamb[:-1]):
                        lamb = np.repeat(lamb, ph.shape[ii], axis=ii)

                    lamb[i0] = -np.inf
                    lambmax = np.max(lamb, axis=(-1, -2))
                    lamb[i0] = np.inf
                    lambmin = np.min(lamb, axis=(-1, -2))
                    out = lambmax - lambmin
                out[np.all(i0, axis=(-1, -2))] = np.nan

            ddata[cc] = out
            dref[cc] = ref
            units = coll.ddata[kout]['units']

    return ddata, dref, units, static