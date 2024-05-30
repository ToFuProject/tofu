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

        # vos_pix_sang_sum
        if data == 'vos_pix_sang_sum':
            kdata = dvos['sang_cross']
            ddata[cc] = np.nansum(coll.ddata[kdata]['data'], axis=-1)
            dref[cc] = ref
            units = coll.ddata[kdata]['units']

        #
        elif data == 'vos_cross_sang':

            # -----------------
            # get mesh sampling

            dsamp = coll.get_sample_mesh(
                key=dvos['keym'],
                res=dvos['res_RZ'],
                mode='abs',
                grid=False,
                in_mesh=True,
                # non-used
                x0=None,
                x1=None,
                Dx0=None,
                Dx1=None,
                imshow=False,
                store=False,
                kx0=None,
                kx1=None,
            )

            # -----------------
            # prepare image

            n0, n1 = dsamp['x0']['data'].size, dsamp['x1']['data'].size
            shape = (n0, n1)
            sang = np.full(shape, np.nan)
            sang_tot = np.full(shape, 0.)

        elif data == 'vos_cross_rz':

            # get indices
            # kindr, kindz = v0['dvos']['ind_cross']
            pass

        # average observation angle per RZ point per pixel in cross section
        elif data == 'vos_vect_cross_ang':

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