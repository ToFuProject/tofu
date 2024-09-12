# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:17:01 2024

@author: dvezinet
"""


import numpy as np
from matplotlib import path as mpath
import datastock as ds


# ###############################################################
# ###############################################################
#                      Main
# ###############################################################


def main(
    coll=None,
    key=None,
    strategy=None,
    nrays=None,
    # storing
    store=None,
):

    # -------------
    # check
    # -------------

    key, strategy, nrays, store = _check(
        coll=coll,
        key=key,
        strategy=strategy,
        nrays=nrays,
        # storing
        store=store,
    )

    # ---------------
    #  prepare
    # ---------------

    wdiag = 'diagnostic'
    key_cam = coll.dobj[wdiag][key]['camera']
    doptics = coll.dobj[wdiag][key]['doptics']

    # -------------------------
    # trivial: nrays=1 => LOS
    # -------------------------

    if nrays == 1:

        dout = {
            kcam: {}
            for kcam in key_cam
        }

        store = False

    # ---------------
    #  compute
    # ---------------

    # initialize
    dout = {}

    # loop on cameras
    for kcam in key_cam:

        # --------------
        # prepare pixel outline

        out0, out1 = coll.dobj['camera'][kcam]['dgeom']['outline']
        out0 = coll.ddata[out0]['data']
        out1 = coll.ddata[out1]['data']
        out_arr = np.array([out0, out1]).T

        # ---------------
        # call routine

        if strategy == 'random':

            dout[kcam] = _random(
                coll=coll,
                kcam=kcam,
                doptics=doptics[kcam],
                out_arr=out_arr,
                nrays=nrays,
            )

        elif strategy == 'outline':

            dout[kcam] = _outline()

        elif strategy == 'mesh':

            dout[kcam] = _mesh()

    # ---------------
    #  store
    # ---------------

    if store is True:
        _store(
            coll=coll,
            dout=dout,
        )

    else:
        return dout


# ###############################################################
# ###############################################################
#                      check
# ###############################################################


def _check(
    coll=None,
    key=None,
    strategy=None,
    nrays=None,
    # storing
    store=None,
    store_key=None,
):

    # ------------
    # key
    # ------------

    wdiag = 'diagnostic'
    lok = list(coll.dobj.get(wdiag, {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # spectro ?
    spectro = coll.dobj[wdiag][key]['spectro']
    if spectro is True:
        raise NotImplementedError()

    # ------------
    # strategy
    # ------------

    strategy = ds._generic_check._check_var(
        strategy, 'strategy',
        types=str,
        default='random',
        allowed=['random', 'mesh', 'outline'],
    )

    # ------------
    # nrays
    # ------------

    if strategy == 'custom':
        nrdef = 10
    else:
        nrdef = 10


    nrays = int(ds._generic_check._check_var(
        nrays, 'nrays',
        types=(int, float),
        default=nrdef,
        sign='>0',
    ))

    # ------------
    # store
    # ------------

    store = ds._generic_check._check_var(
        store, 'store',
        types=bool,
        default=False,
    )

    return key, strategy, nrays, store


# ###############################################################
# ###############################################################
#                      random
# ###############################################################


def _random(
    coll=None,
    kcam=None,
    doptics=None,
    out_arr=None,
    nrays=None,
):

    # ---------------
    # prepare pixel
    # ---------------

    # outline path
    path = mpath.Path(out_arr)

    min0, max0 = np.min(out_arr[:, 0]), np.max(out_arr[:, 0])
    min1, max1 = np.min(out_arr[:, 1]), np.max(out_arr[:, 1])

    # areas to get safety factor
    area = doptics['pix_area']
    area_max = (max0 - min0) * (max1 - min1)

    # start point coords in pixel's plane
    start0, start1 = _seed_pix(
        path=path,
        nrays=nrays,
        nrays_factor=np.ceil(area_max/area) + 2,
        min0=min0,
        max0=max0,
        min1=min1,
        max1=max1,
    )

    # shared apertures ?
    pinhole = doptics['pinhole']

    # shape camera
    shape_cam = coll.dobj['camera'][kcam]['dgeom']['shape']

    # -----------------------------------
    # pinhole camera (shared apertures)
    # -----------------------------------

    if pinhole is True:

        cx, cy, cz = coll.get_camera_cents_xyz(kcam)
        cent_cam = np.r_[np.mean(cx), np.mean(cy), np.mean(cz)]

        end0, end1 = _seed_optics(
            coll=coll,
            cent_cam=cent_cam,
            nrays=nrays,
            optics=doptics['optics'],
            cent_cam=coll.get_camera_cents_xyz(kcam),
        )

        for ind in np.ndindex(shape_cam):
            pass



    # ---------------
    # mesh pixels
    # ---------------

    else:

        for ind in np.ndindex(shape_cam):

            print(ind)


    return


# #########################
# seeding pixels
# #########################


def _seed_pix(
    path=None,
    nrays=None,
    nrays_factor=None,
    min0=None,
    max0=None,
    min1=None,
    max1=None,
):

    seed0 = np.random.random((nrays*nrays_factor,))
    seed1 = np.random.random((nrays*nrays_factor,))

    pts0 = (max0 - min0) * seed0 + min0
    pts1 = (max1 - min1) * seed1 + min1

    # only keep those in the pixel
    iok_pix = path.contains_points(np.array([pts0, pts1]).T).nonzero()[0]

    # imax
    imax = min(nrays, iok_pix.size)

    return pts0[iok_pix[:imax]], pts1[iok_pix[:imax]]


# #########################
# seeding optics
# #########################


def _seed_optics(
    coll=None,
    cent_cam=None,
    nrays=None,
    optics=None,
):

    # -----------------------------
    # select smallest etendue optic
    # -----------------------------

    optics, optics_cls = coll.get_optics_cls(optics)

    etend = [
        (
            coll.dobj[cc][kop]['dgeom']['area']
            / np.linalg.norm(coll.dobj[cc][kop]['dgeom']['cent'] - cent_cam)**2
        )
        for kop, cc in zip(optics, optics_cls)
    ]

    ind = np.nanargmin(etend)
    kop = optics[ind]
    cc = optics_cls[ind]

    # -----------------------------
    # seed
    # -----------------------------

    out0, out1 = coll.get_optics_outline(kop)

    path = mpath.Path(np.array([out0, out1]).T)

    min0, max0 = np.min(out0), np.max(out0)
    min1, max1 = np.min(out1), np.max(out1)

    area = coll.dobj[cc][kop]['dgeom']['area']
    factor = np.ceil((max0 - min0) * (max1 - min1) / area) + 2

    seed0 = np.random.random((nrays * factor,))
    seed1 = np.random.random((nrays * factor,))

    pts0 = (max0 - min0) * seed0 + min0
    pts1 = (max1 - min1) * seed1 + min1

    # only keep those in the pixel
    iok_pix = path.contains_points(np.array([pts0, pts1]).T).nonzero()[0]

    # imax
    imax = min(nrays*2, iok_pix.size)

    return pts0[iok_pix[:imax]], pts1[iok_pix[:imax]]


# ###############################################################
# ###############################################################
#                      outline
# ###############################################################


def _outline():


    return


# ###############################################################
# ###############################################################
#                      mesh
# ###############################################################


def _mesh():


    return


# ###############################################################
# ###############################################################
#                      store
# ###############################################################


def _store():


    return