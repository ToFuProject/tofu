# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:17:01 2024

@author: dvezinet
"""


import numpy as np
from matplotlib import path as mpath
import datastock as ds


from ..geom import _GG
from ._class00_poly2d_sample import main as poly2d_sample


# ###############################################################
# ###############################################################
#                      Main
# ###############################################################


def main(
    coll=None,
    key=None,
    # to sample on a single optics
    key_optics=None,
    # sampling
    dsampling_pixel=None,
    dsampling_optics=None,
    # computing
    config=None,
    # storing
    store=None,
    overwrite=None,
):
    """

    Parameters
    ----------
    coll : TYPE, optional
        DESCRIPTION. The default is None.
    key : TYPE, optional
        DESCRIPTION. The default is None.
    # to sample on a single optics    key_optics : TYPE, optional
        DESCRIPTION. The default is None.
    # sampling    dsampling_pixel : TYPE, optional
        DESCRIPTION. The default is None.
    dsampling_optics : TYPE, optional
        DESCRIPTION. The default is None.
    # computing    config : TYPE, optional
        DESCRIPTION. The default is None.
    # storing    store : TYPE, optional
        DESCRIPTION. The default is None.
    overwrite : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    dout : TYPE
        DESCRIPTION.

    """

    # -------------
    # check
    # -------------

    (
     key,
     dsampling_pixel,
     dsampling_optics,
     store, overwrite,
    ) = _check(
        coll=coll,
        key=key,
        # sampling
        dsampling_pixel=dsampling_pixel,
        dsampling_optics=dsampling_optics,
        # storing
        store=store,
        overwrite=overwrite,
    )

    # ---------------
    #  prepare
    # ---------------

    wdiag = 'diagnostic'
    key_cam = coll.dobj[wdiag][key]['camera']
    doptics = coll.dobj[wdiag][key]['doptics']

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

        # if strategy == 'random':

        #     dout[kcam] = _random(
        #         coll=coll,
        #         kcam=kcam,
        #         doptics=doptics[kcam],
        #         out_arr=out_arr,
        #         nrays=nrays,
        #         strategy=strategy,
        #     )

        # elif strategy == 'outline':

        #     dout[kcam] = _outline()

        # elif strategy == 'mesh':

        #     dout[kcam] = _mesh()

        # --------------------
        # sample generic pixel

        dout[kcam] = _generic(
            coll=coll,
            kcam=kcam,
            doptics=doptics[kcam],
            # sampling
            dsampling_pixel=dsampling_pixel,
            dsampling_optics=dsampling_optics,
        )






    # ---------------
    #  store
    # ---------------

    if store is True:
        _store(
            coll=coll,
            dout=dout,
            config=config,
            overwrite=overwrite,
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
    # sampling
    dsampling_pixel=None,
    dsampling_optics=None,
    # storing
    store=None,
    overwrite=None,
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
    # dsampling_pixel
    # ------------

    lk = ['dedge', 'dsurface']
    c0 = (
        isinstance(dsampling_pixel, dict)
        and any([ss in dsampling_pixel.keys() for ss in lk])
    )
    if not c0:
        lstr = [f"\t- {kk}: dict fed to tf.data.poly2d_sample()" for kk in lk]
        msg = (
            "Arg dsampling_pixel must be a dict with at least one of:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    c0 = (
        isinstance(dsampling_optics, dict)
        and any([ss in dsampling_optics.keys() for ss in lk])
    )
    if not c0:
        lstr = [f"\t- {kk}: dict fed to tf.data.poly2d_sample()" for kk in lk]
        msg = (
            "Arg dsampling_optics must be a dict with at least one of:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # ------------
    # store
    # ------------

    store = ds._generic_check._check_var(
        store, 'store',
        types=bool,
        default=True,
    )

    # ------------
    # overwrite
    # ------------

    overwrite = ds._generic_check._check_var(
        overwrite, 'overwrite',
        types=bool,
        default=False,
    )

    return key, dsampling_pixel, dsampling_optics, store, overwrite


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
    strategy=None,
):

    # ---------------
    # prepare pixel
    # ---------------

    # outline path
    path = mpath.Path(out_arr)

    min0, max0 = np.min(out_arr[:, 0]), np.max(out_arr[:, 0])
    min1, max1 = np.min(out_arr[:, 1]), np.max(out_arr[:, 1])

    # areas to get safety factor
    area = coll.dobj['camera'][kcam]['dgeom']['pix_area']
    area_max = (max0 - min0) * (max1 - min1)

    # start point coords in pixel's plane
    start0, start1 = _seed_pix(
        path=path,
        nrays=nrays,
        nrays_factor=int(np.ceil(area_max/area) + 2),
        min0=min0,
        max0=max0,
        min1=min1,
        max1=max1,
    )

    # shared apertures ?
    pinhole = doptics['pinhole']
    parallel = coll.dobj['camera'][kcam]['dgeom']['parallel']

    # shape camera
    shape_cam = coll.dobj['camera'][kcam]['dgeom']['shape']

    # camera vector
    # get camera vectors
    dvect = coll.get_camera_unit_vectors(kcam)
    lc = ['x', 'y', 'z']
    if parallel is True:
        e0i_x, e0i_y, e0i_z = [dvect[f"e0_{kk}"] for kk in lc]
        e1i_x, e1i_y, e1i_z = [dvect[f"e1_{kk}"] for kk in lc]

    # -----------------------------------
    # prepare output
    # -----------------------------------

    shape_out = shape_cam + (nrays*nrays,)

    dout = {
        'key': f'{kcam}_rays_{strategy}',
        'start_x': np.full(shape_out, np.nan),
        'start_y': np.full(shape_out, np.nan),
        'start_z': np.full(shape_out, np.nan),
        'vect_x': np.full(shape_out, np.nan),
        'vect_y': np.full(shape_out, np.nan),
        'vect_z': np.full(shape_out, np.nan),
    }

    # -----------------------------------
    # pinhole camera (shared apertures)
    # -----------------------------------

    if pinhole is True:

        # get pixel centers
        cx, cy, cz = coll.get_camera_cents_xyz(kcam)
        cent_cam = np.r_[np.mean(cx), np.mean(cy), np.mean(cz)]

        # get end points
        end0, end1, op_cent, op_e0, op_e1 = _seed_optics(
            coll=coll,
            cent_cam=cent_cam,
            nrays=nrays,
            optics=doptics['optics'],
        )

        # full versions
        end0f = np.tile(end0, nrays)
        end1f = np.tile(end1, nrays)

        # full versions
        start0f = np.repeat(start0, end0.size)
        start1f = np.repeat(start1, end0.size)

        # vignetting polygons
        vignett_poly = [
            np.array(coll.get_optics_poly(k0, closed=True, add_points=False))
            for k0 in doptics['optics']
        ]
        lnvert = np.array([vv.shape[1] for vv in vignett_poly], dtype=np.int64)

        # loop on pixels
        ray_orig = np.full((3, nrays*end0.size), np.nan)
        ray_vdir = np.full((3, nrays*end0.size), np.nan)
        for ind in np.ndindex(shape_cam):

            sli_out = ind + (slice(None),)

            # unit vectors
            if parallel is not True:
                e0i_x, e0i_y, e0i_z = [dvect[f"e0_{kk}"][ind] for kk in lc]
                e1i_x, e1i_y, e1i_z = [dvect[f"e1_{kk}"][ind] for kk in lc]

            # ray_orig
            ray_orig[0, :] = cx[ind] + start0f * e0i_x + start1f * e1i_x
            ray_orig[1, :] = cy[ind] + start0f * e0i_y + start1f * e1i_y
            ray_orig[2, :] = cz[ind] + start0f * e0i_z + start1f * e1i_z

            # end points
            ray_vdir[0, :] = op_cent[0] + end0f * op_e0[0] + end1f * op_e1[0]
            ray_vdir[1, :] = op_cent[1] + end0f * op_e0[1] + end1f * op_e1[1]
            ray_vdir[2, :] = op_cent[2] + end0f * op_e0[2] + end1f * op_e1[2]

            # ray_vdir, normalized
            ray_vdir[0, :] = ray_vdir[0, :] - ray_orig[0, :]
            ray_vdir[1, :] = ray_vdir[1, :] - ray_orig[1, :]
            ray_vdir[2, :] = ray_vdir[2, :] - ray_orig[2, :]

            ray_norm = np.sqrt(np.sum(ray_vdir**2, axis=0))
            ray_vdir[:] = ray_vdir / ray_norm

            # vignetting (npoly, nrays) is wrong !!!
            # iok = _GG.vignetting(
            #     ray_orig,
            #     ray_vdir,
            #     vignett_poly,
            #     lnvert,
            #     num_threads=16,
            # )
            # print(ind, iok.sum(), iok.size)

            indrand = (np.random.randint(0, ray_orig.shape[1], nrays*nrays),)

            dout['start_x'][sli_out] = ray_orig[(0,) + indrand]
            dout['start_y'][sli_out] = ray_orig[(1,) + indrand]
            dout['start_z'][sli_out] = ray_orig[(2,) + indrand]
            dout['vect_x'][sli_out] = ray_vdir[(0,) + indrand]
            dout['vect_y'][sli_out] = ray_vdir[(1,) + indrand]
            dout['vect_z'][sli_out] = ray_vdir[(2,) + indrand]

    # ---------------
    # mesh pixels
    # ---------------

    else:

        for ind in np.ndindex(shape_cam):

            print(ind)


    return dout


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

    out0, out1 = coll.get_optics_outline(kop, add_points=False, closed=False)

    path = mpath.Path(np.array([out0, out1]).T)

    min0, max0 = np.min(out0), np.max(out0)
    min1, max1 = np.min(out1), np.max(out1)

    area = coll.dobj[cc][kop]['dgeom']['area']
    factor = int(np.ceil((max0 - min0) * (max1 - min1) / area) + 2)

    seed0 = np.random.random((nrays * factor,))
    seed1 = np.random.random((nrays * factor,))

    pts0 = (max0 - min0) * seed0 + min0
    pts1 = (max1 - min1) * seed1 + min1

    # only keep those in the pixel
    iok_pix = path.contains_points(np.array([pts0, pts1]).T).nonzero()[0]

    # imax
    imax = min(nrays*2, iok_pix.size)

    # optic cent and vectors
    cent = coll.dobj[cc][kop]['dgeom']['cent']
    e0 = coll.dobj[cc][kop]['dgeom']['e0']
    e1 = coll.dobj[cc][kop]['dgeom']['e1']

    return pts0[iok_pix[:imax]], pts1[iok_pix[:imax]], cent, e0, e1


# ###############################################################
# ###############################################################
#                      generic
# ###############################################################


def _generic(
    coll=None,
    kcam=None,
    doptics=None,
    # sampling options
    dsampling_pixel=None,
    dsampling_optics=None,
):

    # ---------------------
    # generic pixel outline
    # ---------------------

    # get pixel outline
    kout0, kout1 = coll.dobj['camera'][kcam]['outline']

    # sample pixel
    dout_pixel = poly2d_sample(
        coll.ddata[kout0]['data'],
        coll.ddata[kout1]['data'],
        dedge=dsampling_pixel.get('dedge'),
        dsurface=dsampling_pixel.get('dsurface'),
    )

    nstart = dout_pixel['x0'].size

    # ---------------------
    # number of pts on optics
    # ---------------------

    nvect = optics_nb[0] * optics_nb[1]

    # ---------------------
    # prepare
    # ---------------------

    # shared apertures ?
    pinhole = doptics['pinhole']
    parallel = coll.dobj['camera'][kcam]['dgeom']['parallel']

    # shape camera
    shape_cam = coll.dobj['camera'][kcam]['dgeom']['shape']

    # camera vector
    # get camera vectors
    dvect = coll.get_camera_unit_vectors(kcam)
    lc = ['x', 'y', 'z']
    if parallel is True:
        e0i_x, e0i_y, e0i_z = [dvect[f"e0_{kk}"] for kk in lc]
        e1i_x, e1i_y, e1i_z = [dvect[f"e1_{kk}"] for kk in lc]

    cents_x, cents_y, cents_z = coll.get_camera_cents_xyz(kcam)

    # -----------------------------------
    # prepare output
    # -----------------------------------

    shape_out = shape_cam + (nrays,)

    dout = {
        'key': f'{kcam}_rays',
        'start_x': np.full(shape_out, np.nan),
        'start_y': np.full(shape_out, np.nan),
        'start_z': np.full(shape_out, np.nan),
        'vect_x': np.full(shape_out, np.nan),
        'vect_y': np.full(shape_out, np.nan),
        'vect_z': np.full(shape_out, np.nan),
        # 'dsang': np.full(shape_out, np.nan),
    }

    # ---------------------
    # pinhole with 1 optic
    # ---------------------

    if len(doptics['optics']) == 1:

        kop = doptics['optics'][0]
        clsop = coll.get_optics_cls(kop)
        out0, out1 = coll.dobj[clsop][kop]['outline']
        out0 = coll.ddata[out0]['data']
        out1 = coll.ddata[out1]['data']

        dout_pin_edge = poly2d_sample_edge(out0, out1, nb=optics_nb)
        dout_pin_surf = poly2d_sample_surface(out0, out1, nb=optics_nb)

    # ---------------------
    # loop on pixels
    # ---------------------

    for ii, ind in enumerate():

        if parallel is False:
            nin = None

        start_x = cents_x[ind] + pts0 * e0[0] + pts1 * e1[0]
        start_y = cents_y[ind] + pts0 * e0[1] + pts1 * e1[1]
        start_z = cents_z[ind] + pts0 * e0[2] + pts1 * e1[2]

        for jj in range(pts0.size):

            # get projected polygon
            continue




            # ------------
            # solid angles

            # solid_angles[ind] = _comp_solidangles.calc_solidangle_apertures(
            #     # observation points
            #     pts_x=centsx[ind],
            #     pts_y=centsy[ind],
            #     pts_z=centsz[ind],
            #     # polygons
            #     apertures=None,
            #     detectors=ldet[ii],
            #     # possible obstacles
            #     config=None,
            #     # parameters
            #     visibility=False,
            #     return_vector=False,
            #     # timing
            #     timing=False,
            # )[0, 0]

    # ---------------
    # optics
    # ---------------

    # -------------
    # adjust
    # -------------

    nok = np.sum(np.isfinite(start_x), axis=-1)
    if nok < nrays:
        pass

    # ---------------
    # return
    # ---------------

    # dout = {
    #     'key': f'{kcam}_rays_generic',
    #     'start_x': np.full(shape_out, np.nan),
    #     'start_y': np.full(shape_out, np.nan),
    #     'start_z': np.full(shape_out, np.nan),
    #     'vect_x': np.full(shape_out, np.nan),
    #     'vect_y': np.full(shape_out, np.nan),
    #     'vect_z': np.full(shape_out, np.nan),
    # }

    return # dout


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


def _store(
    coll=None,
    kdiag=None,
    dout=None,
    config=None,
    overwrite=None,
):

    # -----------------
    # add ref
    # -----------------

    nrays = list(dout.values())[0]['start_x'].shape[-1]

    kref = f"{kdiag}_nrays"
    coll.add_ref(key=kref, size=nrays)

    # --------------
    # add rays
    # --------------

    for kcam, v0 in dout.items():

        ref = coll.dobj['camera'][kcam]['dgeom']['ref'] + (kref,)
        coll.add_rays(
            ref=ref,
            config=config,
            **v0
        )

    return