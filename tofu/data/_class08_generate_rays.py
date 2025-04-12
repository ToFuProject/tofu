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
    # optics (to restrain to certain optics only for faster)
    optics=None,
    # computing
    config=None,
    # storing
    store=None,
    strict=None,
    key_rays=None,
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
     optics,
     store, strict, key_rays, overwrite,
    ) = _check(
        coll=coll,
        key=key,
        # sampling
        dsampling_pixel=dsampling_pixel,
        dsampling_optics=dsampling_optics,
        # optics
        optics=optics,
        # storing
        store=store,
        strict=strict,
        key_rays=key_rays,
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

        # --------------------
        # sample generic pixel

        dout[kcam] = _generic(
            coll=coll,
            key=key,
            kcam=kcam,
            doptics=doptics[kcam],
            # sampling
            dsampling_pixel=dsampling_pixel,
            dsampling_optics=dsampling_optics,
            # optics
            optics=optics.get(kcam),
        )

    # ---------------
    #  store
    # ---------------

    if store is True:
        _store(
            coll=coll,
            dout=dout,
            key_rays=key_rays,
            config=config,
            strict=strict,
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
    # optics
    optics=None,
    # storing
    store=None,
    strict=None,
    key_rays=None,
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

    key_cam = coll.dobj[wdiag][key]['camera']

    # spectro ?
    spectro = coll.dobj[wdiag][key]['spectro']
    if spectro is True:
        raise NotImplementedError()

    # ------------
    # dsampling_pixel
    # ------------

    ldict = [
        ('dsampling_pixel', dsampling_pixel),
        ('dsampling_optics', dsampling_optics),
    ]
    lk = [('dedge', ['res', 'factor']), ('dsurface', ['res', 'nb'])]
    kfunc = 'tf.data.poly2d_sample()'
    for ii, (kdict, vdict) in enumerate(ldict):
        c0 = (
            isinstance(vdict, dict)
            and any([kk in vdict.keys() for (kk, largs) in lk])
        )
        if not c0:
            lstr = [f"\t- {kk}: dict of {largs}" for (kk, largs) in lk]
            msg = (
                f"Arg '{kdict}' must be a dict with at least one of:\n"
                + "\n".join(lstr)
                + f"\nFed to {kfunc}\n"
                + f"\nProvided:\n{vdict}\n"
            )
            raise Exception(msg)

    # ------------
    # optics
    # ------------

    if optics is not None:

        # index => position in optics list
        optics0 = {
            kc: _check_optics_for_kcam(
                doptics=coll.dobj[wdiag][key]['doptics'][kc],
                optics=optics.get(kc) if isinstance(optics, dict) else optics,
                shape_cam=coll.dobj['camera'][kc]['dgeom']['shape'],
            )
            for kc in key_cam
        }

    else:
        optics0 = {}

    # -----------
    # store
    # ------------

    store = ds._generic_check._check_var(
        store, 'store',
        types=bool,
        default=True,
    )

    # -----------
    # strict
    # ------------

    strict = ds._generic_check._check_var(
        strict, 'stict',
        types=bool,
        default=False,
    )

    # ------------
    # overwrite
    # ------------

    overwrite = ds._generic_check._check_var(
        overwrite, 'overwrite',
        types=bool,
        default=False,
    )

    # ------------
    # key_rays
    # ------------

    if store is True:

        if key_rays is None:
            key_rays = {kcam: f"{kcam}_rays" for kcam in key_cam}

        else:
            if len(key_cam) == 1 and isinstance(key_rays, str):
                key_rays = {key_cam[0]: key_rays}

        ncam = len(key_cam)
        c0 = (
            isinstance(key_rays, dict)
            and all([
                isinstance(key_rays.get(kcam), str)
                and len(set([key_rays[kcam] for kcam in key_cam])) == ncam
                and all([key_rays[kcam] not in coll.dobj['camera'].keys()])
                for kcam in key_cam
            ])
        )
        if not c0:
            msg = (
                "Arg key_rays must be a dict of key_ray for each camera:\n"
                f"\t- key_diag = '{key}'\n"
                f"\t- key_cam = {key_cam}\n"
                f"\t- key_rays = {key_rays}\n"
            )
            raise Exception(msg)

    else:
        key_rays = None

    return (
        key,
        dsampling_pixel, dsampling_optics,
        optics0,
        store, strict, key_rays, overwrite,
    )


def _check_optics_for_kcam(
    doptics=None,
    optics=None,
    shape_cam=None,
):

    if isinstance(optics, int):

        ind0 = np.arange(0, len(doptics['optics']))
        if doptics['pinhole'] is True:
            lok = np.r_[ind0, -1]
        else:
            nop = doptics['paths'].sum(axis=1)
            lok = np.r_[
                np.arange(0, np.min(nop)),
                -np.arange(1, np.min(nop)+1),
            ]

        optics = ds._generic_check._check_var(
            optics, 'optics',
            types=int,
            allowed=lok,
        )

        if doptics['pinhole'] is True:
            optics = [doptics['optics'][optics]]
        else:
            assert doptics.get('paths') is not None
            paths = doptics['paths']

            optics = [
                doptics['optics'][ind0[paths[ii, :]][optics]]
                for ii in range(shape_cam[0])
            ]

    # -------------------
    # key or list of keys

    if isinstance(optics, str):
        optics = [optics]

    optics = ds._generic_check._check_var_iter(
        optics, 'optics',
        types=(list, tuple),
        types_iter=str,
        allowed=doptics['optics'],
    )

    # eliminate redundants (ex. collimator with 1 common optic)
    return list(set(optics))


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
        # vignett_poly = [
            # np.array(coll.get_optics_poly(k0, closed=True, add_points=False))
            # for k0 in doptics['optics']
        # ]

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
    key=None,
    kcam=None,
    doptics=None,
    # sampling options
    dsampling_pixel=None,
    dsampling_optics=None,
    # optics
    optics=None,
):

    # -------------------
    # optics
    # -------------------

    if optics is None:
        optics = doptics['optics']

    # ---------------------
    # generic pixel outline
    # ---------------------

    # get pixel outline
    kout0, kout1 = coll.dobj['camera'][kcam]['dgeom']['outline']

    # sample pixel
    dout_pixel = poly2d_sample(
        coll.ddata[kout0]['data'],
        coll.ddata[kout1]['data'],
        key=kcam,
        dedge=dsampling_pixel.get('dedge'),
        dsurface=dsampling_pixel.get('dsurface'),
    )

    nstart = dout_pixel['x0'].size

    # ---------------------
    # prepare
    # ---------------------

    # shared apertures ?
    parallel = coll.dobj['camera'][kcam]['dgeom']['parallel']

    # shape camera
    shape_cam = coll.dobj['camera'][kcam]['dgeom']['shape']
    shape_start = shape_cam + (nstart,)

    # camera vector
    # get camera vectors
    dvect = coll.get_camera_unit_vectors(kcam)
    lc = ['x', 'y', 'z']
    if parallel is True:
        e0i_x, e0i_y, e0i_z = [dvect[f"e0_{kk}"] for kk in lc]
        e1i_x, e1i_y, e1i_z = [dvect[f"e1_{kk}"] for kk in lc]

    cents_x, cents_y, cents_z = coll.get_camera_cents_xyz(kcam)

    # -----------------------
    # prepare buffer for cx, cy, cz
    # -----------------------

    # cx, cy, cz (3d poitns for each pixel)
    cx = np.full((nstart,), np.nan)
    cy = np.full((nstart,), np.nan)
    cz = np.full((nstart,), np.nan)

    # ---------------------
    # Only 1 optic (pinhole or collimator with 1 common optics)
    # ---------------------

    if len(optics) == 1:

        # ------------------------
        # get end points on optics

        endx, endy, endz = _get_end_optics(
            coll=coll,
            kop=optics[0],
            dsampling_optics=dsampling_optics,
        )
        nend = endx.size

        # --------------
        # prepare output

        # rays shape for each pixel
        shape_each = (nstart, nend)
        vx = np.full(shape_each, np.nan)
        vy = np.full(shape_each, np.nan)
        vz = np.full(shape_each, np.nan)

        # oreall
        shape_out = shape_start + (nend,)

        dout = {
            'start_x': np.full(shape_out, np.nan),
            'start_y': np.full(shape_out, np.nan),
            'start_z': np.full(shape_out, np.nan),
            'vect_x': np.full(shape_out, np.nan),
            'vect_y': np.full(shape_out, np.nan),
            'vect_z': np.full(shape_out, np.nan),
            # 'dsang': np.full(shape_out, np.nan),
        }

        # -------------------
        # assign output

        for ii, ind in enumerate(np.ndindex(shape_cam)):

            if not parallel:
                e0i_x, e0i_y, e0i_z = [dvect[f"e0_{kk}"][ind] for kk in lc]
                e1i_x, e1i_y, e1i_z = [dvect[f"e1_{kk}"][ind] for kk in lc]

            # update cx, cy, cz
            _update_cxyz(
                parallel,
                e0i_x, e0i_y, e0i_z,
                e1i_x, e1i_y, e1i_z,
                cents_x, cents_y, cents_z,
                ind, lc,
                dout_pixel,
                cx, cy, cz,
                dvect,
            )

            # vect
            vx[...] = endx[None, :] - cx[:, None]
            vy[...] = endy[None, :] - cy[:, None]
            vz[...] = endz[None, :] - cz[:, None]

            vnorm_inv = 1./np.sqrt(vx**2 + vy**2 + vz**2)

            # slicing
            sli = ind + (slice(None), slice(None))

            # assigning
            dout['start_x'][sli] = np.copy(cx)[:, None]
            dout['start_y'][sli] = np.copy(cy)[:, None]
            dout['start_z'][sli] = np.copy(cz)[:, None]

            dout['vect_x'][sli] = vx * vnorm_inv
            dout['vect_y'][sli] = vy * vnorm_inv
            dout['vect_z'][sli] = vz * vnorm_inv

    # ---------------------
    # loop on pixels
    # ---------------------

    else:

        dout = {
            'start_x': {},
            'start_y': {},
            'start_z': {},
            'vect_x': {},
            'vect_y': {},
            'vect_z': {},
            # 'dsang': np.full(shape_out, np.nan),
        }

        for ii, ind in enumerate(np.ndindex(shape_cam)):

            # check optics per pixel
            lop = np.array(doptics['optics'])[doptics['paths'][ii, :]]
            lop = [
                kop for kop in lop
                if (kop in optics or optics is None)
            ]

            # check number of optics
            if len(lop) > 1:
                msg = (
                    "Multiple optics not handled yet!\n"
                    f"\t- key_diag = '{key}'\n"
                    f"\t- kcam = '{kcam}'\n"
                    f"\t- ii, ind = {ii}, {ind}\n"
                    f"\t- optics = {optics}\n"
                    f"\t- lop = {lop}\n"
                )
                raise NotImplementedError(msg)

            # ------------------------
            # get end points on optics

            endx, endy, endz = _get_end_optics(
                coll=coll,
                kop=lop[0],
                dsampling_optics=dsampling_optics,
            )
            nend = endx.size

            # ------------------
            # update pixel cx, cy, cz

            if not parallel:
                e0i_x, e0i_y, e0i_z = [dvect[f"e0_{kk}"][ind] for kk in lc]
                e1i_x, e1i_y, e1i_z = [dvect[f"e1_{kk}"][ind] for kk in lc]

            _update_cxyz(
                parallel,
                e0i_x, e0i_y, e0i_z,
                e1i_x, e1i_y, e1i_z,
                cents_x, cents_y, cents_z,
                ind, lc,
                dout_pixel,
                cx, cy, cz,
                dvect,
            )

            # vector
            vx = endx[None, :] - cx[:, None]
            vy = endy[None, :] - cy[:, None]
            vz = endz[None, :] - cz[:, None]

            vnorm_inv = 1./np.sqrt(vx**2 + vy**2 + vz**2)

            # assigning
            dout['start_x'][ind] = np.copy(cx)[:, None]
            dout['start_y'][ind] = np.copy(cy)[:, None]
            dout['start_z'][ind] = np.copy(cz)[:, None]

            dout['vect_x'][ind] = vx * vnorm_inv
            dout['vect_y'][ind] = vy * vnorm_inv
            dout['vect_z'][ind] = vz * vnorm_inv

    # -------------
    # adjust
    # -------------

    if isinstance(dout['start_x'], dict):

        dnrays = {k0: v0.shape[1] for k0, v0 in dout['vect_x'].items()}
        nraysu = np.unique([v0 for v0 in dnrays.values()])

        lkstart = [f'start_{cc}' for cc in lc]
        lkvect = [f'vect_{cc}' for cc in lc]

        # initialize
        for kk in lkstart + lkvect:
            oo = np.full(shape_cam + (nstart, nraysu.max()), np.nan)

            for ii, ind in enumerate(np.ndindex(shape_cam)):
                sli = ind + (slice(None), slice(0, dnrays[ind], 1))
                oo[sli] = dout[kk][ind]

            dout[kk] = oo

    # ---------------
    # return
    # ---------------

    return dout


def _get_end_optics(
    coll=None,
    kop=None,
    dsampling_optics=None,
):

    kop, clsop = coll.get_optics_cls(kop)
    clsop, kop = clsop[0], kop[0]
    kout0, kout1 = coll.dobj[clsop][kop]['dgeom']['outline']

    dout_optics = poly2d_sample(
        coll.ddata[kout0]['data'],
        coll.ddata[kout1]['data'],
        key=kop,
        dedge=dsampling_optics.get('dedge'),
        dsurface=dsampling_optics.get('dsurface'),
    )

    # get 3d optics end points coordinates
    func = coll.get_optics_x01toxyz(key=kop, asplane=False)
    return func(dout_optics['x0'], dout_optics['x1'])


def _update_cxyz(
    parallel,
    e0i_x, e0i_y, e0i_z,
    e1i_x, e1i_y, e1i_z,
    cents_x, cents_y, cents_z,
    ind, lc,
    dout_pixel,
    cx, cy, cz,
    dvect,
):

    if parallel is not True:
        e0i_x, e0i_y, e0i_z = [dvect[f"e0_{kk}"][ind] for kk in lc]
        e1i_x, e1i_y, e1i_z = [dvect[f"e1_{kk}"][ind] for kk in lc]

    # start
    cx[...] = (
        cents_x[ind]
        + dout_pixel['x0'] * e0i_x
        + dout_pixel['x1'] * e1i_x
    )
    cy[...] = (
        cents_y[ind]
        + dout_pixel['x0'] * e0i_y
        + dout_pixel['x1'] * e1i_y
    )
    cz[...] = (
        cents_z[ind]
        + dout_pixel['x0'] * e0i_z
        + dout_pixel['x1'] * e1i_z
    )

    return


# ###############################################################
# ###############################################################
#                      store
# ###############################################################


def _store(
    coll=None,
    kdiag=None,
    dout=None,
    key_rays=None,
    config=None,
    strict=None,
    overwrite=None,
):

    # --------------
    # store
    # --------------

    for kcam, v0 in dout.items():

        key = key_rays[kcam]

        if key in coll.dobj['rays'].keys() and overwrite is True:
            coll.remove_rays(key)

        # -----------------
        # add ref

        nstart, nends = v0['vect_x'].shape[-2:]

        krstart = f"{key}_nstart"
        coll.add_ref(key=krstart, size=nstart)

        krend = f"{key}_nend"
        coll.add_ref(key=krend, size=nends)

        # -----------------
        # add rays

        ref = coll.dobj['camera'][kcam]['dgeom']['ref'] + (krstart, krend)

        coll.add_rays(
            key=key,
            ref=ref,
            config=config,
            strict=strict,
            **v0
        )

    return
