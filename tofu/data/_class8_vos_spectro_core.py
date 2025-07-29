

import datetime as dtm


import numpy as np
import scipy.stats as scpstats
import Polygon as plg


from . import _class8_equivalent_apertures as _equivalent_apertures
from . import _class8_reverse_ray_tracing as _reverse_rt


# ################################################
# ################################################
#           Main
# ################################################


def main(
    ddata=None,
    dkeep=None,
    dind_proj=None,
    # 3d pts coords
    xx=None,
    yy=None,
    zz=None,
    pp=None,
    # optics
    lpoly_post=None,
    # parameters
    min_threshold=None,
    # geometry of spectro
    cent_spectro=None,
    nin=None,
    e0=None,
    e1=None,
    # dang
    pix_size=None,
    dist_to_cam=None,
    dang=None,
    # resoluions
    n0=None,
    n1=None,
    # p0
    p0x=None,
    p0y=None,
    p0z=None,
    # camera geometry
    cbin0=None,
    cbin1=None,
    # functions for coords tranform
    ptsvect_plane=None,
    coords_x01toxyz_plane=None,
    ptsvect_spectro=None,
    ptsvect_cam=None,
    # bragg
    bragg=None,
    ang_rel=None,
    pow_interp=None,
    # timing
    timing=None,
    dt111=None,
    dt222=None,
    # verb
    verb=None,
    # debug
    debug=None,
    # unused
    **kwdargs,
):

    # -----------------
    # loop on 3d points
    # -----------------

    pti = np.r_[0., 0., 0.]
    for i0, ipts in enumerate(np.ndindex(xx.shape)):

        if timing:
            t000 = dtm.datetime.now()     # DB

        # ---------------
        # update pti

        pti[0] = xx[ipts]
        pti[1] = yy[ipts]
        pti[2] = zz[ipts]

        # ------------------------------------------
        # initial polygon (crystal on its own plane)

        p0, p1 = ptsvect_plane(
            pts_x=pti[0],
            pts_y=pti[1],
            pts_z=pti[2],
            vect_x=p0x - pti[0],
            vect_y=p0y - pti[1],
            vect_z=p0z - pti[2],
            strict=True,
            return_x01=True,
        )[-2:]
        p_a = plg.Polygon(np.array([p0, p1]).T)

        # post polygons
        if len(lpoly_post) > 0:
            # get equivalent aperture
            p0, p1 = _equivalent_apertures._get_equivalent_aperture(
                p_a=p_a,
                pt=pti,
                nop_pre=len(lpoly_post),
                lpoly_pre=lpoly_post,
                ptsvect=ptsvect_plane,
                min_threshold=min_threshold,
            )

            # skip if no intersection
            if p0 is None or p0.size == 0:
                continue

        # timing
        if timing:
            t111 = dtm.datetime.now()     # DB
            dt111 += (t111-t000).total_seconds()

        # -----------------------------------------------
        # compute image on camera from pti through polygon

        (
            x0c, x1c,
            angles, dsang,
            vectx, vecty, vectz, iok,
            dangmin_str, x0if, x1if,
        ) = _reverse_rt._get_points_on_camera_from_pts(
            p0=p0,
            p1=p1,
            pti=pti,
            # ref
            cent=cent_spectro,
            nin=nin,
            e0=e0,
            e1=e1,
            # dang
            pix_size=pix_size,
            dist_to_cam=dist_to_cam,
            dang=dang,
            phi=pp[ipts],
            # resoluions
            n0=n0,
            n1=n1,
            # functions
            coords_x01toxyz_plane=coords_x01toxyz_plane,
            ptsvect_spectro=ptsvect_spectro,
            ptsvect_cam=ptsvect_cam,
        )[:-6]

        if verb is True:
            msg = (
                f"\t\t3d pt {i0 + 1} / {xx.size}"
                f":\t {iok.sum()} rays   "
                f"\t dangmin: {dangmin_str}"
            )
            print(msg, end='\r')

        if timing:
            # dt1111, dt2222, dt3333, dt4444 = out
            t222 = dtm.datetime.now()     # DB
            dt222 += (t222-t111).total_seconds()

        # safety check - which rays end up within camera frame ?
        iok2 = (
            (x0c[iok] >= cbin0[0])
            & (x0c[iok] <= cbin0[-1])
            & (x1c[iok] >= cbin1[0])
            & (x1c[iok] <= cbin1[-1])
        )

        # ---------- DEBUG ------------
        if debug is True:
            # _plot_debug(
            # coll=coll,
            # key_cam=key_cam,
            # cbin0=cbin0,
            # cbin1=cbin1,
            # x0c=x0c,
            # x1c=x1c,
            # cos=cosi,
            # angles=angles,
            # iok=iok,
            # p0=p0,
            # p1=p1,
            # x0if=x0if,
            # x1if=x1if,
            # )
            # dx0[i0][i1].append(x0c)
            # dx1[i0][i1].append(x1c)
            pass
        # -------- END DEBUG ----------

        if not np.any(iok2):
            continue

        # update index (within camera frame)
        iok[iok] = iok2

        # -----------------------
        # angles vs rocking curve

        dangok = angles[iok][:, None] - bragg[None, :]
        ilamb = (dangok >= ang_rel[0]) & (dangok < ang_rel[-1])

        coefs = np.zeros(dangok.shape)
        coefs[ilamb] = pow_interp(dangok[ilamb])
        ph = dsang[iok][:, None] * coefs

        # -------------------
        # binning into pixels

        ddata = _binning_3d(
            ddata=ddata,
            weights=(
                ('ncounts', None),
                ('sang', dsang[iok]),
                ('vectx', dsang[iok]*vectx[iok]),
                ('vecty', dsang[iok]*vecty[iok]),
                ('vectz', dsang[iok]*vectz[iok]),
            ),
            ipts=ipts,
            cbin0=cbin0,
            cbin1=cbin1,
            x0c=x0c[iok],
            x1c=x1c[iok],
            final=('ph', ph),
            ilamb=ilamb,
            # dind_proj
            dind_proj=dind_proj,
        )

    # -------------------------
    # derive useful quantities
    # -------------------------

    sli = tuple([None]*2 + [slice(None)])
    for kproj in ['3d', 'cross', 'hor']:
        if dkeep[kproj] is True:
            ksang = _get_ddata_key('sang', proj=kproj, din=ddata)
            kdV = f'dV_{kproj}'
            ddata['etendlen']['data'] = np.sum(
                ddata[ksang]['data']
                * ddata[kdV]['data'][sli],
                axis=-1,
            )
            break

    return ddata, dt111, dt222


# ################################################
# ################################################
#           Binning
# ################################################


def _binning_3d(
    ddata=None,
    weights=None,
    ipts=None,
    cbin0=None,
    cbin1=None,
    x0c=None,
    x1c=None,
    # final
    final=None,
    ilamb=None,
    # dind_proj
    dind_proj=None,
):

    # ---------------
    # loop on weights
    # ---------------

    out = None
    for i0, (k0, v0) in enumerate(weights):

        # 2d pixel by binning - counts only
        out = scpstats.binned_statistic_dd(
            [x0c, x1c],
            v0,
            statistic='count' if v0 is None else 'sum',
            bins=(cbin0, cbin1),
            expand_binnumbers=False,
            binned_statistic_result=out,
        )

        # store
        for kproj, vind in dind_proj.items():
            key = _get_ddata_key(k0, proj=kproj, din=ddata)
            ind = vind['all'][ipts]
            sli = (slice(None), slice(None), ind)
            ddata[key]['data'][sli] += out.statistic

    # -----------------
    # add ph
    # -----------------

    if np.any(ilamb):

        for i0 in np.any(ilamb, axis=0).nonzero()[0]:

            # bin
            out = scpstats.binned_statistic_dd(
                [x0c, x1c],
                final[1][:, i0],
                statistic='count' if v0 is None else 'sum',
                bins=(cbin0, cbin1),
                expand_binnumbers=False,
                binned_statistic_result=out,
            )

            # store
            sli = (slice(None), slice(None)) + ipts + (i0,)

            # store
            for kproj, vind in dind_proj.items():
                key = _get_ddata_key(final[0], proj=kproj, din=ddata)
                ind = vind['all'][ipts]
                sli = (slice(None), slice(None), ind, i0)
                ddata[key]['data'][sli] += out.statistic

    return ddata


# ###########################################################
# ###########################################################
#               _get_ddata_key
# ###########################################################


def _get_ddata_key(key=None, proj=None, din=None):

    lk = [kk for kk in din.keys() if kk.endswith(f'{key}_{proj}')]
    if len(lk) != 1:
        msg = (
            f"Looking for key '{key}' with proj '{proj}':\n"
            f"Found: {lk}\n"
            f"Available: {din.keys()}\n"
        )
        raise Exception(msg)

    return lk[0]
