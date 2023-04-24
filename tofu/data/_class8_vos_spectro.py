# -*- coding: utf-8 -*-


import datetime as dtm      # DB
import numpy as np
import scipy.interpolate as scpinterp
import scipy.stats as scpstats
from matplotlib.path import Path
import matplotlib.pyplot as plt       # DB
import Polygon as plg


from . import _class8_equivalent_apertures as _equivalent_apertures
from . import _class8_vos_utilities as _utilities
# from ..geom import _comp_solidangles


# ###########################################################
# ###########################################################
#               Main
# ###########################################################


def _vos(
    # ressources
    coll=None,
    doptics=None,
    key_diag=None,
    key_cam=None,
    dsamp=None,
    # inputs
    x0u=None,
    x1u=None,
    x0f=None,
    x1f=None,
    x0l=None,
    x1l=None,
    # overall polygons
    pcross0=None,
    pcross1=None,
    phor0=None,
    phor1=None,
    dphi_r=None,
    sh=None,
    res=None,
    res_lamb=None,
    res_ang_rocking_curve=None,
    bool_cross=None,
    # parameters
    margin_poly=None,
    visibility=None,
    verb=None,
    # debug
    debug=None,
    # timing
    timing=None,
    dt11=None,
    dt111=None,
    dt1111=None,
    dt2222=None,
    dt3333=None,
    dt4444=None,
    dt222=None,
    dt333=None,
    dt22=None,
    # unused
    **kwdargs,
):
    """ vos computation for spectrometers """

    if timing:
        t00 = dtm.datetime.now()     # DB

    # -----------------
    # prepare optics

    lop = doptics[key_cam]['optics'][::-1]
    lop, lcls = coll.get_optics_cls(lop)

    kspectro = lop[lcls.index('crystal')]
    ispectro = lop.index(kspectro)
    if len(lop[ispectro:]) > 1:
        msg = "Not yet implemented optics between crystal and camera!"
        raise NotImplementedError()

    lpoly_pre = [
        coll.get_optics_poly(
            key=k0,
            add_points=None,
            return_outline=False,
        )
        for k0 in lop[:ispectro]
    ]

    # get initial polygon
    p0x, p0y, p0z = coll.get_optics_poly(key=kspectro, add_points=None)

    # get functions
    ptsvect_plane = coll.get_optics_reflect_ptsvect(key=kspectro, asplane=True)
    ptsvect_spectro = coll.get_optics_reflect_ptsvect(key=kspectro)
    ptsvect_cam = coll.get_optics_reflect_ptsvect(key=key_cam)

    coords_x01toxyz_plane = coll.get_optics_x01toxyz(
        key=kspectro,
        asplane=True,
    )

    # Get centers of crystal and camera to estimate distance
    cent_spectro = coll.get_optics_x01toxyz(key=kspectro)(x0=0, x1=0)
    cent_cam = coll.dobj['camera'][key_cam]['dgeom']['cent']
    dist_to_cam = np.linalg.norm(cent_spectro - cent_cam)
    pix_size = np.sqrt(coll.dobj['camera'][key_cam]['dgeom']['pix_area'])

    # prepare camera bin edges
    kcc = coll.dobj['camera'][key_cam]['dgeom']['cents']
    cc0 = coll.ddata[kcc[0]]['data']
    cc1 = coll.ddata[kcc[1]]['data']
    cout0, cout1 = coll.get_optics_outline(key_cam, total=False)
    cbin0 = np.r_[cc0 + np.min(cout0), cc0[-1] + np.max(cout0)]
    cbin1 = np.r_[cc1 + np.min(cout1), cc1[-1] + np.max(cout1)]

    # --------------------------
    # get overall polygons

    pcross0, pcross1 = _utilities._get_overall_polygons(
        coll=coll,
        doptics=coll.dobj['diagnostic'][key_diag]['doptics'],
        key_cam=key_cam,
        poly='pcross',
    )

    phor0, phor1 = _utilities._get_overall_polygons(
        coll=coll,
        doptics=coll.dobj['diagnostic'][key_diag]['doptics'],
        key_cam=key_cam,
        poly='phor',
    )

    # --------------------------
    # add margins

    pcross0, pcross1 = _utilities._get_poly_margin(
        # polygon
        p0=pcross0,
        p1=pcross1,
        # margin
        margin=margin_poly,
    )

    phor0, phor1 = _utilities._get_poly_margin(
        # polygon
        p0=phor0,
        p1=phor1,
        # margin
        margin=margin_poly,
    )

    # ------------------------
    # get ind in cross-section

    pcross = Path(np.array([pcross0, pcross1]).T)
    ind = (
        dsamp['ind']['data']
        & pcross.contains_points(np.array([x0f, x1f]).T).reshape(sh)
    )

    # R and Z indices
    ir, iz = ind.nonzero()
    iru = np.unique(ir)

    # ----------
    # get dphi_r

    dphi = doptics[key_cam]['dvos']['dphi']
    phimin = np.nanmin(dphi[0, :])
    phimax = np.nanmin(dphi[1, :])

    # get dphi vs phor
    dphi_r = _utilities._get_dphi_from_R_phor(
        R=x0u[iru],
        phor0=phor0,
        phor1=phor1,
        phimin=phimin,
        phimax=phimax,
        res=res,
    )

    # -------------------------------------
    # prepare lambda, angles, rocking_curve

    lamb = coll.get_diagnostic_data(
        key=key_diag,
        key_cam=key_cam,
        data='lamb',
    )[0][key_cam]

    lambmin = np.nanmin(lamb)
    lambmax = np.nanmax(lamb)
    dlamb = (lambmax - lambmin) * 1.1
    lambmean = 0.5*(lambmin + lambmax)
    nlamb = int(np.ceil(dlamb / res_lamb))
    lamb = np.linspace(lambmean - 0.5*dlamb, lambmean + 0.5*dlamb, nlamb)

    bragg = coll.get_crystal_bragglamb(key=kspectro, lamb=lamb)[0]

    # power ratio
    kpow = coll.dobj['crystal'][kspectro]['dmat']['drock']['power_ratio']
    pow_ratio = coll.ddata[kpow]['data']

    # angle relative
    kang_rel = coll.dobj['crystal'][kspectro]['dmat']['drock']['angle_rel']
    ang_rel = coll.ddata[kang_rel]['data']
    if res_ang_rocking_curve is not None:
        nang = int((np.max(ang_rel) - np.min(ang_rel)) / res_ang_rocking_curve)
        ang_rel2 = np.linspace(np.min(ang_rel), np.max(ang_rel), nang)
        dang = np.mean(np.diff(ang_rel))
        pow_ratio = scpinterp.interp1d(
            ang_rel,
            pow_ratio,
            kind='linear',
        )(ang_rel2)
        ang_rel = ang_rel2

    dang = np.mean(np.diff(ang_rel))

    # overall bragg angle with rocking curve
    angbragg = bragg[None, :] + ang_rel[:, None]

    # --------------
    # prepare output

    shape_cam = coll.dobj['camera'][key_cam]['dgeom']['shape']

    shape0 = tuple(np.r_[shape_cam, ind.sum()])
    ncounts = np.full(shape0, 0.)
    cos = np.full(shape0, 0.)
    lambmin = np.full(shape0, np.inf)
    lambmax = np.full(shape0, 0.)
    indr = np.zeros((ind.sum(),), dtype=int)
    indz = np.zeros((ind.sum(),), dtype=int)

    shape1 = tuple(np.r_[shape_cam, ind.sum(), nlamb])
    ph_count = np.full(shape1, 0.)

    if timing:
        t11 = dtm.datetime.now()     # DB
        dt11 += (t11-t00).total_seconds()

    # ----------
    # verb

    if verb is True:
        msg = (
            f"\tlamb.shape: {lamb.shape}\n"
            f"\tang_rel.shape: {ang_rel.shape}\n"
            f"\tiru.size: {iru.size}\n"
            f"\tind.sum(): {ind.sum()}\n"
        )
        print(msg)

    # ---------------------
    # loop in plasma points

    dr = np.mean(np.diff(x0u))
    dz = np.mean(np.diff(x1u))
    ipts = 0
    pti = np.r_[0., 0., 0.]
    nru = iru.size
    for i00, i0 in enumerate(iru):

        if np.all(np.isnan(dphi_r[:, i00])):
            continue

        nphi = np.ceil(x0u[i0]*(dphi_r[1, i00] - dphi_r[0, i00]) / res).astype(int)
        phi = np.linspace(dphi_r[0, i00], dphi_r[1, i00], nphi)
        dphi = phi[1] - phi[0]
        dV = dr * x0u[i0] * dphi * dz
        indiz = ir == i0
        nz = indiz.sum()

        for i11, i1 in enumerate(iz[indiz]):

            indr[ipts] = i0
            indz[ipts] = i1
            pti[2] = x1u[i1]

            for i2, phii in enumerate(phi):

                if timing:
                    t000 = dtm.datetime.now()     # DB

                # set point
                pti[0] = x0u[i0]*np.cos(phii)
                pti[1] = x0u[i0]*np.sin(phii)

                # anuglar resolution associated to pixels
                dist_spectro = np.linalg.norm(pti - cent_spectro)
                dang_pix = pix_size / (dist_to_cam + dist_spectro)

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

                if len(lpoly_pre) > 0:
                    # get equivalent aperture
                    p0, p1 = _equivalent_apertures._get_equivalent_aperture(
                        p_a=p_a,
                        pt=pti,
                        nop_pre=len(lpoly_pre),
                        lpoly_pre=lpoly_pre,
                        ptsvect=ptsvect_plane,
                    )

                    # skip if no intersection
                    if p0 is None or p0.size == 0:
                        continue

                if timing:
                    t111 = dtm.datetime.now()     # DB
                    dt111 += (t111-t000).total_seconds()

                # compute image
                (
                    x0c, x1c, angles, dsang, cosi, iok,
                ) = _get_points_on_camera_from_pts(
                    p0=p0,
                    p1=p1,
                    pti=pti,
                    dist=dist_spectro,
                    dang=dang,
                    dang_pix=dang_pix,
                    phi=phii,
                    # functions
                    coords_x01toxyz_plane=coords_x01toxyz_plane,
                    ptsvect_spectro=ptsvect_spectro,
                    ptsvect_cam=ptsvect_cam,
                )[:6]

                if verb is True:
                    msg = (
                        f"\t\t{i00} / {nru}, {i11} / {nz}, {i2} / {nphi}"
                        f":  {iok.sum()} pts"
                    )
                    print(msg, end='\r')

                if timing:
                    # dt1111, dt2222, dt3333, dt4444 = out
                    t222 = dtm.datetime.now()     # DB
                    dt222 += (t222-t111).total_seconds()

                # ---------- DEBUG ------------
                if debug is True:
                    _plot_debug(
                        coll=coll,
                        key_cam=key_cam,
                        cbin0=cbin0,
                        cbin1=cbin1,
                        x0c=x0c,
                        x1c=x1c,
                        cos=cosi,
                        angles=angles,
                        iok=iok,
                    )
                # -------- END DEBUG ----------

                # safety check
                iok2 = (
                    (x0c[iok] >= cbin0[0])
                    & (x0c[iok] <= cbin0[-1])
                    & (x1c[iok] >= cbin1[0])
                    & (x1c[iok] <= cbin1[-1])
                )
                if not np.any(iok2):
                    continue

                iok[iok] = iok2

                # 2d pixel by binning
                out = scpstats.binned_statistic_2d(
                    x0c[iok],
                    x1c[iok],
                    None,
                    statistic='count',
                    bins=(cbin0, cbin1),
                    expand_binnumbers=True,
                )

                ipixok = out.statistic > 0
                ncounts[ipixok, ipts] += out.statistic[ipixok]

                cosi = cosi[iok]
                angles = angles[iok]

                ip0, ip1 = ipixok.nonzero()
                for ii in np.unique(ip0):
                    indi = out.binnumber[0, :] == ii + 1
                    for jj in np.unique(ip1[ip0 == ii]):

                        # indices
                        indj = indi & (out.binnumber[1, :] == jj + 1)

                        # cos
                        cos[ii, jj, ipts] += np.sum(cosi[indj])

                        # ilamb
                        ilamb = (
                            (angles[indj][:, None] >= angbragg[:1, ...])
                            & (angles[indj][:, None] < angbragg[-1:, ...])
                        )

                        if not np.any(ilamb):
                            continue

                        ilamb_n = np.any(ilamb, axis=0).nonzero()[0]

                        # lambmin
                        lambmin[ii, jj, ipts] = min(
                            lambmin[ii, jj, ipts],
                            np.min(lamb[ilamb_n]),
                        )

                        # lambmax
                        lambmax[ii, jj, ipts] = max(
                            lambmax[ii, jj, ipts],
                            np.max(lamb[ilamb_n]),
                        )

                        # binning of angles
                        for kk in ilamb_n:
                            inds = np.searchsorted(
                                angbragg[:, kk],
                                angles[indj][ilamb[:, kk]],
                            )

                            # update power_ratio
                            ph_count[ii, jj, ipts, kk] += np.sum(pow_ratio[inds])

                # multiply by dsang
                ph_count[ii, jj, ipts, :] *= dsang

                if timing:
                    t333 = dtm.datetime.now()     # DB
                    dt333 += (t333-t222).total_seconds()

            # update index
            ipts += 1

        # multiply by increments
        ph_count[ii, ...] *= (dV * dlamb)

    if timing:
        t22 = dtm.datetime.now()     # DB

    # remove useless points
    iin = np.any(np.any(ncounts > 0, axis=0), axis=0)
    if not np.all(iin):
        ncounts = ncounts[:, :, iin]
        cos = cos[:, :, iin]
        lambmin = lambmin[:, :, iin]
        lambmax = lambmax[:, :, iin]
        ph_count = ph_count[:, :, iin]
        indr = indr[iin]
        indz = indz[iin]

    # remove useless lamb
    iin = ph_count > 0.
    ilamb = np.any(np.any(np.any(iin, axis=0), axis=0), axis=0)
    if not np.all(ilamb):
        ph_count = ph_count[..., ilamb]
        lamb = lamb[ilamb]

    # average cos + adjust
    iout = ncounts == 0
    cos[~iout] = cos[~iout] / ncounts[~iout]
    cos[iout] = np.nan
    lambmin[iout] = np.nan
    lambmax[iout] = np.nan
    ph_count[iout, :] = np.nan

    # ------------
    # get indices

    # for ii, i0 in enumerate(iru):
        # ind0 = irf == i0
        # for i1 in izru[ii]:
            # ind = ind0 & (izf == i1)
            # bool_cross[i0 + 1, i1 + 1] = np.any(out[0, ind] > 0.)

    # dout
    dout = {
        'pcross0': None,
        'pcross1': None,
        'lamb': lamb,
        'indr': indr,
        'indz': indz,
        'cos': cos,
        'lambmin': lambmin,
        'lambmax': lambmax,
        'ph_count': ph_count,
    }

    if timing:
        t33 = dtm.datetime.now()
        dt22 += (t33 - t22).total_seconds()

    return (
        dout,
        dt11, dt22,
        dt111, dt222, dt333,
        dt1111, dt2222, dt3333, dt4444,
    )


# ################################################
# ################################################
#           Sub-routine
# ################################################


def _get_points_on_camera_from_pts(
    p0=None,
    p1=None,
    pti=None,
    dist=None,
    dang=None,
    dang_pix=None,
    phi=None,
    # functions
    coords_x01toxyz_plane=None,
    ptsvect_spectro=None,
    ptsvect_cam=None,
):

    # dang
    dang_min = min(dang, 0.25*dang_pix)

    # set n0, n1
    p0min, p0max = p0.min(), p0.max()
    p1min, p1max = p1.min(), p1.max()
    n0 = int(np.ceil(((p0max - p0min) / dist) / dang_min))
    n1 = int(np.ceil(((p1max - p1min) / dist) / dang_min))
    dsang = ((p0max - p0min) / n0) * ((p1max - p1min) / n1)

    # sample 2d equivalent aperture
    x0i = np.linspace(p0.min(), p0.max(), n0)
    x1i = np.linspace(p1.min(), p1.max(), n1)

    # mesh
    x0if = np.repeat(x0i[:, None], n1, axis=1)
    x1if = np.repeat(x1i[None, :], n0, axis=0)
    ind = Path(np.array([p0, p1]).T).contains_points(
        np.array([x0if.ravel(), x1if.ravel()]).T
    ).reshape((n0, n1))

    # back to 3d
    xx, yy, zz = coords_x01toxyz_plane(
        x0=x0if,
        x1=x1if,
    )

    # get normalized vector from plasma point to crystal
    vectx = xx - pti[0]
    vecty = yy - pti[1]
    vectz = zz - pti[2]
    norm = np.sqrt(vectx**2 + vecty**2 + vectz**2)
    vectx = vectx / norm
    vecty = vecty / norm
    vectz = vectz / norm

    # get local cosine vs toroidal direction (for doppler)
    cos = -vectx*np.sin(phi) + vecty*np.cos(phi)

    # get reflexion
    (
        ptsx, ptsy, ptsz,
        vx, vy, vz,
        angles, iok,
    ) = ptsvect_spectro(
        pts_x=pti[0],
        pts_y=pti[1],
        pts_z=pti[2],
        vect_x=vectx,
        vect_y=vecty,
        vect_z=vectz,
        strict=True,
        return_x01=False,
    )

    # get x0, x1 on camera
    x0c, x1c = ptsvect_cam(
        pts_x=ptsx,
        pts_y=ptsy,
        pts_z=ptsz,
        vect_x=vx,
        vect_y=vy,
        vect_z=vz,
        strict=False,
        return_x01=True,
    )[-2:]

    return x0c, x1c, angles, dsang, cos, ind, ptsx, ptsy, ptsz


# ################################################
# ################################################
#           Debug plot
# ################################################


def _plot_debug(
    coll=None,
    key_cam=None,
    cbin0=None,
    cbin1=None,
    x0c=None,
    x1c=None,
    cos=None,
    angles=None,
    iok=None,
):

    out0, out1 = coll.get_optics_outline(key_cam, total=True)
    ck0f = np.array([cbin0, cbin0, np.full((cbin0.size,), np.nan)])
    ck1f = np.array([cbin1, cbin1, np.full((cbin1.size,), np.nan)])
    ck01 = np.r_[np.min(cbin1), np.max(cbin1), np.nan]
    ck10 = np.r_[np.min(cbin0), np.max(cbin0), np.nan]

    ldata = [
        ('cos vs toroidal', cos),
        ('angles vs crystal', angles),
        ('iok', iok)
    ]

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 8))
    for ii, v0 in enumerate(ldata):
        ax = fig.add_subplot(1, 3, ii + 1, aspect='equal')
        ax.set_title(v0[0], size=12, fontweight='bold')
        ax.set_xlabel('x0 (m)', size=12, fontweight='bold')
        ax.set_xlabel('x1 (m)', size=12, fontweight='bold')

        ax.plot(np.r_[out0, out0[0]], np.r_[out1, out1[0]], '.-k')
        ax.plot(
            ck0f.T.ravel(),
            np.tile(ck01, cbin0.size),
            '-k',
        )
        ax.plot(
            np.tile(ck10, cbin1.size),
            ck1f.T.ravel(),
            '-k',
        )
        im = ax.scatter(
            x0c,
            x1c,
            c=v0[1],
            s=4,
            edgecolors='None',
            marker='o',
        )
        plt.colorbar(im, ax=ax)

    import pdb
    pdb.set_trace()     # DB
