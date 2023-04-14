# -*- coding: utf-8 -*-


import datetime as dtm      # DB
import numpy as np
import scipy.stats as scpstats
from matplotlib.path import Path
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
    sh=None,
    res=None,
    res_lamb=None,
    bool_cross=None,
    # parameters
    margin_poly=None,
    config=None,
    visibility=None,
    verb=None,
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

    # -------------------------------------------------
    # prepare polygons limiting points in cross-section

    # get temporary vos
    kpc0, kpc1 = doptics[key_cam]['vos_pcross']
    shape = coll.ddata[kpc0]['data'].shape
    pcross0 = coll.ddata[kpc0]['data'].reshape((shape[0], -1))
    pcross1 = coll.ddata[kpc1]['data'].reshape((shape[0], -1))
    kph0, kph1 = doptics[key_cam]['vos_phor']
    shapeh = coll.ddata[kph0]['data'].shape
    phor0 = coll.ddata[kph0]['data'].reshape((shapeh[0], -1))
    phor1 = coll.ddata[kph1]['data'].reshape((shapeh[0], -1))

    dphi = doptics[key_cam]['vos_dphi']

    # pix indices
    ipix = ~(np.any(np.isnan(pcross0), axis=0))
    ipix_n = ipix.nonzero()[0]

    # envelop pcross and phor
    pcross = plg.Polygon(
        np.array([pcross0[:, ipix_n[0]], pcross1[:, ipix_n[0]]]).T
    )
    phor = plg.Polygon(
        np.array([phor0[:, ipix_n[0]], phor1[:, ipix_n[0]]]).T
    )
    for ii in ipix_n[1:]:
        pcross |= plg.Polygon(
            np.array([pcross0[:, ii], pcross1[:, ii]]).T
        )
        phor |= plg.Polygon(np.array([phor0[:, ii], phor1[:, ii]]).T)

    pcross0, pcross1 = np.array(pcross)[0].T
    phor0, phor1 = np.array(phor)[0].T

    # --------------------------
    # prepare points and indices

    # get cross-section polygon
    ind, path_hor = _utilities._get_cross_section_indices(
        dsamp=dsamp,
        # polygon
        pcross0=pcross0,
        pcross1=pcross1,
        phor0=phor0,
        phor1=phor1,
        margin_poly=margin_poly,
        # points
        x0f=x0f,
        x1f=x1f,
        sh=sh,
    )

    ir, iz = ind.nonzero()
    iru = np.unique(ir)

    phimin = np.nanmin(dphi[0, :])
    phimax = np.nanmin(dphi[1, :])

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

    kang_rel = coll.dobj['crystal'][kspectro]['dmat']['drock']['angle_rel']
    kpow = coll.dobj['crystal'][kspectro]['dmat']['drock']['power_ratio']
    ang_rel = coll.ddata[kang_rel]['data']
    dang = np.mean(np.diff(ang_rel))
    pow_ratio = coll.ddata[kpow]['data']

    angbragg = bragg[None, :] + ang_rel[:, None]

    # --------------
    # prepare output

    # shape =
    # ph_count = np.full(shape, np.nan)
    # cos = np.full(shape, np.nan)

    # -------------
    # get


    pti = np.r_[0., 0., 0.]
    for i0 in iru:

        nphi = np.ceil(x0u[i0]*(phimax - phimin) / res).astype(int)
        phi = np.linspace(phimin, phimax, nphi)

        for i1 in iz[ir == i0]:

            pti[2] = x1u[i1]

            for i2, phii in enumerate(phi):

                print(f"\t\t{i0}, {i1}, {i2} / {nphi}")
                # set point
                pti[0] = x0u[i0]*np.cos(phii)
                pti[1] = x0u[i0]*np.sin(phii)

                # anuglar resolution associated to pixels
                dist_spectro = np.linalg.norm(pti - cent_spectro)
                dang_pix = pix_size / (dist_to_cam + dist_spectro)

                # compute image
                (
                    x0c, x1c, angles, dsang, cosi, iok,
                ) = _get_points_on_camera_from_pts(
                    p0x=p0x,
                    p0y=p0y,
                    p0z=p0z,
                    pti=pti,
                    lpoly_pre=lpoly_pre,
                    dist=dist_spectro,
                    dang=dang,
                    dang_pix=dang_pix,
                    phi=phii,
                    # functions
                    ptsvect_plane=ptsvect_plane,
                    coords_x01toxyz_plane=coords_x01toxyz_plane,
                    ptsvect_spectro=ptsvect_spectro,
                    ptsvect_cam=ptsvect_cam,
                )[:6]

                if x0c is None:
                    continue

                # ---------- DEBUG ------------
                if True:
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

                # 2d pixel by binning
                out = scpstats.binned_statistic_2d(
                    x0c[iok],
                    x1c[iok],
                    cosi[iok],
                    statistic='mean',
                    bins=(cbin0, cbin1),
                    expand_binnumbers=True,
                )
                import pdb; pdb.set_trace()     # DB

                cos[] = out.statistic

                # get range of ang_rel

                # get power ratio


                # Interpolate per pixel
                power_ratio = None

                # binned photon counts


                # binned average cos

                dV * dlamb * dsang

    import pdb; pdb.set_trace()     # DB




    # ------------
    # get indices

    if timing:
        t0 = dtm.datetime.now()     # DB
        out, dt1, dt2, dt3 = out

    for ii, i0 in enumerate(iru):
        ind0 = irf == i0
        for i1 in izru[ii]:
            ind = ind0 & (izf == i1)
            bool_cross[i0 + 1, i1 + 1] = np.any(out[0, ind] > 0.)

    # timing
    if timing:
        dt4444 += (dtm.datetime.now() - t0).total_seconds()
        dt1111 += dt1
        dt2222 += dt2
        dt3333 += dt3

    return (
        None,
        dt11, dt22,
        dt111, dt222, dt333,
        dt1111, dt2222, dt3333, dt4444,
    )


# ################################################
# ################################################
#           Sub-routine
# ################################################


def _get_points_on_camera_from_pts(
    p0x=None,
    p0y=None,
    p0z=None,
    pti=None,
    lpoly_pre=None,
    dist=None,
    dang=None,
    dang_pix=None,
    phi=None,
    # functions
    ptsvect_plane=None,
    coords_x01toxyz_plane=None,
    ptsvect_spectro=None,
    ptsvect_cam=None,
):

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
            return None, None, None, None, None, None, None

    # back to 3d
    # px, py, pz = coords_x01toxyz(x0=p0, x1=p1)

    # get angles
    # (
        # ptx, pty, ptz,
        # vx, vy, vz,
        # angles, iok,
    # ) = ptsvect_spectro(
        # pts_x=pti[0],
        # pts_y=pti[1],
        # pts_z=pti[2],
        # vect_x=px - pti[0],
        # vect_y=py - pti[1],
        # vect_z=pz - pti[2],
        # strict=True,
        # return_x01=False,
    # )

    # # angles min, max
    # ang_min = np.min(angles)
    # ang_max = np.max(angles)

    # # get lambda from angles and rocking curve
    # indang = (angbragg >= ang_min) & (angbragg <= ang_max)
    # indlamb = np.any(indang, axis=0)

    # lambi = lamb[indlamb]
    # angi = angbragg[:, indlamb]

    # dang
    dang_min = min(dang, 0.2*dang_pix)

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
