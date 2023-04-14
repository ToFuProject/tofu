# -*- coding: utf-8 -*-


import datetime as dtm      # DB
import numpy as np
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

    lpoly_pre = [
        coll.get_optics_poly(
            key=k0,
            add_points=None,
            return_outline=False,
        )
        for k0 in lop[1:]
    ]
    kref = lop[0]
    kspectro = lop[lcls.index('crystal')]

    # intial polygon
    p_a = coll.get_optics_outline(key=kref, add_points=False)
    p_a = plg.Polygon(np.array([p_a[0], p_a[1]]).T)

    # ptsvect func
    ptsvect = coll.get_optics_reflect_ptsvect(key=kref)
    coords_x01toxyz = coll.get_optics_x01toxyz(key=kref)
    ptsvect_spectro = coll.get_optics_reflect_ptsvect(key=kspectro)
    ptsvect_cam = coll.get_optics_reflect_ptsvect(key=key_cam)

    cent = coll.get_optics_x01toxyz(key=kref)(x0=0, x1=0)
    cent_cam = coll.dobj['camera'][key_cam]['dgeom']['cent']
    dist_to_cam = np.linalg.norm(cent - cent_cam)
    pix_size = np.sqrt(coll.dobj['camera'][key_cam]['dgeom']['pix_area'])

    # ---------------
    # prepare spectro

    pts2plane = coll.get_optics_reflect_ptsvect(
        key=kspectro,
        asplane=True,
    )

    # --------------------------
    # prepare overall cross polygon

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


    # irf = np.repeat(ir, nphi)
    # izf = np.repeat(iz, nphi)
    # import pdb; pdb.set_trace()     # DB
    # phi = np.concatenate(tuple([
        # np.linspace(dphi[0], dphi[1], nn) for nn in nphi
    # ]))

    # xx = x0[irf] * np.cos(phi)
    # yy = x0[irf] * np.sin(phi)
    # zz = x1[izf]

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

    # angular resolution associated to pixels
    dang_pix = None

    # -------------
    # get

    lcross = []
    pti = np.r_[0., 0., 0.]
    for i0 in iru:

        nphi = np.ceil(x0u[i0]*(phimax - phimin) / res).astype(int)
        phi = np.linspace(phimin, phimax, nphi)

        for i1 in iz[ir == i0]:

            pti[2] = x1u[i1]

            for i2, phii in enumerate(phi):

                # set point
                pti[0] = x0u[i0]*np.cos(phii)
                pti[1] = x0u[i0]*np.sin(phii)

                # anuglar resolution associated to pixels
                dist_pix = np.linalg.norm(pti - cent) + dist_to_cam
                dang_pix = pix_size / dist_pix

                # compute image
                x0c, x1c, angles, sang, iok = _get_points_on_camera_from_pts(
                    p_a=p_a,
                    pti=pti,
                    lpoly_pre=lpoly_pre,
                    ptsvect=ptsvect,
                    dist=dist,
                    dang=dang,
                    dang_pix=dang_pix,
                    coords_x01toxyz=coords_x01toxyz,
                    ptsvect_spectro=ptsvect_spectro,
                    ptsvect_cam=ptsvect_cam,
                )[:5]

                if x0c is None:
                    continue

                # ge# get power ratio


                # Interpolate per pixel
                pow_ratio = None
                cos = None

                # Integrate per pixel
                dV * dlamb * dang * dalpha

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
    p_a=None,
    pti=None,
    lpoly_pre=None,
    ptsvect=None,
    dist=None,
    dang=None,
    dang_pix=None,
    coords_x01toxyz=None,
    ptsvect_spectro=None,
    ptsvect_cam=None,
):

    # get equivalent aperture
    p0, p1 = _equivalent_apertures._get_equivalent_aperture(
        p_a=p_a,
        pt=pti,
        nop_pre=len(lpoly_pre),
        lpoly_pre=lpoly_pre,
        ptsvect=ptsvect,
    )

    print(i0, i1, i2, p0)
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

    # set n0, n1
    p0min, p0max = p0.min(), p0.max()
    p1min, p1max = p1.min(), p1.max()
    n0 = int(np.ceil(((p0max - p0min) / dist) / min(dang, dang_pix)))
    n1 = int(np.ceil(((p1max - p1min) / dist) / min(dang, dang_pix)))
    sang = ((p0max - p0min) / n0) * ((p1max - p1min) / n1)

    # sample 2d equivalent aperture
    x0i = np.linspace(p0.min(), p0.max(), n0)
    x1i = np.linspace(p1.min(), p1.max(), n1)

    # mesh
    x0if = np.repeat(x0i[:, None], n1, axis=1)
    x1if = np.repeat(x1i[None, :], n0, axis=0)
    ind = Path(np.array([p0, p1].T)).contains_points(
        np.array([x0if.ravel(), x1if.ravel()]).T
    ).reshape((n0, n1))

    # back to 3d
    xx, yy, zz = coords_x01toxyz(
        x0=x0if,
        x1=x1if,
    )

    # get reflexion
    (
        ptsx, ptsy, ptsz,
        vx, vy, vz,
        angles, iok,
    ) = ptsvect_spectro(
        pts_x=pti[0],
        pts_y=pti[1],
        pts_z=pti[2],
        vect_x=xx - pti[0],
        vect_y=yy - pti[1],
        vect_z=zz - pti[2],
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
    )

    import pdb; pdb.set_trace()     # DB
    return x0c, x1c, angles, sang, ind, ptx, pty, ptz

