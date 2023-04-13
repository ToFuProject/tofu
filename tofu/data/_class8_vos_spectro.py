# -*- coding: utf-8 -*-


import datetime as dtm      # DB
import numpy as np
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

    pcross0, pcross1 = np.array(pcross).T
    phor0, phor1 = np.array(phor).T

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
    izru = [iz[ir == i0] for i0 in iru]

    nphi = np.ceil(x0[ir]*(dphi[1] - dphi[0]) / res).astype(int)

    irf = np.repeat(ir, nphi)
    izf = np.repeat(iz, nphi)
    phi = np.concatenate(tuple([
        np.linspace(dphi[0], dphi[1], nn) for nn in nphi
    ]))

    xx = x0[irf] * np.cos(phi)
    yy = x0[irf] * np.sin(phi)
    zz = x1[izf]

    import pdb; pdb.set_trace()     # DB

    # -------------------------------------
    # prepare lambda, angles, rocking_curve

    lamb = np.linspace()
    bragg = None
    ang = np.linspace()
    angbrag = bragg[None, :] + ang[:, None]

    dang = None

    # rock_curve_full = np.repeat(, axis=1)

    # -------------
    # get

    lcross = []
    for pp in pts:

        # get equivalent aperture
        p0, p1 = _equivalent_apertures._get_equivalent_aperture(
            p_a=p_a,
            # pt=np.r_[],
            nop_pre=len(lpoly_pre),
            lpoly_pre=lpoly_pre,
            ptsvect=ptsvect,
        )

        if p0 is None or p0.size == 0:
            continue

        # get angles
        angles = ptsvect(
            pts_x=cxi,
            pts_y=cyi,
            pts_z=czi,
            vect_x=np.tile(exi, nc) - cxi,
            vect_y=np.tile(eyi, nc) - cyi,
            vect_z=np.tile(ezi, nc) - czi,
            strict=True,
            return_x01=False,
        )[6]

        # angles min, max
        ang_min = np.min(angles)
        ang_max = np.max(angles)

        # get lambda from angles and rocking curve
        ind = (angbragg >= ang_min) & (angbragg <= ang_max)
        indlamb = np.any(ind, axis=0)

        angi = angbragg[:, indlamb]

        # get

        # get power ratio
        pow_ratio = rock_curve[:, None][ind]

        # get rays
        rays = None

        # get image on camera
        x0, x1 = pts2plane(
            pts_x=cx[ii],
            pts_y=cy[ii],
            pts_z=cz[ii],
            vect_x=pxi - cx[ii],
            vect_y=pyi - cy[ii],
            vect_z=pzi - cz[ii],
            strict=False,
            return_x01=True,
        )

        import pdb; pdb.set_trace()     # DB

        # Interpolate per pixel
        pow_ratio = None
        cos = None

        # Integrate per pixel
        dV * dlamb * dang * dalpha





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
