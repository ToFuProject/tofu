import numpy as np
import scipy.stats as scpstats
from matplotlib.path import Path
import Polygon as plg


from . import _class8_equivalent_apertures as _equivalent_apertures
from . import _class8_vos_utilities as _utilities


# ################################################
# ################################################
#           Loops
# ################################################


def _loop0(
    iok=None,
    ptsx=None,
    ptsy=None,
    ptsz=None,
    lpoly_post=None,
    min_threshold=None,
    # ref
    cent=None,
    nin=None,
    e0=None,
    e1=None,
    # p0
    p0x=None,
    p0y=None,
    p0z=None,
    # dang
    pix_size=None,
    dist_to_cam=None,
    dang=None,
    phi=None,
    shape_cam=None,
    # lamb
    lamb=None,
    dlamb=None,
    rocking_curve=None,
    bragg_per_pix=None,
    ang_rel=None,
    pow_interp=None,
    bragg=None,
    # optional
    n0=None,
    n1=None,
    # functions
    ptsvect_plane=None,
    coords_x01toxyz_plane=None,
    ptsvect_spectro=None,
    ptsvect_cam=None,
    # others
    ncounts=None,
    cbin0=None,
    cbin1=None,
    append=None,
    verb=True,
    **kwdargs,
):

    # ----------
    # prepare

    # counts, ph_count
    ncounts = np.full(shape_cam, 0.)
    sang = np.full(tuple(np.r_[shape_cam, ptsx.size]), 0.)
    sang0 = np.full(tuple(np.r_[shape_cam, ptsx.size]), 0.)
    sang_lamb = np.full(shape_cam, 0.)

    lsang = None
    if append is True:
        lp0, lp1 = [], []
        lpx, lpy, lpz = [], [], []
        lsang = []
        if lamb is None:
            dpow = None
        else:
            dpow = {ll: [] for ll in lamb}
    else:
        lp0, lp1 = None, None
        lpx, lpy, lpz = None, None, None
        dpow = None
        lsang = None

    # ----------
    # loop

    pti = np.full((3,), np.nan)
    npts0 = iok.sum()
    for ii, i0 in enumerate(iok.nonzero()[0]):

        pti[:] = np.r_[ptsx[i0], ptsy[i0], ptsz[i0]]

        # ------------------------------------------
        # initial polygon (crystal on its own plane)

        p0, p1 = ptsvect_plane(
            pts_x=ptsx[i0],
            pts_y=ptsy[i0],
            pts_z=ptsz[i0],
            vect_x=p0x - ptsx[i0],
            vect_y=p0y - ptsy[i0],
            vect_z=p0z - ptsz[i0],
            strict=True,
            return_x01=True,
        )[-2:]
        p_a = plg.Polygon(np.array([p0, p1]).T)

        if len(lpoly_post) > 0:

            # get equivalent aperture
            p0, p1 = _equivalent_apertures._get_equivalent_aperture(
                p_a=p_a,
                pt=pti,
                nop_pre=len(lpoly_post),
                lpoly_pre=lpoly_post,
                ptsvect=ptsvect_plane,
                min_threshold=min_threshold,
                # debug
                debug=False,
            )

            # skip if no intersection
            if p0 is None or p0.size == 0:
                continue

        # compute image
        (
            x0c, x1c,
            angles, dsang,
            vectx, vecty, vectz, iok,
            dangmin_str, x0if, x1if,
            ptsx1, ptsy1, ptsz1,
            ptsx2, ptsy2, ptsz2,
        ) = _get_points_on_camera_from_pts(
            p0=p0,
            p1=p1,
            pti=pti,
            # ref
            cent=cent,
            nin=nin,
            e0=e0,
            e1=e1,
            # dang
            pix_size=pix_size,
            dist_to_cam=dist_to_cam,
            dang=dang,
            phi=phi[ii],
            # optional
            n0=n0,
            n1=n1,
            # functions
            coords_x01toxyz_plane=coords_x01toxyz_plane,
            ptsvect_spectro=ptsvect_spectro,
            ptsvect_cam=ptsvect_cam,
            # debug
            debug=npts0 == 1,
        )

        if verb is True:
            msg = (
                f"\t\t{ii} / {npts0} pts \t dangmin: {dangmin_str}"
            )
            print(msg, end='\r')

        # stack coordinates
        if append is True:
            npts = iok.sum()
            lpx.append(np.array([
                np.full((npts,), ptsx[i0]),
                ptsx1[iok],
                ptsx2[iok],
            ]))
            lpy.append(np.array([
                np.full((npts,), ptsy[i0]),
                ptsy1[iok],
                ptsy2[iok],
            ]))
            lpz.append(np.array([
                np.full((npts,), ptsz[i0]),
                ptsz1[iok],
                ptsz2[iok],
            ]))

            # p0, p1
            lp0.append(x0c[iok])
            lp1.append(x1c[iok])

            # pow
            if lamb is not None:
                for kk, ll in enumerate(lamb):
                    dpow[ll].append(pow_interp(angles[iok] - bragg[kk]))

        # safety check
        iok2 = (
            (x0c[iok] >= cbin0[0])
            & (x0c[iok] <= cbin0[-1])
            & (x1c[iok] >= cbin1[0])
            & (x1c[iok] <= cbin1[-1])
        )

        if not np.any(iok2):
            continue

        # update index
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
        ncounts[ipixok] += out.statistic[ipixok]

        sang0[ipixok, i0] += scpstats.binned_statistic_2d(
            x0c[iok],
            x1c[iok],
            dsang[iok],
            statistic='sum',
            bins=(cbin0, cbin1),
            expand_binnumbers=True,
        ).statistic[ipixok]

        # taking into account rocking curve
        angles = angles[iok]
        dsang = dsang[iok]

        ip0, ip1 = ipixok.nonzero()
        indi = np.zeros((out.binnumber.shape[1],), dtype=bool)
        indj = np.zeros((out.binnumber.shape[1],), dtype=bool)
        for ii in np.unique(ip0):
            indi[:] = (out.binnumber[0, :] == (ii + 1))
            for jj in np.unique(ip1[ip0 == ii]):

                # indices
                indj[:] = indi & (out.binnumber[1, :] == jj + 1)

                if lamb is None:
                    # solid angle
                    sang[ii, jj, i0] += np.sum(dsang[indj])

                else:
                    # lambref
                    arelj = angles[indj] - bragg_per_pix[ii, jj]

                    # solid angle
                    sang[ii, jj, i0] += np.sum(
                        pow_interp(arelj)
                        * dsang[indj]
                    )

                    # ----------
                    # sang_lamb

                    angj = angles[indj]
                    ilamb = (
                        (angj[:, None] - bragg >= ang_rel[0])
                        & (angj[:, None] - bragg < ang_rel[-1])
                    )

                    if not np.any(ilamb):
                        continue

                    ilamb_n = np.any(ilamb, axis=0).nonzero()[0]

                    # binning of angles
                    for kk in ilamb_n:
                        sang_lamb[ii, jj] += np.sum(
                            pow_interp(angj[ilamb[:, kk]] - bragg[kk])
                            * dsang[indj][ilamb[:, kk]]
                        )

    # adjust sang_lamb
    if lamb is None:
        sang_lamb = np.nansum(sang0, axis=-1)

    return {
        'ncounts': {'data': ncounts, 'units': ''},
        'sang0': {'data': sang0, 'units': 'sr'},
        'sang': {'data': sang, 'units': 'sr'},    # not really useful?
        'sang_lamb': {'data': sang_lamb, 'units': 'sr'},
        'lamb': {'data': lamb, 'units': 'm'},
        'dlamb': {'data': dlamb, 'units': 'm'},
        'lp0': {'data': lp0, 'units': 'm'},
        'lp1': {'data': lp1, 'units': 'm'},
        'lpx': {'data': lpx, 'units': 'm'},
        'lpy': {'data': lpy, 'units': 'm'},
        'lpz': {'data': lpz, 'units': 'm'},
        'dpow': {'data': dpow, 'units': ''},
        'lsang': {'data': lsang, 'units': 'sr'},
    }


def _loop1(
    # specific
    coll=None,
    key=None,
    key_cam=None,
    key_mesh=None,
    margin_poly=None,
    res_phi=None,
    dsamp=None,
    # common
    iok=None,
    ptsx=None,
    ptsy=None,
    ptsz=None,
    lpoly_post=None,
    min_threshold=None,
    # ref
    cent=None,
    nin=None,
    e0=None,
    e1=None,
    # p0
    p0x=None,
    p0y=None,
    p0z=None,
    # dang
    pix_size=None,
    dist_to_cam=None,
    dang=None,
    phi=None,
    shape_cam=None,
    # optional
    n0=None,
    n1=None,
    # functions
    ptsvect_plane=None,
    coords_x01toxyz_plane=None,
    ptsvect_spectro=None,
    ptsvect_cam=None,
    # others
    ncounts=None,
    cbin0=None,
    cbin1=None,
    # others
    verb=True,
    **kwdargs,
):

    # --------------------------
    # get overall polygons

    pcross0, pcross1 = _utilities._get_overall_polygons(
        coll=coll,
        doptics=coll.dobj['diagnostic'][key]['doptics'],
        key_cam=key_cam,
        poly='pcross',
    )

    phor0, phor1 = _utilities._get_overall_polygons(
        coll=coll,
        doptics=coll.dobj['diagnostic'][key]['doptics'],
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

    sh = dsamp['x0']['data'].shape
    x0u = dsamp['x0']['data'][:, 0]
    x1u = dsamp['x1']['data'][0, :]
    x0f = dsamp['x0']['data'].ravel()
    x1f = dsamp['x1']['data'].ravel()

    pcross = Path(np.array([pcross0, pcross1]).T)
    ind = (
        dsamp['ind']['data']
        & pcross.contains_points(np.array([x0f, x1f]).T).reshape(sh)
    )
    nRZ = ind.sum()

    # R and Z indices
    ir, iz = ind.nonzero()
    iru = np.unique(ir)

    # ----------
    # get dphi_r

    phi_hor = np.arctan2(phor1, phor0)
    phimin = np.nanmin(phi_hor)
    phimax = np.nanmax(phi_hor)

    # get dphi vs phor
    dphi_r = _utilities._get_dphi_from_R_phor(
        R=x0u[iru],
        phor0=phor0,
        phor1=phor1,
        phimin=phimin,
        phimax=phimax,
        res=res_phi,
        out=True,
    )

    # --------------
    # prepare output

    indr = np.zeros((nRZ,), dtype=int)
    indz = np.zeros((nRZ,), dtype=int)
    ncounts = np.full(shape_cam, 0.)

    lp0, lp1 = [], []
    xx, yy = [], []

    dr = np.mean(np.diff(x0u))
    dz = np.mean(np.diff(x1u))
    ipts = 0
    pti = np.r_[0., 0., 0.]
    nru = iru.size
    for i00, i0 in enumerate(iru):

        indiz = ir == i0
        nz = indiz.sum()
        if np.all(np.isnan(dphi_r[:, i00])):
            ipts += nz
            continue

        nphi = np.ceil(
            x0u[i0]*(dphi_r[1, i00] - dphi_r[0, i00]) / res_phi
        ).astype(int)
        phir = np.linspace(dphi_r[0, i00], dphi_r[1, i00], nphi)
        cosphi = np.cos(phir)
        sinphi = np.sin(phir)

        dphi = phir[1] - phir[0]
        dv = dr * x0u[i0] * dphi * dz

        for i11, i1 in enumerate(iz[indiz]):

            indr[ipts] = i0
            indz[ipts] = i1
            pti[2] = x1u[i1]
            done_xy = []

            for i2, phii in enumerate(phir):

                # set point
                pti[0] = x0u[i0]*cosphi[i2]
                pti[1] = x0u[i0]*sinphi[i2]

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

                # compute image
                (
                    x0c, x1c,
                    angles, dsang,
                    vectx, vecty, vectz, iok,
                    dangmin_str, x0if, x1if,
                ) = _get_points_on_camera_from_pts(
                    p0=p0,
                    p1=p1,
                    pti=pti,
                    # ref
                    cent=cent,
                    nin=nin,
                    e0=e0,
                    e1=e1,
                    # dang
                    pix_size=pix_size,
                    dist_to_cam=dist_to_cam,
                    dang=dang,
                    phi=phii,
                    # resoluions
                    n0=n0,
                    n1=n1,
                    # functions
                    coords_x01toxyz_plane=coords_x01toxyz_plane,
                    ptsvect_spectro=ptsvect_spectro,
                    ptsvect_cam=ptsvect_cam,
                )[:11]

                # update
                lp0.append(x0c)
                lp1.append(x1c)
                if i2 not in done_xy:
                    xx.append(x0u[i0] * np.cos(phii))
                    yy.append(x0u[i0] * np.sin(phii))
                    done_xy.append(i2)

                if verb is True:
                    msg = (
                        f"\t\t{i00} / {nru}, {i11} / {nz}, {i2} / {nphi}"
                        f":  {iok.sum()} pts   "
                        f"\t dangmin: {dangmin_str}"
                    )
                    print(msg, end='\r')

                # safety check
                iok2 = (
                    (x0c[iok] >= cbin0[0])
                    & (x0c[iok] <= cbin0[-1])
                    & (x1c[iok] >= cbin1[0])
                    & (x1c[iok] <= cbin1[-1])
                )

                if not np.any(iok2):
                    continue

                # update index
                iok[iok] = iok2

                # 2d pixel by binning
                out = scpstats.binned_statistic_2d(
                    x0c[iok],
                    x1c[iok],
                    dsang[iok],
                    statistic='sum',
                    bins=(cbin0, cbin1),
                    expand_binnumbers=True,
                )

                ipixok = out.statistic > 0
                ncounts[ipixok] += out.statistic[ipixok] * dv

    # ------------------
    # length for etendue

    # mesh envelop
    dout = coll.get_mesh_outline(key=key_mesh)
    p0 = dout['x0']['data']
    p1 = dout['x1']['data']

    etendue, length = _get_etendue_length(
        coll=coll,
        doptics=coll.dobj['diagnostic'][key]['doptics'][key_cam],
        poly=np.array([p0, p1]),
    )

    return {
        'ncounts': {'data': ncounts / length, 'units': 'm3/m'},
        'lp0': {'data': lp0, 'units': 'm'},
        'lp1': {'data': lp1, 'units': 'm'},
        'ptsx': {'data': np.array(xx), 'units': 'm'},
        'ptsy': {'data': np.array(yy), 'units': 'm'},
        'ptsr': {'data': x0u[ir], 'units': 'm'},
        'ptsz': {'data': x1u[iz], 'units': 'm'},
    }


def _get_etendue_length(
    coll=None,
    doptics=None,
    poly=None,
):

    # ----------------------
    # get etendue and length

    ketend = doptics['etendue']
    etendue = coll.ddata[ketend]['data']

    # -------------------------
    # los through mesh envelopp

    # los
    klos = doptics['los']
    ptsx, ptsy, ptsz = coll.get_rays_pts(key=klos)
    vectx, vecty, vectz = coll.get_rays_vect(key=klos)

    iok = np.isfinite(vectx[-1, ...])
    length = np.full(vectx.shape[1:], np.nan)

    DD = np.array([
        ptsx[-2, ...][iok],
        ptsy[-2, ...][iok],
        ptsz[-2, ...][iok],
    ])
    uu = np.array([
        vectx[-1, ...][iok],
        vecty[-1, ...][iok],
        vectz[-1, ...][iok],
    ])

    # Prepare structures
    from ..geom import _core
    ves = _core.Ves(
        Poly=poly,
        Name='temp',
        Exp='',
    )

    conf = _core.Config(
        lStruct=[ves],
        Name='temp',
        Exp='',
    )

    # ray-tracing
    cam = _core.CamLOS1D(
        dgeom=(DD, uu),
        config=conf,
        Name='temp',
        Diag='',
        Exp='',
        strict=False,
    )

    # length
    length[iok] = (cam.dgeom['kOut'] - cam.dgeom['kIn'])

    return etendue, length


# ################################################
# ################################################
#           Sub-routine
# ################################################


def _get_points_on_camera_from_pts(
    p0=None,
    p1=None,
    pti=None,
    # ref
    cent=None,
    nin=None,
    e0=None,
    e1=None,
    # dang
    pix_size=None,
    dist_to_cam=None,
    dang=None,
    phi=None,
    # optional
    n0=None,
    n1=None,
    # functions
    coords_x01toxyz_plane=None,
    ptsvect_spectro=None,
    ptsvect_cam=None,
    # output
    ptsx_all=None,
    ptsy_all=None,
    ptsz_all=None,
    # debug
    debug=None,
):

    # anuglar resolution associated to pixels
    vect = cent - pti
    dist = np.linalg.norm(vect)
    vect = vect / dist
    dang_pix = pix_size / (dist_to_cam + dist)

    # dang
    dang_min = min(dang, 0.25*dang_pix)
    dangmin_str = f"rock {dang:.2e} vs {0.25*dang_pix:.2e} 1/4 pixel"

    # set n0, n1
    p0min, p0max = p0.min(), p0.max()
    p1min, p1max = p1.min(), p1.max()

    if n0 is None:
        cos0 = np.linalg.norm(np.cross(e0, vect))
        ang0 = cos0 * (p0max - p0min) / dist
        n0 = int(np.ceil(ang0 / dang_min)) + 2
    if n1 is None:
        cos1 = np.linalg.norm(np.cross(e1, vect))
        ang1 = cos1 * (p1max - p1min) / dist
        n1 = int(np.ceil(ang1 / dang_min)) + 2

    # sample 2d equivalent aperture
    margin0 = 1e-6 * (p0max - p0min)
    margin1 = 1e-6 * (p1max - p1min)
    x0i = np.linspace(p0min + margin0, p0max - margin0, n0)
    x1i = np.linspace(p1min + margin1, p1max - margin1, n1)
    dx0 = x0i[1] - x0i[0]
    dx1 = x1i[1] - x1i[0]

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

    # approx solid angles
    ax = xx - pti[0]
    ay = yy - pti[1]
    az = zz - pti[2]
    di = np.sqrt(ax**2 + ay**2 + az**2)
    cos = np.abs((nin[0] * ax + nin[1] * ay + nin[2] * az) / di)
    dsang = dx0 * dx1 * cos / di**2

    # get normalized vector from plasma point to crystal
    vectx = ax / di
    vecty = ay / di
    vectz = az / di

    # get local cosine vs toroidal direction (for doppler)
    cos = -vectx*np.sin(phi) + vecty*np.cos(phi)

    # get reflexion
    (
        ptsx1, ptsy1, ptsz1,
        vx, vy, vz,
        angles,
    ) = ptsvect_spectro(
        pts_x=pti[0],
        pts_y=pti[1],
        pts_z=pti[2],
        vect_x=vectx,
        vect_y=vecty,
        vect_z=vectz,
        strict=False,
        return_x01=False,
    )[:7]

    # get x0, x1 on camera
    x0c, x1c, ptsx2, ptsy2, ptsz2 = ptsvect_cam(
        pts_x=ptsx1,
        pts_y=ptsy1,
        pts_z=ptsz1,
        vect_x=vx,
        vect_y=vy,
        vect_z=vz,
    )

    return (
        x0c, x1c,
        angles, dsang,
        vectx, vecty, vectz, ind,
        dangmin_str, x0if, x1if,
        ptsx1, ptsy1, ptsz1,
        ptsx2, ptsy2, ptsz2,
    )
