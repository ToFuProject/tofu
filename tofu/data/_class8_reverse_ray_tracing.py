

import numpy as np
import scipy.interpolate as scpinterp
import scipy.stats as scpstats
import Polygon as plg
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import datastock as ds


from . import _class8_equivalent_apertures as _equivalent_apertures
from . import _class8_plot
from . import _generic_check
from . import _generic_plot
from . import _class8_vos_utilities as _utilities


# ################################################
# ################################################
#               main
# ################################################


def _from_pts(
    coll=None,
    # diag
    key=None,
    key_cam=None,
    # mesh sampling
    key_mesh=None,
    res_RZ=None,
    res_phi=None,
    # pts coordinates
    ptsx=None,
    ptsy=None,
    ptsz=None,
    # res
    res_rock_curve=None,
    n0=None,
    n1=None,
    min_threshold=None,
    # optional lamb
    lamb0=None,
    res_lamb=None,
    # options
    append=None,
    plot=None,
    plot_pixels=None,
    plot_config=None,
    plot_rays=None,
    vmin=None,
    vmax=None,
    # options
    dax=None,
    fs=None,
    dmargin=None,
    wintit=None,
):

    # -------------
    # check inputs

    (
        key, key_cam, key_mesh,
        dsamp, res_phi,
        shape, iok,
        ptsx, ptsy, ptsz, phi,
        lamb0,
        append, plot, plot_pixels,
    ) = _check(
        coll=coll,
        # diag
        key=key,
        key_cam=key_cam,
        # mesh sampling
        key_mesh=key_mesh,
        res_RZ=res_RZ,
        res_phi=res_phi,
        # pts coordinates
        ptsx=ptsx,
        ptsy=ptsy,
        ptsz=ptsz,
        # optional lamb
        lamb0=lamb0,
        # options
        append=append,
        plot=plot,
        plot_pixels=plot_pixels,
    )

    # ----------------------
    # prepare optics, camera

    (
        kspectro,
        lpoly_post,
        p0x, p0y, p0z,
        nin, e0, e1,
        ptsvect_plane,
        ptsvect_spectro,
        ptsvect_cam,
        coords_x01toxyz_plane,
        cent,
        cent_cam,
        dist_to_cam,
        pix_size,
        cbin0, cbin1,
    ) = _prepare_optics(
        coll=coll,
        key=key,
        key_cam=key_cam,
    )
    shape_cam = coll.dobj['camera'][key_cam]['dgeom']['shape']

    # -------------------------------------
    # prepare lambda, angles, rocking_curve

    (
        nlamb,
        lamb,
        dlamb,
        pow_ratio,
        ang_rel,
        dang,
        angbragg,
    ) = _prepare_lamb(
        coll=coll,
        key_diag=key,
        key_cam=key_cam,
        kspectro=kspectro,
        res_lamb=res_lamb,
        res_rock_curve=res_rock_curve,
        verb=False,
    )

    angbragg0 = angbragg[:1, :]
    angbragg1 = angbragg[-1:, :]

    # -------
    # prepare lamb per pixel

    lamb_per_pix = coll.get_diagnostic_data(
        key,
        key_cam=key_cam,
        data='lamb',
    )[0][key_cam]

    bragg_per_pix = coll.get_crystal_bragglamb(
        key=kspectro,
        lamb=lamb_per_pix,
    )[0]

    # --------------
    # prepare output

    if key_mesh is None:
        func = _loop0
    else:
        func = _loop1

    # -------------
    # compute

    dout = func(**locals())

    # -------
    # reshape

    if len(shape) > 1:
        shape_new = tuple(np.r_[shape_cam, shape])
        for k0 in ['sang']:
            dout[k0] = dout[k0].reshape(shape_new)
        for k0 in ['sang0']:
            dout[k0] = dout[k0].reshape(shape_new)

    # -------
    # plot

    if plot is True:
        _plot(
            coll=coll,
            key=key,
            key_cam=key_cam,
            dout=dout,
            cbin0=cbin0,
            cbin1=cbin1,
            # options
            plot_config=plot_config,
            plot_pixels=plot_pixels,
            vmin=vmin,
            vmax=vmax,
            # options
            dax=dax,
            fs=fs,
            dmargin=dmargin,
            wintit=wintit,
        )

    return dout


# ################################################
# ################################################
#               check
# ################################################


def _check(
    coll=None,
    # diag
    key=None,
    key_cam=None,
    # mesh sampling
    key_mesh=None,
    res_RZ=None,
    res_phi=None,
    # pts coordinates
    ptsx=None,
    ptsy=None,
    ptsz=None,
    # optional lamb
    lamb0=None,
    # bool
    append=None,
    plot=None,
    plot_pixels=None,
):

    # --------
    # key

    # key
    lok = [
        k0 for k0, v0 in coll._dobj.get('diagnostic', {}).items()
        if v0['spectro'] is True
    ]
    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok,
    )

    # key_cam
    lok = list(coll.dobj['diagnostic'][key]['doptics'].keys())
    key_cam = ds._generic_check._check_var(
        key_cam, 'key_cam',
        types=str,
        allowed=lok,
    )

    # ----------------------------
    # coordinates vs mesh sampling

    lc = [
        key_mesh is not None,
        ptsx is not None
    ]
    if np.sum(lc) != 1:
        msg = "Please provide key_mesh xor (ptsx, ptsy, ptsz)\n"
        raise Exception(msg)

    if lc[0]:

        lok = list(coll.dobj.get('mesh', {}).keys())
        key_mesh = ds._generic_check._check_var(
            key_mesh, 'key_mesh',
            types=str,
            allowed=lok,
        )

        dsamp = coll.get_sample_mesh(
            key=key_mesh,
            res=res_RZ,
            mode='abs',
            grid=True,
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

        # res_RZ
        if res_RZ is None:
            res_RZ = 0.01

        # res_phi
        if res_phi is None:
            res_phi = 0.01

        iok, shape, phi = None, None, None

    else:

        ptsx = np.atleast_1d(ptsx)
        ptsy = np.atleast_1d(ptsy)
        ptsz = np.atleast_1d(ptsz)

        # check shapes
        dshape = {
            'ptsx': ptsx.shape,
            'ptsy': ptsy.shape,
            'ptsz': ptsz.shape,
        }
        maxsize = np.max([np.prod(v0) for v0 in dshape.values()])
        shape = [v0 for v0 in dshape.values() if np.prod(v0) == maxsize][0]

        lout = [
            k0 for k0, v0 in dshape.items()
            if v0 not in [(1,), shape]
        ]
        if len(lout) > 0:
            lstr = [f"\t-{k0}.shape: {v0}" for k0, v0 in dshape.items()]
            msg = (
                "All coordinates arrays must be of same shape:\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

        # uniformize shapes
        if shape != (1,):
            if ptsx.shape == (1,):
                ptsx = np.full(shape, ptsx[0])
            if ptsy.shape == (1,):
                ptsy = np.full(shape, ptsy[0])
            if ptsz.shape == (1,):
                ptsz = np.full(shape, ptsz[0])

        dsamp = None

    # -------
    # lamb

    if lamb0 is not None:
        lamb0 = np.atleast_1d(lamb0)

        if lc[0]:
            if lamb0.shape != (1,):
                msg = "With mesh sampling, lamb0 must be a scalar!"
                raise Exception(msg)

        else:
            if lamb0.shape == (1,):
                lamb0 = np.full(ptsx.shape, lamb0[0])

            elif lamb0.shape != ptsx.shape:
                msg = (
                    "Arg lamb0 must be the same shape as coordinates\n"
                    "\t- ptsx.shape = {ptsx.shape}\n"
                    "\t- lamb.shape = {lamb.shape}\n"
                )
                raise Exception(msg)

    # -----------------------
    # store shape and flatten

    if lc[0]:
        pass

    else:
        if len(shape) > 1:
            ptsx = ptsx.ravel()
            ptsy = ptsy.ravel()
            ptsz = ptsz.ravel()
            if lamb0 is not None:
                lamb0 = lamb0.ravel()

        iok = np.isfinite(ptsx) & np.isfinite(ptsy) & np.isfinite(ptsz)
        phi = np.arctan2(ptsy, ptsx)

    # ---------
    # options

    # plot
    plot = ds._generic_check._check_var(
        plot, 'plot',
        types=bool,
        default=True,
    )

    # plot_pixel
    plot_pixels = ds._generic_check._check_var(
        plot_pixels, 'plot_pixels',
        types=bool,
        default=False,
    )

    # append
    append = ds._generic_check._check_var(
        append, 'append',
        types=bool,
        default=lc[1] and ptsx.size < 100 and plot is True,
    )

    return  (
        key, key_cam, key_mesh,
        dsamp, res_phi,
        shape, iok,
        ptsx, ptsy, ptsz, phi,
        lamb0,
        append, plot, plot_pixels,
    )


# ################################################
# ################################################
#               prepare optics
# ################################################


def _prepare_optics(
    coll=None,
    key=None,
    key_cam=None,
):

    doptics = coll.dobj['diagnostic'][key]['doptics']
    lop = doptics[key_cam]['optics']
    lop, lcls = coll.get_optics_cls(lop)

    cls_spectro = 'crystal'
    kspectro = lop[lcls.index(cls_spectro)]
    ispectro = lop.index(kspectro)
    if len(lop[:ispectro]) > 1:
        msg = "Not yet implemented optics between crystal and camera!"
        raise NotImplementedError()

    # lpoly_post = []
    lpoly_post = [
        coll.get_optics_poly(
            key=k0,
            add_points=None,
            return_outline=False,
        )
        for k0 in lop[ispectro+1:]
    ]

    # get initial polygon
    p0x, p0y, p0z = coll.get_optics_poly(key=kspectro, add_points=None)

    # unit vectors
    nin = coll.dobj[cls_spectro][kspectro]['dgeom']['nin']
    e0 = coll.dobj[cls_spectro][kspectro]['dgeom']['e0']
    e1 = coll.dobj[cls_spectro][kspectro]['dgeom']['e1']

    # get functions
    ptsvect_plane = coll.get_optics_reflect_ptsvect(key=kspectro, asplane=True)
    ptsvect_spectro = coll.get_optics_reflect_ptsvect(key=kspectro, isnorm=True)
    ptsvect_cam = coll.get_optics_reflect_ptsvect(key=key_cam, fast=True)

    coords_x01toxyz_plane = coll.get_optics_x01toxyz(
        key=kspectro,
        asplane=True,
    )

    # Get centers of crystal and camera to estimate distance
    cent_spectro = coll.dobj[cls_spectro][kspectro]['dgeom']['cent']
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

    return (
        kspectro,
        lpoly_post,
        p0x, p0y, p0z,
        nin, e0, e1,
        ptsvect_plane,
        ptsvect_spectro,
        ptsvect_cam,
        coords_x01toxyz_plane,
        cent_spectro,
        cent_cam,
        dist_to_cam,
        pix_size,
        cbin0, cbin1,
    )


# ################################################
# ################################################
#           Prepare lambda
# ################################################


def _prepare_lamb(
    coll=None,
    key_diag=None,
    key_cam=None,
    kspectro=None,
    lamb=None,
    res_lamb=None,
    res_rock_curve=None,
    verb=None,
):

    # ------------------
    # get lamb

    if res_lamb is not None:
        lamb = coll.get_diagnostic_data(
            key=key_diag,
            key_cam=key_cam,
            data='lamb',
        )[0][key_cam]

        lambmin = np.nanmin(lamb)
        lambmax = np.nanmax(lamb)
        Dlamb = (lambmax - lambmin) * 1.1
        nlamb = int(np.ceil(Dlamb / res_lamb)) + 2
        lamb = np.linspace(lambmin - 0.2*Dlamb, lambmax + 0.2*Dlamb, nlamb)
        dlamb = lamb[1] - lamb[0]

        bragg = coll.get_crystal_bragglamb(key=kspectro, lamb=lamb)[0]
    
    elif lamb is not None:
        
        lamb = ds._generic_check._check_flat1darray(
            lamb, 'lamb',
            dtype=float,
            sign='>0',
            unique=True,
        )
        nlamb = lamb.size
        dlamb = None
        bragg = coll.get_crystal_bragglamb(key=kspectro, lamb=lamb)[0]

    # ---------------
    # get bragg angle

    # power ratio
    cls_spectro = 'crystal'
    kpow = coll.dobj[cls_spectro][kspectro]['dmat']['drock']['power_ratio']
    pow_ratio = coll.ddata[kpow]['data']

    # angle relative
    kang_rel = coll.dobj[cls_spectro][kspectro]['dmat']['drock']['angle_rel']
    ang_rel = coll.ddata[kang_rel]['data']
    if res_rock_curve is not None:
        if isinstance(res_rock_curve, int):
            nang = res_rock_curve
        else:
            nang = int(
                (np.max(ang_rel) - np.min(ang_rel)) / res_rock_curve
            )

        ang_rel2 = np.linspace(np.min(ang_rel), np.max(ang_rel), nang)
        pow_ratio = scpinterp.interp1d(
            ang_rel,
            pow_ratio,
            kind='linear',
        )(ang_rel2)
        ang_rel = ang_rel2

    dang = np.mean(np.diff(ang_rel))

    # --------------------------------------
    # overall bragg angle with rocking curve

    if res_lamb is None and lamb is None:
        nlamb, lamb, dlamb, angbragg = None, None, None, None

    else:
        angbragg = bragg[None, :] + ang_rel[:, None]

        # ------------
        # safety check

        FW = coll.dobj[cls_spectro][kspectro]['dmat']['drock']['FW']
        dd0 = bragg[0] + np.r_[0, ang_rel[-1] - ang_rel[0]]
        dd1 = bragg[0] + np.r_[0, FW]
        dd2 = bragg[0] + np.r_[0, ang_rel[1] - ang_rel[0]]

        dlamb_max = np.diff(coll.get_crystal_bragglamb(key=kspectro, bragg=dd0)[1])
        dlamb_mh = np.diff(coll.get_crystal_bragglamb(key=kspectro, bragg=dd1)[1])
        dlamb_res = np.diff(coll.get_crystal_bragglamb(key=kspectro, bragg=dd2)[1])

        if verb is True and res_lamb is not None:
            msg = (
                "Recommended res_lamb to ensure rocking curve overlap:\n"
                f"\t- edge-edge: \t{dlamb_max[0]:.2e}\n"
                f"\t- MH-to-MH: \t{dlamb_mh[0]:.2e}\n"
                f"\t- resolution: \t{dlamb_res[0]:.2e}\n"
                f"\tProvided: \t{dlamb:.2e}\n"
            )
            print(msg)

    return (
        nlamb,
        lamb,
        dlamb,
        pow_ratio,
        ang_rel,
        dang,
        angbragg,
    )


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
    bragg_per_pix=None,
    ang_rel=None,
    pow_ratio=None,
    angbragg=None,
    angbragg0=None,
    angbragg1=None,
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

    if append is True:
        lp0, lp1 = [], []
        lpx, lpy, lpz = [], [], []
    else:
        lp0, lp1 = None, None
        lpx, lpy, lpz = None, None, None

    pwr_interp = scpinterp.interp1d(
        ang_rel,
        pow_ratio,
        kind='linear',
        bounds_error=False,
        fill_value=0.,
    )

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
            x0c, x1c, angles, dsang, cosi, iok,
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

                # lambref
                arelj = angles[indj] - bragg_per_pix[ii, jj]

                sang[ii, jj, i0] += np.sum(
                    pwr_interp(arelj)
                    * dsang[indj]
                )

                # ----------
                # sang_lamb

                angj = angles[indj]
                ilamb = (
                    (angj[:, None] >= angbragg0)
                    & (angj[:, None] < angbragg1)
                )

                if not np.any(ilamb):
                    continue

                ilamb_n = np.any(ilamb, axis=0).nonzero()[0]

                # binning of angles
                for kk in ilamb_n:
                    inds = np.searchsorted(
                        angbragg[:, kk],
                        angj[ilamb[:, kk]],
                    )

                    # update power_ratio * solid angle
                    sang_lamb[ii, jj] += np.sum(
                        pow_ratio[inds]
                        * dsang[indj][ilamb[:, kk]]
                    )

    return {
        'ncounts': ncounts,
        'sang0': sang0,
        'sang': sang,
        'sang_lamb': sang_lamb * dlamb,
        'lamb': lamb,
        'dlamb': dlamb,
        'lp0': lp0,
        'lp1': lp1,
        'lpx': lpx,
        'lpy': lpy,
        'lpz': lpz,
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

        nphi = np.ceil(x0u[i0]*(dphi_r[1, i00] - dphi_r[0, i00]) / res_phi).astype(int)
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
                    x0c, x1c, angles, dsang, cosi, iok,
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
                )[:9]

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
        'ncounts': ncounts / length,
        'lp0': lp0,
        'lp1': lp1,
        'ptsx': np.array(xx),
        'ptsy': np.array(yy),
        'ptsr': x0u[ir],
        'ptsz': x1u[iz],
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
        x0c, x1c, angles, dsang, cos, ind,
        dangmin_str, x0if, x1if,
        ptsx1, ptsy1, ptsz1,
        ptsx2, ptsy2, ptsz2,
    )


# ####################################################
# ####################################################
#               plot
# ####################################################


def _plot(
    coll=None,
    key=None,
    key_cam=None,
    dout=None,
    # auxiliary
    cbin0=None,
    cbin1=None,
    is2d=None,
    # options
    plot_config=None,
    plot_pixels=None,
    vmin=None,
    vmax=None,
    # options
    dax=None,
    fs=None,
    dmargin=None,
    wintit=None,
):

    # --------------
    # add ncounts to ddata

    ktemp = f'{key_cam}_ncounts'
    nn = len([
        k0 for k0 in coll.ddata.keys()
        if k0.startswith(ktemp)
    ])
    if nn > 0:
        ktemp = f'{ktemp}{nn}'

    coll.add_data(
        key=ktemp,
        data=dout['ncounts'],
        ref=coll.dobj['camera'][key_cam]['dgeom']['ref'],
        units='counts',
    )

    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = np.nanmax(dout['ncounts'])

    # --------------
    # prepare

    dplot = coll.get_diagnostic_dplot(
        key=key,
        key_cam=key_cam,
        optics=None,
        elements='o',
    )

    # --------------
    # plot_pixels

    out0, out1 = coll.get_optics_outline(key_cam, total=True)
    ck0f = np.array([cbin0, cbin0, np.full((cbin0.size,), np.nan)])
    ck1f = np.array([cbin1, cbin1, np.full((cbin1.size,), np.nan)])
    ck01 = np.r_[np.min(cbin1), np.max(cbin1), np.nan]
    ck10 = np.r_[np.min(cbin0), np.max(cbin0), np.nan]

    # --------------
    # pepare dax

    if dax is None:
        dax, cax = _get_dax(
            fs=fs,
            dmargin=dmargin,
        )
        # dax = _generic_plot.get_dax_diag(
            # proj=['cross', 'hor', '3d', 'camera'],
            # dmargin=dmargin,
            # fs=fs,
            # wintit=wintit,
            # tit=key,
            # is2d=True,
            # key_cam=['rays', 'binned'],
        # )

    dax = _generic_check._check_dax(dax=dax, main='cross')

    # --------------
    # prepare camera

    kax = 'cam_all'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ax.plot(
            np.r_[out0, out0[0]],
            np.r_[out1, out1[0]],
            c='k',
            ls='-',
            lw=0.5,
        )
        if plot_pixels is True:
            ax.plot(
                ck0f.T.ravel(),
                np.tile(ck01, cbin0.size),
                c='k',
                ls='-',
                lw=0.1,
            )
            ax.plot(
                np.tile(ck10, cbin1.size),
                ck1f.T.ravel(),
                c='k',
                ls='-',
                lw=0.1,
            )


    # -------------
    # mesh sampling

    if dout.get('lpx') is None:

        kax = 'cross'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            ax.plot(
                dout['ptsr'],
                dout['ptsz'],
                ls='None',
                lw=1.,
                marker='.',
            )

        kax = 'hor'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            ax.plot(
                dout['ptsx'],
                dout['ptsy'],
                ls='None',
                lw=1.,
                marker='.',
            )

        kax = 'cam_all'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            p0 = np.concatenate([pp.ravel() for pp in dout['lp0']])
            p1 = np.concatenate([pp.ravel() for pp in dout['lp1']])
            ax.plot(
                p0,
                p1,
                ls='None',
                marker='.',
            )

    else:

        # --------------
        # ray-tracing

        for ii in range(len(dout['lp0'])):

            p0 = dout['lp0'][ii]
            p1 = dout['lp1'][ii]

            # --------------
            # camera image

            kax = 'cam_all'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                ax.plot(
                    p0,
                    p1,
                    ls='None',
                    marker='.',
                )

            # --------------
            # rays

            ind = np.arange(3, dout['lpx'][ii].size, 3)
            px = np.insert(dout['lpx'][ii].T.ravel(), ind, np.nan)
            py = np.insert(dout['lpy'][ii].T.ravel(), ind, np.nan)
            pz = np.insert(dout['lpz'][ii].T.ravel(), ind, np.nan)

            kax = 'cross'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                ax.plot(
                    np.hypot(px, py),
                    pz,
                    ls='-',
                    lw=1.,
                )

            kax = 'hor'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                ax.plot(
                    px,
                    py,
                    ls='-',
                    lw=1.,
                )

            kax = '3d'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                ax.plot(
                    px,
                    py,
                    pz,
                    ls='-',
                    lw=1.,
                )

    # ----------
    # counts

    kax = 'cam_bin'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        extent = (cbin0[0], cbin0[-1], cbin1[0], cbin1[-1])
        im = ax.imshow(
            dout['ncounts'].T,
            extent=extent,
            interpolation='nearest',
            origin='lower',
            vmin=vmin,
            vmax=vmax,
        )

        plt.colorbar(im, ax=ax, cax=cax)

    # ----------
    # diag geom

    _class8_plot._plot_diag_geom(
        dax=dax,
        key_cam=key_cam,
        dplot=dplot,
        is2d=coll.dobj['diagnostic'][key]['is2d'],
    )

    # -------
    # config

    if plot_config.__class__.__name__ == 'Config':

        kax = 'cross'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj=kax, dLeg=False)

        kax = 'hor'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj=kax, dLeg=False)

    coll.remove_data(ktemp)
    return


def _get_dax(
    fs=None,
    dmargin=None,
):

    # ----------
    # check

    if fs is None:
        fs = (15, 9)

    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.05, 'top': 0.95,
            'left': 0.05, 'right': 0.95,
            'wspace': 0.25, 'hspace': 0.2,
        },
    )

    # --------
    # prepare

    ncols = 31
    nh = int((ncols - 1) / 2)
    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(ncols=ncols, nrows=4, **dmargin)

    # --------
    # create

    ax0 = fig.add_subplot(gs[:2, :nh], aspect='equal', adjustable='datalim')
    ax0.set_xlabel('X (m)')
    ax0.set_ylabel('Y (m)')

    ax1 = fig.add_subplot(gs[2:, :nh], aspect='equal', adjustable='datalim')
    ax1.set_xlabel('R (m)')
    ax1.set_ylabel('Z (m)')

    ax2 = fig.add_subplot(gs[:2, nh:-1], projection='3d')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_ylabel('Z (m)')

    ax3 = fig.add_subplot(gs[2, nh:-1], aspect='equal')
    ax3.set_ylabel('x1 (m)')

    ax4 = fig.add_subplot(gs[3, nh:-1], sharex=ax3, sharey=ax3)
    ax4.set_ylabel('x1 (m)')
    ax4.set_xlabel('x0 (m)')

    cax = fig.add_subplot(gs[3, -1])

    dax = {
        'cross': {'handle': ax0, 'type': 'cross'},
        'hor': {'handle': ax1, 'type': 'hor'},
        '3d': {'handle': ax2, 'type': '3d'},
        'cam_all': {'handle': ax3, 'type': 'camera'},
        'cam_bin': {'handle': ax4, 'type': 'camera'},
    }

    return dax, cax
