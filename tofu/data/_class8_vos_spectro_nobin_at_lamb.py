# -*- coding: utf-8 -*-


import warnings


import numpy as np
import scipy.stats as scpstats
from matplotlib.path import Path
import matplotlib.pyplot as plt       # DB
import matplotlib.gridspec as gridspec
import Polygon as plg
import datastock as ds


from . import _class8_equivalent_apertures as _equivalent_apertures
from . import _class8_vos_utilities as _utilities
from . import _class8_reverse_ray_tracing as _reverse_rt
from . import _class8_vos as _vos


# ###############################################################
# ###############################################################
#                       Main
# ###############################################################


def compute_vos_nobin_at_lamb(
    coll=None,
    key_diag=None,
    key_cam=None,
    key_mesh=None,
    # wavelength
    lamb=None,
    # config
    config=None,
    # parameters
    res_RZ=None,
    res_phi=None,
    res_rock_curve=None,
    n0=None,
    n1=None,
    convexHull=None,
    # margins
    margin_poly=None,
    # options
    nmax_rays=None,
    # spectro-only
    rocking_curve_fw=None,
    rocking_curve_max=None,
    ## binning
    dobin=None,
    bin0=None,
    bin1=None,
    remove_raw=None,
    # bool
    visibility=None,
    convex=None,
    verb=None,
    debug=None,
    # plot
    plot=None,
    pix0=None,
    pix1=None,
    tit=None,
):

    # ------------
    # check inputs

    (
        # standard vos
        key_diag,
        key_mesh,
        spectro,
        is2d,
        doptics,
        dcompute,
        res_RZ,
        res_phi,
        convexHull,
        visibility,
        verb,
        debug,
        # specific
        dobin,
        bin0,
        bin1,
        remove_raw,
        lamb,
        nmax_rays,
        plot,
    ) = _check(**locals())

    # ------------
    # sample mesh

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

    sh = dsamp['x0']['data'].shape
    x0f = dsamp['x0']['data'].ravel()
    x1f = dsamp['x1']['data'].ravel()

    sh1 = tuple([ss + 2 for ss in sh])
    bool_cross = np.zeros(sh1, dtype=bool)

    x0u = dsamp['x0']['data'][:, 0]
    x1u = dsamp['x1']['data'][0, :]
    x0l = np.r_[x0u[0] - (x0u[1] - x0u[0]), x0u, x0u[-1] + (x0u[-1] - x0u[-2])]
    x1l = np.r_[x1u[0] - (x1u[1] - x1u[0]), x1u, x1u[-1] + (x1u[-1] - x1u[-2])]
    x0l = np.repeat(x0l[:, None], x1l.size, axis=1)
    x1l = np.repeat(x1l[None, :], x0l.shape[0], axis=0)

    dx0 = x0u[1] - x0u[0]
    dx1 = x1u[1] - x1u[0]

    # --------------
    # prepare optics

    doptics = coll._dobj['diagnostic'][key_diag]['doptics']

    # --------------
    # prepare optics

    dout = {}
    for k0 in dcompute.keys():

            # ------------------
            # call relevant func

            dout[k0] = _vos_nobin_at_lamb(
                # ressources
                coll=coll,
                doptics=doptics,
                key_diag=key_diag,
                key_cam=k0,
                dsamp=dsamp,
                lamb=lamb,
                # inputs sample points
                x0u=x0u,
                x1u=x1u,
                x0f=x0f,
                x1f=x1f,
                x0l=x0l,
                x1l=x1l,
                dx0=dx0,
                dx1=dx1,
                # options
                sh=sh,
                res_RZ=res_RZ,
                res_phi=res_phi,
                res_rock_curve=res_rock_curve,
                n0=n0,
                n1=n1,
                bool_cross=bool_cross,
                convexHull=convexHull,
                # parameters
                nmax_rays=nmax_rays,
                margin_poly=margin_poly,
                config=config,
                visibility=visibility,
                verb=verb,
                # debug
                debug=debug,
            )

            # -----------
            # add binning

            if dobin is True:
                _dobin(
                    coll=coll,
                    # camera
                    k0=k0,
                    # bins
                    bin0=bin0,
                    bin1=bin1,
                    # lamb
                    lamb=lamb,
                    # dout
                    dout=dout,
                    # remove raw data (lightweight)
                    remove_raw=remove_raw,
                )

            # -------------
            # add input info

            dout[k0]['keym'] = key_mesh
            dout[k0]['res_RZ'] = res_RZ
            dout[k0]['res_phi'] = res_phi
            dout[k0]['res_rock_curve'] = res_rock_curve

    # -------------
    # replace

    if plot is True:
        _plot(
            coll=coll,
            key_diag=key_diag,
            # output
            dout=dout,
            dobin=dobin,
            # plot
            pix0=pix0,
            pix1=pix1,
            tit=tit,
        )

    return dout


# ###########################################################
# ###########################################################
#               check
# ###########################################################


def _check(
    coll=None,
    key_diag=None,
    key_cam=None,
    key_mesh=None,
    # lamb
    lamb=None,
    # config
    config=None,
    # parameters
    res_RZ=None,
    res_phi=None,
    res_rock_curve=None,
    n0=None,
    n1=None,
    convexHull=None,
    # margins
    margin_poly=None,
    # options
    nmax_rays=None,
    # spectro-only
    rocking_curve_fw=None,
    rocking_curve_max=None,
    ## binning
    dobin=None,
    bin0=None,
    bin1=None,
    remove_raw=None,
    # bool
    visibility=None,
    convex=None,
    verb=None,
    debug=None,
    # plot
    plot=None,
    pix0=None,
    pix1=None,
    tit=None,
):

    # --------------------
    # standard vos check
    # --------------------

    (
        key_diag,
        key_mesh,
        spectro,
        is2d,
        doptics,
        dcompute,
        res_RZ,
        res_phi,
        _,
        convexHull,
        visibility,
        verb,
        debug,
    ) = _vos._check(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        key_mesh=key_mesh,
        res_RZ=res_RZ,
        res_phi=res_phi,
        convexHull=convexHull,
        visibility=visibility,
        verb=verb,
        debug=debug,
    )[:-2]

    if spectro is False:
        msg = "Routine can only be used for spectro diag, not '{key_diag}'"
        raise Exception(msg)

    # --------------------
    # specific checks
    # --------------------

    # dobin
    dobin = ds._generic_check._check_var(
        dobin, 'dobin',
        default=False,
        types=bool,
    )

    # bin0, bin1
    if dobin is True:

        lc = [
            bin0 is None and bin1 is None,
            np.isscalar(bin0) and np.isscalar(bin1),
            hasattr(bin0, '__iter__') and hasattr(bin1, '__iter__')
        ]
        if np.sum(lc) != 1:
            msg = (
                "If dobin = True, then bin0 and bin1 must either be:\n"
                f"\t- both None:  default to pixels edges\n"
                f"\t- both ints: nb of edges in camera surface\n"
                f"\t- both unique 1d np.ndarray: bins edges\n"
                "Provided:\n"
                f"\t- bin0 = {bin0}\n"
                f"\t- bin1 = {bin1}\n"
            )
            raise Exception(msg)

        if lc[1]:
            bin0, bin1 = int(bin0), int(bin1)
        elif lc[2]:
            bin0 = ds._generic_check._check_flat1darray(
                bin0, 'bin0',
                dtype=float,
                unique=True,
            )
            bin1 = ds._generic_check._check_flat1darray(
                bin1, 'bin1',
                dtype=float,
                unique=True,
            )

    else:
        bin0, nin1 = None, None

    # remove_raw
    remove_raw = ds._generic_check._check_var(
        remove_raw, 'remove_raw',
        default=False,
        types=bool,
    )

    # ------------------
    # check lamb

    # lamb
    lamb = np.unique(ds._generic_check._check_flat1darray(
        lamb, 'lamb',
        dtype=float,
        sign='>0',
    ))

    # --------------------
    # other parameters

    # check nmax_rays
    nmax_rays = int(ds._generic_check._check_var(
        nmax_rays, 'nmax_rays',
        types=(int, float),
        sign='>0',
        default=100000,
    ))

    # plot
    plot = ds._generic_check._check_var(
        plot, 'plot',
        types=bool,
        default=True,
    )

    # ----------
    # verb

    if verb is True:
        msg = f"\nComputing vos with no bin at lamb for diag '{key_diag}':"
        print(msg)

    return (
        # standard vos
        key_diag,
        key_mesh,
        spectro,
        is2d,
        doptics,
        dcompute,
        res_RZ,
        res_phi,
        convexHull,
        visibility,
        verb,
        debug,
        # specific
        dobin,
        bin0,
        bin1,
        remove_raw,
        lamb,
        nmax_rays,
        plot,
    )


# ###########################################################
# ###########################################################
#               Main
# ###########################################################


def _vos_nobin_at_lamb(
    # ressources
    coll=None,
    doptics=None,
    key_diag=None,
    key_cam=None,
    dsamp=None,
    lamb=None,
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
    res_phi=None,
    res_rock_curve=None,
    n0=None,
    n1=None,
    nmax_rays=None,
    convexHull=None,
    bool_cross=None,
    # parameters
    min_threshold=None,
    margin_poly=None,
    visibility=None,
    verb=None,
    # debug
    debug=None,
    # unused
    **kwdargs,
):
    """ vos computation for spectrometers """

    # -----------------
    # prepare optics

    (
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
    ) = _reverse_rt._prepare_optics(
        coll=coll,
        key=key_diag,
        key_cam=key_cam,
    )

    # --------------------------
    # get overall polygons

    pcross0, pcross1 = _utilities._get_overall_polygons(
        coll=coll,
        doptics=coll.dobj['diagnostic'][key_diag]['doptics'],
        key_cam=key_cam,
        poly='pcross',
        convexHull=convexHull,
    )

    phor0, phor1 = _utilities._get_overall_polygons(
        coll=coll,
        doptics=coll.dobj['diagnostic'][key_diag]['doptics'],
        key_cam=key_cam,
        poly='phor',
        convexHull=convexHull,
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
    nRZ = ind.sum()

    # R and Z indices
    ir, iz = ind.nonzero()
    iru = np.unique(ir)

    # ----------
    # get dphi_r

    # dphi = doptics[key_cam]['dvos']['dphi']
    # phimin = np.nanmin(dphi[0, :])
    # phimax = np.nanmax(dphi[1, :])

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

    # -------------------------------------
    # prepare lambda, angles, rocking_curve

    (
        nlamb,
        lamb,
        _,
        pow_ratio,
        ang_rel,
        dang,
        angbragg,
    ) = _reverse_rt._prepare_lamb(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        kspectro=kspectro,
        res_lamb=None,
        lamb=lamb,
        res_rock_curve=res_rock_curve,
        verb=verb,
    )

    angbragg0 = angbragg[:1, :]
    angbragg1 = angbragg[-1:, :]

    # linterpbragg = [
        # scpinterp.interp1d(angbragg[:, kk], pow_ratio, kind='linear')
        # for kk in range(angbragg.shape[1])
    # ]

    # --------------
    # prepare output

    shape_x01 = (nmax_rays, nlamb)
    x0 = np.full(shape_x01, np.nan)
    x1 = np.full(shape_x01, np.nan)
    ph_count = np.full(shape_x01, np.nan)

    nmax_real = np.zeros((nlamb,), dtype=int)
    indx01 = np.zeros((nlamb,), dtype=int)

    # ----------
    # verb

    if verb is True:
        msg = (
            f"\tlamb.shape: {lamb.shape}\n"
            f"\tang_rel.shape: {ang_rel.shape}\n"
            f"\tiru.size: {iru.size}\n"
            f"\tnRZ: {nRZ}\n"
        )
        print(msg)

    # ---------------------
    # loop in plasma points

    dr = np.mean(np.diff(x0u))
    dz = np.mean(np.diff(x1u))
    pti = np.r_[0., 0., 0.]
    nru = iru.size

    for i00, i0 in enumerate(iru):

        indiz = ir == i0
        nz = indiz.sum()
        if np.all(np.isnan(dphi_r[:, i00])):
            continue

        nphi = np.ceil(x0u[i0]*(dphi_r[1, i00] - dphi_r[0, i00]) / res_phi).astype(int)
        phir = np.linspace(dphi_r[0, i00], dphi_r[1, i00], nphi)
        cosphi = np.cos(phir)
        sinphi = np.sin(phir)

        dphi = phir[1] - phir[0]
        dv = dr * x0u[i0] * dphi * dz

        for i11, i1 in enumerate(iz[indiz]):

            pti[2] = x1u[i1]

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
                    phi=phii,
                    # resoluions
                    n0=n0,
                    n1=n1,
                    # functions
                    coords_x01toxyz_plane=coords_x01toxyz_plane,
                    ptsvect_spectro=ptsvect_spectro,
                    ptsvect_cam=ptsvect_cam,
                )[:9]

                if verb is True:
                    msg = (
                        f"\t\t{i00} / {nru}, {i11} / {nz}, {i2} / {nphi}"
                        f":  {iok.sum()} pts   "
                        f"\t dangmin: {dangmin_str}"
                    )
                    print(msg, end='\r')

                if not np.any(iok):
                    continue

                x0c = x0c[iok]
                x1c = x1c[iok]
                angles = angles[iok]
                dsang = dsang[iok]

                # ilamb
                ilamb = (
                    (angles[:, None] >= angbragg0)
                    & (angles[:, None] < angbragg1)
                )

                if not np.any(ilamb):
                    continue

                ilamb_n = np.any(ilamb, axis=0).nonzero()[0]

                for kk in ilamb_n:

                    # npts and indices of pts
                    npts = ilamb[:, kk].sum()

                    # update max
                    nmax_real[kk] += npts

                    # if too many points already => skip
                    if indx01[kk] >= nmax_rays:
                        continue

                    # indices of pts
                    imax = min(indx01[kk] + npts, nmax_rays)
                    indpts = np.arange(indx01[kk], imax)

                    delta = indx01[kk] + npts - nmax_rays
                    if delta > 0:
                        ilamb[ilamb[:, kk].nonzero()[0][-delta:], kk] = False

                    # x0, x1
                    x0[indpts, kk] = x0c[ilamb[:, kk]]
                    x1[indpts, kk] = x1c[ilamb[:, kk]]

                    # indices of power_ratio
                    inds = np.searchsorted(
                        angbragg[:, kk],
                        angles[ilamb[:, kk]],
                    )

                    # update power_ratio * solid angle
                    ph_count[indpts, kk] = (
                        pow_ratio[inds] * dsang[ilamb[:, kk]] * dv
                    )

                    # update index
                    indx01[kk] += npts

    # check nmax_real vs nmax_rays
    imax = nmax_real > nmax_rays
    if np.any(imax):
        lstr = [
            f"\t- {ll}: {nmax_real[ii]} vs {nmax_rays}"
            for ii, ll in enumerate(lamb)
        ]
        msg = (
            "\nSome wavelengths would see more rays than allowed:\n"
            + "\n".join(lstr)
        )
        warnings.warn(msg)

    # remove useless points
    iin = np.any(np.isfinite(ph_count), axis=1)
    if not np.all(iin):
        ph_count = ph_count[iin, :]
        x0 = x0[iin, :]
        x1 = x1[iin, :]

    # remove useless lamb
    iin = ph_count > 0.
    ilamb = np.any(np.any(np.any(iin, axis=0), axis=0), axis=0)
    if not np.all(ilamb):
        ph_count = ph_count[..., ilamb]
        lamb = lamb[ilamb]

    # --------------------
    # return

    # dout
    dout = {
        # lamb
        'lamb': lamb,
        # data
        'x0': x0,
        'x1': x1,
        'ph_count': ph_count,
        # information
        'nmax_real': nmax_real,
        'nmax_rays': nmax_rays,
    }

    return dout


# ################################################
# ################################################
#           do bin
# ################################################


def _dobin(
    coll=None,
    # camera
    k0=None,
    # bins
    bin0=None,
    bin1=None,
    # lamb
    lamb=None,
    # dout
    dout=None,
    # remove raw data
    remove_raw=None,
):

    # ----------------------------------
    # get bins (default = camera pixels)

    c0, c1 = coll.dobj['camera'][k0]['dgeom']['cents']
    if isinstance(bin0, np.ndarray) and isinstance(bin1, np.ndarray):
        pass

    else:
        c0 = coll.ddata[c0]['data']
        c1 = coll.ddata[c1]['data']

        c0min = c0[0] - 0.5*(c0[1] - c0[0])
        c0max = c0[-1] + 0.5*(c0[-1] - c0[-2])
        c1min = c1[0] - 0.5*(c1[1] - c1[0])
        c1max = c1[-1] + 0.5*(c1[-1] - c1[-2])

        if bin0 is None or bin1 is None:
            bin0 = np.r_[c0min, 0.5*(c0[1:] + c0[:-1]), c0max]
            bin1 = np.r_[c1min, 0.5*(c1[1:] + c1[:-1]), c1max]

        elif isinstance(bin0, int) and isinstance(bin1, int):
            bin0 = np.linspace(c0min, c0max, bin0)
            bin1 = np.linspace(c1min, c1max, bin1)

    # --------------------------
    # derive bin centers in 2d

    if isinstance(c0, str):
        c0 = 0.5 * (bin0[1:] + bin0[:-1])
        c1 = 0.5 * (bin1[1:] + bin1[:-1])

    n0, n1 = c0.size, c1.size

    # -----------------------
    # compute

    # points with weight
    bin_ph = np.full((n0, n1, lamb.size), 0.)
    for kk, ll in enumerate(lamb):

        iok = np.isfinite(dout[k0]['x0'][:, kk])
        if not np.any(iok):
            continue

        bin_ph[..., kk] = scpstats.binned_statistic_2d(
            dout[k0]['x0'][iok, kk],
            dout[k0]['x1'][iok, kk],
            dout[k0]['ph_count'][iok, kk],
            statistic='sum',
            bins=(bin0, bin1),
        ).statistic

    # --------
    # store

    dout[k0]['bin0'] = bin0
    dout[k0]['bin1'] = bin1
    dout[k0]['bin_ph'] = bin_ph

    if remove_raw is True:
        for k1 in ['x0', 'x1', 'ph_count']:
            del dout[k0][k1]

    return


# ################################################
# ################################################
#           plot
# ################################################


def _plot(
    coll=None,
    key_diag=None,
    # output
    dout=None,
    dobin=None,
    # plot
    pix0=None,
    pix1=None,
    tit=None,
):

    # -------------
    # prepare

    if tit is None:
        tit = (
            f"vos ray tracing for '{key_diag}'"
        )

    # ---------------------
    # prepare figure / axes

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(tit, size=12, fontweight='bold')

    dax = {}
    for ii, (k0, v0) in enumerate(dout.items()):

        # -------------------
        # prepare image

        if dobin is True:
            extent = (
                v0['bin0'][0], v0['bin0'][-1],
                v0['bin1'][0], v0['bin1'][-1],
            )

            im = np.sum(v0['bin_ph'], axis=-1)
            im[im == 0.] = np.nan

        # ------------
        # prepare axes

        ax0 = fig.add_subplot(2, len(dout), ii + 1, aspect='equal', adjustable='datalim')
        ax0.set_xlabel('x0 (m)', size=12, fontweight='bold')
        ax0.set_xlabel('x1 (m)', size=12, fontweight='bold')

        ax1 = fig.add_subplot(2, len(dout), ii + 2, aspect='equal', adjustable='datalim')
        ax1.set_xlabel('x0 (m)', size=12, fontweight='bold')
        ax1.set_xlabel('x1 (m)', size=12, fontweight='bold')

        dax[f'ax0_{k0}'] = {'handle': ax0}
        dax[f'ax1_{k0}'] = {'handle': ax1}

        # -----------------
        # plot

        # get total camera outline
        out0, out1 = coll.get_optics_outline(k0, total=True, closed=True)

        # camera outline
        ax0.plot(out0, out1, '-k')
        ax1.plot(out0, out1, '-k')

        # points without weight
        for kk, ll in enumerate(v0['lamb']):
            ax0.plot(
                v0['x0'][:, kk],
                v0['x1'][:, kk],
                ls='None',
                marker='.',
                label=r"$\lambda = $" + f" {ll*1e10:5.3f} A",
            )

        # points with weight
        if dobin is True:
            for kk in range(len(v0['lamb'])):
                ax1.imshow(
                    im.T,
                    extent=extent,
                    aspect='equal',
                    origin='lower',
                    interpolation='nearest',
                    vmin=0,
                    cmap=plt.cm.viridis,
                )

        # pixel
        if pix0 is not None:

            # get pixel outline
            out0, out1 = coll.get_optics_outline(k0, total=False, closed=True)

            # multiple pixels
            out0 = (pix0[:, None] + np.r_[out0, np.nan][None, :]).ravel()
            out1 = (pix1[:, None] + np.r_[out1, np.nan][None, :]).ravel()

            ax0.plot(out0, out1, '-k')
            ax1.plot(out0, out1, '-k')

        ax0.legend()

    return dax