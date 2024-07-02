# -*- coding: utf-8 -*-


import datetime as dtm      # DB
import numpy as np
import scipy.stats as scpstats
from matplotlib.path import Path
import matplotlib.pyplot as plt       # DB
import matplotlib.gridspec as gridspec
import Polygon as plg


from . import _class8_equivalent_apertures as _equivalent_apertures
from . import _class8_vos_utilities as _utilities
from . import _class8_reverse_ray_tracing as _reverse_rt


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
    res_phi=None,
    lamb=None,
    res_lamb=None,
    res_rock_curve=None,
    n0=None,
    n1=None,
    convexHull=None,
    bool_cross=None,
    # parameters
    min_threshold=None,
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
        dlamb,
        pow_interp,
        ang_rel,
        dang,
        bragg,
    ) = _reverse_rt._prepare_lamb(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        kspectro=kspectro,
        lamb=lamb,
        res_lamb=res_lamb,
        res_rock_curve=res_rock_curve,
        verb=verb,
    )

    # --------------
    # prepare output

    shape_cam = coll.dobj['camera'][key_cam]['dgeom']['shape']
    is2d = len(shape_cam) == 2

    shape0 = tuple(np.r_[shape_cam, nRZ])
    ncounts = np.full(shape0, 0.)
    cos = np.full(shape0, 0.)
    phi_mean = np.full(shape0, 0.)
    phi_min = np.full(shape0, np.inf)
    phi_max = np.full(shape0, -np.inf)
    indr = np.zeros((nRZ,), dtype=int)
    indz = np.zeros((nRZ,), dtype=int)
    dV = np.full((nRZ,), np.nan)

    shape1 = tuple(np.r_[shape_cam, nRZ, nlamb])
    ph_count = np.full(shape1, 0.)

    etendlen = np.full(shape_cam, 0.)
    # ph_approx = np.full(shape1, 0.)
    # sang = np.full(shape1, 0.)
    # dang_rel = np.full(shape1, 0.)
    # nphi_all = np.full(shape1, 0.)
    # FW = coll.dobj[cls_spectro][kspectro]['dmat']['drock']['FW']
    # kp = coll.dobj[cls_spectro][kspectro]['dmat']['drock']['power_ratio']
    # POW = coll.ddata[kp]['data'].max()

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
            f"\tnRZ: {nRZ}\n"
        )
        print(msg)

    # ---------------------
    # loop in plasma points

    if debug is True:
        dx0 = {
            i0: {
                i1: [] for i1 in np.unique(iz[ir == i0])
            }
            for i0 in iru
        }
        dx1 = {
            i0: {
                i1: [] for i1 in np.unique(iz[ir == i0])
            }
            for i0 in iru
        }

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

        nphi = max(
            np.ceil(
                x0u[i0]*(dphi_r[1, i00] - dphi_r[0, i00]) / res_phi
            ).astype(int),
            3,
        )

        phir = np.linspace(dphi_r[0, i00], dphi_r[1, i00], nphi)
        cosphi = np.cos(phir)
        sinphi = np.sin(phir)

        dphi = phir[1] - phir[0]
        dv = dr * x0u[i0] * dphi * dz

        for i11, i1 in enumerate(iz[indiz]):

            indr[ipts] = i0
            indz[ipts] = i1
            dV[ipts] = dv
            pti[2] = x1u[i1]

            for i2, phii in enumerate(phir):

                if timing:
                    t000 = dtm.datetime.now()     # DB

                # set point
                pti[0] = x0u[i0] * cosphi[i2]
                pti[1] = x0u[i0] * sinphi[i2]
                # phi[ipts] = phii

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

                if timing:
                    t111 = dtm.datetime.now()     # DB
                    dt111 += (t111-t000).total_seconds()

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

                if timing:
                    # dt1111, dt2222, dt3333, dt4444 = out
                    t222 = dtm.datetime.now()     # DB
                    dt222 += (t222-t111).total_seconds()

                # safety check
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
                    dx0[i0][i1].append(x0c)
                    dx1[i0][i1].append(x1c)
                # -------- END DEBUG ----------


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
                ncounts[ipixok, ipts] += out.statistic[ipixok]

                # temporary
                outsa = scpstats.binned_statistic_2d(
                    x0c[iok],
                    x1c[iok],
                    dsang[iok],
                    statistic='sum',
                    bins=(cbin0, cbin1),
                    expand_binnumbers=True,
                )
                etendlen[ipixok] += outsa.statistic[ipixok] * dv

                # adjust phimean
                phi_mean[ipixok, ipts] += phii * out.statistic[ipixok]

                cosi = cosi[iok]
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

                        # phi_min, phi_max
                        phi_min[ii, jj, ipts] = min(phi_min[ii, jj, ipts], phii)
                        phi_max[ii, jj, ipts] = max(phi_max[ii, jj, ipts], phii)

                        # cos
                        cos[ii, jj, ipts] += np.sum(cosi[indj])

                        # ilamb
                        angj = angles[indj]
                        ilamb = (
                            (angj[:, None] - bragg >= ang_rel[0])
                            & (angj[:, None] - bragg < ang_rel[-1])
                        )

                        if not np.any(ilamb):
                            continue

                        ilamb_n = np.any(ilamb, axis=0).nonzero()[0]

                        # nphi_all  # DB
                        # nphi_all[ii, jj, ipts, ilamb_n] += 1

                        # if False:
                        # binning of angles
                        for kk in ilamb_n:
                            ph_count[ii, jj, ipts, kk] += np.sum(
                                pow_interp(angj[ilamb[:, kk]] - bragg[kk])
                                * dsang[indj][ilamb[:, kk]]
                            ) * dv

                            # inds = np.searchsorted(
                            #     angbragg[:, kk],
                            #     angj[ilamb[:, kk]],
                            # )

                            # # update power_ratio * solid angle
                            # ph_count[ii, jj, ipts, kk] += np.sum(
                            #     pow_ratio[inds]
                            #     * dsang[indj][ilamb[:, kk]]
                            # ) * dv

                            # ------  DEBUG -------
                            # ph_approx[ii, jj, ipts, kk] += (
                                # POW
                                # * FW
                                # * np.sum(dang1[0, np.any(iok, axis=0)])
                            # ) * dv

                            # # sang
                            # sang[ii, jj, ipts, kk] += np.sum(
                                # dsang[indj][ilamb[:, kk]]
                            # )

                            # # dang_rel
                            # dang_rel[ii, jj, ipts, kk] += np.sum(
                                # dsang[indj][ilamb[:, kk]]
                            # )

                if timing:
                    t333 = dtm.datetime.now()     # DB
                    dt333 += (t333-t222).total_seconds()

            # update index
            ipts += 1

    # multiply by dlamb
    # Now done during synthetic signal compute (binning vs interp)
    # ph_count *= dlamb

    if timing:
        t22 = dtm.datetime.now()     # DB

    # remove useless points
    iin = np.any(np.any(ncounts > 0, axis=0), axis=0)
    if not np.all(iin):
        ncounts = ncounts[:, :, iin]
        cos = cos[:, :, iin]
        phi_mean = phi_mean[:, :, iin]
        phi_min = phi_min[:, :, iin]
        phi_max = phi_max[:, :, iin]
        ph_count = ph_count[:, :, iin, :]
        indr = indr[iin]
        indz = indz[iin]
        dV = dV[iin]
        # DEBUG
        # sang = sang[:, :, iin]
        # ph_approx = ph_approx[:, :, iin, :]
        # dang_rel = dang_rel[:, :, iin, :]
        # nphi_all = nphi_all[:, :, iin, :]

    # remove useless lamb
    iin = ph_count > 0.
    ilamb = np.any(np.any(np.any(iin, axis=0), axis=0), axis=0)
    if not np.all(ilamb):
        ph_count = ph_count[..., ilamb]
        lamb = lamb[ilamb]
        # DEBUG
        # sang = sang[..., ilamb]
        # ph_approx = ph_approx[..., ilamb]
        # dang_rel = dang_rel[..., ilamb]
        # nphi_all = nphi_all[..., ilamb]

    # average cos and phi_mean
    iout = ncounts == 0
    cos[~iout] = cos[~iout] / ncounts[~iout]
    phi_mean[~iout] = phi_mean[~iout] / ncounts[~iout]

    # clear
    cos[iout] = np.nan
    phi_mean[iout] = np.nan
    phi_min[iout] = np.nan
    phi_max[iout] = np.nan
    # ph_count[iout, :] = np.nan
    # DEBUG
    # sang[iout, :] = np.nan
    # ph_approx[iout, :] = np.nan
    # dang_rel[iout, :] = np.nan
    # nphi_all[iout, :] = np.nan

    lresh = [1, 1, lamb.size]
    if is2d:
        lresh.insert(0, 1)
    # phtot = np.sum(ph_count, axis=(-1, -2))
    # lamb0 = np.sum(ph_count * lamb.reshape(lresh), axis=(-1, -2)) / phtot
    # dlamb0 = dlamb * phtot / np.max(np.sum(ph_count, axis=-2), axis=-1)

    # ------ DEBUG --------
    if debug is True:
        _plot_debug(
            coll=coll,
            key_cam=key_cam,
            cbin0=cbin0,
            cbin1=cbin1,
            dx0=dx0,
            dx1=dx1,
        )
    # ---------------------

    # ----------------
    # prepare output

    # ref
    knpts = f'{key_diag}_{key_cam}_vos_npts'
    knlamb = f'{key_diag}_{key_cam}_vos_nlamb'

    # data
    klamb = f'{key_diag}_{key_cam}_vos_lamb'
    kir = f'{key_diag}_{key_cam}_vos_ir'
    kiz = f'{key_diag}_{key_cam}_vos_iz'
    kph = f'{key_diag}_{key_cam}_vos_ph'
    kcos = f'{key_diag}_{key_cam}_vos_cos'
    knc = f"{key_diag}_{key_cam}_vos_nc"
    kphimin = f'{key_diag}_{key_cam}_vos_phimin'
    kphimax = f'{key_diag}_{key_cam}_vos_phimax'

    # optional data
    kdV = f'{key_diag}_{key_cam}_vos_dV'
    ketl = f"{key_diag}_{key_cam}_vos_etendl"
    # klamb0 = f'{key_cam}_vos_lamb0'
    # kdlamb = f'{key_cam}_vos_dlamb'
    kphimean = f'{key_diag}_{key_cam}_vos_phimean'

    refcam = coll.dobj['camera'][key_cam]['dgeom']['ref']
    ref = tuple(list(refcam) + [knpts])
    refph = tuple(list(ref) + [knlamb])

    # -------------------
    # format output

    # dref
    dref = {
        'npts': {
            'key': knpts,
            'size': indr.shape[-1],
        },
        'nlamb': {
            'key': knlamb,
            'size': lamb.size,
        },
    }


    # dout
    dout = {
        'pcross0': None,
        'pcross1': None,
        'phor0': None,
        'phor1': None,
        # lamb
        'lamb': {
            'key': klamb,
            'data': lamb,
            'ref': (knlamb,),
            'units': 'm',
            'dim': 'distance',
        },

        # coordinates
        'indr_cross': {
            'key': kir,
            'data': indr,
            'ref': (knpts,),
            'units': None,
            'dim': 'index',
        },
        'indz_cross': {
            'key': kiz,
            'data': indz,
            'ref': (knpts,),
            'units': None,
            'dim': 'index',
        },

        # data
        'cos': {
            'key': kcos,
            'data': cos,
            'ref': ref,
            'units': None,
            'dim': 'cos',
        },
        'ncounts': {
            'key': knc,
            'data': ncounts,
            'ref': ref,
            'units': None,
            'dim': 'counts',
        },
        'ph': {
            'key': kph,
            'data': ph_count,
            'ref': refph,
            'units': 'sr.m3',
            'dim': 'transfert',
        },
        'phi_min': {
            'key': kphimin,
            'data': phi_min,
            'ref': ref,
            'units': 'rad',
            'dim': 'angle',
        },
        'phi_max': {
            'key': kphimax,
            'data': phi_max,
            'ref': ref,
            'units': 'rad',
            'dim': 'angle',
        },

        # optional
        'phi_mean': {
            'key': kphimean,
            'data': phi_mean,
            'ref': ref,
            'units': 'rad',
            'dim': 'angle',
        },
        'dV': {
            'key': kdV,
            'data': dV,
            'ref': (knpts,),
            'units': 'm3',
            'dim': 'volume',
        },

        # debug
        'etendlen': {
            'key': ketl,
            'data': etendlen,
            'ref': refcam,
            'units': 'sr.m3',
            'dim': 'etend*len',
        },
    }

    if timing:
        t33 = dtm.datetime.now()
        dt22 += (t33 - t22).total_seconds()

    return (
        dout, dref,
        dt11, dt22,
        dt111, dt222, dt333,
        dt1111, dt2222, dt3333, dt4444,
    )


# ################################################
# ################################################
#           Debug plot
# ################################################


def _plot_debug(
    coll=None,
    key_cam=None,
    cbin0=None,
    cbin1=None,
    dx0=None,
    dx1=None,
    x0c=None,
    x1c=None,
    cos=None,
    angles=None,
    iok=None,
    p0=None,
    p1=None,
    x0if=None,
    x1if=None,
):

    out0, out1 = coll.get_optics_outline(key_cam, total=True)
    ck0f = np.array([cbin0, cbin0, np.full((cbin0.size,), np.nan)])
    ck1f = np.array([cbin1, cbin1, np.full((cbin1.size,), np.nan)])
    ck01 = np.r_[np.min(cbin1), np.max(cbin1), np.nan]
    ck10 = np.r_[np.min(cbin0), np.max(cbin0), np.nan]

    fig = plt.figure(figsize=(14, 8))
    if dx0 is None:
        ldata = [
            ('cos vs toroidal', cos),
            ('angles vs crystal', angles),
            ('iok', iok)
        ]

        for ii, v0 in enumerate(ldata):
            ax = fig.add_subplot(2, 2, ii + 1, aspect='equal')
            ax.set_title(v0[0], size=12, fontweight='bold')
            ax.set_xlabel('x0 (m)', size=12, fontweight='bold')
            ax.set_xlabel('x1 (m)', size=12, fontweight='bold')

            # grid
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

            # points
            im = ax.scatter(
                x0c,
                x1c,
                c=v0[1],
                s=4,
                edgecolors='None',
                marker='o',
            )
            plt.colorbar(im, ax=ax)

        # projected polygon
        ax = fig.add_subplot(2, 2, 4, aspect='equal', adjustable='datalim')
        ax.set_title('Projected polygon', size=12, fontweight='bold')
        ax.set_xlabel('x0 (m)', size=12, fontweight='bold')
        ax.set_xlabel('x1 (m)', size=12, fontweight='bold')

        ax.plot(
            np.r_[p0, p0[0]],
            np.r_[p1, p1[0]],
            c='k',
            ls='-',
            lw=1.,
        )
        ax.plot(
            x0if[iok],
            x1if[iok],
            '.g',
        )
        ax.plot(
            x0if[~iok],
            x1if[~iok],
            '.r',
        )

    else:

        dmargin = {
            'bottom': 0.08, 'top': 0.95,
            'left': 0.10, 'right': 0.95,
            'wspace': 0.50, 'hspace': 0.10,
        }

        gs = gridspec.GridSpec(ncols=20, nrows=2, **dmargin)
        ax0 = fig.add_subplot(gs[0, :-1], aspect='equal')
        ax1 = fig.add_subplot(
            gs[1, :-1],
            sharex=ax0,
            sharey=ax0,
        )
        ax2 = fig.add_subplot(gs[1, -1])

        ax0.set_ylabel('x1 (m)', size=12, fontweight='bold')

        ax1.set_xlabel('x0 (m)', size=12, fontweight='bold')
        ax1.set_ylabel('x1 (m)', size=12, fontweight='bold')

        # grid
        ax0.plot(np.r_[out0, out0[0]], np.r_[out1, out1[0]], '.-k')
        ax0.plot(
            ck0f.T.ravel(),
            np.tile(ck01, cbin0.size),
            '-k',
        )
        ax0.plot(
            np.tile(ck10, cbin1.size),
            ck1f.T.ravel(),
            '-k',
        )

        # pre-concatenate
        for i0, v0 in dx0.items():
            for i1, v1 in v0.items():
                if len(v1) > 0:
                    dx0[i0][i1] = np.concatenate([vv.ravel() for vv in v1])
                    dx1[i0][i1] = np.concatenate([vv.ravel() for vv in dx1[i0][i1]])

        # points
        for i0, v0 in dx0.items():
            for i1, v1 in v0.items():
                if len(v1) > 0:
                    ax0.plot(v1, dx1[i0][i1], '.')

        # concatenate
        x0_all = np.concatenate([
            np.concatenate([v1 for v1 in v0.values() if len(v1) > 0])
            for v0 in dx0.values()
        ])
        x1_all = np.concatenate([
            np.concatenate([v1 for v1 in v0.values() if len(v1) > 0])
            for v0 in dx1.values()
        ])

        # binning
        out = scpstats.binned_statistic_2d(
            x0_all,
            x1_all,
            None,
            statistic='count',
            bins=(cbin0, cbin1),
            expand_binnumbers=True,
        )

        # set binned
        binned = out.statistic

        # plot binning
        im = ax1.imshow(
            binned.T,
            origin='lower',
            interpolation='nearest',
            aspect='equal',
            extent=(cbin0[0], cbin0[-1], cbin1[0], cbin1[-1]),
        )

        plt.colorbar(im, ax=ax1, cax=ax2)
        ax1.set_xlim(cbin0[0], cbin0[-1])
        plt.show()

    import pdb
    pdb.set_trace()     # DB