# -*- coding: utf-8 -*-


# Built-in
import warnings


# Common
import numpy as np
import datastock as ds


# local
from . import _class8_plot_coverage_slice_broadband as _broadband
from . import _class8_plot_coverage_slice_spectro as _spectro


# ##########################################################
# ##########################################################
#              Main
# ##########################################################


def main(
    coll=None,
    key_diag=None,
    key_cam=None,
    indch=None,
    indref=None,
    # parameters
    res=None,
    margin_par=None,
    margin_perp=None,
    vect=None,
    segment=None,
    # mesh slice
    key_mesh=None,
    phi=None,
    Z=None,
    DR=None,
    DZ=None,
    Dphi=None,
    adjust_phi=None,
    # solid angle
    config=None,
    visibility=None,
    # solid angle
    n0=None,
    n1=None,
    # lamb
    res_lamb=None,
    # bool
    verb=None,
    plot=None,
    # plotting
    indplot=None,
    dax=None,
    plot_config=None,
    fs=None,
    dmargin=None,
    dvminmax=None,
    markersize=None,
):

    # ------------
    # check inputs
    # ------------

    (
        key_diag, key_cam, indch, indref,
        is2d, spectro, doptics,
        lop_pre, lop_post,
        res, margin_par, margin_perp,
        vect, segment, phi, Z, key_mesh,
        adjust_phi,
        visibility,
        verb, plot,
        indref, indplot,
        plot_config,
    ) = _check(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        indch=indch,
        indref=indref,
        # parameters
        res=res,
        margin_par=margin_par,
        margin_perp=margin_perp,
        vect=vect,
        segment=segment,
        # mesh slice
        key_mesh=key_mesh,
        phi=phi,
        Z=Z,
        DR=DR,
        DZ=DZ,
        Dphi=Dphi,
        adjust_phi=adjust_phi,
        # solid angle
        config=config,
        visibility=visibility,
        # bool
        verb=verb,
        plot=plot,
        # plotting
        indplot=indplot,
        plot_config=plot_config,
    )

    # -----------------
    # get plane pts
    # -----------------

    if vect is not None:
        (
            ptsx, ptsy, ptsz,
            los_ref, pt_ref, klos,
            e0, e1,
            x0, x1,
            dS,
        ) = _plane_from_LOS(
            # resource
            coll=coll,
            doptics=doptics,
            key_cam=key_cam,
            indref=indref,
            spectro=spectro,
            lop_post=lop_post,
            vect=vect,
            segment=segment,
            # plane params
            res=res,
            margin_par=margin_par,
            margin_perp=margin_perp,
            # options
            indch=indch,
        )

        indr, indz, indphi = None, None, None

    else:

        dpts = coll.get_sample_mesh_3d_slice(
            key=key_mesh,
            res=res[0],
            phi=phi,
            Z=Z,
            DR=DR,
            DZ=DZ,
            Dphi=Dphi,
            reshape_2d=True,
            adjust_phi=adjust_phi,
            plot=False,
            dax=None,
            color=None,
        )

        ptsx = dpts['pts_r']['data'] * np.cos(dpts['pts_phi']['data'])
        ptsy = dpts['pts_r']['data'] * np.sin(dpts['pts_phi']['data'])
        ptsz = dpts['pts_z']['data']

        indr = dpts['indr']['data']
        indz = dpts['indz']['data']
        indphi = dpts['indphi']['data']

        los_ref, pt_ref, klos = None, None, None
        e0, e1 = None, None
        x0, x1 = None, None
        dS = None

    # ---------------------------
    # get compute & plot function
    # ---------------------------

    if spectro is True:
        _compute = _spectro._compute
        if vect is None:
            _plot = _spectro._plot_from_mesh
        else:
            _plot = _spectro._plot_from_los

    else:
        _compute = _broadband._compute
        if vect is None:
            _plot = _broadband._plot_from_mesh
        else:
            _plot = _broadband._plot_from_los

    # ----------
    # compute
    # ----------

    dout = _compute(
        coll=coll,
        key_diag=key_diag,
        key_cam=key_cam,
        doptics=doptics,
        is2d=is2d,
        # pts
        ptsx=ptsx,
        ptsy=ptsy,
        ptsz=ptsz,
        # solid angle
        n0=n0,
        n1=n1,
        # res
        res_lamb=res_lamb,
        # config
        config=config,
        visibility=visibility,
    )

    # -------------
    # format output
    # -------------

    dout.update({
        'key_diag': key_diag,
        'key_cam': key_cam,
        'key_mesh': key_mesh,
        'res': res,
        'is2d': is2d,
        'spectro': spectro,
        'indch': indch,
        'indref': indref,
        'los_ref': los_ref,
        'pt_ref': pt_ref,
        'klos': klos,
        'e0': e0,
        'e1': e1,
        # pts
        'ptsx': ptsx,
        'ptsy': ptsy,
        'ptsz': ptsz,
        'indr': indr,
        'indz': indz,
        'indphi': indphi,
        'x0': x0,
        'x1': x1,
        'dS': dS,
        'vect': vect,
        'Z': Z,
        'phi': phi,
    })

    # -------
    # plot
    # -------

    if plot is True:
        _plot(
            coll=coll,
            key_diag=key_diag,
            key_cam=key_cam,
            # phi, Z
            phi=phi,
            Z=Z,
            vect=vect,
            # plane
            x0=x0,
            x1=x1,
            dS=dS,
            # pts
            ptsx=ptsx,
            ptsy=ptsy,
            ptsz=ptsz,
            # dout
            dout=dout,
            # indices
            indch=indch,
            indref=indref,
            los_ref=los_ref,
            pt_ref=pt_ref,
            klos=klos,
            # extra
            indplot=indplot,
            dax=dax,
            plot_config=plot_config,
            fs=fs,
            dmargin=dmargin,
            dvminmax=dvminmax,
            markersize=markersize,
        )

    return dout


# ###############################################################
# ###############################################################
#                   Check
# ###############################################################


def _check(
    coll=None,
    key_diag=None,
    key_cam=None,
    indch=None,
    indref=None,
    # parameters
    res=None,
    margin_par=None,
    margin_perp=None,
    vect=None,
    segment=None,
    # mesh slice
    key_mesh=None,
    phi=None,
    Z=None,
    DR=None,
    DZ=None,
    Dphi=None,
    adjust_phi=None,
    # solid angle
    config=None,
    visibility=None,
    # bool
    verb=None,
    plot=None,
    # plotting
    indplot=None,
    plot_config=None,
):

    # ----------
    # keys
    # ----------

    # key, key_cam
    key_diag, key_cam = coll.get_diagnostic_cam(
        key=key_diag,
        key_cam=key_cam,
    )

    # properties
    spectro = coll.dobj['diagnostic'][key_diag]['spectro']
    is2d = coll.dobj['diagnostic'][key_diag]['is2d']

    # safety check vs spectro
    if spectro:

        if len(key_cam) > 1:
            msg = (
                "Spectrometer: please select a single key_cam!\n"
                f"\tkey_cam: {key_cam}\n"
            )
            raise Exception(msg)

        if not is2d:
            msg = "Only implemented for 2d spectro"
            raise Exception(msg)
        kcam = key_cam[0]

    # doptics
    if spectro:
        doptics = coll.dobj['diagnostic'][key_diag]['doptics'][kcam]
    else:
        doptics = coll.dobj['diagnostic'][key_diag]['doptics']

    # -----------------
    # loptics
    # -----------------

    if spectro:
        optics, cls_optics = coll.get_optics_cls(doptics[kcam]['optics'])
        ispectro = cls_optics.index('crystal')
        lop_pre = doptics[kcam]['optics'][:ispectro]
        lop_post = doptics[kcam]['optics'][ispectro+1:]

    else:
        lop_pre = None
        lop_post = None

    # -----------------
    # indch
    # -----------------

    if indch is not None:
        indch = _check_index(
            coll=coll,
            key_cam=key_cam,
            ind=indch,
            ind_name='indch',
        )

    # -----------------
    # indref
    # -----------------

    if indref is not None:
        indref = _check_index(
            coll=coll,
            key_cam=key_cam,
            ind=indref,
            ind_name='indref',
        )
        if indch is not None:
            indref = indch

    # ----------
    # res
    # ----------

    if res is None:
        res = 0.001

    if isinstance(res, (int, float)):
        res = np.r_[res, res]

    res = ds._generic_check._check_flat1darray(
        res, 'res',
        dtype=float,
        size=2,
        sign='>0.',
    )

    # -----------
    # margin_par
    # -----------

    margin_par = float(ds._generic_check._check_var(
        margin_par, 'margin_par',
        types=(int, float),
        default=0.5,
    ))

    # -----------
    # margin_perp
    # -----------

    if margin_perp is None:
        margin_perp = 0.02

    if isinstance(margin_perp, (float, int)):
        margin_perp = [margin_perp, margin_perp]

    margin_perp = ds._generic_check._check_flat1darray(
        margin_perp, 'margin_perp',
        dtype=float,
        size=2,
        sign='>=0',
    )

    # -----------
    # vect
    # -----------

    vect = ds._generic_check._check_var(
        vect, 'vect',
        types=str,
        default='nin',
        allowed=['nin', 'e0', 'e1'],
    )

    # -----------
    # vect vs phi, Z
    # -----------

    lc = [
        Z is not None,
        phi is not None,
    ]
    if np.sum(lc) > 1:
        msg = (
            "Provide Z xor phi xor None\n"
            f"\t- phi = {phi}\n"
            f"\t- Z = {Z}\n"
        )
        raise Exception(msg)

    # -----------
    # any(phi, Z)
    # -----------

    if any(lc):

        # key_mesh
        wm = coll._which_mesh
        lok = list(coll.dobj.get(wm, {}).keys())
        key_mesh = ds._generic_check._check_var(
            key_mesh, 'key_mesh',
            types=str,
            allowed=lok,
        )

    else:
        key_mesh = None

    # -----------
    # phi
    # -----------

    if phi is not None:
        phi = float(ds._generic_check._check_var(
            phi, 'phi',
            types=(float, int),
        ))
        vect = None

    # -----------
    # Z
    # -----------

    if Z is not None:
        Z = float(ds._generic_check._check_var(
            Z, 'Z',
            types=(float, int),
        ))
        vect = None

    if vect is not None:
        if len(key_cam) != 1:
            msg = (
                "If a slice is not provided, and len(key_cam) != 1\n"
                f"\t- Z: {Z}\n"
                f"\t- phi: {phi}\n"
                f"\t- vect: {vect}\n"
                f"\t- key_diag: {key_diag}\n"
                f"\t- key_cam: {key_cam}\n"
                f"=> The first camera is used as reference for (nin, e0, e1)"
            )
            warnings.warn(msg)

    # -----------
    # adjust_phi
    # -----------

    adjust_phi = ds._generic_check._check_var(
        adjust_phi, 'adjust_phi',
        types=bool,
        default=True,
    )

    # -----------
    # segment
    # -----------

    defseg = -1 if spectro else 0
    segment = int(ds._generic_check._check_var(
        segment, 'segment',
        types=(int, float),
        default=defseg,
    ))

    # -----------
    # visibility
    # -----------

    visibility = ds._generic_check._check_var(
        visibility, 'visibility',
        types=bool,
        default=vect is None,
    )

    # -----------
    # verb
    # -----------

    verb = ds._generic_check._check_var(
        verb, 'verb',
        types=bool,
        default=True,
    )

    # -----------
    # plot
    # -----------

    plot = ds._generic_check._check_var(
        plot, 'plot',
        types=bool,
        default=True,
    )

    # ------------------
    # get indref for los
    # ------------------

    if indref is None and vect is not None:
        if spectro or is2d or indch is None:
            etend = coll.ddata[doptics[key_cam[0]]['etendue']]['data']
            indref = np.nanargmax(etend)

            if is2d:
                n0, n1 = etend.shape
                indref = tuple(np.r_[indref // n1, indref % n1].astype(int))
            else:
                indref = (indref,)

        else:
            indref = indch

    # -----------
    # indplot
    # -----------

    if indplot is None:
        indplot = indref

    # -----------
    # plot_config
    # -----------

    if plot is True and plot_config is None:
        plot_config = config

    return (
        key_diag, key_cam, indch, indref,
        is2d, spectro, doptics,
        lop_pre, lop_post,
        res, margin_par, margin_perp,
        vect, segment, phi, Z, key_mesh,
        adjust_phi,
        visibility,
        verb, plot,
        indref, indplot,
        plot_config,
    )


def _check_index(coll=None, key_cam=None, ind=None, ind_name=None):

    # safety
    if len(key_cam) > 1:
        msg = (
            "indch cannot be provided for multiple key_cam!\n"
            f"\t- {ind_name}: {ind}\n"
            f"\t- key_cam: {key_cam}\n"
        )
        raise Exception(msg)

    if isinstance(ind, (float, int)):
        ind = (int(ind),)

    shape_cam = coll.dobj['camera'][key_cam[0]]['dgeom']['shape']
    ind = ds._generic_check._check_flat1darray(
        ind, ind_name,
        dtype=int,
        size=len(shape_cam),
    )

    return tuple([
        ind[ii] % shape_cam[ii]
        for ii in range(len(ind))
    ])


# ###############################################################
# ###############################################################
#               Plane vs LOS
# ###############################################################


def _plane_from_LOS(
    # resource
    coll=None,
    doptics=None,
    key_cam=None,
    indref=None,
    spectro=None,
    lop_post=None,
    segment=None,
    vect=None,
    # plane params
    res=None,
    margin_par=None,
    margin_perp=None,
    # options
    indch=None,
):

    # ----------
    # los_ref
    # ----------

    kcam = key_cam[0]
    klos = doptics[kcam]['los']
    parallel = coll.dobj['camera'][kcam]['dgeom']['parallel']

    # start_ref
    ptsx, ptsy, ptsz = coll.get_rays_pts(
        key=klos,
        segment=segment,
    )
    ipts = (0,) + indref
    pt_ref = np.r_[ptsx[ipts], ptsy[ipts], ptsz[ipts]]

    # los_ref
    vx, vy, vz = coll.get_rays_vect(
        key=klos,
        norm=True,
        segment=segment,
    )
    los_ref = np.r_[vx[indref], vy[indref], vz[indref]]

    # -----------------
    # furthest aperture
    # -----------------

    if spectro:
        if len(lop_post) > 0:
            poly = lop_post[-1]
        else:
            kmax = 0.
    else:
        if len(doptics[kcam]['optics']) > 0:
            poly = doptics[kcam]['optics'][-1]
        else:
            kmax = 0.

    # poly => sca
    if poly is not None:
        px, py, pz = coll.get_optics_poly(poly)
        poly = np.array([px, py, pz])
        kmax = np.max(np.sum(
            (poly - pt_ref[:, None])*los_ref[:, None],
            axis=0,
        ))

    # get length along los
    klos = kmax + margin_par

    # get pt plane
    pt_plane = pt_ref + klos * los_ref

    # -------------------------------------
    # create plane perpendicular to los_ref
    # -------------------------------------

    if parallel:
        # e0_cam = coll.dobj['camera'][key_cam]['dgeom']['e0']
        e1_cam = coll.dobj['camera'][kcam]['dgeom']['e1']
    else:
        # ke0x, ke0y, ke0z = coll.dobj['camera'][key_cam]['dgeom']['e0']
        ke1x, ke1y, ke1z = coll.dobj['camera'][kcam]['dgeom']['e1']
        e1_cam = np.r_[
            coll.ddata[ke1x]['data'][indref],
            coll.ddata[ke1y]['data'][indref],
            coll.ddata[ke1z]['data'][indref],
        ]

    e0 = np.cross(e1_cam, los_ref)
    e0 = e0 / np.linalg.norm(e0)

    e1 = np.cross(los_ref, e0)
    e1 = e1 / np.linalg.norm(e1)

    # -------------------------------------
    # create plane perpendicular to los_ref
    # -------------------------------------

    # get limits of plane
    if indch is None:
        # kk = vx[-1, ...]

        x0, x1, iok = _get_intersect_los_plane(
            cent=pt_plane,
            nin=los_ref,
            e0=e0,
            e1=e1,
            ptx=ptsx[0, ...],
            pty=ptsy[0, ...],
            ptz=ptsz[0, ...],
            vx=vx,
            vy=vy,
            vz=vz,
        )

    else:
        x0 = np.r_[0]
        x1 = np.r_[0]

    # dx0, dx1
    dx0 = [
        np.nanmin(x0) - margin_perp[0],
        np.nanmax(x0) + margin_perp[0],
    ]
    dx1 = [
        np.nanmin(x1) - margin_perp[1],
        np.nanmax(x1) + margin_perp[1],
    ]

    # -------------
    # vect
    # -------------

    if vect == 'nin':
        pass

    elif vect == 'e0':
        e0 = los_ref
        dx0 = [kmax + margin_perp[0], klos]
        pt_plane = pt_ref

    else:
        e1 = los_ref
        dx1 = [kmax + margin_perp[1], klos]
        pt_plane = pt_ref

    # ---------------
    # create 2d grid
    # ---------------

    nx0 = int(np.ceil((dx0[1] - dx0[0]) / res[0])) + 2
    nx1 = int(np.ceil((dx1[1] - dx1[0]) / res[1])) + 2

    x0 = np.linspace(dx0[0], dx0[1], nx0)
    x1 = np.linspace(dx1[0], dx1[1], nx1)

    dS = (x0[1] - x0[0]) * (x1[1] - x1[0])

    x0f = np.repeat(x0[:, None], nx1, axis=1)
    x1f = np.repeat(x1[None, :], nx0, axis=0)

    # derive 3d pts
    ptsx = pt_plane[0] + x0f*e0[0] + x1f*e1[0]
    ptsy = pt_plane[1] + x0f*e0[1] + x1f*e1[1]
    ptsz = pt_plane[2] + x0f*e0[2] + x1f*e1[2]

    return (
        ptsx, ptsy, ptsz,
        los_ref, pt_ref, klos,
        e0, e1,
        x0, x1,
        dS,
    )


# ###############################################################
# ###############################################################
#              get intersection los vs plane
# ###############################################################


def _get_intersect_los_plane(
    cent=None,
    nin=None,
    e0=None,
    e1=None,
    ptx=None,
    pty=None,
    ptz=None,
    vx=None,
    vy=None,
    vz=None,
):

    # ---------------
    # prepare output

    shape = vx.shape
    kk = np.full(shape, np.nan)

    # ------------
    # compute kk

    sca0 = vx * nin[0] + vy * nin[1] + vz * nin[2]
    sca1 = (
        (cent[0] - ptx) * nin[0]
        + (cent[1] - pty) * nin[1]
        + (cent[2] - ptz) * nin[2]
    )

    iok = np.abs(sca1) > 1e-6
    kk[iok] = sca1[iok] / sca0[iok]

    # --------------
    # 3d coordinates

    xx = ptx + kk * vx
    yy = pty + kk * vy
    zz = ptz + kk * vz

    # -----------------
    # local coordinates

    dx = xx - cent[0]
    dy = yy - cent[1]
    dz = zz - cent[2]

    x0 = dx * e0[0] + dy * e0[1] + dz * e0[2]
    x1 = dx * e1[0] + dy * e1[1] + dz * e1[2]

    return x0, x1, iok
