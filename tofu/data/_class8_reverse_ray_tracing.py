

import numpy as np
import scipy.interpolate as scpinterp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


from . import _class8_plot
from . import _generic_check
from . import _class8_reverse_ray_tracing_broadband as _broadband
from . import _class8_reverse_ray_tracing_spectro as _spectro


# ################################################
# ################################################
#               main
# ################################################


def main(
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
    n0=None,
    n1=None,
    min_threshold=None,
    # optional lamb
    lamb0=None,
    res_lamb=None,
    rocking_curve=None,
    # options
    append=None,
    # plotting
    plot=None,
    plot_pixels=None,
    plot_config=None,
    plot_rays=None,
    vmin=None,
    vmax=None,
    aspect3d=None,
    elements=None,
    colorbar=None,
    # options
    dax=None,
    fs=None,
    dmargin=None,
    wintit=None,
):

    # -------------
    # check inputs

    (
        key, key_cam, key_mesh, isspectro,
        dsamp, res_phi,
        shape, iok,
        ptsx, ptsy, ptsz, phi,
        lamb0, rocking_curve,
        append, plot, plot_pixels, colorbar,
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
        rocking_curve=rocking_curve,
        # options
        append=append,
        plot=plot,
        plot_pixels=plot_pixels,
        colorbar=colorbar,
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
        isspectro=isspectro,
    )
    shape_cam = coll.dobj['camera'][key_cam]['dgeom']['shape']

    # -------------------------------------
    # prepare lambda, angles, rocking_curve

    if isspectro:
        (
            nlamb,
            lamb,
            dlamb,
            pow_interp,
            ang_rel,
            dang,
            bragg,
        ) = _prepare_lamb(
            coll=coll,
            key_diag=key,
            key_cam=key_cam,
            kspectro=kspectro,
            lamb=lamb0,
            res_lamb=res_lamb,
            rocking_curve=rocking_curve,
            verb=False,
        )
    else:
        nlamb, lamb, dlamb, pow_interp, ang_rel, dang, bragg = (None,) * 7

    # -------------------
    # wavelength specific

    if lamb is not None:

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
        if isspectro:
            func = _spectro._loop0
        else:
            func = _broadband._loop0
    else:
        if isspectro:
            func = _spectro._loop1
        else:
            func = _broadband._loop1

    # -------------
    # compute

    dout = func(**locals())
    dout['rocking_curve'] = rocking_curve

    # -------
    # reshape

    lk_reshape = ['sang', 'sang0']
    if all([kk in dout.keys() for kk in lk_reshape]) and len(shape) > 1:
        shape_new = tuple(np.r_[shape_cam, shape])
        for k0 in lk_reshape:
            dout[k0]['data'] = dout[k0]['data'].reshape(shape_new)

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
            aspect3d=aspect3d,
            elements=elements,
            colorbar=colorbar,
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
    rocking_curve=None,
    # bool
    append=None,
    plot=None,
    plot_pixels=None,
    colorbar=None,
):

    # --------
    # key

    # key
    lok_spectro = [
        k0 for k0, v0 in coll._dobj.get('diagnostic', {}).items()
        if v0['spectro'] is True
    ]
    lok_broadband = [
        k0 for k0, v0 in coll._dobj.get('diagnostic', {}).items()
        if v0['spectro'] is False
    ]

    key = ds._generic_check._check_var(
        key, 'key',
        types=str,
        allowed=lok_spectro + lok_broadband,
    )
    isspectro = key in lok_spectro

    # key_cam
    lok = list(coll.dobj['diagnostic'][key]['doptics'].keys())
    key_cam = ds._generic_check._check_var(
        key_cam, 'key_cam',
        types=str,
        allowed=lok,
    )

    wcam = coll._which_cam
    if coll.dobj[wcam][key_cam]['dgeom']['parallel'] is not True:
        msg = (
            "Reverse ray-tracing not implemented for:\n"
            "\t- non-parallel broadband camera {key_cam}"
        )
        raise NotImplementedError(msg)

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

    if isspectro and lamb0 is not None:
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
                    f"\t- ptsx.shape = {ptsx.shape}\n"
                    f"\t- lamb0.shape = {lamb0.shape}\n"
                )
                raise Exception(msg)

    # --------------
    # rocking_curve

    rocking_curve = ds._generic_check._check_var(
        rocking_curve, 'rocking_curve',
        default=isspectro,
        types=bool,
    )

    if lamb0 is None or not isspectro:
        rocking_curve = False

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
        default=lc[1] and ptsx.size < 100,  # and plot is True,
    )

    # colorbar
    colorbar = ds._generic_check._check_var(
        colorbar, 'colorbar',
        types=bool,
        default=False,
    )

    return (
        key, key_cam, key_mesh, isspectro,
        dsamp, res_phi,
        shape, iok,
        ptsx, ptsy, ptsz, phi,
        lamb0, rocking_curve,
        append, plot, plot_pixels, colorbar,
    )


# ################################################
# ################################################
#               prepare optics
# ################################################


def _prepare_optics(
    coll=None,
    key=None,
    key_cam=None,
    isspectro=None,
):

    # -----------------------
    # get optics and classes
    # -----------------------

    doptics = coll.dobj['diagnostic'][key]['doptics']
    lop = doptics[key_cam]['optics']
    lop, lcls = coll.get_optics_cls(lop)

    # ------------------
    # safety check
    # ------------------

    if isspectro:
        cls_spectro = 'crystal'
        kspectro = lop[lcls.index(cls_spectro)]
        ispectro = lop.index(kspectro)
        if len(lop[:ispectro]) > 1:
            msg = "Not yet implemented optics between crystal and camera!"
            raise NotImplementedError(msg)

    # ------------------
    # list of polygons post crystal
    # ------------------

    if isspectro:
        lpoly_post = [
            coll.get_optics_poly(
                key=k0,
                add_points=None,
                return_outline=False,
            )
            for k0 in lop[ispectro+1:]
        ]

        # get initial polygon of crystal
        p0x, p0y, p0z = coll.get_optics_poly(key=kspectro, add_points=None)

    else:
        lpoly_post = [
            coll.get_optics_poly(
                key=k0,
                add_points=None,
                return_outline=False,
            )
            for k0 in lop
        ]

        # get initial polygon of crystal
        p0x, p0y, p0z = coll.get_optics_poly(
            key=key_cam,
            total=True,
            add_points=None,
        )

    # ------------------
    # unit vectors of crystal
    # ------------------

    if isspectro:
        nin = coll.dobj[cls_spectro][kspectro]['dgeom']['nin']
        e0 = coll.dobj[cls_spectro][kspectro]['dgeom']['e0']
        e1 = coll.dobj[cls_spectro][kspectro]['dgeom']['e1']

    else:
        wcam = coll._which_cam
        nin = coll.dobj[wcam][key_cam]['dgeom']['nin']
        e0 = coll.dobj[wcam][key_cam]['dgeom']['e0']
        e1 = coll.dobj[wcam][key_cam]['dgeom']['e1']

    # ------------------
    # get functions for crystal and camera
    # ------------------

    ptsvect_cam = coll.get_optics_reflect_ptsvect(key=key_cam, fast=True)
    if isspectro:
        ptsvect_plane = coll.get_optics_reflect_ptsvect(
            key=kspectro,
            asplane=True,
        )
        ptsvect_spectro = coll.get_optics_reflect_ptsvect(
            key=kspectro,
            isnorm=True,
        )
        coords_x01toxyz_plane = coll.get_optics_x01toxyz(
            key=kspectro,
            asplane=True,
        )
    else:
        ptsvect_plane, ptsvect_spectro, coords_x01toxyz_plane = (None,)*3

    # ------------------
    # Get centers of crystal and camera to estimate distance
    # ------------------

    cent_cam = coll.dobj['camera'][key_cam]['dgeom']['cent']
    pix_size = np.sqrt(coll.dobj['camera'][key_cam]['dgeom']['pix_area'])
    if isspectro:
        cent_spectro = coll.dobj[cls_spectro][kspectro]['dgeom']['cent']
        dist_to_cam = np.linalg.norm(cent_spectro - cent_cam)
    else:
        cent_spectro, dist_to_cam = None, None

    # ------------------
    # prepare camera bin edges
    # ------------------

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
    rocking_curve=None,
    rocking_curve_step_width=None,
    verb=None,
):

    # ------------------
    # get lamb
    # ------------------

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

        bragg = coll.get_crystal_bragglamb(
            key=kspectro,
            lamb=lamb,
            rocking_curve=False,
        )[0]

    elif lamb is not None:

        lamb = ds._generic_check._check_flat1darray(
            lamb, 'lamb',
            dtype=float,
            sign='>0',
            unique=True,
        )
        nlamb = lamb.size
        dlamb = None
        bragg = coll.get_crystal_bragglamb(
            key=kspectro,
            lamb=lamb,
            rocking_curve=False,
        )[0]

    # ----------------
    # get power ratio
    # ----------------

    # angle relative
    cls_spectro = 'crystal'
    kang_rel = coll.dobj[cls_spectro][kspectro]['dmat']['drock']['angle_rel']
    ang_rel = coll.ddata[kang_rel]['data']

    # power ratio
    if rocking_curve is False:
        if rocking_curve_step_width is None:
            rocking_curve_step_width = np.mean(np.diff(ang_rel))

        ang_rel = 0.5 * rocking_curve_step_width * np.r_[-1, 1]
        pow_ratio = np.r_[1, 1]

    else:
        kpow = coll.dobj[cls_spectro][kspectro]['dmat']['drock']['power_ratio']
        pow_ratio = coll.ddata[kpow]['data']

    # --------------
    # interpolation
    # ---------------

    pow_interp = scpinterp.interp1d(
        ang_rel,
        pow_ratio,
        kind='linear',
        bounds_error=False,
        fill_value=0.,
    )

    dang = np.mean(np.diff(ang_rel))

    # --------------------------------------
    # overall bragg angle with rocking curve
    # --------------------------------------

    if res_lamb is None and lamb is None:
        nlamb, lamb, dlamb, bragg = None, None, None, None

    else:

        # ------------
        # safety check

        FW = coll.dobj[cls_spectro][kspectro]['dmat']['drock']['FW']
        dd0 = bragg[0] + np.r_[0, ang_rel[-1] - ang_rel[0]]
        dd1 = bragg[0] + np.r_[0, FW]
        dd2 = bragg[0] + np.r_[0, ang_rel[1] - ang_rel[0]]

        dlamb_max = np.diff(
            coll.get_crystal_bragglamb(key=kspectro, bragg=dd0)[1]
        )
        dlamb_mh = np.diff(
            coll.get_crystal_bragglamb(key=kspectro, bragg=dd1)[1]
        )
        dlamb_res = np.diff(
            coll.get_crystal_bragglamb(key=kspectro, bragg=dd2)[1]
        )

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
        pow_interp,
        ang_rel,
        dang,
        bragg,
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
    aspect3d=None,
    elements=None,
    colorbar=None,
    # options
    dax=None,
    fs=None,
    dmargin=None,
    wintit=None,
):

    # --------------
    # check inputs

    aspect3d = ds._generic_check._check_var(
        aspect3d, 'aspect3d',
        default='equal',
        allowed=['auto', 'equal'],
    )

    elements = ds._generic_check._check_var(
        elements, 'elements',
        default='o',
        types=str,
    )

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
        ref=coll.dobj['camera'][key_cam]['dgeom']['ref'],
        **dout['ncounts'],
    )

    kd = 'sang_lamb'
    if vmin is None:
        vmin = 0
    if vmax is None and dout.get(kd) is not None:
        vmax = np.nanmax(dout[kd]['data'])

    # --------------
    # prepare

    dplot = coll.get_diagnostic_dplot(
        key=key,
        key_cam=key_cam,
        optics=None,
        elements=elements,
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
                dout['ptsr']['data'],
                dout['ptsz']['data'],
                ls='None',
                lw=1.,
                marker='.',
            )

        kax = 'hor'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            ax.plot(
                dout['ptsx']['data'],
                dout['ptsy']['data'],
                ls='None',
                lw=1.,
                marker='.',
            )

        kax = 'cam_all'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']

            p0 = np.concatenate([pp.ravel() for pp in dout['lp0']['data']])
            p1 = np.concatenate([pp.ravel() for pp in dout['lp1']['data']])
            ax.plot(
                p0,
                p1,
                ls='None',
                marker='.',
            )

    else:

        # --------------
        # ray-tracing

        for ii in range(len(dout['lp0']['data'])):

            p0 = dout['lp0']['data'][ii]
            p1 = dout['lp1']['data'][ii]

            # --------------
            # camera image

            kax = 'cam_all'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                if dout['lamb']['data'] is None:
                    ax.plot(
                        p0,
                        p1,
                        ls='None',
                        marker='.',
                        # c=dout['lcolor'][ii],
                    )
                elif dout['rocking_curve'] is False:
                    ax.plot(
                        p0[dout['lindin'][ii]],
                        p1[dout['lindin'][ii]],
                        ls='None',
                        marker='.',
                        # c=dout['lcolor'][ii],
                    )
                    ax.plot(
                        p0[~dout['lindin'][ii]],
                        p1[~dout['lindin'][ii]],
                        ls='None',
                        marker='.',
                        c=(0.8, 0.8, 0.8),
                    )
                else:
                    ax.scatter(
                        p0,
                        p1,
                        # c=dout['lcolor'][ii],
                        s=8,
                        marker='.',
                    )

            # --------------
            # rays

            ind = np.arange(3, dout['lpx']['data'][ii].size, 3)
            px = np.insert(dout['lpx']['data'][ii].T.ravel(), ind, np.nan)
            py = np.insert(dout['lpy']['data'][ii].T.ravel(), ind, np.nan)
            pz = np.insert(dout['lpz']['data'][ii].T.ravel(), ind, np.nan)

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
    if dax.get(kax) is not None and dout.get(kd) is not None:
        ax = dax[kax]['handle']

        extent = (cbin0[0], cbin0[-1], cbin1[0], cbin1[-1])
        im = ax.imshow(
            dout[kd]['data'].T,
            extent=extent,
            interpolation='nearest',
            origin='lower',
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(kd, size=12, fontweight='bold')
        if colorbar is True:
            cb = plt.colorbar(im, ax=ax, cax=cax)
            cb.set_label(dout[kd]['units'], size=12, weight='bold')

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

    # --------------
    # aspect3d

    if aspect3d == 'equal' and dax.get('3d') is not None:
        ds.set_aspect3d(ax=dax['3d']['handle'])

    # --------------
    # clean up

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
            'left': 0.05, 'right': 0.94,
            'wspace': 0.20, 'hspace': 0.25,
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
