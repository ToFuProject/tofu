# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import Polygon as plg
import datastock as ds


# specific
from . import _generic_check
from . import _generic_plot
from . import _class8_plot as _plot
from ..geom import _core
from . import _class8_plot_vos_spectro as _plot_vos_spectro


# ###############################################################
# ###############################################################
#                           plot main
# ###############################################################


def _plot_diagnostic_vos(
    coll=None,
    key=None,
    key_cam=None,
    optics=None,
    elements=None,
    proj=None,
    los_res=None,
    indch=None,
    indlamb=None,
    # data plot
    dvos=None,
    units=None,
    cmap=None,
    vmin=None,
    vmax=None,
    vmin_tot=None,
    vmax_tot=None,
    vmin_cam=None,
    vmax_cam=None,
    dvminmax=None,
    alpha=None,
    # plot vos polygons
    plot_pcross=None,
    plot_phor=None,
    # colorbar
    plot_colorbar=None,
    # config
    plot_config=None,
    # figure
    dax=None,
    dmargin=None,
    fs=None,
    wintit=None,
    # interactivity
    color_dict=None,
):

    # ------------
    # check inputs

    (
        key,
        key_cam,
        is2d,
        proj,
        ddata,
        dref,
        static,
        daxis,
        _,
        _,
        _,
        alpha,
        units,
        los_res,
        color_dict,
        _,
        _,
        _,
        plot_pcross,
        plot_phor,
        ylab,
        plot_colorbar,
        _,
    ) = _plot._plot_diagnostic_check(
        coll=coll,
        key=key,
        key_cam=key_cam,
        # parameters
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        # plot vos polygons
        plot_pcross=plot_pcross,
        plot_phor=plot_phor,
        # figure
        plot_colorbar=plot_colorbar,
        proj=proj,
        units=units,
        los_res=los_res,
        # interactivity
        color_dict=color_dict,
    )

    # single camera + get dvos
    key_cam = key_cam[:1]
    key, dvos, isstore = coll.check_diagnostic_dvos(
        key,
        key_cam=key_cam,
        dvos=dvos,
    )
    doptics = coll.dobj['diagnostic'][key]['doptics'][key_cam[0]]
    shape_cam = coll.dobj['camera'][key_cam[0]]['dgeom']['shape']

    # indch
    if indch is None:
        if is2d:
            indch = [int(ss/2) for ss in shape_cam]
        else:
            indch = int(shape_cam[0]/2)

    # ------------
    # prepare data

    dplot = coll.get_diagnostic_dplot(
        key=key,
        key_cam=key_cam,
        optics=optics,
        elements=elements,
    )

    # ---------------------
    # prepare los and ddata

    if is2d:
        out0, out1 = coll.get_optics_outline(
            key=key_cam[0],
            add_points=False,
            total=True,
        )

        dgeom = coll.dobj['camera'][key_cam[0]]['dgeom']
        k0, k1 = dgeom['cents']
        x0 = coll.ddata[k0]['data']
        x1 = coll.ddata[k1]['data']
        if x0.size == 1:
            dx0 = coll.ddata[dgeom['outline'][0]]['data']
            dx0 = dx0.max() - dx0.min()
        else:
            dx0 = x0[1] - x0[0]
        if x1.size == 1:
            dx1 = coll.ddata[dgeom['outline'][1]]['data']
            dx1 = dx1.max() - dx1.min()
        else:
            dx1 = x1[1] - x1[0]
        extent_cam = (
            x0[0] - 0.5*dx0,
            x0[-1] + 0.5*dx0,
            x1[0] - 0.5*dx1,
            x1[-1] + 0.5*dx1,
        )
    else:
        x0, x1, extent_cam = None, None, None

    # ---------------------
    # prepare los and vos

    (
        los_x, los_y, los_z, los_r,
        los_xi, los_yi, los_zi, los_ri,
    ) = _prepare_los(
        coll=coll,
        doptics=doptics,
        los_res=los_res,
        is2d=is2d,
        indch=indch,
    )

    (
        pc0, pc1, ph0, ph1, pc0i, pc1i, ph0i, ph1i
    ) = _prepare_vos(
        coll=coll,
        doptics=doptics,
        los_res=los_res,
        is2d=is2d,
        indch=indch,
    )

    # plot vos polygons ?
    if plot_pcross is False:
        pc0, pc1, pc0i, pc1i = None, None, None, None
    if plot_phor is False:
        ph0, ph1, ph0i, ph1i = None, None, None, None

    # mesh envelop
    dout = coll.get_mesh_outline(key=dvos[key_cam[0]]['keym'])
    p0 = dout['x0']['data']
    p1 = dout['x1']['data']

    # etendue and length
    etendue, length = _get_etendue_length(
        coll=coll,
        doptics=doptics,
        poly=np.array([p0, p1]),
    )

    # spectro => call routine
    spectro = coll.dobj['diagnostic'][key]['spectro']
    if spectro:
        return _plot_vos_spectro._plot(**locals())

    # dsamp from mesh
    sang_tot, sang_integ, sang, extent, sang_units = _prepare_sang(
        coll=coll,
        dvos=dvos[key_cam[0]],
        key_cam=key_cam,
        indch=indch,
        spectro=spectro,
        is2d=is2d,
    )

    # vmin, vmax
    if vmin is None:
        vmin = 0.
    if vmax is None:
        vmax = np.nanmax(sang)
    if vmin_tot is None:
        vmin_tot = 0.
    if vmax_tot is None:
        vmax_tot = np.nanmax(sang_tot)
    if vmin_cam is None:
        vmin_cam = 0.
    if vmax_cam is None:
        vmax_cam = np.nanmax(sang_integ)

    # -----------------
    # prepare figure

    if dax is None:

        dax = _get_dax(
            proj=proj,
            dmargin=dmargin,
            fs=fs,
            wintit=wintit,
            tit=key,
            is2d=is2d,
            key_cam=key_cam,
            indch=indch,
        )

    dax = _generic_check._check_dax(dax=dax, main=proj[0])

    # -----------------
    # plot diag elements

    for k0, v0 in dplot.items():

        for k1, v1 in v0.items():

            # cross
            kax = 'cross'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                if k1.startswith('v-'):
                    ax.quiver(
                        v1['r'],
                        v1['z'],
                        v1['ur'],
                        v1['uz'],
                        **v1.get('props', {}),
                    )

                else:
                    ax.plot(
                        v1['r'],
                        v1['z'],
                        **v1.get('props', {}),
                    )

            # hor
            kax = 'hor'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                if k1.startswith('v-'):
                    ax.quiver(
                        v1['x'],
                        v1['y'],
                        v1['ux'],
                        v1['uy'],
                        **v1.get('props', {}),
                    )

                else:
                    ax.plot(
                        v1['x'],
                        v1['y'],
                        **v1.get('props', {}),
                    )

    # ------------------
    # plot sang

    # crosstot
    kax = 'crosstot'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        im = ax.imshow(
            sang_tot.T,
            extent=extent,
            origin='lower',
            aspect='equal',
            interpolation='nearest',
            vmin=vmin_tot,
            vmax=vmax_tot,
        )

        ax.plot(
            np.r_[p0, p0[0]],
            np.r_[p1, p1[0]],
            c='k',
            lw=1.,
            ls='-',
        )

        if plot_colorbar is True:
            plt.colorbar(im, ax=ax, label=sang_units)

    # cross
    kax = 'cross'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        im = ax.imshow(
            sang.T,
            extent=extent,
            origin='lower',
            aspect='equal',
            interpolation='nearest',
            vmin=vmin,
            vmax=vmax,
        )

        if plot_colorbar is True:
            plt.colorbar(im, ax=ax, label=sang_units)

    # ------------------
    # plot los / vos

    # crosstot
    kax = 'crosstot'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _add_camera_los_cross(
            ax=ax,
            is2d=is2d,
            los_r=los_r,
            los_z=los_z,
            pc0=pc0,
            pc1=pc1,
            alpha=alpha,
            color_dict=color_dict,
        )

    # cross
    kax = 'cross'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _add_camera_los_cross(
            ax=ax,
            is2d=is2d,
            los_r=los_ri,
            los_z=los_zi,
            pc0=pc0i,
            pc1=pc1i,
            alpha=alpha,
            color_dict=color_dict,
        )

    # hor
    kax = 'hor'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _add_camera_los_hor(
            ax=ax,
            is2d=is2d,
            los_x=los_x,
            los_y=los_y,
            ph0=ph0,
            ph1=ph1,
            alpha=alpha,
            color_dict=color_dict,
        )

    # camera
    kax = key_cam[0]
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _add_camera_data(
            coll=coll,
            ax=ax,
            sang_integ=sang_integ,
            etendue=etendue,
            length=length,
            indch=indch,
            # vmin, vmax
            vmin_cam=vmin_cam,
            vmax_cam=vmax_cam,
            # 2d only
            is2d=is2d,
            ax_etend=dax.get(f'{key_cam[0]}_etend', {}).get('handle'),
            ax_diff=dax.get(f'{key_cam[0]}_diff', {}).get('handle'),
            x0=x0,
            x1=x1,
            extent_cam=extent_cam,
        )

    # -------
    # config

    if plot_config.__class__.__name__ == 'Config':

        kax = 'cross'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj=kax, dLeg=False)

        kax = 'crosstot'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj='cross', dLeg=False)

        kax = 'hor'
        if dax.get(kax) is not None:
            ax = dax[kax]['handle']
            plot_config.plot(lax=ax, proj=kax, dLeg=False)

    # -------
    # connect

    return dax


# ##################################################################
# ##################################################################
#                       Prepare los and vos
# ##################################################################


def _prepare_los(
    coll=None,
    doptics=None,
    los_res=None,
    is2d=None,
    indch=None,
):

    # -----------
    # safety check

    if doptics['los'] is None:
        return [None] * 8

    # ----------
    # los

    los_x, los_y, los_z = coll.sample_rays(
        key=doptics['los'],
        res=los_res,
        mode='rel',
        concatenate=False,
    )

    # ------------------
    # for chosen index

    if is2d:
        los_xi = los_x[:, indch[0], indch[1]]
        los_yi = los_y[:, indch[0], indch[1]]
        los_zi = los_z[:, indch[0], indch[1]]
        los_ri = np.hypot(los_xi, los_yi)

        los_x, los_y, los_z, los_r = None, None, None, None

    else:
        los_r = np.hypot(los_x, los_y)

        los_xi = los_x[:, indch]
        los_yi = los_y[:, indch]
        los_zi = los_z[:, indch]
        los_ri = los_r[:, indch]

        # concatenate
        sh = tuple(np.r_[1, los_x.shape[1:]])
        los_x = np.append(los_x, np.full(sh, np.nan), axis=0).T.ravel()
        los_y = np.append(los_y, np.full(sh, np.nan), axis=0).T.ravel()
        los_z = np.append(los_z, np.full(sh, np.nan), axis=0).T.ravel()
        los_r = np.append(los_r, np.full(sh, np.nan), axis=0).T.ravel()

    return los_x, los_y, los_z, los_r, los_xi, los_yi, los_zi, los_ri


def _prepare_vos(
    coll=None,
    doptics=None,
    los_res=None,
    is2d=None,
    indch=None,
):

    # -----------
    # safety check

    if doptics['los'] is None:
        return [None] * 8

    # -------
    # vos

    kpc = doptics['dvos']['pcross']
    pc0 = coll.ddata[kpc[0]]['data']
    pc1 = coll.ddata[kpc[1]]['data']
    kph = doptics['dvos']['phor']
    ph0 = coll.ddata[kph[0]]['data']
    ph1 = coll.ddata[kph[1]]['data']

    if is2d:
        pc0i = pc0[:, indch[0], indch[1]]
        pc1i = pc1[:, indch[0], indch[1]]
        ph0i = ph0[:, indch[0], indch[1]]
        ph1i = ph1[:, indch[0], indch[1]]

        # envelop
        pc0 = pc0.reshape(pc0.shape[0], -1)
        pc1 = pc1.reshape(pc1.shape[0], -1)
        ph0 = ph0.reshape(ph0.shape[0], -1)
        ph1 = ph1.reshape(ph1.shape[0], -1)

        iok = np.all(np.isfinite(pc0), axis=0).nonzero()[0]
        pc = plg.Polygon(np.array([pc0[:, iok[0]], pc1[:, iok[0]]]).T)
        ph = plg.Polygon(np.array([ph0[:, iok[0]], ph1[:, iok[0]]]).T)
        for ii in iok[1:]:
            pc = pc | plg.Polygon(np.array([pc0[:, ii], pc1[:, ii]]).T)
            ph = ph | plg.Polygon(np.array([ph0[:, ii], ph1[:, ii]]).T)

        # -----------------------
        # convex hull if distinct

        # pc
        if len(pc) > 1:
            # replace by convex hull
            pts = np.concatenate(
                tuple([np.array(pc.contour(ii)) for ii in range(len(pc))]),
                axis=0,
            )
            pc0, pc1 = pts[ConvexHull(pts).vertices, :].T
        else:
            pc0, pc1 = np.array(pc).T

        # ph
        if len(ph) > 1:
            # replace by convex hull
            pts = np.concatenate(
                tuple([np.array(ph.contour(ii)) for ii in range(len(ph))]),
                axis=0,
            )
            ph0, ph1 = pts[ConvexHull(pts).vertices, :].T
        else:
            ph0, ph1 = np.array(ph).T

    else:
        pc0i = pc0[:, indch]
        pc1i = pc1[:, indch]
        ph0i = ph0[:, indch]
        ph1i = ph1[:, indch]

    # safety check
    if np.any(~np.isfinite(pc0i)):
        pc0i, pc1i = None, None
        ph0i, ph1i = None, None

    return pc0, pc1, ph0, ph1, pc0i, pc1i, ph0i, ph1i


# ################################################################
# ################################################################
#                       Prepare sang
# ################################################################


def _prepare_sang(
    coll=None,
    dvos=None,
    key_cam=None,
    indch=None,
    spectro=None,
    is2d=None,
):

    # -----------------
    # get mesh sampling

    dsamp = coll.get_sample_mesh(
        key=dvos['keym'],
        res=dvos['res_RZ'],
        mode='abs',
        grid=False,
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

    # -----------------
    # prepare image

    n0, n1 = dsamp['x0']['data'].size, dsamp['x1']['data'].size
    shape = (n0, n1)
    sang = np.full(shape, np.nan)
    sang_tot = np.full(shape, 0.)

    if is2d:
        for ii in range(dvos['indr_cross']['data'].shape[0]):
            for jj in range(dvos['indr_cross']['data'].shape[1]):
                iok = dvos['indr_cross']['data'][ii, jj, :] >= 0
                indr = dvos['indr_cross']['data'][ii, jj, iok]
                indz = dvos['indz_cross']['data'][ii, jj, iok]
                sang_tot[indr, indz] += dvos['sang_cross']['data'][ii, jj, iok]

                # sang
                if ii == indch[0] and jj == indch[1]:
                    sang[indr, indz] = dvos['sang_cross']['data'][ii, jj, iok]

    else:
        for ii in range(dvos['indr_cross']['data'].shape[0]):
            iok = dvos['indr_cross']['data'][ii, :] >= 0
            indr = dvos['indr_cross']['data'][ii, iok]
            indz = dvos['indz_cross']['data'][ii, iok]
            sang_tot[indr, indz] += dvos['sang_cross']['data'][ii, iok]

            # sang
            if ii == indch:
                sang[indr, indz] = dvos['sang_cross']['data'][ii, iok]

    sang_tot[sang_tot == 0.] = np.nan
    sang[sang == 0.] = np.nan

    # -------------------
    # get integrated vos

    sang_integ = np.nansum(dvos['sang_cross']['data'], axis=-1)

    # extent
    x0 = dsamp['x0']['data']
    dx0 = x0[1] - x0[0]
    x1 = dsamp['x1']['data']
    dx1 = x1[1] - x1[0]

    extent = (
        x0[0] - 0.5*dx0,
        x0[-1] + 0.5*dx0,
        x1[0] - 0.5*dx1,
        x1[-1] + 0.5*dx1,
    )

    # units
    sang_units = dvos['sang_cross']['units']

    return sang_tot, sang_integ, sang, extent, sang_units


def _get_etendue_length(
    coll=None,
    doptics=None,
    poly=None,
):

    # -----------
    # safety check

    if doptics['etendue'] is None:
        return None, None

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


# ###############################################################
# ###############################################################
#                       add mobile
# ###############################################################


def _add_camera_los_cross(
    ax=None,
    is2d=None,
    color_dict=None,
    los_r=None,
    los_z=None,
    pc0=None,
    pc1=None,
    alpha=None,
):

    # ------
    # los

    if (not is2d) and (los_r is not None):
        l0, = ax.plot(
            los_r,
            los_z,
            c='k',
            ls='-',
            lw=1.,
        )

    # ------
    # vos
    if pc0 is not None:
        if pc0.ndim == 2:
            for ii in range(pc0.shape[1]):
                l0, = ax.fill(
                    pc0[:, ii],
                    pc1[:, ii],
                    fc='k',
                    alpha=alpha,
                    ls='None',
                    lw=0.,
                )

        elif pc0.ndim == 3:
            for ii in range(pc0.shape[1]):
                for jj in range(pc0.shape[2]):
                    l0, = ax.fill(
                        pc0[:, ii, jj],
                        pc1[:, ii, jj],
                        fc='k',
                        alpha=alpha,
                        ls='None',
                        lw=0.,
                    )

        else:
            l0, = ax.fill(
                pc0,
                pc1,
                fc='k',
                alpha=alpha,
                ls='None',
                lw=0.,
            )


def _add_camera_los_hor(
    ax=None,
    is2d=None,
    los_x=None,
    los_y=None,
    ph0=None,
    ph1=None,
    alpha=None,
    color_dict=None,
):

    # ------
    # los

    if (not is2d) and (los_x is not None):
        l0, = ax.plot(
            los_x,
            los_y,
            c=color_dict['x'][0],
            ls='-',
            lw=1.,
        )

    # ------
    # vos

    if ph0 is not None:

        if ph0.ndim == 2:
            for ii in range(ph0.shape[1]):
                l0, = ax.fill(
                    ph0[:, ii],
                    ph1[:, ii],
                    fc='k',
                    alpha=alpha,
                    ls='None',
                    lw=0.,
                )

        elif ph0.ndim == 3:
            for ii in range(ph0.shape[1]):
                for jj in range(ph0.shape[2]):
                    l0, = ax.fill(
                        ph0[:, ii, jj],
                        ph1[:, ii, jj],
                        fc='k',
                        alpha=alpha,
                        ls='None',
                        lw=0.,
                    )

        else:
            l0, = ax.fill(
                ph0,
                ph1,
                fc='k',
                alpha=alpha,
                ls='None',
                lw=0.,
            )


def _add_camera_data(
    coll=None,
    ax=None,
    sang_integ=None,
    etendue=None,
    length=None,
    indch=None,
    color_dict=None,
    # vmin, vmax
    vmin_cam=None,
    vmax_cam=None,
    # 2d only
    ax_diff=None,
    ax_etend=None,
    is2d=None,
    x0=None,
    x1=None,
    extent_cam=None,
):

    if is2d:

        # sang
        if sang_integ is not None:
            mi = ax.imshow(
                sang_integ.T,
                origin='lower',
                extent=extent_cam,
                interpolation='nearest',
                vmin=vmin_cam,
                vmax=vmax_cam,
            )

        # etendue
        if etendue is not None:
            etendle = etendue * length
            mi = ax_etend.imshow(
                etendle.T,
                origin='lower',
                extent=extent_cam,
                interpolation='nearest',
                vmin=vmin_cam,
                vmax=vmax_cam,
            )
            plt.colorbar(mi, ax=[ax, ax_etend])

        # diff
        if sang_integ is not None and etendue is not None:
            diff = (sang_integ - etendle).T
            dmax = np.abs(max(np.nanmin(diff), np.nanmax(diff)))
            imd = ax_diff.imshow(
                (sang_integ - etendle).T,
                origin='lower',
                extent=extent_cam,
                interpolation='nearest',
                cmap=plt.cm.seismic,
                vmin=-dmax,
                vmax=dmax,
            )

            plt.colorbar(imd, ax=ax_diff)

        # marker
        if x0 is not None:
            for aa in [ax, ax_etend, ax_diff]:
                aa.plot(
                    [x0[indch[0]]],
                    [x1[indch[1]]],
                    c='k',
                    marker='s',
                    ms=6,
                    ls='None',
                    lw=1.,
                )

    else:

        nlos = sang_integ.shape[0]
        ind = np.arange(0, nlos)

        # vos
        if sang_integ is not None:
            nlos = sang_integ.shape[0]
            ind = np.arange(0, nlos)
            ax.plot(
                ind,
                sang_integ,
                c='b',
                marker='.',
                ls='-',
                lw=1.,
                label="sang * dV (sr.m3)",
            )

        # los
        if etendue is not None:
            nlos = etendue.shape[0]
            ind = np.arange(0, nlos)
            ax.plot(
                ind,
                etendue * length,
                c='k',
                marker='.',
                ls='-',
                lw=1.,
                label="etendue * length (sr.m3)",
            )

        # indch
        ax.axvline(
            indch,
            c='k',
            lw=1.,
            ls='--',
        )

        ax.set_ylim(vmin_cam, vmax_cam)
        plt.legend()


# ################################################################
# ################################################################
#                   figure
# ################################################################


def _get_dax(
    proj=None,
    dmargin=None,
    fs=None,
    tit=None,
    wintit=None,
    is2d=None,
    key_cam=None,
    indch=None,
):

    # ------------
    # check inputs

    # fs
    if fs is None:
        fs = (14.5, 10)

    # dmargin
    dmargin = ds._generic_check._check_var(
        dmargin, 'dmargin',
        types=dict,
        default={
            'bottom': 0.05, 'top': 0.95,
            'left': 0.05, 'right': 0.98,
            'wspace': 0.20, 'hspace': 0.40,
            # 'width_ratios': [0.6, 0.4],
            # 'height_ratios': [0.4, 0.6],
        },
    )

    # wintit
    wintit = ds._generic_check._check_var(
        wintit, 'wintit',
        types=str,
        default=_generic_plot._WINDEF,
    )

    # tit
    if tit is None:
        tit = f"{key_cam} - {indch}\nVOS"

    # -------------
    # Create figure

    fig = plt.figure(figsize=fs)
    fig.canvas.manager.set_window_title(wintit)
    fig.suptitle(tit, size=12, fontweight='bold')

    if is2d:

        gs = gridspec.GridSpec(ncols=4, nrows=4, **dmargin)

        # -------------
        # hor

        ax0 = fig.add_subplot(gs[2:, 2:], aspect='equal', adjustable='datalim')
        ax0.set_xlabel(r'X (m)', size=12)
        ax0.set_ylabel(r'Y (m)', size=12)

        # -------------
        # cross

        ax1 = fig.add_subplot(gs[:2, :2], aspect='equal')
        ax1.set_xlabel(r'R (m)', size=12)
        ax1.set_ylabel(r'Z (m)', size=12)

        # -------------
        # cross

        ax2 = fig.add_subplot(gs[2:, :2], sharex=ax1, sharey=ax1)
        ax2.set_xlabel(r'R (m)', size=12)
        ax2.set_ylabel(r'Z (m)', size=12)

        # -------------
        # camera

        ax3 = fig.add_subplot(gs[0, 2], aspect='equal')
        ax3.set_xlabel(r'x0 (m)', size=12)
        ax3.set_ylabel(r'x1 (m)', size=12)
        ax3.set_title('sang * dV (m3.sr)', size=12, fontweight='bold')

        ax4 = fig.add_subplot(gs[0, 3], sharex=ax3, sharey=ax3)
        ax4.set_xlabel(r'x0 (m)', size=12)
        ax4.set_title('etendue * length (m3.sr)', size=12, fontweight='bold')

        ax5 = fig.add_subplot(gs[1, 2], sharex=ax3, sharey=ax3)
        ax5.set_xlabel(r'x0 (m)', size=12)
        ax5.set_ylabel(r'x1 (m)', size=12)
        ax5.set_title('difference (m3.sr)', size=12, fontweight='bold')

        # ---------
        # dict

        dax = {
            'hor': {'handle': ax0, 'type': 'hor'},
            'cross': {'handle': ax1, 'type': 'cross'},
            'crosstot': {'handle': ax2, 'type': 'cross'},
            key_cam[0]: {'handle': ax3, 'type': 'camera'},
            f'{key_cam[0]}_etend': {'handle': ax4, 'type': 'camera'},
            f'{key_cam[0]}_diff': {'handle': ax5, 'type': 'camera'},
        }

    else:

        gs = gridspec.GridSpec(ncols=2, nrows=2, **dmargin)

        # -------------
        # hor

        ax0 = fig.add_subplot(gs[1, 1], aspect='equal', adjustable='datalim')
        ax0.set_xlabel(r'X (m)', size=12)
        ax0.set_ylabel(r'Y (m)', size=12)

        # -------------
        # cross

        ax1 = fig.add_subplot(gs[0, 0], aspect='equal')
        ax1.set_xlabel(r'R (m)', size=12)
        ax1.set_ylabel(r'Z (m)', size=12)

        # -------------
        # cross

        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
        ax2.set_xlabel(r'R (m)', size=12)
        ax2.set_ylabel(r'Z (m)', size=12)

        # -------------
        # camera

        ax3 = fig.add_subplot(gs[0, 1])
        ax3.set_xlabel(r'index', size=12)
        ax3.set_ylabel(r'data', size=12)

        # ---------
        # dict

        dax = {
            'hor': {'handle': ax0, 'type': 'hor'},
            'cross': {'handle': ax1, 'type': 'cross'},
            'crosstot': {'handle': ax2, 'type': 'cross'},
            key_cam[0]: {'handle': ax3, 'type': 'camera'},
        }

    return dax
