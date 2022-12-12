# -*- coding: utf-8 -*-


# Built-in


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import datastock as ds

# specific
from . import _generic_check
from . import _class1_compute as _compute


# #############################################################################
# #############################################################################
#                           plot mesh
# #############################################################################


def _plot_mesh_check(
    coll=None,
    key=None,
    ind_knot=None,
    ind_cent=None,
    crop=None,
    bck=None,
    color=None,
    dleg=None,
):

    # key
    key = ds._generic_check._check_var(
        key, 'key',
        default=None,
        types=str,
        allowed=list(coll.dobj.get('mesh', {}).keys()),
    )

    # crop, bck
    crop = ds._generic_check._check_var(crop, 'crop', default=True, types=bool)
    bck = ds._generic_check._check_var(bck, 'bck', default=True, types=bool)

    # ind_knot
    if ind_knot is not None:
        ind_knot = coll.select_mesh_elements(
            key=key, ind=ind_knot, elements='knots',
            returnas='data', return_neighbours=True, crop=crop,
        )

    # ind_cent
    if ind_cent is not None:
        ind_cent = coll.select_mesh_elements(
            key=key, ind=ind_cent, elements='cents',
            returnas='data', return_neighbours=True, crop=crop,
        )

    # color
    if color is None:
        color = 'k'
    if not mcolors.is_color_like(color):
        msg = (
            "Arg color must be a valid matplotlib color identifier!\n"
            f"Provided: {color}"
        )
        raise Exception(msg)

    # dleg
    defdleg = {
        'bbox_to_anchor': (1.1, 1.),
        'loc': 'upper left',
        'frameon': True,
    }
    dleg = ds._generic_check._check_var(
        dleg, 'dleg',
        default=defdleg,
        types=(bool, dict),
    )

    return key, ind_knot, ind_cent, crop, bck, color, dleg


def _plot_mesh_prepare(
    coll=None,
    key=None,
    crop=None,
    bck=None,
):

    # --------
    # prepare

    meshtype = coll.dobj['mesh'][key]['type']

    grid_bck = None
    if meshtype == 'rect':
        Rk, Zk = coll.dobj['mesh'][key]['knots']
        R = coll.ddata[Rk]['data']
        Z = coll.ddata[Zk]['data']

        vert = np.array([
            np.repeat(R, 3),
            np.tile((Z[0], Z[-1], np.nan), R.size),
        ])
        hor = np.array([
            np.tile((R[0], R[-1], np.nan), Z.size),
            np.repeat(Z, 3),
        ])

        # --------
        # compute

        if crop is False or coll.dobj['mesh'][key]['crop'] is False:
            grid = np.concatenate((vert, hor), axis=1)

        else:

            crop = coll.ddata[coll.dobj['mesh'][key]['crop']]['data']

            grid = []
            icropR = np.r_[range(R.size-1), R.size-2]
            jcropZ = np.r_[range(Z.size-1), Z.size-2]

            # vertical lines  TBC
            for ii, ic in enumerate(icropR):
                if np.any(crop[ic, :]):
                    if ii in [0, R.size-1]:
                        cropi = crop[ic, :]
                    else:
                        cropi = crop[ic, :] | crop[ic-1, :]
                    lseg = []
                    for jj, jc in enumerate(jcropZ):
                        if jj == 0 and cropi[jc]:
                            lseg.append(Z[jj])
                        elif jj == Z.size-1 and cropi[jc]:
                            lseg.append(Z[jj])
                        elif cropi[jc] and not cropi[jc-1]:
                            if len(lseg) > 0:
                                lseg.append(np.nan)
                            lseg.append(Z[jj])
                        elif (not cropi[jc]) and cropi[jc-1]:
                            lseg.append(Z[jc])
                    grid.append(np.concatenate(
                        (
                            np.array([R[ii]*np.ones((len(lseg),)), lseg]),
                            np.full((2, 1), np.nan)
                        ),
                        axis=1,
                    ))

            # horizontal lines
            for jj, jc in enumerate(jcropZ):
                if np.any(crop[:, jc]):
                    if jj in [0, Z.size-1]:
                        cropj = crop[:, jc]
                    else:
                        cropj = crop[:, jc] | crop[:, jc-1]
                    lseg = []
                    for ii, ic in enumerate(icropR):
                        if ii in [0, R.size-1] and cropj[ic]:
                            lseg.append(R[ii])
                        elif cropj[ic] and not cropj[ic-1]:
                            if len(lseg) > 0:
                                lseg.append(np.nan)
                            lseg.append(R[ii])
                        elif (not cropj[ic]) and cropj[ic-1]:
                            lseg.append(R[ic])
                    grid.append(np.concatenate(
                        (
                            np.array([lseg, Z[jj]*np.ones((len(lseg),))]),
                            np.full((2, 1), np.nan)
                        ),
                        axis=1,
                    ))

            grid = np.concatenate(tuple(grid), axis=1)

            if bck is True:
                grid_bck = np.concatenate((vert, hor), axis=1)

    else:
        kknots = coll.dobj['mesh'][key]['knots']
        R = coll.ddata[kknots[0]]['data']
        Z = coll.ddata[kknots[1]]['data']

        indtri = coll.ddata[coll.dobj['mesh'][key]['ind']]['data']

        # find unique segments from all triangles
        segs = np.unique(
            np.sort(np.concatenate(
                (indtri[:, 0:2], indtri[:, 1:], indtri[:, ::2]),
                axis=0,
            )),
            axis=0,
        )

        # build long segments if possible
        ind = np.ones((segs.shape[0],), dtype=bool)
        ind[0] = False
        lseg = [segs[0, :]]
        last = segs[0, :]
        while np.any(ind):
            ii = segs[ind, 0] == last[-1]
            if np.any(ii):
                ii = ind.nonzero()[0][ii]
                dR0 = R[last[1]] - R[last[0]]
                dZ0 = Z[last[1]] - Z[last[0]]
                dR = np.diff(R[segs[ii, :]], axis=1)[:, 0]
                dZ = np.diff(Z[segs[ii, :]], axis=1)[:, 0]
                norm0 = np.sqrt(dR0**2 + dZ0**2)
                norm = np.sqrt(dR**2 + dZ**2)
                sca = (dR0*dR + dZ0*dZ) / (norm0 * norm)
                iwin = ii[np.argmax(sca)]
                lseg.append([segs[iwin, 1]])

            else:
                lseg.append([-1])
                iwin = ind.nonzero()[0][0]
                lseg.append(segs[iwin, :])

            last = segs[iwin, :]
            ind[iwin] = False

        lseg = np.concatenate(lseg)
        grid = np.array([R[lseg], Z[lseg]])
        grid[0, lseg == -1] = np.nan

    return grid, grid_bck


def _plot_mesh_prepare_polar_cont(
    coll=None,
    key=None,
    k2d=None,
    RR=None,
    ZZ=None,
    ind=None,
    nn=None,
):

    # ---------------------
    # sample mesh if needed

    # ---------------------
    # get map of rr / angle

    if callable(k2d):

        # check RR
        if RR is None:
            msg = (
                "radius2d / angle2d are callable => provide RR and ZZ!"
            )
            raise Exception(msg)

        # compute map
        rr = k2d(RR, ZZ)[None, ...]
        assert rr.ndim == RR.ndim + 1
        reft = None
        nt = 1

        if nn is None:
            nn = 50

        # create vector
        rad = np.linspace(np.nanmin(rr), np.nanmax(rr), nn)

    else:
        kn = coll.dobj[coll._which_mesh][key]['knots'][ind]
        rad = coll.ddata[kn]['data']
        kb2 = coll.ddata[k2d]['bsplines']

        if RR is None:
            km2 = coll.dobj['bsplines'][kb2]['mesh']
            RR, ZZ = coll.get_sample_mesh(
                key=km2,
                res=None,
                grid=True,
                mode=None,
                R=None,
                Z=None,
                DR=None,
                DZ=None,
                imshow=True,
            )

        rr = coll.interpolate_profile2d(
            key=k2d,
            R=RR,
            Z=ZZ,
            grid=False,
            return_params=False,
        )[0]

        refr2d = coll.ddata[k2d]['ref']
        refbs = coll.dobj['bsplines'][kb2]['ref']
        if refr2d == refbs:
            reft = None
            nt = 1
            rr = rr[None, ...]
        elif len(refr2d) == len(refbs) + 1 and refr2d[1:] == refbs:
            reft = refr2d[0]
            nt = coll.dref[reft]['size']

    assert rr.shape[0] == nt

    # ----------------
    # Compute contours

    contR, contZ = _compute._get_contours(
        RR=RR,
        ZZ=ZZ,
        val=rr,
        levels=rad,
    )

    # refrad
    refrad = coll.dobj[coll._which_mesh][key]['knots'][ind]

    return contR, contZ, rad, reft, refrad, RR, ZZ


def _plot_mesh_prepare_polar(
    coll=None,
    key=None,
    # Necessary for callable radius2d
    RR=None,
    ZZ=None,
):

    # --------
    # prepare

    # create rectangular grid and compute radius at each point
    k2d = coll.dobj[coll._which_mesh][key]['radius2d']
    (
        contRrad, contZrad,
        rad, reft, refrad,
        RR, ZZ,
    ) = _plot_mesh_prepare_polar_cont(
        coll=coll,
        key=key,
        k2d=k2d,
        RR=RR,
        ZZ=ZZ,
        ind=0,
        nn=None,        # nrad if k2d callable
    )

    # -----------
    # contour of angle if angle not None

    contRang, contZang, ang, refang = None, None, None, None
    if len(coll.dobj[coll._which_mesh][key]['shape-c']) == 2:
        # create rectangular grid and compute radius at each point
        k2d = coll.dobj[coll._which_mesh][key]['angle2d']
        (
            contRang, contZang,
            ang, _, refang,
            _, _,
        ) = _plot_mesh_prepare_polar_cont(
            coll=coll,
            key=key,
            k2d=k2d,
            RR=RR,
            ZZ=ZZ,
            ind=1,
            nn=None,        # nang if k2d callable
        )

    return (
        contRrad, contZrad, rad, refrad,
        contRang, contZang, ang, refang,
        reft,
    )


def plot_mesh(
    coll=None,
    key=None,
    ind_knot=None,
    ind_cent=None,
    crop=None,
    bck=None,
    nmax=None,
    color=None,
    dax=None,
    dmargin=None,
    fs=None,
    dleg=None,
    connect=None,
):
    """ Plot the desired mesh

    rect and tri meshes are constant
    polar meshes can vary in time

    """

    # --------------
    # check input

    key, ind_knot, ind_cent, crop, bck, color, dleg = _plot_mesh_check(
        coll=coll,
        key=key,
        ind_knot=ind_knot,
        ind_cent=ind_cent,
        crop=crop,
        bck=bck,
        color=color,
        dleg=dleg,
    )

    # ------------------------
    # call appropriate routine

    if coll.dobj[coll._which_mesh][key]['type'] in ['rect', 'tri']:
        # time-fixed meshes
        return _plot_mesh_recttri(
            coll=coll,
            key=key,
            ind_knot=ind_knot,
            ind_cent=ind_cent,
            crop=crop,
            bck=bck,
            color=color,
            dax=dax,
            fs=fs,
            dmargin=dmargin,
            dleg=dleg,
        )

    else:
        # possibly time-varying mesh
        return _plot_mesh_polar(
            coll=coll,
            key=key,
            nmax=nmax,
            color=color,
            dax=dax,
            fs=fs,
            dmargin=dmargin,
            dleg=dleg,
            connect=connect,
        )


def _plot_mesh_recttri(
    coll=None,
    key=None,
    ind_knot=None,
    ind_cent=None,
    crop=None,
    bck=None,
    color=None,
    dax=None,
    fs=None,
    dmargin=None,
    dleg=None,
):

    # --------------
    #  Prepare data

    grid, grid_bck = _plot_mesh_prepare(
        coll=coll,
        key=key,
        crop=crop,
        bck=bck,
    )

    # --------------
    # plot - prepare

    if dax is None:

        if dmargin is None:
            dmargin = {
                'left': 0.1, 'right': 0.9,
                'bottom': 0.1, 'top': 0.9,
                'hspace': 0.1, 'wspace': 0.1,
            }

        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(ncols=1, nrows=1, **dmargin)
        ax0 = fig.add_subplot(gs[0, 0], aspect='equal')
        ax0.set_xlabel(f'R (m)')
        ax0.set_ylabel(f'Z (m)')

        dax = {'cross': ax0}

    dax = _generic_check._check_dax(dax=dax, main='cross')

    # --------------
    # plot

    kax = 'cross'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        if grid_bck is not None and bck is True:
            ax.plot(
                grid_bck[0, :],
                grid_bck[1, :],
                ls='-',
                lw=0.5,
                color=color,
                alpha=0.5,
                label=key,
            )

        ax.plot(
            grid[0, :],
            grid[1, :],
            color=color,
            ls='-',
            lw=1.,
            label=key,
        )

        if ind_knot is not None:
            ax.plot(
                ind_knot[0][0],
                ind_knot[0][1],
                marker='o',
                ms=8,
                ls='None',
                color=color,
                label='knots',
            )
            ax.plot(
                ind_knot[1][0, :, :],
                ind_knot[1][1, :, :],
                marker='x',
                ms=4,
                ls='None',
                color=color,
            )

        if ind_cent is not None:
            ax.plot(
                ind_cent[0][0],
                ind_cent[0][1],
                marker='x',
                ms=8,
                ls='None',
                color=color,
                label='cents',
            )
            ax.plot(
                ind_cent[1][0, :, :],
                ind_cent[1][1, :, :],
                marker='o',
                ms=4,
                ls='None',
                color=color,
            )

    # --------------
    # dleg

    if dleg is not False:
        for kax in dax.keys():
            dax[kax]['handle'].legend(**dleg)

    return dax


def _plot_mesh_polar(
    coll=None,
    key=None,
    npts=None,
    nmax=None,
    color=None,
    dax=None,
    fs=None,
    dmargin=None,
    dleg=None,
    connect=None,
):

    # --------------
    #  Prepare data

    if nmax is None:
        nmax = 2

    (
        contRrad, contZrad, rad, refrad,
        contRang, contZang, ang, refang,
        reft,
    ) = _plot_mesh_prepare_polar(
        coll=coll,
        key=key,
    )
    refptsr = 'ptsr'
    nt, nr, nptsr = contRrad.shape
    if contRang is not None:
        refptsa = 'ptsa'
        _, nang, nptsa = contRang.shape

    # --------------------
    # Instanciate Plasma2D

    coll2 = coll.__class__()

    # ref
    coll2.add_ref(
        key=reft,
        size=nt,
    )
    reft = list(coll2.dref.keys())[0]
    coll2.add_ref(
        key=refrad,
        size=nr,
    )
    coll2.add_ref(
        key=refptsr,
        size=nptsr,
    )

    if contRang is not None:
        coll2.add_ref(
            key=refang,
            size=nang,
        )
        coll2.add_ref(
            key=refptsa,
            size=nptsa,
        )

    # data
    coll2.add_data(
        key='radius',
        data=rad,
        ref=(refrad,)
    )
    coll2.add_data(
        key='contRrad',
        data=contRrad,
        ref=(reft, refrad, refptsr)
    )
    coll2.add_data(
        key='contZrad',
        data=contZrad,
        ref=(reft, refrad, refptsr)
    )

    if contRang is not None:
        coll2.add_data(
            key='angle',
            data=ang,
            ref=(refang,)
        )
        coll2.add_data(
            key='contRang',
            data=contRang,
            ref=(reft, refang, refptsa)
        )
        coll2.add_data(
            key='contZang',
            data=contZang,
            ref=(reft, refang, refptsa)
        )

    # -----
    # plot

    if contRang is None:
        return coll2.plot_as_mobile_lines(
            keyX='contRrad',
            keyY='contZrad',
            key_time=reft,
            key_chan='radius',
            connect=connect,
        )

    else:

        daxrad, dgrouprad = coll2.plot_as_mobile_lines(
            keyX='contRrad',
            keyY='contZrad',
            key_time=reft,
            key_chan='radius',
            connect=False,
            inplace=False,
        )

        daxang, dgroupang = coll2.plot_as_mobile_lines(
            keyX='contRang',
            keyY='contZang',
            key_time=reft,
            key_chan='angle',
            connect=False,
            inplace=False,
        )

        # connect
        if connect is False:
            return (daxrad, daxang), (dgrouprad, dgroupang)

        else:
            daxrad.setup_interactivity(
                kinter='inter0', dgroup=dgrouprad, dinc=None,
            )
            daxrad.disconnect_old()
            daxrad.connect()

            daxang.setup_interactivity(
                kinter='inter0', dgroup=dgroupang, dinc=None,
            )
            daxang.disconnect_old()
            daxang.connect()

            daxrad.show_commands()
            return daxrad, daxang


# #############################################################################
# #############################################################################
#                           plot bspline
# #############################################################################


def _plot_bsplines_get_dRdZ(coll=None, km=None, meshtype=None):
    # Get minimum distances

    if meshtype == 'rect':
        kR, kZ = coll.dobj['mesh'][km]['knots']
        Rk = coll.ddata[kR]['data']
        Zk = coll.ddata[kZ]['data']
        dR = np.min(np.diff(Rk))
        dZ = np.min(np.diff(Zk))

    elif meshtype == 'tri':
        indtri = coll.ddata[coll.dobj['mesh'][km]['ind']]['data']
        kknots = coll.dobj['mesh'][km]['knots']
        Rk = coll.ddata[kknots[0]]['data']
        Zk = coll.ddata[kknots[1]]['data']
        R = Rk[indtri]
        Z = Zk[indtri]
        dist = np.mean(np.array([
            np.sqrt((R[:, 1] - R[:, 0])**2 + (Z[:, 1] - Z[:, 0])**2),
            np.sqrt((R[:, 2] - R[:, 1])**2 + (Z[:, 2] - Z[:, 1])**2),
            np.sqrt((R[:, 2] - R[:, 0])**2 + (Z[:, 2] - Z[:, 0])**2),
        ]))
        dR, dZ = dist, dist

    else:
        km2 = coll.dobj[coll._which_mesh][km]['submesh']
        meshtype = coll.dobj[coll._which_mesh][km2]['type']
        return _plot_bsplines_get_dRdZ(
            coll=coll, km=km2, meshtype=meshtype,
        )

    Rminmax = [Rk.min(), Rk.max()]
    Zminmax = [Zk.min(), Zk.max()]
    return dR, dZ, Rminmax, Zminmax


def _plot_bspline_check(
    coll=None,
    key=None,
    indbs=None,
    indt=None,
    knots=None,
    cents=None,
    plot_mesh=None,
    cmap=None,
    dleg=None,
):

    # key
    lk = list(coll.dobj.get('bsplines', {}).keys())
    key = ds._generic_check._check_var(
        key, 'key',
        default=None,
        types=str,
        allowed=lk,
    )
    keym0 = coll.dobj['bsplines'][key]['mesh']
    mtype0 = coll.dobj[coll._which_mesh][keym0]['type']
    if mtype0 == 'polar':
        keym = coll.dobj[coll._which_mesh][keym0]['submesh']
        mtype = coll.dobj[coll._which_mesh][keym]['type']
    else:
        keym = keym0
        mtype = mtype0

    # knots, cents
    knots = ds._generic_check._check_var(
        knots, 'knots', default=True, types=bool,
    )
    cents = ds._generic_check._check_var(
        cents, 'cents', default=True, types=bool,
    )

    # ind_bspline
    if indbs is not None:
        indbs = coll.select_bsplines(
            key=key,
            ind=indbs,
            returnas='ind',
            return_knots=False,
            return_cents=False,
            crop=False,
        )

    _, knotsi, centsi = coll.select_bsplines(
        key=key,
        ind=indbs,
        returnas='data',
        return_knots=True,
        return_cents=True,
        crop=False,
    )

    # indt
    nt = False
    if mtype0 == 'polar':
        radius2d = coll.dobj[coll._which_mesh][keym0]['radius2d']
        r2d_reft = coll.get_time(key=radius2d)[2]
        if r2d_reft is not None:
            nt = coll.dref[r2d_reft]['size']

    if nt is False:
        indt = None
    else:
        if indt is None:
            indt = 0
        indt = np.atleast_1d(indt).ravel()[0]

    # plot_mesh
    plot_mesh = ds._generic_check._check_var(
        plot_mesh, 'plot_mesh',
        default=True,
        types=bool,
    )

    # cmap
    if cmap is None:
        cmap = 'viridis'

    # dleg
    defdleg = {
        'bbox_to_anchor': (1.1, 1.),
        'loc': 'upper left',
        'frameon': True,
    }
    dleg = ds._generic_check._check_var(
        dleg, 'dleg',
        default=defdleg,
        types=(bool, dict),
    )

    return (
        key, keym0, keym, mtype0, mtype,
        indbs, indt,
        knots, cents, knotsi, centsi,
        plot_mesh, cmap, dleg,
    )


def _plot_bspline_prepare(
    coll=None,
    # keys
    key=None,
    keym0=None,
    keym=None,
    mtype0=None,
    mtype=None,
    # indices
    indbs=None,
    indt=None,
    # options
    res=None,
    knotsi=None,
    centsi=None,
    val_out=None,
    nan0=None,
):

    # check input
    deg = coll.dobj['bsplines'][key]['deg']

    # get dR, dZ
    dR, dZ, _, _ = _plot_bsplines_get_dRdZ(
        coll=coll, km=keym, meshtype=mtype,
    )

    # resolution of sampling
    if res is None:
        if mtype == 'rect':
            res_coef = 0.05
        else:
            res_coef = 0.25
        res = [res_coef*dR, res_coef*dZ]

    # sampling domain
    if mtype0 == 'polar':
        DR = None
        DZ = None
    else:
        knotsiR, knotsiZ = knotsi
        DR = [np.nanmin(knotsiR) + dR*1.e-10, np.nanmax(knotsiR) - dR*1.e-10]
        DZ = [np.nanmin(knotsiZ) + dZ*1.e-10, np.nanmax(knotsiZ) - dZ*1.e-10]

    # sample
    R, Z = coll.get_sample_mesh(
        key=keym,
        res=res,
        DR=DR,
        DZ=DZ,
        mode='abs', grid=True, imshow=True,
    )

    # bspline
    bspline = coll.interpolate_profile2d(
        key=key,
        R=R,
        Z=Z,
        # coefs=coefs,
        indt=indt,
        indbs=indbs,
        details=indbs is not None,
        grid=False,
        nan0=nan0,
        val_out=val_out,
        return_params=False,
    )[0]

    if indbs is None:
        if bspline.ndim == R.ndim + 1:
            assert bspline.shape[1:] == R.shape
            bspline = bspline[0, ...]
    else:
        if bspline.ndim == R.ndim + 1:
            assert bspline.shape[:-1] == R.shape
            bspline = np.nansum(bspline, axis=-1)
        elif bspline.ndim == R.ndim + 2:
            assert bspline.shape[1:-1] == R.shape
            bspline = np.nansum(bspline[0, ...], axis=-1)

    if bspline.shape != R.shape:
        import pdb; pdb.set_trace() # DB
        pass

    # extent
    if mtype0 == 'polar':
        extent = (
            R.min(), R.max(),
            Z.min(), Z.max(),
        )
    else:
        extent = (
            DR[0], DR[1],
            DZ[0], DZ[1],
        )

    # interpolation
    if deg == 0:
        interp = 'nearest'
    elif deg == 1:
        interp = 'bilinear'
    elif deg >= 2:
        interp = 'bicubic'

    return bspline, extent, interp


def plot_bspline(
    # ressources
    coll=None,
    # inputs
    key=None,
    indbs=None,
    indt=None,
    # parameters
    knots=None,
    cents=None,
    res=None,
    plot_mesh=None,
    val_out=None,
    nan0=None,
    # plot-specific
    cmap=None,
    dax=None,
    dmargin=None,
    fs=None,
    dleg=None,
):

    # --------------
    # check input

    (
        key, keym0, keym, mtype0, mtype,
        indbs, indt,
        knots, cents, knotsi, centsi,
        plot_mesh, cmap, dleg,
    ) = _plot_bspline_check(
        coll=coll,
        key=key,
        indbs=indbs,
        indt=indt,
        knots=knots,
        cents=cents,
        plot_mesh=plot_mesh,
        cmap=cmap,
        dleg=dleg,
    )

    # --------------
    #  Prepare data

    bspline, extent, interp = _plot_bspline_prepare(
        coll=coll,
        key=key,
        keym=keym,
        mtype0=mtype0,
        mtype=mtype,
        indbs=indbs,
        indt=indt,
        knotsi=knotsi,
        centsi=centsi,
        res=res,
        val_out=val_out,
        nan0=nan0,
    )

    # --------------
    # plot - prepare

    if dax is None:

        if dmargin is None:
            dmargin = {
                'left': 0.1, 'right': 0.9,
                'bottom': 0.1, 'top': 0.9,
                'hspace': 0.1, 'wspace': 0.1,
            }

        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(ncols=1, nrows=1, **dmargin)
        ax0 = fig.add_subplot(gs[0, 0], aspect='equal')
        ax0.set_xlabel(f'R (m)')
        ax0.set_ylabel(f'Z (m)')

        dax = {'cross': ax0}

    dax = _generic_check._check_dax(dax=dax, main='cross')

    # --------------
    # plot

    if plot_mesh is True:
        keym = coll.dobj['bsplines'][key]['mesh']
        if mtype0 == 'polar':
            _ = coll.plot_mesh(key=keym, dleg=False)
        else:
            dax = coll.plot_mesh(key=keym, dax=dax, dleg=False)

    kax = 'cross'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ax.imshow(
            bspline,
            extent=extent,
            interpolation=interp,
            origin='lower',
            aspect='equal',
            cmap=cmap,
            vmin=0.,
            vmax=1.,
        )

        if mtype0 != 'polar':
            if knots is not False:
                ax.plot(
                    knotsi[0].ravel(),
                    knotsi[1].ravel(),
                    marker='x',
                    ms=6,
                    ls='None',
                    color='k',
                )

            if cents is not False:
                ax.plot(
                    centsi[0].ravel(),
                    centsi[1].ravel(),
                    marker='o',
                    ms=6,
                    ls='None',
                    color='k',
                )

        ax.relim()
        ax.autoscale()

        # --------------
        # dleg

        if dleg is not False:
            ax.legend(**dleg)

    return dax


# #############################################################################
# #############################################################################
#                           plot profile2d
# #############################################################################


def _plot_profile2d_check(
    coll=None,
    key=None,
    cmap=None,
    dcolorbar=None,
    dleg=None,
):

    # key
    dk = coll.get_profiles2d()
    key = ds._generic_check._check_var(
        key, 'key', types=str, allowed=list(dk.keys())
    )
    keybs = dk[key]
    refbs = coll.dobj['bsplines'][keybs]['ref']
    keym = coll.dobj['bsplines'][keybs]['mesh']
    mtype = coll.dobj[coll._which_mesh][keym]['type']

    # cmap
    if cmap is None:
        cmap = 'viridis'

    # dcolorbar
    defdcolorbar = {
        # 'location': 'right',
        'fraction': 0.15,
        'orientation': 'vertical',
    }
    dcolorbar = ds._generic_check._check_var(
        dcolorbar, 'dcolorbar',
        default=defdcolorbar,
        types=dict,
    )

    # dleg
    defdleg = {
        'bbox_to_anchor': (1.1, 1.),
        'loc': 'upper left',
        'frameon': True,
    }
    dleg = ds._generic_check._check_var(
        dleg, 'dleg',
        default=defdleg,
        types=(bool, dict),
    )

    # polar1d
    polar1d = False
    if mtype  == 'polar':
        if coll.dobj['bsplines'][keybs]['class'].knotsa is None:
            polar1d = True
        elif len(coll.dobj['bsplines'][keybs]['ref']) == 2:
            polar1d = True
        else:
            apbs = coll.dobj['bsplines'][keybs]['class'].apex_per_bs_a
            if np.sum([aa is not None for aa in apbs]) == 1:
                polar1d = True

    return key, keybs, keym, cmap, dcolorbar, dleg, polar1d


def _plot_profiles2d_prepare(
    coll=None,
    key=None,
    keybs=None,
    keym=None,
    coefs=None,
    indt=None,
    res=None,
    mtype=None,
):

    # check input
    deg = coll.dobj['bsplines'][keybs]['deg']

    # get dR, dZ
    dR, dZ, Rminmax, Zminmax = _plot_bsplines_get_dRdZ(
        coll=coll, km=keym, meshtype=mtype,
    )

    if res is None:
        res_coef = 0.2
        res = [res_coef*dR, res_coef*dZ]

    # compute
    coll2 = coll.interpolate_profile2d(
        key=key,
        R=None,
        Z=None,
        coefs=coefs,
        indt=indt,
        res=res,
        details=False,
        nan0=True,
        imshow=False,
        return_params=False,
        store=True,
        inplace=False,
    )

    keymap = [k0 for k0, v0 in coll2.ddata.items() if v0['data'].ndim > 1][0]
    ndim = coll2.ddata[keymap]['data'].ndim

    refmap = coll2.ddata[keymap]['ref']
    dkeys = {
        'key': keymap,
        'keyX': coll2.get_ref_vector(key=keymap, ref=refmap[-2])[3],
        'keyY': coll2.get_ref_vector(key=keymap, ref=refmap[-1])[3],
        'keyZ': None,
    }
    if ndim == 3:
        keyZ = coll2.get_ref_vector(key=keymap, ref=refmap[0])[3]
        import datastock as ds
        uniform = ds._plot_as_array._check_uniform_lin(
            k0=keyZ, ddata=coll2.ddata,
        )
        if not uniform:
            keyZ = None

    if deg == 0:
        interp = 'nearest'
    elif deg == 1:
        interp = 'bilinear'
    elif deg >= 2:
        interp = 'bicubic'

    # radial of polar
    if mtype == 'polar':
        # lcol
        lcol = ['k', 'r', 'b', 'g', 'm', 'c', 'y']

    else:
        lcol = None

    return coll2, dkeys, interp, lcol


def _plot_profile2d_polar_add_radial(
    coll=None,
    key=None,
    keym=None,
    keybs=None,
    dax=None,
):

    # key to radius
    kr2d = coll.dobj[coll._which_mesh][keym]['radius2d']
    kr = coll.dobj[coll._which_mesh][keym]['knots'][0]
    rr = coll.ddata[kr]['data']
    rad = np.linspace(rr[0], rr[-1], rr.size*20)

    # get angle if any
    clas = coll.dobj['bsplines'][keybs]['class']
    if clas.knotsa is None:
        angle = None
    elif len(clas.shapebs) == 2:
        ka = coll.dobj['bsplines'][keybs]['apex'][1]
        angle = coll.ddata[ka]['data']
    elif np.sum(clas.nbs_a_per_r > 1) == 1:
        i0 = (clas.nbs_a_per_r > 1).nonzero()[0][0]
        angle = coll.dobj['bsplines'][keybs]['class'].apex_per_bs_a[i0]
    else:
        pass

    if angle is None:
        radmap = rad
        anglemap = angle
    else:
        radmap = np.repeat(rad[:, None], angle.size, axis=1)
        anglemap = np.repeat(angle[None, :], rad.size, axis=0)

    # reft
    reft, keyt, _, dind = coll.get_time_common(keys=[key, kr2d])[1:]

    # radial total profile
    radial, t_radial, _ = coll.interpolate_profile2d(
        key=key,
        radius=radmap,
        angle=anglemap,
        grid=False,
        t=keyt,
    )

    if reft is not None and radial.ndim == radmap.ndim:
        radial = np.repeat(radial[None, ...], t_radial.size, axis=0)

    # details for purely-radial cases
    if clas.knotsa is None:
        radial_details, t_radial, _ = coll.interpolate_profile2d(
            key=keybs,
            radius=rad,
            angle=None,
            grid=False,
            details=True,
        )

        if reft is None:
            radial_details = radial_details * coll.ddata[key]['data'][None, :]
            refdet = ('nradius',)
        else:
            refdet = (reft, 'nradius')
            if reft == coll.get_time(key)[2]:
                radial_details = (
                    radial_details[None, :, :]
                    * coll.ddata[key]['data'][:, None, :]
                )
            elif key in dind.keys():
                radial_details = (
                    radial_details[None, :, :]
                    * coll.ddata[key]['data'][dind[key]['ind'], None, :]
                )

        nbs = radial_details.shape[-1]

    # add to dax
    dax.add_ref(key='nradius', size=rad.size)
    if angle is not None:
        dax.add_ref(key='nangle', size=angle.size)

    if reft is not None:
        assert radial.ndim > 1 and radial.shape[0] > 1
        # if angle is not None:
            # ref = (reft, 'nangle', 'nradius')
        # else:
        ref = (reft, 'nradius')
    else:
        # if angle is not None:
            # ref = ('nangle', 'nradius')
        # else:
        ref = 'nradius'

    # add to ddata
    kradius = 'radius'
    dax.add_data(key=kradius, data=rad, ref='nradius')
    if angle is None:
        lk = ['radial']
        dax.add_data(key=lk[0], data=radial, ref=ref)
        lkdet = [f'radial-detail-{ii}' for ii in range(nbs)]
        for ii in range(nbs):
            dax.add_data(
                key=lkdet[ii], data=radial_details[..., ii], ref=refdet,
            )

    else:
        kangle = 'angle'
        dax.add_data(key=kangle, data=angle, ref='nangle')
        lkdet = None
        lk = [f'radial-{ii}' for ii in range(angle.size)]
        if reft is None:
            for ii in range(angle.size):
                dax.add_data(key=lk[ii], data=radial[:, ii], ref=ref)
        else:
            for ii in range(angle.size):
                dax.add_data(key=lk[ii], data=radial[:, :, ii], ref=ref)

    return kradius, lk, lkdet, reft


def plot_profile2d(
    # ressources
    coll=None,
    # inputs
    key=None,
    # parameters
    coefs=None,
    indt=None,
    res=None,
    # figure
    vmin=None,
    vmax=None,
    cmap=None,
    dax=None,
    dmargin=None,
    fs=None,
    dcolorbar=None,
    dleg=None,
    # interactivity
    dinc=None,
    connect=None,
):

    # --------------
    # check input

    if connect is None:
        connect = True

    (
        key, keybs, keym, cmap, dcolorbar, dleg, polar1d,
    ) = _plot_profile2d_check(
        coll=coll,
        key=key,
        cmap=cmap,
        dcolorbar=dcolorbar,
        dleg=dleg,
    )
    mtype = coll.dobj[coll._which_mesh][keym]['type']
    hastime = coll.get_time(key=key)[0]

    # --------------
    #  Prepare data

    (
        coll2, dkeys, interp, lcol,
    ) = _plot_profiles2d_prepare(
        coll=coll,
        key=key,
        keybs=keybs,
        keym=keym,
        coefs=coefs,
        indt=indt,
        res=res,
        mtype=mtype,
    )

    # ---------------
    # call right function

    if mtype in ['rect', 'tri']:
        return coll2.plot_as_array(
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
            interp=interp,
            connect=connect,
            **dkeys,
        )

    else:

        if dax is None:
            dax = _plot_profile2d_polar_create_axes(
                fs=fs,
                dmargin=dmargin,
            )

        dax, dgroup = coll2.plot_as_array(
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            dcolorbar=dcolorbar,
            dleg=dleg,
            connect=False,
            interp=interp,
            label=True,
            **dkeys,
        )

        # ------------------
        # add radial profile to dax

        kradius, lkradial, lkdet, reft = _plot_profile2d_polar_add_radial(
            coll=coll,
            key=key,
            keym=keym,
            keybs=keybs,
            dax=dax,
        )


        assert (reft is not None) == ('Z' in dgroup.keys())
        if reft is not None and reft not in dgroup['Z']['ref']:
            dgroup['Z']['ref'].append(reft)
            dgroup['Z']['data'].append('index')

        # ------------------
        # add radial profile

        kax = 'radial'
        if dax.dax.get(kax) is not None:
            ax = dax.dax[kax]['handle']
            for ii in range(len(lkradial)):

                if reft is None:
                    l0, = ax.plot(
                        dax.ddata[kradius]['data'],
                        dax.ddata[lkradial[ii]]['data'],
                        c=lcol[ii],
                        ls='-',
                        lw=2,
                    )
                else:
                    l0, = ax.plot(
                        dax.ddata[kradius]['data'],
                        dax.ddata[lkradial[ii]]['data'][0, :],
                        c=lcol[ii],
                        ls='-',
                        lw=2,
                    )

                    kl = f"radial{ii}"
                    dax.add_mobile(
                        key=kl,
                        handle=l0,
                        refs=(reft,),
                        data=[lkradial[ii]],
                        dtype=['ydata'],
                        axes=kax,
                        ind=0,
                    )

            if lkdet is not None:
                for ii in range(len(lkdet)):
                    if reft is None:
                        l0, = ax.plot(
                            dax.ddata[kradius]['data'],
                            dax.ddata[lkdet[ii]]['data'],
                            ls='-',
                            lw=1,
                        )
                    else:
                        l0, = ax.plot(
                            dax.ddata[kradius]['data'],
                            dax.ddata[lkdet[ii]]['data'][0, :],
                            ls='-',
                            lw=1,
                        )

                        kl = f"radial_det{ii}"
                        dax.add_mobile(
                            key=kl,
                            handle=l0,
                            refs=(reft,),
                            data=[lkdet[ii]],
                            dtype=['ydata'],
                            axes=kax,
                            ind=0,
                        )

            ax.set_xlim(
                dax.ddata[kradius]['data'].min(),
                dax.ddata[kradius]['data'].max(),
            )

            if vmin is not None:
                ax.set_ylim(bottom=vmin)
            if vmax is not None:
                ax.set_ylim(top=vmax)

        # connect
        if connect is True:
            dax.setup_interactivity(kinter='inter0', dgroup=dgroup, dinc=dinc)
            dax.disconnect_old()
            dax.connect()

            dax.show_commands()
            return dax
        else:
            return dax, dgroup



def _plot_profile2d_polar_create_axes(
    fs=None,
    dmargin=None,
):

    if fs is None:
        fs = (15, 9)

    if dmargin is None:
        dmargin = {
            'left': 0.05, 'right': 0.95,
            'bottom': 0.05, 'top': 0.95,
            'hspace': 0.4, 'wspace': 0.3,
        }

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(ncols=6, nrows=6, **dmargin)

    # axes for image
    ax0 = fig.add_subplot(gs[:4, 2:4], aspect='auto')

    # axes for vertical profile
    ax1 = fig.add_subplot(gs[:4, 4], sharey=ax0)

    # axes for horizontal profile
    ax2 = fig.add_subplot(gs[4:, 2:4], sharex=ax0)

    # axes for traces
    ax3 = fig.add_subplot(gs[2:4, :2])

    # axes for traces
    ax7 = fig.add_subplot(gs[:2, :2], sharey=ax2)

    # axes for text
    ax4 = fig.add_subplot(gs[:3, 5], frameon=False)
    ax5 = fig.add_subplot(gs[3:, 5], frameon=False)
    ax6 = fig.add_subplot(gs[4:, :2], frameon=False)

    # dax
    dax = {
        # data
        'matrix': {'handle': ax0, 'type': 'matrix'},
        'vertical': {'handle': ax1, 'type': 'misc'},
        'horizontal': {'handle': ax2, 'type': 'misc'},
        'traces': {'handle': ax3, 'type': 'misc'},
        'radial': {'handle': ax7, 'type': 'misc'},
        # text
        'textX': {'handle': ax4, 'type': 'text'},
        'textY': {'handle': ax5, 'type': 'text'},
        'textZ': {'handle': ax6, 'type': 'text'},
    }
    return dax
