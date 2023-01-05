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
from. import _class1_plot
# from . import _class1_compute as _compute
from . import _spectralunits


# #############################################################################
# #############################################################################
#                           plot mesh
# #############################################################################


def _plot_mesh_prepare_spectral(
    coll=None,
    key=None,
):

    # --------
    # prepare

    kknots = coll.dobj[coll._which_msp][key]['knots'][0]
    knots = coll.ddata[kknots]['data']

    xx = np.array([knots, knots, np.full(knots.shape, np.nan)]).T.ravel()
    yy = np.array([
        np.zeros(knots.shape),
        np.ones(knots.shape),
        np.ones(knots.shape),
    ]).T.ravel()

    return xx, yy


def plot_mesh_spectral(
    coll=None,
    key=None,
    ind_knot=None,
    ind_cent=None,
    units=None,
    nmax=None,
    color=None,
    dax=None,
    dmargin=None,
    fs=None,
    dleg=None,
    connect=None,
):
    """ Plot the desired spectral mesh

    """

    # --------------
    # check input

    (
     key, ind_knot, ind_cent, _, _, color, dleg,
     ) = _class1_plot._plot_mesh_check(
        coll=coll,
        which_mesh=coll._which_msp,
        key=key,
        ind_knot=ind_knot,
        ind_cent=ind_cent,
        color=color,
        dleg=dleg,
    )

    # --------------
    #  Prepare data

    xx, yy = _plot_mesh_prepare_spectral(
        coll=coll,
        key=key,
    )
    
    if units not in [None, 'eV']:
        xx, _, _, cat = _spectralunits.convert_spectral(
            data_in=xx,
            units_in='eV',
            units_out=units,
        )
        xlab = cat + r" ($" + units + "$)"
        
    else:
        xlab = r'energy ($eV$)'

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
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_xlabel(xlab)

        dax = {'spectral': ax0}

    dax = _generic_check._check_dax(dax=dax, main='spectral')

    # --------------
    # plot

    kax = 'spectral'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ax.plot(
            xx,
            yy,
            ls='-',
            lw=0.5,
            color=color,
            alpha=0.5,
            label=key,
        )

        if ind_knot is not None:
            ax.plot(
                ind_knot[0],
                0.5,
                marker='o',
                ms=8,
                ls='None',
                color=color,
                label='knots',
            )
            # ax.plot(
            #     ind_knot[1][0, :, :],
            #     marker='x',
            #     ms=4,
            #     ls='None',
            #     color=color,
            # )

        if ind_cent is not None:
            ax.plot(
                ind_cent[0],
                0.5,
                marker='x',
                ms=8,
                ls='None',
                color=color,
                label='cents',
            )
            # ax.plot(
            #     ind_cent[1][0, :, :],
            #     marker='o',
            #     ms=4,
            #     ls='None',
            #     color=color,
            # )

    # --------------
    # dleg

    if dleg is not False:
        for kax in dax.keys():
            dax[kax]['handle'].legend(**dleg)

    return dax


# #############################################################################
# #############################################################################
#                           plot bspline
# #############################################################################


def _plot_bspline_prepare_spectral(
    coll=None,
    key=None,
    keym=None,
    res=None,
    mode=None,
):
    
    xx = coll.get_sample_mesh_spectral(key=keym, res=res, mode=mode)
    
    # TBF depending on what is practical for LOS
    yy = coll.interpolate_spectrum(
        key=None,
        coefs=None,
        E=xx,
        t=None,
        indt=None,
        indt_strict=None,
        indbs=None,
        details=True,
        reshape=None,
        res=None,
        nan0=None,
        val_out=None,
        return_params=False,
        store=False,
        inplace=False,
    )
    
    return xx, yy


def plot_bspline_spectral(
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
        key, _, keym, mtype0, mtype,
        indbs, indt,
        knots, cents, knotsi, centsi,
        plot_mesh, _, dleg,
    ) = _class1_plot._plot_bspline_check(
        coll=coll,
        key=key,
        indbs=indbs,
        indt=indt,
        knots=knots,
        cents=cents,
        plot_mesh=plot_mesh,
        dleg=dleg,
    )

    # --------------
    #  Prepare data

    xx, yy = _plot_bspline_prepare_spectral(
        coll=coll,
        key=key,
        keym=keym,
        mtype0=mtype0,
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

        dax = {'spectral': ax0}

    dax = _generic_check._check_dax(dax=dax, main='spectral')

    # --------------
    # plot

    kax = 'spectral'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        ax.plot(
            xx,
            np.sum(yy, axis=0),
            c='k',
            lw=2.,
            ls='-',
        )

        ax.plot(
            xx,
            yy.T,
            lw=1.,
            ls='-',
        )

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
#                           plot spectrum 
#               (1d or time-dependent and/or radius dependent)
# #############################################################################


def plot_spectrum(
    coll=None,
    key=None,
):

    # -----------
    # check input
    
    key, keyX, keyY = None, None, None
    
    # ------------------------
    # call appropriate routine
    
    return coll.plot_as_array(
        key=key,
        keyX=keyX,
        keyY=keyY,
    )
    
    
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
