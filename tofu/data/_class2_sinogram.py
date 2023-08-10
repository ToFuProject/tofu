# -*- coding: utf-8 -*-


# Built-in
import warnings
import itertools as itt


# Common
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import datastock as ds


# specific
from . import _generic_check
from . import _generic_plot


# ###############################################################
# ###############################################################
#                       Compute
# ###############################################################


def sinogram(
    coll=None,
    key=None,
    # config
    config=None,
    kVes=None,
    # sinogram ref point
    R0=None,
    Z0=None,
    # sinogram options
    ang=None,
    ang_units=None,
    impact_pos=None,
    pmax=None,
    # plotting options
    plot=None,
    color=None,
    marker=None,
    label=None,
    sketch=None,
    # other options
    verb=None,
    # figure
    dax=None,
    dmargin=None,
    fs=None,
    wintit=None,
):

    # -----------------
    # check
    # -----------------

    (
        key,
        # config
        config,
        kVes,
        # sinogram options
        ang,
        ang_units,
        impact_pos,
        # plotting options
        plot,
        # other options
        verb,
    ) = _check(**locals())

    if len(key) == 0 and config is None:
        return

    # -----------------
    # get R0, Z0
    # -----------------

    R0, Z0 = _get_RZ0(R0=R0, Z0=Z0, config=config, kVes=kVes)

    # -----------------
    # compute for rays
    # -----------------

    dout = {}
    dfail = {}
    for k0 in key:
        dout[k0] = _compute_rays(
            coll=coll,
            kray=k0,
            R0=R0,
            Z0=Z0,
            # options
            verb=verb,
        )

        nans = np.isnan(dout[k0]['ang']).sum()
        if nans == dout[k0]['ang'].size:
            dfail[k0] = 'all nans'
        elif nans > 0:
            dfail[k0] = f"{nans} / {dout[k0]['ang'].size} nans"

    # warnings
    if len(dfail) > 0 and verb == 1:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dfail.items()]
        msg = (
            "Some rays seem to have no impact solution:\n"
            + "\n".join(lstr)
        )
        warnings.warn(msg)

    # -----------------
    # compute for config
    # -----------------

    dout_config = None
    if config is not None:
        dout_config = _compute_config(
            config=config,
            kVes=kVes,
            R0=R0,
            Z0=Z0,
        )

    # ------------
    # adjust
    # ------------

    # convert impact parameter
    if impact_pos is False:
        for k0, v0 in dout.items():
            ineg = v0['ang'] < 0.
            dout[k0]['impact'][ineg] = -v0['impact'][ineg]
            dout[k0]['ang'][ineg] = np.arctan2(
                np.sin(v0['ang'][ineg] + np.pi),
                np.cos(v0['ang'][ineg] + np.pi),
            )

        if dout_config is not None:
            ineg = dout_config['ang'] < 0.
            dout_config['pmax'][ineg] = -dout_config['pmax'][ineg]
            dout_config['ang'][ineg] = np.arctan2(
                np.sin(dout_config['ang'][ineg] + np.pi),
                np.cos(dout_config['ang'][ineg] + np.pi),
            )

    # convert angle
    if ang == 'xi':
        for k0, v0 in dout.items():
            dout[k0]['ang'] = np.arctan2(
                np.sin(v0['ang'] - np.pi/2.),
                np.cos(v0['ang'] - np.pi/2.),
            )

        if dout_config is not None:
            dout_config['ang'] = np.arctan2(
                np.sin(dout_config['ang'] - np.pi/2),
                np.cos(dout_config['ang'] - np.pi/2),
            )

    # convert angle units
    if ang_units == 'deg':
        for k0, v0 in dout.items():
            dout[k0]['ang'] = v0['ang'] * 180 / np.pi

        if dout_config is not None:
            dout_config['ang'] = dout_config['ang'] * 180 / np.pi

    # store in dict
    for k0, v0 in dout.items():
        dout[k0]['ang_var'] = ang
        dout[k0]['ang_units'] = ang_units
        dout[k0]['impact_pos'] = impact_pos

    if dout_config is not None:
        dout_config['ang_var'] = ang
        dout_config['ang_units'] = ang_units
        dout_config['impact_pos'] = impact_pos


    # ------------
    # plot
    # ------------

    if plot is True:

        dax = _plot(
            dout=dout,
            dout_config=dout_config,
            # sinogram ref point
            R0=R0,
            Z0=Z0,
            # sinogram options
            ang=ang,
            ang_units=ang_units,
            impact_pos=impact_pos,
            pmax=pmax,
            # plotting options
            color=color,
            marker=marker,
            label=label,
            sketch=sketch,
            # figure
            dax=dax,
            dmargin=dmargin,
            fs=fs,
            wintit=wintit,
        )

        return dout, dax

    return dout


# ###############################################################
# ###############################################################
#                       Check
# ###############################################################


def _check(
    coll=None,
    key=None,
    # config
    config=None,
    kVes=None,
    # sinogram ref point
    R0=None,
    Z0=None,
    # sinogram options
    ang=None,
    ang_units=None,
    impact_pos=None,
    # plotting options
    plot=None,
    # other options
    verb=None,
    # unused
    **kwdargs,
):

    # -------------
    # key
    # -------------

    # allowed
    lrays = list(coll.dobj.get('rays', {}).keys())
    ldiag = [
        k0 for k0, v0 in coll.dobj.get('diagnostic', {}).items()
        if any([v1.get('los') is not None for v1 in v0['doptics'].values()])
    ]
    ldiag = list(coll.dobj.get('diagnostic', {}).keys())

    if isinstance(key, str):
        key = [key]

    # check
    key = ds._generic_check._check_var_iter(
        key, 'key',
        types=list,
        types_iter=str,
        allowed=ldiag + lrays,
    )

    # if diag => single
    if any([ss in ldiag for ss in key]) and len(key) != 1:
        msg = (
            "Arg key must be either a list of ray keys or a single diag key!\n"
            f"Provided: {key}"
        )
        raise Exception(msg)

    # convrt to list of rays
    if key[0] in ldiag:
        lcam = coll.dobj['diagnostic'][key[0]]['camera']
        doptics = coll.dobj['diagnostic'][key[0]]['doptics']
        key = [
            doptics[k0]['los'] for k0 in lcam
            if doptics[k0]['los'] is not None
        ]

    # -------------------
    # config

    if config is not None:
        lkVes = list(config.dStruct['dObj']['Ves'].keys())
        if kVes is None:
            lkVes = list(config.dStruct['dObj']['Ves'].keys())
            ls = [config.dStruct['dObj']['Ves'][k0].dgeom['Surf'] for k0 in lkVes]
            kVes = lkVes[np.argmin(ls)]

        kVes = ds._generic_check._check_var(
            kVes, 'kVes',
            types=str,
            allowed=lkVes,
        )

    # ------------
    # ang

    ang = ds._generic_check._check_var(
        ang, 'ang',
        types=str,
        default='theta',
        allowed=['xi', 'theta'],
    )

    # ------------
    # ang_units

    ang_units = ds._generic_check._check_var(
        ang_units, 'ang_units',
        types=str,
        default='deg',
        allowed=['deg', 'radian'],
    )

    # ------------
    # impact_pos

    impact_pos = ds._generic_check._check_var(
        impact_pos, 'impact_pos',
        types=bool,
        default=True,
    )

    # ------------
    # plot

    plot = ds._generic_check._check_var(
        plot, 'plot',
        types=bool,
        default=True,
    )

    # --------------
    # verb

    if isinstance(verb, bool):
        verb = 2 if verb else 0

    verb = ds._generic_check._check_var(
        verb, 'verb',
        types=int,
        default=2,
        allowed=[0, 1, 2],
    )

    return (
        key,
        # config
        config,
        kVes,
        # sinogram options
        ang,
        ang_units,
        impact_pos,
        # plotting options
        plot,
        # other options
        verb,
    )


def _get_RZ0(
    R0=None,
    Z0=None,
    config=None,
    kVes=None,
):

    # ------------
    # R0

    if R0 is None:
        if config is None:
            msg = "Please provide a value for R0!"
            raise Exception(msg)

        R0 = config.dStruct['dObj']['Ves'][kVes].dgeom['BaryS'][0]

    # ------------
    # Z0

    if Z0 is None:
        if config is None:
            Z0 = 0.
        else:
            Z0 = config.dStruct['dObj']['Ves'][kVes].dgeom['BaryS'][1]

    return R0, Z0

# ###############################################################
# ###############################################################
#               Compute - rays
# ###############################################################


def _compute_rays(
    coll=None,
    kray=None,
    R0=None,
    Z0=None,
    # options
    verb=None,
):

    # --------------------------------
    # compute (for first segment only)

    # starting points
    ptsx, ptsy, ptsz = coll.get_rays_pts(kray)
    ptsx, ptsy, ptsz = ptsx[0, ...], ptsy[0, ...], ptsz[0, ...]

    # unit vectors
    vectx, vecty, vectz = coll.get_rays_vect(kray, norm=True)
    vectx, vecty, vectz = vectx[0, ...], vecty[0, ...], vectz[0, ...]
    vnorm2d = np.sqrt(vectx**2 + vecty**2)

    # dphi
    phi = np.arctan2(ptsy, ptsx)
    cosdphi = np.cos(phi) * vectx / vnorm2d + np.sin(phi) * vecty / vnorm2d
    dphi = np.arccos(np.abs(cosdphi))

    # ----------------------
    # solve k / AM = k v and AC = ZA ez + RA eRA

    # Z = ZA + k (v.ez)
    # R^2 = RA^2 + k^2 vpar^2 + 2 k RA (v.erA)
    # R eR.v = RA eRA.v + (ZA - Z) vz + k
    #        = RA eRA.v + k vpar^2
    # AM.v = k = AC.v + R0 eR.v + p er.v (=0)
    # ( R0 (R eR.v) )^2 = ( R [k - AC.v]  )^2
    #                   = ( R [k + CA.v]  )^2
    #
    # Thus:
    # R0^2 [ RA eRA.v + k vpar^2 ]^2 = R^2 [ k + CA.v ]^2
    #
    # R0^2 [ k^2 vpar^4 + 2 k vpar^2 RA (eRA.v) + RA^2 (eRA.v)^2 ]
    #    = [RA^2 + k^2 vpar^2 + 2 k RA (erA.v)] [ k^2 + 2k (CA.v) + (CA.v)^2 ]
    #
    # Polynom deg 4 = 0
    # k^4:
    #   vpar^2
    # k^3:
    #   2(CA.v) vpar^2 + 2RA (eRA.v)
    # k^2:
    #   vpar^2 (CA.v)^2 + 4RA(erA.v)(CA.v) + RA^2 - R0^2 vpar^4
    # k:
    #   2RA (eRA.v)(CA.v)^2 + 2(CA.v)RA^2 - 2 R0^2 vpar^2 RA (eRA.v)
    # 0:
    #   RA^2 (CA.v)^2 - R0^2 RA^2 (eRA.v)^2

    CAv = ptsx * vectx + ptsy * vecty + (ptsz - Z0) * vectz
    vpar2 = vnorm2d**2
    RA = np.hypot(ptsx, ptsy)
    eRAv = np.cos(phi) * vectx + np.sin(phi) * vecty

    c4 = vpar2
    c3 = 2*CAv * vpar2 + 2*RA * eRAv
    c2 = vpar2 * CAv**2 + 4*RA*eRAv*CAv + RA**2 - R0**2 * vpar2**2
    c1 = 2*RA * eRAv * CAv**2 + 2*CAv*RA**2 - 2*R0**2 * vpar2 * RA * eRAv
    c0 = RA**2 * CAv**2 - R0**2 * RA**2 * eRAv**2

    # --------------------------------
    # loop on rays

    linds = [range(ss) for ss in ptsx.shape]

    dwarn = {}
    roots0 = np.full(ptsx.shape, np.nan)
    for ind in itt.product(*linds):

        coefs = np.r_[c0[ind], c1[ind], c2[ind], c3[ind], c4[ind]]
        if not np.all(np.isfinite(coefs)):
            continue

        # define polynomial
        poly = np.polynomial.polynomial.Polynomial(coefs)

        # roots
        roots = poly.roots()
        roots = np.real(roots[np.isreal(roots)])

        nsol = roots.size
        if nsol > 0:
            if nsol > 2:
                dwarn[ind] = roots

            elif nsol <= 2:
                roots0[ind] = roots[np.argmin(np.abs(roots))]

    # -----------------------
    # warnings

    if len(dwarn) > 0 and verb == 2:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dwarn.items()]
        msg = (
            "\nSinogram computation\n"
            f"The following rays of '{kray}' have non-unique solutions:\n"
            + "\n".join(lstr)
        )
        warnings.warn(msg)

    # -----------------------
    # derive impact and angle

    impact = np.full(ptsx.shape, np.nan)
    ang = np.full(ptsx.shape, np.nan)

    iok = np.isfinite(roots0)

    rootsx = ptsx[iok] + roots0[iok] * vectx[iok]
    rootsy = ptsy[iok] + roots0[iok] * vecty[iok]
    rootsz = ptsz[iok] + roots0[iok] * vectz[iok]

    rootphi = np.arctan2(rootsy, rootsx)

    erx = rootsx - R0*np.cos(rootphi)
    ery = rootsy - R0*np.sin(rootphi)
    erz = rootsz - Z0

    ern = np.sqrt(erx**2 + ery**2 + erz**2)

    # check dm.v = 0
    sca = (erx * vectx[iok] + ery * vecty[iok] + erz * vectz[iok]) / ern
    if not np.allclose(sca[ern > 1e-10], 0., atol=1e-4):
        msg = (
            f"Some rays in '{kray}' seem to have inconsistent solutions:\n"
            f"sca = {sca}\n"
            f"ern = {ern}"
        )
        raise Exception(msg)

    # impact and ang
    impact[iok] = ern
    ang[iok] = np.arctan2(erz, erx*np.cos(rootphi) + ery*np.sin(rootphi))

    # -------------
    # format ouput

    dout = {
        'R0': R0,
        'Z0': Z0,
        'root_kk': roots0,
        'root_ptx': rootsx,
        'root_pty': rootsy,
        'root_ptz': rootsz,
        'impact': impact,
        'ang': ang,
        'dphi': dphi,
    }

    return dout



# ###############################################################
# ###############################################################
#               Compute - vessel
# ###############################################################


def _compute_config(
    config=None,
    kVes=None,
    R0=None,
    Z0=None,
    # options
    verb=None,
):


    # --------------
    # Select polygon

    polyR, polyZ = config.dStruct['dObj']['Ves'][kVes].dgeom['Poly']

    # ----------------
    # compute

    ang = np.pi * np.linspace(-1, 1, 101)

    pmax = np.max(
        (polyR[:, None] - R0) * np.cos(ang)[None, :]
        + (polyZ[:, None] - Z0) * np.sin(ang)[None, :],
        axis=0,
    )

    dout = {
        'ang': ang,
        'pmax': pmax,
    }

    return dout


# ###############################################################
# ###############################################################
#               Plot
# ###############################################################


def _plot(
    dout=None,
    dout_config=None,
    # sinogram ref point
    R0=None,
    Z0=None,
    # sinogram options
    ang=None,
    ang_units=None,
    impact_pos=None,
    pmax=None,
    # plotting options
    color=None,
    marker=None,
    label=None,
    sketch=None,
    # figure
    dax=None,
    dmargin=None,
    fs=None,
    wintit=None,
):

    # --------------
    # check

    (
        pmax,
        dprops,
        sketch,
        dax,
    ) = _check_plot(**locals())

    # -----------
    # prepare


    # -----------
    # plot rays

    # sinogram
    kax = 'sinogram'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # plot rays
        for k0, v0 in dout.items():
            ax.plot(
                v0['ang'].ravel(),
                v0['impact'].ravel(),
                **dprops[k0],
            )

        # legend
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1, 1),
        )

        # -----------
        # plot config

        if dout_config is not None:

            inds = np.argsort(dout_config['ang'])
            if dout_config['impact_pos'] is True:
                ax.fill_between(
                    dout_config['ang'][inds],
                    dout_config['pmax'][inds],
                    pmax*np.ones((inds.size,)),
                    fc=(0.8, 0.8, 0.8),
                )

            else:
                ipos = dout_config['pmax'] >= 0
                inds = np.argsort(dout_config['ang'][ipos])
                ax.fill_between(
                    dout_config['ang'][ipos][inds],
                    dout_config['pmax'][ipos][inds],
                    pmax*np.ones((ipos.sum(),)),
                    fc=(0.8, 0.8, 0.8),
                )

                inds = np.argsort(dout_config['ang'][~ipos])
                ax.fill_between(
                    dout_config['ang'][~ipos][inds],
                    dout_config['pmax'][~ipos][inds],
                    -pmax*np.ones(((~ipos).sum(),)),
                    fc=(0.8, 0.8, 0.8),
                )

    # -----------
    # plot sketch

    kax = 'sketch'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        _plot_sketch(ax)

    return dax


# ###############################################################
# ###############################################################
#               Check plot
# ###############################################################


def _check_plot(
    dout=None,
    dout_config=None,
    # parameters
    R0=None,
    Z0=None,
    # angles
    ang=None,
    ang_units=None,
    impact_pos=None,
    pmax=None,
    # options
    color=None,
    marker=None,
    label=None,
    sketch=None,
    # figure
    dax=None,
    dmargin=None,
    fs=None,
    wintit=None,
    tit=None,
    # unused
    **kwdargs,
):

    # --------------
    # dprops

    dprops = {}
    for k0 in dout.keys():
        dprops[k0] = {
            'label': k0 if label is None else label,
            'color': color,
            'marker': '.' if marker is None else marker,
        }

    # -------------
    # sketch

    sketch = ds._generic_check._check_var(
        sketch, 'sketch',
        types=bool,
        default=True,
    )

    # -------------
    # pmax

    if pmax is None:

        lpmax = [np.nanmax(np.abs(v0['impact'])) for v0 in dout.values()]

        if dout_config is not None:
            lpmax.append(np.nanmax(np.abs(dout_config['pmax'])))

        if not np.any(np.isfinite(lpmax)):
            lstr = [f"\t- {k0}" for k0 in dout.keys()]
            msg = (
                "Impact parameters are all nans for:\n"
                + "\n".join(lstr)
                + f"\nlen(dout) = {len(dout)}\n"
                + f"dout_config is None: {dout_config is None}"
            )
            raise Exception(msg)

        pmax = np.nanmax(lpmax)

    pmax = float(ds._generic_check._check_var(
        pmax, 'pmax',
        types=(float, int),
        sign='>=0',
    ))

    # -------------
    # figure

    if fs is None:
        fs = (10, 8)

    if wintit is None:
        wintit = 'tofu - sinogram'

    if tit is None:
        tit = f"Sinogram with respect to (R, Z) = ({R0}, {Z0}) m"

    if dmargin is None:
        dmargin = {
            'left': 0.10, 'right': 0.95,
            'bottom': 0.10, 'top': 0.90,
            'wspace': 0.25, 'hspace': 0.1,
        }

    # -------------
    # sketch

    if dax is None:
        dax = _get_ax(
            ang=ang,
            ang_units=ang_units,
            impact_pos=impact_pos,
            pmax=pmax,
            # figure
            fs=fs,
            wintit=wintit,
            tit=tit,
            dmargin=dmargin,
        )

    elif issubclass(dax, plt.Axes):
        dax = {'sinogram': {'handle': dax}}

    return (
        pmax,
        dprops,
        sketch,
        dax,
    )


def _get_ax(
    ang=None,
    ang_units=None,
    impact_pos=None,
    pmax=None,
    # figure
    dmargin=None,
    fs=None,
    wintit=None,
    tit=None,
):

    # create figure and axes
    fig = plt.figure(figsize=fs)
    fig.canvas.manager.set_window_title(wintit)
    fig.suptitle(tit, size=14, fontweight='bold')

    gs = GridSpec(ncols=6, nrows=6, **dmargin)

    # sinogram
    ax0 = fig.add_subplot(gs[:, :-1])
    ax0.set_xlabel(r"$" + f"\{ang}" + r"$" + f" ({ang_units})", size=12, fontweight='bold')
    ax0.set_ylabel(r"$p$ (m)", size=12, fontweight='bold')

    angmax = np.pi
    if ang_units == 'deg':
        angmax = 180

    if impact_pos is True:
        ax0.set_xlim(-angmax, angmax)
        ax0.set_ylim(0, pmax)
    else:
        if ang == 'theta':
            ax0.set_xlim(0, angmax)
        else:
            ax0.set_xlim(-angmax/2., angmax/2.)
        ax0.set_ylim(-pmax, pmax)

    # sketch
    ax1 = fig.add_subplot(
        gs[-1, -1],
        aspect='equal',
        adjustable='datalim',
        frameon=False,
    )
    ax1.set_xticks([])
    ax1.set_yticks([])

    # output dict
    dax = {
        'sinogram': {'handle': ax0},
        'sketch': {'handle': ax1},
    }

    return dax


# ##########################################################
# ##########################################################
#
# ##########################################################


def _plot_sketch(ax=None):

    # get points
    pt = np.array([[0, -0.8], [0, 0.8]])
    line = np.array([[-1.6, 0.1], [0, 1.7]])
    hor = np.array([[-0.4, 0.2], [1.2, 1.2]])
    theta = np.linspace(0, 3.*np.pi/4., 30)
    ksi = np.linspace(0, np.pi/4., 10)

    theta = np.array([0.3*np.cos(theta),0.3*np.sin(theta)])
    ksi = np.array([-0.4+0.4*np.cos(ksi), 1.2+0.4*np.sin(ksi)])

    # plot
    ax.plot(
        pt[0,:], pt[1,:], '+k',
        pt[0,:], pt[1,:], '--k',
        line[0,:], line[1,:], '-k',
        hor[0,:], hor[1,:], '-k',
        theta[0,:], theta[1,:], '-k',
        ksi[0,:], ksi[1,:], '-k',
    )

    # annotate
    ax.annotate(
        r"$\theta$",
        xy=(0.3,0.4),
        xycoords='data',
        va="center",
        ha="center",
    )
    ax.annotate(
        r"$\xi$",
        xy=(0.1,1.4),
        xycoords='data',
        va="center",
        ha="center",
    )
    ax.annotate(
        r"$p$",
        xy=(-0.7,0.3),
        xycoords='data',
        va="center",
        ha="center",
    )
    return
