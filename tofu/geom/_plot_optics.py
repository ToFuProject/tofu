

# Built-in
import itertools as itt
import sys

# Common
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mplcol
import matplotlib.gridspec as gridspec
from matplotlib.axes._axes import Axes
from mpl_toolkits.mplot3d import Axes3D

# tofu
from tofu.version import __version__
from . import _def as _def

_GITHUB = 'https://github.com/ToFuProject/tofu/issues'
_WINTIT = 'tofu-%s        report issues / requests at %s'%(__version__, _GITHUB)

_QUIVERCOLOR = plt.cm.viridis(np.linspace(0, 1, 3))
_QUIVERCOLOR = np.array([[1., 0., 0., 1.],
                         [0., 1., 0., 1.],
                         [0., 0., 1., 1.]])
_QUIVERCOLOR = mplcol.ListedColormap(_QUIVERCOLOR)
_RAYS_NPTS = 10


# Generic
def _check_projdax_mpl(
    dax=None, proj=None,
    dmargin=None, fs=None, wintit=None,
):

    # ----------------------
    # Check inputs
    if proj is None:
        proj = 'all'
    assert isinstance(proj, str)
    proj = proj.lower()
    lproj = ['cross', 'hor', '3d', 'im']
    assert proj in lproj + ['all']

    # ----------------------
    # Check dax
    lc = [
        dax is None,
        issubclass(dax.__class__, Axes),
        isinstance(dax, dict),
        isinstance(dax, list),
    ]
    assert any(lc)
    if lc[0]:
        dax = dict.fromkeys(proj)
    elif lc[1]:
        assert len(proj) == 1
        dax = {proj[0]: dax}
    elif lc[2]:
        lcax = [
            dax.get(pp) is None or issubclass(dax.get(pp).__class__, Axes)
            for pp in proj
        ]
        if not all(lcax):
            msg = (
                "Wrong key or axes in dax:\n"
                + "    - proj = {}".format(proj)
                + "    - dax = {}".format(dax)
            )
            raise Exception(msg)
    else:
        assert len(dax) == 2
        assert all(
            [ax is None or issubclass(ax.__class__, Axes) for ax in dax]
        )
        dax = {'cross': dax[0], 'hor': dax[1]}

    # Populate with default axes if necessary
    if proj == 'cross' and dax['cross'] is None:
        dax['cross'] = _def.Plot_LOSProj_DefAxes(
            'cross', fs=fs, dmargin=dmargin, wintit=wintit,
        )
    elif proj == 'hor' and dax['hor'] is None:
        dax['hor'] = _def.Plot_LOSProj_DefAxes(
            'hor', fs=fs, dmargin=dmargin, wintit=wintit,
        )
    elif proj == '3d' and dax['3d'] is None:
        dax['3d'] = _def.Plot_3D_plt_Tor_DefAxes(
            fs=fs, dmargin=dmargin, wintit=wintit,
        )
    elif proj == 'im' and dax['im'] is None:
        dax['im'] = _def.Plot_CrystIm(
            fs=fs, dmargin=dmargin, wintit=wintit,
        )
    elif proj == 'all' and any([dax.get(k0) is None for k0 in lproj]):
        dax = _def.Plot_AllCryst(
            fs=fs, dmargin=dmargin, wintit=wintit,
        )
    for kk in lproj:
        dax[kk] = dax.get(kk, None)
    return dax


# #################################################################
# #################################################################
#                   Generic geometry plot
# #################################################################
# #################################################################


def _CrystalBragg_plot_check(
    cryst=None, dcryst=None,
    det=None, ddet=None,
    res=None, element=None,
    color=None,
    pts_summit=None, pts1=None, pts2=None,
    xi=None, xj=None,
    rays_color=None, rays_npts=None,
    dleg=None, draw=True,
    use_non_parallelism=None,
    wintit=None, tit=None,
):

    # plotting
    assert type(draw) is bool, "Arg draw must be a bool !"
    assert cryst is None or cryst.__class__.__name__ == 'CrystalBragg'
    if wintit is None:
        wintit = _WINTIT
    if tit is None:
        tit = False
    if dleg is None:
         dleg = _def.TorLegd

    # rays color
    if rays_color is None:
        if pts_summit is None:
            rays_color = 'k'
        else:
            rays_color = 'pts' if pts_summit.shape[2] > 1 else 'lamb'

    if rays_color in ['pts', 'lamb']:
        pass
    else:
        try:
            rays_color = mplcol.to_rgba(rays_color)
        except Exception as err:
            msg = (
                "Arg rays_color must be either:\n"
                + "\t- 'pts': color by pts of origin\n"
                + "\t- 'lamb': color by wavelength\n"
                + "\t- a matplotlib color (e.g.: 'k', (0.1, 0.2, 0.1), ...)\n"
            )
            raise Exception(msg)

    # rays_npts
    if rays_npts is None:
        rays_npts = _RAYS_NPTS
    assert rays_npts >= 2, "Arg rays_npts must be a int >= 2"

    # elements
    lelement = ['s', 'c', 'r', 'o', 'v']
    if element is None:
        element = 'oscrv'
    c0 = (
        isinstance(element, str)
        and all([ss in lelement for ss in element.lower()])
    )
    if not c0:
        msg = ("Arg element must be str contain some of the following:\n"
               + "\t- 'o': outline\n"
               + "\t- 'c': center (of curvature sphere)\n"
               + "\t- 's': summit (geometrical center of crystal piece)\n"
               + "\t- 'r': rowland circle (along e1 direction)\n"
               + "\t- 'v': local unit vectors\n"
               + "You provided:\n{}".format(element))
        raise Exception(msg)
    element = element.lower()

    # cryst
    if element != '' and cryst is None:
        msg = (
            "cryst cannot be None if element contains any of:\n"
            + "\t- {}\n".format(lelement)
            + "You provided: {}".format(element)
        )
        raise Exception(msg)

    # vectors and outline
    if cryst is not None:
        nout, e1, e2, use_non_parallelism = cryst.get_unit_vectors(
            use_non_parallelism=use_non_parallelism,
        )
        nin = -nout
        outline = cryst.sample_outline_plot(
            res=res,
            use_non_parallelism=use_non_parallelism,
        )

    # det
    if det is None:
        det = False
    if det is not False and any([ss in element for ss in 'ocv']):
        c0 = isinstance(det, dict) and 'cent' in det.keys()
        if c0 and 'o' in element:
            c0 = c0 and all(
                [ss in det.keys() for ss in ['outline', 'ei', 'ej']]
            )
        if c0 and 'v' in element:
            c0 = c0 and all([ss in det.keys() for ss in ['nout', 'ei', 'ej']])
        if not c0:
            msg = ("Arg det must be a dict with keys:\n"
                   + "\t- 'cent': center of the detector\n"
                   + "\t- 'nout': outward unit vector (normal to surface)\n"
                   + "\t- 'ei': first local coordinate unit vector\n"
                   + "\t- 'ej': second coordinate unit vector\n"
                   + "\t- 'outline': 2d local coordinates of outline\n"
                   + "\tAll (except outline) are 3d cartesian coordinates\n"
                   + "You provided:\n{}".format(det))
            raise Exception(msg)

    # pts
    lc = [pts_summit is not None, pts1 is not None, pts2 is not None]
    c0 = (
        (not any(lc))
        or all(lc)
    )
    if not c0:
        msg = (
            "pts_summit and pts1 and pts2 must be:\n"
            + "\t- all None\n"
            + "\t- all np.ndarray of same shape, with shape[0] == 3\n"
            + "  You provided:\n"
            + "\t- pts_summit: {}\n".format(pts_summit)
            + "\t- pts1: {}".format(pts1)
            + "\t- pts2: {}".format(pts2)
        )
        raise Exception(msg)
    if pts_summit is not None:
        if not (pts_summit.shape == pts1.shape == pts2.shape):
            msg = (
                "Args pts_summit, pts1 and pts2 must have the same shape!\n"
                + "  You provided:\n"
                + "\t- pts_summit.shape: {}\n".format(pts_summit.shape)
                + "\t- pts1.shape: {}\n".format(pts1.shape)
                + "\t- pts2.shape: {}\n".format(pts2.shape)
            )
            raise Exception(msg)

    if rays_color in ['pts', 'lamb'] and pts_summit is not None:
        if pts_summit.ndim not in [4, 5]:
            msg = (
                "For pts-wise or lambda-wise coloring, "
                + "input pts_summit must be 4d np.ndarray of shape "
                + "(3, nlamb, npts, ndtheta)\n"
                + "  You provided:\n"
                + "\t- pts_summit.shape = {}".format(pts_summit.shape)
            )
            raise Exception(msg)

    # rays
    rays = None
    if pts_summit is not None:
        # pts.shape = (3, nlamb, npts, ndtheta)
        # rays.shape = (3, nlamb, npts, ndtheta, 2, 2*rays_npts)
        shape = np.r_[pts1.shape, 1]
        k = np.linspace(0, 1, rays_npts)
        rays = np.concatenate((
            pts1[..., None] + k*(pts_summit-pts1)[..., None],
            pts_summit[..., None]
            + k[1:]*(pts2-pts_summit)[..., None],
            np.full(shape, np.nan),
        ), axis=-1)

        nlamb, npts, ndtheta, _, nk = rays.shape[1:]
        if rays_color in ['pts', 'lamb']:
            if rays_color == 'lamb':
                rays = rays.reshape(3, nlamb, npts*ndtheta*nk*2).swapaxes(1, 2)
            elif rays_color == 'pts':
                rays = rays.swapaxes(1, 2).reshape(
                    3, npts, nlamb*ndtheta*nk*2,
                ).swapaxes(1, 2)
        else:
            rays = rays.reshape(3, nlamb*npts*ndtheta*nk*2, order='C')

    # xi, xj
    lc = [xi is not None, xj is not None]
    c0 = (
        np.sum(lc) in [0, 2]
        and (
            not any(lc)
            or (lc[0] and xi.shape == xj.shape == pts1.shape[1:])
        )
    )
    if not c0:
        msg = (
            "Args xi, xj must be either both None of 2 array of same shape!\n"
            + "  Provided:\n\t- xi:\n{}\n\t- xj:\n{}".format(xi, xj)
        )
        raise Exception(msg)
    if lc[0]:
        if rays_color in ['pts', 'lamb']:
            if rays_color == 'lamb':
                xi = xi.reshape(nlamb, npts*ndtheta*2).T
                xj = xj.reshape(nlamb, npts*ndtheta*2).T
            elif rays_color == 'pts':
                xi = xi.swapaxes(0, 1).reshape(npts, nlamb*ndtheta*2).T
                xj = xj.swapaxes(0, 1).reshape(npts, nlamb*ndtheta*2).T
        else:
            xi = xi.ravel()
            xj = xj.ravel()

    # dict for plotting
    if color is None:
        color = False
    lkd = ['outline', 'cent', 'summit', 'rowland', 'vectors']
    # Avoid passing default by reference
    if dcryst is None:
        dcryst = dict({k0: dict(v0)
                       for k0, v0 in _def._CRYSTAL_PLOT_DDICT.items()})
    else:
        dcryst = dict({k0: dict(v0) for k0, v0 in dcryst.items()})

    for k0 in lkd:
        if dcryst.get(k0) is None:
            dcryst[k0] = dict(_def._CRYSTAL_PLOT_DDICT[k0])
        if dcryst[k0].get('color') is None:
            if cryst is not None and cryst._dmisc.get('color') is not None:
                dcryst[k0]['color'] = cryst._dmisc['color']
        if color is not False:
            dcryst[k0]['color'] = color
    if ddet is None:
        # Avoid passing default by reference
        ddet = dict({k0: dict(v0)
                     for k0, v0 in _def._DET_PLOT_DDICT.items()})
    else:
        ddet = dict({k0: dict(v0) for k0, v0 in ddet.items()})
    for k0 in lkd:
        if ddet.get(k0) is None:
            ddet[k0] = dict(_def._DET_PLOT_DDICT[k0])
        if color is not False:
            ddet[k0]['color'] = color

    return (
        dcryst, det, ddet,
        nout, nin, e1, e2, xi, xj, outline, element, color,
        rays, rays_color, rays_npts,
        dleg, wintit,
    )


def CrystalBragg_plot(
    cryst=None, dcryst=None,
    det=None, ddet=None,
    dax=None, proj=None, res=None, element=None,
    color=None,
    pts_summit=None, pts1=None, pts2=None,
    xi=None, xj=None,
    rays_color=None, rays_npts=None,
    dleg=None, draw=True, fs=None, dmargin=None,
    use_non_parallelism=None,
    wintit=None, tit=None,
):

    # ---------------------
    # Check / format inputs

    (
        dcryst, det, ddet,
        nout, nin, e1, e2, xi, xj, outline, element, color,
        rays, rays_color, rays_npts,
        dleg, wintit,
    ) = _CrystalBragg_plot_check(
        cryst=cryst, dcryst=dcryst,
        det=det, ddet=ddet,
        res=res, element=element,
        color=color,
        pts_summit=pts_summit, pts1=pts1, pts2=pts2,
        xi=xi, xj=xj,
        rays_color=rays_color, rays_npts=rays_npts,
        dleg=dleg, draw=draw,
        use_non_parallelism=use_non_parallelism,
        wintit=wintit, tit=tit,
    )

    # ---------------------
    # call plotting functions
    dax = _CrystalBragg_plot(
        cryst=cryst, dcryst=dcryst,
        det=det, ddet=ddet,
        nout=nout, nin=nin, e1=e1, e2=e2, outline=outline,
        proj=proj, dax=dax, element=element,
        rays=rays, rays_color=rays_color, rays_npts=rays_npts,
        xi=xi, xj=xj,
        draw=draw, dmargin=dmargin, fs=fs, wintit=wintit,
    )

    # recompute the ax.dataLim
    ax0 = None
    for kk, vv in dax.items():
        if vv is None:
            continue
        dax[kk].relim()
        dax[kk].autoscale_view()
        if dleg is not False:
            dax[kk].legend(**dleg)
        ax0 = vv

    # set title
    if tit != False:
        ax0.figure.suptitle(tit)
    if draw:
        ax0.figure.canvas.draw()
    return dax


def _CrystalBragg_plot(
    cryst=None, dcryst=None,
    det=None, ddet=None,
    nout=None, nin=None, e1=None, e2=None,
    outline=None,
    proj=None, dax=None, element=None,
    rays=None, rays_color=None, rays_npts=None,
    xi=None, xj=None,
    quiver_cmap=None, draw=True,
    dmargin=None, fs=None, wintit=None,
):

    # ---------------------
    # Check / format inputs

    if 'v' in element and quiver_cmap is None:
        quiver_cmap = _QUIVERCOLOR

    # ---------------------
    # Prepare axe and data

    dax = _check_projdax_mpl(
        dax=dax, proj=proj,
        dmargin=dmargin, fs=fs, wintit=wintit,
    )

    if 's' in element or 'v' in element:
        summ = cryst._dgeom['summit']
    if 'c' in element:
        cent = cryst._dgeom['center']
    if 'r' in element:
        ang = np.linspace(0, 2.*np.pi, 200)
        rr = 0.5*cryst._dgeom['rcurve']
        row = cryst._dgeom['summit'] + rr*nin
        row = (row[:, None]
               + rr*(np.cos(ang)[None, :]*nin[:, None]
                     + np.sin(ang)[None, :]*e1[:, None]))

    # ---------------------
    # plot
    cross = dax.get('cross') is not None
    hor = dax.get('hor') is not None
    d3 = dax.get('3d') is not None
    im = dax.get('im') is not None

    if 'o' in element:
        if cross:
            dax['cross'].plot(
                np.hypot(outline[0, :], outline[1, :]),
                outline[2, :],
                label=cryst.Id.NameLTX+' outline',
                **dcryst['outline'],
            )
        if hor:
            dax['hor'].plot(
                outline[0, :], outline[1, :],
                label=cryst.Id.NameLTX+' outline',
                **dcryst['outline'],
            )
        if d3:
            dax['3d'].plot(
                outline[0, :], outline[1, :], outline[2, :],
                label=cryst.Id.NameLTX+' outline',
                **dcryst['outline'],
            )

    if 's' in element:
        if cross:
            dax['cross'].plot(
                np.hypot(summ[0], summ[1]), summ[2],
                label=cryst.Id.NameLTX+" summit",
                **dcryst['summit'],
            )
        if hor:
            dax['hor'].plot(
                summ[0], summ[1],
                label=cryst.Id.NameLTX+" summit",
                **dcryst['summit'],
            )
        if d3:
            dax['3d'].plot(
                summ[0:1], summ[1:2], summ[2:3],
                label=cryst.Id.NameLTX+" summit",
                **dcryst['summit'],
            )

    if 'c' in element:
        if cross:
            dax['cross'].plot(
                np.hypot(cent[0], cent[1]), cent[2],
                label=cryst.Id.NameLTX+" center",
                **dcryst['cent'],
            )
        if hor:
            dax['hor'].plot(
                cent[0], cent[1],
                label=cryst.Id.NameLTX+" center",
                **dcryst['cent'],
            )
        if d3:
            dax['3d'].plot(
                cent[0:1], cent[1:2], cent[2:3],
                label=cryst.Id.NameLTX+" center",
                **dcryst['cent'],
            )

    if 'r' in element:
        if cross:
            dax['cross'].plot(
                np.hypot(row[0, :], row[1, :]), row[2, :],
                label=cryst.Id.NameLTX+' rowland',
                **dcryst['rowland'],
            )
        if hor:
            dax['hor'].plot(
                row[0, :], row[1, :],
                label=cryst.Id.NameLTX+' rowland',
                **dcryst['rowland'],
            )
        if d3:
            dax['3d'].plot(
                row[0, :], row[1, :], row[2, :],
                label=cryst.Id.NameLTX+' rowland',
                **dcryst['rowland'],
            )
    if 'v' in element:
        p0 = np.repeat(summ[:,None], 3, axis=1)
        v = np.concatenate((nout[:, None], e1[:, None], e2[:, None]), axis=1)
        if cross:
            pr = np.hypot(p0[0, :], p0[1, :])
            vr = np.hypot(p0[0, :]+v[0, :], p0[1, :]+v[1, :]) - pr
            dax['cross'].quiver(
                pr, p0[2, :],
                vr, v[2, :],
                np.r_[0., 0.5, 1.], cmap=quiver_cmap,
                angles='xy', scale_units='xy',
                label=cryst.Id.NameLTX+" unit vect",
                **dcryst['vectors'],
            )
        if hor:
            dax['hor'].quiver(
                p0[0, :], p0[1, :],
                v[0, :], v[1, :],
                np.r_[0., 0.5, 1.], cmap=quiver_cmap,
                angles='xy', scale_units='xy',
                label=cryst.Id.NameLTX+" unit vect",
                **dcryst['vectors'],
            )
        if d3:
            dax['3d'].quiver(
                p0[0, :], p0[1, :], p0[2, :],
                v[0, :], v[1, :], v[2, :],
                np.r_[0., 0.5, 1.],
                length=0.1,
                normalize=True,
                cmap=quiver_cmap,
                label=cryst.Id.NameLTX+" unit vect",
                **dcryst['vectors'],
            )

    # -------------
    # Detector
    if det is not False:
        if det.get('cent') is not None and 'c' in element:
            if cross:
                dax['cross'].plot(
                    np.hypot(det['cent'][0], det['cent'][1]),
                    det['cent'][2],
                    label="det_cent",
                    **ddet['cent'],
                )
            if hor:
                dax['hor'].plot(
                    det['cent'][0], det['cent'][1],
                    label="det_cent",
                    **ddet['cent'],
                )
            if d3:
                dax['3d'].plot(
                    det['cent'][0:1],
                    det['cent'][1:2],
                    det['cent'][2:3],
                    label="det_cent",
                    **ddet['cent'],
                )

        if det.get('nout') is not None and 'v' in element:
            assert det.get('ei') is not None and det.get('ej') is not None
            p0 = np.repeat(det['cent'][:, None], 3, axis=1)
            v = np.concatenate((det['nout'][:, None], det['ei'][:, None],
                                det['ej'][:, None]), axis=1)
            if cross:
                pr = np.hypot(p0[0, :], p0[1, :])
                vr = np.hypot(p0[0, :]+v[0, :], p0[1, :]+v[1, :]) - pr
                dax['cross'].quiver(
                    pr, p0[2, :],
                    vr, v[2, :],
                    np.r_[0., 0.5, 1.], cmap=quiver_cmap,
                    angles='xy', scale_units='xy',
                    label="det unit vect",
                    **ddet['vectors'],
                )
            if hor:
                dax['hor'].quiver(
                    p0[0, :], p0[1, :],
                    v[0, :], v[1, :],
                    np.r_[0., 0.5, 1.], cmap=quiver_cmap,
                    angles='xy', scale_units='xy',
                    label="det unit vect",
                    **ddet['vectors'],
                )
            if d3:
                dax['3d'].quiver(
                    p0[0, :], p0[1, :], p0[2, :],
                    v[0, :], v[1, :], v[2, :],
                    np.r_[0., 0.5, 1.],
                    length=0.1,
                    normalize=True,
                    cmap=quiver_cmap,
                    label="det unit vect",
                    **ddet['vectors'],
                )

        if det.get('outline') is not None and 'o' in element:
            det_out = (
                det['outline'][0:1, :]*det['ei'][:, None]
                + det['outline'][1:2, :]*det['ej'][:, None]
                + det['cent'][:, None]
            )

            if cross:
                dax['cross'].plot(
                    np.hypot(det_out[0, :], det_out[1, :]),
                    det_out[2, :],
                    label='det outline',
                    **ddet['outline'],
                )
            if hor:
                dax['hor'].plot(
                    det_out[0, :],
                    det_out[1, :],
                    label='det outline',
                    **ddet['outline'],
                )
            if d3:
                dax['3d'].plot(
                    det_out[0, :],
                    det_out[1, :],
                    det_out[2, :],
                    label='det outline',
                    **ddet['outline'],
                )
            if im:
                dax['im'].plot(
                    det['outline'][0, :],
                    det['outline'][1, :],
                    label='det outline',
                    **ddet['outline'],
                )

    # -------------
    # rays
    if rays is not None:
        if rays_color in ['pts', 'lamb']:
            if cross:
                dax['cross'].set_prop_cycle(None)
                dax['cross'].plot(
                    np.hypot(rays[0, :, :], rays[1, :, :]),
                    rays[2, :, :],
                    lw=1., ls='-',
                )
            if hor:
                dax['hor'].set_prop_cycle(None)
                dax['hor'].plot(
                    rays[0, :, :], rays[1, :, :],
                    lw=1., ls='-',
                )
            if d3:
                dax['3d'].set_prop_cycle(None)
                for ii in range(rays.shape[2]):
                    dax['3d'].plot(
                        rays[0, :, ii],
                        rays[1, :, ii],
                        rays[2, :, ii],
                        lw=1., ls='-',
                    )
            if im:
                dax['3d'].set_prop_cycle(None)
                dax['im'].plot(
                    xi, xj,
                    ls='None',
                    marker='.',
                    ms=6,
                )

        else:
            if cross:
                dax['cross'].set_prop_cycle(None)
                dax['cross'].plot(
                    np.hypot(rays[0, :], rays[1, :]),
                    rays[2, :],
                    color=rays_color, lw=1., ls='-',
                )
            if hor:
                dax['hor'].set_prop_cycle(None)
                dax['hor'].plot(
                    rays[0, :], rays[1, :],
                    color=rays_color, lw=1., ls='-',
                )
            if d3:
                dax['3d'].set_prop_cycle(None)
                dax['3d'].plot(
                    rays[0, :], rays[1, :], rays[2, :],
                    color=rays_color, lw=1., ls='-',
                )
            if im:
                dax['3d'].set_prop_cycle(None)
                dax['im'].plot(
                    xi, xj,
                    ls='None',
                    marker='.',
                    ms=6,
                )
    return dax


# #################################################################
# #################################################################
#                   Rocking curve plot
# #################################################################
# #################################################################


def CrystalBragg_plot_rockingcurve(
    func=None, bragg=None, lamb=None,
    sigma=None, npts=None,
    ang_units=None, axtit=None,
    color=None,
    legend=None, fs=None, ax=None,
):

    # Prepare
    if legend is None:
        legend = True
    if color is None:
        color = 'k'
    if ang_units is None:
        ang_units = 'deg'
    if axtit is None:
        axtit = 'Rocking curve'
    if sigma is None:
        sigma = 0.005*np.pi/180.
    if npts is None:
        npts = 1000
    angle = bragg + 3.*sigma*np.linspace(-1, 1, npts)
    curve = func(angle)
    lab = r"$\lambda = {:9.6} A$".format(lamb*1.e10)
    if ang_units == 'deg':
        angle = angle*180/np.pi
        bragg = bragg*180/np.pi

    # Plot
    if ax is None:
        if fs is None:
            fs = (8, 6)
        fig = plt.figure(figsize=fs)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_title(axtit, size=12)
        ax.set_xlabel('angle ({})'.format(ang_units))
        ax.set_ylabel('reflectivity (adim.)')
    ax.plot(angle, curve, ls='-', lw=1., c=color, label=lab)
    ax.axvline(bragg, ls='--', lw=1, c=color)
    if legend is not False:
        ax.legend()
    return ax


# #################################################################
# #################################################################
#                   Bragg diffraction plot
# #################################################################
# #################################################################

# Deprecated ? re-use ?
def CrystalBragg_plot_approx_detector_params(Rrow, bragg, d, Z,
                                             frame_cent, nn):

    R = 2.*Rrow
    L = 2.*R
    ang = np.linspace(0., 2.*np.pi, 100)

    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8], aspect='equal')

    ax.axvline(0, ls='--', c='k')
    ax.plot(Rrow*np.cos(ang), Rrow + Rrow*np.sin(ang), c='r')
    ax.plot(R*np.cos(ang), R + R*np.sin(ang), c='b')
    ax.plot(L*np.cos(bragg)*np.r_[-1,0,1],
            L*np.sin(bragg)*np.r_[1,0,1], c='k')
    ax.plot([0, d*np.cos(bragg)], [Rrow, d*np.sin(bragg)], c='r')
    ax.plot([0, d*np.cos(bragg)], [Z, d*np.sin(bragg)], 'g')
    ax.plot([0, L/10*nn[1]], [Z, Z+L/10*nn[2]], c='g')
    ax.plot(frame_cent[1]*np.cos(2*bragg-np.pi),
            Z + frame_cent[1]*np.sin(2*bragg-np.pi), c='k', marker='o', ms=10)

    ax.set_xlabel(r'y')
    ax.set_ylabel(r'z')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.), frameon=False)
    return ax


def CrystalBragg_plot_xixj_from_braggangle(bragg=None, xi=None, xj=None,
                                           data=None, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_axes([0.1,0.1,0.8,0.8], aspect='equal')

    for ii in range(len(bragg)):
        deg ='{0:07.3f}'.format(bragg[ii]*180/np.pi)
        ax.plot(xi[:,ii], xj[:,ii], '.', label='bragg %s'%deg)

    ax.set_xlabel(r'xi')
    ax.set_ylabel(r'yi')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.), frameon=False)
    return ax


def CrystalBragg_plot_braggangle_from_xixj(xi=None, xj=None,
                                           bragg=None, angle=None,
                                           ax=None, plot=None,
                                           braggunits='rad', angunits='rad',
                                           leg=None, colorbar=None,
                                           fs=None, wintit=None,
                                           tit=None, **kwdargs):

    # Check inputs
    if isinstance(plot, bool):
        plot = 'contour'
    if fs is None:
        fs = (6, 6)
    if wintit is None:
        wintit = _WINTIT
    if tit is None:
        tit = False
    if colorbar is None:
        colorbar = True
    if leg is None:
        leg = False
    if leg is True:
        leg = {}

    # Prepare axes
    if ax is None:
        fig = plt.figure(figsize=fs)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8],
                          aspect='equal', adjustable='box')
    dobj = {'phi': {'ax': ax}, 'bragg': {'ax': ax}}
    dobj['bragg']['kwdargs'] = dict(kwdargs)
    dobj['phi']['kwdargs'] = dict(kwdargs)
    dobj['phi']['kwdargs']['cmap'] = plt.cm.seismic

    # Clear cmap if colors provided
    if 'colors' in kwdargs.keys():
        if 'cmap' in dobj['bragg']['kwdargs'].keys():
            del dobj['bragg']['kwdargs']['cmap']
        if 'cmap' in dobj['phi']['kwdargs'].keys():
            del dobj['phi']['kwdargs']['cmap']

    # Plot
    if plot == 'contour':
        if 'levels' in kwdargs.keys():
            lvls = kwdargs['levels']
            del kwdargs['levels']
            obj0 = dobj['bragg']['ax'].contour(xi, xj, bragg, lvls,
                                               **dobj['bragg']['kwdargs'])
            obj1 = dobj['phi']['ax'].contour(xi, xj, angle, lvls,
                                             **dobj['phi']['kwdargs'])
        else:
            obj0 = dobj['bragg']['ax'].contour(xi, xj, bragg,
                                               **dobj['bragg']['kwdargs'])
            obj1 = dobj['phi']['ax'].contour(xi, xj, angle,
                                             **dobj['phi']['kwdargs'])
    elif plot == 'imshow':
        extent = (xi.min(), xi.max(), xj.min(), xj.max())
        obj0 = dobj['bragg']['ax'].imshow(bragg, extent=extent, aspect='equal',
                                          adjustable='datalim',
                                          **dobj['bragg']['kwdargs'])
        obj1 = dobj['phi']['ax'].imshow(angle, extent=extent, aspect='equal',
                                        adjustable='datalim',
                                        **dobj['phi']['kwdargs'])
    elif plot == 'pcolor':
        obj0 = dobj['bragg']['ax'].pcolor(xi, xj, bragg,
                                          **dobj['bragg']['kwdargs'])
        obj1 = dobj['phi']['ax'].pcolor(xi, xj, angle,
                                        **dobj['phi']['kwdargs'])
    dobj['bragg']['obj'] = obj0
    dobj['phi']['obj'] = obj1

    # Post polish
    for k0 in set(dobj.keys()):
        dobj[k0]['ax'].set_xlabel(r'xi (m)')
        dobj[k0]['ax'].set_ylabel(r'xj (m)')

    if colorbar is True:
        cax0 = plt.colorbar(dobj['bragg']['obj'], ax=dobj['bragg']['ax'])
        cax1 = plt.colorbar(dobj['phi']['obj'], ax=dobj['phi']['ax'])
        cax0.ax.set_title(r'$\theta_{bragg}$' + '\n' + r'($%s$)' % braggunits)
        cax1.ax.set_title(r'$ang$' + '\n' + r'($%s$)' % angunits)

    if leg is not False:
        ax.legend(**leg)
    if wintit is not False:
        ax.figure.canvas.set_window_title(wintit)
    if tit is not False:
        ax.figure.suptitle(tit, size=10, weight='bold', ha='right')
    return ax


def CrystalBragg_plot_line_tracing_on_det(
    lamb, xi, xj, xi_err, xj_err,
    det=None, ddet=None,
    use_non_parallelism=None,
    johann=None, rocking=None,
    cryst=None, dcryst=None,
    ax=None, dleg=None,
    fs=None, dmargin=None,
    wintit=None, tit=None,
    plot=None,
):

    # Check inputs
    # ------------

    if dleg is None:
            dleg = {'loc': 'upper right', 'bbox_to_anchor': (1.05, 0.99)}
    if fs is None:
        fs = (8, 8)
    if dmargin is None:
        dmargin = {'left': 0.10, 'right': 0.95,
                   'bottom': 0.06, 'top': 0.92,
                   'wspace': None, 'hspace': 0.4}

    if wintit is None:
        wintit = _WINTIT
    if tit is None:
        tit = "line tracing"
        if johann is True:
            tit += " - johann error"
        if rocking is True:
            tit += " - rocking curve"

    plot_err = johann is True or rocking is True

    # Plot
    # ------------

    if ax is None:
        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(1, 1, **dmargin)
        ax = fig.add_subplot(gs[0, 0], aspect='equal', adjustable='datalim')
        if wintit is not False:
            fig.canvas.set_window_title(wintit)
        if tit is not False:
            fig.suptitle(tit, size=14, weight='bold')

    if det.get('outline') is not None:
        ax.plot(
            det['outline'][0, :], det['outline'][1, :],
            ls='-', lw=1., c='k',
        )

    for l in range(lamb.size):
        lab = '$\u03BB$'+' = {:6.3f}'.format(lamb[l]*1.e10)+r'$\AA$'
        alpha = cryst._dmat['alpha']*(180/np.pi)
        lab = lab+' ; $\u03B1$ ='+str(alpha)+'°'
        l0, = ax.plot(xi[l, :], xj[l, :], ls='-', lw=3., label=lab)
        if plot_err:
            ax.plot(
                xi_err[l, ...], xj_err[l, ...],
                ls='None', lw=1., c=l0.get_color(),
                ms=4, marker='.',
            )

    if dleg is not False:
        ax.legend(**dleg)

    return ax

def CrystalBragg_gap_pixels(
    lamb, lamb_unp,
    bragg, bragg_unp,
    phi, pĥi_unp,
    xi, xi_unp,
    xj, xj_unp,
    xii, xjj,
    gap_xi, gap_lamb,
    #lamb_interv, phi_interv,
    #z_plus=None, z_minus=None,
    cryst=None, dcryst=None,
    use_non_parallelism=None,
    det=None,
    split=None,
    val_phi=None, n_val_phi=None,
    ax=None, dleg=None,
    fs=None, dmargin=None,
    wintit=None, tit=None,
):

    # Check inputs
    #-------------

    if dleg is None:
        dleg = {'loc': 'upper right', 'bbox_to_anchor': (1.0, 1.0)}
    if fs is None:
        fs = (14, 10)
    if dmargin is None:
        dmargin = {'left': 0.09, 'right': 0.89,
                   'bottom': 0.06, 'top': 0.92,
                   'wspace': None, 'hspace': 0.4}

    if wintit is None:
        wintit = _WINTIT
    if tit is None:
        if split:
            tit = (
                u"Gap computed between each pixel [m], crystal splitted"
            )
        else:
            tit = u"Gap computed between each pixel [m]"

    # Plot
    #-----

    # gap = f(xi, xj)
    #----------------

    if ax is None:
        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(6, 6, **dmargin)
        ax = fig.add_subplot(gs[:, :2])
        ax1 = fig.add_subplot(gs[:, 2:])
        ax.set_xlabel('Xi [m]', fontsize=14)
        ax.set_ylabel('Xj [m]', fontsize=14)
        ax1.set_xlabel('Xi [m]', fontsize=14)
        ax1.set_ylabel('Xj [m]', fontsize=14)

    if wintit is not False:
        fig.canvas.set_window_title(wintit)
    if tit is not False:
        fig.suptitle(tit, size=12, weight='bold')

    if det.get('outline') is not None:
        ax.plot(
            det['outline'][0, :], det['outline'][1, :],
            ls='-', lw=1., c='k',
        )
        ax1.plot(
            det['outline'][0, :], det['outline'][1, :],
            ls='-', lw=1., c='k',
        )

    ax.set_title(r'$\alpha_{C1}=0 / \alpha_{C2}=+3$', fontsize=14)
    ax1.set_title(r'$\alpha_{C1}=0 / \alpha_{C2}=-3$', fontsize=14)

    extent = (
        np.min(xi), np.max(xi), np.min(xj), np.max(xj),
    )
    errmap = ax.imshow(
        gap_xi[0, ...].T,
        # keep the transposate, according to how xi_unp, xj_unp and
        # gap arrays are shaped
        cmap='viridis',
        origin='lower',
        extent=extent,
        interpolation='nearest',
        aspect='equal',
    )
    errmap1 = ax1.imshow(
        gap_xi[1, ...].T,
        cmap='viridis',
        origin='lower',
        extent=extent,
        interpolation='nearest',
        aspect='equal',
    )
    cbar = plt.colorbar(
        errmap,
        label="Gap (0/3 arcmin) [m]",
        orientation="vertical",
        ax=ax,
    )
    cbar1 = plt.colorbar(
        errmap1,
        label="Gap (0/-3 arcmin) [m]",
        orientation="vertical",
        ax=ax1,
    )

    # gap = f(lamb, phi)
    #-------------------

    fig2 = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(6, 6, **dmargin)
    ax = fig2.add_subplot(gs[:, :2])
    ax1 = fig2.add_subplot(gs[:, 4:])

    ax.set_title(r'$\alpha_{C1}=0 / \alpha_{C2}=+3$', fontsize=14)
    ax1.set_title(r'$\alpha_{C1}=0 / \alpha_{C2}=-3$', fontsize=14)

    ax.set_xlabel(r'$\lambda$ [m]', fontsize=14)
    ax1.set_xlabel(r'$\lambda$ [m]', fontsize=14)
    ax.set_ylabel('phi [rad]', fontsize=14)
    ax1.set_ylabel('phi [rad]', fontsize=14)
    ax.set_xlim(3.92*1e-10, 4.03*1e-10)
    ax1.set_xlim(3.92*1e-10, 4.03*1e-10)

    errmap = ax.scatter(
        lamb[0, ...].flatten(),
        phi[0, ...].flatten(),
        s=6,
        c=gap_lamb[0, ...].flatten(),
        cmap='viridis',
        marker='s',
        edgecolors="None",
    )
    errmap1 = ax1.scatter(
        lamb[1, ...].flatten(),
        phi[1, ...].flatten(),
        s=6,
        c=gap_lamb[1, ...].flatten(),
        cmap='viridis',
        marker='s',
        edgecolors="None",
    )
    cbar = plt.colorbar(
        errmap,
        label="Gap (0/3 arcmin) [m]",
        orientation="vertical",
        ax=ax,
    )
    cbar1 = plt.colorbar(
        errmap1,
        label="Gap (0/-3 arcmin) [m]",
        orientation="vertical",
        ax=ax1,
    )

    # Profiles
    # ------------------------------

    fig3 = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(1, 2, **dmargin)  # 3 rows, 2 columns
    ax1 = fig3.add_subplot(gs[0, 0])
    ax2 = fig3.add_subplot(gs[0, 1])

    ax1.set_title(r'Relation between $x_{i}$ coord. and $\lambda$', fontsize=14)
    ax2.set_title(r'$\phi$ ranges selected into $(\lambda,\phi)$ space', fontsize=14)

    ax1.set_xlabel('Xi [m]', fontsize=14)
    ax1.set_ylabel(r'$\lambda$ [m]', fontsize=14)
    ax2.set_xlabel(r'$\lambda$ [m]', fontsize=14)
    ax2.set_ylabel(r'$\phi$ [m]', fontsize=14)

    fig4 = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(2, 2, **dmargin)
    ax3 = fig4.add_subplot(gs[0, 0])
    ax4 = fig4.add_subplot(gs[0, 1])
    ax5 = fig4.add_subplot(gs[1, 0])
    ax6 = fig4.add_subplot(gs[1, 1])

    ax3.set_title(r'$\alpha_{C1}=0 / \alpha_{C2}=+3$', fontsize=14)
    ax4.set_title(r'$\alpha_{C1}=0 / \alpha_{C2}=-3$', fontsize=14)

    ax3.set_xlabel('Xi [m]', fontsize=14)
    ax3.set_ylabel(' Gap [m]', fontsize=14)
    ax4.set_xlabel('Xi [m]', fontsize=14)
    ax4.set_ylabel(' Gap [m]', fontsize=14)
    ax5.set_xlabel(r'$\lambda$ [m]', fontsize=14)
    ax5.set_ylabel(' Gap [m]', fontsize=14)
    ax6.set_xlabel(r'$\lambda$ [m]', fontsize=14)
    ax6.set_ylabel(' Gap [m]', fontsize=14)

    ## find indices of Phi wanted values
    def find_nearest(arr, val):
        arr = np.asarray(arr);
        ids = (np.abs(arr-val)).argmin();
        return arr[ids]

    nn = int(val_phi.size)
    nearest = np.full((nn), np.nan)
    ind_near, ix, jx = nearest.copy(), nearest.copy(), nearest.copy()
    colors = ['midnightblue', 'royalblue', 'turquoise', 'limegreen', 'gold']
    for bb in np.linspace(0, nn-1, nn):
        bb = int(bb)
        nearest[bb] = find_nearest(phi[0, 0, :], val_phi[bb])
        ind_near[bb] = phi[0, 0, :].tolist().index(nearest[bb])
        ix[bb], jx[bb] = cryst.calc_xixj_from_braggphi(
            phi=val_phi[bb],
            det=det,
            use_non_parallelism=use_non_parallelism,
            plot=False,
        )
        ax3.plot(
            xii[:, int(ind_near[bb])],
            gap_xi[0, :, int(ind_near[bb])],
            color=colors[bb],
        )
        ax4.plot(
            xii[:, int(ind_near[bb])],
            gap_xi[1, :, int(ind_near[bb])],
            label='$X_{j}$='+str(np.round(jx[bb], 3)),
            color=colors[bb],
        )
        for i in np.linspace(0, int(xi.size)-1, int(xi.size)):
            i = int(i)
            ax5.scatter(
                lamb[
                    0, i,
                    (phi[0,i]<nearest[bb]+n_val_phi)&(phi[0,i]>nearest[bb]-n_val_phi)
                ],
                gap_lamb[
                    0, i,
                    (phi[0,i]<nearest[bb]+n_val_phi)&(phi[0,i]>nearest[bb]-n_val_phi)
                ],
                marker='.',
                c=colors[bb],
            )
            ax6.scatter(
                lamb[
                    1, i,
                    (phi[1,i]<nearest[bb]+n_val_phi)&(phi[1,i]>nearest[bb]-n_val_phi)
                ],
                gap_lamb[
                    1, i,
                    (phi[1,i]<nearest[bb]+n_val_phi)&(phi[1,i]>nearest[bb]-n_val_phi)
                ],
                marker='.',
                c=colors[bb],
                label=r'$\phi$='+str(val_phi[bb]) if i == 0 else None,
            )
            ax1.scatter(
                xii[
                    i,
                    (phi[0,i]<nearest[bb]+n_val_phi)&(phi[0,i]>nearest[bb]-n_val_phi)
                ],
                lamb[
                    0, i,
                    (phi[0,i]<nearest[bb]+n_val_phi)&(phi[0,i]>nearest[bb]-n_val_phi)
                ],
                marker='.',
                c=colors[bb],
                label=r'$\phi$='+str(val_phi[bb]) if i == 0 else None,
            )
            ax2.scatter(
                lamb[
                    0, i,
                    (phi[0,i]<nearest[bb]+n_val_phi)&(phi[0,i]>nearest[bb]-n_val_phi)
                ],
                phi[
                    0, i,
                    (phi[0,i]<nearest[bb]+n_val_phi)&(phi[0,i]>nearest[bb]-n_val_phi)
                ],
                marker='.',
                c=colors[bb],
            )

    ax5.set_xlim(lamb.min()-1e-12, lamb.max()+1e-12)
    ax5.set_ylim(
        np.nanmin(gap_lamb[0,...])-1e-15, np.nanmax(gap_lamb[0,...])+1e-15,
    )
    ax6.set_xlim(lamb.min()-1e-12, lamb.max()+1e-12)
    ax6.set_ylim(
        np.nanmin(gap_lamb[1,...])-1e-15, np.nanmax(gap_lamb[1,...])+1e-15,
    )
    ax1.legend(loc='upper right', markerscale=4)
    ax4.legend(loc='best', markerscale=4)
    ax6.legend(loc='best', markerscale=4)

    ## computing analytical expressions of gaps
    """r = cryst._dgeom['rcurve']
    braggzero = cryst._dbragg['braggref']
    lambref = cryst._dbragg['lambref']
    alpha = (3/60)*(np.pi/180)
    dbragg = np.linspace(0.959-braggzero, braggzero-0.959, 487)

    def xi_gap(r, bragg, dbragg):
        return abs(r*np.sin(bragg)*(
            np.cos(bragg) - np.sin(bragg)/np.tan(bragg-dbragg)
        ))
    def diff_gap(r, bragg, dbragg, alpha):
        return abs(r*np.sin(bragg)**2*(
            1/np.tan(bragg-dbragg) - 1/np.tan(bragg-dbragg-alpha)
        ))
    y0 = xi_gap(r=r, bragg=braggzero, dbragg=dbragg)
    y1 = diff_gap(r=r, bragg=braggzero, dbragg=dbragg, alpha=alpha)
    conv = (xi.max()-xi.min())/(lamb.max()-lamb.min())
    print(conv)
    y00 = conv*xi_gap(r=r, bragg=braggzero, dbragg=dbragg)
    y11 = conv*diff_gap(r=r, bragg=braggzero, dbragg=dbragg, alpha=alpha)

    ax4.plot(xi, y0, 'k:', label=r'gap_xi($\theta, \Delta\theta$)')
    ax4.plot(xi, y1, 'r:',
        label=r'gap_xi($\theta, \Delta\theta, \alpha$)'+r'-gap_xi($\theta, \Delta\theta$)')
    ax4.legend()
    ax5.plot(xi, y0, 'k:')
    ax5.plot(xi, y1, 'r:')
    ax6.plot(xi, y00, 'k:', label=r'gap_xi($\theta, \Delta\theta$)')
    ax6.plot(xi, y11, 'r:',
        label=r'gap_xi($\theta, \Delta\theta, \alpha$)'+r'-gap_xi($\theta, \Delta\theta$)')
    ax6.legend()
    ax7.plot(xi, y00, 'k:')
    ax7.plot(xi, y11, 'r:')

    # interpolated gap = f(lamb, xj)
    #-------------------------------

    fig3 = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(6, 6, **dmargin)
    ax = fig3.add_subplot(gs[:, :2])
    ax1 = fig3.add_subplot(gs[:, 4:])

    ax.set_ylabel('phi [rad]', fontsize=14)
    ax1.set_ylabel('phi [rad]', fontsize=14)
    ax.set_xlabel(r'$\lambda$ [m]', fontsize=14)
    ax1.set_xlabel(r'$\lambda$ [m]', fontsize=14)
    ax.set_xlim(3.92*1e-10, 4.03*1e-10)
    ax1.set_xlim(3.92*1e-10, 4.03*1e-10)

    errmap = ax.pcolormesh(lamb_interv, phi_interv, z_plus, shading='flat',)
    errmap1 = ax1.pcolormesh(lamb_interv, phi_interv, z_minus, shading='flat',)

    z_plus = interp_plus(lamb_interv, phi_interv)
    z_minus = interp_minus(lamb_interv, phi_interv)

    errmap = ax.imshow(
        z_plus,
        cmap='viridis',
        origin='lower',
        extent=extent,
        interpolation='nearest',
        aspect='auto',
    )
    errmap = ax.scatter(
        lamb[0, ...].flatten()[::50],
        phi[0,...].flatten()[::50],
        s=6,
        c=z_plus.flatten(),
        cmap='viridis',
        marker='s',
        edgecolors="None",
    )
    cbar = plt.colorbar(
        errmap,
        label="Gap (0/3 arcsec) [m]",
        orientation="vertical",
        ax=ax,
    )"""

    return ax, ax1, ax2, ax3, ax4, ax5, ax6,

def CrystalBragg_gap_ray_tracing(
    lamb, gap_xi,
    xis1, xis2,
    xjs1, xjs2,
    npts, nlamb=None, n=None,
    det=None,
    split=None,
    relation=None,
    ax=None, dleg=None,
    fs=None, dmargin=None,
    wintit=None, tit=None,
):

    # Check inputs
    #-------------

    if dleg is None:
        dleg = {'loc': 'upper right', 'bbox_to_anchor': (1.0, 1.0)}
    if fs is None:
        fs = (15, 13)
    if dmargin is None:
        dmargin = {'left': 0.06, 'right': 0.99,
            'bottom': 0.06, 'top': 0.92,
            'wspace': None, 'hspace': 0.4}

    if wintit is None:
        wintit = _WINTIT
    if tit is None:
        if split:
            tit = (
                u"Gap between each wavelength arc,"
                u" with & without non-parallelism, crystal splitted"
            )
        else:
            tit = (
                u"Gap between each wavelength arc,"
                u" with & without non-parallelism"
            )

    dcolor = ['red', 'orange', 'yellow', 'green',
             'blue', 'pink', 'purple', 'brown', 'black',
             ]
    dmarkers = ['o', 'v', 's', 'x', '*', 'P', '+', 'p']
    dls = [
        '--', ':', '-.',
        '--', ':', '-.',
        '--', ':', '-.',
        '--', ':', '-.',
    ]
    dlab = [r'$\alpha_{c1}$ = 0" & $\alpha_{c2}$ = 3"',
        r'$\alpha_{c1}$ = 0" & $\alpha_{c2}$ = -3"',
    ]

    # Plot
    #-----

    if ax is None:
        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(4, 4, **dmargin)
        ax = fig.add_subplot(gs[:, 0])
        ax1 = fig.add_subplot(gs[:, 1])
        ax2 = fig.add_subplot(gs[:, 2:])

    if wintit is not False:
        fig.canvas.set_window_title(wintit)
    if tit is not False:
        fig.suptitle(tit, size=12, weight='bold')

    if det.get('outline') is not None:
        ax.plot(
            det['outline'][0, :], det['outline'][1, :],
            ls='-', lw=1., c='k',
        )
        ax1.plot(
            det['outline'][0, :], det['outline'][1, :],
            ls='-', lw=1., c='k',
        )
        ax2.plot(
            det['outline'][0, :], det['outline'][1, :],
            ls='-', lw=1., c='k',
        )

    ## Plotting ray-tracing on det (ax2) and
    ## xi's gap on each case of non-parallelism (ax & ax1)
    for l in range(nlamb):
        lab = r'$\lambda$'+' = {:6.3f} A\n'.format(lamb[l]*1.e10)
        ax.plot(
            gap_xi[1, l, :],  # plot diff between 0" and -3" [arcsec]
            xjs1[0, l, ::n],  # xj det coordinates
            ls=dls[1], lw=3.,
            c=dcolor[l],
        )
        ax1.plot(
            gap_xi[0, l, :],  # plot diff between 0" and +3" [arcsec]
            xjs1[0, l, ::n],
            ls=dls[0], lw=3.,
            c=dcolor[l],
        )
        ax2.plot(
            xis1[0, l, :], xjs1[0, l, :],  # plot ray-tracing without non-para
            ls='-', lw=3.,
            c=dcolor[l],
            label=lab
        )
        for ii in range(2):
            ax2.plot(
                xis2[ii, l, :], xjs2[ii, l, :],  # plot rays with non-para
                ls=dls[ii], lw=3.,
                c=dcolor[l],
                label=dlab[ii]
            )

        ax.set_xlim(
            np.nanmin(gap_xi[1, ...])-5e-5, np.nanmax(gap_xi[1, ...])+5e-5,
        )
        ax1.set_xlim(
            np.nanmin(gap_xi[0, ...])-5e-5, np.nanmax(gap_xi[0, ...])+5e-5,
        )
        ax.set_ylabel('Xj [m]')
        ax.set_xlabel('Gap [m]')
        ax1.set_xlabel('Gap [m]')
        ax2.set_xlabel('Xi [m]')
        ax.legend([r'$\alpha_{c1}$ = 0" & $\alpha_{c2}$ = -3"'],
                  loc='lower center')
        ax1.legend([r'$\alpha_{c1}$ = 0" & $\alpha_{c2}$ = +3"'],
                  loc='lower center')

    if dleg is not False:
        ax2.legend(**dleg)

    # plot relation between gap and wavelength, according xj positions
    if relation is True:
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 1,
            left=0.1, right=0.95,
            bottom=0.1, top=0.92,
            wspace=None, hspace=0.2,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax.set_xlabel(r'$\lambda$ [m]')
        ax.set_ylabel('Xj [m]')
        ax2.set_xlabel(r'$\lambda$ [m]')
        ax2.set_ylabel('Xj [m]')

        for aa in range(npts):
            ax.plot(
                lamb,
                gap_xi[0, :, aa],
                ls='-',#dls[aa],
                lw=1.,
                c=dcolor[0],
                ms=4,
            )
            ax2.plot(
                lamb,
                gap_xi[1, :, aa],
                ls='-',#dls[aa],
                lw=1.,
                c=dcolor[1],
                ms=4,
            )
        ax.set_xlim(
            np.nanmin(lamb)-2*1e-12, np.nanmax(lamb)+2*1e-12,
        )
        ax2.set_xlim(
            np.nanmin(lamb)-2*1e-12, np.nanmax(lamb)+2*1e-12,
        )
        ax.set_ylim(
            np.nanmin(gap_xi[0, ...])-1e-4, np.nanmax(gap_xi[0, ...])+1e-4,
        )
        ax2.set_ylim(
            np.nanmin(gap_xi[1, ...])-1e-4, np.nanmax(gap_xi[1, ...])+1e-4,
        )
        ax.set_ylabel('Gap [m]')
        ax2.set_xlabel(r'$\lambda$ [m]')
        ax2.set_ylabel('Gap [m]')
        ax.legend([r'$\alpha_{c1}$ = 0" & $\alpha_{c2}$ = 3"'])
        ax2.legend([r'$\alpha_{c1}$ = 0" & $\alpha_{c2}$ = -3"'])

    return ax, ax1, ax2


def CrystalBragg_plot_johannerror(
    xi, xj, lamb, phi, err_lamb, err_phi,
    err_lamb_units=None,
    err_phi_units=None,
    cmap=None, vmin=None, vmax=None,
    fs=None, dmargin=None, wintit=None, tit=None,
    angunits=None,
):

    # Check inputs
    # ------------

    if fs is None:
        fs = (15, 8)
    if cmap is None:
        cmap = plt.cm.viridis
    if dmargin is None:
        dmargin = {'left': 0.09, 'right': 0.90,
                   'bottom': 0.06, 'top': 0.92,
                   'wspace': None, 'hspace': 0.4}

    if angunits is None:
        angunits = 'rad'
    assert angunits in ['deg', 'rad']
    if angunits == 'deg':
        # bragg = bragg*180./np.pi
        phi = phi*180./np.pi
        err_phi = err_phi*180./np.pi
        err_phi_units = angunits

    if wintit is None:
        wintit = _WINTIT
    if tit is None:
        tit = False

    # pre-compute
    # ------------

    # extent
    extent = (xi.min(), xi.max(), xj.min(), xj.max())

    # Plot
    # ------------
    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(1, 3, **dmargin)
    ax0 = fig.add_subplot(gs[0, 0], aspect='equal')     # adjustable='datalim')
    ax1 = fig.add_subplot(
        gs[0, 1], aspect='equal', sharex=ax0, sharey=ax0,
    )
    ax2 = fig.add_subplot(
        gs[0, 2], aspect='equal', sharex=ax0, sharey=ax0,
    )

    ax0.set_title('Iso-lamb and iso-phi at crystal summit')
    ax1.set_title(f'Focalization error on lamb ({err_lamb_units})')
    ax2.set_title(f'Focalization error on phi ({err_phi_units})')
    ax0.contour(xi, xj, lamb.T, 10, cmap=cmap)
    ax0.contour(xi, xj, phi.T, 10, cmap=cmap, ls='--')
    imlamb = ax1.imshow(
        err_lamb.T,
        extent=extent, aspect='equal',
        origin='lower', interpolation='nearest',
        vmin=vmin, vmax=vmax,
    )
    imphi = ax2.imshow(
        err_phi.T,
        extent=extent, aspect='equal',
        origin='lower', interpolation='nearest',
        vmin=vmin, vmax=vmax,
    )

    plt.colorbar(imlamb, ax=ax1)
    plt.colorbar(imphi, ax=ax2)

    if wintit is not False:
        fig.canvas.set_window_title(wintit)
    if tit is not False:
        fig.suptitle(tit, size=14, weight='bold')

    return [ax0, ax1, ax2]


def CrystalBragg_plot_focal_error_summed(
    cryst=None, dcryst=None,
    lamb=None, bragg=None,
    error_lambda=None,
    ddist=None, di=None,
    ddist0=None, di0=None, dj0=None,
    dtheta0=None, dpsi0=None, tilt0=None,
    angle_nout=None,
    det_ref=None,
    units=None,
    plot_dets=None, nsort=None,
    tangent_to_rowland=None,
    use_non_parallelism=None,
    pts=None,
    test_lamb_interv=None,
    contour=None,
    fs=None,
    cmap=None,
    vmin=None,
    vmax=None,
    ax=None,
):

    if cmap is None:
        # cmap = 'RdYlBu'
        cmap = plt.cm.viridis
    if nsort is None:
        nsort = 5
    if contour is None:
        errmin = np.nanmin(error_lambda)
        contour = [errmin + (np.nanmax(error_lambda) - errmin)/50.]
    if fs is None:
        fs = (6, 8)

    if ax is None:
        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.set_title('Mean focalization error\non detector')
        ax.set_xlabel('ddist (m)')
        ax.set_ylabel('di (m)')

    # plot error map function(ddist, di)
    extent = (ddist.min(), ddist.max(), di.min(), di.max())
    errmap = ax.imshow(
        error_lambda,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        extent=extent,
        interpolation='nearest',
    )
    cbar = plt.colorbar(
        errmap,
        label=f"error on lambda ({units})",
        orientation="vertical",
    )
    ax.contour(
        ddist,
        di,
        error_lambda,
        contour,
        colors='w',
        linewstyles='-',
        linewidths=1.,
    )
    ax.contour(
        ddist,
        di,
        test_lamb_interv,
        contour,
        colors='yellow',
        linewstyles='-',
        linewidths=1.,
    )

    # Computing detector with exact position of det_ref
    if det_ref:
        dpsi0bis = float(dpsi0)
        if tangent_to_rowland:
            dpsi0bis = dpsi0 - angle_nout

        detector_comp = cryst.get_detector_approx(
            ddist=ddist0,
            di=di0,
            dj=dj0,
            dtheta=dtheta0,
            dpsi=dpsi0bis,
            tilt=tilt0,
            lamb=lamb,
            use_non_parallelism=use_non_parallelism,
            tangent_to_rowland=False,
        )
        detector_comp['outline'] = det_ref['outline']
        ax.plot(
            ddist0,
            di0,
            marker='x',
            ls='None',
            color='w',
        )

    if plot_dets:
        indsort = np.argsort(np.ravel(error_lambda))
        inddist = indsort % ddist.size
        inddi = indsort // ddist.size

        # plot nbr of dets on map "mean focalization error = f(ddist, di)"
        ax.plot(
            ddist[inddist[:nsort]],
            di[inddi[:nsort]],
            marker='x',
            ls='None',
            color='r',
        )

        # plot dets geometry with CrystalBragg_plot()
        if det_ref is not None:
            dax = cryst.plot(
                det=det_ref,
                pts=pts,
                color='black',
            )
            dax = cryst.plot(
                det=detector_comp,
                pts=pts,
                color='blue',
                element='ocv',
                dax=dax,
            )
            msg = (
                "Parameters of reference detector:\n"
                + "Center position in (x, y, z): ({})\n".format(
                    np.round(detector_comp['cent'], decimals=4)
                    )
                + "Translations (ddist, di, dj): ({}, {}, {}) [m]\n".format(
                    ddist0, di0, dj0,
                    )
                + "Rotations (dtheta, dpsi, tilt): ({}, {}, {}) [rad]\n".format(
                    dtheta0, dpsi0, tilt0,
                    )
            )
            print(msg)
            det = {}
            for ii in range(nsort):
                det[ii] = cryst.get_detector_approx(
                    ddist=ddist[inddist[ii]],
                    di=di[inddi[ii]],
                    tangent_to_rowland=tangent_to_rowland,
                )
                det[ii]['outline'] = det_ref['outline']
                dax = cryst.plot(
                    det=det[ii],
                    pts=pts,
                    color='red',
                    element='oc',
                    dax=dax,
                )
                print(
                    "det: {}\n".format(det[ii])
                    + "\t ddist: {}\n".format(ddist[inddist[ii]])
                    + "\t di: {}\n".format(di[inddi[ii]])
                )
    return ax


# #################################################################
# #################################################################
#                   Ray tracing plot
# #################################################################
# #################################################################


# To be clarified
def CrystalBragg_plot_raytracing_from_lambpts(xi=None, xj=None, lamb=None,
                                              xi_bounds=None, xj_bounds=None,
                                              pts=None, ptscryst=None,
                                              ptsdet=None,
                                              det_cent=None, det_nout=None,
                                              det_ei=None, det_ej=None,
                                              cryst=None, proj=None,
                                              fs=None, ax=None, dmargin=None,
                                              wintit=None, tit=None,
                                              legend=None, draw=None):
    # Check
    assert xi.shape == xj.shape and xi.ndim == 3
    assert (isinstance(proj, list)
            and all([pp in ['det', '2d', '3d'] for pp in proj]))
    if legend is None or legend is True:
        legend = dict(bbox_to_anchor=(1.02, 1.), loc='upper left',
                      ncol=1, mode="expand", borderaxespad=0.,
                      prop={'size': 6})
    if wintit is None:
        wintit = _WINTIT
    if draw is None:
        draw = True

    # Prepare
    nlamb, npts, ndtheta = xi.shape
    det = np.array([[xi_bounds[0], xi_bounds[1], xi_bounds[1],
                     xi_bounds[0], xi_bounds[0]],
                    [xj_bounds[0], xj_bounds[0], xj_bounds[1],
                     xj_bounds[1], xj_bounds[0]]])
    lcol = ['r', 'g', 'b', 'm', 'y', 'c']
    lm = ['+', 'o', 'x', 's']
    lls = ['-', '--', ':', '-.']
    ncol, nm, nls = len(lcol), len(lm), len(lls)

    if '2d' in proj or '3d' in proj:
        pts = np.repeat(np.repeat(pts[:, None, :], nlamb, axis=1)[..., None],
                        ndtheta, axis=-1)[..., None]
        ptsall = np.concatenate(
            (
                pts,
                ptscryst[..., None],
                ptsdet[..., None],
                np.full((3, nlamb, npts, ndtheta, 1), np.nan),
            ),
            axis=-1,
        ).reshape((3, nlamb, npts, ndtheta*4))
        del pts, ptscryst, ptsdet
        if '2d' in proj:
            R = np.hypot(ptsall[0, ...], ptsall[1, ...])

    # --------
    # Plot
    lax = []
    if 'det' in proj:

        # Prepare
        if ax is None:
            if fs is None:
                fsi = (8, 6)
            else:
                fsi = fs
            if dmargin is None:
                dmargini = {'left': 0.1, 'right': 0.8,
                            'bottom': 0.1, 'top': 0.9,
                            'wspace': None, 'hspace': 0.4}
            else:
                dmargini = dmargin
            if tit is None:
                titi = False
            else:
                titi = tit
            fig = plt.figure(figsize=fsi)
            gs = gridspec.GridSpec(1, 1, **dmargini)
            axi = fig.add_subplot(
                gs[0, 0], aspect='equal', adjustable='datalim',
            )
            axi.set_xlabel(r'$x_i$ (m)')
            axi.set_ylabel(r'$x_j$ (m)')
        else:
            axi = ax

        # plot
        axi.plot(det[0, :], det[1, :], ls='-', lw=1., c='k')
        for pp in range(npts):
            for ll in range(nlamb):
                lab = (
                    r'pts {} - '.format(pp)
                    + r'$\lambda$' + ' = {:6.3f} A'.format(lamb[ll]*1.e10)
                )
                axi.plot(
                    xi[ll, pp, :], xj[ll, pp, :],
                    ls='None', marker=lm[ll % nm],
                    c=lcol[pp % ncol], label=lab,
                )

        # decorate
        if legend is not False:
            axi.legend(**legend)
        if wintit is not False:
            axi.figure.canvas.set_window_title(wintit)
        if titi is not False:
            axi.figure.suptitle(titi, size=14, weight='bold')
        if draw:
            axi.figure.canvas.draw()
        lax.append(axi)

    if '2d' in proj:

        # Prepare
        if tit is None:
            titi = False
        else:
            titi = tit

        # plot
        dax = cryst.plot(lax=ax, proj='all',
                         det_cent=det_cent, det_nout=det_nout,
                         det_ei=det_ei, det_ej=det_ej, draw=False)
        for pp in range(npts):
            for ll in range(nlamb):
                lab = (r'pts {} - '.format(pp)
                       + r'$\lambda$'+' = {:6.3f} A'.format(lamb[ll]*1.e10))
                dax['cross'].plot(
                    R[ll, pp, :], ptsall[2, ll, pp, :],
                    ls=lls[ll % nls], color=lcol[pp % ncol],
                    label=lab,
                )
                dax['hor'].plot(
                    ptsall[0, ll, pp, :], ptsall[1, ll, pp, :],
                    ls=lls[ll % nls], color=lcol[pp % ncol], label=lab,
                )
        # decorate
        if legend is not False:
            dax['cross'].legend(**legend)
        if wintit is not False:
            dax['cross'].figure.canvas.set_window_title(wintit)
        if titi is not False:
            dax['cross'].figure.suptitle(titi, size=14, weight='bold')
        if draw:
            dax['cross'].figure.canvas.draw()
        lax.append(dax['cross'])
        lax.append(dax['hor'])

    return lax
