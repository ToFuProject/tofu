

# Built-in
import itertools as itt

# Common
import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
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
_QUIVERCOLOR = ListedColormap(_QUIVERCOLOR)


# Generic
def _check_projdax_mpl(dax=None, proj=None,
                       dmargin=None, fs=None, wintit=None):

    # ----------------------
    # Check inputs
    if proj is None:
        proj = 'all'
    assert isinstance(proj, str)
    proj = proj.lower()
    lproj = ['cross', 'hor', '3d']
    assert proj in lproj + ['all']
    if proj == 'all':
        proj = ['cross', 'hor']
    else:
        proj = [proj]

    # ----------------------
    # Check dax
    lc = [dax is None,
          issubclass(dax.__class__, Axes),
          isinstance(dax, dict),
          isinstance(dax, list)]
    assert any(lc)
    if lc[0]:
        dax = dict.fromkeys(proj)
    elif lc[1]:
        assert len(proj) == 1
        dax = {proj[0]: dax}
    elif lc[2]:
        lcax = [dax.get(pp) is None or issubclass(dax.get(pp).__class__, Axes)
                for pp in proj]
        if not all(lcax):
            msg = "Wrong key or axes in dax:\n"
            msg += "    - proj = %s"%str(proj)
            msg += "    - dax = %s"%str(dax)
            raise Exception(msg)
    else:
        assert len(dax) == 2
        assert all([ax is None or issubclass(ax.__class__, Axes)
                    for ax in dax])
        dax = {'cross': dax[0], 'hor': dax[1]}

    # Populate with default axes if necessary
    if 'cross' in proj and 'hor' in proj:
        if dax['cross'] is None:
            assert dax['hor'] is None
            lax = _def.Plot_LOSProj_DefAxes('all', fs=fs,
                                            dmargin=dmargin,
                                            wintit=wintit)
            dax['cross'], dax['hor'] = lax
    elif 'cross' in proj and dax['cross'] is None:
        dax['cross'] = _def.Plot_LOSProj_DefAxes('cross', fs=fs,
                                                 dmargin=dmargin,
                                                 wintit=wintit)
    elif 'hor' in proj and dax['hor'] is None:
        dax['hor'] = _def.Plot_LOSProj_DefAxes('hor', fs=fs,
                                               dmargin=dmargin,
                                               wintit=wintit)
    elif '3d' in proj  and dax['3d'] is None:
        dax['3d'] = _def.Plot_3D_plt_Tor_DefAxes(fs=fs,
                                                 dmargin=dmargin,
                                                 wintit=wintit)
    for kk in lproj:
        dax[kk] = dax.get(kk, None)
    return dax



# #################################################################
# #################################################################
#                   Generic geometry plot
# #################################################################
# #################################################################

def CrystalBragg_plot(cryst=None, dax=None, proj=None, res=None, element=None,
                      color=None, dP=None,
                      det_cent=None, det_nout=None,
                      det_ei=None, det_ej=None, det_cont=None,
                      pts0=None, pts1=None, rays_color=None, rays_npts=None,
                      dI=None, dBs=None, dBv=None,
                      dVect=None, dIHor=None, dBsHor=None, dBvHor=None,
                      dleg=None, indices=False,
                      draw=True, fs=None, dmargin=None,
                      wintit=None, tit=None, Test=True):

    # ---------------------
    # Check / format inputs

    if Test:
        msg = "Arg proj must be in ['cross','hor','all','3d'] !"
        assert type(draw) is bool, "Arg draw must be a bool !"
        assert cryst is None or cryst.__class__.__name__ == 'CrystalBragg'
    if wintit is None:
        wintit = _WINTIT
    if dleg is None:
         dleg = _def.TorLegd

    # ---------------------
    # call plotting functions

    kwa = dict(fs=fs, wintit=wintit, Test=Test)
    if proj == '3d':
        # Temporary matplotlib issue
        dax = _CrystalBragg_plot_3d(
            cryst=cryst, proj=proj, res=res, dax=dax, element=element,
            color=color, det_cent=det_cent,
            det_nout=det_nout, det_cont=det_cont,
            pts0=pts0, pts1=pts1, rays_color=rays_color, rays_npts=rays_npts,
            det_ei=det_ei, det_ej=det_ej, draw=draw,
            dmargin=dmargin, fs=fs, wintit=wintit)
    else:
        dax = _CrystalBragg_plot_crosshor(
            cryst=cryst, proj=proj, res=res, dax=dax, element=element,
            color=color, det_cent=det_cent,
            det_nout=det_nout, det_cont=det_cont,
            pts0=pts0, pts1=pts1, rays_color=rays_color, rays_npts=rays_npts,
            det_ei=det_ei, det_ej=det_ej, draw=draw,
            dmargin=dmargin, fs=fs, wintit=wintit)

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


def _CrystalBragg_plot_crosshor(cryst=None, proj=None, dax=None,
                                element=None, res=None,
                                det_cent=None, det_nout=None,
                                det_ei=None, det_ej=None, det_cont=None,
                                pts0=None, pts1=None,
                                rays_color=None, rays_npts=None,
                                Pdict=_def.TorPd, Idict=_def.TorId,
                                Bsdict=_def.TorBsd, Bvdict=_def.TorBvd,
                                Vdict=_def.TorVind, color=None, ms=None,
                                quiver_cmap=None,  LegDict=_def.TorLegd,
                                indices=False, draw=True,
                                dmargin=None, fs=None, wintit=None, Test=True):
    if Test:
        assert type(Pdict) is dict, 'Arg Pdict should be a dictionary !'
        assert type(Idict) is dict, "Arg Idict should be a dictionary !"
        assert type(Bsdict) is dict, "Arg Bsdict should be a dictionary !"
        assert type(Bvdict) is dict, "Arg Bvdict should be a dictionary !"
        assert type(Vdict) is dict, "Arg Vdict should be a dictionary !"
        msg = 'Arg LegDict should be a dictionary !'
        assert type(LegDict) is dict or LegDict is None, msg

    # ---------------------
    # Check / format inputs

    lelement = ['s', 'c', 'r', 'o', 'v']
    if element is None:
        element = 'oscvr'
    element = element.lower()
    if 'v' in element and quiver_cmap is None:
        quiver_cmap = _QUIVERCOLOR
    if color is None:
        if cryst is not None and cryst._dmisc.get('color') is not None:
            color = cryst._dmisc['color']
        else:
            color = 'k'
    if ms is None:
        ms = 6

    lc = [ss in element for ss in lelement]
    if any(lc) and cryst is None:
        msg = ("cryst cannot be None if element contains any of:\n"
               + "\t- {}".format(lelement))
        raise Exception(msg)

    lc = [pts0 is None, pts1 is None]
    c0 = (np.sum(lc) == 1
          or (not any(lc)
              and (pts0.shape != pts1.shape or pts0.shape[0] != 3)))
    if c0:
        msg = ("pts0 and pts1 must be:\n"
               + "\t- both None\n"
               + "\t- both np.ndarray of same shape, with shape[0] == 3\n"
               + "  You provided:\n"
               + "\t- pts0: {}\n".format(pts0)
               + "\t- pts1: {}".format(pts1))
        raise Exception(msg)
    if pts0 is not None:
        if rays_color is None:
            rays_color = 'k'
        if rays_npts is None:
            rays_npts = 10

    # ---------------------
    # Prepare axe and data

    dax = _check_projdax_mpl(dax=dax, proj=proj,
                             dmargin=dmargin, fs=fs, wintit=wintit)

    if 's' in element or 'v' in element:
        summ = cryst._dgeom['summit']
    if 'c' in element:
        cent = cryst._dgeom['center']
    if 'r' in element:
        ang = np.linspace(0, 2.*np.pi, 200)
        rr = 0.5*cryst._dgeom['rcurve']
        row = cryst._dgeom['center'] + rr*cryst._dgeom['nout']
        row = (row[:, None]
               + rr*(np.cos(ang)[None, :]*cryst._dgeom['nout'][:, None]
                     + np.sin(ang)[None, :]*cryst._dgeom['e1'][:, None]))

    # ---------------------
    # plot

    if 'o' in element:
        cont = cryst.sample_outline_plot(res=res)
        if dax['cross'] is not None:
            dax['cross'].plot(np.hypot(cont[0,:], cont[1,:]), cont[2,:],
                              ls='-', c=color, marker='None',
                              label=cryst.Id.NameLTX+' contour')
        if dax['hor'] is not None:
            dax['hor'].plot(cont[0,:], cont[1,:],
                            ls='-', c=color, marker='None',
                            label=cryst.Id.NameLTX+' contour')
    if 's' in element:
        if dax['cross'] is not None:
            dax['cross'].plot(np.hypot(summ[0], summ[1]), summ[2],
                              marker='^', ms=ms, c=color,
                              label=cryst.Id.NameLTX+" summit")
        if dax['hor'] is not None:
            dax['hor'].plot(summ[0], summ[1],
                            marker='^', ms=ms, c=color,
                            label=cryst.Id.NameLTX+" summit")
    if 'c' in element:
        if dax['cross'] is not None:
            dax['cross'].plot(np.hypot(cent[0], cent[1]), cent[2],
                              marker='o', ms=ms, c=color,
                              label=cryst.Id.NameLTX+" center")
        if dax['hor'] is not None:
            dax['hor'].plot(cent[0], cent[1],
                            marker='o', ms=ms, c=color,
                            label=cryst.Id.NameLTX+" center")
    if 'r' in element:
        if dax['cross'] is not None:
            dax['cross'].plot(np.hypot(row[0,:], row[1,:]), row[2,:],
                              ls='--', color=color, marker='None',
                              label=cryst.Id.NameLTX+' rowland')
        if dax['hor'] is not None:
            dax['hor'].plot(row[0,:], row[1,:],
                            ls='--', color=color, marker='None',
                            label=cryst.Id.NameLTX+' rowland')
    if 'v' in element:
        nin = cryst._dgeom['nin']
        e1, e2 = cryst._dgeom['e1'], cryst._dgeom['e2']
        p0 = np.repeat(summ[:,None], 3, axis=1)
        v = np.concatenate((nin[:, None], e1[:, None], e2[:, None]), axis=1)
        if dax['cross'] is not None:
            pr = np.hypot(p0[0, :], p0[1, :])
            vr = np.hypot(p0[0, :]+v[0, :], p0[1, :]+v[1, :]) - pr
            dax['cross'].quiver(pr, p0[2, :],
                                vr, v[2, :],
                                np.r_[0., 0.5, 1.], cmap=quiver_cmap,
                                angles='xy', scale_units='xy',
                                label=cryst.Id.NameLTX+" unit vect", **Vdict)
        if dax['hor'] is not None:
            dax['hor'].quiver(p0[0, :], p0[1, :],
                              v[0, :], v[1, :],
                              np.r_[0., 0.5, 1.], cmap=quiver_cmap,
                              angles='xy', scale_units='xy',
                              label=cryst.Id.NameLTX+" unit vect", **Vdict)

    # -------------
    # Detector
    sc = None
    if det_cent is not None and 'c' in element:
        if dax['cross'] is not None:
            dax['cross'].plot(np.hypot(det_cent[0], det_cent[1]), det_cent[2],
                              marker='x', ms=ms, c=color, label="det_cent")
        if dax['hor'] is not None:
            dax['hor'].plot(det_cent[0], det_cent[1],
                            marker='x', ms=ms, c=color, label="det_cent")

    if det_nout is not None and 'v' in element:
        assert det_ei is not None and det_ej is not None
        p0 = np.repeat(det_cent[:, None], 3, axis=1)
        v = np.concatenate((det_nout[:, None], det_ei[:, None],
                            det_ej[:, None]), axis=1)
        if dax['cross'] is not None:
            pr = np.hypot(p0[0, :], p0[1, :])
            vr = np.hypot(p0[0, :]+v[0, :], p0[1, :]+v[1, :]) - pr
            dax['cross'].quiver(pr, p0[2, :],
                                vr, v[2, :],
                                np.r_[0., 0.5, 1.], cmap=quiver_cmap,
                                angles='xy', scale_units='xy',
                                label="det unit vect", **Vdict)
        if dax['hor'] is not None:
            dax['hor'].quiver(p0[0, :], p0[1, :],
                              v[0, :], v[1, :],
                              np.r_[0., 0.5, 1.], cmap=quiver_cmap,
                              angles='xy', scale_units='xy',
                              label="det unit vect", **Vdict)

    if det_cont is not None and 'c' in element:
        det_cont = (det_cont[0:1, :]*det_ei[:, None]
                    + det_cont[1:2, :]*det_ej[:, None]) + det_cent[:, None]
        if dax['cross'] is not None:
            dax['cross'].plot(np.hypot(det_cont[0, :], det_cont[1, :]),
                              det_cont[2, :],
                              ls='-', c=color, marker='None',
                              label='det contour')
        if dax['hor'] is not None:
            dax['hor'].plot(det_cont[0, :],
                            det_cont[1, :],
                            ls='-', c=color, marker='None',
                            label='det contour')

    # -------------
    # pts0 and pts1
    if pts0 is not None:
        if pts0.ndim == 3:
            pts0 = np.reshape(pts0, (3, pts0.shape[1]*pts0.shape[2]))
            pts1 = np.reshape(pts1, (3, pts1.shape[1]*pts1.shape[2]))
        if dax['cross'] is not None:
            k = np.r_[np.linspace(0, 1, rays_npts), np.nan]
            pts01 = np.reshape((pts0[:, :, None]
                                + k[None, None, :]*(pts1-pts0)[:, :, None]),
                               (3, pts0.shape[1]*(rays_npts+1)))
            linesr = np.hypot(pts01[0, :], pts01[1, :])
            dax['cross'].plot(linesr, pts01[2, :],
                              color=rays_color, lw=1., ls='-')
        if dax['hor'] is not None:
            k = np.r_[0, 1, np.nan]
            pts01 = np.reshape((pts0[:2, :, None]
                                + k[None, None, :]*(pts1-pts0)[:2, :, None]),
                               (2, pts0.shape[1]*3))
            dax['hor'].plot(pts01[0, :], pts01[1, :],
                            color=rays_color, lw=1., ls='-')
    return dax


def _CrystalBragg_plot_3d(cryst=None, proj=None, dax=None,
                          element=None, res=None,
                          det_cent=None, det_nout=None,
                          det_ei=None, det_ej=None, det_cont=None,
                          pts0=None, pts1=None,
                          rays_color=None, rays_npts=None,
                          Pdict=_def.TorPd, Idict=_def.TorId,
                          Bsdict=_def.TorBsd, Bvdict=_def.TorBvd,
                          Vdict=_def.TorVind, color=None, ms=None,
                          quiver_cmap=None,  LegDict=_def.TorLegd,
                          indices=False, draw=True,
                          dmargin=None, fs=None, wintit=None, Test=True):
    if Test:
        assert type(Pdict) is dict, 'Arg Pdict should be a dictionary !'
        assert type(Idict) is dict, "Arg Idict should be a dictionary !"
        assert type(Bsdict) is dict, "Arg Bsdict should be a dictionary !"
        assert type(Bvdict) is dict, "Arg Bvdict should be a dictionary !"
        assert type(Vdict) is dict, "Arg Vdict should be a dictionary !"
        msg = 'Arg LegDict should be a dictionary !'
        assert type(LegDict) is dict or LegDict is None, msg

    # ---------------------
    # Check / format inputs

    lelement = ['s', 'c', 'r', 'o', 'v']
    if element is None:
        element = 'oscvr'
    element = element.lower()
    if 'v' in element and quiver_cmap is None:
        quiver_cmap = _QUIVERCOLOR
    if color is None:
        if cryst is not None and cryst._dmisc.get('color') is not None:
            color = cryst._dmisc['color']
        else:
            color = 'k'
    if ms is None:
        ms = 6

    lc = [ss in element for ss in lelement]
    if any(lc) and cryst is None:
        msg = ("cryst cannot be None if element contains any of:\n"
               + "\t- {}".format(lelement))
        raise Exception(msg)

    lc = [pts0 is None, pts1 is None]
    c0 = (np.sum(lc) == 1
          or (not any(lc)
              and (pts0.shape != pts1.shape or pts0.shape[0] != 3)))
    if c0:
        msg = ("pts0 and pts1 must be:\n"
               + "\t- both None\n"
               + "\t- both np.ndarray of same shape, with shape[0] == 3\n"
               + "  You provided:\n"
               + "\t- pts0: {}\n".format(pts0)
               + "\t- pts1: {}".format(pts1))
        raise Exception(msg)
    if pts0 is not None:
        if rays_color is None:
            rays_color = 'k'
        if rays_npts is None:
            rays_npts = 10

    # ---------------------
    # Prepare axe and data

    dax = _check_projdax_mpl(dax=dax, proj=proj,
                             dmargin=dmargin, fs=fs, wintit=wintit)

    if 's' in element or 'v' in element:
        summ = cryst._dgeom['summit']
    if 'c' in element:
        cent = cryst._dgeom['center']
    if 'r' in element:
        ang = np.linspace(0, 2.*np.pi, 200)
        rr = 0.5*cryst._dgeom['rcurve']
        row = cryst._dgeom['center'] + rr*cryst._dgeom['nout']
        row = (row[:, None]
               + rr*(np.cos(ang)[None, :]*cryst._dgeom['nout'][:, None]
                     + np.sin(ang)[None, :]*cryst._dgeom['e1'][:, None]))

    # ---------------------
    # plot

    if 'o' in element:
        cont = cryst.sample_outline_plot(res=res)
        if dax['3d'] is not None:
            dax['3d'].plot(cont[0, :], cont[1, :], cont[2, :],
                           ls='-', c=color, marker='None',
                           label=cryst.Id.NameLTX+' contour')
    if 's' in element:
        if dax['3d'] is not None:
            dax['3d'].plot(summ[0:1], summ[1:2], summ[2:3],
                           marker='^', ms=ms, c=color,
                           label=cryst.Id.NameLTX+" summit")
    if 'c' in element:
        if dax['3d'] is not None:
            dax['3d'].plot(cent[0:1], cent[1:2], cent[2:3],
                           marker='o', ms=ms, c=color,
                           label=cryst.Id.NameLTX+" center")
    if 'r' in element:
        if dax['3d'] is not None:
            dax['3d'].plot(row[0, :], row[1, :], row[2, :],
                           ls='--', color=color, marker='None',
                           label=cryst.Id.NameLTX+' rowland')
    if 'v' in element:
        nin = cryst._dgeom['nin']
        e1, e2 = cryst._dgeom['e1'], cryst._dgeom['e2']
        p0 = np.repeat(summ[:, None], 3, axis=1)
        v = np.concatenate((nin[:, None], e1[:, None], e2[:, None]), axis=1)
        if dax['3d'] is not None:
            dax['3d'].quiver(p0[0, :], p0[1, :], p0[2, :],
                             v[0, :], v[1, :], v[2, :],
                             np.r_[0., 0.5, 1.], cmap=quiver_cmap,
                             label=cryst.Id.NameLTX+" unit vect", color='r')
            #, **Vdict)
            # angles='xy', scale_units='xy',

    # -------------
    # Detector
    sc = None
    if det_cent is not None and 'c' in element:
        if dax['3d'] is not None:
            dax['3d'].plot(det_cent[0:1], det_cent[1:2], det_cent[2:3],
                           marker='x', ms=ms, c=color, label="det_cent")

    if det_nout is not None and 'v' in element:
        assert det_ei is not None and det_ej is not None
        p0 = np.repeat(det_cent[:, None], 3, axis=1)
        v = np.concatenate((det_nout[:, None], det_ei[:, None],
                            det_ej[:, None]), axis=1)
        if dax['3d'] is not None:
            dax['3d'].quiver(p0[0, :], p0[1, :], p0[2, :],
                             v[0, :], v[1, :], v[2, :],
                             np.r_[0., 0.5, 1.], cmap=quiver_cmap,
                             label="det unit vect", color='r')
            #, **Vdict)
            # angles='xy', scale_units='xy',
    if det_cont is not None and 'o' in element:
        det_cont = (det_cont[0:1, :]*det_ei[:, None]
                    + det_cont[1:2, :]*det_ej[:, None]) + det_cent[:, None]
        if dax['3d'] is not None:
            dax['3d'].plot(det_cont[0, :],
                           det_cont[1, :],
                           det_cont[2, :],
                           ls='-', c=color, marker='None',
                           label='det contour')

    # -------------
    # pts0 and pts1
    if pts0 is not None:
        if pts0.ndim == 3:
            pts0 = np.reshape(pts0, (3, pts0.shape[1]*pts0.shape[2]))
            pts1 = np.reshape(pts1, (3, pts1.shape[1]*pts1.shape[2]))
        if dax['3d'] is not None:
            k = np.r_[0, 1, np.nan]
            pts01 = np.reshape((pts0[:, :, None]
                                + k[None, None, :]*(pts1-pts0)[:, :, None]),
                               (3, pts0.shape[1]*3))
            dax['3d'].plot(pts01[0, :], pts01[1, :], pts01[2, :],
                           color=rays_color, lw=1., ls='-')
    return dax


# #################################################################
# #################################################################
#                   Rocking curve plot
# #################################################################
# #################################################################

def CrystalBragg_plot_rockingcurve(func=None, bragg=None, lamb=None,
                                   sigma=None, npts=None,
                                   ang_units=None, axtit=None,
                                   color=None,
                                   legend=None, fs=None, ax=None):

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
        cax0.ax.set_title(r'$\theta_{bragg}$' + '\n' + r'($%s$)'%braggunits)
        cax1.ax.set_title(r'$ang$' + '\n' + r'($%s$)'%angunits)

    if leg is not False:
        ax.legend(**leg)
    if wintit is not False:
        ax.figure.canvas.set_window_title(wintit)
    if tit is not False:
        ax.figure.suptitle(tit, size=14, weight='bold')
    return ax


def CrystalBragg_plot_line_tracing_on_det(lamb, xi, xj, xi_err, xj_err,
                                          det=None,
                                          johann=None, rocking=None,
                                          fs=None, dmargin=None,
                                          wintit=None, tit=None):

    # Check inputs
    # ------------

    if fs is None:
        fs = (6, 8)
    if dmargin is None:
        dmargin = {'left': 0.05, 'right': 0.99,
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

    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(1, 1, **dmargin)
    ax0 = fig.add_subplot(gs[0, 0], aspect='equal', adjustable='datalim')

    ax0.plot(det[0, :], det[1, :], ls='-', lw=1., c='k')
    for l in range(lamb.size):
        lab = r'$\lambda$'+' = {:6.3f} A'.format(lamb[l]*1.e10)
        l0, = ax0.plot(xi[l, :], xj[l, :], ls='-', lw=1., label=lab)
        if plot_err:
            ax0.plot(xi_err[l, ...], xj_err[l, ...],
                     ls='None', lw=1., c=l0.get_color(),
                     marker='.', ms=1)

    ax0.legend()

    if wintit is not False:
        fig.canvas.set_window_title(wintit)
    if tit is not False:
        fig.suptitle(tit, size=14, weight='bold')
    return [ax0]


def CrystalBragg_plot_johannerror(xi, xj, lamb, phi, err_lamb, err_phi,
                                  cmap=None, vmin=None, vmax=None,
                                  fs=None, dmargin=None, wintit=None, tit=None,
                                  angunits='deg', err=None):

    # Check inputs
    # ------------

    if fs is None:
        fs = (14, 8)
    if cmap is None:
        cmap = plt.cm.viridis
    if dmargin is None:
        dmargin = {'left': 0.05, 'right': 0.99,
                   'bottom': 0.06, 'top': 0.92,
                   'wspace': None, 'hspace': 0.4}
    assert angunits in ['deg', 'rad']
    if angunits == 'deg':
        # bragg = bragg*180./np.pi
        phi = phi*180./np.pi
        err_phi = err_phi*180./np.pi

    if err is None:
        err = 'abs'
    if err == 'rel':
        err_lamb = 100.*err_lamb / (np.nanmax(lamb) - np.nanmin(lamb))
        err_phi = 100.*err_phi / (np.nanmax(phi) - np.nanmin(phi))
        err_lamb_units = '%'
        err_phi_units = '%'
    else:
        err_lamb_units = 'm'
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

    fig = fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(1, 3, **dmargin)
    ax0 = fig.add_subplot(gs[0, 0], aspect='equal', adjustable='datalim')
    ax1 = fig.add_subplot(gs[0, 1], aspect='equal', adjustable='datalim',
                          sharex=ax0, sharey=ax0)
    ax2 = fig.add_subplot(gs[0, 2], aspect='equal', adjustable='datalim',
                          sharex=ax0, sharey=ax0)

    ax0.set_title('Iso-lamb and iso-phi at crystal summit')
    ax1.set_title('Focalization error on lamb ({})'.format(err_lamb_units))
    ax2.set_title('Focalization error on phi ({})'.format(err_phi_units))

    ax0.contour(xi, xj, lamb, 10, cmap=cmap)
    ax0.contour(xi, xj, phi, 10, cmap=cmap, ls='--')
    imlamb = ax1.imshow(err_lamb, extent=extent, aspect='equal',
                        origin='lower', interpolation='nearest',
                        vmin=vmin, vmax=vmax)
    imphi = ax2.imshow(err_phi, extent=extent, aspect='equal',
                       origin='lower', interpolation='nearest',
                       vmin=vmin, vmax=vmax)

    plt.colorbar(imlamb, ax=ax1)
    plt.colorbar(imphi, ax=ax2)
    if wintit is not False:
        fig.canvas.set_window_title(wintit)
    if tit is not False:
        fig.suptitle(tit, size=14, weight='bold')

    return [ax0, ax1, ax2]



# #################################################################
# #################################################################
#                   Ray tracing plot
# #################################################################
# #################################################################

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
        ptsall = np.concatenate((pts,
                                 ptscryst[..., None],
                                 ptsdet[..., None],
                                 np.full((3, nlamb, npts, ndtheta, 1), np.nan)),
                                axis=-1).reshape((3, nlamb, npts, ndtheta*4))
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
            axi = fig.add_subplot(gs[0, 0], aspect='equal', adjustable='datalim')
            axi.set_xlabel(r'$x_i$ (m)')
            axi.set_ylabel(r'$x_j$ (m)')
        else:
            axi = ax

        # plot
        axi.plot(det[0, :], det[1, :], ls='-', lw=1., c='k')
        for pp in range(npts):
            for ll in range(nlamb):
                lab = (r'pts {} - '.format(pp)
                       + '$\lambda$'+' = {:6.3f} A'.format(lamb[ll]*1.e10))
                axi.plot(xi[ll, pp, :], xj[ll, pp, :],
                         ls='None', marker=lm[ll%nm], c=lcol[pp%ncol], label=lab)

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
                       + '$\lambda$'+' = {:6.3f} A'.format(lamb[ll]*1.e10))
                dax['cross'].plot(R[ll, pp, :], ptsall[2, ll, pp, :],
                                  ls=lls[ll%nls], color=lcol[pp%ncol],
                                  label=lab)
                dax['hor'].plot(ptsall[0, ll, pp, :], ptsall[1, ll, pp, :],
                                ls=lls[ll%nls], color=lcol[pp%ncol], label=lab)
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













# #################################################################
# #################################################################
#                   data plot
# #################################################################
# #################################################################



def CrystalBragg_plot_data_vs_lambphi(xi, xj, bragg, lamb, phi, data,
                                      lambfit=None, phifit=None,
                                      spect1d=None, vertsum1d=None,
                                      lambax=None, phiax=None,
                                      phiminmax=None, dlines=None,
                                      lambmin=None, lambmax=None,
                                      xjcut=None, lambcut=None,
                                      phicut=None, spectcut=None,
                                      cmap=None, vmin=None, vmax=None,
                                      fs=None, dmargin=None,
                                      tit=None, wintit=None,
                                      angunits='deg'):

    # Check inputs
    # ------------

    if fs is None:
        fs = (14, 8)
    if tit is None:
        tit = False
    if wintit is None:
        wintit = _WINTIT
    if cmap is None:
        cmap = plt.cm.viridis
    if dmargin is None:
        dmargin = {'left':0.05, 'right':0.99,
                   'bottom':0.07, 'top':0.92,
                   'wspace':0.2, 'hspace':0.3}
    assert angunits in ['deg', 'rad']
    if angunits == 'deg':
        bragg = bragg*180./np.pi
        phi = phi*180./np.pi
        phifit = phifit*180./np.pi
        if phiax is not None:
            phiax = 180*phiax/np.pi
        phiminmax = phiminmax*180./np.pi
        if phicut is not None:
            phicut = phicut*180./np.pi
    nspect = spect1d.shape[0]

    if dlines is not None:
        lines = [k0 for k0, v0 in dlines.items()
                 if (v0['lambda'] >= lambfit[0]
                     and v0['lambda'] <= lambfit[-1])]
        lions = sorted(set([dlines[k0]['ION'] for k0 in lines]))
        nions = len(lions)
        dions = {k0: [k1 for k1 in lines if dlines[k1]['ION'] == k0]
                 for k0 in lions}
        dions = {k0: {'lamb': np.array([dlines[k1]['lambda']
                                        for k1 in dions[k0]]),
                      'symbol': [dlines[k1]['symbol'] for k1 in dions[k0]]}
                 for k0 in lions}
        lcol = ['k', 'r', 'b', 'g', 'm', 'c']
        ncol = len(lcol)
        llamb = np.concatenate([dions[lions[ii]]['lamb']
                                for ii in range(nions)])
        indsort = np.argsort(llamb)
        lsymb = list(itt.chain.from_iterable([dions[lions[ii]]['symbol']
                                              for ii in range(nions)]))
        lsymb = [lsymb[ii] for ii in indsort]
        lcollamb = list(itt.chain.from_iterable(
            [[lcol[ii]]*dions[lions[ii]]['lamb'].size
             for ii in range(nions)]))
        lcollamb = [lcollamb[ii] for ii in indsort]

    lcolspect = ['k', 'r', 'b', 'g', 'y', 'm', 'c']
    ncolspect = len(lcolspect)

    # pre-compute
    # ------------

    # extent
    extent = (xi.min(), xi.max(), xj.min(), xj.max())
    if lambmin is None:
        lambmin = lambfit.min()
    if lambmax is None:
        lambmax = lambfit.max()

    # Plot
    # ------------

    fig = fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(4, 4, **dmargin)
    ax0 = fig.add_subplot(gs[:3, 0], aspect='equal')    #, adjustable='datalim'
    ax1 = fig.add_subplot(gs[:3, 1], aspect='equal',
                          sharex=ax0, sharey=ax0)       #, adjustable='datalim'
    axs1 = fig.add_subplot(gs[3, 1], sharex=ax0)
    ax2 = fig.add_subplot(gs[:3, 2])
    axs2 = fig.add_subplot(gs[3, 2], sharex=ax2, sharey=axs1)
    ax3 = fig.add_subplot(gs[:3, 3], sharey=ax2)

    ax0.set_title('Coordinates transform')
    ax1.set_title('Camera image')
    axs1.set_title('Vertical average')
    ax2.set_title('Camera image transformed')

    ax0.set_ylabel(r'$x_j$ (m)')
    ax0.set_xlabel(r'$x_i$ (m)')
    axs1.set_ylabel(r'data')
    axs1.set_xlabel(r'$x_i$ (m)')
    ax2.set_ylabel(r'incidence angle ($deg$)')
    axs2.set_xlabel(r'$\lambda$ ($m$)')

    ax0.contour(xi, xj, bragg, 10, cmap=cmap)
    ax0.contour(xi, xj, phi, 10, cmap=cmap, ls='--')
    ax1.imshow(data, extent=extent, aspect='equal',
               origin='lower', vmin=vmin, vmax=vmax)
    axs1.plot(xi, np.nanmean(data, axis=0), c='k', ls='-')
    ax2.scatter(lamb.ravel(), phi.ravel(), c=data.ravel(), s=2,
                marker='s', edgecolors='None',
                cmap=cmap, vmin=vmin, vmax=vmax)

    if xjcut is None:
        ax3.plot(vertsum1d, phifit, c='k', ls='-')
    else:
        dphicut = 0.1*(phifit.max() - phifit.min())
        for ii in range(xjcut.size):
            ax1.axhline(xjcut[ii],
                        ls='-', lw=1., c='r')
            ax2.plot(lambcut[ii, :], phicut[ii, :],
                     ls='-', lw=1., c='r')
            ax3.plot(lambcut[ii, :],
                     (np.nanmean(phicut[ii, :])
                      + spectcut[ii, :]*dphicut/np.nanmax(spectcut[ii, :])),
                     ls='-', lw=1., c='r')

    if phiax is not None:
        ax2.plot(lambax, phiax, c='r', ls='-', lw=1.)

    # Plot known spectral lines
    if dlines is not None:
        for ii, k0 in enumerate(lions):
            for jj in range(dions[k0]['lamb'].size):
                x = dions[k0]['lamb'][jj]
                col = lcol[ii%ncol]
                axs2.axvline(x,
                             c=col, ls='--')
                axs2.annotate(dions[k0]['symbol'][jj],
                              xy=(x, 1.01), xytext=None,
                              xycoords=('data', 'axes fraction'),
                              color=col, arrowprops=None,
                              horizontalalignment='center',
                              verticalalignment='bottom')
                if xjcut is not None:
                    ax3.axvline(x,
                                c=col, ls='--')
        hand = [mlines.Line2D([], [], color=lcol[ii%ncol], ls='--')
                for ii in range(nions)]
        axs2.legend(hand, lions,
                    bbox_to_anchor=(1., 1.02), loc='upper left')

    # Plot 1d spectra and associated phi windows
    for ii in range(nspect):
        axs2.plot(lambfit, spect1d[ii], c=lcolspect[ii%ncolspect], ls='-')
        ax2.axhline(phiminmax[ii ,0], c=lcolspect[ii%ncolspect], ls='-', lw=1.)
        ax2.axhline(phiminmax[ii, 1], c=lcolspect[ii%ncolspect], ls='-', lw=1.)
        ax3.axhline(phiminmax[ii ,0], c=lcolspect[ii%ncolspect], ls='-', lw=1.)
        ax3.axhline(phiminmax[ii, 1], c=lcolspect[ii%ncolspect], ls='-', lw=1.)

    ax2.set_xlim(lambmin, lambmax)
    ax2.set_ylim(phifit.min(), phifit.max())
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)

    if tit is not False:
        fig.suptitle(tit, size=14, weight='bold')
    if wintit is not False:
        fig.canvas.set_window_title(wintit)
    return [ax0, ax1]



def CrystalBragg_plot_data_fit1d(dfit1d, dinput=None, showonly=None,
                                 lambmin=None, lambmax=None,
                                 same_spectrum=None,
                                 fs=None, dmargin=None,
                                 tit=None, wintit=None):

    # Check inputs
    # ------------
    if fs is None:
        fs = (15, 8)
    if wintit is None:
        wintit = _WINTIT
    if dmargin is None:
        dmargin = {'left':0.05, 'right':0.85,
                   'bottom':0.07, 'top':0.85,
                   'wspace':0.2, 'hspace':0.3}

    # pre-compute
    # ------------
    if same_spectrum is True:
        nlines = int(dinput['nlines'] / dinput['same_spectrum_nspect'])
        dinput['lines'] = dinput['lines'][:nlines]
        dinput['ion'] = dinput['ion'][:nlines]
        dinput['symb'] = dinput['symb'][:nlines]
        nwidth = int(dinput['width']['keys'].size
                     / dinput['same_spectrum_nspect'])
        dinput['width']['keys'] = dinput['width']['keys'][:nwidth]
        dinput['width']['ind'] = dinput['width']['ind'][:, :nlines]
        dinput['width']['ind'] = dinput['width']['ind'][:nwidth, :]
        dinput['shift']['ind'] = dinput['shift']['ind'][:, :nlines]

    nlines = dinput['lines'].size
    ions_u = sorted(set(dinput['ion'].tolist()))
    nions = len(ions_u)

    indwidth = np.argmax(dinput['width']['ind'], axis=0)
    indshift = np.argmax(dinput['shift']['ind'], axis=0)

    x = dinput['lines'][None, :] + dfit1d['shift']

    lcol = ['k', 'r', 'b', 'g', 'm', 'c']
    ncol = len(lcol)
    if dfit1d['Ti'] is True:
        lfcol = ['y', 'g', 'c', 'm']
    else:
        lfcol = [None]
    nfcol = len(lfcol)
    if dfit1d['vi'] is True:
        lhatch = [None, '/', '\\', '|', '-', '+', 'x', '//']
    else:
        lhatch = [None]
    nhatch = len(lhatch)
    nspect = dfit1d['data'].shape[0]

    # import pdb; pdb.set_trace()     # DB

    # Plot
    # ------------

    for ii in range(nspect):
        if tit is None:
            titi = ("spect1d {}\n".format(ii)
                   + "phi in [{}, {}]".format(round(ii), ii))
        else:
            titi = tit

        fig = fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(1, 1, **dmargin)
        ax = fig.add_subplot(gs[0, 0])
        ax.set_ylabel(r'data (a.u.)')
        ax.set_xlabel(r'$\lambda$ (m)')

        if showonly is not True:
            ax.plot(dfit1d['lamb'], dfit1d['sol_detail'][ii, 0, :],
                    ls='-', c='k')
            ax.set_prop_cycle(None)
            if dfit1d['Ti'] is True or dfit1d['vi'] is True:
                for jj in range(nlines):
                    col = lfcol[indwidth[jj]%nfcol]
                    hatch = lhatch[indshift[jj]%nhatch]
                    ax.fill_between(dfit1d['lamb'],
                                    dfit1d['sol_detail'][ii, 1+jj, :],
                                    alpha=0.3, color=col, hatch=hatch)
            else:
                ax.plot(dfit1d['lamb'], dfit1d['sol_detail'][ii, 1:, :].T)
            ax.plot(dfit1d['lamb'], dfit1d['sol'][ii, :], c='k', lw=2.)
            ax.plot(dfit1d['lamb'], dfit1d['data'][ii, :],
                    marker='.', c='k', ls='None', ms=8)
        else:
            ax.plot(dfit1d['lamb'], dfit1d['data'][ii, :],
                    marker='.', c='k', ls='-', ms=8)

        # Annotate lines
        for jj, k0 in enumerate(ions_u):
            col = lcol[jj%ncol]
            ind = (dinput['ion'] == k0).nonzero()[0]
            for nn in ind:
                ax.axvline(x[ii, nn],
                           c=col, ls='--')
                lab = (dinput['symb'][nn]
                       + '\n{:4.2e}'.format(dfit1d['coefs'][ii, nn])
                       + '\n({:+4.2e} A)'.format(dfit1d['shift'][ii, nn]*1.e10))
                ax.annotate(lab,
                            xy=(x[ii, nn], 1.01), xytext=None,
                            xycoords=('data', 'axes fraction'),
                            color=col, arrowprops=None,
                            horizontalalignment='center',
                            verticalalignment='bottom')

        # Ion legend
        hand = [mlines.Line2D([], [], color=lcol[jj%ncol], ls='--',
                              label=ions_u[jj])
                for jj in range(nions)]
        legi = ax.legend(handles=hand,
                         title='ions',
                         bbox_to_anchor=(1.01, 1.), loc='upper left')
        ax.add_artist(legi)

        # Ti legend
        if dfit1d['Ti'] is True:
            hand = [mpatches.Patch(color=lfcol[jj%nfcol])
                    for jj in range(dinput['width']['ind'].shape[0])]
            lleg = [dinput['width']['keys'][jj]
                    + '  {:4.2f}'.format(dfit1d['kTiev'][ii, jj]*1.e-3)
                    for jj in range(dinput['width']['ind'].shape[0])]
            legT = ax.legend(handles=hand, labels=lleg,
                             title='Ti (keV)',
                             bbox_to_anchor=(1.01, 0.8), loc='upper left')
            ax.add_artist(legT)

        # vi legend
        if dfit1d['vi'] is True:
            hand = [mpatches.Patch(facecolor='w', edgecolor='k',
                                   hatch=lhatch[jj%nhatch])
                    for jj in range(dinput['shift']['ind'].shape[0])]
            lleg = [dinput['shift']['keys'][jj]
                    + '  {:4.2f}'.format(dfit1d['vims'][ii, jj]*1.e-3)
                    for jj in range(dinput['shift']['ind'].shape[0])]
            legv = ax.legend(handles=hand, labels=lleg,
                             title='vi (km/s)',
                             bbox_to_anchor=(1.01, 0.5), loc='upper left')
            ax.add_artist(legv)

        # Ratios legend
        if dfit1d['ratio'] is not None:
            nratio = len(dfit1d['ratio']['up'])
            hand = [mlines.Line2D([], [], c='k', ls='None')]*nratio
            lleg = ['{} =  {:4.2e}'.format(dfit1d['ratio']['str'][jj],
                                           dfit1d['ratio']['value'][ii, jj])
                    for jj in range(nratio)]
            legr = ax.legend(handles=hand,
                             labels=lleg,
                             title='line ratio',
                             bbox_to_anchor=(1.01, 0.11), loc='lower left')
            ax.add_artist(legr)

        # double legend
        if dfit1d['double'] is True:
            hand = [mlines.Line2D([], [], c='k', ls='None')]*2
            lleg = ['ratio = {:4.2f}'.format(dfit1d['dratio'][ii]),
                    ('shift ' + r'$\approx$'
                     + ' {:4.2e}'.format(dfit1d['dshift'][0]))]
            legr = ax.legend(handles=hand,
                             labels=lleg,
                             title='double',
                             bbox_to_anchor=(1.01, 0.), loc='lower left')

        ax.set_xlim(lambmin, lambmax)

        if titi is not False:
            fig.suptitle(titi, size=14, weight='bold')
        if wintit is not False:
            fig.canvas.set_window_title(wintit)
    return ax


def CrystalBragg_plot_data_fit2d(xi, xj, data, lamb, phi, indok=None,
                                 dfit2d=None,
                                 dax=None, indspect=None,
                                 spect1d=None, fit1d=None,
                                 lambfit=None, phiminmax=None,
                                 cmap=None, vmin=None, vmax=None,
                                 plotmode=None, fs=None, dmargin=None,
                                 angunits='deg', tit=None, wintit=None):

    # Check inputs
    # ------------

    if dax is None:
        if fs is None:
            fs = (18, 9)
        if cmap is None:
            cmap = plt.cm.viridis
        if dmargin is None:
            dmargin = {'left':0.05, 'right':0.99,
                       'bottom':0.06, 'top':0.95,
                       'wspace':None, 'hspace':0.8}
        if wintit is None:
            wintit = _WINTIT
        if tit is None:
            tit = '2D fitting of X-Ray Crystal Bragg spectrometer'
    if dfit2d['dinput']['symmetry'] is True:
        symaxis = dfit2d['dinput']['symmetry_axis'][indspect]
    if angunits is None:
        angunits = 'deg'
    assert angunits in ['deg', 'rad']




    phiflat = dfit2d['phi']
    pts_phi = dfit2d['pts_phi']
    ylim = np.r_[dfit2d['dinput']['phiminmax']]
    if angunits == 'deg':
        phiflat = phiflat*180./np.pi
        phi = phi*180./np.pi
        pts_phi = pts_phi*180./np.pi
        ylim = ylim*180./np.pi
        phiminmax = phiminmax*180./np.pi
        if dfit2d['dinput']['symmetry'] is True:
            symaxis = symaxis*180./np.pi
    if plotmode == 'transform':
        xlab = r'$\lambda$ (m)'
        ylab = r'$\phi$ ({})'.format(angunits)
    else:
        xlab = r'$x_i$' + r' (m)'
        ylab = r'$x_j$' + r' ({})'.format(angunits)
        spect1d = None

    # pre-compute
    # ------------

    if vmin is None:
        vmin = 0.
    if vmax is None:
        vmax = max(np.nanmax(dfit2d['data'][indspect, :]),
                   np.nanmax(dfit2d['sol_tot'][indspect, :]))
    err = dfit2d['sol_tot'][indspect, :] - dfit2d['data'][indspect, :]
    errm = vmax/10.


    # Prepare figure if dax not provided
    # ------------
    if dax is None:
        fig = plt.figure(figsize=fs)
        naxh = (3*3
                + 2*(dfit2d['dinput']['Ti'] + dfit2d['dinput']['vi']
                     + (dfit2d['ratio'] not in [None, False])))
        naxv = 1
        gs = gridspec.GridSpec((3+naxv)*3, naxh, **dmargin)
        if plotmode == 'transform':
            ax0 = fig.add_subplot(gs[:9, :3])
            ax0.set_xlim(dfit2d['dinput']['lambminmax'])
            ax0.set_ylim(ylim)
        else:
            ax0 = fig.add_subplot(gs[:9, :3], aspect='equal')
        ax0c = fig.add_subplot(gs[10, :3])
        ax1 = fig.add_subplot(gs[:9, 3:6],
                          sharex=ax0, sharey=ax0)
        ax2 = fig.add_subplot(gs[:9, 6:9],
                          sharex=ax0, sharey=ax0)
        ax1c = fig.add_subplot(gs[10, 3:6])

        ax0.set_title('Residu')
        ax0.set_xlabel(xlab)
        ax0.set_ylabel(ylab)
        ax1.set_title('Original data')
        ax1.set_xlabel(xlab)
        ax2.set_title('2d fit')
        dax = {'err': ax0, 'err_colorbar': ax0c,
               'data': ax1, 'data_colorbar': ax1c,
               'fit': ax2}
        if tit is not False:
            fig.suptitle(tit)
        if wintit is not False:
            fig.canvas.set_window_title(wintit)

        if naxv == 1:
            dax['fit1d'] = fig.add_subplot(gs[9:11, 6:9], sharex=ax2)
            dax['fit1d'].set_title('1d sliced spectrum vs fit')
            dax['fit1d'].set_ylabel(r'data')
            dax['err1d'] = fig.add_subplot(gs[11, 6:9], sharex=ax2)
            dax['err1d'].set_ylabel(r'err')
            dax['err1d'].set_xlabel(xlab)

        nn = 1
        if dfit2d['dinput']['Ti'] is True:
            dax['Ti'] = fig.add_subplot(gs[:9, 9+2*(nn-1):9+2*nn], sharey=ax1)
            dax['Ti'].set_title(r'Width')
            dax['Ti'].set_xlabel(r'$\hat{T_i}$' + r' (keV)')
            nn += 1
        if dfit2d['dinput']['vi'] is True:
            dax['vi'] = fig.add_subplot(gs[:9, 9+2*(nn-1):9+2*nn], sharey=ax1)
            dax['vi'].set_title(r'Shift')
            dax['vi'].set_xlabel(r'$\hat{v_i}$' + r' (km/s)')
            dax['vi'].axvline(0, ls='-', lw=1., c='k')
            nn += 1
        if dfit2d['ratio'] not in [None, False]:
            dax['ratio'] = fig.add_subplot(gs[:9, 9+2*(nn-1):], sharey=ax1)
            dax['ratio'].set_title(r'Intensity Ratio')
            dax['ratio'].set_xlabel(r'ratio (a.u)')

    # Plot main images
    # ------------
    if plotmode == 'transform':
        if dax.get('err') is not None:
            errax = dax['err'].scatter(dfit2d['lamb'],
                                       phiflat,
                                       c=err,
                                       s=6, marker='s', edgecolors='None',
                                       vmin=-errm, vmax=errm,
                                       cmap=plt.cm.seismic)
        if dax.get('data') is not None:
            dataax = dax['data'].scatter(dfit2d['lamb'],
                                         phiflat,
                                         c=dfit2d['data'][indspect, :],
                                         s=6, marker='s', edgecolors='None',
                                         vmin=vmin, vmax=vmax, cmap=cmap)
        if dax.get('fit') is not None:
            dax['fit'].scatter(dfit2d['lamb'],
                               phiflat,
                               c=dfit2d['sol_tot'][indspect, :],
                               s=6, marker='s', edgecolors='None',
                               vmin=vmin, vmax=vmax, cmap=cmap)
            if dfit2d['dinput']['symmetry'] is True:
                dax['fit'].axhline(symaxis,
                                   c='k', ls='--', lw=2.)
        if dax.get('fit1d') is not None:
            # dax['fit1d'].plot()
            pass
    else:
        extent = (xi.min(), xi.max(), xj.min(), xj.max())
        if dax.get('err') is not None:
            dax['err'].imshow(err,
                              extent=extent, cmap=plt.cm.seismic,
                              origin='lower', vmin=errm, vmax=errm)
        if dax.get('data') is not None:
            dax['data'].imshow(data,
                               extent=extent, cmap=cmap,
                               origin='lower', vmin=vmin, vmax=vmax)
        if dax.get('fit') is not None:
            dax['fit'].imshow(fit,
                              extent=extent, cmap=cmap,
                              origin='lower', vmin=vmin, vmax=vmax)

    # Plot profiles
    # ------------
    if dax.get('Ti') is not None and dfit2d['dinput']['Ti'] is True:
        for ii in range(dfit2d['kTiev'].shape[1]):
            dax['Ti'].plot(dfit2d['kTiev'][indspect, ii, :]*1.e-3,
                           pts_phi,
                           ls='-', marker='.', ms=4,
                           label=dfit2d['dinput']['width']['keys'][ii])
        dax['Ti'].set_xlim(left=0.)
        dax['Ti'].legend(frameon=True,
                         loc='upper left', bbox_to_anchor=(0., -0.1))

    if dax.get('vi') is not None and dfit2d['dinput']['vi'] is True:
        for ii in range(dfit2d['vims'].shape[1]):
            dax['vi'].plot(dfit2d['vims'][indspect, ii, :]*1.e-3,
                           pts_phi,
                           ls='-', marker='.', ms=4,
                           label=dfit2d['dinput']['shift']['keys'][ii])
        dax['vi'].legend(frameon=True,
                         loc='upper left', bbox_to_anchor=(0., -0.1))

    if dax.get('ratio') not in [None, False]:
        for ii in range(dfit2d['ratio']['value'].shape[1]):
            dax['ratio'].plot(dfit2d['ratio']['value'][indspect, ii, :],
                              pts_phi,
                              ls='-', marker='.', ms=4,
                              label=dfit2d['ratio']['str'][ii])
        dax['ratio'].legend(frameon=True,
                            loc='upper left', bbox_to_anchor=(0., -0.1))

    # Add 1d prof
    # ------------
    if spect1d is not None:
        lcol = ['r', 'k', 'm']
        for ii in range(spect1d.shape[0]):
            l, = dax['fit1d'].plot(lambfit, fit1d[ii, :],
                                   ls='-', lw=1., c=lcol[ii%len(lcol)])
            col = l.get_color()
            dax['fit1d'].plot(lambfit, spect1d[ii, :],
                              ls='None', marker='.', c=col, ms=4)
            dax['err1d'].plot(lambfit, fit1d[ii, :] - spect1d[ii, :],
                              ls='-', lw=1., c=col)
            dax['fit'].axhline(phiminmax[ii][0], c=col, ls='-', lw=2.)
            dax['fit'].axhline(phiminmax[ii][1], c=col, ls='-', lw=2.)

    # double legend
    if dfit2d['dinput']['double'] is not False:
        hand = [mlines.Line2D([], [], c='k', ls='None')]*2
        c0 = (dfit2d['dinput']['double'] is True
              or dfit2d['dinput']['double'].get('dratio') is None)
        if c0:
            dratio = dfit2d['dratio'][indspect]
            dratiostr = ''
        else:
            dratio = dfit2d['dinput']['double']['dratio']
            dratiostr = '  (fixed)'
        c0 = (dfit2d['dinput']['double'] is True
              or dfit2d['dinput']['double'].get('dshift') is None)
        if c0:
            dshift = dfit2d['dshift'][indspect]
            dshiftstr = ''
        else:
            dshift = dfit2d['dinput']['double']['dshift']
            dshiftstr = '  (fixed)'
        lleg = ['dratio = {:4.2f}{}'.format(dratio, dratiostr),
                ('dshift = {:4.2e} * '.format(dshift)
                 + r'$\lambda${}'.format(dshiftstr))]
        legr = dax['err1d'].legend(handles=hand, labels=lleg, title='double',
                                   bbox_to_anchor=(1.01, 0.),
                                   loc='center left')

    # Polishing
    # ------------
    for kax in dax.keys():
        if kax != 'err':
            plt.setp(dax[kax].get_yticklabels(), visible=False)
    if dax.get('fit') is not None:
        plt.setp(dax['fit'].get_xticklabels(), visible=False)
    if dax.get('fit1d') is not None:
        plt.setp(dax['fit1d'].get_xticklabels(), visible=False)
    if dax.get('err_colorbar') is not None and dax['err'] is not None:
        plt.colorbar(errax, ax=dax['err'], cax=dax['err_colorbar'],
                     orientation='horizontal', extend='both')
    if (dax.get('data_colorbar') is not None
        and (dax['data'] is not None or dax['fit'] is not None)):
        plt.colorbar(dataax, ax=dax['data'], cax=dax['data_colorbar'],
                     orientation='horizontal')
    return dax
