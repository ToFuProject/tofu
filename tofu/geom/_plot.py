# Built-in
import os
import itertools as itt
import warnings


# Generic common libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mPolygon
from matplotlib.patches import Wedge as mWedge
from matplotlib.patches import  Rectangle as mRectangle
from matplotlib.axes._axes import Axes
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# ToFu-specific
try:
    from tofu.version import __version__
    import tofu.utils as utils
    import tofu.geom._def as _def
    import tofu.geom._GG as _GG
except Exception:
    from tofu.version import __version__
    from .. import utils as utils
    from . import _def as _def
    from . import _GG as _GG


#__author_email__ = 'didier.vezinet@cea.fr'
_fontsize = 8
_labelpad = 0
__github = 'https://github.com/ToFuProject/tofu/issues'
_wintit = 'tofu-%s        report issues / requests at %s'%(__version__, __github)
_nchMax = 4
_cdef = 'k'
_cbck = (0.8,0.8,0.8)
_lcch = [plt.cm.tab20.colors[ii] for ii in [6,8,10,7,9,11]]

# Generic
def _check_Lax(lax=None, n=2):
    assert n in [1,2]
    c0 = lax is None
    c1 = issubclass(lax.__class__,Axes)
    c2 = hasattr(lax, '__iter__')
    if c2:
        c2 = all([aa is None or issubclass(aa.__class__,Axes) for aa in lax])
        c2 = c2 and len(lax) in [1,2]
    if n==1:
        assert c0 or c1, "Arg ax must be None or a plt.Axes instance !"
    else:
        assert c0 or c1 or c2, "Arg lax must be an Axes or a list/tuple of such !"
        if c0:
            lax = [None,None]
        elif c1:
            lax = [lax,None]
        elif c2 and len(lax)==1:
            lax = [lax[0],None]
        else:
            lax = list(lax)
    return lax, c0, c1, c2





"""
###############################################################################
###############################################################################
                        Ves class and functions
###############################################################################
"""

############################################
##### Plotting functions
############################################


def _Struct_plot_format(ss, proj='all', **kwdargs):

    # Local default
    defplot = {'cross':{'Elt':'PIBsBvV',
                        'dP':{'empty':_def.TorPd , 'full':_def.StructPd},
                        'dI':_def.TorId,
                        'dBs':_def.TorBsd,
                        'dBv':_def.TorBvd,
                        'dVect':_def.TorVind},
               'hor':{'Elt':'PIBsBvV',
                      'dP':{'empty':_def.TorPd , 'full':_def.StructPd_Tor},
                      'dI':_def.TorITord,
                      'dBs':_def.TorBsTord,
                      'dBv':_def.TorBvTord,
                      'Nstep':_def.TorNTheta},
               '3d':{'Elt':'P',
                     'dP':{'color':(0.8,0.8,0.8,1.),
                           'rstride':1,'cstride':1,
                           'linewidth':0., 'antialiased':False},
                     'Lim':None,
                     'Nstep':_def.TorNTheta}}

    # Select keys for proj
    lproj = ['cross','hor'] if proj=='all' else [proj]

    # Match with kwdargs
    dk = {}
    dk['cross']= dict([(k,k) for k in defplot['cross'].keys()])
    dk['hor']= dict([(k,k+'Hor') for k in defplot['hor'].keys()])
    dk['hor']['Elt'] = 'Elt'
    dk['hor']['dP'] = 'dP'
    dk['hor']['Nstep'] = 'Nstep'
    dk['3d']= dict([(k,k) for k in defplot['3d'].keys()])

    # Rename keys (retro-compatibility)
    lrepl = [('Elt','Elt'),('dP','Pdict'),('dI','Idict'),('dBs','Bsdict'),
             ('dBv','Bvdict'),('dVect','Vdict'),('dIHor','IdictHor'),
             ('dBsHor','BsdictHor'),('dBvHor','BvdictHor'),
             ('Nstep','Nstep'),('Lim','Lim')]
    dnk = dict(lrepl)

    # Map out dict
    dout = {}
    for pp in lproj:
        dout[pp] = {}
        for k in defplot[pp].keys():
            v = kwdargs[dk[pp][k]]
            if v is None:
                if k in ss._dplot[pp].keys():
                    dout[pp][dnk[k]] = ss._dplot[pp][k]
                else:
                    if k=='dP':
                        if ss.Id.Cls=='Ves':
                            dout[pp][dnk[k]] = defplot[pp][k]['empty']
                        else:
                            dout[pp][dnk[k]] = defplot[pp][k]['full']
                    else:
                        dout[pp][dnk[k]] = defplot[pp][k]
            else:
                dout[pp][dnk[k]] = v

    return dout


def Struct_plot(lS, lax=None, proj=None, element=None, dP=None,
                dI=None, dBs=None, dBv=None,
                dVect=None, dIHor=None, dBsHor=None, dBvHor=None,
                Lim=None, Nstep=None, dLeg=None, indices=False,
                draw=True, fs=None, wintit=None, tit=None, Test=True):
    """ Plot the projections of a list of Struct subclass instances

    D. VEZINET, Aug. 2014
    Inputs :
        V           A Ves instance
        Nstep      An int (the number of points for evaluation of theta by np.linspace)
        axP         A plt.Axes instance (if given) on which to plot the poloidal projection, otherwise ('None') a new figure/axes is created
        axT         A plt.Axes instance (if given) on which to plot the toroidal projection, otherwise ('None') a new figure/axes is created
        Tdict       A dictionnary specifying the style of the polygon plot
        dLeg     A dictionnary specifying the style of the legend box (if None => no legend)
    Outputs :
        axP          The plt.Axes instance on which the poloidal plot was performed
        axT          The plt.Axes instance on which the toroidal plot was performed
    """
    if proj is None:
        proj = 'all'
    proj = proj.lower()
    if Test:
        msg = "Arg proj must be in ['cross','hor','all','3d'] !"
        assert proj in ['cross','hor','all','3d'], msg
        lax, C0, C1, C2 = _check_Lax(lax,n=2)
        assert type(draw) is bool, "Arg draw must be a bool !"

    C0 = issubclass(lS.__class__, utils.ToFuObject)
    C1 = (isinstance(lS,list)
          and all([issubclass(ss.__class__, utils.ToFuObject) for ss in lS]))
    msg = "Arg lves must be a Struct subclass or a list of such !"
    assert C0 or C1, msg
    if C0:
        lS = [lS]
    nS = len(lS)
    if wintit is None:
        wintit = _wintit

    kwa = dict(fs=fs, wintit=wintit, Test=Test)
    if proj=='3d':
        # Temporary matplotlib issue
        dLeg = None

    for ii in  range(0,nS):
        dplot = _Struct_plot_format(lS[ii], proj=proj, Elt=element,
                                    dP=dP, dI=dI, dBs=dBs,
                                    dBv=dBv, dVect=dVect, dIHor=dIHor,
                                    dBsHor=dBsHor, dBvHor=dBvHor,
                                    Lim=Lim, Nstep=Nstep)
        for k in dplot.keys():
            dplot[k].update(kwa)

        if proj=='3d':
            lax[0] = _Plot_3D_plt_Ves(lS[ii], ax=lax[0], LegDict=None,
                                      draw=False, **dplot[proj])
        else:
            if proj=='cross':
                lax[0] = _Plot_CrossProj_Ves(lS[ii], ax=lax[0],
                                             indices=indices, LegDict=None,
                                             draw=False, **dplot[proj])
            elif proj=='hor':
                lax[0] = _Plot_HorProj_Ves(lS[ii], ax=lax[0],
                                           indices=indices, LegDict=None,
                                           draw=False, **dplot[proj])
            elif proj=='all':
                if lax[0] is None or lax[1] is None:
                    lax = list(_def.Plot_LOSProj_DefAxes('All', fs=fs,
                                                         wintit=wintit,
                                                         Type=lS[ii].Id.Type))
                lax[0] = _Plot_CrossProj_Ves(lS[ii], ax=lax[0], LegDict=None,
                                             indices=indices,
                                             draw=False, **dplot['cross'])
                lax[1] = _Plot_HorProj_Ves(lS[ii], ax=lax[1], LegDict=None,
                                           indices=indices,
                                           draw=False, **dplot['hor'])

    # recompute the ax.dataLim
    lax[0].relim()
    if proj=='all':
        lax[1].relim()
    # update ax.viewLim using the new dataLim
    lax[0].autoscale_view()
    if proj=='all':
        lax[1].autoscale_view()

    if tit != False:
        lax[0].figure.suptitle(tit)

    if dLeg not in [None, False]:
        lax[0].legend(**dLeg)
    if draw:
        lax[0].relim()
        lax[0].autoscale_view()
        if len(lax)==2 and lax[1] is not None:
            lax[1].relim()
            lax[1].autoscale_view()
        lax[0].figure.canvas.draw()
    lax = lax if proj=='all' else lax[0]
    return lax



def _Plot_CrossProj_Ves(V, ax=None, Elt='PIBsBvV',
                        Pdict=_def.TorPd, Idict=_def.TorId, Bsdict=_def.TorBsd,
                        Bvdict=_def.TorBvd, Vdict=_def.TorVind,
                        LegDict=_def.TorLegd, indices=False,
                        draw=True, fs=None, wintit=_wintit, Test=True):
    """ Plot the poloidal projection of a Ves instance

    Parameters
    ----------
        V       :   tfg.Ves / tfg.Struct
            A Ves instance
        ax      :   None / plt.Axes
            A plt.Axes instance (if given) on which to plot, otherwise ('None') a new figure/axes is created
        Pdict   :   dict
            A dictionnary specifying the style of the polygon plot
        LegDict :   None / dict
            A dictionnary specifying the style of the legend box (if None => no legend)

    Returns
    -------
        ax          The plt.Axes instance on which the plot was performed
    """
    if Test:
        ax, C0, C1, C2 = _check_Lax(ax,n=1)
        assert type(Pdict) is dict, 'Arg Pdict should be a dictionary !'
        assert type(Idict) is dict, "Arg Idict should be a dictionary !"
        assert type(Bsdict) is dict, "Arg Bsdict should be a dictionary !"
        assert type(Bvdict) is dict, "Arg Bvdict should be a dictionary !"
        assert type(Vdict) is dict, "Arg Vdict should be a dictionary !"
        assert type(LegDict) is dict or LegDict is None, 'Arg LegDict should be a dictionary !'
        assert type(indices) is bool
        if indices:
            assert 'P' in Elt
    if ax is None:
        ax = _def.Plot_LOSProj_DefAxes('Cross', fs=fs,
                                       wintit=wintit, Type=V.Id.Type)
    if 'P' in Elt or 'V' in Elt:
        P_closed = V.Poly_closed
    if 'V' in Elt or indices:
        midX = (P_closed[0,:-1]+P_closed[0,1:])/2.
        midY = (P_closed[1,:-1]+P_closed[1,1:])/2.
        VInX, VInY = V.dgeom['VIn'][0,:], V.dgeom['VIn'][1,:]
    if 'P' in Elt:
        if V._InOut=='in':
            ax.plot(P_closed[0,:], P_closed[1,:],
                    label=V.Id.NameLTX,**Pdict)
        elif V._InOut=='out':
            ax.add_patch(mPolygon(V.Poly.T, closed=True,
                                  label=V.Id.NameLTX, **Pdict))
        else:
            msg = "self._InOut not defined !"
            raise Exception(msg)
    if 'I' in Elt:
        ax.plot(V.dsino['RefPt'][0], V.dsino['RefPt'][1],
                label=V.Id.NameLTX+" Imp", **Idict)
    if 'Bs' in Elt:
        ax.plot(V.dgeom['BaryS'][0], V.dgeom['BaryS'][1],
                label=V.Id.NameLTX+" Bs", **Bsdict)
    if 'Bv' in Elt and V.Id.Type=='Tor':
        ax.plot(V.dgeom['BaryV'][0], V.dgeom['BaryV'][1],
                label=V.Id.NameLTX+" Bv", **Bvdict)
    if 'V' in Elt:
        ax.quiver(midX, midY, VInX, VInY,
                  angles='xy', scale_units='xy',
                  label=V.Id.NameLTX+" Vin", **Vdict)
    if indices:
        for ii in range(0,V.dgeom['nP']):
            ax.annotate(r"{0}".format(ii), size=10,
                        xy = (midX[ii],midY[ii]),
                        xytext = (midX[ii]-0.01*VInX[ii],
                                  midY[ii]-0.01*VInY[ii]),
                        horizontalalignment='center',
                        verticalalignment='center')
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.relim()
        ax.autoscale_view()
        ax.figure.canvas.draw()
    return ax




def _Plot_HorProj_Ves(V, ax=None, Elt='PI', Nstep=_def.TorNTheta,
                      Pdict=_def.TorPd, Idict=_def.TorITord,
                      Bsdict=_def.TorBsTord, Bvdict=_def.TorBvTord,
                      LegDict=_def.TorLegd, indices=False,
                      draw=True, fs=None, wintit=_wintit, Test=True):
    """ Plotting the toroidal projection of a Ves instance

    Parameters
    ----------
        V           A Ves instance
        Nstep      An int (the number of points for evaluation of theta by np.linspace)
        ax          A plt.Axes instance (if given) on which to plot, otherwise ('None') a new figure/axes is created
        Tdict       A dictionnary specifying the style of the polygon plot
        LegDict     A dictionnary specifying the style of the legend box (if None => no legend)

    Returns
    -------
        ax          The plt.Axes instance on which the plot was performed
    """
    if Test:
        assert type(Nstep) is int
        ax, C0, C1, C2 = _check_Lax(ax,n=1)
        assert type(Pdict) is dict, 'Arg Pdict should be a dictionary !'
        assert type(Idict) is dict, 'Arg Idict should be a dictionary !'
        assert type(LegDict) is dict or LegDict is None, 'Arg LegDict should be a dictionary !'

    if ax is None:
        ax = _def.Plot_LOSProj_DefAxes('Hor', Type=V.Id.Type,
                                       fs=fs, wintit=wintit)
    P1Min = V.dgeom['P1Min']
    P1Max = V.dgeom['P1Max']
    if 'P' in Elt:
        if V._InOut=='in':
            if V.Id.Type=='Tor':
                Theta = np.linspace(0, 2*np.pi, num=Nstep,
                                    endpoint=True, retstep=False)
                lx = np.concatenate((P1Min[0]*np.cos(Theta),np.array([np.nan]),
                                     P1Max[0]*np.cos(Theta)))
                ly = np.concatenate((P1Min[0]*np.sin(Theta),np.array([np.nan]),
                                     P1Max[0]*np.sin(Theta)))
            elif V.Id.Type=='Lin':
                lx = np.array([V.Lim[0,0],V.Lim[0,1],V.Lim[0,1],
                               V.Lim[0,0],V.Lim[0,0]])
                ly = np.array([P1Min[0],P1Min[0],P1Max[0],P1Max[0],P1Min[0]])
            ax.plot(lx,ly,label=V.Id.NameLTX,**Pdict)
        elif V._InOut=='out':
            if V.Id.Type=='Tor':
                Theta = np.linspace(0, 2*np.pi, num=Nstep,
                                    endpoint=True, retstep=False)
                if V.noccur==0:
                    lx = np.concatenate((P1Min[0]*np.cos(Theta),
                                         P1Max[0]*np.cos(Theta[::-1])))
                    ly = np.concatenate((P1Min[0]*np.sin(Theta),
                                         P1Max[0]*np.sin(Theta[::-1])))
                    Lp = [mPolygon(np.array([lx,ly]).T, closed=True,
                                   label=V.Id.NameLTX, **Pdict)]
                else:
                    Lp = [mWedge((0,0), P1Max[0],
                                 V.Lim[ii][0]*180./np.pi,
                                 V.Lim[ii][1]*180./np.pi,
                                 width=P1Max[0]-P1Min[0],
                                 label=V.Id.NameLTX, **Pdict)
                          for ii in range(0,len(V.Lim))]
            elif V.Id.Type=='Lin':
                    ly = np.array([P1Min[0],P1Min[0],
                                   P1Max[0],P1Max[0],P1Min[0]])
                    Lp = []
                    for ii in range(0,len(V.Lim)):
                        lx = np.array([V.Lim[ii][0],V.Lim[ii][1],
                                       V.Lim[ii][1],V.Lim[ii][0],
                                       V.Lim[ii][0]])
                        Lp.append(mPolygon(np.array([lx,ly]).T,
                                           closed=True, label=V.Id.NameLTX,
                                           **Pdict))
            for pp in Lp:
                ax.add_patch(pp)
        else:
            msg = "Unknown self._InOut !"
            raise Exception(msg)

    if 'I' in Elt:
        if V.Id.Type=='Tor':
            lx = V.dsino['RefPt'][0]*np.cos(Theta)
            ly = V.dsino['RefPt'][0]*np.sin(Theta)
        elif V.Id.Type=='Lin':
            lx = np.array([np.min(V.Lim),np.max(V.Lim)])
            ly = V.dsino['RefPt'][0]*np.ones((2,))
        ax.plot(lx,ly,label=V.Id.NameLTX+" Imp",**Idict)
    if 'Bs' in Elt:
        if V.Id.Type=='Tor':
            lx = V.dgeom['BaryS'][0]*np.cos(Theta)
            ly = V.dgeom['BaryS'][0]*np.sin(Theta)
        elif V.Id.Type=='Lin':
            lx = np.array([np.min(V.Lim),np.max(V.Lim)])
            ly = V.dgeom['BaryS'][0]*np.ones((2,))
        ax.plot(lx,ly,label=V.Id.NameLTX+" Bs", **Bsdict)
    if 'Bv' in Elt and V.Type=='Tor':
        lx = V.dgeom['BaryV'][0]*np.cos(Theta)
        ly = V.dgeom['BaryV'][0]*np.sin(Theta)
        ax.plot(lx,ly,label=V.Id.NameLTX+" Bv", **Bvdict)

    if indices and V.noccur>1:
        if V.Id.Type=='Tor':
            for ii in range(0,V.noccur):
                R, theta = V.dgeom['P1Max'][0], np.mean(V.Lim[ii])
                X, Y = R*np.cos(theta), R*np.sin(theta)
                ax.annotate(r"{0}".format(ii), size=10,
                            xy = (X,Y),
                            xytext = (X+0.02*np.cos(theta),
                                      Y+0.02*np.sin(theta)),
                            horizontalalignment='center',
                            verticalalignment='center')
        elif V.Id.Type=='Lin':
            for ii in range(0,V.noccur):
                X, Y = np.mean(V.Lim[ii]), V.dgeom['P1Max'][0]
                ax.annotate(r"{0}".format(ii), size=10,
                            xy = (X,Y),
                            xytext = (X, Y+0.02),
                            horizontalalignment='center',
                            verticalalignment='center')

    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.relim()
        ax.autoscale_view()
        ax.figure.canvas.draw()
    return ax



def _Plot_3D_plt_Ves(V, ax=None, Elt='P', Lim=None,
                     Nstep=_def.Tor3DThetamin, Pdict=_def.TorP3Dd,
                     LegDict=_def.TorLegd,
                     draw=True, fs=None, wintit=_wintit, Test=True):
    if Test:
        msg = 'Arg ax should a plt.Axes instance !'
        assert isinstance(ax,Axes3D) or ax is None, msg
        assert Lim is None or (hasattr(Lim,'__iter__') and len(Lim)==2), "Arg Lim should be an iterable of 2 elements !"
        assert type(Pdict) is dict and (type(LegDict) is dict or LegDict is None), "Args Pdict and LegDict should be dictionnaries !"
        assert type(Elt)is str, "Arg Elt must be a str !"

    if ax is None:
        ax = _def.Plot_3D_plt_Tor_DefAxes(fs=fs, wintit=wintit)
    if V.Type=='Lin':
        lim = np.array(V.Lim)
        lim = lim.reshape((1,2)) if lim.ndim==1 else lim
        if Lim is not None:
            for ii in range(lim.shape[0]):
                lim[ii,:] = [max(Lim[0],lim[ii,0]),min(lim[1],Lim[1])]
    else:
        lim = np.array([[0.,2.*np.pi]]) if V.noccur==0 else np.array(V.Lim)
        lim = lim.reshape((1,2)) if lim.ndim==1 else lim
        if Lim is not None and V.Id.Cls=='Ves':
            Lim[0] = np.arctan2(np.sin(Lim[0]),np.cos(Lim[0]))
            Lim[1] = np.arctan2(np.sin(Lim[1]),np.cos(Lim[1]))
            for ii in range(lim.shape[0]):
                lim[ii,:] = Lim
    if 'P' in Elt:
        handles, labels = ax.get_legend_handles_labels()
        for ii in range(lim.shape[0]):
            theta = np.linspace(lim[ii,0],lim[ii,1],Nstep)
            theta = theta.reshape((1,Nstep))
            if V.Type=='Tor':
                X = np.dot(V.Poly_closed[0:1,:].T,np.cos(theta))
                Y = np.dot(V.Poly_closed[0:1,:].T,np.sin(theta))
                Z = np.dot(V.Poly_closed[1:2,:].T,np.ones(theta.shape))
            elif V.Type=='Lin':
                X = np.dot(theta.reshape((Nstep,1)),
                           np.ones((1,V.Poly_closed.shape[1]))).T
                Y = np.dot(V.Poly_closed[0:1,:].T,np.ones((1,Nstep)))
                Z = np.dot(V.Poly_closed[1:2,:].T,np.ones((1,Nstep)))
            ax.plot_surface(X,Y,Z, label=V.Id.NameLTX, **Pdict)
        proxy = plt.Rectangle((0, 0), 1, 1, fc=Pdict['color'])
        handles.append(proxy)
        labels.append(V.Id.NameLTX)
    return ax

"""
def Plot_3D_mlab_Tor(T,fig='None',thetaLim=(np.pi/2,2*np.pi),Tdict=Dict_3D_mlab_Tor_Def,LegDict=LegDict_Def):

    if fig=='None':
        fig = Plot_3D_mlab_Tor_DefFig()
    thetamin = np.pi/20
    N = np.ceil((thetaLim[1]-thetaLim[0])/thetamin)
    theta = np.linspace(thetaLim[0],thetaLim[1],N).reshape((1,N))
    X = np.dot(T.Poly[0:1,:].T,np.cos(theta))
    Y = np.dot(T.Poly[0:1,:].T,np.sin(theta))
    Z = np.dot(T.POly[1:2,:].T,np.ones(theta.shape))
    Theta = np.dot(np.ones(T.Poly[1:2,:].T.shape),theta)
    S = mlab.mesh(X,Y,Z,figure=fig,name=T.Id.NameLTX, scalars=Theta,**Tdict)
    mlab.orientation_axes(figure=fig)
    #mlab.axes(S)
    return fig,S
"""




def Plot_Impact_PolProjPoly(lS, Leg="", ax=None, Ang='theta', AngUnit='rad',
                            Sketch=True, dP=None,
                            dLeg=_def.TorLegd, draw=True, fs=None,
                            wintit=None, tit=None, Test=True):
    """ Plotting the toroidal projection of a Ves instance

    D. VEZINET, Aug. 2014
    Inputs :
        T           A Ves instance
        Leg         A str (the legend label to be used if T is not a Ves instance)
        ax          A plt.Axes instance (if given) on which to plot the projection space, otherwise ('None') a new figure/axes is created
        Dict        A dictionnary specifying the style of the boundary polygon plot
        dLeg        A dictionnary specifying the style of the legend box
    Outputs :
        ax          The plt.Axes instance on which the poloidal plot was performed
    """
    if Test:
        Lax, C0, C1, C2 = _check_Lax(ax,n=1)
        assert C0 or C1, 'Arg ax should a plt.Axes instance !'
        assert dP is None or type(dP) is dict, "Arg dP must be a dictionary !"
        assert dLeg is None or type(dLeg) is dict, "Arg dLeg must be a dictionary !"
        assert Ang in ['theta','xi'], "Arg Ang must be in ['theta','xi'] !"
        assert AngUnit in ['rad','deg'], "Arg AngUnit must be in ['rad','deg'] !"
    C0 = issubclass(lS.__class__, utils.ToFuObject)
    C1 = (isinstance(lS,list)
          and all([issubclass(ss.__class__, utils.ToFuObject) for ss in lS]))
    msg = "Arg lves must be a Struct subclass or a list of such !"
    assert C0 or C1, msg
    if C0:
        lS = [lS]
    nS = len(lS)

    # Get Sketch
    if ax is None:
        if wintit is None:
            wintit = _wintit
        ax, axsketch = _def.Plot_Impact_DefAxes('Cross', fs=fs, wintit=wintit,
                                                Ang=Ang, AngUnit=AngUnit,
                                                Sketch=Sketch)

    if dP is not None:
        dp = dP

    # Get up/down limits
    pPmax, pPmin = 0, 0
    for ss in lS:
        pmax = np.max(ss.dsino['EnvMinMax'])
        if pmax>pPmax:
            pPmax = pmax
        pmin = np.min(ss.dsino['EnvMinMax'])
        if pmin<pPmin:
            pPmin = pmin
    if nS>0:
        DoUp = (pPmin,pPmax)
        nP = pmax.size

    handles, labels = ax.get_legend_handles_labels()
    for ii in range(0,nS):

        Theta, pP = lS[ii].dsino['EnvTheta'], lS[ii].dsino['EnvMinMax'][0,:]
        pN = lS[ii].dsino['EnvMinMax'][1,:]
        if Ang=='xi':
            Theta, pP, pN = _GG.ConvertImpact_Theta2Xi(Theta, pP, pN)
        Theta = Theta.ravel()

        if dP is None:
            dp = {'facecolor':lS[ii].get_color(), 'edgecolor':'k',
                  'linewidth':1., 'linestyle':'-'}

        if lS[ii]._InOut=='in':
            ax.fill_between(Theta, pP, DoUp[1]*np.ones((nP,)),**dp)
            ax.fill_between(Theta, DoUp[0]*np.ones((nP,)), pN,**dp)
        elif lS[ii]._InOut=='out':
            ax.fill_between(Theta, pP, pN, **dp)
        else:
            msg = "self._InOut not defined for {0}".format(lS[ii].Id.Cls)
            raise Exception(msg)
        proxy = plt.Rectangle((0,0),1,1, fc=dp['facecolor'])
        handles.append(proxy)
        labels.append(lS[ii].Id.Cls+' '+lS[ii].Id.Name)

    if nS>0:
        ax.set_ylim(DoUp)
    if not dLeg is None:
        ax.legend(handles,labels,**dLeg)
    if draw:
        ax.figure.canvas.draw()
    return ax


# Deprecated ?
def Plot_Impact_3DPoly(T, Leg="", ax=None, Ang=_def.TorPAng,
                       AngUnit=_def.TorPAngUnit, Pdict=_def.TorP3DFilld,
                       dLeg=_def.TorLegd,
                       draw=True, fs=None, wintit=_wintit, Test=True):
    """ Plotting the toroidal projection of a Ves instance

    D. VEZINET, Aug. 2014
    Inputs :
        T           A Ves instance
        Leg         A str (the legend label to be used if T is not a Ves instance)
        ax          A plt.Axes instance (if given) on which to plot the projection space, otherwise ('None') a new figure/axes is created
        Dict        A dictionnary specifying the style of the boundary polygon plot
        dLeg        A dictionnary specifying the style of the legend box
    Outputs :
        ax          The plt.Axes instance on which the poloidal plot was performed
    """

    if Test:
        assert T.Id.Cls in ['Ves','Struct'] or (isinstance(T,tuple) and len(T)==3), "Arg T must be Ves instance or tuple with (Theta,pP,pN) 3 ndarrays !"
        assert isinstance(ax,Axes3D) or ax is None, "Arg ax must be a Axes instance !"
        assert type(Pdict) is dict, "Arg Pdict must be a dictionary !"
        assert type(dLeg) is dict or dLeg is None, "Arg dLeg must be a dictionary !"
        assert Ang in ['theta','xi'], "Arg Ang must be in ['theta','xi'] !"
        assert AngUnit in ['rad','deg'], "Arg AngUnit must be in ['rad','deg'] !"
    if ax is None:
        ax = _def.Plot_Impact_DefAxes('3D', fs=fs, wintit=wintit)
    handles, labels = ax.get_legend_handles_labels()
    if T.Id.Cls == "Ves":
        Leg = T.Id.NameLTX
        Theta, pP, pN = T._Imp_EnvTheta, T._Imp_EnvMinMax[0,:], T._Imp_EnvMinMax[1,:]
    else:
        assert isinstance(T[0],np.ndarray) and isinstance(T[1],np.ndarray) and isinstance(T[2],np.ndarray), "Args Theta, pP and pN should be np.ndarrays !"
        assert T[0].shape==T[1].shape==T[2].shape, "Args Theta, pP and pN must have same shape !"
        Theta, pP, pN = T
    AngName = r"$\theta$"
    if Ang=='xi':
        Theta, pP, pN = _GG.ConvertImpact_Theta2Xi(Theta, pP, pN)
        AngName = r"$\xi$"
    yDoUp, zDoUp = ax.get_ylim(), ax.get_zlim()
    x = np.outer(Theta.flatten(),np.ones(zDoUp.shape))
    yP = np.outer(pP.flatten(),np.ones(zDoUp.shape))
    yN = np.outer(pN.flatten(),np.ones(zDoUp.shape))
    z = np.outer(np.ones(pP.flatten().shape),zDoUp)
    ax.plot_surface(x,yP,z,rstride=1,cstride=1,label=Leg,**Pdict)
    ax.plot_surface(x,yN,z,rstride=1,cstride=1,label=Leg,**Pdict)
    proxy = plt.Rectangle((0,0),1,1, fc=Pdict['color'])
    handles.append(proxy)
    labels.append(Leg)
    ax.set_xticks([0,np.pi/4.,np.pi/2.,3.*np.pi/4.,np.pi])
    ax.set_zticks([-np.pi/2.,-np.pi/4.,0.,np.pi/4.,np.pi/2.])
    if AngUnit=='rad':
        ax.set_xticklabels([r"$0$",r"$\pi/4$",r"$\pi/2$",r"$3\pi/4$",r"$\pi$"])
        ax.set_zticklabels([r"$-\pi/2$",r"$-\pi/4$",r"$0$",r"$\pi/4$",r"$\pi/2$"])
        AngUnit = r"$(rad.)$"
    elif AngUnit=='deg':
        ax.set_xticklabels([r"$0$",r"$90$",r"$180$",r"$270$",r"$360$"])
        ax.set_zticklabels([r"$-180$",r"$-90$",r"$0$",r"$90$",r"$180$"])
        AngUnit = r"$(deg.)$"
    ax.set_xlabel(AngName+r" "+AngUnit)
    ax.set_zlabel(r"$\phi$ "+AngUnit)
    if not dLeg is None:
        ax.legend(handles,labels,**dLeg)
    if draw:
        ax.figure.canvas.draw()
    return ax



############################################
#       Phi Theta Prof dist plotting
############################################


def Config_phithetaproj_dist(config, refpt, dist, indStruct,
                             distonly=False,
                             cmap=None, vmin=None, vmax=None, invertx=None,
                             ax=None, fs=None, cbck=(0.8,0.8,0.8,0.8),
                             tit=None, wintit=None, legend=None, draw=None):
    if cmap is None:
        cmap = 'touch'
    lS = config.lStruct
    indsu = np.unique(indStruct)
    if invertx is None:
        invertx = True

    # set extent
    ratio = refpt[0] / np.nanmin(dist)
    extent = np.pi*np.r_[-1., 1., -1.,1.]

    # set colors
    vmin = np.nanmin(dist) if vmin is None else vmin
    vmax = np.nanmax(dist) if vmax is None else vmax
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    colshape = (dist.shape[0], dist.shape[1], 4)
    if cmap == 'touch':
        cols = np.array(np.broadcast_to(mpl.colors.to_rgba(cbck), colshape))
        for ii in indsu:
            ind = indStruct == ii
            cols[ind,:] = np.r_[mpl.colors.to_rgba(lS[ii].get_color())][None,None,:]
    else:
        cols = np.tile(mpl.colors.to_rgba(cmap), colshape)
    cols[:,:,-1] = 1.-norm(dist)


    # Plotting
    if not distonly or ax is None:
        fig, dax = _def._Config_phithetaproj_default()
    if tit is not None:
        fig.suptitle(tit)

    dax['dist'][0].imshow(cols, extent=extent, aspect='auto',
                          interpolation='nearest', origin='lower', zorder=-1)

    dax['cross'][0], dax['hor'][0] = config.plot(lax=[dax['cross'][0],
                                                      dax['hor'][0]],
                                                 draw=False)
    dax['dist'][0].set_xlim(np.pi*np.r_[-1.,1.])
    dax['dist'][0].set_ylim(np.pi*np.r_[-1.,1.])
    dax['dist'][0].set_aspect(aspect=1./ratio)



    # legend proxy
    # if legend != False:
        # handles, labels = dax['cross'][0].get_legend_handles_labels()
        # for ii in indsu:
            # handles.append( mRectangle((0.,0.), 1, 1, fc=lS[ii].get_color()) )
            # labels.append( '%s_%s'%(lS[ii].Id.Cls, lS[ii].Id.Name) )
        # dax['cross'][0].legend(handles, labels, frameon=False,
                               # bbox_to_anchor=(1.01,1.), loc=2, borderaxespad=0.)
    if invertx is True:
        dax['dist'][0].invert_xaxis()

    if draw:
        fig.canvas.draw()
    return dax



############################################
#       Solid angles - particles
############################################


def Config_plot_solidangle_map_particle(
    config=None,
    part_traj=None,
    part_radius=None,
    ptsRZ=None,
    sang=None,
    indices=None,
    reseff=None,
    vmin=None,
    vmax=None,
    scale=None,
    fs=None,
    dmargin=None,
):

    # ---------
    # Prepare
    npart = sang.shape[1]

    if scale is None:
        scale = 'lin'
    if scale == 'log':
        sang2 = np.full(sang.shape, np.nan)
        indok = (~np.isnan(sang)) & (sang>0.)
        sang2[indok] = np.log10(sang[indok])
        sang = sang2

    if vmin  is None:
        vmin = min(0., np.nanmin(sang))
    if vmax is None:
        vmax = np.nanmax(sang)
    if vmax is False:
        vmax = None

    if fs is None:
        fs = (12, 6)
    if dmargin is None:
        dmargin = {
            'left': 0.06, 'right': 0.95,
            'bottom': 0.06, 'top': 0.85,
            'wspace': 0.03, 'hspace': 0.1,
        }

    # -----------------
    # reshape RZ
    R = np.unique(ptsRZ[0, :])
    Z = np.unique(ptsRZ[1, :])
    dR2 = np.mean(np.diff(R))/2.
    dZ2 = np.mean(np.diff(Z))/2.

    shape = tuple([npart, Z.size, R.size])
    sangmap = np.full(shape, np.nan)
    for ii in range(sang.shape[0]):
        iR = (R == ptsRZ[0, ii]).nonzero()[0][0]
        iZ = (Z == ptsRZ[1, ii]).nonzero()[0][0]
        sangmap[:, iZ, iR] = sang[ii, :]

    extent = (R.min()-dR2, R.max()+dR2, Z.min()-dZ2, Z.max()+dZ2)

    # -----------------
    # plot
    fig = plt.figure(figsize=fs)
    gs = gridspec.GridSpec(1, npart, **dmargin)
    dax = {}
    sharex, sharey = None, None
    for ii in range(npart):
        k0 = f'map{ii}'
        dax[k0] = fig.add_subplot(gs[0, ii], sharex=sharex, sharey=sharey)
        dax[k0] = config.plot(lax=dax[k0], proj='cross', dLeg=False)
        obj = dax[k0].imshow(
            sangmap[ii, :, :],
            origin='lower',
            interpolation='nearest',
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        dax[k0].set_title(f'radius = {part_radius[ii]}')
        dax[k0].set_xlabel(r'$R$ (m)')
        if ii == 0:
            dax[k0].set_ylabel(r'$Z$ (m)')
        if ii == 0:
            sharex = dax[k0]
            sharey = dax[k0]
            dax[k0].set_aspect('equal')
    return dax


"""
###############################################################################
###############################################################################
                        LOS class and functions
###############################################################################
"""

############################################
#       Utility functions
############################################

def _LOS_calc_InOutPolProj_Debug(config, Ds, us ,PIns, POuts,
                                 L=3, nptstot=None, Lim=None, Nstep=100,
                                 fs=None, wintit=_wintit, draw=True):
    # Preformat
    assert Ds.shape==us.shape==PIns.shape==POuts.shape
    if Ds.ndim==1:
        Ds, us = Ds.reshape((3,1)), us.reshape((3,1))
        PIns, POuts = PIns.reshape((3,1)), POuts.reshape((3,1))
    nP = Ds.shape[1]
    pts = (Ds[:,:,None]
           + np.r_[0., L, np.nan][None,None,:]*us[:,:,None]).reshape((3,nP*3))

    # Plot
    ax = config.plot(element='P', proj='3d', Lim=Lim, Nstep=Nstep, dLeg=None,
                     fs=fs, wintit=wintit, draw=False)
    msg = '_LOS_calc_InOutPolProj - Debugging %s / %s pts'%(str(nP),str(nptstot))
    ax.set_title(msg)
    ax.plot(pts[0,:], pts[1,:], pts[2,:], c='k', lw=1, ls='-')
    # ax.plot(PIns[0,:],PIns[1,:],PIns[2,:],
    #         c='b', ls='None', marker='o', label=r"PIn")
    # ax.plot(POuts[0,:],POuts[1,:],POuts[2,:],
    #         c='r', ls='None', marker='x', label=r"POut")
    # ax.legend(**_def.TorLegd)
    if draw:
        ax.figure.canvas.draw()


def _get_LLOS_Leg(GLLOS, Leg=None, ind=None, Val=None, Crit='Name', PreExp=None,
                  PostExp=None, Log='any', InOut='In'):

    # Get Legend
    if type(Leg) is not str:
        Leg = GLLOS.Id.NameLTX
    if not type(GLLOS) is list and GLLOS.Id.Cls=='LOS':
        GLLOS = [GLLOS]
    elif type(GLLOS) is list:
        assert all([dd.Id.Cls=='LOS' for dd in GLLOS]), "GLD must be a list of TFG.LOS instances !"
    elif GLLOS.Id.Cls=='GLOS':
        Leg = GLLOS.Id.NameLTX if Leg is None else Leg
        if ind is None and Val is None and PreExp is None and PostExp is None:
            ind = np.arange(0,GLLOS.nLOS)
        elif not ind is None:
            assert type(ind) is np.ndarray and ind.ndim==1, "Arg ind must be a np.ndarray with ndim=1 !"
            ind = ind.nonzero()[0] if ind.dtype==bool else ind
        elif not (Val is None and PreExp is None and PostExp is None):
            ind = GLLOS.select(Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut, Out=int)
        GLLOS = [GLLOS.LLOS[ii] for ii in ind]
    return GLLOS, Leg


def Get_FieldsFrom_LLOS(L,Fields):
    # Returns a list of outputs
    assert type(L) is list and all([ll.Id.Cls=='LOS' for ll in L]), "Arg L should be a list of LOS"
    assert type(Fields) is list and all([type(ff) in [str,tuple] for ff in Fields]), "Arg Fields must be a list of str or tuples !"
    Out = []
    for ii in range(len(Fields)):
        if type(Fields[ii]) is str:
            F = getattr(L[0],Fields[ii])
        else:
            F = getattr(L[0],Fields[ii][0])[Fields[ii][1]]
        if type(F) is np.ndarray:
            ndim, shape = F.ndim, F.shape
            Shape = tuple([1]+[shape[ss] for ss in range(ndim-1,-1,-1)])
            if type(Fields[ii]) is str:
                F = np.concatenate(tuple([np.resize(getattr(ll,Fields[ii]).T,Shape).T for ll in L]),axis=ndim)
            else:
                F = np.concatenate(tuple([np.resize(getattr(ll,Fields[ii][0])[Fields[ii][1]].T,Shape).T for ll in L]),axis=ndim)
        elif type(F) in [int,float,np.int64,np.float64]:
            if type(Fields[ii]) is str:
                F = np.asarray([getattr(ll,Fields[ii]) for ll in L])
            else:
                F = np.asarray([getattr(ll,Fields[ii][0])[Fields[ii][1]] for ll in L])
        else:
            if type(Fields[ii]) is str:
                for ij in range(1,len(L)):
                    F += getattr(L[ij],Fields[ii])
            else:
                for ij in range(1,len(L)):
                    F += getattr(L[ij],Fields[ii][0])[Fields[ii][1]]
        Out = Out + [F]
    return Out




############################################
#       Plotting functions
############################################



def Rays_plot(GLos, Lax=None, Proj='all', reflections=True,
              Lplot=_def.LOSLplot, element='LDIORP', element_config='P',
              Leg=None, dL=None, dPtD=_def.LOSMd,
              dPtI=_def.LOSMd, dPtO=_def.LOSMd, dPtR=_def.LOSMd,
              dPtP=_def.LOSMd, dLeg=_def.TorLegd, multi=False,
              draw=True, fs=None, wintit=None, tit=None, Test=True, ind=None):

    if Test:
        C = GLos.Id.Cls in ['Rays','CamLOS1D','CamLOS2D']
        assert C, "Arg GLos must be an object child of tfg.Rays !"
        Proj = Proj.lower()
        C = Proj in ['cross','hor','all','3d']
        assert C, "Arg Proj must be in ['cross','hor','all','3d'] !"
        Lax, C0, C1, C2 = _check_Lax(Lax, n=2)
        assert type(element) is str, "Arg element must be a str !"
        C = element_config is None or type(element_config) is str
        msg = "Arg element must be None or a str !"
        assert C, msg
        assert Lplot in ['Tot','In'], "Arg Lplot must be in ['Tot','In']"
        C = all([type(dd) is dict for dd in [dPtD,dPtI,dPtO,dPtR,dPtP]])
        assert C, "Args dPtD,dPtI,dPtO,dPtR,dPtP must all be dict !"
        assert dL is None or type(dL) is dict, "Arg dL must be None or a dict"
        assert Leg is None or type(Leg) is str, "Arg Leg must be a str !"
        assert type(dLeg) is dict or dLeg is None, 'dLeg must be dict !'
        assert type(draw) is bool, "Arg draw must be a bool !"

    if wintit is None:
        wintit = _wintit

    if element_config != '':
        Lax = GLos.config.plot(lax=Lax, element=element_config,
                               proj=Proj, indices=False, fs=fs, tit=False,
                               draw=False, dLeg=None, wintit=wintit, Test=Test)
        Lax, C0, C1, C2 = _check_Lax(Lax, n=2)

    # Select subset
    if ind is None:
        ind = np.arange(0,GLos.nRays)
    ind = np.asarray(ind)

    Leg = GLos.Id.NameLTX if Leg is None else Leg
    dL = _def.LOSLd if dL is None else dL
    if multi:
        if GLos.Id.Cls in ['Rays','LOS']:
            Leg = [None for ii in ind]
        else:
            Leg = GLos.LNames
        if 'c' in dL.keys():
            del dL['c']

    # Check sino
    if GLos._dsino['RefPt'] is None:
        element = element.replace('P','')

    if len(ind)>0 and not element=='':
        if Proj=='3d':
            Lax[0] = _Rays_plot_3D(GLos, ax=Lax[0],
                                   reflections=reflections,
                                   Elt=element, Lplot=Lplot,
                                   Leg=Leg, dL=dL, dPtD=dPtD, dPtI=dPtI,
                                   dPtO=dPtO, dPtR=dPtR, dPtP=dPtP, dLeg=None,
                                   multi=multi, ind=ind,
                                   draw=False, fs=fs, wintit=wintit, Test=Test)
        else:
            if Proj=='all' and None in Lax:
                Lax = list(_def.Plot_LOSProj_DefAxes('All',
                                                     fs=fs, wintit=wintit,
                                                     Type=GLos.config.Type))
            if Proj in ['cross','all']:
                Lax[0] = _Rays_plot_Cross(GLos, ax=Lax[0],
                                          reflections=reflections,
                                          Elt=element, Lplot=Lplot,
                                          Leg=Leg, dL=dL, dPtD=dPtD, dPtI=dPtI,
                                          dPtO=dPtO, dPtR=dPtR, dPtP=dPtP,
                                          dLeg=None, multi=multi, ind=ind,
                                          draw=False, fs=fs, wintit=wintit,
                                          Test=Test)
            if Proj in ['hor','all']:
                ii = 0 if Proj=='hor' else 1
                Lax[ii] = _Rays_plot_Hor(GLos, ax=Lax[ii],
                                         reflections=reflections,
                                         Elt=element, Lplot=Lplot,
                                         Leg=Leg, dL=dL, dPtD=dPtD, dPtI=dPtI,
                                         dPtO=dPtO, dPtR=dPtR, dPtP=dPtP,
                                         dLeg=None, multi=multi, ind=ind,
                                         draw=False, fs=fs, wintit=wintit,
                                         Test=Test)
    if dLeg is not None:
        Lax[0].legend(**dLeg)
    if draw:
        Lax[0].figure.canvas.draw()
    Lax = Lax if Proj=='all' else Lax[0]
    return Lax



def _Rays_plot_Cross(L,Leg=None, reflections=True,
                     Lplot='Tot', Elt='LDIORP',ax=None,
                     dL=_def.LOSLd, dPtD=_def.LOSMd, dPtI=_def.LOSMd,
                     dPtO=_def.LOSMd, dPtR=_def.LOSMd, dPtP=_def.LOSMd,
                     dLeg=_def.TorLegd, multi=False, ind=None,
                     draw=True, fs=None, wintit=_wintit, Test=True):
    assert ax is None or isinstance(ax,plt.Axes), 'Wrong input for ax !'
    dPts = {'D':('D',dPtD), 'I':('PkIn',dPtI), 'O':('PkOut',dPtO),
            'R':('PRMin',dPtR),'P':('RefPt',dPtP)}
    if ax is None:
        ax = _def.Plot_LOSProj_DefAxes('Cross', fs=fs, wintit=wintit,
                                       Type=L.Ves.Type)

    if 'L' in Elt:
        R, Z, _, _, _ = L._get_plotL(Lplot=Lplot, proj='Cross',
                                     reflections=reflections, ind=ind, multi=multi)
        if multi:
            for ii in range(0,len(R)):
                ax.plot(R[ii], Z[ii], label=Leg[ii], **dL)
        else:
            ax.plot(R, Z, label=Leg, **dL)

    for kk in dPts.keys():
        if kk in Elt:
            if kk=='P' and L._dsino['RefPt'] is not None:
                P = L._dsino['pts'][:,ind]
            elif kk=='D':
                P = L.D[:,ind]
            elif not (kk == 'R' and L.config.Id.Type == 'Lin'):
                P = L._dgeom[dPts[kk][0]][:,ind]
            if len(ind)==1:
                P = P.reshape((3,1))
            if L.config.Id.Type=='Tor':
                P = np.array([np.hypot(P[0,:],P[1,:]),P[2,:]])
            else:
                P = P[1:,:]
            if multi:
                for ii in range(0,len(ind)):
                    leg = kk if Leg[ii] is None else Leg[ii]+""+kk
                    ax.plot(P[0,ii],P[1,ii], label=leg, **dPts[kk][1])
            else:
                ax.plot(P[0,:],P[1,:], label=Leg, **dPts[kk][1])

    if dLeg is not None:
        ax.legend(**dLeg)
    if draw:
        ax.figure.canvas.draw()
    return ax



def _Rays_plot_Hor(L, Leg=None, reflections=True,
                   Lplot='Tot', Elt='LDIORP',ax=None,
                   dL=_def.LOSLd, dPtD=_def.LOSMd, dPtI=_def.LOSMd,
                   dPtO=_def.LOSMd, dPtR=_def.LOSMd, dPtP=_def.LOSMd,
                   dLeg=_def.TorLegd, multi=False, ind=None,
                   draw=True, fs=None, wintit=_wintit, Test=True):
    assert ax is None or isinstance(ax,plt.Axes), 'Wrong input for ax !'
    dPts = {'D':('D',dPtD), 'I':('PkIn',dPtI), 'O':('PkOut',dPtO),
            'R':('PRMin',dPtR),'P':('RefPt',dPtP)}

    if ax is None:
        ax = _def.Plot_LOSProj_DefAxes('Hor', fs=fs,
                                       wintit=wintit, Type=L.Ves.Type)
    if 'L' in Elt:
        _, _, x, y, _ = L._get_plotL(Lplot=Lplot, proj='hor',
                                     reflections=reflections, ind=ind, multi=multi)
        if multi:
            for ii in range(0,len(x)):
                ax.plot(x[ii], y[ii], label=Leg[ii], **dL)
        else:
            ax.plot(x, y, label=Leg, **dL)

    for kk in dPts.keys():
        if kk in Elt:
            if kk=='P' and L._dsino['RefPt'] is not None:
                P = L._dsino['pts'][:,ind]
            elif kk=='D':
                P = L.D[:,ind]
            elif not (kk=='R' and L.config.Id.Type=='Lin'):
                P = L._dgeom[dPts[kk][0]][:,ind]
            if len(ind)==1:
                P = P.reshape((3,1))
            if multi:
                for ii in range(0,len(ind)):
                    leg = kk if Leg[ii] is None else Leg[ii]+""+kk
                    ax.plot(P[0,ii],P[1,ii], label=leg, **dPts[kk][1])
            else:
                ax.plot(P[0,:],P[1,:], label=Leg, **dPts[kk][1])

    if dLeg is not None:
        ax.legend(**dLeg)
    if draw:
        ax.figure.canvas.draw()
    return ax



def  _Rays_plot_3D(L, Leg=None, reflections=True,
                   Lplot='Tot',Elt='LDIORr',ax=None,
                   dL=_def.LOSLd, dPtD=_def.LOSMd, dPtI=_def.LOSMd,
                   dPtO=_def.LOSMd, dPtR=_def.LOSMd, dPtP=_def.LOSMd,
                   dLeg=_def.TorLegd, multi=False, ind=None,
                   draw=True, fs=None, wintit=_wintit, Test=True):
    assert ax is None or isinstance(ax,Axes3D), 'Arg ax should be plt.Axes instance !'
    dPts = {'D':('D',dPtD), 'I':('PkIn',dPtI), 'O':('PkOut',dPtO),
            'R':('PRMin',dPtR),'P':('RefPt',dPtP)}

    if ax is None:
        ax = _def.Plot_3D_plt_Tor_DefAxes(fs=fs, wintit=wintit)

    if 'L' in Elt:
        _, _, x, y, z = L._get_plotL(Lplot=Lplot, proj='3d',
                                     reflections=reflections, ind=ind, multi=multi)
        if multi:
            for ii in range(0,len(x)):
                ax.plot(x[ii], y[ii], z[ii],
                        label=Leg[ii], **dL)
        else:
            ax.plot(x, y, z, label=Leg, **dL)

    for kk in dPts.keys():
        if kk in Elt:
            P = L._dsino['RefPt'][:,ind] if kk=='P' else L._dgeom[dPts[kk][0]][:,ind]
            if len(ind)==1:
                P = P.reshape((3,1))
            if multi:
                for ii in range(0,len(ind)):
                    leg = kk if Leg[ii] is None else Leg[ii]+""+kk
                    ax.plot(P[0,ii],P[1,ii],P[2,ii], label=leg, **dPts[kk][1])
            else:
                ax.plot(P[0,:],P[1,:],P[2,:], label=Leg, **dPts[kk][1])

    if not dLeg is None:
        ax.legend(**dLeg)
    if draw:
        ax.figure.canvas.draw()
    return ax



"""
def  Plot_3D_mlab_GLOS(L,Leg ='',Lplot='Tot',PDIOR='DIOR',fig='None', dL=dL_mlab_Def, Mdict=Mdict_mlab_Def,LegDict=LegDict_Def):
    assert isinstance(L,LOS) or isinstance(L,list) or isinstance(L,GLOS), 'Arg L should a LOS instance or a list of LOS !'
    assert Lplot=='Tot' or Lplot=='In', "Arg Lplot should be str 'Tot' or 'In' !"
    assert isinstance(PDIOR,basestring), 'Arg PDIOR should be string !'
    #assert fig=='None' or isinstance(fig,mlab.Axes), 'Arg ax should be plt.Axes instance !'
    assert type(dL) is dict and type(Mdict) is dict and type(LegDict) is dict, 'dL, Mdict and LegDict should be dictionaries !'
    LegDict['frameon'] = LegDict['frameon']=='True' or (type(LegDict['frameon']) is bool and LegDict['frameon'])
    if isinstance(L,LOS):
        L = [L]
    elif isinstance(L, GLOS):
    Leg = L.Id.NameLTX
        L = L.LLOS
    if Lplot=='Tot':
        Pfield = 'PplotOut'
    else:
        Pfield = 'PplotIn'
    PDIORind = np.array(['D' in PDIOR, 'I' in PDIOR, 'O' in PDIOR, 'R' in PDIOR],dtype=np.bool_)

    if fig=='None':
        fig = Plot_3D_mlab_Tor_DefFig()
    if Leg == '':
        for i in range(len(L)):
            P = getattr(L[i],Pfield)
            mlab.plot3d(P[0,:],P[1,:],P[2,:],name=L[i].Id.NameLTX, figure=fig, **dL)
        if np.any(PDIORind):
            for i in range(len(L)):
                P = np.concatenate((L[i].D,L[i].PIn,L[i].POut,L[i].P1Min),axis=1)
                P = P[:,PDIORind]
                mlab.points3d(P[0,:],P[1,:],P[2,:],name=L[i].Id.NameLTX+' '+PDIOR, figure=fig, **Mdict)
    else:
        Pl,Pm = np.nan*np.ones((3,1)), np.nan*np.ones((3,1))
        for i in range(len(L)):
            P = getattr(L[i],Pfield)
            Pl = np.concatenate((Pl,P,np.nan*np.ones((3,1))),axis=1)
            P = np.concatenate((L[i].D,L[i].PIn,L[i].POut,L[i].P1Min),axis=1)
            P = P[:,PDIORind]
            Pm = np.concatenate((Pm,P),axis=1)
        mlab.plot3d(Pl[0,:],Pl[1,:],Pl[2,:],name=Leg, figure=fig, **dL)
        if np.any(PDIORind):
            mlab.points3d(Pm[0,:],Pm[1,:],Pm[2,:],name=Leg+' '+PDIOR, figure=fig, **Mdict)
    #ax.legend(**LegDict)
    return fig
"""



def GLOS_plot_Sino(GLos, Proj='Cross', ax=None, Elt=_def.LOSImpElt,
                   Sketch=True, Ang=_def.LOSImpAng, AngUnit=_def.LOSImpAngUnit,
                   Leg=None, dL=_def.LOSMImpd, dVes=_def.TorPFilld,
                   dLeg=_def.TorLegd, ind=None, multi=False,
                   draw=True, fs=None, tit=None, wintit=None, Test=True):
    if Test:
        assert Proj in ['Cross','3d'], "Arg Proj must be in ['Pol','3d'] !"
        assert Ang in ['theta','xi'], "Arg Ang must be in ['theta','xi'] !"
        assert AngUnit in ['rad','deg'], "Arg Ang must be in ['rad','deg'] !"
    if wintit is None:
        wintit = _wintit
    if not GLos.dsino['RefPt'] is None:
        ax = GLos.config.plot_sino(ax=ax, dP=dVes, Ang=Ang,
                                   AngUnit=AngUnit, Sketch=Sketch,
                                   dLeg=None, draw=False, fs=fs,
                                   wintit=wintit, Test=Test)

        # Select subset
        if ind is None:
            ind = np.arange(0,GLos.nRays)
        ind = np.asarray(ind)

        Leg = GLos.Id.NameLTX if Leg is None else Leg
        dL = _def.LOSLd if dL is None else dL
        if multi:
            if GLos.Id.Cls in ['Rays','LOS']:
                Leg = [None for ii in ind]
            else:
                Leg = GLos.LNames
            if 'c' in dL.keys():
                del dL['c']

        if 'L' in Elt:
            if Proj=='Cross':
                ax = _Plot_Sinogram_CrossProj(GLos, ax=ax, Ang=Ang, AngUnit=AngUnit,
                                              Sketch=Sketch, dL=dL, LegDict=dLeg,
                                              ind=ind, draw=False, fs=fs,
                                              wintit=wintit, Test=Test)
            else:
                ax = _Plot_Sinogram_3D(GLos, ax=ax, Ang=Ang, AngUnit=AngUnit,
                                       dL=dL, ind=ind, LegDict=dLeg, draw=False,
                                       fs=fs, wintit=wintit, Test=Test)
        if draw:
            ax.figure.canvas.draw()
    return ax



def _Plot_Sinogram_CrossProj(L, ax=None, Leg ='', Ang='theta', AngUnit='rad',
                             Sketch=True, dL=_def.LOSMImpd, LegDict=_def.TorLegd,
                             ind=None, multi=False,
                             draw=True, fs=None, wintit=_wintit, Test=True):
    if Test:
        assert ax is None or isinstance(ax,plt.Axes), 'Arg ax should be Axes instance !'
    if ax is None:
        ax, axSketch = _def.Plot_Impact_DefAxes('Cross', fs=fs, wintit=wintit,
                                                Ang=Ang, AngUnit=AngUnit,
                                                Sketch=Sketch)
    Impp, Imptheta = L._dsino['p'][ind], L._dsino['theta'][ind]
    if Ang=='xi':
        Imptheta, Impp, bla = _GG.ConvertImpact_Theta2Xi(Imptheta, Impp, Impp)
    if multi:
        for ii in range(0,len(ind)):
            if not L[ii]._dsino['RefPt'] is None:
                ax.plot(Imptheta[ii],Impp[ii],label=Leg[ind[ii]], **dL)
    else:
        ax.plot(Imptheta,Impp,label=Leg, **dL)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax


def _Plot_Sinogram_3D(L,ax=None,Leg ='', Ang='theta', AngUnit='rad',
                      dL=_def.LOSMImpd, ind=None, multi=False,
                      draw=True, fs=None, wintit=_wintit, LegDict=_def.TorLegd):
    assert ax is None or isinstance(ax,plt.Axes), 'Arg ax should be Axes instance !'
    if ax is None:
        ax = _def.Plot_Impact_DefAxes('3D', fs=fs, wintit=wintit)
    Impp, Imptheta = L.sino['p'][ind], L.sino['theta'][ind]
    ImpPhi = L.sino['Phi'][ind]
    if Ang=='xi':
        Imptheta, Impp, bla = _GG.ConvertImpact_Theta2Xi(Imptheta, Impp, Impp)
    if multi:
        for ii in range(len(ind)):
            if not L[ii].Sino_RefPt is None:
                ax.plot([Imptheta[ii]], [Impp[ii]], [ImpPhi[ii]], zdir='z',
                        label=Leg[ind[ii]], **dL)
    else:
        ax.plot(Imptheta,Impp,ImpPhi, zdir='z', label=Leg, **dL)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax



########################################################
########################################################
########################################################
#           plot_touch
########################################################
########################################################


def Rays_plot_touch(cam, key=None, ind=None, quant='lengths', cdef=_cdef,
                    invert=None, Bck=True, cbck=_cbck, Lplot=None,
                    incch=[1,10], ms=4, cmap='touch', vmin=None, vmax=None,
                    fmt_ch='02.0f', labelpad=_labelpad, dmargin=None,
                    nchMax=_nchMax, lcch=_lcch, fs=None, wintit=None, tit=None,
                    fontsize=_fontsize, draw=True, connect=True):

    ########
    # Prepare
    if ind is not None:
        ind = cam._check_indch(ind, out=bool)
    if wintit is None:
        wintit = _wintit
    assert (issubclass(cam.__class__, utils.ToFuObject)
            and 'cam' in cam.Id.Cls.lower())

    nD = 2 if cam._is2D() else 1
    if nD == 2:
        invert = True if invert is None else invert


    assert type(quant) in [str,np.ndarray]
    if type(quant) is str:
        lok = ['lengths', 'indices', 'angles', 'Etendues', 'Surfaces']
        if not quant in lok:
            msg = "Valid flags for kwarg quant are:\n"
            msg += "    [" + ", ".join(lok) + "]\n"
            msg += "    Provided: %s"%quant
            raise Exception(msg)
        if quant in ['Etendues','Surfaces'] and getattr(cam,quant) is None:
            msg = "Required quantity is not set:\n"
            msg += "    self.%s = None\n"%quant
            msg += "  => use self.set_%s() first"%quant
            raise Exception(msg)
    else:
        quant = quant.ravel()
        if quant.shape != (cam.nRays,):
            msg = "Provided quant has wrong shape!\n"
            msg += "    - Expected: (%s,)"%cam.nRays
            msg += "    - Provided: %s"%quant.shape
            raise Exception(msg)

    ########
    # Plot
    out = _Cam12D_plottouch(cam, key=key, ind=ind, quant=quant, nchMax=nchMax,
                            Bck=Bck, lcch=lcch, cbck=cbck, Lplot=Lplot,
                            incch=incch, ms=ms, cmap=cmap, vmin=vmin, vmax=vmax,
                            fmt_ch=fmt_ch, invert=invert,
                            fontsize=fontsize, labelpad=labelpad,
                            fs=fs, dmargin=dmargin, wintit=wintit, tit=tit,
                            draw=draw, connect=connect, nD=nD)
    return out



def _Cam12D_plot_touch_init(fs=None, dmargin=None, fontsize=8,
                            wintit=_wintit, nchMax=_nchMax, nD=1):

    # Figure
    axCol = "w"
    if fs is None:
        fs = (10,7)
    elif type(fs) is str and fs.lower()=='a4':
        fs = (8.27,11.69)
    fig = plt.figure(facecolor=axCol,figsize=fs)
    if wintit != False:
        fig.canvas.manager.set_window_title(wintit)
    if dmargin is None:
        dmargin = {'left':0.03, 'right':0.99,
                   'bottom':0.05, 'top':0.92,
                   'wspace':None, 'hspace':0.4}

    # Axes
    gs1 = gridspec.GridSpec(6, 3, **dmargin)
    if nD == 1:
        axp = fig.add_subplot(gs1[:,:-1], fc='w')
    else:
        pos = list(gs1[5,:-1].get_position(fig).bounds)
        pos[-1] = pos[-1]/2.
        cax = fig.add_axes(pos, fc='w')
        axp = fig.add_subplot(gs1[:5,:-1], fc='w')
    axH = fig.add_subplot(gs1[0:2,2], fc='w')
    axC = fig.add_subplot(gs1[2:,2], fc='w')

    axC.set_aspect('equal', adjustable='datalim')
    axH.set_aspect('equal', adjustable='datalim')

    Ytxt = axp.get_position().bounds[1] + axp.get_position().bounds[3]
    DY = 0.02
    Xtxt = axp.get_position().bounds[0]
    DX = axp.get_position().bounds[2]
    axtxtch = fig.add_axes([Xtxt, Ytxt, DX, DY], fc='w')

    xtxt, Ytxt, dx, DY = 0.01, 0.98, 0.15, 0.02
    axtxtg = fig.add_axes([xtxt, Ytxt, dx, DY], fc='None')

    # Dict
    dax = {'X':[axp],
           'cross':[axC],
           'hor':[axH],
           'txtg':[axtxtg],
           'txtch':[axtxtch]}
    if nD == 2:
        dax['colorbar'] = [cax]

    # Formatting
    for kk in dax.keys():
        for ii in range(0,len(dax[kk])):
            dax[kk][ii].tick_params(labelsize=fontsize)
            if 'txt' in kk:
                dax[kk][ii].patch.set_alpha(0.)
                for ss in ['left','right','bottom','top']:
                    dax[kk][ii].spines[ss].set_visible(False)
                dax[kk][ii].set_xticks([]), dax[kk][ii].set_yticks([])
                dax[kk][ii].set_xlim(0,1),  dax[kk][ii].set_ylim(0,1)

    return dax


def _Cam12D_plottouch(cam, key=None, ind=None, quant='lengths', nchMax=_nchMax,
                      Bck=True, lcch=_lcch, cbck=_cbck, Lplot=None,
                      incch=[1,5], ms=4, plotmethod='imshow',
                      cmap=None, vmin=None, vmax=None,
                      fmt_ch='01.0f', invert=True, Dlab=None,
                      fontsize=_fontsize, labelpad=_labelpad,
                      fs=None, dmargin=None, wintit=_wintit, tit=None,
                      draw=True, connect=True, nD=1):

    assert plotmethod == 'imshow', "plotmethod %s not coded yet !"%plotmethod

    #########
    # Prepare
    #########
    fldict = dict(fontsize=fontsize, labelpad=labelpad)


    # ---------
    # Check nch and X
    nch = cam.nRays

    nan2 = np.full((2,1),np.nan)
    if nD == 1:
        Xlab = r"index"
        Xtype = 'x'
        DX = [-1., nch]
    else:
        x1, x2, indr, extent = cam.get_X12plot('imshow')
        if Bck:
            indbck = np.r_[indr[0,0], indr[0,-1], indr[-1,0], indr[-1,-1]]
        idx12 = id((x1,x2))
        n12 = [x1.size, x2.size]
        Xtype = 'x'

    X = np.arange(0,nch)
    idX = id(X)

    # dchans
    if key is None:
        dchans = np.arange(0,nch)
    else:
        dchans = cam.dchans(key)
    idchans = id(dchans)

    # ---------
    # Check colors

    dElt = cam.get_touch_dict(ind=ind, out=int)

    # ---------
    # Check data

    # data
    if type(quant) is str:
        if quant == 'lengths':
            if cam._isLOS():
                Dlab = r'LOS length'+r'$m$'
                data = cam.kOut-cam.kIn
                data[np.isinf(data)] = np.nan
            else:
                Dlab = r'VOS volume'+r'$m^3$'
                data = None
                raise Exception("Not coded yet !")
        elif quant == 'indices':
            Dlab = r'index' + r' ($a.u.$)'
            data = np.arange(0,cam.nRays)
        elif quant == 'angles':
            Dlab = r'angle of incidence (rad.)'
            data = np.arccos(-np.sum(cam.u*cam.dgeom['vperp'], axis=0))
            assert np.all(data >= 0.) and np.all(data <= np.pi/2.)
        else:
            data = getattr(cam, quant)
            Dlab = quant
            Dlab += r' ($m^2/sr$)' if quant == 'Etendues' else r' ($m^2$)'
    else:
        data = quant
        Dlab = '' if Dlab is None else Dlab

    iddata = id(data)

    vmin = np.nanmin(data) if vmin is None else vmin
    vmax = np.nanmax(data) if vmax is None else vmax
    if nD == 1:
        Dlim = [min(0.,vmin), max(0.,vmax)]
        Dd = [Dlim[0]-0.05*np.diff(Dlim), Dlim[1]+0.05*np.diff(Dlim)]
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        if cmap == 'touch':
            cols = cam.get_touch_colors(dElt=dElt)
        else:
            cols = np.tile(mpl.colors.to_rgba(cmap), (cam.nRays, 1)).T
        cols[-1,:] = 1.-norm(data)
        cols = np.swapaxes(cols[:,indr.T], 0,2)

    #########
    # Plot
    #########

    # Format axes
    dax = _Cam12D_plot_touch_init(fs=fs, wintit=wintit, nchMax=nchMax,
                                  dmargin=dmargin, fontsize=fontsize, nD=nD)

    fig = dax['X'][0].figure

    if tit is None:
        tit = r"%s - %s - %s"%(cam.Id.Exp, cam.Id.Diag, cam.Id.Name)
    if tit != False:
        fig.suptitle(tit)

    # -----------------
    # Plot conf and bck
    if cam.config is not None:
        out = cam.config.plot(lax=[dax['cross'][0], dax['hor'][0]],
                              element='P', tit=False, wintit=False, dLeg=None, draw=False)
        dax['cross'][0], dax['hor'][0] = out

    if cam._isLOS():
        lCross = cam._get_plotL(Lplot=Lplot, proj='cross',
                                return_pts=True, multi=True)
        lHor = cam._get_plotL(Lplot=Lplot, proj='hor',
                              return_pts=True, multi=True)
        if Bck and nD == 2:
            crossbck = [lCross[indbck[0]],nan2,lCross[indbck[1]],nan2,
                        lCross[indbck[2]],nan2,lCross[indbck[3]]]
            crossbck = np.concatenate(crossbck,axis=1)
            horbck = [lHor[indbck[0]],nan2,lHor[indbck[1]],nan2,
                      lHor[indbck[2]],nan2,lHor[indbck[3]]]
            horbck = np.concatenate(horbck,axis=1)
            dax['cross'][0].plot(crossbck[0,:], crossbck[1,:],
                                 c=cbck, ls='-', lw=1.)
            dax['hor'][0].plot(horbck[0,:], horbck[1,:],
                                 c=cbck, ls='-', lw=1.)
        elif nD == 1:
            for kn, v in dElt.items():
                if np.any(v['indok']):
                    crok = [np.concatenate((lCross[ii],nan2), axis=1)
                            for ii in v['indok']]
                    crok = np.concatenate(crok, axis=1)
                    dax['cross'][0].plot(crok[0,:],  crok[1,:],  c=v['col'], lw=1.)
                    crok = [np.concatenate((lHor[ii],nan2), axis=1)
                            for ii in v['indok']]
                    crok = np.concatenate(crok, axis=1)
                    dax['hor'][0].plot(crok[0,:],  crok[1,:],  c=v['col'], lw=1.)
                if np.any(v['indout']):
                    crout = [np.concatenate((lCross[ii],nan2), axis=1)
                             for ii in v['indout']]
                    crout = np.concatenate(crout, axis=1)
                    dax['cross'][0].plot(crout[0,:], crout[1,:], c=cbck, lw=1.)
                    crout = [np.concatenate((lHor[ii],nan2), axis=1)
                             for ii in v['indout']]
                    crout = np.concatenate(crout, axis=1)
                    dax['hor'][0].plot(crout[0,:], crout[1,:], c=cbck, lw=1.)
        lHor = np.stack(lHor)
        idlCross = id(lCross)
        idlHor = id(lHor)
    else:
        lCross, lHor = None, None

    # data, TBF
    if nD == 1:
        for kn,v in dElt.items():
            dax['X'][0].plot(X[v['indok']], data[v['indok']],
                             marker='o', ms=ms, mfc='None',
                             c=v['col'], ls='-', lw=1.)
            dax['X'][0].plot(X[v['indout']], data[v['indout']],
                             marker='o', ms=ms, mfc='None',
                             c=cbck, ls='-', lw=1.)
    elif nD == 2:
        dax['X'][0].imshow(cols, extent=extent, aspect='equal',
                           interpolation='nearest', origin='lower', zorder=-1)
        cmapdef = plt.cm.gray if cmap == 'touch' else cmap
        cb = mpl.colorbar.ColorbarBase(dax['colorbar'][0],
                                       cmap=cmapdef, norm=norm,
                                       orientation='horizontal')
        cb.set_label(Dlab)
        # Define datanorm because colorbar => xlim in (0,1)
        if dax['colorbar'][0].get_xlim() == (0.,1.):
            datanorm = np.asarray(norm(data))
        else:
            datanorm = data
        iddatanorm= id(datanorm)


    # ---------------
    # Lims and labels
    if nD == 1:
        dax['X'][0].set_xlim(DX)
        dax['X'][0].set_xlabel(Xlab, **fldict)
    else:
        dax['X'][0].set_xlim(extent[:2])
        dax['X'][0].set_ylim(extent[2:])
        if invert:
            dax['X'][0].invert_xaxis()
            dax['X'][0].invert_yaxis()

    ##################
    # Interactivity dict
    dgroup = {'channel':   {'nMax':nchMax, 'key':'f1',
                            'defid':idX, 'defax':dax['X'][0]}}

    # Group info (make dynamic in later versions ?)
    msg = '  '.join(['%s: %s'%(v['key'],k) for k, v in dgroup.items()])
    l0 = dax['txtg'][0].text(0., 0., msg,
                             color='k', fontweight='bold',
                             fontsize=6., ha='left', va='center')

    # dref
    dref = {idX:{'group':'channel', 'val':X, 'inc':incch}}

    if nD == 2:
        dref[idX]['2d'] = (x1,x2)

    # ddata
    ddat = {iddata:{'val':data, 'refids':[idX]}}
    ddat[idchans] = {'val':dchans, 'refids':[idX]}
    if lCross is not None:
        ddat[idlCross] = {'val':lCross, 'refids':[idX]}
        ddat[idlHor] = {'val':lHor, 'refids':[idX]}
    if nD == 2:
        ddat[idx12] = {'val':(x1,x2), 'refids':[idX]}
        if iddatanorm not in ddat.keys():
            ddat[iddatanorm] = {'val':datanorm, 'refids':[idX]}

    # dax
    lax_fix = [dax['cross'][0], dax['hor'][0],
               dax['txtg'][0], dax['txtch'][0]]

    dax2 = {}
    if nD == 1:
        dax2[dax['X'][0]] = {'ref':{idX:'x'}}
    else:
        dax2[dax['X'][0]] = {'ref':{idX:'2d'},'invert':invert}

    dobj = {}


    ##################
    # Populating dobj

    # -------------
    # One-shot channels
    for jj in range(0,nchMax):

        # Channel text
        l0 = dax['txtch'][0].text((0.5+jj)/nchMax, 0., r'',
                                 color=lcch[jj], fontweight='bold',
                                 fontsize=6., ha='center', va='bottom')
        dobj[l0] = {'dupdate':{'txt':{'id':idchans, 'lrid':[idX],
                                      'bstr':'{0:%s}'%fmt_ch}},
                    'drefid':{idX:jj}}
        # los
        if cam._isLOS():
            l, = dax['cross'][0].plot([np.nan,np.nan], [np.nan,np.nan],
                                      c=lcch[jj], ls='-', lw=2.)
            dobj[l] = {'dupdate':{'data':{'id':idlCross, 'lrid':[idX]}},
                        'drefid':{idX:jj}}
            l, = dax['hor'][0].plot([np.nan,np.nan], [np.nan,np.nan],
                                    c=lcch[jj], ls='-', lw=2.)
            dobj[l] = {'dupdate':{'data':{'id':idlHor, 'lrid':[idX]}},
                        'drefid':{idX:jj}}

    # -------------
    # Data-specific


    # Channel
    for jj in range(0,nchMax):

        # Channel vlines or pixels
        if nD == 1:
            l0 = dax['X'][0].axvline(np.nan, c=lcch[jj], ls='-', lw=1.)
            dobj[l0] = {'dupdate':{'xdata':{'id':idX, 'lrid':[idX]}},
                        'drefid':{idX:jj}}
        else:
            l0, = dax['X'][0].plot([np.nan],[np.nan],
                                   mec=lcch[jj], ls='None', marker='s', mew=2.,
                                   ms=ms, mfc='None', zorder=10)
            dobj[l0] = {'dupdate':{'data':{'id':idx12, 'lrid':[idX]}},
                        'drefid':{idX:jj}}

            # Channel colorbar indicators
            l0 = dax['colorbar'][0].axvline([np.nan], ls='-', c=lcch[jj])
            dobj[l0] = {'dupdate':{'xdata':{'id':iddatanorm, 'lrid':[idX]}},
                        'drefid':{idX:jj}}


    ##################
    # Instanciate KeyHandler
    can = fig.canvas
    can.draw()

    kh = utils.KeyHandler_mpl(can=can,
                              dgroup=dgroup, dref=dref, ddata=ddat,
                              dobj=dobj, dax=dax2, lax_fix=lax_fix,
                              groupinit='channel', follow=True)

    if connect:
        kh.disconnect_old()
        kh.connect()
    if draw:
        can.draw()
    return kh
