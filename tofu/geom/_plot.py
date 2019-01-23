

# Built-in
import os
import itertools as itt
import warnings


# Generic common libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mPolygon, Wedge as mWedge
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


__author_email__ = 'didier.vezinet@cea.fr'
_fontsize = 8
_wintit = 'tofu-{0}    {1}'.format(__version__,__author_email__)
_nchMax = 4
_cdef = (0.8,0.8,0.8)
_lcch = [plt.cm.tab20.colors[ii] for ii in [6,8,10,7,9,11]]


# Generic
def _check_Lax(Lax=None, n=2):
    assert n in [1,2]
    C0 = Lax is None
    C1 = issubclass(Lax.__class__,Axes)
    C2 = type(Lax) in [list,tuple]
    if C2:
        C2 = all([aa is None or issubclass(aa.__class__,Axes) for aa in Lax])
        C2 = C2 and len(Lax) in [1,2]
    if n==1:
        assert C0 or C1, "Arg ax must be None or a plt.Axes instance !"
    else:
        assert C0 or C1 or C2, "Arg Lax must be an Axes or a list/tuple of such !"
        if C0:
            Lax = [None,None]
        elif C1:
            Lax = [Lax,None]
        elif C2 and len(Lax)==1:
            Lax = [Lax[0],None]
        else:
            Lax = list(Lax)
    return Lax, C0, C1, C2





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



def Struct_plot(lS, lax=None, proj='all', element=None, dP=None,
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

    if tit is not None:
        lax[0].figure.suptitle(tit)

    if not dLeg is None:
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
        proxy = plt.Rectangle((0,0),1,1, fc=Pdict['color'])
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
    if isinstance(T,Ves):
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







"""
###############################################################################
###############################################################################
                        LOS class and functions
###############################################################################
"""

############################################
#       Utility functions
############################################

def _LOS_calc_InOutPolProj_Debug(Ves, Ds, us ,PIns, POuts, L=3,
                                 fs=None, wintit=_wintit, draw=True):
    # Preformat
    assert Ds.shape==us.shape==PIns.shape==POuts.shape
    if Ds.ndim==1:
        Ds, us = Ds.reshape((3,1)), us.reshape((3,1))
        PIns, POuts = PIns.reshape((3,1)), POuts.reshape((3,1))
    Ps = Ds + L*us
    nP = Ds.shape[1]
    l0 = np.array([Ds[0,:], Ps[0,:], np.full((nP,),np.nan)]).T.ravel()
    l1 = np.array([Ds[1,:], Ps[1,:], np.full((nP,),np.nan)]).T.ravel()
    l2 = np.array([Ds[2,:], Ps[2,:], np.full((nP,),np.nan)]).T.ravel()

    # Plot
    ax = Ves.plot(element='P', proj='3d', dLeg=None,
                  fs=fs, wintit=wintit, draw=False)
    ax.set_title('_LOS_calc_InOutPolProj / Debugging')
    ax.plot(l0,l1,l2, c='k', lw=1, ls='-')
    ax.plot(PIns[0,:],PIns[1,:],PIns[2,:], c='b', ls='None', marker='o', label=r"PIn")
    ax.plot(POuts[0,:],POuts[1,:],POuts[2,:], c='r', ls='None', marker='x', label=r"POut")
    #ax.legend(**_def.TorLegd)
    if draw:
        ax.figure.canvas.draw()
    print("")
    print("Debugging...")
    print("    D, u = ", Ds, us)
    print("    PIn, POut = ", PIns, POuts)


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



def Rays_plot(GLos, Lax=None, Proj='all', Lplot=_def.LOSLplot,
              element='LDIORP', element_config='P',
              Leg=None, dL=None, dPtD=_def.LOSMd,
              dPtI=_def.LOSMd, dPtO=_def.LOSMd, dPtR=_def.LOSMd,
              dPtP=_def.LOSMd, dLeg=_def.TorLegd, multi=False,
              draw=True, fs=None, wintit=None, Test=True, ind=None):

    if Test:
        C = GLos.Id.Cls in ['Rays','LOS','LOSCam1D','LOSCam2D']
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
                               proj=Proj, indices=False, fs=fs, tit=None,
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
            Lax[0] = _Rays_plot_3D(GLos, ax=Lax[0], Elt=element, Lplot=Lplot,
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
                Lax[0] = _Rays_plot_Cross(GLos, ax=Lax[0], Elt=element, Lplot=Lplot,
                                          Leg=Leg, dL=dL, dPtD=dPtD, dPtI=dPtI,
                                          dPtO=dPtO, dPtR=dPtR, dPtP=dPtP,
                                          dLeg=None, multi=multi, ind=ind,
                                          draw=False, fs=fs, wintit=wintit,
                                          Test=Test)
            if Proj in ['hor','all']:
                ii = 0 if Proj=='hor' else 1
                Lax[ii] = _Rays_plot_Hor(GLos, ax=Lax[ii], Elt=element, Lplot=Lplot,
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



def _Rays_plot_Cross(L,Leg=None,Lplot='Tot', Elt='LDIORP',ax=None,
                     dL=_def.LOSLd, dPtD=_def.LOSMd, dPtI=_def.LOSMd,
                     dPtO=_def.LOSMd, dPtR=_def.LOSMd, dPtP=_def.LOSMd,
                     dLeg=_def.TorLegd, multi=False, ind=None,
                     draw=True, fs=None, wintit=_wintit, Test=True):
    assert ax is None or isinstance(ax,plt.Axes), 'Wrong input for ax !'
    dPts = {'D':('D',dPtD), 'I':('PkMin',dPtI), 'O':('PkMax',dPtO),
            'R':('PRMin',dPtR),'P':('RefPt',dPtP)}
    if ax is None:
        ax = _def.Plot_LOSProj_DefAxes('Cross', fs=fs, wintit=wintit,
                                       Type=L.Ves.Type)

    if 'L' in Elt:
        pts = L._get_plotL(Lplot=Lplot, proj='Cross', ind=ind, multi=multi)
        if multi:
            for ii in range(0,len(pts)):
                ax.plot(pts[ii][0,:], pts[ii][1,:], label=Leg[ii], **dL)
        else:
            ax.plot(pts[0,:], pts[1,:], label=Leg, **dL)

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



def _Rays_plot_Hor(L, Leg=None, Lplot='Tot', Elt='LDIORP',ax=None,
                   dL=_def.LOSLd, dPtD=_def.LOSMd, dPtI=_def.LOSMd,
                   dPtO=_def.LOSMd, dPtR=_def.LOSMd, dPtP=_def.LOSMd,
                   dLeg=_def.TorLegd, multi=False, ind=None,
                   draw=True, fs=None, wintit=_wintit, Test=True):
    assert ax is None or isinstance(ax,plt.Axes), 'Wrong input for ax !'
    dPts = {'D':('D',dPtD), 'I':('PkMin',dPtI), 'O':('PkMax',dPtO),
            'R':('PRMin',dPtR),'P':('RefPt',dPtP)}

    if ax is None:
        ax = _def.Plot_LOSProj_DefAxes('Hor', fs=fs,
                                       wintit=wintit, Type=L.Ves.Type)
    if 'L' in Elt:
        pts = L._get_plotL(Lplot=Lplot, proj='hor', ind=ind, multi=multi)
        if multi:
            for ii in range(0,len(pts)):
                ax.plot(pts[ii][0,:], pts[ii][1,:], label=Leg[ii], **dL)
        else:
            ax.plot(pts[0,:], pts[1,:], label=Leg, **dL)

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



def  _Rays_plot_3D(L,Leg=None,Lplot='Tot',Elt='LDIORr',ax=None,
                   dL=_def.LOSLd, dPtD=_def.LOSMd, dPtI=_def.LOSMd,
                   dPtO=_def.LOSMd, dPtR=_def.LOSMd, dPtP=_def.LOSMd,
                   dLeg=_def.TorLegd, multi=False, ind=None,
                   draw=True, fs=None, wintit=_wintit, Test=True):
    assert ax is None or isinstance(ax,Axes3D), 'Arg ax should be plt.Axes instance !'
    dPts = {'D':('D',dPtD), 'I':('PkMin',dPtI), 'O':('PkMax',dPtO),
            'R':('PRMin',dPtR),'P':('RefPt',dPtP)}

    if ax is None:
        ax = _def.Plot_3D_plt_Tor_DefAxes(fs=fs, wintit=wintit)

    if 'L' in Elt:
        pts = L._get_plotL(Lplot=Lplot, proj='3d', ind=ind, multi=multi)
        if multi:
            for ii in range(0,len(pts)):
                ax.plot(pts[ii][0,:], pts[ii][1,:], pts[ii][2,:],
                        label=Leg[ii], **dL)
        else:
            ax.plot(pts[0,:], pts[1,:], pts[2,:], label=Leg, **dL)

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
                   draw=True, fs=None, wintit=None, Test=True):
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
#           plot_touch
########################################################

def _make_cmap(c):

    c0 = mpl.colors.to_rgb(c)
    dc = {'red':((0.,c0[0],c0[0]),(1.,1.,1.)),
          'green':((0,c0[1],c0[1]),(1.,1.,1.)),
          'blue':((0.,c0[2],c0[2]),(1.,1.,1.))}
    cm = mpl.colors.LinearSegmentedColormap(c, dc)
    return cm



def Rays_plot_touch(Cam, key=None, ind=None, cdef=_cdef,
                    invert=None, plotmethod='imshow',
                    nchMax=_nchMax, lcch=_lcch, fs=None, wintit=None, tit=None,
                    fontsize=_fontsize, draw=True, connect=True):

    if type(Cam) is list:
        assert all([cc.config==Cam[0].config for cc in Cam])
    if ind is not None and type(Cam) is list and len(Cam)>1:
        msg = "Cannot provide ind if Cam is a list !"
        raise Exception(msg)
    if ind is not None:
        if type(Cam) is list:
            ind = Cam[0]._check_indch(ind,out=bool)
        else:
            ind = Cam._check_indch(ind,out=bool)

    if wintit is None:
        wintit = _wintit

    if type(Cam) is list or '1D' in Cam.Id.Cls:
        if not type(Cam) is list:
            Cam = [Cam]
        out = _Cam1D_plot_touch(Cam, key=key, ind=ind, cdef=cdef,
                                fs=fs, wintit=wintit, tit=tit, draw=draw)
    else:
        invert = True if invert is None else invert
        out = _Cam2D_plot_touch(Cam, ind=ind, cdef=cdef, nchMax=nchMax,
                                invert=invert, plotmethod=plotmethod,
                                lcch=lcch, fs=fs, wintit=wintit, tit=tit,
                                fontsize=fontsize, draw=draw, connect=connect)
    return out


def  _Cam1D_plot_touch_init(fs=None, wintit=_wintit, nchMax=_nchMax):
    axCol = "w"
    if fs is None:
        fs = (10,7)
    elif type(fs) is str and fs.lower()=='a4':
        fs = (8.27,11.69)
    fig = plt.figure(facecolor=axCol,figsize=fs)
    if wintit is not None:
        fig.canvas.set_window_title(wintit)
    gs1 = gridspec.GridSpec(6, 3,
                            left=0.05, bottom=0.05, right=0.99, top=0.94,
                            wspace=None, hspace=0.4)
    axp = fig.add_subplot(gs1[:,:-1], fc='w')
    axH = fig.add_subplot(gs1[0:2,2], fc='w')
    axC = fig.add_subplot(gs1[2:,2], fc='w')

    axC.set_aspect('equal', adjustable='datalim')
    axH.set_aspect('equal', adjustable='datalim')
    dax = {'prof':[axp], '2D':[axC,axH]}
    for kk in dax.keys():
        for ii in range(0,len(dax[kk])):
            dax[kk][ii].tick_params(labelsize=8)
    return dax


def _Cam1D_plot_touch(Cam, key=None, ind=None, cdef=_cdef,
                      nchMax=_nchMax, fs=None, tit=None, wintit=_wintit, draw=True):

    # Prepare
    if 'LOS' in Cam[0].Id.Cls:
        Dname = 'LOS length'
        Dunits = r"$m$"
        data = [cc._dgeom['kMax']-cc._dgeom['kMin'] for cc in Cam]
        data = np.concatenate(tuple(data))
    else:
        Dname = 'VOS volume'
        Dunits = r"$m^3$"
        data = None
        raise Exception("Not codd yet !")
    Dd = [min(0,np.nanmin(data)), 1.2*np.nanmax(data)]

    nch = data.size
    nchmax = min(nch,50)
    if nch>nchmax:
        indchans = np.arange(0,nch, nch//nchmax)
    else:
        indchans = np.arange(0,nch)
    chans = np.arange(0,nch)
    Dchans = [-1,nch]
    if key is None:
        chlab = chans
    else:
        chlab = itt.chain.from_iterable([cc._dchans[kk] for cc in Cam])

    lS = [cc.lStruct_computeInOut for cc in Cam]
    lS = lS[0] if len(lS)==1 else list(itt.chain.from_iterable(lS))
    lSP = [os.path.join(s.Id.SavePath,s.Id.SaveName) for s in lS]
    lS = [lS[lSP.index(ss)] for ss in list(set(lSP))]

    if ind is not None:
        indl = np.zeros((Cam[0].nRays,),dtype=int)
        indl[ind] = np.arange(0,ind.sum())

    dElt = {}
    for ss in lS:
        kn = ss.Id.Cls+'_'+ss.Id.Name
        inde = []
        for cc in Cam:
            try:
                ii = cc.select(touch=kn,out=bool)
            except:
                ii = np.zeros((cc.nRays,),dtype=bool)
            inde.append(ii)
        inde = np.concatenate(tuple(inde))
        if ind is not None:
            indi = indl[inde]
            inde[:] = inde & ind
        else:
            indi = inde.nonzero()[0]
        dElt[kn] = {'inde':inde.nonzero()[0], 'indi':indi, 'c':ss.get_color()}

    nan = np.full((2,1),np.nan)
    if 'LOS' in Cam[0].Id.Cls:
        if ind is not None:
            lCross = Cam[0]._get_plotL(Lplot='In', proj='cross', ind=ind, multi=True)
            lHor = Cam[0]._get_plotL(Lplot='In', proj='hor', ind=ind, multi=True)
        else:
            lCross = [cc._get_plotL(Lplot='In', proj='Cross', multi=True)
                      for cc in Cam]
            lHor = [cc._get_plotL(Lplot='In', proj='Hor', multi=True)
                    for cc in Cam]
            lCross = list(itt.chain.from_iterable(lCross))
            lHor = list(itt.chain.from_iterable(lHor))
        for ee in dElt.keys():
            if dElt[ee]['inde'].size>0:
                cr = [np.concatenate((lCross[ii],nan),axis=1)
                      for ii in dElt[ee]['indi']]
                cr = np.concatenate(tuple(cr),axis=1)
                hh = [np.concatenate((lHor[ii],nan),axis=1)
                      for ii in dElt[ee]['indi']]
                hh = np.concatenate(tuple(hh),axis=1)
                dElt[ee]['cr'] = cr
                dElt[ee]['hh'] = hh

    else:
        raise Exception("Not coded yet !")


    # Format axes
    dax = _Cam1D_plot_touch_init(fs=fs, wintit=wintit)

    dax['prof'][0].set_xlim(Dchans),   dax['prof'][0].set_ylim(Dd)
    dax['prof'][0].set_xlabel(r"", fontsize=8)
    dax['prof'][0].set_ylabel(r"%s (%s)"%(Dname,Dunits), fontsize=8)
    dax['prof'][0].set_xticks(chans[indchans])
    dax['prof'][0].set_xticklabels([chlab[ii] for ii in indchans], rotation=45)

    # Plot fixed parts
    if Cam[0].config is not None:
        dax['2D'] = Cam[0].config.plot(lax=dax['2D'], element='P', dLeg=None)
    if tit is None:
        tit = r"%s - %s"%(Cam[0].Id.Exp,Cam[0].Id.Diag)
        if len(Cam)==1:
            tit += r" - %s"%Cam[0].Id.Name
    dax['prof'][0].figure.suptitle(tit)

    for ee in dElt.keys():
        if dElt[ee]['inde'].size>0:
            c, inde = dElt[ee]['c'], dElt[ee]['inde']
            cr, hh = dElt[ee]['cr'], dElt[ee]['hh']
            dax['prof'][0].plot(chans[inde], data[inde],
                                ls='None', marker='x', ms=8, c=c)
            if 'LOS' in Cam[0].Id.Cls:
                dax['2D'][0].plot(cr[0,:], cr[1,:], ls='-', lw=1., c=c)
                dax['2D'][1].plot(hh[0,:], hh[1,:], ls='-', lw=1., c=c)

    if draw:
        dax['prof'][0].figure.canvas.draw()
    return dax


###################################################
###################################################
#           Data2D
###################################################
###################################################

class KH2D(utils.KeyHandler):

    def __init__(self, can, daxT, ntMax=3, nchMax=3):

        utils.KeyHandler.__init__(self, can, daxT=daxT,
                                  ntMax=ntMax, nchMax=nchMax, nlambMax=1)

    def update(self):

        # Restore background
        self._update_restore_Bck(list(self.daxr.keys()))

        # Update and get lax
        lax = self._update_vlines_and_Eq()

        # Blit
        self._update_blit(lax)

def _prepare_pcolormeshimshow(X12_1d, out='imshow'):
    assert out.lower() in ['pcolormesh','imshow']
    x1, x2, ind, dX12 = utils.get_X12fromflat(X12_1d)
    if out=='pcolormesh':
        x1 = np.r_[x1-dX12[0]/2., x1[-1]+dX12[0]/2.]
        x2 = np.r_[x2-dX12[1]/2., x2[-1]+dX12[1]/2.]
    return x1, x2, ind, dX12

def _Cam2D_plot_touch_init(fs=None, dmargin=None,
                           fontsize=8, wintit=_wintit, nchMax=_nchMax):
    axCol = "w"
    if fs is None:
        fs = (10,7)
    elif type(fs) is str and fs.lower()=='a4':
        fs = (8.27,11.69)
    fig = plt.figure(facecolor=axCol,figsize=fs)
    if wintit is not None:
        fig.canvas.set_window_title(wintit)
    if dmargin is None:
        dmargin = {'left':0.03, 'right':0.99,
                   'bottom':0.05, 'top':0.92,
                   'wspace':None, 'hspace':0.4}
    gs1 = gridspec.GridSpec(6, 3, **dmargin)
    pos = list(gs1[5,:-1].get_position(fig).bounds)
    pos[-1] = pos[-1]/2.
    cax = fig.add_axes(pos, fc='w')
    axp = fig.add_subplot(gs1[:5,:-1], fc='w')
    axH = fig.add_subplot(gs1[0:2,2], fc='w')
    axC = fig.add_subplot(gs1[2:,2], fc='w')

    axC.set_aspect('equal', adjustable='datalim')
    axH.set_aspect('equal', adjustable='datalim')

    Ytxt = axp.get_position().bounds[1]+axp.get_position().bounds[3]
    DY = (axp.get_position().bounds[1]
          - cax.get_position().bounds[1] - cax.get_position().bounds[3])
    Xtxt = axp.get_position().bounds[0]
    DX = axp.get_position().bounds[2]
    axtxtch = fig.add_axes([Xtxt, Ytxt, DX, DY], fc='w')
    for ax in [axtxtch]:
        ax.patch.set_alpha(0.)
        for ss in ['left','right','bottom','top']:
            ax.spines[ss].set_visible(False)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlim(0,1),  ax.set_ylim(0,1)

    dax = {'chan2D':[{'ax':axp,'dh':{'vline':[]}}],
           'cross':[{'ax':axC, 'dh':{}}],
           'hor':[{'ax':axH, 'dh':{}}],
           'colorbar':[{'ax':cax, 'dh':{}}],
           'txtch':[{'ax':axtxtch, 'dh':{}}]}
    for kk in dax.keys():
        for ii in range(0,len(dax[kk])):
            dax[kk][ii]['ax'].tick_params(labelsize=fontsize)
    return dax



def _Cam2D_plot_touch(Cam, key=None, ind=None, ms=4, lcch=_lcch, cdef=_cdef,
                      plotmethod='scatter', invert=False, nchMax=_nchMax,
                      dmargin=None, fs=None, wintit=_wintit, tit=None,
                      fontsize=_fontsize, draw=True, connect=True):

    # Prepare
    if 'LOS' in Cam.Id.Cls:
        Dname = 'LOS length'
        Dunits = r"$m$"
        data = Cam.kMax-Cam.kMin
        data[np.isinf(data)] = np.nan
    else:
        Dname = 'VOS volume'
        Dunits = r"$m^3$"
        data = None
        raise Exception("Not coded yet !")

    nch = data.size
    chans = np.arange(0,nch)
    Dchans = [-1,nch]
    if key is None:
        chlab = chans
    else:
        chlab = Cam.dchans[kk]
    X12, DX12 = Cam.get_X12(out='1d')
    X12T = X12.T

    X1p, X2p, indp, dX12 = _prepare_pcolormeshimshow(X12, out=plotmethod)
    DX1 = [np.nanmin(X1p),np.nanmax(X1p)]
    DX2 = [np.nanmin(X2p),np.nanmax(X2p)]

    incx = {'left':np.r_[-dX12[0],0.], 'right':np.r_[dX12[0],0.],
            'down':np.r_[0.,-dX12[1]], 'up':np.r_[0.,dX12[1]]}

    if 'LOS' in Cam.Id.Cls:
        lCross = Cam._get_plotL(Lplot='In', proj='cross', multi=True)
        lHor = Cam._get_plotL(Lplot='In', proj='hor', multi=True)
    else:
        raise Exception("Not coded yet !")

    # Get colors
    datamin, datamax = np.nanmin(data), np.nanmax(data)
    norm = mpl.colors.Normalize(vmin=datamin, vmax=datamax)
    cmapdef = _make_cmap(cdef)
    scamapdef = mpl.cm.ScalarMappable(norm=norm, cmap=cmapdef)
    lS = Cam.lStruct_computeInOut
    cols = np.full((data.size,4),np.nan)
    dElt = {}
    for ss in lS:
        kn = ss.Id.Cls+'_'+ss.Id.Name
        inde = Cam.select(touch=kn,out=bool)
        if ind is not None:
            indin = inde & ind
            indout = inde & (~ind)
        c = ss.get_color()[:-1]
        cmap = _make_cmap(c)
        scamap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        if ind is None:
            cols[inde,:] = scamap.to_rgba(data[inde])
        else:
            cols[indin,:] = scamap.to_rgba(data[indin])
            cols[indout,:] = scamapdef.to_rgba(data[indout])

    # Prepare colors
    if plotmethod=='imshow':
        x1u, x2u, ind, DX12 = utils.get_X12fromflat(X12)
        nx1, nx2 = x1u.size, x2u.size
        extent = (x1u.min(),x1u.max(),x2u.min(),x2u.max())

    # Format axes
    dax = _Cam2D_plot_touch_init(fs=fs, wintit=wintit, nchMax=nchMax,
                                 dmargin=dmargin, fontsize=fontsize)

    # Plot fixed parts
    out = Cam.config.plot(lax=[dax['cross'][0]['ax'],
                               dax['hor'][0]['ax']],
                          element='P', dLeg=None, draw=False)
    dax['cross'][0]['ax'], dax['hor'][0]['ax'] = out

    if tit is None:
        tit = r"%s - %s - %s"%(Cam.Id.Exp, Cam.Id.Diag, Cam.Id.Name)
    dax['chan2D'][0]['ax'].figure.suptitle(tit)

    dax['chan2D'][0]['xref'] = X12T
    if plotmethod=='scatter':
        dax['chan2D'][0]['ax'].set_xlim(DX1)
        dax['chan2D'][0]['ax'].set_ylim(DX2)
        dax['chan2D'][0]['ax'].scatter(X12[0,:],X12[1,:], c=cols,
                                       s=8, marker='s', edgecolors='None',
                                       zorder=-1)
    elif plotmethod=='imshow':
        dax['chan2D'][0]['ax'].set_xlim(extent[0:2])
        dax['chan2D'][0]['ax'].set_ylim(extent[2:])
        cols = cols.reshape((nx1,nx2,4)).swapaxes(0,1)
        dax['chan2D'][0]['ax'].imshow(cols, extent=extent, aspect='equal',
                                      interpolation='nearest', origin='lower',
                                      zorder=-1)

    # Plot colorbar
    cb = mpl.colorbar.ColorbarBase(dax['colorbar'][0]['ax'],
                                   cmap=cmapdef, norm=norm,
                                   orientation='horizontal')
    cb.set_label(r"LOS length (m)")
    # Define datanorm because colorbar => xlim in (0,1)
    if dax['colorbar'][0]['ax'].get_xlim()==(0.,1.):
        datanorm = ((data-datamin)/(datamax-datamin))[:,np.newaxis]
    else:
        datanorm = data[:,np.newaxis]

    # Plot LOS
    if 'LOS' in Cam.Id.Cls:
        lCross = Cam._get_plotL(Lplot='In', proj='cross', multi=True)
        lHor = Cam._get_plotL(Lplot='In', proj='hor', multi=True)
        if 'Name' in Cam.dchans.keys():
            llab = [Cam.Id.Name + s for s in Cam.dchans['Name']]
        else:
            llab = [Cam.Id.Name + '-{0}'.format(ii)
                    for ii in range(0,Cam.nRays)]

        lv = []
        dlosc = {'losc':[{'h':[],'xy':lCross, 'xref':X12T}]}
        lHor= np.stack(tuple(lHor),axis=0)
        dlosh = {'losh':[{'h':[],'x':lHor[:,0,:], 'y':lHor[:,1,:], 'xref':X12T}]}
        dcolb = {'vline':[{'h':[],'x':datanorm, 'xref':X12T}]}
        dchtxt = {'txt':[{'h':[],'txt':llab, 'xref':X12T}]}
        for jj in range(0,nchMax):
            lab = r"ch{0}".format(jj)
            l, = dax['chan2D'][0]['ax'].plot([np.nan],[np.nan],
                                              mec=lcch[jj], ls='None',
                                              marker='s', mew=2.,
                                              ms=ms, mfc='None',
                                              label=lab, zorder=10)
            lv.append(l)
            l, = dax['cross'][0]['ax'].plot([np.nan,np.nan],
                                           [np.nan,np.nan],
                                           c=lcch[jj], ls='-', lw=2.)
            dlosc['losc'][0]['h'].append(l)
            l, = dax['hor'][0]['ax'].plot([np.nan,np.nan],
                                          [np.nan,np.nan],
                                          c=lcch[jj], ls='-', lw=2.)
            dlosh['losh'][0]['h'].append(l)
            l = dax['colorbar'][0]['ax'].axvline(np.nan, ls='-', lw=1,
                                                 c=lcch[jj], zorder=10)
            dcolb['vline'][0]['h'].append(l)
            l = dax['txtch'][0]['ax'].text((0.5+jj)/nchMax,0., r"",
                                       color=lcch[jj],
                                       fontweight='bold', fontsize=6.,
                                       ha='center', va='bottom')
            dchtxt['txt'][0]['h'].append(l)
        dax['chan2D'][0]['dh']['vline'] = [{'h':lv, 'xref':X12T,
                                            'trig':{}}]
        dax['hor'][0]['dh'].update(dlosh)
        dax['cross'][0]['dh'].update(dlosc)
        dax['colorbar'][0]['dh'].update(dcolb)
        dax['txtch'][0]['dh'].update(dchtxt)
        dax['chan2D'][0]['dh']['vline'][0]['trig'].update(dlosh)
        dax['chan2D'][0]['dh']['vline'][0]['trig'].update(dlosc)
        dax['chan2D'][0]['dh']['vline'][0]['trig'].update(dcolb)
        dax['chan2D'][0]['dh']['vline'][0]['trig'].update(dchtxt)
    else:
        raise Exception("Not coded yet !")
    dax['chan2D'][0]['incx'] = incx
    dax['chan2D'][0]['ax'].set_ylabel(r"pix.", fontsize=fontsize)

    dax['chan2D'][0]['ax'].set_xlabel(r"$X_1$", fontsize=8)
    dax['chan2D'][0]['ax'].set_ylabel(r"$X_2$", fontsize=8)
    if invert:
        dax['chan2D'][0]['ax'].invert_xaxis()
        dax['chan2D'][0]['ax'].invert_yaxis()
    dax['chan2D'][0]['invert'] = invert

    # Plot mobile parts
    can = dax['chan2D'][0]['ax'].figure.canvas
    can.draw()
    KH = KH2D(can, dax, nchMax=nchMax)

    if connect:
        KH.disconnect_old()
        KH.connect()
    if draw:
        can.draw()
    return KH
