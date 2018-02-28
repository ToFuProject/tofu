

# Built-in
import itertools as itt
import warnings


# Generic common libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mPolygon, Wedge as mWedge
from matplotlib.axes._axes import Axes
from mpl_toolkits.mplot3d import Axes3D

# ToFu-specific
try:
    import tofu.geom._def as _def
    import tofu.geom._GG as _GG
except Exception:
    from . import _def as _def
    from . import _GG as _GG



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



def Ves_plot(Ves, Lax=None, Proj='All', Elt='PIBsBvV', Pdict=None,
             Idict=_def.TorId, Bsdict=_def.TorBsd, Bvdict=_def.TorBvd,
             Vdict=_def.TorVind, IdictHor=_def.TorITord,
             BsdictHor=_def.TorBsTord, BvdictHor=_def.TorBvTord,
             Lim=_def.Tor3DThetalim, Nstep=_def.TorNTheta, LegDict=_def.TorLegd,
             draw=True, a4=False, Test=True):
    """ Plotting the toroidal projection of a Ves instance

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
    if Test:
        assert Proj in ['Cross','Hor','All','3d'], "Arg Proj must be in ['Cross','Hor','All','3d'] !"
        Lax, C0, C1, C2 = _check_Lax(Lax,n=2)
        assert type(draw) is bool, "Arg draw must be a bool !"

    if any(['P' in Elt, 'I' in Elt]):
        if Proj=='3d':
            Pdict = dict(_def.TorP3Dd) if Pdict is None else Pdict
            Lax[0] = _Plot_3D_plt_Ves(Ves,ax=Lax[0], Elt=Elt, Lim=Lim, Nstep=Nstep, Pdict=Pdict, LegDict=None, a4=a4, draw=False, Test=Test)
        else:
            if Pdict is None:
                if Ves.Id.Cls=='Ves':
                    Pdict = _def.TorPd
                else:
                    Pdict = _def.StructPd_Tor if Ves.Lim is None else _def.StructPd
            if Proj=='Cross':
                Lax[0] = _Plot_CrossProj_Ves(Ves, ax=Lax[0], Elt=Elt, Pdict=Pdict, Idict=Idict, Bsdict=Bsdict, Bvdict=Bvdict, Vdict=Vdict, LegDict=None, draw=False, a4=a4, Test=Test)
            elif Proj=='Hor':
                Lax[0] = _Plot_HorProj_Ves(Ves, ax=Lax[0], Elt=Elt, Nstep=Nstep, Pdict=Pdict, Idict=IdictHor, Bsdict=BsdictHor, Bvdict=BvdictHor, LegDict=None, draw=False, a4=a4, Test=Test)
            elif Proj=='All':
                if Lax[0] is None or Lax[1] is None:
                    Lax = list(_def.Plot_LOSProj_DefAxes('All', a4=a4, Type=Ves.Type))
                Lax[0] = _Plot_CrossProj_Ves(Ves, ax=Lax[0], Elt=Elt, Pdict=Pdict, Idict=Idict, Bsdict=Bsdict, Bvdict=Bvdict, Vdict=Vdict, LegDict=None, draw=False, a4=a4, Test=Test)
                Lax[1] = _Plot_HorProj_Ves(Ves, ax=Lax[1], Elt=Elt, Nstep=Nstep, Pdict=Pdict, Idict=IdictHor, Bsdict=BsdictHor, Bvdict=BvdictHor, LegDict=None, draw=False, a4=a4, Test=Test)
    if not LegDict is None:
        Lax[0].legend(**LegDict)
    if draw:
        Lax[0].figure.canvas.draw()
    Lax = Lax if Proj=='All' else Lax[0]
    return Lax



def _Plot_CrossProj_Ves(V, ax=None, Elt='PIBsBvV', Pdict=_def.TorPd, Idict=_def.TorId, Bsdict=_def.TorBsd, Bvdict=_def.TorBvd, Vdict=_def.TorVind, LegDict=_def.TorLegd, draw=True, a4=False, Test=True):
    """ Plot the poloidal projection of a Ves instance

    D. VEZINET, Aug. 2014
    Inputs :
    --------
        V           A Ves instance
        ax          A plt.Axes instance (if given) on which to plot, otherwise ('None') a new figure/axes is created
        Tdict       A dictionnary specifying the style of the polygon plot
        LegDict     A dictionnary specifying the style of the legend box (if None => no legend)

    Outputs :
    ---------
        ax          The plt.Axes instance on which the plot was performed
    """
    if Test:
        assert V.Id.Cls in ['Ves','Struct'], 'Arg V should a Ves instance !'
        ax, C0, C1, C2 = _check_Lax(ax,n=1)
        assert type(Pdict) is dict, 'Arg Pdict should be a dictionary !'
        assert type(Idict) is dict, "Arg Idict should be a dictionary !"
        assert type(Bsdict) is dict, "Arg Bsdict should be a dictionary !"
        assert type(Bvdict) is dict, "Arg Bvdict should be a dictionary !"
        assert type(Vdict) is dict, "Arg Vdict should be a dictionary !"
        assert type(LegDict) is dict or LegDict is None, 'Arg LegDict should be a dictionary !'
    if ax is None:
        ax = _def.Plot_LOSProj_DefAxes('Cross', a4=a4, Type=V.Type)
    if 'P' in Elt:
        if V.Id.Cls=='Ves':
            ax.plot(V.Poly[0,:],V.Poly[1,:],label=V.Id.NameLTX,**Pdict)
        elif V.Id.Cls=='Struct':
            ax.add_patch(mPolygon(V.Poly.T, closed=True, **Pdict))
    if 'I' in Elt:
        ax.plot(V.sino['RefPt'][0],V.sino['RefPt'][1], label=V.Id.NameLTX+" Imp", **Idict)
    if 'Bs' in Elt:
        ax.plot(V.geom['BaryS'][0],V.geom['BaryS'][1], label=V.Id.NameLTX+" Bs", **Bsdict)
    if 'Bv' in Elt and V.Type=='Tor':
        ax.plot(V.geom['BaryV'][0],V.geom['BaryV'][1], label=V.Id.NameLTX+" Bv", **Bvdict)
    if 'V' in Elt:
        ax.quiver(0.5*(V.Poly[0,:-1]+V.Poly[0,1:]), 0.5*(V.Poly[1,:-1]+V.Poly[1,1:]), V.geom['VIn'][0,:],V.geom['VIn'][1,:], angles='xy',scale_units='xy', label=V.Id.NameLTX+" Vin", **Vdict)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax




def _Plot_HorProj_Ves(V, ax=None, Elt='PI', Nstep=_def.TorNTheta, Pdict=_def.TorPd, Idict=_def.TorITord, Bsdict=_def.TorBsTord, Bvdict=_def.TorBvTord, LegDict=_def.TorLegd, draw=True, a4=False, Test=True):
    """ Plotting the toroidal projection of a Ves instance

    D. VEZINET, Aug. 2014
    Inputs :
        V           A Ves instance
        Nstep      An int (the number of points for evaluation of theta by np.linspace)
        ax          A plt.Axes instance (if given) on which to plot, otherwise ('None') a new figure/axes is created
        Tdict       A dictionnary specifying the style of the polygon plot
        LegDict     A dictionnary specifying the style of the legend box (if None => no legend)
    Outputs :
        ax          The plt.Axes instance on which the plot was performed
    """
    if Test:
        assert V.Id.Cls in ['Ves','Struct'], 'Arg V should a Ves instance !'
        assert type(Nstep) is int
        ax, C0, C1, C2 = _check_Lax(ax,n=1)
        assert type(Pdict) is dict, 'Arg Pdict should be a dictionary !'
        assert type(Idict) is dict, 'Arg Idict should be a dictionary !'
        assert type(LegDict) is dict or LegDict is None, 'Arg LegDict should be a dictionary !'

    if ax is None:
        ax = _def.Plot_LOSProj_DefAxes('Hor', a4=a4, Type=V.Type)
    P1Min = V.geom['P1Min']
    P1Max = V.geom['P1Max']
    if 'P' in Elt:
        if V.Id.Cls=='Ves':
            if V.Type=='Tor':
                Theta = np.linspace(0, 2*np.pi, num=Nstep, endpoint=True, retstep=False)
                lx = np.concatenate((P1Min[0]*np.cos(Theta),np.array([np.nan]),P1Max[0]*np.cos(Theta)))
                ly = np.concatenate((P1Min[0]*np.sin(Theta),np.array([np.nan]),P1Max[0]*np.sin(Theta)))
            elif V.Type=='Lin':
                lx = np.array([V.Lim[0],V.Lim[1],V.Lim[1],V.Lim[0],V.Lim[0]])
                ly = np.array([P1Min[0],P1Min[0],P1Max[0],P1Max[0],P1Min[0]])
            ax.plot(lx,ly,label=V.Id.NameLTX,**Pdict)
        elif V.Id.Cls=='Struct':
            if V.Type=='Tor':
                Theta = np.linspace(0, 2*np.pi, num=Nstep, endpoint=True, retstep=False)
                if V.Lim is None:
                    lx = np.concatenate((P1Min[0]*np.cos(Theta),P1Max[0]*np.cos(Theta[::-1])))
                    ly = np.concatenate((P1Min[0]*np.sin(Theta),P1Max[0]*np.sin(Theta[::-1])))
                    Lp = [mPolygon(np.array([lx,ly]).T, closed=True, label=V.Id.NameLTX, **Pdict)]
                else:
                    if V._Multi:
                        Lp = [mWedge((0,0), P1Max[0], V.Lim[ii][0]*180./np.pi, V.Lim[ii][1]*180./np.pi, width=P1Max[0]-P1Min[0], label=V.Id.NameLTX, **Pdict) for ii in range(0,len(V.Lim))]
                    else:
                        Lp = [mWedge((0,0), P1Max[0], V.Lim[0]*180./np.pi, V.Lim[1]*180./np.pi, width=P1Max[0]-P1Min[0], label=V.Id.NameLTX, **Pdict)]
            elif V.Type=='Lin':
                    ly = np.array([P1Min[0],P1Min[0],P1Max[0],P1Max[0],P1Min[0]])
                    if V._Multi:
                        Lp = []
                        for ii in range(0,len(V.Lim)):
                            lx = np.array([V.Lim[ii][0],V.Lim[ii][1],V.Lim[ii][1],V.Lim[ii][0],V.Lim[ii][0]])
                            Lp.append(mPolygon(np.array([lx,ly]).T, closed=True, label=V.Id.NameLTX, **Pdict))
                    else:
                        lx = np.array([V.Lim[0],V.Lim[1],V.Lim[1],V.Lim[0],V.Lim[0]])
                        Lp = [mPolygon(np.array([lx,ly]).T, closed=True, label=V.Id.NameLTX, **Pdict)]
            for pp in Lp:
                ax.add_patch(pp)
    if 'I' in Elt:
        if V.Type=='Tor':
            lx, ly = V.sino['RefPt'][0]*np.cos(Theta), V.sino['RefPt'][0]*np.sin(Theta)
        elif V.Type=='Lin':
            lx, ly = np.array([np.min(V.Lim),np.max(V.Lim)]), V.sino['RefPt'][0]*np.ones((2,))
        ax.plot(lx,ly,label=V.Id.NameLTX+" Imp",**Idict)
    if 'Bs' in Elt:
        if V.Type=='Tor':
            lx, ly = V.geom['BaryS'][0]*np.cos(Theta), V.geom['BaryS'][0]*np.sin(Theta)
        elif V.Type=='Lin':
            lx, ly = np.array([np.min(V.Lim),np.max(V.Lim)]), V.geom['BaryS'][0]*np.ones((2,))
        ax.plot(lx,ly,label=V.Id.NameLTX+" Bs", **Bsdict)
    if 'Bv' in Elt and V.Type=='Tor':
        lx, ly = V.geom['BaryV'][0]*np.cos(Theta), V.geom['BaryV'][0]*np.sin(Theta)
        ax.plot(lx,ly,label=V.Id.NameLTX+" Bv", **Bvdict)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax



def _Plot_3D_plt_Ves(V,ax=None, Elt='P', Lim=_def.Tor3DThetalim, Nstep=_def.Tor3DThetamin, Pdict=_def.TorP3Dd, LegDict=_def.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert V.Id.Cls in ['Ves','Struct'], "Arg V should be a Ves instance !"
        assert isinstance(ax,Axes3D) or ax is None, 'Arg ax should a plt.Axes instance !'
        assert hasattr(Lim,'__iter__') and len(Lim)==2, "Arg Lim should be an iterable of 2 elements !"
        assert type(Pdict) is dict and (type(LegDict) is dict or LegDict is None), "Args Pdict and LegDict should be dictionnaries !"
        assert type(Elt)is str, "Arg Elt must be a str !"
    if ax is None:
        ax = _def.Plot_3D_plt_Tor_DefAxes(a4=a4)
    Lim = [-np.inf,np.inf] if Lim is None else Lim
    if 'P' in Elt:
        handles, labels = ax.get_legend_handles_labels()
        Lim0 = V.Lim[0] if (V.Type=='Lin' and Lim[0]>V.Lim[1]) else Lim[0]
        Lim1 = V.Lim[1] if (V.Type=='Lin' and Lim[1]<V.Lim[0]) else Lim[1]
        theta = np.linspace(max(Lim0,0.),min(Lim1,2.*np.pi),Nstep).reshape((1,Nstep)) if V.Type=='Tor' else np.linspace(max(Lim0,V.Lim[0]),min(Lim1,V.Lim[1]),Nstep).reshape((1,Nstep))
        if V.Type=='Tor':
            X = np.dot(V.Poly[0:1,:].T,np.cos(theta))
            Y = np.dot(V.Poly[0:1,:].T,np.sin(theta))
            Z = np.dot(V.Poly[1:2,:].T,np.ones(theta.shape))
        elif V.Type=='Lin':
            X = np.dot(theta.reshape((Nstep,1)),np.ones((1,V.Poly.shape[1]))).T
            Y = np.dot(V.Poly[0:1,:].T,np.ones((1,Nstep)))
            Z = np.dot(V.Poly[1:2,:].T,np.ones((1,Nstep)))
        ax.plot_surface(X,Y,Z, label=V.Id.NameLTX, **Pdict)
        proxy = plt.Rectangle((0,0),1,1, fc=Pdict['color'])
        handles.append(proxy)
        labels.append(V.Id.NameLTX)
    if not LegDict is None:
        ax.legend(handles, labels, **LegDict)
    if draw:
        ax.figure.canvas.draw()
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




def Plot_Impact_PolProjPoly(T, Leg="", ax=None, Ang='theta', AngUnit='rad',
                            Sketch=True, Pdict=_def.TorPFilld,
                            dLeg=_def.TorLegd, draw=True, a4=False, Test=True):
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
        assert T.Id.Cls in ['Ves','Struct'] or (isinstance(T,tuple) and len(T)==3), "Arg T must be Ves instance or tuple with (Theta,pP,pN) 3 np.ndarrays !"
        Lax, C0, C1, C2 = _check_Lax(ax,n=1)
        assert C0 or C1, 'Arg ax should a plt.Axes instance !'
        assert type(Pdict) is dict, "Arg Pdict must be a dictionary !"
        assert dLeg is None or type(dLeg) is dict, "Arg dLeg must be a dictionary !"
        assert Ang in ['theta','xi'], "Arg Ang must be in ['theta','xi'] !"
        assert AngUnit in ['rad','deg'], "Arg AngUnit must be in ['rad','deg'] !"
    if ax is None:
        ax, axsketch = _def.Plot_Impact_DefAxes('Cross', a4=a4, Ang=Ang, AngUnit=AngUnit, Sketch=Sketch)

    if type(T) is tuple:
        assert isinstance(T[0],np.ndarray) and isinstance(T[1],np.ndarray) and isinstance(T[2],np.ndarray), "Args Theta, pP and pN should be np.ndarrays !"
        assert T[0].shape==T[1].shape==T[2].shape, "Args Theta, pP and pN must have same shape !"
        Theta, pP, pN = T
    elif T.Id.Cls in ['Ves','Struct']:
        Leg = T.Id.NameLTX
        Theta, pP, pN = T.sino['EnvTheta'], T.sino['EnvMinMax'][0,:], T.sino['EnvMinMax'][1,:]
    if Ang=='xi':
        Theta, pP, pN = _GG.ConvertImpact_Theta2Xi(Theta, pP, pN)
    DoUp = (pN.min(),pP.max())
    handles, labels = ax.get_legend_handles_labels()
    ax.fill_between(Theta.flatten(),pP.flatten(),DoUp[1]*np.ones(pP.flatten().shape),**Pdict)
    ax.fill_between(Theta.flatten(),DoUp[0]*np.ones(pP.flatten().shape),pN.flatten(),**Pdict)
    ax.set_ylim(DoUp)
    proxy = plt.Rectangle((0,0),1,1, fc=Pdict['facecolor'])
    handles.append(proxy)
    labels.append(Leg)
    if not dLeg is None:
        ax.legend(handles,labels,**dLeg)
    if draw:
        ax.figure.canvas.draw()
    return ax



def Plot_Impact_3DPoly(T, Leg="", ax=None, Ang=_def.TorPAng,
                       AngUnit=_def.TorPAngUnit, Pdict=_def.TorP3DFilld,
                       dLeg=_def.TorLegd, draw=True, a4=False, Test=True):
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
        ax = _def.Plot_Impact_DefAxes('3D', a4=a4)
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

def _LOS_calc_InOutPolProj_Debug(Ves, Ds, us ,PIns, POuts, L=3):
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
    ax = Ves.plot(Elt='P', Proj='3d', dLeg=None)
    ax.set_title('_LOS_calc_InOutPolProj / Debugging')
    ax.plot(l0,l1,l2, c='k', lw=1, ls='-')
    ax.plot(PIns[0,:],PIns[1,:],PIns[2,:], c='b', ls='None', marker='o', label=r"PIn")
    ax.plot(POuts[0,:],POuts[1,:],POuts[2,:], c='r', ls='None', marker='x', label=r"POut")
    #ax.legend(**_def.TorLegd)
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



def Rays_plot(GLos, Lax=None, Proj='All', Lplot=_def.LOSLplot, Elt='LDIORP',
              EltVes='', EltStruct='', Leg=None, dL=None, dPtD=_def.LOSMd,
              dPtI=_def.LOSMd, dPtO=_def.LOSMd, dPtR=_def.LOSMd,
              dPtP=_def.LOSMd, dLeg=_def.TorLegd, dVes=_def.Vesdict,
              dStruct=_def.Structdict, multi=False, draw=True, a4=False,
              Test=True, ind=None):

    if Test:
        C = GLos.Id.Cls in ['Rays','LOS','LOSCam1D','LOSCam2D']
        assert C, "Arg GLos must be an object child of tfg.Rays !"
        C = Proj.lower() in ['cross','hor','all','3d']
        assert C, "Arg Proj must be in ['Cross','Hor','All','3d'] !"
        Lax, C0, C1, C2 = _check_Lax(Lax, n=2)
        assert type(Elt) is str, "Arg Elt must be a str !"
        assert Lplot in ['Tot','In'], "Arg Lplot must be in ['Tot','In']"
        C = all([type(dd) is dict for dd in [dPtD,dPtI,dPtO,dPtR,dPtP]])
        assert C, "Args dPtD,dPtI,dPtO,dPtR,dPtP must all be dict !"
        assert dL is None or type(dL) is dict, "Arg dL must be None or a dict"
        assert Leg is None or type(Leg) is str, "Arg Leg must be a str !"
        assert type(dLeg) is dict or dLeg is None, 'dLeg must be dict !'
        assert type(draw) is bool, "Arg draw must be a bool !"

    if GLos.Ves is not None:
        if EltVes is None:
            if (not 'Elt' in dVes.keys() or dVes['Elt'] is None):
                dVes['Elt'] = ''
        else:
            dVes['Elt'] = EltVes
        dVes['Lax'], dVes['Proj'], dVes['dLeg'] = Lax, Proj, None
        dVes['draw'], dVes['a4'], dVes['Test'] = False, a4, Test
        Lax = GLos.Ves.plot(**dVes)
        Lax, C0, C1, C2 = _check_Lax(Lax, n=2)

    if GLos.LStruct is not None:
        if EltStruct is None:
            if (not 'Elt' in dStruct.keys() or dStruct['Elt'] is None):
                dStruct['Elt'] = ''
        else:
            dStruct['Elt'] = EltStruct
        dStruct['Lax'], dStruct['Proj'], dStruct['dLeg'] = Lax, Proj, None
        dStruct['draw'], dStruct['a4'], dStruct['Test'] = False, a4, Test
        for ii in range(0,len(GLos.LStruct)):
            Lax = GLos.LStruct[ii].plot(**dStruct)
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
    if GLos._sino is None:
        Elt = Elt.replace('P','')

    if len(ind)>0 and not Elt=='':
        if Proj=='3d':
            Lax[0] = _Rays_plot_3D(GLos, ax=Lax[0], Elt=Elt, Lplot=Lplot,
                                   Leg=Leg, dL=dL, dPtD=dPtD, dPtI=dPtI,
                                   dPtO=dPtO, dPtR=dPtR, dPtP=dPtP, dLeg=None,
                                   multi=multi, ind=ind,
                                   draw=False, a4=a4, Test=Test)
        else:
            if Proj=='All' and None in Lax:
                Lax = list(_def.Plot_LOSProj_DefAxes('All', a4=a4,
                                                     Type=GLos.Ves.Type))
            if Proj in ['Cross','All']:
                Lax[0] = _Rays_plot_Cross(GLos, ax=Lax[0], Elt=Elt, Lplot=Lplot,
                                          Leg=Leg, dL=dL, dPtD=dPtD, dPtI=dPtI,
                                          dPtO=dPtO, dPtR=dPtR, dPtP=dPtP,
                                          dLeg=None, multi=multi, ind=ind,
                                          draw=False,a4=a4,Test=Test)
            if Proj in ['Hor','All']:
                ii = 0 if Proj=='Hor' else 1
                Lax[ii] = _Rays_plot_Hor(GLos, ax=Lax[ii], Elt=Elt, Lplot=Lplot,
                                         Leg=Leg, dL=dL, dPtD=dPtD, dPtI=dPtI,
                                         dPtO=dPtO, dPtR=dPtR, dPtP=dPtP,
                                         dLeg=None, multi=multi, ind=ind,
                                         draw=False, a4=a4, Test=Test)
    if dLeg is not None:
        Lax[0].legend(**dLeg)
    if draw:
        Lax[0].figure.canvas.draw()
    Lax = Lax if Proj=='All' else Lax[0]
    return Lax



def _Rays_plot_Cross(L,Leg=None,Lplot='Tot', Elt='LDIORP',ax=None,
                     dL=_def.LOSLd, dPtD=_def.LOSMd, dPtI=_def.LOSMd,
                     dPtO=_def.LOSMd, dPtR=_def.LOSMd, dPtP=_def.LOSMd,
                     dLeg=_def.TorLegd, multi=False, ind=None,
                     draw=True, a4=False, Test=True):
    assert ax is None or isinstance(ax,plt.Axes), 'Wrong input for ax !'
    dPts = {'D':('D',dPtD), 'I':('PIn',dPtI), 'O':('POut',dPtO),
            'R':('PRMin',dPtR),'P':('Pt',dPtP)}
    if ax is None:
        ax = _def.Plot_LOSProj_DefAxes('Cross', a4=a4, Type=L.Ves.Type)

    if 'L' in Elt:
        pts = L._get_plotL(Lplot=Lplot, Proj='Cross', ind=ind, multi=multi)
        if multi:
            for ii in range(0,len(pts)):
                ax.plot(pts[ii][0,:], pts[ii][1,:], label=Leg[ii], **dL)
        else:
            ax.plot(pts[0,:], pts[1,:], label=Leg, **dL)

    for kk in dPts.keys():
        if kk in Elt:
            P = L._sino['Pt'][:,ind] if kk=='P' else L.geom[dPts[kk][0]][:,ind]
            if len(ind)==1:
                P = P.reshape((3,1))
            if L.Ves.Type=='Tor':
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
                   draw=True, a4=False, Test=True):
    assert ax is None or isinstance(ax,plt.Axes), 'Wrong input for ax !'
    dPts = {'D':('D',dPtD), 'I':('PIn',dPtI), 'O':('POut',dPtO),
            'R':('PRMin',dPtR),'P':('Pt',dPtP)}

    if ax is None:
        ax = _def.Plot_LOSProj_DefAxes('Hor', a4=a4, Type=L.Ves.Type)
    if 'L' in Elt:
        pts = L._get_plotL(Lplot=Lplot, Proj='Hor', ind=ind, multi=multi)
        if multi:
            for ii in range(0,len(pts)):
                ax.plot(pts[ii][0,:], pts[ii][1,:], label=Leg[ii], **dL)
        else:
            ax.plot(pts[0,:], pts[1,:], label=Leg, **dL)

    for kk in dPts.keys():
        if kk in Elt:
            P = L._sino['Pt'][:,ind] if kk=='P' else L.geom[dPts[kk][0]][:,ind]
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
                   draw=True, a4=False, Test=True):
    assert ax is None or isinstance(ax,Axes3D), 'Arg ax should be plt.Axes instance !'
    dPts = {'D':('D',dPtD), 'I':('PIn',dPtI), 'O':('POut',dPtO),
            'R':('PRMin',dPtR),'P':('Pt',dPtP)}

    if ax is None:
        ax = _def.Plot_3D_plt_Tor_DefAxes(a4=a4)

    if 'L' in Elt:
        pts = L._get_plotL(Lplot=Lplot, Proj='3d', ind=ind, multi=multi)
        if multi:
            for ii in range(0,len(pts)):
                ax.plot(pts[ii][0,:], pts[ii][1,:], pts[ii][2,:],
                        label=Leg[ii], **dL)
        else:
            ax.plot(pts[0,:], pts[1,:], pts[2,:], label=Leg, **dL)

    for kk in dPts.keys():
        if kk in Elt:
            P = L._sino['Pt'][:,ind] if kk=='P' else L.geom[dPts[kk][0]][:,ind]
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
                   dLeg=_def.TorLegd, ind=None, multi=False, draw=True, a4=False,
                   Test=True):
    if Test:
        assert Proj in ['Cross','3d'], "Arg Proj must be in ['Pol','3d'] !"
        assert Ang in ['theta','xi'], "Arg Ang must be in ['theta','xi'] !"
        assert AngUnit in ['rad','deg'], "Arg Ang must be in ['rad','deg'] !"
    if not GLos.sino['RefPt'] is None:
        if 'V' in Elt:
            ax = GLos.Ves.plot_sino(ax=ax, Proj=Proj, Pdict=dVes, Ang=Ang,
                                    AngUnit=AngUnit, Sketch=Sketch,
                                    LegDict=None, draw=False, a4=a4, Test=Test)

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
                                              ind=ind, draw=False, a4=a4, Test=Test)
            else:
                ax = _Plot_Sinogram_3D(GLos, ax=ax, Ang=Ang, AngUnit=AngUnit, dL=dL,
                                       ind=ind, LegDict=dLeg, draw=False, a4=a4, Test=Test)
        if draw:
            ax.figure.canvas.draw()
    return ax



def _Plot_Sinogram_CrossProj(L, ax=None, Leg ='', Ang='theta', AngUnit='rad',
                             Sketch=True, dL=_def.LOSMImpd, LegDict=_def.TorLegd,
                             ind=None, multi=False, draw=True, a4=False, Test=True):
    if Test:
        assert ax is None or isinstance(ax,plt.Axes), 'Arg ax should be Axes instance !'
    if ax is None:
        ax, axSketch = _def.Plot_Impact_DefAxes('Cross', a4=a4, Ang=Ang, AngUnit=AngUnit, Sketch=Sketch)
    Impp, Imptheta = L.sino['p'][ind], L.sino['theta'][ind]
    if Ang=='xi':
        Imptheta, Impp, bla = _GG.ConvertImpact_Theta2Xi(Imptheta, Impp, Impp)
    if multi:
        for ii in range(0,len(ind)):
            if not L[ii]._sino['RefPt'] is None:
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
                      draw=True, a4=False, LegDict=_def.TorLegd):
    assert ax is None or isinstance(ax,plt.Axes), 'Arg ax should be Axes instance !'
    if ax is None:
        ax = _def.Plot_Impact_DefAxes('3D', a4=a4)
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
