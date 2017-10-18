

# Built-in
import warnings


# Generic common libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ToFu-specific
try:
    import tofu.geom._def as _def
    import tofu.geom._GG as _GG
except Exception:
    from . import _def as _def
    from . import _GG as _GG






"""
###############################################################################
###############################################################################
                        Ves class and functions
###############################################################################
"""

############################################
##### Plotting functions
############################################



def Ves_plot(Ves, Lax=None, Proj='All', Elt='PIBsBvV', Pdict=None, Idict=_def.TorId, Bsdict=_def.TorBsd, Bvdict=_def.TorBvd, Vdict=_def.TorVind,
        IdictHor=_def.TorITord, BsdictHor=_def.TorBsTord, BvdictHor=_def.TorBvTord, Lim=_def.Tor3DThetalim, Nstep=_def.TorNTheta, LegDict=_def.TorLegd, draw=True, a4=False, Test=True):
    """ Plotting the toroidal projection of a Ves instance

    D. VEZINET, Aug. 2014
    Inputs :
        V           A Ves instance
        Nstep      An int (the number of points for evaluation of theta by np.linspace)
        axP         A plt.Axes instance (if given) on which to plot the poloidal projection, otherwise ('None') a new figure/axes is created
        axT         A plt.Axes instance (if given) on which to plot the toroidal projection, otherwise ('None') a new figure/axes is created
        Tdict       A dictionnary specifying the style of the polygon plot
        LegDict     A dictionnary specifying the style of the legend box (if None => no legend)
    Outputs :
        axP          The plt.Axes instance on which the poloidal plot was performed
        axT          The plt.Axes instance on which the toroidal plot was performed
    """
    if Test:
        assert Proj in ['Cross','Hor','All','3d'], "Arg Proj must be in ['Cross','Hor','All','3d'] !"
        assert Lax is None or type(Lax) in [list,tuple,plt.Axes,Axes3D], "Arg Lax must be a plt.Axes or a list of such !"
        assert type(draw) is bool, "Arg draw must be a bool !"

    Lax = list(Lax) if hasattr(Lax,'__iter__') else [Lax]

    if any(['P' in Elt, 'I' in Elt]):
        if Proj=='3d':
            Pdict = _def.TorP3Dd if Pdict is None else Pdict
            Lax[0] = _Plot_3D_plt_Ves(Ves,ax=Lax[0], Elt=Elt, Lim=Lim, Nstep=Nstep, Pdict=Pdict, LegDict=None, a4=a4, draw=False, Test=Test)
        else:
            Pdict = _def.TorPd if Pdict is None else Pdict
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
        assert type(ax) is plt.Axes or ax is None, 'Arg ax should a plt.Axes instance !'
        assert type(Pdict) is dict, 'Arg Pdict should be a dictionary !'
        assert type(Idict) is dict, "Arg Idict should be a dictionary !"
        assert type(Bsdict) is dict, "Arg Bsdict should be a dictionary !"
        assert type(Bvdict) is dict, "Arg Bvdict should be a dictionary !"
        assert type(Vdict) is dict, "Arg Vdict should be a dictionary !"
        assert type(LegDict) is dict or LegDict is None, 'Arg LegDict should be a dictionary !'
    if ax is None:
        ax = _def.Plot_LOSProj_DefAxes('Cross', a4=a4, Type=V.Type)
    if 'P' in Elt:
        ax.plot(V.Poly[0,:],V.Poly[1,:],label=V.Id.NameLTX,**Pdict)
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
        assert type(ax) is plt.Axes or ax is None, 'Arg ax should a plt.Axes instance !'
        assert type(Pdict) is dict, 'Arg Pdict should be a dictionary !'
        assert type(Idict) is dict, 'Arg Idict should be a dictionary !'
        assert type(LegDict) is dict or LegDict is None, 'Arg LegDict should be a dictionary !'
    Theta = np.linspace(0, 2*np.pi, num=Nstep, endpoint=True, retstep=False) if V.Type=='Tor' else np.linspace(V.Lim[0],V.Lim[1],num=Nstep, endpoint=True, retstep=False)
    if ax is None:
        ax = _def.Plot_LOSProj_DefAxes('Hor', a4=a4, Type=V.Type)
    P1Min = V.geom['P1Min']
    P1Max = V.geom['P1Max']
    if 'P' in Elt:
        if V.Type=='Tor':
            lx = np.concatenate((P1Min[0]*np.cos(Theta),np.array([np.nan]),P1Max[0]*np.cos(Theta)))
            ly = np.concatenate((P1Min[0]*np.sin(Theta),np.array([np.nan]),P1Max[0]*np.sin(Theta)))
        elif V.Type=='Lin':
            lx = np.concatenate((Theta,np.array([np.nan]),Theta))
            ly = np.concatenate((P1Min[0]*np.ones((Nstep,)),np.array([np.nan]),P1Max[0]*np.ones((Nstep,))))
        ax.plot(lx,ly,label=V.Id.NameLTX,**Pdict)
    if 'I' in Elt:
        if V.Type=='Tor':
            lx, ly = V.sino['RefPt'][0]*np.cos(Theta), V.sino['RefPt'][0]*np.sin(Theta)
        elif V.Type=='Lin':
            lx, ly = Theta, V.sino['RefPt'][0]*np.ones((Nstep,))
        ax.plot(lx,ly,label=V.Id.NameLTX+" Imp",**Idict)
    if 'Bs' in Elt:
        if V.Type=='Tor':
            lx, ly = V.geom['BaryS'][0]*np.cos(Theta), V.geom['BaryS'][0]*np.sin(Theta)
        elif V.Type=='Lin':
            lx, ly = Theta, V.geom['BaryS'][0]*np.ones((Nstep,))
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




def Plot_Impact_PolProjPoly(T, Leg="", ax=None, Ang='theta', AngUnit='rad', Sketch=True, Pdict=_def.TorPFilld, LegDict=_def.TorLegd, draw=True, a4=False, Test=True):
    """ Plotting the toroidal projection of a Ves instance

    D. VEZINET, Aug. 2014
    Inputs :
        T           A Ves instance
        Leg         A str (the legend label to be used if T is not a Ves instance)
        ax          A plt.Axes instance (if given) on which to plot the projection space, otherwise ('None') a new figure/axes is created
        Dict        A dictionnary specifying the style of the boundary polygon plot
        LegDict     A dictionnary specifying the style of the legend box
    Outputs :
        ax          The plt.Axes instance on which the poloidal plot was performed
    """
    if Test:
        assert T.Id.Cls in ['Ves','Struct'] or (isinstance(T,tuple) and len(T)==3), "Arg T must be Ves instance or tuple with (Theta,pP,pN) 3 np.ndarrays !"
        assert isinstance(ax,plt.Axes) or ax is None, "Arg ax must be a Axes instance !"
        assert type(Pdict) is dict, "Arg Pdict must be a dictionary !"
        assert LegDict is None or type(LegDict) is dict, "Arg LegDict must be a dictionary !"
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
    ax.fill_between(Theta.flatten(),pP.flatten(),DoUp[1]*np.ones(pP.flatten().shape),label=Leg,**Pdict)
    ax.fill_between(Theta.flatten(),DoUp[0]*np.ones(pP.flatten().shape),pN.flatten(),label=Leg,**Pdict)
    ax.set_ylim(DoUp)
    proxy = plt.Rectangle((0,0),1,1, fc=Pdict['facecolor'])
    handles.append(proxy)
    labels.append(Leg)
    if not LegDict is None:
        ax.legend(handles,labels,**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax



def Plot_Impact_3DPoly(T, Leg="", ax=None, Ang=_def.TorPAng, AngUnit=_def.TorPAngUnit, Pdict=_def.TorP3DFilld, LegDict=_def.TorLegd, draw=True, a4=False, Test=True):
    """ Plotting the toroidal projection of a Ves instance

    D. VEZINET, Aug. 2014
    Inputs :
        T           A Ves instance
        Leg         A str (the legend label to be used if T is not a Ves instance)
        ax          A plt.Axes instance (if given) on which to plot the projection space, otherwise ('None') a new figure/axes is created
        Dict        A dictionnary specifying the style of the boundary polygon plot
        LegDict     A dictionnary specifying the style of the legend box
    Outputs :
        ax          The plt.Axes instance on which the poloidal plot was performed
    """

    if Test:
        assert isinstance(T,Ves) or (isinstance(T,tuple) and len(T)==3), "Arg T must be Ves instance or tuple with (Theta,pP,pN) 3 ndarrays !"
        assert isinstance(ax,plt.Axes) or ax is None, "Arg ax must be a Axes instance !"
        assert type(Pdict) is dict, "Arg Pdict must be a dictionary !"
        assert type(LegDict) is dict or LegDict is None, "Arg LegDict must be a dictionary !"
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
    if not LegDict is None:
        ax.legend(handles,labels,**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax




































