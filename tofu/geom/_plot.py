

# Built-in
import itertools as itt
import warnings


# Generic common libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon as mPolygon, Wedge as mWedge

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
        assert type(ax) is plt.Axes or ax is None, 'Arg ax should a plt.Axes instance !'
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
                ly = np.array([P1Min[0],P1Min[0],P1Max[1],P1Max[1],P1Min[0]])
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







"""
###############################################################################
###############################################################################
                        LOS class and functions
###############################################################################
"""

############################################
#       Utility functions
############################################

def _LOS_calc_InOutPolProj_Debug(Los,PIn,POut):
    axP, axT = Los.Ves.plot()
    axP.set_title('LOS '+ Los.Id.NameLTX + ' / _LOS_calc_InOutPolProj / Debugging')
    axT.set_title('LOS '+ Los.Id.NameLTX + ' / _LOS_calc_InOutPolProj / Debugging')
    P = np.array([Los.D, Los.D+2*Los.u]).T
    axP.plot(np.sqrt(P[0,:]**2+P[1,:]**2),P[2,:],color='k',ls='solid',marker='x',markersize=8,mew=2,label=Los.Id.NameLTX)
    axP.plot([np.sqrt(PIn[0]**2+PIn[1]**2), np.sqrt(POut[0]**2+POut[1]**2)], [PIn[2],POut[2]], 'or', label=r"PIn, POut")
    axT.plot(P[0,:],P[1,:],color='k',ls='solid',marker='x',markersize=8,mew=2,label=Los.Id.NameLTX)
    axT.plot([PIn[0],POut[0]], [PIn[1],POut[1]], 'or', label=r"PIn, POut")
    axP.legend(**_def.TorLegd), axT.legend(**_def.TorLegd)
    axP.figure.canvas.draw()
    print("")
    print("Debugging...")
    print("    LOS.D, LOS.u = ", Los.D, Los.u)
    print("    PIn, POut = ", PIn, POut)
    assert not (np.any(np.isnan(PIn)) or np.any(np.isnan(POut))), "Error in computation of In/Out points !"


def _get_LLOS_Leg(GLLOS, Leg=None,
        ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):

    # Convert to list of Detect, with common legend if GDetect

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



def GLLOS_plot(GLos, Lax=None, Proj='All', Lplot=_def.LOSLplot, Elt='LDIORr', EltVes='', Leg=None,
            Ldict=_def.LOSLd, MdictD=_def.LOSMd, MdictI=_def.LOSMd, MdictO=_def.LOSMd, MdictR=_def.LOSMd, MdictP=_def.LOSMd, LegDict=_def.TorLegd,
            Vesdict=_def.Vesdict, draw=True, a4=False, Test=True,
            ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):

    if Test:
        assert type(GLos) is list or GLos.Id.Cls in ['LOS','GLOS'], "Arg GLos must be a LOS or a GLOS instance !"
        assert Proj in ['Cross','Hor','All','3d'], "Arg Proj must be in ['Cross','Hor','All','3d'] !"
        assert Lax is None or type(Lax) in [list,tuple,plt.Axes,Axes3D], "Arg Lax must be a plt.Axes or a list of such !"
        assert type(draw) is bool, "Arg draw must be a bool !"

    GLos, Leg = _get_LLOS_Leg(GLos, Leg, ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)

    if EltVes is None:
        Vesdict['Elt'] = '' if (not 'Elt' in Vesdict.keys() or Vesdict['Elt'] is None) else Vesdict['Elt']
    else:
        Vesdict['Elt'] = EltVes
    Vesdict['Lax'], Vesdict['Proj'], Vesdict['LegDict'] = Lax, Proj, None
    Vesdict['draw'], Vesdict['a4'], Vesdict['Test'] = False, a4, Test
    Lax = GLos[0].Ves.plot(**Vesdict)

    Lax = list(Lax) if hasattr(Lax,'__iter__') else [Lax]

    if not Elt=='':
        if Proj=='3d':
            Lax[0] = _Plot_3D_plt_GLOS(GLos, ax=Lax[0], Elt=Elt, Lplot=Lplot, Leg=Leg, Ldict=Ldict, MdictD=MdictD, MdictI=MdictI, MdictO=MdictO, MdictR=MdictR, MdictP=MdictP, LegDict=None, draw=False, a4=a4, Test=Test)
        else:
            if Proj=='Cross':
                Lax[0] = _Plot_CrossProj_GLOS(GLos, ax=Lax[0], Elt=Elt, Lplot=Lplot, Leg=Leg, Ldict=Ldict, MdictD=MdictD, MdictI=MdictI, MdictO=MdictO, MdictR=MdictR, MdictP=MdictP, LegDict=None, draw=False, a4=a4, Test=Test)
            elif Proj=='Hor':
                Lax[0] = _Plot_HorProj_GLOS(GLos, ax=Lax[0], Elt=Elt, Lplot=Lplot, Leg=Leg, Ldict=Ldict, MdictD=MdictD, MdictI=MdictI, MdictO=MdictO, MdictR=MdictR, MdictP=MdictP, LegDict=None, draw=False, a4=a4, Test=Test)
            elif Proj=='All':
                if Lax[0] is None or Lax[1] is None:
                    Lax = list(_def.Plot_LOSProj_DefAxes('All', a4=a4, Type=GLos[0].Ves.Type))
                Lax[0] = _Plot_CrossProj_GLOS(GLos,ax=Lax[0],Leg=Leg,Lplot=Lplot,Elt=Elt,Ldict=Ldict,MdictD=MdictD,MdictI=MdictI,MdictO=MdictO,MdictR=MdictR,MdictP=MdictP,LegDict=LegDict, draw=draw, a4=a4, Test=Test)
                Lax[1] = _Plot_HorProj_GLOS(GLos,ax=Lax[1],Leg=Leg,Lplot=Lplot,Elt=Elt,Ldict=Ldict,MdictD=MdictD,MdictI=MdictI,MdictO=MdictO,MdictR=MdictR,MdictP=MdictP,LegDict=LegDict, draw=draw, a4=a4, Test=Test)
    if not LegDict is None:
        Lax[0].legend(**LegDict)
    if draw:
        Lax[0].figure.canvas.draw()
    Lax = Lax if Proj=='All' else Lax[0]
    return Lax



def _Plot_CrossProj_GLOS(L,Leg=None,Lplot='Tot',Elt='LDIORP',ax=None, Ldict=_def.LOSLd, MdictD=_def.LOSMd, MdictI=_def.LOSMd, MdictO=_def.LOSMd, MdictR=_def.LOSMd, MdictP=_def.LOSMd, LegDict=_def.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(L) is list or L.Id.Cls in ['LOS','GLOS'], 'Arg L should a LOS instance or a list of LOS !'
        assert Lplot=='Tot' or Lplot=='In', "Arg Lplot should be str 'Tot' or 'In' !"
        assert type(Elt) is str, 'Arg Elt must be str !'
        assert ax is None or isinstance(ax,plt.Axes), 'Wrong input for Arg4 !'
        assert all([type(Di) is dict for Di in [Ldict,MdictD,MdictI,MdictO,MdictR,MdictP]]) and (type(LegDict) is dict or LegDict is None), 'Ldict, MdictD,MdictI,MdictO,MdictR,MdictP and LegDict should be dictionaries !'
    if not type(L) is list and L.Id.Cls=='LOS':
        L = [L]
    elif not type(L) is list and L.Id.Cls=='GLOS':
        Leg = L.Id.NameLTX
        L = L.LLOS
    Pfield = 'kplotTot' if Lplot=='Tot' else 'kplotIn'
    DIORrFields = ['D','I','O','R','P']
    DIORrAttr = ['D','PIn','POut','PRMin','Sino_P']
    DIORrind = np.array([Let in Elt for Let in DIORrFields],dtype=bool)
    Mdict = [MdictD, MdictI, MdictO, MdictR, MdictP]
    nDIORr = np.sum(DIORrind)
    DIORrInd = DIORrind.nonzero()[0]
    if ax is None:
        ax = _def.Plot_LOSProj_DefAxes('Cross', a4=a4, Type=L.Ves.Type)
    if Leg is None:
        if 'L' in Elt:
            if L[0].Ves.Type=='Tor':
                for ll in L:
                    P = ll.D[:,np.newaxis] + ll.geom[Pfield][np.newaxis,:]*ll.u[:,np.newaxis]
                    ax.plot(np.hypot(P[0,:],P[1,:]),P[2,:],label=ll.Id.NameLTX, **Ldict)
            elif L[0].Ves.Type=='Lin':
                for ll in L:
                    P = ll.D[:,np.newaxis] + ll.geom[Pfield][np.newaxis,:]*ll.u[:,np.newaxis]
                    ax.plot(P[1,:],P[2,:],label=ll.Id.NameLTX, **Ldict)
        if np.any(DIORrind):
            if L[0].Ves.Type=='Tor':
                for jj in range(0,nDIORr):
                    for ll in L:
                        P = ll.geom[DIORrAttr[DIORrInd[jj]]] if not DIORrFields[jj]=='P' else ll._sino['Pt']
                        ax.plot(np.hypot(P[0],P[1]),P[2],label=ll.Id.NameLTX+" "+DIORrFields[DIORrInd[jj]], **Mdict[DIORrInd[jj]])
            elif L[0].Ves.Type=='Lin':
                for jj in range(0,nDIORr):
                    for ll in L:
                        P = ll.geom[DIORrAttr[DIORrInd[jj]]] if not DIORrFields[jj]=='P' else ll._sino['Pt']
                        ax.plot(P[1],P[2],label=ll.Id.NameLTX+" "+DIORrFields[DIORrInd[jj]], **Mdict[DIORrInd[jj]])
    else:
        if 'L' in Elt:
            P = [[ll.D[:,np.newaxis] + ll.geom[Pfield][np.newaxis,:]*ll.u[:,np.newaxis],np.nan*np.ones((3,1))] for ll in L]
            P = np.concatenate(tuple(list(itt.chain.from_iterable(P))),axis=1)
            if L[0].Ves.Type=='Tor':
                ax.plot(np.hypot(P[0,:],P[1,:]),P[2,:],label=Leg, **Ldict)
            elif L[0].Ves.Type=='Lin':
                ax.plot(P[1,:],P[2,:],label=Leg, **Ldict)
        if np.any(DIORrind):
            for jj in range(0,nDIORr):
                if not DIORrFields[jj]=='P':
                    P = np.concatenate(tuple([ll.geom[DIORrAttr[DIORrInd[jj]]].reshape(3,1) for ll in L]),axis=1)
                else:
                    P = np.concatenate(tuple([ll._sino['Pt'].reshape(3,1) for ll in L]),axis=1)
            if L[0].Ves.Type=='Tor':
                ax.plot(np.hypot(P[0,:],P[1,:]),P[2,:],label=Leg+" "+DIORrFields[DIORrInd[jj]], **Mdict[DIORrInd[jj]])
            elif L[0].Ves.Type=='Lin':
                ax.plot(P[1,:],P[2,:],label=Leg+" "+DIORrFields[DIORrInd[jj]], **Mdict[DIORrInd[jj]])
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax



def _Plot_HorProj_GLOS(L, Leg=None, Lplot='Tot',Elt='LDIORP',ax=None, Ldict=_def.LOSLd, MdictD=_def.LOSMd, MdictI=_def.LOSMd, MdictO=_def.LOSMd, MdictR=_def.LOSMd, MdictP=_def.LOSMd, LegDict=_def.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(L) is list or L.Id.Cls in ['LOS','GLOS'], 'Arg L should a LOS instance or a list of LOS !'
        assert Lplot=='Tot' or Lplot=='In', "Arg Lplot should be str 'Tot' or 'In' !"
        assert type(Elt) is str, 'Arg Elt must be str !'
        assert ax is None or isinstance(ax,plt.Axes), 'Wrong input for Arg4 !'
        assert all([type(Di) is dict for Di in [Ldict,MdictD,MdictI,MdictO,MdictR,MdictP]]) and (type(LegDict) is dict or LegDict is None), 'Ldict, MdictD,MdictI,MdictO,MdictR,MdictP and LegDict should be dictionaries !'
    if not type(L) is list and L.Id.Cls=='LOS':
        L = [L]
    elif not type(L) is list and L.Id.Cls=='GLOS':
        Leg = L.Id.NameLTX
        L = L.LLOS
    Pfield = 'kplotTot' if Lplot=='Tot' else 'kplotIn'
    DIORrFields = ['D','I','O','R','P']
    DIORrAttr = ['D','PIn','POut','PRMin','Sino_P']
    DIORrind = np.array([Let in Elt for Let in DIORrFields],dtype=bool)
    Mdict = [MdictD, MdictI, MdictO, MdictR, MdictP]
    nDIORr = np.sum(DIORrind)
    DIORrInd = DIORrind.nonzero()[0]
    if ax is None:
        ax = _def.Plot_LOSProj_DefAxes('Hor', a4=a4, Type=L.Ves.Type)
    if Leg is None:
        if 'L' in Elt:
            for ll in L:
                P = ll.D[:,np.newaxis] + ll.geom[Pfield][np.newaxis,:]*ll.u[:,np.newaxis]
                ax.plot(P[0,:],P[1,:],label=ll.Id.NameLTX, **Ldict)
        if np.any(DIORrind):
            for jj in range(0,nDIORr):
                for ll in L:
                    P = ll._sino['Pt'] if DIORrFields[jj]=='P' else ll.geom[DIORrAttr[DIORrInd[jj]]]
                    ax.plot(P[0],P[1],label=ll.Id.NameLTX+" "+DIORrFields[DIORrInd[jj]], **Mdict[DIORrInd[jj]])
    else:
        if 'L' in Elt:
            P = [[ll.D[:,np.newaxis] + ll.geom[Pfield][np.newaxis,:]*ll.u[:,np.newaxis],np.nan*np.ones((3,1))] for ll in L]
            P = np.concatenate(tuple(list(itt.chain.from_iterable(P))),axis=1)
            ax.plot(P[0,:],P[1,:],label=Leg, **Ldict)
        if np.any(DIORrind):
            for jj in range(0,nDIORr):
                if not DIORrFields[jj]=='P':
                    P = np.concatenate(tuple([ll.geom[DIORrAttr[DIORrInd[jj]]].reshape(3,1) for ll in L]),axis=1)
                else:
                    P = np.concatenate(tuple([ll._sino['Pt'].reshape(3,1) for ll in L]),axis=1)
                ax.plot(P[0,:],P[1,:],label=Leg+" "+DIORrFields[DIORrInd[jj]], **Mdict[DIORrInd[jj]])
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax


def  _Plot_3D_plt_GLOS(L,Leg=None,Lplot='Tot',Elt='LDIORr',ax=None, Ldict=_def.LOSLd, MdictD=_def.LOSMd, MdictI=_def.LOSMd, MdictO=_def.LOSMd, MdictR=_def.LOSMd, MdictP=_def.LOSMd, LegDict=_def.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(L) is list or L.Id.Cls in ['LOS','GLOS'], 'Arg L should a LOS instance or a list of LOS !'
        assert Lplot=='Tot' or Lplot=='In', "Arg Lplot should be str 'Tot' or 'In' !"
        assert type(Elt) is str, 'Arg Elt should be string !'
        assert ax is None or isinstance(ax,Axes3D), 'Arg ax should be plt.Axes instance !'
        assert all([type(Di) is dict for Di in [Ldict,MdictD,MdictI,MdictO,MdictR,MdictP]]) and (type(LegDict) is dict or LegDict is None), 'Ldict, Mdict and LegDict should be dictionaries !'
    if not type(L) is list and L.Id.Cls=='LOS':
        L = [L]
    elif not type(L) is list and L.Id.Cls=='GLOS':
        Leg = L.Id.NameLTX
        L = L.LLOS
    Pfield = 'kplotTot' if Lplot=='Tot' else 'kplotIn'
    DIORrFields = ['D','I','O','R','r']
    DIORrAttr = ['D','PIn','POut','PRMin','Sino_P']
    DIORrind = np.array([Let in Elt for Let in DIORrFields],dtype=bool)
    Mdict = [MdictD, MdictI, MdictO, MdictR, MdictP]
    nDIORr = np.sum(DIORrind)
    DIORrInd = DIORrind.nonzero()[0]
    if ax is None:
        ax = _def.Plot_3D_plt_Tor_DefAxes(a4=a4)
    if Leg is None:
        if 'L' in Elt:
            for ll in L:
                P = ll.D[:,np.newaxis] + ll.geom[Pfield][np.newaxis,:]*ll.u[:,np.newaxis]
                ax.plot(P[0,:],P[1,:],P[2,:],label=ll.Id.NameLTX, **Ldict)
        if np.any(DIORrind):
            for jj in range(0,nDIORr):
                for ll in L:
                    P = ll._sino['Pt'] if DIORrFields[jj]=='P' else ll.geom[DIORrAttr[DIORrInd[jj]]]
                    ax.plot(P[0:1],P[1:2],P[2:3],label=ll.Id.NameLTX+" "+DIORrFields[DIORrInd[jj]], **Mdict[DIORrInd[jj]])
    else:
        if 'L' in Elt:
            P = [[ll.D[:,np.newaxis] + ll.geom[Pfield][np.newaxis,:]*ll.u[:,np.newaxis],np.nan*np.ones((3,1))] for ll in L]
            P = np.concatenate(tuple(list(itt.chain.from_iterable(P))),axis=1)
            ax.plot(P[0,:],P[1,:],P[2,:],label=Leg, **Ldict)
        if np.any(DIORrind):
            for jj in range(0,nDIORr):
                if not DIORrFields[jj]=='P':
                    P = np.concatenate(tuple([ll.geom[DIORrAttr[DIORrInd[jj]]].reshape(3,1) for ll in L]),axis=1)
                else:
                    P = np.concatenate(tuple([ll._sino['Pt'].reshape(3,1) for ll in L]),axis=1)
                ax.plot(P[0,:],P[1,:],P[2,:],label=Leg+" "+DIORrFields[DIORrInd[jj]], **Mdict[DIORrInd[jj]])
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax



"""
def  Plot_3D_mlab_GLOS(L,Leg ='',Lplot='Tot',PDIOR='DIOR',fig='None', Ldict=Ldict_mlab_Def, Mdict=Mdict_mlab_Def,LegDict=LegDict_Def):
    assert isinstance(L,LOS) or isinstance(L,list) or isinstance(L,GLOS), 'Arg L should a LOS instance or a list of LOS !'
    assert Lplot=='Tot' or Lplot=='In', "Arg Lplot should be str 'Tot' or 'In' !"
    assert isinstance(PDIOR,basestring), 'Arg PDIOR should be string !'
    #assert fig=='None' or isinstance(fig,mlab.Axes), 'Arg ax should be plt.Axes instance !'
    assert type(Ldict) is dict and type(Mdict) is dict and type(LegDict) is dict, 'Ldict, Mdict and LegDict should be dictionaries !'
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
            mlab.plot3d(P[0,:],P[1,:],P[2,:],name=L[i].Id.NameLTX, figure=fig, **Ldict)
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
        mlab.plot3d(Pl[0,:],Pl[1,:],Pl[2,:],name=Leg, figure=fig, **Ldict)
        if np.any(PDIORind):
            mlab.points3d(Pm[0,:],Pm[1,:],Pm[2,:],name=Leg+' '+PDIOR, figure=fig, **Mdict)
    #ax.legend(**LegDict)
    return fig
"""



def GLOS_plot_Sinogram(GLos, Proj='Cross', ax=None, Elt=_def.LOSImpElt, Sketch=True, Ang=_def.LOSImpAng, AngUnit=_def.LOSImpAngUnit, Leg=None,
            Ldict=_def.LOSMImpd, Vdict=_def.TorPFilld, LegDict=_def.TorLegd, draw=True, a4=False, Test=True,
            ind=None, Val=None, Crit='Name', PreExp=None, PostExp=None, Log='any', InOut='In'):
    if Test:
        assert Proj in ['Cross','3d'], "Arg Proj must be in ['Pol','3d'] !"
        assert Ang in ['theta','xi'], "Arg Ang must be in ['theta','xi'] !"
        assert AngUnit in ['rad','deg'], "Arg Ang must be in ['rad','deg'] !"
    if 'V' in Elt:
        ax = GLos.Ves.plot_sino(ax=ax, Proj=Proj, Pdict=Vdict, Ang=Ang, AngUnit=AngUnit, Sketch=Sketch, LegDict=None, draw=False, a4=a4, Test=Test)
    if 'L' in Elt:
        GLos, Leg = _get_LLOS_Leg(GLos, Leg, ind=ind, Val=Val, Crit=Crit, PreExp=PreExp, PostExp=PostExp, Log=Log, InOut=InOut)
        if Proj=='Cross':
            ax = _Plot_Sinogram_CrossProj(GLos, ax=ax, Ang=Ang, AngUnit=AngUnit, Sketch=Sketch, Ldict=Ldict, LegDict=LegDict, draw=False, a4=a4, Test=Test)
        else:
            ax = _Plot_Sinogram_3D(GLos, ax=ax, Ang=Ang, AngUnit=AngUnit, Ldict=Ldict, LegDict=LegDict, draw=False, a4=a4, Test=Test)
    if draw:
        ax.figure.canvas.draw()
    return ax



def _Plot_Sinogram_CrossProj(L, ax=None, Leg ='', Ang='theta', AngUnit='rad', Sketch=True, Ldict=_def.LOSMImpd, LegDict=_def.TorLegd, draw=True, a4=False, Test=True):
    if Test:
        assert type(L) is list or L.Id.Cls in ['LOS','GLOS'], "Arg L must be a GLOs, a LOS or a list of such !"
        assert ax is None or isinstance(ax,plt.Axes), 'Arg ax should be Axes instance !'
    if not type(L) is list and L.Id.Cls=='LOS':
        L = [L]
    elif not type(L) is list and L.Id.Cls=='GLOS':
        Leg = L.Id.NameLTX
        L = L.LLOS
    if ax is None:
        ax, axSketch = _def.Plot_Impact_DefAxes('Cross', a4=a4, Ang=Ang, AngUnit=AngUnit, Sketch=Sketch)
    Impp, Imptheta = Get_FieldsFrom_LLOS(L,[('_sino','p'),('_sino','theta')])
    if Ang=='xi':
        Imptheta, Impp, bla = _GG.ConvertImpact_Theta2Xi(Imptheta, Impp, Impp)
    if Leg == '':
        for ii in range(0,len(L)):
            if not L[ii]._sino['RefPt'] is None:
                ax.plot(Imptheta[ii],Impp[ii],label=L[ii].Id.NameLTX, **Ldict)
    else:
        ax.plot(Imptheta,Impp,label=Leg, **Ldict)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax


def _Plot_Sinogram_3D(L,ax=None,Leg ='', Ang='theta', AngUnit='rad', Ldict=_def.LOSMImpd, draw=True, a4=False, LegDict=_def.TorLegd):
    assert ax is None or isinstance(ax,plt.Axes), 'Arg ax should be Axes instance !'
    if not type(L) is list and L.Id.Cls=='LOS':
        L = [L]
    elif not type(L) is list and L.Id.Cls=='GLOS':
        Leg = L.Id.NameLTX
        L = L.LLOS
    if ax is None:
        ax = _def.Plot_Impact_DefAxes('3D', a4=a4)
    Impp, Imptheta, ImpPhi = Get_FieldsFrom_LLOS(L,[('_sino','p'),('_sino','theta'),('_sino','Phi')])
    if Ang=='xi':
        Imptheta, Impp, bla = _GG.ConvertImpact_Theta2Xi(Imptheta, Impp, Impp)
    if Leg == '':
        for ii in range(len(L)):
            if not L[ii].Sino_RefPt is None:
                ax.plot([Imptheta[ii]], [Impp[ii]], [ImpPhi[ii]], zdir='z', label=L[ii].Id.NameLTX, **Ldict)
    else:
        ax.plot(Imptheta,Impp,ImpPhi, zdir='z', label=Leg, **Ldict)
    if not LegDict is None:
        ax.legend(**LegDict)
    if draw:
        ax.figure.canvas.draw()
    return ax
