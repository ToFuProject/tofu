import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import datetime as dtm



# ToFu-specific
import General_Geom_cy as GG


C, a, NS = [3.,0.], 1., 100

marg, N = 0.01, 100
Lthet, LDthet, Lphi, Ldphi = [0.,np.pi/2.,-np.pi/2,np.pi]*2, [np.pi/4.]*8, np.linspace(0,2.*np.pi,8), [0.]*4+[np.pi/4.]*4


def makeVesPoly(NS=NS, C=C, a=a):
    theta = np.linspace(0,2.*np.pi,NS+1)
    VP = np.array([C[0]+a*np.cos(theta), C[1]+a*np.sin(theta)])
    ind = (VP[1,:]<C[1]-0.85*a).nonzero()[0]
    div = np.array([[C[0]-0.2*a, C[0], C[0]+0.2*a],[C[1]-1.2*a, C[1]-0.75*a, C[1]-1.2*a]])
    VP = np.concatenate((VP[:,:ind[0]],div,VP[:,ind[-1]+1:-1],VP[:,0:1]), axis=1)
    return VP



def get_Pb():

    Ds, dus = [], []
    # ECE on WEST
    D_RZP, A_RZP = np.array([4.264881,0.,83.5]), np.array([4.471881, 0., 83.5])
    D = np.array([D_RZP[0]*np.cos(D_RZP[2]*np.pi/180.), D_RZP[0]*np.sin(D_RZP[2]*np.pi/180.), D_RZP[1]])
    A = np.array([A_RZP[0]*np.cos(A_RZP[2]*np.pi/180.), A_RZP[0]*np.sin(A_RZP[2]*np.pi/180.), A_RZP[1]])
    u = (D-A)/np.linalg.norm(D-A)
    Ds.append(D.reshape((3,1)))
    dus.append(u.reshape((3,1)))

    Ds = np.concatenate(tuple(Ds),axis=1)
    dus = np.concatenate(tuple(dus),axis=1)

    return Ds, dus

def makeLOS(marg=marg, thet=0., Dthet=np.pi/4., phi=0., dphi=0., N=N, C=C, a=a):
    D = np.array([(C[0]+(a+marg)*np.cos(thet))*np.cos(phi), (C[0]+(a+marg)*np.cos(thet))*np.sin(phi), C[1]+(a+marg)*np.sin(thet)])
    Ref = np.array([C[0]*np.cos(phi), C[0]*np.sin(phi), C[1]])
    duRef = (Ref-D)/np.linalg.norm(Ref-D)
    if N==1:
        dus = duRef.reshape((3,1))
    else:
        ephi = np.array([-np.sin(phi),np.cos(phi),0.])
        eperp = np.cross(ephi,duRef)/np.linalg.norm(np.cross(ephi,duRef))
        epar = duRef*np.cos(dphi) + ephi*np.sin(dphi)
        DThet = np.linspace(-Dthet,Dthet,N)[np.newaxis,:]
        dus = epar[:,np.newaxis]*np.cos(DThet) + eperp[:,np.newaxis]*np.sin(DThet)
        dus = np.concatenate((duRef.reshape((3,1)),dus),axis=1)  
    dus = dus/(np.sqrt(np.sum(dus**2,axis=0))[np.newaxis,:]) 
    return D, dus



def run(marg=marg, Lthet=Lthet, LDthet=LDthet, Lphi=Lphi, Ldphi=Ldphi, N=N, NS=NS, C=C, a=a, Pb=False, Run=['original','Single','Multi','Multi_Flat'], debug=False):

    VP = makeVesPoly(NS=NS, C=C, a=a)
    VIn = np.array([-(VP[1,1:]-VP[1,:-1]), VP[0,1:]-VP[0,:-1]]) 

    Ds, dus = [],[]
    for ii in range(0,len(Lthet)):
        D, du = makeLOS(marg=marg, thet=Lthet[ii], Dthet=LDthet[ii], phi=Lphi[ii], dphi=Ldphi[ii], N=N, C=C, a=a)
        Ds.append(np.tile(D,(du.shape[1],1)).T)
        dus.append(du)        
    Ds = np.concatenate(tuple(Ds),axis=1)
    dus = np.concatenate(tuple(dus),axis=1)
    if Pb:
        ds, us = get_Pb()
        Ds = np.concatenate((Ds,ds),axis=1)
        dus = np.concatenate((dus,us),axis=1)
        NbPb = ds.shape[1]
    else:
        NbPb = 0
    NL = Ds.shape[1]

    PIn0, POut0, Vin0 = np.nan*np.ones((3,NL)), np.nan*np.ones((3,NL)), np.nan*np.ones((3,NL))
    PIn1, POut1, Vin1 = np.nan*np.ones((3,NL)), np.nan*np.ones((3,NL)), np.nan*np.ones((3,NL))
    PIn2, POut2, Vin2 = np.nan*np.ones((3,NL)), np.nan*np.ones((3,NL)), np.nan*np.ones((3,NL))

    Dt = {}
    LPIn, LPOut = [0 for mm in range(0,len(Run))], [0 for mm in range(0,len(Run))]
    for mm in range(0,len(Run)):
        if Run[mm]=='original':
            Dt[Run[mm]] = [dtm.datetime.now(),0]
            LPIn[mm], LPOut[mm] = GG.Calc_InOut_LOS_PIO(Ds, dus, VP, VIn)
            Dt[Run[mm]][1] = dtm.datetime.now()
        else:
            Dt[Run[mm]] = [dtm.datetime.now(),0]
            LPIn[mm], LPOut[mm] = GG.Calc_LOS_PInOut_New(Ds, dus, VP, VIn, mode=Run[mm])
            Dt[Run[mm]][1] = dtm.datetime.now()
        print("        "+Run[mm]+"    {0}/{1}    {2}/{1}".format(np.all(~np.isnan(LPIn[mm]),axis=0).sum(), Ds.shape[1], np.all(~np.isnan(LPOut[mm]),axis=0).sum()))
    if debug and len(Run)==2:
        dd = np.sqrt(np.sum((LPIn[0]-LPIn[1])**2,axis=0))
        ind = ((dd>1.e-8) | ((np.isnan(dd)) & (~np.isnan(LPIn[0][0,:]))) | ((np.isnan(dd)) & (~np.isnan(LPIn[1][0,:])))).nonzero()[0]
        for ii in range(0,ind.size):
            print("")
            print("D,du", Ds[:,ind[ii]], dus[:,ind[ii]])
            print("PIn, PIn", LPIn[0][:,ind[ii]], LPIn[1][:,ind[ii]]) 
        dd = np.sqrt(np.sum((LPOut[0]-LPOut[1])**2,axis=0))
        ind = ((dd>1.e-8) | ((np.isnan(dd)) & (~np.isnan(LPOut[0][0,:]))) | ((np.isnan(dd)) & (~np.isnan(LPOut[1][0,:])))).nonzero()[0]
        for ii in range(0,ind.size):
            print("")
            print("D,du", Ds[:,ind[ii]], dus[:,ind[ii]])
            print("POut, POut", LPOut[0][:,ind[ii]], LPOut[1][:,ind[ii]])

    return (Ds,dus), LPIn, LPOut, Dt, NbPb
    



def stats(marg=marg, Lthet=Lthet, LDthet=LDthet, Lphi=Lphi, Ldphi=Ldphi, N=[1,4,10,40,100,400,1000,4000], NS=[10,50,200], C=C, a=a, Pb=True, Run=['original','Single','Multi','Multi_Flat'], a4=False, save=False, draw=True):

    Nn, Ns, Nm = len(N), len(NS), len(Run)
    Nl = np.nan*np.ones((Nn,))
    Lt = np.nan*np.ones((Nn,Ns,Nm))
    Diff = ''
    for ii in range(0,Ns):
        print('')
        print('NS = {0}'.format(NS[ii]))
        for jj in range(0,Nn):
            print("    NL = {0}".format(N[jj]))
            pb = (ii==Ns-1) and (jj==0) and Pb
            (Ds,dus), LPIn, LPOut, Dt, NbPb = run(marg=marg, Lthet=Lthet, LDthet=LDthet, Lphi=Lphi, Ldphi=Ldphi, N=N[jj], NS=NS[ii], C=C, a=a, Pb=pb, Run=Run) 
            Nl[jj] = dus.shape[1]
            Lt[jj,ii,:] =  [(Dt[Run[mm]][1]-Dt[Run[mm]][0]).total_seconds() for mm in range(0,Nm)]
      
            # Compare
            LIn = [np.sqrt(np.sum((LPIn[mm]-LPIn[mm+1])**2,axis=0)) for mm in range(0,Nm-1)]
            LIn = [(LIn[mm][~np.isnan(LIn[mm])]>1.e-6).sum() for mm in range(0,Nm-1)]
            LInn = [Run[mm]+' vs '+Run[mm+1] for mm in range(0,Nm-1) if LIn[mm]>0]
            LIn = [LIn[mm] for mm in range(0,Nm-1) if LIn[mm]>0]
            LOut = [np.sqrt(np.sum((LPOut[mm]-LPOut[mm+1])**2,axis=0)) for mm in range(0,Nm-1)]
            LOut = [(LOut[mm][~np.isnan(LOut[mm])]>1.e-6).sum() for mm in range(0,Nm-1)]
            LOutn = [Run[mm]+' vs '+Run[mm+1] for mm in range(0,Nm-1) if LOut[mm]>0]
            LOut = [LOut[mm] for mm in range(0,Nm-1) if LOut[mm]>0]
            nbpb = 1 if pb else 0
            msg = ""
            if len(LInn)>0:
                msg = "\n Non In "
                for mm in range(0,len(LIn)):
                    msg += "    {0}/{1} ".format(LIn[mm],nbpb) + LInn[mm]
            if len(LOutn)>0:
                msg = "\n Non Out "
                for mm in range(0,len(LOut)):
                    msg += "    {0}/{1} ".format(LOut[mm],nbpb) + LOutn[mm]
            if len(msg)>0:
                print(msg)
    

    cols = ['r','b','g','k'] 
    mk = ['o','+','^','o','x']

    (fw,fh) = (11.7,8.3) if a4 else (14,10)
    f = plt.figure(figsize=(fw,fh), facecolor="b", frameon=False)
    ax = f.add_axes([0.1,0.1,0.8,0.8], axis_bgcolor='w', frameon=True, xscale='log', yscale='log')
    ax.set_xlabel(r"Number of LOS")
    ax.set_ylabel(r"CPU time")

    for kk in range(0,Nm):
        for ii in range(0,Ns):
            ax.plot(Nl, Lt[:,ii,kk], ls='-', lw=1., c=cols[kk], marker=mk[ii], markersize=10)
    
    # Legend proxys
    for kk in range(0,Nm):
        ax.plot([],[], ls='-', lw=1., c=cols[kk], label=Run[kk])
    for ii in range(0,Ns):
        ax.plot([],[], ls='None', lw=0., marker=mk[ii], markersize=10, c='k', label=str(NS[ii]))
    ax.legend(loc=2, prop={'size':10})

    if save:
        np.savez('./Stats_TimeLOSPInOut.npz', Lt=Lt, Nl=Nl, NS=NS, Run=Run)

    if draw:
        f.canvas.draw()
        if save:
            f.savefig('./Fig_Stats_Timing_last.png',format='png')

    return ax, Lt, Nl, NS








    

