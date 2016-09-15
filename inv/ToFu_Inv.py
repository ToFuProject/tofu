# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 16:22:47 2014

@author: didiervezinet
"""

import numpy as np
import scipy.sparse as scpsp
import scipy.sparse.linalg as scpsplin
import scipy.optimize as scpop
import ToFu_Mesh as TFM
import ToFu_MatComp as TFMC
import ToFu_Defaults as TFD
import ToFu_PathFile as TFPF
import ToFu_Treat as TFT
import ToFu_PostTreat as TFPT
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.mlab as mlab
import matplotlib as mpl
from matplotlib import path
from matplotlib import _cntr as cntr
import datetime as dtm
import types


# Nomenclature :        'Linearity_Parameter_Version'
#
#   Linearity : Function name starts with 'InvLin' if it solves linear problems, 'InvNonLin' else
#   Parameter : A short name or acronym, sufficently explicit, to identify the method used to determine the regularisation parameter (e.g.: DisPrinc, GenCross, AugTikho...)
#   Version   : 'V' + version number of the function

# Each function should include a short description of its philosophy + one or two relevant references


####### Intermediate pre-conditioning ###########

SpFcNames = ['csr','csc','coo','bsr','dia','dok','lil']
SpFcl = [scpsp.csr_matrix, scpsp.csc_matrix, scpsp.coo_matrix, scpsp.bsr_matrix, scpsp.dia_matrix, scpsp.dok_matrix, scpsp.lil_matrix]
SpIsl = [scpsp.isspmatrix_csr, scpsp.isspmatrix_csc, scpsp.isspmatrix_coo, scpsp.isspmatrix_bsr, scpsp.isspmatrix_dia, scpsp.isspmatrix_dok, scpsp.isspmatrix_lil]

SpFc, SpIs = {}, {}
for ii in range(0,len(SpFcNames)):
    SpFc[SpFcNames[ii]] = SpFcl[ii]
    SpIs[SpFcNames[ii]] = SpIsl[ii]


def Default_sol(BF2, TMat, data0, sigma0=None, N=3):
    def sol0_Mode0(Points):
        R = np.hypot(Points[0,:],Points[1,:])
        return np.exp(-(R-1.70)**2/0.10**2 - (Points[2,:]-0.05)**2/0.15**2)

    def sol0_Mode1(Points):
        R = np.hypot(Points[0,:],Points[1,:])
        Z = Points[2,:]
        return np.exp(-(R-1.70)**2/0.20**2 - (Z-0.05)**2/0.30**2) - 0.50*np.exp(-(R-1.67)**2/0.10**2 - (Z-0.05)**2/0.15**2)

    def sol0_Mode2(Points):
        R = np.hypot(Points[0,:],Points[1,:])
        Z = Points[2,:]
        return (1.+np.tanh((R-1.35)/0.10))*(1.-np.tanh((R-1.85)/0.15)) * (1.+np.tanh((Z+0.35)/0.20))*(1.-np.tanh((Z-0.45)/0.20))

    if not sigma0 is None:
        data0bis = data0/sigma0
        TMatbis = np.diag(1./sigma0).dot(TMat)
    else:
        data0bis = data0[:]
        TMatbis = TMat[:]

    sol, a, Chi2 = np.nan*np.ones((N,TMat.shape[1])), np.nan*np.ones((N,)), np.nan*np.ones((N,))
    for ii in range(0,N):
        sol[ii,:] = BF2.get_Coefs( ff = locals()['sol0_Mode'+str(ii)] )[0]
        a[ii] = ((sol[ii,:].dot(TMatbis.T)).dot(data0bis))/((sol[ii,:].dot(TMatbis.T)).dot(TMatbis.dot(sol[ii,:])))
        Chi2[ii] = (a[ii]*TMatbis.dot(sol[ii,:])-data0bis).T.dot(a[ii]*TMatbis.dot(sol[ii,:])-data0bis)
    return a[np.nanargmin(Chi2)]*sol[np.nanargmin(Chi2),:]


##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
###########            Main routines
##############################################################################################################################################
##############################################################################################################################################




def InvChoose(TMat, data, t, BF2, sigma=None, Dt=None, mu0=TFD.mu0, SolMethod='InvLin_AugTikho_V1', Deriv='D2N2', IntMode='Vol', Cond=None, ConvCrit=TFD.ConvCrit, Sparse=True, SpType='csr', Pos=True, KWARGS=None, timeit=True, Verb=False, VerbNb=None):
    if timeit:
        t1 = dtm.datetime.now()

    t = t.flatten()
    if sigma is None:
        sigma = 0.5*np.mean(np.mean(data))*np.ones((TMat.shape[0],))
    NdimSig = sigma.ndim

    if Dt is None:
        indt = np.ones((t.size,),dtype=bool)
    else:
        if type(Dt) is float:
            indt = np.nanargmin(np.abs(t-Dt))
        elif len(Dt)==2:
            indt = (t>=Dt[0]) & (t<=Dt[1])
    t = t[indt]
    data = data[indt,:]
    if data.ndim==1:
        data = data.reshape((1,data.size))
    if NdimSig==2:
        sigma = sigma[indt,:]
    #SigMean = np.mean(np.mean(sigma))

    NMes, Nbf = TMat.shape
    Nt = t.size
    if SolMethod=='InvLin_AugTikho_V1' and Pos:
        SolMethod = 'InvLinQuad_AugTikho_V1'

    if KWARGS is None:
        if SolMethod in 'InvLin_AugTikho_V1':
            KWARGS = {'a0':TFD.AugTikho_a0, 'b0':TFD.AugTikho_b0, 'a1':TFD.AugTikho_a1, 'b1':TFD.AugTikho_b1, 'd':TFD.AugTikho_d, 'ConvReg':True, 'FixedNb':True}
        elif SolMethod == 'InvLin_DisPrinc_V1':
            KWARGS = {'chi2Tol':TFD.chi2Tol, 'chi2Obj':TFD.chi2Obj}
        elif SolMethod == 'InvLinQuad_AugTikho_V1':
            KWARGS = {'a0':TFD.AugTikho_a0, 'b0':TFD.AugTikho_b0, 'a1':TFD.AugTikho_a1, 'b1':TFD.AugTikho_b1, 'd':TFD.AugTikho_d, 'ConvReg':True, 'FixedNb':True}
        elif SolMethod == 'InvQuad_AugTikho_V1':
            KWARGS = {'a0':TFD.AugTikho_a0, 'b0':TFD.AugTikho_b0, 'a1':TFD.AugTikho_a1, 'b1':TFD.AugTikho_b1, 'd':TFD.AugTikho_d, 'ConvReg':True, 'FixedNb':True}

    if VerbNb is None:
        VerbNb = np.ceil(Nt/20.)

    # Getting initial solution - step 1/2
    if scpsp.issparse(TMat):
        tmat = TMat.toarray()
    else:
        tmat = TMat.copy()
    if NdimSig==2:
        ssigm = np.mean(sigma,axis=0)
    else:
        ssigm = sigma.copy()
    if data.ndim==1:
        ddat = data.copy()
    else:
        ddat = np.copy(data[0,:])
    sol0 = Default_sol(BF2, tmat, ddat, sigma0=ssigm, N=3)

    # Using discrepancy principle on averaged signal
    LL, mm = BF2.get_IntOp(Deriv=Deriv, Mode=IntMode)

    if mm==2:
        LL = LL[0]+LL[1]
    b = ddat/ssigm                      # DB
    Tm = np.diag(1./ssigm).dot(tmat)    # DB
    Tb = Tm.T.dot(b)                    # DB
    TT = Tm.T.dot(Tm)                   # DB
    sol0, mu0 = InvLin_DisPrinc_V1_Sparse(NMes, Nbf, SpFc[SpType](Tm), SpFc[SpType](TT), SpFc[SpType](LL), Tb, b, sol0, ConvCrit=ConvCrit, mu0=mu0, chi2Tol=0.2, chi2Obj=1.2, M=None, Verb=True)[0:2]    # DB

    # Launching inversions with chosen method - step 2/2 of initial solution included below
    Sol, Mu, Chi2N, R, Nit, Spec = np.nan*np.ones((Nt,Nbf)), np.nan*np.ones((Nt,)), np.nan*np.ones((Nt,)), np.nan*np.ones((Nt,)), np.nan*np.ones((Nt,)), [[] for ii in range(0,Nt)]
    NormL = sol0.dot(LL.dot(sol0))
    if Sparse:
        func = globals()[SolMethod+'_Sparse']
        if not SpIs[SpType](TMat):
            TMat = SpFc[SpType](TMat)
        if not SpIs[SpType](LL):
            LL = SpFc[SpType](LL)
        b = ddat/ssigm
        Tm = np.diag(1./ssigm).dot(tmat)
        Tb = Tm.T.dot(b)
        TT = SpFc[SpType](Tm.T.dot(Tm))
        sol0, mu0, chi2N0, R0, Nit0, spec0 = func(NMes, Nbf, SpFc[SpType](Tm), TT, LL, Tb, b, sol0, ConvCrit=ConvCrit, mu0=mu0, M=None, Verb=False, **KWARGS)
        M = scpsplin.inv((TT+mu0*LL/sol0.dot(LL.dot(sol0))).tocsc())
        # Beware of element-wise operations vs matrix operations !!!!
        assert not mm==0, "Inversion-regularisation routines require a quadratic operator !"
        if NdimSig==2:
            b = data/sigma
            for ii in range(0,Nt):
                if ii==0 or (ii+1)%VerbNb==0:
                    print "    Running inversion for time step ", ii+1,' / ', Nt
                Tm = np.diag(1./sigma[ii,:]).dot(tmat)
                Tb = np.concatenate(tuple([(Tm.T.dot(b[ii,:])).reshape((1,Nbf)) for ii in range(0,Nt)]),axis=0)
                TT = SpFc[SpType](Tm.T.dot(Tm))
                Sol[ii,:], Mu[ii], Chi2N[ii], R[ii], Nit[ii], Spec[ii] = func(NMes, Nbf, SpFc[SpType](Tm), TT, LL, Tb[ii,:], b[ii,:], sol0, ConvCrit=ConvCrit, mu0=mu0, NormL=NormL, M=M, Verb=Verb, **KWARGS)
                sol0[:] = Sol[ii,:]
                mu0 = Mu[ii]
                #print '    Nit or Ntry : ', Spec[ii][0]
        else:
            b = data.dot(np.diag(1./sigma))
            Tm = np.diag(1./sigma).dot(tmat)
            Tb = np.concatenate(tuple([(Tm.T.dot(b[ii,:])).reshape((1,Nbf)) for ii in range(0,Nt)]),axis=0)
            TT = SpFc[SpType](Tm.T.dot(Tm))
            Tm = SpFc[SpType](Tm)
            for ii in range(0,Nt):
                if ii==0 or (ii+1)%VerbNb==0:
                    print "    Running inversion for time step ", ii+1,' / ', Nt, KWARGS['ConvReg'], KWARGS['FixedNb']
                Sol[ii,:], Mu[ii], Chi2N[ii], R[ii], Nit[ii], Spec[ii] = func(NMes, Nbf, Tm, TT, LL, Tb[ii,:], b[ii,:], sol0, ConvCrit=ConvCrit, mu0=mu0, NormL=NormL, M=M, Verb=Verb, **KWARGS)
                sol0[:] = Sol[ii,:]
                mu0 = Mu[ii]
                #print '    Nit or Ntry : ', Spec[ii][0]

    else:
        func = globals()[SolMethod]
        if scpsp.issparse(TMat):
            TMat = TMat.toarray()
        if scpsp.issparse(LL):
            LL = LL.toarray()

        assert not mm==0, "Inversion-regularisation routines require a quadratic operator !"
        if NdimSig==2:
            b = data/sigma
            for ii in range(0,Nt):
                if ii==0 or (ii+1)%VerbNb==0:
                    print "    Running inversion for time step ", ii+1,' / ', Nt
                Tm = np.diag(1./sigma[ii,:]).dot(TMat)
                Tb = np.concatenate(tuple([(Tm.T.dot(b[ii,:])).reshape((1,Nbf)) for ii in range(0,Nt)]),axis=0)
                TT = Tm.T.dot(Tm)
                Sol[ii,:], Mu[ii], Chi2N[ii], R[ii], Nit[ii], Spec[ii] = func(NMes, Nbf, Tm, TT, LL, Tb[ii,:], b[ii,:], sol0, ConvCrit=ConvCrit, mu0=mu0, Verb=Verb, **KWARGS)
                sol0[:] = Sol[ii,:]
                mu0 = Mu[ii]
                #print '    Nit or Ntry : ', Spec[ii][0]
        else:
            b = data.dot(np.diag(1./sigma))
            Tm = np.diag(1./sigma).dot(TMat)
            Tb = np.concatenate(tuple([(Tm.T.dot(b[ii,:])).reshape((1,Nbf)) for ii in range(0,Nt)]),axis=0)
            TT = Tm.T.dot(Tm)
            for ii in range(0,Nt):
                if ii==0 or (ii+1)%VerbNb==0:
                    print "    Running inversion for time step ", ii+1,' / ', Nt
                Sol[ii,:], Mu[ii], Chi2N[ii], R[ii], Nit[ii], Spec[ii] = func(NMes, Nbf, Tm, TT, LL, Tb[ii,:], b[ii,:], sol0, ConvCrit=ConvCrit, mu0=mu0, Verb=Verb, **KWARGS)
                sol0[:] = Sol[ii,:]
                mu0 = Mu[ii]
                #print '    Nit or Ntry : ', Spec[ii][0]

    """# ------- DB ---------
    indview = 0
    f, (ax1,ax2) = plt.subplots(1,2)
    ax1.axis('equal')
    ax1 = BF2.plot(Coefs=Sol[indview,:], ax=ax1)
    ax2.plot(data[indview,:],'k+')
    ax2.plot(TMat.dot(Sol[indview,:]),'r-')
    #print TMat.dot(Sol[indview,:])/data[indview,:]
    plt.show()
    """# --------------------
    if timeit:
        t2 = dtm.datetime.now()-t1
        print t2
    else:
        t2 = None
    return Sol, t, data, sigma, Mu, Chi2N, R, Nit, Spec, t2



def InvRatio(TMat, data, t, BF2, indCorr, sigma=None, Dt=None, mu0=TFD.mu0, SolMethod='InvLin_AugTikho_V1', Deriv='D2N2', IntMode='Vol', Cond=None, ConvCrit=TFD.ConvCrit, Sparse=True, SpType='csr', Pos=True, KWARGS=None, timeit=True, Verb=False, VerbNb=None):
    assert isinstance(indCorr,np.ndarray) and indCorr.ndim==1, "Arg indCorr must be a 1D np.ndarray !"
    if indCorr.dtype in ['int','int64']:
        ind = np.zeros((TMat.shape[0],),dtype=bool)
        ind[indCorr] = True
        indCorr = ind

    datatemp = data[:,~indCorr]
    TMattemp = TMat[(~indCorr).nonzero()[0],:]
    TMatCorr = TMat[indCorr.nonzero()[0],:]
    if not sigma is None:
        if sigma.ndim==2:
            sigmatemp = sigma[:,~indCorr]
        else:
            sigmatemp = sigma[~indCorr]
    else:
        sigmatemp = None
    Sol, tbis, datatemp, sigmatemp, Mu, Chi2N, R, Nit, Spec, t2 = InvChoose(TMattemp, datatemp, t, BF2, sigma=sigmatemp, Dt=Dt, mu0=mu0, SolMethod=SolMethod, Deriv=Deriv, IntMode=IntMode, Cond=Cond, ConvCrit=ConvCrit, Sparse=Sparse, SpType=SpType, Pos=Pos, KWARGS=KWARGS, timeit=timeit, Verb=Verb, VerbNb=VerbNb)
    if Dt is None:
        indt = np.ones((t.size,),dtype=bool)
    else:
        indt = (t>=Dt[0]) & (t<=Dt[1])
    dataCorr = data[:,indCorr]
    Retro = np.empty((tbis.size,np.sum(indCorr)))
    for ii in range(0,tbis.size):
        Retro[ii,:] = TMatCorr.dot(Sol[ii,:])
    return dataCorr[indt,:], Retro, indCorr


def Corrdata(data, retro, sigma, Method='Poly', Deg=1, Group=True, plot=False, LNames=None, Com=''):
    assert isinstance(data,np.ndarray) and isinstance(retro,np.ndarray) and data.shape==retro.shape, "Args data and retro must be np.ndarrays of same shape !"
    Nt, Nm = data.shape
    if Method=='Const':
        Mean = np.mean(retro/data,axis=0)
        if Group:
            def ffcorr(x):
                return np.mean(Mean)*x, Mean
        else:
            def ffcorr(x):
                if x.ndim==2:
                    return np.tile(Mean,(Nt,1))*x, Mean
                else:
                    return Mean*x, Mean
    elif Method=='Poly':
        if Group:
            pp = np.polyfit(np.abs(data.flatten()), np.abs(retro.flatten()), Deg, rcond=None, full=False, w=None, cov=False)
            def ffcorr(x):
                return np.polyval(pp, x), pp
        else:
            PP = [np.polyfit(np.abs(data[:,ii]), np.abs(retro[:,ii]), Deg, rcond=None, full=False, w=None, cov=False) for ii in range(0,Nm)]
            def ffcorr(x):
                if x.ndim==2:
                    return np.concatenate(tuple([np.polyval(PP[ii], x[:,ii]).reshape((Nt,1)) for ii in range(0,Nm)]),axis=1), PP
                else:
                    return np.array([np.polyval(PP[ii], x[ii]) for ii in range(0,Nm)]), PP
    sigmabis, pp = ffcorr(sigma)
    databis, pp = ffcorr(data)
    if plot:
        fW,fH,fdpi,axCol = 17,10,80,'w'
        f = plt.figure(facecolor=axCol,figsize=(fW,fH),dpi=fdpi)
        ax = f.add_axes([0.1,0.1,0.75,0.8],frameon=True,axisbg=axCol)
        ax.set_title(r"Correction Coef. estimation for "+Com)
        ax.set_xlabel(r"Measurements (a.u.)")
        ax.set_ylabel(r"Retrofit (a.u.)")
        if LNames is None:
            LNames = range(0,Nm)
        for ii in range(0,Nm):
            ll = ax.plot(data[:,ii], retro[:,ii], ls='none', lw=2, marker='x', label=LNames[ii])
            if not Group:
                indsort = np.argsort(data[:,ii])
                ax.plot(np.sort(data[:,ii]), (databis[:,ii])[indsort], ls='solid', lw=2, c=ll[0].get_color(), label=LNames[ii]+' fit')
        if Group:
            indsort = np.argsort(data.flatten())
            ax.plot(np.sort(data.flatten()), databis.flatten()[indsort], ls='solid', lw=2, c='k', label='Fit')
        X = np.sort(data.flatten())
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,prop={'size':10})
        ax.plot(X, X, lw=1, c='k', ls='dashed', label='y = x')
        return databis, sigmabis, pp, ax
    else:
        return databis, sigmabis, pp, None


def InvCorrdata(TMat, data, t, BF2, indCorr, sigma=None, Dt=None, mu0=TFD.mu0, SolMethod='InvLin_AugTikho_V1', Deriv='D2N2', IntMode='Vol', Cond=None, ConvCrit=TFD.ConvCrit, Sparse=True, SpType='csr', Pos=True, KWARGS=None, timeit=True, Verb=False, VerbNb=None, Method='Poly', Deg=1, Group=True, plot=False, LNames=None, Com=''):
    if sigma is None:
        sigma = 0.5*np.mean(np.mean(np.abs(data)))*np.ones((TMat.shape[0],))
    dataCorr, Retro, indCorr = InvRatio(TMat, data, t, BF2, indCorr, sigma=sigma, Dt=Dt, mu0=mu0, SolMethod=SolMethod, Deriv=Deriv, IntMode=IntMode, Cond=Cond, ConvCrit=ConvCrit, Sparse=Sparse, SpType=SpType, Pos=Pos, KWARGS=KWARGS, timeit=timeit, Verb=Verb, VerbNb=VerbNb)
    if sigma.ndim==2:
        sigmaCorr = sigma[:,indCorr]
    else:
        sigmaCorr = sigma[indCorr]
    databis, sigmabis, pp, ax = Corrdata(dataCorr, Retro, sigmaCorr, Method=Method, Deg=Deg, Group=Group, plot=plot, LNames=LNames, Com=Com)
    if Dt is None:
        indt = np.ones((t.size,),dtype=bool)
    else:
        indt = (t>=Dt[0]) & (t<=Dt[1])
    tin = t[indt]
    datain = data[indt,:]
    datain[:,indCorr] = databis

    # Implement corrected sigma, taking care it should not be too low !
    if sigma.ndim==2:
        sigmain = sigma[indt,:]
        indNonCorrbool = np.ones((sigma.shape[1],),dtype=bool)
        indNonCorrbool[indCorr] = False
        sigmaMin = np.nanmean(sigmain[:,indNonCorrbool],axis=1)
        indMin = sigmabis < np.tile(sigmaMin,(sigmabis.shape[1],1)).T
        sigmabis[indMin] = sigmaMin
        sigmain[:,indCorr] = sigmabis
    else:
        sigmain = sigma
        indNonCorrbool = np.ones((sigma.size,),dtype=bool)
        indNonCorrbool[indCorr] = False
        sigmaMin = np.nanmean(sigmain[indNonCorrbool])
        sigmabis[sigmabis<sigmaMin] = sigmaMin
        sigmain[indCorr] = sigmabis

    Sol, t, data, sigma, Mu, Chi2N, R, Nit, Spec, t2 = InvChoose(TMat, datain, tin, BF2, sigma=sigmain, Dt=Dt, mu0=mu0, SolMethod=SolMethod, Deriv=Deriv, IntMode=IntMode, Cond=Cond, ConvCrit=ConvCrit, Sparse=Sparse, SpType=SpType, Pos=Pos, KWARGS=KWARGS, timeit=timeit, Verb=Verb, VerbNb=VerbNb)
    return Sol, t, data, sigma, Mu, Chi2N, R, Nit, Spec, t2, pp


def Inv(TMat, data, t, BF2, indCorr=None, sigma=None, Dt=None, mu0=TFD.mu0, SolMethod='InvLin_AugTikho_V1', Deriv='D2N2', IntMode='Vol', Cond=None, ConvCrit=TFD.ConvCrit, Sparse=True, SpType='csr', Pos=True, KWARGS=None, timeit=True, Verb=False, VerbNb=None, Method='Poly', Deg=1, Group=True, plot=False, LNames=None, Com='', Sep=None):
    if indCorr is None or (type(indCorr) is list and len(indCorr)==0) or (isinstance(indCorr,np.ndarray) and indCorr.size==0):
        Sol, t, data, sigma, Mu, Chi2N, R, Nit, Spec, t2 = InvChoose(TMat, data, t, BF2, sigma=sigma, Dt=Dt, mu0=mu0, SolMethod=SolMethod, Deriv=Deriv, IntMode=IntMode, Cond=Cond, ConvCrit=ConvCrit, Sparse=Sparse, SpType=SpType, Pos=Pos, KWARGS=KWARGS, timeit=timeit, Verb=Verb, VerbNb=VerbNb)
        pp = None
    else:
        Sol, t, data, sigma, Mu, Chi2N, R, Nit, Spec, t2, pp = InvCorrdata(TMat, data, t, BF2, indCorr, sigma=sigma, Dt=Dt, mu0=mu0, SolMethod=SolMethod, Deriv=Deriv, IntMode=IntMode, Cond=Cond, ConvCrit=ConvCrit, Sparse=Sparse, SpType=SpType, Pos=Pos, KWARGS=KWARGS, timeit=timeit, Verb=Verb, VerbNb=VerbNb, Method=Method, Deg=Deg, Group=Group, plot=plot, LNames=LNames, Com=Com)
    return Sol, t, data, sigma, Mu, Chi2N, R, Nit, Spec, t2, pp






##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
###########            Inversion routines
##############################################################################################################################################
##############################################################################################################################################

####### For Tomas ###########



def InvLin_GenCross_V1(TT, LL, Tb, a0, b0, a1, b1, mu0, ConvCrit=TFD.ConvCrit):
    # Tikhonov regularisation for linear problems, with Generalized Cross-Validation for regularisation parameter determination
    print "To do..."




def InvNonLin_GenCross_V1(TT, LL, Tb, a0, b0, a1, b1, mu0, ConvCrit=TFD.ConvCrit):
    # Tikhonov regularisation for non-linear problems (MFI type), with Generalized Cross-Validation for regularisation parameter determination
    print "To do..."




    # .....







####### Others ##########




def InvLin_AugTikho_V1(NMes, Nbf, Tm, TT, LL, Tb, b, sol0, mu0=TFD.mu0, ConvCrit=TFD.ConvCrit, a0=TFD.AugTikho_a0, b0=TFD.AugTikho_b0, a1=TFD.AugTikho_a1, b1=TFD.AugTikho_b1, d=TFD.AugTikho_d, NormL=None, Verb=False, ConvReg=True, FixedNb=True):
    """ Linear algorithm for Phillips-Tikhonov regularisation, called "Augmented Tikhonov"
    augmented in the sense that bayesian statistics are combined with standard Tikhonov regularisation
    Determines both noise (common multiplicative coefficient) and regularisation paremeter automatically

    We assume here that all arrays have previously been scaled (noise, conditioning...)
    Sparse matrixes are also prefered to speed-up the computation
    In this method:
      tau is an approximation of the inverse of the noise coefficient
      lamb is an approximation of the regularisation parameter

    N.B.: The noise and reg. param. have probability densities of the form : f(x) = x^(a-1) * exp(-bx)
    This function maximum is in x = (a-1)/b, so a = b+1 gives a maximum at 1.
    (a0,b0) for the reg. param. and (a1,b1) for the noise estimate

    Ref:
      [1] Jin B., Zou J., Inverse Problems, vol.25, nb.2, 025001, 2009
      [2] http://www.math.uni-bremen.de/zetem/cms/media.php/250/nov14talk_jin%20bangti.pdf
      [3] Kazufumi Ito, Bangti Jin, Jun Zou, "A New Choice Rule for Regularization Parameters in Tikhonov Regularization", Research report, University of Hong Kong, 2008
    """

    a0bis = a0-1.+Nbf/2. if not FixedNb else a0-1.+1200./2.
    a1bis = a1-1.+NMes/2.

    Conv = 0.                                                   # Initialise convergence variable
    Nit = 0                                                     # Initialise number of iterations
    chi2N, R = [], []                                           # Initialise residu list and regularisation term list
    lmu = [mu0]                                                 # Initialise regularisation parameter
    if NormL is None:
        NormL = sol0.dot(LL.dot(sol0))
    LL = LL/NormL                                               # Scale the regularity operator
    if Verb:
        print "        Init. guess : N*Chi2N + mu*RN = ", NMes,'*',np.sum((Tm.dot(sol0)-b)**2)/NMes,'+',lmu[-1],'*',sol0.dot(LL.dot(sol0)), ' = ', np.sum((Tm.dot(sol0)-b)**2)+lmu[-1]*sol0.dot(LL.dot(sol0))
        while  Nit<2 or Conv>ConvCrit:                              # Continue until convergence criterion is fulfilled, and do at least 2 iterations
            if np.linalg.det(TT + lmu[-1]*LL)==0.:
                sol = np.linalg.lstsq(TT + lmu[-1]*LL,Tb)[0]
            else:
                Chol = np.linalg.cholesky(TT + lmu[-1]*LL)                   # Use Choleski factorisation (Chol(A).T * Chol(A) = A) for faster computation
                sol = np.linalg.lstsq(Chol,np.linalg.lstsq(Chol.T,Tb)[0])[0]  # Use np.linalg.lstsq for double-solving the equation

            Res = np.sum((Tm.dot(sol)-b)**2)                        # Compute residu**2
            chi2N.append(Res/NMes)                                  # Record normalised residu history
            R.append(sol.dot(LL.dot(sol)))                          # Compute and record regularity term

            lamb = a0bis/(0.5*R[Nit]+b0)                            # Update reg. param. estimate
            tau = a1bis/(0.5*Res+b1)                                # Update noise coef. estimate
            lmu.append((lamb/tau) * (2*a1bis/Res)**d)               # Update regularisation parameter taking into account rescaling with noise estimate
            if ConvReg:
                Conv = np.abs(lmu[-1]-lmu[-2])/lmu[-1]
            else:
                Conv = np.sqrt(np.sum((sol-sol0)**2/np.max([sol**2,0.001*np.max(sol**2)*np.ones((Nbf,))],axis=0))/Nbf)        # Compute convergence variable
            print "        Nit = ",str(Nit),"   N*Chi2N + mu*R = ", NMes,'*',chi2N[Nit],'+',lmu[-1],'*',R[Nit], ' = ', Res+lmu[-1]*R[Nit], '    tau = ',tau, '    Conv = ',Conv
            sol0[:] = sol[:]                                           # Update reference solution
            Nit += 1                                                # Update number of iterations
    else:
        while  Nit<2 or Conv>ConvCrit:                              # Continue until convergence criterion is fulfilled, and do at least 2 iterations
            if np.linalg.det(TT + lmu[-1]*LL)==0.:
                sol = np.linalg.lstsq(TT + lmu[-1]*LL,Tb)[0]
            else:
                Chol = np.linalg.cholesky(TT + lmu[-1]*LL)                   # Use Choleski factorisation (Chol(A).T * Chol(A) = A) for faster computation
                sol = np.linalg.lstsq(Chol,np.linalg.lstsq(Chol.T,Tb)[0])[0]  # Use np.linalg.lstsq for double-solving the equation

            Res = np.sum((Tm.dot(sol)-b)**2)                        # Compute residu**2
            chi2N.append(Res/NMes)                                  # Record normalised residu history
            R.append(sol.dot(LL.dot(sol)))                          # Compute and record regularity term

            lamb = a0bis/(0.5*R[Nit]+b0)                            # Update reg. param. estimate
            tau = a1bis/(0.5*Res+b1)                                # Update noise coef. estimate
            lmu.append((lamb/tau) * (2*a1bis/Res)**d)               # Update regularisation parameter taking into account rescaling with noise estimate
            if ConvReg:
                Conv = np.abs(lmu[-1]-lmu[-2])/lmu[-1]
            else:
                Conv = np.sqrt(np.sum((sol-sol0)**2/np.max([sol**2,0.001*np.max(sol**2)*np.ones((Nbf,))],axis=0))/Nbf)        # Compute convergence variable
            sol0[:] = sol[:]                                           # Update reference solution
            Nit += 1                                                # Update number of iterations

    return sol, lmu[-1], chi2N[-1], R[-1], Nit, [tau, lamb]


def InvLin_AugTikho_V1_Sparse(NMes, Nbf, Tm, TT, LL, Tb, b, sol0, mu0=TFD.mu0, ConvCrit=TFD.ConvCrit, a0=TFD.AugTikho_a0, b0=TFD.AugTikho_b0, a1=TFD.AugTikho_a1, b1=TFD.AugTikho_b1, d=TFD.AugTikho_d, atol=TFD.AugTkLsmrAtol, btol=TFD.AugTkLsmrBtol, conlim=TFD.AugTkLsmrConlim, maxiter=TFD.AugTkLsmrMaxiter, NormL=None, M=None, Verb=False, ConvReg=True, FixedNb=True):       # Install scikit !
    """ Linear algorithm for Phillips-Tikhonov regularisation, called "Augmented Tikhonov", sparese matrix version
    see TFI.InvLin_AugTikho_V1.__doc__ for details
    """

    a0bis = a0-1.+Nbf/2. if not FixedNb else a0-1.+1200./2.
    a1bis = a1-1.+NMes/2.

    Conv = 0.                                                   # Initialise convergence variable
    Nit = 0                                                     # Initialise number of iterations
    chi2N, R = [], []                                           # Initialise residu list and regularisation term list
    lmu = [mu0]                                                 # Initialise regularisation parame
    if NormL is None:
        NormL = sol0.dot(LL.dot(sol0))
    LL = LL/NormL                                               # Scale the regularity operator
    if M is None:
        M = scpsplin.inv((TT + lmu[-1]*LL).tocsc())
    if Verb:
        print "        Initial guess : N*Chi2N + mu*RN = ", NMes,'*',np.sum((Tm.dot(sol0)-b)**2)/NMes,'+',lmu[-1],'*',sol0.dot(LL.dot(sol0)), ' = ', np.sum((Tm.dot(sol0)-b)**2)+lmu[-1]*sol0.dot(LL.dot(sol0))
        while  Nit<2 or Conv>ConvCrit:                              # Continue until convergence criterion is fulfilled, and do at least 2 iterations
            #sol, itconv = scpsplin.minres(TT + mu*LL, Tb, x0=sol0, shift=0.0, tol=1e-10, maxiter=None, M=M, show=False, check=False)   # 2
            sol, itconv = scpsplin.cg(TT + lmu[-1]*LL, Tb, x0=sol0, tol=1e-08, maxiter=None, M=M)    # 1
            #sol = scpsplin.spsolve(TT + lmu[-1]*LL, Tb, permc_spec=None, use_umfpack=False) # 3
            #sol, isstop, itn, normr, normar, norma, conda, normx = scpsplin.lsmr(TT + lmu[-1]*LL, Tb, atol=atol, btol=btol, conlim=conlim, maxiter=maxiter, show=False) # Install scikits and use CHOLMOD fast cholesky factorization !!!

            Res = np.sum((Tm.dot(sol)-b)**2)                        # Compute residu**2
            chi2N.append(Res/NMes)                                  # Record normalised residu history
            R.append(sol.dot(LL.dot(sol)))                          # Compute and record regularity term

            lamb = a0bis/(0.5*R[Nit]+b0)                            # Update reg. param. estimate
            tau = a1bis/(0.5*Res+b1)                                # Update noise coef. estimate
            lmu.append((lamb/tau) * (2*a1bis/Res)**d)               # Update regularisation parameter taking into account rescaling with noise estimate
            if ConvReg:
                Conv = np.abs(lmu[-1]-lmu[-2])/lmu[-1]
            else:
                Conv = np.sqrt(np.sum((sol-sol0)**2/np.max([sol**2,0.001*np.max(sol**2)*np.ones((Nbf,))],axis=0))/Nbf)        # Compute convergence variable
            print "        Nit = ",str(Nit),"  N*Chi2N + mu*R = ", NMes,'*',chi2N[Nit],'+',lmu[-1],'*',R[Nit], ' = ', Res+lmu[-1]*R[Nit], '  tau = ',tau, '  Conv = ',Conv
            #print '                     isstop,itn=',isstop,itn, '  normRArAX=',normr, normar, norma,normx, 'condA=',conda
            sol0[:] = sol[:]                                           # Update reference solution
            Nit += 1                                                # Update number of iterations
    else:
        while  Nit<2 or Conv>ConvCrit:                              # Continue until convergence criterion is fulfilled, and do at least 2 iterations
            #sol, itconv = scpsplin.minres(TT + lmu[-1]*LL, Tb, x0=sol0, shift=0.0, tol=1e-10, maxiter=None, M=M, show=False, check=False)   # 2
            sol, itconv = scpsplin.cg(TT + lmu[-1]*LL, Tb, x0=sol0, tol=1e-08, maxiter=None, M=M)    # 1
            #sol = scpsplin.spsolve(TT + lmu[-1]*LL, Tb, permc_spec=None, use_umfpack=False) # 3
            #sol, isstop, itn, normr, normar, norma, conda, normx = scpsplin.lsmr(TT + lmu[-1]*LL, Tb, atol=atol, btol=btol, conlim=conlim, maxiter=maxiter, show=False)

            Res = np.sum((Tm.dot(sol)-b)**2)                        # Compute residu**2
            chi2N.append(Res/NMes)                                  # Record normalised residu history
            R.append(sol.dot(LL.dot(sol)))                          # Compute and record regularity term

            lamb = a0bis/(0.5*R[Nit]+b0)                            # Update reg. param. estimate
            tau = a1bis/(0.5*Res+b1)                                # Update noise coef. estimate
            lmu.append((lamb/tau) * (2*a1bis/Res)**d)               # Update regularisation parameter taking into account rescaling with noise estimate
            if ConvReg:
                Conv = np.abs(lmu[-1]-lmu[-2])/lmu[-1]
            else:
                Conv = np.sqrt(np.sum((sol-sol0)**2/np.max([sol**2,0.001*np.max(sol**2)*np.ones((Nbf,))],axis=0))/Nbf)        # Compute convergence variable
            sol0[:] = sol[:]                                        # Update reference solution
            Nit += 1

    return sol, lmu[-1], chi2N[-1], R[-1], Nit, [tau, lamb]


def InvLinQuad_AugTikho_V1_Sparse(NMes, Nbf, Tm, TT, LL, Tb, b, sol0, mu0=TFD.mu0, ConvCrit=TFD.ConvCrit, a0=TFD.AugTikho_a0, b0=TFD.AugTikho_b0, a1=TFD.AugTikho_a1, b1=TFD.AugTikho_b1, d=TFD.AugTikho_d, atol=TFD.AugTkLsmrAtol, btol=TFD.AugTkLsmrBtol, conlim=TFD.AugTkLsmrConlim, maxiter=TFD.AugTkLsmrMaxiter, NormL=None, M=None, Verb=False, ConvReg=True, FixedNb=True):       # Install scikit !
    """ Quadratic algorithm for Phillips-Tikhonov regularisation, alternative to the linear version with positivity constraint
    see TFI.InvLin_AugTikho_V1.__doc__ for details
    """

    a0bis = a0-1.+Nbf/2. if not FixedNb else a0-1.+1200./2.
    a1bis = a1-1.+NMes/2.

    Conv = 0.                                                   # Initialise convergence variable
    Nit = 0                                                     # Initialise number of iterations
    chi2N, R = [], []                                           # Initialise residu list and regularisation term list
    lmu = [mu0]                                                 # Initialise regularisation parame
    if NormL is None:
        NormL = sol0.dot(LL.dot(sol0))
    LL = LL/NormL                                               # Scale the regularity operator
    if M is None:
        M = scpsplin.inv((TT + lmu[-1]*LL).tocsc())

    Bds = tuple([(-0.2,None) for ii in range(0,sol0.size)])
    F = lambda X, mu=lmu[-1]: np.sum((Tm.dot(X)-b)**2) + mu*X.dot(LL.dot(X))
    F_g = lambda X, mu=lmu[-1]: 2.*(TT + mu*LL).dot(X) - 2.*Tb
    F_h = lambda X, mu=lmu[-1]: 2.*(TT + mu*LL)

    Method = 'L-BFGS-B'
    if Method == 'L-BFGS-B':
        options={'ftol':ConvCrit/100., 'disp':False}
    elif Method == 'TNC':
        options={'ftol':ConvCrit/100., 'disp':False, 'minfev':1, 'rescale':1., 'maxCGit':0, 'offset':0.}  # Not working....
    elif Method == 'SLSQP':
        options={'ftol':ConvCrit/100., 'disp':False}      # Very slow... (maybe not updated ?)

    if Verb:
        print "        Initial guess : N*Chi2N + mu*RN = ", NMes,'*',np.sum((Tm.dot(sol0)-b)**2)/NMes,'+',lmu[-1],'*',sol0.dot(LL.dot(sol0)), ' = ', np.sum((Tm.dot(sol0)-b)**2)+lmu[-1]*sol0.dot(LL.dot(sol0))
    while  Nit<2 or Conv>ConvCrit:                              # Continue until convergence criterion is fulfilled, and do at least 2 iterations
        Out = scpop.minimize(F, sol0, args=([lmu[-1]]), jac=F_g, hess=F_h, method=Method, bounds=Bds, options=options)       # Minimisation de la fonction (method quadratique)
        sol = Out.x
        Res = np.sum((Tm.dot(sol)-b)**2)                        # Compute residu**2
        chi2N.append(Res/NMes)                                  # Record normalised residu history
        R.append(sol.dot(LL.dot(sol)))                          # Compute and record regularity term

        lamb = a0bis/(0.5*R[Nit]+b0)                            # Update reg. param. estimate
        tau = a1bis/(0.5*Res+b1)                                # Update noise coef. estimate
        lmu.append((lamb/tau) * (2*a1bis/Res)**d)               # Update regularisation parameter taking into account rescaling with noise estimate
        if ConvReg:
            Conv = np.abs(lmu[-1]-lmu[-2])/lmu[-1]
        else:
            Conv = np.sqrt(np.sum((sol-sol0)**2/np.max([sol**2,0.001*np.max(sol**2)*np.ones((Nbf,))],axis=0))/Nbf)        # Compute convergence variable
        if Verb:
            print "        Nit = ",str(Nit),"  N*Chi2N + mu*R = ", NMes,'*',chi2N[Nit],'+',lmu[-1],'*',R[Nit], ' = ', Res+lmu[-1]*R[Nit], '  tau = ',tau, '  Conv = ',Conv
        sol0[:] = sol[:]                                           # Update reference solution
        Nit += 1                                                # Update number of iterations

    return sol, lmu[-1], chi2N[-1], R[-1], Nit, [tau, lamb]




def estimate_jacob_scalar_Spec(ff, x0, ratio=0.05):
    nV = x0.size
    jac = np.zeros((nV,))
    dx = np.zeros((nV,))
    for ii in range(0,nV):
        dx[ii] = ratio*x0[ii]
        jac[ii] = (ff(x0+dx)[-1]-ff(x0)[-1])/dx[ii]
        dx[ii] = 0.
    return jac



def InvQuad_AugTikho_V1(NMes, Nbf, Tm, TT, LL, Tb, b, sol0, mu0=TFD.mu0, ConvCrit=TFD.ConvCrit, a0=TFD.AugTikho_a0, b0=TFD.AugTikho_b0, a1=TFD.AugTikho_a1, b1=TFD.AugTikho_b1, d=TFD.AugTikho_d, NormL=None, Verb=False, ConvReg=True, ratiojac=0.05, Method='Newton-CG', FixedNb=True):
    """ Non-linear (quadratic) algorithm for Phillips-Tikhonov regularisation, inspired from "Augmented Tikhonov"
    see TFI.InvLin_AugTikho_V1.__doc__ for details
    """

    a0bis = a0-1.+Nbf/2. if not FixedNb else a0-1.+1200./2.
    a1bis = a1-1.+NMes/2.

    Conv = 0.                                                   # Initialise convergence variable
    Nit = 0                                                     # Initialise number of iterations
    chi2N, R = [], []                                           # Initialise residu list and regularisation term list
    mu = mu0                                                    # Initialise regularisation parameter
    if NormL is None:
        NormL = sol0.dot(LL.dot(sol0))
    LL = LL/NormL                                               # Scale the regularity operator
    if Verb:
        print "        Init. guess : N*Chi2N + mu*RN = ", NMes,'*',np.sum((Tm.dot(sol0)-b)**2)/NMes,'+',mu,'*',sol0.dot(LL.dot(sol0)), ' = ', np.sum((Tm.dot(sol0)-b)**2)+mu*sol0.dot(LL.dot(sol0))
        while  Nit<2 or Conv>ConvCrit:                              # Continue until convergence criterion is fulfilled, and do at least 2 iterations
            F = lambda X, mu=mu: np.sum((Tm.dot(X)-b)**2) + mu*X.dot(LL.dot(X))
            Out = scpop.minimize(F, sol0, tol=None, method=Method, options={'maxiter':100, 'disp':True})       # Minimisation de la fonction (method quadratique)
            sol = Out.x
            Res = np.sum((Tm.dot(sol)-b)**2)                        # Compute residu**2
            chi2N.append(Res/NMes)                                  # Record normalised residu history
            R.append(sol.dot(LL.dot(sol)))                          # Compute and record regularity term

            lamb = a0bis/(0.5*R[Nit]+b0)                            # Update reg. param. estimate
            tau = a1bis/(0.5*Res+b1)                                # Update noise coef. estimate
            mu = (lamb/tau) * (2*a1bis/Res)**d                      # Update regularisation parameter taking into account rescaling with noise estimate
            if ConvReg:
                Conv = np.abs(lmu[-1]-lmu[-2])/lmu[-1]
            else:
                Conv = np.sqrt(np.sum((sol-sol0)**2/np.max([sol**2,0.001*np.max(sol**2)*np.ones((Nbf,))],axis=0))/Nbf)        # Compute convergence variable
            print "        Nit = ",str(Nit),"   N*Chi2N + mu*R = ", NMes,'*',chi2N[Nit],'+',mu,'*',R[Nit], ' = ', Res+mu*R[Nit], '    tau = ',tau, '    Conv = ',Conv
            sol0[:] = sol[:]                                        # Update reference solution
            Nit += 1                                                # Update number of iterations
    else:
        xx = np.zeros((Nbf,))
        F = lambda X : np.sum((Tm.dot(X[:-1])-b)**2) + X[-1]*X[:-1].dot(LL.dot(X[:-1]))
        Out = scpop.minimize(F, np.append(sol0,mu), tol=None, options={'maxiter':100, 'disp':False})
        sol = Out.x[:-1]
        Res = np.sum((Tm.dot(sol)-b)**2)
        chi2N.append(np.sum((Tm.dot(sol)-b)**2)/NMes)
        R.append(sol.dot(LL.dot(sol)))
        lamb = a0bis/(0.5*R[-1]+b0)                            # Update reg. param. estimate
        tau = a1bis/(0.5*Res+b1)                                # Update noise coef. estimate
        mu = (lamb/tau) * (2*a1bis/Res)**d
        print "    Done :", chi2N

    return sol, mu, chi2N[-1], R[-1], Nit, [tau, lamb]



def InvLin_DisPrinc_V1_Sparse(NMes, Nbf, Tm, TT, LL, Tb, b, sol0, mu0=TFD.mu0, ConvCrit=TFD.ConvCrit, chi2Tol=TFD.chi2Tol, chi2Obj=TFD.chi2Obj, M=None, Verb=False, NormL=None):
    """ Discrepancy principle """

    Ntry = 0
    LogTol = np.log(chi2Obj)

    chi2N = []                                                  # Initialise residu list and regularisation term list
    lmu = [mu0]                                                 # Initialise regularisation parame
    if NormL is None:
        NormL = sol0.dot(LL.dot(sol0))
    LL = LL/NormL                                               # Scale the regularity operator
    if M is None:
        M = scpsplin.inv((TT + lmu[-1]*LL).tocsc())
    #try:
    if Verb:
        print "        Initial guess : N*Chi2N + mu*RN = ", NMes,'*',np.sum((Tm.dot(sol0)-b)**2)/NMes,'+',lmu[-1],'*',sol0.dot(LL.dot(sol0)), ' = ', np.sum((Tm.dot(sol0)-b)**2)+lmu[-1]*sol0.dot(LL.dot(sol0))
        while Ntry==0 or np.abs(chi2N[-1]-chi2Obj)>chi2Tol:
            sol, itconv = scpsplin.cg(TT + lmu[-1]*LL, Tb, x0=sol0, tol=1e-08, maxiter=None, M=M)    # 1
            Ntry += 1
            chi2N.append(np.sum((Tm.dot(sol)-b)**2)/NMes)
            if Ntry==1:
                if chi2N[-1]>=chi2Obj+chi2Tol:
                    lmu.append(lmu[-1]/50.)
                elif chi2N[-1]<=chi2Obj-chi2Tol:
                    lmu.append(lmu[-1]*50.)
                else:
                    lmu.append(lmu[-1])
            elif all([chi>=chi2Obj+chi2Tol for chi in chi2N]) or all([chi<=chi2Obj-chi2Tol for chi in chi2N]):
                if chi2N[-1]>=chi2Obj+chi2Tol:
                    lmu.append(lmu[-1]/50.)
                else:
                    lmu.append(lmu[-1]*50.)
            else:
                indsort = np.argsort(chi2N)
                lmu.append(np.exp(np.interp(LogTol, np.log(np.asarray(chi2N)[indsort]), np.log(np.asarray(lmu)[indsort]))))
            print "        InvLin_DisPrinc_V1 : try ",Ntry-1, "mu = ",lmu[-2],"chi2N = ",chi2N[-1], "New mu = ", lmu[-1]
    else:
        while Ntry==0 or np.abs(chi2N[-1]-chi2Obj)>chi2Tol:
            sol, itconv = scpsplin.cg(TT + lmu[-1]*LL, Tb, x0=sol0, tol=1e-08, maxiter=None, M=M)
            Ntry += 1
            chi2N.append(np.sum((Tm.dot(sol)-b)**2)/NMes)
            if Ntry==1:
                if chi2N[-1]>=chi2Obj+chi2Tol:
                    lmu.append(lmu[-1]/50.)
                elif chi2N[-1]<=chi2Obj-chi2Tol:
                    lmu.append(lmu[-1]*50.)
                else:
                    lmu.append(lmu[-1])
            elif all([chi>=chi2Obj+chi2Tol for chi in chi2N]) or all([chi<=chi2Obj-chi2Tol for chi in chi2N]):
                if chi2N[-1]>=chi2Obj+chi2Tol:
                    lmu.append(mu[-1]/50.)
                else:
                    lmu.append(mu[-1]*50.)
            else:
                indsort = np.argsort(chi2N)
                #mu.append(np.exp(np.log(mu[-1]) - np.log(chi2N[-1])*(np.log(mu[-1])-np.log(mu[-2]))/(np.log(chi2N[-1])-np.log(chi2N[-2])))
                lmu.append(np.exp(np.interp(LogTol, np.log(np.asarray(chi2N)[indsort]),np.log(np.asarray(lmu[:-1])[indsort]))))
    """
    except:
        print np.asarray(lmu[:-1]), np.asarray(chi2N), np.arange(0,len(chi2N))
        plt.figure()
        plt.scatter(np.asarray(lmu[:-1]),np.asarray(chi2N), c=np.arange(0,len(chi2N)),edgecolors='none')
        plt.gca().set_xscale('log'), plt.gca().set_yscale('log')
        plt.gca().set_xlim(r"$\log_10(\mu)$"), plt.gca().set_ylim(r"$\log_10(\chi^2_N)$")
        plt.figure()
        plt.plot(np.asarray(LL),'b-')
        plt.figure()
        plt.plot(TT,'ko'), plt.plot(TT+lmu[-1]*LL,'r--')
        plt.show()
    """

    return sol, lmu[-1], chi2N[-1], sol.dot(LL.dot(sol)), Ntry, []









##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############            Sol Objects and properties
##############################################################################################################################################
##############################################################################################################################################



class Sol2D(object):
    def __init__(self, Id=None, Diag='SXR', PreData=None, GMat=None, InvParam=TFD.SolInvParDef, StorePreData=True, StoreGMat=True, StoreBF=True, dtime=None):
        self.set_Id(Id, Diag=Diag, GMat=GMat, PreData=PreData, InvParam=InvParam, dtime=dtime)
        self.set_GMat(GMat, StoreGMat=StoreGMat, StoreBF=StoreBF)
        self.set_PreData(PreData, Store=StorePreData)
        self.set_InvParam(InvParam=dict(InvParam))
        self._PostTreat = []
        self._init_run()

    @property
    def Id(self):
        return self._Id
    @Id.setter
    def Id(self,Val):
        self.set_Id(Val)
    @property
    def GMat(self):
        if self._GMat is None:
            self._GMat = TPFP.Open(self.Id.LObj['GMat2D']['SavePath'][0]+self.Id.LObj['GMat2D']['SaveName'][0]+'.npz')
        return self._GMat
    @property
    def BF2(self):
        if self._BF2 is None:
            self._BF2 = TFPF.Open(self.Id.LObj['BF2D']['SavePath'][0]+self.Id.LObj['BF2D']['SaveName'][0]+'.npz')
        return self._BF2
    @property
    def LD(self):
        return self.GMat.LD
    @property
    def PreData(self):
        if self._PreData is None:
            self._PreData = TFPF.Open(self.Id.LObj['PreData']['SavePath'][0]+self.Id.LObj['PreData']['SaveName'][0]+'.npz')
        return self._PreData
    @property
    def shot(self):
        return self._shot
    @property
    def InvParam(self):
        return self._InvParam
    @InvParam.setter
    def InvParam(self,Val):
        self.set_InvParam(Val)
    @property
    def Coefs(self):
        return self._Coefs
    @property
    def t(self):
        return self._t
    @property
    def data(self):
        return self._data
    @property
    def Dt(self):
        return self._Dt
    @property
    def PostTreat(self):
        return self._PostTreat

    def set_Id(self, Val=None, Diag='SXR', GMat=None, PreData=None, InvParam=None, dtime=None):
        assert Val is None or type(Val) is str or isinstance(Val,TFPF.ID), "Arg Id should be string or an TFPF.ID instance !"
        if Val is None and not PreData is None and not GMat is None and not InvParam is None:
            Val = PreData.Id.Name + '_' + GMat.Id.Name + '_Deriv' + InvParam['Deriv'] + '_Sep'+str(InvParam['Sep']['In']) + '_Pos'+str(InvParam['Pos'])
        if type(Val) is str:
            Val = TFPF.ID('Sol2D',Val, Diag=Diag, dtime=dtime)
        self._Id = Val

    def set_GMat(self, GMat=None, StoreGMat=True, StoreBF=True):
        if not GMat is None:
            assert isinstance(GMat,TFMC.GMat2D), "Arg GMat must be a TFMC.GMat2D instance !"
            self._Id.set_LObj([GMat.Id])
            self._GMat = GMat if StoreGMat else None
            BF2 = GMat.BF2
            self._Id.set_LObj([BF2.Id])
            self._BF2 = BF2 if StoreBF else None
            self._BF2_NFunc = BF2.NFunc
            self._BF2_Deg = BF2.Deg

    def set_PreData(self, PreData=None, Store=True):
        #assert isinstance(PreData,TFT.PreData), "Arg PreData must be a TFT.PreData instance !"
        if not PreData is None:
            self._Id.set_LObj([PreData.Id])
            self._shot = PreData.shot
            self._PreData = PreData if Store else None
            self._GMat = self.GMat.get_SubGMat2D(Val=PreData.Sig, Crit='Name',InOut='In')
            self._Dt = PreData._Dt
            self._LNames = PreData.In_list()

    def Store(StorePreData=True, StoreGMat=True, StoreBF=True):
        if self._PreData is None and StorePreData:
            self._PreData = self.PreData
        elif not self._PreData is None and not StorePreData:
            self._PreData = None
        if self._GMat is None and StoreGMat:
            self._GMat = self.GMat
        elif not self._GMat is None and not StoreGMat:
            self._GMat = None
        if self._BF2 is None and StoreBF:
            self._BF2 = self.BF2
        elif not self._BF is None and not StoreBF:
            self._BF2 = None

    def set_InvParam(self,InvParam=TFD.SolInvParDef):
        assert type(InvParam) is dict, "Arg InvParam must be a dict containing all necessary parameters for TFI.Inv() !"
        self._InvParam = dict(InvParam)
        self._InvParam['KWARGS'] = dict(InvParam['KWARGS'])
        if 'Deriv' in self.Id.Name:
            Name = self.Id.Name
            ind1 = Name.index('Deriv')
            if '_' in Name[ind1:]:
                nn = Name[ind1:].index('_')
                Name = Name[:ind1] + 'Deriv'+self._InvParam['Deriv'] + Name[ind1+nn:]
            else:
                Name = Name[:ind1] + 'Deriv'+self._InvParam['Deriv']
            self._Id.Name = Name

    def _init_run(self):
        self._run = False
        self._t, self._data = None, None
        self._Coefs, self._sigma = None, None
        self._Mu, self._Chi2N, self._R = None, None, None
        self._Spec = None
        self._timing = None
        self._timing = None

    def run_Inv(self, InvParam=None, LOS=False, Corr=None):
        if not InvParam is None:
            self.set_InvParam(InvParam=dict(InvParam))
        self._LOS = LOS
        Tmat = self.GMat.MatLOS if LOS else self.GMat.Mat
        if self.InvParam['Sep']['In']:
            Sep = self.InvParam['Sep']['Poly']
            indFin = self.BF2.get_SubBFPolygon_indin(Sep, NLim=self.InvParam['Sep']['NLim'], Out=int)
            Tmat = Tmat[:,indFin]
            BF2 = self._BF2.get_SubBFPolygonind(indFin=indFin)
        else:
            BF2 = self.BF2
        Corr = self.PreData.Corr_list() if Corr is None else Corr
        indCorr = self.PreData.select(Val=Corr, Crit='Name',InOut='In',Out=int, ToIn=True)
        Coefs, t, data, sigma, Mu, Chi2N, R, Nit, Spec, timing, pp = Inv(Tmat, self.PreData.data, self.PreData.t, BF2, indCorr=indCorr, sigma=self.PreData._Noise, **self.InvParam)
        if self.InvParam['Sep']['In']:
            coefs = np.zeros((t.size,self.BF2.NFunc))
            coefs[:,indFin] = Coefs
            Coefs = coefs
        self._t, self._data = t, data
        self._Coefs, self._sigma = Coefs, sigma
        self._Mu, self._Chi2N, self._R = Mu, Chi2N, R
        self._Nit = Nit
        self._Spec = Spec
        self._timing = timing
        self._Corr = pp
        self._run = True

    def get_Val(self, PtsRZ, t=None, indt=None, Deriv='D0', DVect=TFD.BF2_DVect_DefR, Test=True):
        assert isinstance(PtsRZ,np.ndarray) and PtsRZ.ndim==2 and PtsRZ.shape[0]==2, "Arg PtsRZ must be a (2,N) np.ndarray representing (R,Z) coordinates of points"
        if t is None and indt is None:
            indt = np.arange(0,self.t.size)
        elif not t is None:
            indt = np.asarray([np.argmin(self.t-tt) for tt in t]) if hasattr(t,'__getitem__') else np.argmin(self.t-t)
        elif not indt is None:
            indt = np.asarray(indt)
        Vals = self.BF2.get_TotVal(PtsRZ, Deriv=Deriv, DVect=DVect, Coefs=self.Coefs[indt,:], Test=Test)
        return Vals

    def get_MinMax(self, Ratio=0.05, SubP=0.004, SubP1=0.015, SubP2=0.001, TwoSteps=False, SubMode='abs', Deriv='D0', Test=True):
        return self.BF2.get_MinMax(Coefs=self._Coefs, Ratio=Ratio, SubP=SubP, SubP1=SubP1, SubP2=SubP2, TwoSteps=TwoSteps, SubMode=SubMode, Deriv=Deriv, Test=Test)

    def get_Extrema(self, Ratio=0.95, SubP=0.002, SubMode='abs', D1N2=True, D2N2=True):
        return self.BF2.get_Extrema(Coefs=self._Coefs, Ratio=Ratio, SubP=SubP, SubMode=SubMode, D1N2=D1N2, D2N2=D2N2)

    def get_fft(self, PtsRZ=None, FreqIn=[10.e3,11.e3], FreqOut=None, HarmIn=True, HarmOut=True, Deriv='D0', DVect=TFD.BF2_DVect_DefR, Test=True):
        assert isinstance(PtsRZ,np.ndarray) and PtsRZ.ndim==2 and PtsRZ.shape[0]==2, "Arg PtsRZ must be a (2,N) np.ndarray representing (R,Z) coordinates of points"
        Vals = self.BF.get_TotVal(PtsRZ, Deriv=Deriv, DVect=DVect, Coefs=self.Coefs, Test=Test)
        Phys, Noise = FourierExtract(Vals, self._t, DF=FreqInt, DFEx=FreqOut, Harm=HarmIn, HarmEx=HarmOut, Test=Test)
        return Phys, Noise

    def get_fftDom(self, PtsRZ):
        assert isinstance(PtsRZ,np.ndarray) and PtsRZ.ndim==2 and PtsRZ.shape[0]==2, "Arg PtsRZ must be a (2,N) np.ndarray representing (R,Z) coordinates of points"
        Vals = self.BF.get_TotVal(PtsRZ, Deriv=Deriv, DVect=DVect, Coefs=self.Coefs, Test=Test)

    def get_InvRad(self, PreDt, PostDt, Test=True):
        """
        Compute the inversion radius and inversion contour from the 2D inversions by contour plot of the difference between two chosen times or time intervals, usable for sawtooth crashes

        Call:
        -----
            = Sol2D,get_InvRad(PreDt, PostDt, Test=True)

        Inputs:
        -------
            PreDt   int, float or list      Time index (int), value (float) or interval (list with len()=2) to be considered as the reference profile before the crash, if interval the profile is averaged
            PostDt  int, float or list      Time index (int), value (float) or interval (list with len()=2) to be considered as the reference profile after the crash, if interval the profile is averaged
            Test    bool                    specifies wether to Test the input data for consistency (default : True)

        Outputs:
        --------
            D
        """
        return _Sol2D_get_InvRad(self, PreDt, PostDt, Test=True)

    def plot(self, V='basic', prop=None, Test=True):
        """
        Plot the TFI.Sol2D instance with the chosen post-treatment plot configuration
        Inputs :
            Sol2        TFI.Sol2D object,   for which the inversion as been carried out
            V           str,                specifies the version of the plot configuration in ['basic','technical','profiles'] (default : 'basic')
            prop        dict,               specifies the properties to be used for the various artists of the plot (default : None)
            FreqIn      list,
            FreqOut     list,
            HarmIn      bool,
            HarmOut     bool,
            Test        bool,               specifies wether to Test the input data for consistency (default : True)
        Outputs :
            La      dict,               a dict of plt.Axes instances on which the various plots were made, classified depending of the type of plot (i.e.: 2D constant, 2D changing, 1D changing profiles, 1D time traces, 1D constant)
        """
        return TFPT.Dispatch_Inv_Plot(self, V=V, prop=prop, Test=Test)

    def compare(self, LSol2=[], V='basic', prop=None, Test=True):
        """
        Plot a comparison of the TFI.Sol2D instance with a list of other TFI.Sol2D instances with the chosen post-treatment plot configuration
        Inputs :
            Sol2        TFI.Sol2D object,   for which the inversion as been carried out
            V           str,                specifies the version of the plot configuration in ['basic','technical','profiles'] (default : 'basic')
            prop        dict,               specifies the properties to be used for the various artists of the plot (default : None)
            FreqIn      list,
            FreqOut     list,
            HarmIn      bool,
            HarmOut     bool,
            Test        bool,               specifies wether to Test the input data for consistency (default : True)
        Outputs :
            La      dict,               a dict of plt.Axes instances on which the various plots were made, classified depending of the type of plot (i.e.: 2D constant, 2D changing, 1D changing profiles, 1D time traces, 1D constant)
        """

        if type(LSol2) is list and len(LSol2)>0:
            return TFPT.Dispatch_Compare([self]+LSol2, V=V, prop=prop, Test=Test)


    def plot_diff(self, Ref, V='basic', prop=None, Test=True):
        """
        Plot the difference between the TFI.Sol2D instance and a reference time, time interval, coefs array or Sol2D with the chosen post-treatment plot configuration

        Call:
        -----
            La = Sol2D.plot_diff(Ref, V='basic', prop=None, Test=True)

        Inputs :
        --------
            Ref         float, list, np.ndarray or TFI.Sol2D    The reference solution to be substracted from the Sol2D solution
                                                                    int:        interpreted as the index of the reference time step
                                                                    float:      interpreted as the reference time step
                                                                    list:       of len()=2, computes the average profile over the chosen time interval
                                                                    np.ndarray: interpreted as a reference set of Coefs, must be Coefs.shape==Sol2D.Coefs.shape
                                                                    TFI.Sol2D:  interpreted as the reference solution, must have shape==Sol2D.Coefs.shape
            V           str,                                    specifies the version of the plot configuration in ['basic','technical','profiles'] (default : 'basic')
            prop        dict,                                   specifies the properties to be used for the various artists of the plot, if None sets to defaults depending on the chosen V (default : None)
            Test        bool,                                   specifies wether to Test the input data for consistency (default : True)
        Outputs :
            La      dict,               a dict of plt.Axes instances on which the various plots were made, classified depending of the type of plot (i.e.: 2D constant, 2D changing, 1D changing profiles, 1D time traces, 1D constant)
        """
        if Test:
            assert type(Ref) in [int,float,list,np.ndarray,Sol2D], "Arg Ref must be of type in [int,float,list,np.ndarray,TFI.Sol2D] !"
        if type(Ref) is int:
            Ref = np.tile(self.Coefs[Ref,:],(self.t.size,1))
        elif type(Ref) is float:
            Ref = np.tile(self.Coefs[np.nanargmin(np.abs(self.t-Ref)),:],(self.t.size,1))
        elif type(Ref) is list:
            Ref = (self.t>=Ref[0]) & (self.t<=Ref[1])
            Ref = np.tile(np.nanmean(self.Coefs[Ref,:],axis=0),(self.t.size,1))
        elif type(Ref) is np.ndarray:
            Ref = Ref
        elif type(Ref) is Sol2D:
            Ref = Ref.Coefs
        CoefsI = self.Coefs
        CoefsD = CoefsI - Ref
        self._Coefs = CoefsD
        if prop is None:
            prop = dict(TFD.TFPT_Lprop[V])
            Max = np.nanmax(np.abs(CoefsD))
            prop['InvLvls'] = np.linspace(-Max,Max,29)
            prop['Norm'] = False
            prop['VMinMax'] = [-Max,Max]
            prop['Invd'] = {'cmap':plt.cm.seismic,'edgecolor':None}
        Out = TFPT.Dispatch_Inv_Plot(self, V=V, prop=prop, Test=Test)
        self._Coefs = CoefsI
        return Out



    def plot_fft(self, V='basic', prop=None, FreqIn=None, FreqOut=None, HarmIn=True, HarmOut=True, Test=True):
        return TFPT.Dispatch_Inv_Plot(self, V=V, prop=prop, FreqIn=FreqIn, FreqOut=FreqOut, HarmIn=HarmIn, HarmOut=HarmOut, Test=Test)

    def anim(self, DVect=TFD.BF2_DVect_DefR, SubP=TFD.InvPlotSubP, SubMode=TFD.InvPlotSubMode, blit=TFD.InvAnimBlit, interval=TFD.InvAnimIntervalms, repeat=TFD.InvAnimRepeat, repeat_delay=TFD.InvAnimRepeatDelay,
            InvPlotFunc=TFD.InvPlotF, InvLvls=TFD.InvLvls, Invd=TFD.Invdict, Tempd=TFD.Tempd, Retrod=TFD.Retrod, TimeScale=TFD.InvAnimTimeScale, VMinMax=[None,None], Test=True, Hybrid=False, FName=None, Com='', Norm=False):
        TMat = self.GMat.MatLOS if self._LOS else self.GMat.Mat
        Nit = np.array([ss[0] for ss in self._Spec])
        ani, axInv, axTMat, Laxtemp = TFPT.Inv_MakeAnim(self.BF2, self.Coefs, t=self.t, Com=Com, shot=self.shot, Ves=self.GMat.Ves, SXR=self.data, sigma=self._sigma, TMat=TMat, Chi2N=self._Chi2N,
                Mu=self._Mu, R=self._R, Nit=Nit, Deriv=0, indt0=0, DVect=DVect, SubP=SubP, SubMode=SubMode, blit=blit, interval=interval, repeat=repeat, repeat_delay=repeat_delay, InvPlotFunc=InvPlotFunc, InvLvls=InvLvls,
                Invd=Invd, Tempd=Tempd, Retrod=Retrod, TimeScale=TimeScale, VMinMax=VMinMax, Test=Test, Hybrid=Hybrid, FName=FName)
        return ani

    def plot_fftpow(self, PtsRZ=None, V='user', Deriv=0, DVect=TFD.BF2_DVect_DefR, SubP=TFD.InvPlotSubP, SubMode=TFD.InvPlotSubMode, InvPlotFunc=TFD.InvPlotF, InvLvls=TFD.InvLvls, Invd=TFD.Invdict,
            Tempd=TFD.Tempd, Retrod=TFD.Retrod, VMinMax=[None,None], Test=True, Com='', DTF=None, RatDef=100, Method='Max', Trunc=0.60, Inst=True, SpectNorm=True, cmapPow=plt.cm.gray_r, a4=False):
        TMat = self.GMat.MatLOS if self._LOS else self.GMat.Mat
        Nit = np.array([ss[0] for ss in self._Spec])
        axInv, axTMat, Laxtemp = TFPT.Inv_PlotFFTPow(self.BF2, self.Coefs, RZPts=PtsRZ, t=self.t, Com=Com, shot=self.shot, Ves=self.GMat.Ves, SXR=self.data, sigma=self._sigma, TMat=TMat, Deriv=Deriv, DVect=DVect, SubP=SubP, SubMode=SubMode, InvPlotFunc=InvPlotFunc, InvLvls=InvLvls, Invd=Invd, Tempd=Tempd, Retrod=Retrod, VMinMax=VMinMax, DTF=DTF, RatDef=RatDef, Method=Method, Trunc=Trunc, Inst=Inst, SpectNorm=SpectNorm, cmapPow=cmapPow, a4=a4, Test=Test)
        return axInv, axTMat, Laxtemp

    def posttreat(self, pathfileext, kwdargs={}, overwrite=False, update=False):
        assert type(pathfileext) is str, "Arg pathfileext must be a str !"
        import os
        strrev = pathfileext[-1::-1]
        Path = pathfileext[:-strrev.index('/')]
        Mod = pathfileext[-strrev.index('/'):-strrev.index('.')-1]
        cwd = os.getcwd()
        os.chdir(Path)
        Path = os.getcwd()
        exec "import "+Mod+" as pt"
        os.chdir(cwd)
        Out = pt.Init(self, kwdargs=kwdargs, overwrite=overwrite, update=update)
        if not Out is None:
            Out['Path'] = Path
            Out['Mod'] = Mod
            if not 'Name' in Out.keys():
                Out['Name'] = Mod
            ind = [ii for ii in range(0,len(self._PostTreat)) if self._PostTreat[ii]['Name']==Out['Name']]
            assert len(ind) in [0,1], "Several matching Sol2D.PostTreat  !"
            if len(ind)==1:
                self._PostTreat[ind[0]] = Out
            else:
                self._PostTreat.append(Out)

    def posttreat_plot(self, name=None, ind=-1, **kwdargs):
        if not name is None:
            ind = [ii for ii in range(0,len(self.PostTreat)) if self.PostTreat[ii]['Name']==name][0]
        import os
        gcwd = os.getcwd()
        os.chdir(self.PostTreat[ind]['Path'])
        exec "import "+self.PostTreat[ind]['Mod']+" as pt"
        os.chdir(gcwd)
        return pt.plot(self, self.PostTreat[ind], **kwdargs)

    def save(self,SaveName=None,Path=None,Mode='npz'):
        """
        Save the obj to specified path and name, with numpy of cPickle

        Call :
        ------

        obj.save( SaveName=None, Path=None, Mode='npz' )

        Arguments:
        ----------

            Inputs :
                SaveName    str or None     Specifies the name under which the file should be saved, if None uses obj.Id.SaveName (default: None)
                Path        str or None     Specifies the relative path in which to save the file, if None uses obj.Id.SavePath (default: None)
                Mode        str             Specifies the format used for saving, 'npz' uses numpy to store the data in numpy arrays while 'pck' uses cPickle to store the whole object structure (default: 'npz')

        Examples:
        ---------

        >> Sol.save(Path='./')
            Will save the obj called Sol in the current folder using its defaults Id.SaveName and numpy
        """
        if Path is None:
            Path = self.Id.SavePath
        else:
            assert type(Path) is str, "Arg Path must be a str !"
            self._Id.SavePath = Path
        if SaveName is None:
            SaveName = self.Id.SaveName
        else:
            assert type(SaveName) is str, "Arg SaveName must be a str !"
            self.Id.SaveName = SaveName
        Ext = '.npz' if 'npz' in Mode else '.pck'
        TFPF.save(self, Path+SaveName+Ext)









##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############            Routines
##############################################################################################################################################
##############################################################################################################################################



def _Sol2D_get_InvRad(Sol, PreDt, PostDt, SubP=TFD.InvPlotSubP, SubMode=TFD.InvPlotSubMode, Test=True):
    if Test:
        assert type(PreDt) in [int,float,list], "Arg PreDt must be of type in [int,float,list] !"
        assert type(PostDt) in [int,float,list], "Arg PostDt must be of type in [int,float,list] !"
        if type(PreDt) is list:
            assert len(PreDt)==2, "Arg PreDt must be a list of len()==2 !"
        if type(PostDt) is list:
            assert len(PostDt)==2, "Arg PostDt must be a list of len()==2 !"

    # Get coefficients
    Coefs = [PreDt,PostDt]
    for ii in range(0,2):
        if type(Coefs[ii]) is int:
            Coefs[ii] = Sol.Coefs[Coefs[ii]]
        elif type(Coefs[ii]) is float:
            Coefs[ii] = Sol.Coefs[np.nanargmin(np.abs(Sol.t-Coefs[ii]))]
        elif type(Coefs[ii]) is list:
            indt = (Sol.t>=Coefs[ii][0]) & (Sol.t<=Coefs[ii][1])
            Coefs[ii] = np.nanmean(Sol.Coefs[indt,:],axis=0)
    Coefs = Coefs[0]-Coefs[1]

    # Get mesh and values
    Rplot, Zplot, nR, nZ = Sol.BF2.get_XYplot(SubP=SubP, SubMode=SubMode)
    Pts = np.array([Rplot.flatten(), np.zeros((nR*nZ,)), Zplot.flatten()])
    indnan = ~Sol.BF2.Mesh.isInside(np.array([Rplot.flatten(),Zplot.flatten()]))
    Vals = np.nan*np.ma.ones((nR*nZ,))
    Vals[~indnan] = Sol.BF2.get_TotVal(Pts[:,~indnan], Deriv='D0', Coefs=Coefs, Test=Test)
    indpeak = np.nanargmax(np.abs(Vals))
    Peak = Pts[:,indpeak]
    Vals = np.reshape(Vals,(nZ,nR))
    print Peak

    # Get contours containing peak
    Lcont = cntr.Cntr(Rplot, Zplot, Vals)
    Lcont = Lcont.trace(0.)
    print Lcont
    print len(Lcont)
    nseg = len(Lcont) // 2
    print nseg
    segments, codes = Lcont[:nseg], Lcont[nseg:]
    print segments
    print codes

    inds = np.zeros((len(Lcont)),dtype=bool)
    for ii in range(0,len(Lcont)):
        Pth = path.Path(Lcont[ii])
        print Pth
        print Pth.contains_points(Peak)
        if Pth.contains_points(Peak):
            inds[ii] = True
    assert inds.sum()>=1, "TFI._Sol2D_get_InvRad : could not find a 0-contour containing the peak for "+Sol.Id.Name+" with specified intput !"
    Lcont = [Lcont[ii] for ii in range(0,len(Lcont)) if inds[ii]]
    print Lcont

    # Select innermost contour
    #if inds.sum()>1:
        #Area =


    return Pts#, Area

















