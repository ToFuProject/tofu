# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 18:12:05 2014

@author: didiervezinet
"""

# Module used for data treatment


import numpy as np
import scipy.interpolate as scpinterp
import matplotlib.pyplot as plt
import itertools as itt


# ToFu-specific
import tofu.pathfile as tfpf



####################################################
####################################################
#       Helper routines
####################################################

def get_DefName(t=None, Dt=None):
    Dtstr = 'DtNaN' if Dt is None else 'Dt{0:07.4f}-{1:07.4f}s'.format(Dt[0],Dt[1])
    if not t is None:
        f = 1./np.mean(np.diff(t))
        fcoefs = [([1.,1000.],1.,'Hz'),([1000.,1.e6],0.001,'kHz'),([1.e6,1.e9],1.e-6,'MHz')]
        ind = [ii for ii in range(0,3) if f>=fcoefs[ii][0][0] and f<fcoefs[ii][0][1]][0]
        f = fcoefs[ind][1]*f
        fstr = '{0:03.0f}'.format(int(f))+fcoefs[ind][2]
    else:
        fstr = 'NaNHz'
    Name = Dtstr+'_'+fstr
    return Name



####################################################
####################################################
#       Routines for PreData class
####################################################


def _PreData_set_data(data, t=None, Chans=None, DtRef=None, LIdDet=None):

    # Formatting
    if t is None:
        t = np.arange(0,data.shape[0])
    if Chans is None:
        Chans = map(str,np.arange(0,data.shape[1]))
    if not LIdDet is None:
        # Make sure LIdDet is is the good order
        LIdN = [Id.Name for Id in LIdDet]
        ind = [LIdN.index(Chans[ii]) for ii in range(0,len(Chans))]
        LIdDet = [LIdDet[ii] for ii in ind]
    else:
        LIdDet = [tfpf.ID('Detect',Chans[ii],Exp='Misc',Diag='Misc',shot=0) for ii in range(0,data.shape[1])]

    dataRef, tRef, ChanRef, NChanRef, LIdDetRef = np.copy(data), np.copy(t), list(Chans), len(Chans), list(LIdDet)
    data2, t2, Chans2, NChans, LIdDet2 = None, None, None, None, None

    if DtRef is None:
        DtRef, Dt = [t[0],t[-1]], None
    else:
        DtRef, Dt = list(DtRef), list(DtRef)

    indOut = np.zeros((NChanRef,),dtype=bool)
    indCorr = np.zeros((NChanRef,),dtype=bool)

    return (dataRef, tRef, ChanRef, NChanRef, LIdDetRef, DtRef), (data2, t2, Chans2, NChans, LIdDet2, Dt), (indOut,indCorr)



def _PreData_interp(lt=[], lNames=[]):
    # Formatting input
    assert type(lt) in [list, np.ndarray, float], "Arg lt must be a np.ndarray, a float or a list of such !"
    lt = [lt] if type(lt) is float else lt
    assert lNames==[] or (type(lNames) is str and len(lt)==1) or (type(lNames) is list and len(lNames)==len(lt) and all([type(nn) in [list,str] for nn in lNames])), "Arg lNames must be a str or a list, with len=len(lt) !"
    if lNames==[]:
        lNames = ['All' for ii in range(0,len(lt))]
    if type(lNames) is str:
        lNames = [lNames]
    ltU = sorted(list(set(lt)))
    for ii in range(0,len(lNames)):
        lNames[ii] = [lNames[ii]] if type(lNames[ii]) is str else lNames[ii]

    # Computing
    lN = []
    for ii in range(0,len(ltU)):
        indt = [jj for jj in range(0,len(lt)) if lt[jj]==ltU[ii]]
        ln = sorted(list(set(itt.chain.from_iterable([lNames[jj] for jj in indt]))))
        lN.append(ln)
    UNames = sorted(list(set(itt.chain.from_iterable(lN))))
    for ii in range(0,len(lN)):
        lN[ii] = lN[ii][0] if len(lN[ii])==1 else lN[ii]

    return ltU, lN, UNames


def _PreData_doAll_interp(data, t, lt, lNames, UNames, inds):
    Indt = np.asarray([np.argmin(np.abs(t-tt)) for tt in lt])

    data = np.copy(data)
    # Interpolate 'All'
    indAll = np.asarray([lNames[jj]=='All' for jj in range(0,len(lt))])
    if np.any(indAll):
        indt = np.zeros((t.size,),dtype=bool)
        indt[Indt[indAll]] = True
        ff = scpinterp.interp1d(t[~indt], data[~indt,:], axis=0, kind='linear')
        data[indt,:] = ff(t[Indt[indAll]])

    # Interpolate each individual channel
    for ii in range(0,len(UNames)):
        indt = np.zeros((t.size,),dtype=bool)
        # Identify the time points corresponding to one given channel
        for jj in range(0,len(Indt)):
            if UNames[ii] in lNames[jj] or UNames[ii]==lNames[jj]:
                indt[Indt[jj]] = True
        # Interpolate the channel
        data[indt,inds[ii]] = np.interp(t[indt], t[~indt], data[~indt,inds[ii]])
    return data



def _PreData_doAll_Resamp(DtRef, tRef, data, Resamp_t, Resamp_f, Resamp_Method, Resamp_interpkind):
    """ Compute the data time re-sampling """

    # Compute the new time vector (t being the priority)
    if not Resamp_t is None:
        assert Resamp_t.ndim==1 and np.all(Resamp_t==np.unique(Resamp_t)) and Resamp_t[0]>=tRef[0] and Resamp_t[-1]<=tRef[-1], "The time vector chosen for re-sampling is not fit !"
        t = Resamp_t
    elif not Resamp_f is None:
        assert 1./Resamp_f < (tRef[-1]-tRef[0])/2., "The chosen frequency is not fit !"
        t = np.linspace(DtRef[0],DtRef[-1], np.ceil((DtRef[-1]-DtRef[0])*Resamp_f))

    # Compute data
    if Resamp_Method=='movavrg':
        data = MovAverage(data, tRef, Resamp_f, tResamp=False)[0]

    ff = scpinterp.interp1d(tRef, data, axis=0, kind=Resamp_interpkind)
    data = ff(t)
    return data, t


def _PreData_doAll_Subtract(data, t, tsub):
    if hasattr(tsub,'__iter__'):
        indt = (t>=tsub[0]) & (t<=tsub[-1])
        datasub = np.mean(data[indt,:],axis=0)
    else:
        indt = np.nanargmin(np.abs(t-tsub))
        datasub = data[indt,:]
    return data - np.tile(datasub,(t.size,1))



def _PreData_doAll_FFT(data, t, FFTPar):
    if not FFTPar['DF'] is None or not FFTPar['DFEx'] is None:
        data = FourierExtract(t, data, **FFTPar)[0]
    return data



def _PreData_set_NoiseModel(PhysNoise, Chans, NChans, Nt, Deg, Nbin, LimRatio, Plot=False):
    assert not PhysNoise is None, "PhysNoise must be estimated first (use self.set_PhysNoise() method) !"
    NoiseCoefs = np.zeros((Deg+1,NChans))
    Err = np.zeros((Nbin,))
    if Plot:
        f = plt.figure(facecolor='w',figsize=(10,8))
        ax = f.add_axes([0.05, 0.05, 0.85, 0.85],frameon=True,axisbg='w')
    if Deg>=1:
        Noise = np.zeros((Nt,NChans))
        for ii in range(0,NChans):
            bins =  np.linspace(PhysNoise['Phys'][:,ii].min(),PhysNoise['Phys'][:,ii].max(),Nbin+1)
            inds = np.digitize(PhysNoise['Phys'][:,ii], bins)-1
            for jj in range(0,Nbin):
                Err[jj] = np.std(PhysNoise['Noise'][inds==jj,ii])
            NoiseCoefs[:,ii] = np.polyfit(0.5*(bins[1:]+bins[:-1]), Err, Deg)
            if Plot:
                ax.plot(0.5*(bins[1:]+bins[:-1]), Err, label=Chans[ii])
            Noise[:,ii] = np.polyval(NoiseCoefs,PhysNoise['Phys'][:,ii])
            if not LimRatio is None:
                Lim1 = LimRatio*np.mean(Noise[:,ii])
                Lim2 = np.exp(np.mean(np.log(Noise[:,ii]))-2.*np.std(np.log(Noise[:,ii])))
                Lim3 = np.sort(Noise[:,ii])[1]
                Lim = max([Lim1,Lim2,Lim3])
                Noise[Noise[:,ii]<Lim,ii] = Lim
    else:
        Nbin, NoiseCoefs = 1, NoiseCoefs.flatten()
        for ii in range(0,NChans):
            NoiseCoefs[ii] = np.std(PhysNoise['Noise'][:,ii])
        Noise = np.copy(NoiseCoefs)
        if not LimRatio is None:
            Lim1 = LimRatio*np.mean(Noise)
            Lim2 = np.exp(np.mean(np.log(Noise))-2.*np.std(np.log(Noise)))
            Lim3 = np.sort(Noise)[1]
            Lim = max([Lim1,Lim2,Lim3])
            Noise[Noise<Lim] = Lim

    NoiseModel = {'Deg':Deg, 'Nbin':Nbin, 'LimRatio':LimRatio}
    return NoiseModel, Noise, NoiseCoefs






####################################################
####################################################
#       Data treatment routines
####################################################




# --------- Moving average ---------------------------

def MovAverage(data, time, MovMeanfreq, tResamp=False, interpkind='linear', Mode=None, Test=True):
    if Test:
        assert isinstance(time,np.ndarray) and time.ndim==1, "Arg time must be a (N,) np.ndarray !"
        assert isinstance(data,np.ndarray) and data.shape[0]==time.size, "Arg data must be a (N,M) np.ndarray !"
        assert type(MovMeanfreq) in [int, float, np.float64], "Arg MovMeanfreq must be a int or a float !"
        assert type(tResamp) in [bool,np.ndarray], "Arg Resamp must be a bool or an array !"
    Nt = time.size
    time = np.copy(time)
    data = np.copy(data)
    if data.ndim==1:
        data = data.reshape((Nt,1))
    NP = int(round(1./(np.mean(np.diff(time))*MovMeanfreq)))       # Getting the number of points for the convolution
    window = np.ones((NP,),dtype=float)/float(NP)               # Building the convolution window with that number of points
    for ii in range(0,data.shape[1]):
        data[:,ii] = np.convolve(data[:,ii], window, mode='same')
    if not Mode is None:
        #Ind = np.zeros((Nt,),dtype=bool)
        if NP%2==0:
            ind = np.arange(int(NP/2),int(Nt-NP/2))
        else:
            ind = np.arange(int(np.ceil(NP/2)),int(Nt-np.floor(NP/2)))
        #Ind[ind] = True
        time = time[ind]
    if Mode=='trunc':
        data = data[ind,:]
    # if required, interpolate on a different time base
    if not tResamp is False:
        tResamp = np.linspace(time[0],time[-1], np.ceil((time[-1]-time[0])*MovMeanfreq)) if tResamp is True else tResamp
        ff = scpinterp.interp1d(time, data, axis=0, kind=interpkind)   # Interpolate data to match new time base
        data = ff(tResamp)
        time = tResamp
    return data, time


def Plot_MovAverage(data, time, MovMeanfreq, Resamp=True, interpkind='linear', Test=True):
    databis, timebis = MovAverage(data, time, MovMeanfreq, Resamp=Resamp, interpkind=interpkind, Test=Test)
    plt.figure()
    plt.plot(time, data, c=(0.8,0.8,0.8))
    plt.plot(timebis,databis, c=(1.,0.,0.))
    return plt.gca()





# --------- SVD ------------------------------------

def SVDExtractPhysNoise(data, Modes=range(0,10)):
    u,s,v = np.linalg.svd(data, full_matrices=1, compute_uv=1)
    indPhys = Modes
    indNoise = np.arange(0,s.size)
    indNoise = np.delete(indNoise,Modes)

    Physic = u[:,indPhys].dot(np.diag(s[indPhys])).dot(v[indPhys,:])
    Noise = u[:,indNoise].dot(np.diag(s[indNoise])).dot(v[indNoise,:])
    return Physic, Noise


def Calc_NoiseModelFromSVD(noise, Type='Const', physics=None, data=None):
    if Type=='Const':
        sigma = np.std(noise,axis=0)
    return sigma





# --------- Fourier analysis ------------------------------------


def FourierMainFreq(data, t, Method='Max', Trunc=0.60, Plot=False, Test=True):
    if Test:
        assert isinstance(data,np.ndarray) and isinstance(t,np.ndarray) and t.ndim==1 and data.shape[0]==t.size, "Args t and data must be np.ndarray with data.shape[0]==t.size !"
    if data.ndim==1:
        data = np.reshape(data,(data.size,1))
    Dt = np.mean(np.diff(t))
    Nt = t.size
    Freq = np.fft.fftfreq(Nt, Dt).flatten()
    Freq = Freq[Freq>=0]
    if Nt%2==0:
        Freq = np.concatenate((Freq,np.array([Dt*Nt/2])))
    NF = Freq.size
    A = np.fft.rfft(data, axis=0).flatten()
    assert A.shape[0]==NF, "Problem with frequency scale !"
    if Plot:
        ind = np.abs(A)**2 >= Trunc*np.max(np.abs(A)**2)
        f = plt.figure()
        plt.plot(Freq, np.abs(A)**2, c='k', lw=2)
        F1 = Freq[np.argmax(np.abs(A)**2, axis=0)]
        F2 = np.sum(Freq[ind]*np.abs(A[ind])**2)/np.sum(np.abs(A[ind])**2)
        F3 = np.sum(Freq*np.abs(A)**2)/np.sum(np.abs(A)**2)
        plt.axvline(F1, 0, 1, c='k', ls='--', label='Max')
        plt.axvline(F2, 0, 1, c='k', ls='--', label='Trunc')
        plt.axvline(F3, 0, 1, c='k', ls='--', label='Average')
        plt.show()
    if Method=='Max':
        Freq = Freq[np.argmax(np.abs(A)**2, axis=0)]
    elif Method=='Trunc':
        ind = np.abs(A)**2 >= Trunc*np.max(np.abs(A)**2)
        Freq = np.sum(Freq[ind]*np.abs(A[ind])**2)/np.sum(np.abs(A[ind])**2)
    elif Method=='Average':
        Freq = np.sum(Freq*np.abs(A)**2)/np.sum(np.abs(A)**2)
    return Freq

def FourierExtract(t, data, DF=None, DFEx=None, Harm=True, HarmEx=True, Test=True):
    """ Return the FFT-filtered signal (and the rest) in the chosen frequency interval (in Hz) and in all the higher harmonics (optional)

    Can also exclude a given interval and its higher harmonics from the filtering (optional)

    Parameters
    ----------
    t :         np.ndarray
        1D array, monotonously increasing time vector with regular spacing
    data :      np.ndarray
        1 or 2D array, with shape[0]=t.size, the data to be filtered
    DF :        list
        List or tuple of len()=2, containing the lower and upper bounds of the frequency interval to be used for filtering
    Harm :      bool
        If True all the higher harmonics of the interval DF will also be included, default=True
    DFEx :      list
        List or tuple of len()=2, containing the lower and upper bounds of the frequency interval to be excluded from filtering (in case it overlaps with some high harmonics of DF), default=None
    HarmEx :    bool
        If True all the higher harmonics of the interval DFEx will also be excluded, default=True
    Test :      bool
        If True tests all the input arguments for any mistake

    Returns
    -------
    In :        np.ndarray
        Array with shape=data.shape, contains the filtered signal retrieved from reverse FFT
    Out :       np.ndarray
        Array with shape=data.shape, contains the signal excluded from filtering retrieved from reverse FFT

    """

    if Test:
        assert isinstance(data,np.ndarray) and isinstance(t,np.ndarray) and t.ndim==1 and data.shape[0]==t.size, "Args t and data must be np.ndarray with data.shape[0]==t.size !"
        assert DF is None or (hasattr(DF,'__getitem__') and len(DF)==2), "Arg DF must be a list, tuple or np.ndarray of size 2 !"
        assert DFEx is None or (hasattr(DFEx,'__getitem__') and len(DFEx)==2), "Arg DFEx must be a list, tuple or np.ndarray of size 2 !"
        assert type(Harm) is bool and type(HarmEx) is bool, "Args Harm and HarmEx must be bool !"
    if data.ndim==1:
        data = np.reshape(data,(data.size,1))
    Dt = np.mean(np.diff(t))
    Nt = t.size
    Freq = np.fft.fftfreq(Nt, Dt)
    Freq = Freq[Freq>=0]
    if Nt%2==0:
        Freq = np.concatenate((Freq,np.array([Dt*Nt/2])))
    NF = Freq.size
    A = np.fft.rfft(data, axis=0)
    assert A.shape[0]==NF, "Problem with frequency scale !"

    # Defining the intervals of interest for reconstruction
    if not DF is None:
        indin = (Freq>=DF[0]) & (Freq<=DF[1])
        if Harm:
            for ii in range(1,int(np.floor(Freq.max()/DF[0]))):
                indin = indin | ((Freq>=DF[0]*ii) & (Freq<=DF[1]*ii))
    else:
        indin = np.ones((NF,),dtype=bool)
    if not DFEx is None:
        indin = indin & ~((Freq>=DFEx[0]) & (Freq<=DFEx[1]))
        if HarmEx:
            for ii in range(1,int(np.floor(Freq.max()/DFEx[0]))):
                indin = indin & ~((Freq>=DFEx[0]*ii) & (Freq<=DFEx[1]*ii))
    # Reconstructing
    Aphys = np.copy(A)
    Aphys[~indin,:] = 0
    A[indin,:] = 0
    return np.fft.irfft(Aphys, Nt, axis=0), np.fft.irfft(A, Nt, axis=0)


def FourierPowSpect(data, t, DTF=None, RatDef=100., Test=True):
    if Test:
        assert isinstance(data,np.ndarray) and isinstance(t,np.ndarray) and t.ndim==1 and data.shape[0]==t.size, "Args t and data must be np.ndarray with data.shape[0]==t.size !"
        assert data.ndim==1 or 1 in data.shape, "Arg data must be 1D !"
    if not data.ndim==1:
        data = data.flatten()
    Dt = np.mean(np.diff(t))
    Nt = t.size
    if DTF is None:
        DTF = (t.max()-t.min())/RatDef
    Nnt = int(np.ceil(DTF/Dt))
    Freq = np.fft.fftfreq(Nnt, Dt)
    Freq = Freq[Freq>=0]
    if Nnt%2==0:
        Freq = np.concatenate((Freq,np.array([Dt*Nnt/2])))
    NF = Freq.size
    Pow = np.nan*np.zeros((Nt,NF))
    nn = Nnt/2
    Res = Nnt%2
    for ii in range(0,Nt):
        if ii>=nn and ii<=Nt-nn-Res:
            indt = np.arange(ii-nn,ii+nn+Res)
            A = np.fft.rfft(data[indt], axis=0)
            assert A.size==NF, "Problem with frequency scale !"
            Pow[ii,:] = np.abs(A)**2
    ind = np.argsort(Freq)
    return Pow[:,ind], Freq[ind]


def FourierPowSpect_V2(data, t, DTF=None, RatDef=100., Test=True):
    if Test:
        assert isinstance(data,np.ndarray) and isinstance(t,np.ndarray) and t.ndim==1 and data.shape[0]==t.size, "Args t and data must be np.ndarray with data.shape[0]==t.size !"
        assert data.ndim==1 or 1 in data.shape, "Arg data must be 1D !"
    if not data.ndim==1:
        data = data.flatten()
    Dt = np.mean(np.diff(t))
    Nt = t.size
    if DTF is None:
        DTF = (t.max()-t.min())/RatDef
    Nnt = int(np.ceil(DTF/Dt))
    Freq = np.fft.fftfreq(Nnt, Dt)
    Freq = Freq[Freq>=0]
    if Nnt%2==0:
        Freq = np.concatenate((Freq,np.array([Dt*Nnt/2])))
    NF = Freq.size
    Pow = np.nan*np.zeros((Nt,NF))
    nn = Nnt/2
    Res = Nnt%2
    for ii in range(0,Nt):
        if ii>=nn and ii<=Nt-nn-Res:
            indt = np.arange(ii-nn,ii+nn+Res)
            A = np.fft.rfft(data[indt]-np.mean(data[indt]), axis=0)
            assert A.size==NF, "Problem with frequency scale !"
            Pow[ii,:] = np.abs(A)**2
    ind = np.argsort(Freq)
    return Pow[:,ind], Freq[ind]


def Fourier_MainFreqPowSpect(data, t, DTF=None, RatDef=100, Method='Max', Trunc=0.60, V=2, Test=True):
    """ Get the dominant frequency of the running FFT of a 1D signal, and the power spectrum """
    Nt = t.size
    if V==1:
        Pow, Freq = FourierPowSpect(data, t, DTF=DTF, RatDef=RatDef, Test=Test)
    elif V==2:
        Pow, Freq = FourierPowSpect_V2(data, t, DTF=DTF, RatDef=RatDef, Test=Test)
    indnan = np.any(np.isnan(Pow),axis=1)
    MainFreq = np.nan*np.ones((t.size,))
    if Method=='Max':
        MainFreq[~indnan] = Freq[np.argmax(Pow[~indnan,:], axis=1)]
    elif Method=='Trunc':
        ind = (Pow >= Trunc*np.tile(np.nanmax(Pow,axis=1),(Nt,1))) & (~indnan)
        MainFreq[ind] = np.sum(Freq[ind]*Pow[ind],axis=1)/np.sum(Pow,axis=1)
    elif Method=='Average':
        MainFreq[~indnan] = np.sum(Freq*Pow[~indnan,:],axis=1)/np.sum(Pow[~indnan,:],axis=1)
    return Pow, MainFreq, Freq





# ------------- Hybrid smoothing methods ------------------------

def ModeSmoothing(t, data, Freq, N=5, Test=True):
    """ Smoothe a mode by doing a running average of N data points separated by the estimated period """

    if Test:
        assert isinstance(t,np,ndarray) and t.ndim==1, "Arg t must be a (Nt,) np.ndarray !"
        assert isinstance(data,np.ndarray) and data.ndim in [1,2] and data.shape[0]==t.size, "Arg data must be a (Nt,) or (Nt,Np) np.ndarray !"
        assert (not hasattr(Freq,'__getitem__') and type(Freq) in [float, int]) or (isinstance(Freq,np.ndarray) and Freq.ndim==1 and Freq.size==t.size), "Arg Freq must be a single frequency value (in Hz) or the time trace of a frequency value !"
        assert type(N) is int and N>0 and N%2==1, "Arg N must be a strictly positive odd int !"

    NDim = data.ndim
    Nt = t.size
    if NDim==1:
        data = data.reshape((Nt,1))
    if not hasattr(Freq,'__getitem__'):
        Freq = Freq*np.ones((Nt,))

    T = 1./Freq
    nn = (N-1)/2
    indtbis = []
    t1, t2 = t.min(), t.max()
    for ii in range(0,Nt):
        indt = [ii]
        ttmp1, temp2 = np.copy(t[ii]), np.copy(t[ii])
        indt1, indt2 = np.copy([ii]), np.copy([ii])
        for jj in range(0,nn):
            ttmp1 = ttmp1-T[indt1]
            ttmp2 = ttmp2+T[indt2]
            if ttmp1>=t1:
                indt1 = np.argmin(np.abs(t-ttmp1))
                indt.append(indt1)
            if ttmp2<=t2:
                indt2 = np.argmin(np.abs(t-ttmp2))
                indt.append(indt2)
        indtbis.append(np.sort(indt).astype(int))

    databis = np.nan*np.ones(data.shape)
    for ii in range(0,Nt):
        databis[ii,:] = np.mean(data[ii,indtbis[ii]])
    return databis



