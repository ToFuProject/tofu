# -*- coding: utf-8 -*-

# Builtin
import warnings

# Common
import numpy as np
import scipy.signal as scpsig
import scipy.interpolate as scpinterp



#############################################
#       Spectrograms
#############################################


def spectrogram(data, t=None, method='',
                **kwdargs):

    lm = ['scipy-fourier', 'scipy-welch']

    nt, nch = data.shape
    if t is not None:
        assert t.shape==(nt,)
    else:
        t = np.arange(0,nt)
    if not np.allclose(t,t[0]+np.mean(np.diff(t))*np.arange(0,nt)):
        msg = "Time vector does not seem to be regularly increasing !"
        raise Exception(msg)

    if method=='scipy-fourier':
        out = _spectrogram_scipy_fourier(data, 1./dt, nt, nch, **kwdargs)
    elif method=='scipy-wavelet':
        out = _spectrogram_scipy_wavelet(data, 1./dt, nt, nch, **kwdargs)
    elif method=='irfm-ece':
        out = _spectrogram_irfm_ece(data, 1./dt, nt, nch, **kwdargs)

    return t, f, psd


def _spectrogram_scipy_fourier(data, fs, nt, ch, fmin=None, fmax=None,
                               window=('tukey', 0.25), detrend='linear'):

    # Format inputs
    if fmin is None:
        fmin = 10.*(fs/nt)
    assert fmin > fs/nt
    if fmax is None:
        fmax = fs/2.01
    assert fmax < fs/2.

    # Deduce parameters
    nperseg = int(np.ceil(fs/fmin))
    noverlap = nperseg - 1
    n = int(np.ceil(np.log(nperseg)/np.log(2)))
    nfft = 2**n

    # Prepare output

    f, tf, ssx = scpsig.spectrogram(data[:,ii], fs=fs,
                                    window=window, nperseg=nperseg,
                                    noverlap=noverlap, nfft=nfft,
                                    detrend=detrend, return_onesided=True,
                                    scaling='density', axis=-1, mode='psd')

    lpsd = [np.full((nt,f.size),np.nan) for ii in range(0,nch)]
    ind = np.arange(nperseg/2, nt-nperseg/2)
    lssx = np.split(ssx, ind, axis=1)

    for ii in lssx = [ss.squeeze() for ss in lssx]

    return f, lssx







def FourierExtract(t, data, df=None, dfEx=None, Harm=True, HarmEx=True, Test=True):
    """ Return the FFT-filtered signal (and the rest) in the chosen frequency interval (in Hz) and in all the higher harmonics (optional)

    Can also exclude a given interval and its higher harmonics from the filtering (optional)

    Parameters
    ----------
    t :         np.ndarray
        1D array, monotonously increasing time vector with regular spacing
    data :      np.ndarray
        1 or 2D array, with shape[0]=t.size, the data to be filtered
    df :        list
        List or tuple of len()=2, containing the lower and upper bounds of the frequency interval to be used for filtering
    Harm :      bool
        If True all the higher harmonics of the interval df will also be included, default=True
    dfEx :      list
        List or tuple of len()=2, containing the lower and upper bounds of the frequency interval to be excluded from filtering (in case it overlaps with some high harmonics of df), default=None
    HarmEx :    bool
        If True all the higher harmonics of the interval dfEx will also be excluded, default=True
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
        assert df is None or (hasattr(df,'__getitem__') and len(df)==2), "Arg df must be a list, tuple or np.ndarray of size 2 !"
        assert dfEx is None or (hasattr(dfEx,'__getitem__') and len(dfEx)==2), "Arg dfEx must be a list, tuple or np.ndarray of size 2 !"
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
    if not df is None:
        indin = (Freq>=df[0]) & (Freq<=df[1])
        if Harm:
            for ii in range(1,int(np.floor(Freq.max()/df[0]))):
                indin = indin | ((Freq>=df[0]*ii) & (Freq<=df[1]*ii))
    else:
        indin = np.ones((NF,),dtype=bool)
    if not dfEx is None:
        indin = indin & ~((Freq>=dfEx[0]) & (Freq<=dfEx[1]))
        if HarmEx:
            for ii in range(1,int(np.floor(Freq.max()/dfEx[0]))):
                indin = indin & ~((Freq>=dfEx[0]*ii) & (Freq<=dfEx[1]*ii))
    # Reconstructing
    Aphys = np.copy(A)
    Aphys[~indin,:] = 0
    A[indin,:] = 0
    return np.fft.irfft(Aphys, Nt, axis=0), np.fft.irfft(A, Nt, axis=0)
