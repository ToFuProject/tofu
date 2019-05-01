# -*- coding: utf-8 -*-

# Builtin
import warnings

# Common
import numpy as np
import scipy.signal as scpsig
import scipy.interpolate as scpinterp
import scipy.linalg as scplin
import scipy.stats as scpstats

_fmin_coef = 5.



#############################################
#############################################
#############################################
#       Spectrograms
#############################################
#############################################


def spectrogram(data, t,
                fmin=None, method='scipy-fourier', deg=False,
                window='hann', detrend='linear',
                nperseg=None, noverlap=None,
                boundary='constant', padded=True, wave='morlet', warn=True):

    # Format/check inputs
    lm = ['scipy-fourier', 'scipy-stft']#, 'scipy-wavelet']
    if not method in lm:
        msg = "Alowed methods are:"
        msg += "\n    - scipy-fourier: scipy.signal.spectrogram()"
        msg += "\n    - scipy-stft: scipy.signal.stft()"
        #msg += "\n    - scpipy-wavelet: scipy.signal.cwt()"
        raise Exception(msg)

    nt, nch = data.shape
    if t is not None:
        assert t.shape==(nt,)
    else:
        t = np.arange(0,nt)
    if not np.allclose(t,t[0]+np.mean(np.diff(t))*np.arange(0,nt)):
        msg = "Time vector does not seem to be regularly increasing !"
        raise Exception(msg)
    dt = np.mean(np.diff(t))
    fs = 1./dt

    # Compute
    if method in ['scipy-fourier', 'scipy-stft']:
        stft = 'stft' in method
        f, tf, lpsd, lang = _spectrogram_scipy_fourier(data, fs, nt, nch, fmin=fmin,
                                                       stft=stft, deg=deg,
                                                       window=window,
                                                       nperseg=nperseg,
                                                       noverlap=noverlap,
                                                       detrend=detrend,
                                                       boundary=boundary,
                                                       padded=padded, warn=warn)
        tf = tf + t[0]
    elif method=='scipy-wavelet':
        f, lspect = _spectrogram_scipy_wavelet(data, fs, nt, nch,
                                               fmin=fmin, wave=wave, warn=warn)
        tf = t.copy()

    return tf, f, lpsd, lang


def _spectrogram_scipy_fourier(data, fs, nt, nch, fmin=None,
                               window=('tukey', 0.25), deg=False,
                               nperseg=None, noverlap=None,
                               detrend='linear', stft=False,
                               boundary='constant', padded=True, warn=True):
    """ Return a spectrogram for each channel, and a common frequency vector

    The min frequency of interest fmin fixes the nb. of pt. per seg. (if None)
    The number of overlapping points is set to nperseg-1 if None
    The choice of the window type is a trade-off between:
        Spectral resolution between similar frequencies/amplitudes:
            =>
        Dynamic range (lots of != frequencies of != amplitudes):
            =>
        Compromise:
            => 'hann'
    """

    # Check inputs
    if nperseg is None and fmin is None:
        fmin = _fmin_coef*(fs/nt)
        if warn:
            msg = "nperseg and fmin were not provided\n"
            msg += "    => fmin automatically set to 10.*fs/nt:\n"
            msg += "       fmin = 10.*{0} / {1} = {2} Hz".format(fs,nt,fmin)
            warnings.warn(msg)

    # Format inputs
    if nperseg is None:
        assert fmin > fs/nt
        nperseg = int(np.ceil(fs/fmin))

    if nperseg%2==1:
        nperseg = nperseg + 1
    if noverlap is None:
        noverlap = nperseg - 1
    n = int(np.ceil(np.log(nperseg)/np.log(2)))
    nfft = 2**n

    # Prepare output
    if stft:
        f, tf, ssx = scpsig.stft(data, fs=fs,
                                 window=window, nperseg=nperseg,
                                 noverlap=noverlap, nfft=nfft, detrend=detrend,
                                 return_onesided=True, boundary=boundary,
                                 padded=padded, axis=0)
    else:
        f, tf, ssx = scpsig.spectrogram(data, fs=fs,
                                        window=window, nperseg=nperseg,
                                        noverlap=noverlap, nfft=nfft,
                                        detrend=detrend, return_onesided=True,
                                        scaling='density', axis=0,
                                        mode='complex')

    # Split in list (per channel)
    lssx = np.split(ssx, np.arange(1,nch), axis=1)
    lssx = [ss.squeeze().T for ss in lssx]
    lpsd = [np.abs(ss)**2 for ss in lssx]
    lang = [np.angle(ss, deg=deg) for ss in lssx]

    return f, tf, lpsd, lang


def _spectrogram_scipy_wavelet(data, fs, nt, nch, fmin=None, wave='morlet',
                               warn=True):

    if wave!='morlet':
        msg = "Only the morlet wavelet implmented so far !"
        raise Exception(msg)


    # Check inputs
    if fmin is None:
        fmin = _fmin_coef*(fs/nt)
        if warn:
            msg = "fmin was not provided => set to 10.*fs/nt"
            warnings.warn(msg)

    nw = int((1./fmin-2./fs)*fs)
    widths = 2.*np.pi*np.linspace(fmin,fs/2.,nw)
    wave = eval('scpsig.%s'%wave)

    for ii in range(0,nch):
        cwt = scpsig.cwt(data[:,ii], wave, widths)
        lcwt.append(np.abs(cwt)**2)

    f = widths/(2.*np.pi)
    return f, lcwt



#############################################
#############################################
#############################################
#       SVD
#############################################
#############################################

def calc_svd(data, lapack_driver='gesdd'):
    chronos, s, topos = scplin.svd(data, full_matrices=True, compute_uv=True,
                                   overwrite_a=False, check_finite=True,
                                   lapack_driver=lapack_driver)

    # Test if reversed correlation
    lind = [np.nanargmax(np.std(data,axis=0)),
            np.nanargmax(np.mean(data,axis=0)),
            0, data.shape[1]//2, -1]

    corr = np.zeros((len(lind),2))
    for ii in range(0,len(lind)):
        corr[ii,:] = scpstats.pearsonr(chronos[:,0], data[:,lind[ii]])

    ind = corr[:,1] < 0.05
    if np.any(ind):
        corr = corr[ind,0]
        if corr[np.argmax(np.abs(corr))] < 0.:
            chronos, topos = -chronos, -topos

    return chronos, s, topos





#############################################
#############################################
#############################################
#       Filtering
#############################################
#############################################


def filter_bandpass_fourier(t, data, method='stft', detrend='linear',
                            df=None, harm=True,
                            df_out=None, harm_out=True):
    """ Return bandpass FFT-filtered signal (and the rest)

    Optionnally include all higher harmonics
    Can also exclude a frequency interval and its higher harmonics

    Parameters
    ----------
    t :         np.ndarray
        1D array, monotonously increasing time vector with regular spacing
    data :      np.ndarray
        1 or 2D array, with shape[0]=t.size, the data to be filtered
    method:     str
        Flag indicating which method to use:
            - 'rfft':   scipy.fftpack.rfft
            - 'stft':   scipy.signal.stft
    df :        None / list
        List or tuple of len()=2, containing the bandpass lower / upper bounds
    harm :      bool
        If True all the higher harmonics of df will also be included
    df_out :    None / list
        List or tuple of len()=2, containing the bandpass lower / upper bounds
        to be excluded from filtering (if it overlaps with high harmonics of df)
    harm_out :  bool
        If True, the higher harmonics of the interval df_out are also excluded
    Test :      bool
        If True tests all the input arguments for any mistake

    Returns
    -------
    In :        np.ndarray
        Array with shape=data.shape, filtered signal retrieved from inverse FFT
    Out :       np.ndarray
        Array with shape=data.shape, excluded signal from filtering

    """
    # Check / format inputs
    assert data.ndim==2 and t.ndim==1
    assert data.shape[0]==(t.size,)
    assert np.allclose(t,np.unique(t)) and np.std(np.diff(t))<=1.e-12
    assert method in ['rfft','stft']
    lC = [df is None, df_out is None]
    assert np.sum(lC)<=1, "At least one of df or df_out must be provided !"
    assert type(harm) is bool and type(harm_out) is bool
    if df is not None:
        df = np.unique(df)
        assert df.shape==(2,)
    if df_out is not None:
        df_out = np.unique(df_out)
        assert df_out.shape==(2,)

    nt, nch = data.shape
    dt = np.mean(np.diff(t))
    fs = 1./dt

    if method=='rfft':
        data_in, data_out = _filter_bandpass_rfft(data, t, dt, fs, nt, nch,
                                                  df=df, harm=harm,
                                                  df_out=df_out,
                                                  harm_out=harm_out)
    elif methd=='stft':
        data_in, data_out = _filter_bandpass_stft(data, t, dt, fs, nt, nch,
                                                  df=df, harm=harm,
                                                  df_out=df_out,
                                                  harm_out=harm_out)


def _filter_bandpass_rfft(data, t, dt, fs,
                          nt, nch, df):

    # Get frequency
    f = np.fft.fftfreq(nt, dt)
    f = f[f>=0]
    if nt%2==0:
        f = np.append(f,dt*nt/2)
    nf = f.size

    # Get rfft
    A = np.fft.rfft(data, axis=0)
    assert A.shape[0]==nf, "Problem with frequency scale !"

    # Intervals of interest for reconstruction
    if df is None:
        indin = np.ones((nf,),dtype=bool)
    else:
        indin = (f>=df[0]) & (f<=df[1])
        if harm:
            for ii in range(1,int(np.floor(f.max()/df[0]))):
                indin = indin | ((f>=df[0]*ii) & (f<=df[1]*ii))
    if df_out is not None:
        indin = indin & ~((f>=df_out[0]) & (f<=df_out[1]))
        if harm_out:
            for ii in range(1,int(np.floor(f.max()/df_out[0]))):
                indin = indin & ~((f>=df_out[0]*ii) & (f<=df_out[1]*ii))

    # Reconstructing
    Aphys = np.copy(A)
    Aphys[~indin,:] = 0
    A[indin,:] = 0
    return np.fft.irfft(Aphys, nt, axis=0), np.fft.irfft(A, nt, axis=0)



def _filter_bandpass_stft(data, t, dt, fs,
                          nt, nch, df):

    # Get stft
    f, tf, ssx = scpsig.stft(data, fs=fs,
                             window=window, nperseg=nperseg,
                             noverlap=noverlap, nfft=nfft, detrend=False,
                             return_onesided=True, boundary=boundary,
                             padded=padded, axis=0)
    ssx = np.abs(ssx)**2
    nf = f.size

    # Intervals of interest for reconstruction
    if df is None:
        indin = np.ones((nf,),dtype=bool)
    else:
        indin = (f>=df[0]) & (f<=df[1])
        if harm:
            for ii in range(1,int(np.floor(f.max()/df[0]))):
                indin = indin | ((f>=df[0]*ii) & (f<=df[1]*ii))
    if df_out is not None:
        indin = indin & ~((f>=df_out[0]) & (f<=df_out[1]))
        if harm_out:
            for ii in range(1,int(np.floor(f.max()/df_out[0]))):
                indin = indin & ~((f>=df_out[0]*ii) & (f<=df_out[1]*ii))

    # Reconstructing
    ssxphys = np.copy(ssx)
    ssxphys[~indin,:] = 0
    ssx[indin,:] = 0
    data_in = scpsig.istft(ssxphys, fs=fs, window=window, nperseg=nperseg,
                           noverlap=noverlap, nfft=nfft,
                           input_onesided=True, boundary=boundary,
                           time_axis=0, freq_axis=0)
    data_out = scpsig.istft(ssx, fs=fs, window=window, nperseg=nperseg,
                            noverlap=noverlap, nfft=nfft,
                            input_onesided=True, boundary=boundary,
                            time_axis=0, freq_axis=0)

    return data_in, data_out




def filter_svd(data, lapack_driver='gesdd', modes=[]):
    """ Return the svd-filtered signal using only the selected mode

    Provide the indices of the modes desired

    """
    # Check input
    modes = np.asarray(modes,dtype=int)
    assert modes.ndim==1
    assert modes.size>=1, "No modes selected !"

    u, s, v = scplin.svd(data, full_matrices=False, compute_uv=True,
                         overwrite_a=False, check_finite=True,
                         lapack_driver=lapack_driver)
    indout = np.arange(0,s.size)
    indout = np.delete(indout, modes)

    data_in = np.dot(u[:,modes]*s[modes],v[modes,:])
    data_out = np.dot(u[:,indout]*s[indout],v[indout,:])
    return data_in, data_out
