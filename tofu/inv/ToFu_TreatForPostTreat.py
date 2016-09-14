








#################################################################
#################################################################
# --------- Miscellaneous Post-treatment ------------------------
#################################################################



def SpiralRadiusCenter(t, Pts, Freq, Ord=3, Method='CvH', Test=True):
    """ Compute the estimated generalized geometric radius from surface spanned by one period of a spiral, and compute the estimated center of rotation
    Inputs :
        t       (Nt,) np.ndarray, a 1D array of Nt time points
        Pts     (2,Nt) np.ndarray, trajectory of the point to be tracked, in 2D cartesian coordinates
        Freq    (Nt,) np.ndarray, a 1D array of the instantaneous estimated frequency of rotation
        Ord     int, the number of times the algorithm for estimating the center of rotation should be used recursively (optionnal, default=3, recommended >= 3)
        Method  str in ['Pure', 'CvH', 'FAv'], Method used for computing the center of rotation, 'Pure' assumes all polygons are simple, 'CvH' computes their convex hull and 'FAv' computes a Moving Average at chosen frequency
        Test    bool, if True then all the inputs are tested for conformity
    Outputs :
        r       (Nt,) np.ndarray, a 1D array of the estimated generalized geometric radius, defined as sqrt(S/(2pi)), with S the surface spanned by one period
        C       (2,Nt) np.ndarray, a 2D array represneting the 2D cartesian coordinates, as a functio of time of the estimated center of rotation

    Method described further in [1]
    [1] D. Vezinet, personal Notes, 'Deriving a radius from spiraling hot core with unknown center, unknown growth rate and non-circular base', May 2015
    """

    if Test:
        assert isinstance(t,np.ndarray) and t.ndim==1,                                  "Arg t must be a (Nt,) np.ndarray !"
        assert isinstance(Pts,np.ndarray) and Pts.ndim==2 and Pts.shape==(2,t.size),    "Arg Pts must be a (2,Nt) np.ndarray !"
        assert isinstance(Freq,np.ndarray) and Freq.ndim==1 and Freq.size==t.size,      "Arg Freq must be a (Nt,) np.ndarray !"
        assert type(Ord) is int and Ord>0,                                              "Arg Ord must be a strictly positive int !"
        assert Method in ['Pure', 'CvH', 'FAv'],                                        "Arg Method must be a str in ['Pure', 'CvH', 'FAv'] !"

    # Initialising
    Nt = t.size
    SS = np.nan*np.ones((Nt,))
    CC = np.nan*np.ones((2,Nt))
    t = np.copy(t)

    indnan = ~(np.any(np.isnan(Pts),axis=0) | np.isnan(Freq))
    t = t[indnan]
    Pts = Pts[:,indnan]
    Freq = Freq[indnan]

    Nt = t.size
    T2 = 0.5/Freq
    ss = np.nan*np.ones((Nt,))
    cc = np.nan*np.ones((2,Nt))
    LC = []

    # Getting the interval on which the surface can be calculated
    tbis = np.array([t+T2,t-T2])
    indtS = (t>=t[np.argmin(np.abs(t[~np.isnan(t)].min()-tbis[1,:]))]) & (t<=t[np.argmin(np.abs(t[~np.isnan(t)].max()-tbis[0,:]))])
    IndtS = indtS.nonzero()[0]

    # Computing
    for ii in range(0,IndtS.size):
        indt = (t >= t[IndtS[ii]]-T2[IndtS[ii]]) & (t <= t[IndtS[ii]]+T2[IndtS[ii]])
        pp = plg.Polygon(Pts[:,indt].T)
        ss[IndtS[ii]] = pp.area()
        if Method=='Pure':
            cc[:,IndtS[ii]] = pp.center()
        elif Method=='CvH':
            cc[:,IndtS[ii]] = plgUtils.convexHull(pp).center()
        elif Method=='FAv':
            cc[:,IndtS[ii]] = np.mean(Pts[:,indt],axis=1)

    SS[indnan] = ss
    if Ord==1:
        CC[:,indnan] = cc
        return np.sqrt(SS/np.pi), CC

    t1, t2 = t[indtS].min(), t[indtS].max()
    for oo in range(2,Ord+1):
        indtC = (t>=t[np.argmin(np.abs(t1-tbis[1,:]))]) & (t<=t[np.argmin(np.abs(t2-tbis[0,:]))])
        IndtC = indtC.nonzero()[0]
        cctemp = np.nan*np.ones((2,Nt))
        if Method=='Pure':
            for ii in range(0,IndtC.size):
                indt = (t >= t[IndtC[ii]]-T2[IndtC[ii]]) & (t <= t[IndtC[ii]]+T2[IndtC[ii]])
                pp = plg.Polygon(cc[:,indt].T)
                cctemp[:,IndtC[ii]] = pp.center()
        elif Method=='CvH':
            for ii in range(0,IndtC.size):
                indt = (t >= t[IndtC[ii]]-T2[IndtC[ii]]) & (t <= t[IndtC[ii]]+T2[IndtC[ii]])
                pp = plgUtils.convexHull(plg.Polygon(cc[:,indt].T))
                cctemp[:,IndtC[ii]] = pp.center()
        elif Method=='FAv':
            for ii in range(0,IndtC.size):
                indt = (t >= t[IndtC[ii]]-T2[IndtC[ii]]) & (t <= t[IndtC[ii]]+T2[IndtC[ii]])
                cctemp[:,IndtC[ii]] = np.mean(cc[:,indt],axis=1)
        cc[:] = np.copy(cctemp)
        t1, t2 = t[indtC].min(), t[indtC].max()
    CC[:,indnan] = cc
    return np.sqrt(SS/np.pi), CC



def SpiralRadiusCenter_V2(t, Pts, Freq, Frac=None, Ord=3, ElMet='Perim', Test=True):
    """ Compute the estimated generalized geometric radius from surface spanned by a fraction of a period of a spiral, and compute the estimated center of rotation
    Inputs :
        t       (Nt,) np.ndarray    a 1D array of Nt time points
        Pts     (2,Nt) np.ndarray   trajectory of the point to be tracked, in 2D cartesian coordinates
        Freq    (Nt,) np.ndarray    a 1D array of the instantaneous estimated frequency of rotation
        Frac    float in ]0.5,1.]   fraction of a period to use for the averaging, if None (default) uses the maximum value compatible witrh sampling frequency and <= 1
        Ord     int                 the number of times the algorithm for estimating the center of rotation should be used recursively (optionnal, default=3, recommended >= 3)
        ElMet   str                 flag indicating which method to use for estimating the ellipticity
        Test    bool                if True then all the inputs are tested for conformity
    Outputs :
        r       (Nt,) np.ndarray, a 1D array of the estimated generalized geometric radius, defined as sqrt(S/(2pi)), with S the surface spanned by one period
        C       (2,Nt) np.ndarray, a 2D array represneting the 2D cartesian coordinates, as a functio of time of the estimated center of rotation
        Ellipt  (Nt,) np.ndarray, estimate of the ellipticity of the trajectory

    Method described further in [1]
    [1] D. Vezinet, personal Notes, 'Deriving a radius from spiraling hot core with unknown center, unknown growth rate and non-circular base', May 2015
    """

    if Test:
        assert isinstance(t,np.ndarray) and t.ndim==1,                                  "Arg t must be a (Nt,) np.ndarray !"
        assert isinstance(Pts,np.ndarray) and Pts.ndim==2 and Pts.shape==(2,t.size),    "Arg Pts must be a (2,Nt) np.ndarray !"
        assert isinstance(Freq,np.ndarray) and Freq.ndim==1 and Freq.size==t.size,      "Arg Freq must be a (Nt,) np.ndarray !"
        assert type(Ord) is int and Ord>0,                                              "Arg Ord must be a strictly positive int !"
        assert Frac is None or (type(Frac) is float and Frac>0.5 and Frac<=1.),         "Arg Frac must be None or a float in ]0.5,1.]"

    # Initialising
    Nt = t.size
    SS = np.nan*np.ones((Nt,))
    CC = np.nan*np.ones((2,Nt))
    LL = np.nan*np.ones((Nt,))
    t = np.copy(t)

    indnan = ~(np.any(np.isnan(Pts),axis=0) | np.isnan(Freq))
    t = t[indnan]
    Pts = Pts[:,indnan]
    Freq = Freq[indnan]

    Nt = t.size
    T2 = 0.5/Freq
    ss = np.nan*np.ones((Nt,))
    cc = np.nan*np.ones((2,Nt))
    ll = np.nan*np.ones((Nt,))
    LC = []

    # Getting the good Frac value
    Frac = Get_AlphaFromtFreq(t,Freq, Frac=Frac)
    assert  np.all(Frac>0.5) and np.all(Frac<=1.), "Frac could not be computed in ]0.5,1], fsampling or fmode must be incompatible !"

    # Getting the interval on which the surface can be calculated
    tbis = np.array([t+Frac*T2,t-Frac*T2])
    indtS = (t>=t[np.argmin(np.abs(np.nanmin(t)-tbis[1,:]))]) & (t<=t[np.argmin(np.abs(np.nanmax(t)-tbis[0,:]))])
    IndtS = indtS.nonzero()[0]

    # Computing
    for ii in range(0,IndtS.size):
        indt = (t >= t[IndtS[ii]]-Frac[IndtS[ii]]*T2[IndtS[ii]]) & (t <= t[IndtS[ii]]+Frac[IndtS[ii]]*T2[IndtS[ii]])
        pp = np.concatenate((Pts[:,indt],(Pts[:,indt])[:,0:1]),axis=1)
        try:
            li = np.hypot(pp[0,:-1]-pp[0,1:], pp[1,:-1]-pp[1,1:])
            pp = plgUtils.convexHull(plg.Polygon(Pts[:,indt].T))
            ss[IndtS[ii]] = pp.area()
            cc[:,IndtS[ii]] = pp.center()
            ll[IndtS[ii]] = np.sum(li)
        except:
            pp = list(set([tuple(pppp) for pppp in Pts[:,indt].T]))
            if len(pp)>=3:
                try:
                    pp.append(pp[0])
                    pp = np.array(pp).T
                    li = np.hypot(pp[0,:-1]-pp[0,1:], pp[1,:-1]-pp[1,1:])
                    pp = plgUtils.convexHull(plg.Polygon(pp.T))
                    ss[IndtS[ii]] = pp.area()
                    cc[:,IndtS[ii]] = pp.center()
                    ll[IndtS[ii]] = np.sum(li)
                except:
                    ss[IndtS[ii]] = 0
                    cc[:,IndtS[ii]] = np.mean(np.array(list(set([tuple(pppp) for pppp in Pts[:,indt].T]))).T,axis=1)
            else:
                ss[IndtS[ii]] = 0
                cc[:,IndtS[ii]] = np.mean(np.array(pp).T,axis=1)

    SS[indnan] = ss
    LL[indnan] = ll
    if Ord==1:
        CC[:,indnan] = cc
        return np.sqrt(SS/np.pi), CC

    t1, t2 = t[indtS].min(), t[indtS].max()
    for oo in range(2,Ord+1):
        indtC = (t>=t[np.argmin(np.abs(t1-tbis[1,:]))]) & (t<=t[np.argmin(np.abs(t2-tbis[0,:]))])
        IndtC = indtC.nonzero()[0]
        cctemp = np.nan*np.ones((2,Nt))
        for ii in range(0,IndtC.size):
            indt = (t >= t[IndtC[ii]]-Frac[IndtC[ii]]*T2[IndtC[ii]]) & (t <= t[IndtC[ii]]+Frac[IndtC[ii]]*T2[IndtC[ii]])
            pp = plgUtils.convexHull(plg.Polygon(cc[:,indt].T))
            try:
                cctemp[:,IndtC[ii]] = pp.center()
            except Exception:
                pp = list(set([tuple(cccc) for cccc in cc[:,indt].T]))
                if len(pp)>=3:
                    try:
                        cctemp[:,IndtC[ii]] = plgUtils.convexHull(plg.Polygon(pp)).center()
                    except:
                        cctemp[:,IndtC[ii]] = np.mean(np.array(pp).T,axis=1)
                else:
                    cctemp[:,IndtC[ii]] = np.mean(np.array(pp).T,axis=1)
        cc[:] = np.copy(cctemp)
        t1, t2 = t[indtC].min(), t[indtC].max()
    CC[:,indnan] = cc

    Ellip = LL**2/(4.*np.pi*SS) + np.sqrt((LL**2/(4.*np.pi*SS))**2-1.)

    return np.sqrt(SS/np.pi), CC, Ellip


def Get_AlphaFromtFreq(t,Freq, Frac=None):
    assert Frac is None or (type(Frac) is float and Frac>0.5 and Frac<=1.),         "Arg Frac must be None or a float in ]0.5,1.]"
    fs = 1./np.mean(np.diff(t))
    N = np.floor(0.5*fs/Freq) if Frac is None else np.floor(Frac*0.5*fs/Freq)
    return N*2.*Freq/fs


def SpiralAsym(t, Pts, Freq, Val, Frac=1., Test=True):
    """ Compute the estimated generalized geometric radius from surface spanned by a fraction of a period of a spiral, and compute the estimated center of rotation
    Inputs :
        t       (Nt,) np.ndarray    a 1D array of Nt time points
        Pts     (2,Nt) np.ndarray   trajectory of the point to be tracked, in 2D cartesian coordinates
        Freq    (Nt,) np.ndarray    a 1D array of the instantaneous estimated frequency of rotation
        Frac    float in ]0.5,1.]   fraction of a period to use for the averaging, if None (default) uses the maximum value compatible witrh sampling frequency and <= 1
        Test    bool                if True then all the inputs are tested for conformity
    Outputs :
        r       (Nt,) np.ndarray, a 1D array of the estimated generalized geometric radius, defined as sqrt(S/(2pi)), with S the surface spanned by one period
        C       (2,Nt) np.ndarray, a 2D array represneting the 2D cartesian coordinates, as a functio of time of the estimated center of rotation

    Method described further in [1]
    [1] D. Vezinet, personal Notes, 'Deriving a radius from spiraling hot core with unknown center, unknown growth rate and non-circular base', May 2015
    """
    if Test:
        assert isinstance(t,np.ndarray) and t.ndim==1,                                  "Arg t must be a (Nt,) np.ndarray !"
        assert isinstance(Pts,np.ndarray) and Pts.ndim==2 and Pts.shape==(2,t.size),    "Arg Pts must be a (2,Nt) np.ndarray !"
        assert isinstance(Freq,np.ndarray) and Freq.ndim==1 and Freq.size==t.size,      "Arg Freq must be a (Nt,) np.ndarray !"
        assert Frac is None or (type(Frac) is float and Frac>0.5 and Frac<=1.),         "Arg Frac must be None or a float in ]0.5,1.]"

    # Initialising
    t = np.copy(t)
    Nt = t.size

    indnan = ~(np.any(np.isnan(Pts),axis=0) | np.isnan(Freq))
    t = t[indnan]
    Pts = Pts[:,indnan]
    Freq = Freq[indnan]
    Val = Val[indnan]

    EmissMoy = np.nan*np.ones((Nt,))
    Asym = np.nan*np.ones((Nt,))
    Ellip = np.nan*np.ones((Nt,))
    RMin, RMax = np.nan*np.ones((Nt,)), np.nan*np.ones((Nt,))

    Nt = t.size
    T2 = 0.5/Freq
    emissmoy = np.nan*np.ones((Nt,))
    asym = np.nan*np.ones((Nt,))
    ellip = np.nan*np.ones((Nt,))
    Rmin, Rmax = np.nan*np.ones((Nt,)), np.nan*np.ones((Nt,))

    # Getting the good Frac value
    Frac = Get_AlphaFromtFreq(t,Freq, Frac=Frac)
    assert  np.all(Frac>0.5) and np.all(Frac<=1.), "Frac could not be computed in ]0.5,1], fsampling or fmode must be incompatible !"

    # Getting the interval on which the surface can be calculated
    tbis = np.array([t+Frac*T2,t-Frac*T2])
    indtS = (t>=t[np.argmin(np.abs(np.nanmin(t)-tbis[1,:]))]) & (t<=t[np.argmin(np.abs(np.nanmax(t)-tbis[0,:]))])
    IndtS = indtS.nonzero()[0]

    # Computing
    for ii in range(0,IndtS.size):
        indt = (t >= t[IndtS[ii]]-Frac[IndtS[ii]]*T2[IndtS[ii]]) & (t <= t[IndtS[ii]]+Frac[IndtS[ii]]*T2[IndtS[ii]])
        emissmoy[IndtS[ii]] = np.mean(Val[indt])
        indsortR = np.argsort(Pts[0,indt])
        indsortZ = np.argsort(Pts[1,indt])
        vals = Val[indt][indsortR]
        asym[IndtS[ii]] = (np.mean(vals[-3:]) - np.mean(vals[:3]))/emissmoy[IndtS[ii]]
        Rmin[IndtS[ii]] = np.mean(Pts[0,indt][indsortR][:3])
        Rmax[IndtS[ii]] = np.mean(Pts[0,indt][indsortR][-3:])
        Zmin = np.mean(Pts[1,indt][indsortZ][:3])
        Zmax = np.mean(Pts[1,indt][indsortZ][-3:])
        ellip[IndtS[ii]] = (Zmax-Zmin)/(Rmax[IndtS[ii]]-Rmin[IndtS[ii]])


    EmissMoy[indnan] = emissmoy
    Asym[indnan] = asym
    Ellip[indnan] = ellip
    RMin[indnan], RMax[indnan] = Rmin, Rmax

    return EmissMoy, Asym, Ellip, RMin, RMax



def FindrbinEdges(t, r, Freq, LDt, Drmin=0.005, NTmin=1., Test=True):

    for ii in range(0,len(LDt)):
        indt = (t>=LDt[ii][0]) & (t<=LDt[ii][1])
        t0 = (LDt[ii][1]+LDt[ii][0])/2.
        indt0 = np.nanargmin(np.abs(t-t0))
        indt1, indt2 = np.nanargmin(np.abs(t-LDt[ii][0])), np.nanargmin(np.abs(t-LDt[ii][1]))
        dt = LDt[ii][1]-LDt[ii][0]
        r1, r2 = r[indt1], r[indt2]
        dr = np.abs(r1-r2)
        if dt < NTmin/Freq[indt0] or dr<Drmin:
            N = 1
        else:
            N = int(dt*Freq[indt0]/NTmin)
        rE = np.linspace(r1,r2,N+1)
        tE = np.interp(rE, r[indt], t[indt])
        rC = np.mean(np.array([Dr[:-1],Dr[1:]]),axis=0)
        LrE.append(Dr)
        LrC.append(rC)
        LrLm.append(rC-rE[:-1])
        LrRm.append(rE[1:]-rC)
    return LrE, LrC, LrLm, LrRm




