

import os
import numpy as np
import matplotlib.pyplot as plt

pwd = os.getcwd()
os.chdir('/afs/ipp-garching.mpg.de/home/d/didiv/Python/tofu')
import tofu.mesh._bsplines_cy as _tfm_bs
os.chdir(pwd)






def Fig01(Deriv=0, a4=False, figNb=1, save=True):

    # Computing
    Deg = 2
    ND = 1
    knt = np.linspace(0.,7.,8)
    NK = len(knt)
    NF = NK-1-Deg
    x = np.linspace(knt[0],knt[-1],500)

    LFunc = _tfm_bs.BSpline_LFunc(Deg, knt, Deriv=0, Test=True)[0]

    yy = np.vstack([LFunc[jj](x) for jj in range(0,len(LFunc))])

    # Plotting
    fdpi,axCol = 80,'w'
    (fW,fH) = (11.69,8.27) if a4 else (6,4)
    f, axarr = plt.subplots(nrows=1, ncols=1, facecolor='w',figsize=(fW,fH), subplot_kw={'adjustable':'datalim','frameon':True,'axisbg':axCol})
    plt.subplots_adjust(left=0.08, bottom=0.07, right=0.96, top=0.98, wspace=0.05, hspace=0.05)

    axarr.plot(x, yy.T, ls='-', lw=1.)
    axarr.plot(x, np.nansum(yy,axis=0), ls='-', lw=2., c='k')

    axarr.set_xlim(knt[0],knt[-1])
    axarr.set_ylim(0.,1.05)
    #axarr.set_title(r"D{0}".format(Deg), size=12)
    axarr.set_xticks(knt)
    axarr.set_yticks([0.,0.2,0.4,0.6,0.8,1.])
    labx = [r"$x_{"+str(kk)+r"}$" for kk in range(0,NK)]
    axarr.set_xticklabels(labx, size=14)

    if save:
        f.savefig('/afs/ipp-garching.mpg.de/home/d/didiv/Documents/Notes/Bsplines_GaussianQuadrature/Fig{0:02.0f}_BSplines_Int_D{1:01.0f}.pdf'.format(figNb,Deg), format='pdf')
    return axarr
