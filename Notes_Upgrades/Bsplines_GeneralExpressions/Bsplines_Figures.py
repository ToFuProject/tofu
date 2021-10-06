

import os
import numpy as np
import matplotlib.pyplot as plt


pwd = os.getcwd()
os.chdir('/afs/ipp-garching.mpg.de/home/d/didiv/Python/tofu')
import tofu.mesh._bsplines_cy as _tfm_bs
os.chdir(pwd)


def Fig01(Deriv=0, a4=False, save=True):

    # Computing
    Deg = [0, 1, 2, 3]
    knt = np.linspace(0., 10., 11)
    x = np.linspace(knt[0], knt[-1], 500)

    LF = {}
    for jj in range(0, len(Deg)):
        LFunc = _tfm_bs.BSpline_LFunc(Deg[jj], knt, Deriv=Deriv, Test=True)[0]
        LF[Deg[jj]] = LFunc

    # Plotting
    fdpi, axCol = 80, 'w'
    (fW, fH) = (11.69, 8.27) if a4 else (8, 6)
    f = plt.figure(facecolor="w", figsize=(fW, fH), dpi=fdpi)
    ax0 = f.add_axes([0.06, 0.53, 0.43, 0.42], frameon=True, axisbg=axCol)
    ax1 = f.add_axes([0.54, 0.53, 0.43, 0.42], frameon=True, axisbg=axCol)
    ax2 = f.add_axes([0.06, 0.06, 0.43, 0.42], frameon=True, axisbg=axCol)
    ax3 = f.add_axes([0.54, 0.06, 0.43, 0.42], frameon=True, axisbg=axCol)

    La = [ax0, ax1, ax2, ax3]
    for ii in range(0, len(Deg)):
        for jj in range(0, len(LF[Deg[ii]])):
            La[ii].plot(x, LF[Deg[ii]][jj](x), ls='-', lw=2.)

        La[ii].set_xlim(3., 7.)
        La[ii].set_ylim(0., 1.05)
        La[ii].set_title(r"D{0}".format(Deg[ii]), size=12)
        La[ii].set_xticks([3., 4., 5., 6., 7.])
        La[ii].set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1.])
        if ii in [2, 3]:
            La[ii].set_xticklabels(
                [r"$x_{0}$", r"$x_{1}$", r"$x_{2}$", r"$x_{3}$", r"$x_{4}$"],
                size=15,
            )
        else:
            La[ii].set_xticklabels([])
        if ii in [1, 3]:
            La[ii].set_yticklabels([])

    if save:
        path = os.path.dirname(__file__)
        pfe = os.path.join(
            path,
            f'BSplines_GeneralExpression_Deriv{Deriv}.pdf',
        )
        f.savefig(pfe, format='pdf')
    return La
