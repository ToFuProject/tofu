""" Analytical formulation of a 2d shielding cylinder


"""


import os


import numpy as np
import matplotlib.pyplot as plt


# ################################################################
# ################################################################
#                DEFAULTS
# ################################################################


_PATH_HERE = os.path.dirname(__file__)

_DMUR = {
    'SS1008': 2000,
    'mumetal': 1,   # 470000,
}
# _B0T = 0.015
# _B0A = 0.003
_B0T = 0.011
_B0A = 0.002


_DGEOM = {
    # 'a': 66e-3,
    'a': 69e-3,
    'a_tol': 0.5e-3,
    # 'b': 139.725e-3,
    'b': 140e-3,
    'b_tol': 0.1e-3,
    'thick': 1.5e-3,
    # length
    # 'len0': 250e-3,           # S1008
    'len0': 230e-3,
    'len1': (240-24.5)*1e-3,  # mumetal
}


# ################################################################
# ################################################################
#       Scan inner diameter of single shield
# ################################################################


def main(
    b=_DGEOM['b'],
    b_tol=_DGEOM['b_tol'],
    a0=_DGEOM['a'],
    a0_tol=_DGEOM['a_tol'],
    B0=_B0T,
    mur0='SS1008',
):

    # ---------------
    # Initialize values
    # ---------------

    nmur = 51
    mur = np.linspace(500, 8000, nmur)

    na = 100
    a = np.linspace(0.06, 0.13, na)

    # ---------------
    # Compute
    # ---------------

    shield = shielding_trans(a=a[:, None], b=b, mur=mur[None, :])
    Bin = B0 * shield

    # ---------------
    # prepare figure
    # ---------------

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_axes([.06, 0.08, 0.6, 0.80])

    # title
    ax.set_title(
        "B field in a 2d infinite shielding cylinder\n"
        f"Outer diameter b = {b} m\n"
        f"External field B0 = {B0} T  ({int(B0*1e4)} gauss)",
        size=14,
        fontweight='bold',
    )

    # xlabel
    ax.set_xlabel(
        'inner diameter a (m)',
        size=12,
        fontweight='bold',
    )

    # ylabel
    ax.set_ylabel(
        "Relative permeability  " + r"$\mu_r$",
        size=12,
        fontweight='bold',
    )

    # ---------------
    # contours
    # ---------------

    # contourf plot
    ax.contourf(
        a,
        mur,
        Bin.T*1e4,
        [0, 0.5, np.max(Bin*1e4)],
        cmap='RdYlGn_r',
    )

    # contour plot
    cs = ax.contour(a, mur, Bin.T*1e4, [0.1, 0.2, 0.5, 1, 10], ec='k')

    # labelled contours
    ax.clabel(
        cs,
        cs.levels,
        inline=True,
        fontsize=14,
        fmt=lambda x: f'{x} gauss',
    )

    # ---------------
    # add mur0
    # ---------------

    ax.axhline(
        _DMUR[mur0],
        ls='--',
        c='k',
        lw=1,
    )

    ax.text(
        a[-1], _DMUR[mur0], mur0,
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transData,
        size=12,
        fontweight='bold',
    )

    # ------------------
    # add inner diameter
    # ------------------

    ax.axvspan(
        a0 - a0_tol,
        a0 + a0_tol,
        fc=(0.8, 0.8, 0.8, 0.5),
    )

    # ---------------
    # add expression
    # ---------------

    exp = (
        r"$\frac{B_{in}}{B_0} = $"
        + r"$\frac{4\mu_r}{(1+\mu_r)^2 - \left(\frac{a}{b}\right)^2(1-\mu_r)^2}$"
    )

    ax.text(
        1.28, 0.3, exp,
        horizontalalignment='center',
        transform=ax.transAxes,
        size=24,
    )

    # shield value
    imur = np.argmin(np.abs(mur - _DMUR[mur0]))
    ia = np.argmin(np.abs(a - a0))

    ax.plot(
        [a[ia]],
        [mur[imur]],
        marker='s',
        ms=12,
        c='r'
    )

    exp = (
        r"$\frac{B_{in}}{B_0} = $"
        + f"{shield[ia, imur]:3.2e}"
    )

    ax.text(
        1.28, 0.05, exp,
        horizontalalignment='center',
        transform=ax.transAxes,
        size=24,
    )

    # -----------------
    # add image
    # -----------------

    pfe = os.path.join(_PATH_HERE, '2dproblem.png')
    image = plt.imread(pfe)
    axim = fig.add_axes([0.72, 0.4, 0.25, 0.55])
    axim.imshow(image)
    axim.axis('off')

    return


# ################################################################
# ################################################################
#       shielding formula
# ################################################################


def shielding_trans(a=None, b=None, mur=None):

    num = 4 * mur
    denom0 = (1 + mur)**2
    denom1 = (a/b)**2 * (1 - mur)**2

    return num / (denom0 - denom1)


def shielding_axial(p=None):

    p2m1 = p**2 - 1
    sqp2m1 = np.sqrt(p2m1)

    return (1./p2m1) * ((p / sqp2m1) * np.log(p + sqp2m1) - 1)


# ################################################################
# ################################################################
#           __main__
# ################################################################


if __name__ == '__main__':
    main()
