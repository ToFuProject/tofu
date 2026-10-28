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
#       Compute double shield
# ################################################################


def main(
    dmur=_DMUR,
    dgeom=_DGEOM,
    B0T=_B0T,
    B0A=_B0A,
):

    # --------------
    # compute
    # --------------

    dshield = {
        'SS1008': shielding_trans(
            a=dgeom['a'] + dgeom['a_tol'],
            b=dgeom['b'] - _DGEOM['b_tol'],
            mur=dmur['SS1008'],
        ),
        'mumetal': shielding_trans(
            a=dgeom['a'] - dgeom['a_tol'] - dgeom['thick'],
            b=dgeom['a'] - dgeom['a_tol'],
            mur=dmur['mumetal'],
        ),
    }

    shield = dshield['mumetal'] * dshield['SS1008']
    ST = int(1/shield)

    # --------------
    # axial component
    # --------------

    p_S1008 = dgeom['len0'] / dgeom['b']
    p_mumetal = dgeom['len1'] / dgeom['a']

    N_S1008 = shielding_axial(p=p_S1008)
    N_mumetal = shielding_axial(p=p_mumetal)

    SA_S1008 = 4*N_S1008/dshield['SS1008'] + 1
    SA_mumetal = 4*N_mumetal/dshield['mumetal'] + 1
    SA = SA_S1008 * SA_mumetal

    # ---------------
    # prepare figure
    # ---------------

    fig = plt.figure(figsize=(15, 8))

    # transverse
    axT = fig.add_axes(
        [.06, 0.50, 0.4, 0.40],
        aspect='equal',
        adjustable='datalim',
    )

    # axial
    axA = fig.add_axes(
        [.56, 0.50, 0.4, 0.40],
        adjustable='datalim',
    )

    # axial - shapes
    axS = fig.add_axes(
        [.70, 0.60, 0.25, 0.25],
        aspect='equal',
        adjustable='datalim',
    )

    # transverse title
    ref0 = "[1] A. Mager, IEEE Transactions on Magnetics, vol. 6, 1970"
    axT.set_title(
        "B field in a 2d infinite shielding cylinder\n"
        f"External transverse field B0 = {B0T} T  ({int(B0T*1e4)} gauss)\n"
        + ref0,
        size=14,
        fontweight='bold',
    )

    ref1 = "[2] A. Mager, Journal of Applied Physics, vol. 39, 1968"
    axA.set_title(
        "B field in a 2d finite shielding cylinder\n"
        f"External axial field B0 = {B0A} T  ({int(B0A*1e4)} gauss)\n"
        + ref1,
        size=14,
        fontweight='bold',
    )

    # xlabel
    axT.set_xlabel(
        'x (m)',
        size=12,
        fontweight='bold',
    )

    # ylabel
    axT.set_ylabel(
        "y (m)",
        size=12,
        fontweight='bold',
    )

    # xlabel
    axA.set_xlabel(
        'L/D',
        size=12,
        fontweight='bold',
    )

    # ylabel
    axA.set_ylabel(
        "N",
        size=12,
        fontweight='bold',
    )

    axA.set_ylim(0, 1)

    # xlabel
    axS.set_xlabel(
        'l (m)',
        size=12,
        fontweight='bold',
    )

    # ylabel
    axS.set_ylabel(
        "y (m)",
        size=12,
        fontweight='bold',
    )

    # ---------------
    # add circles
    # ---------------

    theta = np.pi * np.linspace(-1, 1, 101)

    # SS1008
    ss_x_out = 0.5 * dgeom['b'] * np.cos(theta)
    ss_y_out = 0.5 * dgeom['b'] * np.sin(theta)
    ss_x_in = 0.5 * (dgeom['a'] + dgeom['a_tol']) * np.cos(theta)
    ss_y_in = 0.5 * (dgeom['a'] + dgeom['a_tol']) * np.sin(theta)
    ss_x = np.r_[ss_x_out, ss_x_in[::-1], ss_x_out[0]]
    ss_y = np.r_[ss_y_out, ss_y_in[::-1], ss_y_out[0]]

    axT.fill(
        ss_x,
        ss_y,
        fc=(0.8, 0.8, 0.8, 0.5),
    )

    # mumetal
    mm_x_out = 0.5 * (dgeom['a'] - dgeom['a_tol']) * np.cos(theta)
    mm_y_out = 0.5 * (dgeom['a'] - dgeom['a_tol']) * np.sin(theta)
    mm_x_in = 0.5 * (dgeom['a'] - dgeom['a_tol'] - dgeom['thick']) * np.cos(theta)
    mm_y_in = 0.5 * (dgeom['a'] - dgeom['a_tol'] - dgeom['thick']) * np.sin(theta)
    mm_x = np.r_[mm_x_out, mm_x_in[::-1], mm_x_out[0]]
    mm_y = np.r_[mm_y_out, mm_y_in[::-1], mm_y_out[0]]

    axT.fill(
        mm_x,
        mm_y,
        fc=(0.8, 0., 0.8, 0.5),
    )

    # ---------------
    # add shapes - axial
    # ---------------

    axS.fill(
        dgeom['len0'] * np.r_[0, 1, 1, 0],
        dgeom['b'] * np.r_[0, 0, 1, 1],
        fc=(0.8, 0.8, 0.8),
    )
    axS.fill(
        dgeom['len1'] * np.r_[0, 1, 1, 0] + 0.5*(dgeom['len0'] - dgeom['len1']),
        dgeom['a'] * np.r_[0, 0, 1, 1] + 0.5*(dgeom['b'] - dgeom['a']),
        fc=(0.8, 0., 0.8),
    )

    # ---------------
    # add expression - transverse
    # ---------------

    exp0 = (
        r"$S_T = \frac{B_0}{B_{in}} = $"
        + r"$\frac{(1+\mu_r)^2 - \left(\frac{a}{b}\right)^2(1-\mu_r)^2}{4\mu_r}$"
        + "\n   "
    )

    exp1 = (
        r"$ = \frac{B_0}{B_{inter}}\frac{B_{inter}}{B_{in}}$"
        + "\n   "
        + f" = {int(1/dshield['mumetal'])} " + r"$\times$" + f" {int(1/dshield['SS1008'])}"
        + "\n   "
        + f" = {ST}"
    )

    axT.text(
        -0.1, -0.2, exp0 + exp1,
        horizontalalignment='left',
        verticalalignment='top',
        transform=axT.transAxes,
        size=18,
    )

    exp2 = (
        r"$B_{in} = \frac{B_0}{" + f"{ST}" + r"}$"
        + r"$ = \frac{"
        + f"{B0T*dshield['SS1008']*1e4:3.2} gauss"
        + r"}{" + f"{int(1/dshield['mumetal'])}" + r"}$"
        + "\n" + r"$B_{in} = " + f"{B0T/ST * 1e4:3.2e}" + r"\text{  gauss}$"
    )

    axT.text(
        -0.1, -0.8, exp2,
        horizontalalignment='left',
        verticalalignment='top',
        transform=axT.transAxes,
        size=18,
    )

    # ---------------
    # add constants
    # ---------------

    const = (
        "S1008\n"
        f"    length = {dgeom['len0'] * 1e3} mm\n"
        f"    outer diam = {dgeom['b']*1e3} +/- {dgeom['b_tol']*1e3} mm\n"
        f"    inner diam = {dgeom['a']*1e3} +/- {dgeom['a_tol']*1e3} mm\n"
        f"    mur = {dmur['SS1008']}\n\n"
        "mumetal\n"
        f"    length = {dgeom['len1'] * 1e3} mm\n"
        f"    outer diam = {dgeom['a']*1e3} +/- {dgeom['a_tol']*1e3} mm\n"
        f"    thickness = {dgeom['thick']*1e3} mm\n"
        f"    mur = {dmur['mumetal']}\n"
    )

    axT.text(
        0.6, -0.2, const,
        horizontalalignment='left',
        verticalalignment='top',
        transform=axT.transAxes,
        size=14,
    )

    # ---------------
    # add plot - axial
    # ---------------

    pp = np.linspace(1.000001, 15, 101)
    NN = shielding_axial(pp)

    axA.plot(
        pp,
        NN,
        ls='-',
        c='k',
    )

    axA.axvline(p_S1008, c='k', ls='--', lw=1)
    axA.axvline(p_mumetal, c='k', ls='--', lw=1)

    axA.text(
        p_S1008, 1, 'S1008',
        horizontalalignment='left',
        verticalalignment='top',
        transform=axA.transData,
        rotation=90,
        size=14,
    )

    axA.text(
        p_mumetal, 1, 'mumetal',
        horizontalalignment='left',
        verticalalignment='top',
        transform=axA.transData,
        rotation=90,
        size=14,
    )

    # ---------------
    # add expression - axial
    # ---------------

    exp0 = (
        r"$S_A = 4NS_T + 1$"
        + "\n"
        r"$p = \frac{L}{D}$"
        + "\n"
        + r"$N = \frac{1}{p^2-1}\left(\frac{p}{\sqrt{p^2-1}}\log\left(p + \sqrt{p^2-1}\right)-1\right)$"
        + "\n   "
    )

    axA.text(
        0.1, -0.2, exp0,
        horizontalalignment='left',
        verticalalignment='top',
        transform=axA.transAxes,
        size=18,
    )

    exp1 = (
        r"$S_A = $" + f"{int(SA_S1008)} x {int(SA_mumetal)} = {int(SA_S1008*SA_mumetal)}"
        + "\n\n"
        + r"$B_{in} = $" + f"{B0A / SA * 1e4: 3.2e} gauss"
    )

    axA.text(
        0.1, -0.8, exp1,
        horizontalalignment='left',
        verticalalignment='top',
        transform=axA.transAxes,
        size=18,
    )

    return dshield


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
    dshield = main()
