

import os
import sys


import numpy as np
import matplotlib.pyplot as plt
import tofu as tf


# #####################################################
# #####################################################
#               DEFAULTS
# #####################################################


_PATH_HERE = os.path.dirname(__file__)


_PFE_DCROSS = os.path.join(
    _PATH_HERE,
    'd2cross_Ee01eV-100MeV-80log_Eph1eV-100MeV-81log_ntheta61_EH.npz',
)


# #####################################################
# #####################################################
#       Fig 1 - cross-section
# #####################################################


def fig01_cross_section(
    pfe=None,
    Eph_eV=[5e3, 50e3, 500e3],
    Ee0_eV=[1e5, 1e6],
):

    # --------------
    # load
    # --------------

    dout = {
        kk: vv.tolist()
        for kk, vv in dict(np.load(pfe, allow_pickle=True)).items()
    }

    # --------------
    # extract
    # --------------

    indEph = [
        np.argmin(np.abs(dout['E_ph']['data'] - eph))
        for eph in Eph_eV
    ]

    indEe0 = [
        np.argmin(np.abs(dout['E_e0']['data'] - e0))
        for e0 in Ee0_eV
    ][0]

    Ee0 = dout['E_e0']['data'][0, indEe0, 0]
    Eph = dout['E_ph']['data'][0, 0, indEph]
    theta_ph = dout['theta_ph']['data'].squeeze()*180/np.pi

    cross = dout['cross']['EH']['data'][:, indEe0, indEph]

    units = dout['cross']['EH']['units']

    # --------------
    # prepare plot
    # --------------

    fig = plt.figure(figsize=(13, 7))
    gs = None
    dax = {}

    ax0 = fig.add_subplot(121)
    ax0.set_xlabel(
        "Angle of photon emission" + r"$\theta_{ph}$" + " (deg)",
        fontweight='bold',
        fontsize=14,
    )
    ax0.set_ylabel(
        f'Cross-section {units}',
        fontweight='bold',
        fontsize=14,
    )

    ax1 = fig.add_subplot(122, sharex=ax0)
    ax1.set_xlabel(
        "Angle of photon emission" + r"$\theta_{ph}$" + " (deg)",
        fontweight='bold',
        fontsize=14,
    )
    ax1.set_ylabel(
        f'Cross-section {units}',
        fontweight='bold',
        fontsize=14,
    )

    # --------------
    # plot
    # --------------

    for ii, eph in enumerate(Eph):

        l0, = ax0.plot(
            theta_ph,
            cross[:, ii],
            label=f"Eph = {eph*1e-3:3.0f} keV",
        )

        ax1.plot(
            theta_ph,
            cross[:, ii] / np.max(cross[:, ii]),
            color=l0.get_color(),
            label=f"Eph = {eph*1e-3:03.0f} keV",
        )

    # --------------
    # adjust
    # --------------

    ax0.set_xlim(0, 180)
    ax0.set_ylim(bottom=0)
    ax1.set_ylim(0, 1)
    ax0.legend()

    return dax
