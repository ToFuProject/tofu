

import os


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


import tofu as tf
tfphysemis = tf.physics_tools.electrons.emission


# #####################################################
# #####################################################
#               DEFAULTS
# #####################################################


_PATH_HERE = os.path.dirname(__file__)


_DPFE_DCROSS = {
    'EH0': os.path.join(
        _PATH_HERE,
        'd2cross_Ee01eV-100MeV-80log_Eph1eV-100MeV-81log_ntheta61_EH.npz',
    ),
    'EH1': os.path.join(
        _PATH_HERE,
        'd2cross_Ee0100eV-10MeV-80log_Eph100eV-10MeV-81log_ntheta61_EH.npz',
    ),
    'BHE': os.path.join(
        _PATH_HERE,
        'd2cross_Ee01eV-100MeV-400log_Eph1eV-100MeV-401log_ntheta181_BHE.npz',
    ),
}


# #####################################################
# #####################################################
#       Fig 1 - cross-section
# #####################################################


def fig01_cross_section(
    figsize=(15, 7),
    pfe_cross='EH0',
    version='EH',
    Eph_eV=np.r_[1e3, 10e3, 500e3],
    Ee0_eV=np.r_[20e3, 1e6],
    fontsize=14,
    pfe_save=None,
):

    # --------------
    # load
    # --------------

    pfe = _DPFE_DCROSS[pfe_cross]

    dout = {
        kk: vv.tolist()
        for kk, vv in dict(np.load(pfe, allow_pickle=True)).items()
    }
    units = dout['cross'][version]['units']
    Z = dout.get('Z', {'data': 1})['data']

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.06, 'right': 0.99,
        'bottom': 0.08, 'top': 0.93,
        'wspace': 0.25, 'hspace': 0.10,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=4, nrows=2, **dmargin)
    dax = {}

    # --------------
    # prepare axes
    # --------------

    # --------------
    # ax - isolines

    ax = fig.add_subplot(
        gs[:, -2:],
        xscale='log',
        yscale='log',
        aspect='equal',
    )
    ax.set_xlabel(
        r"$E_{e,0}$ (keV)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$E_{ph}$ (keV)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        r"$d^2\sigma(E_{e0}, E_{ph}, \theta_{ph}, Z)$"
        + f"\n Z = {Z} - version = {version}",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['map'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # ax - theta_norm

    # theta_norm0
    ax = fig.add_subplot(
        gs[0, 0],
        xscale='linear',
    )
    ax.set_ylabel(
        "normalized cross-section (adim.)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        r"$E_{e0}$" + f" = {Ee0_eV[0]*1e-3:2.0f} keV",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['theta_norm0'] = {'handle': ax, 'type': 'isolines'}

    # theta_norm1
    ax = fig.add_subplot(
        gs[0, 1],
        sharex=dax['theta_norm0']['handle'],
        sharey=dax['theta_norm0']['handle'],
    )
    ax.set_title(
        r"$E_{e0}$" + f" = {Ee0_eV[1]*1e-6:2.0f} MeV",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['theta_norm1'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # ax - theta_abs

    # theta_abs0
    ax = fig.add_subplot(
        gs[1, 0],
        sharex=dax['theta_norm0']['handle'],
    )
    ax.set_xlabel(
        r"$\theta_{ph}$ (deg)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$d\sigma$" + f"({units})",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['theta_abs0'] = {'handle': ax, 'type': 'isolines'}

    # theta_abs1
    ax = fig.add_subplot(
        gs[1, 1],
        sharex=dax['theta_norm0']['handle'],
        sharey=dax['theta_abs0']['handle'],
    )
    ax.set_xlabel(
        r"$\theta_{ph}$ (deg)",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['theta_abs1'] = {'handle': ax, 'type': 'isolines'}

    # ------------------
    # call built-in
    # ------------------

    # cases 100 keV
    lc = ['r', 'g', 'b']
    for i0, e0 in enumerate(Ee0_eV):
        iphok = Eph_eV < e0
        dcases = {
            i1: {
                'E_e0_eV': e0,
                'E_ph_eV': eph,
                'color': lc[i1],
                'label': f"Eph = {eph*1e-3:3.0f} keV",
            }
            for i1, eph in enumerate(Eph_eV[iphok])
        }
        _dax, _d2cross = tfphysemis.plot_xray_thin_d2cross_ei_anisotropy(
            d2cross=pfe,
            dcases=dcases,
            dax={
                'map': dax['map']['handle'],
                'theta_norm': dax[f'theta_norm{i0}']['handle'],
                'theta_abs': dax[f'theta_abs{i0}']['handle'],
            },
        )

        # remove contour plots
        if i0 == 0:
            for cc in dax['map']['handle'].get_children():
                if cc.__class__.__name__ == 'QuadContourSet':
                    cc.remove()

            # remove legend
            dax['theta_norm0']['handle'].get_legend().remove()

    # ---------------------
    # Adjust map x/y scales
    # ---------------------

    dax['map']['handle'].set_xlim(0.1, 10e3)
    dax['map']['handle'].set_ylim(0.1, 10e3)
    dax['theta_abs1']['handle'].set_ylabel('')
    dax['theta_norm1']['handle'].legend(loc='lower right')

    # --------------
    # add a, b, c, d, e
    # --------------

    dabc = {
        'theta_norm0': '(a)',
        'theta_norm1': '(c)',
        'theta_abs0': '(b)',
        'theta_abs1': '(d)',
    }
    for kax, abc in dabc.items():
        dax[kax]['handle'].grid(visible=True, which='major', axis='both')
        dax[kax]['handle'].text(
            0.95, 0.95,
            abc,
            fontsize=fontsize,
            fontweight='bold',
            horizontalalignment='right',
            verticalalignment='top',
            transform=dax[kax]['handle'].transAxes,
        )

    dax['map']['handle'].text(
        0., 1.02,
        "(e)",
        fontsize=fontsize,
        fontweight='bold',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=dax['map']['handle'].transAxes,
    )

    # --------------
    # save
    # --------------

    if pfe_save is not False:
        if pfe_save is None:
            name = 'fig01_crosssection.png'
            pfe_save = os.path.join(_PATH_HERE, name)
        fig.savefig(pfe_save, format='png', dpi=300)
        msg = f"Saved figure in:\n\t{pfe_save}\n"
        print(msg)

    return dax


# #####################################################
# #####################################################
#       Fig 1 - cross-section
# #####################################################


_DR = {
    'R0': 1.8,
    'rplasma': 0.6,
    'RFW': [1.2, 2.45],
    'Rcryo': 4.6,
    'PP_length': 2.2,
    'PP_width': None,
    'PP_phi': np.r_[-30, 30] * np.pi/180,
}


def fig02_tokamak(
    # tokamak
    R0=None,
    rplasma=None,
    RFW=None,
    Rcryo=None,
    PP_length=None,
    PP_width=None,
    PP_phi=None,
    # plot
    figsize=(15, 7),
    fontsize=14,
    pfe_save=None,
):

    # --------------
    # Load SPARC
    # --------------

    config = tf.load_config('SPARC')

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.06, 'right': 0.99,
        'bottom': 0.08, 'top': 0.93,
        'wspace': 0.25, 'hspace': 0.10,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=1, nrows=1, **dmargin)
    dax = {}

    # --------------
    # prepare axes
    # --------------

    ax = fig.add_subplot(gs[0, 0], aspect='equal')
    ax.set_xlabel('X (m)', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=fontsize, fontweight='bold')
    ax.set_title('SPARC-like tokamak', fontsize=fontsize, fontweight='bold')

    dax['hor'] = ax

    dax = ds._generic_check._check_dax(dax)

    # --------------
    # plot tokamak
    # --------------

    config.plot(lax=dax['hor']['handle'], proj='hor')

    # --------------
    # add port plug
    # --------------

    # --------------
    # save
    # --------------

    if pfe_save is not False:
        if pfe_save is None:
            name = 'fig02_tokamak.png'
            pfe_save = os.path.join(_PATH_HERE, name)
        fig.savefig(pfe_save, format='png', dpi=300)
        msg = f"Saved figure in:\n\t{pfe_save}\n"
        print(msg)

    return dax

