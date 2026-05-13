

import os


import numpy as np
import scipy.integrate as scpinteg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import datastock as ds


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
#       Fig 02 - RE dist
# #####################################################


_DDIST = {
    # maxwell
    'Te_eV': np.r_[1e3, 1e3, 3e3, 3e3],
    'ne_m3': 1e20,
    'jp_Am2': 1e6,
    # RE
    'jp_fraction_re': np.r_[0.2, 0.8, 0.2, 0.8],
    'dominant': 'bump',
    'Ekin_max_eV': 20e6,
    'E_eV': np.logspace(0, 8, 80),
    'theta': np.linspace(0, 180, 181) * np.pi / 180,
}


def fig02_distributions(
    E_eV=None,
    theta=None,
    ne_m3=None,
    jp_Am2=None,
    jp_fraction_re=None,
    Ekin_max_eV=None,
    # plot
    figsize=(5, 7),
    fontsize=12,
    pfe_save=None,
):

    # ------------
    # inputs
    # ------------

    din = locals()
    din = {
        kk: vv if din.get(kk) is None else din[kk]
        for kk, vv in _DDIST.items()
    }

    # ------------
    # compute
    # ------------

    # dout = {'dist': dict, 'plasma': dist, 'coords': dist}
    dout = tf.physics.electrons.distribution.get_distribution(**din)

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.11, 'right': 0.97,
        'bottom': 0.06, 'top': 0.99,
        'wspace': 0.25, 'hspace': 0.20,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=1, nrows=2, **dmargin)
    dax = {}

    # --------------
    # axes - 2d
    # --------------

    ax = fig.add_subplot(gs[0, 0], aspect='auto')
    ax.set_xlabel('E (keV)', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel(
        r'$\theta_{e0}$ (deg)',
        fontsize=fontsize,
        fontweight='bold',
    )
    ax.text(
        0.01,
        0.99,
        '(a)',
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=fontsize,
        fontweight='bold',
        transform=ax.transAxes,
    )

    dax['2d'] = ax

    # --------------
    # axes - 1d
    # --------------

    ax = fig.add_subplot(gs[1, 0], aspect='auto', sharex=ax)
    ax.set_xlabel('E (keV)', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel(
        '',
        fontsize=fontsize,
        fontweight='bold',
    )
    ax.text(
        0.01,
        0.99,
        '(b)',
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=fontsize,
        fontweight='bold',
        transform=ax.transAxes,
    )

    dax['1d'] = ax

    # ------------
    # plot 1d
    # ------------

    kax = '1d'
    dcolor = {}
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # prepare
        dataRE = scpinteg.trapezoid(
            dout['dist']['RE']['dist']['data'],
            x=dout['coords']['x1']['data'],
            axis=-1,
        )
        dataMax = scpinteg.trapezoid(
            dout['dist']['maxwell']['dist']['data'],
            x=dout['coords']['x1']['data'],
            axis=-1,
        )

        # loop plot
        for ind in np.ndproduct(dataRE.shape[:-2]):
            sli = ind + (slice(None),)

            # Max
            l0, = ax.plot(
                dout['coords']['x0']['data']*1e-3,
                dataMax[sli],
                ls='-',
                lw=1,
                color=dcolor[ind],
            )
            dcolor[ind] = l0.get_color()

            # RE
            ax.plot(
                dout['coords']['x0']['data']*1e-3,
                dataRE[sli],
                ls='--',
                lw=1,
                color=dcolor[ind],
            )

            # Total
            ax.plot(
                dout['coords']['x0']['data']*1e-3,
                dataMax[sli] + dataRE[sli],
                ls='-',
                lw=2,
                color=dcolor[ind],
                label=str(ind),
            )

        ax.legend()

    # ------------
    # plot 2d
    # ------------

    kax = '2d'
    dcolor = {}
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # prepare
        dataRE = dout['dist']['RE']['dist']['data']
        dataMax = dout['dist']['maxwell']['dist']['data']

        # loop plot
        for ind in np.ndproduct(dataRE.shape[:-2]):
            sli = ind + (slice(None), slice(None))

            cc = ax.contour(
                dout['coords']['x0']['data'],
                dout['coords']['x1']['data'],
                dataRE[sli] + dataMax[sli],
                20,
                ls='-',
                vmin=0,
                vmax=None,
                label=str(ind),
                color=dcolor[ind],
            )

    # --------------
    # save
    # --------------

    if pfe_save is not False:
        if pfe_save is None:
            name = 'fig02_distributions.png'
            pfe_save = os.path.join(_PATH_HERE, name)
        fig.savefig(pfe_save, format='png', dpi=300)
        msg = f"Saved figure in:\n\t{pfe_save}\n"
        print(msg)

    return dax


# #####################################################
# #####################################################
#       Fig 03 - cross-section
# #####################################################


_DR = {
    'R0': 1.8,
    'rplasma': 0.60,
    'RVes': [1.2, 2.66],
    'Rcryo': 4.6,
    'PP_R': np.r_[2.50, 4.7],  # 4.2
    'PP_width': 0.47,
    'PP_phi': np.r_[-180, 0] * np.pi/180,
}


_DSENSORS = {
    'in': {
        'pp': 0,
        'R': 2.55,
        'cw': False,
        'rplasma_ratio': 0.7,
        'color': 'b',
        'marker': '.',
        'ms': 2,
        'alpha': 0.6,
    },
    'ex': {
        'pp': 1,
        'R': 6,
        'cw': False,
        'color': 'g',
        'width': 0.10,
        'dist': 4,
        'marker': '.',
        'ms': 2,
        'alpha': 0.6,
        'wall': True,
        'beamdump': True,
    },
}


def fig03_tokamak(
    # tokamak
    R0=None,
    rplasma=None,
    RVes=None,
    Rcryo=None,
    # port plug
    PP_R=None,
    PP_width=None,
    PP_phi=None,
    # sensors
    in_pp=None,
    in_R=None,
    in_cw=None,
    in_rplasma_ratio=None,
    in_color=None,
    in_marker=None,
    in_ms=None,
    # ex
    res=None,
    ex_pp=None,
    ex_R=None,
    ex_cw=None,
    ex_width=None,
    ex_dist=None,
    ex_color=None,
    ex_marker=None,
    ex_ms=None,
    # plot
    figsize=(5, 7),
    fontsize=12,
    pfe_save=None,
):

    # --------------
    # Load SPARC
    # --------------

    config, dinput = _fig02_check(**locals())

    phi = np.pi * np.linspace(-1, 1, 181)
    cos = np.cos(phi)
    sin = np.sin(phi)

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.11, 'right': 0.97,
        'bottom': 0.06, 'top': 0.99,
        'wspace': 0.25, 'hspace': 0.20,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=1, nrows=2, **dmargin)
    dax = {}

    # --------------
    # axes - hor
    # --------------

    ax = fig.add_subplot(gs[0, 0], aspect='equal')
    ax.set_xlabel('X (m)', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=fontsize, fontweight='bold')
    ax.text(
        0.01,
        0.99,
        '(a)',
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=fontsize,
        fontweight='bold',
        transform=ax.transAxes,
    )

    dax['hor'] = ax

    # --------------
    # axes - ang vs rplasma
    # --------------

    ax = fig.add_subplot(gs[1, 0], aspect='auto')
    ax.set_xlabel('r / a', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel(
        r'$\theta_{ph,B}$ (deg)',
        fontsize=fontsize,
        fontweight='bold',
    )
    ax.text(
        0.01,
        0.99,
        '(b)',
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=fontsize,
        fontweight='bold',
        transform=ax.transAxes,
    )

    dax['theta_vs_B'] = ax

    dax = ds._generic_check._check_dax(dax)

    # --------------
    # plot hor
    # --------------

    kax = 'hor'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # --------------
        # plot R

        lk = [kk for kk in dinput.keys() if kk[0] == 'R']
        for k0 in lk:

            if 'plasma' in k0:
                inner = dinput[k0]['data'][0] * np.array([cos, sin]).T
                outer = dinput[k0]['data'][1] * np.array([cos, sin]).T
                vertices = np.concatenate((inner, outer[::-1]), axis=0)

                codes = np.ones(
                    len(inner),
                    dtype=mpath.Path.code_type,
                ) * mpath.Path.LINETO
                codes[0] = mpath.Path.MOVETO
                all_codes = np.concatenate((codes, codes))

                path = mpath.Path(vertices, all_codes)
                patch = mpatches.PathPatch(
                    path,
                    facecolor='r',
                    alpha=0.1,
                    edgecolor='r',
                )
                ax.add_patch(patch)

            else:
                for v1 in dinput[k0]['data']:
                    ax.plot(
                        v1*cos,
                        v1*sin,
                        **dinput[k0]['prop'],
                    )

        # --------------
        # add arrows

        R = dinput['R0']['data'][0] + 0.5 * dinput['rplasma']['data'][0]
        phi = np.r_[100, 160] * np.pi / 180
        # dist = R * np.hypot(
        # np.cos(phi[0]) - np.cos(phi[1]),
        # np.sin(phi[0]) - np.sin(phi[1]),
        # )
        # rad = (R * (1 - np.cos(np.abs(np.diff(phi)/2))) / dist)[0]
        rad = -0.3
        ax.annotate(
            "RE",
            xy=(R*np.cos(phi[0]), R*np.sin(phi[0])),
            xycoords='data',
            xytext=(R*np.cos(phi[1]), R*np.sin(phi[1])),
            textcoords='data',
            color='r',
            fontweight='bold',
            fontsize=fontsize,
            arrowprops=dict(
                arrowstyle="->",
                lw=1.5,
                color='r',
                shrinkA=5, shrinkB=5,
                patchA=None, patchB=None,
                connectionstyle=f'arc3,rad={rad}',
            ),
        )

        # --------------
        # plot port plug

        for ii, phi in enumerate(dinput['PP_phi']['data']):

            # edges
            width = dinput['PP_width']['data'][0]
            ppR0 = dinput['PP_R']['data'][0]
            ppR1 = dinput['PP_R']['data'][1]
            length = ppR1 - ppR0
            cent = 0.5 * (ppR0+ppR1) * np.r_[np.cos(phi), np.sin(phi)]
            xy = (
                cent[0] - 0.5 * length,
                cent[1] - 0.5 * width,
            )

            # patch
            patch = mpatches.Rectangle(
                xy,
                length,
                width,
                angle=phi*180/np.pi,
                rotation_point='center',
                facecolor='w',
                alpha=1.,
                edgecolor='None',
                zorder=10,
            )
            ax.add_patch(patch)

            # central line
            ppR0 = dinput['PP_R']['data'][0]
            ppR1 = dinput['PP_R']['data'][1]

            centx = np.r_[ppR0, ppR1] * np.cos(phi)
            centy = np.r_[ppR0, ppR1] * np.sin(phi)

            ax.plot(
                centx,
                centy,
                c='k',
                lw=1,
                ls='--',
                alpha=0.3,
                zorder=15,
            )

            # edges
            ephi = np.r_[-np.sin(phi), np.cos(phi)]
            edgex = (
                centx[None, :]
                + 0.5 * width * ephi[0] * np.r_[1, np.nan, -1][:, None]
            ).ravel()
            edgey = (
                centy[None, :]
                + 0.5 * width * ephi[1] * np.r_[1, np.nan, -1][:, None]
            ).ravel()

            ax.plot(
                edgex,
                edgey,
                c='k',
                lw=1,
                ls='-',
                zorder=20,
            )

        # --------------
        # add sensors

        dsensors = _sensors(**locals())

        for k0, v0 in dsensors.items():

            # FOV
            patch = mpatches.PathPatch(
                v0['path'],
                facecolor=v0['color'],
                alpha=v0['alpha'],
                zorder=25,
                edgecolor=v0['color'],
            )
            ax.add_patch(patch)

            # sensor
            ax.plot(
                [v0['cent'][0]],
                [v0['cent'][1]],
                c=v0['color'],
                ls='None',
                lw=2,
                marker=v0['marker'],
                zorder=30,
                label=v0.get('label', k0),
            )

            # FOV sampling
            ax.plot(
                v0['ptsx'],
                v0['ptsy'],
                c=v0['color'],
                marker=v0.get('marker', '.'),
                ls='None',
                zorder=30,
                ms=v0.get('ms', 4),
            )

    # ----------------
    # plot theta_vs_B
    # ----------------

    kax = 'theta_vs_B'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for k0, v0 in dsensors.items():

            ax.plot(
                v0['rplasma_norm'],
                v0['theta_vs_B'] * 180 / np.pi,
                marker=v0.get('marker', '.'),
                ms=v0.get('ms', 6),
                color=v0['color'],
                ls=v0.get('ls', 'None'),
                label=v0.get('label', k0),
            )

        ax.axhline(90, c='k', ls='--', lw=1)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 180)
        ax.grid(True)

        # comments
        ax.text(
            -0.97,
            55,
            'forward',
            horizontalalignment='left',
            verticalalignment='top',
            rotation=90,
            fontsize=fontsize,
            fontweight='bold',
            transform=ax.transData,
        )
        ax.text(
            -0.97,
            110,
            'backward',
            horizontalalignment='left',
            verticalalignment='bottom',
            rotation=90,
            fontsize=fontsize,
            fontweight='bold',
            transform=ax.transData,
        )

    # --------------
    # save
    # --------------

    if pfe_save is not False:
        if pfe_save is None:
            name = 'fig03_tokamak.png'
            pfe_save = os.path.join(_PATH_HERE, name)
        fig.savefig(pfe_save, format='png', dpi=300)
        msg = f"Saved figure in:\n\t{pfe_save}\n"
        print(msg)

    return dax


def _fig02_check(
    config=None,
    # tokamak
    R0=None,
    rplasma=None,
    RVes=None,
    Rcryo=None,
    PP_R=None,
    PP_width=None,
    PP_phi=None,
    # unused
    **kwdargs,
):

    # ------------------
    # config
    # ------------------

    if config is None:
        config = 'SPARC'

    if isinstance(config, str):
        config = tf.load_config(config)

    # ------------------
    # Geometry - R
    # ------------------

    lk = ['R0', 'rplasma', 'RVes', 'Rcryo', 'PP_R', 'PP_width', 'PP_phi']
    dinput = {
        kk: {
            'data': None,
            'prop': {'color': 'k', 'ls': '-', 'lw': 1, 'label': None}
        }
        for kk in lk
    }

    # R0
    dinput['R0']['data'] = np.r_[float(ds._generic_check._check_var(
        R0, 'R0',
        types=(float, int),
        sign='>0',
        default=_DR['R0'],
    ))]
    dinput['R0']['prop']['ls'] = '--'

    # rplasma
    dinput['rplasma']['data'] = np.r_[float(ds._generic_check._check_var(
        rplasma, 'rplasma',
        types=(float, int),
        sign='>0',
        default=_DR['rplasma'],
    ))]
    assert dinput['rplasma']['data'] < dinput['R0']['data']

    dinput['Rplasma'] = {
        'data': dinput['R0']['data'] + dinput['rplasma']['data']*np.r_[-1, 1],
        'prop': {
            'color': 'r',
            'lw': 1,
            'ls': '-',
            'label': 'plasma',
        }
    }

    # RVes
    if RVes is None:
        RVes = _DR['RVes']
    dinput['RVes']['data'] = ds._generic_check._check_flat1darray(
        RVes, 'RVes',
        dtype=float,
        sign='>0',
        unique=True,
        size=2,
    )
    Rlim = dinput['R0']['data'] - dinput['rplasma']['data']
    assert dinput['RVes']['data'][0] < Rlim
    Rlim = dinput['R0']['data'] + dinput['rplasma']['data']
    assert dinput['RVes']['data'][1] > Rlim
    dinput['RVes']['prop']['lw'] = 2

    # rplasma
    dinput['Rcryo']['data'] = np.r_[float(ds._generic_check._check_var(
        Rcryo, 'Rcryo',
        types=(float, int),
        sign='>0',
        default=_DR['Rcryo'],
    ))]
    assert dinput['Rcryo']['data'] > dinput['RVes']['data'][1]
    dinput['Rcryo']['prop']['lw'] = 2

    # ------------------
    # Geometry - Port plug
    # ------------------

    # PP_R
    if PP_R is None:
        PP_R = _DR['PP_R']
    dinput['PP_R']['data'] = ds._generic_check._check_flat1darray(
        PP_R, 'PP_R',
        dtype=float,
        sign='>0',
        unique=True,
        size=2,
    )
    Rin = dinput['R0']['data'] + dinput['rplasma']['data']
    assert dinput['PP_R']['data'][0] > Rin
    assert dinput['PP_R']['data'][1] > dinput['Rcryo']['data']

    # PP_width
    dinput['PP_width']['data'] = np.r_[float(ds._generic_check._check_var(
        PP_width, 'PP_width',
        types=(float, int),
        sign='>0',
        default=_DR['PP_width'],
    ))]

    # PP_phi
    if PP_phi is None:
        PP_phi = _DR['PP_phi']
    PP_phi = ds._generic_check._check_flat1darray(
        PP_phi, 'PP_phi',
        dtype=float,
        unique=True,
        size=2,
    )
    dinput['PP_phi']['data'] = np.arctan2(np.sin(PP_phi), np.cos(PP_phi))

    return config, dinput


def _sensors(
    res=None,
    # sensors - in
    in_pp=None,
    in_R=None,
    in_cw=None,
    in_rplasma_ratio=None,
    in_color=None,
    in_marker=None,
    in_ms=None,
    # sensors - ex
    ex_pp=None,
    ex_R=None,
    ex_cw=None,
    ex_width=None,
    ex_dist=None,
    ex_color=None,
    ex_marker=None,
    ex_ms=None,
    # dinput
    dinput=None,
    # unused
    **kwdargs,
):

    # --------------
    # inputs
    # --------------

    res = ds._generic_check._check_var(
        res, 'res',
        types=float,
        sign='>0',
        default=0.01,
    )

    # --------------
    # initialize
    # --------------

    dsensors = {
        'in': {
            kk.replace('in_', ''): vv for kk, vv in locals().items()
            if kk.startswith('in_')
        },
        'ex': {
            kk.replace('ex_', ''): vv for kk, vv in locals().items()
            if kk.startswith('ex_')
        },
    }

    # --------------
    # check
    # --------------

    for k0, v0 in dsensors.items():
        for k1, v1 in v0.items():
            if v1 is None:
                dsensors[k0][k1] = _DSENSORS[k0][k1]
        for k1, v1 in _DSENSORS[k0].items():
            if dsensors[k0].get(k1) is None:
                dsensors[k0][k1] = v1

    # --------------
    # Derive
    # --------------

    for k0, v0 in dsensors.items():

        phi = dinput['PP_phi']['data'][v0['pp']]
        eR = np.r_[np.cos(phi), np.sin(phi)]
        ephi = np.r_[-np.sin(phi), np.cos(phi)]
        sign = v0["cw"] * 2 - 1
        width = dinput['PP_width']['data']

        # ----------
        # cent

        if k0 == 'in':
            cent = v0['R'] * eR + sign * 0.5 * width * ephi
        else:
            length = dinput['PP_R']['data'][1] - dinput['PP_R']['data'][0]
            dphi = np.arctan2(width - v0['width'], length)
            eRs = eR * np.cos(dphi) + sign * ephi * np.sin(dphi)
            ppc = np.mean(dinput['PP_R']['data']) * eR
            cent = ppc + v0["dist"] * eRs

            # store
            dsensors[k0]["dphi"] = dphi
            dsensors[k0]["ppc"] = ppc
            dsensors[k0]["eRs"] = eRs

        dsensors[k0]['cent'] = cent

        # ----------
        # FOV

        if k0 == 'in':

            R0 = dinput['R0']['data'][0]
            rplasma = dinput['rplasma']['data'][0]

            # out
            R = R0 + rplasma * v0["rplasma_ratio"]
            vect_out = _tangent(cent, R, sign)

            # in
            R = R0 - rplasma * v0["rplasma_ratio"]
            vect_in = _tangent(cent, R, sign)

        else:
            ephis = np.r_[-eRs[1], eRs[0]]
            vect_out = (
                (length + v0["dist"]) * (-eRs) + 0.5 * v0['width'] * ephis
            )
            vect_in = (
                (length + v0["dist"]) * (-eRs) - 0.5 * v0['width'] * ephis
            )
            vect_out = vect_out / np.linalg.norm(vect_out)
            vect_in = vect_in / np.linalg.norm(vect_in)

        # Get FOV from cent + 2 vect
        xx, yy = _FOV(cent, vect_out, vect_in, R0, rplasma)
        path = mpath.Path(np.array([xx, yy]).T)

        # Sample FOV
        DX = np.max(xx) - np.min(xx)
        DY = np.max(yy) - np.min(yy)
        nptsx = int(DX / res)
        nptsy = int(DY / res)
        ptsx = np.linspace(np.min(xx), np.max(xx), nptsx)
        ptsy = np.linspace(np.min(yy), np.max(yy), nptsy)
        ptsx = np.repeat(ptsx[:, None], nptsy, axis=1).ravel()
        ptsy = np.repeat(ptsy[None, :], nptsx, axis=0).ravel()
        iok = (
            path.contains_points(np.array([ptsx, ptsy]).T)
            & (np.hypot(ptsx, ptsy) >= R0 - rplasma)
            & (np.hypot(ptsx, ptsy) <= R0 + rplasma)
        )
        ptsx = ptsx[iok]
        ptsy = ptsy[iok]

        # Angle
        pts_phi = np.arctan2(ptsy, ptsx)
        pts_ephi0 = -np.sin(pts_phi)
        pts_ephi1 = np.cos(pts_phi)
        vect0 = ptsx - cent[0]
        vect1 = ptsy - cent[1]
        vectn = np.sqrt(vect0**2 + vect1**2)
        vect0 = vect0 / vectn
        vect1 = vect1 / vectn
        theta_vs_B = np.arccos(vect0 * pts_ephi0 + vect1 * pts_ephi1)

        # store
        dsensors[k0]["ptsx"] = ptsx
        dsensors[k0]["ptsy"] = ptsy
        dsensors[k0]["rplasma_norm"] = (
            (np.hypot(ptsx, ptsy) - R0) / dinput['rplasma']['data']
        )
        dsensors[k0]["theta_vs_B"] = theta_vs_B
        dsensors[k0]["path"] = path

    return dsensors


def _tangent(
    cent=None,
    R=None,
    sign=None,
):

    # ---------
    #

    phi = np.arctan2(cent[1], cent[0])
    eR = np.r_[np.cos(phi), np.sin(phi)]
    ephi = np.r_[-np.sin(phi), np.cos(phi)]

    ang = np.arcsin(R / np.hypot(*cent))
    vect = (-eR) * np.cos(ang) - sign * ephi * np.sin(ang)
    vect = vect / np.linalg.norm(vect)

    return vect


def _FOV(cent, vect_out, vect_in, R0, rplasma):

    # ------------------
    # intersect vect_out
    # ------------------

    kk_out_out, isout_out_out = _intersect(cent, vect_out, R0 + rplasma)
    kk_out_in, isout_out_in = _intersect(cent, vect_out, R0 - rplasma)

    kk_out = np.r_[kk_out_out, kk_out_in]
    iok_out = np.r_[isout_out_out, ~isout_out_in]
    iok = np.isfinite(kk_out) & iok_out
    assert iok.sum() >= 1
    kout = np.min(kk_out[iok])
    pt_out = cent + kout * vect_out

    # ------------------
    # intersect vect_in
    # ------------------

    kk_in_out, isout_in_out = _intersect(cent, vect_in, R0 + rplasma)
    kk_in_in, isout_in_in = _intersect(cent, vect_in, R0 - rplasma)

    kk_in = np.r_[kk_in_out, kk_in_in]
    iok_in = np.r_[isout_in_out, ~isout_in_in]
    iok = np.isfinite(kk_in) & iok_in
    assert iok.sum() >= 1
    kin = np.min(kk_in[iok])
    pt_in = cent + kin * vect_in

    assert np.allclose(np.linalg.norm(pt_out), np.linalg.norm(pt_in))
    Rpts = np.linalg.norm(pt_out)

    # ------------------
    # polyx, polyy
    # ------------------

    # ang
    ang_out = np.arctan2(pt_out[1], pt_out[0])
    ang_in = np.arctan2(pt_in[1], pt_in[0])
    ang_min = min(ang_out, ang_in)
    ang_max = max(ang_out, ang_in)
    if np.abs(ang_min - ang_max) > np.pi:
        ang_min, ang_max = ang_max, ang_min + 2*np.pi
    ang = np.linspace(ang_min, ang_max, 31)

    polyx = np.r_[cent[0], Rpts * np.cos(ang), cent[0]]
    polyy = np.r_[cent[1], Rpts * np.sin(ang), cent[1]]

    return polyx, polyy


def _intersect(cent, vect, R):

    # ----------
    # kk

    # AM = ku
    # R^2 = (OA + AM)^2 = RA^2 + k^2 + 2 ku OA
    a = 1
    b = 2 * np.sum(vect * cent)
    c = np.sum(cent**2) - R**2
    delta = b**2 - 4 * a * c

    kk = np.full((2,), np.nan)
    if delta == 0:
        kk[0] = -b / (2*a)
    elif delta > 0:
        kk = (-b + np.r_[1, -1] * np.sqrt(delta)) / (2 * a)

    # ----------
    # isout

    xx = cent[0] + kk * vect[0]
    yy = cent[1] + kk * vect[1]
    phi = np.arctan2(yy, xx)
    isout = (vect[0] * np.cos(phi) + vect[1] * np.sin(phi)) > 0.

    assert isout.sum() <= 1

    return kk, isout
