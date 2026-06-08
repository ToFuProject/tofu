

import os
import sys


import numpy as np
import scipy.integrate as scpinteg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import astropy.units as asunits
import datastock as ds


_PATH_HERE = os.path.dirname(__file__)
_PATH_TF = os.path.dirname(os.path.dirname(_PATH_HERE))
sys.path.insert(0, _PATH_TF)
import tofu as tf
sys.path.pop(0)

tfphysdist = tf.physics_tools.electrons.distribution
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
    'Te_eV': np.r_[0.1e3, 0.1e3, 1e3, 1e3],
    'ne_m3': 1e19,
    'jp_Am2': 1e6,
    # RE
    'jp_fraction_re': np.r_[0.1, 0.9, 0.1, 0.9],
    'dominant': 'bump',
    'Ekin_max_eV': 1e6,
    'Ekin_min_eV': 100,
    'step': 1,
    'pnormW': 5,
    'theta_width': 20*np.pi/180,
    # coords
    'E_eV': np.logspace(0, 8, 80),
    'theta': np.linspace(0, 180, 181) * np.pi / 180,
}


def fig02_distributions(
    # coords
    E_eV=None,
    theta=None,
    # Maxwell
    ne_m3=None,
    jp_Am2=None,
    # RE
    dominant=None,
    jp_fraction_re=None,
    Ekin_max_eV=None,
    Ekin_min_eV=None,
    step=None,
    pnormW=None,
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
    dout = tfphysdist.get_distribution(**din)

    # units
    units2d = asunits.Unit(dout['dist']['RE']['dist']['units'])
    units1d = units2d * asunits.Unit(dout['coords']['x1']['units'])

    # ------------
    # Derive 1d data
    # ------------

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

    # ------------
    # Derive levels, vmin, vmax
    # ------------

    Ekin_max = dout['plasma']['Ekin_max_eV']['data']
    vminRE_2d = np.inf
    vminRE_1d = np.inf
    for ind in np.ndindex(dataRE.shape[:-1]):
        indE = np.argmin(np.abs(dout['coords']['x0']['data'] - Ekin_max[ind]))
        sli = ind + (indE, slice(None))
        vmaxRE_2d = np.nanmax(dout['dist']['RE']['dist']['data'][sli])
        vminRE_2d = min(vminRE_2d, vmaxRE_2d)
        sli = ind + (indE,)
        vmaxRE_1d = dataRE[sli]
        vminRE_1d = min(vminRE_1d, vmaxRE_1d)
    vmaxRE_2d = np.nanmax(dout['dist']['RE']['dist']['data'])
    vmaxRE_1d = np.nanmax(dataRE)
    vmaxMax_2d = np.nanmax(dout['dist']['maxwell']['dist']['data'])
    vmaxMax_1d = np.nanmax(dataMax)

    # 1d
    vmaxlog10_1d = np.log10(max(vmaxRE_1d, vmaxMax_1d))
    dlog10_1d = vmaxlog10_1d - np.log10(vminRE_1d)
    vmaxlog10_1d = np.ceil(vmaxlog10_1d)
    vminlog10_1d = np.floor(vmaxlog10_1d - 1 - 1.2*dlog10_1d)
    vmax_1d = 10**vmaxlog10_1d
    vmin_1d = 10**vminlog10_1d

    # 2d
    vmaxlog10_2d = np.log10(max(vmaxRE_2d, vmaxMax_2d))
    dlog10_2d = vmaxlog10_2d - np.log10(vminRE_2d)
    vmaxlog10_2d = np.ceil(vmaxlog10_2d)
    vminlog10_2d = np.floor(vmaxlog10_2d - 1 - 1.2*dlog10_2d)
    levels_2d = np.logspace(vminlog10_2d, vmaxlog10_2d - 1, 6)

    # --------------
    # labels
    # --------------

    dlabel = {}
    for ind in np.ndindex(dout['dist']['RE']['dist']['data'].shape[:-2]):
        Te = dout['plasma']['Te_eV']['data'][ind] * 1e-3
        jpf = dout['plasma']['jp_fraction_re']['data'][ind]

        dlabel[ind] = f"{jpf:2.1f}  ,    {Te:2.1f} keV"

    # title
    ne = np.unique(dout['plasma']['ne_m3']['data'])
    assert ne.size == 1
    jp = np.unique(dout['plasma']['jp_Am2']['data'])
    assert jp.size == 1
    tit = f"ne = {ne[0]:2.1e}, jp_tot = {jp[0]*1e-6:2.1f} MA/m2"

    # --------------
    # print
    # --------------

    _print(dout)

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.12, 'right': 0.98,
        'bottom': 0.06, 'top': 0.97,
        'wspace': 0.25, 'hspace': 0.10,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=1, nrows=2, **dmargin)
    dax = {}

    # --------------
    # axes - 2d
    # --------------

    ax = fig.add_subplot(gs[0, 0], aspect='auto', xscale='log')
    ax.set_ylabel(
        r'$\theta_{e_0,B}$ (deg)',
        fontsize=fontsize,
        fontweight='bold',
    )
    ax.set_title(tit, fontsize=fontsize, fontweight='bold')
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

    ax = fig.add_subplot(gs[1, 0], aspect='auto', sharex=ax, yscale='log')
    ax.set_xlabel('E (keV)', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel(
        f" ({units1d})",
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

    dax = ds._generic_check._check_dax(dax)

    # ------------
    # plot 1d
    # ------------

    kax = '1d'
    dcolor = {}
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # for legend
        ax.plot([], [], c='w', ls='-', lw=1, label='j_frac, Te')

        # loop plot
        for ind in np.ndindex(dataRE.shape[:-1]):
            sli = ind + (slice(None),)

            # Max
            l0, = ax.plot(
                dout['coords']['x0']['data']*1e-3,
                dataMax[sli],
                ls='-',
                lw=1,
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
                label=dlabel[ind],
            )

        # Add critical energy
        Ec = tf.physics_tools.electrons.convert_momentum_velocity_energy(
            momentum_normalized=dout['dist']['RE']['p_crit']['data'],
        )['energy_kinetic_eV']['data']
        for ec in np.unique(Ec):
            ax.axvline(ec*1e-3, c='k', lw=1, ls='--')

        # decorate
        ax.legend()
        ax.grid(True)
        ax.set_ylim(vmin_1d, vmax_1d)

    # ------------
    # plot 2d
    # ------------

    kax = '2d'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # loop plot
        for ind in np.ndindex(dataRE.shape[:-1]):
            sli = ind + (slice(None), slice(None))

            # data
            data = (
                dout['dist']['maxwell']['dist']['data'][sli]
                + dout['dist']['RE']['dist']['data'][sli]
            )

            # contour
            ax.contour(
                dout['coords']['x0']['data']*1e-3,
                dout['coords']['x1']['data']*180/np.pi,
                data.T,
                levels_2d,
                colors=dcolor[ind],
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

    return dax, dout


def _print(dout, sep='  '):

    # -----------
    # header

    head = [
        'ind',
        'Te (keV)',
        'ne (1e20/m3)', 'Max / RE',
        'jp (MA/m2)', 'Max / RE',
    ]
    lmax = np.max([len(ss) for ss in head])

    # -----------
    # header

    lc = []
    for ind in np.ndindex(dout['dist']['RE']['dist']['data'].shape[:-2]):
        Te = dout['plasma']['Te_eV']['data'][ind]*1e-3
        ne = dout['plasma']['ne_m3']['data'][ind]*1e-20
        jp = dout['plasma']['jp_Am2']['data'][ind]*1e-6
        ne_max = dout['dist']['maxwell']['integ_ne']['data'][ind]*1e-20
        ne_RE = dout['dist']['RE']['integ_ne']['data'][ind]*1e-20
        jp_max = dout['dist']['maxwell']['integ_jp']['data'][ind]*1e-6
        jp_RE = dout['dist']['RE']['integ_jp']['data'][ind]*1e-6

        cc = [
            str(ind),
            f'{Te:2.1f}',
            f'{ne:2.2f}', f"{ne_max:2.2f} / {ne_RE:2.2f}",
            f'{jp:2.2f}', f"{jp_max:2.2f} / {jp_RE:2.2f}",
        ]
        lc.append(cc)
        lmax = max(lmax, np.max([len(ss) for ss in cc]))

    # ----------------
    # concatenate

    line = sep.join(['-'*lmax for ss in head])
    head = sep.join([ss.ljust(lmax) for ss in head])
    lc = [
        sep.join([ss.ljust(lmax) for ss in cc])
        for cc in lc
    ]

    msg = '\n'.join([head, line] + lc)
    print(msg)

    return


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


# #####################################################
# #####################################################
#       Fig 04 - Bremsstrahlung
# #####################################################


_PFE_D2CROSS_PHI = os.path.join(
    _PATH_HERE,
    'd2cross_phi_Ee01eV-100MeV-80log_Eph1eV-100MeV-81log_nthetaph61_nthetae060_EH.npz',
)


_CASES = {
    'case': {
        '0': {'Te': 0.1e3, 'jp_frac': 0.9, 'color': 'r'},
        '1': {'Te': 1e3, 'jp_frac': 0.1, 'color': 'b'},
    },
    'theta_ph_vsB': {
        'val': np.r_[0, 0.5, 1]*np.pi,
        'ls': ['-', '--', ':'],
    },
    'E_ph_eV': {
        'val': np.r_[0.1, 1, 10]*1e3,
        'ls': ['-', '--', ':'],
    },
}


def fig04_Bremsstrahlung(
    d2cross_phi=None,
    # cases
    cases=None,
    # plot
    figsize=(7, 4),
    fontsize=14,
    # save
    pfe_save=None,
):

    # ------------
    # d2cross_phi
    # ------------

    if d2cross_phi is None:
        d2cross_phi = _PFE_D2CROSS_PHI

    # ------------
    # ddist
    # ------------

    ddist = locals()
    ddist = {
        kk: vv if ddist.get(kk) is None else ddist[kk]
        for kk, vv in _DDIST.items()
        if kk not in ['E_eV', 'theta']
    }
    ddist['Te_eV'] = 1e3 * np.linspace(0.1, 2.5, 25)[:, None]
    ddist['jp_fraction_re'] = np.linspace(0., 1., 11)[None, :]

    # --------------
    # integrated cross-section
    # --------------

    demiss, ddist, d2cross_phi = tfphysemis.get_xray_thin_integ_dist(
        # ----------------
        # cross-section
        # tabulated d2cross_phi
        d2cross_phi=d2cross_phi,
        # d2cross_phi computation
        E_ph_eV=None,
        E_e0_eV=None,
        E_e0_eV_npts=None,
        theta_e0_vsB_npts=None,
        phi_e0_vsB_npts=None,
        theta_ph_vsB=None,
        # inputs
        Z=None,
        # hypergeometric parameter
        ninf=None,
        source=None,
        # integration parameters
        nthetae=None,
        ndphi=None,
        # output customization
        version_cross=None,
        # save / load
        save_d2cross_phi=False,
        # ---------------------
        # optional responsivity
        dresponsivity=None,
        plot_responsivity_integration=None,
        # -----------
        # verb
        debug=False,
        verb=True,
        # ----------------
        # electron distribution
        **ddist,
    )

    units = demiss['emiss']['RE']['emiss']['units']

    # --------------
    # cases
    # --------------

    if cases is None:
        cases = _CASES

    # --------------
    # Elim
    # --------------

    shape = ddist['plasma']['Te_eV']['data'].shape
    Elim = np.full(shape, np.nan)
    theta_Elim = 0
    for ii, ind in enumerate(np.ndindex(shape)):

        sli_emiss = ind + (slice(None), 0)
        emiss_RE = demiss['emiss']['RE']['emiss']['data'][sli_emiss]
        emiss_max = demiss['emiss']['maxwell']['emiss']['data'][sli_emiss]

        ilim = (emiss_RE > emiss_max)
        if np.any(ilim):
            Elim[ind] = np.min(demiss['E_ph_eV']['data'][ilim])

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.11, 'right': 0.97,
        'bottom': 0.12, 'top': 0.98,
        'wspace': 0.25, 'hspace': 0.20,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=2, nrows=2, **dmargin)
    dax = {}

    # ----------------
    # ax - spectra
    # ----------------

    ax = fig.add_subplot(gs[0, 0], aspect='auto')
    ax.set_xlabel('E (keV)', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel(f'emiss ({units})', fontsize=fontsize, fontweight='bold')

    dax['spectra'] = ax

    # ----------------
    # ax - theta
    # ----------------

    ax = fig.add_subplot(gs[1, 0], aspect='auto')
    ax.set_xlabel(
        r'$\theta_{ph,B}$' + ' (deg)',
        fontsize=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(f'emiss ({units})', fontsize=fontsize, fontweight='bold')
    ax.set_xlim(0, 180)

    dax['theta'] = ax

    # ----------------
    # ax - Elim
    # ----------------

    ax = fig.add_subplot(gs[:, 1], aspect='auto')
    ax.set_xlabel('Te (keV)', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel('jp_frac', fontsize=fontsize, fontweight='bold')

    dax['Elim'] = ax

    dax = ds._generic_check._check_dax(dax)

    # --------------
    # plot - cases
    # --------------

    Teu = np.unique(ddist['plasma']['Te_eV']['data'])
    jp_fracu = np.unique(ddist['plasma']['jp_fraction_re']['data'])
    for kdist in demiss['emiss'].keys():

        # loop on spectra
        for i0, (k0, v0) in enumerate(cases['case'].items()):

            # slice
            Te = Teu[np.argmin(np.abs(Teu - v0['jp_frac']))]
            jpf = jp_fracu[np.argmin(np.abs(jp_fracu - v0['jp_frac']))]
            ic = (
                (ddist['plasma']['jp_fraction_re']['data'] == jpf)
                & (ddist['plasma']['Te_eV']['data'] == Te)
            )
            assert ic.sum() == 1

            # --------
            # spectra

            kax = 'spectra'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                for i1, cc in enumerate(cases['theta_ph_vsB']['val']):
                    it = np.argmin(np.abs(demiss['theta_ph_vsB']['data'] - cc))
                    sli = tuple([cc[0] for cc in ic.nonzero()]) + (slice(None), it)
                    emiss_E = demiss['emiss'][kdist]['emiss']['data'][sli]

                    # plot
                    ax.loglog(
                        demiss['E_ph_eV']['data']*1e-3,
                        emiss_E,
                        ls=cases['theta_ph_vsB']['ls'][i1],
                        lw=1 if kdist == 'RE' else 2,
                        marker='None',
                        color=v0['color'],
                    )

            # --------
            # theta

            kax = 'theta'
            if dax.get(kax) is not None:
                ax = dax[kax]['handle']

                for i1, cc in enumerate(cases['E_ph_eV']['val']):
                    iE = np.argmin(np.abs(demiss['E_ph_eV']['data'] - cc))
                    sli = sli[:-2] + (iE, slice(None))
                    emiss_theta = demiss['emiss'][kdist]['emiss']['data'][sli]

                    # plot
                    ax.plot(
                        demiss['theta_ph_vsB']['data']*180/np.pi,
                        emiss_theta / emiss_theta.max(),
                        ls=cases['theta_ph_vsB']['ls'][i1],
                        lw=1 if kdist == 'RE' else 2,
                        marker='None',
                        color=v0['color'],
                    )

    # --------------
    # plot - Elim
    # --------------

    kax = 'Elim'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        cs = ax.contour(
            ddist['plasma']['Te_eV']['data'] * 1e-3,
            ddist['plasma']['jp_fraction_re']['data'],
            Elim * 1e-3,
            cmap=plt.cm.viridis,
            levels=np.r_[0.1, 0.2, 0.5, 1, 2, 5, 10, 20]*1e3,
            vmin=0.1,
            vmax=20,
        )

        ax.clabel(cs, cs.levels, fontsize=12)

        ax.set_xlim(0, ddist['plasma']['Te_eV']['data'].max()*1e-3)
        ax.set_ylim(0, 1)

    return dax, demiss, ddist, d2cross_phi


# #####################################################
# #####################################################
#       Fig 05 - responsivities
# #####################################################


_PFE_RESPONSIVITIES = os.path.join(_PATH_HERE, 'responsivities.npz')


def fig05_responsivities(
    pfe=None,
    lw=2,
    figsize=(7, 4),
    fontsize=14,
    pfe_save=None,
):

    # --------------
    # load
    # --------------

    if pfe is None:
        pfe = _PFE_RESPONSIVITIES

    dresp = {
        k0: v0.tolist()
        for k0, v0 in np.load(pfe, allow_pickle=True).items()
    }

    # --------------
    # prepare axes
    # --------------

    dmargin = {
        'left': 0.11, 'right': 0.97,
        'bottom': 0.12, 'top': 0.98,
        'wspace': 0.25, 'hspace': 0.20,
    }

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(ncols=1, nrows=1, **dmargin)
    dax = {}

    # --------------
    # axes - resp
    # --------------

    ax = fig.add_subplot(gs[0, 0], aspect='auto')
    ax.set_xlabel('E (keV)', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel('responsivity', fontsize=fontsize, fontweight='bold')

    dax['resp'] = ax

    dax = ds._generic_check._check_dax(dax)

    # --------------
    # plot - resp
    # --------------

    kax = 'resp'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # ---------------
        # loop on sensors

        dme = {'mesxr': False, 'mehxr': False, 'cvd': False}
        for k0, v0 in dresp.items():

            # resp, color, lab
            lk = [kk for kk in dme.keys() if kk in k0]
            if len(lk) == 1:
                kk = lk[0]
                if dme[kk] is False:
                    dme[kk] = v0.get('color', 'k')
                    lab = f"{kk}  -  {v0['responsivity']['units']}"
                else:
                    v0['color'] = dme[kk]
                    v0['ls'] = '--'
                    lab = None
            else:
                lab = f"{k0}  -  {v0['responsivity']['units']}"

            # lw
            if lw is None:
                lwi = v0.get('lw', 1)
            else:
                lwi = lw

            # plot
            ax.loglog(
                v0['E_eV']['data']*1e-3,
                v0['responsivity']['data'],
                ls=v0.get('ls', '-'),
                lw=lwi,
                c=v0.get('color', 'k'),
                marker=v0.get('marker', 'None'),
                label=lab,
            )

        ax.set_ylim(1e-4, 2)
        ax.grid(True)
        ax.legend()

    # --------------
    # save
    # --------------

    if pfe_save is not False:
        if pfe_save is None:
            name = 'fig05_responsivities.png'
            pfe_save = os.path.join(_PATH_HERE, name)
        fig.savefig(pfe_save, format='png', dpi=300)
        msg = f"Saved figure in:\n\t{pfe_save}\n"
        print(msg)

    return dax, dresp
