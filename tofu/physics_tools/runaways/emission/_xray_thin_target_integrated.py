

import os


import numpy as np
import scipy.integrate as scpinteg
import astropy.units as asunits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


from . import _xray_thin_target


# ####################################################
# ####################################################
#           DEFAULT
# ####################################################


_PATH_HERE = os.path.dirname(__file__)


_E_E0_EV = 45e3
_E_PH_EV = 40e3
_THETA_PH = np.linspace(0, np.pi, 31)
_NTHETAE = 31
_NDPHI = 51


# ####################################################
# ####################################################
#        main
# ####################################################


def get_xray_thin_d2cross_ei_integrated_thetae_dphi(
    # inputs
    Z=None,
    E_e0_eV=None,
    E_ph_eV=None,
    theta_ph=None,
    # hypergeometric parameter
    ninf=None,
    source=None,
    # integration parameters
    nthetae=None,
    ndphi=None,
    # output customization
    per_energy_unit=None,
    # version
    version=None,
    # verb
    verb=None,
):

    # ------------
    # inputs
    # ------------

    (
        E_e0_eV, E_ph_eV, theta_ph,
        nthetae, ndphi,
        shape, shape_theta_e, shape_dphi,
        verb,
    ) = _check(
        # inputs
        E_e0_eV=E_e0_eV,
        E_ph_eV=E_ph_eV,
        theta_ph=theta_ph,
        # integration parameters
        nthetae=nthetae,
        ndphi=ndphi,
        # verb
        verb=verb,
    )

    # ------------------
    # Derive angles
    # ------------------

    # E_e1_eV
    E_e1_eV = E_e0_eV - E_ph_eV

    # angles
    theta_e = np.pi * np.linspace(0, 1, nthetae)
    dphi = np.pi * np.linspace(-1, 1, ndphi)
    theta_ef = theta_e.reshape(shape_theta_e)
    dphif = dphi.reshape(shape_dphi)

    # derived
    sinte = np.sin(theta_ef)

    # ------------------
    # get d3cross
    # ------------------

    if verb is True:
        msg = "Computing d3cross..."
        print(msg)

    d3cross = _xray_thin_target.get_xray_thin_d3cross_ei(
        # inputs
        Z=Z,
        E_e0_eV=E_e0_eV[..., None, None],
        E_e1_eV=E_e1_eV[..., None, None],
        # directions
        theta_ph=theta_ph[..., None, None],
        theta_e=theta_ef,
        dphi=dphif,
        # hypergeometric parameter
        ninf=ninf,
        source=source,
        # output customization
        per_energy_unit=per_energy_unit,
        # version
        version=version,
        # debug
        debug=False,
    )

    # ------------------
    # prepare output
    # ------------------

    d2cross = {
        # energies
        'E_e0': {
            'data': E_e0_eV,
            'units': 'eV',
        },
        'E_ph': {
            'data': E_ph_eV,
            'units': 'eV',
        },
        # angles
        'theta_ph': {
            'data': theta_ph,
            'units': 'rad',
        },
        'theta_e': {
            'data': theta_e,
            'units': 'rad',
        },
        'dphi': {
            'data': dphi,
            'units': 'rad',
        },
        # cross-section
        'cross': {
            vv: {
                'data': np.full(shape, 0.),
                'units': asunits.Unit(vcross['units']) * asunits.Unit('sr'),
            }
            for vv, vcross in d3cross['cross'].items()
        },
    }

    # ------------------
    # integrate
    # ------------------

    if verb is True:
        msg = "Integrating..."
        print(msg)

    for vv, vcross in d3cross['cross'].items():
        d2cross['cross'][vv]['data'][...] = scpinteg.simpson(
            scpinteg.simpson(
                vcross['data'] * sinte,
                x=theta_e,
                axis=-1,
            ),
            x=dphi,
            axis=-1,
        )

    return d2cross


# ####################################################
# ####################################################
#        check
# ####################################################


def _check(
    # inputs
    E_e0_eV=None,
    E_ph_eV=None,
    theta_ph=None,
    # integration parameters
    nthetae=None,
    ndphi=None,
    # verb
    verb=None,
):

    # -----------
    # arrays
    # -----------

    # --------
    # E_e0_eV

    if E_e0_eV is None:
        E_e0_eV = _E_E0_EV
    E_e0_eV = np.atleast_1d(E_e0_eV)

    # --------
    # E_ph_eV

    if E_ph_eV is None:
        E_ph_eV = _E_PH_EV
    E_ph_eV = np.atleast_1d(E_ph_eV)

    # -------
    # theta_e

    if theta_ph is None:
        theta_ph = _THETA_PH
    theta_ph = np.atleast_1d(theta_ph)

    # -------------
    # Broadcastable

    dout, shape = ds._generic_check._check_all_broadcastable(
        return_full_arrays=False,
        E_e0_eV=E_e0_eV,
        E_ph_eV=E_ph_eV,
        # directions
        theta_ph=theta_ph,
    )

    # -----------
    # shapes
    # -----------

    shape = np.broadcast_shapes(E_e0_eV.shape, E_ph_eV.shape, theta_ph.shape)
    shape_theta_e = (1,) * (len(shape)+1) + (-1,)
    shape_dphi = (1,) * len(shape) + (-1, 1)

    # -----------
    # integers
    # -----------

    # nthetae
    nthetae = ds._generic_check._check_var(
        nthetae, 'nthetae',
        types=int,
        sign='>0',
        default=_NTHETAE,
    )

    # ndphi
    ndphi = ds._generic_check._check_var(
        ndphi, 'ndphi',
        types=int,
        sign='>0',
        default=_NDPHI,
    )

    # -----------
    # verb
    # -----------

    verb = ds._generic_check._check_var(
        verb, 'verb',
        types=bool,
        default=True,
    )

    return (
        E_e0_eV, E_ph_eV, theta_ph,
        nthetae, ndphi,
        shape, shape_theta_e, shape_dphi,
        verb,
    )


# ####################################################
# ####################################################
#        plot vs litterature
# ####################################################


def plot_xray_thin_d2cross_ei_vs_literature():
    """ Plot electron-angle-integrated cross section vs

    [1] G. Elwert and E. Haug, Phys. Rev., 183, pp. 90–105, 1969
        doi: 10.1103/PhysRev.183.90.

    """

    # --------------
    # Load literature data
    # --------------

    # isolines
    pfe_fig12 = os.path.join(
        _PATH_HERE,
        'RE_HXR_CrossSection_ThinTarget_PhotonAngle_ElwertHaug_fig12.csv',
    )
    out_fig12 = np.loadtxt(pfe_fig12, delimiter=',')

    # --------------------
    # prepare data
    # --------------------

    msg = "\nComputing data for fig12 (1/3):"
    print(msg)

    theta_ph = np.linspace(0, 1, 31)*np.pi

    # -----------
    # fig 12

    msg = "\t- For Z = 8... (1/2)"
    print(msg)

    d2cross_fig12_Z8 = get_xray_thin_d2cross_ei_integrated_thetae_dphi(
        # inputs
        Z=8,
        E_e0_eV=45e3,
        E_ph_eV=40e3,
        theta_ph=theta_ph,
        # output customization
        per_energy_unit=None,
        # version
        version=['EH', 'BH'],
        # verb
        verb=False,
    )

    msg = "\t- For Z = 13... (1/2)"
    print(msg)

    d2cross_fig12_Z13 = get_xray_thin_d2cross_ei_integrated_thetae_dphi(
        # inputs
        Z=13,
        E_e0_eV=45e3,
        E_ph_eV=40e3,
        theta_ph=theta_ph,
        # output customization
        per_energy_unit=None,
        # version
        version=['EH', 'BH'],
        # verb
        verb=False,
    )

    # --------------
    # prepare axes
    # --------------

    fontsize = 14
    tit = (
        "[1] G. Elwert and E. Haug, Phys. Rev., 183, p.90, 1969\n"
    )

    dmargin = {
        'left': 0.08, 'right': 0.95,
        'bottom': 0.06, 'top': 0.85,
        'wspace': 0.2, 'hspace': 0.40,
    }

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(tit, size=fontsize+2, fontweight='bold')

    gs = gridspec.GridSpec(ncols=2, nrows=1, **dmargin)
    dax = {}

    # --------------
    # prepare axes
    # --------------

    # --------------
    # ax - isolines

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(
        r"$\theta_{ph}$ (photon emission angle, deg)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$\frac{k}{Z^2}\frac{d^2\sigma}{dkd\Omega_{ph}}$ (mb/sr)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        "[1] Fig 12. Integrated cross-section (vs theta_e and phi)\n"
        "Comparisation between experimental values and models\n"
        + r"$Z = O$ (O) and $Z = 13$ (Al), "
        + r"$E_{e0} = 45 keV$, $E_{e1} = 5 keV$"
        + "\nTarget was " + r"$Al_2O_3$",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['fig12'] = {'handle': ax, 'type': 'isolines'}

    # ------------
    # ax - ph_dist

    # ---------------
    # plot fig 12
    # ---------------

    kax = 'fig12'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # literature data
        inan = np.r_[0, np.any(np.isnan(out_fig12), axis=1).nonzero()[0], -1]
        dls = {
            0: {'ls': '--', 'lab': 'Born approx'},
            1: {'ls': '-.', 'lab': 'Z = 8, EH'},
            2: {'ls': '-', 'lab': 'Z = 13, EH'},
            3: {'ls': '-', 'lab': 'Z = 13, Non-rel.'},
            4: {'ls': 'None', 'lab': 'exp.'},
        }
        for ii, ia in enumerate(inan[:-1]):
            ax.plot(
                out_fig12[inan[ii]:inan[ii+1], 0],
                out_fig12[inan[ii]:inan[ii+1], 1],
                c='k',
                ls=dls[ii]['ls'],
                marker='o' if ii == 4 else 'None',
                ms=10,
                label=dls[ii]['lab'],
            )

        # -------------
        # computed data

        # Z = 13
        Z = 13
        for k0, v0 in d2cross_fig12_Z13['cross'].items():
            ax.plot(
                theta_ph * 180/np.pi,
                v0['data']*1e28*1e3 * 40e3 / Z**2,
                ls='-',
                label=f'computed - {k0} Z = {Z}',
            )

        # Z = 8
        Z = 8
        for k0, v0 in d2cross_fig12_Z8['cross'].items():
            ax.plot(
                theta_ph * 180/np.pi,
                v0['data']*1e28*1e3 * 40e3 / Z**2,
                ls='-',
                label=f'computed - {k0} Z = {Z}',
            )

        ax.set_xlim(0, 180)

        # add legend
        ax.legend()

    # ------------------------
    # plot photon distribution
    # ------------------------

    return dax, d2cross_fig12_Z13, d2cross_fig12_Z8


# ####################################################
# ####################################################
#        plot anisotropy
# ####################################################


def plot_xray_thin_d2cross_ei_anisotropy(
    Z=None,
    theta_ph0=None,
    theta_ph1=None,
    E_e0_eV=None,
    E_ph_eV=None,
    version=None,
):

    # ---------------
    # check inputs
    # ---------------

    # E_e0_eV
    if E_e0_eV is None:
        E_e0_eV = np.linspace(10, 10000, 20) * 1e3

    E_e0_eV = ds._generic_check._check_flat1darray(
        E_e0_eV, 'E_e0_eV',
        dtype=float,
        sign='>0',
    )

    # E_ph_eV
    if E_ph_eV is None:
        E_ph_eV = np.linspace(10, 100, 1000, 10) * 1e3

    E_ph_eV = ds._generic_check._check_flat1darray(
        E_ph_eV, 'E_ph_eV',
        dtype=float,
        sign='>0',
    )

    # theta_ph0
    theta_ph0 = float(ds._generic_check._check_var(
        theta_ph0, 'theta_ph0',
        types=(float, int),
        default=0,
    ))
    theta_ph0 = np.arctan2(np.sin(theta_ph0), np.cos(theta_ph0))

    # theta_ph1
    theta_ph1 = float(ds._generic_check._check_var(
        theta_ph1, 'theta_ph1',
        types=(float, int),
        default=np.pi/2.,
    ))
    theta_ph1 = np.arctan2(np.sin(theta_ph1), np.cos(theta_ph1))

    # ---------------
    # prepare data
    # ---------------

    d2cross_map = get_xray_thin_d2cross_ei_integrated_thetae_dphi(
        # inputs
        Z=Z,
        E_e0_eV=E_e0_eV[None, :, None],
        E_ph_eV=E_ph_eV[None, None, :],
        theta_ph=np.r_[theta_ph0, theta_ph1][:, None, None],
        # output customization
        per_energy_unit=None,
        # version
        version=version,
        # verb
        verb=False,
    )

    # ---------------
    # prepare norm
    # ---------------

    E_e0_eV_norm = np.r_[20, 200, 2000] * 1e3
    E_ph_eV_norm = np.r_[10, 100] * 1e3
    theta_ph = np.linspace(0, np.pi, 31)

    d2cross_norm = get_xray_thin_d2cross_ei_integrated_thetae_dphi(
        # inputs
        Z=Z,
        E_e0_eV=E_e0_eV_norm[None, :, None],
        E_ph_eV=E_ph_eV_norm[None, None, :],
        theta_ph=theta_ph[:, None, None],
        # output customization
        per_energy_unit=None,
        # version
        version=version,
        # verb
        verb=False,
    )

    # --------------
    # prepare axes
    # --------------

    fontsize = 14
    tit = (
        "Anisotropy = d2cross(angle0) / d2cross(angle1)"
    )

    dmargin = {
        'left': 0.08, 'right': 0.95,
        'bottom': 0.06, 'top': 0.85,
        'wspace': 0.2, 'hspace': 0.40,
    }

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(tit, size=fontsize+2, fontweight='bold')

    gs = gridspec.GridSpec(ncols=2, nrows=1, **dmargin)
    dax = {}

    # --------------
    # prepare axes
    # --------------

    # --------------
    # ax - isolines

    ax = fig.add_subplot(gs[0, 0])
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
        r"$d^2\sigma(\theta_0, Z) / d^2\sigma(\theta_1, Z)$"
        + f"\n Z = {Z}, "
        + r"$\theta_0 = $" + f"{theta_ph0*180/np.pi:3.1f} deg, "
        + r"$\theta_1 = $" + f"{theta_ph1*180/np.pi:3.1f} deg",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['map'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # ax - profiles

    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlabel(
        r"$\theta_{ph}$ (deg)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        "normalized cross-section (adim.)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_title(
        "Normalized cross-section vs photon emission angle",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['norm'] = {'handle': ax, 'type': 'isolines'}

    # ---------------
    # plot - map
    # ---------------

    kax = 'map'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for iv, (kk, vv) in enumerate(d2cross_map['cross'].items()):

            ax.contour(
                E_e0_eV * 1e-3,
                E_ph_eV * 1e-3,
                (vv['data'][0, ...] / vv['data'][1, ...]).T,
                50,
            )

    # ---------------
    # plot - norm
    # ---------------

    kax = 'norm'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for iv, (kk, vv) in enumerate(d2cross_norm['cross'].items()):

            for ie, ee0 in enumerate(E_e0_eV_norm):
                for iph, eph in enumerate(E_ph_eV_norm):
                    lab = (
                        r"$E_{ph} / E_{e0}$ = "
                        + f"{ee0*1e-3:3.0f} keV / {eph*1e-3:3.0f} keV = "
                        + f"{ee0 / eph} - {kk}"
                    )
                    yy = vv['data'][:, ie, iph]

                    ax.plot(
                        theta_ph * 180/np.pi,
                        yy / np.max(yy),
                        label=lab,
                    )

    return dax, d2cross_map, d2cross_norm
