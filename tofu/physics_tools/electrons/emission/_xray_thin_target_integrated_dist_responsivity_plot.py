

import copy
from typing import Optional   # Any, Dict


import numpy as np
import astropy.units as asunits
import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import datastock as ds


# from . import _xray_thin_target_integrated as _mod
from . import _xray_thin_target_integrated_plot as _mod_plot
# from ..distribution import get_distribution


TupleDict = tuple[dict]


# ####################################################
# ####################################################
#           DEFAULT
# ####################################################


# Energy vectors
_E_E0_EV = np.logspace(3, 6, 51)
_E_PH_EV = np.linspace(1, 100, 25) * 1e3
_THETA_PH = np.linspace(0, np.pi, 41)
_VERSION = 'BHE'


# DCASES
_DCASES = {
    'cvd no filter maxwell': {
        'E_ph': {},
        'responsivity': {},
        'dist': {},
        'E_e0': {},
    },
}


# ####################################################
# ####################################################
#        plot integrated filtered anisotropy
# ####################################################


def plot_xray_thin_integ_dist_filter_anisotropy(
    # optional input d2cross file
    d2cross: Optional[str | dict] = None,
    # target ion charge
    Z: Optional[int] = None,
    # Energy
    E_e0_eV=None,
    E_ph_eV=None,
    theta_ph=None,
    # hypergeometric parameter
    ninf: Optional[int] = None,
    source: Optional[str] = None,
    # output customization
    per_energy_unit: Optional[str] = None,
    # version
    version: Optional[str] = None,
    # selected cases
    dcases: Optional[dict[int, dict]] = None,
    # plot
    dax: Optional[dict] = None,
    fs: Optional[tuple] = None,
    fontsize: Optional[int] = None,
    dplot_forbidden=None,
    dplot_peaking=None,
    dplot_thetamax=None,
    dplot_mean=None,
) -> TupleDict:
    """ Compute and plot a (E_e0, E_ph) countour map of the d2cross section

    Where d2cross is the fully differentiated cross-section (d3cross),
    integrated over one of the two the emission angle (dphi)

    Actually 3 overlayed contour plots with:
        - integral of of the cross-section (over photon emission angle)
        - angle of max cross-section
        - peaking of the cross-section (std vs angle)

    Can overlay a few selected cases and plot them vs angle of emission
    In normalized-linear and log scales

    """

    # ---------------
    # check inputs
    # ---------------

    dcases, fs, fontsize = _check(
        dcases=dcases,
        fs=fs,
        fontsize=fontsize,
    )

    # ---------------
    # prepare data
    # ---------------

    # --------------
    # prepare axes
    # --------------

    if dax is None:
        dax = _dax(
            Z=Z,
            version=version,
            fs=fs,
            fontsize=fontsize,
        )

    dax = ds._generic_check._check_dax(dax)

    # -------------------
    # plot cross-section
    # -------------------

    dax_cross, d2cross_cross = _mod_plot.plot_xray_thin_d2cross_ei_anisotropy(
        # optional input d2cross file
        d2cross=d2cross,
        # target ion charge
        Z=Z,
        # Energy
        E_e0_eV=E_e0_eV,
        E_ph_eV=E_ph_eV,
        theta_ph=theta_ph,
        # hypergeometric parameter
        ninf=ninf,
        source=source,
        # output customization
        per_energy_unit=per_energy_unit,
        # version
        version=version,
        # selected cases
        dcases=False,
        # plot
        dax=dax,
        dplot_forbidden=dplot_forbidden,
        dplot_peaking=dplot_peaking,
        dplot_thetamax=dplot_thetamax,
        dplot_mean=dplot_mean,
    )

    # -------------------
    # plot responsivity
    # -------------------

    kax = 'responsivity'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for kk, vv in dcases.items():

            l0, = ax.plot(
                vv['responsivity']['data'],
                vv['E_ph']['data']*1e-3,
                c=vv.get('color'),
                ls=vv.get('ls', '-'),
                marker=vv.get('marker'),
                lw=vv.get('lw', 1.),
                label=kk,
            )
            dcases[kk]['color'] = l0.get_color()

        ax.set_ylabel(
            vv['responsivity']['units'],
            fontsize=fontsize,
            fontweight='bold',
        )

    # -------------------
    # plot distribution
    # -------------------

    kax = 'dist'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for kk, vv in dcases.items():

            ax.semilogy(
                vv['E_e0']['data']*1e-3,
                vv['dist']['data'],
                c=vv['color'],
                ls=vv.get('ls', '-'),
                marker=vv.get('marker'),
                lw=vv.get('lw', 1.),
                label=kk,
            )

        ax.set_ylabel(
            vv['dist']['units'],
            fontsize=fontsize,
            fontweight='bold',
        )

    return dax


# #############################################
# #############################################
#        Axes for anisotropy
# #############################################


def _check(
    dcases=None,
    fs=None,
    fontsize=None,
):

    # ------------
    # dcases
    # ------------

    ddef = copy.deepcopy(_DCASES)
    if dcases in [None, False]:
        dcases = {}
    else:
        for k0, v0 in dcases.items():
            dcases[k0] = _check_case(
                v0,
                f"dcases['{k0}']",
                ddef[list(ddef.keys())[0]],
            )

    # ------------
    # fs
    # ------------

    if fs is None:
        fs = (15, 12)

    # ------------
    # fontsize
    # ------------

    fontsize = ds._generic_check._check_var(
        fontsize, 'fontsize',
        types=int,
        default=14,
        sign='>0',
    )

    return dcases, fs, fontsize


def _check_case(
    case=None,
    key=None,
    ddef=None,
):

    # --------------
    # general structure
    # --------------

    dfail = {}
    lok = list(ddef.keys())
    ltunits = (str, asunits.Unit, asunits.CompositeUnit)
    for kk in lok:
        if not isinstance(case.get(kk), dict):
            typ = type(case.get(kk))
            dfail[kk] = f'absent or not a dict ({typ})'
        elif not isinstance(case[kk].get('data'), np.ndarray):
            typ = type(case[kk].get('data'))
            dfail[kk] = f'data key not a np.ndarray ({typ})'
        elif not isinstance(case[kk].get('units'), ltunits):
            typ = type(case[kk].get('units'))
            dfail[kk] = f"units not a str ({typ})"
        else:
            dfail[kk] = 'ok'

    if any([vv != 'ok' for vv in dfail.values()]):
        lstr = [f"\t- {kk}: {vv}" for kk, vv in dfail.items()]
        msg = (
            f"Arg {key} must be a dict with keys {lok}, "
            "where each is a {'data': np.ndarray, 'units': str} subdict!\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # --------------
    # shape consistency
    # --------------

    shape_Eph = case['E_ph']['data'].shape
    shape_resp = case['responsivity']['data'].shape
    if shape_Eph != shape_resp:
        msg = (
            "The 2 fields below must have the same shape:\n"
            f"{key}['E_ph']['data'].shape = {shape_Eph}\n"
            f"{key}['responsivity']['data'].shape = {shape_resp}\n"
        )
        raise Exception(msg)

    shape_Ee0 = case['E_e0']['data'].shape
    shape_dist = case['dist']['data'].shape
    if shape_Ee0 != shape_dist:
        msg = (
            "The 2 fields below must have the same shape:\n"
            f"{key}['E_e0']['data'].shape = {shape_Ee0}\n"
            f"{key}['dist']['data'].shape = {shape_dist}\n"
        )
        raise Exception(msg)

    return case


# #############################################
# #############################################
#        Axes for anisotropy
# #############################################


def _dax(
    Z=None,
    version=None,
    fs=None,
    fontsize=None,
):

    # --------------
    # prepare figure
    # --------------

    tit = (
        "Thin-target Bremsstrahlung emission anisotropy"
    )

    dmargin = {
        'left': 0.06, 'right': 0.95,
        'bottom': 0.06, 'top': 0.90,
        'wspace': 0.20, 'hspace': 0.20,
    }

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(tit, size=fontsize+2, fontweight='bold')

    nh0, nh1, nh2 = 2, 5, 4
    nv0, nv1, nv2 = 3, 1, 2
    gs = gridspec.GridSpec(
        ncols=nh0+nh1+nh2 + 1,
        nrows=nv0 + nv1,
        **dmargin,
    )
    dax = {}

    # --------------
    # prepare axes
    # --------------

    # --------------
    # ax - isolines

    ax = fig.add_subplot(gs[:nv0, nh0:nh0+nh1], xscale='log')
    ax.set_title(
        r"$d^2\sigma(E_{e0}, E_{ph}, \theta_{ph}, Z)$"
        + f"\n Z = {Z}, version = {version}",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['map'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # ax - responsivity

    ax = fig.add_subplot(gs[:nv0, :nh0], sharey=dax['map']['handle'])
    ax.set_title(
        "responsivity",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        r"$E_{ph}$ (keV)",
        size=fontsize,
        fontweight='bold',
    )
    ax.grid(True)

    # store
    dax['responsivity'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # ax - dist

    ax = fig.add_subplot(gs[nv0:, nh0:nh0 + nh1], sharex=dax['map']['handle'])
    ax.set_xlabel(
        r"$E_{e,0}$ (keV)",
        size=fontsize,
        fontweight='bold',
    )
    ax.grid(True)

    # store
    dax['dist'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # ax - theta

    ax = fig.add_subplot(gs[:nv2, nh0 + nh1 + 1:])
    ax.set_xlabel(
        r"$\theta_{ph}^B$ (deg)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        "emiss (ph/sr/s/m3)",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['theta'] = {'handle': ax, 'type': 'isolines'}

    return dax
