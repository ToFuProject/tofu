

import copy
from typing import Optional   # Any, Dict


import numpy as np
import astropy.units as asunits
import scipy.constants as scpct
import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import datastock as ds


# from . import _xray_thin_target_integrated as _mod
from . import _xray_thin_target_integrated_plot as _mod_plot
from ... import transmission
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


# DSCALES
_DSCALES = {
    'E_ph': 'log',
    'E_e0': 'log',
    'theta': 'linear',
    'dist': 'log',
    'resp': 'log',
}


# DDIST
_UNITS = (str, asunits.Unit, asunits.CompositeUnit)
_DDIST = {}
_DDIST_FORMAT = {
    'E_e0': {'data': np.ndarray, 'units': _UNITS},
    'dist': {'data': np.ndarray, 'units': _UNITS},
    'marker': 'None',
    'lw': 1,
    'color': 'k',
    'ls': None,     # cycle
}


# DRESP
_DRESP = {}
_DRESP_FORMAT = {
    'E_ph': {'data': np.ndarray, 'units': _UNITS},
    'responsivity': {'data': np.ndarray, 'units': _UNITS},
    'marker': 'None',
    'lw': 1,
    'ls': '-',
    'color': None,    # cycle
}


# DCASES
_DCASES = {}
_DCASE_FORMAT = {
    'dist': str,
    'resp': str,
}


# -----------
# DTRANS

_DTRANS = {
    'Al\n10 um': {
        'mat': 'Al',
        'thick': 10e-6,
    },
    'Steel\n1 cm': {
        'mat': 'StainlessSteel',
        'thick': 0.01,
    },
}

_DTRANS_DEF = {
    'fontsize': 12,
    'fontweight': 'bold',
    'ls': '--',
    'lw': 1,
    'color': 'k',
}

# -----------
# DRANGES

_DRANGES = {
    'visible': {
        'E': np.sort(scpct.h * scpct.c / (np.r_[380, 750]*1e-9) / scpct.e),
    },
    'UV': {
        'E': np.sort(np.r_[
            scpct.h * scpct.c / (350*1e-9) / scpct.e,
            30e15 * scpct.h / scpct.e,
        ]),
    },
}

_DRANGES_DEF = {
    'fontsize': 12,
    'fontweight': 'bold',
    'color': 'k',
    'facecolor': (0.8, 0.8, 0.8),
    'alpha': 0.5,
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
    dcases_cross: Optional[dict[int, dict]] = None,
    # distributions, responsivities and cases
    ddist: Optional[dict] = None,
    dresp: Optional[dict] = None,
    dcases_dist_resp: Optional[dict[int, dict]] = None,
    # decorative
    dtrans: Optional[dict] = None,
    dranges: Optional[dict] = None,
    # verb
    verb: Optional[bool] = None,
    # plot
    dax: Optional[dict] = None,
    fs: Optional[tuple] = None,
    fontsize: Optional[int] = None,
    E_e0_scale: Optional[str] = None,
    E_ph_scale: Optional[str] = None,
    dist_scale: Optional[str] = None,
    resp_scale: Optional[str] = None,
    theta_scale: Optional[str] = None,
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

    (
        ddist, dresp,
        dcases_dist_resp,
        dscales,
        fs, fontsize,
    ) = _check(**locals())

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
            dscales=dscales,
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
        dcases=dcases_cross,
        # verb
        verb=verb,
        # plot
        dax=dax,
        dplot_forbidden=dplot_forbidden,
        dplot_peaking=dplot_peaking,
        dplot_thetamax=dplot_thetamax,
        dplot_mean=dplot_mean,
    )

    # -------------------
    # Compute integrand
    # -------------------

    dinteg = _integrand(
        d2cross=d2cross_cross,
        dcases_dist_resp=dcases_dist_resp,
    )

    # -----------
    # decorative
    # -----------

    # transmissions
    dtrans = _dtrans(
        d2cross=d2cross_cross,
        dtrans=dtrans,
    )

    # dranges
    dranges = _dranges(
        d2cross=d2cross_cross,
        dranges=dranges,
    )

    # -------------------
    # plot responsivity
    # -------------------

    kax = 'responsivity'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for kk, vv in dresp.items():

            l0, = ax.plot(
                vv['responsivity']['data'],
                vv['E_ph']['data']*1e-3,
                c=vv.get('color'),
                ls=vv.get('ls', '-'),
                marker=vv.get('marker'),
                lw=vv.get('lw', 1.),
                label=f"{kk}_{vv['responsivity']['units']}",
            )
            dresp[kk]['color'] = l0.get_color()

        ax.invert_xaxis()

        # legend
        ax.legend(
            bbox_to_anchor=(0, -0.1),
            loc='upper left',
            borderaxespad=0,
        )

    # -------------------
    # plot comments
    # -------------------

    kax = 'comments'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        # ------------
        # transmission

        for ktrans, vtrans in dtrans.items():

            ax.axhline(
                vtrans['E']*1e-3,
                color=vtrans['color'],
                ls=vtrans['ls'],
                lw=vtrans['lw'],
            )

            ax.text(
                0.5,
                vtrans['E']*1e-3,
                ktrans,
                fontweight=vtrans['fontweight'],
                fontsize=vtrans['fontsize'],
                verticalalignment='center',
                horizontalalignment='center',
                color=vtrans['color'],
            )

        # ------------
        # ranges

        for krang, vrang in dranges.items():

            ax.axhspan(
                vrang['E'][0]*1e-3,
                vrang['E'][1]*1e-3,
                facecolor=vrang['facecolor'],
                alpha=vrang['alpha']
            )

            ax.text(
                0.5,
                np.sqrt(vrang['E'][0] * vrang['E'][1])*1e-3,
                krang,
                fontweight=vtrans['fontweight'],
                fontsize=vtrans['fontsize'],
                verticalalignment='center',
                horizontalalignment='center',
                color=vtrans['color'],
            )

    # -------------------
    # plot distribution
    # -------------------

    kax = 'dist'
    if dax.get(kax) is not None:
        ax = dax[kax]['handle']

        for kk, vv in ddist.items():

            l0, = ax.semilogy(
                vv['E_e0']['data']*1e-3,
                vv['dist']['data'],
                c=vv['color'],
                ls=vv.get('ls', '-'),
                marker=vv.get('marker'),
                lw=vv.get('lw', 1.),
                label=kk,
            )
            ddist[kk]['color'] = l0.get_color()

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
    # distributions, responsivity, cases
    ddist=None,
    dresp=None,
    dcases_dist_resp=None,
    # plotting
    fs=None,
    fontsize=None,
    E_e0_scale=None,
    E_ph_scale=None,
    dist_scale=None,
    resp_scale=None,
    theta_scale=None,
    # unused
    **kwdargs,
):

    # ------------
    # dresp
    # ------------

    if dresp is None:
        dresp = _DRESP

    if dresp is False:
        dresp = {}

    dresp = _check_dict(
        din=dresp,
        din_name='dresp',
        ddef=_DRESP_FORMAT,
    )

    # ------------
    # ddist
    # ------------

    if ddist is None:
        ddist = _DDIST

    if ddist is False:
        ddist = {}

    ddist = _check_dict(
        din=ddist,
        din_name='ddist',
        ddef=_DDIST_FORMAT,
    )

    # ------------
    # dcases
    # ------------

    dcases_dist_resp = _check_dcases(
        dresp=dresp,
        ddist=ddist,
        dcases=dcases_dist_resp,
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

    # ------------
    # scales
    # ------------

    dscales = {
        'E_ph': E_ph_scale,
        'E_e0': E_e0_scale,
        'theta': theta_scale,
        'dist': dist_scale,
        'resp': resp_scale,
    }

    for kk, vv in dscales.items():
        dscales[kk] = ds._generic_check._check_var(
            vv, f'{kk}_scale',
            types=str,
            allowed=['log', 'linear'],
            default=_DSCALES[kk],
        )

    return (
        ddist, dresp,
        dcases_dist_resp,
        dscales,
        fs, fontsize,
    )


def _check_dict(
    din=None,
    din_name=None,
    ddef=None,
):

    # --------------
    # overall structure
    # --------------

    c0 = (
        isinstance(din, dict)
        and all([
            isinstance(kk, str)
            and isinstance(vv, dict)
            for kk, vv in din.items()
        ])
    )
    if not c0:
        msg = (
            f"Arg '{din_name}' must be a dict of sub-dicts!\n"
            f"Provided:\n{din}\n"
        )
        raise Exception(msg)

    # --------------
    # each key structure
    # --------------

    dfail = {}
    lok = [kk for kk, vv in ddef.items() if isinstance(vv, dict)]
    for k0, v0 in din.items():
        for kk in lok:
            if not isinstance(din[k0].get(kk), dict):
                typ = type(din[k0].get(kk))
                dfail[kk] = f'absent or not a dict ({typ})'
            elif not isinstance(din[k0][kk].get('data'), ddef[kk]['data']):
                typ = type(din[k0][kk].get('data'))
                dfail[kk] = f'data not a np.ndarray ({typ})'
            elif not isinstance(din[k0][kk].get('units'), ddef[kk]['units']):
                typ = type(din[k0][kk].get('units'))
                dfail[kk] = f"units not a str ({typ})"
            else:
                dfail[kk] = 'ok'

        if any([vv != 'ok' for vv in dfail.values()]):
            lstr = [f"\t- {kk}: {vv}" for kk, vv in dfail.items()]
            msg = (
                f"Arg {din_name}['{k0}'] must be a dict with keys {lok}, "
                "where each is {'data': np.ndarray, 'units': str} subdict!\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

    # --------------
    # shapes: all flat
    # --------------

    dfail = {}
    for k0, v0 in din.items():

        # squeeze
        for kk in lok:

            shape = v0[kk]['data'].shape
            if np.prod(shape) != np.max(shape):
                dfail[kk] = (
                    'data must be squeeze-able to a flat 1d array!'
                    f'  (shape = {shape})'
                )
                continue

            din[k0][kk]['data'] = v0[kk]['data'].squeeze()

        # consistency
        lsize = list(set([din[k0][kk]['data'].size for kk in lok]))
        if len(lsize) != 1:
            msg = (
                "All keys in {din_name}['{k0}'] must have the same data shape"
            )
            raise Exception(msg)

    # ------------
    # units
    # ------------

    dfail = {}
    for k0, v0 in din.items():
        for kk in lok:

            if not kk.startswith('E_'):
                continue

            if str(v0[kk]['units']) != 'eV':
                dfail[kk] = f"units should be eV ({v0[kk]['units']})"

        if len(dfail) > 0:
            lstr = [
                f"\t- {din_name}['{k0}']['{kk}']: {vv}"
                for kk, vv in dfail.items()
            ]
            msg = (
                "Units of the following keys is incorrect:\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

    # --------------
    # plotting
    # --------------

    lok = [kk for kk, vv in ddef.items() if not isinstance(vv, dict)]
    for k0, v0 in din.items():
        for kk in lok:
            din[k0][kk] = din[k0].get(kk, ddef[kk])

    return din


def _check_dcases(
    dcases=None,
    dresp=None,
    ddist=None,
):

    # --------------
    # defaults
    # --------------

    if dcases is False:
        dcases = {}

    if dcases is None:
        dcases = {}
        for kdist in ddist.keys():
            for kresp in dresp.keys():
                key = f"{kresp}_{kdist}"
                dcases[key] = {
                    'resp': kresp,
                    'dist': kdist,
                }

    # --------------
    # general structure
    # --------------

    c0 = (
        isinstance(dcases, dict)
        and all([
            isinstance(kcase, str)
            and (
                isinstance(vcase.get('resp'), str)
                and vcase['resp'] in dresp.keys()
            )
            and (
                isinstance(vcase.get('dist'), str)
                and vcase['dist'] in ddist.keys()
            )
            for kcase, vcase in dcases.items()
        ])
    )
    if not c0:
        msg = (
            "Arg dcase must be a dict of subdicts of the form "
            "{'dist': key0, 'resp': key1}\n"
            "Where key0 (resp. key1) refer to an existing key in "
            "ddist (resp. dresp)\n"
            f"Provided: {dcases}\n"
        )
        raise Exception(msg)

    return dcases


# #############################################
# #############################################
#        Integrand
# #############################################


def _integrand(
    d2cross=None,
    dcases_dist_resp=None,
):

    # --------------
    # loop on cases
    # --------------

    for kcase, vcase in dcases_dist_resp.items():

        pass

    return


# #############################################
# #############################################
#           Decorative
# #############################################


def _dtrans(
    dtrans=None,
    d2cross=None,
):

    # ----------------
    # default
    # ----------------

    # default
    if dtrans is None:
        dtrans = copy.deepcopy(_DTRANS)

    # ----------------
    # get transmission
    # ----------------

    # False
    if dtrans is False:
        dtrans = {}

    else:
        dout = transmission.get_xray_transmission(
            dthick=dtrans,
            E=d2cross['E_ph']['data'],
            plot=False,
        )

        # extract info
        E_inv = d2cross['E_ph']['data'].ravel()[::-1]
        for kk, vv in dtrans.items():

            # get last time > 0.5
            trans = dout['keys'][kk]['trans'].ravel()[::-1]
            ii = np.nonzero(trans < 0.5)[0][0]
            dtrans[kk]['E'] = E_inv[ii]

    # ----------------
    # check format
    # ----------------

    for ktrans, vtrans in dtrans.items():

        for kk, vv in _DTRANS_DEF.items():
            dtrans[ktrans][kk] = vtrans.get(kk, vv)

    return dtrans


def _dranges(
    dranges=None,
    d2cross=None,
):

    # ----------------
    # default
    # ----------------

    # default
    if dranges is None:
        dranges = copy.deepcopy(_DRANGES)

    # ----------------
    # get transmission
    # ----------------

    # False
    if dranges is False:
        dranges = {}

    # ----------------
    # check format
    # ----------------

    for krang, vrang in dranges.items():

        for kk, vv in _DRANGES_DEF.items():
            dranges[krang][kk] = vrang.get(kk, vv)

    return dranges


# #############################################
# #############################################
#        Axes for anisotropy
# #############################################


def _dax(
    Z=None,
    version=None,
    fs=None,
    fontsize=None,
    dscales=None,
):

    # --------------
    # prepare figure
    # --------------

    tit = (
        "Thin-target Bremsstrahlung emission anisotropy"
    )

    dmargin = {
        'left': 0.05, 'right': 0.98,
        'bottom': 0.05, 'top': 0.92,
        'wspace': 0.70, 'hspace': 0.20,
    }

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(tit, size=fontsize+2, fontweight='bold')

    nhvert = 3
    nhcom = 1
    nhlarge = 8
    nhint = 1
    nhtheta = 5
    nv0, nv1, nv2 = 3, 1, 2
    gs = gridspec.GridSpec(
        ncols=nhvert + nhcom + nhlarge + nhint + nhtheta + nhtheta,
        nrows=nv0 + nv1,
        **dmargin,
    )
    dax = {}

    # --------------
    # prepare axes
    # --------------

    # --------------
    # ax - isolines

    nh = nhvert + nhcom
    ax = fig.add_subplot(
        gs[:nv0, nh:nh + nhlarge],
        xscale=dscales['E_e0'],
        yscale=dscales['E_ph'],
    )
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

    ax = fig.add_subplot(
        gs[:nv0, :nhvert],
        sharey=dax['map']['handle'],
        xscale=dscales['resp'],
    )
    ax.set_title(
        "responsivity",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_xlabel(
        'responsivity',
        fontsize=fontsize,
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
    # ax - comments

    ax = fig.add_subplot(
        gs[:nv0, nhvert:nhvert+nhcom],
        sharey=dax['map']['handle'],
        yscale=dscales['resp'],
        frameon=False,
    )
    ax.axis('off')

    # store
    dax['comments'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # ax - dist

    nh = nhvert + nhcom
    ax = fig.add_subplot(
        gs[nv0:, nh:nh + nhlarge],
        sharex=dax['map']['handle'],
        yscale=dscales['dist'],
    )
    ax.set_xlabel(
        r"$E_{e,0}$ (keV)",
        size=fontsize,
        fontweight='bold',
    )
    ax.grid(True)

    # store
    dax['dist'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # ax - theta_cross - norm

    nh = nhvert + nhcom + nhlarge + nhint
    ax = fig.add_subplot(
        gs[:nv2, nh:nh + nhtheta],
        xscale=dscales['theta'],
        yscale='linear',
    )
    ax.set_xlabel(
        r"$\theta_{ph}^{e0}$ (deg)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        "cross-section norm.",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['theta_norm'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # ax - theta_cross - abs

    nh = nhvert + nhcom + nhlarge + nhint
    ax = fig.add_subplot(
        gs[nv2:, nh:nh + nhtheta],
        sharex=dax['theta_norm']['handle'],
        yscale='log',
    )
    ax.set_xlabel(
        r"$\theta_{ph}^{e0}$ (deg)",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['theta_abs'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # ax - theta_emiss - norm

    nh = nhvert + nhcom + nhlarge + nhint + nhtheta
    ax = fig.add_subplot(
        gs[:nv2, nh:],
        xscale=dscales['theta'],
        yscale='linear',
    )
    ax.set_xlabel(
        r"$\theta_{ph}^B$ (deg)",
        size=fontsize,
        fontweight='bold',
    )
    ax.set_ylabel(
        "emiss norm.",
        size=fontsize,
        fontweight='bold',
    )

    # store
    dax['theta_emiss_norm'] = {'handle': ax, 'type': 'isolines'}

    # --------------
    # ax - theta_emiss - abs

    nh = nhvert + nhcom + nhlarge + nhint + nhtheta
    ax = fig.add_subplot(
        gs[nv2:, nh:],
        sharex=dax['theta_norm']['handle'],
        yscale='log',
    )
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
    dax['theta_emiss_abs'] = {'handle': ax, 'type': 'isolines'}

    return dax
