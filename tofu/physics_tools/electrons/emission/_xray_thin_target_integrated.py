

import os
import copy
import warnings
from typing import Optional   # Dict, Any


import numpy as np
import scipy.integrate as scpinteg
import astropy.units as asunits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datastock as ds


from . import _xray_thin_target


TupleDict = tuple[dict]


# ####################################################
# ####################################################
#           DEFAULT
# ####################################################


_PATH_HERE = os.path.dirname(__file__)


_E_E0_EV = 45e3
_E_PH_EV = 40e3
_THETA_PH = np.linspace(0, np.pi, 31)


# Integration
_NTHETAE = 91
_NDPHI = 181

# VERSION
_VERSION = 'EH'   # best with cfsem.hyp2f1()


# Default naming
_DSCALE = {
    12: 'T',
    9: 'G',
    6: 'M',
    3: 'k',
    0: '',
    -3: 'm',
    -6: 'u',
    -9: 'n',
}


_DFORMAT = {
    # energies
    'E_e0': {
        'data': None,
        'units': 'eV',
    },
    'E_ph': {
        'data': None,
        'units': 'eV',
    },
    # angles
    'theta_ph': {
        'data': None,
        'units': 'rad',
    },
    'theta_e': {
        'data': None,
        'units': 'rad',
    },
    'dphi': {
        'data': None,
        'units': 'rad',
    },
}


# ####################################################
# ####################################################
#        main
# ####################################################


def get_xray_thin_d2cross_ei_integrated_thetae_dphi(
    # optional input d2cross file
    d2cross: Optional[str | dict] = None,
    # target ion charge
    Z: Optional[int] = None,
    # energies
    E_e0_eV=None,
    E_ph_eV=None,
    theta_ph=None,
    # hypergeometric parameter
    ninf: Optional[int] = None,
    source: Optional[str] = None,
    # integration parameters
    nthetae=None,
    ndphi=None,
    # output customization
    per_energy_unit: Optional[str] = None,
    # version
    version: Optional[str] = None,
    # verb
    verb: Optional[bool] = None,
    verb_tab: Optional[str] = None,
    # saving
    save: Optional[bool] = None,
    pfe_save: Optional[str] = None,
    overwrite: Optional[bool] = None,
) -> dict:
    """ Compute d2cross, which is d3cross integrated over dphi

    Optionally loads / checks formatting of a pre-existing d2cross
        - d2cross = str, should be a pfe to a local .npz
        - d2cross = dict, will use as-is

    """

    # ------------
    # inputs
    # ------------

    (
        E_e0_eV, E_ph_eV, theta_ph,
        nthetae, ndphi,
        version,
        shape,
        verb, verb_tab,
        save, pfe_save, overwrite,
    ) = _check(**locals())

    # ------------------
    # compute
    # ------------------

    if d2cross is None:

        d2cross = _compute(**locals())

        # optional save
        if save is True:
            _save(
                d2cross,
                pfe_save=pfe_save,
                overwrite=overwrite,
                verb=verb,
            )

    # ------------------
    # load
    # ------------------

    else:

        d2cross = _load(d2cross)

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
    version=None,
    # verb
    verb=None,
    verb_tab=None,
    # saving
    save=None,
    pfe_save=None,
    overwrite=None,
    # unused
    **kwdargs,
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

    # ------------
    # version
    # ------------

    if version is None:
        version = _VERSION
    if isinstance(version, str):
        version = [version]

    version = ds._generic_check._check_var_iter(
        version, 'version',
        types=(list, tuple),
        types_iter=str,
        allowed=['EH', 'BH', 'BHE'],
    )

    # -----------
    # verb
    # -----------

    lok = [False, True, 0, 1, 2]
    verb = int(ds._generic_check._check_var(
        verb, 'verb',
        types=(int, bool),
        default=lok[-1],
        allowed=lok,
    ))

    # -----------
    # verb_tab
    # -----------

    verb_tab = ds._generic_check._check_var(
        verb_tab, 'verb_tab',
        types=int,
        default=0,
        sign='>=0',
    )
    verb_tab = '\t'*verb_tab

    # -----------
    # save
    # -----------

    savedef = pfe_save not in [None, False]
    save = ds._generic_check._check_var(
        save, 'save',
        types=bool,
        default=savedef,
    )

    # -----------
    # pfe_save
    # -----------

    if save is True:
        if pfe_save is not None:
            pfe_save = int(ds._generic_check._check_var(
                pfe_save, 'pfe_save',
                types=str,
            ))

            try:
                pfe_save = os.path.abspath(pfe_save)
            except Exception as err:
                msg = (
                    "Arg 'pfe_save' must point to an valid path/file.ext\n"
                    f"Provided: {pfe_save}\n"
                )
                raise Exception(msg) from err

            if not pfe_save.endswith('.npz'):
                msg = (
                    "Arg 'pfe_save' must point to an valid path/file.npz\n"
                    f"Provided: {pfe_save}\n"
                )
                raise Exception(msg)

    else:
        pfe_save = None

    # -----------
    # overwrite
    # -----------

    overwrite = ds._generic_check._check_var(
        overwrite, 'overwrite',
        types=bool,
        default=False,
    )

    return (
        E_e0_eV, E_ph_eV, theta_ph,
        nthetae, ndphi,
        version,
        shape,
        verb, verb_tab,
        save, pfe_save, overwrite,
    )


# ####################################################
# ####################################################
#        compute
# ####################################################


def _compute(
    Z=None,
    E_e0_eV=None,
    E_e1_eV=None,
    E_ph_eV=None,
    theta_ph=None,
    # shapes
    nthetae=None,
    ndphi=None,
    # parameters
    ninf=None,
    source=None,
    version=None,
    per_energy_unit=None,
    # misc
    shape=None,
    daxis=None,
    verb=None,
    verb_tab=None,
    # unused
    **kwdargs,
):

    # ------------------
    # get angles and shape
    # ------------------

    # E_e1_eV
    E_e1_eV = E_e0_eV - E_ph_eV

    # angles
    theta_e = np.pi * np.linspace(0, 1, nthetae)
    dphi = np.pi * np.linspace(-1, 1, ndphi)

    # ---------------------
    # determine how to loop
    # ---------------------

    # loop on largest dimension
    iloop = np.argmax(shape)
    sli = np.array([slice(None)]*len(shape) + [None, None])
    sli_Ee0 = np.copy(sli)
    sli_Ee1 = np.copy(sli)
    sli_theta = np.copy(sli)
    slistr = [':'] * (len(shape) + 2)

    sli_ang = (None,) * (len(shape) - 1) + (slice(None),)*2
    theta_ef = theta_e[None, :][sli_ang]
    dphif = dphi[:, None][sli_ang]

    # derived
    sinte = np.sin(theta_ef)

    # ------------------
    # prepare output
    # ------------------

    d2cross = copy.deepcopy(_DFORMAT)
    for kk, vv in _DFORMAT.items():
        kv = f"{kk}_{vv['units']}" if vv['units'] == 'eV' else kk
        d2cross[kk]['data'] = eval(kv)

    # cross-sections
    d2cross['cross'] = {vv: {'data': np.full(shape, 0.)} for vv in version}
    srunits = asunits.Unit('sr')

    # ------------------
    # get d3cross
    # ------------------

    if verb >= 1:
        msg = f"{verb_tab}Computing d3cross for shape {shape}... "
        print(msg)

    # -------------------------------
    # loop on all but phi and theta_e

    size = shape[iloop]
    for ii in range(size):

        # sli
        sli[iloop] = ii
        sli_Ee0[iloop] = min(ii, E_e0_eV.shape[iloop]-1)
        sli_Ee1[iloop] = min(ii, E_e1_eV.shape[iloop]-1)
        sli_theta[iloop] = min(ii, theta_ph.shape[iloop]-1)

        # verb
        if verb >= 2:
            slistr[iloop] = str(ii)
            str0 = ', '.join(slistr)
            str1 = str(shape + (None, None))
            end = '\n' if ii == size - 1 else '\r'
            msg = f"\t{ii+1} / {size}, index ({str0}) / {str1}"
            print(msg, end=end)

        # d3cross
        d3cross = _xray_thin_target.get_xray_thin_d3cross_ei(
            # inputs
            Z=Z,
            E_e0_eV=E_e0_eV[tuple(sli_Ee0)],
            E_e1_eV=E_e1_eV[tuple(sli_Ee1)],
            # directions
            theta_ph=theta_ph[tuple(sli_theta)],
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

        # integrate of theta_e
        for vv, vcross in d3cross['cross'].items():
            d2cross['cross'][vv]['data'][tuple(sli[:-2])] = scpinteg.trapezoid(
                scpinteg.trapezoid(
                    vcross['data'] * sinte,
                    x=theta_e,
                    axis=-1,
                ),
                x=dphi,
                axis=-1,
            )
            d2cross['cross'][vv]['units'] = (
                srunits * asunits.Unit(vcross['units'])
            )

        # Add Z
        d2cross['Z'] = d3cross['Z']

    return d2cross


# ####################################################
# ####################################################
#        save
# ####################################################


def _save(
    d2cross=None,
    pfe_save=None,
    overwrite=None,
    verb=None,
):

    # ----------
    # pfe_save
    # ----------

    if pfe_save is None:

        # extract sizes
        ntheta = d2cross['theta_ph']['data'].size
        Eph = _format_vect2str(
            d2cross['E_ph']['data'],
            base=d2cross['E_ph']['units'],
        )
        Ee0 = _format_vect2str(
            d2cross['E_e0']['data'],
            base=d2cross['E_e0']['units'],
        )

        # extract versions
        versions = '-'.join(sorted(d2cross['cross'].keys()))

        # extract boundaries
        path = os.path.abspath(_PATH_HERE)
        fname = f"d2cross_Ee0{Ee0}_Eph{Eph}_ntheta{ntheta}_{versions}"
        pfe_save = os.path.join(path, f"{fname}.npz")

    # ----------
    # overwrite
    # ----------

    if os.path.isfile(pfe_save):
        if overwrite is True:
            if verb is True:
                msg = f"Overwritting file {pfe_save}\n"
                warnings.warn(msg)
        else:
            msg = (
                "File {pfe_save} already exists!\n"
                "\t=> use overwrite=True to overwrite\n"
            )
            raise Exception(msg)

    # ----------
    # save
    # ----------

    np.savez(pfe_save, **d2cross)

    # ----------
    # verb
    # ----------

    if verb >= 1:
        msg = f"Saved d2cross in:\n\t{pfe_save}\n"
        print(msg)

    return


# ####################################################
# ####################################################
#        Built str vector
# ####################################################


def _format_vect2str(vect, base='eV'):

    # -------------
    # extract
    # -------------

    v0 = vect.min()
    v1 = vect.max()
    nv = vect.size

    # -------------
    # test scale
    # -------------

    if np.max(vect.shape) == vect.size:
        dlog = np.diff(np.log(vect))
        dlin = np.diff(vect)
        islog = np.allclose(dlog, dlog[0])
        islin = np.allclose(dlin, dlin[0])

        if islog is True:
            scale = 'log'
        elif islin is True:
            scale = 'lin'
        else:
            scale = 'pts'
    else:
        scale = 'pts'

    # -------------
    # format
    # -------------

    v0 = _format(v0, base=base)
    v1 = _format(v1, base=base)

    return f"{v0}-{v1}-{nv}{scale}"


def _format(vv, base='eV'):

    ls = sorted(_DSCALE.keys())
    ind = np.searchsorted(ls, np.log10(vv), side='right') - 1
    key = ls[ind]
    factor = 10**(-key)

    return f"{vv*factor:3.0f}{_DSCALE[key]}{base}".strip()


# ####################################################
# ####################################################
#           load
# ####################################################


def _load(
    d2cross=None,
):

    # ---------
    # str
    # ---------

    if isinstance(d2cross, str):

        if not (os.path.isfile(d2cross) and d2cross.endswith('.npz')):
            msg = (
                "Arg 'd2cross', if a str, should be valid path/file.npz\n"
                f"Provided: {d2cross}\n"
            )
            raise Exception(msg)

        d2cross = {
            kk: vv.tolist()
            for kk, vv in np.load(d2cross, allow_pickle=True).items()
        }

    # ---------
    # dict
    # ---------

    if not isinstance(d2cross, dict):
        msg = "Arg 'd2cross' must be a dict!\nProvided: {type(d2cross)}\n"
        raise Exception(msg)

    # ----------------
    # inner formatting
    # ----------------

    dfail = {}
    for kk in _DFORMAT.keys():
        if not isinstance(d2cross.get(kk), dict):
            dfail[kk] = 'not a key or value not a dict ()'
        elif str(d2cross[kk].get('units')) != _DFORMAT[kk]['units']:
            dfail[kk] = 'no or wrong units ()'
        elif not isinstance(d2cross[kk]['data'], np.ndarray):
            dfail[kk] = "data is not a np.ndarray (type(d2cross[kk]['data']))"
        else:
            dfail[kk] = 'ok'

    if any([vv != 'ok' for vv in dfail.values()]) > 0:
        lstr = [f"\t- {kk}: {vv}" for kk, vv in dfail.items()]
        msg = (
            "Arg 'd2cross' must be a dict of subdicts "
            "{'data': np.ndarray, 'units': str}, with keys:\n"
            + "\n".join(lstr)
        )
        raise Exception(msg)

    # ----------------
    # cross formatting
    # ----------------

    lok = ['BHE', 'BH', 'EH']
    typunits = (str, asunits.Unit, asunits.CompositeUnit)
    c0 = (
        isinstance(d2cross.get('cross'), dict)
        and all([
            kk in lok
            and isinstance(vv, dict)
            and isinstance(vv.get('data'), np.ndarray)
            and isinstance(vv.get('units'), typunits)
            for kk, vv in d2cross['cross'].items()
        ])
    )
    if not c0:
        msg = (
            "Arg d2cross['cross'] must be a dict "
            "with {'version': dict} subdict with:\n"
            f"\t- 'version' in {lok}\n"
            f"Provided:\n{d2cross.get('cross')}\n"
        )
        raise Exception(msg)

    return d2cross


# ####################################################
# ####################################################
#        plot vs litterature
# ####################################################


def plot_xray_thin_d2cross_ei_vs_literature() -> TupleDict:
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
                lw=3 if v0 == 'EH' else 1.5,
                alpha=0.5,
                label=f'computed - {k0} Z = {Z}',
            )

        # Z = 8
        Z = 8
        for k0, v0 in d2cross_fig12_Z8['cross'].items():
            ax.plot(
                theta_ph * 180/np.pi,
                v0['data']*1e28*1e3 * 40e3 / Z**2,
                ls='-',
                lw=3 if v0 == 'EH' else 1.5,
                alpha=0.5,
                label=f'computed - {k0} Z = {Z}',
            )

        ax.set_xlim(0, 180)
        ax.set_ylim(0, 8)

        # add legend
        ax.legend()

    # ------------------------
    # plot photon distribution
    # ------------------------

    return dax, d2cross_fig12_Z13, d2cross_fig12_Z8
