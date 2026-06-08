

import os
import warnings


import numpy as np
import scipy.integrate as scpinteg
import scipy.interpolate as scpinterp
import astropy.units as asunits
import datastock as ds


from . import _xray_thin_target_integrated as _mod


# ############################################
# ############################################
#             Default
# ############################################


_PATH_HERE = os.path.dirname(__file__)


_THETA_PH_VSB = np.linspace(0, np.pi, 37)
_THETA_E0_VSB_NPTS = 31
_E_PH_EV = np.r_[
    np.logspace(np.log10(100), np.log10(50e3), 51),
]
_E_E0_EV_NPTS = 61


# ###########################################
# ###########################################
#        get d2cross_phi
# ###########################################


def get_d2cross_phi(
    # load from file
    d2cross_phi=None,
    # tabulated d2cross
    d2cross=None,
    # params
    Z=None,
    E_ph_eV=None,
    E_e0_eV=None,
    E_e0_eV_npts=None,
    theta_ph_vsB=None,
    theta_e0_vsB_npts=None,
    phi_e0_vsB_npts=None,
    # hypergeometric parameter
    ninf=None,
    source=None,
    # integration parameters
    nthetae=None,
    ndphi=None,
    # iok
    iok=None,
    # version
    version_cross=None,
    # verb
    verb=None,
    # load / save
    save=None,
    pfe_save=None,
    overwrite=None,
    # unused
    **kwdargs,
):
    """ Integrates d2cross over phi_e0_vsB

    Intermediate between d2cross and integration over an electron distribution

    """

    # ----------------
    # inputs
    # ----------------

    (
        save, overwrite, d2cross_phi, verb,
    ) = _check(
        d2cross_phi=d2cross_phi,
        save=save,
        overwrite=overwrite,
        pfe_save=pfe_save,
        verb=verb,
    )

    # ----------------
    # compute
    # ----------------

    if d2cross_phi is None:

        # check compute
        (
            E_ph_eV, E_e0_eV, iok,
            theta_e0_vsB, theta_ph_vsB,
            phi_e0_vsB,
            version_cross,
            pfe_save,
        ) = _check_compute(**locals())

        # compute
        d2cross_phi = _compute(**locals())

        # optional save
        if save is True:
            _save(
                d2cross_phi=d2cross_phi,
                pfe_save=pfe_save,
                overwrite=overwrite,
                verb=verb,
            )

    # ----------------
    # load
    # ----------------

    elif isinstance(d2cross_phi, str):
        d2cross_phi = _load(pfe=d2cross_phi, **locals())

    return d2cross_phi


# ###########################################
# ###########################################
#        check
# ###########################################


def _check(
    d2cross_phi=None,
    save=None,
    overwrite=None,
    pfe_save=None,
    verb=None,
):

    # -------------
    # save
    # -------------

    # save
    save = ds._generic_check._check_var(
        save, 'save',
        types=bool,
        default=pfe_save is not None,
    )

    # -------------
    # overwrite
    # -------------

    # overwrite
    overwrite = ds._generic_check._check_var(
        overwrite, 'overwrite',
        types=bool,
        default=False,
    )

    # -------------
    # pfe
    # -------------

    if d2cross_phi is not None and not isinstance(d2cross_phi, dict):

        dc = {
            'str': isinstance(d2cross_phi, str),
            'file': (
                isinstance(d2cross_phi, str)
                and os.path.isfile(d2cross_phi)
            ),
            '.npz': (
                isinstance(d2cross_phi, str)
                and d2cross_phi.endswith('.npz')
            ),
        }
        if not all([vv for vv in dc.values()]):
            lstr = [f"\t- {kk}: {vv}" for kk, vv in dc.items()]
            msg = (
                "Arg pfe must be a valid path to a .npz file!\n"
                + "\n".join(lstr)
                + f"\nProvided: {d2cross_phi}\n"
            )
            raise Exception(msg)
        save = False

    # --------------------
    # verb
    # --------------------

    lok = [False, True, 0, 1, 2, 3]
    verb = int(ds._generic_check._check_var(
        verb, 'verb',
        types=(int, bool),
        default=lok[-1],
        allowed=lok,
    ))

    return (
        save, overwrite, d2cross_phi, verb,
    )


# ###########################################
# ###########################################
#        check_compute
# ###########################################


def _check_compute(
    # tabulated d2cross
    d2cross=None,
    # params
    E_ph_eV=None,
    E_e0_eV=None,
    E_e0_eV_npts=None,
    theta_e0_vsB_npts=None,
    phi_e0_vsB_npts=None,
    theta_ph_vsB=None,
    # version
    version_cross=None,
    # saving
    pfe_save=None,
    # unused
    **kwdargs,
):

    # ----------
    # E_ph_eV
    # ----------

    if E_ph_eV is None:
        if isinstance(d2cross, dict):
            E_ph_eV = d2cross['E_ph']['data']
        else:
            E_ph_eV = _E_PH_EV

    E_ph_eV = ds._generic_check._check_flat1darray(
        E_ph_eV, 'E_ph_eV',
        dtype=float,
        sign='>=0',
    )

    # ----------
    # E_e0_eV
    # ----------

    E_e0_eV_npts = int(ds._generic_check._check_var(
        E_e0_eV_npts, 'E_e0_eV_npts',
        types=(int, float),
        sign='>=3',
        default=_E_E0_EV_NPTS,
    ))

    if E_e0_eV is None:
        if isinstance(d2cross, dict):
            E_e0_eV = d2cross['E_e0']['data']
        else:
            E_e0_eV = np.logspace(
                np.log10(E_ph_eV.min()),
                np.ceil(np.log10(E_ph_eV.max())) + 2,
                E_e0_eV_npts,
            )

    E_e0_eV = np.unique(ds._generic_check._check_flat1darray(
        E_e0_eV, 'E_e0_eV',
        dtype=float,
        unique=True,
        sign='>=0',
    ))

    iok = E_e0_eV >= E_ph_eV.min()
    nok = np.sum(iok)
    if nok < E_e0_eV.size:
        if nok == 0:
            msg = (
                f"All points ({E_e0_eV.size}) "
                "are removed from E_e0_eV (< E_ph_eV.min())"
            )
            raise Exception(msg)

        else:
            msg = (
                f"Some points ({E_e0_eV.size - nok} / {E_e0_eV.size}) "
                "are removed from E_e0_eV (< E_ph_eV.min())"
            )
            warnings.warn(msg)

    if E_e0_eV.max() < E_ph_eV.max():
        msg = (
            "Arg E_e0_eV should not have a max value below E_ph_eV.min()!\n"
            f"\t- E_ph_eV.max() = {E_ph_eV.max()}\n"
            f"\t- E_e0_eV.max() = {E_e0_eV.max()}\n"
        )
        raise Exception(msg)

    # ------------
    # theta_ph_vsB
    # ------------

    if theta_ph_vsB is None:
        if isinstance(d2cross, dict):
            theta_ph_vsB = d2cross['theta_ph']['data']
        else:
            theta_ph_vsB = _THETA_PH_VSB

    theta_ph_vsB = ds._generic_check._check_flat1darray(
        theta_ph_vsB, 'theta_ph_vsB',
        dtype=float,
    )
    theta_ph_vsB = np.arctan2(np.sin(theta_ph_vsB), np.cos(theta_ph_vsB))
    iout = (theta_ph_vsB < 0.) | (theta_ph_vsB > np.pi)
    if np.any(iout):
        msg = (
            "Arg theta_ph_vsB must be within [0, pi]\n"
            f"Provided:\n{theta_ph_vsB}\n"
        )
        raise Exception(msg)

    # ------------
    # theta_e0_vsB
    # ------------

    if isinstance(d2cross, dict):
        npdef = d2cross['theta_e']['data'].size
    else:
        npdef = _THETA_E0_VSB_NPTS

    theta_e0_vsB_npts = int(ds._generic_check._check_var(
        theta_e0_vsB_npts, 'theta_e0_vsB_npts',
        types=(int, float),
        sign='>=3',
        default=npdef,
    ))
    theta_e0_vsB = np.linspace(0, np.pi, theta_e0_vsB_npts)

    # --------------------
    # phi_e0_vsB
    # --------------------

    phi_e0_vsB_npts = int(ds._generic_check._check_var(
        phi_e0_vsB_npts, 'phi_e0_vsB_npts',
        types=(int, float),
        sign='>=5',
        default=2*theta_e0_vsB_npts + 1,
    ))
    phi_e0_vsB = np.linspace(-np.pi, np.pi, phi_e0_vsB_npts)

    # --------------------
    # version_cross
    # --------------------

    version_cross = ds._generic_check._check_var(
        version_cross, 'version_cross',
        types=str,
        default='BHE',
    )

    # ----------
    # pfe
    # ----------

    if pfe_save is not None:
        c0 = (
            isinstance(pfe_save, str)
            and os.path.isdir(os.path.split(pfe_save)[0])
        )
        if not c0:
            msg = (
                "Arg pfe_save should be a path/file.ext with a valid path!\n"
                f"Provided: {pfe_save}\n"
            )
            raise Exception(msg)

        if not pfe_save.endswith('.npz'):
            pfe_save = f"{pfe_save}.npz"

    return (
        E_ph_eV, E_e0_eV, iok,
        theta_e0_vsB, theta_ph_vsB,
        phi_e0_vsB,
        version_cross,
        pfe_save,
    )


# ###########################################
# ###########################################
#        compute
# ###########################################


def _compute(
    # tabulated d2cross
    d2cross=None,
    # parameters
    E_ph_eV=None,
    E_e0_eV=None,
    theta_e0_vsB=None,
    phi_e0_vsB=None,
    theta_ph_vsB=None,
    iok=None,
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
    verb=None,
    # unused
    **kwdargs,
):

    # -----------
    # prepare
    # -----------

    # theta_ph_vs_e in (theta_ph_vsB, theta_e0_vsB, phi_e0_vsB)
    cos = (
        np.cos(theta_e0_vsB[None, :, None])
        * np.cos(theta_ph_vsB[:, None, None])
        + np.sin(theta_e0_vsB[None, :, None])
        * np.sin(theta_ph_vsB[:, None, None])
        * np.cos(phi_e0_vsB[None, None, :])
    )

    ieps = np.abs(cos) > 1.
    assert np.all(np.abs(cos[ieps]) - 1. < 1e-13)
    cos[ieps] = np.sign(cos[ieps])
    theta_ph_vs_e = np.arccos(cos)

    shape_emiss = (E_ph_eV.size, theta_ph_vsB.size)
    shape_integ = (E_e0_eV.size, theta_e0_vsB.size, phi_e0_vsB.size)

    d2cross_phi = np.zeros(shape_emiss + shape_integ[:-1], dtype=float)

    # -------------
    # load d2cross
    # -------------

    if d2cross is not None:
        d2cross0, nthetae, ndphi = _get_interpolator(d2cross)
    else:
        d2cross0 = None

    # -------------
    # loop on index
    # -------------

    for i0, ind in enumerate(np.ndindex(shape_emiss)):

        if verb >= 2:
            iEstr = f"({ind[0] + 1} / {shape_emiss[0]})"
            itstr = f"({ind[1] + 1} / {shape_emiss[1]})"
            ish = f"{iok.sum()} / {shape_integ[0]}"
            ish = f"({ish}, {shape_integ[1]}, {shape_integ[2]})"
            msg = f"\tE_ph_eV {iEstr}, theta_ph_vsB {itstr} for shape {ish}"
            print(msg)

        # -----------
        # get integrated cross-section
        # theta_ph_vs_e = (theta_ph_vsB, theta_e0_vsB, phi_e0_vsB)
        # d2cross = (E_ph_eV, E_e0_eV, theta_ph_vsB)
        #         = (E_ph_eV, E_e0_eV, theta_ph_vsB, theta_e0_vsB, phi_e0_vsB)
        if d2cross0 is None:
            d2cross = _mod.get_xray_thin_d2cross_ei_integrated_thetae_dphi(
                # tabulated d2cross
                d2cross=None,
                # inputs
                Z=Z,
                E_ph_eV=E_ph_eV[ind[0]],
                E_e0_eV=E_e0_eV[iok, None, None],
                theta_ph=theta_ph_vs_e[None, ind[1], :, :],
                # hypergeometric parameter
                ninf=ninf,
                source=source,
                # integration parameters
                nthetae=nthetae,
                ndphi=ndphi,
                # output customization
                per_energy_unit='eV',
                # version
                version=version_cross,
                # verb
                verb=verb > 2,
                verb_tab=2,
            )

            if i0 == 0:
                units = d2cross['cross'][version_cross]['units']
                nthetae = d2cross['theta_e']['data'].size
                ndphi = d2cross['dphi']['data'].size

            cross = d2cross['cross'][version_cross]['data']

        # -----------
        # interpolate
        else:

            xx = d2cross0[version_cross]['get_xx'](
                E_ph_eV=E_ph_eV[ind[0]],
                E_e0_eV=E_e0_eV[iok, None, None],
                theta_ph=theta_ph_vs_e[None, ind[1], :, :],
            )

            cross = d2cross0[version_cross]['interp'](xx)
            units = d2cross0[version_cross]['units']

        # -----------
        # integrate over phi
        # MULTIPLY BY SIN PHI ?????
        d2cross_phi[ind[0], ind[1], iok, :] = scpinteg.trapezoid(
            cross,
            x=phi_e0_vsB,
            axis=-1,
        )

    # ----------
    # units
    # ----------

    units *= asunits.Unit('rad')

    # -------------
    # format output
    # -------------

    dout = {
        'd2cross_phi': {
            'data': d2cross_phi,
            'units': units,
        },
        'E_e0_eV': E_e0_eV,
        'E_ph_eV': E_ph_eV,
        'theta_e0_vsB': theta_e0_vsB,
        'theta_ph_vsB': theta_ph_vsB,
        'phi_e0_vsB': phi_e0_vsB,
        'Z': Z,
        'nthetae': nthetae,
        'ndphi': ndphi,
        'version_cross': version_cross,
        'ninf': ninf,
        'source': source,
    }

    return dout


# ###########################################
# ###########################################
#        Interpolator
# ###########################################


def _get_interpolator(
    d2cross=None,
):

    # ---------------
    # load tabulated
    # ---------------

    d2cross0 = _mod.get_xray_thin_d2cross_ei_integrated_thetae_dphi(
        d2cross=d2cross,
    )

    # extract npts
    nthetae = d2cross0['theta_e']['data'].size
    ndphi = d2cross0['dphi']['data'].size

    # loop on versions
    dinterp = {}
    laxis = ['theta_ph', 'E_ph', 'E_e0']
    for version in d2cross0['cross'].keys():

        # ----------------------
        # extract data and units

        dinterp[version] = {
            'axis': {},
            'islog': {},
            'interp': None,
            'units': d2cross0['cross'][version]['units'],
        }
        cross = d2cross0['cross'][version]['data']

        # --------------
        # safety check

        if np.unique(cross.shape).size != cross.ndim:
            msg = (
                "d2cross from file has ambiguous dimensions!\n"
                f"\t- pfe: {d2cross}\n"
                f"\t- cross shape: {cross.shape}\n"
            )
            raise Exception(msg)

        # ---------------
        # axes

        for kk in laxis:
            data = d2cross0[kk]['data']
            assert np.prod(data.shape) == data.size
            data = np.squeeze(data)

            dinterp[version]['axis'][kk] = cross.shape.index(data.size)

            diff = np.diff(data)
            if np.allclose(diff, diff[0]):
                dinterp[version]['islog'][kk] = False
            else:
                difflog = np.diff(np.log10(data))
                if np.allclose(difflog, difflog[0]):
                    dinterp[version]['islog'][kk] = True
                else:
                    msg = (
                        "Not linear nor log!\n"
                        f"\t- pfe = {d2cross}\n"
                        f"\t- kk = {kk}\n"
                    )
                    raise Exception(msg)

        # ---------------
        # grid point
        # ---------------

        inds = np.argsort([dinterp[version]['axis'][kk] for kk in laxis])
        keys = [laxis[ii] for ii in inds]
        xx = tuple([
            np.log10(d2cross0[kk]['data'].ravel())
            if dinterp[version]['islog'][kk]
            else d2cross0[kk]['data'].ravel()
            for kk in keys
        ])

        # ---------------
        # interpolator
        # ---------------

        dinterp[version]['interp'] = scpinterp.RegularGridInterpolator(
            xx,
            cross,
            method='linear',
            bounds_error=False,
            fill_value=0.,
        )

        # ---------------
        # get_xx
        # ---------------

        islog = [dinterp[version]['islog'][kk] for kk in keys]
        keys = [f"{kk}_eV" if kk[0] == 'E' else kk for kk in keys]
        dinterp[version]['get_xx'] = _get_xx(keys, islog)

    return dinterp, nthetae, ndphi


def _get_xx(keys, islog):

    def func(**kwdargs):
        args = [
            np.log10(kwdargs[kk]) if islog[ii]
            else kwdargs[kk]
            for ii, kk in enumerate(keys)
        ]
        return np.moveaxis(
            np.array(np.broadcast_arrays(*args)),
            0,
            -1,
        )

    return func


# ###########################################
# ###########################################
#        load
# ###########################################


def _load(
    pfe=None,
    **kwdargs,
):

    # ----------
    # load
    # ----------

    d2cross_phi = {
        kk: vv.tolist() if vv.dtype == 'object' else vv
        for kk, vv in dict(np.load(pfe, allow_pickle=True)).items()
    }

    # ----------
    # compare with input
    # ----------

    dout = {}
    for k0, v0 in d2cross_phi.items():

        if k0 == 'd2cross_phi':
            continue

        # npts vs field
        knpts = [k1 for k1 in kwdargs.keys() if k1 == f"{k0}_npts"]
        if len(knpts) == 1:
            n1 = kwdargs[knpts[0]]
            if n1 is not None:
                n0 = v0.size
                if n0 != n1:
                    dout[k0] = f"wrong number of points: {n1} vs {n0}"
            continue

        elif len(knpts) > 1:
            msg = "weird glitch"
            raise Exception(msg)

        if kwdargs.get(k0) is None:
            continue

        if not isinstance(kwdargs[k0], v0.__class__):
            dout[k0] = f"wrong type: {type(kwdargs[k0])} vs {type(v0)}"
            continue

        if isinstance(v0, np.ndarray):
            if v0.shape != kwdargs[k0].shape:
                dout[k0] = f"wrong shape: {kwdargs[k0].shape} vs {v0.shape}"
                continue
            if not np.allclose(v0, kwdargs[k0]):
                dout[k0] = "Wrong array values"
                continue

        else:
            if v0 != kwdargs[k0]:
                dout[k0] = "wrong value"

    # -----------------------
    # Raise wraning if needed
    # -----------------------

    if len(dout) > 0:
        lstr = [f"\t- {k0}: {v0}" for k0, v0 in dout.items()]
        msg = (
            "Specified args do not match loaded from file:\n"
            f"pfe: {pfe}\n"
            + "\n".join(lstr)
        )
        warnings.warn(msg)

    return d2cross_phi


# ###########################################
# ###########################################
#        save
# ###########################################


def _save(
    d2cross_phi=None,
    pfe_save=None,
    overwrite=None,
    verb=None,
):

    # ----------
    # pfe_save
    # ----------

    if pfe_save is None:

        # extract sizes
        ntheta_ph = d2cross_phi['theta_ph_vsB'].size
        ntheta_e0 = d2cross_phi['theta_e0_vsB'].size
        Eph = _mod._format_vect2str(
            d2cross_phi['E_ph_eV'],
            base='eV',
        )
        Ee0 = _mod._format_vect2str(
            d2cross_phi['E_e0_eV'],
            base='eV',
        )

        # extract versions
        versions = d2cross_phi['version_cross']

        # extract boundaries
        path = os.path.abspath(_PATH_HERE)
        fname = (
            f"d2cross_phi_Ee0{Ee0}_Eph{Eph}"
            f"_nthetaph{ntheta_ph}_nthetae0{ntheta_e0}_{versions}"
        )
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

    np.savez(pfe_save, **d2cross_phi)
    msg = f"Saved in\n\t{pfe_save}"
    print(msg)

    return
