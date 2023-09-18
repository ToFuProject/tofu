

# Built-in
import os
import warnings


# Common
import numpy as np
import scipy.constants as scpct
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt


# specific
from . import _fit12d_funccostjac as _funccostjac


__all__ = [
    'fit1d_extract',
    'fit2d_extract',
]


# Think this through again:
# automatically load all ?
# width => Ti?
# shift => vi?
_D3 = {
    'bck_amp': {
        'types': ['x'],
        'unit': 'a.u.',
        'field': 'bck_amp',
    },
    'bck_rate': {
        'types': ['x'],
        'unit': 'a.u.',
        'field': 'bck_rate',
    },
    'amp': {
        'types': ['x', 'lines'],
        'units': 'a.u.',
        'field': 'amp',
    },
    'width': {
        'types': ['x', 'lines'],
        'units': 'a.u.',
        'field': 'width',
    },
    'shift': {
        'types': ['x', 'lines'],
        'units': 'a.u.',
        'field': 'shift',
    },
    'ratio': {
        'types': ['lines'],
        'units': 'a.u.',
        'field': 'amp',
    },
    'Ti': {
        'types': ['lines'],
        'units': 'eV',
        'field': 'width',
    },
    'vi': {
        'types': ['x'],
        'units': 'm.s^-1',
        'field': 'shift',
    },
    'dratio': {
        'types': ['x'],
        'units': 'a.u.',
        'field': 'dratio',
    },
    'dshift': {
        'types': ['x'],
        'units': 'a.u.',
        'field': 'dshift',
    },
}
_ALLOW_PICKLE = True


###########################################################
###########################################################
#
#   Extract data from pre-computed dict of fitted results
#
###########################################################
###########################################################


def fit12d_get_data_checkformat(
    dfit=None,
    bck=None,
    amp=None, ratio=None,
    Ti=None, width=None,
    vi=None, shift=None,
    sol_total=None,
    sol_detail=None,
    sol_pts=None,
    phi_prof=None,
    phi_npts=None,
    vs_nbs=None,
    allow_pickle=None,
):

    # ----------------
    # load file if str

    if isinstance(dfit, str):
        if not os.path.isfile(dfit) or not dfit[-4:] == '.npz':
            msg = ("Provided dfit must be either a dict or "
                   + "the absolute path to a saved .npz\n"
                   + "  You provided: {}".format(dfit))
            raise Exception(msg)
        if allow_pickle is None:
            allow_pickle = _ALLOW_PICKLE
        dfit = dict(np.load(dfit, allow_pickle=allow_pickle))
        _rebuild_dict(dfit)

    # ----------------
    # check dfit basic structure

    lk = ['dprepare', 'dinput', 'dind', 'sol_x', 'jac', 'scales']
    c0 = isinstance(dfit, dict) and all([ss in dfit.keys() for ss in lk])
    if not isinstance(dfit, dict):
        msg = ("\ndfit must be a dict with at least the following keys:\n"
               + "\t- {}\n".format(lk)
               + "\t- provided: {}".format(dfit))
        raise Exception(msg)

    # ----------------
    # Identify if fit1d or fit2d

    is2d = 'nbs' in dfit['dinput'].keys()
    if is2d is True:
        if 'symmetry' not in dfit['dinput'].keys():
            msg = "dfit is a fit2d but does not have key 'symmetry'!"
            raise Exception(msg)
        if dfit['dinput']['symmetry']:
            c0 = dfit['dinput'].get('symmetry_axis', False) is False
            if c0:
                msg = "dfit is a fit2d but does not have key 'symmetry_axis'!"
                raise Exception(msg)
    else:
        phi_prof = False

    # ----------------
    # Extract dinput and dprepare (more readable)

    dinput = dfit['dinput']
    dprepare = dfit['dinput']['dprepare']

    # ----------------
    # ratio

    if ratio is None:
        ratio = False
    if ratio is not False:
        amp = ['lines', 'x']

    if ratio is not False:
        lkeys = dfit['dinput']['keys']
        if isinstance(ratio, tuple):
            ratio = [ratio]
        lc = [
            isinstance(ratio, list)
            and all([isinstance(tt, tuple) and len(tt) == 2 for tt in ratio]),
            isinstance(ratio, np.ndarray)
            and ratio.ndim == 2
            and ratio.shape[0] == 2
        ]
        msg = (
            "\nArg ratio (spectral lines magnitude ratio) must be either:\n"
            "\t- False:  no line ration computed\n"
            "\t- tuple of len=2: upper and lower keys of the lines\n"
            "\t- list of tuple of len=2: upper and lower keys pairs\n"
            "\t- np.ndarray of shape (2, N): upper keys and lower keys\n"
            f"  Available keys: {lkeys}\n"
            f"  Provided: {ratio}\n"
        )
        if not any(lc):
            warnings.warn(msg)
            ratio = False
        elif lc[0]:
            ratio = np.atleast_2d(ratio).T

        # Remove ratio using non-aviailable lines
        indokratio = np.array([
            ratio[0, ii] in lkeys and ratio[1, ii] in lkeys
            for ii in range(ratio.shape[1])
        ])
        if np.any(indokratio):
            ratio = ratio[:, indokratio]
        else:
            ratio = False

    # ----------------
    # Check / format amp, Ti, vi

    # check if double
    isdouble = dfit['dinput']['double']

    d3 = {k0: dict(v0) for k0, v0 in _D3.items()}
    lval = [
        [bck, 'bck_amp'], [bck, 'bck_rate'],
        [amp, 'amp'], [width, 'width'], [shift, 'shift'],
        [ratio, 'ratio'], [Ti, 'Ti'], [vi, 'vi'],
        [isdouble, 'dratio'], [isdouble, 'dshift'],
    ]
    for (v0, k0) in lval:
        if v0 is None or v0 is True:
            d3[k0]['requested'] = _D3[k0]['types']
        else:
            d3[k0]['requested'] = v0

    # remove non-requested
    lout = [k0 for k0, v0 in d3.items() if v0['requested'] is False]
    for k0 in lout:
        del d3[k0]

    # ----------------
    # amp, Ti, vi from d3

    lkkeys = ['amp', 'width', 'shift', 'Ti', 'vi']
    for k0 in d3.keys():

        if k0 == 'ratio':
            v0 = d3[k0]['types']
        else:
            v0 = d3[k0]['requested']

        # basic conformity check
        if isinstance(v0, str):
            v0 = [v0]
            d3[k0]['requested'] = v0

        c0 = (
            k0 != 'ratio'
            and isinstance(v0, list)
            and all([isinstance(ss, str) for ss in v0])
        )
        if not (k0 == 'ratio' or c0):
            msg = (
                f"Arg {k0} must be a list of str!\n"
                f"Provided: {v0}"
            )
            raise Exception(msg)

        # check if trying to get all/some lines and / or all/some x
        ltypes = d3[k0]['types']
        c0 = all([ss in ltypes for ss in v0]),               # all lines/x
        c1 = (
            not c0
            and 'lines' in ltypes
            and all([ss in dinput['keys'] for ss in v0]),       # some lines
        )
        c2 = (
            not c0
            and not c1
            and 'x' in ltypes
            and all([ss in dinput[k0]['keys'] for ss in v0]),   # some x
        )

        if not any([c0, c1, c2]):
            msg = (
                f"Arg {k0} elements must be either:\n"
                f"\t- 'x': return all unique {k0}\n"
                f"\t- 'lines': return {k0} for all lines (inc. duplicates)\n"
                "\t- str: a key in:\n"
                f"\t\t lines: {dinput['keys']}\n"
                f"\t\t variables: {dinput[d3[k0][1]]['keys']}\n\n"
                f"Provided: {d3[k0][0]}"
            )
            raise Exception(msg)

        if c0:
            # 'lines' and/or 'x'
            for k1 in v0:

                if k0 in lkkeys:
                    if k1 == 'lines':
                        keys = dinput['keys']
                    else:
                        keys = dinput[d3[k0]['field']]['keys']

                    d3[k0][k1] = {
                        'keys': keys,
                        'ind': np.arange(0, len(keys)),
                    }

                elif k0 != 'ratio':
                    d3[k0][k1] = {
                        'ind': np.r_[0],
                    }
                else:
                    d3[k0][k1] = {}

        else:
            if c1:
                # a selection of lines
                typ = 'lines'
                keysok = dinput['keys']
                keys = v0
                if k0 == 'amp' and ratio is not False:
                    for rr in set(ratio.ravel().tolist()):
                        if rr not in keys:
                            keys.append(rr)

            elif c2:
                # a selection of variables 'x'
                typ = 'x'
                keysok = dinput[d3[k0][1]]['keys']
                keys = v0

            d3[k0][typ] = {
                'keys': keys,
                'ind': np.array(
                    [(keysok == ss).nonzero()[0][0] for ss in keys],
                    dtype=int,
                )
            }

    # ----------------
    # phi_prof, phi_npts

    if is2d is True:
        c1 = [phi_prof is not None, phi_npts is not None]
        if all(c1):
            msg = "Arg phi_prof and phi_npts cannot be both provided!"
            raise Exception(msg)

        if phi_npts is False or phi_prof is False:
            phi_prof = False
        else:
            if not any(c1):
                phi_npts = (2*dinput['deg']-1)*(dinput['knots'].size-1) + 1
            if phi_npts is not None:
                phi_npts = int(phi_npts)
                if dfit['dinput']['symmetry'] is True:
                    phimin = (
                        np.mean(dfit['dinput']['symmetry_axis'])
                        - dprepare['domain']['phi']['minmax'][1]
                    )
                else:
                    phimin = dprepare['domain']['phi']['minmax'][0]
                phi_prof = np.linspace(
                    phimin,
                    dprepare['domain']['phi']['minmax'][1],
                    phi_npts,
                )
            else:
                phi_prof = np.atleast_1d(phi_prof).ravel()

        # vs_nbs
        if vs_nbs is None:
            vs_nbs = True
        if not isinstance(vs_nbs, bool):
            msg = "Arg vs_nbs must be a bool!"
            raise Exception(msg)

    # ----------------
    # sol_total, sol_detail, sol_pts

    if sol_pts is not None:
        if is2d is True:
            c0 = (
                isinstance(sol_pts, (list, np.ndarray))
                and len(sol_pts) == 2
                and all([isinstance(ss, np.ndarray) for ss in sol_pts])
                and sol_pts[0].shape == sol_pts[1].shape
            )
            if not c0:
                msg = (
                    "Arg sol_lamb_phi must be a tuple of 2 np.ndarray"
                    " of same shape!"
                )
                raise Exception(msg)
        else:
            c0 = isinstance(sol_pts, np.ndarray)
            if not c0:
                msg = "Arg sol_lamb must be a np.ndarray!"
                raise Exception(msg)

    if sol_total is None:
        sol_total = sol_pts is not None
    if sol_detail is None:
        sol_detail = False

    if not isinstance(sol_total, bool):
        msg = f"Arg sol_total must be a bool!\nProvided: {sol_total}"
        raise Exception(msg)
    if not isinstance(sol_detail, bool):
        msg = f"Arg sol_detail must be a bool!\nProvided: {sol_detail}"
        raise Exception(msg)

    c0 = (sol_total is True or sol_detail is True) and sol_pts is None
    if c0:
        if dprepare is None:
            sol_pts = False
        else:
            if is2d is True:
                sol_pts = [dprepare['lamb'], dprepare['phi']]
            else:
                sol_pts = dprepare['lamb']
    if any([sol_total, sol_detail]):
        assert sol_pts is not None

    return dfit, d3, sol_total, sol_detail, sol_pts, phi_prof, vs_nbs


def fit1d_extract(
    dfit1d=None,
    bck=None,
    amp=None, ratio=None,
    Ti=None, width=None,
    vi=None, shift=None,
    sol_total=None,
    sol_detail=None,
    sol_lamb=None,
):
    """
    Return a dict with extarcted data of interest

    bck_amp:    (nt,) array
    bck_rate:   (nt,) array
    amp:        (nt, namp) array
    coefs:      (nt, nlines) array
    ratio:      (nt, nratio) array
    width:      (nt, nwidth) array
    Ti:         (nt, nlines) array
    shift:      (nt, nshift) array
    vi:         (nt, nlines) array

    """

    # -------------------
    # Check format input
    (
        dfit1d, d3,
        sol_total, sol_detail, sol_lamb,
        _, _,
    ) = fit12d_get_data_checkformat(
        dfit=dfit1d,
        bck=bck,
        amp=amp, ratio=ratio,
        Ti=Ti, width=width,
        vi=vi, shift=shift,
        sol_total=sol_total,
        sol_detail=sol_detail,
        sol_pts=sol_lamb,
    )

    # Extract dprepare and dind (more readable)
    dprepare = dfit1d['dinput']['dprepare']
    dind = dfit1d['dinput']['dind']
    nspect = dprepare['data'].shape[0]

    # Prepare extract func
    def _get_values(
        k0=None,
        k1=None,
        d3=d3,
        dinput=dfit1d['dinput'],
        dind=dind,
        sol_x=dfit1d['sol_x'],
        scales=dfit1d['scales'],
    ):
        ind = dind[d3[k0]['field']][k1][0, d3[k0][k1]['ind']]
        coefs = sol_x[:, ind] * scales[:, ind]

        # rebuild from constraints
        if k0 in ['amp', 'width', 'shift']:
            if k1 == 'lines':
                coefs[...] = (
                    coefs[...] * dinput[k0]['coefs'][None, :]
                    + dinput[k0]['offset'][None, :]
                )

        return coefs

    # -------------------
    # Prepare output

    # multiple-value, direct computation
    lk_direct = ['bck_amp', 'bck_rate', 'amp', 'width', 'shift']
    for k0 in set(lk_direct).intersection(d3.keys()):
        for k1 in set(['x', 'lines']).intersection(d3[k0].keys()):
            d3[k0][k1]['values'] = _get_values(k0=k0, k1=k1)

    # multiple-value, indirect computation
    k0 = 'Ti'
    if k0 in d3.keys():
        k1 = d3[k0]['types'][0]
        val = _get_values(k0=k0, k1=k1)
        conv = np.sqrt(scpct.mu_0*scpct.c / (2.*scpct.h*scpct.alpha))
        d3[k0][k1]['values'] = (
            val * conv * scpct.c**2
            * dfit1d['dinput']['mz'][d3[k0][k1]['ind']][None, :]
        )

    # vi
    k0 = 'vi'
    if k0 in d3.keys():
        k1 = d3[k0]['types'][0]
        val = _get_values(k0=k0, k1=k1)
        d3[k0][k1]['values'] = val * scpct.c

    # ratio
    k0 = 'ratio'
    if k0 in d3.keys():
        k1 = d3[k0]['types'][0]
        nratio = d3[k0]['requested'].shape[1]
        indup = np.r_[[
            (d3['amp']['lines']['keys'] == kk).nonzero()[0][0]
            for kk in d3['ratio']['requested'][0, :]
        ]]
        indlo = np.r_[[
            (d3['amp']['lines']['keys'] == kk).nonzero()[0][0]
            for kk in d3['ratio']['requested'][1, :]]
        ]
        val = (
            d3['amp']['lines']['values'][:, indup]
            / d3['amp']['lines']['values'][:, indlo]
        )
        lab = np.r_[
            [
                '{} / {}'.format(
                    dfit1d['dinput']['symb'][indup[ii]],
                    dfit1d['dinput']['symb'][indlo[ii]],
                )
                for ii in range(nratio)
            ]
        ]
        d3['ratio']['lines']['values'] = val
        d3['ratio']['lines']['lab'] = lab

    # double
    if dfit1d['dinput']['double'] is not False:
        double = dfit1d['dinput']['double']
        for k0 in ['dratio', 'dshift']:
            if double is True or double.get(k0) is None:
                val = _get_values(k0=k0, k1='x')
            else:
                val = np.full((nspect, 1), double[k0])
            d3[k0]['x']['values'] = val

    # -------------------
    # sol_detail and sol_tot
    sold, solt = False, False
    if any([sol_total, sol_detail]):

        (
            func_detail, func_cost,
        ) = _funccostjac.multigausfit1d_from_dlines_funccostjac(
            lamb=sol_lamb,
            indx=None,      # because dfit['sol_x' built with const]
            dinput=dfit1d['dinput'],
            dind=dind,
            jac=None,
        )[:2]

        # sol_details
        if sol_detail:
            shape = tuple(np.r_[
                nspect,
                sol_lamb.shape,
                dfit1d['dinput']['nlines'] + 1,
            ])
            sold = np.full(shape, np.nan)
            for ii in range(nspect):
                if dfit1d['validity'][ii] < 0:
                    continue
                sold[ii, ...] = func_detail(
                    dfit1d['sol_x'][ii, :],
                    scales=dfit1d['scales'][ii, :],
                    indok=None,
                    const=None,
                )

        # sol_total
        if sol_total:
            shape = tuple(np.r_[nspect, sol_lamb.shape])
            solt = np.full(shape, np.nan)
            for ii in range(nspect):
                if dfit1d['validity'][ii] < 0:
                    continue
                solt[ii, ...] = func_cost(
                    dfit1d['sol_x'][ii, :],
                    scales=dfit1d['scales'][ii, :],
                    indok=None,
                    const=None,
                    data=0.,
                )

            # Double-check consistency if possible
            if sol_detail:
                soldsum = np.nansum(sold, axis=-1)
                iok = (~np.isnan(solt)) & (~np.isnan(soldsum))
                c1 = np.allclose(solt[iok], soldsum[iok], equal_nan=True)
                if not c1:
                    msg = "Inconsistent computations detail vs total"
                    raise Exception(msg)

    dout = {
        'sol_detail': sold,
        'sol_total': solt,
        'units': 'a.u.',
        'd3': d3,
        'sol_lamb': sol_lamb,
    }
    return dout


def fit2d_extract(
    dfit2d=None,
    bck=None,
    amp=None, ratio=None,
    Ti=None, width=None,
    vi=None, shift=None,
    sol_total=None,
    sol_detail=None,
    sol_lamb_phi=None,
    phi_prof=None,
    phi_npts=None,
    lines_indphi=None,
    amp_on_bck_thresh=None,
    vs_nbs=None,
):
    """
    Return a dict with extarcted data of interest

    bck_amp:    (nt, nbs) array
    bck_rate:   (nt, nbs) array
    amp:        (nt, nbs, namp)   and/or (nt, phi_npts, namp) array
    coefs:      (nt, nbs, nlines) and/or (nt, phi_npts, nlines) array
    ratio:      (nt, nratio) array
    width:      (nt, nwidth) array
    Ti:         (nt, nlines) array
    shift:      (nt, nshift) array
    vi:         (nt, nlines) array

    """

    # -------------------
    # Check format input
    (
        dfit2d, d3,
        sol_total, sol_detail, sol_lamb_phi,
        phi_prof, vs_nbs,
    ) = fit12d_get_data_checkformat(
        dfit=dfit2d,
        bck=bck,
        amp=amp, ratio=ratio,
        Ti=Ti, width=width,
        vi=vi, shift=shift,
        sol_total=sol_total,
        sol_detail=sol_detail,
        sol_pts=sol_lamb_phi,
        phi_prof=phi_prof,
        phi_npts=phi_npts,
        vs_nbs=vs_nbs,
    )

    # lines_indphi
    if lines_indphi is None:
        lines_indphi = dfit2d['dinput']['keys']
    if isinstance(lines_indphi, str):
        lines_indphi = [lines_indphi]
    if not all([ss in dfit2d['dinput']['keys'] for ss in lines_indphi]):
        msg = (
            "Arg lines_indphi must be a list of lines keys:\n"
            f"\t- lines_indphi: {lines_indphi}\n"
            f"\t- available: {dfit2d['dinput']['keys']}\n"
        )
        raise Exception(msg)

    if amp_on_bck_thresh is None:
        amp_on_bck_thresh = 2.

    # Extract dprepare and dind (more readable)
    dprepare = dfit2d['dinput']['dprepare']
    dind = dfit2d['dinput']['dind']
    nspect = dprepare['data'].shape[0]

    # Prepare Bsplines
    nbs = dfit2d['dinput']['nbs']
    BS = BSpline(
        dfit2d['dinput']['knots_mult'],
        np.ones((nbs, 1), dtype=float),
        dfit2d['dinput']['deg'],
        extrapolate=False,
        axis=0,
    )

    # phi_prof symmetric
    if dfit2d['dinput']['symmetry'] is True:
        phi_prof2 = np.abs(
            phi_prof - np.mean(dfit2d['dinput']['symmetry_axis'])
        )
    else:
        phi_prof2 = phi_prof

    # Prepare extract func
    def _get_values(
        k0,
        k1=None,
        indtok=dfit2d['validity'] == 0,
        phi_prof2=phi_prof2,
        dinput=dfit2d['dinput'],
        d3=d3,
        nspect=nspect,
        BS=BS,
        dind=dind,
        sol_x=dfit2d['sol_x'],
        scales=dfit2d['scales'],
    ):

        # For bck_amp, bck_rate and dratio, dshift
        # => need to make ind 2d !! [nbs, 1] and not 1d [nbs,]
        ind = dind[d3[k0]['field']][k1][:, d3[k0][k1]['ind']]
        assert k0 in ['dratio', 'dshift'] or ind.shape[0] == nbs

        # coefs
        shape = tuple(np.r_[nspect, ind.shape])
        coefs = np.full(shape, np.nan)

        coefs[indtok, ...] = (
            sol_x[indtok, ...][:, ind] * scales[indtok, ...][:, ind]
        )

        # rebuild from constraints
        if k0 in ['amp', 'width', 'shift']:
            if k1 == 'lines':
                coefs[indtok, ...] = (
                    coefs[indtok, ...] * dinput[k0]['coefs'][None, :]
                    + dinput[k0]['offset'][None, :]
                )

        # values at phi_prof
        shape = tuple(np.r_[nspect, phi_prof2.size, ind.shape[1]])
        val = np.full(shape, np.nan)
        for ii in indtok.nonzero()[0]:
            BS.c = coefs[ii, :, :]
            val[ii, :, :] = BS(phi_prof2)

        return coefs, val

    # multiple-value, direct computation
    lk_direct = ['bck_amp', 'bck_rate', 'amp', 'width', 'shift']
    for k0 in set(lk_direct).intersection(d3.keys()):
        for k1 in set(['x', 'lines']).intersection(d3[k0].keys()):
            d3[k0][k1]['coefs'], d3[k0][k1]['values'] = _get_values(
                k0=k0, k1=k1,
            )

    # multiple-value, indirect computation
    k0 = 'Ti'
    if k0 in d3.keys():
        k1 = d3[k0]['types'][0]
        coefs, val = _get_values(k0=k0, k1=k1)
        conv = np.sqrt(scpct.mu_0*scpct.c / (2.*scpct.h*scpct.alpha))
        mz = dfit2d['dinput']['mz'][d3[k0][k1]['ind']]
        d3[k0][k1]['coefs'] = (
            coefs * conv * scpct.c**2 * mz[None, None, :]
        )
        d3[k0][k1]['values'] = (
            val * conv * scpct.c**2 * mz[None, None, :]
        )

    # vi
    k0 = 'vi'
    if k0 in d3.keys():
        k1 = d3[k0]['types'][0]
        coefs, val = _get_values(k0=k0, k1=k1)
        d3[k0][k1]['coefs'] = coefs * scpct.c
        d3[k0][k1]['values'] = val * scpct.c

    # ratio
    k0 = 'ratio'
    if k0 in d3.keys():
        k1 = d3[k0]['types'][0]
        nratio = d3[k0]['requested'].shape[1]

        # safeguard against non-available lines
        indrequest = np.array([
            ii for ii in range(d3['ratio']['requested'].shape[1])
            if d3['ratio']['requested'][0, ii] in d3['amp']['lines']['keys']
            and d3['ratio']['requested'][1, ii] in d3['amp']['lines']['keys']
        ])

        # get indices for all ratios
        indup = np.r_[[
            (d3['amp']['lines']['keys'] == kk).nonzero()[0][0]
            for kk in d3['ratio']['requested'][0, indrequest]
        ]]
        indlo = np.r_[[
            (d3['amp']['lines']['keys'] == kk).nonzero()[0][0]
            for kk in d3['ratio']['requested'][1, indrequest]]
        ]

        # get values and labels
        val = (
            d3['amp']['lines']['values'][:, :, indup]
            / d3['amp']['lines']['values'][:, :, indlo]
        )
        lab = np.r_[
            [
                '{} / {}'.format(
                    dfit2d['dinput']['symb'][indup[ii]],
                    dfit2d['dinput']['symb'][indlo[ii]],
                )
                for ii in range(nratio)
            ]
        ]
        d3['ratio']['lines']['values'] = val
        d3['ratio']['lines']['lab'] = lab

    # double
    if dfit2d['dinput']['double'] is not False:
        double = dfit2d['dinput']['double']
        for k0 in ['dratio', 'dshift']:
            if double is True or double.get(k0) is None:
                coefs, val = _get_values(k0=k0, k1='x')
            else:
                coefs = np.full((nspect, nbs, 1), double[k0])
                val = np.full((nspect, nbs, 1), double[k0])
            d3[k0]['x']['coefs'] = coefs
            d3[k0]['x']['values'] = val

    # -------------------
    # check lines / bck ratio
    # a fit is usable if the lines amplitudes are >= background
    # using lambda0 for simplicity

    # get background for each line
    bcki = (
        d3['bck_amp']['x']['values']
        * np.exp(
            d3['bck_rate']['x']['values']
            * (
                dfit2d['dinput']['lines'][None, None, :]
                - dfit2d['dinput']['lambmin_bck']
            )
        )
    )

    amp_on_bck = d3['amp']['lines']['values'][:, :, :] / bcki

    # get indices of each line desired in lines_indphi
    iiphi = np.array([
        (dfit2d['dinput']['keys'] == ss).nonzero()[0][0]
        for ss in lines_indphi
    ])

    # check validity of amp/bck ratio
    indphi = amp_on_bck[:, :, iiphi] >= amp_on_bck_thresh
    indphi = np.all(indphi, axis=-1)

    for ii in range(nspect):
        indphi_no = np.copy(indphi[ii, ...])
        for jj in range(len(dfit2d['dinput']['valid']['ldphi'][ii])):
            indphi_no &= (
                (phi_prof < dfit2d['dinput']['valid']['ldphi'][ii][jj][0])
                | (phi_prof >= dfit2d['dinput']['valid']['ldphi'][ii][jj][1])
            )
        indphi[ii, indphi_no] = False

    if not np.any(indphi):
        msg = (
            "No usable vertical profile!\n"
            "Conditions for usability include:\n"
            f"\t- lines amplitude > = {amp_on_bck_thresh} * background\n"
            f"\t- phi in a an interval of valid data, for lines {lines_indphi}"
        )
        warnings.warn(msg)

    # ----------
    # update phi profiles
    for k0 in d3.keys():
        for k1 in {'lines', 'x'}.intersection(d3[k0].keys()):
            if 'values' in d3[k0][k1].keys():
                for jj in range(d3[k0][k1]['values'].shape[-1]):
                    d3[k0][k1]['values'][~indphi, jj] = np.nan

    # update validity according to indphi
    dfit2d['validity'][
        (dfit2d['validity'] == 0) & np.all(~indphi, axis=1)
    ] = -3

    # ----------
    # func
    (
        func_detail, func_sum,
    ) = _funccostjac.multigausfit2d_from_dlines_funccostjac(
        dinput=dfit2d['dinput'],
        dind=dind,
        jac=None,
    )[:2]

    # -------------------
    # func_tot

    def func_tot(
        lamb=None,
        phi=None,
        nspect=nspect,
        dfit2d=dfit2d,
        func_sum=func_sum,
    ):
        assert lamb.shape == phi.shape
        if dfit2d['dinput']['symmetry'] is True:
            phi = np.abs(phi - np.nanmean(dfit2d['dinput']['symmetry_axis']))

        shape = tuple(np.r_[nspect, lamb.shape])
        solt = np.full(shape, np.nan)
        for ii in range(nspect):
            if dfit2d['validity'][ii] < 0:
                continue
            # Separate and reshape output
            solt[ii, ...] = func_sum(
                dfit2d['sol_x'][ii, :],
                lamb=lamb,
                phi=phi,
                scales=dfit2d['scales'][ii, :],
                const=None,
            )
        return solt

    # -------------------
    # sol_detail and sol_tot

    sold, solt = False, False
    if sol_total or sol_detail:
        if dfit2d['dinput']['symmetry'] is True:
            sol_lamb_phi[1] = np.abs(
                sol_lamb_phi[1] - np.nanmean(dfit2d['dinput']['symmetry_axis'])
            )

        # sol_details
        if sol_detail:
            shape = tuple(np.r_[
                nspect,
                sol_lamb_phi[0].shape,
                dfit2d['dinput']['nlines'] + 1,
                nbs,
            ])
            sold = np.full(shape, np.nan)
            if dfit2d['dinput']['symmetry'] is True:
                phi = np.abs(
                    sol_lamb_phi[1]
                    - np.nanmean(dfit2d['dinput']['symmetry_axis'])
                )
            else:
                phi = sol_lamb_phi[1]

            for ii in range(nspect):
                if dfit2d['validity'][ii] < 0:
                    continue
                sold[ii, ...] = func_detail(
                    dfit2d['sol_x'][ii, :],
                    lamb=sol_lamb_phi[0],
                    phi=phi,
                    scales=dfit2d['scales'][ii, :],
                    const=None,
                    # indbs=None,
                )

        # sol_total
        if sol_total:
            solt = func_tot(
                lamb=sol_lamb_phi[0],
                phi=sol_lamb_phi[1],
            )

            # Double-check consistency if possible
            if sol_detail:
                soldsum = np.nansum(np.nansum(sold, axis=-1), axis=-1)
                iok = (~np.isnan(solt)) & (~np.isnan(soldsum))
                c1 = np.allclose(solt[iok], soldsum[iok], equal_nan=True)
                if not c1:
                    msg = "Inconsistent computations detail vs total"
                    raise Exception(msg)

    # -------------------
    # Add input args

    dout = {
        'sol_detail': sold,
        'sol_tot': solt,
        'func_tot': func_tot,
        'units': 'a.u.',
        'd3': d3,
        'bcki': bcki,
        'amp_on_bck': amp_on_bck,
        'amp_on_bck_thresh': amp_on_bck_thresh,
        'phi_prof': phi_prof,
        'indphi': indphi,
        'lines_indphi': lines_indphi,
        'sol_lamb_phi': sol_lamb_phi,
    }

    return dout
