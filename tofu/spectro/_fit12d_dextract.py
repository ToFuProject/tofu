

# Built-in
import os

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
        'types': ['x'],
        'units': 'a.u.',
        'field': 'width',
    },
    'shift': {
        'types': ['x'],
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
        'field': 'shift',
    },
    'vi': {
        'types': ['lines'],  # necessarily by line for de-normalization (*lamb0)
        'units': 'm.s^-1',
    },
    'dratio': {
        'types': ['x'],
        'units': 'a.u.',
    },
    'dshift': {
        'types': ['x'],
        'units': 'a.u.',
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
    amp=None, coefs=None, ratio=None,
    Ti=None, width=None,
    vi=None, shift=None,
    pts_total=None,
    pts_detail=None,
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

    is2d = 'nbsplines' in dfit['dinput'].keys()
    if is2d is True:
        if 'phi2' not in dfit.keys():
            msg = "dfit is a fit2d output but does not have key 'phi2'!"
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

    # ----------------
    # Check / format amp, Ti, vi

    d3 = {k0: dict(v0) for k0, v0 in _D3}

    d3 = {
        'bck_amp': [bck, 'bck_amp'],
        'bck_rate': [bck, 'bck_rate'],
        'amp': [amp, 'amp'],
        'coefs': [coefs, 'amp'],
        'Ti': [Ti, 'width'],
        'width': [width, 'width'],
        'vi': [vi, 'shift'],
        'shift': [shift, 'shift'],
    }

    # ----------------
    # Ratio

    if ratio is not False:
        lkeys = dfit['dinput']['keys']
        if isinstance(ratio, tuple):
            ratio = [ratio]
        lc = [
            isinstance(ratio, list)
            and all([isinstance(tt, tuple) and len(tt) == 2 for tt in ratio])
            and all([all([ss in lkeys for ss in tt]) for tt in ratio]),
            isinstance(ratio, np.ndarray)
            and ratio.ndim == 2
            and ratio.shape[0] == 2
            and all([ss in lkeys for ss in ratio[0, :]])
            and all([ss in lkeys for ss in ratio[1, :]]),
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
            raise Exception(msg)

        if lc[0]:
            ratio = np.atleast_2d(ratio).T

    d3['ratio'] = ratio

    # ----------------
    # amp, Ti, vi from d3

    for k0 in d3.keys():

        # default
        if d3[k0][0] is None:
            d3[k0][0] = True
        if d3[k0][0] is True:
            d3[k0][0] = _D3[k0]
        if d3[k0][0] is False:
            d3[k0] = d3[k0][0]
            continue

        if 'bck' in k0:
            continue

        # basic conformity check
        if isinstance(d3[k0][0], str):
            d3[k0][0] = [d3[k0][0]]
        c0 = (
            isinstance(d3[k0][0], list)
            and all([isinstance(ss, str) for ss in d3[k0][0]])
        )
        if not c0:
            msg = (
                f"Arg {k0} must be a list of str!\n"
                f"Provided: {d3[k0][0]}"
            )
            raise Exception(msg)

        # check if trying to get all/some lines and / or all/some x
        lc = [
            all([ss in ['x', 'lines'] for ss in d3[k0][0]]),    # all lines/x
            all([ss in dinput['keys'] for ss in d3[k0][0]]),    # some lines
            all([ss in dinput[d3[k0][1]]['keys'] for ss in d3[k0][0]]), # x
        ]

        if not any(lc):
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

        if lc[0]:
            # 'lines' and/or 'x'
            lok = [('lines', dinput['keys']), ('x', dinput[d3[k0][1]]['keys'])]
            d3[k0][0] = {
                k1: {
                    'keys': v1,
                    'ind': np.arange(0, len(v1)),
                }
                for (k1, v1) in lkok
                if k0 in d3[k0][0]
            }

        else:
            if lc[1]:
                # a selection of lines
                keysok = dinput['keys']
                keys = d3[k0][0]
                if k0 == 'amp' and ratio is not False:
                    keys += list(set(ratio.ravel().tolist()))

            elif lc[2]:
                # a selection of variables 'x'
                keysok = dinput[d3[k0][1]]['keys']
                keys = d3[k0][0]

            d3[k0][0] = {
                k1: {
                    'keys': keys,
                    'ind': np.array(
                        [(keysok == ss).nonzero()[0][0] for ss in keys],
                        dtype=int,
                    )
                }
                for k1 in d3[k0][0]
            }

        d3[k0][0]['field'] = d3[k0][1]
        d3[k0] = d3[k0][0]

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
                phi_npts = int(npts_phi)
                phi_prof = np.linspace(
                    dprepare['domain']['phi']['minmax'][0],
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
    # pts_total, pts_detail

    if pts_total is None:
        if dprepare is None:
            pts_total = False
        else:
            if is2d is True:
                pts_total = np.array([dprepare['lamb'], dprepare['phi']])
            else:
                pts_total = dprepare['lamb']
    if pts_detail is None:
        pts_detail = False
    if pts_detail is True and pts_total is not False:
        pts_detail = pts_total
    if pts_detail is not False:
        pts_detail = np.array(pts_detail)
    if pts_total is not False:
        pts_total = np.array(pts_total)

    return dfit, d3, pts_total, pts_detail, phi_prof, vs_nbs


def fit1d_extract(
    dfit1d=None,
    bck=None,
    amp=None, coefs=None, ratio=None,
    Ti=None, width=None,
    vi=None, shift=None,
    pts_lamb_total=None, pts_lamb_detail=None,
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
        pts_lamb_total, pts_lamb_detail,
        _, _,
    ) = fit12d_get_data_checkformat(
        dfit=dfit1d,
        bck=bck,
        amp=amp, coefs=coefs, ratio=ratio,
        Ti=Ti, width=width,
        vi=vi, shift=shift,
        pts_total=pts_lamb_total,
        pts_detail=pts_lamb_detail,
    )

    # Extract dprepare and dind (more readable)
    dprepare = dfit1d['dinput']['dprepare']
    dind = dfit1d['dinput']['dind']
    nspect = dprepare['data'].shape[0]

    # Prepare extract func
    def _get_values(
        k0,
        d3=d3,
        dind=dind,
        sol_x=dfit1d['sol_x'],
        scales=dfit1d['scales'],
    ):
        for k1, v1 in d3[k0].items():
            ind = dind[d3[k0]['field']][k1][d3[k0]['ind']]
            d3[k0]['values'] = sol_x[:, ind] * scales[:, ind]

    # -------------------
    # Prepare output
    dout = dict(d3)
    dout.update(dict.fromkeys(['dratio', 'dshift'], False))

    # bck
    if d3['bck_amp'] is not False:
        dout['bck_amp'] = {
            'values': (
                dfit1d['sol_x'][:, dind['bck_amp']['x'][0]]
                * dfit1d['scales'][:, dind['bck_amp']['x'][0]]
            ),
            'units': 'a.u.',
        }
        dout['bck_rate'] = {
            'values': (
                dfit1d['sol_x'][:, dind['bck_rate']['x'][0]]
                * dfit1d['scales'][:, dind['bck_rate']['x'][0]]
            ),
            'units': 'a.u.',
        }

    # amp
    if d3['amp'] is not False:
        _get_values('amp')
        dout['amp'] = dict(d3['amp'])
        dout['amp']{'units': 'a.u.'}

    # ratio
    if d3['ratio'] is not False:
        nratio = d3['ratio'].shape[1]
        indup = np.r_[[
            (dout['amp']['lines']['keys'] == kk).nonzero()[0][0]
            for kk in d3['ratio'][0, :]
        ]]
        indlo = np.r_[[
            (dout['amp']['lines']['keys'] == kk).nonzero()[0][0]
            for kk in d3['ratio'][1, :]]
        ]
        val = (
            dout['amp']['lines']['values'][:, indup]
            / dout['amp']['lines']['values'][:, indlo]
        )
        lab = np.r_[
            ['{} / {}'.format(
                dfit1d['dinput']['symb'][indup[ii]],
                dfit1d['dinput']['symb'][indlo[ii]],
            )
            for ii in range(nratio)]
        ]
        dout['ratio'] = {
            'keys': dout['ratio'],
            'values': val,
            'lab': lab,
            'units': 'a.u.',
        }

    # Ti
    if d3['Ti'] is not False:
        _get_values('Ti')
        conv = np.sqrt(scpct.mu_0*scpct.c / (2.*scpct.h*scpct.alpha))
        indTi = np.array([iit[0] for iit in dind['width']['jac']])  # TBC
        indTi = indTi[d3['Ti']['lines']['ind']]
        dout['Ti'] = dict(d3['Ti'])
        dout['Ti']['values'] = (
            dout['Ti']['values']
            * conv * scpct.c**2
            * dfit1d['dinput']['mz'][indTi][None, :]
        )
        dout['Ti']['units'] = 'eV'

    # width
    if d3['width'] is not False:
        _get_values('width')
        dout['width'] = dict(d3['vi'])
        dout['width']['units'] = 'a.u.'

    # vi
    if d3['vi'] is not False:
        _get_values('vi')
        val = val * scpct.c
        dout['vi'] = dict(d3['vi'])
        dout['vi']['values'] = dout['vi']['values'] * scpct.c
        dout['vi']['units'] = 'm.s^-1'

    # shift
    if d3['shift'] is not False:
        _get_values('shift')
        val = val * dfit1d['dinput']['lines'][None, :]
        dout['shift'] = {'keys': keys, 'values': val, 'units': 'm'}

    # double
    if dfit1d['dinput']['double'] is not False:
        double = dfit1d['dinput']['double']
        if double is True or double.get('dratio') is None:
            dout['dratio'] = dfit1d['sol_x'][:, dind['dratio']['x']]
        else:
            dout['dratio'] = np.full((nspect,), double['dratio'])
        if double is True or double.get('dratio') is None:
            dout['dshift'] = dfit1d['sol_x'][:, dind['dshift']['x']]
        else:
            dout['dshift'] = np.full((nspect,), double['dshift'])

    # -------------------
    # sol_detail and sol_tot
    sold, solt = False, False
    if pts_lamb_detail is not False or pts_lamb_total is not False:

        (
            func_detail, func_cost, _,
        ) = _funccostjac.multigausfit1d_from_dlines_funccostjac(
            dprepare['lamb'],
            dinput=dfit1d['dinput'],
            dind=dind, jac=dfit1d['jac'])

        if pts_lamb_detail is not False:
            shape = tuple(np.r_[nspect, pts_lamb_detail.shape,
                                dfit1d['dinput']['nlines']+1])
            sold = np.full(shape, np.nan)
            for ii in range(nspect):
                sold[ii, dprepare['indok'][ii, :], :] = func_detail(
                    dfit1d['sol_x'][ii, :],
                    scales=dfit1d['scales'][ii, :],
                    indok=dprepare['indok'][ii, :],
                )
                # indok_var=dprepare['indok_var'][ii])

        if pts_lamb_total is not False:
            shape = tuple(np.r_[nspect, pts_lamb_total.shape])
            solt = np.full(shape, np.nan)
            for ii in range(nspect):
                solt[ii, dprepare['indok'][ii, :]] = func_cost(
                    dfit1d['sol_x'][ii, :],
                    scales=dfit1d['scales'][ii, :],
                    indok=dprepare['indok'][ii, :],
                    data=0.)

            # Double-check consistency if possible
            c0 = (pts_lamb_detail is not False
                  and np.allclose(pts_lamb_total, pts_lamb_detail))
            if c0:
                if not np.allclose(solt, np.sum(sold, axis=-1),
                                   equal_nan=True):
                    msg = "Inconsistent computations detail vs total"
                    raise Exception(msg)

    dout['sol_detail'] = sold
    dout['sol_tot'] = solt
    dout['units'] = 'a.u.'

    # -------------------
    # Add input args
    dout['d3'] = d3
    dout['pts_lamb_detail'] = pts_lamb_detail
    dout['pts_lamb_total'] = pts_lamb_total
    return dout


def _get_phi_profile(key,
                     nspect=None, dinput=None,
                     dind=None, sol_x=None, scales=None,
                     typ=None, ind=None, pts_phi=None):
    ncoefs = ind.size
    val = np.full((nspect, pts_phi.size, ncoefs), np.nan)
    BS = BSpline(
        dinput['knots_mult'],
        np.ones((dinput['nbs'], ncoefs), dtype=float),
        dinput['deg'],
        extrapolate=False,
        axis=0,
    )

    if typ == 'lines':
        keys = dinput['keys'][ind]
    else:
        keys = dinput[key]['keys'][ind]
    indbis = dind[key][typ][:, ind]
    for ii in range(nspect):
        BS.c = sol_x[ii, indbis] * scales[ii, indbis]
        val[ii, :, :] = BS(pts_phi)
    return keys, val


# Debugging, TBF
def fit2d_extract(
    dfit2d=None,
    bck=None,
    amp=None, coefs=None, ratio=None,
    Ti=None, width=None,
    vi=None, shift=None,
    pts_lamb_phi_total=None,
    pts_lamb_phi_detail=None,
    phi_prof=None,
    phi_npts=None,
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
        dfit1d, d3,
        pts_lamb_total, pts_lamb_detail,
        phi_prof, vs_nbs,
    ) = fit12d_get_data_checkformat(
        dfit=dfit2d,
        bck=bck,
        amp=amp, coefs=coefs, ratio=ratio,
        Ti=Ti, width=width,
        vi=vi, shift=shift,
        pts_total=pts_lamb_phi_total,
        pts_detail=pts_lamb_phi_detail,
        phi_prof=phi_prof,
        phi_npts=phi_npts,
        vs_nbs=vs_nbs,
    )

    # Extract dprepare and dind (more readable)
    dprepare = dfit1d['dinput']['dprepare']
    dind = dfit1d['dinput']['dind']
    nspect = dprepare['data'].shape[0]

    # Prepare extract func
    def _get_values(
        key,
        phi_prof=phi_prof,
        d3=d3,
        nspect=nspect,
        dinput=dfit1d['dinput'],
        dind=dind,
        sol_x=dfit1d['sol_x'],
        scales=dfit1d['scales'],
    ):
        if d3[key]['type'] == 'lines':
            keys = dinput['keys'][d3[key]['ind']]
        else:
            keys = dinput[d3[key]['field']]['keys'][d3[key]['ind']]
        indbis = dind[d3[key]['field']][d3[key]['type']][d3[key]['ind']]

        # coefs
        coefs = sol_x[:, indbis] * scales[:, indbis]

        # values at phi_prof
        BS = BSpline(
            dinput['knots_mult'],
            np.ones((dinput['nbs'], ncoefs), dtype=float),
            dinput['deg'],
            extrapolate=False,
            axis=0,
        )
        val = np.full((sol_x.shape[0], phi_prof.size), np.nan)
        for ii in range(nspect):
            BS.c = sol_x[ii, indbis] * scales[ii, indbis]
            val[ii, :, :] = BS(phi_prof)

        return keys, coefs, val

    # -------------------
    # Prepare output
    lk = [
        'bck_amp', 'bck_rate',
        'amp', 'coefs', 'ratio', 'Ti', 'width', 'vi', 'shift',
        'dratio', 'dshift',
    ]
    dout = dict.fromkeys(lk, False)

    # bck_amp
    if d3['bck_amp'] is not False:
        keys, coefs, val = _get_values(key='bck_amp')
        dout['bck_amp'] = {
            'keys': keys,
            'coefs': coefs,
            'values': val,
            'units': 'a.u.',
        }
            # 'coefs': (
                # dfit1d['sol_x'][:, dind['bck_amp']['x']]
                # * dfit1d['scales'][:, dind['bck_amp']['x']]
            # ),
        # dout['bck_rate'] = {
            # 'coefs': (
                # dfit1d['sol_x'][:, dind['bck_rate']['x']]
                # * dfit1d['scales'][:, dind['bck_rate']['x']]
            # ),
        # }

    # bck_rate
    if d3['bck_rate'] is not False:
        keys, coefs, val = _get_values(key='bck_rate')
        dout['bck_rate'] = {
            'keys': keys,
            'coefs': coefs,
            'values': val,
            'units': 'a.u.',
        }

    import pdb; pdb.set_trace()     # DB
    # amp
    if d3['amp'] is not False:
        keys, coefs, val = _get_values('amp')
        dout['amp'] = {
            'keys': keys,
            'coefs': coefs,
            'values': val,
            'units': 'a.u.',
        }

    # coefs
    if d3['coefs'] is not False:
        keys, coefs, val = _get_values('coefs')
        dout['coefs'] = {
            'keys': keys,
            'coefs': coefs,
            'values': val,
            'units': 'a.u.',
        }

    # ratio
    if d3['ratio'] is not False:
        nratio = d3['ratio'].shape[1]
        indup = np.r_[[(dout['coefs']['keys'] == kk).nonzero()[0][0]
                       for kk in d3['ratio'][0, :]]]
        indlo = np.r_[[(dout['coefs']['keys'] == kk).nonzero()[0][0]
                       for kk in d3['ratio'][1, :]]]
        val = (dout['coefs']['values'][:, indup]
               / dout['coefs']['values'][:, indlo])
        lab = np.r_[['{} / {}'.format(dfit1d['dinput']['symb'][indup[ii]],
                                      dfit1d['dinput']['symb'][indlo[ii]])
                     for ii in range(nratio)]]
        dout['ratio'] = {'keys': dout['ratio'], 'values': val,
                         'lab': lab, 'units': 'a.u.'}

    dout = {}
    # amp
    if d3['amp'] is not False:
        keys, val = _get_phi_profile(
            d3['amp']['field'], nspect=nspect,
            dinput=dfit2d['dinput'],
            dind=dfit2d['dind'], sol_x=dfit2d['sol_x'],
            scales=dfit2d['scales'], pts_phi=pts_phi,
            typ=d3['amp']['type'], ind=d3['amp']['ind'])
        dout['amp'] = {'keys': keys, 'values': val, 'units': 'a.u.'}

    # Ti
    if d3['Ti'] is not False:
        keys, val = _get_phi_profile(
            d3['Ti']['field'], nspect=nspect,
            dinput=dfit2d['dinput'],
            dind=dfit2d['dind'], sol_x=dfit2d['sol_x'],
            scales=dfit2d['scales'], pts_phi=pts_phi,
            typ=d3['Ti']['type'], ind=d3['Ti']['ind'])
        conv = np.sqrt(scpct.mu_0*scpct.c / (2.*scpct.h*scpct.alpha))
        if d3['Ti']['type'] == 'lines':
            indTi = np.arange(0, dfit2d['dinput']['nlines'])
        else:
            indTi = np.array([iit[0]
                              for iit in dfit2d['dind']['width']['jac']])
        indTi = indTi[d3['Ti']['ind']]
        val = (conv * val
               * dfit2d['dinput']['mz'][indTi][None, None, :]
               * scpct.c**2)
        dout['Ti'] = {'keys': keys, 'values': val, 'units': 'eV'}

    # vi
    if d3['vi'] is not False:
        keys, val = _get_phi_profile(
            d3['vi']['field'], nspect=nspect,
            dinput=dfit2d['dinput'],
            dind=dfit2d['dind'], sol_x=dfit2d['sol_x'],
            scales=dfit2d['scales'], pts_phi=pts_phi,
            typ=d3['vi']['type'], ind=d3['vi']['ind'])
        val = val * scpct.c
        dout['vi'] = {'keys': keys, 'values': val, 'units': 'm.s^-1'}

    # -------------------
    # sol_detail and sol_tot
    sold, solt = False, False
    if pts_lamb_phi_detail is not False or pts_lamb_phi_total is not False:

        func_detail = _funccostjac.multigausfit2d_from_dlines_funccostjac(
            dfit2d['dprepare']['lamb'], dfit2d['phi2'],
            indok=dfit2d['dprepare']['indok'],
            binning=dfit2d['dprepare']['binning'],
            dinput=dfit2d['dinput'],
            dind=dfit2d['dind'], jac=dfit2d['jac'])[0]

        if pts_lamb_phi_detail is not False:
            shape = tuple(np.r_[nspect, pts_lamb_phi_detail.shape,
                                dfit2d['dinput']['nlines']+1,
                                dfit2d['dinput']['nbs']])
            sold = np.full(shape, np.nan)
        if pts_lamb_phi_total is not False:
            shape = tuple(np.r_[nspect, pts_lamb_phi_total.shape])
            solt = np.full(shape, np.nan)

        for ii in range(nspect):

            # Separate and reshape output
            fd = func_detail(dfit2d['sol_x'][ii, :],
                             scales=dfit2d['scales'][ii, :],
                             indok_var=dfit2d['dprepare']['indok_var'][ii])

            if pts_lamb_phi_detail is not False:
                sold[ii, ...] = fd
            if pts_lamb_phi_total is not False:
                solt[ii, ...] = np.nansum(np.nansum(fd, axis=-1), axis=-1)

    dout['sol_detail'] = sold
    dout['sol_tot'] = solt
    dout['units'] = 'a.u.'

    # -------------------
    # Add input args
    dout['d3'] = d3
    dout['pts_phi'] = pts_phi
    dout['pts_lamb_phi_detail'] = pts_lamb_phi_detail
    dout['pts_lamb_phi_total'] = pts_lamb_phi_total
    return dout
