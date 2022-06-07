
# Built-in
import os
import warnings
import datetime as dtm      # DB

# Common
import numpy as np
import scipy.optimize as scpopt
import scipy.interpolate as scpinterp
import scipy.sparse as sparse
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt


# ToFu-specific
from ._fit12d_dinput import *
from ._fit12d_dinput import _dict2vector_dscalesx0bounds
from . import _fit12d_funccostjac as _funccostjac
from ._fit12d_dextract import *
from . import _plot


__all__ = [
    'fit1d_dinput',
    'fit2d_dinput',
    'fit12d_dvalid',
    'fit12d_dscales',
    'fit1d',
    'fit2d',
    'fit1d_extract',
    'fit2d_extract',
]


_TOL1D = {'x': 1e-10, 'f': 1.e-10, 'g': 1.e-10}
_TOL2D = {'x': 1e-6, 'f': 1.e-6, 'g': 1.e-6}
_CHAIN = True
_METHOD = 'trf'
_LOSS = 'linear'
_SIGMA_MARGIN = 3.
_DVALIDITY = {
    0: 'ok',
    -1: 'non-valid input data',
    -2: 'convergence failed',
    -3: 'insufficient lines / bck',
}



###########################################################
###########################################################
#
#           Load dinput
#
###########################################################
###########################################################


def _rebuild_dict(dd):
    for k0, v0 in dd.items():
        if isinstance(v0, np.ndarray) and v0.shape == ():
            dd[k0] = v0.tolist()
        if isinstance(dd[k0], dict):
            _rebuild_dict(dd[k0])


def _checkformat_dinput(dinput, allow_pickle=True):
    if isinstance(dinput, str):
        if not (os.path.isfile(dinput) and dinput[-4:] == '.npz'):
            msg = ("Arg dinput must be aither a dict or "
                   + "the absolute path to a .npz\n"
                   + "  You provided: {}".format(dinput))
            raise Exception(msg)
        dinput = dict(np.load(dinput, allow_pickle=allow_pickle))

    if not isinstance(dinput, dict):
        msg = (
            "dinput must be a dict!\n"
            + "  You provided: {}".format(type(dinput))
        )

    _rebuild_dict(dinput)
    return dinput


###########################################################
###########################################################
#
#           Main fitting sub-routines
#
###########################################################
###########################################################


def _checkformat_options(
    chain, method, tr_solver, tr_options,
    xtol, ftol, gtol, loss, max_nfev, verbose, strict,
):
    if chain is None:
        chain = _CHAIN
    if method is None:
        method = _METHOD
    assert method in ['trf', 'dogbox'], method
    if tr_solver is None:
        tr_solver = None
    if tr_options is None:
        tr_options = {}
    if xtol is None:
        xtol = _TOL1D['x']
    if ftol is None:
        ftol = _TOL1D['f']
    if gtol is None:
        gtol = _TOL1D['g']
    if loss is None:
        loss = _LOSS
    if max_nfev is None:
        max_nfev = None
    if verbose is None:
        verbose = 1
    if verbose == 3:
        verbscp = 2
    else:
        verbscp = 0
    if strict is None:
        strict = False

    return (chain, method, tr_solver, tr_options,
            xtol, ftol, gtol, loss, max_nfev, verbose, verbscp, strict)


def multigausfit1d_from_dlines(
    dinput=None,
    method=None, tr_solver=None, tr_options=None,
    xtol=None, ftol=None, gtol=None,
    max_nfev=None, chain=None, verbose=None,
    loss=None, jac=None,
    strict=None,
):
    """ Solve multi_gaussian fit in 1d from dlines

    If double is True, all lines are double with common shift and ratio

    Unknowns are:
        x = [bck, w0, v0, c00, c01, ..., c0n, w1, v1, c10, c11, ..., c1N, ...]

        - bck : constant background
        - wi  : spectral width of a group of lines (ion): wi^2 = 2kTi / m*c**2
                This way, it is dimensionless
        - vni : normalised velicity of the ion: vni = vi / c
        - cij : normalised coef (intensity) of line: cij = Aij

    Scaling is done so each quantity is close to unity:
        - bck: np.mean(data[data < mean(data)/2])
        - wi : Dlamb / 20
        - vni: 10 km/s
        - cij: np.mean(data)

    """

    # ---------------------------
    # Check format options
    (
        chain, method, tr_solver, tr_options,
        xtol, ftol, gtol, loss, max_nfev,
        verbose, verbscp, strict,
    ) = _checkformat_options(
         chain, method, tr_solver, tr_options,
         xtol, ftol, gtol, loss, max_nfev, verbose, strict,
    )

    # ---------------------------
    # Load dinput if necessary
    dinput = _checkformat_dinput(dinput)
    dprepare, dind = dinput['dprepare'], dinput['dind']
    nspect = dprepare['data'].shape[0]

    # ---------------------------
    # If same spectrum => consider a single data set
    if dinput['same_spectrum'] is True:
        lamb = (
            dinput['same_spectrum_dlamb']*np.arange(0, nspect)[:, None]
            + dprepare['lamb'][None, :]
        ).ravel()
        datacost = dprepare['data'].ravel()[None, :]
        nspect = data.shape[0]
        chain = False
    else:
        lamb = dprepare['lamb']
        datacost = dprepare['data']

    # ---------------------------
    # Get scaling, x0, bounds from dict
    scales = _dict2vector_dscalesx0bounds(
        dd=dinput['dscales'], dd_name='dscales', dinput=dinput,
    )
    x0 = _dict2vector_dscalesx0bounds(
        dd=dinput['dx0'], dd_name='dx0', dinput=dinput,
    )
    boundmin = _dict2vector_dscalesx0bounds(
        dd=dinput['dbounds']['min'], dd_name="dbounds['min']", dinput=dinput,
    )
    boundmax = _dict2vector_dscalesx0bounds(
        dd=dinput['dbounds']['max'], dd_name="dbounds['max']", dinput=dinput,
    )
    bounds = np.array([boundmin[0, :], boundmax[0, :]])

    # ---------------------------
    # Separate free from constant parameters
    const = _dict2vector_dscalesx0bounds(
        dd=dinput['dconstants'], dd_name='dconstants', dinput=dinput,
    )
    indx = np.any(np.isnan(const), axis=0)
    const = const[:, ~indx]
    x0[:, ~indx] = const / scales[:, ~indx]

    # ---------------------------
    # Get function, cost function and jacobian
    (
        func_detail, func_cost, func_jac,
    ) = _funccostjac.multigausfit1d_from_dlines_funccostjac(
        lamb, dinput=dinput, dind=dind, jac=jac, indx=indx,
    )

    # ---------------------------
    # Prepare output
    sol_x = np.full((nspect, dind['sizex']), np.nan)
    success = np.full((nspect,), np.nan)
    time = np.full((nspect,), np.nan)
    cost = np.full((nspect,), np.nan)
    nfev = np.full((nspect,), np.nan)
    validity = np.zeros((nspect,), dtype=int)
    message = ['' for ss in range(nspect)]
    errmsg = ['' for ss in range(nspect)]

    # Prepare msg
    if verbose in [1, 2]:
        col = np.char.array([
            'spect', 'time (s)', 'cost', 'nfev', 'njev', 'msg',
        ])
        maxl = max(np.max(np.char.str_len(col)), 10)
        msg = '\n'.join([' '.join([cc.ljust(maxl) for cc in col]),
                         ' '.join(['-'*maxl]*6)])
        print(msg)

    # ------------
    # Main loop

    end = '\r'
    t0 = dtm.datetime.now()     # DB
    for ii in range(nspect):

        if verbose == 3:
            msg = "\nspect {} / {}".format(ii+1, nspect)
            print(msg)

        try:
            dti = None
            t0i = dtm.datetime.now()     # DB
            if not dinput['valid']['indt'][ii]:
                validity[ii] = -1
                continue

            # optimization
            res = scpopt.least_squares(
                func_cost,
                x0[ii, indx],
                jac=func_jac,
                bounds=bounds[:, indx],
                method=method,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                x_scale=1.0,
                f_scale=1.0,
                loss=loss,
                diff_step=None,
                tr_solver=tr_solver,
                tr_options=tr_options,
                jac_sparsity=None,
                max_nfev=max_nfev,
                verbose=verbscp,
                args=(),
                kwargs={
                    'data': datacost[ii, :],
                    'scales': scales[ii, :],
                    'const': const[ii, :],
                    'indok': dprepare['indok_bool'][ii, :],
                },
            )
            dti = (dtm.datetime.now() - t0i).total_seconds()

            if chain is True and ii < nspect-1:
                x0[ii+1, indx] = res.x

            # cost, message, time
            success[ii] = res.success
            cost[ii] = res.cost
            nfev[ii] = res.nfev
            message[ii] = res.message
            time[ii] = round(
                (dtm.datetime.now()-t0i).total_seconds(),
                ndigits=3,
            )
            sol_x[ii, indx] = res.x
            sol_x[ii, ~indx] = const[ii, :] / scales[ii, ~indx]

        except Exception as err:
            if strict:
                raise err
            else:
                errmsg[ii] = str(err)
                validity[ii] = -2

        # verbose
        if verbose in [1, 2]:
            if validity[ii] == 0:
                col = np.char.array([
                    '{} / {}'.format(ii+1, nspect),
                    '{}'.format(dti),
                    '{:5.3e}'.format(res.cost),
                    str(res.nfev),
                    str(res.njev),
                    res.message,
                ])
            else:
                col = np.char.array([
                    '{} / {}'.format(ii+1, nspect),
                    '{}'.format(dti),
                    ' - ', ' - ', ' - ',
                    errmsg[ii],
                ])
            msg = ' '.join([cc.ljust(maxl) for cc in col])
            if verbose == 1:
                if ii == nspect - 1:
                    end = '\n'
                print(msg, end=end, flush=True)
            else:
                print(msg, end='\n')

    # ---------------------------
    # Reshape in case of same_spectrum
    if dinput['same_spectrum'] is True:
        nspect0 = dinput['same_spectrum_nspect']

        def reshape_custom(aa, nspect0=nspect0):
            return aa.reshape((nspect0, int(aa.size/nspect0)))

        nlamb = int(lamb.size / nspect0)
        nlines = int((sol_detail.shape[1]-1)/nspect0)
        lamb = lamb[:nlamb]

        nxbis = int(
            dind['bck_amp']['x'].size
            + dind['bck_rate']['x'].size
            + (dind['amp']['x'].size + dind['width']['x'].size)/nspect0
            + dind['shift']['x'].size
        )
        if dinput['double'] is not False:
            if dinput['double'] is True:
                nxbis += 2
            else:
                nxbis += (
                    dinput['double'].get('dratio') is not None
                    + dinput['double'].get('dshift') is not None
                )
        nba = dind['bck_amp']['x'].size
        nbr = dind['bck_rate']['x'].size
        nb = nba+nbr
        na = int(dind['amp']['x'].size/nspect0)
        nw = int(dind['width']['x'].size/nspect0)
        ns = dind['shift']['x'].size
        x2 = np.full((nspect0, nxbis), np.nan)
        x2[:, :nba] = sol_x[0, dind['bck_amp']['x']][None, :]
        x2[:, nba:nbr] = sol_x[0, dind['bck_rate']['x']][None, :]
        x2[:, nb:nb+na] = reshape_custom(sol_x[0, dind['amp']['x']])
        x2[:, nb+na:nb+na+nw] = reshape_custom(sol_x[0, dind['width']['x']])
        x2[:, nb+na+nw:nb+na+nw+ns] = sol_x[:, dind['shift']['x']]
        if dinput['double'] is True:
            x2[:, dind['dratio']['x']] = sol_x[:, dind['dratio']['x']]
            x2[:, dind['dshift']['x']] = sol_x[:, dind['dshift']['x']]
        import pdb; pdb.set_trace()     # DB
        sol_x = x2

    # ---------------------------
    # Isolate dratio and dshift
    dratio, dshift = None, None
    if dinput['double'] is not False:
        if dinput['double'] is True:
            dratio = (
                sol_x[:, dind['dratio']['x']] * scales[:, dind['dratio']['x']]
            )
            dshift = (
                sol_x[:, dind['dshift']['x']] * scales[:, dind['dshift']['x']]
            )
        else:
            if dinput['double'].get('dratio') is None:
                dratio = (
                    sol_x[:, dind['dratio']['x']]
                    * scales[:, dind['dratio']['x']]
                )
            else:
                dratio = np.full((nspect,), dinput['double']['dratio'])

            if dinput['double'].get('dshift') is None:
                dshift = (
                    sol_x[:, dind['dshift']['x']]
                    * scales[:, dind['dshift']['x']]
                )
            else:
                dshift = np.full((nspect,), dinput['double']['dshift'])

    if verbose > 0:
        dt = (dtm.datetime.now() - t0).total_seconds()
        msg = (
            "Total computation time:"
            + "\t{} s for {} spectra ({} s per spectrum)".format(
                round(dt, ndigits=3),
                nspect,
                round(dt/nspect, ndigits=3),
            )
        )
        print(msg)

    # ---------------------------
    # Format output as dict
    dfit = {
        'dinput': dinput,
        'scales': scales, 'x0': x0, 'bounds': bounds,
        'jac': jac, 'sol_x': sol_x,
        'dratio': dratio, 'dshift': dshift,
        'indx': indx,
        'time': time, 'success': success,
        'validity': validity,
        'dvalidity': _DVALIDITY,
        'errmsg': np.array(errmsg),
        'cost': cost, 'nfev': nfev, 'msg': np.array(message),
        'const': const,
        'xtol': xtol, 'ftol': ftol, 'gtol': gtol,
    }
    return dfit


def multigausfit2d_from_dlines(
    dinput=None,
    method=None, tr_solver=None, tr_options=None,
    xtol=None, ftol=None, gtol=None,
    max_nfev=None, chain=None, verbose=None,
    loss=None, jac=None,
    strict=None,
):
    """ Solve multi_gaussian fit in 1d from dlines

    If double is True, all lines are double with common shift and ratio

    Unknowns are:
        x = [bck, w0, v0, c00, c01, ..., c0n, w1, v1, c10, c11, ..., c1N, ...]

        - bck : constant background
        - wi  : spectral width of a group of lines (ion): wi^2 = 2kTi / m*c**2
                This way, it is dimensionless
        - vni : normalised velicity of the ion: vni = vi / c
        - cij : normalised coef (intensity) of line: cij = Aij

    Scaling is done so each quantity is close to unity:
        - bck: np.mean(data[data < mean(data)/2])
        - wi : Dlamb / 20
        - vni: 10 km/s
        - cij: np.mean(data)

    """

    # ---------------------------
    # Check format options
    (
        chain, method, tr_solver, tr_options,
        xtol, ftol, gtol, loss, max_nfev,
        verbose, verbscp, strict,
    ) = _checkformat_options(
         chain, method, tr_solver, tr_options,
         xtol, ftol, gtol, loss, max_nfev, verbose, strict,
    )

    # ---------------------------
    # Load dinput if necessary
    dinput = _checkformat_dinput(dinput)
    dprepare, dind = dinput['dprepare'], dinput['dind']
    nspect = dprepare['data'].shape[0]

    # ---------------------------
    # lamb and phi (symmetry axis?)
    lamb = dprepare['lamb']
    if dinput['symmetry'] is True:
        phi = np.abs(dprepare['phi'] - np.nanmean(dinput['symmetry_axis']))
    else:
        phi = dprepare['phi']

    # ---------------------------
    # Get scaling, x0, bounds from dict
    scales = _dict2vector_dscalesx0bounds(
        dd=dinput['dscales'], dd_name='dscales', dinput=dinput,
    )
    # muliply amplitudes by scales bs in [0, 1]
    scales[:, dinput['dind']['amp']['x']] *= dinput['dscales']['bs'][..., None]

    x0 = _dict2vector_dscalesx0bounds(
        dd=dinput['dx0'], dd_name='dx0', dinput=dinput,
    )
    boundmin = _dict2vector_dscalesx0bounds(
        dd=dinput['dbounds']['min'], dd_name="dbounds['min']", dinput=dinput,
    )
    boundmax = _dict2vector_dscalesx0bounds(
        dd=dinput['dbounds']['max'], dd_name="dbounds['max']", dinput=dinput,
    )
    bounds = np.array([boundmin[0, :], boundmax[0, :]])

    # ---------------------------
    # Prepare const

    const = _dict2vector_dscalesx0bounds(
        dd=dinput['dconstants'], dd_name='dconstants', dinput=dinput,
    )

    # -----------------------
    # Prepare indbs

    # bck_amp, bck_rate, all amp, width and shift are affected
    indbs = dinput['valid']['indbs']
    indbsfull = ~np.tile(indbs, dinput['dind']['nvar_bs'])
    ndiff = const.shape[1] - indbsfull.shape[1]
    assert ndiff <= 2
    if ndiff > 0:
        indbsfull = np.concatenate(
            (indbsfull, np.zeros((nspect, ndiff), dtype=bool)),
            axis=1,
        )
    const[indbsfull] = 1.

    # ---------------------------
    # Separate free from constant parameters

    # const = nan => x valid
    indx = np.isnan(const)
    x0[~indx] = const[~indx] / scales[~indx]

    # -----------------------
    # Prepare flattened data

    indok_all = np.any(dprepare['indok_bool'], axis=0)
    indok_flat = dprepare['indok_bool'][:, indok_all].reshape((nspect, -1))
    data_flat = dprepare['data'][:, indok_all].reshape((nspect, -1))
    phi_flat = phi[indok_all].ravel()
    lamb_flat = lamb[indok_all].ravel()

    # prepare lambrel and lambn
    lambrel_flat = lamb_flat - dinput['lambmin_bck']
    lambn_flat = lamb_flat[:, None] / dinput['lines'][None, ...]

    # jac0
    jac0 = np.zeros((phi_flat.size, dind['sizex']), dtype=float)

    # libs
    libs = np.array([
        (phi_flat >= dinput['knots_mult'][ii])
        & (phi_flat <= dinput['knots_mult'][ii + dinput['nknotsperbs'] - 1])
        for ii in range(dinput['nbs'])
    ])

    # ---------------------------
    # Get function, cost function and jacobian

    (
        func_cost, func_jac,
    ) = _funccostjac.multigausfit2d_from_dlines_funccostjac(
        phi_flat=phi_flat,
        dinput=dinput, dind=dind, jac=jac,
    )[2:]

    # ---------------------------
    # Prepare output
    sol_x = np.full((nspect, dind['sizex']), np.nan)
    sol_x[~indx] = const[~indx] / scales[~indx]
    success = np.full((nspect,), np.nan)
    time = np.full((nspect,), np.nan)
    cost = np.full((nspect,), np.nan)
    nfev = np.full((nspect,), np.nan)
    validity = np.zeros((nspect,), dtype=int)
    saturated = np.zeros((nspect, dind['sizex']), dtype=bool)
    message = ['' for ss in range(nspect)]
    errmsg = ['' for ss in range(nspect)]

    indamp = np.zeros((dind['sizex'],), dtype=bool)
    indamp[dinput['dind']['amp']['x'].T.ravel()] = True

    # Prepare msg
    if verbose in [1, 2]:
        col = np.char.array(['Spect', 'time (s)', 'cost',
                             'nfev', 'njev', 'msg'])
        maxl = max(np.max(np.char.str_len(col)), 10)
        msg = '\n'.join([
            ' '.join([cc.ljust(maxl) for cc in col]),
            ' '.join(['-'*maxl]*6),
        ])
        print(msg)

    # ------------
    # Main loop

    end = '\r'
    t0 = dtm.datetime.now()     # DB
    for ii in range(nspect):

        if verbose == 3:
            msg = "\nSpect {} / {}".format(ii+1, nspect)
            print(msg)

        try:
            dti = None
            t0i = dtm.datetime.now()     # DB
            if not dinput['valid']['indt'][ii]:
                validity[ii] = -1
                continue

            deltab = bounds[1, indx[ii, :]] - bounds[0, indx[ii, :]]

            # optimization
            res = scpopt.least_squares(
                func_cost,
                x0[ii, indx[ii, :]],
                jac=func_jac,
                bounds=bounds[:, indx[ii, :]],
                method=method,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                x_scale=1.0,
                f_scale=1.0,
                loss=loss,
                diff_step=None,
                tr_solver=tr_solver,
                tr_options=tr_options,
                jac_sparsity=None,
                max_nfev=max_nfev,
                verbose=verbscp,
                args=(),
                kwargs={
                    'indx': indx[ii, :],
                    'data_flat': data_flat[ii, indok_flat[ii]],
                    'scales': scales[ii, :],
                    'const': const[ii, ~indx[ii, :]],
                    'indok_flat': indok_flat[ii],
                    'phi_flat': phi_flat[indok_flat[ii]],
                    'lambrel_flat': lambrel_flat[indok_flat[ii]],
                    'lambn_flat': lambn_flat[indok_flat[ii], :],
                    'jac0': jac0[indok_flat[ii], :],
                    'libs': [ibs[indok_flat[ii]] for ibs in libs],
                }
            )
            dti = (dtm.datetime.now() - t0i).total_seconds()

            if chain is True and ii < nspect-1:
                x0[ii+1, indx[ii, :]] = res.x

            # cost, message, time
            success[ii] = res.success
            cost[ii] = res.cost
            nfev[ii] = res.nfev
            message[ii] = res.message
            time[ii] = round(
                (dtm.datetime.now()-t0i).total_seconds(),
                ndigits=3,
            )
            sol_x[ii, indx[ii, :]] = res.x

            # detect saturated values
            # amp at 0 are ok
            saturated[ii, indx[ii, :]] = (
                res.x > bounds[1, indx[ii, :]] - deltab*1e-4
            )
            saturated[ii, indx[ii, :] & indamp] |= (
                res.x[indamp[indx[ii, :]]] < 0.
            )
            saturated[ii, indx[ii, :] & (~indamp)] |= (
                res.x[(~indamp)[indx[ii, :]]]
                < (
                    bounds[0, indx[ii, :] & (~indamp)]
                    + 1.e-4 * deltab[(~indamp)[indx[ii, :]]]
                )
            )

        except Exception as err:
            if strict:
                raise err
            else:
                errmsg[ii] = str(err)
                validity[ii] = -2

        # verbose
        if verbose in [1, 2]:
            if validity[ii] == 0:
                col = np.char.array([
                    '{} / {}'.format(ii+1, nspect),
                    '{}'.format(dti),
                    '{:5.3e}'.format(res.cost),
                    str(res.nfev),
                    str(res.njev),
                    res.message,
                ])
            else:
                col = np.char.array([
                    '{} / {}'.format(ii+1, nspect),
                    '{}'.format(dti),
                    ' - ', ' - ', ' - ',
                    errmsg[ii],
                ])
            msg = ' '.join([cc.ljust(maxl) for cc in col])
            if verbose == 1:
                if ii == nspect - 1:
                    end = '\n'
                print(msg, end=end, flush=True)
            else:
                print(msg, end='\n')

    # ---------------
    # Display saturated values
    dsat = None
    if np.any(saturated):
        lksat = [
            'bck_amp', 'bck_rate',
            'amp', 'width', 'shift',
            'dshift', 'dratio',
        ]
        dsat = {
            k0: {
                'ind': np.sum(saturated[:, dind[k0]['x']], axis=1),
            }
            for k0 in lksat
            if dind.get(k0) is not None
            and np.any(saturated[:, dind[k0]['x']])
        }

        for k0 in dsat.keys():
            if k0 in ['amp', 'width', 'shift']:
                dsat[k0]['str'] = {}
                for ik, k1 in enumerate(dinput[k0]['keys']):
                    indk1 = dsat[k0]['ind'][:, ik] > 0
                    if np.any(indk1):
                        dsat[k0]['str'][k1] = (
                            indk1.sum(),
                            np.mean(dsat[k0]['ind'][indk1, ik]),
                        )
            else:
                indk1 = dsat[k0]['ind'][:, 0] > 0
                if np.any(indk1):
                    dsat[k0]['str'] = {
                        '': (indk1.sum(), np.mean(dsat[k0]['ind'][indk1, 0])),
                    }

        lstr = [
            "\n".join([
                f"\t{k0} {k1}: {v1[0]} / {nspect} "
                f"(mean {v1[1]} / {dinput['nbs']} bsplines)"
                for k1, v1 in v0['str'].items()
            ])
            for k0, v0 in dsat.items()
        ]
        msg = (
            "The following variables seem to have saturated:\n"
            + "\n".join(lstr)
        )
        print(msg)

    # ---------------------------
    # Isolate dratio and dshift
    dratio, dshift = None, None
    if dinput['double'] is not False:
        if dinput['double'] is True:
            dratio = (
                sol_x[:, dind['dratio']['x']] * scales[:, dind['dratio']['x']]
            )
            dshift = (
                sol_x[:, dind['dshift']['x']] * scales[:, dind['dshift']['x']]
            )
        else:
            if dinput['double'].get('dratio') is None:
                dratio = (
                    sol_x[:, dind['dratio']['x']]
                    * scales[:, dind['dratio']['x']]
                )
            else:
                dratio = np.full((nspect,), dinput['double']['dratio'])

            if dinput['double'].get('dshift') is None:
                dshift = (
                    sol_x[:, dind['dshift']['x']]
                    * scales[:, dind['dshift']['x']]
                )
            else:
                dshift = np.full((nspect,), dinput['double']['dshift'])

    if verbose > 0:
        dt = (dtm.datetime.now() - t0).total_seconds()
        msg = (
            "Total computation time:"
            + "\t{} s for {} spectra ({} s per spectrum)".format(
                round(dt, ndigits=3),
                nspect,
                round(dt/nspect, ndigits=3)
            )
        )
        print(msg)

    # ---------------------------
    # Format output as dict
    dfit = {
        'dinput': dinput,
        'scales': scales, 'x0': x0, 'bounds': bounds,
        'jac': jac, 'sol_x': sol_x,
        'dratio': dratio, 'dshift': dshift,
        'indx': indx,
        'time': time, 'success': success,
        'validity': validity,
        'dvalidity': _DVALIDITY,
        'errmsg': np.array(errmsg),
        'cost': cost, 'nfev': nfev, 'msg': np.array(message),
        'phi': phi,
        'const': const,
        'dsat': dsat,
        'xtol': xtol, 'ftol': ftol, 'gtol': gtol,
    }
    return dfit


###########################################################
###########################################################
#
#   Main fit functions
#
###########################################################
###########################################################


def fit1d(
    dinput=None,
    method=None, tr_solver=None, tr_options=None,
    xtol=None, ftol=None, gtol=None,
    max_nfev=None, loss=None, chain=None,
    jac=None, verbose=None, strict=None,
    # saving
    save=None,
    name=None,
    path=None,
    # extract for plotting
    amp=None, ratio=None,
    Ti=None, width=None,
    vi=None, shift=None,
    sol_total=None, sol_detail=None, sol_lamb=None,
    # plotting
    plot=None,
    showonly=None,
    fs=None, wintit=None, tit=None, dmargin=None,
    return_dax=None,
):

    # ----------------------
    # Check / format
    if showonly is None:
        showonly = False
    if save is None:
        save = False
    if plot is None:
        plot = False
    if return_dax is None:
        return_dax = False

    # ----------------------
    # Get dinput for 1d fitting from dlines, dconstraints, dprepare...
    if not isinstance(dinput, dict):
        msg = ("Please provide a properly formatted dict of inputs!\n"
               + "fit1d() needs the problem to be given as a dinput dict\n"
               + "  => Use dinput = fit1d_dinput()")
        raise Exception(msg)

    # ----------------------
    # Perform 2d fitting
    if showonly is True:
        msg = "TBF: lambfit and spect1d not defined"
        raise Exception(msg)

        dfit1d = {
            'shift': np.zeros((1, dinput['nlines'])),
            'coefs': np.zeros((1, dinput['nlines'])),
            'lamb': lambfit,
            'data': spect1d,
            'double': False,
            'Ti': False,
            'vi': False,
            'ratio': None,
        }
    else:
        dfit1d = multigausfit1d_from_dlines(
            dinput=dinput,
            method=method,
            max_nfev=max_nfev,
            tr_solver=tr_solver,
            tr_options=tr_options,
            xtol=xtol,
            ftol=ftol,
            gtol=gtol,
            loss=loss,
            chain=chain,
            verbose=verbose,
            jac=jac,
            strict=strict,
        )

    # ----------------------
    # Optional saving
    if save is True:
        if name is None:
            name = 'custom'
        name = 'TFS_fit1d_doutput_{}_{}_tol{}.npz'.format(
            name, dinput['method'], dinput['xtol'],
        )
        if not name.endswith('.npz'):
            name = name + '.npz'
        if path is None:
            path = os.getcwd()
        pfe = os.path.join(os.path.abspath(path), name)
        np.savez(pfe, **dfit1d)
        print(f"Saved in:\n\t{pfe}")

    # ----------------------
    # Optional plotting
    if plot is True:
        dout = fit1d_extract(
            dfit1d,
            amp=amp, ratio=ratio,
            Ti=Ti, width=width,
            vi=vi, shift=shift,
            sol_total=sol_total,
            sol_detail=sol_detail,
            sol_lamb=sol_lamb,
        )
        # TBF
        dax = _plot.plot_fit1d(
            dfit1d=dfit1d, dout=dout, showonly=showonly,
            fs=fs, dmargin=dmargin,
            tit=tit, wintit=wintit,
        )

    # ----------------------
    # return
    if return_dax is True:
        return dfit1d, dax
    else:
        return dfit1d


def fit2d(
    dinput=None,
    method=None, tr_solver=None, tr_options=None,
    xtol=None, ftol=None, gtol=None,
    max_nfev=None, loss=None, chain=None,
    jac=None,
    verbose=None, strict=None,
    # saving
    save=None,
    name=None,
    path=None,
    # extract for plotting
    amp=None, ratio=None,
    Ti=None, width=None,
    vi=None, shift=None,
    sol_total=None, sol_detail=None, sol_lamb_phi=None,
    # plotting
    plot=None,
    showonly=None,
    fs=None, wintit=None, tit=None, dmargin=None,
    return_dax=None,
):

    # ----------------------
    # Check / format
    if showonly is None:
        showonly = False
    if save is None:
        save = False
    if plot is None:
        plot = False
    if return_dax is None:
        return_dax = False

    # ----------------------
    # Get dinput for 1d fitting from dlines, dconstraints, dprepare...
    if not isinstance(dinput, dict):
        msg = ("Please provide a properly formatted dict of inputs!\n"
               + "fit2d() needs the problem to be given as a dinput dict\n"
               + "  => Use dinput = fit2d_dinput()")
        raise Exception(msg)

    # ----------------------
    # Perform 2d fitting
    if showonly is True:
        msg = "TBF: lambfit and spect1d not defined"
        raise Exception(msg)

        dfit2d = {
            'shift': np.zeros((1, dinput['nlines'])),
            'coefs': np.zeros((1, dinput['nlines'])),
            'lamb': None,
            'phi': None,
            'data': spect1d,
            'double': False,
            'Ti': False,
            'vi': False,
            'ratio': None,
        }
    else:
        dfit2d = multigausfit2d_from_dlines(
            dinput=dinput,
            method=method,
            tr_solver=tr_solver,
            tr_options=tr_options,
            xtol=xtol,
            ftol=ftol,
            gtol=gtol,
            max_nfev=max_nfev,
            chain=chain,
            verbose=verbose,
            loss=loss,
            jac=jac,
            strict=strict,
        )

    # ----------------------
    # Optional saving
    if save is True:
        if name is None:
            name = 'custom'
        name = 'TFS_fit2d_doutput_{}_nbs{}_{}_tol{}.npz'.format(
            name, dinput['nbs'], dinput['method'], dinput['xtol'],
        )
        if not name.endswith('.npz'):
            name = name + '.npz'
        if path is None:
            path = os.getcwd()
        pfe = os.path.join(os.path.abspath(path), name)
        np.savez(pfe, **dfit2d)
        print(f"Saved in:\n\t{pfe}")

    # ----------------------
    # Optional plotting
    if plot is True:
        dout = fit2d_extract(
            dfit2d,
            amp=amp, ratio=ratio,
            Ti=Ti, width=width,
            vi=vi, shift=shift,
            sol_total=sol_total,
            sol_detail=sol_detail,
            sol_lamb_phi=sol_lamb_phi,
        )
        # TBF
        dax = _plot.plot_fit2d(
            dfit2d=dfit2d, dout=dout, showonly=showonly,
            fs=fs, dmargin=dmargin,
            tit=tit, wintit=wintit,
        )

    # ----------------------
    # return
    if return_dax is True:
        return dfit2d, dax
    else:
        return dfit2d


###########################################################
###########################################################
#
#   Plot fitted data from pre-computed dict of fitted results
#
###########################################################
###########################################################

def fit2d_plot(dout=None):

    # ----------------------
    # Optional plotting
    if plot is True:
        if plotmode is None:
            plotmode = 'transform'
        if indspect is None:
            indspect = 0

        if spect1d is not None:
            # Compute lambfit / phifit and spectrum1d
            if nlambfit is None:
                nlambfit = 200
            ((spect1d, fit1d), lambfit,
             phifit, _, phiminmax) = self._calc_spect1d_from_data2d(
                 [dataflat[indspect, :], dfit2d['sol_tot'][indspect, :]],
                 lambflat, phiflat,
                 nlambfit=nlambfit, nphifit=10,
                 spect1d=spect1d, mask=None, vertsum1d=False)
        else:
            fit1d, lambfit, phiminmax = None, None, None

        dax = _plot_optics.CrystalBragg_plot_data_fit2d(
            xi=xi, xj=xj, data=dfit2d['data'],
            lamb=dfit2d['lamb'], phi=dfit2d['phi'], indspect=indspect,
            indok=indok, dfit2d=dfit2d,
            dax=dax, plotmode=plotmode, angunits=angunits,
            cmap=cmap, vmin=vmin, vmax=vmax,
            spect1d=spect1d, fit1d=fit1d,
            lambfit=lambfit, phiminmax=phiminmax,
            dmargin=dmargin, tit=tit, wintit=wintit, fs=fs)
    return dax


###########################################################
###########################################################
#
#           1d vertical fitting for noise analysis
#
###########################################################
###########################################################


def get_noise_costjac(deg=None, nbsplines=None, dbsplines=None, phi=None,
                      phiminmax=None, symmetryaxis=None, sparse=None):

    if sparse is None:
        sparse = False

    if dbsplines is None:
        dbsplines = multigausfit2d_from_dlines_dbsplines(
            knots=None, deg=deg, nbsplines=nbsplines,
            phimin=phiminmax[0], phimax=phiminmax[1],
            symmetryaxis=symmetryaxis)

    def cost(x,
             km=dbsplines['knots_mult'],
             deg=dbsplines['deg'],
             data=0., phi=phi):
        return scpinterp.BSpline(km, x, deg,
                                 extrapolate=False, axis=0)(phi) - data

    jac = np.zeros((phi.size, dbsplines['nbs']), dtype=float)
    km = dbsplines['knots_mult']
    kpb = dbsplines['nknotsperbs']
    lind = [(phi >= km[ii]) & (phi < km[ii+kpb-1])
            for ii in range(dbsplines['nbs'])]
    if sparse is True:
        def jac_func(x, jac=jac, km=km, data=None,
                     phi=phi, kpb=kpb, lind=lind):
            for ii in range(x.size):
                jac[lind[ii], ii] = scpinterp.BSpline.basis_element(
                    km[ii:ii+kpb], extrapolate=False)(phi[lind[ii]])
            return scpsparse.csr_matrix(jac)
    else:
        def jac_func(x, jac=jac, km=km, data=None,
                     phi=phi, kpb=kpb, lind=lind):
            for ii in range(x.size):
                jac[lind[ii], ii] = scpinterp.BSpline.basis_element(
                    km[ii:ii+kpb], extrapolate=False)(phi[lind[ii]])
            return jac
    return cost, jac_func


def _basic_loop(ilambu=None, ilamb=None, phi=None, data=None, mask=None,
                domain=None, nbs=None, dbsplines=None, nspect=None,
                method=None, tr_solver=None, tr_options=None, loss=None,
                xtol=None, ftol=None, gtol=None, max_nfev=None, verbose=None):

    # ---------------
    # Check inputs
    if method is None:
        method = _METHOD
    assert method in ['trf', 'dogbox', 'lm'], method
    if tr_solver is None:
        tr_solver = None
    if tr_options is None:
        tr_options = {}
    if xtol is None:
        xtol = _TOL2D['x']
    if ftol is None:
        ftol = _TOL2D['f']
    if gtol is None:
        gtol = _TOL2D['g']
    if loss is None:
        loss = _LOSS
    if max_nfev is None:
        max_nfev = None

    x0 = 1. - (2.*np.arange(nbs)/nbs - 1.)**2

    # ---------------
    # Prepare outputs
    dataint = np.full((nspect, ilambu.size), np.nan)
    fit = np.full(data.shape, np.nan)
    indsort = np.zeros((2, phi.size), dtype=int)
    indout_noeval = np.zeros(phi.shape, dtype=bool)
    chi2n = np.full((nspect, ilambu.size), np.nan)
    chi2_meandata = np.full((nspect, ilambu.size), np.nan)

    # ---------------
    # Main loop
    i0, indnan = 0, []
    for jj in range(ilambu.size):
        ind = ilamb == ilambu[jj]
        nind = ind.sum()
        isort = i0 + np.arange(0, nind)

        # skips cases with no points
        if not np.any(ind):
            continue

        inds = np.argsort(phi[ind])
        inds_rev = np.argsort(inds)
        indsort[0, isort] = ind.nonzero()[0][inds]
        indsort[1, isort] = ind.nonzero()[1][inds]

        phisort = phi[indsort[0, isort], indsort[1, isort]]
        datasort = data[:, indsort[0, isort], indsort[1, isort]]
        dataint[:, jj] = np.nanmean(datasort, axis=1)

        # skips cases with to few points
        indok = ~np.any(np.isnan(datasort), axis=0)
        if mask is not None:
            indok &= mask[indsort[0, isort], indsort[1, isort]]

        # Check there are enough phi vs bsplines
        indphimin = np.searchsorted(np.linspace(domain['phi']['minmax'][0],
                                                domain['phi']['minmax'][1],
                                                nbs + 1),
                                    phisort[indok])
        if np.unique(indphimin).size < nbs:
            indout_noeval[ind] = True
            continue
        indout_noeval[ind] = ~indok[inds_rev]

        # get bsplines func
        func_cost, func_jac = get_noise_costjac(phi=phisort[indok],
                                                dbsplines=dbsplines,
                                                sparse=False,
                                                symmetryaxis=False)
        for tt in range(nspect):
            if verbose > 0:
                msg = ("\tlambbin {} / {}".format(jj+1, ilambu.size)
                       + "    "
                       + "time step = {} / {}".format(tt+1, nspect))
                print(msg.ljust(50), end='\r', flush=True)

            if dataint[tt, jj] == 0.:
                continue

            datai = datasort[tt, indok] / dataint[tt, jj]
            res = scpopt.least_squares(
                func_cost, x0, jac=func_jac,
                method=method, ftol=ftol, xtol=xtol, gtol=gtol,
                x_scale='jac', f_scale=1.0, loss=loss, diff_step=None,
                tr_solver=tr_solver, tr_options={}, jac_sparsity=None,
                max_nfev=max_nfev, verbose=0, args=(),
                kwargs={'data': datai})

            # Store in original shape
            fit[tt, ind] = (
                func_cost(res.x, phi=phisort, data=0.)
                * dataint[tt, jj]
            )[inds_rev]
            chi2_meandata[tt, jj] = np.nanmean(fit[tt, ind])
            chi2n[tt, jj] = np.nanmean(func_cost(x=res.x, data=datai)**2)

        i0 += nind
        indnan.append(i0)
    return (fit, dataint, indsort, np.array(indnan), indout_noeval,
            chi2n, chi2_meandata)


def noise_analysis_2d(
    data, lamb, phi, mask=None, margin=None, valid_fraction=None,
    deg=None, knots=None, nbsplines=None, nxerrbin=None,
    nlamb=None, loss=None, max_nfev=None,
    xtol=None, ftol=None, gtol=None,
    method=None, tr_solver=None, tr_options=None,
    verbose=None, plot=None,
    ms=None, dcolor=None,
    dax=None, fs=None, dmargin=None,
    wintit=None, tit=None, sublab=None,
    save_fig=None, name_fig=None, path_fig=None, fmt=None,
    return_dax=None,
):

    # -------------
    # Check inputs
    if not isinstance(nbsplines, int):
        msg = "Please provide a (>0) integer value for nbsplines"
        raise Exception(msg)

    if deg is None:
        deg = 2
    if plot is None:
        plot = True
    if verbose is None:
        verbose = 1
    if return_dax is None:
        return_dax = False

    c0 = lamb.shape == phi.shape == data.shape[1:]
    if c0 is not True:
        msg = (
            "input data, lamb, phi are non-conform!\n"
            + "\t- expected lamb.shape == phi.shape == data.shape[1:]\n"
            + "\t- provided:\n"
            + "\t\tlamb.shape = {}\n".format(lamb.shape)
            + "\t\tphi.shape = {}\n".format(phi.shape)
            + "\t\tdata.shape = {}\n".format(data.shape)
        )
        raise Exception(msg)

    nspect = data.shape[0]
    domain = {'lamb': {'minmax': [np.nanmin(lamb), np.nanmax(lamb)]},
              'phi': {'minmax': [np.nanmin(phi), np.nanmax(phi)]}}

    if nlamb is None:
        if lamb.ndim == 2:
            nlamb = lamb.shape[0]
        else:
            msg = ("Please provide a value for nlamb (nb of bins)!")
            raise Exception(msg)
    nlamb = int(nlamb)

    # -------------
    # lamb binning
    lambedges = np.linspace(domain['lamb']['minmax'][0],
                            domain['lamb']['minmax'][1], nlamb+1)
    ilamb = np.searchsorted(lambedges, lamb)
    ilambu = np.unique(ilamb)

    # -------------
    # bspline dict and plotting utilities
    dbsplines = multigausfit2d_from_dlines_dbsplines(
        knots=None, deg=deg, nbsplines=nbsplines,
        phimin=domain['phi']['minmax'][0],
        phimax=domain['phi']['minmax'][1],
        symmetryaxis=False)

    # plotting utils
    bs_phi = np.linspace(domain['phi']['minmax'][0],
                         domain['phi']['minmax'][1], 101)
    bs_val = np.array([
        scpinterp.BSpline.basis_element(
            dbsplines['knots_mult'][ii:ii+dbsplines['nknotsperbs']],
            extrapolate=False)(bs_phi)
        for ii in range(nbsplines)]).T

    # -------------
    # Perform fits
    (fit, dataint, indsort, indnan, indout_noeval,
     chi2n, chi2_meandata) = _basic_loop(
        ilambu=ilambu, ilamb=ilamb, phi=phi, data=data, mask=mask,
        domain=domain, nbs=nbsplines, dbsplines=dbsplines, nspect=nspect,
        method=method, tr_solver=tr_solver, tr_options=tr_options, loss=loss,
        xtol=xtol, ftol=ftol, gtol=gtol,
        max_nfev=max_nfev, verbose=verbose)

    # -------------
    # Identify outliers with respect to noise model
    (mean, var, xdata, const,
     indout_var, _, margin, valid_fraction) = get_noise_analysis_var_mask(
         fit=fit, data=data, mask=(mask & (~indout_noeval)),
         margin=margin, valid_fraction=valid_fraction)

    # Safety check
    if mask is None:
        indout_mask = np.zeros(lamb.shape, dtype=bool)
    else:
        indout_mask = ~mask
    indout_noeval[~mask] = False
    indout_tot = np.array([~mask,
                           indout_noeval,
                           np.any(indout_var, axis=0)])
    c0 = np.all(np.sum(indout_tot.astype(int), axis=0) <= 1)
    if not c0:
        msg = "Overlapping indout!"
        raise Exception(msg)

    indin = ~np.any(indout_tot, axis=0)

    # -------------
    # output dict
    dnoise = {
        'data': data, 'phi': phi, 'fit': fit,
        'chi2n': chi2n, 'chi2_meandata': chi2_meandata, 'dataint': dataint,
        'domain': domain, 'indin': indin, 'indout_mask': indout_mask,
        'indout_noeval': indout_noeval, 'indout_var': indout_var,
        'mask': mask, 'ind_noeval': None,
        'indsort': indsort, 'indnan': np.array(indnan),
        'nbsplines': nbsplines, 'bs_phi': bs_phi, 'bs_val': bs_val,
        'deg': deg, 'lambedges': lambedges, 'deg': deg,
        'ilamb': ilamb, 'ilambu': ilambu,
        'var_mean': mean, 'var': var, 'var_xdata': xdata,
        'var_const': const, 'var_margin': margin,
        'var_fraction': valid_fraction}

    # Plot
    if plot is True:
        try:
            dax = _plot.plot_noise_analysis(
                dnoise=dnoise,
                ms=ms, dcolor=dcolor,
                dax=dax, fs=fs, dmargin=dmargin,
                wintit=wintit, tit=tit, sublab=sublab,
                save=save_fig, name=name_fig, path=path_fig, fmt=fmt)
        except Exception as err:
            msg = ("Plotting failed: {}".format(str(err)))
            warnings.warn(msg)
    if return_dax is True:
        return dnoise, dax
    else:
        return dnoise


def noise_analysis_2d_scannbs(
    data, lamb, phi, mask=None, nxerrbin=None,
    deg=None, knots=None, nbsplines=None, lnbsplines=None,
    nlamb=None, loss=None, max_nfev=None,
    xtol=None, ftol=None, gtol=None,
    method=None, tr_solver=None, tr_options=None,
    verbose=None, plot=None,
    dax=None, fs=None, dmargin=None,
    wintit=None, tit=None, ms=None, sublab=None,
    save_fig=None, name_fig=None, path_fig=None,
    fmt=None, return_dax=None,
):

    # -------------
    # Check inputs
    if lnbsplines is None:
        lnbsplines = np.arange(5, 21)
    else:
        lnbsplines = np.atleast_1d(lnbsplines).ravel().astype(int)
    if nbsplines is None:
        nbsplines = int(lnbsplines.size/2)
    if nbsplines is not None:
        nbsplines = np.unique(np.atleast_1d(nbsplines)).astype(int)
    nlnbs = lnbsplines.size
    if nxerrbin is None:
        nxerrbin = 100

    if deg is None:
        deg = 2
    if plot is None:
        plot = True
    if verbose is None:
        verbose = 1
    if return_dax is None:
        return_dax = False

    c0 = lamb.shape == phi.shape == data.shape[1:]
    if c0 is not True:
        msg = ("input data, lamb, phi are non-conform!\n"
               + "\t- expected lamb.shape == phi.shape == data.shape[1:]\n"
               + "\t- provided: ")
        raise Exception(msg)

    nspect = data.shape[0]
    domain = {'lamb': {'minmax': [np.nanmin(lamb), np.nanmax(lamb)]},
              'phi': {'minmax': [np.nanmin(phi), np.nanmax(phi)]}}

    if nlamb is None:
        if lamb.ndim == 2:
            nlamb = lamb.shape[0]
        else:
            msg = ("Please provide a value for nlamb (nb of bins)!")
            raise Exception(msg)
    nlamb = int(nlamb)

    # -------------
    # lamb binning
    lambedges = np.linspace(domain['lamb']['minmax'][0],
                            domain['lamb']['minmax'][1], nlamb+1)
    ilamb = np.searchsorted(lambedges, lamb)
    ilambu = np.unique(ilamb)

    # -------------
    # Perform fits
    xdata_edge = np.linspace(0, np.nanmax(data[:, mask]), nxerrbin+1)
    xdata = 0.5*(xdata_edge[1:] + xdata_edge[:-1])
    dataint = np.full((nspect, ilambu.size), np.nan)
    # fit = np.full(data.shape, np.nan)
    indsort = np.zeros((2, phi.size), dtype=int)
    # indout_noeval = np.zeros(phi.shape, dtype=bool)
    chi2n = np.full((nlnbs, nspect, ilambu.size), np.nan)
    chi2_meandata = np.full((nlnbs, nspect, ilambu.size), np.nan)
    const = np.full((nlnbs,), np.nan)
    mean = np.full((nlnbs, nxerrbin), np.nan)
    var = np.full((nlnbs, nxerrbin), np.nan)
    bs_phidata, bs_data, bs_fit, bs_indin = [], [], [], []
    for ii in range(lnbsplines.size):
        nbs = int(lnbsplines[ii])
        # -------------
        # bspline dict and plotting utilities
        dbsplines = multigausfit2d_from_dlines_dbsplines(
            knots=None, deg=deg, nbsplines=nbs,
            phimin=domain['phi']['minmax'][0],
            phimax=domain['phi']['minmax'][1],
            symmetryaxis=False,
        )

        # -------------
        # Perform fits
        if verbose > 0:
            msg = "nbs = {} ({} / {})".format(nbs, ii+1, lnbsplines.size)
            print(msg)
        (fiti, dataint, indsort, indnan, indout_noeval,
         chi2n[ii, ...], chi2_meandata[ii, ...]) = _basic_loop(
             ilambu=ilambu, ilamb=ilamb, phi=phi, data=data, mask=mask,
             domain=domain, nbs=nbs, dbsplines=dbsplines, nspect=nspect,
             method=method, tr_solver=tr_solver, tr_options=tr_options,
             loss=loss, xtol=xtol, ftol=ftol, gtol=gtol,
             max_nfev=max_nfev, verbose=verbose)

        if ii == 0:
            ind_intmax = np.unravel_index(np.argmax(dataint, axis=None),
                                          dataint.shape)

        if nbs in nbsplines:
            isi = np.split(indsort, indnan, axis=1)[ind_intmax[1]]
            bs_phidata.append(phi[isi[0], isi[1]])
            bs_data.append(data[ind_intmax[0], isi[0], isi[1]])
            bs_fit.append(fiti[ind_intmax[0], isi[0], isi[1]])
            indini = ~np.any(np.array([~mask, indout_noeval]), axis=0)
            bs_indin.append(indini[isi[0], isi[1]])

        # -------------
        # Identify outliers with respect to noise model
        (meani, vari, xdatai, consti,
         _, inderrui, _, _) = get_noise_analysis_var_mask(
             fit=fiti, data=data, xdata_edge=xdata_edge,
             mask=(mask & (~indout_noeval)),
             margin=None, valid_fraction=False)

        const[ii] = consti
        mean[ii, inderrui] = meani
        var[ii, inderrui] = vari

    # -------------
    # output dict
    dnoise_scan = {
        'data': data,
        'chi2n': chi2n, 'chi2_meandata': chi2_meandata, 'dataint': dataint,
        'domain': domain, 'lnbsplines': lnbsplines, 'nbsplines': nbsplines,
        'deg': deg, 'lambedges': lambedges, 'deg': deg,
        'ilamb': ilamb, 'ilambu': ilambu,
        'bs_phidata': bs_phidata, 'bs_data': bs_data,
        'bs_fit': bs_fit, 'bs_indin': bs_indin,
        'var_mean': mean, 'var': var, 'var_xdata': xdata,
        'var_const': const,
    }

    # Plot
    if plot is True:
        try:
            dax = _plot.plot_noise_analysis_scannbs(
                dnoise=dnoise_scan, ms=ms,
                dax=dax, fs=fs, dmargin=dmargin,
                wintit=wintit, tit=tit, sublab=sublab,
                save=save_fig, name=name_fig, path=path_fig, fmt=fmt)
        except Exception as err:
            msg = ("Plotting failed: {}".format(str(err)))
            warnings.warn(msg)

    if return_dax is True:
        return dnoise_scan, dax
    else:
        return dnoise_scan


def get_noise_analysis_var_mask(fit=None, data=None,
                                xdata_edge=None, nxerrbin=None,
                                valid_fraction=None,
                                mask=None, margin=None):
    if margin is None:
        margin = _SIGMA_MARGIN
    if valid_fraction is None:
        valid_fraction = False
    if nxerrbin is None:
        nxerrbin = 100

    err = fit - data
    if mask is None:
        mask = np.ones(err.shape[1:], dtype=bool)
    if xdata_edge is None:
        xdata_edge = np.linspace(0, np.nanmax(fit[:, mask]), nxerrbin)
    inderr = np.searchsorted(xdata_edge[1:-1], fit[:, mask])
    inderru = np.unique(inderr[~np.isnan(err[:, mask])])
    xdata = 0.5*(xdata_edge[1:] + xdata_edge[:-1])[inderru]
    mean = np.full((inderru.size,), np.nan)
    var = np.full((inderru.size,), np.nan)
    nn = np.full((inderru.size,), np.nan)
    for ii in range(inderru.size):
        ind = inderr == inderru[ii]
        indok = ~np.isnan(err[:, mask][ind])
        nn[ii] = np.sum(indok)
        mean[ii] = np.nanmean(err[:, mask][ind])
        var[ii] = nn[ii] * np.nanmean(err[:, mask][ind]**2) / (nn[ii] - 1)

    # fit sqrt on sigma (weight by log10 to take into account diff. nb. of pts)
    const = np.nansum(
        (np.log10(nn)*np.sqrt(var / xdata)) / np.nansum(np.log10(nn))
    )

    # indout
    indok = (~np.isnan(err)) & mask[None, ...]
    indout = np.zeros(err.shape, dtype=bool)
    indout[indok] = (np.abs(err[indok])
                     > margin*const*np.sqrt(np.abs(fit[indok])))
    if valid_fraction is not False:
        indout = np.sum(indout, axis=0)/float(indout.shape[0]) > valid_fraction
    return mean, var, xdata, const, indout, inderru, margin, valid_fraction
