

import numpy as np
try:
    from sksparse.cholmod import cholesky
except Exception:
    cholesky = False


import tomotok.core as tmtkc


# #############################################################################
# #############################################################################
#           Define DALGO for tomotok
# #############################################################################


def _get_sparse_from_docstr(doc, k0=None):
    gmat = [ss for ss in doc if ss.strip().startswith('gmat : ')]
    if len(gmat) != 1:
        msg = (
            f"tomotok class {k0}\n"
            "No/too many matches found in docstr for 'gmat : '\n"
            f"\t- matches: {gmat}"
        )
        raise Exception(msg)
    if any([ss in gmat[0] for ss in ['scipy', 'sparse', 'csr_matrix']]):
        sparse = True
    elif any([ss in gmat[0] for ss in ['np.ndarray', 'numpy']]):
        sparse = False
    else:
        msg = (
            f"tomotok class {k0}\n"
            f"Unreckognized type of gmat: {gmat}"
        )
        raise Exception(msg)
    return sparse


def get_dalgo():

    # ----------------
    # Initialize dict

    dalgo = dict.fromkeys([
        ss for ss in dir(tmtkc.inversions)
        if type(getattr(tmtkc.inversions, ss)) == type
        and not (cholesky is False and ss == 'CholmodMfr')
        and 'bob' not in ss.lower()
    ])

    # ---------
    # Complete

    for k0 in dalgo.keys():

        # family
        fam = 'Non-regularized' if 'bob' in k0.lower() else 'Phillips-Tikhonov'

        # reg. operator
        if 'mfr' in k0.lower():
            regoper = 'MinFisher'
        elif 'algebraic' in k0.lower():
            regoper = 'any linear'
        elif 'bob' in k0.lower():
            regoper = False

        # reg. parameter
        if 'algebraic' in k0.lower():
            regparam = 'linear estimate'
        else:
            regparam = ''

        # matrix decomposition
        if 'chol' in k0.lower():
            decomp = 'cholesky'
        elif 'svd' in k0.lower():
            decomp = 'svd'
        elif 'Gev' in k0:
            decomp = 'Gen. Eigen val.'
        else:
            decomp = ''

        # positivity constraint
        pos = False

        # sparse
        doc = getattr(tmtkc.inversions, k0).__call__.__doc__.split('\n')
        sparse = _get_sparse_from_docstr(doc, k0=k0)

        # fill
        dalgo[k0] = {
            'source': 'tomotok',
            'family': fam,
            'reg_operator': regoper,
            'reg_param': regparam,
            'decomposition': decomp,
            'positive': pos,
            'sparse': sparse,
            'isotropic': True,
            'func': k0,
        }

    return dalgo


# #############################################################################
# #############################################################################
#           Define functions to be called by tofu
# #############################################################################


def SvdFastAlgebraic(
    sig_norm=None,
    gmat_norm=None,
    deriv=None,
    method=None,
    num=None,
    # additional
    nchan=None,
    **kwdargs,
):

    # solve
    solver = tmtkc.SvdFastAlgebraic()
    sol = solver.invert(
        sig_norm,
        gmat_norm,
        deriv,
        method=None,
        num=None,
    )

    # compute residue
    chi2n = np.sum((gmat_norm.dot(sol) - sig_norm)**2) / nchan
    reg = sol.dot(deriv.dot(sol))

    return sol, chi2n, reg


def GevFastAlgebraic(
    sig_norm=None,
    gmat_norm=None,
    deriv=None,
    method=None,
    num=None,
    # additional
    nchan=None,
    **kwdargs,
):
    """
    Expects deriv to be provided as tgradR.dot(gradR) + tgradZ.dot(gradZ)

    """

    # solve
    solver = tmtkc.GevFastAlgebraic()
    sol = solver.invert(
        sig_norm,
        gmat_norm,
        deriv,
        method=None,
        num=None,
    )

    # compute residue
    chi2n = np.sum((gmat_norm.dot(sol) - sig_norm)**2) / nchan
    reg = sol.dot(deriv.dot(sol))

    return sol, chi2n, reg


def Mfr(
    sig_norm=None,
    gmat_norm=None,
    deriv=None,
    method=None,
    num=None,
    # additional
    nchan=None,
    **kwdargs,
):
    """
    Expects deriv to be provided as a tuple (gradR, gradZ)

    """

    # solve
    solver = tmtkc.Mfr()
    sol = solver.invert(
        sig_norm,
        gmat_norm,
        deriv,          # here must be provided as (gradR, gradZ)
        w_factor=None,
        mfi_num=3,
        bounds=(-15, 0),
        iter_max=10,
        w_max=1,
        danis=0,
    )

    # compute residue
    chi2n = np.sum((gmat_norm.dot(sol) - sig_norm)**2) / nchan
    reg = sol.dot(deriv[0][0].dot(sol)) + sol.dot(deriv[0][1].dot(sol))

    return sol, chi2n, reg


def CholmodMfr(
    sig_norm=None,
    gmat_norm=None,
    deriv=None,
    method=None,
    num=None,
    # additional
    nchan=None,
    **kwdargs,
):
    """
    Expects deriv to be provided as a tuple (gradR, gradZ)

    """

    # solve
    solver = tmtkc.CholmodMfr()
    sol = solver.invert(
        sig_norm,
        gmat_norm,
        deriv,          # here must be provided as (gradR, gradZ)
        w_factor=None,
        mfi_num=3,
        bounds=(-15, 0),
        iter_max=10,
        w_max=1,
        danis=0,
    )

    # compute residue
    chi2n = np.sum((gmat_norm.dot(sol) - sig_norm)**2) / nchan
    reg = sol.dot(deriv[0][0].dot(sol)) + sol.dot(deriv[0][1].dot(sol))

    return sol, chi2n, reg
