

import tomotok.core as tmtkc





# #############################################################################
# #############################################################################
#                       
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
        sparse  = True
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


    #----------------
    # Initialize dict

    dalgo = dict.fromkeys([
        ss for ss in dir(tmtkc.inversions)
        if type(getattr(tmtkc.inversions, ss)) == type
    ])

    #---------
    # Complete

    for k0 in dalgo.keys():

        # family
        fam = 'Non-regularized' if 'bob' in k0.lower() else 'Phillips-Tikhonov'

        # reg. operator
        regoper = 'MinFisher' if 'mfr' in k0.lower() else ''

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
            'reg. operator': regoper,
            'reg. param': regparam,
            'decomposition': decomp,
            'positivity': pos,
            'sparse': sparse,
            'isotropic': True,
        }

    return dalgo
