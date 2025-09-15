

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ##################################################################
# ##################################################################
#               debug singular matrix
# ##################################################################


def _debug_singular(
    Tn=None,
    TTn=None,
    mu0=None,
    R=None,
    det=None,
    # debug
    debug=None,
    key_diag=None,
    key_matrix=None,
    key_bs=None,
    key_data=None,
    operator=None,
    algo=None,
    it=None,
    # unused
    **kwdargs,
):

    # --------------
    # prepare figure
    # --------------

    fontsize = 14
    dmargin = {
        'left': 0.05, 'right': 0.98,
        'bottom': 0.05, 'top': 0.90,
        'hspace': 0.2, 'wspace': 0.2,
    }

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(ncols=2, nrows=2, **dmargin)

    tit = (
        "Debug singular inversion matrix\n"
        f"key_diag = {key_diag}\n"
        f"key_data = {key_data}\n"
        f"key_matrix = {key_matrix}\n"
        f"key_bs = {key_bs}\n"
        f"operator = {operator}\n"
        f"algo = {algo}\n"
        f"it = {it}\n"
    )
    fig.suptitle(tit, size=fontsize+2, fontweight='bold')

    # ----------
    # Tn

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(
        Tn,
        aspect='auto',
        interpolation='nearest',
    )
    plt.colorbar(im)
    ax0.set_title('Tn', size=fontsize, fontweight='bold')

    # ----------
    # TTn

    ax1 = fig.add_subplot(gs[1, 0])
    im = ax1.imshow(
        TTn,
        aspect='auto',
        interpolation='nearest',
    )
    plt.colorbar(im)
    ax1.set_title('TTn', size=fontsize, fontweight='bold')

    # ----------
    # mu0*R

    ax = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    im = ax.imshow(
        mu0*R,
        aspect='auto',
        interpolation='nearest',
    )
    plt.colorbar(im)
    ax.set_title(f'{mu0:4.3e} * R', size=fontsize, fontweight='bold')

    # -----------
    # TTn + mu0*R

    ax = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax1)
    im = ax.imshow(
        TTn + mu0*R,
        aspect='auto',
        interpolation='nearest',
    )
    plt.colorbar(im)
    rank = np.linalg.matrix_rank(TTn + mu0*R)
    tit = (
        "TTn + mu0*R\n"
        f"\ndet(TTn + mu*0R) = {det}\n"
        f"rank(TTn + mu0*R) = {rank} / {TTn.shape[1]}"
    )
    ax.set_title('TTn + mu0R', size=fontsize, fontweight='bold')

    return
