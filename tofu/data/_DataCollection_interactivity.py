


# #############################################################################
# #############################################################################
#           Update number of visible indices
# #############################################################################


def _get_nn_ii_group(
    nmax=None,
    nmaxcur=None,
    indcur=None,
    ctrl=None,
    shift=None,
):
    if shift and nmaxcur == nmax:
        msg = "Max nb. of plots reached for group '{group}': {nmax}"
        Exception(msg)

    if ctrl:
        nn = 1
        ii = 0
    elif shift:
        nn = int(nmaxcur) + 1
        ii = nn
    else:
        nn = int(nmaxcur)
        ii = int(indcur)
    return nn, ii


def _update_indices_nb(group=None, dinter=None, ctrl=None, shift=None):
    igx = dinter['lgroup'].index(group)
    dinter['nmaxcur'][igx], dinter['indcur'][igx] = _get_nn_ii_group(
        nmax=dinter['nmax'][igx],
        nmaxcur=dinter['nmaxcur'][igx],
        indcur=dinter['indcur'][igx],
        ctrl=ctrl,
        shift=shift,
    )
