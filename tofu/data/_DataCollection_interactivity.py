

# #############################################################################
# #############################################################################
#           data of mobile based on indices
# #############################################################################


def _set_dbck(lax=None, daxes=None, dcanvas=None, dmobile=None):
    """ Update background of relevant axes (ex: in case of resizing) """

    # Make all invisible
    for k0 in lax:
        for k1 in daxes[k0]['mobile']:
            dmobile[k1]['handle'].set_visible(False)

    # Draw and reset bck
    lcan = set([daxes[k0]['canvas'] for k0 in lax])
    for k0 in lcan:
        dcanvas[k0]['handle'].draw()

    # set bck (= bbox copy)
    for k0 in lax:
        # ax.draw(self.can.renderer)
        daxes[k0]['bck'] = dcanvas[
            daxes[k0]['canvas']
        ]['handle'].copy_from_bbox(daxes[k0]['handle'].bbox)

    # Redraw
    for k0 in lax:
        for k1 in daxes[k0]['mobile']:
            dmobile[k1]['handle'].set_visible(dmobile[k1]['visible'])
            # ax.draw(self.can.renderer)

    for k0 in lcan:
        dcanvas[k0]['handle'].draw()


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
    group=None,
):
    """"""

    if shift and nmaxcur == nmax:
        msg = f"Max nb. of plots reached for group '{group}': {nmax}"
        print(msg)
        return False

    if ctrl:
        nn = 0
        ii = 0
    elif shift:
        nn = int(nmaxcur) + 1
        ii = nn - 1
    else:
        nn = int(nmaxcur)
        ii = int(indcur)
    return nn, ii


def _update_indices_nb(group=None, dgroup=None, ctrl=None, shift=None):
    """"""
    out = _get_nn_ii_group(
        nmax=dgroup[group]['nmax'],
        nmaxcur=dgroup[group]['nmaxcur'],
        indcur=dgroup[group]['indcur'],
        ctrl=ctrl,
        shift=shift,
        group=group,
    )
    if out is False:
        return False
    else:
        dgroup[group]['nmaxcur'] = out[0]
        dgroup[group]['indcur'] = out[1]


# #############################################################################
# #############################################################################
#           data of mobile based on indices
# #############################################################################


def get_fupdate(handle=None, dtype=None, norm=None, bstr=None):
    if dtype == 'xdata':
        func = lambda val, handle=handle: handle.set_xdata(val)
    elif dtype == 'ydata':
        func = lambda val, handle=handle: handle.set_ydata(val)
    elif dtype in ['data']:   # Also works for imshow
        func = lambda val, handle=handle: handle.set_data(val)
    elif dtype in ['alpha']:   # Also works for imshow
        func = lambda val, handle=handle, norm=norm: handle.set_alpha(norm(val))
    elif dtype == 'txt':
        func = lambda val, handle=handle, bstr=bstr: handle.set_text(bstr.format(val))
    return func


def _update_mobile_data(
    func=None,
    kref=None,
    kdata=None,
    iref=None,
    ddata=None,
):
    """"""

    # if handle.__class__.__name__ == 'Line2D':

    if kdata == 'index':
        func(iref)

    elif ddata[kdata]['data'].ndim == 1:
        func(ddata[kdata]['data'][iref])

    elif ddata[kdata]['data'].ndim == 2:

        idim = ddata[kdata]['ref'].index(kref)
        if idim == 0:
            func(ddata[kdata]['data'][iref, :])
        else:
            func(ddata[kdata]['data'][:, iref])

    elif ddata[kdata]['data'].ndim == 3:

        idim = ddata[kdata]['ref'].index(kref)
        if idim == 0:
            func(ddata[kdata]['data'][iref, :, :])
        elif idim == 1:
            func(ddata[kdata]['data'][:, iref, :])
        elif idim == 2:
            func(ddata[kdata]['data'][:, :, iref])


def _update_mobile(k0=None, dmobile=None, dref=None, ddata=None):
    """ Update mobile objects data """

    func = dmobile[k0]['func']
    dtype = dmobile[k0]['dtype']

    kref = dmobile[k0]['ref']
    kdata = dmobile[k0]['data']
    iref = [dref[rr]['indices'][dmobile[k0]['ind']] for rr in kref]

    if kref[0] is not None:
        _update_mobile_data(
            func=func,
            kref=kref[0],
            kdata=kdata[0],
            iref=iref[0],
            ddata=ddata,
        )

    if len(kref) > 1 and kref[1] is not None:
        _update_mobile_data(
            func=func,
            kref=kref[1],
            kdata=kdata[1],
            iref=iref[1],
            ddata=ddata,
        )
