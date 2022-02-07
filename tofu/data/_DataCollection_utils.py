

from . import _generic_check


# #############################################################################
# #############################################################################
#               Selection
# #############################################################################


def _select(dd=None, dd_name=None, log=None, returnas=None, **kwdargs):
    """ Return the indices / keys of data matching criteria

    The selection is done comparing the value of all provided parameters
    The result is a boolean indices array, optionally with the keys list
    It can include:
        - log = 'all': only the data matching all criteria
        - log = 'any': the data matching any criterion

    If log = 'raw', a dict of indices arrays is returned, showing the
    details for each criterion

    """

    # -----------
    # check input

    # log
    log = _generic_var._check_var(
        log,
        'log',
        types=str,
        default='all',
        allowed=['all', 'any', 'raw'],
    )

    # returnas
    # 'raw' => return the full 2d array of boolean indices
    returnas = _generic_var._check_var(
        returnas,
        'returnas',
        default=bool if log == 'raw' else int,
        allowed=[int, bool, str, 'key'],
    )

    kwdargs = {k0: v0 for k0, v0 in kwdargs.items() if v0 is not None}

    # Get list of relevant criteria
    lp = [kk for kk in list(dd.values())[0].keys() if kk != 'data']
    lk = list(kwdargs.keys())
    lk = _generic_var._check_var_iter(
        lk,
        'lk',
        types_iter=str,
        default=lp,
        allowed=lp,
    )

    # --------------------
    # Get raw bool indices

    # Get list of accessible param
    ltypes = [float, np.float_]
    lquant = [
        kk for kk in kwdargs.keys()
        if any([type(dd[k0][kk]) in ltypes for k0 in dd.keys()])
    ]

    # Prepare array of bool indices and populate
    ind = np.zeros((len(kwdargs), len(dd)), dtype=bool)
    for ii, kk in enumerate(kwdargs.keys()):
        try:
            par = _get_param(
                dd=dd, dd_name=dd_name,
                param=kk,
                returnas=np.ndarray,
            )[kk]
            if kk in lquant:
                # list => in interval
                if isinstance(kwdargs[kk], list) and len(kwdargs[kk]) == 2:
                    ind[ii, :] = (
                        (kwdargs[kk][0] <= par) & (par <= kwdargs[kk][1])
                    )

                # tuple => out of interval
                elif isinstance(kwdargs[kk], tuple) and len(kwdargs[kk]) == 2:
                    ind[ii, :] = (
                        (kwdargs[kk][0] > par) | (par > kwdargs[kk][1])
                    )

                # float / int => equal
                else:
                    ind[ii, :] = par == kwdargs[kk]
            else:
                ind[ii, :] = par == kwdargs[kk]
        except Exception as err:
            try:
                ind[ii, :] = [
                    dd[k0][kk] == kwdargs[kk] for k0 in dd.keys()
                ]
            except Exception as err:
                msg = (
                    "Could not determine whether:\n"
                    + "\t- {}['{}'] == {}".format(
                        dd_name, kk, kwdargs[kk],
                    )
                )
                raise Exception(msg)

    # -----------------
    # Format output ind

    # return raw 2d array of bool indices
    if log == 'raw':
        if returnas in [str, 'key']:
            ind = {
                kk: [k0 for jj, k0 in enumerate(dd.keys()) if ind[ii, jj]]
                for ii, kk in enumerate(kwdargs.keys())
            }
        if returnas == int:
            ind = {
                kk: ind[ii, :].nonzero()[0]
                for ii, kk in enumerate(kwdargs.keys())
            }
        else:
            ind = {kk: ind[ii, :] for ii, kk in enumerate(kwdargs.keys())}

    else:
        # return all or any
        if log == 'all':
            ind = np.all(ind, axis=0)
        else:
            ind = np.any(ind, axis=0)

        if returnas == int:
            ind = ind.nonzero()[0]
        elif returnas in [str, 'key']:
            ind = np.array(
                [k0 for jj, k0 in enumerate(dd.keys()) if ind[jj]],
                dtype=str,
            )
    return ind


# #############################################################################
# #############################################################################
#               Identify references
# #############################################################################


def _get_keyingroup_ddata(
    dd=None, dd_name='data',
    key=None, monot=None,
    msgstr=None, raise_=False,
):
    """ Return the unique data key matching key

    Here, key can be interpreted as name / source / units / quant...
    All are tested using select() and a unique match is returned
    If not unique match an error message is either returned or raised

    """

    # ------------------------
    # Trivial case: key is actually a ddata key

    if key in dd.keys():
        return key, None

    # ------------------------
    # Non-trivial: check for a unique match on other params

    dind = _select(
        dd=dd, dd_name=dd_name,
        dim=key, quant=key, name=key, units=key, source=key,
        monot=monot,
        log='raw',
        returnas=bool,
    )
    ind = np.array([ind for kk, ind in dind.items()])

    # Any perfect match ?
    nind = np.sum(ind, axis=1)
    sol = (nind == 1).nonzero()[0]
    key_out, msg = None, None
    if sol.size > 0:
        if np.unique(sol).size == 1:
            indkey = ind[sol[0], :].nonzero()[0]
            key_out = list(dd.keys())[indkey]
        else:
            lstr = "[dim, quant, name, units, source]"
            msg = "Several possible matches in {} for {}".format(lstr, key)
    else:
        lstr = "[dim, quant, name, units, source]"
        msg = "No match in {} for {}".format(lstr, key)

    # Complement error msg and optionally raise
    if msg is not None:
        lk = ['dim', 'quant', 'name', 'units', 'source']
        dk = {
            kk: (
                dind[kk].sum(),
                sorted(set([vv[kk] for vv in dd.values()]))
            ) for kk in lk
        }
        msg += (
            "\n\nRequested {} could not be identified!\n".format(msgstr)
            + "Please provide a valid (unique) key/name/dim/quant/units:\n\n"
            + '\n'.join([
                '\t- {} ({} matches): {}'.format(kk, dk[kk][0], dk[kk][1])
                for kk in lk
            ])
            + "\nProvided:\n\t'{}'".format(key)
        )
        if raise_:
            raise Exception(msg)
    return key_out, msg


def _get_possible_ref12d(
    dd=None,
    key=None, ref1d=None, ref2d=None,
    group1d='radius',
    group2d='mesh2d',
):

    # Get relevant lists
    kq, msg = _get_keyingroup_ddata(
        dd=dd,
        key=key, group=group2d, msgstr='quant', raise_=False,
    )

    if kq is not None:
        # The desired quantity is already 2d
        k1d, k2d = None, None

    else:
        # Check if the desired quantity is 1d
        kq, msg = _get_keyingroup_ddata(
            dd=dd,
            key=key, group=group1d,
            msgstr='quant', raise_=True,
        )

        # Get dict of possible {ref1d: lref2d}
        ref = [rr for rr in dd[kq]['ref'] if dd[rr]['group'] == (group1d,)][0]
        lref1d = [
            k0 for k0, v0 in dd.items()
            if ref in v0['ref'] and v0['monot'][v0['ref'].index(ref)] is True
        ]

        # Get matching ref2d with same quant and good group
        lquant = list(set([dd[kk]['quant'] for kk in lref1d]))
        dref2d = {
            k0: [
                kk for kk in _select(
                    dd=dd, quant=dd[k0]['quant'],
                    log='all', returnas=str,
                )
                if group2d in dd[kk]['group']
                and not isinstance(dd[kk]['data'], dict)
            ]
            for k0 in lref1d
        }
        dref2d = {k0: v0 for k0, v0 in dref2d.items() if len(v0) > 0}

        if len(dref2d) == 0:
            msg = (
                "No match for (ref1d, ref2d) for ddata['{}']".format(kq)
            )
            raise Exception(msg)

        # check ref1d
        if ref1d is None:
            if ref2d is not None:
                lk = [k0 for k0, v0 in dref2d.items() if ref2d in v0]
                if len(lk) == 0:
                    msg = (
                        "\nNon-valid interpolation intermediate\n"
                        + "\t- provided:\n"
                        + "\t\t- ref1d = {}, ref2d = {}\n".format(ref1d, ref2d)
                        + "\t- valid:\n{}".format(
                            '\n'.join([
                                '\t\t- ref1d = {}  =>  ref2d in {}'.format(
                                    k0, v0
                                )
                                for k0, v0 in dref2d.items()
                            ])
                        )
                    )
                    raise Exception(msg)
                if kq in lk:
                    ref1d = kq
                else:
                    ref1d = lk[0]
            else:
                if kq in dref2d.keys():
                    ref1d = kq
                else:
                    ref1d = list(dref2d.keys())[0]
        else:
            ref1d, msg = _get_keyingroup_ddata(
                dd=dd,
                key=ref1d, group=group1d,
                msgstr='ref1d', raise_=False,
            )
        if ref1d not in dref2d.keys():
            msg = (
                "\nNon-valid interpolation intermediate\n"
                + "\t- provided:\n"
                + "\t\t- ref1d = {}, ref2d = {}\n".format(ref1d, ref2d)
                + "\t- valid:\n{}".format(
                    '\n'.join([
                        '\t\t- ref1d = {}  =>  ref2d in {}'.format(
                            k0, v0
                        )
                        for k0, v0 in dref2d.items()
                    ])
                )
            )
            raise Exception(msg)

        # check ref2d
        if ref2d is None:
            ref2d = dref2d[ref1d][0]
        else:
            ref2d, msg = _get_keyingroup_ddata(
                dd=dd,
                key=ref2d, group=group2d,
                msgstr='ref2d', raise_=False,
            )
        if ref2d not in dref2d[ref1d]:
            msg = (
                "\nNon-valid interpolation intermediate\n"
                + "\t- provided:\n"
                + "\t\t- ref1d = {}, ref2d = {}\n".format(ref1d, ref2d)
                + "\t- valid:\n{}".format(
                    '\n'.join([
                        '\t\t- ref1d = {}  =>  ref2d in {}'.format(
                            k0, v0
                        )
                        for k0, v0 in dref2d.items()
                    ])
                )
            )
            raise Exception(msg)

    return kq, ref1d, ref2d
