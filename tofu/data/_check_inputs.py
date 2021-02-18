

# #############################################################################
# #############################################################################
#                           dgroup
# #############################################################################


def _check_remove_group(group=None, dgroup=None):
    c0 = isinstance(group, str) and group in dgroup.keys()
    c1 = (
        isinstance(goup, list)
        and all([isinstance(gg, str) and gg in dgroup.keys()])
    )
    if not c0:
        msg = (
            """
            Removed group must be a str already in self.dgroup
            It can also be a list of such
            \t- provided: {}
            \t- already available: {}
            """.format(group, sorted(dgroup.keys()))
        )
        raise Exception(msg)
    if c0:
        group = [group]
    return group


def _check_dgroup(dgroup=None, dgroup0=None):
    """ dgroup must be
    - str: turned to list
    - list of str
    - dict of dict
    """

    # Check conformity
    c0 = isinstance(dgroup, str) and dgroup in dgroup.keys()
    c1 = (
        isinstance(goup, list)
        and all([isinstance(gg, str) and gg in dgroup.keys()])
    )
    c2 = (
        isinstance(dgroup, dict)
        and all([
            isinstance(k0, str)
            and k0 not in dgroup0.keys()
            and isinstance(v0, dict)
            and all([
                isinstance(k1, str)
                and k1 in ['lref', 'ldata']
                and isinstance(v1, list)
                and all([isinstance(v2, str) for v2 in v1])
                for k1, v1 in v0.items()
            ])
            for k0, v0 not in dgroup.items()
        ])
    )
    if not (c0 or c1 or c2):
        msg = (
            """
            Added group must be either:
            \t- str: not already in self.dgroup
            \t- list of str: each not already in self.dgroup
            \t- dict: each key not already in self.dgroup, and each vlue a dict
            You provided:
            \t- {}
            Already available in self.dgroup:
            {}
            """.format(
                group,
                '\t- ' + '\n\t- '.join(sorted(dgroup.keys()))
            )
        )
        raise Exception(msg)

    # Convert if necessary
    if c0:
        dgroup = {dgroup: {}}
    elif c1:
        dgroup = {k0: {} for k0 in dgroup}
    return dgroup


# #############################################################################
# #############################################################################
#                           dref
# #############################################################################


def _check_dref(dref=None, dref0=None, dgroup0=None):
    """ Check and format dref

    dref can be:
        - dict
    """

    # ----------------
    # Check conformity
    ngroup = len(dgroup0)
    if ngroup == 1:
        groupref = list(dgroup0.keys())[0]

    # Basis
    # lk_opt = ['ldata', 'size', 'group', 'data']
    c0 = (
        isinstance(dref, dict)
        and all([
            isinstance(k0, str)
            and k0 not in dref0.keys()
            for k0, v0 in dref.items()
        ])
    )

    # Case {ref0: {'group': g0, 'data': d0, ...}, ...}
    c00 = (
        c0
        and ngroup > 1
        and all([
            and isinstance(v0, dict)
            and all([
                isinstance(k1, str)
                for k1, v1 in v0.items()
            ])
            for k0, v0 in dref.items()
        ])
    )

    # Case {ref0: data0, ...}
    c01 = (
        c0
        and ngroup == 1
        and all([isinstance(v0, np.ndarray) for k0, v0 in dref.items()])
    )

    if len(dgroup0) == 1:
        groupref = list(dgroup0.keys())[0]
        for k0, v0 in dref.items():
            if 'group' not in v0.keys():
                dref[k0]['group'] = groupref
    elif len(dgroup0) > 1:
        c0 = c0 and all(['group' in v0.keys() for v0 in dref.values()])

    # Raise exception if non-conformity
    if not (c00 or c01):
        msg = (
            """
            Arg dref must be a dict of the form:
            {
                'ref0': {'group': str, 'size': int, 'ldata': list},
                'ref1': {'group': str, 'size': int, 'ldata': list},
                ...
                'refn': {'group': str, 'size': int, 'ldata': list},
            }

            or of the form (only if len(dgroup) == 1):
            {
                'ref0': data0,
                'ref1': data1,
                ...
                'refn': datan,
            }

            Where:
                - each 'refi' is a unique str identifier
                - the {'group': str} pair refers to a key in self.dgroup
                    and is compulsory if self.dgroup has several keys

            """
        )
        raise Exception(msg)

    # ----------------
    # Convert if necessary
    if c01:
        dref = {
            k0: {'data': v0} for k0, v0 in dref.items()
        }

    # Add missing groups
    lgroups = [v0['group'] for v0 in dref.values()
               if 'group' in v0.keys() and v0['group'] not in dgroup0.keys()]
    if len(lgroups) > 0:
        dgroup0.update(_check_dgroup(lgroups, dgroup0=dgroup0))

    # Check groups
    lnogroup = [k0 for k0, v0 in dref.items() if 'group' not in v0.keys()]
    if ngroup == 1:
        for k0 in lnogroup:
            dref[k0]['group'] = groupref

    else:
        if len(lnogroup) > 0:
            msg = (
                """
                The following refs have no assigned group:
                {}

                The available groups are:
                {}
                """.format(
                    '\t- ' + '\n\t- '.join(lnogroup),
                    '\t- ' + '\n\t- '.join(sorted(dgroup0.keys())),
                )
            )
            raise Exception(msg)

    # Add data if relevant   TBF
    for k0, v0 in dref.items():
        if 'data' in v0.keys():
            pass


    return dref, dgroup0

    # Back-up

    # Options:
    #   (A)  - {'group0': {'t0': {'data': t0, 'units': 's'}, 't1':...}}
    #   (B)  - {'t0': {'data': t0, 'units': 's', 'group': 'group0'}, 't1':...}
    #   (C)  - {'t0': {'data': t0, 'units': 's'}, 't1':...}
    #   (D)  - {'t0': t0, 't1': t1, ...}

    # Check cB = normal case
    for kk, vv in dref.items():

        # Check data
        c0 = 'data' in vv.keys()
        data = vv['data']
        if not isinstance(data, np.ndarray):
            if isinstance(data, list) or isinstance(data, tuple):
                try:
                    data = np.atleast_1d(data).ravel()
                    size = data.size
                except Exception as err:
                    c0 = False
            else:
                size = data.__class__.__name__
        else:
            if data.ndim != 1:
                data = np.atleast_1d(data).ravel()
            size = data.size

        if not c0:
            msg = "dref[%s]['data'] must be array-convertible\n"%kk
            msg += "The following array conversion failed:\n"
            msg += "    - np.atleast_1d(dref[%s]['data']).ravel()"%kk
            raise Exception(msg)

        # Fill self._dref
        self._dref['dict'][kk] = {'size': size, 'group': vv['group']}
        self._dref['lkey'].append(kk)

        # Extract and check parameters
        dparams = self._extract_known_params(kk, vv, ref=True,
                                             group=vv['group'])

        # Fill self._ddata
        self._ddata['dict'][kk] = dict(data=data, refs=(kk,),
                                       shape=(size,), **dparams)
        self._ddata['lkey'].append(kk)


def _check_remove_ref(ref=None, dref=None):
    c0 = isinstance(ref, str) and ref in dref.keys()
    c1 = (
        isinstance(ref, list)
        and all([isinstance(k0, str) and k0 in dref.keys()])
    )
    if not c0:
        msg = (
            """
            Removed ref must be a str already in self.dref
            It can also be a list of such
            \t- provided: {}
            \t- already available: {}
            """.format(ref, sorted(dref.keys()))
        )
        raise Exception(msg)
    if c0:
        ref = [ref]
    return ref
