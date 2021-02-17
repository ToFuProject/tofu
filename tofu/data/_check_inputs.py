

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
    # Check conformity
    lk_opt = ['ldata', 'size', 'group']
    c0 = (
        isinstance(dref, dict)
        and all([
            isinstance(k0, str)
            and k0 not in dref0.keys()
            and isinstance(v0, dict)
            and all([
                isinstance(k1, str)
                k1 in lk_opt
                for k1, v1 in v0.items()
            ])
            for k0, v0 in dref.items()
        ])
    )
    if len(dgroup0) == 1:
        groupref = list(dgroup0.keys())[0]
        for k0, v0 in dref.items():
            if 'group' not in v0.keys():
                dref[k0]['group'] = groupref
    elif len(dgroup0) > 1:
        c0 = c0 and all(['group' in v0.keys() for v0 in dref.values()])

    c0 = c0 and all([v0['group'] in dgroup0.keys() for v0 in self.values()])

    if not c0:
        msg = (
            """
            Arg dref must be a dict of the form:
            {
                'ref0': {'group': str, 'size': int, 'ldata': list},
                'ref1': {'group': str, 'size': int, 'ldata': list},
                ...
                'refn': {'group': str, 'size': int, 'ldata': list},
            }

            Where:
                - each 'refi' is a unique str identifier
                - the {'group': str} pair refers to a key in self.dgroup
                    and is compulsory if self.dgroup has several keys

            """
        )
        raise Exception(msg)


    # TBF

    # Options:
    #   (A)  - {'group0': {'t0': {'data': t0, 'units': 's'}, 't1':...}}
    #   (B)  - {'t0': {'data': t0, 'units': 's', 'group': 'group0'}, 't1':...}
    #   (C)  - {'t0': {'data': t0, 'units': 's'}, 't1':...}
    #   (D)  - {'t0': t0, 't1': t1, ...}

    cA = all([all([(isinstance(v1, dict) and 'group' not in v1.keys())
                   or not isinstance(v1, dict)
                   for v1 in v0.values()])
              and 'group' not in v0.keys() for v0 in dref.values()])
    cB = all([isinstance(v0, dict) and isinstance(v0.get('group', None), str)
              for v0 in dref.values()])
    cC = (not cA and self._forced_group is not None
          and all([isinstance(v0, dict) and 'group' not in v0.keys()
                   for v0 in dref.values()]))
    cD = (self._forced_group is not None
          and all([not isinstance(v0, dict) for v0 in dref.values()]))
    assert np.sum([cA, cB, cC, cD]) <= 1
    if not (cA or cB or cC or cD):
        msg = "Provided dref must formatted either as a dict with:\n\n"
        msg += "    - keys = group, values = {ref: data}:\n"
        msg += "        {'time':{'t0':{'data':t0, 'units':'s'},\n"
        msg += "                 't1':{'data':t1, 'units':'h'}},\n"
        msg += "         'dist':{'x0':{'data':x0, 'units':'m'}}}\n\n"
        msg += "    - keys = ref, values = {data, group, ...}:\n"
        msg += "        {'t0':{'data':t0, 'units':'s', 'group':'time'},\n"
        msg += "         't1':{'data':t1, 'units':'h', 'group':'time'},\n"
        msg += "         'x0':{'data':x0, 'units':'m', 'group':'dist'}\n\n"
        msg += "    If self._forced_group is not None, 2 more options:\n"
        msg += "    - keys = ref, values = {data, ...}:\n"
        msg += "        {'t0':{'data':t0, 'units':'s'},\n"
        msg += "         't1':{'data':t1, 'units':'h'},\n"
        msg += "         'x0':{'data':x0, 'units':'m'}\n"
        msg += "    - keys = ref, values = data:\n"
        msg += "        {'t0':t0,\n"
        msg += "         't1':t1,\n"
        msg += "         'x0':x0}\n"
        raise Exception(msg)

    if cA:
        # Convert to cB
        drbis = {}
        for k0, v0 in dref.items():
            for k1, v1 in v0.items():
                if isinstance(v1, dict):
                    drbis[k1] = v1
                    drbis['group'] = k0
                else:
                    drbis[k1] = {'data': v1, 'group': k0}
        dref = drbis

    # Check cC and cD and convert to cB
    if cC:
        # Convert to cB
        for k0 in dref.keys():
            dref[k0]['group'] = self._forced_group
    elif cD:
        # Convert to cB
        for k0, v0 in dref.items():
            dref[k0] = {'data': v0, 'group': self._forced_group}


    # Check cB = normal case
    for kk, vv in dref.items():

        # Check if new group
        if vv['group'] not in self._dgroup['lkey']:
            self._dgroup['dict'][vv['group']] = {}
            self._dgroup['lkey'].append(vv['group'])

        # Check key unicity
        if kk in self._ddata['lkey']:
            msg = "key '%s' already used !\n"%kk
            msg += "  => each key must be unique !"
            raise Exception(msg)

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
