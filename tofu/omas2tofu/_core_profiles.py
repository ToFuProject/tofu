

# ################################################
# ################################################
#              1d-2d mesh
# ################################################


def _get_subkey(
    coll=None,
    ids=None,
    shape=None,
    axis=None,
    ddata=None,
    ldata=None,
    lk2d=None,
    k1d=None,
    key_1d=None,
    q1d=None,
    # unused
    **kwdargs,
):

    # ------------------
    # check expectations
    # ------------------

    # check no 2d mesh in ids
    if len(lk2d) != 0:
        msg = "Expected 'core_profiles' to be empty of 2d meshes!"
        raise Exception(msg)

    # check equilibrium is already added
    lk2d = [
        k0 for k0, v0 in coll.ddata.items()
        if v0.get('bsplines') is not None
        and 'eq_2d' in k0
    ]
    if len(lk2d) == 0:
        msg = "No 2d equilibrium data found!"
        raise Exception(msg)

    # ------------------
    # identify 2d data from equilibrium matching k1d
    # ------------------

    lkey_2d = [
        kk for kk in lk2d
        if coll.ddata[kk]['name'] == ddata[k1d]['name']
        or coll.ddata[kk]['quant'] == ddata[k1d]['quant']
    ]

    if len(lkey_2d) == 1:
        k2dn = lkey_2d[0]

    # ------------------
    # identify 1d time-varying data from core_profiles matching 2d equilibrium
    # ------------------

    else:
        d1d_all = {}
        for k0, v0 in ddata.items():
            l2d = [
                k1 for k1 in lk2d
                if coll.ddata[k1]['name'] == v0['name']
            ]
            if len(l2d) == 1:
                d1d_all[k0] = l2d[0]

        if len(d1d_all) == 0:
            msg = (
                "Could not identify a matching quantity between:\n"
                "\t- equilibrium (2d)"
                "\t- core_profiles (1d)"
            )
            raise Exception(msg)

        # ------------
        # interpolate

        elif len(d1d_all) > 1:

            keep, ii = True, 0
            lprio = ['rhopn', 'rhotn', 'psi', 'phi']
            while keep is True and ii < len(lprio):
                lk1 = [
                    kk for kk, vv in d1d_all.items()
                    if lprio[ii] in kk and lprio[ii] in vv
                ]
                if len(lk1) == 1:
                    k1d = lk1[0]
                    k2dn = d1d_all[k1d]
                    keep = False
                else:
                    ii += 1
            if keep is True:
                msg = "Multiple matching 1d (cprof) <=> 2d (eq)"
                raise NotImplementedError(msg)

        else:
            msg = "Time-varying radial interpolation"
            raise NotImplementedError(msg)

    return k1d, q1d, k2dn
