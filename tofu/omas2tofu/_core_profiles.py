

# ###########################################################
# ###########################################################
#              1d-2d mesh
# ###########################################################


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
            l2d = [k1 for k1 in lk2d if coll.ddata[k1]['name'] == v0['name']]
            if len(l2d) == 1:
                d1d_all[k0] = l2d[0]

        if len(d1d_all) == 0:
            msg = (
                "Could not identify a single matching quantity between "
                "equilibrium (2d) and core_profiles (1d)"
            )
            raise Exception(msg)

        # ------------
        # interpolate

        raise NotImplementedError()

    return k1d, q1d, k2dn
