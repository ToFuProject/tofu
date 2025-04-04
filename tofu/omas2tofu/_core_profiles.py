






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
):

    # ------------------
    # check expectations
    # ------------------

    import pdb; pdb.set_trace()     # DB

    # ------------------
    # back-up
    # ------------------

    # get k2
    lk1 = [kk for kk in ldata if kk.endswith(k1d[:-1])]
    lk2 = [kk for kk in lk2d if kk.endswith(k1d.replace('1d', '2d')[:-1])]
    if len(lk1) != 1:
        msg = f"Unidentified 1d base for 2d subkey: {lk1}"
        raise Exception(msg)
    if len(lk2) != 1:
        msg = f"Unidentified 2d subkey: {lk2}"
        raise Exception(msg)
    k2d = lk2[0]

    # slices
    sli0 = tuple([
        0 if ii == axis else slice(None) for ii in range(len(shape))
    ])
    sli1 = tuple([
        -1 if ii == axis else slice(None) for ii in range(len(shape))
    ])
    sli2 = tuple(itt.chain.from_iterable([
        [None, None] if ii == axis
        else [slice(None)] for ii in range(len(shape))
    ]))

    # compute normalization
    q1d0 = ddata[lk1[0]]['data'][sli0][sli2]
    q1d1 = ddata[lk1[0]]['data'][sli1][sli2]
    q2dn = (coll.ddata[k2d]['data'] - q1d0) / (q1d1 - q1d0)

    # add data2d
    k2dn = f"{k2d}n"
    coll.add_data(
        key=k2dn,
        data=q2dn,
        ref=coll.ddata[k2d]['ref'],
    )


    return k1d, q1d, k2dn
