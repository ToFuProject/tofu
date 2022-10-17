

import numpy as np
import datastock as ds


# ##################################################################
# ##################################################################
#                   plot text fror ref on axes
# ##################################################################


def plot_text(
    coll=None,
    kax=None,
    ax=None,
    ref=None,
    group=None,
    ind=None,
    lkeys=None,
    nmax=None,
    color_dict=None,
    bstr_dict=None,
):

    # ------------------
    # Get list of data

    lk0 = ['index'] + [
        k0 for k0, v0 in coll._ddata.items()
        if v0['ref'] == (ref,)
    ]

    if isinstance(lkeys, str):
        lkeys = [lkeys]
    lk0 = ds._generic_check._check_var_iter(
        lkeys, 'lkeys',
        default=lk0,
        types=list,
        types_iter=str,
        allowed=lk0,
    )

    # ------------------
    # Get list of data

    if bstr_dict is None:
        bstr_dict = '{}'
    if isinstance(bstr_dict, str):
        bstr_dict = {k0: bstr_dict for k0 in lk0}
    c0 = (
        isinstance(bstr_dict, dict)
        and all([
            k0 in bstr_dict.keys()
            and isinstance(bstr_dict.get(k0), str)
            for k0 in lk0
        ])
    )
    if not c0:
        msg = (
            "Arg bstr_dict must be a dict with keys:\n"
            f"\t- keys: {lk0}\n"
            f"\t- Provided: {bstr_dict}"
        )
        raise Exception(msg)

    # ------------------
    # Compute placements

    nx = nmax + 1
    ny = len(lk0) + 1
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)

    # ----------------
    # Plot fixed text

    ax.text(
        np.mean(x),
        y[-1],
        f'{ref}',
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.transAxes,
        size=11,
        color='k',
        fontweight='bold',
    )

    for ii, k0 in enumerate(lk0):
        ax.text(
            x[0],
            y[-(ii + 2)],
            k0,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes,
            size=8,
            color='k',
            fontweight='bold',
        )

    # ----------------
    # Plot mobile text

    for ii, k0 in enumerate(lk0):
        for jj in range(nmax):
            if k0 == 'index':
                datastr = ind
                data = k0
            else:
                datastr = coll._ddata[k0]['data'][ind]
                data = k0
            ht = ax.text(
                x[jj + 1],
                y[-(ii + 2)],
                data,
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes,
                size=8,
                color=color_dict[group][jj],
                fontweight='bold',
            )

            coll.add_mobile(
                key=f'txt-{ref}-{k0}-{jj}',
                handle=ht,
                ref=(ref,),
                data=data,
                dtype='txt',
                bstr=bstr_dict[k0],
                ax=kax,
                ind=jj,
            )
