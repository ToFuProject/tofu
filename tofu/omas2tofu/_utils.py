

import numpy as np
import datastock as ds


from . import _ddef


# ############################################
# ############################################
#             Make key
# ############################################


def _make_key(
    prefix=None,
    ids=None,
    short=None,
    sep="_",
):

    # -----------
    # check

    if prefix in [None, False]:
        prefix = ''

    if not isinstance(prefix, str):
        msg = f"Arg prefix must be a str!\nProvided: {prefix}"
        raise Exception(msg)

    return sep.join([prefix, _ddef._DIDS[ids], short]).strip(sep)


# ############################################
# ############################################
#             get short
# ############################################


def _get_short(
    din=None,
    ids=None,
    short=None,
    dshort=None,
    strict=None,
    prefix=None,
):

    # ------------
    # check inputs
    # ------------

    short, strict = _check(
        din=din,
        short=short,
        strict=strict,
    )

    # ------------
    # get
    # ------------

    try:

        # -----------
        # ddata

        out = _short(din[ids], dshort[ids][short]['long'])

        # key
        key = _make_key(
            prefix=prefix,
            ids=ids,
            short=short,
        )

        # ref
        if 'ref' in dshort[ids][short].keys():
            ref = tuple([
                rr if rr == 'im2d'
                else _make_key(
                    prefix=prefix,
                    ids=ids,
                    short=rr,
                )
                for rr in dshort[ids][short]['ref']
            ])
        else:
            ref = None

        # ddata
        ddata = {
            short: {
                'key': key,
                'data': out,
                'ref': ref,
                'units': dshort[ids][short].get('units'),
                'dim': dshort[ids][short].get('dim'),
                'quant': dshort[ids][short].get('quant'),
                'name': dshort[ids][short].get('name'),
            }
        }

        # ----------
        # dref

        if dshort[ids][short].get('ref0') is not None:

            # key of ref
            kref = _make_key(
                prefix=prefix,
                ids=ids,
                short=dshort[ids][short]['ref0'],
            )

            # supposed to be an array
            nref = dshort[ids][short]['long'].count('[')
            out = np.array(out)
            if out.ndim != nref:
                msg = "Something weird..."
                raise Exception(msg)

            # dref
            indr = ref.index(kref)
            dref = {
                kref: {
                    'key': kref,
                    'size': out.shape[indr],
                },
            }

        else:
            dref = None

    except Exception as err:

        if strict is True:
            raise err
        else:
            return err, None

    return ddata, dref


# ############################################
# ############################################
#             Elementary
# ############################################


def _short(din=None, elem=None):

    # ---------
    # trivial
    # ---------

    if elem is None:
        return din

    # ---------------
    # split round '.'
    # ---------------

    if '.' in elem:
        ind = elem.index('.')
        e0, e1 = elem[:ind], elem[ind+1:]
    else:
        e0, e1 = elem, None

    # ---------------
    # sequence or not
    # ---------------

    if '[' in e0:
        i0 = e0.index('[')
        i1 = e0.index(']')
        ee = e0[:i0]
        rin = e0[i0+1:i1]

        if rin in ['im2d']:
            out = _short(din[ee][0], e1)
        else:
            nn = len(din[ee])
            out = np.array([_short(din[ee][ii], e1) for ii in range(nn)])

    else:
        out = _short(din[e0], e1)

    return out


# ############################################
# ############################################
#             Check
# ############################################


def _check(
    din=None,
    short=None,
    strict=None,
):

    # -------------
    # din
    # -------------

    if not isinstance(din, dict):
        msg = (
            "Arg din must be a dict\n"
            f"Provided:\n{din}\n"
        )
        raise Exception(msg)

    # -------------
    # short
    # -------------

    if not isinstance(short, str):
        msg = (
            "Arg short must be a str in din.keys()!\n"
            f"Provided:\n{short}\n"
        )
        raise Exception(msg)

    # -------------
    # strict
    # -------------

    strict = ds._generic_check._check_var(
        strict, 'strict',
        types=bool,
        default=False,
    )

    return short, strict
