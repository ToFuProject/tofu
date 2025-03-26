

import datastock as ds


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

    return sep.join([prefix, ids, short]).strip(sep)


# ############################################
# ############################################
#             get short
# ############################################


def _get_short(
    din=None,
    short=None,
    ddef=None,
    strict=None,
):

    # ------------
    # check inputs
    # ------------

    short, ddef, strict = _check(
        din=din,
        short=short,
        ddef=ddef,
        strict=strict,
    )

    # ------------
    # get
    # ------------

    try:

        out = _short(din, ddef[short])

        dout = {
            'data': out,
            'units': ddef[short]['units'],
        }

    except Exception as err:
        if strict is True:
            raise err
        else:
            return err

    return dout


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
        e0, e1 = elem[:elem.index('.')]
    else:
        e0, e1 = elem, None

    # ---------------
    # sequence or not
    # ---------------

    if '[' in e0:
        ee = e0[:e0.index['[']]
        nn = len(din[ee])
        out = [_short(din[ee][ii], e1) for ii in range(nn)]

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
    ddef=None,
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

    strict = ds._generic._check_var(
        strict, 'strict',
        types=bool,
        default=False,
    )

    return short, ddef, strict
