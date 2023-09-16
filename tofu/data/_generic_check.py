# -*- coding: utf-8 -*-


# common
import matplotlib.pyplot as plt


_LALLOWED_AXESTYPES = [
    None,
    'cross',
    'hor',
    '3d',
    'camera',
    'matrix',
    'timetrace',
    'profile1d',
    'image',
    'text',
    'misc',
]


# ##################################################################
# ##################################################################
#                   Utilities for plotting
# ##################################################################


def _check_dax(dax=None, main=None):

    # ------------
    # trivial case

    if dax is None:
        return dax

    # -------------
    # Axes provided

    if issubclass(dax.__class__, plt.Axes):
        if main is None:
            msg = (
            )
            raise Exception(msg)
        else:
            return {main: dax}

    # --------------
    # check dict

    c0 = (
        isinstance(dax, dict)
        and all([
            isinstance(k0, str)
            and (
                (
                    issubclass(v0.__class__, plt.Axes)
                )
                or (
                    isinstance(v0, dict)
                    and issubclass(v0.get('handle').__class__, plt.Axes)
                    # and v0.get('type') in _LALLOWED_AXESTYPES
                )
            )
            for k0, v0 in dax.items()
        ])
    )
    if not c0:
        msg = (
        )
        import pdb; pdb.set_trace()     # DB
        raise Exception(msg)

    # --------------
    # make dict

    for k0, v0 in dax.items():
        if issubclass(v0.__class__, plt.Axes):
            dax[k0] = {'handle': v0, 'type': k0}
        if isinstance(v0, dict):
            dax[k0]['type'] = v0.get('type')

    return dax
