# -*- coding: utf-8 -*-


import numpy as np


# #################################################################
# #################################################################
#               Unit orthonormal vectors
# #################################################################


def _check_unitvector(uv=None, uv_name=None):
    try:
        uv = np.atleast_1d(uv).ravel().astype(float)
        assert uv.shape == (3,)
    except Exception as err:
        msg = str(err) + (
            f"\nArg {uv_name} not convertible to (3,) float np.ndarray!"
            "Provided: {uv}"
        )
        raise Exception(msg)

    # enforce normalization
    return uv / np.linalg.norm(uv)


def _check_nine0e1(nin=None, e0=None, e1=None, key=None):

    # e0 or e0 provided => compute missing one
    if e0 is None and e1 is not None:
        e0 = np.cross(e1, nin)
    elif e0 is not None and e1 is None:
        e1 = np.cross(nin, e0)

    # either e0 and e1 provided or none
    if e0 is not None:
        dv = {
            'nin.e0': np.abs(np.sum(nin*e0)),
            'nin.e1': np.abs(np.sum(nin*e1)),
            'e0.e1': np.abs(np.sum(e0*e1)),
            '|nin.(e0 x e1)|': np.linalg.norm(np.cross(nin, np.cross(e0, e1))),
        }
        dv = {k0: v0 for k0, v0 in dv.items() if v0 > 1.e-15}
        if len(dv) > 0:
            lstr = [f'\t- {k0}: {v0}' for k0, v0 in dv.items()]
            msg = (
                f"Args (e0, e1, nin) for '{key}' are non-direct orthonormal!\n"
                + "\n".join(lstr)
            )
            raise Exception(msg)

    return nin, e0, e1
